"""
Карты времён активации и реполяризации.
========================================

Регистратор смотрит на потенциал после КАЖДОГО электрического шага
(`on_electric_step`) и отмечает для каждого узла:

    активация      — пересечение порога `activation_threshold` модели
                     клетки снизу вверх;
    реполяризация  — спад до уровня V_base + (1 − apd_level)·(V_peak − V_base),
                     где V_peak — максимум ЭТОГО удара в этом узле, а
                     V_base — минимум потенциала в диастоле перед ним (в
                     этом же узле). При apd_level = 0.9 это APD90.

Базовый уровень берётся по узлу, а не из модели: в ишемической зоне
покой деполяризован (при K_o = 9.4 мМ — около −71.6 мВ вместо −85.9), и
уровень, отсчитанный от покоя модели, там не достигался бы никогда —
APD в самой интересной области молча пропал бы. `v_rest` модели нужен
только узлам, возбуждённым уже в момент старта.

Моменты уточняются линейной интерполяцией между соседними шагами, так
что точность — доли dt_эл, а не dt_мех, на котором работают остальные
наблюдатели. Для скорости проведения это существенно: при 0.12 мм/мс
фронт проходит шаг сетки 0.25 мм за ~2 мс.

Удары считаются в каждом узле отдельно: k-я активация узла — его k-й
удар. Порог реполяризации заметно ниже порога активации, поэтому шум
у порога не порождает ложных ударов (гистерезис).

Файл activation.npz (пишет ранг 0 в конце счёта; упорядочено по (y, x)):

    coords (N, 2), region (N,)        — узлы эл. сетки, 0 = базовая ткань
    act, repol, peak (N, B)           — по ударам; NaN — не было
    threshold, apd_level, v_rest      — как считалось
    t_start_ms, t_end_ms, dt_ms

Узлы, уже возбуждённые в момент старта (продолжение с чекпоинта
посреди потенциала действия), получают удар с act = NaN: момент их
активации в этом прогоне неизвестен.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..runtime.observers import Observer

__all__ = ["ThresholdDetector", "ActivationRecorder", "ACTIVATION_FILE"]

ACTIVATION_FILE = "activation.npz"


class ThresholdDetector:
    """
    Счётчик пересечений для набора узлов — чистый numpy, без MPI и
    DOLFINx. Отдельно от наблюдателя, чтобы его можно было проверить на
    искусственных сигналах.
    """

    def __init__(self, v0: np.ndarray, t0: float, threshold: float,
                 v_rest: float, apd_level: float = 0.9):
        v0 = np.asarray(v0, dtype=np.float64).copy()
        self.threshold = float(threshold)
        self.v_rest = float(v_rest)
        self.apd_level = float(apd_level)
        self.t0 = float(t0)
        self._v = v0
        self._t = float(t0)
        n = len(v0)
        self.active = v0 >= self.threshold
        self.peak = np.where(self.active, v0, -np.inf)
        # минимум в текущей диастоле и базовый уровень текущего удара
        self.base = np.where(self.active, np.inf, v0)
        self.beat_base = np.where(self.active, min(self.v_rest, float(v0.min(initial=self.v_rest))),
                                  v0)
        self.count = self.active.astype(np.int64)
        # события: (узел, номер удара, время[, пик])
        self._act = [(np.flatnonzero(self.active),
                      np.zeros(int(self.active.sum()), dtype=np.int64),
                      np.full(int(self.active.sum()), np.nan))]
        self._rep: list[tuple] = []
        self.n = n

    def update(self, v: np.ndarray, t: float) -> None:
        """Новое значение потенциала на момент t (> предыдущего)."""
        v = np.asarray(v, dtype=np.float64)
        prev, t_prev = self._v, self._t
        dt = t - t_prev
        th = self.threshold

        # ── активация: снизу вверх через порог ────────────────────────
        up = ~self.active & (prev < th) & (v >= th)
        rest = ~self.active & ~up
        self.base[rest] = np.minimum(self.base[rest], v[rest])
        if up.any():
            idx = np.flatnonzero(up)
            frac = (th - prev[idx]) / (v[idx] - prev[idx])
            self._act.append((idx, self.count[idx].copy(), t_prev + frac * dt))
            self.count[idx] += 1
            self.active[idx] = True
            self.peak[idx] = v[idx]
            self.beat_base[idx] = np.minimum(self.base[idx], prev[idx])
            self.base[idx] = np.inf

        # ── пик и реполяризация у уже активных ────────────────────────
        act = self.active & ~up
        self.peak[act] = np.maximum(self.peak[act], v[act])
        level = self.beat_base + (1.0 - self.apd_level) * (self.peak - self.beat_base)
        down = act & (v < level)
        if down.any():
            idx = np.flatnonzero(down)
            lv = level[idx]
            denom = prev[idx] - v[idx]
            frac = np.where(denom > 0, (prev[idx] - lv) / np.where(denom > 0, denom, 1.0), 1.0)
            frac = np.clip(frac, 0.0, 1.0)
            self._rep.append((idx, self.count[idx] - 1, t_prev + frac * dt,
                              self.peak[idx].copy()))
            self.active[idx] = False
            self.peak[idx] = -np.inf
            self.base[idx] = v[idx]

        self._v = v.copy()
        self._t = float(t)

    @property
    def n_beats(self) -> int:
        return int(self.count.max()) if self.n else 0

    def maps(self, n_beats: int | None = None) -> tuple[np.ndarray, ...]:
        """(act, repol, peak), каждая (n, B), NaN там, где события не было."""
        B = self.n_beats if n_beats is None else n_beats
        act = np.full((self.n, B), np.nan)
        rep = np.full((self.n, B), np.nan)
        peak = np.full((self.n, B), np.nan)
        for idx, beat, t in self._act:
            act[idx, beat] = t
        for idx, beat, t, pk in self._rep:
            rep[idx, beat] = t
            peak[idx, beat] = pk
        # пик удара, не успевшего реполяризоваться к концу счёта
        still = np.flatnonzero(self.active)
        if still.size:
            peak[still, self.count[still] - 1] = self.peak[still]
        return act, rep, peak


class ActivationRecorder(Observer):
    """Наблюдатель: ведёт ThresholdDetector по владеемым узлам и пишет карты."""

    def __init__(self, out_dir, apd_level: float = 0.9,
                 threshold: float | None = None):
        self.out_dir = Path(out_dir)
        self.apd_level = apd_level
        self.threshold_override = threshold
        self.detector: ThresholdDetector | None = None
        self._t_start = 0.0

    @property
    def path(self) -> Path:
        return self.out_dir / ACTIVATION_FILE

    def _owned_v(self, sim) -> np.ndarray:
        e = sim.electrics
        return e.potential()[:e.n_owned]

    def on_start(self, sim, schedule):
        cell = sim.cell_model
        th = (cell.activation_threshold if self.threshold_override is None
              else self.threshold_override)
        v_rest = float(cell.resting_state()[cell.v_index])
        self._t_start = schedule.t_start_ms
        self._dt = schedule.dt
        self.detector = ThresholdDetector(self._owned_v(sim), schedule.t_start_ms,
                                          th, v_rest, self.apd_level)

    def on_electric_step(self, sim, tick):
        self.detector.update(self._owned_v(sim), tick.t_end_ms)

    def on_finish(self, sim):
        from ._points import gather_owned
        from mpi4py import MPI

        det = self.detector
        comm = sim.comm
        B = comm.allreduce(det.n_beats, op=MPI.MAX)
        act, rep, peak = det.maps(B)
        e = sim.electrics
        n = e.n_owned
        # гало не участвуют в детекторе — дополняем массивы до локальных
        # узлов нулями, gather_owned всё равно их отрежет
        pad = e.n_local - n

        def full(a):
            return np.vstack([a, np.zeros((pad, a.shape[1]))]) if pad else a

        region = sim.tissue_e.region_id_p1[:e.n_local] + 1
        stacked = np.column_stack([full(act), full(rep), full(peak),
                                   region.astype(np.float64)])
        coords, vals = gather_owned(e.P1, stacked, comm)
        if comm.rank == 0:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.savez_compressed(
                    fh, coords=coords,
                    act=vals[:, :B], repol=vals[:, B:2 * B], peak=vals[:, 2 * B:3 * B],
                    region=vals[:, 3 * B].astype(np.int32),
                    threshold=np.array(det.threshold),
                    apd_level=np.array(det.apd_level),
                    v_rest=np.array(det.v_rest),
                    t_start_ms=np.array(self._t_start),
                    t_end_ms=np.array(float(sim.t_ms)),
                    dt_ms=np.array(self._dt))
            tmp.replace(self.path)
        comm.Barrier()
