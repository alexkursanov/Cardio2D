"""
Чекпоинты: сохранение и продолжение расчёта.
=============================================

Состояние хранится как пары (координата → значение) — см. раздел 8
ARCHITECTURE.md. Поэтому чекпоинт, снятый на одном ядре, читается
прогоном на восьми и наоборот, и продолжить можно в ДРУГОМ эксперименте
(другая папка вывода, другой протокол стимуляции, другие параметры
ткани).

Состав файла (.npz, без pickle)
-------------------------------
    format, version        — "cardiac_em.checkpoint", 1
    stage                  — "preloaded" (сразу после преднагрузки)
                             или "running"
    t_ms                   — момент, к которому относится состояние
    mech_index             — глобальный номер механического шага
    config_json            — полная конфигурация прогона-источника
    cell_model, state_names
    e_coords (N, 2), e_state (N, n_states) — клетки на электрической сетке
    m_coords (M, 2), m_u (M, 2)            — перемещения на механической

T_act не хранится: при продолжении он пересчитывается из электрического
состояния, то есть ровно так же, как в исходном прогоне.

Запись атомарная (во временный файл, затем переименование): если счёт
убьют посреди записи, прежний ckpt_last.npz останется целым.

Продолжение
-----------
    sim = Simulation(config)
    info = restore_checkpoint(sim, "old/ckpt_last.npz")
    sim.run(t_start_ms=info.t_start_ms)

Параметры `allow_interp` и `reset_time` берутся из `config.restart`, если
не заданы явно.

Совместимость с монолитной версией
----------------------------------
Чекпоинты legacy/electromechanics_dolfinx.py читаются (модель
Роджерса–МакКаллоха, переменные u_ap, v_ap). Одна тонкость: там момент
в файле — НАЧАЛО последнего электрического шага, а состояние уже в
конце шага. При чтении время сдвигается на dt, чтобы оно
соответствовало состоянию.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config.simulation import SimulationConfig
from ..runtime.observers import Observer
from ._points import gather_owned, local_coordinates, match_points

__all__ = [
    "CheckpointData",
    "RestoreInfo",
    "save_checkpoint",
    "pack_checkpoint",
    "read_checkpoint",
    "restore_checkpoint",
    "CheckpointObserver",
]

FORMAT = "cardiac_em.checkpoint"
VERSION = 1

# Разделы конфигурации, расхождение в которых делает загруженное
# состояние «чужим» для текущей физики. О них предупреждаем. Протокол
# стимуляции и шаги по времени в список не входят: менять их при
# продолжении — обычное дело.
_PHYSICS_SECTIONS = ("mesh", "tissue_base", "regions", "preload",
                     "cell_model", "passive_material")


@dataclass
class CheckpointData:
    """Содержимое чекпоинта, одинаковое на всех рангах."""

    path: Path
    stage: str
    t_ms: float
    mech_index: int
    cell_model: str
    state_names: tuple[str, ...]
    e_coords: np.ndarray
    e_state: np.ndarray
    m_coords: np.ndarray
    m_u: np.ndarray
    config_dict: dict | None = None      # None — у чекпоинтов монолитной версии
    created: str | None = None
    legacy: bool = False

    def config(self) -> SimulationConfig | None:
        """Конфигурация прогона-источника (если сохранена)."""
        if self.config_dict is None:
            return None
        return SimulationConfig.from_dict(self.config_dict)


@dataclass
class RestoreInfo:
    """Что произошло при восстановлении — для печати и манифеста."""

    source: Path
    stage: str
    t_checkpoint_ms: float
    t_start_ms: float
    nearest_nodes: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"source": str(self.source), "stage": self.stage,
                "t_checkpoint_ms": self.t_checkpoint_ms,
                "t_start_ms": self.t_start_ms,
                "nearest_nodes": dict(self.nearest_nodes),
                "warnings": list(self.warnings)}


# ═══════════════════════════════════════════════════════════════════════
#  ЗАПИСЬ
# ═══════════════════════════════════════════════════════════════════════

def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:                 # файловый объект: numpy
        np.savez_compressed(fh, **arrays)       # не допишет второй .npz
    os.replace(tmp, path)


def save_checkpoint(sim, path, *, stage: str = "running",
                    mech_index: int = 0) -> Path:
    """
    Сохранить состояние `sim` в `path`. КОЛЛЕКТИВНАЯ операция: вызывать
    на всех рангах. Пишет нулевой ранг.
    """
    if stage not in ("preloaded", "running"):
        raise ValueError(f"stage должен быть 'preloaded' или 'running', "
                         f"получено {stage!r}")
    path = Path(path)
    comm = sim.comm
    e = sim.electrics
    m = sim.mechanics

    e_coords, e_state = gather_owned(e.P1, e.state, comm)
    m_coords, m_u = gather_owned(m.V, m.u.x.array.reshape(-1, 2), comm)

    if comm.rank == 0:
        _atomic_savez(path, **pack_checkpoint(
            stage=stage, t_ms=sim.t_ms, mech_index=mech_index,
            config_dict=sim.config.to_dict(skip_unserializable=True),
            cell_model=sim.cell_model.name,
            state_names=sim.cell_model.state_names,
            e_coords=e_coords, e_state=e_state, m_coords=m_coords, m_u=m_u))
    comm.Barrier()
    return path


def pack_checkpoint(*, stage, t_ms, mech_index, config_dict, cell_model,
                    state_names, e_coords, e_state, m_coords, m_u) -> dict:
    """
    Массивы для .npz в формате чекпоинта. Отдельно от `save_checkpoint`,
    чтобы формат можно было проверить без DOLFINx и чтобы сторонние
    инструменты могли собрать совместимый чекпоинт.
    """
    return dict(
        format=np.array(FORMAT), version=np.array(VERSION),
        stage=np.array(stage),
        t_ms=np.array(float(t_ms)),
        mech_index=np.array(int(mech_index)),
        config_json=np.array(json.dumps(config_dict, ensure_ascii=False)),
        cell_model=np.array(cell_model),
        state_names=np.array(tuple(state_names)),
        created=np.array(datetime.now(timezone.utc).isoformat(timespec="seconds")),
        e_coords=np.asarray(e_coords, dtype=np.float64),
        e_state=np.asarray(e_state, dtype=np.float64),
        m_coords=np.asarray(m_coords, dtype=np.float64),
        m_u=np.asarray(m_u, dtype=np.float64),
    )


# ═══════════════════════════════════════════════════════════════════════
#  ЧТЕНИЕ
# ═══════════════════════════════════════════════════════════════════════

def _from_npz(path: Path, d: dict) -> CheckpointData:
    if "format" in d:
        fmt = str(d["format"])
        if fmt != FORMAT:
            raise ValueError(f"{path}: неизвестный формат {fmt!r}")
        version = int(d["version"])
        if version > VERSION:
            raise ValueError(
                f"{path}: чекпоинт версии {version}, этот код понимает до "
                f"{VERSION} — обновите код")
        return CheckpointData(
            path=path, stage=str(d["stage"]), t_ms=float(d["t_ms"]),
            mech_index=int(d["mech_index"]), cell_model=str(d["cell_model"]),
            state_names=tuple(str(s) for s in d["state_names"]),
            e_coords=d["e_coords"], e_state=d["e_state"],
            m_coords=d["m_coords"], m_u=d["m_u"],
            config_dict=json.loads(str(d["config_json"])),
            created=str(d["created"]) if "created" in d else None,
        )

    if {"u_ap", "v_ap", "coords_p1", "coords_u", "u"} <= set(d):
        # Монолитная версия: см. докстринг модуля о сдвиге времени.
        mech_count = int(d["mech_count"])
        t = float(d["t"])
        if mech_count > 0:
            t = round(t + float(d["dt_e"]), 12)
        return CheckpointData(
            path=path, stage="preloaded" if mech_count == 0 else "running",
            t_ms=t, mech_index=mech_count,
            cell_model="rogers_mcculloch", state_names=("u", "v"),
            e_coords=d["coords_p1"][:, :2],
            e_state=np.column_stack([d["u_ap"].reshape(-1), d["v_ap"].reshape(-1)]),
            m_coords=d["coords_u"][:, :2], m_u=d["u"].reshape(-1, 2),
            legacy=True,
        )

    raise ValueError(f"{path}: это не чекпоинт cardiac_em и не чекпоинт "
                     f"монолитной версии (ключи: {sorted(d)})")


def read_checkpoint(path, comm=None) -> CheckpointData:
    """
    Прочитать чекпоинт. С `comm` читает нулевой ранг и рассылает
    остальным (КОЛЛЕКТИВНАЯ операция); без него — просто читает.
    """
    path = Path(path)
    if comm is None or comm.rank == 0:
        try:
            with np.load(path, allow_pickle=False) as npz:
                payload = _from_npz(path, {k: npz[k] for k in npz.files})
        except Exception as exc:  # noqa: BLE001 — переслать ошибку всем рангам
            payload = exc
    else:
        payload = None
    if comm is not None:
        payload = comm.bcast(payload, root=0)
    if isinstance(payload, Exception):
        raise payload
    return payload


# ═══════════════════════════════════════════════════════════════════════
#  ВОССТАНОВЛЕНИЕ
# ═══════════════════════════════════════════════════════════════════════

def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): obj}


def _config_differences(src: dict, cur: dict) -> list[str]:
    diffs = []
    for section in _PHYSICS_SECTIONS:
        a = _flatten({section: src.get(section)})
        b = _flatten({section: cur.get(section)})
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                diffs.append(f"{key}: {a.get(key)!r} → {b.get(key)!r}")
    return diffs


def _same_geometry(a: dict, b: dict) -> bool:
    return (np.isclose(a["lx_mm"], b["lx_mm"]) and np.isclose(a["ly_mm"], b["ly_mm"]))


def _check_geometry(data: CheckpointData, sim) -> None:
    """Размеры области обязаны совпадать — перенос это не исправит."""
    if data.config_dict is None:
        return
    src_mesh = data.config_dict["mesh"]
    cur_mesh = sim.config.mesh.to_dict()
    for part in ("electric", "mechanical"):
        if not _same_geometry(src_mesh[part], cur_mesh[part]):
            raise ValueError(
                f"область {part}-сетки в чекпоинте "
                f"{src_mesh[part]['lx_mm']}×{src_mesh[part]['ly_mm']} мм, "
                f"в текущей конфигурации "
                f"{cur_mesh[part]['lx_mm']}×{cur_mesh[part]['ly_mm']} мм — "
                f"продолжить на другой геометрии нельзя")


def restore_checkpoint(sim, source, *, allow_interp: bool | None = None,
                       reset_time: bool | None = None) -> RestoreInfo:
    """
    Загрузить состояние в свежесобранную `sim` и подготовить её к `run()`
    вместо преднагрузки. КОЛЛЕКТИВНАЯ операция.

    source       : путь к .npz или уже прочитанный CheckpointData
    allow_interp : разрешить ближайшего соседа при несовпадении сеток
                   (по умолчанию — из config.restart)
    reset_time   : начать отсчёт с нуля (по умолчанию — из config.restart)

    Возвращает RestoreInfo; `info.t_start_ms` передаётся в `sim.run()`.
    """
    if sim.is_preloaded:
        # Проверка до загрузки: иначе состояние готовой симуляции было бы
        # испорчено, и только потом resume() отказался бы работать.
        raise RuntimeError(
            "состояние уже подготовлено (preload или resume) — "
            "восстанавливать чекпоинт нужно в свежесобранную Simulation")

    rc = sim.config.restart
    allow_interp = rc.allow_interp if allow_interp is None else allow_interp
    reset_time = rc.reset_time if reset_time is None else reset_time

    data = (source if isinstance(source, CheckpointData)
            else read_checkpoint(source, sim.comm))

    # ── модель клетки должна совпадать: иначе состояние бессмысленно ──
    cell = sim.cell_model
    if data.cell_model != cell.name or data.state_names != tuple(cell.state_names):
        raise ValueError(
            f"чекпоинт записан для модели клетки {data.cell_model!r} "
            f"(переменные {data.state_names}), текущая — {cell.name!r} "
            f"({tuple(cell.state_names)}). Переносить состояние между "
            f"разными моделями клетки нельзя")
    _check_geometry(data, sim)

    warnings: list[str] = []
    if data.legacy:
        warnings.append(
            "чекпоинт монолитной версии: в нём нет конфигурации, поэтому "
            "расхождения параметров с текущей конфигурацией не проверены")
    elif data.config_dict is not None:
        diffs = _config_differences(data.config_dict,
                                    sim.config.to_dict(skip_unserializable=True))
        if diffs:
            warnings.append(
                "физика текущей конфигурации отличается от прогона-источника "
                "(загруженное u будет приведено к равновесию для текущей): "
                + "; ".join(diffs))

    # ── сопоставление по координатам ──────────────────────────────────
    e_spec, m_spec = sim.config.mesh.electric, sim.config.mesh.mechanical
    tol_e = 1e-6 * min(e_spec.hx_mm, e_spec.hy_mm)
    tol_m = 1e-6 * min(m_spec.hx_mm, m_spec.hy_mm)

    def _match(ref, V, tol, label):
        try:
            idx, n_near = match_points(ref, local_coordinates(V), tol,
                                       allow_interp, label)
            err = None
        except ValueError as exc:
            idx, n_near, err = None, 0, exc
        # ошибка на одном ранге — ошибка на всех, иначе остальные зависнут
        errors = sim.comm.allgather(err)
        first = next((e for e in errors if e is not None), None)
        if first is not None:
            raise first
        return idx, sim.comm.allreduce(n_near)

    idx_e, near_e = _match(data.e_coords, sim.electrics.P1, tol_e, "клетки")
    idx_m, near_m = _match(data.m_coords, sim.mechanics.V, tol_m, "перемещения")

    sim.electrics.load_state(data.e_state[idx_e])
    u = sim.mechanics.u
    u.x.array[:] = data.m_u[idx_m].reshape(-1)
    u.x.scatter_forward()

    nearest = {}
    if near_e:
        nearest["electric"] = near_e
    if near_m:
        nearest["mechanical"] = near_m
    if nearest:
        warnings.append(
            f"сетки не совпадают с чекпоинтом: по ближайшему соседу взято "
            f"{near_e} эл. и {near_m} мех. узлов — состояние приближённое")

    t_start = 0.0 if reset_time else data.t_ms
    sim.resume(t_start)

    info = RestoreInfo(source=data.path, stage=data.stage,
                       t_checkpoint_ms=data.t_ms, t_start_ms=t_start,
                       nearest_nodes=nearest, warnings=warnings)
    sim.restart_info = info.to_dict()
    sim.warnings.extend(warnings)
    return info


# ═══════════════════════════════════════════════════════════════════════
#  НАБЛЮДАТЕЛЬ
# ═══════════════════════════════════════════════════════════════════════

class CheckpointObserver(Observer):
    """
    Пишет чекпоинты по расписанию (`Tick.checkpoint`) и один раз после
    преднагрузки.

        ckpt_preloaded.npz      — сразу после преднагрузки: общая
                                  стартовая точка для серии экспериментов
        ckpt_last.npz           — последнее сохранённое состояние
        ckpt_t0000123.000ms.npz — при keep_all: каждый чекпоинт отдельно
    """

    def __init__(self, out_dir, keep_all: bool = False):
        self.out_dir = Path(out_dir)
        self.keep_all = keep_all
        self.written: list[tuple[float, Path]] = []

    @property
    def last_path(self) -> Path:
        return self.out_dir / "ckpt_last.npz"

    def on_preload_done(self, sim):
        path = save_checkpoint(sim, self.out_dir / "ckpt_preloaded.npz",
                               stage="preloaded", mech_index=0)
        self.written.append((sim.t_ms, path))

    def on_mech_tick(self, sim, tick):
        if not tick.checkpoint:
            return
        if self.keep_all:
            path = self.out_dir / f"ckpt_t{tick.t_end_ms:012.3f}ms.npz"
        else:
            path = self.last_path
        save_checkpoint(sim, path, stage="running", mech_index=tick.mech_index)
        if self.keep_all and sim.comm.rank == 0:
            tmp = self.last_path.with_name(self.last_path.name + ".tmp")
            shutil.copyfile(path, tmp)
            os.replace(tmp, self.last_path)
        self.written.append((tick.t_end_ms, path))
