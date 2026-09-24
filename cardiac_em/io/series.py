"""
Временные ряды диагностики в CSV.
==================================

    series.csv   — строка на каждый механический шаг (и начальная)
    preload.csv  — строка на каждый шаг преднагрузки

Столбцы — ключи `Simulation.diagnostics()`: t_ms, mech_index, u_min/max,
T_act на обеих сетках (максимум и интеграл), λ_f, J, σ_xx, итерации
Ньютона. Файл дописывается и сбрасывается на диск после каждой строки,
так что за идущим счётом можно следить (`tail -f`), а при аварии
история до сбоя остаётся.

Читается чем угодно: `numpy.genfromtxt(..., names=True, delimiter=",")`
или pandas.
"""

from __future__ import annotations

from pathlib import Path

from ..runtime.observers import Observer

__all__ = ["SeriesWriter", "SERIES_COLUMNS"]

SERIES_COLUMNS = (
    "t_ms", "mech_index",
    "u_min", "u_max",
    "t_act_electric_max", "t_act_electric_integral",
    "t_act_mech_max", "t_act_mech_integral",
    "lambda_f_min", "lambda_f_max",
    "J_min", "J_max",
    "sigma_xx_min", "sigma_xx_max",
    "newton_iterations",
)

_PRELOAD_COLUMNS = ("step", "lambda_f_min", "lambda_f_max", "J_min", "J_max",
                    "sigma_xx_min", "sigma_xx_max", "newton_iterations")


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v)                # полная точность, без потерь при чтении
    return str(v)


class SeriesWriter(Observer):
    """
    Диагностика в CSV. Диагностика — коллективная операция, поэтому
    считается на всех рангах, а пишет только нулевой.

    every_mech_steps — писать каждый n-й механический шаг (последний
    пишется всегда).
    """

    def __init__(self, out_dir, every_mech_steps: int = 1):
        if every_mech_steps < 1:
            raise ValueError("every_mech_steps должен быть >= 1")
        self.out_dir = Path(out_dir)
        self.every = every_mech_steps
        self._fh = None
        self._fh_pre = None

    @property
    def path(self) -> Path:
        return self.out_dir / "series.csv"

    @staticmethod
    def _open(path: Path, columns):
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "w", encoding="utf-8")
        fh.write(",".join(columns) + "\n")
        fh.flush()
        return fh

    @staticmethod
    def _row(fh, columns, d) -> None:
        fh.write(",".join(_fmt(d.get(c)) for c in columns) + "\n")
        fh.flush()

    # ── преднагрузка ──────────────────────────────────────────────────
    def on_preload_step(self, sim, k, n):
        d = sim.diagnostics(electric=False)
        d["step"] = k
        if sim.comm.rank == 0:
            if self._fh_pre is None:
                self._fh_pre = self._open(self.out_dir / "preload.csv",
                                          _PRELOAD_COLUMNS)
            self._row(self._fh_pre, _PRELOAD_COLUMNS, d)

    def on_preload_done(self, sim):
        if self._fh_pre is not None:
            self._fh_pre.close()
            self._fh_pre = None

    # ── активация ─────────────────────────────────────────────────────
    def on_start(self, sim, schedule):
        d = sim.diagnostics()
        d["mech_index"] = None
        if sim.comm.rank == 0:
            self._fh = self._open(self.path, SERIES_COLUMNS)
            self._row(self._fh, SERIES_COLUMNS, d)

    def on_mech_tick(self, sim, tick):
        if not (tick.mech_index % self.every == 0 or tick.is_last):
            return
        d = sim.diagnostics()
        d["mech_index"] = tick.mech_index
        if sim.comm.rank == 0:
            self._row(self._fh, SERIES_COLUMNS, d)

    def on_finish(self, sim):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def on_abort(self, sim, exc):
        self.on_finish(sim)
