"""
Управление прогонами: как их запускают.
========================================

    overrides.py — изменение параметров по пути «tissue_base.active.t_max»
    api.py       — run_simulation(config): прогон целиком одной функцией
    sweep.py     — параметрические серии (SweepSpec, run_sweep)
    cli.py       — командная строка: python -m cardiac_em …

Разбор конфигураций, серий и команды init/show/status работают без
DOLFINx; сам счёт (run, sweep) — с ним. Будущий интерфейс управления
строится поверх `run_simulation` и `SweepSpec`, а за ходом счёта следит
по run.json / sweep.json.
"""

from __future__ import annotations

from .api import RunResult, run_simulation
from .overrides import apply_overrides, parse_assignment, parse_value, set_path
from .sweep import SWEEP_INDEX, SweepPoint, SweepSpec, run_sweep

__all__ = [
    "run_simulation",
    "RunResult",
    "apply_overrides",
    "parse_assignment",
    "parse_value",
    "set_path",
    "SweepSpec",
    "SweepPoint",
    "run_sweep",
    "SWEEP_INDEX",
]
