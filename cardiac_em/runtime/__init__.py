"""
Ход расчёта: расписание, наблюдатели, связанная задача.
========================================================

    schedule.py    — какие события на каком шаге (без DOLFINx)
    observers.py   — наблюдатели за ходом счёта (без DOLFINx)
    simulation.py  — сборка и ведение связанной задачи (DOLFINx)

`Simulation` импортируется ЛЕНИВО. Причина: импорт любого подмодуля
сначала выполняет этот `__init__.py`, и если бы здесь стоял обычный
`from .simulation import Simulation`, то даже `cardiac_em.runtime.schedule`
нельзя было бы загрузить без DOLFINx. А расписание полезно проверять и
использовать отдельно — например, чтобы заранее посчитать, сколько шагов
и записей даст конфигурация.
"""

from __future__ import annotations

from .observers import ConsoleObserver, Observer, TraceObserver
from .schedule import Schedule, Tick

__all__ = [
    "Schedule",
    "Tick",
    "Observer",
    "ConsoleObserver",
    "TraceObserver",
    "Simulation",
]


def __getattr__(name: str):
    if name == "Simulation":
        from .simulation import Simulation
        return Simulation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
