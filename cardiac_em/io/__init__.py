"""
Ввод-вывод: манифест, временные ряды, поля, чекпоинты.
=======================================================

    manifest.py    — run.json: контракт с анализом (без DOLFINx)
    series.py      — series.csv / preload.csv (без DOLFINx)
    fields.py      — XDMF для ParaView и .npz-снимки для анализа
    checkpoint.py  — сохранение и продолжение расчёта
    outputs.py     — attach_outputs(sim): всё вышеперечисленное разом

Всё, кроме `restore_checkpoint`, — наблюдатели (runtime/observers.py):
расчётный цикл о файлах ничего не знает.

Модули, которым нужен DOLFINx, импортируются ЛЕНИВО (как Simulation в
runtime/): манифест и ряды можно использовать и проверять без
расчётного стека.
"""

from __future__ import annotations

from .manifest import MANIFEST_NAME, ManifestObserver, collect_environment
from .outputs import attach_outputs
from .series import SERIES_COLUMNS, SeriesWriter

__all__ = [
    "MANIFEST_NAME",
    "ManifestObserver",
    "collect_environment",
    "SeriesWriter",
    "SERIES_COLUMNS",
    "attach_outputs",
    # ленивые (DOLFINx):
    "FieldWriter",
    "write_snapshot",
    "CheckpointObserver",
    "CheckpointData",
    "RestoreInfo",
    "save_checkpoint",
    "pack_checkpoint",
    "read_checkpoint",
    "restore_checkpoint",
]

_LAZY = {
    "FieldWriter": "fields",
    "write_snapshot": "fields",
    "CheckpointObserver": "checkpoint",
    "CheckpointData": "checkpoint",
    "RestoreInfo": "checkpoint",
    "save_checkpoint": "checkpoint",
    "pack_checkpoint": "checkpoint",
    "read_checkpoint": "checkpoint",
    "restore_checkpoint": "checkpoint",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f".{module}", __name__), name)
