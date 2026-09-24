"""
Стандартный набор вывода одним вызовом.
========================================

    sim = Simulation(config, observers=[ConsoleObserver()])
    attach_outputs(sim)          # до preload(): чтобы попали preload.csv
    sim.preload()                # и ckpt_preloaded.npz
    sim.run()

Что появится в `config.output.out_dir`:

    run.json                 — манифест (io/manifest.py)
    series.csv, preload.csv  — диагностика по времени (io/series.py)
    electrics.xdmf/.h5, mechanics.xdmf/.h5, regions_*.xdmf
                             — поля для ParaView (io/fields.py)
    snapshots/snap_t…ms.npz  — снимки для анализа (io/fields.py)
    activation.npz           — карты активации и APD (io/activation.py)
    ckpt_preloaded.npz, ckpt_last.npz [, ckpt_t…ms.npz]
                             — чекпоинты, если включены (io/checkpoint.py)

Защита от затирания: если в папке уже есть run.json, это ошибка —
прежний прогон не пропадёт из-за того, что забыли сменить out_dir.
Перезапись — явным `overwrite=True`.
"""

from __future__ import annotations

from pathlib import Path

from ..runtime.observers import Observer
from .activation import ActivationRecorder
from .manifest import MANIFEST_NAME, ManifestObserver
from .series import SeriesWriter

__all__ = ["attach_outputs"]


def attach_outputs(sim, *, overwrite: bool = False,
                   series_every_mech_steps: int = 1) -> list[Observer]:
    """
    Подключить к `sim` запись манифеста, рядов, полей и чекпоинтов по
    `sim.config.output`. КОЛЛЕКТИВНАЯ операция. Возвращает список
    подключённых наблюдателей (в том же порядке, в каком они вызываются).
    """
    out = sim.config.output
    out_dir = Path(out.out_dir)

    exists = (out_dir / MANIFEST_NAME).exists() if sim.comm.rank == 0 else None
    exists = sim.comm.bcast(exists, root=0)
    if exists and not overwrite:
        raise FileExistsError(
            f"в {out_dir} уже есть {MANIFEST_NAME} от другого прогона. "
            f"Укажите другую папку (config.output.out_dir) или "
            f"attach_outputs(sim, overwrite=True)")

    # Импорт здесь, а не наверху: запись полей требует DOLFINx, а
    # манифест и ряды — нет (см. io/__init__.py).
    from .checkpoint import CheckpointObserver
    from .fields import FieldWriter

    observers: list[Observer] = [
        SeriesWriter(out_dir, every_mech_steps=series_every_mech_steps),
        FieldWriter(out_dir, write_region_maps=out.write_region_maps),
    ]
    if out.checkpoints_enabled:
        observers.append(CheckpointObserver(out_dir, keep_all=out.ckpt_keep_all))
    if out.record_activation:
        observers.append(ActivationRecorder(out_dir, apd_level=out.apd_level))
    # Манифест — ПОСЛЕДНИМ: при завершении он перечисляет файлы папки, и
    # к этому моменту остальные наблюдатели должны свои файлы дописать
    # (карты активации пишутся именно в on_finish).
    observers.append(ManifestObserver(out_dir))

    for obs in observers:
        sim.add_observer(obs)
    return observers
