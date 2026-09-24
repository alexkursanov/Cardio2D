"""
Анализ результатов — только по файлам, без DOLFINx.
====================================================

    reader.py      — open_run / open_sweep: манифест, ряды, снимки, карты
    biomarkers.py  — скорость проведения, APD, блок, реституция, механика
    plots.py       — быстрые графики (matplotlib по желанию)

Работает в лёгком окружении (`pip install -e ".[analysis]"`): на
ноутбуке, в Jupyter, на скопированной с сервера папке прогона. Слой не
импортирует расчётные слои — это проверяет tests/test_architecture.py.

Пример: скорость проведения и APD первого удара

    from cardiac_em.analysis import open_run, biomarkers as bm
    run = open_run("runs/a")
    maps = run.activation()
    cv = bm.conduction_velocity(maps.coords, maps.act[:, 0])
    apd = bm.apd_summary(maps.apd[:, 0], maps.region)
"""

from __future__ import annotations

from . import biomarkers, plots
from .reader import ActivationMaps, Run, Snapshot, Sweep, open_run, open_sweep

__all__ = ["open_run", "open_sweep", "Run", "Sweep", "Snapshot", "ActivationMaps",
           "biomarkers", "plots"]
