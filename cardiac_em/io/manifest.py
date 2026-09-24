"""
Манифест прогона — run.json.
=============================

Контракт между расчётом и анализом (раздел 7 ARCHITECTURE.md): всё, что
нужно, чтобы понять и воспроизвести прогон, и где лежат его файлы.
`analysis/` читает манифест, а не угадывает пути.

Содержимое
----------
    format, version        — "cardiac_em.run", 1
    status                 — "running" | "finished" | "failed"
    started, finished      — время UTC (ISO 8601); elapsed_s
    progress               — {t_ms, mech_index} последнего шага; во время
                             счёта обновляется не чаще раза в
                             `progress_interval_s` секунд — это же и
                             статус-файл для внешнего наблюдения
    config                 — полная конфигурация (SimulationConfig.to_dict)
    unserializable_regions — имена областей, не попавших в config
    models                 — модель клетки (и её переменные), материал,
                             перенос T_act
    meshes                 — число ячеек и вершин обеих сеток
    schedule               — t_start, t_end, число шагов
    restart                — откуда взято состояние (или null)
    warnings               — предупреждения, собранные до и во время счёта
    environment            — версии Python/numpy/DOLFINx/PETSc, число
                             рангов, хост, git-коммит кода
    outputs                — перечень файлов в папке вывода (при завершении)
    error                  — текст ошибки, если status = "failed"

Файл пишет только нулевой ранг, запись атомарная.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..runtime.observers import Observer

__all__ = ["ManifestObserver", "collect_environment", "write_json_atomic",
           "MANIFEST_NAME"]

FORMAT = "cardiac_em.run"
VERSION = 1
MANIFEST_NAME = "run.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _git_info() -> dict:
    """Коммит и «грязность» рабочей копии кода; пусто, если не git."""
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=5)
        if head.returncode != 0:
            return {}
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                 "--untracked-files=no"],
                                capture_output=True, text=True, timeout=5)
        return {"commit": head.stdout.strip(),
                "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {}


def _version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
    except ImportError:
        return None
    return str(getattr(mod, "__version__", None))


def collect_environment(comm=None) -> dict:
    """Версии и окружение — чтобы через год понять, чем считали."""
    from .. import __version__

    env = {
        "cardiac_em": __version__,
        "python": sys.version.split()[0],
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "dolfinx": _version("dolfinx"),
        "petsc4py": _version("petsc4py"),
        "mpi_size": comm.size if comm is not None else 1,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "git": _git_info(),
    }
    return env


def _list_outputs(out_dir: Path) -> list[str]:
    files = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and not p.name.endswith(".tmp") and p.name != MANIFEST_NAME:
            files.append(p.relative_to(out_dir).as_posix())
    return files


class ManifestObserver(Observer):
    """
    Ведёт run.json на протяжении прогона. Пишет только ранг 0; сам не
    делает коллективных операций, поэтому безопасен и в `on_abort`.
    """

    def __init__(self, out_dir, progress_interval_s: float = 2.0):
        self.out_dir = Path(out_dir)
        self.progress_interval_s = progress_interval_s
        self.data: dict = {}
        self._t0 = None
        self._last_write = 0.0

    @property
    def path(self) -> Path:
        return self.out_dir / MANIFEST_NAME

    def _write(self, sim, force: bool = True) -> None:
        if sim.comm.rank != 0:
            return
        now = time.monotonic()
        if not force and now - self._last_write < self.progress_interval_s:
            return
        write_json_atomic(self.path, self.data)
        self._last_write = now

    # ── события ───────────────────────────────────────────────────────
    def on_start(self, sim, schedule):
        cfg = sim.config
        self._t0 = time.perf_counter()
        self.data = {
            "format": FORMAT,
            "version": VERSION,
            "status": "running",
            "started": _now(),
            "finished": None,
            "elapsed_s": None,
            "progress": {"t_ms": schedule.t_start_ms, "mech_index": None},
            "config": cfg.to_dict(skip_unserializable=True),
            "unserializable_regions": [r.name or type(r).__name__
                                       for r in cfg.unserializable_regions()],
            "models": {
                "cell": sim.cell_model.name,
                "cell_states": list(sim.cell_model.state_names),
                "cell_description": sim.cell_model.describe(),
                "material": sim.mechanics.material.name,
                "transfer": sim.transfer.describe(),
                "transfer_exact": bool(sim.transfer.exact),
            },
            "meshes": {
                "electric": {"n_cells": sim.pair.n_cells_electric,
                             **cfg.mesh.electric.to_dict()},
                "mechanical": {"n_cells": sim.pair.n_cells_mechanical,
                               **cfg.mesh.mechanical.to_dict()},
            },
            "schedule": {"t_start_ms": schedule.t_start_ms,
                         "t_end_ms": schedule.t_end_ms,
                         "n_electric_steps": len(schedule),
                         "n_mech_steps": schedule.n_mech_ticks,
                         "dt_electric_ms": schedule.dt,
                         "unreachable_snapshots_ms": list(schedule.unreachable_snapshots)},
            "restart": sim.restart_info,
            "warnings": list(sim.warnings),
            "environment": collect_environment(sim.comm),
            "outputs": [],
            "error": None,
        }
        self._write(sim)

    def on_mech_tick(self, sim, tick):
        self.data["progress"] = {"t_ms": tick.t_end_ms,
                                 "mech_index": tick.mech_index}
        self._write(sim, force=tick.is_last)

    def _finish(self, sim, status: str, error: str | None = None) -> None:
        self.data["status"] = status
        self.data["finished"] = _now()
        if self._t0 is not None:
            self.data["elapsed_s"] = round(time.perf_counter() - self._t0, 3)
        self.data["progress"] = {"t_ms": sim.t_ms,
                                 "mech_index": self.data["progress"].get("mech_index")}
        self.data["warnings"] = list(sim.warnings)
        self.data["error"] = error
        if sim.comm.rank == 0:
            self.data["outputs"] = _list_outputs(self.out_dir)
        self._write(sim)

    def on_finish(self, sim):
        self._finish(sim, "finished")

    def on_abort(self, sim, exc):
        if self.data:
            self._finish(sim, "failed", f"{type(exc).__name__}: {exc}")
