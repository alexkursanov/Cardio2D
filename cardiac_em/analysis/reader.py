"""
Чтение результатов: прогон и серия.
====================================

    from cardiac_em.analysis import open_run, open_sweep

    run = open_run("runs/baseline")          # папка или путь к run.json
    run.status, run.config["tissue_base"]    # из манифеста
    s = run.series()                         # series.csv → структурный массив
    s["t_ms"], s["sigma_xx_max"]
    snap = run.snapshot(300.0)               # ближайший снимок
    V = snap.grid("e_state", "u")            # (ny+1, nx+1) — карта потенциала
    maps = run.activation()                  # карты активации и APD
    maps.grid(maps.act[:, 0])                # время активации 1-го удара

    sweep = open_sweep("runs/tmax_sweep")
    for point, run in sweep.runs(): ...

Только numpy (pandas — по желанию, `series(as_frame=True)`). Никаких
импортов расчётных слоёв: анализ работает на ноутбуке с одними файлами.
Пути берутся из манифеста и соглашений io/, а не угадываются.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

__all__ = ["Run", "Snapshot", "ActivationMaps", "Sweep", "open_run", "open_sweep"]

_SNAP_RE = re.compile(r"snap_t(-?\d+(?:\.\d+)?)ms\.npz$")


def _grid_shape(mesh: dict, kind: str) -> tuple[int, int]:
    """Форма двумерной карты: узлы (ny+1, nx+1) или ячейки (ny, nx)."""
    nx, ny = int(mesh["nx"]), int(mesh["ny"])
    if kind == "nodes":
        return ny + 1, nx + 1
    if mesh.get("cell_type", "quadrilateral") != "quadrilateral":
        raise ValueError("карта ячеек как (ny, nx) есть только у "
                         "четырёхугольной сетки; у треугольной ячеек вдвое больше")
    return ny, nx


# ═══════════════════════════════════════════════════════════════════════
#  СНИМОК
# ═══════════════════════════════════════════════════════════════════════

class Snapshot:
    """
    Снимок полей в момент t (snapshots/snap_t…ms.npz). Массивы — по
    ключам (`snap["m_lambda_f"]`), список ключей — `snap.keys()`.

    Префикс e_ — электрическая сетка, m_ — механическая; *_coords —
    узлы, *_cell_coords — ячейки.
    """

    def __init__(self, path: Path, meshes: dict | None = None):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as z:
            self._data = {k: z[k] for k in z.files}
        self.meshes = meshes or {}
        self.t_ms = float(self._data["t_ms"])
        self.state_names = tuple(str(s) for s in self._data.get("e_state_names", ()))

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def state(self, name: str) -> np.ndarray:
        """Переменная клетки по имени, на узлах электрической сетки."""
        if name not in self.state_names:
            raise KeyError(f"нет переменной {name!r}; есть {self.state_names}")
        return self._data["e_state"][:, self.state_names.index(name)]

    def grid(self, key: str, component=None) -> np.ndarray:
        """
        Поле как двумерная карта (строки — y, столбцы — x).

            snap.grid("e_state", "u")     — переменная клетки
            snap.grid("m_u", 0)           — x-компонента перемещения
            snap.grid("m_lambda_f")       — по ячейкам механики
        """
        if key == "e_state" and isinstance(component, str):
            arr = self.state(component)
        else:
            arr = self._data[key]
            if component is not None:
                arr = arr[:, component]
        mesh_part = "electric" if key.startswith("e_") else "mechanical"
        cell_keys = {"e_t_act", "m_t_act", "m_lambda_f", "m_J", "m_sigma_xx", "m_region"}
        kind = "cells" if key in cell_keys else "nodes"
        if mesh_part not in self.meshes:
            raise ValueError("нет описания сетки (манифест не прочитан) — "
                             "откройте снимок через Run.snapshot()")
        return np.asarray(arr).reshape(_grid_shape(self.meshes[mesh_part], kind))


# ═══════════════════════════════════════════════════════════════════════
#  КАРТЫ АКТИВАЦИИ
# ═══════════════════════════════════════════════════════════════════════

class ActivationMaps:
    """
    activation.npz: по узлам электрической сетки и по ударам.

        act, repol, peak : (N, B), NaN — события не было
        apd              : repol − act
        coords (N, 2), region (N,)
    """

    def __init__(self, path: Path, mesh: dict | None = None):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        self.coords = d["coords"]
        self.region = d["region"]
        self.act = d["act"]
        self.repol = d["repol"]
        self.peak = d["peak"]
        self.threshold = float(d["threshold"])
        self.apd_level = float(d["apd_level"])
        self.v_rest = float(d["v_rest"])
        self.t_start_ms = float(d["t_start_ms"])
        self.t_end_ms = float(d["t_end_ms"])
        self.dt_ms = float(d["dt_ms"])
        self.mesh = mesh

    @property
    def n_beats(self) -> int:
        return self.act.shape[1]

    @property
    def apd(self) -> np.ndarray:
        return self.repol - self.act

    def grid(self, values: np.ndarray) -> np.ndarray:
        """Значения по узлам → карта (ny+1, nx+1)."""
        if self.mesh is None:
            raise ValueError("нет описания сетки — откройте через Run.activation()")
        return np.asarray(values).reshape(_grid_shape(self.mesh, "nodes"))


# ═══════════════════════════════════════════════════════════════════════
#  ПРОГОН
# ═══════════════════════════════════════════════════════════════════════

class Run:
    """Один прогон — папка с run.json."""

    def __init__(self, path):
        path = Path(path)
        if path.is_file():
            path = path.parent
        manifest = path / "run.json"
        if not manifest.exists():
            raise FileNotFoundError(f"в {path} нет run.json — это не папка прогона")
        self.dir = path
        self.manifest = json.loads(manifest.read_text(encoding="utf-8"))

    def __repr__(self) -> str:
        return f"Run({str(self.dir)!r}, status={self.status!r})"

    # ── из манифеста ──────────────────────────────────────────────────
    @property
    def status(self) -> str:
        return self.manifest.get("status", "unknown")

    @property
    def config(self) -> dict:
        return self.manifest["config"]

    @property
    def meshes(self) -> dict:
        return self.manifest.get("meshes") or self.config["mesh"]

    @property
    def warnings(self) -> list[str]:
        return list(self.manifest.get("warnings", []))

    def param(self, path: str):
        """Параметр конфигурации по пути: run.param('tissue_base.active.t_max')."""
        node = self.config
        for key in path.split("."):
            node = node[int(key)] if isinstance(node, list) else node[key]
        return node

    # ── ряды ──────────────────────────────────────────────────────────
    def _csv(self, name: str, as_frame: bool):
        path = self.dir / name
        if not path.exists():
            raise FileNotFoundError(f"нет {path}")
        if as_frame:
            import pandas as pd
            return pd.read_csv(path)
        data = np.genfromtxt(path, delimiter=",", names=True)
        return np.atleast_1d(data)

    def series(self, as_frame: bool = False):
        """series.csv: структурный массив (или pandas.DataFrame)."""
        return self._csv("series.csv", as_frame)

    def preload_series(self, as_frame: bool = False):
        return self._csv("preload.csv", as_frame)

    # ── снимки ────────────────────────────────────────────────────────
    @property
    def snapshot_times(self) -> list[float]:
        folder = self.dir / "snapshots"
        if not folder.exists():
            return []
        times = []
        for p in folder.iterdir():
            m = _SNAP_RE.search(p.name)
            if m:
                times.append(float(m.group(1)))
        return sorted(times)

    def snapshot(self, t_ms: float, tol_ms: float = 1e-6) -> Snapshot:
        """Снимок ровно на момент t (с допуском tol_ms)."""
        for t in self.snapshot_times:
            if abs(t - t_ms) <= tol_ms:
                return Snapshot(self.dir / "snapshots" / f"snap_t{t:012.3f}ms.npz",
                                self.meshes)
        raise KeyError(f"нет снимка на t = {t_ms} мс; есть {self.snapshot_times}")

    def snapshots(self):
        """Все снимки по порядку времени."""
        for t in self.snapshot_times:
            yield self.snapshot(t)

    # ── карты активации ───────────────────────────────────────────────
    def activation(self) -> ActivationMaps:
        path = self.dir / "activation.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"нет {path}: прогон без записи активации "
                f"(output.record_activation) или не завершён")
        return ActivationMaps(path, self.meshes.get("electric"))

    @property
    def checkpoints(self) -> list[Path]:
        return sorted(self.dir.glob("ckpt_*.npz"))


# ═══════════════════════════════════════════════════════════════════════
#  СЕРИЯ
# ═══════════════════════════════════════════════════════════════════════

class Sweep:
    """Серия — папка с sweep.json."""

    def __init__(self, path):
        path = Path(path)
        if path.is_file():
            path = path.parent
        index = path / "sweep.json"
        if not index.exists():
            raise FileNotFoundError(f"в {path} нет sweep.json — это не папка серии")
        self.dir = path
        self.index = json.loads(index.read_text(encoding="utf-8"))

    @property
    def parameters(self) -> dict:
        return self.index["parameters"]

    @property
    def points(self) -> list[dict]:
        return self.index["points"]

    def _run_dir(self, point: dict) -> Path:
        d = Path(point["out_dir"])
        # пути в сводке — как их задали при запуске; если папку серии
        # перенесли, ищем прогон рядом со сводкой
        return d if (d / "run.json").exists() else self.dir / point["name"]

    def runs(self, only_finished: bool = True):
        """Пары (точка, Run) — по умолчанию только успешно завершённые."""
        for p in self.points:
            if only_finished and p["status"] != "finished":
                continue
            d = self._run_dir(p)
            if (d / "run.json").exists():
                yield p, Run(d)

    def table(self, metrics, only_finished: bool = True) -> list[dict]:
        """
        Сводная таблица: по строке на точку — её параметры и значения
        `metrics(run) -> dict`. Удобно передать в pandas.DataFrame.
        """
        rows = []
        for p, run in self.runs(only_finished):
            row = {"name": p["name"], **p["overrides"]}
            row.update(metrics(run))
            rows.append(row)
        return rows


def open_run(path) -> Run:
    return Run(path)


def open_sweep(path) -> Sweep:
    return Sweep(path)
