"""
Общие помощники ноутбуков notebooks/*.ipynb.
=============================================

    from nbtools import ROOT, RUNS, run_stream, pytest_run, list_runs, style

* ROOT, RUNS          — корень репозитория и папка прогонов runs/
* run_stream(cmd)     — запустить команду (mpirun, CLI, pytest) и показывать
                        вывод по мере появления; вернуть код выхода
* pytest_run(args)    — pytest в отдельном процессе (при mpi=N — под mpirun)
                        со сводкой: сколько прошло/упало, время
* list_runs()         — прогоны в runs/ с их статусом
* style               — цвета и шкалы для графиков (одна система во всех
                        ноутбуках)

Модуль не требует DOLFINx.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _find_root() -> Path:
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").exists() and (p / "cardiac_em").is_dir():
            return p
    raise RuntimeError("не найден корень репозитория (pyproject.toml + cardiac_em/) — "
                       "откройте ноутбук из папки проекта")


ROOT = _find_root()
RUNS = ROOT / "runs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════
#  ЗАПУСК КОМАНД
# ═══════════════════════════════════════════════════════════════════════

def run_stream(cmd, cwd=ROOT, quiet: bool = False, tail: int = 0) -> int:
    """
    Запустить команду и печатать вывод построчно, пока она работает.
    quiet — не печатать, tail — напечатать только последние N строк.
    Возвращает код выхода; сам вывод — в run_stream.last_output.
    """
    cmd = [str(c) for c in cmd]
    if not quiet:
        print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if not quiet and not tail:
            print(line, end="", flush=True)
    code = proc.wait()
    if tail and not quiet:
        print("".join(lines[-tail:]), end="")
    run_stream.last_output = "".join(lines)
    if not quiet:
        print(f"[код выхода {code}]")
    return code


run_stream.last_output = ""


def mpirun(required: bool = True) -> str:
    """Путь к mpirun/mpiexec. required=False — вернуть "mpirun", даже если
    его нет (для показа команды, которую не собираемся запускать)."""
    exe = shutil.which("mpirun") or shutil.which("mpiexec")
    if exe is None and not required:
        return "mpirun"
    if exe is None:
        raise RuntimeError("mpirun/mpiexec не найден в PATH окружения ядра")
    return exe


_SUMMARY = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)")


def pytest_run(args, mpi: int = 0, label: str | None = None, tail: int = 15) -> dict:
    """
    pytest в отдельном процессе: тесты не делят память с ядром ноутбука,
    а при mpi > 0 идут под mpirun, как в терминале. Возвращает сводку:
    {"набор", "код", "passed", "failed", "errors", "skipped", "секунд"}.
    """
    args = [str(a) for a in args]
    base = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args]
    cmd = [mpirun(), "-n", str(mpi), *base] if mpi else base
    t0 = time.perf_counter()
    code = run_stream(cmd, tail=tail)
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    last = [ln for ln in run_stream.last_output.splitlines() if _SUMMARY.search(ln)]
    if last:
        for n, kind in _SUMMARY.findall(last[-1]):
            key = "errors" if kind.startswith("error") else kind
            if key in counts:
                counts[key] = int(n)
    return {"набор": label or " ".join(args) + (f" (MPI ×{mpi})" if mpi else ""),
            "код": code, **counts, "секунд": round(time.perf_counter() - t0, 1)}


# ═══════════════════════════════════════════════════════════════════════
#  ПРОГОНЫ
# ═══════════════════════════════════════════════════════════════════════

def list_runs(root: Path = RUNS) -> list[dict]:
    """Прогоны и серии в runs/: имя, тип, статус, модель, время."""
    rows = []
    if not root.exists():
        return rows
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "run.json").exists():
            m = json.loads((d / "run.json").read_text(encoding="utf-8"))
            rows.append({"папка": d.name, "тип": "прогон", "статус": m.get("status"),
                         "модель": m.get("models", {}).get("cell"),
                         "t, мс": m.get("progress", {}).get("t_ms"),
                         "счёт, с": m.get("elapsed_s")})
        elif (d / "sweep.json").exists():
            s = json.loads((d / "sweep.json").read_text(encoding="utf-8"))
            pts = s.get("points", [])
            ok = sum(p["status"] == "finished" for p in pts)
            rows.append({"папка": d.name, "тип": "серия", "статус": f"{ok}/{len(pts)} готово",
                         "модель": s.get("base", {}).get("cell_model"),
                         "t, мс": None, "счёт, с": None})
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  СТИЛЬ ГРАФИКОВ
# ═══════════════════════════════════════════════════════════════════════

class _Style:
    """
    Одна система цветов для всех ноутбуков.

    * series      — категориальные цвета в ФИКСИРОВАННОМ порядке (проверены
                    на различимость при цветовой слепоте; у двух последних
                    контраст с фоном ниже 3:1 — поэтому на графиках с
                    несколькими линиями всегда есть легенда)
    * seq         — одноцветная шкала для величин (светлое → тёмное)
    * div         — расходящаяся шкала синий ↔ красный с серой серединой
                    (для величин со знаком)
    * time        — шкала для времени (карты активации)
    """

    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    surface = "#fcfcfb"
    ink = "#2b2b2a"
    muted = "#8a8984"
    grid = "#e4e3df"

    def __init__(self):
        self._cmaps = None

    def _build(self):
        from matplotlib.colors import LinearSegmentedColormap
        blue = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
        red = ["#fbd9d9", "#f3a8a7", "#e97877", "#e34948", "#c23433", "#98201f"]
        self._cmaps = {
            "seq": LinearSegmentedColormap.from_list("cem_seq", blue),
            "div": LinearSegmentedColormap.from_list(
                "cem_div", list(reversed(blue[1:])) + ["#f0efec"] + red[1:]),
            "time": LinearSegmentedColormap.from_list("cem_time", blue[1:] + ["#0a2748"]),
        }

    @property
    def seq(self):
        if self._cmaps is None:
            self._build()
        return self._cmaps["seq"]

    @property
    def div(self):
        if self._cmaps is None:
            self._build()
        return self._cmaps["div"]

    @property
    def time(self):
        if self._cmaps is None:
            self._build()
        return self._cmaps["time"]

    def apply(self):
        """Настроить matplotlib: тонкие линии, спокойные оси и сетка."""
        import matplotlib as mpl
        mpl.rcParams.update({
            "figure.facecolor": self.surface, "axes.facecolor": self.surface,
            "savefig.facecolor": self.surface, "figure.dpi": 110,
            "axes.edgecolor": self.muted, "axes.labelcolor": self.ink,
            "axes.titlecolor": self.ink, "axes.titlesize": 11, "axes.titleweight": "medium",
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "axes.axisbelow": True, "grid.color": self.grid, "grid.linewidth": 0.6,
            "xtick.color": self.muted, "ytick.color": self.muted,
            "xtick.labelcolor": self.ink, "ytick.labelcolor": self.ink,
            "lines.linewidth": 2.0, "lines.markersize": 8,
            "legend.frameon": False, "font.size": 10,
            "axes.prop_cycle": mpl.cycler(color=self.series),
        })


style = _Style()
