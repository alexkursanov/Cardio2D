"""
Параметрические серии.
=======================

Серия — базовая конфигурация плюс набор изменяемых параметров. Описание
в JSON:

    {
      "base": "baseline.json",              // путь (от файла серии) или словарь
      "out_root": "runs/tmax_sweep",
      "mode": "grid",                       // "grid" — все сочетания,
                                            // "zip"  — i-е значения вместе
      "parameters": {
        "tissue_base.active.t_max": [60, 90, 120],
        "stimulus.times_ms": [[0, 500], [0, 400]]
      },
      "stop_on_error": false
    }

Каждая точка серии — обычный прогон в out_root/run_000, run_001, … со
своим run.json. В out_root/sweep.json — сводка: какие параметры у какой
точки, где лежит результат, чем закончилось. Её читает анализ.

Свойства
--------
* Все точки проверяются ДО начала счёта: опечатка в пути параметра или
  недопустимое значение в последней точке обнаружатся сразу, а не через
  сутки счёта.
* Повторный запуск той же серии пропускает точки, уже завершённые
  успешно (status = finished в их run.json), и пересчитывает остальные.
  Серию, прерванную лимитом времени на кластере, достаточно запустить
  ещё раз.
* Прогоны идут последовательно в одном процессе — это возможно, потому
  что расчётное ядро не хранит глобального состояния (шаг 8).

Разбор и раскрутка серии не требуют DOLFINx.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config.simulation import OutputConfig, SimulationConfig
from .overrides import apply_overrides

__all__ = ["SweepSpec", "SweepPoint", "run_sweep", "SWEEP_INDEX"]

SWEEP_INDEX = "sweep.json"
_FORMAT = "cardiac_em.sweep"


@dataclass
class SweepPoint:
    index: int
    name: str
    overrides: dict
    config: SimulationConfig

    @property
    def out_dir(self) -> Path:
        return Path(self.config.output.out_dir)

    @property
    def label(self) -> str:
        return ", ".join(f"{k.rsplit('.', 1)[-1]}={v}" for k, v in self.overrides.items())


@dataclass
class SweepSpec:
    base: SimulationConfig
    parameters: dict[str, list]
    out_root: Path
    mode: str = "grid"
    stop_on_error: bool = False
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out_root = Path(self.out_root)
        if self.mode not in ("grid", "zip"):
            raise ValueError(f"mode должен быть 'grid' или 'zip', получено {self.mode!r}")
        if not self.parameters:
            raise ValueError("в серии нет изменяемых параметров")
        for path, values in self.parameters.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"значения параметра {path!r} должны быть "
                                 f"непустым списком, получено {values!r}")
        if self.mode == "zip":
            lengths = {len(v) for v in self.parameters.values()}
            if len(lengths) != 1:
                raise ValueError(
                    f"в режиме zip списки значений должны быть одной длины, "
                    f"получено {({k: len(v) for k, v in self.parameters.items()})}")

    # ── чтение ────────────────────────────────────────────────────────
    @staticmethod
    def from_dict(d: dict, base_dir: Path | None = None) -> "SweepSpec":
        base = d.get("base")
        if isinstance(base, str):
            path = Path(base)
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            base = SimulationConfig.from_json(path)
        elif isinstance(base, dict):
            base = SimulationConfig.from_dict(base)
        else:
            raise ValueError("в серии нужен 'base': путь к конфигурации или словарь")
        if "out_root" not in d:
            raise ValueError("в серии нужен 'out_root' — папка для прогонов")
        return SweepSpec(base=base, parameters=dict(d.get("parameters", {})),
                         out_root=Path(d["out_root"]), mode=d.get("mode", "grid"),
                         stop_on_error=bool(d.get("stop_on_error", False)))

    @staticmethod
    def from_json(path) -> "SweepSpec":
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            return SweepSpec.from_dict(json.load(fh), base_dir=path.parent)

    # ── раскрутка ─────────────────────────────────────────────────────
    def combinations(self) -> list[dict]:
        keys = list(self.parameters)
        values = [self.parameters[k] for k in keys]
        rows = (itertools.product(*values) if self.mode == "grid" else zip(*values))
        return [dict(zip(keys, row)) for row in rows]

    def expand(self) -> list[SweepPoint]:
        """
        Все точки серии с готовыми конфигурациями. Любая ошибка в
        параметрах — здесь, до начала счёта, с номером точки.
        """
        points = []
        for i, combo in enumerate(self.combinations()):
            name = f"run_{i:03d}"
            try:
                cfg = apply_overrides(self.base, combo)
            except (KeyError, ValueError, TypeError) as exc:
                msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
                raise ValueError(f"точка {i} ({combo}): {msg}") from exc
            out = cfg.output
            cfg.output = OutputConfig(
                out_dir=self.out_root / name,
                save_every_mech_steps=out.save_every_mech_steps,
                snapshot_times_ms=out.snapshot_times_ms,
                ckpt_every_mech_steps=out.ckpt_every_mech_steps,
                ckpt_keep_all=out.ckpt_keep_all,
                write_region_maps=out.write_region_maps)
            points.append(SweepPoint(i, name, combo, cfg))
        return points


# ═══════════════════════════════════════════════════════════════════════
#  ВЫПОЛНЕНИЕ
# ═══════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_status(out_dir: Path) -> str | None:
    """Статус прогона из его run.json; None — прогона не было."""
    path = out_dir / "run.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        return "unreadable"


def run_sweep(spec: SweepSpec, *, console: bool = True, dry_run: bool = False,
              skip_finished: bool = True, comm=None) -> dict:
    """
    Выполнить серию. КОЛЛЕКТИВНАЯ операция. Возвращает сводку (то же,
    что пишется в sweep.json).

    dry_run       : только проверить и записать сводку, ничего не считать
    skip_finished : не пересчитывать точки с status = finished
    """
    from ..io.manifest import write_json_atomic

    points = spec.expand()                       # все ошибки параметров — здесь

    if comm is None:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except ImportError:
            comm = None
    rank = comm.rank if comm is not None else 0

    def say(text):
        if console and rank == 0:
            print(text, flush=True)

    index = {
        "format": _FORMAT, "version": 1,
        "started": _now(), "updated": _now(),
        "mode": spec.mode,
        "parameters": spec.parameters,
        "base": spec.base.to_dict(skip_unserializable=True),
        "points": [{"index": p.index, "name": p.name, "overrides": p.overrides,
                    "label": p.label, "out_dir": str(p.out_dir),
                    "status": "pending", "error": None, "elapsed_s": None}
                   for p in points],
    }
    index_path = spec.out_root / SWEEP_INDEX

    def save_index():
        index["updated"] = _now()
        if rank == 0:
            write_json_atomic(index_path, index)

    save_index()
    say(f"Серия: {len(points)} прогонов ({spec.mode}) → {spec.out_root}")
    if dry_run:
        for p in points:
            say(f"  {p.name}: {p.label}")
        return index

    from .api import run_simulation

    for p, rec in zip(points, index["points"]):
        status = _read_status(p.out_dir) if rank == 0 else None
        if comm is not None:
            status = comm.bcast(status, root=0)

        if status == "finished" and skip_finished:
            rec["status"] = "finished"
            rec["skipped"] = True
            say(f"── {p.name} [{p.label}] — уже посчитан, пропуск")
            save_index()
            continue

        say(f"── {p.name} [{p.label}]")
        rec["status"] = "running"
        save_index()
        t0 = time.perf_counter()
        try:
            run_simulation(p.config, console=console,
                           overwrite=status is not None, comm=comm)
            rec["status"] = "finished"
        except KeyboardInterrupt:
            rec["status"] = "failed"
            rec["error"] = "прервано вручную"
            save_index()
            raise
        except Exception as exc:  # noqa: BLE001 — точка серии, а не вся серия
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            say(f"   ОШИБКА: {rec['error']}")
            if spec.stop_on_error:
                rec["elapsed_s"] = round(time.perf_counter() - t0, 3)
                save_index()
                raise
        rec["elapsed_s"] = round(time.perf_counter() - t0, 3)
        save_index()

    n_ok = sum(r["status"] == "finished" for r in index["points"])
    say(f"Серия завершена: {n_ok} из {len(points)} успешно")
    return index
