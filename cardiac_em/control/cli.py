"""
Командная строка.
==================

    python -m cardiac_em <команда> …      (или `cardiac-em` после pip install -e .)

Команды
-------
    init   CONFIG.json            создать конфигурацию по умолчанию для правки
    show   CONFIG.json [--set …]  сводка и предупреждения — без счёта и без DOLFINx
    run    CONFIG.json [--set …]  прогон (под mpirun — параллельно)
    sweep  SWEEP.json             параметрическая серия
    status DIR                    состояние прогона или серии по run.json / sweep.json

Изменение параметров без правки файла
-------------------------------------
    --set tissue_base.active.t_max=60
    --set stimulus.times_ms=[0,500,1000]
    --set regions.0.overrides.AP_c1=0.1

Путь — ключи из CONFIG.json через точку, значение — JSON (иначе строка).
Опечатка в пути — ошибка. Частые параметры имеют короткие флаги:
--out, --t-end, --restart, --allow-interp, --reset-time.

Примеры
-------
    python -m cardiac_em init base.json --nx 80 --coarsening 4
    python -m cardiac_em show base.json --set time.t_end_ms=3000
    mpirun -n 4 python -m cardiac_em run base.json --out runs/a
    python -m cardiac_em run base.json --out runs/b \\
        --restart runs/a/ckpt_last.npz --t-end 3000 --set stimulus.times_ms=[2000,2400]
    python -m cardiac_em sweep tmax.json
    python -m cardiac_em status runs/tmax_sweep

Коды выхода: 0 — успех, 1 — прогон или часть серии не удались,
2 — ошибка в конфигурации или аргументах.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

__all__ = ["main", "build_parser"]


def _rank() -> int:
    """Номер ранга без импорта mpi4py (для лёгких команд)."""
    for var in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MV2_COMM_WORLD_RANK"):
        if var in os.environ:
            try:
                return int(os.environ[var])
            except ValueError:
                pass
    return 0


def _say(text: str = "", *, err: bool = False) -> None:
    if _rank() == 0:
        print(text, file=sys.stderr if err else sys.stdout, flush=True)


class _UsageError(Exception):
    """Ошибка в аргументах или конфигурации — код выхода 2."""


def _text(exc: BaseException) -> str:
    """Текст исключения; у KeyError str() добавляет лишние кавычки."""
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


# ═══════════════════════════════════════════════════════════════════════
#  РАЗБОР АРГУМЕНТОВ
# ═══════════════════════════════════════════════════════════════════════

def _add_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="ПУТЬ=ЗНАЧЕНИЕ",
                   help="изменить параметр конфигурации (можно несколько раз)")
    p.add_argument("--out", metavar="DIR", help="папка вывода (output.out_dir)")
    p.add_argument("--t-end", type=float, metavar="МС",
                   help="конец счёта, мс (time.t_end_ms); при продолжении — абсолютное время")
    p.add_argument("--restart", metavar="CKPT.npz",
                   help="продолжить с чекпоинта (restart.checkpoint_path)")
    p.add_argument("--allow-interp", action="store_true",
                   help="разрешить перенос состояния на другую сетку по ближайшему соседу")
    p.add_argument("--reset-time", action="store_true",
                   help="при продолжении начать отсчёт времени с нуля")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cardiac-em",
        description="Двумерная электромеханика сердечной ткани (DOLFINx).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Подробности: python -m cardiac_em <команда> -h")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="создать конфигурацию по умолчанию")
    p.add_argument("config", type=Path)
    p.add_argument("--nx", type=int, default=40, help="узлов электрической сетки по x и y")
    p.add_argument("--coarsening", type=int, default=2,
                   help="во сколько раз механическая сетка грубее")
    p.add_argument("--force", action="store_true", help="перезаписать существующий файл")

    p = sub.add_parser("show", help="сводка конфигурации без счёта")
    p.add_argument("config", type=Path)
    _add_overrides(p)
    p.add_argument("--json", action="store_true",
                   help="вывести итоговую конфигурацию в JSON (с учётом --set)")

    p = sub.add_parser("run", help="выполнить прогон")
    p.add_argument("config", type=Path)
    _add_overrides(p)
    p.add_argument("--overwrite", action="store_true",
                   help="разрешить запись в папку, где уже есть run.json")
    p.add_argument("--quiet", action="store_true", help="не печатать ход счёта")
    p.add_argument("--every", type=int, default=20, metavar="N",
                   help="печатать строку каждые N механических шагов")

    p = sub.add_parser("sweep", help="выполнить параметрическую серию")
    p.add_argument("spec", type=Path)
    p.add_argument("--dry-run", action="store_true",
                   help="только проверить серию и показать точки")
    p.add_argument("--rerun-finished", action="store_true",
                   help="пересчитать и уже завершённые точки")
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("status", help="состояние прогона или серии")
    p.add_argument("path", type=Path)

    return parser


def _load_config(args):
    """Конфигурация из файла с учётом --set и коротких флагов."""
    from ..config.simulation import SimulationConfig
    from .overrides import apply_overrides, parse_assignment

    if not args.config.exists():
        raise _UsageError(f"нет файла конфигурации {args.config}")
    try:
        cfg = SimulationConfig.from_json(args.config)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise _UsageError(f"{args.config}: {_text(exc)}") from exc

    overrides = {}
    try:
        for text in args.overrides:
            path, value = parse_assignment(text)
            overrides[path] = value
    except ValueError as exc:
        raise _UsageError(str(exc)) from exc
    if args.out is not None:
        overrides["output.out_dir"] = args.out
    if args.t_end is not None:
        overrides["time.t_end_ms"] = args.t_end
    if args.restart is not None:
        overrides["restart.checkpoint_path"] = args.restart
    if args.allow_interp:
        overrides["restart.allow_interp"] = True
    if args.reset_time:
        overrides["restart.reset_time"] = True

    try:
        return apply_overrides(cfg, overrides)
    except (KeyError, ValueError, TypeError) as exc:
        raise _UsageError(f"--set: {_text(exc)}") from exc


# ═══════════════════════════════════════════════════════════════════════
#  КОМАНДЫ
# ═══════════════════════════════════════════════════════════════════════

def _cmd_init(args) -> int:
    from ..config.simulation import SimulationConfig

    if args.config.exists() and not args.force:
        raise _UsageError(f"{args.config} уже существует (--force — перезаписать)")
    try:
        cfg = SimulationConfig.default(nx_electric=args.nx, coarsening=args.coarsening)
    except ValueError as exc:
        raise _UsageError(str(exc)) from exc
    if _rank() == 0:
        args.config.parent.mkdir(parents=True, exist_ok=True)
        cfg.to_json(args.config)
    _say(f"записано {args.config}")
    _say(cfg.summary())
    return 0


def _cmd_show(args) -> int:
    from ..runtime.schedule import Schedule

    cfg = _load_config(args)
    if args.json:
        _say(json.dumps(cfg.to_dict(skip_unserializable=True),
                        ensure_ascii=False, indent=2))
        return 0
    _say(cfg.summary())
    if not cfg.restart.is_restart:
        try:
            _say(Schedule(cfg.time, cfg.output).summary())
        except ValueError as exc:
            _say(f"  [!] {exc}")
    warnings = cfg.check()
    for w in warnings:
        _say(f"  [!] {w}")
    if not warnings:
        _say("  предупреждений нет")
    return 0


def _cmd_run(args) -> int:
    from .api import run_simulation

    cfg = _load_config(args)
    try:
        result = run_simulation(cfg, console=not args.quiet,
                                console_every_mech_steps=args.every,
                                overwrite=args.overwrite)
    except FileExistsError as exc:
        raise _UsageError(str(exc)) from exc
    _say(f"готово: {result.out_dir}  (t = {result.t_start_ms:g} → {result.t_end_ms:g} мс)")
    return 0


def _cmd_sweep(args) -> int:
    from .sweep import SweepSpec, run_sweep

    if not args.spec.exists():
        raise _UsageError(f"нет файла серии {args.spec}")
    try:
        spec = SweepSpec.from_json(args.spec)
        spec.expand()
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise _UsageError(f"{args.spec}: {_text(exc)}") from exc
    index = run_sweep(spec, console=not args.quiet, dry_run=args.dry_run,
                      skip_finished=not args.rerun_finished)
    if args.dry_run:
        return 0
    failed = [p for p in index["points"] if p["status"] != "finished"]
    return 1 if failed else 0


def _cmd_status(args) -> int:
    path = args.path
    if path.is_file():
        path = path.parent
    sweep = path / "sweep.json"
    run = path / "run.json"
    if sweep.exists():
        idx = json.loads(sweep.read_text(encoding="utf-8"))
        counts: dict[str, int] = {}
        for p in idx["points"]:
            counts[p["status"]] = counts.get(p["status"], 0) + 1
        _say(f"серия {path}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        for p in idx["points"]:
            line = f"  {p['name']}  {p['status']:9s}  {p['label']}"
            if p["status"] in ("running", "pending"):
                st = _run_line(Path(p["out_dir"]) / "run.json")
                if st:
                    line += f"  [{st}]"
            if p.get("error"):
                line += f"  — {p['error']}"
            _say(line)
        return 0
    if run.exists():
        _say(f"{path}: {_run_line(run)}")
        return 0
    raise _UsageError(f"в {path} нет ни run.json, ни sweep.json")


def _run_line(run_json: Path) -> str:
    if not run_json.exists():
        return ""
    m = json.loads(run_json.read_text(encoding="utf-8"))
    t = m.get("progress", {}).get("t_ms")
    t_end = m.get("schedule", {}).get("t_end_ms")
    text = f"{m.get('status')}, t = {t} из {t_end} мс"
    if m.get("elapsed_s") is not None:
        text += f", {m['elapsed_s']:.0f} с"
    if m.get("error"):
        text += f", ошибка: {m['error']}"
    return text


_COMMANDS = {"init": _cmd_init, "show": _cmd_show, "run": _cmd_run,
             "sweep": _cmd_sweep, "status": _cmd_status}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except _UsageError as exc:
        _say(f"ошибка: {exc}", err=True)
        return 2
    except Exception as exc:  # noqa: BLE001 — сбой счёта: сообщение и код 1
        _say(f"прогон не удался: {type(exc).__name__}: {exc}", err=True)
        if os.environ.get("CARDIAC_EM_TRACEBACK"):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
