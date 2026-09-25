#!/usr/bin/env python
"""
Очередь расчётов: серии и одиночные прогоны по списку в текстовом файле.
=========================================================================

    python scripts/queue.py QUEUE.txt --np 8 --parallel 3

Файл очереди — по заданию в строке, `#` — комментарий:

    # серия: JSON с ключами "base", "out_root", "parameters"
    examples/sweep_tnnpm_1hz.json
    # одиночный прогон: конфигурация и, при желании, флаги как у `cardiac_em run`
    examples/tnnpm_ischemia.json --out runs/isch_long --t-end 3000
    examples/tnnpm_ischemia.json --out runs/isch_s2 --set stimulus.times_ms=[1,1001,1301]

Как работает
------------
* Серия раскладывается на точки (run_000, run_001, …); каждая точка и
  каждый одиночный прогон — отдельное задание `mpirun -n NP … cardiac_em run`.
* Одновременно идут до `--parallel` заданий (всего NP·parallel ядер), в
  порядке строк файла. Вывод каждого — в `<папка прогона>.log`.
* Готовые прогоны (status = finished в run.json) пропускаются, поэтому
  очередь можно перезапускать сколько угодно: досчитается только
  недостающее. Упавшее задание в этом запуске повторно не берётся.
* Файл очереди перечитывается, когда освобождается место, — строки
  можно ДОПИСЫВАТЬ, пока очередь работает.
* Когда все точки серии готовы, пишется её сводка sweep.json (её читают
  ноутбуки 03–05 и `cardiac_em status`).
* Остановка: создайте файл `QUEUE.txt.stop` — новые задания не
  начнутся, идущие досчитаются. `--watch` — не выходить, когда задания
  кончились, а ждать новых строк (до файла .stop).

Состояние очереди без запуска:

    python scripts/queue.py QUEUE.txt --status

Запуск на ночь (не зависит от закрытия терминала):

    nohup python scripts/queue.py QUEUE.txt --np 8 --parallel 3 > runs/queue.log 2>&1 &

DOLFINx самому скрипту не нужен (он только собирает команды и следит
за процессами); нужен он запускаемым прогонам.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cardiac_em.control.cli import _load_config, build_parser  # noqa: E402
from cardiac_em.control.sweep import SweepSpec  # noqa: E402


def say(text: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  ЗАДАНИЯ
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Task:
    key: str                     # уникальное имя: папка прогона
    label: str
    config: Path                 # JSON конфигурации для `cardiac_em run`
    extra: list[str]             # дополнительные флаги run
    out_dir: Path
    sweep: Path | None = None    # файл серии, если задание — её точка
    cfg_obj: object = None       # конфигурация точки серии: пишется в config перед запуском

    @property
    def log(self) -> Path:
        return self.out_dir.with_name(self.out_dir.name + ".log")


@dataclass
class SweepGroup:
    spec_path: Path
    out_root: Path
    keys: list[str] = field(default_factory=list)


def run_status(out_dir: Path) -> str | None:
    p = out_dir / "run.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        return "unreadable"


def run_progress(out_dir: Path) -> str:
    p = out_dir / "run.json"
    if not p.exists():
        return ""
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    t = m.get("progress", {}).get("t_ms")
    t_end = m.get("schedule", {}).get("t_end_ms")
    if t is None or not t_end:
        return ""
    return f"t = {t:g} из {t_end:g} мс ({100 * t / t_end:.0f} %)"


def _is_sweep(path: Path) -> bool:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(d, dict) and "parameters" in d and "out_root" in d


def expand_line(line: str, base_dir: Path) -> tuple[list[Task], SweepGroup | None]:
    """Строка очереди → задания. Ошибка в строке — ValueError с пояснением."""
    parts = shlex.split(line)
    path = Path(parts[0])
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        raise ValueError(f"нет файла {parts[0]}")

    if _is_sweep(path):
        if len(parts) > 1:
            raise ValueError("у серии не бывает дополнительных флагов — "
                             "меняйте параметры в её JSON")
        spec = SweepSpec.from_json(path)
        out_root = spec.out_root if spec.out_root.is_absolute() else ROOT / spec.out_root
        group = SweepGroup(path, out_root)
        tasks = []
        for p in spec.expand():
            out_dir = out_root / p.name
            cfg_file = out_root / f"{p.name}.config.json"
            cfg = p.config
            cfg.output = dataclasses.replace(cfg.output, out_dir=out_dir)
            tasks.append(Task(key=str(out_dir), label=f"{path.stem}/{p.name}: {p.label}",
                              config=cfg_file, extra=[], out_dir=out_dir, sweep=path,
                              cfg_obj=cfg))
            group.keys.append(str(out_dir))
        return tasks, group

    # одиночный прогон: разбор флагов тем же парсером, что у `cardiac_em run`
    args = build_parser().parse_args(["run", str(path), *parts[1:]])
    cfg = _load_config(args)
    out_dir = Path(cfg.output.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    return [Task(key=str(out_dir), label=f"{path.stem} → {out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir}",
                 config=path, extra=parts[1:], out_dir=out_dir)], None


def read_queue(queue: Path, reported: set[str]) -> tuple[list[Task], list[SweepGroup]]:
    """Все задания файла очереди по порядку; ошибки строк — в лог (один раз)."""
    tasks: list[Task] = []
    groups: list[SweepGroup] = []
    seen: set[str] = set()
    for n, raw in enumerate(queue.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            new, group = expand_line(line, ROOT)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 — строка очереди с ошибкой не роняет очередь
            msg = f"строка {n} «{line}»: {exc}"
            if msg not in reported:
                reported.add(msg)
                say(f"[!] {msg} — пропускаю")
            continue
        for t in new:
            if t.key in seen:
                continue
            seen.add(t.key)
            tasks.append(t)
        if group is not None:
            groups.append(group)
    return tasks, groups


# ═══════════════════════════════════════════════════════════════════════
#  ЗАПУСК
# ═══════════════════════════════════════════════════════════════════════

def mpi_command(mpirun: str) -> list[str]:
    exe = shutil.which(mpirun) or mpirun
    opts: list[str] = []
    try:
        ver = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
        if "open mpi" in (ver.stdout + ver.stderr).lower():
            # несколько mpirun одновременно не должны садиться на одни ядра
            opts = ["--bind-to", "none"]
    except (OSError, subprocess.SubprocessError):
        pass
    return [exe, *opts]


def start(task: Task, mpi: list[str], np_: int, every: int) -> subprocess.Popen:
    task.out_dir.parent.mkdir(parents=True, exist_ok=True)
    if task.cfg_obj is not None:
        task.cfg_obj.to_json(task.config)
    cmd = [*mpi, "-n", str(np_), sys.executable, "-m", "cardiac_em", "run",
           str(task.config), *task.extra, "--overwrite", "--every", str(every)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    log = open(task.log, "w", encoding="utf-8")
    log.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
    log.flush()
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=env)


def write_sweep_index(group: SweepGroup) -> None:
    """Все точки готовы → sweep.json (сама серия ничего не пересчитает)."""
    code = subprocess.run([sys.executable, "-m", "cardiac_em", "sweep",
                           str(group.spec_path), "--quiet"], cwd=ROOT).returncode
    say(f"сводка серии {group.out_root.relative_to(ROOT) if group.out_root.is_relative_to(ROOT) else group.out_root}"
        f"/sweep.json {'записана' if code == 0 else f'НЕ записана (код {code})'}")


def print_status(queue: Path) -> None:
    tasks, groups = read_queue(queue, set())
    stop = queue.with_name(queue.name + ".stop")
    print(f"очередь {queue}: заданий {len(tasks)}" + ("  [остановлена: есть .stop]" if stop.exists() else ""))
    for t in tasks:
        st = run_status(t.out_dir) or "ждёт"
        extra = run_progress(t.out_dir) if st == "running" else ""
        print(f"  {st:9s} {t.label}  {extra}")
    for g in groups:
        idx = g.out_root / "sweep.json"
        print(f"  сводка {g.spec_path.name}: {'есть' if idx.exists() else 'нет'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue", type=Path, help="файл очереди")
    ap.add_argument("--np", type=int, default=8, help="рангов MPI на одно задание (8)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="сколько заданий одновременно (1); всего ядер — np·parallel")
    ap.add_argument("--watch", action="store_true",
                    help="не выходить, когда задания кончились: ждать новых строк")
    ap.add_argument("--poll", type=float, default=30.0, help="опрос процессов, с (30)")
    ap.add_argument("--every", type=int, default=250,
                    help="строка хода счёта в логе прогона каждые N мех. шагов (250)")
    ap.add_argument("--mpirun", default="mpirun", help="команда запуска MPI")
    ap.add_argument("--status", action="store_true", help="показать состояние и выйти")
    ap.add_argument("--dry-run", action="store_true",
                    help="показать, что будет запущено, и выйти")
    args = ap.parse_args(argv)

    queue = args.queue.resolve()
    if not queue.exists():
        print(f"нет файла очереди {args.queue}", file=sys.stderr)
        return 2
    if args.status:
        print_status(queue)
        return 0

    stop = queue.with_name(queue.name + ".stop")
    ncpu = os.cpu_count() or 1
    say(f"очередь {queue.name}: {args.np} рангов × {args.parallel} заданий = "
        f"{args.np * args.parallel} ядер из {ncpu}")
    if args.np * args.parallel > ncpu:
        say(f"[!] заказано больше ядер, чем есть ({ncpu}) — счёт будет медленнее")

    if args.dry_run:
        tasks, _ = read_queue(queue, set())
        for t in tasks:
            st = run_status(t.out_dir)
            print(f"  {'пропуск (готово)' if st == 'finished' else 'запуск':17s} {t.label}")
        return 0

    mpi = mpi_command(args.mpirun)
    running: dict[str, tuple[Task, subprocess.Popen, float]] = {}
    failed: set[str] = set()
    indexed: set[str] = set()
    reported: set[str] = set()
    n_ok = n_fail = 0

    while True:
        tasks, groups = read_queue(queue, reported)

        # закончившиеся процессы
        for key, (task, proc, t0) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[key]
            dt = (time.time() - t0) / 60
            if code == 0 and run_status(task.out_dir) == "finished":
                n_ok += 1
                say(f"готово  {task.label}  ({dt:.0f} мин)")
            else:
                n_fail += 1
                failed.add(key)
                say(f"ОШИБКА  {task.label}  (код {code}, {dt:.0f} мин) — см. {task.log}")

        # сводки серий, у которых готово всё
        for g in groups:
            if str(g.spec_path) in indexed:
                continue
            if all(run_status(Path(k)) == "finished" for k in g.keys) and not \
                    any(k in running for k in g.keys):
                write_sweep_index(g)
                indexed.add(str(g.spec_path))

        # новые задания
        pending = [t for t in tasks if t.key not in running and t.key not in failed
                   and run_status(t.out_dir) != "finished"]
        if stop.exists():
            if not running:
                say(f"остановлено файлом {stop.name}; готово {n_ok}, ошибок {n_fail}, "
                    f"не начато {len(pending)}")
                return 1 if n_fail else 0
        else:
            while pending and len(running) < args.parallel:
                t = pending.pop(0)
                running[t.key] = (t, start(t, mpi, args.np, args.every), time.time())
                say(f"старт   {t.label}  → {t.log.relative_to(ROOT) if t.log.is_relative_to(ROOT) else t.log}")

        if not running and not pending and not args.watch:
            say(f"очередь пуста: готово {n_ok}, ошибок {n_fail}")
            return 1 if n_fail else 0
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
