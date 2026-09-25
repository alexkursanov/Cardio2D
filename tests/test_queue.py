"""
Очередь расчётов scripts/queue.py — без DOLFINx и MPI.

Вместо mpirun подставляется скрипт, который «считает» мгновенно: пишет
run.json со status = finished (или failed, если в папке вывода есть
FAIL). Проверяется логика очереди: раскладка серий, параллельность,
пропуск готового, сводка серии, остановка, ошибки в строках.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from cardiac_em.config import SimulationConfig

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts" / "queue.py"

FAKE_MPIRUN = r'''#!{python}
import json, sys, time
from pathlib import Path
a = sys.argv[1:]
if a and a[0] == "--version":
    print("fake mpirun"); sys.exit(0)
i = a.index("run")
cfg = json.loads(Path(a[i + 1]).read_text(encoding="utf-8"))
out = Path(cfg["output"]["out_dir"])
if "--out" in a:
    out = Path(a[a.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
time.sleep(0.3)
bad = "FAIL" in str(out)
(out / "run.json").write_text(json.dumps({{"status": "failed" if bad else "finished",
    "config": cfg, "progress": {{"t_ms": 1.0}}, "schedule": {{"t_end_ms": 1.0}}}}))
print("fake run", out)
sys.exit(1 if bad else 0)
'''


def _setup(d: Path):
    fake = d / "mpirun"
    fake.write_text(FAKE_MPIRUN.format(python=sys.executable), encoding="utf-8")
    fake.chmod(0o755)
    base = SimulationConfig.default()
    base_file = d / "base.json"
    base.to_json(base_file)
    sweep = d / "sweep.json"
    sweep.write_text(json.dumps({
        "base": str(base_file), "out_root": str(d / "sw"), "mode": "zip",
        "parameters": {"tissue_base.active.t_max": [60, 90, 120]}}), encoding="utf-8")
    return fake, base_file, sweep


def _run(queue: Path, fake: Path, *extra):
    return subprocess.run([sys.executable, str(QUEUE), str(queue), "--np", "1",
                           "--parallel", "2", "--poll", "0.1", "--mpirun", str(fake), *extra],
                          capture_output=True, text=True, cwd=ROOT, timeout=120)


def _status(p: Path):
    f = p / "run.json"
    return json.loads(f.read_text())["status"] if f.exists() else None


def test_queue_runs_sweep_and_single_and_skips_finished():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fake, base, sweep = _setup(d)
        q = d / "q.txt"
        q.write_text(f"# серия\n{sweep}\n{base} --out {d / 'single'} --t-end 100  # прогон\n",
                     encoding="utf-8")
        r = _run(q, fake)
        assert r.returncode == 0, r.stdout + r.stderr
        for n in ("run_000", "run_001", "run_002"):
            assert _status(d / "sw" / n) == "finished"
            assert (d / "sw" / f"{n}.log").exists()
            assert (d / "sw" / f"{n}.config.json").exists()
        assert _status(d / "single") == "finished"
        idx = json.loads((d / "sw" / "sweep.json").read_text())
        assert [p["status"] for p in idx["points"]] == ["finished"] * 3, "сводка серии записана"
        assert "--t-end" in (d / "single.log").read_text(), "флаги строки передаются в run"

        r2 = _run(q, fake)
        assert r2.returncode == 0 and "старт" not in r2.stdout, "готовое не пересчитывается"


def test_queue_failure_bad_line_and_stop_file():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fake, base, _ = _setup(d)
        q = d / "q.txt"
        q.write_text(f"{d / 'nope.json'}\n{base} --out {d / 'FAIL_a'}\n{base} --out {d / 'ok_b'}\n",
                     encoding="utf-8")
        r = _run(q, fake)
        assert r.returncode == 1, "упавшее задание — код 1"
        assert "нет файла" in r.stdout and "ОШИБКА" in r.stdout
        assert _status(d / "ok_b") == "finished", "остальные задания идут"

        q2 = d / "q2.txt"
        q2.write_text(f"{base} --out {d / 'never'}\n", encoding="utf-8")
        (d / "q2.txt.stop").write_text("")
        _run(q2, fake)
        assert not (d / "never").exists(), "с файлом .stop новые задания не начинаются"

        st = subprocess.run([sys.executable, str(QUEUE), str(q), "--status"],
                            capture_output=True, text=True, cwd=ROOT)
        assert "finished" in st.stdout and "failed" in st.stdout


def _main() -> int:
    test_queue_runs_sweep_and_single_and_skips_finished()
    test_queue_failure_bad_line_and_stop_file()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
