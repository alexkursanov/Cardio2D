"""
Счёт через слой control: API, командная строка, серии (Шаг 10а).
================================================================

ТРЕБУЮТ DOLFINx.

    pytest tests/test_control_fem.py -v
    mpirun -n 2 python -m pytest tests/test_control_fem.py -q

Командная строка вызывается в том же процессе (`main([...])`), так что
под mpirun она работает так же, как `mpirun -n 2 python -m cardiac_em …`.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.config import (  # noqa: E402
    DualMeshConfig,
    OutputConfig,
    PreloadProtocol,
    RectangleMeshSpec,
    SimulationConfig,
    StimulusProtocol,
    TimeStepping,
)

pytestmark = pytest.mark.fem


def _comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD


def _shared(path: Path) -> Path:
    comm = _comm()
    return Path(comm.bcast(str(path) if comm.rank == 0 else None, root=0))


def _config(out_dir, t_end=6.0) -> SimulationConfig:
    return SimulationConfig(
        mesh=DualMeshConfig.nested(
            RectangleMeshSpec(nx=24, ny=8, lx_mm=6.0, ly_mm=2.0), coarsening=4),
        stimulus=StimulusProtocol(times_ms=(1.0,), duration_ms=2.0,
                                  amplitude=20.0, rise_time_ms=0.5,
                                  decay_length_mm=0.5, x_max_mm=2.0),
        time=TimeStepping(dt_electric_ms=0.05, n_electric_per_mech=20,
                          t_end_ms=t_end),
        preload=PreloadProtocol(stretch=1.2575, n_steps=4),
        output=OutputConfig(out_dir=out_dir, save_every_mech_steps=2,
                            ckpt_every_mech_steps=3, ckpt_keep_all=True),
    )


def _write_json(path: Path, data: dict) -> None:
    if _comm().rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    _comm().Barrier()


def _cli(*argv) -> tuple[int, str]:
    from cardiac_em.control.cli import main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = main([str(a) for a in argv])
    _comm().Barrier()
    return code, buf.getvalue()


def _manifest(out: Path) -> dict:
    return json.loads((out / "run.json").read_text(encoding="utf-8"))


def _series(out: Path):
    return np.genfromtxt(out / "series.csv", delimiter=",", names=True)


# ═══════════════════════════════════════════════════════════════════════
#  API
# ═══════════════════════════════════════════════════════════════════════

def test_run_simulation_end_to_end(tmp_path):
    from cardiac_em.control import run_simulation

    out = _shared(tmp_path) / "api"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run_simulation(_config(out), console=True, console_every_mech_steps=2)
    _comm().Barrier()

    assert result.t_start_ms == 0.0 and result.t_end_ms == 6.0
    assert result.manifest_path == out / "run.json"
    assert result.restart is None
    m = _manifest(out)
    assert m["status"] == "finished"
    assert (out / "ckpt_preloaded.npz").exists() and (out / "ckpt_last.npz").exists()
    if _comm().rank == 0:
        text = buf.getvalue()
        assert "Сетка эл." in text and "Монодомен" in text and "готово за" in text


def test_run_simulation_continues_from_config_restart(tmp_path):
    from cardiac_em.control import apply_overrides, run_simulation

    root = _shared(tmp_path)
    run_simulation(_config(root / "a"), console=False)
    cfg = apply_overrides(_config(root / "b", t_end=8.0),
                          {"restart.checkpoint_path": str(root / "a" / "ckpt_last.npz")})
    result = run_simulation(cfg, console=False)
    _comm().Barrier()

    assert result.t_start_ms == 6.0 and result.t_end_ms == 8.0
    assert result.restart["t_checkpoint_ms"] == 6.0
    assert _series(root / "b")["t_ms"][0] == 6.0


# ═══════════════════════════════════════════════════════════════════════
#  КОМАНДНАЯ СТРОКА
# ═══════════════════════════════════════════════════════════════════════

def test_cli_run_with_overrides_then_continue(tmp_path):
    root = _shared(tmp_path)
    base = root / "base.json"
    _write_json(base, _config("unused").to_dict())

    code, out = _cli("run", base, "--out", root / "r1", "--t-end", 4,
                     "--set", "tissue_base.active.t_max=60", "--quiet")
    assert code == 0, out
    m = _manifest(root / "r1")
    assert m["status"] == "finished"
    assert m["config"]["tissue_base"]["active"]["t_max"] == 60
    assert m["schedule"]["t_end_ms"] == 4.0

    code, out = _cli("run", base, "--out", root / "r2", "--t-end", 7,
                     "--set", "tissue_base.active.t_max=60",
                     "--restart", root / "r1" / "ckpt_last.npz", "--quiet")
    assert code == 0, out
    m2 = _manifest(root / "r2")
    assert m2["restart"]["t_checkpoint_ms"] == 4.0
    assert m2["restart"]["warnings"] == [], "та же физика — без предупреждений"
    np.testing.assert_allclose(_series(root / "r2")["t_ms"], [4, 5, 6, 7])


def test_cli_refuses_to_overwrite_a_run(tmp_path):
    root = _shared(tmp_path)
    base = root / "base.json"
    _write_json(base, _config(str(root / "r")).to_dict())

    assert _cli("run", base, "--t-end", 2, "--quiet")[0] == 0
    code, out = _cli("run", base, "--t-end", 2, "--quiet")
    assert code == 2
    if _comm().rank == 0:                 # печатает только нулевой ранг
        assert "overwrite" in out
    assert _cli("run", base, "--t-end", 2, "--quiet", "--overwrite")[0] == 0


def test_cli_status_after_run(tmp_path):
    root = _shared(tmp_path)
    base = root / "base.json"
    _write_json(base, _config(str(root / "r")).to_dict())
    _cli("run", base, "--t-end", 2, "--quiet")
    code, out = _cli("status", root / "r")
    assert code == 0
    if _comm().rank == 0:
        assert "finished" in out and "2.0 из 2.0" in out


# ═══════════════════════════════════════════════════════════════════════
#  СЕРИИ
# ═══════════════════════════════════════════════════════════════════════

def test_sweep_runs_points_and_skips_them_on_rerun(tmp_path):
    """
    Две точки с T_max = 60 и 120. Электрика от механики не зависит,
    поэтому электрический T_act второй точки ровно вдвое больше — это
    проверяет, что каждая точка получила СВОИ параметры.
    """
    root = _shared(tmp_path)
    spec = root / "sweep.json"
    _write_json(spec, {
        "base": _config("unused", t_end=5.0).to_dict(),
        "out_root": str(root / "sw"),
        "parameters": {"tissue_base.active.t_max": [60, 120]},
    })

    code, out = _cli("sweep", spec, "--quiet")
    assert code == 0, out
    idx = json.loads((root / "sw" / "sweep.json").read_text(encoding="utf-8"))
    assert [p["status"] for p in idx["points"]] == ["finished", "finished"]

    s0 = _series(root / "sw" / "run_000")
    s1 = _series(root / "sw" / "run_001")
    assert s1["t_act_electric_max"][-1] > 1.0
    assert s1["t_act_electric_max"][-1] == pytest.approx(
        2 * s0["t_act_electric_max"][-1], rel=1e-10)
    assert _manifest(root / "sw" / "run_001")["config"]["tissue_base"]["active"]["t_max"] == 120

    code, _ = _cli("sweep", spec, "--quiet")
    assert code == 0
    idx = json.loads((root / "sw" / "sweep.json").read_text(encoding="utf-8"))
    assert all(p.get("skipped") for p in idx["points"]), "повтор серии ничего не пересчитывает"


def test_sweep_failed_point_gives_exit_code_1(tmp_path):
    root = _shared(tmp_path)
    spec = root / "sweep.json"
    _write_json(spec, {
        "base": _config("unused", t_end=3.0).to_dict(),
        "out_root": str(root / "sw"),
        "mode": "zip",
        "parameters": {"restart.checkpoint_path": [None, "/nonexistent/ckpt.npz"]},
    })

    code, _ = _cli("sweep", spec, "--quiet")
    assert code == 1
    idx = json.loads((root / "sw" / "sweep.json").read_text(encoding="utf-8"))
    assert [p["status"] for p in idx["points"]] == ["finished", "failed"]
    assert "FileNotFoundError" in idx["points"][1]["error"]
