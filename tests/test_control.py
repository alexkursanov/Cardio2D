"""
Тесты слоя control без DOLFINx (Шаг 10а).
==========================================

    python tests/test_control.py
    pytest tests/test_control.py -v

Изменение параметров по пути, раскрутка серий, ведение sweep.json
(с подменённым прогоном), командная строка: init / show / status / коды
выхода. Настоящий счёт через control — в tests/test_control_fem.py.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.config import (  # noqa: E402
    CustomRegion,
    RectRegion,
    SimulationConfig,
)
from cardiac_em.control import (  # noqa: E402
    SweepSpec,
    apply_overrides,
    parse_assignment,
    parse_value,
    run_sweep,
)
from cardiac_em.control import api as control_api  # noqa: E402
from cardiac_em.control.cli import main  # noqa: E402


def _raises(exc_type, fn, *args, match=None, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as e:
        if match is not None and match not in str(e):
            raise AssertionError(f"в сообщении нет {match!r}: {e}") from e
        return e
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            f"ожидалось {exc_type.__name__}, получено {type(e).__name__}: {e}") from e
    raise AssertionError(f"ожидалось {exc_type.__name__}, но исключения не было")


def _cli(*argv) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


# ═══════════════════════════════════════════════════════════════════════
#  ПЕРЕОПРЕДЕЛЕНИЕ ПАРАМЕТРОВ
# ═══════════════════════════════════════════════════════════════════════

def test_parse_value_and_assignment():
    assert parse_value("60") == 60 and parse_value("1.5") == 1.5
    assert parse_value("[0, 500]") == [0, 500]
    assert parse_value("true") is True and parse_value("null") is None
    assert parse_value("runs/a") == "runs/a"
    assert parse_assignment("time.t_end_ms=300") == ("time.t_end_ms", 300)
    assert parse_assignment("output.out_dir=a=b") == ("output.out_dir", "a=b")
    _raises(ValueError, parse_assignment, "time.t_end_ms")


def test_overrides_reach_the_config_and_leave_original_intact():
    base = SimulationConfig.default()
    new = apply_overrides(base, {"tissue_base.active.t_max": 60,
                                 "stimulus.times_ms": [0, 500],
                                 "time.t_end_ms": 900.0})
    assert new.tissue_base.active.t_max == 60
    assert new.stimulus.times_ms == (0.0, 500.0)
    assert new.time.t_end_ms == 900.0
    assert base.tissue_base.active.t_max == 120.0, "исходная не меняется"


def test_typo_in_path_is_an_error():
    base = SimulationConfig.default()
    e = _raises(KeyError, apply_overrides, base, {"tissue_base.active.tmax": 60})
    assert "t_max" in str(e), "в сообщении — список допустимых ключей"
    _raises(KeyError, apply_overrides, base, {"time.t_end_ms.x": 1})


def test_invalid_value_caught_by_config_validation():
    base = SimulationConfig.default()
    _raises(ValueError, apply_overrides, base, {"output.save_every_mech_steps": 0})


def test_region_overrides_by_index():
    base = SimulationConfig.default()
    base.regions = [RectRegion(name="isch", x0=0, x1=5, y0=0, y1=5,
                               overrides={"AP_c1": 0.2})]
    new = apply_overrides(base, {"regions.0.overrides.AP_c1": 0.1,
                                 "regions.0.overrides.T_MAX": 30.0})
    assert new.regions[0].overrides == {"AP_c1": 0.1, "T_MAX": 30.0}
    _raises(KeyError, apply_overrides, base, {"regions.1.overrides.AP_c1": 0.1})
    # новый ключ региона проверяется самим регионом
    _raises(ValueError, apply_overrides, base, {"regions.0.overrides.AP_C1": 0.1})


def test_custom_regions_survive_overrides():
    base = SimulationConfig.default()
    base.regions = [CustomRegion(name="клин", predicate=lambda x, y: x < y,
                                 overrides={"T_MAX": 10.0})]
    new = apply_overrides(base, {"time.t_end_ms": 100.0})
    assert [r.name for r in new.regions] == ["клин"]


# ═══════════════════════════════════════════════════════════════════════
#  СЕРИИ
# ═══════════════════════════════════════════════════════════════════════

def _spec(tmp: Path, **kw) -> SweepSpec:
    d = {"base": SimulationConfig.default().to_dict(), "out_root": str(tmp / "sw"),
         "parameters": {"tissue_base.active.t_max": [60, 90, 120],
                        "time.t_end_ms": [100.0, 200.0]}}
    d.update(kw)
    return SweepSpec.from_dict(d)


def test_grid_and_zip_expansion():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        grid = _spec(tmp).expand()
        assert len(grid) == 6
        assert [p.name for p in grid][:2] == ["run_000", "run_001"]
        assert grid[1].overrides == {"tissue_base.active.t_max": 60,
                                     "time.t_end_ms": 200.0}
        assert grid[5].config.tissue_base.active.t_max == 120
        assert grid[5].out_dir == tmp / "sw" / "run_005"

        z = _spec(tmp, mode="zip", parameters={
            "tissue_base.active.t_max": [60, 90],
            "time.t_end_ms": [100.0, 200.0]}).expand()
        assert [(p.config.tissue_base.active.t_max, p.config.time.t_end_ms)
                for p in z] == [(60, 100.0), (90, 200.0)]


def test_bad_sweeps_rejected_before_running():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _raises(ValueError, _spec, tmp, mode="zip", match="одной длины")
        _raises(ValueError, _spec, tmp, parameters={}, match="нет изменяемых")
        _raises(ValueError, _spec, tmp, mode="random")
        bad = _spec(tmp, parameters={"tissue_base.active.t_max": [60, -1]})
        e = _raises(ValueError, bad.expand, match="точка 1")
        assert "t_max" in str(e)


def test_sweep_base_path_is_relative_to_sweep_file():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "cfg").mkdir()
        SimulationConfig.default().to_json(tmp / "cfg" / "base.json")
        (tmp / "cfg" / "s.json").write_text(json.dumps({
            "base": "base.json", "out_root": str(tmp / "out"),
            "parameters": {"time.t_end_ms": [10.0]}}), encoding="utf-8")
        spec = SweepSpec.from_json(tmp / "cfg" / "s.json")
        assert spec.expand()[0].config.time.t_end_ms == 10.0


def _fake_runner(fail_names=()):
    """Подмена run_simulation: пишет run.json, как настоящий прогон."""
    calls = []

    def fake(config, *, console=True, overwrite=False, comm=None, **kw):
        out = Path(config.output.out_dir)
        calls.append((out.name, overwrite))
        out.mkdir(parents=True, exist_ok=True)
        status = "failed" if out.name in fail_names else "finished"
        (out / "run.json").write_text(json.dumps({"status": status}))
        if status == "failed":
            raise RuntimeError("механика не сошлась")

    return fake, calls


def _with_fake(fake, fn, *args, **kwargs):
    original = control_api.run_simulation
    control_api.run_simulation = fake
    try:
        return fn(*args, **kwargs)
    finally:
        control_api.run_simulation = original


def test_sweep_index_records_every_point_and_failures():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        spec = _spec(tmp)
        fake, calls = _fake_runner(fail_names=("run_002",))
        index = _with_fake(fake, run_sweep, spec, console=False)

        saved = json.loads((tmp / "sw" / "sweep.json").read_text(encoding="utf-8"))
        assert saved["points"] == index["points"]
        statuses = [p["status"] for p in saved["points"]]
        assert statuses == ["finished", "finished", "failed",
                            "finished", "finished", "finished"], \
            "сбой одной точки не останавливает серию"
        assert "механика не сошлась" in saved["points"][2]["error"]
        assert len(calls) == 6


def test_sweep_rerun_skips_finished_and_retries_failed():
    """Серию, прерванную лимитом времени, достаточно запустить ещё раз."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        spec = _spec(tmp)
        fake, _ = _fake_runner(fail_names=("run_002",))
        _with_fake(fake, run_sweep, spec, console=False)

        fake2, calls2 = _fake_runner()
        index = _with_fake(fake2, run_sweep, spec, console=False)
        assert calls2 == [("run_002", True)], \
            "пересчитывается только неудавшаяся точка, с разрешением перезаписи"
        assert all(p["status"] == "finished" for p in index["points"])
        assert sum(bool(p.get("skipped")) for p in index["points"]) == 5


def test_sweep_stop_on_error():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        spec = _spec(tmp, stop_on_error=True)
        fake, calls = _fake_runner(fail_names=("run_001",))
        _raises(RuntimeError, _with_fake, fake, run_sweep, spec, console=False)
        assert len(calls) == 2
        saved = json.loads((tmp / "sw" / "sweep.json").read_text(encoding="utf-8"))
        assert [p["status"] for p in saved["points"]][:3] == ["finished", "failed", "pending"]


def test_dry_run_computes_nothing():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        fake, calls = _fake_runner()
        index = _with_fake(fake, run_sweep, _spec(tmp), console=False, dry_run=True)
        assert calls == [] and len(index["points"]) == 6
        assert (tmp / "sw" / "sweep.json").exists()


# ═══════════════════════════════════════════════════════════════════════
#  КОМАНДНАЯ СТРОКА
# ═══════════════════════════════════════════════════════════════════════

def test_cli_init_and_show():
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "base.json"
        code, out, _ = _cli("init", cfg, "--nx", 24, "--coarsening", 4)
        assert code == 0 and cfg.exists()
        assert SimulationConfig.from_json(cfg).mesh.electric.nx == 24

        code, _, err = _cli("init", cfg)
        assert code == 2 and "--force" in err

        code, out, _ = _cli("show", cfg, "--set", "tissue_base.active.t_max=60",
                            "--t-end", 300)
        assert code == 0 and "T_max=60" in out and "300" in out

        code, out, _ = _cli("show", cfg, "--out", "runs/x", "--json")
        assert json.loads(out)["output"]["out_dir"] == "runs/x"


def test_cli_usage_errors_exit_2():
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "base.json"
        _cli("init", cfg)
        code, _, err = _cli("show", cfg, "--set", "tissue_base.activ.t_max=60")
        assert code == 2 and "tissue_base.activ" in err and "'" + "active" in err
        code, _, err = _cli("run", Path(d) / "нет.json")
        assert code == 2 and "нет файла" in err
        code, _, err = _cli("status", d)
        assert code == 2


def test_cli_status_of_run_and_sweep():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "r").mkdir()
        (tmp / "r" / "run.json").write_text(json.dumps({
            "status": "running", "progress": {"t_ms": 120.0},
            "schedule": {"t_end_ms": 1500.0}, "elapsed_s": None}))
        code, out, _ = _cli("status", tmp / "r")
        assert code == 0 and "running" in out and "120.0 из 1500.0" in out

        fake, _ = _fake_runner(fail_names=("run_001",))
        _with_fake(fake, run_sweep, _spec(tmp), console=False)
        code, out, _ = _cli("status", tmp / "sw")
        assert code == 0 and "failed 1" in out and "finished 5" in out


def test_cli_sweep_dry_run():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        s = tmp / "s.json"
        s.write_text(json.dumps({
            "base": SimulationConfig.default().to_dict(), "out_root": str(tmp / "o"),
            "parameters": {"tissue_base.active.t_max": [60, 90]}}))
        code, out, _ = _cli("sweep", s, "--dry-run")
        assert code == 0 and "t_max=90" in out
        s.write_text(json.dumps({
            "base": SimulationConfig.default().to_dict(), "out_root": str(tmp / "o"),
            "parameters": {"tissue_base.active.tmax": [60]}}))
        code, _, err = _cli("sweep", s, "--dry-run")
        assert code == 2 and "точка 0" in err


# ═══════════════════════════════════════════════════════════════════════

def _main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:  # noqa: BLE001
            failed.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} тестов прошло")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
