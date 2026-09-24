"""
Тесты слоя io, не требующие DOLFINx (Шаг 9).
=============================================

    python tests/test_io_light.py
    pytest tests/test_io_light.py -v

Проверяется всё, что можно проверить без расчётного стека: формат
чекпоинта (запись → чтение), чтение чекпоинтов монолитной версии,
сопоставление точек, сравнение конфигураций, манифест и CSV-ряды на
поддельной «симуляции». Запись полей и само восстановление состояния —
в tests/test_io.py (нужен DOLFINx).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.config import (  # noqa: E402
    ActiveStressParams,
    OutputConfig,
    SimulationConfig,
    StimulusProtocol,
    TimeStepping,
    TissueBaseParams,
)
from cardiac_em.io import (  # noqa: E402
    SERIES_COLUMNS,
    ManifestObserver,
    SeriesWriter,
    attach_outputs,
    pack_checkpoint,
    read_checkpoint,
)
from cardiac_em.io._points import (  # noqa: E402
    _nearest,
    _nearest_bruteforce,
    match_points,
    yx_order,
)
from cardiac_em.io.checkpoint import _config_differences  # noqa: E402
from cardiac_em.models.cell import make_cell_model  # noqa: E402
from cardiac_em.runtime import Schedule  # noqa: E402


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


def _grid(nx, ny, lx=6.0, ly=2.0):
    x, y = np.meshgrid(np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1))
    return np.column_stack([x.ravel(), y.ravel()])


# ═══════════════════════════════════════════════════════════════════════
#  СОПОСТАВЛЕНИЕ ТОЧЕК
# ═══════════════════════════════════════════════════════════════════════

def test_match_exact_under_permutation():
    """Порядок узлов у другого числа рангов другой — индексы обязаны найтись."""
    ref = _grid(12, 4)
    perm = np.random.default_rng(0).permutation(len(ref))
    pts = ref[perm] + 1e-12                 # шум округления
    idx, n_near = match_points(ref, pts, tol=1e-6, allow_nearest=False, label="t")
    assert n_near == 0
    np.testing.assert_array_equal(idx, perm)


def test_match_mismatch_is_an_error_by_default():
    ref = _grid(6, 2)
    pts = _grid(12, 4)
    _raises(ValueError, match_points, ref, pts, 1e-6, False, "клетки",
            match="allow_interp")


def test_match_nearest_when_allowed():
    ref = _grid(6, 2)
    pts = _grid(12, 4)
    idx, n_near = match_points(ref, pts, 1e-6, True, "t")
    # узлы грубой сетки входят в мелкую ровно: (7·3) совпадают
    assert n_near == len(pts) - 7 * 3
    d = np.linalg.norm(ref[idx] - pts, axis=1)
    assert d.max() <= np.hypot(0.5, 0.5) / 1 + 1e-12


def test_bruteforce_agrees_with_kdtree():
    rng = np.random.default_rng(1)
    ref = rng.random((300, 2))
    pts = rng.random((200, 2))
    i1, d1 = _nearest(ref, pts)
    i2, d2 = _nearest_bruteforce(ref, pts)
    np.testing.assert_allclose(d1, d2, atol=1e-14)
    # при равных расстояниях индексы могут отличаться — сравниваем расстояния
    np.testing.assert_allclose(np.linalg.norm(ref[i2] - pts, axis=1), d2, atol=1e-14)


def test_yx_order_tolerates_roundoff_noise():
    """
    Воспроизводит сбой, найденный на сервере: у DOLFINx узлы одной строки
    отличаются по y на ~1e-17, и сортировка по точным значениям ставила
    узел (6, −1e-17) перед (0, 0). Порядок обязан быть построчным.
    """
    nx, ny = 24, 8
    grid = _grid(nx, ny)
    rng = np.random.default_rng(2)
    noisy = grid + rng.normal(scale=1e-16, size=grid.shape)
    perm = rng.permutation(len(grid))

    order = yx_order(noisy[perm])
    back = noisy[perm][order].reshape(ny + 1, nx + 1, 2)
    np.testing.assert_allclose(back[0, :, 0], np.linspace(0, 6, nx + 1), atol=1e-12)
    np.testing.assert_allclose(back[:, 0, 1], np.linspace(0, 2, ny + 1), atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════
#  ФОРМАТ ЧЕКПОИНТА
# ═══════════════════════════════════════════════════════════════════════

def _write_ckpt(tmp: Path, name="c.npz", **over) -> Path:
    e = _grid(24, 8)
    m = _grid(6, 2)
    arrays = dict(
        stage="running", t_ms=12.5, mech_index=12,
        config_dict=SimulationConfig.default().to_dict(),
        cell_model="rogers_mcculloch", state_names=("u", "v"),
        e_coords=e, e_state=np.column_stack([e[:, 0], e[:, 1]]),
        m_coords=m, m_u=0.1 * m)
    arrays.update(over)
    path = tmp / name
    np.savez_compressed(path, **pack_checkpoint(**arrays))
    return path


def test_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = _write_ckpt(Path(d))
        c = read_checkpoint(path)

    assert c.stage == "running" and c.t_ms == 12.5 and c.mech_index == 12
    assert c.cell_model == "rogers_mcculloch" and c.state_names == ("u", "v")
    assert not c.legacy
    np.testing.assert_array_equal(c.e_state[:, 0], c.e_coords[:, 0])
    np.testing.assert_array_equal(c.m_u, 0.1 * c.m_coords)
    assert c.config().to_dict() == SimulationConfig.default().to_dict(), \
        "конфигурация источника обязана восстанавливаться целиком"


def test_checkpoint_needs_no_pickle():
    """Чекпоинт читается с allow_pickle=False — никакого исполнения кода."""
    with tempfile.TemporaryDirectory() as d:
        path = _write_ckpt(Path(d))
        with np.load(path, allow_pickle=False) as z:
            assert {"e_state", "m_u", "config_json"} <= set(z.files)


def test_future_version_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = _write_ckpt(Path(d))
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        data["version"] = np.array(99)
        np.savez(path, **data)
        _raises(ValueError, read_checkpoint, path, match="обновите код")


def test_foreign_file_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.npz"
        np.savez(path, a=np.zeros(3))
        _raises(ValueError, read_checkpoint, path, match="не чекпоинт")


def test_missing_file_raises():
    _raises(FileNotFoundError, read_checkpoint, "/nonexistent/ckpt.npz")


def _legacy_npz(path: Path, *, t, mech_count, dt_e=0.05):
    e = _grid(24, 8)
    m = _grid(6, 2)
    np.savez_compressed(
        path, coords_u=np.column_stack([m, np.zeros(len(m))]), u=(0.2 * m).ravel(),
        coords_dg0=np.zeros((12, 3)), T_act=np.zeros(12),
        coords_p1=np.column_stack([e, np.zeros(len(e))]),
        u_ap=e[:, 0] / 6, v_ap=e[:, 1] / 2,
        t=float(t), step=int(round(t / dt_e)), mech_count=int(mech_count),
        dt_e=dt_e, n_mech=20, Lx=6.0, Ly=2.0, Nx_e=24, Ny_e=8, Nx_m=6, Ny_m=2)


def test_legacy_checkpoint_is_read_with_time_shift():
    """
    В монолитной версии в файл писалось время НАЧАЛА последнего шага, а
    состояние — уже в его конце. При чтении время сдвигается на dt.
    """
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt_last.npz"
        _legacy_npz(path, t=499.95, mech_count=25)
        c = read_checkpoint(path)

    assert c.legacy and c.stage == "running" and c.config_dict is None
    assert c.t_ms == 500.0
    assert c.cell_model == "rogers_mcculloch" and c.state_names == ("u", "v")
    assert c.e_state.shape == (25 * 9, 2) and c.m_u.shape == (7 * 3, 2)
    np.testing.assert_allclose(c.e_state[:, 0], c.e_coords[:, 0] / 6)
    np.testing.assert_allclose(c.m_u, 0.2 * c.m_coords)


def test_legacy_preloaded_checkpoint_has_no_shift():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt_preloaded.npz"
        _legacy_npz(path, t=0.0, mech_count=0)
        c = read_checkpoint(path)
    assert c.stage == "preloaded" and c.t_ms == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  СРАВНЕНИЕ КОНФИГУРАЦИЙ
# ═══════════════════════════════════════════════════════════════════════

def test_config_differences_physics_only():
    base = SimulationConfig.default()
    other = SimulationConfig.default()
    other.stimulus = StimulusProtocol(times_ms=(0.0, 500.0))
    other.time = TimeStepping(t_end_ms=2000.0)
    other.output = OutputConfig(out_dir="elsewhere")
    assert _config_differences(base.to_dict(), other.to_dict()) == [], \
        "новый протокол и новый t_end при продолжении — не повод предупреждать"

    other.tissue_base = TissueBaseParams(active=ActiveStressParams(t_max=60.0))
    diffs = _config_differences(base.to_dict(), other.to_dict())
    assert len(diffs) == 1 and "t_max" in diffs[0] and "60.0" in diffs[0]


# ═══════════════════════════════════════════════════════════════════════
#  МАНИФЕСТ И РЯДЫ НА ПОДДЕЛЬНОЙ СИМУЛЯЦИИ
# ═══════════════════════════════════════════════════════════════════════

class _Comm:
    rank, size = 0, 1

    def bcast(self, obj, root=0):
        return obj


def _fake_sim(out_dir: Path):
    cfg = SimulationConfig.default()
    cfg.output = OutputConfig(out_dir=out_dir)
    cfg.time = TimeStepping(dt_electric_ms=0.05, n_electric_per_mech=20, t_end_ms=5.0)
    sim = SimpleNamespace(
        comm=_Comm(), config=cfg, t_ms=0.0,
        cell_model=make_cell_model("rogers_mcculloch"),
        mechanics=SimpleNamespace(material=SimpleNamespace(
            name="transversely_isotropic_exponential")),
        transfer=SimpleNamespace(describe=lambda: "averaging (точный)", exact=True),
        pair=SimpleNamespace(n_cells_electric=1600, n_cells_mechanical=400),
        restart_info=None, warnings=["пример предупреждения"],
        observers=[],
    )
    sim.add_observer = sim.observers.append

    def diagnostics(electric=True):
        d = {k: 1.5 for k in SERIES_COLUMNS}
        d.update(t_ms=sim.t_ms, newton_iterations=3, mech_index=None)
        if not electric:
            for k in ("u_min", "u_max", "t_act_electric_max",
                      "t_act_electric_integral"):
                d[k] = None
        return d

    sim.diagnostics = diagnostics
    return sim


def _drive(sim, observers, fail_at=None):
    """Прогнать события так, как их выдаёт Simulation.run()."""
    sched = Schedule(sim.config.time, sim.config.output)
    for o in observers:
        for k in (1, 2, 3):
            o.on_preload_step(sim, k, 3)
        o.on_preload_done(sim)
        o.on_start(sim, sched)
    for tick in sched:
        if tick.is_mech:
            sim.t_ms = tick.t_end_ms
            if fail_at is not None and tick.mech_index == fail_at:
                exc = RuntimeError("механика не сошлась")
                for o in observers:
                    o.on_abort(sim, exc)
                return sched
            for o in observers:
                o.on_mech_tick(sim, tick)
    for o in observers:
        o.on_finish(sim)
    return sched


def test_manifest_lifecycle():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        sim = _fake_sim(out)
        obs = ManifestObserver(out)
        _drive(sim, [obs])
        (out / "series.csv").write_text("x\n")
        m = json.loads((out / "run.json").read_text(encoding="utf-8"))

    assert m["format"] == "cardiac_em.run" and m["status"] == "finished"
    assert m["progress"]["t_ms"] == 5.0 and m["progress"]["mech_index"] == 5
    assert m["models"]["cell"] == "rogers_mcculloch"
    assert m["models"]["cell_states"] == ["u", "v"]
    assert m["schedule"]["n_mech_steps"] == 5
    assert m["warnings"] == ["пример предупреждения"]
    assert m["environment"]["python"] and m["environment"]["mpi_size"] == 1
    assert m["elapsed_s"] is not None and m["error"] is None
    cfg = SimulationConfig.from_dict(m["config"])
    assert cfg.to_dict() == sim.config.to_dict(), "конфиг из манифеста — тот же"


def test_manifest_marks_failure():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        sim = _fake_sim(out)
        _drive(sim, [ManifestObserver(out)], fail_at=3)
        m = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert m["status"] == "failed"
    assert "механика не сошлась" in m["error"]
    assert m["progress"]["t_ms"] == 3.0


def test_manifest_lists_outputs_without_itself():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        (out / "snapshots").mkdir()
        (out / "snapshots" / "snap_t1.npz").write_bytes(b"")
        (out / "junk.tmp").write_bytes(b"")
        sim = _fake_sim(out)
        _drive(sim, [ManifestObserver(out)])
        m = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert m["outputs"] == ["snapshots/snap_t1.npz"]


def test_series_csv_is_readable_and_complete():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        sim = _fake_sim(out)
        _drive(sim, [SeriesWriter(out)])
        data = np.genfromtxt(out / "series.csv", delimiter=",", names=True)
        pre = np.genfromtxt(out / "preload.csv", delimiter=",", names=True)

    assert tuple(data.dtype.names) == SERIES_COLUMNS
    assert len(data) == 1 + 5, "начальная строка + 5 механических шагов"
    np.testing.assert_array_equal(data["t_ms"], [0, 1, 2, 3, 4, 5])
    assert np.isnan(data["mech_index"][0])
    np.testing.assert_array_equal(data["mech_index"][1:], [1, 2, 3, 4, 5])
    assert len(pre) == 3 and list(pre["step"]) == [1, 2, 3]


def test_series_decimation_keeps_last():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        sim = _fake_sim(out)
        _drive(sim, [SeriesWriter(out, every_mech_steps=2)])
        data = np.genfromtxt(out / "series.csv", delimiter=",", names=True)
    np.testing.assert_array_equal(data["t_ms"], [0, 2, 4, 5])


def test_series_survives_abort():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        sim = _fake_sim(out)
        _drive(sim, [SeriesWriter(out)], fail_at=4)
        data = np.genfromtxt(out / "series.csv", delimiter=",", names=True)
    np.testing.assert_array_equal(data["t_ms"], [0, 1, 2, 3])


def test_attach_outputs_refuses_to_overwrite():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        (out / "run.json").write_text("{}")
        sim = _fake_sim(out)
        _raises(FileExistsError, attach_outputs, sim, match="overwrite=True")
        assert sim.observers == [], "при отказе ничего не подключается"


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
