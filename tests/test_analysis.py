"""
Тесты анализа и регистратора активации без DOLFINx (Шаг 10б).
==============================================================

    python tests/test_analysis.py
    pytest tests/test_analysis.py -v

Детектор активации проверяется на искусственных сигналах с точно
известными ответами (кусочно-линейный потенциал действия: линейная
интерполяция между шагами даёт точное время). Чтение и биомаркеры —
на синтетической папке прогона с аналитическими полями: плоская волна
с заданной скоростью, известные APD по регионам, известный ряд
механики. Настоящий прогон — в tests/test_analysis_fem.py.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.analysis import biomarkers as bm  # noqa: E402
from cardiac_em.analysis import open_run, open_sweep  # noqa: E402
from cardiac_em.io.activation import ThresholdDetector  # noqa: E402

DT = 0.05


def _ap(t, t_up, apd_plateau=100.0, rise=1.0, fall=20.0, peak=1.0):
    """
    Кусочно-линейный ПД: подъём [t_up, t_up+rise], плато до
    t_up+rise+apd_plateau, линейный спад за `fall` мс до нуля.
    """
    t = np.asarray(t, dtype=np.float64)
    v = np.zeros_like(t)
    a = t_up
    b = a + rise
    c = b + apd_plateau
    d = c + fall
    v = np.where((t >= a) & (t < b), peak * (t - a) / rise, v)
    v = np.where((t >= b) & (t < c), peak, v)
    v = np.where((t >= c) & (t < d), peak * (1 - (t - c) / fall), v)
    return v


def _run_detector(signal, t_end, t0=0.0, apd_level=0.9, threshold=0.5):
    """signal(t) → массив по узлам; прогоняет детектор шагами DT."""
    n_steps = int(round((t_end - t0) / DT))
    det = ThresholdDetector(signal(t0), t0, threshold, v_rest=0.0, apd_level=apd_level)
    for k in range(1, n_steps + 1):
        t = t0 + k * DT
        det.update(signal(t), t)
    return det


# ═══════════════════════════════════════════════════════════════════════
#  ДЕТЕКТОР
# ═══════════════════════════════════════════════════════════════════════

def test_activation_time_is_interpolated_exactly():
    """Порог 0.5 при подъёме за 1 мс с t_up — ровно t_up + 0.5, а не шаг сетки."""
    t_up = np.array([2.013, 5.0, 7.777])
    det = _run_detector(lambda t: np.array([_ap(t, tu) for tu in t_up]), 150.0)
    act, rep, peak = det.maps()
    np.testing.assert_allclose(act[:, 0], t_up + 0.5, atol=1e-9)


def test_apd90_is_exact_for_linear_decay():
    """
    Уровень APD90 = 0.1 от пика достигается на 90 % спада:
    repol = t_up + rise + plateau + 0.9·fall.
    """
    t_up = np.array([1.0, 3.3])
    det = _run_detector(lambda t: np.array([_ap(t, tu) for tu in t_up]), 200.0)
    act, rep, peak = det.maps()
    np.testing.assert_allclose(rep[:, 0], t_up + 1 + 100 + 0.9 * 20, atol=1e-9)
    np.testing.assert_allclose(peak[:, 0], 1.0)

    # уровень считается от пика ЭТОГО удара: при пике 0.8 — 0.08, и доля
    # спада снова ровно 0.9 (при уровне от «типичного» пика 1.0 было бы 0.875)
    low = _run_detector(lambda t: np.array([_ap(t, 1.0, peak=0.8)]), 200.0)
    np.testing.assert_allclose(low.maps()[1][0, 0], 1 + 1 + 100 + 0.9 * 20, atol=1e-9)

    det50 = _run_detector(lambda t: np.array([_ap(t, tu) for tu in t_up]), 200.0,
                          apd_level=0.5)
    np.testing.assert_allclose(det50.maps()[1][:, 0], t_up + 1 + 100 + 0.5 * 20, atol=1e-9)


def test_two_beats_are_separated():
    def sig(t):
        return np.array([_ap(t, 1.0) + _ap(t, 301.0), _ap(t, 1.0)])

    det = _run_detector(sig, 500.0)
    act, rep, _ = det.maps()
    assert det.n_beats == 2
    np.testing.assert_allclose(act[0], [1.5, 301.5], atol=1e-9)
    assert np.isnan(act[1, 1]) and np.isnan(rep[1, 1]), "второй узел бился один раз"


def test_noise_on_plateau_and_subthreshold_bumps_do_not_count():
    rng = np.random.default_rng(0)

    def sig(t):
        noise = 0.05 * rng.standard_normal()        # шум вокруг плато
        bump = 0.45 * np.exp(-((t - 200.0) / 2.0) ** 2)   # подпороговый всплеск
        return np.array([_ap(t, 1.0) + (noise if 5 < t < 100 else 0.0), bump])

    det = _run_detector(sig, 300.0)
    act, _, _ = det.maps(2)
    assert np.isfinite(act[0, 0]) and np.isnan(act[0, 1]), "шум на плато — не новый удар"
    assert np.isnan(act[1]).all(), "подпороговый всплеск — не активация"


def test_node_active_at_start_has_unknown_activation():
    """Продолжение посреди ПД: удар есть, момент активации неизвестен."""
    det = _run_detector(lambda t: np.array([_ap(t, -50.0), _ap(t, 10.0)]), 200.0)
    act, rep, _ = det.maps()
    assert np.isnan(act[0, 0]) and np.isfinite(rep[0, 0])
    np.testing.assert_allclose(rep[0, 0], -50 + 1 + 100 + 18, atol=1e-9)
    np.testing.assert_allclose(act[1, 0], 10.5, atol=1e-9)


def test_unfinished_beat_keeps_peak():
    det = _run_detector(lambda t: np.array([_ap(t, 10.0, peak=0.9)]), 50.0)
    act, rep, peak = det.maps()
    assert np.isfinite(act[0, 0]) and np.isnan(rep[0, 0])
    np.testing.assert_allclose(peak[0, 0], 0.9)


# ═══════════════════════════════════════════════════════════════════════
#  СИНТЕТИЧЕСКИЙ ПРОГОН
# ═══════════════════════════════════════════════════════════════════════

NX, NY, LX, LY = 30, 10, 6.0, 2.0
CV_TRUE = 0.4          # мм/мс
MX, MY = 6, 2


def _nodes(nx, ny, lx, ly):
    x, y = np.meshgrid(np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1))
    return np.column_stack([x.ravel(), y.ravel()])


def _cells(nx, ny, lx, ly):
    x, y = np.meshgrid((np.arange(nx) + 0.5) * lx / nx, (np.arange(ny) + 0.5) * ly / ny)
    return np.column_stack([x.ravel(), y.ravel()])


def _make_run(root: Path, name="r", t_max=120.0) -> Path:
    """
    Папка прогона с аналитическими данными:
      * плоская волна вдоль x со скоростью CV_TRUE, два удара (1 и 301 мс);
      * APD = 200 мс в регионе 1 (x < 3) и 250 мс в базовой ткани;
      * во втором ударе узлы с x > 4.5 не возбуждаются (блок);
      * ряд механики с известными пиками.
    """
    out = root / name
    (out / "snapshots").mkdir(parents=True)
    meshes = {"electric": {"nx": NX, "ny": NY, "lx_mm": LX, "ly_mm": LY,
                           "cell_type": "quadrilateral", "n_cells": NX * NY},
              "mechanical": {"nx": MX, "ny": MY, "lx_mm": LX, "ly_mm": LY,
                             "cell_type": "quadrilateral", "n_cells": MX * MY}}
    config = {"tissue_base": {"active": {"t_max": t_max}}, "mesh": meshes,
              "regions": [{"shape": "rect", "name": "left"}]}
    (out / "run.json").write_text(json.dumps({
        "status": "finished", "config": config, "meshes": meshes,
        "warnings": ["пример"]}), encoding="utf-8")

    xy = _nodes(NX, NY, LX, LY)
    region = (xy[:, 0] < 3.0 - 1e-9).astype(np.int32)
    apd = np.where(region == 1, 200.0, 250.0)
    act = np.column_stack([1.0 + xy[:, 0] / CV_TRUE, 301.0 + xy[:, 0] / CV_TRUE])
    act[xy[:, 0] > 4.5 + 1e-9, 1] = np.nan
    rep = act + apd[:, None]
    np.savez(out / "activation.npz", coords=xy, region=region, act=act, repol=rep,
             peak=np.ones_like(act), threshold=0.5, apd_level=0.9, v_rest=0.0,
             t_start_ms=0.0, t_end_ms=600.0, dt_ms=DT)

    t = np.arange(0.0, 11.0)
    tact = t_max / 4 * np.exp(-((t - 4.0) / 2.0) ** 2)
    sigma = 5.0 + tact
    lam = 1.2575 - 0.01 * tact / t_max
    cols = {"t_ms": t, "mech_index": np.r_[np.nan, t[1:]],
            "u_min": 0 * t, "u_max": 0 * t + 1,
            "t_act_electric_max": tact, "t_act_electric_integral": tact * 12,
            "t_act_mech_max": tact, "t_act_mech_integral": tact * 12,
            "lambda_f_min": lam, "lambda_f_max": 0 * t + 1.2575,
            "J_min": 0 * t + 1, "J_max": 0 * t + 1,
            "sigma_xx_min": sigma, "sigma_xx_max": sigma,
            "newton_iterations": 0 * t + 3}
    header = ",".join(cols)
    rows = np.column_stack(list(cols.values()))
    lines = [header] + [",".join("" if np.isnan(v) else repr(float(v)) for v in r) for r in rows]
    (out / "series.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    m_c = _nodes(MX, MY, LX, LY)
    m_cc = _cells(MX, MY, LX, LY)
    e_cc = _cells(NX, NY, LX, LY)
    np.savez(out / "snapshots" / "snap_t00000005.000ms.npz",
             t_ms=5.0, requested_ms=np.array([4.9]),
             e_coords=xy, e_state=np.column_stack([xy[:, 0], xy[:, 1]]),
             e_state_names=np.array(["u", "v"]), e_region=region,
             e_cell_coords=e_cc, e_t_act=e_cc[:, 0],
             m_coords=m_c, m_u=0.1 * m_c, m_cell_coords=m_cc,
             m_t_act=m_cc[:, 0], m_lambda_f=1 + m_cc[:, 1],
             m_J=np.ones(len(m_cc)), m_sigma_xx=m_cc[:, 0],
             m_region=(m_cc[:, 0] < 3).astype(np.int32))
    return out


def test_run_reader_basics():
    with tempfile.TemporaryDirectory() as d:
        run = open_run(_make_run(Path(d)) / "run.json")
        assert run.status == "finished" and run.warnings == ["пример"]
        assert run.param("tissue_base.active.t_max") == 120.0
        s = run.series()
        assert len(s) == 11 and np.isnan(s["mech_index"][0])
        assert run.snapshot_times == [5.0]
        try:
            run.snapshot(6.0)
        except KeyError as e:
            assert "5.0" in str(e)
        else:
            raise AssertionError("несуществующий снимок должен давать KeyError")


def test_snapshot_grids():
    with tempfile.TemporaryDirectory() as d:
        run = open_run(_make_run(Path(d)))
        snap = run.snapshot(5.0)
        u = snap.grid("e_state", "u")
        assert u.shape == (NY + 1, NX + 1)
        np.testing.assert_allclose(u[0], np.linspace(0, LX, NX + 1))
        np.testing.assert_allclose(snap.grid("e_state", "v")[:, 0], np.linspace(0, LY, NY + 1))
        assert snap.grid("m_u", 0).shape == (MY + 1, MX + 1)
        lam = snap.grid("m_lambda_f")
        assert lam.shape == (MY, MX)
        np.testing.assert_allclose(lam[:, 0], 1 + (np.arange(MY) + 0.5) * LY / MY)
        assert snap.grid("e_t_act").shape == (NY, NX)
        assert list(run.snapshots())[0].t_ms == 5.0


def test_conduction_velocity_of_planar_wave():
    with tempfile.TemporaryDirectory() as d:
        maps = open_run(_make_run(Path(d))).activation()
    assert maps.n_beats == 2
    cv = bm.conduction_velocity(maps.coords, maps.act[:, 0])
    assert abs(cv - CV_TRUE) < 1e-9

    field = bm.cv_field(maps.grid(maps.act[:, 0]), LX / NX, LY / NY)
    np.testing.assert_allclose(field, CV_TRUE, rtol=1e-9)

    # поперёк волны скорость «бесконечна» — наклон нулевой
    assert np.isnan(bm.conduction_velocity(maps.coords, 0 * maps.act[:, 0], axis=1)) \
        or np.isinf(bm.conduction_velocity(maps.coords, 0 * maps.act[:, 0], axis=1))


def test_oblique_wave_cv_field():
    """Волна под углом 30°: локальная скорость — та же по модулю."""
    xy = _nodes(40, 40, 4.0, 4.0)
    ang = np.deg2rad(30)
    t = (xy[:, 0] * np.cos(ang) + xy[:, 1] * np.sin(ang)) / 0.25
    field = bm.cv_field(t.reshape(41, 41), 0.1, 0.1)
    np.testing.assert_allclose(field, 0.25, rtol=1e-9)


def test_activation_summary_and_block():
    with tempfile.TemporaryDirectory() as d:
        maps = open_run(_make_run(Path(d))).activation()
    s0 = bm.activation_summary(maps.act[:, 0])
    assert s0["fraction_activated"] == 1.0
    assert abs(s0["total_activation_time_ms"] - LX / CV_TRUE) < 1e-9

    blk = bm.conduction_block(maps.act, beat=0)
    x = maps.coords[:, 0]
    np.testing.assert_array_equal(blk["blocked_mask"], x > 4.5 + 1e-9)
    assert blk["n_blocked"] == int((x > 4.5 + 1e-9).sum())
    assert bm.conduction_block(maps.act, beat=1)["n_blocked"] == int((x <= 4.5 + 1e-9).sum()), \
        "после последнего удара «следующего» нет — все его узлы считаются заблокированными"


def test_apd_by_region():
    with tempfile.TemporaryDirectory() as d:
        maps = open_run(_make_run(Path(d))).activation()
    s = bm.apd_summary(maps.apd[:, 0], maps.region)
    assert s["min_ms"] == 200.0 and s["max_ms"] == 250.0 and s["dispersion_ms"] == 50.0
    assert s["by_region"][1]["mean"] == 200.0 and s["by_region"][0]["mean"] == 250.0


def test_restitution_pairs():
    with tempfile.TemporaryDirectory() as d:
        maps = open_run(_make_run(Path(d))).activation()
    r = bm.restitution(maps.act, maps.repol)
    n2 = int(np.isfinite(maps.act[:, 1]).sum())
    assert len(r["di_ms"]) == n2
    x = maps.coords[r["node"], 0]
    apd = np.where(x < 3.0 - 1e-9, 200.0, 250.0)
    np.testing.assert_allclose(r["di_ms"], 300.0 - apd)
    np.testing.assert_allclose(r["apd_ms"], apd)
    assert (r["beat"] == 1).all()


def test_mechanics_summary():
    with tempfile.TemporaryDirectory() as d:
        s = open_run(_make_run(Path(d))).series()
    m = bm.mechanics_summary(s)
    assert m["peak_t_act_kpa"] == 30.0 and m["time_to_peak_t_act_ms"] == 4.0
    assert m["peak_sigma_xx_kpa"] == 35.0 and m["sigma_rise_kpa"] == 35.0 - s["sigma_xx_max"][0]
    np.testing.assert_allclose(m["min_lambda_f"], 1.2575 - 0.01 / 4)
    np.testing.assert_allclose(m["max_shortening"], 1 - m["min_lambda_f"] / s["lambda_f_min"][0])
    t = s["t_ms"]
    np.testing.assert_allclose(m["t_act_time_integral_kpa_mm2_ms"],
                               np.trapezoid(s["t_act_mech_integral"], t)
                               if hasattr(np, "trapezoid") else np.trapz(s["t_act_mech_integral"], t))
    w = bm.mechanics_summary(s, t_from_ms=6.0)
    assert w["time_to_peak_t_act_ms"] == 0.0


def test_sweep_table():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "sw"
        root.mkdir()
        _make_run(root, "run_000", t_max=60.0)
        _make_run(root, "run_001", t_max=120.0)
        (root / "sweep.json").write_text(json.dumps({
            "parameters": {"tissue_base.active.t_max": [60, 120, 180]},
            "points": [
                {"name": "run_000", "overrides": {"tissue_base.active.t_max": 60},
                 "out_dir": "/moved/elsewhere/run_000", "status": "finished"},
                {"name": "run_001", "overrides": {"tissue_base.active.t_max": 120},
                 "out_dir": "/moved/elsewhere/run_001", "status": "finished"},
                {"name": "run_002", "overrides": {"tissue_base.active.t_max": 180},
                 "out_dir": "/x", "status": "failed"}]}), encoding="utf-8")
        sweep = open_sweep(root)
        rows = sweep.table(lambda run: bm.mechanics_summary(run.series()))
    assert [r["tissue_base.active.t_max"] for r in rows] == [60, 120], \
        "неудавшиеся точки не попадают; перенесённая папка находится"
    assert [r["peak_t_act_kpa"] for r in rows] == [15.0, 30.0]


def test_plots_render():
    try:
        import matplotlib
    except ImportError:
        return
    matplotlib.use("Agg")
    from cardiac_em.analysis import plots

    with tempfile.TemporaryDirectory() as d:
        run = open_run(_make_run(Path(d)))
        fig, ax = plots.series(run, ["t_act_mech_max", "sigma_xx_max"])
        assert len(ax.lines) == 2
        maps = run.activation()
        fig2, _ = plots.field_map(maps.grid(maps.act[:, 0]), run.meshes["electric"],
                                  title="t_act")
        fig2.savefig(Path(d) / "map.png")
        assert (Path(d) / "map.png").stat().st_size > 1000


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
