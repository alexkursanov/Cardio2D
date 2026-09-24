"""
Тесты модели TNNPM без DOLFINx (Шаг 11).
=========================================

    python tests/test_tnnpm.py
    pytest tests/test_tnnpm.py -v

Проверка в три слоя:

1. ПЕРЕНОС ФОРМУЛ. Правые части векторной модели сравниваются с
   эталоном tests/tnnpm_reference.py — это model.py автора модели с
   минимальными правками (удалены двухкомпартментный СР и блок CaMKII).
   Совпадение до ~1e-9 во множестве состояний, включая все ветви
   механики и ишемические параметры, означает, что формулы перенесены
   без ошибок.

2. ИНТЕГРИРОВАНИЕ. Один удар при четырёх стадиях ишемии (сценарий
   автора: K_o, ATP_i, KmATP) против эталонных чисел, полученных
   жёстким интегратором (solve_ivp, BDF, rtol 1e-8) по РАБОЧЕЙ версии
   модели автора — независимой от этого пакета.

3. ВСТРАИВАНИЕ. Параметры по регионам (cell:…), обрезка потенциала.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardiac_em.config import (  # noqa: E402
    RectRegion,
    SimulationConfig,
)
from cardiac_em.control import apply_overrides  # noqa: E402
from cardiac_em.models.cell import make_cell_model  # noqa: E402
from tnnpm_reference import TNNPM as Reference  # noqa: E402

DT = 0.05

# Стадии ишемии из сценария автора; эталон — рабочая версия model.py,
# solve_ivp BDF (rtol 1e-8, atol 1e-10), стимул 52 пА/пФ на [10, 11] мс.
STAGES = {
    #          K_o   ATP_i KmATP   APD до −60 мВ  V(700 мс)  пик F, мН
    "норма":  (5.4,  6.8,  0.09,   254.9,        -85.77,    37.97),
    "5 мин":  (6.2,  6.0,  0.30,   236.2,        -82.29,    35.91),
    "10 мин": (8.0,  4.5,  0.35,   206.6,        -75.75,    31.93),
    "15 мин": (9.4,  4.0,  0.38,   188.4,        -71.56,    28.37),
}


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


def _params(model, n, **arrays):
    p = {k: np.full(n, v) for k, v in model.default_params().items()}
    for k, v in arrays.items():
        p[k] = np.asarray(v, dtype=np.float64)
    return p


def _full_rhs(model, y, t, stim, params):
    """Полный вектор производных векторной модели (ворота — как (g∞−g)/τ)."""
    Y = y[None, :]
    inf, tau = model.gate_inf_tau(t, Y, params)
    d = np.empty(model.n_states)
    gi = list(model.gate_indices)
    d[gi] = ((inf - Y[:, gi]) / tau)[0]
    d[list(model.non_gate_indices)] = model.rhs_non_gate(t, Y, np.array([stim]), params)[0]
    return d


# ═══════════════════════════════════════════════════════════════════════
#  1. ПЕРЕНОС ФОРМУЛ
# ═══════════════════════════════════════════════════════════════════════

def test_interface():
    m = make_cell_model("tnnpm")
    assert m.n_states == 29 and len(m.gate_indices) == 12
    assert m.state_names[m.v_index] == "V"
    assert set(m.default_params()) == set(m.param_names)
    assert {"ATP_i", "KmATP", "K_o", "K_mNa", "g_Na", "g_CaL"} <= set(m.param_names)
    assert m.potential_clip is None, "в мВ обрезать потенциал нельзя"
    assert m.activation_threshold == -40.0
    _raises(KeyError, make_cell_model, "tnnpm", constants={"g_Naa": 1.0})


def test_initial_state_matches_reference():
    """Включая механику: деление пополам по l₂ при силе преднагрузки r0."""
    m = make_cell_model("tnnpm")
    ref = Reference().calculate_init_conditions()
    np.testing.assert_array_equal(m.resting_state(), ref)


def _states_to_check(m):
    """Состояния, покрывающие все ветви механики и разные фазы ПД."""
    rng = np.random.default_rng(0)
    rest = m.resting_state()
    ix = m.state_names.index
    vmax = m.c["v_max"]
    states = []
    # фазы ПД: разные V и кальций
    for V, ca_i, ca_ss in [(-85.9, 4e-5, 1.6e-4), (-60.0, 1e-4, 3e-4), (-30.0, 3e-4, 5e-3),
                           (10.0, 6e-4, 0.05), (30.0, 9e-4, 0.2)]:
        y = rest.copy()
        y[ix("V")], y[ix("Ca_i")], y[ix("Ca_ss")] = V, ca_i, ca_ss
        states.append(y)
    # механика: v во всех ветвях p(v), q(v), G*(v); w по обе стороны от v
    for v in (-1.5 * vmax, -0.5 * vmax, 0.0, 0.05 * vmax, 0.5 * vmax, 0.98 * vmax, 1.2 * vmax):
        for dw in (-0.3 * vmax, 0.3 * vmax):
            y = rest.copy()
            y[ix("v")], y[ix("w")] = v, v + dw
            y[ix("N")], y[ix("A")] = 0.12, 0.03
            y[ix("l_1")] -= 0.01
            states.append(y)
    # случайные возмущения всего состояния
    for _ in range(40):
        y = rest * (1 + 0.05 * rng.standard_normal(m.n_states))
        y[ix("V")] = rng.uniform(-90, 40)
        states.append(y)
    return states


def test_rhs_matches_reference_in_all_branches():
    m = make_cell_model("tnnpm")
    ref = Reference()
    ref.calculate_init_conditions()
    params = _params(m, 1)
    worst = 0.0
    for y in _states_to_check(m):
        for t, stim in ((5.0, 0.0), (10.5, 52.0)):           # вне и внутри стимула
            r = np.array(ref.diff_equations(t, y))
            d = _full_rhs(m, y, t, stim, params)
            scale = np.maximum(np.abs(r), 1e-12 + 1e-9 * np.abs(y))
            worst = max(worst, float(np.max(np.abs(d - r) / scale)))
    assert worst < 1e-8, f"расхождение с эталоном {worst:.2e}"


def test_rhs_matches_reference_with_ischemic_parameters():
    """Региональные параметры доходят туда же, куда атрибуты эталона."""
    m = make_cell_model("tnnpm")
    ref = Reference()
    ref.calculate_init_conditions()
    ref.K_o, ref.ATPi, ref.KmATP, ref.K_mNa, ref.g_Na, ref.g_CaL = 9.4, 4.0, 0.38, 80.0, 10.0, 4e-5
    params = _params(m, 1, K_o=[9.4], ATP_i=[4.0], KmATP=[0.38], K_mNa=[80.0],
                     g_Na=[10.0], g_CaL=[4e-5])
    for y in _states_to_check(m)[:12]:
        r = np.array(ref.diff_equations(5.0, y))
        d = _full_rhs(m, y, 5.0, 0.0, params)
        scale = np.maximum(np.abs(r), 1e-12 + 1e-9 * np.abs(y))
        assert np.max(np.abs(d - r) / scale) < 1e-8


# ═══════════════════════════════════════════════════════════════════════
#  2. ИНТЕГРИРОВАНИЕ: ОДИН УДАР, ЧЕТЫРЕ СТАДИИ ИШЕМИИ
# ═══════════════════════════════════════════════════════════════════════

_BEAT = {}


def _beat():
    """Один удар 700 мс, четыре стадии в одном векторном прогоне (кэш)."""
    if _BEAT:
        return _BEAT
    m = make_cell_model("tnnpm")
    names = list(STAGES)
    P = _params(m, 4, K_o=[STAGES[n][0] for n in names],
                ATP_i=[STAGES[n][1] for n in names], KmATP=[STAGES[n][2] for n in names])
    y = m.initial_state(4)
    ix = m.state_names.index
    V, F, T_act, ts = [], [], [], []
    for k in range(int(round(700.0 / DT))):
        t = k * DT
        y = m.step(t, y, DT, np.full(4, 52.0 if 10.0 <= t < 11.0 else 0.0), P)
        V.append(y[:, ix("V")].copy())
        F.append(m.force(y))
        T_act.append(m.active_tension(y, P))
        ts.append(t + DT)
    _BEAT.update(names=names, V=np.array(V), F=np.array(F), T_act=np.array(T_act),
                 t=np.array(ts), y_end=y, model=m)
    return _BEAT


def test_beat_matches_stiff_reference_at_all_ischemia_stages():
    b = _beat()
    for n, name in enumerate(b["names"]):
        _, _, _, apd_ref, v_end_ref, f_ref = STAGES[name]
        V, t = b["V"][:, n], b["t"]
        up = np.argmax(V > -40)
        k = up + np.argmax((V[up:] < -60) & (t[up:] > t[up] + 5))
        apd = t[k] - t[up]
        assert abs(apd - apd_ref) / apd_ref < 0.005, f"{name}: APD {apd:.1f} против {apd_ref}"
        assert abs(V[-1] - v_end_ref) < 0.05, f"{name}: V(700) {V[-1]:.3f} против {v_end_ref}"
        f = b["F"][:, n].max()
        assert abs(f - f_ref) / f_ref < 0.01, f"{name}: пик силы {f:.2f} против {f_ref}"


def test_ischemia_shortens_apd_depolarizes_rest_and_weakens_force():
    b = _beat()
    V, t, F = b["V"], b["t"], b["F"]
    apd = []
    for n in range(4):
        up = np.argmax(V[:, n] > -40)
        k = up + np.argmax((V[up:, n] < -60) & (t[up:] > t[up] + 5))
        apd.append(t[k] - t[up])
    assert np.all(np.diff(apd) < 0), f"APD должен укорачиваться: {apd}"
    assert np.all(np.diff(V[-1]) > 0), "покой должен деполяризоваться"
    assert np.all(np.diff(F.max(axis=0)) < 0), "сила должна падать"


def test_active_tension_is_normalized_to_healthy_peak():
    """T_act/T_MAX ≈ 1 на пике здорового удара и ≈ 0 в покое."""
    b = _beat()
    ta = b["T_act"][:, 0]
    assert abs(ta.max() - 1.0) < 0.01
    assert abs(ta[0]) < 1e-3 and abs(ta[-1]) < 0.01
    m = b["model"]
    assert np.all(m.calcium(b["y_end"]) > 0)


# ═══════════════════════════════════════════════════════════════════════
#  3. ВСТРАИВАНИЕ В КОНФИГУРАЦИЮ
# ═══════════════════════════════════════════════════════════════════════

def test_region_cell_keys_are_accepted_and_separated():
    r = RectRegion(name="ишемия", x0=5, x1=10, y0=0, y1=10,
                   overrides={"cell:ATP_i": 4.0, "cell:KmATP": 0.38, "D_LONG": 0.1})
    assert r.cell_overrides == {"ATP_i": 4.0, "KmATP": 0.38}
    _raises(ValueError, RectRegion, x0=0, x1=1, y0=0, y1=1, overrides={"cell:": 1.0})
    _raises(ValueError, RectRegion, x0=0, x1=1, y0=0, y1=1, overrides={"ATP_i": 1.0},
            match="cell:")


def test_cell_params_roundtrip_and_overrides():
    cfg = SimulationConfig.default()
    cfg.cell_model = "tnnpm"
    cfg.regions = [RectRegion(name="isch", x0=5, x1=10, y0=0, y1=10,
                              overrides={"cell:ATP_i": 4.0})]
    new = apply_overrides(cfg, {"cell_params.K_o": 5.0,
                                "regions.0.overrides.cell:KmATP": 0.38})
    assert new.cell_params == {"K_o": 5.0}
    assert new.regions[0].cell_overrides == {"ATP_i": 4.0, "KmATP": 0.38}
    back = SimulationConfig.from_dict(new.to_dict())
    assert back.cell_params == {"K_o": 5.0} and "K_o=5" in back.summary()
    _raises(ValueError, SimulationConfig, mesh=cfg.mesh, cell_params={"K_o": "много"})


def test_cell_relax_setting():
    from cardiac_em.config import PreloadProtocol
    p = PreloadProtocol(cell_relax_ms=300.0)
    assert PreloadProtocol.from_dict(p.to_dict()).cell_relax_ms == 300.0
    assert PreloadProtocol.from_dict({}).cell_relax_ms == 0.0, "старые конфиги — без подготовки"
    _raises(ValueError, PreloadProtocol, cell_relax_ms=-1.0)


def test_rogers_mcculloch_keeps_its_potential_clip():
    rm = make_cell_model("rogers_mcculloch")
    assert rm.potential_clip == (0.0, 1.5) and rm.default_params() == {}


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
