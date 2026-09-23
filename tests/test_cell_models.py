"""
Тесты моделей клетки (Шаг 5).
==============================

DOLFINx НЕ ТРЕБУЕТСЯ — чистые ОДУ на numpy.

    python tests/test_cell_models.py
    pytest tests/test_cell_models.py -v

Проверяется три вещи:

  1. ФИЗИОЛОГИЯ конкретной модели: покой устойчив, есть порог
     возбуждения, длительность ПД в разумных пределах, после ПД клетка
     возвращается к покою, возбудимость восстанавливается постепенно.
     Числа взяты не с потолка — они измерены на этой же модели и
     согласуются с заявленными в исходном коде (~240 мс).

  2. ИНТЕГРАТОР базового класса, в первую очередь схема Раша–Ларсена.
     У Роджерса–МакКаллоха воротных переменных нет, поэтому эта ветка
     осталась бы непроверенной до появления подробной ионной модели.
     Чтобы она не была непроверенным кодом, здесь заведена
     синтетическая модель с одним воротом и известным точным решением.

  3. КОНТРАКТ абстракции: формы массивов, понятные ошибки при
     рассогласовании, независимость узлов друг от друга.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.models.cell import (  # noqa: E402
    CELL_MODELS,
    CellModel,
    RogersMcCullochModel,
    available_models,
    make_cell_model,
)

# Рабочий шаг солвера. Характеристики модели от него практически не
# зависят (APD90 = 222.7 мс при dt = 0.05 против 222.8 мс при dt = 0.01),
# поэтому тесты идут на том же шаге, что и расчёт: и ближе к реальному
# режиму, и впятеро быстрее.
DT = 0.05
PARAMS = {"AP_c1": 0.26, "AP_c2": 0.10, "AP_a": 0.13,
          "AP_b": 0.013, "AP_d": 1.0}


def _raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            f"ожидалось {exc_type.__name__}, получено {type(e).__name__}: {e}") from e
    raise AssertionError(f"ожидалось {exc_type.__name__}, но исключения не было")


def _run(model, y, duration_ms, amp=0.0, t_stim=None, stim_dur=2.0,
         dt=DT, params=PARAMS):
    """Проинтегрировать заданное время, вернуть (состояние, история u)."""
    n = int(round(duration_ms / dt))
    trace = np.empty(n)
    n_nodes = y.shape[0]
    for k in range(n):
        t = k * dt
        active = t_stim is not None and t_stim <= t < t_stim + stim_dur
        stim = np.full(n_nodes, amp if active else 0.0)
        y = model.step(t, y, dt, stim, params)
        trace[k] = y[0, 0]
    return y, trace


# ═══════════════════════════════════════════════════════════════════════
#  ФИЗИОЛОГИЯ: РОДЖЕРС–МАККАЛЛОХ
# ═══════════════════════════════════════════════════════════════════════

def test_rest_is_an_equilibrium():
    """
    Без стимула клетка обязана оставаться в покое. Если покой «ползёт»,
    дальше всё бессмысленно: тканевая задача будет заряжаться сама собой.
    """
    model = make_cell_model("rogers_mcculloch")
    y = model.initial_state(1)
    y, trace = _run(model, y, 200.0)

    assert np.max(np.abs(trace)) < 1e-12
    np.testing.assert_allclose(y[0], model.resting_state(), atol=1e-12)


def test_subthreshold_stimulus_does_not_excite():
    model = make_cell_model("rogers_mcculloch")
    _, trace = _run(model, model.initial_state(1), 200.0, amp=0.05, t_stim=1.0)
    assert trace.max() < 0.2, f"подпороговый стимул вызвал ответ: u_max={trace.max():.3f}"


def test_suprathreshold_stimulus_excites():
    model = make_cell_model("rogers_mcculloch")
    _, trace = _run(model, model.initial_state(1), 100.0, amp=0.5, t_stim=1.0)
    assert trace.max() > 0.9, f"надпороговый стимул не вызвал ПД: u_max={trace.max():.3f}"


def test_threshold_lies_between_measured_bounds():
    """
    Порог существует и лежит между 0.05 и 0.10 при длительности 2 мс.
    Тест фиксирует само НАЛИЧИЕ порога: монотонность отклика по
    амплитуде с резким переходом.
    """
    model = make_cell_model("rogers_mcculloch")
    peaks = {}
    for amp in (0.05, 0.10):
        _, trace = _run(model, model.initial_state(1), 200.0,
                        amp=amp, t_stim=1.0)
        peaks[amp] = trace.max()

    assert peaks[0.05] < 0.2
    assert peaks[0.10] > 0.7
    assert peaks[0.10] > 5 * peaks[0.05], "переход через порог должен быть резким"


def test_action_potential_duration_is_physiological():
    """
    APD90 ≈ 223 мс для этих параметров — согласуется с заявленными в
    исходном коде «≈ 240 мс». Границы взяты с запасом, чтобы тест ловил
    грубые ошибки (перепутанные параметры, неверный знак), а не мигал
    от мелких изменений схемы.
    """
    model = make_cell_model("rogers_mcculloch")
    _, trace = _run(model, model.initial_state(1), 400.0, amp=0.5, t_stim=1.0)

    u_max = trace.max()
    above = np.flatnonzero(trace > 0.1 * u_max)
    apd90 = (above[-1] - above[0]) * DT

    assert 180.0 < apd90 < 280.0, f"APD90 = {apd90:.1f} мс вне разумного диапазона"


def test_cell_returns_to_rest_after_action_potential():
    model = make_cell_model("rogers_mcculloch")
    y, _ = _run(model, model.initial_state(1), 500.0, amp=0.5, t_stim=1.0)
    assert abs(y[0, 0]) < 1e-3, f"u не вернулось к покою: {y[0, 0]:.2e}"
    assert abs(y[0, 1]) < 2e-2, f"v не вернулось к покою: {y[0, 1]:.2e}"


def test_excitability_recovers_gradually():
    """
    Рефрактерность: околопороговый второй стимул блокируется при
    коротком интервале сцепления и проходит при длинном, а амплитуда
    ответа восстанавливается монотонно.

    Измерено: блок на 250 мс, ответ на 300 мс, насыщение к уровню
    покоящейся клетки (~0.85) к 600 мс.

    Все интервалы сцепления считаются ОДНИМ прогоном: узлы независимы,
    поэтому каждый интервал — просто свой узел со своим расписанием
    стимула. Четыре последовательных интегрирования превращаются в одно
    (7.8 с → 0.7 с), а проверяемое поведение не меняется.
    """
    model = make_cell_model("rogers_mcculloch")
    s2_amp = 0.12                 # чуть выше порога покоящейся клетки
    gaps = np.array([250.0, 300.0, 400.0, 600.0])
    n = len(gaps)

    y = model.initial_state(n)
    peaks = np.zeros(n)
    for k in range(int((gaps.max() + 300.0) / DT)):
        t = k * DT
        s1 = 0.5 if 1.0 <= t < 3.0 else 0.0
        s2 = np.where((gaps <= t) & (t < gaps + 2.0), s2_amp, 0.0)
        y = model.step(t, y, DT, np.full(n, s1) + s2, PARAMS)
        peaks = np.where(t >= gaps, np.maximum(peaks, y[:, 0]), peaks)

    by_gap = dict(zip(gaps, peaks))
    assert by_gap[250.0] < 0.5, f"на 250 мс ожидался блок, вышло {by_gap[250.0]:.3f}"
    assert by_gap[300.0] > 0.6, f"на 300 мс ожидался ответ, вышло {by_gap[300.0]:.3f}"
    assert all(b >= a - 1e-9 for a, b in zip(peaks, peaks[1:])), (
        f"восстановление не монотонно: {peaks}")


def test_active_tension_shape_and_range():
    """
    T_act = T_max·u·(1−u): безразмерный множитель лежит в [0, 1/4],
    максимум при u = 1/2. Пиковое напряжение — T_max/4, как в исходном
    скрипте.
    """
    model = make_cell_model("rogers_mcculloch")
    assert model.tension_kind == "scaled"

    y = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.25, 0.0]])
    act = model.active_tension(y, PARAMS)

    np.testing.assert_allclose(act, [0.0, 0.25, 0.0, 0.1875])
    assert act.shape == (4,)


def test_heterogeneous_parameters_give_different_behaviour():
    """
    Параметры приходят массивами по узлам: узел с c1 = 0 («рубец») ведёт
    себя иначе соседнего здорового. Это то, ради чего параметры вынесены
    из модели наружу.

    Важно правильно сформулировать, ЧТО именно отличается. Стимул тянет
    потенциал напрямую, поэтому во время импульса деполяризуются оба
    узла — c1 = 0 убирает не это, а РЕГЕНЕРАТИВНЫЙ ток. Без него клетка
    не удерживает плато и спадает сразу после окончания импульса:
    к 100 мс здоровая держит ~0.6, рубец — уже ~0.08.
    """
    model = make_cell_model("rogers_mcculloch")
    params = dict(PARAMS)
    params["AP_c1"] = np.array([0.26, 0.0])       # здоровая клетка и рубец

    y = model.initial_state(2)
    snapshots: dict[int, np.ndarray] = {}
    for k in range(int(150.0 / DT)):
        t = k * DT
        stim = np.full(2, 0.5 if 1.0 <= t < 3.0 else 0.0)
        y = model.step(t, y, DT, stim, params)
        for mark in (3, 100):
            if k == int(mark / DT):
                snapshots[mark] = y[:, 0].copy()

    # во время импульса стимул дотягивает оба узла — это ожидаемо
    assert snapshots[3][0] > 0.9
    assert snapshots[3][1] > 0.9

    # а вот плато держит только узел с регенеративным током
    healthy, scar = snapshots[100]
    assert healthy > 0.4, f"здоровая клетка должна держать плато, вышло {healthy:.3f}"
    assert scar < 0.2, f"узел с c1 = 0 не должен держать плато, вышло {scar:.3f}"
    assert healthy > 3 * scar


def test_nodes_are_independent():
    """
    Узлы не связаны между собой: диффузия — забота солвера, а не модели.
    Прогон одного узла и одного из многих должен совпадать до битов.
    """
    model = make_cell_model("rogers_mcculloch")

    y1 = model.initial_state(1)
    y1, trace1 = _run(model, y1, 100.0, amp=0.5, t_stim=1.0)

    y5 = model.initial_state(5)
    y5, trace5 = _run(model, y5, 100.0, amp=0.5, t_stim=1.0)

    np.testing.assert_allclose(trace1, trace5, atol=0.0)
    for i in range(5):
        np.testing.assert_allclose(y5[i], y1[0], atol=0.0)


# ═══════════════════════════════════════════════════════════════════════
#  ИНТЕГРАТОР: РАШ–ЛАРСЕН
# ═══════════════════════════════════════════════════════════════════════

class _OneGateModel(CellModel):
    """
    Синтетическая модель для проверки схемы Раша–Ларсена.

    Одна воротная переменная с ПОСТОЯННЫМИ g_∞ и τ:

        dg/dt = (g_∞ − g)/τ      →      g(t) = g_∞ + (g₀ − g_∞)·e^{−t/τ}

    Точное решение известно, поэтому схему можно проверить не «на
    глаз», а сравнением с аналитикой при любом шаге, включая dt ≫ τ,
    где явный Эйлер разваливается.
    """

    name = "synthetic_one_gate"
    state_names = ("V", "g")
    v_index = 0
    gate_indices = (1,)
    param_names = ("g_inf", "tau")
    tension_kind = "scaled"

    def resting_state(self) -> np.ndarray:
        return np.array([0.0, 0.0])

    def rhs_non_gate(self, t, y, stim, params):
        return np.zeros((len(y), 1))         # V заморожен

    def gate_inf_tau(self, t, y, params):
        n = len(y)
        return (np.full((n, 1), params["g_inf"]),
                np.full((n, 1), params["tau"]))

    def active_tension(self, y, params):
        return np.zeros(len(y))


def test_rush_larsen_is_exact_for_constant_gate():
    """
    При постоянных g_∞ и τ схема Раша–Ларсена даёт ТОЧНОЕ решение —
    композиция точных шагов остаётся точной.
    """
    model = _OneGateModel()
    params = {"g_inf": 0.8, "tau": 5.0}

    y = model.initial_state(1)
    dt, n_steps = 1.0, 20
    for k in range(n_steps):
        y = model.step(k * dt, y, dt, np.zeros(1), params)

    t_end = dt * n_steps
    exact = 0.8 + (0.0 - 0.8) * math.exp(-t_end / 5.0)
    assert math.isclose(float(y[0, 1]), exact, rel_tol=1e-14)


def test_rush_larsen_stable_where_euler_would_diverge():
    """
    Смысл схемы: шаг много больше постоянной времени. Явный Эйлер дал бы
    множитель (1 − dt/τ) = −9 за шаг и разлетелся бы; Раш–Ларсен
    спокойно сходится к g_∞.
    """
    model = _OneGateModel()
    params = {"g_inf": 0.8, "tau": 0.1}       # τ на порядок меньше шага

    y = model.initial_state(1)
    dt = 1.0
    for k in range(10):
        y = model.step(k * dt, y, dt, np.zeros(1), params)

    g = float(y[0, 1])
    assert math.isclose(g, 0.8, abs_tol=1e-12), f"ожидалось насыщение к g_∞, вышло {g}"

    # для сравнения: тот же шаг явным Эйлером
    g_euler = 0.0
    for _ in range(10):
        g_euler = g_euler + dt * (0.8 - g_euler) / 0.1
    assert abs(g_euler) > 1e6, "контрольный расчёт: Эйлер должен разойтись"


def test_rush_larsen_relaxes_monotonically():
    model = _OneGateModel()
    params = {"g_inf": 1.0, "tau": 3.0}

    y = model.initial_state(1)
    values = []
    for k in range(50):
        y = model.step(k * 0.5, y, 0.5, np.zeros(1), params)
        values.append(float(y[0, 1]))

    assert all(b > a for a, b in zip(values, values[1:])), "релаксация не монотонна"
    assert values[-1] < 1.0
    assert math.isclose(values[-1], 1.0, abs_tol=1e-3)


def test_gate_indices_and_non_gate_split():
    model = _OneGateModel()
    assert model.gate_indices == (1,)
    assert model.non_gate_indices == (0,)
    assert model.n_states == 2

    rm = RogersMcCullochModel()
    assert rm.gate_indices == ()
    assert rm.non_gate_indices == (0, 1)


def test_missing_gate_implementation_is_reported():
    """Объявил воротные переменные — обязан определить gate_inf_tau."""

    class _Broken(CellModel):
        name = "broken"
        state_names = ("V", "g")
        gate_indices = (1,)

        def resting_state(self):
            return np.zeros(2)

        def rhs_non_gate(self, t, y, stim, params):
            return np.zeros((len(y), 1))

        def active_tension(self, y, params):
            return np.zeros(len(y))

    model = _Broken()
    _raises(NotImplementedError, model.step, 0.0, model.initial_state(1),
            0.1, np.zeros(1), {})


def test_non_positive_tau_is_reported():
    model = _OneGateModel()
    _raises(FloatingPointError, model.step, 0.0, model.initial_state(1),
            0.1, np.zeros(1), {"g_inf": 0.5, "tau": 0.0})


# ═══════════════════════════════════════════════════════════════════════
#  КОНТРАКТ АБСТРАКЦИИ
# ═══════════════════════════════════════════════════════════════════════

def test_state_shapes():
    model = make_cell_model("rogers_mcculloch")
    assert model.n_states == 2
    assert model.state_names == ("u", "v")
    assert model.resting_state().shape == (2,)
    assert model.initial_state(7).shape == (7, 2)


def test_state_index_lookup():
    model = make_cell_model("rogers_mcculloch")
    assert model.state_index("u") == 0
    assert model.state_index("v") == 1
    _raises(KeyError, model.state_index, "Ca_i")


def test_wrong_state_shape_is_reported():
    model = make_cell_model("rogers_mcculloch")
    _raises(ValueError, model.step, 0.0, np.zeros((5, 3)), DT,
            np.zeros(5), PARAMS)
    _raises(ValueError, model.step, 0.0, np.zeros(5), DT,
            np.zeros(5), PARAMS)


def test_step_does_not_mutate_input():
    """Шаг возвращает новый массив: исходное состояние должно уцелеть."""
    model = make_cell_model("rogers_mcculloch")
    y = model.initial_state(3)
    y[:, 0] = 0.4
    before = y.copy()
    model.step(0.0, y, DT, np.zeros(3), PARAMS)
    np.testing.assert_array_equal(y, before)


def test_check_params_reports_missing():
    model = make_cell_model("rogers_mcculloch")
    model.check_params(PARAMS, n_nodes=10)          # полный набор — ок

    incomplete = {k: v for k, v in PARAMS.items() if k != "AP_a"}
    _raises(KeyError, model.check_params, incomplete, 10)


def test_check_params_reports_short_array():
    model = make_cell_model("rogers_mcculloch")
    params = dict(PARAMS)
    params["AP_c1"] = np.zeros(5)
    _raises(ValueError, model.check_params, params, 10)


def test_state_bounds_are_applied():
    """Ограничение на u должно срабатывать, иначе явная схема разгоняется."""
    model = make_cell_model("rogers_mcculloch")
    lo, hi = model.state_bounds[0]

    y = model.initial_state(1)
    y[0, 0] = 0.5
    y = model.step(0.0, y, 1.0, np.array([1e3]), PARAMS)   # заведомый перебор
    assert y[0, 0] <= hi + 1e-12


def test_registry():
    assert "rogers_mcculloch" in available_models()
    assert CELL_MODELS["rogers_mcculloch"] is RogersMcCullochModel
    assert isinstance(make_cell_model("rogers_mcculloch"), RogersMcCullochModel)
    _raises(KeyError, make_cell_model, "нет_такой_модели")


def test_describe_renders():
    text = make_cell_model("rogers_mcculloch").describe()
    assert "rogers_mcculloch" in text and "u" in text


def test_calcium_absent_by_default():
    """У феноменологической модели кальция нет — это законно."""
    model = make_cell_model("rogers_mcculloch")
    assert model.calcium(model.initial_state(3)) is None


# ═══════════════════════════════════════════════════════════════════════
#  ЗАПУСК БЕЗ PYTEST
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
