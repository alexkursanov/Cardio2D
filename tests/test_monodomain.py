"""
Тесты монодоменного солвера (Шаг 6).
=====================================

ТРЕБУЮТ DOLFINx.

    pytest tests/test_monodomain.py -v
    mpirun -n 4 python -m pytest tests/test_monodomain.py -q

Главная проверка — СКОРОСТЬ ПРОВЕДЕНИЯ. Она хороша тем, что зависит
сразу от всего: от сборки матриц, от тензора диффузии, от расщепления,
от кинетики. Ошибка в любом из звеньев сдвигает скорость.

Эталон получен НЕЗАВИСИМЫМ методом — одномерной конечно-разностной
схемой на чистом numpy (см. `_reference_cv_1d` ниже). Это важно: если бы
ожидаемое значение считалось тем же кодом, который проверяется, тест
подтверждал бы сам себя. Согласованные числа:

    D = 0.300 мм²/мс  →  0.1165 мм/мс
    D = 0.075 мм²/мс  →  0.0581 мм/мс      (отношение 2.006 при ожидаемом 2)

Скорость масштабируется как √D и составляет 0.796 от классической
формулы √(kD/2)·(1−2a) для чистого бистабильного уравнения —
систематическая поправка на переменную восстановления, одинаковая при
всех проверенных D.

ЗАМЕЧАНИЕ ПО ПАРАМЕТРАМ. 0.116 мм/мс — это 0.116 м/с, тогда как
комментарии в исходном скрипте обещали 0.4–0.5 м/с, а физиологическая
продольная скорость в миокарде 0.3–0.7 м/с. Тесты проверяют фактическое
поведение при нынешних параметрах, а не цифру из комментария. Для
0.3 м/с понадобился бы D ≈ 2.0 мм²/мс либо c1 ≈ 1.7 /мс.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.config import (  # noqa: E402
    ConductionParams,
    RectRegion,
    RectangleMeshSpec,
    StimulusProtocol,
    TissueBaseParams,
)

pytestmark = pytest.mark.fem

DT = 0.05


# ═══════════════════════════════════════════════════════════════════════
#  НЕЗАВИСИМЫЙ ЭТАЛОН: ОДНОМЕРНЫЕ КОНЕЧНЫЕ РАЗНОСТИ
# ═══════════════════════════════════════════════════════════════════════

def _reference_cv_1d(d_long: float, c1: float = 0.26, c2: float = 0.10,
                     a: float = 0.13, b: float = 0.013, d: float = 1.0,
                     length_mm: float = 25.0, h: float = 0.15) -> float:
    """
    Скорость плоского фронта, посчитанная явной конечно-разностной схемой.

    Намеренно НЕ использует ни DOLFINx, ни код пакета: эталон должен
    быть получен другим способом, иначе он проверяет сам себя. Шаг по
    времени берётся из условия устойчивости явной схемы (λ = 0.2).

    Разрешение подобрано по компромиссу точность/время: при h = 0.15
    против h = 0.05 скорость меняется с 0.1162 на 0.1165 мм/мс (0.3 %),
    а расчёт идёт 0.5 с вместо 9 с.
    """
    dt = 0.2 * h ** 2 / d_long
    n = int(length_mm / h)
    x = np.arange(n) * h
    u = np.zeros(n)
    v = np.zeros(n)
    u[x < 1.0] = 1.0                      # затравка слева
    lam = d_long * dt / h ** 2

    t = 0.0
    arrival = np.full(n, np.nan)
    t_end = 2.0 * length_mm / (math.sqrt(c1 * d_long / 2) * (1 - 2 * a))

    for _ in range(int(t_end / dt)):
        lap = np.empty_like(u)
        lap[1:-1] = u[2:] - 2 * u[1:-1] + u[:-2]
        lap[0] = 2 * (u[1] - u[0])        # нулевой поток на концах
        lap[-1] = 2 * (u[-2] - u[-1])
        du = c1 * u * (u - a) * (1 - u) - c2 * u * v
        u_new = u + dt * du + lam * lap
        v = v + dt * b * (u - d * v)
        u = np.clip(u_new, 0.0, 1.1)
        t += dt
        fresh = np.isnan(arrival) & (u > 0.5)
        arrival[fresh] = t

    known = ~np.isnan(arrival)
    window = known & (x > 0.25 * length_mm) & (x < 0.75 * length_mm)
    if window.sum() < 10:
        raise RuntimeError("фронт не прошёл рабочее окно — увеличьте время")
    slope = np.polyfit(x[window], arrival[window], 1)[0]
    return 1.0 / slope


# ═══════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНОЕ
# ═══════════════════════════════════════════════════════════════════════

def _make_solver(spec, base=None, regions=(), stim=None, theta=0.5):
    from cardiac_em.fem import Tissue, build_mesh
    from cardiac_em.models.cell import make_cell_model
    from cardiac_em.solvers import MonodomainSolver

    mesh = build_mesh(spec)
    tissue = Tissue.for_electrics(mesh, base or TissueBaseParams(), regions)
    cell = make_cell_model("rogers_mcculloch")
    # Параметры подобраны так, чтобы зародыш был ЗАВЕДОМО надкритическим:
    # ширина фронта при D = 0.3 равна √(2D/c1) ≈ 1.5 мм, поэтому стимул
    # уже́ этой величины волну не запускает. Проверено одномерным расчётом
    # при h = 0.25…0.5 мм: потенциал после импульса ~1.09, волна доходит
    # до края во всех случаях.
    stim = stim or StimulusProtocol(times_ms=(1.0,), duration_ms=2.0,
                                    amplitude=5.0, rise_time_ms=0.2,
                                    decay_length_mm=1.0, x_max_mm=3.0)
    return MonodomainSolver(tissue, cell, stim, theta=theta)


def _measure_cv(solver, spec, t_end_ms, axis=0, lo_frac=0.39, hi_frac=0.72):
    """
    Скорость фронта по временам прихода: прогоняем, фиксируем для каждого
    узла момент, когда потенциал впервые превысил 0.5, и берём наклон
    зависимости «время прихода от координаты» в рабочем окне.

    ОКНО ВЫБРАНО НЕ ПРОИЗВОЛЬНО. Фронт выходит на асимптотическую
    скорость не сразу, а вблизи дальней границы с нулевым потоком
    РАЗГОНЯЕТСЯ. Измеренная в неудачном окне скорость завышена на
    десятки процентов, причём тем сильнее, чем шире фронт. Проверено
    одномерным расчётом: на области 12 мм с окном 5.4…10.2 мм
    получается 0.1464 вместо асимптотических 0.1165 (+26 %), а на
    области 18 мм с окном 7…13 мм — 0.1173 (+0.7 %).
    """
    from cardiac_em.fem import dof_coordinates

    coords = dof_coordinates(solver.P1)[:solver.n_owned, axis]
    arrival = np.full(solver.n_owned, np.nan)

    n_steps = int(round(t_end_ms / DT))
    for k in range(n_steps):
        t = k * DT
        solver.step(t, DT)
        u = solver.potential()[:solver.n_owned]
        fresh = np.isnan(arrival) & (u > 0.5)
        arrival[fresh] = t + DT

    comm = solver.comm
    all_x = np.concatenate(comm.allgather(coords))
    all_t = np.concatenate(comm.allgather(arrival))

    length = spec.lx_mm if axis == 0 else spec.ly_mm
    window = (~np.isnan(all_t)) & (all_x > lo_frac * length) & (all_x < hi_frac * length)
    if window.sum() < 10:
        raise RuntimeError(
            f"фронт не прошёл рабочее окно: покрыто {window.sum()} узлов")
    slope = np.polyfit(all_x[window], all_t[window], 1)[0]
    return 1.0 / slope


# ═══════════════════════════════════════════════════════════════════════
#  БАЗОВОЕ ПОВЕДЕНИЕ
# ═══════════════════════════════════════════════════════════════════════

def test_rest_state_stays_at_rest():
    """
    Без стимула решение обязано остаться нулевым. Если ткань заряжается
    сама — ошибка в сборке матриц или в знаке диффузионного члена.
    """
    spec = RectangleMeshSpec(nx=20, ny=20, lx_mm=10.0, ly_mm=10.0)
    solver = _make_solver(spec, stim=StimulusProtocol(times_ms=(1e6,)))

    for k in range(200):
        solver.step(k * DT, DT)

    lo, hi = solver.potential_range()
    assert abs(lo) < 1e-12 and abs(hi) < 1e-12, f"покой уехал: [{lo}, {hi}]"


def test_stimulus_depolarizes_the_stimulated_zone():
    """
    Диагностический тест: отделяет ЗАПУСК от РАСПРОСТРАНЕНИЯ.

    Сразу после импульса ткань в зоне стимула обязана быть
    деполяризована. Если этот тест зелёный, а тесты проведения красные —
    проблема в распространении; если красный и он — в подаче стимула.
    """
    spec = RectangleMeshSpec(nx=32, ny=6, lx_mm=12.0, ly_mm=2.0)
    solver = _make_solver(spec)

    for k in range(int(4.0 / DT)):        # импульс идёт с 1 до 3 мс
        solver.step(k * DT, DT)

    lo, hi = solver.potential_range()
    assert hi > 0.9, f"зона стимула не деполяризовалась: максимум {hi:.3f}"
    assert lo >= -1e-9, f"потенциал ушёл в минус: {lo:.3e}"


def test_stimulus_launches_a_propagating_wave():
    """
    Максимум берётся ЗА ВСЁ ВРЕМЯ прогона, а не в конечный момент:
    `potential_range()` возвращает мгновенное состояние, и к сотой
    миллисекунде волна уже прошла бы мимо большей части ткани.
    """
    spec = RectangleMeshSpec(nx=32, ny=6, lx_mm=12.0, ly_mm=2.0)
    solver = _make_solver(spec)

    peak = 0.0
    floor = 0.0
    for k in range(int(100.0 / DT)):
        solver.step(k * DT, DT)
        lo, hi = solver.potential_range()
        peak = max(peak, hi)
        floor = min(floor, lo)

    assert peak > 0.9, f"волна не запустилась: максимум за прогон {peak:.3f}"
    assert floor >= -1e-9, f"потенциал уходил в минус: {floor:.3e}"


def test_wave_reaches_the_far_end():
    """Фронт должен дойти до дальнего края, а не затухнуть по дороге."""
    from mpi4py import MPI

    from cardiac_em.fem import dof_coordinates

    spec = RectangleMeshSpec(nx=32, ny=6, lx_mm=12.0, ly_mm=2.0)
    solver = _make_solver(spec)

    x = dof_coordinates(solver.P1)[:solver.n_owned, 0]
    far = x > 0.9 * spec.lx_mm

    reached = False
    for k in range(int(150.0 / DT)):
        solver.step(k * DT, DT)
        u = solver.potential()[:solver.n_owned]
        if far.any() and np.any(u[far] > 0.5):
            reached = True
            break
    reached = solver.comm.allreduce(reached, op=MPI.LOR)
    assert reached, "фронт не дошёл до дальнего края за 150 мс"


def test_potential_stays_bounded():
    """Ни разгон, ни отрицательные значения недопустимы."""
    spec = RectangleMeshSpec(nx=32, ny=6, lx_mm=12.0, ly_mm=2.0)
    solver = _make_solver(spec)

    for k in range(int(120.0 / DT)):
        solver.step(k * DT, DT)
        lo, hi = solver.potential_range()
        assert -1e-9 <= lo, f"потенциал ушёл в минус: {lo:.3e}"
        assert hi <= 1.6, f"потенциал разогнался: {hi:.3f}"


# ═══════════════════════════════════════════════════════════════════════
#  СКОРОСТЬ ПРОВЕДЕНИЯ
# ═══════════════════════════════════════════════════════════════════════

def test_conduction_velocity_matches_independent_reference():
    """
    Главная проверка солвера: скорость плоского фронта должна совпасть с
    эталоном, посчитанным конечными разностями.

    Допуск 15 % — расхождение двух разных дискретизаций (P1 + Кранк–
    Николсон против явной FD) на сетке, где фронт покрыт несколькими
    ячейками. Более узкий допуск сделал бы тест хрупким к мелким
    изменениям схемы, более широкий — пропустил бы ошибку в тензоре
    диффузии.
    """
    d_long = 0.3
    spec = RectangleMeshSpec(nx=72, ny=4, lx_mm=18.0, ly_mm=1.0)
    base = TissueBaseParams(
        conduction=ConductionParams(d_long=d_long, d_trans=d_long,
                                    fiber_angle_deg=0.0))
    solver = _make_solver(spec, base=base)

    cv = _measure_cv(solver, spec, t_end_ms=130.0)
    reference = _reference_cv_1d(d_long)

    assert math.isclose(cv, reference, rel_tol=0.15), (
        f"скорость {cv:.4f} мм/мс против эталона {reference:.4f} мм/мс "
        f"(расхождение {abs(cv / reference - 1) * 100:.1f} %)")


def test_conduction_velocity_scales_as_sqrt_of_diffusivity():
    """
    Скорость должна идти как √D. Проверка независима от абсолютного
    значения и ловит ошибку масштабирования в тензоре диффузии —
    например, если бы D входил линейно.
    """
    spec = RectangleMeshSpec(nx=72, ny=4, lx_mm=18.0, ly_mm=1.0)

    speeds = {}
    for d in (0.3, 0.075):           # отношение 4 → скорости как 2
        base = TissueBaseParams(
            conduction=ConductionParams(d_long=d, d_trans=d))
        solver = _make_solver(spec, base=base)
        t_end = 130.0 if d > 0.1 else 250.0
        speeds[d] = _measure_cv(solver, spec, t_end_ms=t_end)

    ratio = speeds[0.3] / speeds[0.075]
    assert math.isclose(ratio, 2.0, rel_tol=0.1), (
        f"отношение скоростей {ratio:.3f}, ожидалось 2.0 "
        f"(D: {speeds[0.3]:.4f} и {speeds[0.075]:.4f} мм/мс)")


def test_anisotropy_slows_transverse_propagation():
    """
    При D_l ≠ D_t фронт вдоль волокон обязан идти быстрее поперечного,
    и отношение скоростей — как √(D_l/D_t).

    Геометрия и стимул в обоих прогонах ОДИНАКОВЫ, меняется только угол
    волокон: 0° (фронт идёт вдоль волокон) и 90° (тот же фронт идёт
    поперёк). Так проверяется не только анизотропия тензора, но и то,
    что поворот на угол волокон действительно применяется: при 90°
    получается D = diag(D_t, D_l), и распространение по x видит D_t.

    Поворачивать геометрию вместо волокон было бы хуже: пространственный
    профиль стимула задан вдоль оси x, и на вытянутой по y области он
    накрыл бы всю ткань разом — фронта бы просто не было.
    """
    d_long, d_trans = 0.3, 0.075          # отношение 4 → скорости как 2
    spec = RectangleMeshSpec(nx=72, ny=4, lx_mm=18.0, ly_mm=1.0)

    def cv_at_angle(angle_deg, t_end):
        base = TissueBaseParams(
            conduction=ConductionParams(d_long=d_long, d_trans=d_trans,
                                        fiber_angle_deg=angle_deg))
        return _measure_cv(_make_solver(spec, base=base), spec,
                           t_end_ms=t_end, axis=0)

    cv_along = cv_at_angle(0.0, 130.0)     # волокна вдоль x
    cv_across = cv_at_angle(90.0, 250.0)   # волокна поперёк x

    assert cv_along > cv_across, (
        f"вдоль волокон {cv_along:.4f} не быстрее поперёк {cv_across:.4f}")
    ratio = cv_along / cv_across
    assert math.isclose(ratio, 2.0, rel_tol=0.15), (
        f"анизотропия скоростей {ratio:.3f}, ожидалось √(D_l/D_t) = 2.0 "
        f"({cv_along:.4f} и {cv_across:.4f} мм/мс)")


# ═══════════════════════════════════════════════════════════════════════
#  НЕОДНОРОДНАЯ ТКАНЬ
# ═══════════════════════════════════════════════════════════════════════

def test_inexcitable_region_blocks_propagation():
    """
    Полоса с нулевой возбудимостью и почти нулевой проводимостью должна
    остановить фронт. Это связка всего сделанного ранее: регионы
    (шаг 3) реально влияют на решение.
    """
    from mpi4py import MPI

    from cardiac_em.fem import dof_coordinates

    spec = RectangleMeshSpec(nx=48, ny=4, lx_mm=12.0, ly_mm=1.0)
    barrier = RectRegion(x0=6.0, x1=8.5, y0=0.0, y1=1.0, name="barrier",
                         overrides={"AP_c1": 0.0, "D_LONG": 1e-6,
                                    "D_TRANS": 1e-6})
    solver = _make_solver(spec, regions=(barrier,))

    x = dof_coordinates(solver.P1)[:solver.n_owned, 0]
    before = x < 5.0
    beyond = x > 9.5

    # Максимумы копим ЗА ВСЁ ВРЕМЯ: к концу прогона ткань уже
    # реполяризована, и мгновенное состояние ничего не скажет о том,
    # проходила ли волна.
    peak_before = peak_beyond = 0.0
    for k in range(int(200.0 / DT)):
        solver.step(k * DT, DT)
        u = solver.potential()[:solver.n_owned]
        if before.any():
            peak_before = max(peak_before, float(u[before].max()))
        if beyond.any():
            peak_beyond = max(peak_beyond, float(u[beyond].max()))

    comm = solver.comm
    max_before = comm.allreduce(peak_before, op=MPI.MAX)
    max_beyond = comm.allreduce(peak_beyond, op=MPI.MAX)

    assert max_before > 0.5, "волна не возникла даже до барьера"
    assert max_beyond < 0.3, (
        f"барьер не остановил проведение: за ним потенциал {max_beyond:.3f}")


# ═══════════════════════════════════════════════════════════════════════
#  АКТИВНОЕ НАПРЯЖЕНИЕ
# ═══════════════════════════════════════════════════════════════════════

def test_active_tension_is_zero_at_rest():
    spec = RectangleMeshSpec(nx=20, ny=20)
    solver = _make_solver(spec, stim=StimulusProtocol(times_ms=(1e6,)))

    t_act = solver.active_tension_dg0()
    assert t_act.shape == (solver.DG0.dofmap.index_map.size_local,)
    np.testing.assert_allclose(t_act, 0.0, atol=1e-12)


def test_active_tension_peaks_near_quarter_of_t_max():
    """
    T_act = T_max·u·(1−u) достигает максимума T_max/4 при u = 1/2. Фронт
    проходит через это значение, поэтому где-то в ткани пик обязан
    появиться.
    """
    from mpi4py import MPI

    spec = RectangleMeshSpec(nx=32, ny=6, lx_mm=12.0, ly_mm=2.0)
    base = TissueBaseParams()
    solver = _make_solver(spec, base=base)

    t_max = base.to_flat_dict()["T_MAX"]
    peak = 0.0
    for k in range(int(90.0 / DT)):
        solver.step(k * DT, DT)
        local = solver.active_tension_dg0()
        peak = max(peak, float(local.max()) if local.size else 0.0)
    peak = solver.comm.allreduce(peak, op=MPI.MAX)

    assert peak > 0.2 * t_max, f"пик активного напряжения мал: {peak:.2f} кПа"
    assert peak <= 0.25 * t_max + 1e-9, (
        f"пик превысил T_max/4 = {0.25 * t_max:.2f}: {peak:.2f} кПа")


def test_active_tension_respects_regional_t_max():
    """Регион с T_MAX = 0 не должен давать напряжения даже при возбуждении."""
    from cardiac_em.fem import owned_dof_coordinates

    spec = RectangleMeshSpec(nx=32, ny=6, lx_mm=12.0, ly_mm=2.0)
    dead = RectRegion(x0=7.0, x1=12.0, y0=0.0, y1=2.0, name="no_tension",
                      overrides={"T_MAX": 0.0})
    solver = _make_solver(spec, regions=(dead,))

    xy = owned_dof_coordinates(solver.DG0)
    right = xy[:, 0] > 7.0

    peak_right = 0.0
    for k in range(int(100.0 / DT)):
        solver.step(k * DT, DT)
        t_act = solver.active_tension_dg0()
        if right.any():
            peak_right = max(peak_right, float(t_act[right].max()))

    assert peak_right < 1e-9, (
        f"в зоне с T_MAX = 0 возникло напряжение {peak_right:.3e} кПа")


def test_t_act_function_matches_array():
    spec = RectangleMeshSpec(nx=20, ny=10, lx_mm=10.0, ly_mm=5.0)
    solver = _make_solver(spec)
    for k in range(int(30.0 / DT)):
        solver.step(k * DT, DT)

    arr = solver.active_tension_dg0()
    fn = solver.t_act_function()
    n = solver.DG0.dofmap.index_map.size_local
    np.testing.assert_allclose(fn.x.array[:n], arr, rtol=1e-12)


# ═══════════════════════════════════════════════════════════════════════
#  КОНТРАКТ И УСТОЙЧИВОСТЬ
# ═══════════════════════════════════════════════════════════════════════

def test_invalid_theta_rejected():
    from cardiac_em.fem import Tissue, build_mesh
    from cardiac_em.models.cell import make_cell_model
    from cardiac_em.solvers import MonodomainSolver

    mesh = build_mesh(RectangleMeshSpec(nx=4, ny=4))
    tissue = Tissue.for_electrics(mesh, TissueBaseParams())
    with pytest.raises(ValueError, match="theta"):
        MonodomainSolver(tissue, make_cell_model("rogers_mcculloch"),
                         StimulusProtocol(), theta=1.5)


def test_implicit_scheme_also_works():
    """θ = 1 (полностью неявная схема) должна давать близкую скорость."""
    spec = RectangleMeshSpec(nx=72, ny=4, lx_mm=18.0, ly_mm=1.0)
    base = TissueBaseParams(conduction=ConductionParams(d_long=0.3, d_trans=0.3))

    cv_cn = _measure_cv(_make_solver(spec, base=base, theta=0.5), spec, 130.0)
    cv_impl = _measure_cv(_make_solver(spec, base=base, theta=1.0), spec, 130.0)

    assert math.isclose(cv_cn, cv_impl, rel_tol=0.1), (
        f"схемы расходятся сильнее ожидаемого: КН {cv_cn:.4f}, "
        f"неявная {cv_impl:.4f}")


def test_state_and_potential_stay_consistent():
    """
    Потенциал в массиве состояния и в Function обязаны совпадать после
    шага — иначе расщепление рассинхронизировано.
    """
    spec = RectangleMeshSpec(nx=20, ny=10, lx_mm=10.0, ly_mm=5.0)
    solver = _make_solver(spec)

    for k in range(int(40.0 / DT)):
        solver.step(k * DT, DT)

    np.testing.assert_allclose(
        solver.v_fn.x.array[:solver.n_local],
        solver.state[:, solver.cell.v_index],
        atol=1e-14)


def test_summary_renders():
    spec = RectangleMeshSpec(nx=8, ny=8)
    text = _make_solver(spec).summary()
    assert "Монодомен" in text and "rogers_mcculloch" in text
