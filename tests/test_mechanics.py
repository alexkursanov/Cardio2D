"""
Тесты механики (Шаг 7).
========================

ТРЕБУЮТ DOLFINx.

    pytest tests/test_mechanics.py -v
    mpirun -n 4 python -m pytest tests/test_mechanics.py -q

Основной приём проверки — АФФИННАЯ ДЕФОРМАЦИЯ. Если материал однороден,
а на всей границе задано аффинное перемещение, то точное решение — то же
аффинное поле во всей области: градиент деформации постоянен, значит
постоянно и напряжение, а дивергенция постоянного поля равна нулю.
Аффинное поле лежит в пространстве P1 точно, поэтому метод конечных
элементов обязан воспроизвести его с точностью решателя, а не с
точностью дискретизации.

Это даёт сразу две независимые проверки:
  * решение внутри области совпадает с аффинным полем (проверяются
    невязка, якобиан, наложение граничных условий и сам решатель);
  * напряжение совпадает с посчитанным ОТДЕЛЬНО на numpy по тем же
    формулам (проверяется реализация материала в UFL).

ВАЖНОЕ СВОЙСТВО МОДЕЛИ. При F = I напряжение НЕ равно нулю: изотропный
член даёт ∂Ψ/∂I₁ = μ·e^{b(I₁−2)}, что при I₁ = 2 равно μ, откуда
σ = 2μ·I = 2 кПа гидростатики. Недеформированное состояние в этой
энергии не свободно от напряжений. Это характерно для экспоненциальных
энергий типа Фунга без явного выделения объёмной части (в формулировке
с множителем Лагранжа гидростатика поглощается давлением, а в
штрафной — нет). Свойство унаследовано из исходной модели; тесты его
фиксируют, чтобы изменение заметили сразу.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.config import (  # noqa: E402
    PassiveMechParams,
    RectRegion,
    RectangleMeshSpec,
    TissueBaseParams,
)

pytestmark = pytest.mark.fem


# ═══════════════════════════════════════════════════════════════════════
#  НЕЗАВИСИМЫЙ ЭТАЛОН: НАПРЯЖЕНИЕ НА NUMPY
# ═══════════════════════════════════════════════════════════════════════

def _cauchy_numpy(F: np.ndarray, t_act: float = 0.0, *, mu=1.0, b_iso=1.0,
                  mu_f=3.0, b_f=2.0, kappa=100.0, theta_deg=0.0):
    """
    Тензор напряжений Коши для постоянного F, посчитанный отдельно от
    UFL — по тем же формулам, но другим инструментом. Возвращает
    (σ, J, λ_f).
    """
    th = np.deg2rad(theta_deg)
    f0 = np.array([np.cos(th), np.sin(th)])

    C = F.T @ F
    J = float(np.linalg.det(F))
    I1 = float(np.trace(C))
    I4 = float(f0 @ C @ f0)

    S = 2 * (
        mu * np.exp(b_iso * (I1 - 2)) * np.eye(2)
        + mu_f * (I4 - 1) * np.exp(b_f * (I4 - 1) ** 2) * np.outer(f0, f0)
        + kappa * (J - 1) * J / 2 * np.linalg.inv(C)
    )
    sigma = (F @ S @ F.T) / J

    if t_act:
        f_cur = (F @ f0) / math.sqrt(I4)
        sigma = sigma + (t_act / J) * np.outer(f_cur, f_cur)

    return sigma, J, math.sqrt(I4)


# ═══════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНОЕ
# ═══════════════════════════════════════════════════════════════════════

SPEC = RectangleMeshSpec(nx=8, ny=8, lx_mm=4.0, ly_mm=4.0)


def _make_solver(spec=SPEC, base=None, regions=()):
    from cardiac_em.fem import Tissue, build_mesh
    from cardiac_em.solvers import MechanicsSolver

    mesh = build_mesh(spec)
    tissue = Tissue.for_mechanics(mesh, base or TissueBaseParams(), regions)
    return MechanicsSolver(tissue)


def _affine(grad_u: np.ndarray):
    """Callable для dolfinx: перемещение u(X) = grad_u · X."""
    def _expr(x):
        return np.vstack([grad_u[0, 0] * x[0] + grad_u[0, 1] * x[1],
                          grad_u[1, 0] * x[0] + grad_u[1, 1] * x[1]])
    return _expr


def _solve_affine(solver, spec, grad_u, t_act=0.0):
    """Наложить аффинное поле на границе и решить."""
    from cardiac_em.solvers import bcs_prescribed_on_boundary

    if t_act:
        solver.set_active_tension(np.full(solver.n_cells_owned, t_act))
    solver.set_bcs(bcs_prescribed_on_boundary(solver, spec, _affine(grad_u)))
    return solver.solve_or_raise()


def _global_range(solver, fn):
    return solver.value_range(fn)


# ═══════════════════════════════════════════════════════════════════════
#  РЕФЕРЕНСНОЕ СОСТОЯНИЕ
# ═══════════════════════════════════════════════════════════════════════

def test_rest_state_is_not_stress_free():
    """
    Фиксируем свойство модели: при нулевой деформации остаётся
    гидростатическое напряжение 2μ = 2 кПа. Это не дефект реализации, а
    особенность выбранной энергии — но знать о ней нужно, иначе «нулевое
    перемещение» ошибочно принимают за «нулевое напряжение».
    """
    sigma, J, lam = _cauchy_numpy(np.eye(2))
    assert math.isclose(J, 1.0)
    assert math.isclose(lam, 1.0)
    assert math.isclose(sigma[0, 0], 2.0, rel_tol=1e-12)
    assert math.isclose(sigma[1, 1], 2.0, rel_tol=1e-12)
    assert abs(sigma[0, 1]) < 1e-15


def test_clamped_at_zero_stays_at_zero():
    """
    Если всю границу держать в нуле и не подавать активного напряжения,
    решение — тождественный ноль. Напряжение при этом постоянно
    (те самые 2 кПа), поэтому его дивергенция равна нулю и равновесие
    выполняется.
    """
    from cardiac_em.solvers import bcs_prescribed_on_boundary

    solver = _make_solver()
    solver.set_bcs(bcs_prescribed_on_boundary(
        solver, SPEC, lambda x: np.zeros((2, x.shape[1]))))
    solver.solve_or_raise()

    assert np.max(np.abs(solver.u.x.array)) < 1e-10


# ═══════════════════════════════════════════════════════════════════════
#  АФФИННЫЕ ДЕФОРМАЦИИ
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name,grad_u", [
    ("растяжение по x", np.array([[0.05, 0.0], [0.0, -0.02]])),
    ("сдвиг", np.array([[0.0, 0.03], [0.0, 0.0]])),
    ("двуосное", np.array([[0.04, 0.0], [0.0, 0.04]])),
])
def test_affine_field_is_reproduced_exactly(name, grad_u):
    """
    Аффинное поле — точное решение, и оно лежит в P1. Значит FEM обязан
    воспроизвести его во всей области, а не только на границе.
    """
    from cardiac_em.fem import dof_coordinates

    solver = _make_solver()
    _solve_affine(solver, SPEC, grad_u)

    coords = dof_coordinates(solver.V)
    n_nodes = solver.V.dofmap.index_map.size_local
    xy = coords[:n_nodes, :2]

    expected = np.stack([grad_u[0, 0] * xy[:, 0] + grad_u[0, 1] * xy[:, 1],
                         grad_u[1, 0] * xy[:, 0] + grad_u[1, 1] * xy[:, 1]],
                        axis=1)
    got = solver.u.x.array[:n_nodes * 2].reshape(-1, 2)

    np.testing.assert_allclose(got, expected, atol=1e-9, err_msg=name)


def test_stress_matches_analytic_on_affine_state():
    """
    Реализация материала в UFL сверяется с расчётом на numpy по тем же
    формулам. Совпадать должны все три компоненты — иначе где-то
    перепутаны индексы или потерян множитель.
    """
    grad_u = np.array([[0.05, 0.0], [0.0, -0.02]])
    F = np.eye(2) + grad_u

    solver = _make_solver()
    _solve_affine(solver, SPEC, grad_u)

    sigma_ref, J_ref, lam_ref = _cauchy_numpy(F)

    for (i, j) in ((0, 0), (1, 1), (0, 1)):
        lo, hi = _global_range(solver, solver.cauchy_stress(i, j))
        assert math.isclose(lo, hi, abs_tol=1e-8), (
            f"σ_{i}{j} неоднородно при аффинной деформации: [{lo}, {hi}]")
        assert math.isclose(lo, sigma_ref[i, j], rel_tol=1e-6, abs_tol=1e-8), (
            f"σ_{i}{j}: солвер {lo:.6f}, эталон {sigma_ref[i, j]:.6f}")


def test_jacobian_and_stretch_match_analytic():
    grad_u = np.array([[0.05, 0.0], [0.0, -0.02]])
    F = np.eye(2) + grad_u
    _, J_ref, lam_ref = _cauchy_numpy(F)

    solver = _make_solver()
    _solve_affine(solver, SPEC, grad_u)

    J_lo, J_hi = _global_range(solver, solver.jacobian_determinant())
    lam_lo, lam_hi = _global_range(solver, solver.fiber_stretch())

    assert math.isclose(J_lo, J_hi, abs_tol=1e-10)
    assert math.isclose(J_lo, J_ref, rel_tol=1e-9)
    assert math.isclose(lam_lo, lam_hi, abs_tol=1e-10)
    assert math.isclose(lam_lo, lam_ref, rel_tol=1e-9)


def test_active_tension_enters_stress_analytically():
    """
    При аффинной деформации и однородном T_act активное напряжение тоже
    постоянно, поэтому равновесие сохраняется, а σ_xx должно совпасть с
    аналитикой вместе с активной добавкой.
    """
    grad_u = np.array([[0.05, 0.0], [0.0, -0.02]])
    t_act = 30.0
    F = np.eye(2) + grad_u

    solver = _make_solver()
    _solve_affine(solver, SPEC, grad_u, t_act=t_act)

    sigma_ref, _, _ = _cauchy_numpy(F, t_act=t_act)
    lo, hi = _global_range(solver, solver.cauchy_stress(0, 0))

    assert math.isclose(lo, hi, abs_tol=1e-8)
    assert math.isclose(lo, sigma_ref[0, 0], rel_tol=1e-6)
    # активная часть вдоль волокна много больше пассивной
    assert lo > 30.0


# ═══════════════════════════════════════════════════════════════════════
#  ОДНООСНОЕ РАСТЯЖЕНИЕ
# ═══════════════════════════════════════════════════════════════════════

def test_uniaxial_stretch_gives_prescribed_lambda():
    """
    При растяжении вдоль волокон λ_f должно равняться заданному
    отношению длин. Поперечная грань свободна, поэтому ткань сужается —
    но продольное растяжение задано жёстко.
    """
    from cardiac_em.solvers import bcs_uniaxial_stretch

    stretch = 1.1
    delta = (stretch - 1.0) * SPEC.lx_mm

    solver = _make_solver()
    solver.set_bcs(bcs_uniaxial_stretch(solver, SPEC, delta))
    solver.solve_or_raise()

    lo, hi = _global_range(solver, solver.fiber_stretch())
    assert math.isclose(lo, stretch, rel_tol=1e-6), f"λ_f = [{lo}, {hi}]"
    assert math.isclose(hi, stretch, rel_tol=1e-6)


def test_uniaxial_stretch_contracts_transversally():
    """Эффект Пуассона: при растяжении вдоль x ткань должна сузиться по y."""
    from cardiac_em.fem import dof_coordinates
    from cardiac_em.solvers import bcs_uniaxial_stretch

    solver = _make_solver()
    solver.set_bcs(bcs_uniaxial_stretch(solver, SPEC, 0.4))   # λ_f = 1.1
    solver.solve_or_raise()

    n_nodes = solver.V.dofmap.index_map.size_local
    xy = dof_coordinates(solver.V)[:n_nodes, :2]
    u = solver.u.x.array[:n_nodes * 2].reshape(-1, 2)

    top = xy[:, 1] > SPEC.ly_mm - 1e-9
    if top.any():
        assert np.all(u[top, 1] < 0.0), "верхняя грань не опустилась"


def test_stretch_is_progressive():
    """Большее растяжение — большее напряжение вдоль волокна."""
    from cardiac_em.solvers import bcs_uniaxial_stretch

    stresses = []
    for stretch in (1.02, 1.05, 1.10):
        solver = _make_solver()
        solver.set_bcs(bcs_uniaxial_stretch(
            solver, SPEC, (stretch - 1.0) * SPEC.lx_mm))
        solver.solve_or_raise()
        lo, _ = _global_range(solver, solver.cauchy_stress(0, 0))
        stresses.append(lo)

    assert all(b > a for a, b in zip(stresses, stresses[1:])), (
        f"напряжение не растёт с растяжением: {stresses}")


# ═══════════════════════════════════════════════════════════════════════
#  АКТИВНОЕ НАПРЯЖЕНИЕ
# ═══════════════════════════════════════════════════════════════════════

def test_active_tension_increases_axial_stress_when_clamped():
    """Изометрический режим: ткань зажата, рост T_act поднимает σ_xx."""
    from cardiac_em.solvers import bcs_prescribed_on_boundary

    solver = _make_solver()
    solver.set_bcs(bcs_prescribed_on_boundary(
        solver, SPEC, lambda x: np.zeros((2, x.shape[1]))))

    stresses = []
    for t_act in (0.0, 10.0, 30.0):
        solver.set_active_tension(np.full(solver.n_cells_owned, t_act))
        solver.solve_or_raise()
        lo, _ = _global_range(solver, solver.cauchy_stress(0, 0))
        stresses.append(lo)

    assert all(b > a for a, b in zip(stresses, stresses[1:])), (
        f"σ_xx не растёт с T_act: {stresses}")
    # при зажатых границах деформации нет, поэтому прибавка равна самому T_act
    assert math.isclose(stresses[2] - stresses[0], 30.0, rel_tol=1e-6)


def test_active_tension_shortens_free_tissue():
    """
    Изотонический режим: правая грань свободна, активное напряжение
    вдоль волокон должно стянуть ткань.

    Сравнение идёт с состоянием при T_act = 0, а не с нулём: из-за
    остаточной гидростатики ткань и без активации слегка меняет размер.
    """
    import dolfinx
    import dolfinx.fem as fem

    from cardiac_em.fem import dof_coordinates
    from cardiac_em.solvers.mechanics import _TOL, _boundary_facets

    def free_end_bcs(solver):
        mesh, V = solver.mesh, solver.V
        Vx, Vy = V.sub(0), V.sub(1)
        fdim, left = _boundary_facets(
            mesh, lambda x: np.isclose(x[0], 0.0, atol=_TOL))
        zero = fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
        bc_left = fem.dirichletbc(
            zero, fem.locate_dofs_topological(Vx, fdim, left), Vx)

        Vy_c, _ = Vy.collapse()
        u_y_zero = fem.Function(Vy_c)
        dofs_pt = fem.locate_dofs_geometrical(
            (Vy, Vy_c),
            lambda x: (np.isclose(x[0], 0.0, atol=_TOL)
                       & np.isclose(x[1], SPEC.ly_mm / 2.0, atol=SPEC.hy_mm)))
        return [bc_left, fem.dirichletbc(u_y_zero, dofs_pt, Vy)]

    tip = {}
    for t_act in (0.0, 40.0):
        solver = _make_solver()
        solver.set_active_tension(np.full(solver.n_cells_owned, t_act))
        solver.set_bcs(free_end_bcs(solver))
        solver.solve_or_raise()

        n_nodes = solver.V.dofmap.index_map.size_local
        xy = dof_coordinates(solver.V)[:n_nodes, :2]
        u = solver.u.x.array[:n_nodes * 2].reshape(-1, 2)
        right = xy[:, 0] > SPEC.lx_mm - 1e-9
        local = float(u[right, 0].mean()) if right.any() else np.nan
        vals = [v for v in solver.comm.allgather(local) if not np.isnan(v)]
        tip[t_act] = float(np.mean(vals))

    assert tip[40.0] < tip[0.0], (
        f"активное напряжение не укоротило ткань: "
        f"u_x на свободном крае {tip[0.0]:.5f} → {tip[40.0]:.5f} мм")


# ═══════════════════════════════════════════════════════════════════════
#  НЕОДНОРОДНАЯ ТКАНЬ
# ═══════════════════════════════════════════════════════════════════════

def test_stiff_region_deforms_less():
    """
    Жёсткий рубец при том же растяжении должен деформироваться слабее
    окружающей ткани — связка с регионами из шага 3.
    """
    from cardiac_em.fem import owned_dof_coordinates
    from cardiac_em.solvers import bcs_uniaxial_stretch

    scar = RectRegion(x0=1.5, x1=2.5, y0=0.0, y1=4.0, name="scar",
                      overrides={"MU": 50.0, "MU_F": 150.0})
    solver = _make_solver(base=TissueBaseParams(), regions=(scar,))
    solver.set_bcs(bcs_uniaxial_stretch(solver, SPEC, 0.4))
    solver.solve_or_raise()

    lam = solver.fiber_stretch()
    xy = owned_dof_coordinates(solver.DG0)
    vals = lam.x.array[:solver.n_cells_owned]

    inside = (xy[:, 0] > 1.5) & (xy[:, 0] < 2.5)
    outside = (xy[:, 0] < 1.0) | (xy[:, 0] > 3.0)

    from mpi4py import MPI
    mean_in = solver.comm.allreduce(
        float(vals[inside].sum()) if inside.any() else 0.0, op=MPI.SUM)
    n_in = solver.comm.allreduce(int(inside.sum()), op=MPI.SUM)
    mean_out = solver.comm.allreduce(
        float(vals[outside].sum()) if outside.any() else 0.0, op=MPI.SUM)
    n_out = solver.comm.allreduce(int(outside.sum()), op=MPI.SUM)

    assert n_in > 0 and n_out > 0
    lam_in, lam_out = mean_in / n_in, mean_out / n_out
    assert lam_in < lam_out, (
        f"жёсткая область растянулась не меньше окружения: "
        f"λ_f рубца {lam_in:.4f}, ткани {lam_out:.4f}")


def test_soft_region_deforms_more():
    """Обратная проверка: мягкая область растягивается сильнее."""
    from cardiac_em.fem import owned_dof_coordinates
    from cardiac_em.solvers import bcs_uniaxial_stretch

    soft = RectRegion(x0=1.5, x1=2.5, y0=0.0, y1=4.0, name="soft",
                      overrides={"MU": 0.2, "MU_F": 0.5})
    solver = _make_solver(regions=(soft,))
    solver.set_bcs(bcs_uniaxial_stretch(solver, SPEC, 0.4))
    solver.solve_or_raise()

    xy = owned_dof_coordinates(solver.DG0)
    vals = solver.fiber_stretch().x.array[:solver.n_cells_owned]
    inside = (xy[:, 0] > 1.5) & (xy[:, 0] < 2.5)
    outside = (xy[:, 0] < 1.0) | (xy[:, 0] > 3.0)

    from mpi4py import MPI
    s_in = solver.comm.allreduce(float(vals[inside].sum()) if inside.any() else 0.0, op=MPI.SUM)
    n_in = solver.comm.allreduce(int(inside.sum()), op=MPI.SUM)
    s_out = solver.comm.allreduce(float(vals[outside].sum()) if outside.any() else 0.0, op=MPI.SUM)
    n_out = solver.comm.allreduce(int(outside.sum()), op=MPI.SUM)

    assert s_in / n_in > s_out / n_out


# ═══════════════════════════════════════════════════════════════════════
#  КОНТРАКТ
# ═══════════════════════════════════════════════════════════════════════

def test_electric_tissue_is_rejected():
    """
    Механике нужны модули упругости; ткань, собранная для электрики, их
    не содержит. Ошибка должна быть понятной и в момент создания.
    """
    from cardiac_em.fem import Tissue, build_mesh
    from cardiac_em.solvers import MechanicsSolver

    mesh = build_mesh(SPEC)
    tissue = Tissue.for_electrics(mesh, TissueBaseParams())
    with pytest.raises(KeyError, match="не построен"):
        MechanicsSolver(tissue)


def test_active_tension_size_is_checked():
    solver = _make_solver()
    with pytest.raises(ValueError, match="T_act"):
        solver.set_active_tension(np.zeros(solver.n_cells_owned - 1))


def test_material_registry():
    from cardiac_em.models.passive import (
        TransverselyIsotropicExponential,
        available_materials,
        make_passive_material,
    )

    assert "transversely_isotropic_exponential" in available_materials()
    mat = make_passive_material("transversely_isotropic_exponential")
    assert isinstance(mat, TransverselyIsotropicExponential)
    assert "MU" in mat.param_names
    with pytest.raises(KeyError, match="неизвестный материал"):
        make_passive_material("резина")


def test_softer_material_gives_lower_stress():
    """Параметры ткани действительно доходят до напряжения."""
    from cardiac_em.solvers import bcs_uniaxial_stretch

    results = {}
    for mu in (0.5, 2.0):
        base = TissueBaseParams(passive=PassiveMechParams(mu=mu))
        solver = _make_solver(base=base)
        solver.set_bcs(bcs_uniaxial_stretch(solver, SPEC, 0.4))
        solver.solve_or_raise()
        results[mu], _ = _global_range(solver, solver.cauchy_stress(0, 0))

    assert results[0.5] < results[2.0]


def test_summary_renders():
    text = _make_solver().summary()
    assert "Механика" in text and "transversely_isotropic" in text
