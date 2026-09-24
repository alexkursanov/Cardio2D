"""
Квазистатическая гиперупругость с активным напряжением.
========================================================

Ищется поле перемещений u, при котором ткань находится в равновесии:

    ∫ (P_pass + P_act) : ∇v dΩ = 0   ∀v,      P = F·S

Инерция не учитывается: характерное время механического отклика много
меньше длительности сокращения, поэтому на каждом механическом шаге
решается стационарная задача при текущем активном напряжении.

Активное напряжение
--------------------
    σ_act = T_act/J · f̂⊗f̂,   f̂ = F·f₀/λ_f
    S_act = J·F⁻¹·σ_act·F⁻ᵀ = T_act/I₄ · f₀⊗f₀

Вторая форма и используется: она не содержит обращения F, а потому
устойчивее и дешевле. Обе части, пассивная и активная, складываются на
уровне S — там у них одинаковый смысл.

Откуда берётся T_act
---------------------
С ЭЛЕКТРИЧЕСКОЙ сетки через `coupling/`. Солвер механики получает
готовый массив значений на своих ячейках и не знает, что тот пришёл с
более мелкой сетки.

Граничные условия
------------------
Функции `bcs_*` ниже привязаны к прямоугольной геометрии: они ищут
грани по координатам. Для реальной геометрии из файла их придётся
переписать на маркеры граней — это ожидаемая точка расширения, а не
недосмотр.
"""

from __future__ import annotations

import numpy as np

import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc as fem_petsc
import dolfinx.mesh as dmesh
from ufl import (
    Identity,
    TestFunction,
    conditional,
    gt,
    TrialFunction,
    derivative,
    det,
    dx,
    grad,
    inner,
    outer,
    sqrt,
)

from ..config.mesh_spec import RectangleMeshSpec
from ..fem.tissue import Tissue
from ..models.active import ActiveTensionLaw, IsometricLaw
from ..models.passive import (
    PassiveMaterial,
    TransverselyIsotropicExponential,
    kinematics,
)

__all__ = [
    "MechanicsSolver",
    "bcs_uniaxial_stretch",
    "bcs_uniaxial_stretch_symmetric",
    "bcs_free_contraction",
    "bcs_clamped_at_current_state",
    "bcs_prescribed_on_boundary",
]


class MechanicsSolver:
    """
    Нелинейная квазистатическая задача на механической сетке.

    Параметры
    ---------
    tissue : Tissue
        Поля параметров на механической сетке (`Tissue.for_mechanics`).
    material : PassiveMaterial, optional
        Модель пассивной упругости; по умолчанию трансверсально-
        изотропная экспоненциальная.
    active_law : ActiveTensionLaw, optional
        Как действующее активное напряжение зависит от скорости волокна:
        T_act = T_iso · φ(λ_f(u), λ_f,пред, Δt). По умолчанию φ ≡ 1 (модели
        без механической обратной связи). У TNNPM — сила–скорость,
        вычисляемая НЕЯВНО, внутри равновесия (см. models/active).
    dt_mech : float
        Шаг механики, мс — для скорости в законе φ; меняется
        `set_time_step` (последний шаг может быть короче).
    petsc_options : dict, optional
        Настройки SNES. По умолчанию — Ньютон с возвратом по шагу и
        прямым решателем: на задачах такого размера он надёжнее
        итерационного, а стоимость несущественна.
    """

    _DEFAULT_OPTIONS = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_stol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    def __init__(self, tissue: Tissue,
                 material: PassiveMaterial | None = None,
                 petsc_options: dict | None = None,
                 active_law: ActiveTensionLaw | None = None,
                 dt_mech: float = 1.0):
        self.tissue = tissue
        self.mesh = tissue.mesh
        self.comm = self.mesh.comm
        self.material = material or TransverselyIsotropicExponential()

        for key in self.material.param_names:
            if key not in tissue.keys:
                raise KeyError(
                    f"материалу {self.material.name} нужен параметр {key!r}, "
                    f"но он не построен на этой сетке "
                    f"(есть: {sorted(tissue.keys)})")

        self.V = fem.functionspace(self.mesh, ("Lagrange", 1, (2,)))
        self.u = fem.Function(self.V, name="u")
        self._du = TrialFunction(self.V)
        self._v = TestFunction(self.V)

        # DG0 берём из tissue: так T_act поячеечно совпадает с полями
        # параметров, и перепутать порядок невозможно.
        self.DG0 = tissue.DG0
        # T_act — сила, переданная клетками (у моделей с законом φ — при
        # нулевой скорости, T_iso). Действующее напряжение T_iso·φ —
        # `active_tension_actual()`.
        self.T_act = fem.Function(self.DG0, name="T_input_kPa")
        self.n_cells_owned = self.DG0.dofmap.index_map.size_local

        self.law = active_law or IsometricLaw()
        # Выключатель закона φ: на преднагрузке активного напряжения нет, и
        # скорость растяжения к силе клеток отношения не имеет.
        self._law_on = fem.Constant(self.mesh, dolfinx.default_scalar_type(1.0))
        # Перемещения прошлого механического шага. Прошлое растяжение
        # берётся как √I₄(u_пред) в ТЕХ ЖЕ квадратурных точках, что и
        # текущее: если хранить его одним значением на ячейку, разброс λ_f
        # внутри ячейки (у зажатых граней) превращается в ложную скорость —
        # 0.003 по λ за 1 мс это уже ~0.9·v_max, и Ньютон не сходится.
        self.u_old = fem.Function(self.V, name="u_prev")
        self.dt = fem.Constant(self.mesh, dolfinx.default_scalar_type(dt_mech))

        self.bcs: list = []
        self._bcs_dirty = True
        self._problem = None
        self._options = dict(petsc_options or self._DEFAULT_OPTIONS)
        self._post_cache: dict[str, tuple] = {}

        self._build_forms()

    # ── формы ─────────────────────────────────────────────────────────
    def _build_forms(self) -> None:
        u, v, du = self.u, self._v, self._du

        F = Identity(2) + grad(u)
        S_pass = self.material.second_piola(u, self.tissue)
        S_act = self._active_second_piola(u)

        # P = F·S для суммарного напряжения
        self.residual = inner(F * (S_pass + S_act), grad(v)) * dx
        self.jacobian = derivative(self.residual, u, du)

    def _effective_tension(self, u):
        """T_act · φ(v(u)) — действующее активное напряжение (UFL)."""
        if not self.law.uses_velocity:
            return self.T_act
        _, _, _, _, I4 = kinematics(u, self.tissue)
        _, _, _, _, I4_old = kinematics(self.u_old, self.tissue)
        phi = self.law.factor(sqrt(I4), sqrt(I4_old), self.dt)
        return self.T_act * conditional(gt(self._law_on, 0.5), phi, 1.0)

    def _active_second_piola(self, u):
        """S_act = T_eff/I₄ · f₀⊗f₀ — без обращения F."""
        f0 = self.tissue.fiber_vector()
        _, _, _, _, I4 = kinematics(u, self.tissue)
        return (self._effective_tension(u) / I4) * outer(f0, f0)

    # ── активное напряжение ───────────────────────────────────────────
    def set_active_tension(self, values_owned: np.ndarray) -> None:
        """
        Задать T_act по значениям на ВЛАДЕЕМЫХ ячейках — ровно в том
        виде, в каком их отдаёт перенос с электрической сетки.
        """
        values = np.asarray(values_owned, dtype=np.float64)
        if values.size < self.n_cells_owned:
            raise ValueError(
                f"ожидалось {self.n_cells_owned} значений T_act на владеемых "
                f"ячейках, получено {values.size}")
        self.T_act.x.array[:self.n_cells_owned] = values[:self.n_cells_owned]
        self.T_act.x.scatter_forward()

    def clear_active_tension(self) -> None:
        self.T_act.x.array[:] = 0.0
        self.T_act.x.scatter_forward()

    # ── граничные условия и решение ───────────────────────────────────
    def set_bcs(self, bcs: list) -> None:
        """
        Сменить граничные условия. Задача пересобирается при следующем
        решении: в DOLFINx граничные условия вшиты в объект задачи, и
        подменить их на лету нельзя.
        """
        self.bcs = list(bcs)
        self._bcs_dirty = True

    def solve(self) -> tuple[int, bool]:
        """
        Решить при текущих T_act и граничных условиях.
        Возвращает (число итераций Ньютона, сошлось ли).
        """
        if self._problem is None or self._bcs_dirty:
            self._problem = fem_petsc.NonlinearProblem(
                self.residual, self.u, bcs=self.bcs, J=self.jacobian,
                petsc_options=self._options,
                petsc_options_prefix="mech_")
            self._bcs_dirty = False

        self._problem.solve()
        snes = self._problem.solver
        n_it = snes.getIterationNumber()
        converged = snes.getConvergedReason() > 0
        return n_it, converged

    def solve_or_raise(self) -> int:
        """Как `solve`, но несходимость — ошибка, а не тихий возврат False."""
        n_it, ok = self.solve()
        if not ok:
            reason = self._problem.solver.getConvergedReason()
            raise RuntimeError(
                f"механика не сошлась: SNES reason={reason} за {n_it} итераций. "
                f"Обычные причины — слишком большой шаг нагружения или "
                f"вырожденная деформация (проверьте det F)")
        return n_it

    # ── постобработка ─────────────────────────────────────────────────
    def _interpolate_dg0(self, expr_ufl, name: str) -> fem.Function:
        """
        Интерполировать выражение на DG0.

        Скомпилированное выражение и функция-приёмник кэшируются по имени:
        выражения ссылаются на `u` и `T_act`, чьи значения меняются, так
        что компиляция нужна один раз на весь расчёт.

        СЛЕДСТВИЕ: повторный вызов возвращает ТУ ЖЕ функцию, обновлённую
        на месте. Если нужны значения на конкретный момент — скопируйте
        массив (`fn.x.array.copy()`), а не держите ссылку на функцию.
        """
        cached = self._post_cache.get(name)
        if cached is None:
            expr = fem.Expression(expr_ufl, self.DG0.element.interpolation_points)
            fn = fem.Function(self.DG0, name=name)
            cached = self._post_cache[name] = (expr, fn)
        expr, fn = cached
        fn.interpolate(expr)
        return fn

    def fiber_stretch(self) -> fem.Function:
        """λ_f = √I₄ — растяжение вдоль волокна, на ячейках."""
        _, _, _, _, I4 = kinematics(self.u, self.tissue)
        return self._interpolate_dg0(sqrt(I4), "lambda_f")

    def jacobian_determinant(self) -> fem.Function:
        """J = det F. Для почти несжимаемого материала должен быть ≈ 1."""
        F = Identity(2) + grad(self.u)
        return self._interpolate_dg0(det(F), "J")

    def cauchy_stress(self, i: int, j: int) -> fem.Function:
        """
        Компонента σ_ij тензора напряжений Коши (пассивная + активная),
        кПа, на ячейках.
        """
        F, _, J, _, I4 = kinematics(self.u, self.tissue)
        S = self.material.second_piola(self.u, self.tissue)
        sigma_pass = (1 / J) * F * S * F.T

        f0 = self.tissue.fiber_vector()
        f_cur = (F * f0) / sqrt(I4)          # направление волокна после деформации
        sigma_act = (self._effective_tension(self.u) / J) * outer(f_cur, f_cur)

        return self._interpolate_dg0((sigma_pass + sigma_act)[i, j],
                                     f"sigma_{i}{j}")

    def active_tension_actual(self) -> fem.Function:
        """Действующее активное напряжение T_iso·φ, кПа, на ячейках."""
        return self._interpolate_dg0(self._effective_tension(self.u), "T_act_kPa")

    # ── шаг по времени и прошлое растяжение (для закона φ) ────────────
    def enable_active_law(self, on: bool) -> None:
        """Включить/выключить зависимость T_act от скорости (закон φ)."""
        self._law_on.value = 1.0 if on else 0.0

    def set_time_step(self, dt_ms: float) -> None:
        if dt_ms <= 0:
            raise ValueError(f"шаг механики должен быть > 0, получено {dt_ms}")
        self.dt.value = dt_ms

    def stretch_rate(self) -> np.ndarray:
        """
        dλ_f/dt на владеемых ячейках: (λ_f − λ_f,пред)/Δt — та самая
        скорость, при которой решено равновесие (до следующего
        `commit_step`, который вызывается в начале следующего шага).
        """
        _, _, _, _, I4 = kinematics(self.u, self.tissue)
        _, _, _, _, I4_old = kinematics(self.u_old, self.tissue)
        rate = self._interpolate_dg0((sqrt(I4) - sqrt(I4_old)) / self.dt, "lambda_f_rate")
        return rate.x.array[:self.n_cells_owned].copy()

    def commit_step(self) -> None:
        """Запомнить текущие перемещения как «прошлые» для следующего шага."""
        self.u_old.x.array[:] = self.u.x.array
        self.u_old.x.scatter_forward()

    def value_range(self, fn: fem.Function) -> tuple[float, float]:
        """Глобальный диапазон значений поля на ячейках."""
        from mpi4py import MPI

        owned = fn.x.array[:self.n_cells_owned]
        lo = self.comm.allreduce(
            float(owned.min()) if owned.size else np.inf, op=MPI.MIN)
        hi = self.comm.allreduce(
            float(owned.max()) if owned.size else -np.inf, op=MPI.MAX)
        return lo, hi

    def summary(self) -> str:
        return (f"  Механика     : {self.material.describe()}\n"
                f"                 {self.law.describe()}\n"
                f"                 SNES {self._options.get('snes_type')}, "
                f"{self._options.get('pc_type')}")


# ═══════════════════════════════════════════════════════════════════════
#  ГРАНИЧНЫЕ УСЛОВИЯ
# ═══════════════════════════════════════════════════════════════════════
#
#  Все функции ниже находят грани ПО КООРДИНАТАМ и потому привязаны к
#  прямоугольной области. При переходе на геометрию из файла их нужно
#  будет переписать на маркеры граней (`meshtags`), которые придут
#  вместе с сеткой. Это ожидаемая точка расширения.
# ═══════════════════════════════════════════════════════════════════════

_TOL = 1e-10


def _boundary_facets(mesh, marker):
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    return fdim, dmesh.locate_entities_boundary(mesh, fdim, marker)


def bcs_uniaxial_stretch(solver: MechanicsSolver, spec: RectangleMeshSpec,
                         delta_x: float) -> list:
    """
    Одноосное растяжение вдоль x на величину `delta_x`.

        x = 0   : u_x = 0
        x = L_x : u_x = delta_x

    Поперечная компонента u_y на этих гранях СВОБОДНА — ткань может
    сужаться (эффект Пуассона). Чтобы задача не была вырожденной
    относительно сдвига всей области по y, в одной точке (0, L_y/2)
    дополнительно закрепляется u_y.

    ВНИМАНИЕ: точечная связь в двумерной упругости даёт сосредоточенную
    реакцию и потому сингулярность напряжения — деформация перестаёт
    быть однородной даже там, где должна. Измерено на сетке 8×8:
    λ_f = 1.055…1.108 при заданном 1.1. Вариант унаследован из исходного
    скрипта и оставлен для сверки с ним; для новых расчётов берите
    `bcs_uniaxial_stretch_symmetric`.
    """
    mesh, V = solver.mesh, solver.V
    Vx, Vy = V.sub(0), V.sub(1)

    fdim, left = _boundary_facets(
        mesh, lambda x: np.isclose(x[0], 0.0, atol=_TOL))
    _, right = _boundary_facets(
        mesh, lambda x: np.isclose(x[0], spec.lx_mm, atol=_TOL))

    zero = fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    shift = fem.Constant(mesh, dolfinx.default_scalar_type(float(delta_x)))

    bc_left = fem.dirichletbc(
        zero, fem.locate_dofs_topological(Vx, fdim, left), Vx)
    bc_right = fem.dirichletbc(
        shift, fem.locate_dofs_topological(Vx, fdim, right), Vx)

    # точечное закрепление u_y: убирает сдвиг как жёсткого целого
    Vy_c, _ = Vy.collapse()
    u_y_zero = fem.Function(Vy_c)
    dofs_point = fem.locate_dofs_geometrical(
        (Vy, Vy_c),
        lambda x: (np.isclose(x[0], 0.0, atol=_TOL)
                   & np.isclose(x[1], spec.ly_mm / 2.0, atol=spec.hy_mm)))
    bc_point = fem.dirichletbc(u_y_zero, dofs_point, Vy)

    return [bc_left, bc_right, bc_point]


def bcs_uniaxial_stretch_symmetric(solver: MechanicsSolver,
                                   spec: RectangleMeshSpec,
                                   delta_x: float) -> list:
    """
    Одноосное растяжение вдоль x с условиями СИММЕТРИИ вместо точечной
    связи:

        x = 0   : u_x = 0
        x = L_x : u_x = delta_x
        y = 0   : u_y = 0        (вся нижняя грань)

    Верхняя грань свободна, поэтому ткань сужается; левая и правая
    свободны по y.

    ЧЕМ ЭТО ЛУЧШЕ ТОЧЕЧНОГО ЗАКРЕПЛЕНИЯ. Вариант `bcs_uniaxial_stretch`
    убирает смещение как жёсткого целого, фиксируя u_y в ОДНОМ узле. В
    двумерной упругости такая связь создаёт сосредоточенную реакцию, а
    значит сингулярность напряжения: решение перестаёт быть однородным,
    и λ_f «гуляет» вокруг заданного значения (измерено: 1.055…1.108
    вместо 1.1 на сетке 8×8). Условие симметрии убирает ту же моду, не
    создавая сосредоточенной силы, и однородная деформация становится
    точным решением.

    ЧЕМ ХУЖЕ. Нижняя грань теперь не может сужаться, то есть задача
    описывает половину симметричного образца, а не полосу со свободными
    краями. Для преднагрузки, которая должна быть однородной, это как
    раз то, что нужно.
    """
    mesh, V = solver.mesh, solver.V
    Vx, Vy = V.sub(0), V.sub(1)

    fdim, left = _boundary_facets(
        mesh, lambda x: np.isclose(x[0], 0.0, atol=_TOL))
    _, right = _boundary_facets(
        mesh, lambda x: np.isclose(x[0], spec.lx_mm, atol=_TOL))
    _, bottom = _boundary_facets(
        mesh, lambda x: np.isclose(x[1], 0.0, atol=_TOL))

    zero = fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    shift = fem.Constant(mesh, dolfinx.default_scalar_type(float(delta_x)))

    return [
        fem.dirichletbc(zero, fem.locate_dofs_topological(Vx, fdim, left), Vx),
        fem.dirichletbc(shift, fem.locate_dofs_topological(Vx, fdim, right), Vx),
        fem.dirichletbc(zero, fem.locate_dofs_topological(Vy, fdim, bottom), Vy),
    ]


def bcs_free_contraction(solver: MechanicsSolver,
                         spec: RectangleMeshSpec) -> list:
    """
    Изотонический режим: ткань может свободно менять длину.

        x = 0 : u_x = 0        y = 0 : u_y = 0

    Правая и верхняя грани свободны. Смещение как жёсткого целого убрано
    условиями симметрии, без сосредоточенных реакций, поэтому однородная
    деформация остаётся точным решением и здесь.

    Используется, когда нужно увидеть УКОРОЧЕНИЕ под действием активного
    напряжения, а не развиваемую силу.
    """
    mesh, V = solver.mesh, solver.V
    Vx, Vy = V.sub(0), V.sub(1)

    fdim, left = _boundary_facets(
        mesh, lambda x: np.isclose(x[0], 0.0, atol=_TOL))
    _, bottom = _boundary_facets(
        mesh, lambda x: np.isclose(x[1], 0.0, atol=_TOL))

    zero = fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    return [
        fem.dirichletbc(zero, fem.locate_dofs_topological(Vx, fdim, left), Vx),
        fem.dirichletbc(zero, fem.locate_dofs_topological(Vy, fdim, bottom), Vy),
    ]


def bcs_clamped_at_current_state(solver: MechanicsSolver,
                                 spec: RectangleMeshSpec) -> list:
    """
    Зажать все четыре грани в ТЕКУЩЕМ положении.

    Используется после преднагрузки: растянутое состояние фиксируется, и
    дальше активное напряжение развивается изометрически. Предписанное
    значение — копия текущего u, поэтому условие не «возвращает» ткань,
    а удерживает её там, где она есть.

    Копия нужна обязательно: если передать саму `solver.u`, DOLFINx будет
    брать граничные значения из решения, которое меняется в процессе
    решения, — условие перестанет быть условием.
    """
    mesh, V = solver.mesh, solver.V

    def on_boundary(x):
        return (np.isclose(x[0], 0.0, atol=_TOL)
                | np.isclose(x[0], spec.lx_mm, atol=_TOL)
                | np.isclose(x[1], 0.0, atol=_TOL)
                | np.isclose(x[1], spec.ly_mm, atol=_TOL))

    fdim, facets = _boundary_facets(mesh, on_boundary)

    u_fixed = fem.Function(V)
    u_fixed.x.array[:] = solver.u.x.array[:]
    u_fixed.x.scatter_forward()

    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return [fem.dirichletbc(u_fixed, dofs)]


def bcs_prescribed_on_boundary(solver: MechanicsSolver,
                               spec: RectangleMeshSpec,
                               displacement) -> list:
    """
    Задать перемещение на ВСЕЙ границе по функции `displacement(x)`,
    где x — массив координат формы (3, n), а результат — (2, n).

    Нужно для проверок: при однородном материале и аффинном перемещении
    на всей границе точное решение — то же аффинное поле во всей
    области, потому что напряжение тогда постоянно, а значит его
    дивергенция равна нулю.
    """
    mesh, V = solver.mesh, solver.V

    def on_boundary(x):
        return (np.isclose(x[0], 0.0, atol=_TOL)
                | np.isclose(x[0], spec.lx_mm, atol=_TOL)
                | np.isclose(x[1], 0.0, atol=_TOL)
                | np.isclose(x[1], spec.ly_mm, atol=_TOL))

    fdim, facets = _boundary_facets(mesh, on_boundary)

    u_bc = fem.Function(V)
    u_bc.interpolate(displacement)
    u_bc.x.scatter_forward()

    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return [fem.dirichletbc(u_bc, dofs)]
