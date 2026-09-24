"""
Монодоменная задача: реакция + анизотропная диффузия.
======================================================

    ∂u/∂t = ∇·(D ∇u) + f(u, …)                D = D_l·f₀⊗f₀ + D_t·s₀⊗s₀

Расщепление Странга за шаг dt:

    реакция(dt/2)  →  диффузия(dt)  →  реакция(dt/2)

Реакция — поточечная, её делает `CellModel` (см. `models/cell/`).
Диффузия — Кранка–Николсон по одной переменной состояния (потенциалу):

    (M + θ·dt·K)·uⁿ⁺¹ = (M − (1−θ)·dt·K)·uⁿ,      θ = 1/2

Матрицы M и K собираются один раз; оператор A = M + θ·dt·K и его
решатель (CG + HYPRE) пересобираются только при смене шага.

Распределение по рангам
------------------------
Состояние клеток хранится для ВСЕХ локальных узлов, включая гало.
Реакция локальна, поэтому на гало она считается повторно и даёт в
точности те же значения, что у владельца, — обмена не требуется. После
диффузии потенциал обновляется из решения на владеемых узлах, а гало
приводятся в соответствие через `scatter_forward()`. Остальные
переменные состояния меняются только реакцией и потому остаются
согласованными без единого дополнительного обмена.

Отличие от прежнего скрипта
----------------------------
Активное напряжение вычисляется НА P1, где живёт полное состояние
клетки, и только затем интерполируется на DG0. В монолитной версии
порядок был обратный: сначала потенциал интерполировался на DG0, потом
из него считалось u·(1−u). Для нелинейной функции это не одно и то же,
расхождение порядка O(h²). Выбран нынешний порядок, потому что он
обобщается на любую модель клетки: у подробной модели активное
напряжение зависит от нескольких переменных состояния, и переносить на
DG0 пришлось бы все.
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem as fem
import dolfinx.fem.petsc as fem_petsc
from ufl import TestFunction, TrialFunction, dot, dx, grad, inner, outer

from ..config.protocol import StimulusProtocol
from ..fem.mesh import dof_coordinates
from ..fem.tissue import Tissue
from ..models.cell.base import CellModel

__all__ = ["MonodomainSolver"]


class MonodomainSolver:
    """
    Монодоменная задача на электрической сетке.

    Параметры
    ---------
    tissue : Tissue
        Поля параметров на электрической сетке (см. `Tissue.for_electrics`).
    cell_model : CellModel
        Модель возбуждения. Её параметры берутся из `tissue` по именам,
        объявленным в `cell_model.param_names`.
    stimulus : StimulusProtocol
        Протокол стимуляции; пространственный профиль вычисляется один
        раз при построении.
    theta : float
        Вес неявной части: 0.5 — Кранк–Николсон (по умолчанию),
        1.0 — полностью неявная схема.
    clip_after_diffusion : tuple | None | "model"
        Ограничение на потенциал после шага диффузии. Схема Кранка–
        Николсона с согласованной матрицей масс не удовлетворяет
        дискретному принципу максимума, и на крутом фронте возможен
        небольшой выброс. По умолчанию ("model") берётся
        `cell_model.potential_clip`: (0, 1.5) у безразмерной модели
        Роджерса–МакКаллоха, без обрезки у моделей в мВ. Прежде значение
        (0, 1.5) было зашито здесь и уничтожило бы потенциал в мВ.
    """

    def __init__(self, tissue: Tissue, cell_model: CellModel,
                 stimulus: StimulusProtocol, theta: float = 0.5,
                 clip_after_diffusion="model"):
        if not 0.0 <= theta <= 1.0:
            raise ValueError(f"theta должна быть в [0, 1], получено {theta}")

        self.tissue = tissue
        self.cell = cell_model
        self.stimulus = stimulus
        self.theta = float(theta)
        if isinstance(clip_after_diffusion, str):
            if clip_after_diffusion != "model":
                raise ValueError(f"clip_after_diffusion: пределы, None или 'model'; "
                                 f"получено {clip_after_diffusion!r}")
            clip_after_diffusion = cell_model.potential_clip
        self.clip_after_diffusion = clip_after_diffusion

        self.mesh = tissue.mesh
        self.comm = self.mesh.comm
        self.P1 = tissue.P1
        self.DG0 = tissue.DG0

        imap = self.P1.dofmap.index_map
        self.n_owned = imap.size_local
        self.n_local = imap.size_local + imap.num_ghosts

        # ── параметры модели клетки, разложенные по узлам ─────────────
        self.params = {name: tissue.p1_array(name)
                       for name in cell_model.param_names}
        cell_model.check_params(self.params, self.n_local)

        # ── состояние клеток: владеемые узлы И гало ───────────────────
        # Реакция локальна, поэтому на гало она считается повторно и даёт
        # те же значения, что у владельца. Это дешевле обмена.
        self.state = cell_model.initial_state(self.n_local)

        # ── вспомогательные функции ───────────────────────────────────
        self.v_fn = fem.Function(self.P1, name="u_ap")     # потенциал
        self._act_p1 = fem.Function(self.P1)                # сигнал активации
        self._t_act = fem.Function(self.DG0, name="T_act_kPa")
        # Выражение P1 → DG0 компилируется ОДИН раз. Оно ссылается на
        # функцию `_act_p1`, чьи значения меняются, поэтому скомпилированный
        # объект годится на весь расчёт. Создавать его на каждом вызове
        # значит на каждом механическом шаге хэшировать форму и загружать
        # модуль из кэша FFCx.
        self._act_expr = fem.Expression(
            self._act_p1, self.DG0.element.interpolation_points)
        self._sync_potential_to_function()

        # ── пространственный профиль стимула ──────────────────────────
        # Считается ОДИН раз. В прежней версии координаты DOF
        # запрашивались на каждом шаге — это заметная доля времени при
        # dt = 0.05 мс.
        self._stim_profile = self._build_stimulus_profile()

        # ── матрицы ───────────────────────────────────────────────────
        self._assemble_matrices()
        self._dt_cached: float | None = None
        self._A: PETSc.Mat | None = None
        self._ksp: PETSc.KSP | None = None
        self._rhs = self._M.createVecRight()
        self._tmp = self._M.createVecRight()
        self._sol = self._M.createVecRight()

    # ── сборка ────────────────────────────────────────────────────────
    def _assemble_matrices(self) -> None:
        u = TrialFunction(self.P1)
        phi = TestFunction(self.P1)

        f0 = self.tissue.fiber_vector()
        s0 = self.tissue.sheet_vector()
        D = (self.tissue.d_long * outer(f0, f0)
             + self.tissue.d_trans * outer(s0, s0))

        self._M = fem_petsc.assemble_matrix(fem.form(inner(u, phi) * dx))
        self._M.assemble()
        self._K = fem_petsc.assemble_matrix(
            fem.form(inner(dot(D, grad(u)), grad(phi)) * dx))
        self._K.assemble()

    def _setup_solver(self, dt: float) -> None:
        """Пересобрать A и решатель. Делается только при смене шага."""
        if self._dt_cached == dt:
            return
        self._dt_cached = dt

        self._A = self._M.copy()
        self._A.axpy(self.theta * dt, self._K)
        self._A.assemble()

        self._ksp = PETSc.KSP().create(self.comm)
        self._ksp.setOperators(self._A)
        self._ksp.setType(PETSc.KSP.Type.CG)
        self._ksp.getPC().setType(PETSc.PC.Type.HYPRE)
        self._ksp.setTolerances(rtol=1e-10, atol=1e-12)
        self._ksp.setFromOptions()

    def _build_stimulus_profile(self) -> np.ndarray:
        """
        Пространственная часть стимула на узлах P1: экспоненциальный спад
        от левого края, обрезанный при x > x_max.
        """
        x = dof_coordinates(self.P1)[:self.n_local, 0]
        profile = np.exp(-x / self.stimulus.decay_length_mm)
        profile[x > self.stimulus.x_max_mm] = 0.0
        return profile

    # ── состояние ↔ функция ───────────────────────────────────────────
    def _sync_potential_to_function(self) -> None:
        self.v_fn.x.array[:self.n_local] = self.state[:, self.cell.v_index]

    def _sync_function_to_potential(self) -> None:
        self.state[:, self.cell.v_index] = self.v_fn.x.array[:self.n_local]

    # ── подготовка ────────────────────────────────────────────────────
    def relax(self, duration_ms: float, dt: float) -> int:
        """
        Счёт БЕЗ стимула (с диффузией): ткань приходит к покою при своих
        параметрах, включая пограничную зону, где здоровые узлы
        электротонически подтянуты деполяризованной ишемической областью
        (см. PreloadProtocol.cell_relax_ms). Время расчёта не сдвигается.
        Возвращает число шагов.

        Именно с диффузией: при подготовке изолированных клеток
        пограничные узлы приходили бы к «чужому» покою, и после включения
        связи их первый удар начинался бы с переходного процесса.
        """
        n = int(round(duration_ms / dt))
        for _ in range(n):
            self._split_step(0.0, dt, stimulate=False)
        return n

    # ── шаг ───────────────────────────────────────────────────────────
    def step(self, t: float, dt: float) -> None:
        """
        Один шаг расщепления Странга: реакция(dt/2) → диффузия(dt) →
        реакция(dt/2).

        Стимул подаётся на обоих полушагах реакции со значением
        огибающей в соответствующий момент времени.
        """
        self._split_step(t, dt, stimulate=True)

    def _split_step(self, t: float, dt: float, stimulate: bool) -> None:
        self._setup_solver(dt)
        zero = None if stimulate else np.zeros(self.n_local)

        self.state = self.cell.step(
            t, self.state, dt / 2,
            self._stimulus_at(t) if stimulate else zero, self.params)
        self._sync_potential_to_function()

        self._diffuse(dt)

        self._sync_function_to_potential()
        self.state = self.cell.step(
            t + dt / 2, self.state, dt / 2,
            self._stimulus_at(t + dt / 2) if stimulate else zero, self.params)
        self._sync_potential_to_function()
        self.v_fn.x.scatter_forward()

    def _stimulus_at(self, t: float) -> np.ndarray:
        env = self.stimulus.envelope(t)
        if env == 0.0:
            return np.zeros(self.n_local)
        return (self.stimulus.amplitude * env) * self._stim_profile

    def _diffuse(self, dt: float) -> None:
        """Шаг Кранка–Николсона по потенциалу."""
        vec = self.v_fn.x.petsc_vec

        self._M.mult(vec, self._rhs)
        self._K.mult(vec, self._tmp)
        self._rhs.axpy(-(1.0 - self.theta) * dt, self._tmp)

        self._ksp.solve(self._rhs, self._sol)
        reason = self._ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(
                f"диффузионный шаг не сошёлся: KSP reason={reason}")

        new = self._sol.array_r[:self.n_owned]
        if self.clip_after_diffusion is not None:
            lo, hi = self.clip_after_diffusion
            new = np.clip(new, lo, hi)
        self.v_fn.x.array[:self.n_owned] = new
        self.v_fn.x.scatter_forward()

    def load_state(self, state_local: np.ndarray) -> None:
        """
        Заменить состояние клеток целиком — на ВСЕХ локальных узлах,
        включая гало, (n_local, n_states). Используется при
        восстановлении из чекпоинта: сопоставление там идёт по
        координатам, и гало получают те же значения, что их владельцы.
        """
        arr = np.asarray(state_local, dtype=np.float64)
        expected = (self.n_local, self.cell.n_states)
        if arr.shape != expected:
            raise ValueError(
                f"состояние формы {arr.shape}, ожидалось {expected} "
                f"(узлы × переменные модели {self.cell.name})")
        self.state = arr.copy()
        self._sync_potential_to_function()
        self.v_fn.x.scatter_forward()

    # ── наблюдаемые величины ──────────────────────────────────────────
    def potential(self) -> np.ndarray:
        """Потенциал на локальных узлах (включая гало)."""
        return self.state[:, self.cell.v_index]

    def active_tension_dg0(self) -> np.ndarray:
        """
        T_act на ВЛАДЕЕМЫХ ячейках DG0, кПа — в том виде, в каком его
        ждёт перенос на механическую сетку.

        Сигнал активации вычисляется на P1 (там полное состояние
        клетки) и затем интерполируется на DG0.
        """
        # Сила, которую клетки передают ткани: у моделей с законом φ(v)
        # в ткани — при нулевой скорости (см. CellModel.isometric_tension)
        act = self.cell.isometric_tension(self.state, self.params)
        self._act_p1.x.array[:self.n_local] = act
        self._act_p1.x.scatter_forward()

        self._t_act.interpolate(self._act_expr)

        n_cells = self.DG0.dofmap.index_map.size_local
        values = self._t_act.x.array[:n_cells]

        if self.cell.tension_kind == "scaled":
            values = values * self.tissue.t_max_array[:n_cells]
        elif self.cell.tension_kind != "absolute":
            raise ValueError(
                f"неизвестный вид активного напряжения "
                f"{self.cell.tension_kind!r} у модели {self.cell.name}")

        return np.clip(values, 0.0, None)

    def t_act_function(self) -> fem.Function:
        """
        То же, но как Function на DG0 — для записи в файл. Значения
        пересчитываются при каждом вызове.
        """
        n_cells = self.DG0.dofmap.index_map.size_local
        self._t_act.x.array[:n_cells] = self.active_tension_dg0()
        self._t_act.x.scatter_forward()
        return self._t_act

    # ── диагностика ───────────────────────────────────────────────────
    def potential_range(self) -> tuple[float, float]:
        """Глобальный диапазон потенциала — дешёвая проверка на разнос."""
        owned = self.potential()[:self.n_owned]
        lo = self.comm.allreduce(
            float(owned.min()) if owned.size else np.inf, op=MPI.MIN)
        hi = self.comm.allreduce(
            float(owned.max()) if owned.size else -np.inf, op=MPI.MAX)
        return lo, hi

    def summary(self) -> str:
        return (f"  Монодомен    : {self.cell.describe()}\n"
                f"                 θ = {self.theta} "
                f"({'Кранк–Николсон' if self.theta == 0.5 else 'θ-схема'}), "
                f"CG + HYPRE")
