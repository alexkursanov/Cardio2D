"""
Связанный расчёт: электрика + механика.
========================================

`Simulation` собирает всё из `SimulationConfig` и ведёт счёт в два этапа:

    1. ПРЕДНАГРУЗКА — ткань постепенно растягивается вдоль волокон до
       заданного λ_f, затем все границы зажимаются в достигнутом
       положении.
    2. АКТИВАЦИЯ — электрика идёт каждый шаг; на механических шагах
       активное напряжение переносится на механическую сетку и решается
       равновесие.

    sim = Simulation(config, observers=[ConsoleObserver()])
    sim.preload()
    sim.run()

Никаких глобальных параметров: всё берётся из `config`, поэтому в одном
процессе можно последовательно выполнить сколько угодно разных
конфигураций — это основа будущих параметрических серий.

Ни печати, ни записи на диск здесь нет. О ходе счёта оповещаются
наблюдатели (`runtime/observers.py`); запись полей, чекпоинтов и
манифеста — тоже наблюдатели, из слоя `io/`.

Продолжение с чекпоинта: вместо `preload()` —

    info = io.restore_checkpoint(sim, "old/ckpt_last.npz")
    sim.run(t_start_ms=info.t_start_ms)

Отличия от монолитной версии
-----------------------------
* Преднагрузка идёт с условиями симметрии, а не с точечной связью:
  точечная связь давала сингулярность напряжения и неоднородное
  растяжение (λ_f = 1.055…1.108 при заданном 1.1 на сетке 8×8).

* Граничные условия зажима ставятся ОДИН раз после преднагрузки. В
  монолитной версии они пересоздавались на каждом механическом шаге
  (с копированием u), что заставляло каждый раз пересобирать объект
  нелинейной задачи. Граничные узлы при зажиме не двигаются, так что
  результат тот же, а работы меньше.

* Механика решается в моменты, кратные dt_мех, а не со сдвигом на один
  электрический шаг (подробнее — `runtime/schedule.py`).
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI

from ..config.simulation import SimulationConfig
from ..coupling import CellToNodeSampler, make_transfer
from ..fem import Tissue, build_mesh_pair
from ..models.active import make_active_law
from ..models.cell import CellModel, make_cell_model
from ..models.passive import PassiveMaterial, make_passive_material
from ..solvers import (
    MechanicsSolver,
    MonodomainSolver,
    bcs_clamped_at_current_state,
    bcs_uniaxial_stretch_symmetric,
)
from .observers import Observer
from .schedule import Schedule

__all__ = ["Simulation"]


class Simulation:
    """
    Связанная электромеханическая задача.

    Параметры
    ---------
    config : SimulationConfig
    cell_model : CellModel, optional
        Готовый объект модели клетки вместо `config.cell_model`. Нужен,
        когда модель создаётся с нестандартными аргументами. Если её имя
        расходится с конфигурацией — предупреждение: в run.json попадёт
        имя из конфигурации, и повторный прогон соберёт другую физику.
    material : PassiveMaterial, optional
        То же для `config.passive_material`.
    transfer_strategy : "auto" | "averaging" | "nearest"
        Как переносить T_act на механическую сетку (см. `coupling/`).
    observers : последовательность Observer
    comm : MPI.Comm, optional
    """

    def __init__(self, config: SimulationConfig, *,
                 cell_model: CellModel | None = None,
                 material: PassiveMaterial | None = None,
                 transfer_strategy: str = "auto",
                 observers: tuple[Observer, ...] = (),
                 comm: MPI.Comm | None = None):
        self.config = config
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.observers = list(observers)

        # ── модели (до ткани: ткани нужны параметры модели клетки) ────
        self._explicit_models = []
        if cell_model is None:
            cell_model = make_cell_model(config.cell_model)
        elif cell_model.name != config.cell_model:
            self._explicit_models.append(
                f"модель клетки {cell_model.name!r} передана объектом, а в "
                f"конфигурации указана {config.cell_model!r} — сохранённая "
                f"конфигурация не воспроизведёт этот прогон")
        if material is None:
            material = make_passive_material(config.passive_material)
        elif material.name != config.passive_material:
            self._explicit_models.append(
                f"материал {material.name!r} передан объектом, а в "
                f"конфигурации указан {config.passive_material!r} — "
                f"сохранённая конфигурация не воспроизведёт этот прогон")

        self.cell_model = cell_model

        cell_params = dict(cell_model.default_params())
        unknown = set(config.cell_params) - set(cell_params)
        if unknown:
            raise KeyError(
                f"config.cell_params: у модели {cell_model.name!r} нет параметров "
                f"{sorted(unknown)}; есть: {sorted(cell_params) or 'ни одного'}")
        cell_params.update(config.cell_params)

        # ── сетки и ткань ─────────────────────────────────────────────
        self.pair = build_mesh_pair(config.mesh, self.comm)
        regions = tuple(config.regions)
        self.tissue_e = Tissue.for_electrics(
            self.pair.electric, config.tissue_base, regions,
            cell_params=cell_params)
        self.tissue_m = Tissue.for_mechanics(
            self.pair.mechanical, config.tissue_base, regions)

        # ── солверы и связь ───────────────────────────────────────────
        self.electrics = MonodomainSolver(
            self.tissue_e, self.cell_model, config.stimulus)
        self.active_law = make_active_law(self.cell_model)
        self.mechanics = MechanicsSolver(self.tissue_m, material,
                                         active_law=self.active_law,
                                         dt_mech=config.time.dt_mech_ms)
        self.transfer = make_transfer(
            self.tissue_e.DG0, self.tissue_m.DG0,
            config.mesh.electric, config.mesh.mechanical,
            strategy=transfer_strategy)
        # Обратный канал: растяжение волокна и его скорость — клеткам.
        self.stretch_sampler = (
            CellToNodeSampler(self.tissue_m.DG0, self.tissue_e.P1, config.mesh.mechanical)
            if self.cell_model.stretch_sensitive else None)
        self._t_last_mech = 0.0

        # ── состояние хода счёта ──────────────────────────────────────
        self.t_ms = 0.0
        self.is_preloaded = False
        self.last_newton_iterations = 0
        # Заполняется io.restore_checkpoint: откуда взято состояние.
        # Попадает в манифест. None — счёт с нуля.
        self.restart_info: dict | None = None

        self._area_e = config.mesh.electric.hx_mm * config.mesh.electric.hy_mm
        self._area_m = config.mesh.mechanical.hx_mm * config.mesh.mechanical.hy_mm
        if config.mesh.electric.cell_type == "triangle":
            self._area_e /= 2
        if config.mesh.mechanical.cell_type == "triangle":
            self._area_m /= 2

        self.warnings = self._collect_warnings()

    # ── предупреждения при старте ─────────────────────────────────────
    def _collect_warnings(self) -> list[str]:
        """
        Всё подозрительное, что можно обнаружить до начала счёта.
        Считается на всех рангах (есть коллективные операции).
        """
        found = list(self.config.check())
        found.extend(self._explicit_models)

        for label, tissue in (("электрической", self.tissue_e),
                              ("механической", self.tissue_m)):
            for i in tissue.empty_regions():
                name = self.config.regions[i].name or f"#{i}"
                found.append(
                    f"регион {name} не покрыл ни одной ячейки {label} сетки — "
                    f"он мельче её шага")

        if not self.transfer.exact:
            found.append(
                f"перенос T_act на механическую сетку приближённый "
                f"({self.transfer.describe()})")
        return found

    # ── оповещение наблюдателей ───────────────────────────────────────
    def _notify(self, method: str, *args) -> None:
        for obs in self.observers:
            getattr(obs, method)(self, *args)

    def add_observer(self, observer: Observer) -> None:
        self.observers.append(observer)

    # ── этап 1: преднагрузка ──────────────────────────────────────────
    def preload(self) -> None:
        """
        Подготовка клеток (если задана `preload.cell_relax_ms`), затем
        постепенное растяжение до λ_f = config.preload.stretch и зажим
        границ в достигнутом положении.

        При λ_f = 1 растягивать нечего: границы зажимаются сразу.
        """
        if self.is_preloaded:
            raise RuntimeError("преднагрузка уже выполнена")

        spec = self.config.mesh.mechanical
        p = self.config.preload
        if p.cell_relax_ms > 0:
            # До растяжения: клетки приходят к своему покою (регионы с
            # другими параметрами — к своему). См. PreloadProtocol.
            self.electrics.relax(p.cell_relax_ms, self.config.time.dt_electric_ms)
        self.mechanics.clear_active_tension()

        if not p.is_trivial:
            delta_total = (p.stretch - 1.0) * spec.lx_mm
            for k in range(1, p.n_steps + 1):
                delta = delta_total * k / p.n_steps
                self.mechanics.set_bcs(
                    bcs_uniaxial_stretch_symmetric(self.mechanics, spec, delta))
                self.last_newton_iterations = self.mechanics.solve_or_raise()
                self._notify("on_preload_step", k, p.n_steps)

        self.mechanics.set_bcs(
            bcs_clamped_at_current_state(self.mechanics, spec))
        self.last_newton_iterations = self.mechanics.solve_or_raise()
        # клетки узнают растяжение преднагрузки (скорость — ноль)
        self._feed_stretch(zero_rate=True)
        self.mechanics.commit_step()
        self.is_preloaded = True
        self._notify("on_preload_done")

    def resume(self, t_ms: float) -> None:
        """
        Принять ТЕКУЩЕЕ состояние солверов (обычно только что загруженное
        из чекпоинта) как стартовое на момент `t_ms` вместо преднагрузки.

        Границы зажимаются там, где стоит загруженное u; T_act
        пересчитывается из загруженного электрического состояния и
        переносится на механику; равновесие решается заново. Если
        чекпоинт снят этим же кодом с той же конфигурацией, состояние
        уже равновесное и Ньютон сходится за 0–1 итерацию. Если сетка
        или параметры другие — решение приводит u к равновесию для
        текущей конфигурации.
        """
        if self.is_preloaded:
            raise RuntimeError(
                "состояние уже подготовлено (preload или resume) — "
                "resume нужно вызывать на свежесобранной Simulation")
        spec = self.config.mesh.mechanical
        self.mechanics.set_bcs(bcs_clamped_at_current_state(self.mechanics, spec))
        # «Прошлое» растяжение в чекпоинте не хранится: берём текущее, то
        # есть первый шаг после продолжения считается от покоя волокна.
        # Для моделей без закона φ(v) это не влияет ни на что.
        self.mechanics.commit_step()
        self._couple_and_solve()
        self.t_ms = float(t_ms)
        self.is_preloaded = True

    # ── этап 2: активация ─────────────────────────────────────────────
    def run(self, t_start_ms: float = 0.0) -> Schedule:
        """
        Связанный расчёт от t_start до config.time.t_end_ms.

        `t_start_ms` отличен от нуля при продолжении с чекпоинта; состояние
        к этому моменту должно быть уже загружено (шаг 9).
        """
        if not self.is_preloaded:
            raise RuntimeError(
                "перед run() нужна преднагрузка: вызовите preload() "
                "(или загрузите состояние из чекпоинта)")

        schedule = Schedule(self.config.time, self.config.output, t_start_ms)
        self.t_ms = schedule.t_start_ms
        self._t_last_mech = schedule.t_start_ms
        self._notify("on_start", schedule)

        dt = schedule.dt
        # Только наблюдатели, переопределившие on_electric_step: вызов
        # пустого метода на каждом из десятков тысяч шагов ни к чему.
        per_step = [o for o in self.observers
                    if type(o).on_electric_step is not Observer.on_electric_step]
        try:
            for tick in schedule:
                self.electrics.step(tick.t_ms, dt)
                for obs in per_step:
                    obs.on_electric_step(self, tick)
                if tick.is_mech:
                    self._couple_and_solve(dt_ms=tick.t_end_ms - self._t_last_mech)
                    self._t_last_mech = tick.t_end_ms
                    self.t_ms = tick.t_end_ms
                    self._notify("on_mech_tick", tick)
        except BaseException as exc:
            # Наблюдатели узнают о сбое (манифест помечает прогон как
            # failed, файлы закрываются), после чего ошибка идёт дальше.
            for obs in self.observers:
                try:
                    obs.on_abort(self, exc)
                except Exception:  # noqa: BLE001 — не заслонять исходную ошибку
                    pass
            raise

        self._notify("on_finish")
        return schedule

    def _couple_and_solve(self, dt_ms: float | None = None) -> None:
        """
        T_act с электрической сетки → на механическую → равновесие →
        (если клетки чувствуют деформацию) растяжение и скорость волокна
        обратно клеткам.
        """
        if dt_ms is not None:
            self.mechanics.set_time_step(dt_ms)
        t_act_e = self.electrics.active_tension_dg0()
        t_act_m = self.transfer.apply(t_act_e)
        self.mechanics.set_active_tension(t_act_m)
        self.last_newton_iterations = self.mechanics.solve_or_raise()
        self._feed_stretch()
        self.mechanics.commit_step()

    def _feed_stretch(self, zero_rate: bool = False) -> None:
        """λ_f и dλ_f/dt с ячеек механики — в узлы электрики, клеткам."""
        if self.stretch_sampler is None:
            return
        m = self.mechanics
        lam = m.fiber_stretch().x.array[:m.n_cells_owned]
        rate = np.zeros_like(lam) if zero_rate else m.stretch_rate()
        self.cell_model.apply_stretch(self.electrics.state,
                                      self.stretch_sampler.apply(lam),
                                      self.stretch_sampler.apply(rate))

    # ── диагностика ───────────────────────────────────────────────────
    def diagnostics(self, electric: bool = True) -> dict:
        """
        Сводка текущего состояния — глобальные величины по всем рангам.

        КОЛЛЕКТИВНАЯ операция: вызывать на всех рангах одновременно.

        Интегралы T_act по области нужны для контроля связи: при точном
        переносе они на двух сетках совпадают, и расхождение сразу
        покажет, что перенос сломан.
        """
        m = self.mechanics
        lam_lo, lam_hi = m.value_range(m.fiber_stretch())
        J_lo, J_hi = m.value_range(m.jacobian_determinant())
        s_lo, s_hi = m.value_range(m.cauchy_stress(0, 0))

        t_m = m.T_act.x.array[:m.n_cells_owned]
        t_real = m.active_tension_actual().x.array[:m.n_cells_owned]
        d = {
            "t_ms": self.t_ms,
            "lambda_f_min": lam_lo, "lambda_f_max": lam_hi,
            "J_min": J_lo, "J_max": J_hi,
            "sigma_xx_min": s_lo, "sigma_xx_max": s_hi,
            "t_act_mech_max": self._gmax(t_m),
            "t_act_mech_integral": self._gsum(t_m) * self._area_m,
            # действующее напряжение (с законом φ(v)); без него — то же
            "t_act_actual_max": self._gmax(t_real),
            "t_act_actual_integral": self._gsum(t_real) * self._area_m,
            "newton_iterations": int(self.last_newton_iterations),
        }

        if electric:
            u_lo, u_hi = self.electrics.potential_range()
            t_e = self.electrics.active_tension_dg0()
            d.update({
                "u_min": u_lo, "u_max": u_hi,
                "t_act_electric_max": self._gmax(t_e),
                "t_act_electric_integral": self._gsum(t_e) * self._area_e,
            })
        else:
            d.update({"u_min": None, "u_max": None,
                      "t_act_electric_max": None,
                      "t_act_electric_integral": None})
        return d

    def _gmax(self, arr: np.ndarray) -> float:
        local = float(arr.max()) if arr.size else -np.inf
        return self.comm.allreduce(local, op=MPI.MAX)

    def _gsum(self, arr: np.ndarray) -> float:
        return self.comm.allreduce(float(arr.sum()), op=MPI.SUM)

    def summary(self) -> str:
        lines = [
            self.pair.summary(),
            self.electrics.summary(),
            self.mechanics.summary(),
            f"  Перенос T_act: {self.transfer.describe()}",
        ]
        for w in self.warnings:
            lines.append(f"  [!] {w}")
        return "\n".join(lines)
