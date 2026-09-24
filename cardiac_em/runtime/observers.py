"""
Наблюдатели за ходом расчёта.
==============================

Расчётный цикл ничего не печатает и ничего не пишет на диск сам. Всё,
что должно происходить «по ходу» — вывод прогресса, запись полей,
чекпоинты, сбор истории для анализа, а в будущем — передача состояния
интерфейсу управления, — делается наблюдателями, которых цикл
оповещает о событиях.

Так цикл остаётся коротким, а новые способы следить за расчётом
добавляются без правки цикла: достаточно написать ещё одного
наблюдателя.

ВАЖНО: наблюдатели вызываются на ВСЕХ рангах MPI. Диагностика
(`Simulation.diagnostics()`) содержит коллективные операции, и если
вызвать её только на нулевом ранге, программа зависнет. Печатает только
нулевой ранг, но считают все. Правило для своих наблюдателей: любая
коллективная операция — на всех рангах в одном и том же порядке.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:                                # только для подсказок типов
    from .schedule import Schedule, Tick
    from .simulation import Simulation

__all__ = ["Observer", "ConsoleObserver", "TraceObserver"]


class Observer:
    """
    Базовый наблюдатель: все методы ничего не делают. Наследник
    переопределяет только то, что ему нужно.
    """

    def on_preload_step(self, sim: "Simulation", k: int, n: int) -> None:
        """После k-го из n шагов преднагрузки."""

    def on_preload_done(self, sim: "Simulation") -> None:
        """Преднагрузка закончена, границы зажаты."""

    def on_start(self, sim: "Simulation", schedule: "Schedule") -> None:
        """Перед первым шагом активации. Состояние — на t_start."""

    def on_electric_step(self, sim: "Simulation", tick: "Tick") -> None:
        """
        После КАЖДОГО электрического шага (до механики, если тик
        механический). Для регистраторов, которым нужно временное
        разрешение dt_эл, — например, карт времён активации. Вызывается
        десятки тысяч раз: здесь только дешёвые локальные операции, без
        коллективных вызовов MPI.
        """

    def on_mech_tick(self, sim: "Simulation", tick: "Tick") -> None:
        """После механического тика: обе задачи согласованы на t_end."""

    def on_finish(self, sim: "Simulation") -> None:
        """После последнего тика."""

    def on_abort(self, sim: "Simulation", exc: BaseException) -> None:
        """
        Счёт прерван исключением (несходимость, Ctrl+C, …). После этого
        исключение пробрасывается дальше.

        ЗДЕСЬ НЕЛЬЗЯ делать коллективных операций MPI: ошибка могла
        случиться не на всех рангах, и коллективный вызов зависнет.
        Только локальные действия — закрыть файл, записать статус с
        нулевого ранга.
        """


class ConsoleObserver(Observer):
    """
    Печать прогресса в консоль (только нулевой ранг).

    `every_mech_steps` — как часто печатать строку таблицы. Диагностика
    считается только на печатаемых тиках: на ней есть интерполяции, и
    делать их каждый механический шаг ради вывода незачем.
    """

    def __init__(self, every_mech_steps: int = 20, stream: TextIO | None = None):
        if every_mech_steps < 1:
            raise ValueError("every_mech_steps должен быть >= 1")
        self.every = every_mech_steps
        self.stream = stream or sys.stdout
        self._t0 = None

    def _print(self, sim, text: str) -> None:
        if sim.comm.rank == 0:
            print(text, file=self.stream, flush=True)

    def on_preload_step(self, sim, k, n):
        d = sim.diagnostics(electric=False)
        self._print(sim, f"  преднагрузка {k:2d}/{n}: λ_f ∈ "
                         f"[{d['lambda_f_min']:.4f}, {d['lambda_f_max']:.4f}], "
                         f"Ньютон {d['newton_iterations']}")

    def on_preload_done(self, sim):
        self._print(sim, "  границы зажаты в растянутом положении")

    def on_start(self, sim, schedule):
        self._t0 = time.perf_counter()
        for w in getattr(sim, "warnings", ()):
            self._print(sim, f"  [!] {w}")
        self._print(sim, schedule.summary())
        self._print(sim, f"  {'t, мс':>8}  {'u_max':>6}  {'T_эл':>7}  "
                         f"{'T_мех':>7}  {'σ_xx':>7}  {'λ_f':>13}  {'Ньют':>4}")

    def on_mech_tick(self, sim, tick):
        if not (tick.mech_index % self.every == 0 or tick.is_last):
            return
        d = sim.diagnostics()
        self._print(sim,
                    f"  {tick.t_end_ms:8.1f}  {d['u_max']:6.3f}  "
                    f"{d['t_act_electric_max']:7.2f}  {d['t_act_mech_max']:7.2f}  "
                    f"{d['sigma_xx_max']:7.2f}  "
                    f"{d['lambda_f_min']:.4f}–{d['lambda_f_max']:.4f}  "
                    f"{d['newton_iterations']:4d}")

    def on_finish(self, sim):
        elapsed = time.perf_counter() - (self._t0 or time.perf_counter())
        self._print(sim, f"  готово за {elapsed:.1f} с")

    def on_abort(self, sim, exc):
        self._print(sim, f"  [прервано на t = {sim.t_ms:g} мс] "
                         f"{type(exc).__name__}: {exc}")


class TraceObserver(Observer):
    """
    Собирает историю диагностики на каждом механическом тике.

    Нужен для тестов и для быстрого анализа прямо в процессе — например,
    посмотреть кривую T_act(t) без записи полей на диск. На всех рангах
    история одинакова (диагностика глобальная).
    """

    def __init__(self):
        self.records: list[dict] = []
        self.preload: list[dict] = []

    def on_preload_step(self, sim, k, n):
        d = sim.diagnostics(electric=False)
        d["step"] = k
        self.preload.append(d)

    def on_start(self, sim, schedule):
        d = sim.diagnostics()
        d["t_ms"] = schedule.t_start_ms
        d["mech_index"] = None
        self.records.append(d)

    def on_mech_tick(self, sim, tick):
        d = sim.diagnostics()
        d["t_ms"] = tick.t_end_ms
        d["mech_index"] = tick.mech_index
        self.records.append(d)

    def series(self, key: str) -> list:
        """Временной ряд одной величины: [(t, значение), …]."""
        return [(r["t_ms"], r[key]) for r in self.records]
