"""
Расписание шагов по времени.
=============================

Расписание отвечает на вопросы «что делать на этом шаге»: решать ли
механику, писать ли поля, сохранять ли чекпоинт, снимать ли снимок. Сам
расчёт оно не делает — только выдаёт последовательность `Tick`.

Модуль намеренно не зависит от DOLFINx: логика расписания — место
классических ошибок на единицу (какой шаг первый механический, где
кончается счёт, не теряется ли фаза при рестарте), и её полезно
проверять отдельно от солверов.

Соглашения
----------
* Шаг k переводит состояние из момента t_k = k·dt в t_{k+1} = (k+1)·dt.
  `Tick.t_ms` — начало шага, `Tick.t_end_ms` — конец. Всё, что
  наблюдается после обработки тика, относится к `t_end_ms`.

* Механика решается, когда КОНЕЦ шага приходится на кратное dt_мех:
  (k + 1) % n == 0. Так моменты механики — ровно dt_мех, 2·dt_мех, …
  В монолитной версии механика решалась на шагах k % n == 0, то есть в
  моменты dt, dt_мех + dt, … — со сдвигом на один электрический шаг.

* Последний тик ВСЕГДА механический и всегда пишет поля (и чекпоинт,
  если они включены), даже если t_end не кратно dt_мех: финальное
  состояние должно быть согласованным и сохранённым.

* Всё, что пишется на диск, привязано к механическим тикам: только на
  них состояние обеих задач согласовано между собой.

* Фаза механики ГЛОБАЛЬНА: номер механического шага считается от нуля
  времени, а не от точки старта. При продолжении с чекпоинта моменты
  механики и частоты записи не сдвигаются.

* Снимок в момент ts берётся с первого согласованного состояния не
  раньше ts: для ts ≤ t_start — это начальное состояние
  (`initial_snapshots`), иначе — первый механический тик с
  t_end ≥ ts. Снимки после конца счёта не срабатывают и перечислены в
  `unreachable_snapshots`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config.protocol import TimeStepping
from ..config.simulation import OutputConfig

__all__ = ["Tick", "Schedule"]

_EPS = 1e-9


@dataclass(frozen=True)
class Tick:
    """Один электрический шаг и всё, что к нему приписано."""

    index: int                     # глобальный номер электрического шага
    t_ms: float                    # начало шага
    t_end_ms: float                # конец шага
    is_mech: bool                  # решать ли механику после этого шага
    mech_index: int | None         # глобальный номер механического шага
    write_fields: bool             # писать поля в файл
    checkpoint: bool               # сохранять чекпоинт
    snapshots: tuple[float, ...]   # заказанные моменты снимков, выпадающие сюда
    is_last: bool


class Schedule:
    """
    Последовательность тиков от t_start до t_end.

    Параметры
    ---------
    time : TimeStepping
    output : OutputConfig, optional
        Частоты записи и моменты снимков. Без него расписание выдаёт
        только электрические и механические шаги.
    t_start_ms : float
        Момент старта; при продолжении с чекпоинта — время чекпоинта.
        Обязан лежать на сетке электрических шагов.
    """

    def __init__(self, time: TimeStepping, output: OutputConfig | None = None,
                 t_start_ms: float = 0.0):
        self.time = time
        self.output = output
        self.dt = time.dt_electric_ms
        self.n_per_mech = time.n_electric_per_mech

        k0 = round(t_start_ms / self.dt)
        if abs(k0 * self.dt - t_start_ms) > _EPS * max(1.0, abs(t_start_ms)):
            raise ValueError(
                f"момент старта {t_start_ms} мс не лежит на сетке шагов "
                f"dt = {self.dt} мс")
        k_end = round(time.t_end_ms / self.dt)
        if k_end <= k0:
            raise ValueError(
                f"нечего считать: конец счёта {time.t_end_ms} мс не позже "
                f"старта {t_start_ms} мс. При продолжении с чекпоинта "
                f"t_end — АБСОЛЮТНОЕ время")

        self.k_start = k0
        self.k_end = k_end
        self.t_start_ms = k0 * self.dt

        self._snapshot_map, self.initial_snapshots, self.unreachable_snapshots = \
            self._assign_snapshots()

    # ── основные свойства ─────────────────────────────────────────────
    def __len__(self) -> int:
        return self.k_end - self.k_start

    @property
    def t_end_ms(self) -> float:
        return self.k_end * self.dt

    def _is_mech(self, k: int) -> bool:
        return (k + 1) % self.n_per_mech == 0 or k == self.k_end - 1

    def _mech_index(self, k: int) -> int:
        """
        Глобальный номер механического шага: сколько полных механических
        интервалов уложилось в [0, t_end шага]. Для внеочередного
        финального тика это номер следующего по счёту интервала.
        """
        return -(-(k + 1) // self.n_per_mech)          # ceil((k+1)/n)

    @property
    def mech_ticks(self) -> list[int]:
        """Номера электрических шагов, после которых решается механика."""
        return [k for k in range(self.k_start, self.k_end) if self._is_mech(k)]

    @property
    def n_mech_ticks(self) -> int:
        return len(self.mech_ticks)

    # ── снимки ────────────────────────────────────────────────────────
    def _assign_snapshots(self):
        if self.output is None:
            return {}, (), ()

        mech_ticks = self.mech_ticks
        mapping: dict[int, list[float]] = {}
        initial, unreachable = [], []

        for ts in self.output.snapshot_times_ms:
            if ts <= self.t_start_ms + _EPS:
                initial.append(ts)
                continue
            for k in mech_ticks:
                if (k + 1) * self.dt >= ts - _EPS:
                    mapping.setdefault(k, []).append(ts)
                    break
            else:
                unreachable.append(ts)

        return ({k: tuple(v) for k, v in mapping.items()},
                tuple(initial), tuple(unreachable))

    # ── итерация ──────────────────────────────────────────────────────
    def __iter__(self):
        out = self.output
        for k in range(self.k_start, self.k_end):
            is_last = k == self.k_end - 1
            is_mech = self._is_mech(k)
            m = self._mech_index(k) if is_mech else None

            write = checkpoint = False
            if is_mech and out is not None:
                write = is_last or m % out.save_every_mech_steps == 0
                checkpoint = out.checkpoints_enabled and (
                    is_last or m % out.ckpt_every_mech_steps == 0)

            yield Tick(
                index=k,
                t_ms=round(k * self.dt, 12),
                t_end_ms=round((k + 1) * self.dt, 12),
                is_mech=is_mech,
                mech_index=m,
                write_fields=write,
                checkpoint=checkpoint,
                snapshots=self._snapshot_map.get(k, ()),
                is_last=is_last,
            )

    def summary(self) -> str:
        return (f"  Расписание   : {len(self)} эл. шагов, "
                f"{self.n_mech_ticks} мех., "
                f"{self.t_start_ms:g} → {self.t_end_ms:g} мс")
