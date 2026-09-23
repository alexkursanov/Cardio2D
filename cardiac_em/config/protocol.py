"""
Протоколы: стимуляция, временная шкала, преднагрузка.
======================================================

Здесь лежит всё, что описывает ХОД эксперимента, в отличие от свойств
ткани (tissue_spec.py) и геометрии (mesh_spec.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["StimulusProtocol", "TimeStepping", "PreloadProtocol"]


@dataclass(frozen=True)
class StimulusProtocol:
    """
    Стимуляция левого края области серией импульсов.

    Пространственный профиль: экспоненциальный спад от x = 0 с
    характерной длиной `decay_length_mm`, обрезанный при
    x > `x_max_mm`. Временной профиль каждого импульса — трапеция с
    фронтами длительности `rise_time_ms`.

    Поля
    ----
    times_ms        : моменты НАЧАЛА импульсов, мс (например,
                      (0.0, 1000.0, 2000.0) для протокола S1-S1-S1)
    duration_ms     : длительность каждого импульса, мс
    amplitude       : амплитуда (б/р, в единицах модели Rogers–McCulloch)
    rise_time_ms    : время нарастания/спада фронта, мс
    decay_length_mm : характерная длина пространственного спада, мм
    x_max_mm        : за этим x стимул тождественно нулевой, мм

    ВАЛИДАЦИЯ: окна импульсов [t, t + duration] не должны пересекаться.
    В прежней реализации пересечение приводило к тому, что молча
    срабатывал только первый подходящий импульс — теперь это ошибка.
    """

    times_ms: tuple[float, ...] = (50.0,)
    duration_ms: float = 2.0
    amplitude: float = 20.0
    rise_time_ms: float = 0.5
    decay_length_mm: float = 0.5
    x_max_mm: float = 2.0

    def __post_init__(self) -> None:
        # нормализуем к кортежу: защищает от изменяемого списка в frozen-объекте
        object.__setattr__(self, "times_ms", tuple(float(t) for t in self.times_ms))

        if self.duration_ms <= 0:
            raise ValueError(f"duration_ms должна быть > 0, получено {self.duration_ms}")
        if self.rise_time_ms < 0:
            raise ValueError(f"rise_time_ms не может быть < 0, получено {self.rise_time_ms}")
        if 2 * self.rise_time_ms > self.duration_ms:
            raise ValueError(
                f"фронты не помещаются в импульс: 2·rise_time_ms="
                f"{2 * self.rise_time_ms} > duration_ms={self.duration_ms}")
        if self.decay_length_mm <= 0:
            raise ValueError(
                f"decay_length_mm должна быть > 0, получено {self.decay_length_mm}")
        if self.x_max_mm <= 0:
            raise ValueError(f"x_max_mm должна быть > 0, получено {self.x_max_mm}")
        if any(t < 0 for t in self.times_ms):
            raise ValueError(f"моменты стимуляции не могут быть < 0: {self.times_ms}")

        ordered = tuple(sorted(self.times_ms))
        if ordered != self.times_ms:
            raise ValueError(
                f"моменты стимуляции должны идти по возрастанию, получено "
                f"{self.times_ms}; отсортированный вариант: {ordered}")

        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt < prev + self.duration_ms:
                raise ValueError(
                    f"импульсы на t={prev} и t={nxt} мс перекрываются "
                    f"(длительность {self.duration_ms} мс); разнесите их "
                    f"минимум на {self.duration_ms} мс")

    def active_onset(self, t_ms: float) -> float | None:
        """
        Момент начала импульса, в чьё окно попадает t. None — вне импульсов.
        Окна не пересекаются (гарантировано валидацией), поэтому ответ
        однозначен.
        """
        for t0 in self.times_ms:
            if t0 <= t_ms <= t0 + self.duration_ms:
                return t0
        return None

    def envelope(self, t_ms: float) -> float:
        """Временная огибающая в момент t: 0 вне импульса, до 1 внутри."""
        t0 = self.active_onset(t_ms)
        if t0 is None:
            return 0.0
        if self.rise_time_ms == 0.0:
            return 1.0
        t_rel = t_ms - t0
        if t_rel < self.rise_time_ms:
            return t_rel / self.rise_time_ms
        if t_rel > self.duration_ms - self.rise_time_ms:
            return (self.duration_ms - t_rel) / self.rise_time_ms
        return 1.0

    def to_dict(self) -> dict:
        return {"times_ms": list(self.times_ms),
                "duration_ms": self.duration_ms,
                "amplitude": self.amplitude,
                "rise_time_ms": self.rise_time_ms,
                "decay_length_mm": self.decay_length_mm,
                "x_max_mm": self.x_max_mm}

    @staticmethod
    def from_dict(d: dict) -> "StimulusProtocol":
        return StimulusProtocol(
            times_ms=tuple(d.get("times_ms", (50.0,))),
            duration_ms=float(d.get("duration_ms", 2.0)),
            amplitude=float(d.get("amplitude", 20.0)),
            rise_time_ms=float(d.get("rise_time_ms", 0.5)),
            decay_length_mm=float(d.get("decay_length_mm", 0.5)),
            x_max_mm=float(d.get("x_max_mm", 2.0)),
        )


@dataclass(frozen=True)
class TimeStepping:
    """
    Временная шкала.

    Электрический шаг `dt_electric_ms` — основной. Механика решается
    реже, раз в `n_electric_per_mech` электрических шагов: поле
    перемещений меняется существенно медленнее фронта возбуждения, а
    каждый механический шаг — это нелинейное решение SNES.

    `t_end_ms` — АБСОЛЮТНОЕ время конца счёта. При продолжении с
    чекпоинта, снятого на t = 1500 мс, указывать нужно что-то большее
    1500, иначе считать будет нечего.
    """

    dt_electric_ms: float = 0.05
    n_electric_per_mech: int = 20
    t_end_ms: float = 1500.0

    def __post_init__(self) -> None:
        if self.dt_electric_ms <= 0:
            raise ValueError(
                f"dt_electric_ms должен быть > 0, получено {self.dt_electric_ms}")
        if self.n_electric_per_mech < 1:
            raise ValueError(
                f"n_electric_per_mech должен быть >= 1, получено "
                f"{self.n_electric_per_mech}")
        if self.t_end_ms <= 0:
            raise ValueError(f"t_end_ms должен быть > 0, получено {self.t_end_ms}")

    @property
    def dt_mech_ms(self) -> float:
        return self.dt_electric_ms * self.n_electric_per_mech

    @property
    def n_electric_steps(self) -> int:
        """Полное число электрических шагов от нуля до t_end."""
        return int(round(self.t_end_ms / self.dt_electric_ms)) + 1

    def step_index(self, t_ms: float) -> int:
        """Номер электрического шага, соответствующий времени t."""
        return int(round(t_ms / self.dt_electric_ms))

    def to_dict(self) -> dict:
        return {"dt_electric_ms": self.dt_electric_ms,
                "n_electric_per_mech": self.n_electric_per_mech,
                "t_end_ms": self.t_end_ms}

    @staticmethod
    def from_dict(d: dict) -> "TimeStepping":
        return TimeStepping(
            dt_electric_ms=float(d.get("dt_electric_ms", 0.05)),
            n_electric_per_mech=int(d.get("n_electric_per_mech", 20)),
            t_end_ms=float(d.get("t_end_ms", 1500.0)),
        )


@dataclass(frozen=True)
class PreloadProtocol:
    """
    Предварительное растяжение ткани до фиксации границ (этапы 1-2
    расчёта: одноосное растяжение вдоль волокон, затем зажим всех граней
    в достигнутом положении).

    `stretch` — целевое λ_f, `n_steps` — число шагов нагружения
    (растяжение подаётся постепенно, иначе Ньютон не сходится).
    """

    stretch: float = 1.2575
    n_steps: int = 10

    def __post_init__(self) -> None:
        if self.stretch <= 0:
            raise ValueError(f"stretch должен быть > 0, получено {self.stretch}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps должен быть >= 1, получено {self.n_steps}")

    @property
    def is_trivial(self) -> bool:
        """λ_f = 1 — растягивать нечего, этап можно пропустить."""
        return abs(self.stretch - 1.0) < 1e-12

    def to_dict(self) -> dict:
        return {"stretch": self.stretch, "n_steps": self.n_steps}

    @staticmethod
    def from_dict(d: dict) -> "PreloadProtocol":
        return PreloadProtocol(
            stretch=float(d.get("stretch", 1.2575)),
            n_steps=int(d.get("n_steps", 10)),
        )
