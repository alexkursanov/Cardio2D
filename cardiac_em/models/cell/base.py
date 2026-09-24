"""
Абстракция модели клетки.
==========================

`CellModel` описывает систему ОДУ одной клетки, но работает СРАЗУ НАД
ВСЕМИ узлами сетки: состояние — массив формы (n_nodes, n_states), все
операции векторизованы. Явных циклов по узлам здесь быть не должно —
на них уходит всё время счёта.

DOLFINx этот слой не использует: он получает numpy-массивы и возвращает
numpy-массивы. Благодаря этому модель клетки можно разрабатывать и
проверять отдельно от тканевой задачи — например, прогнать одиночную
клетку и посмотреть форму потенциала действия.

Интегратор
----------
Базовый класс реализует ОДИН шаг по времени, разделяя переменные на два
сорта:

  * ВОРОТНЫЕ (gating, хатчкин-хаксливского вида), для которых
    справедливо  dg/dt = (g_∞ − g)/τ. Они интегрируются схемой
    Раша–Ларсена: g ← g_∞ + (g − g_∞)·exp(−dt/τ). Это точное решение
    уравнения при замороженных g_∞ и τ, поэтому шаг устойчив даже
    когда τ много меньше dt — а именно на этом явный Эйлер и взрывается
    в подробных ионных моделях;

  * ОСТАЛЬНЫЕ (концентрации, медленные переменные восстановления) —
    явным Эйлером.

Модель, у которой воротных переменных нет (`gate_indices = ()`), просто
не заходит в первую ветку: накладных расходов это не создаёт. Механизм
Раша–Ларсена проверяется тестами на синтетической модели с одним
воротом, чтобы он не оставался непроверенным кодом до появления первой
подробной модели.

Параметры
---------
Параметры НЕ принадлежат модели: они приходят извне как словарь
numpy-массивов по одному значению на узел (из `fem/tissue.py`, где
разложены по регионам). Модель объявляет, какие имена ей нужны
(`param_names`), и получает их в `rhs`/`gate_inf_tau`. Так одна и та же
модель работает и на однородной ткани, и на ткани с ишемической зоной,
ничего не зная о регионах.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["CellModel"]


class CellModel(ABC):
    """
    Базовый класс модели клетки.

    Атрибуты класса, которые обязана определить реализация
    -------------------------------------------------------
    name           : короткое имя для реестра и сообщений
    state_names    : имена переменных состояния, по порядку
    v_index        : индекс трансмембранного потенциала в состоянии
    gate_indices   : индексы воротных переменных (может быть пустым)
    param_names    : имена нужных модели параметров ткани
    tension_kind   : "scaled" — `active_tension` безразмерна и её
                     умножают на T_MAX; "absolute" — уже в кПа
    state_bounds   : необязательные ограничения (индекс → (lo, hi)),
                     применяемые после каждого шага
    suggested_dt_ms: ориентир по устойчивому шагу, для предупреждений
    activation_threshold: порог потенциала (в единицах модели), при
                     пересечении которого вверх узел считается
                     активированным — для карт времён активации. Для
                     безразмерных моделей 0.5; модели в мВ задают своё
                     (например, −40 мВ)
    """

    name: str = "abstract"
    state_names: tuple[str, ...] = ()
    v_index: int = 0
    gate_indices: tuple[int, ...] = ()
    param_names: tuple[str, ...] = ()
    tension_kind: str = "scaled"
    state_bounds: dict[int, tuple[float, float]] = {}
    suggested_dt_ms: float = 0.05
    activation_threshold: float = 0.5

    # ── размеры и начальное состояние ─────────────────────────────────
    @property
    def n_states(self) -> int:
        return len(self.state_names)

    @property
    def non_gate_indices(self) -> tuple[int, ...]:
        gates = set(self.gate_indices)
        return tuple(i for i in range(self.n_states) if i not in gates)

    @abstractmethod
    def resting_state(self) -> np.ndarray:
        """Состояние покоя одной клетки, (n_states,)."""

    def initial_state(self, n_nodes: int) -> np.ndarray:
        """Состояние покоя, размноженное по узлам, (n_nodes, n_states)."""
        return np.tile(self.resting_state(), (n_nodes, 1))

    def state_index(self, name: str) -> int:
        """Индекс переменной по имени — чтобы не зашивать числа в код."""
        try:
            return self.state_names.index(name)
        except ValueError:
            raise KeyError(
                f"у модели {self.name} нет переменной {name!r}; "
                f"есть {self.state_names}") from None

    # ── правые части ──────────────────────────────────────────────────
    @abstractmethod
    def rhs_non_gate(self, t: float, y: np.ndarray, stim: np.ndarray,
                     params: dict[str, np.ndarray]) -> np.ndarray:
        """
        Производные НЕворотных переменных, (n_nodes, len(non_gate_indices)),
        в порядке `non_gate_indices`.

        `stim` — стимулирующий ток на узлах, (n_nodes,).
        """

    def gate_inf_tau(self, t: float, y: np.ndarray,
                     params: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Для воротных переменных — (g_∞, τ), обе формы
        (n_nodes, len(gate_indices)).

        Модель без воротных переменных этот метод не переопределяет.
        """
        if self.gate_indices:
            raise NotImplementedError(
                f"модель {self.name} объявила воротные переменные "
                f"{self.gate_indices}, но не определила gate_inf_tau")
        n = len(y)
        empty = np.zeros((n, 0))
        return empty, empty

    # ── шаг по времени ────────────────────────────────────────────────
    def step(self, t: float, y: np.ndarray, dt: float, stim: np.ndarray,
             params: dict[str, np.ndarray]) -> np.ndarray:
        """
        Один шаг: Раш–Ларсен для воротных переменных, явный Эйлер для
        остальных. Возвращает НОВЫЙ массив, исходный не меняется.

        Порядок важен: правые части и (g_∞, τ) вычисляются от состояния
        НА НАЧАЛО шага, до всякого обновления. Иначе часть переменных
        пошла бы по уже обновлённым значениям, и схема перестала бы
        быть тем, чем заявлена.
        """
        self._check_state(y)
        y_new = y.copy()

        gates = self.gate_indices
        non_gates = self.non_gate_indices

        if gates:
            g_inf, tau = self.gate_inf_tau(t, y, params)
            if np.any(tau <= 0):
                raise FloatingPointError(
                    f"модель {self.name}: τ должна быть > 0, получено "
                    f"минимум {float(np.min(tau))}")
            g = y[:, gates]
            y_new[:, gates] = g_inf + (g - g_inf) * np.exp(-dt / tau)

        if non_gates:
            dydt = self.rhs_non_gate(t, y, stim, params)
            y_new[:, non_gates] = y[:, non_gates] + dt * dydt

        for idx, (lo, hi) in self.state_bounds.items():
            np.clip(y_new[:, idx], lo, hi, out=y_new[:, idx])

        return y_new

    def _check_state(self, y: np.ndarray) -> None:
        if y.ndim != 2 or y.shape[1] != self.n_states:
            raise ValueError(
                f"состояние модели {self.name} должно иметь форму "
                f"(n_nodes, {self.n_states}), получено {y.shape}")

    # ── наблюдаемые величины ──────────────────────────────────────────
    def V(self, y: np.ndarray) -> np.ndarray:
        """Трансмембранный потенциал (в единицах модели), (n_nodes,)."""
        return y[:, self.v_index]

    @abstractmethod
    def active_tension(self, y: np.ndarray,
                       params: dict[str, np.ndarray]) -> np.ndarray:
        """
        Сигнал активного напряжения, (n_nodes,).

        Если `tension_kind == "scaled"` — безразмерная величина, которую
        вызывающий умножит на T_MAX. Если `"absolute"` — уже в кПа
        (так будет у моделей со встроенной механикой сокращения).
        """

    def calcium(self, y: np.ndarray) -> np.ndarray | None:
        """
        Концентрация кальция, если модель её считает. None — если нет.
        Нужна для моделей активного напряжения, берущих на вход Ca.
        """
        return None

    # ── проверка параметров ───────────────────────────────────────────
    def check_params(self, params: dict[str, np.ndarray], n_nodes: int) -> None:
        """
        Убедиться, что переданы все нужные параметры и они согласованы по
        длине с состоянием. Дешевле проверить один раз при старте, чем
        ловить рассогласование по длине в середине счёта.
        """
        missing = [k for k in self.param_names if k not in params]
        if missing:
            raise KeyError(
                f"модели {self.name} не переданы параметры {missing}; "
                f"есть {sorted(params)}")
        for key in self.param_names:
            arr = np.asarray(params[key])
            if arr.ndim == 0:
                continue          # скаляр допустим — он broadcast'ится
            if arr.shape[0] < n_nodes:
                raise ValueError(
                    f"параметр {key!r}: длина {arr.shape[0]} меньше числа "
                    f"узлов {n_nodes}")

    def describe(self) -> str:
        gates = (f"{len(self.gate_indices)} воротных"
                 if self.gate_indices else "без воротных переменных")
        return (f"{self.name}: {self.n_states} переменных "
                f"({', '.join(self.state_names)}), {gates}, "
                f"напряжение — {self.tension_kind}")
