"""
Параметры ткани и описание неоднородных областей.
==================================================

Два уровня:

  * БАЗОВЫЕ параметры (`TissueBaseParams`) — значения «по умолчанию»
    для всей ткани;
  * ОБЛАСТИ (`RegionSpec`) — геометрические фигуры с переопределениями
    отдельных параметров поверх базовых.

Области применяются по порядку списка: более поздние перекрывают более
ранние (и базовую ткань) в местах пересечения.

Плоские ключи переопределений
------------------------------
Переопределения задаются ПЛОСКИМ словарём с ключами вида "AP_c1",
"D_LONG", "MU" — имена совпадают с историческим форматом JSON-файлов
областей, поэтому старые файлы работают без правки. Соответствие
плоских ключей и полей dataclass задано в `FLAT_KEYS` каждой группы
параметров; `TissueBaseParams.flat_keys()` собирает полный набор.

Ключи переопределений ВАЛИДИРУЮТСЯ при создании `RegionSpec`: опечатка
вроде "AP_C1" или "D_Long" раньше молча не делала ничего, а теперь
немедленно даёт ошибку с перечнем допустимых имён.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Callable, Mapping

import numpy as np

__all__ = [
    "ReactionParams",
    "ConductionParams",
    "ActiveStressParams",
    "PassiveMechParams",
    "TissueBaseParams",
    "RegionSpec",
    "RectRegion",
    "CircleRegion",
    "CustomRegion",
    "region_from_dict",
    "load_regions_json",
]


# ═══════════════════════════════════════════════════════════════════════
#  ГРУППЫ ПАРАМЕТРОВ
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ReactionParams:
    """
    Параметры реакционной части (Rogers–McCulloch 1994), всё в мс.

        du/dt = c1·u·(u−a)·(1−u) − c2·u·v + I_stim
        dv/dt = b·(u − d·v)

    ЗАМЕЧАНИЕ: это параметры конкретной двухпеременной модели. Когда
    появится слой `models/cell/` с абстракцией CellModel, эта группа
    переедет в спецификацию модели клетки (`CellModelSpec`), а здесь
    останется только то, что относится к ткани как к среде.
    """

    c1: float = 0.26     # мс⁻¹ — скорость нарастания
    c2: float = 0.10     # мс⁻¹ — купирование через v
    a: float = 0.13      # б/р  — порог возбуждения
    b: float = 0.013     # мс⁻¹ — скорость восстановления v
    d: float = 1.0       # б/р  — коэффициент восстановления

    FLAT_KEYS = {"c1": "AP_c1", "c2": "AP_c2", "a": "AP_a",
                 "b": "AP_b", "d": "AP_d"}


@dataclass(frozen=True)
class ConductionParams:
    """Проводимость среды и ориентация волокон."""

    d_long: float = 0.3          # мм²/мс — диффузия вдоль волокон
    d_trans: float = 0.03        # мм²/мс — диффузия поперёк волокон
    fiber_angle_deg: float = 0.0  # градусы от оси x

    FLAT_KEYS = {"d_long": "D_LONG", "d_trans": "D_TRANS",
                 "fiber_angle_deg": "FIBER_ANGLE_DEG"}

    def __post_init__(self) -> None:
        if self.d_long < 0 or self.d_trans < 0:
            raise ValueError(
                f"коэффициенты диффузии не могут быть отрицательными: "
                f"d_long={self.d_long}, d_trans={self.d_trans}")


@dataclass(frozen=True)
class ActiveStressParams:
    """Активное напряжение, генерируемое возбуждением."""

    t_max: float = 120.0   # кПа — пиковое активное напряжение

    FLAT_KEYS = {"t_max": "T_MAX"}

    def __post_init__(self) -> None:
        # Ноль допустим (несокращающаяся ткань, рубец); отрицательное
        # значение превратило бы сокращение в активное растяжение.
        if self.t_max < 0:
            raise ValueError(f"t_max не может быть отрицательным, получено {self.t_max}")


@dataclass(frozen=True)
class PassiveMechParams:
    """
    Пассивная гиперупругость (трансверсально-изотропная):

        Ψ = μ/b·exp(b·(I₁−2)) + μ_f/(2b_f)·exp(b_f·(I₄−1)²) + κ/2·(J−1)²
    """

    mu: float = 1.0        # кПа — изотропная жёсткость матрицы
    b_iso: float = 1.0     # б/р
    mu_f: float = 3.0      # кПа — жёсткость вдоль волокон
    b_f: float = 2.0       # б/р
    kappa: float = 100.0   # кПа — штраф несжимаемости

    FLAT_KEYS = {"mu": "MU", "b_iso": "B_ISO", "mu_f": "MU_F",
                 "b_f": "B_F", "kappa": "KAPPA"}

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError(f"kappa должна быть > 0, получено {self.kappa}")


@dataclass(frozen=True)
class TissueBaseParams:
    """
    Базовые (фоновые) параметры ткани — то, что действует там, где
    не задано ни одной области.
    """

    reaction: ReactionParams = field(default_factory=ReactionParams)
    conduction: ConductionParams = field(default_factory=ConductionParams)
    active: ActiveStressParams = field(default_factory=ActiveStressParams)
    passive: PassiveMechParams = field(default_factory=PassiveMechParams)

    _GROUPS = ("reaction", "conduction", "active", "passive")

    def to_flat_dict(self) -> dict[str, float]:
        """
        Плоский словарь {"AP_c1": 0.26, "D_LONG": 0.3, ...}.

        Именно в таком виде параметры попадают в `fem/tissue.py`, где
        раскладываются по DOF, и в таком же виде задаются переопределения
        областей — один словарь ключей на всё.
        """
        flat: dict[str, float] = {}
        for group_name in self._GROUPS:
            group = getattr(self, group_name)
            for attr, flat_key in group.FLAT_KEYS.items():
                flat[flat_key] = float(getattr(group, attr))
        return flat

    @classmethod
    def flat_keys(cls) -> tuple[str, ...]:
        """Все допустимые ключи переопределений — для валидации областей."""
        keys: list[str] = []
        for group_name, f in zip(cls._GROUPS,
                                 (ReactionParams, ConductionParams,
                                  ActiveStressParams, PassiveMechParams)):
            keys.extend(f.FLAT_KEYS.values())
        return tuple(keys)

    # Ключи, относящиеся к ЭЛЕКТРИЧЕСКОЙ задаче — нужны на мелкой сетке.
    ELECTRIC_KEYS = ("AP_c1", "AP_c2", "AP_a", "AP_b", "AP_d",
                     "D_LONG", "D_TRANS", "FIBER_ANGLE_DEG", "T_MAX")

    # Ключи, относящиеся к МЕХАНИЧЕСКОЙ задаче — нужны на крупной сетке.
    MECHANICAL_KEYS = ("MU", "B_ISO", "MU_F", "B_F", "KAPPA",
                       "FIBER_ANGLE_DEG")

    def to_dict(self) -> dict:
        return {
            "reaction": {f.name: getattr(self.reaction, f.name)
                         for f in fields(self.reaction)},
            "conduction": {f.name: getattr(self.conduction, f.name)
                           for f in fields(self.conduction)},
            "active": {f.name: getattr(self.active, f.name)
                       for f in fields(self.active)},
            "passive": {f.name: getattr(self.passive, f.name)
                        for f in fields(self.passive)},
        }

    @staticmethod
    def from_dict(d: dict) -> "TissueBaseParams":
        return TissueBaseParams(
            reaction=ReactionParams(**d.get("reaction", {})),
            conduction=ConductionParams(**d.get("conduction", {})),
            active=ActiveStressParams(**d.get("active", {})),
            passive=PassiveMechParams(**d.get("passive", {})),
        )


# ═══════════════════════════════════════════════════════════════════════
#  ОБЛАСТИ
# ═══════════════════════════════════════════════════════════════════════

#: Префикс ключей переопределений, относящихся к модели клетки.
CELL_PREFIX = "cell:"


@dataclass(frozen=True)
class RegionSpec:
    """
    База для областей. Сама область умеет две вещи: сказать, какие точки
    ей принадлежат (`contains`), и сериализоваться (`to_dict`).

    `overrides` — плоский словарь переопределений, ключи проверяются по
    `TissueBaseParams.flat_keys()` при создании. Параметры модели клетки
    задаются ключами `cell:<имя>` (например, `cell:ATP_i` для ишемии);
    их имена зависят от модели и проверяются при сборке расчёта — слой
    конфигурации моделей не знает.

    `name` — необязательная метка для логов и карт регионов.
    """

    overrides: Mapping[str, float] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        allowed = set(TissueBaseParams.flat_keys())
        empty_cell = [k for k in self.overrides
                      if k.startswith(CELL_PREFIX) and not k[len(CELL_PREFIX):].strip()]
        if empty_cell:
            raise ValueError(f"область {self.name or type(self).__name__}: "
                             f"пустое имя параметра клетки в {empty_cell}")
        unknown = {k for k in self.overrides if not k.startswith(CELL_PREFIX)} - allowed
        if unknown:
            raise ValueError(
                f"область {self.name or type(self).__name__}: неизвестные "
                f"ключи переопределений {sorted(unknown)}; допустимы "
                f"{sorted(allowed)} и параметры клетки как 'cell:<имя>'")

    @property
    def cell_overrides(self) -> dict[str, float]:
        """Переопределения параметров клетки: {имя без префикса: значение}."""
        return {k[len(CELL_PREFIX):]: float(v) for k, v in self.overrides.items()
                if k.startswith(CELL_PREFIX)}

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def is_serializable(self) -> bool:
        """
        Можно ли записать область в JSON/чекпоинт. Для областей с
        произвольным Python-предикатом — нельзя, и об этом нужно
        предупреждать при сохранении, а не молча терять неоднородность.
        """
        return True

    def to_dict(self) -> dict:
        raise NotImplementedError


@dataclass(frozen=True)
class RectRegion(RegionSpec):
    """Прямоугольник [x0, x1] × [y0, y1], мм."""

    x0: float = 0.0
    x1: float = 0.0
    y0: float = 0.0
    y1: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(
                f"границы прямоугольника перепутаны: "
                f"x[{self.x0}, {self.x1}], y[{self.y0}, {self.y1}]")

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x >= self.x0) & (x <= self.x1) & (y >= self.y0) & (y <= self.y1)

    def to_dict(self) -> dict:
        return {"shape": "rect", "name": self.name,
                "params": {"x0": self.x0, "x1": self.x1,
                           "y0": self.y0, "y1": self.y1},
                "overrides": dict(self.overrides)}


@dataclass(frozen=True)
class CircleRegion(RegionSpec):
    """Круг радиуса r с центром (cx, cy), мм."""

    cx: float = 0.0
    cy: float = 0.0
    r: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.r <= 0:
            raise ValueError(f"радиус должен быть > 0, получено {self.r}")

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.r ** 2

    def to_dict(self) -> dict:
        return {"shape": "circle", "name": self.name,
                "params": {"cx": self.cx, "cy": self.cy, "r": self.r},
                "overrides": dict(self.overrides)}


@dataclass(frozen=True)
class CustomRegion(RegionSpec):
    """
    Область произвольной формы, заданная Python-функцией
    `predicate(x, y) -> bool-массив` (x, y — numpy-массивы координат в мм).

    Такая область НЕ сериализуется: при сохранении конфигурации или
    чекпоинта она будет пропущена с предупреждением. Если неоднородность
    должна переживать рестарт — выражайте её через RectRegion/CircleRegion
    или добавьте новый сериализуемый тип области.
    """

    predicate: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.predicate is None:
            raise ValueError("CustomRegion требует predicate(x, y)")

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(self.predicate(x, y), dtype=bool)

    @property
    def is_serializable(self) -> bool:
        return False

    def to_dict(self) -> dict:
        raise TypeError(
            "CustomRegion не сериализуется — используйте RectRegion/"
            "CircleRegion либо добавьте свой сериализуемый тип области")


# ═══════════════════════════════════════════════════════════════════════
#  ЗАГРУЗКА ОБЛАСТЕЙ ИЗ JSON
# ═══════════════════════════════════════════════════════════════════════

_SHAPE_CLASSES = {"rect": RectRegion, "circle": CircleRegion}


def region_from_dict(d: dict) -> RegionSpec:
    """
    Собрать область из словаря вида
        {"shape": "rect", "params": {...}, "overrides": {...}}
    """
    shape = d.get("shape")
    if shape not in _SHAPE_CLASSES:
        raise ValueError(
            f"неизвестная форма области {shape!r}; "
            f"доступны {sorted(_SHAPE_CLASSES)}")
    cls = _SHAPE_CLASSES[shape]
    params = dict(d.get("params", {}))
    try:
        return cls(overrides=dict(d.get("overrides", {})),
                   name=str(d.get("name", "")), **params)
    except TypeError as e:
        raise ValueError(f"область {shape}: неверные параметры {params} ({e})") from e


def load_regions_json(path) -> list[RegionSpec]:
    """
    Загрузить список областей из JSON-файла.

    Формат — список записей:
        [
          {"shape": "rect", "name": "ischemia",
           "params": {"x0": 6, "x1": 10, "y0": 0, "y1": 10},
           "overrides": {"AP_c1": 0.13, "D_LONG": 0.05, "T_MAX": 20.0}},
          {"shape": "circle", "name": "scar",
           "params": {"cx": 8, "cy": 5, "r": 1.0},
           "overrides": {"AP_c1": 0.0, "T_MAX": 0.0, "MU": 5.0}}
        ]
    """
    import json
    from pathlib import Path

    with open(Path(path), encoding="utf-8") as fh:
        spec = json.load(fh)
    if not isinstance(spec, list):
        raise ValueError(f"{path}: ожидался список областей, получен {type(spec).__name__}")

    regions: list[RegionSpec] = []
    for i, entry in enumerate(spec):
        try:
            regions.append(region_from_dict(entry))
        except ValueError as e:
            raise ValueError(f"{path}, запись {i}: {e}") from e
    return regions
