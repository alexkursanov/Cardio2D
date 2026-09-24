"""
Модели клетки и их реестр.
===========================

DOLFINx здесь не используется: модели работают с numpy-массивами формы
(n_nodes, n_states). Поэтому одиночную клетку можно прогнать и
исследовать где угодно, без расчётного окружения.

Добавление новой модели
------------------------
1. Новый файл в этом каталоге, класс — наследник `CellModel`.
2. Регистрация в `CELL_MODELS` ниже.
3. Тест в `tests/test_cell_models.py`: как минимум покой является точкой
   равновесия, надпороговый стимул вызывает ответ, подпороговый — нет.

Больше ничего править не нужно: солвер получает модель через
`make_cell_model(name)` и обращается к ней только через интерфейс
базового класса.
"""

from __future__ import annotations

from .base import CellModel
from .rogers_mcculloch import RogersMcCullochModel
from .tnnpm import TNNPMIsometricModel, TNNPMModel

__all__ = [
    "CellModel",
    "RogersMcCullochModel",
    "TNNPMModel",
    "TNNPMIsometricModel",
    "CELL_MODELS",
    "make_cell_model",
    "available_models",
]


#: Реестр доступных моделей: имя → класс.
CELL_MODELS: dict[str, type[CellModel]] = {
    RogersMcCullochModel.name: RogersMcCullochModel,
    TNNPMModel.name: TNNPMModel,
    TNNPMIsometricModel.name: TNNPMIsometricModel,
}


def available_models() -> tuple[str, ...]:
    return tuple(sorted(CELL_MODELS))


def make_cell_model(name: str, **kwargs) -> CellModel:
    """
    Создать модель клетки по имени.

    Имена намеренно совпадают с атрибутом `name` класса, чтобы
    конфигурация прогона и сообщения об ошибках говорили на одном языке.
    """
    try:
        cls = CELL_MODELS[name]
    except KeyError:
        raise KeyError(
            f"неизвестная модель клетки {name!r}; "
            f"доступны {list(available_models())}") from None
    return cls(**kwargs)
