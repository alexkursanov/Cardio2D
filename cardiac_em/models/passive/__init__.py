"""
Модели пассивной упругости и их реестр.
========================================

В отличие от `models/cell/`, этот слой требует DOLFINx: материалы
возвращают UFL-выражения, которые подставляются в вариационную форму.

Добавление нового материала
----------------------------
1. Новый файл в этом каталоге, класс — наследник `PassiveMaterial` с
   методом `second_piola(u, tissue)`.
2. Регистрация в `PASSIVE_MATERIALS` ниже.
3. Тест: при нулевой деформации напряжение должно быть нулевым, при
   растяжении вдоль волокна — больше, чем поперёк.

Солвер механики обращается к материалу только через базовый интерфейс,
поэтому больше ничего править не нужно.
"""

from __future__ import annotations

from .base import PassiveMaterial, kinematics
from .transversely_isotropic import TransverselyIsotropicExponential

__all__ = [
    "PassiveMaterial",
    "kinematics",
    "TransverselyIsotropicExponential",
    "PASSIVE_MATERIALS",
    "make_passive_material",
    "available_materials",
]


#: Реестр доступных материалов: имя → класс.
PASSIVE_MATERIALS: dict[str, type[PassiveMaterial]] = {
    TransverselyIsotropicExponential.name: TransverselyIsotropicExponential,
}


def available_materials() -> tuple[str, ...]:
    return tuple(sorted(PASSIVE_MATERIALS))


def make_passive_material(name: str, **kwargs) -> PassiveMaterial:
    try:
        cls = PASSIVE_MATERIALS[name]
    except KeyError:
        raise KeyError(
            f"неизвестный материал {name!r}; "
            f"доступны {list(available_materials())}") from None
    return cls(**kwargs)
