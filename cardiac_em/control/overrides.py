"""
Переопределение параметров конфигурации по пути.
=================================================

    apply_overrides(config, {"tissue_base.active.t_max": 60,
                             "stimulus.times_ms": [0, 500, 1000]})

Путь — ключи словаря `SimulationConfig.to_dict()` через точку; элементы
списков — по номеру (`regions.0.overrides.T_MAX`). Изменение делается на
словаре, затем конфигурация собирается заново через `from_dict`, так что
все проверки групп параметров срабатывают как при ручном создании.

Опечатка в пути — ОШИБКА, а не новый ключ, который никто не прочитает
(та же логика, что с ключами регионов). Единственное исключение — словарь
`overrides` региона: туда можно добавить новый параметр, его имя
проверит сам регион.

Модуль не требует DOLFINx.
"""

from __future__ import annotations

import copy
import json

from ..config.simulation import SimulationConfig

__all__ = ["apply_overrides", "parse_assignment", "parse_value", "set_path"]

_OPEN_DICTS = ("overrides",)   # словари, куда разрешено добавлять ключи


def parse_value(text: str):
    """
    Значение из командной строки: JSON, если разбирается, иначе строка.

        "60" → 60, "1.5" → 1.5, "[0, 500]" → [0, 500], "true" → True,
        "null" → None, "runs/a" → "runs/a"
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def parse_assignment(text: str) -> tuple[str, object]:
    """'путь=значение' → (путь, значение)."""
    if "=" not in text:
        raise ValueError(f"ожидалось путь=значение, получено {text!r}")
    path, value = text.split("=", 1)
    path = path.strip()
    if not path:
        raise ValueError(f"пустой путь в {text!r}")
    return path, parse_value(value.strip())


def set_path(tree: dict, path: str, value) -> None:
    """Записать значение по пути в словарь (на месте)."""
    parts = path.split(".")
    node = tree
    walked = []
    for i, key in enumerate(parts):
        last = i == len(parts) - 1
        here = ".".join(walked) or "<корень>"
        if isinstance(node, list):
            try:
                idx = int(key)
            except ValueError:
                raise KeyError(f"{here} — список, нужен номер элемента, "
                               f"получено {key!r}") from None
            if not -len(node) <= idx < len(node):
                raise KeyError(f"{here}: нет элемента {idx} "
                               f"(всего {len(node)})")
            if last:
                node[idx] = value
                return
            node = node[idx]
        elif isinstance(node, dict):
            if key not in node:
                open_dict = walked and walked[-1] in _OPEN_DICTS
                if not (last and open_dict):
                    raise KeyError(
                        f"нет параметра {'.'.join(walked + [key])!r}; "
                        f"в {here} есть: {sorted(node)}")
            if last:
                node[key] = value
                return
            node = node[key]
        else:
            raise KeyError(f"{here} — значение, а не группа параметров; "
                           f"путь {path!r} слишком длинный")
        walked.append(key)


def apply_overrides(config: SimulationConfig, overrides: dict) -> SimulationConfig:
    """
    Новая конфигурация с изменёнными параметрами. Исходная не меняется.

    CustomRegion (Python-предикаты) сохраняются: они не сериализуются, но
    при изменении параметров копируются в новую конфигурацию как есть.
    """
    if not overrides:
        return copy.deepcopy(config)
    tree = config.to_dict(skip_unserializable=True)
    for path, value in overrides.items():
        set_path(tree, path, value)
    new = SimulationConfig.from_dict(tree)
    custom = config.unserializable_regions()
    if custom:
        new.regions = list(new.regions) + list(custom)
    return new
