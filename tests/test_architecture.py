"""
Тесты архитектурных инвариантов.
=================================

DOLFINx не требуется — анализируется исходный текст, а не выполнение.

    python tests/test_architecture.py
    pytest tests/test_architecture.py -v

Зачем это нужно. Правило направления зависимостей (см. ARCHITECTURE.md)
держится ровно до первого «сейчас быстро поправлю»: кто-нибудь
импортирует `dolfinx` в `config/`, чтобы не дублировать константу, и
через месяц выясняется, что конфигурации серий больше нельзя готовить
на ноутбуке, а `analysis/` тянет за собой весь FEM-стек. Такие вещи
дешевле ловить тестом, чем ревью.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "cardiac_em"

# Тяжёлые зависимости расчётной части: их не должно быть в слоях,
# которые обязаны работать в лёгком окружении.
HEAVY = ("dolfinx", "petsc4py", "mpi4py", "ufl", "basix")

# Порядок слоёв: модуль может импортировать только то, что СТРОГО ЛЕВЕЕ
# него в этом списке (плюс самого себя). `analysis` намеренно отсутствует
# — он не должен зависеть ни от одного расчётного слоя.
LAYERS = ["config", "fem", "models", "solvers", "coupling", "runtime", "io", "control"]


def _iter_modules(subpkg: str):
    """Все .py файлы подпакета (если он уже существует)."""
    root = PKG / subpkg
    if not root.exists():
        return
    for path in sorted(root.rglob("*.py")):
        yield path


def _imported_names(path: Path) -> list[str]:
    """Полные имена всех импортов в файле, включая относительные."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # относительный импорт: ..config.mesh_spec
                names.append("." * node.level + (node.module or ""))
            elif node.module:
                names.append(node.module)
    return names


def _top_level(name: str) -> str:
    return name.split(".", 1)[0]


# ═══════════════════════════════════════════════════════════════════════
#  ЛЁГКИЕ СЛОИ НЕ ТЯНУТ FEM-СТЕК
# ═══════════════════════════════════════════════════════════════════════

def test_config_does_not_import_fem_stack():
    """
    `config/` обязан работать без DOLFINx — на этом держится подготовка
    конфигураций отдельно от расчётной машины.
    """
    violations = []
    for path in _iter_modules("config"):
        for name in _imported_names(path):
            if _top_level(name) in HEAVY:
                violations.append(f"{path.name}: import {name}")
    assert not violations, (
        "config/ не должен зависеть от FEM-стека:\n  " + "\n  ".join(violations))


def test_analysis_does_not_import_fem_stack():
    """
    `analysis/` читает файлы результатов и обязан запускаться в лёгком
    окружении (ноутбук, другая машина, через месяцы после прогона).
    """
    violations = []
    for path in _iter_modules("analysis"):
        for name in _imported_names(path):
            if _top_level(name) in HEAVY:
                violations.append(f"{path.name}: import {name}")
    assert not violations, (
        "analysis/ не должен зависеть от FEM-стека:\n  " + "\n  ".join(violations))


def test_analysis_does_not_import_solver_layers():
    """Постобработка работает с файлами, а не с живыми объектами солвера."""
    forbidden = {"fem", "models", "solvers", "coupling", "runtime", "io", "control"}
    violations = []
    for path in _iter_modules("analysis"):
        for name in _imported_names(path):
            if name.startswith(".."):
                # относительный импорт из соседнего подпакета: ..runtime.x
                target = name.lstrip(".").split(".")[0]
                if target in forbidden:
                    violations.append(f"{path.name}: from {name} import …")
                continue
            if name.startswith("."):
                continue  # внутри самого analysis/
            parts = name.split(".")
            if parts[0] == "cardiac_em" and len(parts) > 1 and parts[1] in forbidden:
                violations.append(f"{path.name}: import {name}")
    assert not violations, (
        "analysis/ не должен импортировать расчётные слои:\n  "
        + "\n  ".join(violations))


# ═══════════════════════════════════════════════════════════════════════
#  НАПРАВЛЕНИЕ ЗАВИСИМОСТЕЙ
# ═══════════════════════════════════════════════════════════════════════

def test_layer_dependency_direction():
    """
    Модуль слоя N может импортировать только слои < N. Обратных и
    «боковых вверх» импортов быть не должно — иначе слои перестают
    быть слоями.
    """
    index = {name: i for i, name in enumerate(LAYERS)}
    violations = []

    for layer in LAYERS:
        for path in _iter_modules(layer):
            for name in _imported_names(path):
                target = None

                if name.startswith("."):
                    # относительный: ..config.mesh_spec → config
                    rest = name.lstrip(".")
                    if rest:
                        target = rest.split(".")[0]
                elif name.startswith("cardiac_em."):
                    parts = name.split(".")
                    if len(parts) > 1:
                        target = parts[1]

                if target is None or target not in index:
                    continue
                if target == layer:
                    continue

                if index[target] > index[layer]:
                    violations.append(
                        f"{layer}/{path.name}: импортирует {target} "
                        f"(слой ниже по графу)")

    assert not violations, (
        "нарушено направление зависимостей (см. ARCHITECTURE.md §1.1):\n  "
        + "\n  ".join(violations))


def test_package_init_does_not_import_subpackages():
    """
    Корневой `cardiac_em/__init__.py` не должен импортировать подпакеты:
    иначе `import cardiac_em.config` в окружении без DOLFINx утащит
    расчётные слои и упадёт.
    """
    init = PKG / "__init__.py"
    names = _imported_names(init)
    subpkg_imports = [n for n in names
                      if n.startswith(".") or n.startswith("cardiac_em.")]
    assert not subpkg_imports, (
        "cardiac_em/__init__.py должен оставаться пустым по импортам, "
        f"найдено: {subpkg_imports}")


# ═══════════════════════════════════════════════════════════════════════
#  ЕДИНИЦЫ ИЗМЕРЕНИЯ В ИМЕНАХ
# ═══════════════════════════════════════════════════════════════════════

def test_config_time_and_length_fields_carry_units():
    """
    Поля конфигурации, несущие размерность, должны иметь суффикс единицы
    (`_mm`, `_ms`, `_deg`) — чтобы не приходилось сверяться с
    комментарием. Проверяем на dataclass'ах слоя config.
    """
    sys.path.insert(0, str(PKG.parent))
    from dataclasses import fields as dc_fields

    from cardiac_em.config import (
        PreloadProtocol,
        RectangleMeshSpec,
        StimulusProtocol,
        TimeStepping,
    )

    # (класс, поля, которые обязаны нести единицу)
    expectations = [
        (RectangleMeshSpec, {"lx_mm", "ly_mm"}),
        (StimulusProtocol, {"times_ms", "duration_ms", "rise_time_ms",
                            "decay_length_mm", "x_max_mm"}),
        (TimeStepping, {"dt_electric_ms", "t_end_ms"}),
    ]

    for cls, expected in expectations:
        present = {f.name for f in dc_fields(cls)}
        missing = expected - present
        assert not missing, f"{cls.__name__}: ожидались поля {missing}"

    # безразмерные поля единицу не несут — проверяем, что не переусердствовали
    assert {"stretch", "n_steps"} <= {f.name for f in dc_fields(PreloadProtocol)}


# ═══════════════════════════════════════════════════════════════════════
#  ЗАПУСК БЕЗ PYTEST
# ═══════════════════════════════════════════════════════════════════════

def _main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:  # noqa: BLE001
            failed.append((name, e))
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} тестов прошло")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
