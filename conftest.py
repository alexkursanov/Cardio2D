"""
Общая настройка тестов.
========================

ЛЕЖИТ В КОРНЕ РЕПОЗИТОРИЯ намеренно. Хук `pytest_report_header`
срабатывает только для conftest.py в rootdir; из `tests/conftest.py` он
молча не вызывается, и тогда пропадает единственный видимый признак
того, что настройка вообще подхватилась.

Задача файла — сделать так, чтобы `pytest` запускался в ЛЮБОМ окружении:

  * где DOLFINx есть (расчётный сервер) — идут все тесты;
  * где его нет (ноутбук, окружение постобработки) — тесты с меткой
    `@pytest.mark.fem` пропускаются, а не падают с ImportError.

То же для `@pytest.mark.mpi`: такие тесты осмысленны только под mpirun
с числом рангов больше одного.

ВАЖНО: автопропуск здесь — удобство, а не гарантия. Тест, которому
нужны особые условия, обязан проверять их сам (см. `_require_mpi` в
tests/test_mesh.py). Если корректность теста зависит от того,
подхватился ли conftest, — это хрупкий тест.
"""

from __future__ import annotations

import pytest


def _dolfinx_available() -> bool:
    try:
        import dolfinx  # noqa: F401
    except Exception:  # noqa: BLE001 — ловим и ImportError, и ошибки сборки
        return False
    return True


def _mpi_size() -> int:
    try:
        from mpi4py import MPI
    except Exception:  # noqa: BLE001
        return 1
    return MPI.COMM_WORLD.size


HAVE_DOLFINX = _dolfinx_available()
MPI_SIZE = _mpi_size()


def pytest_configure(config: pytest.Config) -> None:
    # Маркеры продублированы в pyproject.toml; здесь — на случай запуска
    # pytest из каталога без конфигурации.
    config.addinivalue_line("markers", "fem: требует установленного DOLFINx")
    config.addinivalue_line("markers", "mpi: осмысленен только под mpirun")
    config.addinivalue_line("markers", "slow: длинный тест")


def pytest_collection_modifyitems(config: pytest.Config,
                                  items: list[pytest.Item]) -> None:
    skip_fem = pytest.mark.skip(reason="DOLFINx недоступен в этом окружении")
    skip_mpi = pytest.mark.skip(
        reason=f"нужен запуск под mpirun (рангов сейчас: {MPI_SIZE})")

    for item in items:
        if "fem" in item.keywords and not HAVE_DOLFINX:
            item.add_marker(skip_fem)
        if "mpi" in item.keywords and MPI_SIZE < 2:
            item.add_marker(skip_mpi)


def pytest_report_header(config: pytest.Config) -> list[str]:
    """
    Печатается в шапке отчёта. Отсутствие этой строки — признак того,
    что conftest.py не подхватился; тогда автопропуск тоже не работает.
    """
    if HAVE_DOLFINX:
        import dolfinx
        fem_line = f"DOLFINx {dolfinx.__version__}"
    else:
        fem_line = "DOLFINx отсутствует (тесты с меткой fem пропускаются)"
    return [f"окружение: {fem_line}, MPI-рангов: {MPI_SIZE}"]
