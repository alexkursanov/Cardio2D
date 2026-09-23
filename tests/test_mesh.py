"""
Тесты построения сеток (Шаг 2).
================================

ТРЕБУЮТ DOLFINx — помечены `@pytest.mark.fem` и автоматически
пропускаются там, где его нет.

    pytest tests/test_mesh.py -v
    mpirun -n 4 python -m pytest tests/test_mesh.py -v     # параллельно

Что здесь проверяется и зачем:

  * построенная сетка соответствует спецификации (число ячеек зависит от
    типа элемента — у треугольников их вдвое больше при том же nx×ny, и
    ошибка тут тихо исказила бы всё последующее);

  * `dof_coordinates` для DG0 возвращает СЕРЕДИНЫ ЯЧЕЕК, а не вершины —
    на этом держится и координатное сопоставление в чекпоинтах, и
    перенос полей между сетками. Тест проверяет это напрямую, поэтому
    если поведение `tabulate_dof_coordinates` на вашей сборке DOLFINx
    отличается от ожидаемого, вы увидите это здесь, а не через три шага;

  * для ВЕКТОРНОГО пространства возвращается одна координата на УЗЕЛ, а
    не на компоненту — классический источник ошибок вдвое;

  * во вложенной паре каждая механическая ячейка содержит ровно kx×ky
    электрических. Это предпосылка точного осреднения T_act на шаге 4:
    если она не выполняется, «точный» перенос точным не будет.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.config import (  # noqa: E402
    DualMeshConfig,
    RectangleMeshSpec,
)

pytestmark = pytest.mark.fem

# Импорты DOLFINx делаются внутри тестов/фикстур, чтобы сам файл
# импортировался и в окружении без DOLFINx (там тесты просто пропустятся).


def _require_mpi(min_ranks: int = 2) -> None:
    """
    Предусловие «запущено под mpirun».

    Проверяется ВНУТРИ теста, а не только меткой `@pytest.mark.mpi`:
    автопропуск по метке работает лишь если подхватился conftest.py, и
    тест, чья корректность от этого зависит, — хрупкий. Метка остаётся,
    но нужна она для выборки (`pytest -m mpi`), а не для правильности.
    """
    from mpi4py import MPI

    size = MPI.COMM_WORLD.size
    if size < min_ranks:
        pytest.skip(
            f"нужен запуск под mpirun с числом рангов >= {min_ranks} "
            f"(сейчас: {size})")


# ═══════════════════════════════════════════════════════════════════════
#  ПОСТРОЕНИЕ
# ═══════════════════════════════════════════════════════════════════════

def test_quadrilateral_mesh_matches_spec():
    from cardiac_em.fem import build_mesh, n_cells_global, n_vertices_global

    spec = RectangleMeshSpec(nx=8, ny=6, lx_mm=10.0, ly_mm=5.0)
    mesh = build_mesh(spec)

    assert n_cells_global(mesh) == spec.n_cells == 48
    assert n_vertices_global(mesh) == spec.n_vertices == 9 * 7
    assert mesh.topology.dim == 2


def test_triangle_mesh_has_twice_the_cells():
    from cardiac_em.fem import build_mesh, n_cells_global, n_vertices_global

    spec = RectangleMeshSpec(nx=8, ny=6, cell_type="triangle")
    mesh = build_mesh(spec)

    assert n_cells_global(mesh) == spec.n_cells == 96
    # вершины те же, что у квадратной сетки того же разрешения
    assert n_vertices_global(mesh) == spec.n_vertices == 9 * 7


def test_mesh_covers_requested_domain():
    from mpi4py import MPI

    from cardiac_em.fem import build_mesh

    spec = RectangleMeshSpec(nx=10, ny=4, lx_mm=12.0, ly_mm=3.0)
    mesh = build_mesh(spec)

    x = mesh.geometry.x
    comm = mesh.comm
    # границы считаем глобально: в параллели у ранга лишь часть области
    xmin = comm.allreduce(x[:, 0].min(), op=MPI.MIN)
    xmax = comm.allreduce(x[:, 0].max(), op=MPI.MAX)
    ymin = comm.allreduce(x[:, 1].min(), op=MPI.MIN)
    ymax = comm.allreduce(x[:, 1].max(), op=MPI.MAX)

    assert math.isclose(xmin, 0.0, abs_tol=1e-12)
    assert math.isclose(xmax, spec.lx_mm, rel_tol=1e-12)
    assert math.isclose(ymin, 0.0, abs_tol=1e-12)
    assert math.isclose(ymax, spec.ly_mm, rel_tol=1e-12)


def test_unsupported_spec_type_rejected():
    from cardiac_em.fem import build_mesh

    with pytest.raises(TypeError):
        build_mesh("не спецификация")


# ═══════════════════════════════════════════════════════════════════════
#  КООРДИНАТЫ DOF
# ═══════════════════════════════════════════════════════════════════════

def test_dg0_coordinates_are_cell_midpoints():
    """
    Главная проверка модуля: значение DG0 должно жить в центре ячейки.

    Сверяем поэлементно с `compute_midpoints`, проходя через dofmap —
    то есть проверяем не только сами координаты, но и то, что они
    разложены по номерам DOF в правильном порядке.
    """
    import dolfinx.fem as fem

    from cardiac_em.fem import build_mesh, cell_midpoints, dof_coordinates

    mesh = build_mesh(RectangleMeshSpec(nx=6, ny=4, lx_mm=6.0, ly_mm=4.0))
    DG0 = fem.functionspace(mesh, ("DG", 0))

    coords = dof_coordinates(DG0)
    mids = cell_midpoints(mesh)

    tdim = mesh.topology.dim
    imap = mesh.topology.index_map(tdim)
    n_cells = imap.size_local + imap.num_ghosts

    for cell in range(n_cells):
        dof = DG0.dofmap.cell_dofs(cell)
        assert len(dof) == 1, "у DG0 должен быть ровно один DOF на ячейку"
        np.testing.assert_allclose(coords[dof[0], :2], mids[cell, :2], atol=1e-12)


def test_dg0_coordinates_are_strictly_inside_domain():
    """
    Дешёвая проверка «это середины, а не вершины»: середина ячейки
    никогда не лежит на границе области.
    """
    import dolfinx.fem as fem

    from cardiac_em.fem import build_mesh, dof_coordinates

    spec = RectangleMeshSpec(nx=8, ny=8, lx_mm=10.0, ly_mm=10.0)
    mesh = build_mesh(spec)
    DG0 = fem.functionspace(mesh, ("DG", 0))

    n_owned = DG0.dofmap.index_map.size_local
    xy = dof_coordinates(DG0)[:n_owned, :2]

    half_x, half_y = spec.hx_mm / 2, spec.hy_mm / 2
    assert np.all(xy[:, 0] > half_x - 1e-9)
    assert np.all(xy[:, 0] < spec.lx_mm - half_x + 1e-9)
    assert np.all(xy[:, 1] > half_y - 1e-9)
    assert np.all(xy[:, 1] < spec.ly_mm - half_y + 1e-9)


def test_p1_dof_count_equals_vertex_count():
    import dolfinx.fem as fem

    from cardiac_em.fem import build_mesh, n_vertices_global

    spec = RectangleMeshSpec(nx=5, ny=3)
    mesh = build_mesh(spec)
    P1 = fem.functionspace(mesh, ("Lagrange", 1))

    assert P1.dofmap.index_map.size_global == n_vertices_global(mesh)
    assert P1.dofmap.index_map.size_global == spec.n_vertices


def test_vector_space_gives_one_coordinate_per_node():
    """
    Для пространства с block size = 2 координат должно быть столько же,
    сколько УЗЛОВ, а не компонент. Ошибка вдвое здесь ломает и чекпоинты,
    и перенос полей, причём незаметно — массивы «почти» подходят по длине.
    """
    import dolfinx.fem as fem

    from cardiac_em.fem import build_mesh, dof_coordinates

    mesh = build_mesh(RectangleMeshSpec(nx=4, ny=4))
    V = fem.functionspace(mesh, ("Lagrange", 1, (2,)))

    imap = V.dofmap.index_map
    n_nodes = imap.size_local + imap.num_ghosts

    coords = dof_coordinates(V)
    assert coords.shape[0] == n_nodes
    # а сам массив значений функции — вдвое длиннее
    assert V.dofmap.index_map_bs == 2
    u = fem.Function(V)
    assert u.x.array.size == n_nodes * 2


def test_owned_coordinates_exclude_ghosts():
    import dolfinx.fem as fem
    from mpi4py import MPI

    from cardiac_em.fem import build_mesh, owned_dof_coordinates

    mesh = build_mesh(RectangleMeshSpec(nx=8, ny=8))
    DG0 = fem.functionspace(mesh, ("DG", 0))

    owned = owned_dof_coordinates(DG0)
    assert owned.shape == (DG0.dofmap.index_map.size_local, 2)

    # сумма владеемых по всем рангам = глобальное число ячеек,
    # то есть гало не посчитаны дважды
    total = mesh.comm.allreduce(owned.shape[0], op=MPI.SUM)
    assert total == DG0.dofmap.index_map.size_global


# ═══════════════════════════════════════════════════════════════════════
#  ПАРА СЕТОК
# ═══════════════════════════════════════════════════════════════════════

def test_mesh_pair_builds_and_verifies():
    from cardiac_em.fem import build_mesh_pair

    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=16, ny=16, lx_mm=10.0, ly_mm=10.0), coarsening=4)
    pair = build_mesh_pair(cfg)

    assert pair.n_cells_electric == 16 * 16
    assert pair.n_cells_mechanical == 4 * 4
    assert pair.nesting.is_nested
    assert (pair.nesting.kx, pair.nesting.ky) == (4, 4)
    assert pair.nesting.cells_per_mech_cell == 16


def test_mesh_pair_summary_renders():
    from cardiac_em.fem import build_mesh_pair

    pair = build_mesh_pair(DualMeshConfig.nested(
        RectangleMeshSpec(nx=8, ny=8), coarsening=2))
    text = pair.summary()
    assert "Построено" in text and "Вложенность" in text


def test_non_nested_pair_still_builds():
    """Невложенная пара законна — просто перенос будет приближённым."""
    from cardiac_em.fem import build_mesh_pair

    cfg = DualMeshConfig(
        electric=RectangleMeshSpec(nx=12, ny=12),
        mechanical=RectangleMeshSpec(nx=5, ny=5))
    pair = build_mesh_pair(cfg)

    assert pair.n_cells_electric == 144
    assert pair.n_cells_mechanical == 25
    assert not pair.nesting.is_nested


def test_nested_cells_contain_exactly_k_by_k_electric_cells():
    """
    Предпосылка точного переноса T_act (шаг 4): каждая механическая
    ячейка должна содержать ровно kx×ky электрических.

    Сетки партиционируются независимо, поэтому проверка делается на
    ПОЛНЫХ наборах середин, собранных со всех рангов.
    """
    import dolfinx.fem as fem

    from cardiac_em.fem import build_mesh_pair, owned_dof_coordinates

    kx = ky = 3
    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=9, ny=9, lx_mm=9.0, ly_mm=9.0), coarsening=kx)
    pair = build_mesh_pair(cfg)

    DG0_e = fem.functionspace(pair.electric, ("DG", 0))
    DG0_m = fem.functionspace(pair.mechanical, ("DG", 0))

    comm = pair.comm
    e_all = np.vstack(comm.allgather(owned_dof_coordinates(DG0_e)))
    m_all = np.vstack(comm.allgather(owned_dof_coordinates(DG0_m)))

    assert e_all.shape[0] == 81
    assert m_all.shape[0] == 9

    hx_m = cfg.mechanical.hx_mm
    hy_m = cfg.mechanical.hy_mm
    tol = 1e-9

    for cx, cy in m_all:
        inside = (
            (e_all[:, 0] > cx - hx_m / 2 - tol) & (e_all[:, 0] < cx + hx_m / 2 + tol) &
            (e_all[:, 1] > cy - hy_m / 2 - tol) & (e_all[:, 1] < cy + hy_m / 2 + tol)
        )
        assert inside.sum() == kx * ky, (
            f"механическая ячейка с центром ({cx:.3f}, {cy:.3f}) содержит "
            f"{inside.sum()} электрических вместо {kx * ky}")

    # и каждая электрическая ячейка попала ровно в одну механическую
    counted = 0
    for cx, cy in m_all:
        counted += int((
            (e_all[:, 0] > cx - hx_m / 2 - tol) & (e_all[:, 0] < cx + hx_m / 2 + tol) &
            (e_all[:, 1] > cy - hy_m / 2 - tol) & (e_all[:, 1] < cy + hy_m / 2 + tol)
        ).sum())
    assert counted == e_all.shape[0]


def test_mesh_pair_uses_given_communicator():
    """Сетку должно быть можно построить на COMM_SELF — это нужно в тестах."""
    from mpi4py import MPI

    from cardiac_em.fem import build_mesh_pair

    pair = build_mesh_pair(
        DualMeshConfig.nested(RectangleMeshSpec(nx=4, ny=4), coarsening=2),
        comm=MPI.COMM_SELF)

    # на COMM_SELF каждый ранг держит всю сетку целиком
    assert pair.n_cells_electric == 16
    assert pair.n_cells_mechanical == 4


@pytest.mark.mpi
def test_partitioning_splits_cells_across_ranks():
    """
    Под mpirun сетка действительно распределена, а не продублирована.

    При одном ранге утверждение бессмысленно (там ранг ОБЯЗАН держать
    всю сетку — это не ошибка), поэтому тест сам объявляет предусловие.
    """
    _require_mpi(2)

    from mpi4py import MPI

    from cardiac_em.fem import build_mesh, n_cells_global

    mesh = build_mesh(RectangleMeshSpec(nx=16, ny=16))
    tdim = mesh.topology.dim
    local = mesh.topology.index_map(tdim).size_local

    assert local < n_cells_global(mesh), "ранг держит всю сетку — партиционирования нет"
    assert mesh.comm.allreduce(local, op=MPI.SUM) == n_cells_global(mesh)


def test_owned_cells_sum_to_global_on_any_rank_count():
    """
    Инвариант, осмысленный при ЛЮБОМ числе рангов, в отличие от теста
    выше: сумма владеемых ячеек по рангам равна глобальному числу.
    При одном ранге это вырождается в тождество, но при восьми ловит
    двойной учёт гало.
    """
    from mpi4py import MPI

    from cardiac_em.fem import build_mesh, n_cells_global

    mesh = build_mesh(RectangleMeshSpec(nx=16, ny=16))
    tdim = mesh.topology.dim
    local = mesh.topology.index_map(tdim).size_local

    assert mesh.comm.allreduce(local, op=MPI.SUM) == n_cells_global(mesh)
