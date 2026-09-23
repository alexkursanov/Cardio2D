"""
Построение сеток из спецификаций и работа с координатами DOF.
==============================================================

Этот модуль — единственное место, где спецификация сетки (чистые данные
из `config/`) превращается в объект DOLFINx. Всё, что ниже по слоям,
работает уже с готовыми сетками и не знает, как они были построены.

Ключевой объект — `MeshPair`: пара сеток (электрическая + механическая)
вместе с породившей её конфигурацией и результатом анализа вложенности.
Именно он передаётся дальше в `fem/tissue.py`, солверы и `coupling/`.

О координатах DOF
-----------------
`dof_coordinates(V)` возвращает по одной координате на УЗЕЛ (блок), а не
на компоненту. Для векторного пространства перемещений (block size = 2)
это значит: массив длины `index_map.size_local + num_ghosts`, а не вдвое
больше. Соответственно `V.dofmap.index_map.size_local` тоже считает
БЛОКИ, а не отдельные компоненты — эти две величины согласованы между
собой, и именно на этом держится координатное сопоставление DOF в
чекпоинтах и в переносе полей между сетками.

Владение и гало
---------------
Локальные массивы DOLFINx устроены как [владеемые | гало]. Функции этого
модуля возвращают ПОЛНЫЙ локальный массив (с гало); срезать до
владеемой части — задача вызывающего, и он должен делать это явно через
`index_map.size_local`. Молчаливое срезание здесь приводило бы к тому,
что часть кода считает с гало, часть без, и расхождение всплывает только
в параллельном запуске.
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI

import dolfinx.mesh as dmesh

from ..config.mesh_spec import DualMeshConfig, MeshNesting, RectangleMeshSpec

__all__ = [
    "build_mesh",
    "build_mesh_pair",
    "MeshPair",
    "cell_midpoints",
    "dof_coordinates",
    "n_cells_global",
    "n_vertices_global",
]


# Соответствие строкового имени из спецификации и типа ячейки DOLFINx.
# Спецификация намеренно хранит строку, а не объект DOLFINx: `config/`
# обязан работать в окружении без установленного DOLFINx.
_CELL_TYPES = {
    "quadrilateral": dmesh.CellType.quadrilateral,
    "triangle": dmesh.CellType.triangle,
}


# ═══════════════════════════════════════════════════════════════════════
#  ПОСТРОЕНИЕ СЕТОК
# ═══════════════════════════════════════════════════════════════════════

def build_mesh(spec: RectangleMeshSpec,
               comm: MPI.Comm | None = None) -> dmesh.Mesh:
    """
    Построить сетку по спецификации.

    Параметры
    ---------
    spec : RectangleMeshSpec
        Описание сетки (разрешение, размеры области, тип ячейки).
    comm : MPI.Comm, optional
        Коммуникатор; по умолчанию COMM_WORLD. Передаётся явно, чтобы
        можно было строить сетку на COMM_SELF — это нужно и в тестах, и
        в будущих задачах, где часть работы делается на одном ранге.

    ТОЧКА РАСШИРЕНИЯ: при добавлении `FileMeshSpec` (реальная геометрия
    из .msh/.xdmf) сюда добавляется ветка по типу спецификации, а сам
    выбор остаётся в этой функции — вызывающий код о нём не знает.
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    if not isinstance(spec, RectangleMeshSpec):
        raise TypeError(
            f"build_mesh пока умеет только RectangleMeshSpec, получено "
            f"{type(spec).__name__}")

    try:
        cell_type = _CELL_TYPES[spec.cell_type]
    except KeyError:
        raise ValueError(
            f"тип ячейки {spec.cell_type!r} не поддержан; "
            f"доступны {sorted(_CELL_TYPES)}") from None

    return dmesh.create_rectangle(
        comm,
        [[0.0, 0.0], [spec.lx_mm, spec.ly_mm]],
        [spec.nx, spec.ny],
        cell_type=cell_type,
    )


class MeshPair:
    """
    Пара сеток одной физической области: электрическая (мелкая) и
    механическая (крупная).

    Держит вместе три вещи, которые дальше всегда нужны одновременно:
    сами сетки, породившую их конфигурацию и результат анализа
    вложенности (по нему `coupling/` выбирает оператор переноса).

    Сетки строятся на одном коммуникаторе, но партиционируются
    НЕЗАВИСИМО: ячейки, соседние в физическом пространстве, могут
    оказаться на разных рангах в электрической и механической сетках.
    Поэтому любой перенос данных между ними обязан идти через
    координаты, а не через локальные индексы.
    """

    def __init__(self, config: DualMeshConfig,
                 comm: MPI.Comm | None = None):
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.config = config
        self.electric = build_mesh(config.electric, self.comm)
        self.mechanical = build_mesh(config.mechanical, self.comm)
        self.nesting: MeshNesting = config.nesting()

    # ── проверка, что построенное соответствует заказанному ───────────
    def verify(self) -> None:
        """
        Сверить фактически построенные сетки со спецификацией.

        Нужно потому, что число ячеек зависит от типа элемента (у
        треугольников их вдвое больше, чем у квадов при том же nx×ny), и
        ошибка здесь тихо исказит всё последующее — от числа DOF до
        стоимости шага.
        """
        for name, mesh, spec in (("электрическая", self.electric, self.config.electric),
                                 ("механическая", self.mechanical, self.config.mechanical)):
            got = n_cells_global(mesh)
            if got != spec.n_cells:
                raise RuntimeError(
                    f"{name} сетка: построено {got} ячеек, спецификация "
                    f"обещала {spec.n_cells} ({spec})")

    @property
    def n_cells_electric(self) -> int:
        return n_cells_global(self.electric)

    @property
    def n_cells_mechanical(self) -> int:
        return n_cells_global(self.mechanical)

    def summary(self) -> str:
        lines = [self.config.summary()]
        lines.append(
            f"  Построено    : эл. {self.n_cells_electric} ячеек, "
            f"мех. {self.n_cells_mechanical} ячеек, "
            f"MPI-рангов: {self.comm.size}")
        return "\n".join(lines)


def build_mesh_pair(config: DualMeshConfig,
                    comm: MPI.Comm | None = None) -> MeshPair:
    """Построить пару сеток и сразу проверить её соответствие спецификации."""
    pair = MeshPair(config, comm)
    pair.verify()
    return pair


# ═══════════════════════════════════════════════════════════════════════
#  КООРДИНАТЫ И РАЗМЕРЫ
# ═══════════════════════════════════════════════════════════════════════

def n_cells_global(mesh: dmesh.Mesh) -> int:
    """Полное число ячеек по всем рангам."""
    return mesh.topology.index_map(mesh.topology.dim).size_global


def n_vertices_global(mesh: dmesh.Mesh) -> int:
    """Полное число вершин по всем рангам."""
    mesh.topology.create_entities(0)
    return mesh.topology.index_map(0).size_global


def cell_midpoints(mesh: dmesh.Mesh, include_ghosts: bool = True) -> np.ndarray:
    """
    Середины ячеек, (n_cells, 3). Порядок — по локальным индексам ячеек.

    Для прямоугольных ячеек середина совпадает с геометрическим центром,
    то есть ровно с точкой, в которой «живёт» значение DG0.
    """
    tdim = mesh.topology.dim
    imap = mesh.topology.index_map(tdim)
    n = imap.size_local + (imap.num_ghosts if include_ghosts else 0)
    return dmesh.compute_midpoints(mesh, tdim, np.arange(n, dtype=np.int32))


def dof_coordinates(V) -> np.ndarray:
    """
    Координаты DOF пространства V — по одной на УЗЕЛ (блок), (n_nodes, 3),
    включая гало.

    Для P1 и векторного P1 работает штатный `tabulate_dof_coordinates()`.
    Для DG0 значение живёт в центре ячейки; если `tabulate_dof_coordinates`
    на конкретной сборке DOLFINx для DG0 недоступен, координаты собираются
    из середин ячеек через dofmap — результат тот же, но не зависит от
    деталей реализации элемента.
    """
    try:
        coords = V.tabulate_dof_coordinates()
        if coords.size:
            return coords
    except Exception:  # noqa: BLE001 — сборки DOLFINx различаются
        pass

    return _dg0_coordinates_via_dofmap(V)


def _dg0_coordinates_via_dofmap(V) -> np.ndarray:
    """
    Запасной путь: разложить середины ячеек по номерам DOF.

    Работает для пространств с ОДНИМ узлом на ячейку (DG0). Для
    остальных случаев честно падает, а не возвращает правдоподобный
    мусор.
    """
    mesh = V.mesh
    tdim = mesh.topology.dim
    imap_c = mesh.topology.index_map(tdim)
    n_cells = imap_c.size_local + imap_c.num_ghosts

    dofmap = V.dofmap
    cell_dofs = np.asarray(dofmap.list)
    if cell_dofs.ndim == 1:
        cell_dofs = cell_dofs.reshape(n_cells, -1)
    if cell_dofs.shape[1] != 1:
        raise RuntimeError(
            f"не удалось получить координаты DOF: tabulate_dof_coordinates "
            f"недоступен, а запасной путь работает только для пространств с "
            f"одним узлом на ячейку (получено {cell_dofs.shape[1]})")

    mids = cell_midpoints(mesh)
    n_nodes = dofmap.index_map.size_local + dofmap.index_map.num_ghosts
    coords = np.zeros((n_nodes, 3), dtype=np.float64)
    coords[cell_dofs[:, 0]] = mids[:n_cells]
    return coords


def owned_dof_coordinates(V) -> np.ndarray:
    """
    Координаты только ВЛАДЕЕМЫХ этим рангом узлов, (n_owned, 2).

    Именно эта форма нужна для координатного сопоставления DOF — в
    чекпоинтах и при переносе полей между сетками: гало дублируют узлы
    соседних рангов, и без срезания те попали бы в обмен дважды.
    """
    n_owned = V.dofmap.index_map.size_local
    return np.ascontiguousarray(dof_coordinates(V)[:n_owned, :2])
