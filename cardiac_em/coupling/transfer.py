"""
Перенос полей между электрической и механической сетками.
===========================================================

Задача: значение, посчитанное на одной сетке, подать в форму на другой.
Сейчас единственный реальный канал — активное напряжение T_act,
считаемое на мелкой электрической сетке и нужное механике на крупной.

Обе сетки статичны в референсной конфигурации, поэтому отображение
между ними строится ОДИН РАЗ при старте и переиспользуется каждый
механический шаг.

Две стратегии
-------------
`AveragingTransfer` — осреднение по ячейке-приёмнику. Работает только
для ВЛОЖЕННЫХ сеток (каждая крупная ячейка содержит целое число
мелких) и в этом случае ТОЧЕН: среднее по ячейке — это в точности
интеграл поля по ней, делённый на площадь. Для линейного поля результат
совпадает со значением в центре крупной ячейки до машинной точности.

`NearestCellTransfer` — значение ближайшей по центру ячейки. Работает
для любых сеток, но теряет информацию: крупная ячейка «видит» одну
мелкую вместо среднего по всем. В момент прохождения фронта, когда
T_act внутри крупной ячейки меняется от нуля до пика, это даёт
реальную ошибку в суммарной силе, а не косметическую (см. тесты
`test_averaging_is_exact_on_linear_field` и
`test_nearest_is_worse_than_averaging_on_a_front`).

`make_transfer()` выбирает точную стратегию, когда сетки это позволяют,
и честно сообщает, когда приходится довольствоваться приближением.

О масштабируемости
------------------
Обе стратегии собирают поле-источник целиком на каждый ранг
(`comm.allgather`) — на каждый механический шаг. При нынешних размерах
задачи (тысячи ячеек, шаг механики раз в 20 электрических) это стоит
пренебрежимо мало. Для действительно крупных параллельных расчётов
нужен обмен только с теми рангами, которые реально владеют нужными
ячейками, либо родная межсеточная интерполяция DOLFINx. Место для
замены — метод `apply()`; интерфейс от этого не меняется.

ТОЧКА РАСШИРЕНИЯ: цель переноса сейчас — ячейки DG0 приёмника. Более
точный вариант для невложенных сеток — интерполяция прямо в
квадратурные точки механической формы. Это меняет только реализацию
стратегии: сигнатура `apply()` остаётся прежней.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from mpi4py import MPI

from ..config.mesh_spec import RectangleMeshSpec
from ..fem.mesh import owned_dof_coordinates

__all__ = [
    "FieldTransfer",
    "AveragingTransfer",
    "NearestCellTransfer",
    "make_transfer",
]


class FieldTransfer(ABC):
    """
    Оператор переноса значений DG0 с сетки-источника на сетку-приёмник.

    Строится один раз; `apply()` вызывается каждый механический шаг.

    Соглашение о массивах: `apply()` принимает значения на ВЛАДЕЕМЫХ
    ячейках источника (ровно `index_map.size_local`, без гало) и
    возвращает значения на владеемых ячейках приёмника. Гало на обоих
    концах — забота вызывающего: после присваивания в Function нужен
    `scatter_forward()`.
    """

    #: человекочитаемое имя стратегии для сводок и сообщений
    kind: str = "abstract"

    #: точен ли перенос (в смысле сохранения среднего по ячейке)
    exact: bool = False

    def __init__(self, DG0_src, DG0_dst):
        self.DG0_src = DG0_src
        self.DG0_dst = DG0_dst
        self.n_src_owned = DG0_src.dofmap.index_map.size_local
        self.n_dst_owned = DG0_dst.dofmap.index_map.size_local
        self.comm = DG0_src.mesh.comm

        if DG0_dst.mesh.comm.size != self.comm.size:
            raise ValueError(
                "сетки построены на разных коммуникаторах — перенос между "
                "ними не определён")

        # Размеры владеемых кусков фиксируются при построении: порядок
        # allgather детерминирован, и на нём держится соответствие
        # индексов между построением и каждым применением.
        self._src_sizes = self.comm.allgather(self.n_src_owned)

    # ── обмен ─────────────────────────────────────────────────────────
    def _gather_source(self, values_owned_src: np.ndarray) -> np.ndarray:
        """Собрать значения источника со всех рангов в фиксированном порядке."""
        expected = self._src_sizes[self.comm.rank]
        v = np.ascontiguousarray(values_owned_src[:expected], dtype=np.float64)
        if v.size != expected:
            raise ValueError(
                f"ожидалось {expected} значений на владеемых ячейках "
                f"источника, получено {values_owned_src.size}")
        return np.concatenate(self.comm.allgather(v))

    @abstractmethod
    def apply(self, values_owned_src: np.ndarray) -> np.ndarray:
        """Перенести значения на сетку-приёмник."""

    def describe(self) -> str:
        mark = "точный" if self.exact else "приближённый"
        return f"{self.kind} ({mark})"


# ═══════════════════════════════════════════════════════════════════════
#  ТОЧНЫЙ ПЕРЕНОС: ОСРЕДНЕНИЕ ПО ЯЧЕЙКЕ-ПРИЁМНИКУ
# ═══════════════════════════════════════════════════════════════════════

class AveragingTransfer(FieldTransfer):
    """
    Среднее по всем ячейкам источника, попавшим в ячейку приёмника.

    Требует вложенных структурированных сеток: каждая ячейка приёмника
    должна содержать одинаковое число ячеек источника. Это проверяется
    при построении — иначе «точный» перенос точным не будет, и лучше
    узнать об этом сразу.

    Идентификация ячеек идёт через ГЛОБАЛЬНЫЙ структурный индекс,
    вычисленный из координаты центра: он не зависит от того, как сетки
    разложены по рангам, а раскладки у двух сеток разные.
    """

    kind = "осреднение по ячейке"
    exact = True

    def __init__(self, DG0_src, DG0_dst,
                 src_spec: RectangleMeshSpec,
                 dst_spec: RectangleMeshSpec):
        super().__init__(DG0_src, DG0_dst)

        if not dst_spec.domain_matches(src_spec):
            raise ValueError("сетки покрывают разные области")

        self.dst_spec = dst_spec
        self.n_dst_cells_global = dst_spec.nx * dst_spec.ny

        # Ключ ячейки приёмника, в которую попадает каждая владеемая
        # ячейка источника.
        src_xy = owned_dof_coordinates(DG0_src)
        src_keys_local = self._cell_key(src_xy, dst_spec)
        self._src_keys = np.concatenate(self.comm.allgather(src_keys_local))

        # Ключ каждой владеемой ячейки приёмника.
        dst_xy = owned_dof_coordinates(DG0_dst)
        self._dst_keys = self._cell_key(dst_xy, dst_spec)

        # Сколько ячеек источника пришлось на каждую ячейку приёмника.
        self._counts = np.bincount(self._src_keys,
                                   minlength=self.n_dst_cells_global)
        self._verify_uniform_coverage(src_spec, dst_spec)

    @staticmethod
    def _cell_key(xy: np.ndarray, spec: RectangleMeshSpec) -> np.ndarray:
        """
        Глобальный линейный индекс ячейки сетки `spec`, содержащей точку.

        Берём floor по каждой оси: для вложенных сеток центр ячейки
        источника гарантированно лежит СТРОГО внутри ячейки приёмника,
        поэтому попадание на границу означало бы нарушение вложенности —
        его поймает проверка покрытия.
        """
        if len(xy) == 0:
            return np.zeros(0, dtype=np.int64)

        ix = np.floor(xy[:, 0] / spec.hx_mm).astype(np.int64)
        iy = np.floor(xy[:, 1] / spec.hy_mm).astype(np.int64)
        np.clip(ix, 0, spec.nx - 1, out=ix)
        np.clip(iy, 0, spec.ny - 1, out=iy)
        return iy * spec.nx + ix

    def _verify_uniform_coverage(self, src_spec, dst_spec) -> None:
        """
        Каждая ячейка приёмника обязана получить одинаковое, отличное от
        нуля число ячеек источника. Иначе осреднение — не осреднение по
        площади, и называть его точным нельзя.
        """
        expected = src_spec.n_cells // dst_spec.n_cells
        bad = np.flatnonzero(self._counts != expected)
        if bad.size:
            got = np.unique(self._counts[bad])
            raise ValueError(
                f"сетки не вложены как требуется: ожидалось по {expected} "
                f"ячеек источника на ячейку приёмника, встречено {got.tolist()} "
                f"(ячеек приёмника с отклонением: {bad.size}). Точное "
                f"осреднение здесь невозможно — используйте "
                f"NearestCellTransfer")
        self.cells_per_target = expected

    def apply(self, values_owned_src: np.ndarray) -> np.ndarray:
        values = self._gather_source(values_owned_src)
        sums = np.bincount(self._src_keys, weights=values,
                           minlength=self.n_dst_cells_global)
        means = sums / self._counts
        return means[self._dst_keys]


# ═══════════════════════════════════════════════════════════════════════
#  ПРИБЛИЖЁННЫЙ ПЕРЕНОС: БЛИЖАЙШАЯ ЯЧЕЙКА
# ═══════════════════════════════════════════════════════════════════════

class NearestCellTransfer(FieldTransfer):
    """
    Значение ячейки источника, чей центр ближе всего к центру ячейки
    приёмника.

    Работает для любых сеток и в обе стороны. Применяется, когда
    вложенности нет. Теряет информацию при огрублении: значение одной
    ячейки вместо среднего по всем попавшим.
    """

    kind = "ближайшая ячейка"
    exact = False

    def __init__(self, DG0_src, DG0_dst):
        super().__init__(DG0_src, DG0_dst)

        try:
            from scipy.spatial import cKDTree
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "NearestCellTransfer требует scipy: pip install 'cardiac-em[solver]'"
            ) from e

        src_xy = np.vstack(self.comm.allgather(owned_dof_coordinates(DG0_src)))
        dst_xy = owned_dof_coordinates(DG0_dst)

        if src_xy.shape[0] == 0:
            raise ValueError("сетка-источник не содержит ячеек")

        tree = cKDTree(src_xy)
        if dst_xy.shape[0]:
            dist, idx = tree.query(dst_xy)
            self._idx = idx
            self._max_dist_local = float(dist.max())
        else:
            self._idx = np.zeros(0, dtype=np.int64)
            self._max_dist_local = 0.0

        #: наибольшее расстояние до ближайшей ячейки источника, мм.
        #: Величина порядка шага сетки — норма; заметно больше означает,
        #: что сетки покрывают разные области.
        self.max_distance = self.comm.allreduce(self._max_dist_local, op=MPI.MAX)

    def apply(self, values_owned_src: np.ndarray) -> np.ndarray:
        values = self._gather_source(values_owned_src)
        return values[self._idx]


# ═══════════════════════════════════════════════════════════════════════
#  ВЫБОР СТРАТЕГИИ
# ═══════════════════════════════════════════════════════════════════════

def make_transfer(DG0_src, DG0_dst,
                  src_spec: RectangleMeshSpec,
                  dst_spec: RectangleMeshSpec,
                  strategy: str = "auto") -> FieldTransfer:
    """
    Построить оператор переноса.

    strategy
    --------
    "auto"       — точное осреднение, если сетки вложены; иначе
                   ближайшая ячейка (по умолчанию)
    "averaging"  — требовать точное осреднение; ошибка, если невозможно
    "nearest"    — всегда ближайшая ячейка

    Возвращаемый объект несёт `exact`, по которому вызывающий код может
    предупредить пользователя, что перенос приближённый.
    """
    if strategy not in ("auto", "averaging", "nearest"):
        raise ValueError(
            f"неизвестная стратегия {strategy!r}; "
            f"доступны 'auto', 'averaging', 'nearest'")

    if strategy == "nearest":
        return NearestCellTransfer(DG0_src, DG0_dst)

    nestable = (
        src_spec.cell_type == dst_spec.cell_type
        and src_spec.nx % dst_spec.nx == 0
        and src_spec.ny % dst_spec.ny == 0
        and src_spec.domain_matches(dst_spec)
    )

    if strategy == "averaging":
        if not nestable:
            raise ValueError(
                f"точное осреднение невозможно: сетка-источник "
                f"{src_spec.nx}×{src_spec.ny} не является целым измельчением "
                f"{dst_spec.nx}×{dst_spec.ny}")
        return AveragingTransfer(DG0_src, DG0_dst, src_spec, dst_spec)

    # auto
    if nestable:
        return AveragingTransfer(DG0_src, DG0_dst, src_spec, dst_spec)
    return NearestCellTransfer(DG0_src, DG0_dst)
