"""
Спецификации сеток и конфигурация ПАРЫ сеток (электрика + механика).
=====================================================================

Электрика и механика считаются на разных сетках одной и той же
физической области:

  * электрической сетке нужен мелкий шаг, чтобы разрешить фронт волны
    возбуждения (характерная ширина фронта ~0.1-0.3 мм);
  * механической сетке мелкий шаг не нужен — поле перемещений гладкое,
    а стоимость решения нелинейной задачи SNES растёт с числом DOF
    быстрее линейного.

Два способа задать пару (оба поддержаны, но они НЕ равноценны):

  (а) ВЛОЖЕННЫЕ сетки — `DualMeshConfig.nested(spec, coarsening=k)`.
      Механическая сетка получается делением электрической на целый
      коэффициент k, так что каждая механическая ячейка точно содержит
      k×k электрических. Согласованность областей гарантирована
      построением, а не проверкой.

  (б) ДВЕ НЕЗАВИСИМЫЕ спецификации — `DualMeshConfig(electric=...,
      mechanical=...)`. Общий случай (понадобится, когда механика
      придёт из отдельного файла сетки). Согласованность областей —
      проверяемый инвариант, а не гарантия.

Почему это важно, а не формальность: при вложенности перенос T_act с
мелкой сетки на крупную — ТОЧНОЕ осреднение по k² ячейкам. Без
вложенности приходится выбирать значение ближайшей ячейки, и крупная
механическая ячейка «видит» одну мелкую вместо среднего по всем. В
момент прохождения фронта, когда T_act внутри крупной ячейки меняется
от нуля до пика, это даёт реальную ошибку в суммарной силе. Поэтому
`DualMeshConfig` хранит происхождение сеток (см. `nesting()`), чтобы
слой `coupling/` мог выбрать точный оператор, когда он законно доступен.

ТОЧКА РАСШИРЕНИЯ: под реальную геометрию из файла (.msh/.xdmf) сюда
добавляется класс `FileMeshSpec` и включается в объединение `MeshSpec`
ниже; `fem/mesh.py` получает соответствующую ветку построения. Пока
такой класс не заведён намеренно — неиспользуемый код тоже нужно
сопровождать.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

__all__ = [
    "RectangleMeshSpec",
    "MeshSpec",
    "MeshNesting",
    "DualMeshConfig",
]


@dataclass(frozen=True)
class RectangleMeshSpec:
    """
    Равномерная прямоугольная сетка на [0, lx_mm] × [0, ly_mm].

    Поля
    ----
    nx, ny        : число ЭЛЕМЕНТОВ вдоль x и y (не узлов)
    lx_mm, ly_mm  : размеры области, мм
    cell_type     : "quadrilateral" или "triangle"
    """

    nx: int
    ny: int
    lx_mm: float = 10.0
    ly_mm: float = 10.0
    cell_type: str = "quadrilateral"

    _ALLOWED_CELL_TYPES = ("quadrilateral", "triangle")

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError(
                f"число элементов должно быть >= 1, получено nx={self.nx}, ny={self.ny}")
        if self.lx_mm <= 0.0 or self.ly_mm <= 0.0:
            raise ValueError(
                f"размеры области должны быть > 0, получено "
                f"lx_mm={self.lx_mm}, ly_mm={self.ly_mm}")
        if self.cell_type not in self._ALLOWED_CELL_TYPES:
            raise ValueError(
                f"cell_type={self.cell_type!r}; допустимы "
                f"{self._ALLOWED_CELL_TYPES}")

    # ── производные величины ──────────────────────────────────────────
    @property
    def n_cells(self) -> int:
        """Число ячеек (для треугольников — вдвое больше, чем квадов)."""
        base = self.nx * self.ny
        return 2 * base if self.cell_type == "triangle" else base

    @property
    def n_vertices(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def hx_mm(self) -> float:
        """Шаг сетки вдоль x, мм."""
        return self.lx_mm / self.nx

    @property
    def hy_mm(self) -> float:
        """Шаг сетки вдоль y, мм."""
        return self.ly_mm / self.ny

    @property
    def h_min_mm(self) -> float:
        """Минимальный шаг — ориентир для оценки устойчивости и CFL."""
        return min(self.hx_mm, self.hy_mm)

    def domain_matches(self, other: "RectangleMeshSpec",
                       rel_tol: float = 1e-12) -> bool:
        """Покрывают ли две спецификации одну и ту же физическую область."""
        return (math.isclose(self.lx_mm, other.lx_mm, rel_tol=rel_tol) and
                math.isclose(self.ly_mm, other.ly_mm, rel_tol=rel_tol))

    def coarsened(self, factor: int) -> "RectangleMeshSpec":
        """
        Огрублённая версия этой же сетки в `factor` раз по обоим
        направлениям. nx и ny обязаны делиться на factor нацело —
        иначе вложенность не получится, и лучше узнать об этом здесь,
        чем потом разбираться с неточным переносом полей.
        """
        if factor < 1:
            raise ValueError(f"коэффициент огрубления должен быть >= 1, получен {factor}")
        if self.nx % factor or self.ny % factor:
            raise ValueError(
                f"nx={self.nx}, ny={self.ny} не делятся нацело на {factor}; "
                f"вложенные сетки требуют целого отношения — возьмите "
                f"другой коэффициент или другое разрешение базовой сетки")
        return RectangleMeshSpec(
            nx=self.nx // factor,
            ny=self.ny // factor,
            lx_mm=self.lx_mm,
            ly_mm=self.ly_mm,
            cell_type=self.cell_type,
        )

    # ── сериализация ──────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "kind": "rectangle",
            "nx": self.nx, "ny": self.ny,
            "lx_mm": self.lx_mm, "ly_mm": self.ly_mm,
            "cell_type": self.cell_type,
        }

    @staticmethod
    def from_dict(d: dict) -> "RectangleMeshSpec":
        kind = d.get("kind", "rectangle")
        if kind != "rectangle":
            raise ValueError(f"ожидался kind='rectangle', получено {kind!r}")
        return RectangleMeshSpec(
            nx=int(d["nx"]), ny=int(d["ny"]),
            lx_mm=float(d.get("lx_mm", 10.0)),
            ly_mm=float(d.get("ly_mm", 10.0)),
            cell_type=str(d.get("cell_type", "quadrilateral")),
        )

    def __str__(self) -> str:
        return (f"{self.nx}×{self.ny} {self.cell_type} "
                f"на {self.lx_mm}×{self.ly_mm} мм "
                f"(h = {self.hx_mm:.3g}×{self.hy_mm:.3g} мм)")


# Объединение типов спецификаций сетки. Сейчас поддержан один тип;
# при добавлении FileMeshSpec его включают сюда, и ветка построения
# появляется в fem/mesh.py.
MeshSpec = Union[RectangleMeshSpec]


@dataclass(frozen=True)
class MeshNesting:
    """
    Результат анализа вложенности пары сеток.

    Поля
    ----
    is_nested : вложены ли сетки (каждая механическая ячейка содержит
                целое число электрических)
    kx, ky    : коэффициенты огрубления по x и y (только при is_nested)
    reason    : если не вложены — почему (для сообщений пользователю)
    """

    is_nested: bool
    kx: int | None = None
    ky: int | None = None
    reason: str = ""

    @property
    def cells_per_mech_cell(self) -> int | None:
        """Сколько электрических ячеек приходится на механическую."""
        if not self.is_nested:
            return None
        return self.kx * self.ky


@dataclass(frozen=True)
class DualMeshConfig:
    """
    Пара сеток: электрическая (мелкая) и механическая (крупная).

    Инвариант: обе сетки покрывают одну и ту же физическую область.
    Проверяется в __post_init__ — несогласованные области дают ошибку
    сразу при сборке конфигурации, а не молчаливо кривой перенос полей
    на десятой минуте счёта.

    Создание
    --------
    Предпочтительно — вложенные сетки:

        cfg = DualMeshConfig.nested(
            RectangleMeshSpec(nx=80, ny=80, lx_mm=10, ly_mm=10),
            coarsening=4)                     # механика станет 20×20

    Общий случай — две независимые спецификации:

        cfg = DualMeshConfig(
            electric=RectangleMeshSpec(nx=80, ny=80),
            mechanical=RectangleMeshSpec(nx=25, ny=25))
    """

    electric: MeshSpec
    mechanical: MeshSpec

    def __post_init__(self) -> None:
        if not self.electric.domain_matches(self.mechanical):
            raise ValueError(
                "электрическая и механическая сетки должны покрывать одну "
                f"область: электрика {self.electric.lx_mm}×{self.electric.ly_mm} мм, "
                f"механика {self.mechanical.lx_mm}×{self.mechanical.ly_mm} мм")

    # ── конструкторы ──────────────────────────────────────────────────
    @classmethod
    def nested(cls, electric: MeshSpec, coarsening: int = 2) -> "DualMeshConfig":
        """
        Построить пару, где механическая сетка — огрубление электрической
        в `coarsening` раз. Гарантирует вложенность, а значит — точный
        оператор переноса T_act (осреднение по ячейке).
        """
        return cls(electric=electric, mechanical=electric.coarsened(coarsening))

    @classmethod
    def uniform(cls, spec: MeshSpec) -> "DualMeshConfig":
        """Одна сетка на обе задачи — поведение исходного монолитного кода."""
        return cls(electric=spec, mechanical=spec)

    # ── анализ ────────────────────────────────────────────────────────
    def nesting(self) -> MeshNesting:
        """
        Определить, вложены ли сетки. Слой coupling/ использует это,
        чтобы выбрать точный оператор переноса вместо приближённого.
        """
        e, m = self.electric, self.mechanical

        if e.cell_type != m.cell_type:
            return MeshNesting(False, reason=(
                f"разные типы ячеек: {e.cell_type} и {m.cell_type}"))
        if e.nx % m.nx or e.ny % m.ny:
            return MeshNesting(False, reason=(
                f"разрешения не кратны: {e.nx}×{e.ny} и {m.nx}×{m.ny}"))

        return MeshNesting(True, kx=e.nx // m.nx, ky=e.ny // m.ny)

    @property
    def refinement_ratio(self) -> float:
        """Во сколько раз электрическая сетка мельче механической по шагу."""
        return self.mechanical.h_min_mm / self.electric.h_min_mm

    # ── сериализация ──────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "electric": self.electric.to_dict(),
            "mechanical": self.mechanical.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict) -> "DualMeshConfig":
        return DualMeshConfig(
            electric=RectangleMeshSpec.from_dict(d["electric"]),
            mechanical=RectangleMeshSpec.from_dict(d["mechanical"]),
        )

    def summary(self) -> str:
        nest = self.nesting()
        if nest.is_nested:
            note = (f"вложенные, {nest.kx}×{nest.ky} эл. ячеек "
                    f"на механическую → возможен точный перенос")
        else:
            note = f"не вложены ({nest.reason}) → перенос приближённый"
        return (f"  Сетка эл.    : {self.electric}\n"
                f"  Сетка мех.   : {self.mechanical}\n"
                f"  Вложенность  : {note}")
