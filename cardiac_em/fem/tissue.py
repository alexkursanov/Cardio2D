"""
Поля параметров ткани: базовые значения + переопределения по регионам.
=======================================================================

`Tissue` превращает спецификацию (`TissueBaseParams` + список
`RegionSpec`) в поля на КОНКРЕТНОЙ сетке. Поскольку сеток две, объектов
`Tissue` тоже два — по одному на сетку:

    tissue_e = Tissue.for_electrics(mesh_e, base, regions)
    tissue_m = Tissue.for_mechanics(mesh_m, base, regions)

Переносить параметры между сетками НЕ НУЖНО: регионы заданы
аналитически по физическим координатам, поэтому каждая сетка
раскрашивается независимо и одинаково корректно. Это свойство стоит
беречь — как только регион начнёт зависеть от нумерации ячеек, двойная
сетка сломается.

В каком пространстве живёт какой параметр
------------------------------------------
Выбор не произволен, а следует из того, как параметр используется:

    AP_c1, AP_c2, AP_a, AP_b, AP_d   P1, numpy   поточечный шаг реакции
                                                  идёт по узлам P1, где
                                                  живут u_ap и v_ap
    D_LONG, D_TRANS, FIBER_ANGLE_DEG DG0, Function  входят в UFL-форму
    T_MAX                            DG0, both      T_act считается на DG0
    MU, B_ISO, MU_F, B_F, KAPPA      DG0, Function  функция энергии в UFL

Разные пространства означают и разную границу региона: на P1 регион
«захватывает» узлы, попавшие внутрь, на DG0 — ячейки, чей ЦЕНТР попал
внутрь. Для гладких регионов разница в полклетки; для мелких деталей
это существенно (см. `empty_regions()`).

Осторожно с мелкими регионами на крупной сетке
-----------------------------------------------
Появление второй, более грубой сетки создаёт новый режим отказа:
регион может покрывать ячейки на электрической сетке и НИ ОДНОЙ на
механической — если он мельче шага крупной сетки или проскочил между
центрами ячеек. Молча это даёт «рубец, который электрически есть, а
механически отсутствует». `empty_regions()` такие случаи находит;
вызывающий код обязан о них предупредить.
"""

from __future__ import annotations

import numpy as np

import dolfinx.fem as fem
import ufl
from ufl import as_vector

from ..config.tissue_spec import RegionSpec, TissueBaseParams
from .mesh import dof_coordinates

__all__ = ["Tissue"]

# Точность, до которой округляются координаты при отнесении точек к
# регионам (знаков после запятой, в мм). См. Tissue._region_ids.
_COORD_DECIMALS = 10


# Параметры, которые нужны ПОТОЧЕЧНО на узлах P1 (явный шаг реакции).
_P1_KEYS = ("AP_c1", "AP_c2", "AP_a", "AP_b", "AP_d")

# Параметры, которые входят в UFL-формы или считаются на ячейках.
_DG0_KEYS = ("D_LONG", "D_TRANS", "T_MAX", "FIBER_ANGLE_DEG",
             "MU", "B_ISO", "MU_F", "B_F", "KAPPA")


class Tissue:
    """
    Поля параметров ткани на одной сетке.

    Параметры
    ---------
    mesh : dolfinx.mesh.Mesh
    base : TissueBaseParams
        Фоновые значения — действуют там, где не задано ни одного региона.
    regions : последовательность RegionSpec
        Применяются ПО ПОРЯДКУ: более поздние перекрывают более ранние.
    keys : последовательность строк, optional
        Какие параметры строить. По умолчанию — все. Электрической сетке
        не нужны модули упругости, механической — проводимость; строить
        лишнее дёшево, но путает при отладке, поэтому набор задаётся явно
        через `for_electrics` / `for_mechanics`.
    name : str
        Метка для сообщений («электрическая» / «механическая»).
    """

    def __init__(self, mesh, base: TissueBaseParams,
                 regions: tuple[RegionSpec, ...] = (),
                 keys: tuple[str, ...] | None = None,
                 name: str = ""):
        self.mesh = mesh
        self.base = base
        self.regions = tuple(regions)
        self.name = name

        flat_base = base.to_flat_dict()
        requested = tuple(flat_base) if keys is None else tuple(keys)

        unknown = set(requested) - set(flat_base)
        if unknown:
            raise ValueError(
                f"неизвестные параметры {sorted(unknown)}; "
                f"допустимы {sorted(flat_base)}")
        self.keys = requested

        # ── пространства ──────────────────────────────────────────────
        self.P1 = fem.functionspace(mesh, ("Lagrange", 1))
        self.DG0 = fem.functionspace(mesh, ("DG", 0))

        p1_xy = dof_coordinates(self.P1)[:, :2]
        dg0_xy = dof_coordinates(self.DG0)[:, :2]

        # ── принадлежность регионам ───────────────────────────────────
        # -1 = базовая ткань, i = regions[i]. Считается независимо для
        # каждого пространства: у них разные точки (узлы против центров).
        self.region_id_p1 = self._region_ids(p1_xy, self.regions)
        self.region_id_dg0 = self._region_ids(dg0_xy, self.regions)

        # ── раскладка значений ────────────────────────────────────────
        self._p1_arrays: dict[str, np.ndarray] = {}
        self._dg0_arrays: dict[str, np.ndarray] = {}
        self._dg0_functions: dict[str, fem.Function] = {}

        for key in self.keys:
            if key in _P1_KEYS:
                self._p1_arrays[key] = self._build_array(
                    key, flat_base[key], p1_xy, self.region_id_p1)
            if key in _DG0_KEYS:
                arr = self._build_array(
                    key, flat_base[key], dg0_xy, self.region_id_dg0)
                self._dg0_arrays[key] = arr
                self._dg0_functions[key] = self._as_function(key, arr)

        # ── карта регионов для визуализации ───────────────────────────
        # 0 = базовая ткань, i+1 = regions[i]; так «нет региона» отличимо
        # от «регион номер ноль» при просмотре в ParaView.
        self.region_id_fn = self._as_function(
            "region_id", (self.region_id_dg0 + 1).astype(np.float64))

    # ── конструкторы под конкретную задачу ────────────────────────────
    @classmethod
    def for_electrics(cls, mesh, base: TissueBaseParams,
                      regions: tuple[RegionSpec, ...] = ()) -> "Tissue":
        """Параметры, нужные монодоменной задаче, на электрической сетке."""
        return cls(mesh, base, regions,
                   keys=TissueBaseParams.ELECTRIC_KEYS, name="электрическая")

    @classmethod
    def for_mechanics(cls, mesh, base: TissueBaseParams,
                      regions: tuple[RegionSpec, ...] = ()) -> "Tissue":
        """Параметры, нужные задаче механики, на механической сетке."""
        return cls(mesh, base, regions,
                   keys=TissueBaseParams.MECHANICAL_KEYS, name="механическая")

    # ── построение ────────────────────────────────────────────────────
    @staticmethod
    def _region_ids(xy: np.ndarray,
                    regions: tuple[RegionSpec, ...]) -> np.ndarray:
        """
        Номер региона для каждой точки.

        Координаты перед проверкой ОКРУГЛЯЮТСЯ до 1e-10 мм. У DOLFINx
        узлы несут шум округления (y = 2.0000000000000004 вместо 2.0), и
        без округления регион с границей ровно по краю области [0, 2]
        терял бы отдельные граничные узлы — параметры в них молча
        оставались бы базовыми. Найдено тестом снимков на шаге 9.
        Разрешение 1e-10 мм на порядки мельче любого шага сетки.
        """
        rid = np.full(len(xy), -1, dtype=np.int64)
        if len(xy) == 0:
            return rid
        xr = np.round(np.asarray(xy, dtype=np.float64), _COORD_DECIMALS)
        x, y = xr[:, 0] + 0.0, xr[:, 1] + 0.0          # −0.0 → 0.0
        for i, region in enumerate(regions):
            rid[region.contains(x, y)] = i
        return rid

    def _build_array(self, key: str, base_value: float,
                     xy: np.ndarray, rid: np.ndarray) -> np.ndarray:
        arr = np.full(len(xy), float(base_value), dtype=np.float64)
        for i, region in enumerate(self.regions):
            if key in region.overrides:
                arr[rid == i] = float(region.overrides[key])
        return arr

    def _as_function(self, name: str, arr: np.ndarray) -> fem.Function:
        fn = fem.Function(self.DG0, name=name)
        n_owned = self.DG0.dofmap.index_map.size_local
        fn.x.array[:n_owned] = arr[:n_owned]
        fn.x.scatter_forward()
        return fn

    # ── доступ ────────────────────────────────────────────────────────
    def _check_key(self, key: str) -> None:
        if key not in self.keys:
            raise KeyError(
                f"параметр {key!r} не построен на {self.name or 'этой'} сетке "
                f"(построены: {sorted(self.keys)}). Если он действительно "
                f"нужен здесь — расширьте набор ключей при создании Tissue")

    def dg0(self, key: str) -> fem.Function:
        """Поле параметра на DG0 — для подстановки в UFL-формы."""
        self._check_key(key)
        try:
            return self._dg0_functions[key]
        except KeyError:
            raise KeyError(
                f"параметр {key!r} не живёт на DG0 (он поточечный, на P1); "
                f"используйте p1_array({key!r})") from None

    def dg0_array(self, key: str) -> np.ndarray:
        """Значения на DG0 как numpy, полный локальный массив (с гало)."""
        self._check_key(key)
        try:
            return self._dg0_arrays[key]
        except KeyError:
            raise KeyError(f"параметр {key!r} не живёт на DG0") from None

    def p1_array(self, key: str) -> np.ndarray:
        """Значения на узлах P1 как numpy — для явного шага реакции."""
        self._check_key(key)
        try:
            return self._p1_arrays[key]
        except KeyError:
            raise KeyError(
                f"параметр {key!r} не живёт на P1 (он на ячейках, DG0); "
                f"используйте dg0({key!r})") from None

    # частые параметры — короткими именами, чтобы формы читались
    @property
    def d_long(self) -> fem.Function: return self.dg0("D_LONG")
    @property
    def d_trans(self) -> fem.Function: return self.dg0("D_TRANS")
    @property
    def t_max(self) -> fem.Function: return self.dg0("T_MAX")
    @property
    def t_max_array(self) -> np.ndarray: return self.dg0_array("T_MAX")
    @property
    def fiber_angle(self) -> fem.Function: return self.dg0("FIBER_ANGLE_DEG")
    @property
    def mu(self) -> fem.Function: return self.dg0("MU")
    @property
    def b_iso(self) -> fem.Function: return self.dg0("B_ISO")
    @property
    def mu_f(self) -> fem.Function: return self.dg0("MU_F")
    @property
    def b_f(self) -> fem.Function: return self.dg0("B_F")
    @property
    def kappa(self) -> fem.Function: return self.dg0("KAPPA")

    # ── направления волокон ───────────────────────────────────────────
    def fiber_vector(self):
        """
        UFL-вектор f0(x) — направление волокна, из поля угла на DG0.
        Может меняться по ткани, если регион переопределяет
        FIBER_ANGLE_DEG.
        """
        theta = self.fiber_angle * (np.pi / 180.0)
        return as_vector([ufl.cos(theta), ufl.sin(theta)])

    def sheet_vector(self):
        """UFL-вектор s0(x) ⟂ f0(x) — поперечное направление."""
        theta = self.fiber_angle * (np.pi / 180.0)
        return as_vector([-ufl.sin(theta), ufl.cos(theta)])

    # ── диагностика ───────────────────────────────────────────────────
    def cells_per_region(self) -> dict[int, int]:
        """
        Сколько ячеек (владеемых этим рангом) попало в каждый регион.
        Ключ -1 — базовая ткань.
        """
        n_owned = self.DG0.dofmap.index_map.size_local
        rid = self.region_id_dg0[:n_owned]
        return {int(i): int((rid == i).sum())
                for i in range(-1, len(self.regions))}

    def empty_regions(self) -> list[int]:
        """
        Индексы регионов, не покрывших НИ ОДНОЙ ячейки (по всем рангам).

        Главный режим отказа при двух сетках: регион мельче шага крупной
        сетки либо проскочил между центрами ячеек. Электрически он есть,
        механически его нет — и это не видно, пока не начнёшь искать.
        """
        from mpi4py import MPI

        if not self.regions:
            return []

        n_owned = self.DG0.dofmap.index_map.size_local
        rid = self.region_id_dg0[:n_owned]
        local = np.array([(rid == i).sum() for i in range(len(self.regions))],
                         dtype=np.int64)
        total = np.zeros_like(local)
        self.mesh.comm.Allreduce(local, total, op=MPI.SUM)
        return [i for i, n in enumerate(total) if n == 0]

    def is_uniform(self, key: str) -> bool:
        """
        Одинаков ли параметр по всей ткани (по всем рангам).

        Полезно для диагностики и как задел на оптимизацию: однородный
        параметр можно было бы подставлять в форму как `fem.Constant`
        вместо DG0-поля. Сейчас такая оптимизация не делается — выигрыш
        невелик, а различие типов усложнило бы API.
        """
        from mpi4py import MPI

        self._check_key(key)
        arr = (self._dg0_arrays.get(key)
               if key in self._dg0_arrays else self._p1_arrays[key])
        space = self.DG0 if key in self._dg0_arrays else self.P1
        n_owned = space.dofmap.index_map.size_local
        owned = arr[:n_owned]

        lo = self.mesh.comm.allreduce(
            owned.min() if owned.size else np.inf, op=MPI.MIN)
        hi = self.mesh.comm.allreduce(
            owned.max() if owned.size else -np.inf, op=MPI.MAX)
        return bool(np.isclose(lo, hi))

    def value_range(self, key: str) -> tuple[float, float]:
        """Глобальный диапазон значений параметра — для сводки."""
        from mpi4py import MPI

        self._check_key(key)
        arr = (self._dg0_arrays.get(key)
               if key in self._dg0_arrays else self._p1_arrays[key])
        space = self.DG0 if key in self._dg0_arrays else self.P1
        n_owned = space.dofmap.index_map.size_local
        owned = arr[:n_owned]

        lo = self.mesh.comm.allreduce(
            owned.min() if owned.size else np.inf, op=MPI.MIN)
        hi = self.mesh.comm.allreduce(
            owned.max() if owned.size else -np.inf, op=MPI.MAX)
        return float(lo), float(hi)

    def summary(self) -> str:
        head = f"  Ткань ({self.name or 'без имени'}): "
        lines = [head + f"{len(self.regions)} регионов + базовая ткань"]
        for key in sorted(self.keys):
            lo, hi = self.value_range(key)
            mark = "" if np.isclose(lo, hi) else "   ← неоднородно"
            lines.append(f"    {key:16s}: [{lo:.4g} … {hi:.4g}]{mark}")

        empty = self.empty_regions()
        if empty:
            names = [self.regions[i].name or f"#{i}" for i in empty]
            lines.append(
                f"    [ВНИМАНИЕ] регионы {names} не покрыли ни одной ячейки "
                f"этой сетки — слишком мелкие для её шага")
        return "\n".join(lines)
