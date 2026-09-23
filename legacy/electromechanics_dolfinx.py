"""
Электромеханика миокарда — DOLFINx 0.10 + MPI + PETSc SNES + PyVista
=======================================================================

Физика
------
  Электрика  : Алиев–Панфилов / Rogers–McCulloch (монодоменная PDE)
               Strang splitting: реакция(dt/2) → диффузия(dt) → реакция(dt/2)
               Диффузия: Crank–Nicolson (PETSc KSP CG + HYPRE)

  Активная   : T_act(u) = T_max·u·(1−u)  [кПа]
               σ_act = T_act/J · (f̂⊗f̂)

  Пассивная  : Ψ = μ/b·exp(b·(I₁−2)) + μ_f/(2b_f)·exp(b_f·(I₄−1)²) + κ/2·(J−1)²
               UFL auto-diff → PETSc SNES (Newton, LU/MUMPS)

Раздельные сетки электрики и механики
--------------------------------------
Электрика и механика решаются на РАЗНЫХ сетках одной физической
области (Nx_e×Ny_e — мелкая, для фронта возбуждения; Nx_m×Ny_m —
крупная, дешевле для SNES). Каждый механический шаг T_act переносится
с электрической DG0 на механическую DG0 по ближайшему центру ячейки
(класс FieldTransferDG0, строится один раз при старте). REGIONS
заданы аналитически по координатам — применяются к обеим сеткам
независимо, без искажений на стыке.

  python3 electromechanics_dolfinx.py --nx-e 80 --ny-e 80 --nx-m 20 --ny-m 20

Запуск
------
  python3 electromechanics_dolfinx.py                    # с нуля
  mpirun -n 8 python3 electromechanics_dolfinx.py        # параллельно

Продолжение расчёта (рестарт)
-----------------------------
  # продолжить с последней точки другого эксперимента
  python3 electromechanics_dolfinx.py \
      --restart output_em/ckpt_last.npz --out output_em_run2 --t-end 3000

  # то же, но с обнулением времени и новым стимулом на t = 20 мс
  python3 electromechanics_dolfinx.py \
      --restart output_em/ckpt_t000500.00ms.npz --out output_em_s2 \
      --reset-time --stim-times "20" --t-end 1000

  # серия стимулов (например, протокол S1-S1-S1 с периодом 1000 мс)
  python3 electromechanics_dolfinx.py --stim-times "0,1000,2000,3000" \
      --t-end 3500

  # перенос состояния на сетку другого разрешения (ближайший сосед)
  python3 electromechanics_dolfinx.py --restart old/ckpt_last.npz --allow-interp

Неоднородные области ткани (REGIONS)
-------------------------------------
Ткань может быть разбита на области с собственными параметрами
клеток — возбудимостью, проводимостью, сократимостью, жёсткостью —
поверх базовых (глобальных) значений. Задаётся списком REGIONS в
разделе параметров (см. ниже) или файлом --regions-json:

  python3 electromechanics_dolfinx.py --regions-json ischemia.json

Формат JSON — список областей, каждая с формой и переопределениями:
  [
    {"shape": "rect",   "params": {"x0": 6, "x1": 10, "y0": 0, "y1": 10},
     "overrides": {"AP_c1": 0.13, "AP_a": 0.20, "D_LONG": 0.05,
                   "D_TRANS": 0.01, "T_MAX": 20.0}},
    {"shape": "circle", "params": {"cx": 8, "cy": 5, "r": 1.0},
     "overrides": {"AP_c1": 0.0, "D_LONG": 0.001, "D_TRANS": 0.001,
                   "T_MAX": 0.0, "MU": 5.0, "MU_F": 15.0}}
  ]
Области накладываются по порядку списка — более поздние перекрывают
более ранние (и базовую ткань) в местах пересечения. Доступные
формы: rect(x0,x1,y0,y1), circle(cx,cy,r); произвольные фигуры —
через REGIONS в коде (предикат — любая функция numpy-массивов x,y).
Переопределять можно любые из: AP_c1, AP_c2, AP_a, AP_b, AP_d,
D_LONG, D_TRANS, T_MAX, FIBER_ANGLE_DEG, MU, B_ISO, MU_F, B_F, KAPPA.
Карты регионов и ключевые поля сохраняются раздельно в
<out>/tissue_regions_electric.xdmf (эл. сетка) и
<out>/tissue_regions_mechanical.xdmf (мех. сетка) для проверки в ParaView.

При рестарте этапы 1 (растяжение) и 2 (фиксация) ПРОПУСКАЮТСЯ:
предварительно растянутое состояние уже содержится в сохранённом поле u,
а граничные условия восстанавливаются как «зажим» на загруженном u.

Чекпоинт хранит поля в виде пар (координата → значение), поэтому
не зависит от числа MPI-рангов и порядка нумерации DOF.

Результаты
----------
  <out>/mechanics.xdmf                  — перемещения + T_act, мех. сетка (ParaView)
  <out>/electrics.xdmf                  — u_ap, v_ap, эл. сетка
  <out>/tissue_regions_electric.xdmf    — регионы + D_long/D_trans/T_max (эл. сетка)
  <out>/tissue_regions_mechanical.xdmf  — регионы + MU/MU_F/KAPPA (мех. сетка)
  <out>/snap_mech_*.vtk / snap_elec_*.vtk — снапшоты полей (PyVista), раздельно
  <out>/ckpt_*.npz                      — чекпоинты для продолжения расчёта

Версия: DOLFINx 0.10.x
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc as fem_petsc
import dolfinx.io as io
import dolfinx.log as dlog
import dolfinx.mesh as dmesh

import ufl
from ufl import (
    Identity, TestFunction, TrialFunction,
    conditional, derivative, det, dot, dx, exp, grad, inner, inv, le,
    outer, sqrt, tr,
    as_vector,
)

# ═══════════════════════════════════════════════════════════════════════
#  ПАРАМЕТРЫ  (мм, мс, кПа)
# ═══════════════════════════════════════════════════════════════════════

Lx, Ly = 10.0, 10.0   # мм, геометрия (общая для обеих сеток)

# Раздельные сетки электрики и механики (см. класс FieldTransferDG0 ниже).
# Электрике нужна мелкая сетка для разрешения фронта волны возбуждения
# (характерная ширина фронта ~0.1-0.3 мм при D~0.1-0.3 мм²/мс), механике —
# как правило, вдвое-впятеро крупнее: поле перемещений гораздо более
# гладкое, а квадратичный рост стоимости решения SNES с числом DOF делает
# избыточно мелкую механическую сетку дорогой без выигрыша в точности.
Nx_e, Ny_e = 40, 40     # электрическая сетка (мелкая)
Nx_m, Ny_m = 20, 20     # механическая сетка (крупная)

FIBER_ANGLE_DEG = 0.0  # ориентация волокон, градусы от оси x

# Пассивная механика (трансверсально-изотропная гиперупругость)
MU    = 1.0    # кПа — изотропная жёсткость матрицы
B_ISO = 1.0    # б/р
MU_F  = 3.0    # кПа — жёсткость вдоль волокон
B_F   = 2.0    # б/р
KAPPA = 100.0  # кПа — штраф несжимаемости

STRETCH   = 1.2575   # λ_f — целевое предварительное растяжение
N_STRETCH = 10    # шагов нагружения

# Rogers–McCulloch 1994 (параметры в мс, для кардиомиоцитов)
AP_c1   = 0.26    # мс⁻¹ — скорость нарастания
AP_c2   = 0.1     # мс⁻¹ — купирование через v
AP_a    = 0.13    # б/р  — порог возбуждения
AP_b    = 0.013   # мс⁻¹ — скорость восстановления v
AP_d    = 1.0     # б/р  — коэффициент восстановления
D_LONG  = 0.3     # мм²/мс — диффузия вдоль волокон
D_TRANS = 0.03    # мм²/мс — диффузия поперёк

T_MAX = 120.0   # кПа — пиковое активное напряжение

# Стимул
STIM_XMAX = 0.0   # мм — ширина стимульной полосы (левая грань)
STIM_TIMES = [50.0]   # мс — моменты начала стимулов, список
                       # (можно переопределить ключом --stim-times "0,1000,2000")
STIM_DUR  = 2.0   # мс — длительность каждого стимула
STIM_AMP  = 20.0   # б/р — амплитуда стимула (Rogers-McCulloch)
STIM_RISE_TIME = 0.5 # мс — время нарастания (важно для плавности!)

# Временная шкала
DT_E   = 0.05    # мс — шаг электрики
T_END  = 1500.0   # мс
N_MECH = 20      # электрических шагов на 1 механический → dt_mech = 1 мс

# Вывод
OUT_DIR    = Path("output_em")
SAVE_EVERY = 10                               # мех.шагов между записями
SNAP_TIMES = {0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0,
              500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1200.0, 1300.0, 1400.0}

# ── Чекпоинты ─────────────────────────────────────────────────────────
CKPT_EVERY    = 100    # мех.шагов между чекпоинтами (0 → не сохранять)
CKPT_KEEP_ALL = False  # True → хранить все ckpt_t*.npz, False → только последний
CKPT_ROUND    = 6      # знаков при округлении координат для сопоставления DOF

# ───────────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.rank


# ═══════════════════════════════════════════════════════════════════════
#  СЕТКА
# ═══════════════════════════════════════════════════════════════════════

def make_mesh(nx: int, ny: int) -> dolfinx.mesh.Mesh:
    return dmesh.create_rectangle(
        comm,
        [[0.0, 0.0], [Lx, Ly]],
        [nx, ny],
        cell_type=dmesh.CellType.quadrilateral,
    )


# ═══════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════════════

def _dof_coords(V) -> np.ndarray:
    """
    Координаты DOF (включая ghost) для P1/векторного P1/DG0.
    Для DG0 при отсутствии tabulate_dof_coordinates используем
    середины ячеек, разложенные по dofmap.
    """
    try:
        return V.tabulate_dof_coordinates()
    except Exception:
        mesh = V.mesh
        tdim = mesh.topology.dim
        imap = mesh.topology.index_map(tdim)
        n_cells = imap.size_local + imap.num_ghosts
        mids = dmesh.compute_midpoints(mesh, tdim,
                                       np.arange(n_cells, dtype=np.int32))
        n_dofs = (V.dofmap.index_map.size_local +
                  V.dofmap.index_map.num_ghosts)
        coords = np.zeros((n_dofs, 3), dtype=np.float64)
        dofs = np.asarray(V.dofmap.list).reshape(n_cells, -1)[:, 0]
        coords[dofs] = mids
        return coords


# ═══════════════════════════════════════════════════════════════════════
#  ОБЛАСТИ С НЕОДНОРОДНЫМИ ПАРАМЕТРАМИ КЛЕТОК (REGIONS)
# ═══════════════════════════════════════════════════════════════════════
#
#  REGIONS — список (предикат(x, y) -> bool-массив, {переопределения}),
#  применяемых по порядку: более поздние перекрывают более ранние
#  (и базовую, глобально заданную выше ткань) там, где области
#  пересекаются. Пустой список REGIONS = однородная ткань (как раньше).
#
#  Переопределять можно любые из параметров:
#    электрика (Rogers–McCulloch): AP_c1, AP_c2, AP_a, AP_b, AP_d
#    проводимость / волокна      : D_LONG, D_TRANS, FIBER_ANGLE_DEG
#    сократимость                : T_MAX
#    пассивная механика          : MU, B_ISO, MU_F, B_F, KAPPA
#
#  Готовые формы: rect(x0,x1,y0,y1), circle(cx,cy,r). Произвольная
#  область — любая функция predicate(x, y) -> bool-массив (numpy).
# ═══════════════════════════════════════════════════════════════════════

def rect(x0: float, x1: float, y0: float, y1: float):
    """Прямоугольная область [x0,x1]×[y0,y1], мм."""
    def _pred(x, y):
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return _pred


def circle(cx: float, cy: float, r: float):
    """Круглая область радиуса r с центром (cx, cy), мм."""
    def _pred(x, y):
        return (x - cx)**2 + (y - cy)**2 <= r**2
    return _pred


# Пример (закомментирован): ишемическая зона справа + рубец в центре.
# Снижены возбудимость и проводимость в зоне ишемии, контрактильность
# почти подавлена; в рубце ткань невозбудима, непроводяща и жёстче.
#
# REGIONS = [
#     (rect(6.0, 10.0, 0.0, 10.0), {
#         "AP_c1": 0.13, "AP_a": 0.20,
#         "D_LONG": 0.05, "D_TRANS": 0.01,
#         "T_MAX": 20.0,
#     }),
#     (circle(8.0, 5.0, 1.0), {
#         "AP_c1": 0.0, "D_LONG": 0.001, "D_TRANS": 0.001,
#         "T_MAX": 0.0, "MU": 5.0, "MU_F": 15.0,
#     }),
# ]
REGIONS: list = []   # по умолчанию — однородная ткань (переопределяется
                       # через REGIONS выше или ключом --regions-json)


class Tissue:
    """
    Пространственно-неоднородные параметры ткани, построенные из REGIONS.

    Хранит:
      - параметры реакции (AP_c1/c2/a/b/d) как numpy-массивы на P1
        (используются в явном полушаге _reaction);
      - T_max, D_long, D_trans, угол волокон, пассивные модули
        (MU, B_ISO, MU_F, B_F, KAPPA) как fem.Function на DG0
        (используются напрямую в UFL-формах механики и диффузии);
      - region_id (DG0) — карта регионов для визуализации/контроля.

    Регион для каждого DOF определяется по его физическим координатам
    независимо для P1 и для DG0 (центры ячеек), поэтому неоднородность
    корректно передаётся в оба пространства без проекции.
    """

    _REACTION_KEYS = ("AP_c1", "AP_c2", "AP_a", "AP_b", "AP_d")
    _FIELD_KEYS = ("D_LONG", "D_TRANS", "T_MAX", "FIBER_ANGLE_DEG",
                   "MU", "B_ISO", "MU_F", "B_F", "KAPPA")

    def __init__(self, mesh, regions: list | None = None):
        self.mesh = mesh
        self.regions = REGIONS if regions is None else regions

        base = dict(AP_c1=AP_c1, AP_c2=AP_c2, AP_a=AP_a, AP_b=AP_b, AP_d=AP_d,
                    D_LONG=D_LONG, D_TRANS=D_TRANS, T_MAX=T_MAX,
                    FIBER_ANGLE_DEG=FIBER_ANGLE_DEG,
                    MU=MU, B_ISO=B_ISO, MU_F=MU_F, B_F=B_F, KAPPA=KAPPA)

        P1  = fem.functionspace(mesh, ("Lagrange", 1))
        DG0 = fem.functionspace(mesh, ("DG", 0))
        self.P1, self.DG0 = P1, DG0

        p1_xy  = _dof_coords(P1)[:, :2]
        dg0_xy = _dof_coords(DG0)[:, :2]

        rid_p1  = self._region_ids(p1_xy,  self.regions)
        rid_dg0 = self._region_ids(dg0_xy, self.regions)
        self.region_id_p1  = rid_p1
        self.region_id_dg0 = rid_dg0

        # ── параметры реакции: numpy-массивы на P1 ─────────────────
        for key in self._REACTION_KEYS:
            arr = np.full(len(p1_xy), base[key], dtype=np.float64)
            for i, (_pred, ov) in enumerate(self.regions):
                if key in ov:
                    arr[rid_p1 == i] = ov[key]
            setattr(self, "_arr_" + key.lower(), arr)

        # ── параметры ткани: fem.Function на DG0 (для UFL) ─────────
        self._dg0_fields: dict[str, fem.Function] = {}
        n_dg0 = DG0.dofmap.index_map.size_local
        for key in self._FIELD_KEYS:
            arr = np.full(len(dg0_xy), base[key], dtype=np.float64)
            for i, (_pred, ov) in enumerate(self.regions):
                if key in ov:
                    arr[rid_dg0 == i] = ov[key]
            fn = fem.Function(DG0, name=key)
            fn.x.array[:n_dg0] = arr[:n_dg0]
            fn.x.scatter_forward()
            self._dg0_fields[key] = fn
            setattr(self, "_arr_" + key.lower(), arr)

        # карта регионов для визуализации (0 = базовая ткань, i+1 = REGIONS[i])
        rid_fn = fem.Function(DG0, name="region_id")
        rid_fn.x.array[:n_dg0] = (rid_dg0[:n_dg0] + 1).astype(np.float64)
        rid_fn.x.scatter_forward()
        self.region_id_fn = rid_fn

        self.n_regions = len(self.regions)

    @staticmethod
    def _region_ids(xy: np.ndarray, regions: list) -> np.ndarray:
        """-1 = базовая ткань, i = regions[i] (позднее перекрывает раннее)."""
        rid = np.full(len(xy), -1, dtype=np.int64)
        x, y = xy[:, 0], xy[:, 1]
        for i, (pred, _ov) in enumerate(regions):
            mask = np.asarray(pred(x, y), dtype=bool)
            rid[mask] = i
        return rid

    # ── доступ к DG0-полям параметров (для UFL-форм) ────────────────
    @property
    def D_long_fn(self) -> fem.Function:      return self._dg0_fields["D_LONG"]
    @property
    def D_trans_fn(self) -> fem.Function:     return self._dg0_fields["D_TRANS"]
    @property
    def T_max_fn(self) -> fem.Function:       return self._dg0_fields["T_MAX"]
    @property
    def fiber_angle_fn(self) -> fem.Function: return self._dg0_fields["FIBER_ANGLE_DEG"]
    @property
    def mu_fn(self) -> fem.Function:          return self._dg0_fields["MU"]
    @property
    def b_iso_fn(self) -> fem.Function:       return self._dg0_fields["B_ISO"]
    @property
    def mu_f_fn(self) -> fem.Function:        return self._dg0_fields["MU_F"]
    @property
    def b_f_fn(self) -> fem.Function:         return self._dg0_fields["B_F"]
    @property
    def kappa_fn(self) -> fem.Function:       return self._dg0_fields["KAPPA"]

    # ── доступ к numpy-массивам параметров реакции (для _reaction) ─
    @property
    def ap_c1_arr(self) -> np.ndarray: return self._arr_ap_c1
    @property
    def ap_c2_arr(self) -> np.ndarray: return self._arr_ap_c2
    @property
    def ap_a_arr(self) -> np.ndarray:  return self._arr_ap_a
    @property
    def ap_b_arr(self) -> np.ndarray:  return self._arr_ap_b
    @property
    def ap_d_arr(self) -> np.ndarray:  return self._arr_ap_d
    @property
    def t_max_arr(self) -> np.ndarray: return self._arr_t_max

    def fiber_vector(self):
        """UFL-вектор направления волокна f0(x), из поля угла (градусы) на DG0."""
        theta = self.fiber_angle_fn * (np.pi / 180.0)
        return as_vector([ufl.cos(theta), ufl.sin(theta)])

    def sheet_vector(self):
        """UFL-вектор поперечного (листового) направления s0(x) ⟂ f0(x)."""
        theta = self.fiber_angle_fn * (np.pi / 180.0)
        return as_vector([-ufl.sin(theta), ufl.cos(theta)])

    def summary(self) -> str:
        lines = [f"  Регионы      : {self.n_regions} "
                 f"(+ базовая ткань), диапазоны параметров по всей ткани:"]
        for key in self._FIELD_KEYS:
            arr = getattr(self, "_arr_" + key.lower())
            lo, hi = float(arr.min()), float(arr.max())
            mark = "" if np.isclose(lo, hi) else "  ← неоднородно"
            lines.append(f"    {key:16s}: [{lo:.4g} … {hi:.4g}]{mark}")
        return "\n".join(lines)


def load_regions_json(path: Path) -> list:
    """
    Загрузить REGIONS из JSON-файла (см. пример в docstring модуля).
    Формат: список {"shape": "rect"|"circle", "params": {...},
                     "overrides": {...}}.
    """
    import json
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)

    shape_builders = {"rect": rect, "circle": circle}
    regions = []
    for i, entry in enumerate(spec):
        shape = entry.get("shape")
        if shape not in shape_builders:
            raise ValueError(
                f"[regions-json] запись {i}: неизвестная форма {shape!r} "
                f"(доступны: {sorted(shape_builders)})")
        params = entry.get("params", {})
        try:
            pred = shape_builders[shape](**params)
        except TypeError as e:
            raise ValueError(f"[regions-json] запись {i} ({shape}): {e}") from e
        overrides = entry.get("overrides", {})
        regions.append((pred, overrides))
    return regions


# ═══════════════════════════════════════════════════════════════════════
#  ПЕРЕНОС ПОЛЕЙ МЕЖДУ НЕСОВПАДАЮЩИМИ СЕТКАМИ (ЭЛЕКТРИКА ↔ МЕХАНИКА)
# ═══════════════════════════════════════════════════════════════════════
#
#  Электрика и механика решаются на РАЗНЫХ сетках одной и той же
#  физической области (обе — в референсной конфигурации, недеформированной,
#  поэтому отображение между сетками не меняется со временем и строится
#  один раз при старте). Единственное поле, которое нужно передавать
#  каждый механический шаг — T_act (DG0, электрика) → T_act (DG0, механика).
#
#  Метод: ближайший центр ячейки (nearest-cell), т.к. T_act кусочно-
#  постоянно на DG0 — интерполяция более высокого порядка тут не имеет
#  смысла. Реализация MPI-простая: каждый механический шаг собирает
#  T_act целиком через comm.allgather и раздаёт всем рангам — при
#  текущем масштабе (~1-5 тыс. узлов) это стоит копейки; при переходе
#  к действительно крупным параллельным расчётам стоит заменить на
#  dolfinx.fem.create_nonmatching_meshes_interpolation_data — но точная
#  сигнатура этого API плавает между версиями DOLFINx, поэтому не
#  подставляю её вслепую без проверки на вашей установке 0.10.
# ═══════════════════════════════════════════════════════════════════════

class FieldTransferDG0:
    """
    Перенос кусочно-постоянного DG0-поля с сетки-источника на сетку-
    приёмник по ближайшему центру ячейки. Строится один раз (обе сетки
    статичны в референсной конфигурации), после чего apply() дёшев.
    """

    def __init__(self, DG0_src, DG0_dst):
        from scipy.spatial import cKDTree

        n_src = DG0_src.dofmap.index_map.size_local
        n_dst = DG0_dst.dofmap.index_map.size_local
        coords_src_local = _dof_coords(DG0_src)[:n_src, :2]
        coords_dst_local = _dof_coords(DG0_dst)[:n_dst, :2]

        # Фиксированный порядок: [ранк0, ранк1, ...] — используется
        # одинаково при построении карты и при каждом apply().
        gathered = comm.allgather(np.ascontiguousarray(coords_src_local))
        self._src_sizes = [len(c) for c in gathered]
        coords_src_global = np.vstack(gathered)

        tree = cKDTree(coords_src_global)
        dist, idx = tree.query(coords_dst_local)
        self._idx = idx
        self.n_dst_local = n_dst

        # Диагностика: насколько велик шаг переноса относительно
        # шага целевой сетки — большие значения означают, что сетки
        # сильно рассинхронизированы (например, разное Lx/Ly).
        self.max_dist = float(dist.max()) if len(dist) else 0.0

    def apply(self, values_owned_src: np.ndarray) -> np.ndarray:
        """
        values_owned_src — значения на владеемых DOF источника ЭТОГО ранга
        (ровно size_local, как в _gather_owned). Возвращает значения на
        владеемых DOF приёмника этого же ранга.
        """
        n_expect = self._src_sizes[rank]
        v = np.ascontiguousarray(values_owned_src[:n_expect], dtype=np.float64)
        gathered = comm.allgather(v)
        values_global = np.concatenate(gathered)
        return values_global[self._idx]


# ═══════════════════════════════════════════════════════════════════════
#  UFL: КИНЕМАТИКА И ЭНЕРГИЯ
# ═══════════════════════════════════════════════════════════════════════

def kinematics(u, tissue: "Tissue"):
    """Вернуть (F, C, J, I1, I4). f0 берётся из tissue (может быть неоднородным)."""
    I  = Identity(len(u))
    F  = I + grad(u)
    C  = F.T * F
    J  = det(F)
    I1 = tr(C)
    f0 = tissue.fiber_vector()
    I4 = dot(f0, C * f0)
    return F, C, J, I1, I4


def P_passive(u, tissue: "Tissue"):
    """
    Первый тензор Пиолы-Кирхгофа пассивной ткани: P = F·S
    S = 2·dΨ/dC  (аналитически, без variable())

    MU, B_ISO, MU_F, B_F, KAPPA — поля на DG0 (tissue), могут различаться
    по регионам ткани (например, жёсткий фиброзный рубец).
    """
    f0 = tissue.fiber_vector()
    F, C, J, I1, I4 = kinematics(u, tissue)
    Cinv = inv(C)

    dPsi_dI1 = tissue.mu_fn   * exp(tissue.b_iso_fn * (I1 - 2))
    dPsi_dI4 = tissue.mu_f_fn * (I4 - 1) * exp(tissue.b_f_fn * (I4 - 1)**2)
    dPsi_dJ  = tissue.kappa_fn * (J - 1)

    S = 2 * (
        dPsi_dI1 * Identity(2)
      + dPsi_dI4 * outer(f0, f0)
      + dPsi_dJ  * J / 2 * Cinv
    )
    return F * S   # P = F·S  (первый тензор Пиолы-Кирхгофа)


def S_active(u, T_act_fn, tissue: "Tissue"):
    """
    Второй тензор Пиолы-Кирхгофа активного напряжения (без inv(F)):
      S_act = J·F⁻¹·σ_act·F⁻ᵀ = T_act/I4 · f0⊗f0
    """
    f0   = tissue.fiber_vector()
    _, _, _, _, I4 = kinematics(u, tissue)
    return (T_act_fn / I4) * outer(f0, f0)


# ═══════════════════════════════════════════════════════════════════════
#  МЕХАНИКА
# ═══════════════════════════════════════════════════════════════════════

class Mechanics:
    """
    Нелинейная квазистатика:
      find u: δΨ/δu·[v] + ∫ P_act:∇v dΩ = 0  ∀v
    Решатель: PETSc SNES (v0.10 — новый NonlinearProblem API).
    """

    def __init__(self, mesh, tissue: "Tissue"):
        self.mesh = mesh
        self.tissue = tissue
        V = fem.functionspace(mesh, ("Lagrange", 1, (2,)))
        self.V  = V
        self.u  = fem.Function(V, name="u")
        self.du = TrialFunction(V)
        self.v  = TestFunction(V)

        # Переиспользуем DG0-пространство tissue: гарантирует поэлементное
        # совпадение dof между T_act и полями параметров (T_max, MU, ...).
        DG0 = tissue.DG0
        self.DG0   = DG0
        self.T_act = fem.Function(DG0, name="T_act_kPa")

        self.bcs: list[fem.DirichletBC] = []
        self._bcs_changed = True
        self._build_forms()

    def _build_forms(self):
        u, v, du = self.u, self.v, self.du

        P_pass = P_passive(u, self.tissue)
        F_ufl  = Identity(2) + grad(u)

        S_act = S_active(u, self.T_act, self.tissue)
        P_act = F_ufl * S_act   # P = F·S

        self.F_form = inner(P_pass + P_act, grad(v)) * dx
        self.J_form = derivative(self.F_form, u, du)

    def update_T_act(self, arr: np.ndarray):
        n = self.DG0.dofmap.index_map.size_local
        self.T_act.x.array[:n] = arr[:n]
        self.T_act.x.scatter_forward()

    def _make_problem(self):
        """Создать NonlinearProblem с текущими BC (этапы 1→2→3)."""
        petsc_opts = {
            "snes_type":            "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol":            1e-6,
            "snes_atol":            1e-8,
            "snes_stol":            1e-10,
            "snes_max_it":          50,
            "ksp_type":             "preonly",
            "pc_type":              "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        self._problem = fem_petsc.NonlinearProblem(
            self.F_form, self.u,
            bcs=self.bcs,
            J=self.J_form,
            petsc_options=petsc_opts,
            petsc_options_prefix="mech_",
        )

    def solve(self) -> tuple[int, bool]:
        """Решить с текущими T_act и BC."""
        if not hasattr(self, "_problem") or self._bcs_changed:
            self._make_problem()
            self._bcs_changed = False

        self._problem.solve()
        snes = self._problem.solver
        n_it = snes.getIterationNumber()
        ok   = snes.getConvergedReason() > 0
        if not ok and rank == 0:
            print(f"  [WARN] SNES reason={snes.getConvergedReason()}")
        return n_it, ok

    # ── Постобработка ─────────────────────────────────────────────────

    def lambda_f_dg0(self) -> fem.Function:
        f0 = self.tissue.fiber_vector()
        F_ufl = Identity(2) + grad(self.u)
        C_ufl = F_ufl.T * F_ufl
        expr  = fem.Expression(sqrt(dot(f0, C_ufl * f0)),
                                self.DG0.element.interpolation_points)
        fn = fem.Function(self.DG0)
        fn.interpolate(expr)
        return fn

    def sigma_xx_dg0(self) -> fem.Function:
        """Полное σ_xx = σ_pass_xx + σ_act_xx на DG0."""
        t = self.tissue
        f0 = t.fiber_vector()
        F_ufl = Identity(2) + grad(self.u)
        C_ufl = F_ufl.T * F_ufl
        J_ufl = det(F_ufl)
        I1    = tr(C_ufl)
        I4    = dot(f0, C_ufl * f0)
        Cinv  = inv(C_ufl)

        S = 2 * (
            t.mu_fn   * exp(t.b_iso_fn * (I1 - 2))         * Identity(2)
          + t.mu_f_fn * (I4-1) * exp(t.b_f_fn*(I4-1)**2)   * outer(f0, f0)
          + t.kappa_fn * (J_ufl-1) * J_ufl/2               * Cinv
        )
        sigma_pass = (1/J_ufl) * F_ufl * S * F_ufl.T

        lam_f = sqrt(I4)
        f_cur = (F_ufl * f0) / lam_f
        sigma_act  = (self.T_act / J_ufl) * outer(f_cur, f_cur)

        expr = fem.Expression((sigma_pass + sigma_act)[0, 0],
                               self.DG0.element.interpolation_points)
        fn = fem.Function(self.DG0)
        fn.interpolate(expr)
        return fn


# ═══════════════════════════════════════════════════════════════════════
#  ГРАНИЧНЫЕ УСЛОВИЯ
# ═══════════════════════════════════════════════════════════════════════

def bcs_stretch(mech: Mechanics, dL: float) -> list[fem.DirichletBC]:
    """
    Этап 1: чистое одноосное растяжение вдоль x.

      x = 0  : u_x = 0    (все узлы левой грани, только компонент x)
      x = Lx : u_x = ΔL   (все узлы правой грани, только компонент x)

    u_y свободен на всех гранях; трансляция по y подавлена одним
    точечным BC: u_y = 0 в узле (x=0, y=Ly/2).
    """
    V, mesh = mech.V, mech.mesh
    tol  = 1e-10
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    V0, V1 = V.sub(0), V.sub(1)

    zero = fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    dL_c = fem.Constant(mesh, dolfinx.default_scalar_type(dL))

    f_left  = dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[0], 0.0, atol=tol))
    f_right = dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[0], Lx,  atol=tol))

    bc_left  = fem.dirichletbc(
        zero, fem.locate_dofs_topological(V0, fdim, f_left),  V0)
    bc_right = fem.dirichletbc(
        dL_c, fem.locate_dofs_topological(V0, fdim, f_right), V0)

    V1c, dmap1 = V1.collapse()
    u1_zero = fem.Function(V1c)   # нулевая функция
    dofs_pt = fem.locate_dofs_geometrical(
        (V1, V1c),
        lambda x: (np.isclose(x[0], 0.0,    atol=tol) &
                   np.isclose(x[1], Ly/2.0,  atol=Ly/Ny_m)))
    bc_pt = fem.dirichletbc(u1_zero, dofs_pt, V1)

    return [bc_left, bc_right, bc_pt]


def bcs_fixed(mech: Mechanics) -> list[fem.DirichletBC]:
    """
    Этап 2+3: все 4 грани зафиксированы в текущем (растянутом) положении.

    При рестарте это же условие восстанавливает зажим по загруженному u:
    граничные узлы на этапе 3 не двигались, поэтому их значения в
    чекпоинте в точности равны растянутому состоянию.
    """
    V, mesh = mech.V, mech.mesh
    tol  = 1e-10
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)

    def on_any_boundary(x):
        return (np.isclose(x[0], 0.0, atol=tol) | np.isclose(x[0], Lx, atol=tol) |
                np.isclose(x[1], 0.0, atol=tol) | np.isclose(x[1], Ly, atol=tol))

    facets = dmesh.locate_entities_boundary(mesh, fdim, on_any_boundary)

    u_bc = fem.Function(V)
    u_bc.x.array[:] = mech.u.x.array[:]

    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return [fem.dirichletbc(u_bc, dofs)]


# ═══════════════════════════════════════════════════════════════════════
#  ЭЛЕКТРИКА: АЛИЕВ–ПАНФИЛОВ / ROGERS–McCULLOCH (Strang splitting)
# ═══════════════════════════════════════════════════════════════════════

class AlievPanfilov:
    def __init__(self, mesh, tissue: "Tissue"):
        self.mesh = mesh
        self.tissue = tissue
        # Переиспользуем P1-пространство tissue: гарантирует поэлементное
        # совпадение dof между u_ap/v_ap и массивами параметров реакции.
        P1 = tissue.P1
        self.P1    = P1
        self.u_ap  = fem.Function(P1, name="u_ap")
        self.v_ap  = fem.Function(P1, name="v_ap")
        self.u_tmp = fem.Function(P1)

        self._build_matrices()
        self._dt: float | None = None
        self.A: PETSc.Mat | None = None
        self.ksp: PETSc.KSP | None = None

    def _build_matrices(self):
        P1  = self.P1
        u_t = TrialFunction(P1)
        phi = TestFunction(P1)

        # Проводимость и направление волокна — поля на DG0 (tissue),
        # могут быть неоднородными (например, снижены в зоне ишемии).
        f0 = self.tissue.fiber_vector()
        s0 = self.tissue.sheet_vector()
        D  = self.tissue.D_long_fn * outer(f0, f0) + self.tissue.D_trans_fn * outer(s0, s0)

        self.M = fem_petsc.assemble_matrix(fem.form(inner(u_t, phi) * dx))
        self.M.assemble()
        self.K = fem_petsc.assemble_matrix(
            fem.form(inner(dot(D, grad(u_t)), grad(phi)) * dx))
        self.K.assemble()

    def _setup_ksp(self, dt: float, theta: float = 0.5):
        if self._dt == dt:
            return
        self._dt = dt
        self.A = self.M.copy()
        self.A.axpy(theta * dt, self.K)
        self.A.assemble()
        self.ksp = PETSc.KSP().create(self.mesh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType(PETSc.KSP.Type.CG)
        self.ksp.getPC().setType(PETSc.PC.Type.HYPRE)
        self.ksp.setTolerances(rtol=1e-10, atol=1e-12)
        self.ksp.setFromOptions()

    def _reaction(self, u, v, dt_h, stim):
        """
        Явный полушаг реакции — Rogers-McCulloch 1994 (в мс).
        du/dt = c1·u·(u−a)·(1−u) − c2·u·v + I_stim
        dv/dt = b·(u − d·v)
        Параметры c1,c2,a,b,d — numpy-массивы на P1 (tissue), могут
        различаться по регионам ткани.
        """
        t = self.tissue
        du = t.ap_c1_arr * u * (u - t.ap_a_arr) * (1.0 - u) - t.ap_c2_arr * u * v + stim
        dv = t.ap_b_arr  * (u - t.ap_d_arr * v)
        return np.clip(u + dt_h * du, 0.0, 1.1), v + dt_h * dv

    def _active_stim_t0(self, t: float) -> float | None:
        """
        Найти момент начала стимула из STIM_TIMES, в чьё окно
        [t0, t0+STIM_DUR] попадает t. Стимулы должны быть разнесены
        минимум на STIM_DUR, иначе окна перекрываются — используется
        первое совпадение.
        """
        for t0 in STIM_TIMES:
            if t0 <= t <= t0 + STIM_DUR:
                return t0
        return None

    def _make_stim_array(self, t: float) -> np.ndarray:
        """
        Стимуляция левого края с плавным запуском.
        Прямой доступ к координатам DOF (без Expression).
        Активируется на каждый момент из списка STIM_TIMES.
        """
        t0 = self._active_stim_t0(t)
        if t0 is None:
            return np.zeros(self.P1.dofmap.index_map.size_local)

        t_rel = t - t0
        rise  = STIM_RISE_TIME

        if t_rel < rise:
            envelope = t_rel / rise
        elif t_rel > STIM_DUR - rise:
            envelope = (STIM_DUR - t_rel) / rise
        else:
            envelope = 1.0

        amp = STIM_AMP * envelope

        dof_coords = self.P1.tabulate_dof_coordinates()

        decay_length = 0.5  # мм — характерная длина спада
        stim_vals = amp * np.exp(-dof_coords[:, 0] / decay_length)
        stim_vals[dof_coords[:, 0] > 2.0] = 0.0

        return stim_vals

    def step(self, t: float, dt: float, theta: float = 0.5):
        self._setup_ksp(dt, theta)

        stim = self._make_stim_array(t)

        u = self.u_ap.x.array.copy()
        v = self.v_ap.x.array.copy()

        # 1. Реакция dt/2
        u, v = self._reaction(u, v, dt / 2, stim)
        self.u_tmp.x.array[:] = u
        self.u_tmp.x.scatter_forward()

        # 2. Диффузия Crank–Nicolson
        rhs = self.M.createVecRight()
        self.M.mult(self.u_tmp.x.petsc_vec, rhs)
        tmp = self.K.createVecRight()
        self.K.mult(self.u_tmp.x.petsc_vec, tmp)
        rhs.axpy(-(1.0 - theta) * dt, tmp)

        sol = self.A.createVecRight()
        self.ksp.solve(rhs, sol)
        n_local = self.P1.dofmap.index_map.size_local
        self.u_tmp.x.array[:n_local] = np.clip(sol.array_r[:n_local], 0.0, 1.5)
        self.u_tmp.x.scatter_forward()

        u = self.u_tmp.x.array.copy()

        # 3. Реакция dt/2
        u, v = self._reaction(u, v, dt / 2, stim)
        self.u_ap.x.array[:] = np.clip(u, 0.0, 1.5)
        self.v_ap.x.array[:] = v
        self.u_ap.x.scatter_forward()
        self.v_ap.x.scatter_forward()

    def T_act_on_dg0(self) -> np.ndarray:
        """
        Проецируем u_ap на СВОЮ (электрическую) DG0, возвращаем
        T_act = T_max(x)·u·(1−u). Результат живёт на электрической сетке;
        для механики его нужно перенести через FieldTransferDG0.
        """
        DG0 = self.tissue.DG0
        expr = fem.Expression(self.u_ap,
                               DG0.element.interpolation_points)
        u_dg0 = fem.Function(DG0)
        u_dg0.interpolate(expr)
        u = u_dg0.x.array.copy()
        return np.clip(self.tissue.t_max_arr * u * (1.0 - u), 0.0, None)


# ═══════════════════════════════════════════════════════════════════════
#  ЧЕКПОИНТЫ: СОХРАНЕНИЕ / ЗАГРУЗКА СОСТОЯНИЯ
# ═══════════════════════════════════════════════════════════════════════
#
#  Формат .npz: для каждого поля хранятся координаты DOF (владеемых
#  данным рангом, собранные на rank 0) и соответствующие значения.
#  Такое представление инвариантно к числу MPI-рангов и к внутренней
#  нумерации DOF в DOLFINx, поэтому чекпоинт из прогона на 1 ядре
#  корректно читается прогоном на 8 ядрах и наоборот.
#
#  Состав состояния:
#    u      (V,   bs=2) — перемещение, включает предварительное растяжение
#    T_act  (DG0, bs=1) — активное напряжение (пересчитывается, но полезно
#                         для диагностики и для «горячего» старта SNES)
#    u_ap   (P1,  bs=1) — трансмембранный потенциал (безразмерный)
#    v_ap   (P1,  bs=1) — переменная восстановления
#    t, step, mech_count + метаданные сетки и параметров
# ═══════════════════════════════════════════════════════════════════════

def _gather_owned(V, arr: np.ndarray, bs: int = 1):
    """Собрать на rank 0 (координаты, значения) владеемых DOF со всех рангов."""
    n = V.dofmap.index_map.size_local
    c = np.ascontiguousarray(_dof_coords(V)[:n, :2], dtype=np.float64)
    v = np.ascontiguousarray(arr.reshape(-1, bs)[:n], dtype=np.float64)
    cs = comm.gather(c, root=0)
    vs = comm.gather(v, root=0)
    if rank == 0:
        return np.vstack(cs), np.vstack(vs)
    return None, None


def _match_coords(coords: np.ndarray, ref: np.ndarray,
                  allow_interp: bool, label: str) -> np.ndarray:
    """
    Для каждой локальной координаты найти индекс в опорном массиве.
    Точное сопоставление по округлённому ключу; при промахе — ближайший
    сосед (только если allow_interp=True).
    """
    key = {}
    for i, r in enumerate(np.round(ref[:, :2], CKPT_ROUND)):
        key[(r[0], r[1])] = i

    idx = np.empty(len(coords), dtype=np.int64)
    miss = []
    for i, c in enumerate(np.round(coords[:, :2], CKPT_ROUND)):
        j = key.get((c[0], c[1]))
        if j is None:
            miss.append(i)
            idx[i] = -1
        else:
            idx[i] = j

    if miss:
        if not allow_interp:
            raise RuntimeError(
                f"[checkpoint] поле '{label}': {len(miss)} из {len(coords)} DOF "
                f"не найдены в чекпоинте (сетка не совпадает). "
                f"Запустите с --allow-interp для переноса по ближайшему соседу.")
        if rank == 0:
            print(f"  [ckpt] '{label}': {len(miss)}/{len(coords)} DOF "
                  f"перенесены по ближайшему соседу")
        for i in miss:
            d2 = np.sum((ref[:, :2] - coords[i, :2])**2, axis=1)
            idx[i] = int(np.argmin(d2))
    return idx


def _restore(V, fn: fem.Function, ref_c: np.ndarray, ref_v: np.ndarray,
             bs: int, allow_interp: bool, label: str):
    coords = _dof_coords(V)
    idx = _match_coords(coords, ref_c, allow_interp, label)
    fn.x.array[:] = ref_v[idx].reshape(-1)
    fn.x.scatter_forward()


def save_checkpoint(path: Path, t: float, step: int, mech_count: int,
                    mech: "Mechanics", elec: "AlievPanfilov",
                    link_last: Path | None = None):
    """Сохранить полное состояние расчёта в .npz (MPI-safe)."""
    cu, vu = _gather_owned(mech.V,   mech.u.x.array,    2)
    ct, vt = _gather_owned(mech.DG0, mech.T_act.x.array, 1)
    ce, ve = _gather_owned(elec.P1,  elec.u_ap.x.array,  1)
    _,  vw = _gather_owned(elec.P1,  elec.v_ap.x.array,  1)

    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            # состояние
            coords_u=cu,    u=vu,
            coords_dg0=ct,  T_act=vt,
            coords_p1=ce,   u_ap=ve, v_ap=vw,
            # время
            t=float(t), step=int(step), mech_count=int(mech_count),
            dt_e=DT_E, n_mech=N_MECH,
            # метаданные (для проверки совместимости и истории)
            Lx=Lx, Ly=Ly, Nx_e=Nx_e, Ny_e=Ny_e, Nx_m=Nx_m, Ny_m=Ny_m,
            fiber_angle=FIBER_ANGLE_DEG, stretch=STRETCH,
            T_max=T_MAX, D_long=D_LONG, D_trans=D_TRANS,
            mu=MU, b_iso=B_ISO, mu_f=MU_F, b_f=B_F, kappa=KAPPA,
            stim_times=np.asarray(STIM_TIMES, dtype=np.float64),
            stim_amp=STIM_AMP, stim_dur=STIM_DUR,
        )
        if link_last is not None:
            shutil.copyfile(path, link_last)
        print(f"  [ckpt] t = {t:.1f} мс → {path}")
    comm.Barrier()


def load_checkpoint(path: Path, mech: "Mechanics", elec: "AlievPanfilov",
                    allow_interp: bool = False) -> dict:
    """
    Загрузить состояние из чекпоинта (в т.ч. из другого эксперимента).
    Возвращает словарь метаданных: t, step, mech_count, ...
    """
    if rank == 0:
        if not path.exists():
            raise FileNotFoundError(f"чекпоинт не найден: {path}")
        d = dict(np.load(path, allow_pickle=False))
    else:
        d = None
    d = comm.bcast(d, root=0)

    # Совместимость со старыми чекпоинтами (единая сетка Nx/Ny)
    Nx_e_ckpt = int(d["Nx_e"]) if "Nx_e" in d else int(d["Nx"])
    Ny_e_ckpt = int(d["Ny_e"]) if "Ny_e" in d else int(d["Ny"])
    Nx_m_ckpt = int(d["Nx_m"]) if "Nx_m" in d else int(d["Nx"])
    Ny_m_ckpt = int(d["Ny_m"]) if "Ny_m" in d else int(d["Ny"])

    meta = dict(t=float(d["t"]), step=int(d["step"]),
                mech_count=int(d["mech_count"]),
                Nx_e=Nx_e_ckpt, Ny_e=Ny_e_ckpt,
                Nx_m=Nx_m_ckpt, Ny_m=Ny_m_ckpt,
                Lx=float(d["Lx"]), Ly=float(d["Ly"]),
                stretch=float(d["stretch"]), T_max=float(d["T_max"]),
                fiber_angle=float(d["fiber_angle"]))

    if rank == 0:
        same = (meta["Nx_e"] == Nx_e and meta["Ny_e"] == Ny_e and
                meta["Nx_m"] == Nx_m and meta["Ny_m"] == Ny_m and
                np.isclose(meta["Lx"], Lx) and np.isclose(meta["Ly"], Ly))
        print(f"\n[ Рестарт ] {path}")
        print(f"  Источник     : эл.сетка {meta['Nx_e']}×{meta['Ny_e']}, "
              f"мех.сетка {meta['Nx_m']}×{meta['Ny_m']} "
              f"на {meta['Lx']}×{meta['Ly']} мм, λ_f={meta['stretch']}, "
              f"базовый T_max={meta['T_max']} кПа, базовые волокна {meta['fiber_angle']}°")
        print(f"  Текущий      : эл.сетка {Nx_e}×{Ny_e}, "
              f"мех.сетка {Nx_m}×{Ny_m} на {Lx}×{Ly} мм, "
              f"λ_f={STRETCH}, базовый T_max={T_MAX} кПа, "
              f"базовые волокна {FIBER_ANGLE_DEG}°")
        if not same:
            print("  [WARN] геометрия/разрешение хотя бы одной сетки отличаются — "
                  f"{'перенос по ближайшему соседу' if allow_interp else 'будет ошибка'}")
        if not np.isclose(meta["fiber_angle"], FIBER_ANGLE_DEG):
            print("  [WARN] базовый угол волокон отличается: загруженное поле u "
                  "не является равновесным для текущей анизотропии")
        print("  [WARN] чекпоинт хранит только БАЗОВЫЕ (глобальные) параметры "
              "для справки — заданные в REGIONS неоднородности не сохраняются "
              "и восстанавливаются заново из текущего REGIONS/--regions-json")

    _restore(mech.V,   mech.u,     d["coords_u"],   d["u"],     2, allow_interp, "u")
    _restore(mech.DG0, mech.T_act, d["coords_dg0"], d["T_act"], 1, allow_interp, "T_act")
    _restore(elec.P1,  elec.u_ap,  d["coords_p1"],  d["u_ap"],  1, allow_interp, "u_ap")
    _restore(elec.P1,  elec.v_ap,  d["coords_p1"],  d["v_ap"],  1, allow_interp, "v_ap")

    return meta


# ═══════════════════════════════════════════════════════════════════════
#  ВЫВОД
# ═══════════════════════════════════════════════════════════════════════

class XDMFOut:
    def __init__(self, path: Path, mesh):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = io.XDMFFile(mesh.comm, str(path), "w")
        self.file.write_mesh(mesh)

    def write(self, fns, t):
        for f in fns:
            self.file.write_function(f, t)

    def close(self):
        self.file.close()


def pyvista_snap_mech(mesh_m, u_fn, T_act_fn, t, out_dir):
    """Снапшот механической сетки: перемещения + активное напряжение."""
    try:
        import pyvista as pv
        from dolfinx.plot import vtk_mesh
        topo, ct, pts = vtk_mesh(mesh_m, mesh_m.topology.dim)
        g = pv.UnstructuredGrid(topo, ct, pts)
        u2 = u_fn.x.array.reshape(-1, 2)
        g.point_data["u_mm"]  = np.pad(u2, ((0,0),(0,1)))
        g.cell_data["T_act_kPa"] = T_act_fn.x.array[:]
        out_dir.mkdir(parents=True, exist_ok=True)
        g.save(str(out_dir / f"snap_mech_t{t:.1f}ms.vtk"))
        print(f"  [PyVista] t = {t:.1f} мс → {out_dir}/snap_mech_t{t:.1f}ms.vtk")
    except Exception as e:
        print(f"  [PyVista] ошибка (механика): {e}")


def pyvista_snap_elec(mesh_e, u_ap_fn, v_ap_fn, t, out_dir):
    """Снапшот электрической сетки: потенциал + переменная восстановления."""
    try:
        import pyvista as pv
        from dolfinx.plot import vtk_mesh
        topo, ct, pts = vtk_mesh(mesh_e, mesh_e.topology.dim)
        g = pv.UnstructuredGrid(topo, ct, pts)
        g.point_data["u_ap"] = u_ap_fn.x.array[:]
        g.point_data["v_ap"] = v_ap_fn.x.array[:]
        out_dir.mkdir(parents=True, exist_ok=True)
        g.save(str(out_dir / f"snap_elec_t{t:.1f}ms.vtk"))
        print(f"  [PyVista] t = {t:.1f} мс → {out_dir}/snap_elec_t{t:.1f}ms.vtk")
    except Exception as e:
        print(f"  [PyVista] ошибка (электрика): {e}")


# ═══════════════════════════════════════════════════════════════════════
#  ОСНОВНОЙ РАСЧЁТ
# ═══════════════════════════════════════════════════════════════════════

def run(restart: Path | None = None,
        out_dir: Path = OUT_DIR,
        t_end: float = T_END,
        ckpt_every: int = CKPT_EVERY,
        keep_all_ckpt: bool = CKPT_KEEP_ALL,
        allow_interp: bool = False,
        reset_time: bool = False,
        regions: list | None = None):

    dlog.set_log_level(dlog.LogLevel.WARNING)

    if rank == 0:
        print("=" * 70)
        print("  ЭЛЕКТРОМЕХАНИКА МИОКАРДА")
        print("  DOLFINx 0.10  +  MPI  +  PETSc SNES  +  PyVista")
        print("=" * 70)
        print(f"  MPI ranks    : {comm.size}")
        print(f"  Сетка эл.    : {Nx_e}×{Ny_e}  Q1  ({Nx_e*Ny_e} эл., "
              f"{(Nx_e+1)*(Ny_e+1)} уз.)")
        print(f"  Сетка мех.   : {Nx_m}×{Ny_m}  Q1  ({Nx_m*Ny_m} эл., "
              f"{(Nx_m+1)*(Ny_m+1)} уз.)")
        print(f"  dt_e / dt_m  : {DT_E} мс / {DT_E*N_MECH:.2f} мс    T_end : {t_end} мс")
        print(f"  Базовые      : T_max={T_MAX} кПа  D_l/t={D_LONG}/{D_TRANS} мм²/мс  "
              f"волокна {FIBER_ANGLE_DEG}°")
        print(f"  Стимулы      : t = {STIM_TIMES} мс, A = {STIM_AMP}, "
              f"длит. {STIM_DUR} мс")
        print(f"  Режим        : {'РЕСТАРТ из ' + str(restart) if restart else 'расчёт с нуля'}")
        print(f"  Каталог      : {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)

    # Раздельные сетки: электрика — мелкая (разрешает фронт), механика —
    # крупная (поле перемещений гладкое, дешевле для SNES). REGIONS
    # заданы аналитически по физическим координатам, поэтому применяются
    # к каждой сетке независимо и корректно — без переноса между ними.
    mesh_e = make_mesh(Nx_e, Ny_e)
    mesh_m = make_mesh(Nx_m, Ny_m)
    tissue_e = Tissue(mesh_e, regions)
    tissue_m = Tissue(mesh_m, regions)
    if rank == 0:
        print("  Ткань (электрическая сетка):")
        print(tissue_e.summary())
        print("  Ткань (механическая сетка):")
        print(tissue_m.summary())

    mech = Mechanics(mesh_m, tissue_m)
    elec = AlievPanfilov(mesh_e, tissue_e)

    # Перенос T_act с электрической DG0 на механическую DG0 — строится
    # один раз (обе сетки статичны в референсной конфигурации).
    transfer_e2m = FieldTransferDG0(elec.tissue.DG0, mech.DG0)
    if rank == 0 and transfer_e2m.max_dist > 0.5 * min(Lx / Nx_m, Ly / Ny_m):
        print(f"  [WARN] перенос T_act эл.→мех.: макс. расстояние до "
              f"ближайшей эл.ячейки {transfer_e2m.max_dist:.3f} мм — "
              f"сравнимо с размером мех.ячейки, проверьте геометрию сеток")

    xm = XDMFOut(out_dir / "mechanics.xdmf", mesh_m)
    xe = XDMFOut(out_dir / "electrics.xdmf",  mesh_e)

    # Карты регионов — отдельно для каждой сетки (для контроля в ParaView)
    xt_e = XDMFOut(out_dir / "tissue_regions_electric.xdmf", mesh_e)
    xt_e.write([tissue_e.region_id_fn, tissue_e.T_max_fn,
                tissue_e.D_long_fn, tissue_e.D_trans_fn], 0.0)
    xt_e.close()
    xt_m = XDMFOut(out_dir / "tissue_regions_mechanical.xdmf", mesh_m)
    xt_m.write([tissue_m.region_id_fn, tissue_m.mu_fn,
                tissue_m.mu_f_fn, tissue_m.kappa_fn], 0.0)
    xt_m.close()

    DG0 = mech.DG0
    n_dg0 = DG0.dofmap.index_map.size_local + DG0.dofmap.index_map.num_ghosts
    T0 = np.zeros(n_dg0)

    ckpt_last = out_dir / "ckpt_last.npz"

    # ═══ ВЕТКА A: РЕСТАРТ ═════════════════════════════════════════════
    if restart is not None:
        meta = load_checkpoint(Path(restart), mech, elec, allow_interp)

        t_start = 0.0 if reset_time else meta["t"]
        step0   = int(round(t_start / DT_E))

        # Зажим границ по загруженному (растянутому) состоянию
        mech.bcs = bcs_fixed(mech); mech._bcs_changed = True

        # Согласовать механику с загруженной электрикой (перенос эл.→мех.)
        mech.update_T_act(transfer_e2m.apply(elec.T_act_on_dg0()))
        n_it, ok = mech.solve()
        if rank == 0:
            lam = mech.lambda_f_dg0().x.array
            uap = elec.u_ap.x.array
            print(f"  ✓ Состояние восстановлено: t_ckpt = {meta['t']:.1f} мс "
                  f"→ старт с t = {t_start:.1f} мс")
            print(f"  ✓ λ_f_c = {lam[len(lam)//2]:.4f}   "
                  f"u_ap: min={uap.min():.3f} max={uap.max():.3f}   "
                  f"Newton={n_it}  {'OK' if ok else 'FAIL'}")

    # ═══ ВЕТКА B: РАСЧЁТ С НУЛЯ ═══════════════════════════════════════
    else:
        # ── Этап 1: пассивное растяжение ──────────────────────────────
        if rank == 0:
            print(f"\n[ Этап 1 ] Пассивное растяжение → λ_f = {STRETCH}")

        dL_target = (STRETCH - 1.0) * Lx
        mech.update_T_act(T0)
        for s in range(1, N_STRETCH + 1):
            dL = dL_target * s / N_STRETCH
            mech.bcs = bcs_stretch(mech, dL); mech._bcs_changed = True
            n_it, ok = mech.solve()
            if rank == 0:
                lam = mech.lambda_f_dg0().x.array
                print(f"  Шаг {s:2d}/{N_STRETCH}  ΔL={dL:.3f}  "
                      f"λ_f_c={lam[len(lam)//2]:.4f}  Newton={n_it}  "
                      f"{'OK' if ok else 'FAIL'}")

        # ── Этап 2: фиксация ──────────────────────────────────────────
        if rank == 0:
            print(f"\n[ Этап 2 ] Фиксация всех граней")
        mech.bcs = bcs_fixed(mech); mech._bcs_changed = True
        mech.solve()
        if rank == 0:
            lam = mech.lambda_f_dg0().x.array
            print(f"  ✓ λ_f_c = {lam[len(lam)//2]:.4f}")

        t_start, step0 = 0.0, 0

        # Чекпоинт «преднагруженного» состояния — удобная точка старта
        # для серии экспериментов без повторного растяжения.
        if ckpt_every:
            save_checkpoint(out_dir / "ckpt_preloaded.npz",
                            0.0, 0, 0, mech, elec)

    # ═══ Этап 3: активация ════════════════════════════════════════════
    if rank == 0:
        print(f"\n[ Этап 3 ] Активация  {t_start:.1f} → {t_end} мс\n")
        hdr = f"  {'t,мс':>7}  {'u_ap_c':>7}  {'T_act':>9}  {'σ_xx':>9}  {'λ_f':>7}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

    n_steps    = int(t_end / DT_E) + 1
    mech_count = step0 // N_MECH
    snaps_done: set[float] = {ts for ts in SNAP_TIMES if ts < t_start - 1e-9}

    for step in range(step0, n_steps):
        t = round(step * DT_E, 6)

        # 1. Электрика
        elec.step(t, DT_E)

        # 2. Механика
        if step % N_MECH == 0:
            mech_count += 1
            T_arr_e = elec.T_act_on_dg0()             # на электрической сетке
            T_arr_m = transfer_e2m.apply(T_arr_e)      # перенос на мех. сетку
            mech.update_T_act(T_arr_m)
            mech.bcs = bcs_fixed(mech); mech._bcs_changed = True
            mech.solve()

            # XDMF
            if mech_count % SAVE_EVERY == 0:
                xm.write([mech.u, mech.T_act], t)
                xe.write([elec.u_ap, elec.v_ap], t)

            # Чекпоинт
            if ckpt_every and mech_count % ckpt_every == 0:
                name = (out_dir / f"ckpt_t{t:09.2f}ms.npz" if keep_all_ckpt
                        else ckpt_last)
                save_checkpoint(name, t, step, mech_count, mech, elec,
                                link_last=ckpt_last if keep_all_ckpt else None)

            # Прогресс (индексы «центра» — отдельные для каждой сетки,
            # т.к. электрическая и механическая сетки теперь разного размера)
            if rank == 0 and step % (N_MECH * 20) == 0:
                n_c_e = len(T_arr_e) // 2
                n_c_m = len(T_arr_m) // 2
                u_c   = float(elec.u_ap.x.array[n_c_e])
                T_c   = float(mech.T_act.x.array[n_c_m])
                lam   = mech.lambda_f_dg0().x.array
                sxx   = mech.sigma_xx_dg0().x.array
                print(f"  {t:7.1f}  {u_c:7.3f}  {T_c:9.2f}  "
                      f"{sxx[n_c_m]:9.3f}  {lam[n_c_m]:7.4f}")

        # 3. Снапшоты (отдельные файлы для эл. и мех. сеток)
        for ts in SNAP_TIMES:
            if abs(t - ts) < DT_E * 0.55 and ts not in snaps_done:
                snaps_done.add(ts)
                if rank == 0:
                    pyvista_snap_mech(mesh_m, mech.u, mech.T_act, t, out_dir)
                    pyvista_snap_elec(mesh_e, elec.u_ap, elec.v_ap, t, out_dir)

    # Финальный вывод
    xm.write([mech.u, mech.T_act], t_end)
    xe.write([elec.u_ap, elec.v_ap], t_end)
    xm.close()
    xe.close()

    if ckpt_every:
        save_checkpoint(out_dir / "ckpt_final.npz", t_end, n_steps - 1,
                        mech_count, mech, elec, link_last=ckpt_last)

    if rank == 0:
        print(f"\n  ✓ XDMF → {out_dir}/mechanics.xdmf  /  electrics.xdmf")
        print(f"  ✓ VTK  → {out_dir}/snap_mech_*.vtk  /  snap_elec_*.vtk")
        if ckpt_every:
            print(f"  ✓ CKPT → {out_dir}/ckpt_final.npz  (и ckpt_last.npz)")
            print(f"    продолжить:  --restart {out_dir}/ckpt_last.npz "
                  f"--out <новый_каталог> --t-end <больше {t_end}>")
        print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Электромеханика миокарда (DOLFINx) с поддержкой рестарта",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--restart", type=Path, default=None, metavar="FILE.npz",
                   help="продолжить расчёт с чекпоинта (в т.ч. из другого прогона); "
                        "этапы растяжения и фиксации пропускаются")
    p.add_argument("--out", type=Path, default=OUT_DIR,
                   help="каталог результатов")
    p.add_argument("--t-end", type=float, default=T_END,
                   help="конечное время, мс (абсолютное)")
    p.add_argument("--ckpt-every", type=int, default=CKPT_EVERY,
                   help="мех.шагов между чекпоинтами (0 — не сохранять)")
    p.add_argument("--keep-all-ckpt", action="store_true",
                   help="хранить все чекпоинты, а не только ckpt_last.npz")
    p.add_argument("--allow-interp", action="store_true",
                   help="разрешить перенос состояния на другую сетку "
                        "(ближайший сосед)")
    p.add_argument("--reset-time", action="store_true",
                   help="начать отсчёт времени с 0, сохранив поля "
                        "(удобно для нового протокола стимуляции)")
    p.add_argument("--stim-times", type=str, default=None, metavar="T1,T2,...",
                   help="список моментов стимуляции через запятую, мс "
                        '(например "0,1000,2000"); переопределяет STIM_TIMES')
    p.add_argument("--stim-t0", type=float, default=None,
                   help="[устарел, используйте --stim-times] один момент "
                        "старта стимула, мс")
    p.add_argument("--t-max", type=float, default=None,
                   help="переопределить T_MAX, кПа (для серии экспериментов)")
    p.add_argument("--regions-json", type=Path, default=None, metavar="FILE.json",
                   help="загрузить неоднородные области ткани (REGIONS) из "
                        "JSON-файла вместо заданных в коде (см. docstring "
                        "модуля за формат); переопределяет REGIONS")
    p.add_argument("--nx-e", type=int, default=None,
                   help="переопределить Nx_e (число элементов эл. сетки по x)")
    p.add_argument("--ny-e", type=int, default=None,
                   help="переопределить Ny_e (число элементов эл. сетки по y)")
    p.add_argument("--nx-m", type=int, default=None,
                   help="переопределить Nx_m (число элементов мех. сетки по x)")
    p.add_argument("--ny-m", type=int, default=None,
                   help="переопределить Ny_m (число элементов мех. сетки по y)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Переопределения параметров для серий экспериментов на общем чекпоинте
    if args.stim_times is not None:
        STIM_TIMES = [float(x) for x in args.stim_times.split(",") if x.strip() != ""]
    elif args.stim_t0 is not None:
        STIM_TIMES = [args.stim_t0]
    if args.t_max is not None:
        T_MAX = args.t_max
    if args.nx_e is not None:
        Nx_e = args.nx_e
    if args.ny_e is not None:
        Ny_e = args.ny_e
    if args.nx_m is not None:
        Nx_m = args.nx_m
    if args.ny_m is not None:
        Ny_m = args.ny_m

    regions = load_regions_json(args.regions_json) if args.regions_json else None

    run(restart=args.restart,
        out_dir=args.out,
        t_end=args.t_end,
        ckpt_every=args.ckpt_every,
        keep_all_ckpt=args.keep_all_ckpt,
        allow_interp=args.allow_interp,
        regions=regions,
        reset_time=args.reset_time)
