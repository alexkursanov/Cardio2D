"""
Тесты полей параметров ткани (Шаг 3).
======================================

ТРЕБУЮТ DOLFINx.

    pytest tests/test_tissue.py -v
    mpirun -n 4 python -m pytest tests/test_tissue.py -v

Главные проверяемые свойства:

  * значения внутри региона равны переопределению, снаружи — базовым;
  * при пересечении регионов выигрывает ПОСЛЕДНИЙ в списке;
  * одни и те же регионы на двух разных сетках дают согласованную
    картину — это основа того, что параметры не нужно переносить между
    сетками;
  * регион, не покрывший ни одной ячейки, ОБНАРУЖИВАЕТСЯ. Это новый
    режим отказа, появившийся вместе со второй, более грубой сеткой:
    рубец может быть электрически, но отсутствовать механически.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.config import (  # noqa: E402
    CircleRegion,
    ConductionParams,
    PassiveMechParams,
    RectRegion,
    RectangleMeshSpec,
    TissueBaseParams,
)

pytestmark = pytest.mark.fem


def _owned_dg0(tissue, key):
    """Значения параметра на владеемых ячейках (без гало)."""
    n = tissue.DG0.dofmap.index_map.size_local
    return tissue.dg0_array(key)[:n]


def _owned_xy(space):
    from cardiac_em.fem import owned_dof_coordinates
    return owned_dof_coordinates(space)


# ═══════════════════════════════════════════════════════════════════════
#  ОДНОРОДНАЯ ТКАНЬ
# ═══════════════════════════════════════════════════════════════════════

def test_uniform_tissue_equals_base_everywhere():
    from cardiac_em.fem import Tissue, build_mesh

    base = TissueBaseParams()
    mesh = build_mesh(RectangleMeshSpec(nx=8, ny=8))
    tissue = Tissue.for_electrics(mesh, base)

    flat = base.to_flat_dict()
    for key in ("D_LONG", "D_TRANS", "T_MAX"):
        vals = _owned_dg0(tissue, key)
        np.testing.assert_allclose(vals, flat[key])
        assert tissue.is_uniform(key)

    for key in ("AP_c1", "AP_a"):
        n = tissue.P1.dofmap.index_map.size_local
        np.testing.assert_allclose(tissue.p1_array(key)[:n], flat[key])


def test_base_params_reach_fields():
    """Нестандартные базовые значения должны доезжать до полей."""
    from cardiac_em.fem import Tissue, build_mesh

    base = TissueBaseParams(
        conduction=ConductionParams(d_long=0.7, d_trans=0.07,
                                    fiber_angle_deg=30.0),
        passive=PassiveMechParams(mu=2.5, kappa=250.0))
    mesh = build_mesh(RectangleMeshSpec(nx=4, ny=4))

    te = Tissue.for_electrics(mesh, base)
    np.testing.assert_allclose(_owned_dg0(te, "D_LONG"), 0.7)
    np.testing.assert_allclose(_owned_dg0(te, "FIBER_ANGLE_DEG"), 30.0)

    tm = Tissue.for_mechanics(mesh, base)
    np.testing.assert_allclose(_owned_dg0(tm, "MU"), 2.5)
    np.testing.assert_allclose(_owned_dg0(tm, "KAPPA"), 250.0)


# ═══════════════════════════════════════════════════════════════════════
#  РЕГИОНЫ
# ═══════════════════════════════════════════════════════════════════════

def test_rect_region_overrides_inside_only():
    from cardiac_em.fem import Tissue, build_mesh

    base = TissueBaseParams()
    mesh = build_mesh(RectangleMeshSpec(nx=20, ny=20, lx_mm=10.0, ly_mm=10.0))
    region = RectRegion(x0=5.0, x1=10.0, y0=0.0, y1=10.0, name="right",
                        overrides={"D_LONG": 0.05, "T_MAX": 20.0})
    tissue = Tissue.for_electrics(mesh, base, (region,))

    xy = _owned_xy(tissue.DG0)
    inside = xy[:, 0] > 5.0
    d_long = _owned_dg0(tissue, "D_LONG")
    t_max = _owned_dg0(tissue, "T_MAX")

    np.testing.assert_allclose(d_long[inside], 0.05)
    np.testing.assert_allclose(d_long[~inside], 0.3)
    np.testing.assert_allclose(t_max[inside], 20.0)
    np.testing.assert_allclose(t_max[~inside], 120.0)


def test_unoverridden_params_keep_base_inside_region():
    """Регион меняет только перечисленное, остальное остаётся базовым."""
    from cardiac_em.fem import Tissue, build_mesh

    mesh = build_mesh(RectangleMeshSpec(nx=10, ny=10))
    region = RectRegion(x0=0.0, x1=10.0, y0=0.0, y1=10.0, name="everything",
                        overrides={"D_LONG": 0.01})
    tissue = Tissue.for_electrics(mesh, TissueBaseParams(), (region,))

    np.testing.assert_allclose(_owned_dg0(tissue, "D_LONG"), 0.01)
    np.testing.assert_allclose(_owned_dg0(tissue, "D_TRANS"), 0.03)  # база
    np.testing.assert_allclose(_owned_dg0(tissue, "T_MAX"), 120.0)   # база


def test_later_region_wins_on_overlap():
    from cardiac_em.fem import Tissue, build_mesh

    mesh = build_mesh(RectangleMeshSpec(nx=20, ny=20, lx_mm=10.0, ly_mm=10.0))
    ischemia = RectRegion(x0=5.0, x1=10.0, y0=0.0, y1=10.0, name="ischemia",
                          overrides={"T_MAX": 20.0})
    scar = CircleRegion(cx=7.5, cy=5.0, r=1.0, name="scar",
                        overrides={"T_MAX": 0.0})
    tissue = Tissue.for_electrics(mesh, TissueBaseParams(), (ischemia, scar))

    xy = _owned_xy(tissue.DG0)
    t_max = _owned_dg0(tissue, "T_MAX")

    in_scar = (xy[:, 0] - 7.5) ** 2 + (xy[:, 1] - 5.0) ** 2 <= 1.0
    in_isch_only = (xy[:, 0] > 5.0) & ~in_scar
    healthy = xy[:, 0] < 5.0

    np.testing.assert_allclose(t_max[in_scar], 0.0)
    np.testing.assert_allclose(t_max[in_isch_only], 20.0)
    np.testing.assert_allclose(t_max[healthy], 120.0)


def test_region_map_matches_geometry():
    from cardiac_em.fem import Tissue, build_mesh

    mesh = build_mesh(RectangleMeshSpec(nx=16, ny=16, lx_mm=8.0, ly_mm=8.0))
    r0 = RectRegion(x0=4.0, x1=8.0, y0=0.0, y1=8.0, name="a",
                    overrides={"T_MAX": 10.0})
    r1 = CircleRegion(cx=6.0, cy=4.0, r=1.0, name="b",
                      overrides={"T_MAX": 0.0})
    tissue = Tissue.for_electrics(mesh, TissueBaseParams(), (r0, r1))

    n = tissue.DG0.dofmap.index_map.size_local
    rid = tissue.region_id_dg0[:n]
    xy = _owned_xy(tissue.DG0)

    in_b = (xy[:, 0] - 6.0) ** 2 + (xy[:, 1] - 4.0) ** 2 <= 1.0
    in_a_only = (xy[:, 0] > 4.0) & ~in_b

    assert np.all(rid[in_b] == 1)
    assert np.all(rid[in_a_only] == 0)
    assert np.all(rid[xy[:, 0] < 4.0] == -1)


def test_cells_per_region_counts():
    from mpi4py import MPI

    from cardiac_em.fem import Tissue, build_mesh

    mesh = build_mesh(RectangleMeshSpec(nx=10, ny=10, lx_mm=10.0, ly_mm=10.0))
    region = RectRegion(x0=5.0, x1=10.0, y0=0.0, y1=10.0, name="half",
                        overrides={"T_MAX": 0.0})
    tissue = Tissue.for_electrics(mesh, TissueBaseParams(), (region,))

    counts = tissue.cells_per_region()
    total_region = mesh.comm.allreduce(counts[0], op=MPI.SUM)
    total_base = mesh.comm.allreduce(counts[-1], op=MPI.SUM)

    assert total_region == 50   # ровно половина ячеек
    assert total_base == 50
    assert total_region + total_base == 100


# ═══════════════════════════════════════════════════════════════════════
#  ДВЕ СЕТКИ
# ═══════════════════════════════════════════════════════════════════════

def test_same_region_consistent_on_both_meshes():
    """
    Один и тот же регион, раскрашенный независимо на двух сетках, должен
    покрывать одинаковую ДОЛЮ площади. Это и есть причина, по которой
    параметры не нужно переносить между сетками.
    """
    from mpi4py import MPI

    from cardiac_em.config import DualMeshConfig
    from cardiac_em.fem import Tissue, build_mesh_pair

    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=20, ny=20, lx_mm=10.0, ly_mm=10.0), coarsening=2)
    pair = build_mesh_pair(cfg)

    # граница по x = 5.0 лежит на грани ячеек обеих сеток
    region = RectRegion(x0=5.0, x1=10.0, y0=0.0, y1=10.0, name="right",
                        overrides={"T_MAX": 0.0, "MU": 9.0})

    te = Tissue.for_electrics(pair.electric, TissueBaseParams(), (region,))
    tm = Tissue.for_mechanics(pair.mechanical, TissueBaseParams(), (region,))

    frac_e = mesh_fraction(te, MPI)
    frac_m = mesh_fraction(tm, MPI)
    assert math.isclose(frac_e, 0.5, abs_tol=1e-12)
    assert math.isclose(frac_m, 0.5, abs_tol=1e-12)


def mesh_fraction(tissue, MPI) -> float:
    """Доля владеемых ячеек, попавших хоть в какой-то регион."""
    n = tissue.DG0.dofmap.index_map.size_local
    rid = tissue.region_id_dg0[:n]
    in_region = tissue.mesh.comm.allreduce(int((rid >= 0).sum()), op=MPI.SUM)
    total = tissue.DG0.dofmap.index_map.size_global
    return in_region / total


def test_electric_and_mechanical_build_different_keys():
    """
    Электрической сетке не нужны модули упругости, механической —
    проводимость. Запрос непостроенного параметра должен давать
    понятную ошибку, а не KeyError из недр словаря.
    """
    from cardiac_em.fem import Tissue, build_mesh

    mesh = build_mesh(RectangleMeshSpec(nx=4, ny=4))
    te = Tissue.for_electrics(mesh, TissueBaseParams())
    tm = Tissue.for_mechanics(mesh, TissueBaseParams())

    te.dg0("D_LONG")          # есть у электрики
    tm.dg0("MU")               # есть у механики

    with pytest.raises(KeyError, match="не построен"):
        te.dg0("MU")
    with pytest.raises(KeyError, match="не построен"):
        tm.dg0("D_LONG")

    # угол волокон нужен обеим
    te.dg0("FIBER_ANGLE_DEG")
    tm.dg0("FIBER_ANGLE_DEG")


def test_p1_and_dg0_access_are_not_confused():
    """Поточечные параметры живут на P1, полевые — на DG0; путаница явная."""
    from cardiac_em.fem import Tissue, build_mesh

    mesh = build_mesh(RectangleMeshSpec(nx=4, ny=4))
    tissue = Tissue.for_electrics(mesh, TissueBaseParams())

    assert tissue.p1_array("AP_c1").size >= tissue.P1.dofmap.index_map.size_local
    with pytest.raises(KeyError, match="не живёт на DG0"):
        tissue.dg0("AP_c1")
    with pytest.raises(KeyError, match="не живёт на P1"):
        tissue.p1_array("D_LONG")


def test_unknown_key_rejected_at_construction():
    from cardiac_em.fem import Tissue, build_mesh

    mesh = build_mesh(RectangleMeshSpec(nx=4, ny=4))
    with pytest.raises(ValueError, match="неизвестные параметры"):
        Tissue(mesh, TissueBaseParams(), keys=("D_LONG", "НЕТ_ТАКОГО"))


def test_every_parameter_has_a_space():
    """
    Каждый параметр обязан жить хоть в каком-то пространстве. Если
    добавить новый в TissueBaseParams и забыть приписать его к P1 или
    DG0, он молча не построится — этот тест такую забывчивость ловит.
    """
    from cardiac_em.fem.tissue import _DG0_KEYS, _P1_KEYS

    all_keys = set(TissueBaseParams.flat_keys())
    assigned = set(_P1_KEYS) | set(_DG0_KEYS)
    assert assigned == all_keys, (
        f"без пространства остались: {sorted(all_keys - assigned)}; "
        f"лишние приписки: {sorted(assigned - all_keys)}")


# ═══════════════════════════════════════════════════════════════════════
#  МЕЛКИЕ РЕГИОНЫ НА КРУПНОЙ СЕТКЕ
# ═══════════════════════════════════════════════════════════════════════

def test_tiny_region_missing_on_coarse_mesh_is_detected():
    """
    Ключевой новый режим отказа при двух сетках: регион покрывает ячейки
    на мелкой сетке и НИ ОДНОЙ на крупной. Электрически рубец есть,
    механически его нет — и это не видно, пока не начнёшь искать.
    """
    from cardiac_em.config import DualMeshConfig
    from cardiac_em.fem import Tissue, build_mesh_pair

    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=40, ny=40, lx_mm=10.0, ly_mm=10.0), coarsening=8)
    pair = build_mesh_pair(cfg)      # эл. h = 0.25 мм, мех. h = 2.0 мм

    # Центр (4, 4) выбран так, чтобы попасть МЕЖДУ центрами крупных ячеек:
    # они стоят в 1, 3, 5, 7, 9 мм, ближайший — в 1 мм от центра региона,
    # что больше радиуса 0.3 мм. На мелкой сетке (центры через 0.25 мм)
    # регион при этом покрывает 4 ячейки.
    tiny = CircleRegion(cx=4.0, cy=4.0, r=0.3, name="tiny_scar",
                        overrides={"T_MAX": 0.0, "MU": 50.0})

    te = Tissue.for_electrics(pair.electric, TissueBaseParams(), (tiny,))
    tm = Tissue.for_mechanics(pair.mechanical, TissueBaseParams(), (tiny,))

    assert te.empty_regions() == [], "на мелкой сетке регион должен быть виден"
    assert tm.empty_regions() == [0], "на крупной сетке регион должен пропасть"

    assert "ВНИМАНИЕ" in tm.summary()
    assert "tiny_scar" in tm.summary()


def test_adequate_region_not_flagged_as_empty():
    from cardiac_em.fem import Tissue, build_mesh

    mesh = build_mesh(RectangleMeshSpec(nx=20, ny=20, lx_mm=10.0, ly_mm=10.0))
    region = CircleRegion(cx=5.0, cy=5.0, r=2.0, name="big",
                          overrides={"T_MAX": 0.0})
    tissue = Tissue.for_electrics(mesh, TissueBaseParams(), (region,))

    assert tissue.empty_regions() == []
    assert "ВНИМАНИЕ" not in tissue.summary()


# ═══════════════════════════════════════════════════════════════════════
#  ВОЛОКНА И СВОДКА
# ═══════════════════════════════════════════════════════════════════════

def test_fiber_vector_is_usable_in_a_form():
    """
    Направление волокна должно собираться в UFL-выражение и
    интегрироваться. При угле 0° поле f0 = (1, 0), значит ∫ f0·f0 dΩ
    равен площади области.
    """
    import dolfinx.fem as fem
    import ufl

    from cardiac_em.fem import Tissue, build_mesh

    spec = RectangleMeshSpec(nx=8, ny=8, lx_mm=4.0, ly_mm=3.0)
    mesh = build_mesh(spec)
    tissue = Tissue.for_electrics(mesh, TissueBaseParams())

    f0 = tissue.fiber_vector()
    s0 = tissue.sheet_vector()

    area = fem.assemble_scalar(fem.form(ufl.dot(f0, f0) * ufl.dx))
    area = mesh.comm.allreduce(area)
    assert math.isclose(area, spec.lx_mm * spec.ly_mm, rel_tol=1e-10)

    # f0 ⟂ s0 в каждой точке
    cross = fem.assemble_scalar(fem.form(ufl.dot(f0, s0) * ufl.dx))
    cross = mesh.comm.allreduce(cross)
    assert abs(cross) < 1e-10


def test_fiber_angle_varies_by_region():
    """Регион может переопределять угол волокон — поле становится кусочным."""
    from cardiac_em.fem import Tissue, build_mesh

    mesh = build_mesh(RectangleMeshSpec(nx=10, ny=10, lx_mm=10.0, ly_mm=10.0))
    region = RectRegion(x0=5.0, x1=10.0, y0=0.0, y1=10.0, name="rotated",
                        overrides={"FIBER_ANGLE_DEG": 90.0})
    tissue = Tissue.for_electrics(mesh, TissueBaseParams(), (region,))

    xy = _owned_xy(tissue.DG0)
    ang = _owned_dg0(tissue, "FIBER_ANGLE_DEG")
    np.testing.assert_allclose(ang[xy[:, 0] > 5.0], 90.0)
    np.testing.assert_allclose(ang[xy[:, 0] < 5.0], 0.0)
    assert not tissue.is_uniform("FIBER_ANGLE_DEG")


def test_value_range_and_summary():
    from cardiac_em.fem import Tissue, build_mesh

    mesh = build_mesh(RectangleMeshSpec(nx=10, ny=10, lx_mm=10.0, ly_mm=10.0))
    region = RectRegion(x0=5.0, x1=10.0, y0=0.0, y1=10.0, name="half",
                        overrides={"T_MAX": 20.0})
    tissue = Tissue.for_electrics(mesh, TissueBaseParams(), (region,))

    lo, hi = tissue.value_range("T_MAX")
    assert math.isclose(lo, 20.0) and math.isclose(hi, 120.0)
    assert not tissue.is_uniform("T_MAX")
    assert tissue.is_uniform("D_TRANS")

    text = tissue.summary()
    assert "неоднородно" in text
    assert "T_MAX" in text


def test_dg0_function_matches_array():
    """Function на DG0 и numpy-массив должны нести одни и те же значения."""
    from cardiac_em.fem import Tissue, build_mesh

    mesh = build_mesh(RectangleMeshSpec(nx=12, ny=12, lx_mm=6.0, ly_mm=6.0))
    region = CircleRegion(cx=3.0, cy=3.0, r=1.5, name="c",
                          overrides={"D_LONG": 0.01})
    tissue = Tissue.for_electrics(mesh, TissueBaseParams(), (region,))

    n = tissue.DG0.dofmap.index_map.size_local
    np.testing.assert_allclose(
        tissue.dg0("D_LONG").x.array[:n], tissue.dg0_array("D_LONG")[:n])
