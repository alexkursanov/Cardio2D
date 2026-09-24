"""
TNNPM в ткани (Шаг 11).
=======================

ТРЕБУЮТ DOLFINx.

    pytest tests/test_tnnpm_fem.py -v
    mpirun -n 2 python -m pytest tests/test_tnnpm_fem.py -q

Модель "tnnpm" — вариант для ткани: в клетке только сократительный
элемент, пассивная механика — в ткани; ткань сообщает клеткам длину
саркомера (l₁ = 1.67·(λ_f − 1)) и скорость, а сила–скорость учитывается
неявно в равновесии (models/active).

Полоска 8 × 0.8 мм: электрика 80×8 (h = 0.1 мм), механика 20×4 — у неё
есть внутренние узлы (при сетке в одну ячейку по высоте все узлы на
зажатых гранях, и ткань не деформируется вовсе). Правая половина —
15-я минута ишемии
по сценарию автора модели (K_o 9.4 мМ, ATP_i 4.0 мМ, KmATP 0.38).
Перед счётом 300 мс подготовки без стимула (preload.cell_relax_ms):
ишемическая зона и граница приходят к своему покою. Затем связанный
расчёт 350 мс: волна проходит всю полоску, в ишемической
зоне потенциал действия короче и покой деполяризован, механика
сходится. Точные значения здесь не проверяются — они проверены на
уровне клетки (tests/test_tnnpm.py); здесь — что модель правильно
встроена в ткань: единицы (мВ, без обрезки потенциала), параметры по
регионам, регистратор активации с деполяризованным покоем.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.config import (  # noqa: E402
    ActiveStressParams,
    ConductionParams,
    DualMeshConfig,
    OutputConfig,
    PreloadProtocol,
    RectangleMeshSpec,
    RectRegion,
    SimulationConfig,
    StimulusProtocol,
    TimeStepping,
    TissueBaseParams,
)

pytestmark = pytest.mark.fem

ISCHEMIA_15MIN = {"cell:K_o": 9.4, "cell:ATP_i": 4.0, "cell:KmATP": 0.38}


def _shared(path: Path) -> Path:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    return Path(comm.bcast(str(path) if comm.rank == 0 else None, root=0))


def _config(out, t_end=350.0, regions=None, cell_params=None) -> SimulationConfig:
    return SimulationConfig(
        mesh=DualMeshConfig(
            electric=RectangleMeshSpec(nx=80, ny=8, lx_mm=8.0, ly_mm=0.8),
            mechanical=RectangleMeshSpec(nx=20, ny=4, lx_mm=8.0, ly_mm=0.8)),
        cell_model="tnnpm",
        cell_params=cell_params or {},
        tissue_base=TissueBaseParams(
            conduction=ConductionParams(d_long=0.15, d_trans=0.15, fiber_angle_deg=0.0),
            active=ActiveStressParams(t_max=60.0)),
        regions=[RectRegion(name="ишемия 15 мин", x0=4.05, x1=8.0, y0=0.0, y1=0.8,
                            overrides=ISCHEMIA_15MIN)] if regions is None else regions,
        stimulus=StimulusProtocol(times_ms=(1.0,), duration_ms=2.0, amplitude=100.0,
                                  rise_time_ms=0.1, decay_length_mm=0.5, x_max_mm=1.0),
        time=TimeStepping(dt_electric_ms=0.05, n_electric_per_mech=20, t_end_ms=t_end),
        preload=PreloadProtocol(stretch=1.2575, n_steps=4, cell_relax_ms=300.0),
        output=OutputConfig(out_dir=out, save_every_mech_steps=50,
                            snapshot_times_ms=(t_end,), ckpt_every_mech_steps=0),
    )


@pytest.fixture(scope="module")
def strip(tmp_path_factory):
    from cardiac_em.control import run_simulation

    out = _shared(tmp_path_factory.mktemp("tnnpm"))
    result = run_simulation(_config(out), console=False)
    return {"out": out, "result": result}


def test_run_finished_with_tnnpm(strip):
    from cardiac_em.analysis import open_run

    run = open_run(strip["out"])
    assert run.status == "finished"
    assert run.manifest["models"]["cell"] == "tnnpm"
    assert len(run.manifest["models"]["cell_states"]) == 26, \
        "вариант для ткани: 18 электрических + v, N, A, l_1 + 4 RyR"
    assert run.warnings == [], run.warnings
    s = run.series()
    assert s["u_max"].max() > 20.0, "потенциал в мВ: пик ПД выше +20 мВ (обрезки нет)"
    assert s["u_min"].min() < -80.0
    assert np.all(s["newton_iterations"][1:] < 20)


def test_wave_crosses_the_ischemic_border(strip):
    from cardiac_em.analysis import biomarkers as bm
    from cardiac_em.analysis import open_run

    maps = open_run(strip["out"]).activation()
    assert maps.threshold == -40.0
    s = bm.activation_summary(maps.act[:, 0])
    assert s["fraction_activated"] == 1.0, "волна должна пройти всю полоску"
    cv = bm.conduction_velocity(maps.coords, maps.act[:, 0], window=(1.5, 3.5))
    assert 0.3 < cv < 1.0, f"скорость в здоровой зоне {cv:.3f} мм/мс вне физиологического диапазона"


def test_ischemic_zone_has_shorter_apd_and_depolarized_rest(strip):
    from cardiac_em.analysis import biomarkers as bm
    from cardiac_em.analysis import open_run

    run = open_run(strip["out"])
    maps = run.activation()
    assert np.isfinite(maps.repol[:, 0]).all(), \
        "реполяризация найдена во всех узлах, включая деполяризованную зону"
    apd = bm.apd_summary(maps.apd[:, 0], maps.region)["by_region"]
    # одномерная прикидка той же постановки (numpy, явная диффузия):
    # APD90 250 мс в норме, 214 в ишемии; покой −80.5 и −69.7 мВ к 350 мс
    assert apd[1]["mean"] < apd[0]["mean"] - 20.0, (
        f"APD90: ишемия {apd[1]['mean']:.1f} мс, норма {apd[0]['mean']:.1f} мс")

    snap = run.snapshot(350.0)
    V = snap.state("V")
    x = snap["e_coords"][:, 0]
    healthy, ischemic = V[x < 2.0].mean(), V[x > 6.0].mean()
    assert ischemic > healthy + 6.0, (
        f"покой: ишемия {ischemic:.1f} мВ, норма {healthy:.1f} мВ")


def test_active_tension_reaches_mechanics(strip):
    from cardiac_em.analysis import open_run

    s = open_run(strip["out"]).series()
    # переданная клетками сила (при нулевой скорости) сохраняется переносом
    np.testing.assert_allclose(s["t_act_electric_integral"], s["t_act_mech_integral"],
                               rtol=1e-10, atol=1e-10)
    assert 20.0 < s["t_act_mech_max"].max() < 80.0


def test_coupling_is_two_way(strip):
    """
    Здоровая половина укорачивается, растягивая ослабленную ишемическую;
    действующее напряжение ниже переданного (укорочение снижает силу);
    клетки видят ту длину, что у ткани; колебаний от связи нет.
    """
    from cardiac_em.analysis import open_run

    s = open_run(strip["out"]).series()
    assert s["lambda_f_min"].min() < 1.2575 - 0.01, "активная зона должна укорачиваться"
    assert s["lambda_f_max"].max() > 1.2575 + 0.01, "ослабленная зона — растягиваться"
    assert s["t_act_actual_max"].max() < s["t_act_mech_max"].max(), \
        "сила при укорочении меньше изометрической"
    turns = np.sum(np.diff(np.sign(np.diff(s["lambda_f_min"]))) != 0)
    assert turns <= 6, f"λ_f колеблется: {turns} смен направления"

    sim = strip["result"].sim
    cell, e, m = sim.cell_model, sim.electrics, sim.mechanics
    lam_nodes = sim.stretch_sampler.apply(m.fiber_stretch().x.array[:m.n_cells_owned])
    np.testing.assert_allclose(cell.sarcomere_length(e.state), 1.67 * lam_nodes, atol=1e-12)


def test_force_velocity_law_matches_cell_function(tmp_path):
    """
    φ(v) в UFL совпадает с p(v) клетки (numpy) во всех ветвях. Закон
    вычисляется на ячейках по заданным λ и λ_пред (поля DG0) — так
    скорости покрывают все участки кривой.
    """
    from dolfinx import fem

    from cardiac_em.fem import build_mesh
    from cardiac_em.models.active import make_active_law
    from cardiac_em.models.cell import make_cell_model

    cell = make_cell_model("tnnpm")
    c = cell.c
    law = make_active_law(cell)
    mesh = build_mesh(RectangleMeshSpec(nx=6, ny=2, lx_mm=6.0, ly_mm=2.0))
    DG0 = fem.functionspace(mesh, ("DG", 0))
    n = DG0.dofmap.index_map.size_local
    x = np.linspace(-1.5, 2.0, 12)[:n] if n else np.zeros(0)
    lam, lam_old = fem.Function(DG0), fem.Function(DG0)
    lam.x.array[:] = 1.2
    lam_old.x.array[:n] = 1.2 - x * c["v_max"] * 1.0 / c["SL_slack"]
    lam_old.x.scatter_forward()
    phi = fem.Function(DG0)
    phi.interpolate(fem.Expression(law.factor(lam, lam_old, 1.0),
                                   DG0.element.interpolation_points))
    np.testing.assert_allclose(phi.x.array[:n], cell._p(x * c["v_max"]),
                               rtol=1e-12, atol=1e-12)


def test_no_spurious_velocity_in_a_deformed_static_state(tmp_path):
    """
    Неоднородная деформация в покое (u = u_пред) не должна давать
    скорости ни в одной квадратурной точке: φ ≡ 1, действующее
    напряжение равно переданному. Раньше прошлое растяжение хранилось
    по ячейкам, и разброс λ_f внутри ячейки выглядел как скорость.
    """
    import ufl
    from dolfinx import fem

    from cardiac_em.fem import Tissue, build_mesh
    from cardiac_em.models.active import make_active_law
    from cardiac_em.models.cell import make_cell_model
    from cardiac_em.solvers import MechanicsSolver

    cell = make_cell_model("tnnpm")
    mesh = build_mesh(RectangleMeshSpec(nx=6, ny=2, lx_mm=6.0, ly_mm=2.0))
    solver = MechanicsSolver(Tissue.for_mechanics(mesh, TissueBaseParams()),
                             active_law=make_active_law(cell), dt_mech=1.0)
    xs = ufl.SpatialCoordinate(mesh)
    solver.u.interpolate(fem.Expression(ufl.as_vector([0.05 * xs[0] * xs[1], 0.02 * xs[0] ** 2]),
                                        solver.V.element.interpolation_points))
    solver.commit_step()
    solver.set_active_tension(np.full(solver.n_cells_owned, 10.0))
    # по квадратурным точкам формы (не только в центрах ячеек):
    # ∫ T_act·φ dx обязан равняться 10 · площадь
    from mpi4py import MPI
    local = fem.assemble_scalar(fem.form(solver._effective_tension(solver.u) * ufl.dx))
    total = mesh.comm.allreduce(local, op=MPI.SUM)
    assert abs(total - 10.0 * 12.0) < 1e-9, f"∫T_act = {total}, ожидалось 120"
    np.testing.assert_allclose(solver.stretch_rate(), 0.0, atol=1e-12)


def test_cell_to_node_sampler_on_nested_meshes():
    """Узел внутри ячейки — её значение; на границе — среднее соседних."""
    from cardiac_em.coupling import CellToNodeSampler
    from cardiac_em.fem import Tissue, build_mesh, dof_coordinates

    spec_m = RectangleMeshSpec(nx=6, ny=2, lx_mm=6.0, ly_mm=2.0)
    spec_e = RectangleMeshSpec(nx=24, ny=8, lx_mm=6.0, ly_mm=2.0)
    tm = Tissue.for_mechanics(build_mesh(spec_m), TissueBaseParams())
    te = Tissue.for_electrics(build_mesh(spec_e), TissueBaseParams())
    sampler = CellToNodeSampler(tm.DG0, te.P1, spec_m)

    n_m = tm.DG0.dofmap.index_map.size_local
    cxy = dof_coordinates(tm.DG0)[:n_m, :2]
    ix, iy = np.floor(cxy[:, 0]).astype(int), np.floor(cxy[:, 1]).astype(int)
    values = ix + 10.0 * iy                          # значение ячейки (ix, iy)
    got = sampler.apply(values)

    xy = dof_coordinates(te.P1)[:len(got), :2]
    expected = []
    for x, y in xy:
        xs = {min(max(int(np.floor(x + d)), 0), 5) for d in (-1e-9, 1e-9)}
        ys = {min(max(int(np.floor(y + d)), 0), 1) for d in (-1e-9, 1e-9)}
        expected.append(np.mean([a + 10.0 * b for a in xs for b in ys]))
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_isometric_variant_runs_without_feedback(tmp_path):
    from cardiac_em.control import apply_overrides
    from cardiac_em.runtime import Simulation

    cfg = apply_overrides(_config(_shared(tmp_path), t_end=5.0, regions=[]),
                          {"cell_model": "tnnpm_isometric", "preload.cell_relax_ms": 0.0})
    sim = Simulation(cfg)
    assert sim.stretch_sampler is None and not sim.active_law.uses_velocity
    sim.preload()
    sim.run()
    assert sim.t_ms == 5.0


# ═══════════════════════════════════════════════════════════════════════
#  ОШИБКИ В ПАРАМЕТРАХ КЛЕТКИ — ДО НАЧАЛА СЧЁТА
# ═══════════════════════════════════════════════════════════════════════

def test_unknown_cell_param_in_config(tmp_path):
    from cardiac_em.runtime import Simulation

    with pytest.raises(KeyError, match="ATPi"):
        Simulation(_config(_shared(tmp_path), cell_params={"ATPi": 4.0}))


def test_unknown_cell_param_in_region(tmp_path):
    from cardiac_em.runtime import Simulation

    bad = [RectRegion(name="x", x0=0, x1=1, y0=0, y1=0.8, overrides={"cell:gNa": 1.0})]
    with pytest.raises(ValueError, match="cell:gNa"):
        Simulation(_config(_shared(tmp_path), regions=bad))


def test_cell_keys_rejected_for_model_without_cell_params(tmp_path):
    from cardiac_em.control import apply_overrides
    from cardiac_em.runtime import Simulation

    cfg = apply_overrides(_config(_shared(tmp_path)), {"cell_model": "rogers_mcculloch"})
    with pytest.raises(ValueError, match="ни одного"):
        Simulation(cfg)
