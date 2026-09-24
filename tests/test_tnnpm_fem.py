"""
TNNPM в ткани (Шаг 11).
=======================

ТРЕБУЮТ DOLFINx.

    pytest tests/test_tnnpm_fem.py -v
    mpirun -n 2 python -m pytest tests/test_tnnpm_fem.py -q

Полоска 8 × 0.4 мм (h = 0.1 мм), правая половина — 15-я минута ишемии
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
        mesh=DualMeshConfig.nested(
            RectangleMeshSpec(nx=80, ny=4, lx_mm=8.0, ly_mm=0.4), coarsening=4),
        cell_model="tnnpm",
        cell_params=cell_params or {},
        tissue_base=TissueBaseParams(
            conduction=ConductionParams(d_long=0.15, d_trans=0.15, fiber_angle_deg=0.0),
            active=ActiveStressParams(t_max=60.0)),
        regions=[RectRegion(name="ишемия 15 мин", x0=4.05, x1=8.0, y0=0.0, y1=0.4,
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
    assert len(run.manifest["models"]["cell_states"]) == 29
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
    assert 40.0 < s["t_act_electric_max"].max() < 70.0, "T_act/T_MAX ≈ 1 на пике здоровой ткани"
    np.testing.assert_allclose(s["t_act_electric_integral"], s["t_act_mech_integral"],
                               rtol=1e-10, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════
#  ОШИБКИ В ПАРАМЕТРАХ КЛЕТКИ — ДО НАЧАЛА СЧЁТА
# ═══════════════════════════════════════════════════════════════════════

def test_unknown_cell_param_in_config(tmp_path):
    from cardiac_em.runtime import Simulation

    with pytest.raises(KeyError, match="ATPi"):
        Simulation(_config(_shared(tmp_path), cell_params={"ATPi": 4.0}))


def test_unknown_cell_param_in_region(tmp_path):
    from cardiac_em.runtime import Simulation

    bad = [RectRegion(name="x", x0=0, x1=1, y0=0, y1=0.4, overrides={"cell:gNa": 1.0})]
    with pytest.raises(ValueError, match="cell:gNa"):
        Simulation(_config(_shared(tmp_path), regions=bad))


def test_cell_keys_rejected_for_model_without_cell_params(tmp_path):
    from cardiac_em.control import apply_overrides
    from cardiac_em.runtime import Simulation

    cfg = apply_overrides(_config(_shared(tmp_path)), {"cell_model": "rogers_mcculloch"})
    with pytest.raises(ValueError, match="ни одного"):
        Simulation(cfg)
