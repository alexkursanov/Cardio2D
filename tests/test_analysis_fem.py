"""
Анализ настоящих прогонов (Шаг 10б).
====================================

ТРЕБУЮТ DOLFINx.

    pytest tests/test_analysis_fem.py -v
    mpirun -n 2 python -m pytest tests/test_analysis_fem.py -q

Цепочка целиком: прогон с регистратором активации → файлы → чтение
через analysis/ → биомаркеры. Проверки независимые:

* времена активации сверяются с прямым пошаговым замером (первый шаг,
  на котором потенциал выше порога) — регистратор обязан дать момент
  внутри этого шага;
* скорость проведения из карты — с конечно-разностным эталоном из
  tests/test_monodomain.py (тот же допуск и та же геометрия);
* APD90 в ткани — с APD90 одиночной клетки, посчитанным прямым
  интегрированием модели без DOLFINx.

Механика в электрических тестах отключена (T_max = 0): здесь она не
предмет проверки и только удлиняла бы счёт.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardiac_em.config import (  # noqa: E402
    ActiveStressParams,
    ConductionParams,
    DualMeshConfig,
    OutputConfig,
    PreloadProtocol,
    RectangleMeshSpec,
    SimulationConfig,
    StimulusProtocol,
    TimeStepping,
    TissueBaseParams,
)
from cardiac_em.runtime import Observer  # noqa: E402  (без DOLFINx)

pytestmark = pytest.mark.fem

DT = 0.05


def _shared(path: Path) -> Path:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    return Path(comm.bcast(str(path) if comm.rank == 0 else None, root=0))


def _electric_config(out, *, spec, coarsening, times, t_end, t_max=0.0,
                     stretch=1.0) -> SimulationConfig:
    return SimulationConfig(
        mesh=DualMeshConfig.nested(spec, coarsening=coarsening),
        tissue_base=TissueBaseParams(
            conduction=ConductionParams(d_long=0.3, d_trans=0.3, fiber_angle_deg=0.0),
            active=ActiveStressParams(t_max=t_max)),
        stimulus=StimulusProtocol(times_ms=times, duration_ms=2.0, amplitude=5.0,
                                  rise_time_ms=0.2, decay_length_mm=1.0, x_max_mm=3.0),
        time=TimeStepping(dt_electric_ms=DT, n_electric_per_mech=20, t_end_ms=t_end),
        preload=PreloadProtocol(stretch=stretch, n_steps=4),
        output=OutputConfig(out_dir=out, save_every_mech_steps=50,
                            ckpt_every_mech_steps=0),
    )


class _StepArrival(Observer):
    """Независимый замер: конец первого шага, где потенциал > порога."""

    def __init__(self):
        self.arrival = None

    def on_start(self, sim, schedule):
        self.arrival = np.full(sim.electrics.n_owned, np.nan)

    def on_electric_step(self, sim, tick):
        e = sim.electrics
        u = e.potential()[:e.n_owned]
        fresh = np.isnan(self.arrival) & (u > sim.cell_model.activation_threshold)
        self.arrival[fresh] = tick.t_end_ms

    def gathered(self, sim):
        from cardiac_em.io._points import gather_owned
        return gather_owned(sim.electrics.P1, self.arrival, sim.comm)


# ═══════════════════════════════════════════════════════════════════════
#  ПЛОСКАЯ ВОЛНА: ВРЕМЕНА АКТИВАЦИИ И СКОРОСТЬ
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def planar(tmp_path_factory):
    from cardiac_em.io import attach_outputs
    from cardiac_em.runtime import Simulation

    out = _shared(tmp_path_factory.mktemp("planar"))
    spec = RectangleMeshSpec(nx=72, ny=4, lx_mm=18.0, ly_mm=1.0)
    cfg = _electric_config(out, spec=spec, coarsening=4, times=(1.0,), t_end=130.0)
    probe = _StepArrival()
    sim = Simulation(cfg, observers=[probe])
    attach_outputs(sim)
    sim.preload()
    sim.run()
    coords, arrival = probe.gathered(sim)
    return {"out": out, "sim": sim, "coords": coords, "arrival": arrival}


def test_activation_file_is_listed_in_manifest(planar):
    from cardiac_em.analysis import open_run

    run = open_run(planar["out"])
    assert "activation.npz" in run.manifest["outputs"]
    maps = run.activation()
    assert maps.n_beats == 1
    assert maps.threshold == 0.5 and maps.apd_level == 0.9
    assert maps.grid(maps.act[:, 0]).shape == (5, 73)


def test_recorded_times_fall_inside_the_crossing_step(planar):
    """
    Интерполированный момент обязан лежать в том шаге, где пересечение
    произошло: arrival − dt < act ≤ arrival.
    """
    from cardiac_em.analysis import open_run
    from mpi4py import MPI

    maps = open_run(planar["out"]).activation()
    if MPI.COMM_WORLD.rank != 0:
        return
    np.testing.assert_allclose(maps.coords, planar["coords"], atol=1e-12)
    act, arr = maps.act[:, 0], planar["arrival"][:, 0]
    both = np.isfinite(act) & np.isfinite(arr)
    assert both.sum() == np.isfinite(arr).sum() == np.isfinite(act).sum()
    assert np.all(act[both] <= arr[both] + 1e-9)
    assert np.all(act[both] > arr[both] - DT - 1e-9)


def test_conduction_velocity_from_map_matches_reference(planar):
    from test_monodomain import _reference_cv_1d

    from cardiac_em.analysis import biomarkers as bm
    from cardiac_em.analysis import open_run

    maps = open_run(planar["out"]).activation()
    s = bm.activation_summary(maps.act[:, 0])
    assert s["fraction_activated"] == 1.0

    cv = bm.conduction_velocity(maps.coords, maps.act[:, 0], window=(7.0, 13.0))
    reference = _reference_cv_1d(0.3)
    assert math.isclose(cv, reference, rel_tol=0.15), (
        f"CV по карте {cv:.4f} против эталона {reference:.4f} мм/мс")

    # та же оценка по пошаговому замеру отличается лишь смещением на
    # доли шага — на наклон это почти не влияет
    if planar["arrival"] is not None:
        cv_steps = bm.conduction_velocity(planar["coords"], planar["arrival"][:, 0],
                                          window=(7.0, 13.0))
        assert math.isclose(cv, cv_steps, rel_tol=0.01)

    field = bm.cv_field(maps.grid(maps.act[:, 0]), 18.0 / 72, 1.0 / 4)
    mid = field[:, 28:52]
    assert math.isclose(np.nanmedian(mid), cv, rel_tol=0.05)


# ═══════════════════════════════════════════════════════════════════════
#  ДВА УДАРА: APD И РЕСТИТУЦИЯ
# ═══════════════════════════════════════════════════════════════════════

def _single_cell_apd90() -> float:
    """APD90 одиночной клетки — прямое интегрирование модели, без DOLFINx."""
    from cardiac_em.io.activation import ThresholdDetector
    from cardiac_em.models.cell import make_cell_model

    cell = make_cell_model("rogers_mcculloch")
    flat = TissueBaseParams().to_flat_dict()
    params = {k: np.array([flat[k]]) for k in cell.param_names}
    y = cell.initial_state(1)
    det = ThresholdDetector(y[:, cell.v_index], 0.0, cell.activation_threshold,
                            float(cell.resting_state()[cell.v_index]), 0.9)
    for k in range(int(round(400.0 / DT))):
        t = k * DT
        stim = np.array([5.0 if 1.0 <= t < 3.0 else 0.0])
        y = cell.step(t, y, DT, stim, params)
        det.update(y[:, cell.v_index], t + DT)
    act, rep, _ = det.maps()
    return float(rep[0, 0] - act[0, 0])


def test_apd_and_restitution_over_two_beats(tmp_path):
    from cardiac_em.analysis import biomarkers as bm
    from cardiac_em.analysis import open_run
    from cardiac_em.io import attach_outputs
    from cardiac_em.runtime import Simulation

    out = _shared(tmp_path)
    spec = RectangleMeshSpec(nx=24, ny=4, lx_mm=6.0, ly_mm=1.0)
    cfg = _electric_config(out, spec=spec, coarsening=4, times=(1.0, 400.0),
                           t_end=700.0)
    sim = Simulation(cfg)
    attach_outputs(sim)
    sim.preload()
    sim.run()

    maps = open_run(out).activation()
    assert maps.n_beats == 2
    assert np.isfinite(maps.act).all() and np.isfinite(maps.repol).all(), \
        "оба удара проведены и завершились во всех узлах"

    apd_cell = _single_cell_apd90()
    for beat in (0, 1):
        s = bm.apd_summary(maps.apd[:, beat], maps.region)
        assert math.isclose(s["mean_ms"], apd_cell, rel_tol=0.10), (
            f"удар {beat}: APD90 в ткани {s['mean_ms']:.1f} мс, "
            f"у одиночной клетки {apd_cell:.1f} мс")

    r = bm.restitution(maps.act, maps.repol)
    assert len(r["di_ms"]) == len(maps.coords)
    assert np.all(r["di_ms"] > 0)
    # короткий интервал отдыха — второй ПД не длиннее первого
    assert np.median(r["apd_ms"]) <= np.median(maps.apd[:, 0]) + 1e-6
    assert bm.conduction_block(maps.act, 0)["n_blocked"] == 0


# ═══════════════════════════════════════════════════════════════════════
#  МЕХАНИКА И СЕРИЯ
# ═══════════════════════════════════════════════════════════════════════

def test_mechanics_summary_on_real_run(tmp_path):
    from cardiac_em.analysis import biomarkers as bm
    from cardiac_em.analysis import open_run
    from cardiac_em.control import run_simulation

    out = _shared(tmp_path)
    cfg = _electric_config(out, spec=RectangleMeshSpec(nx=24, ny=8, lx_mm=6.0, ly_mm=2.0),
                           coarsening=4, times=(1.0,), t_end=30.0,
                           t_max=120.0, stretch=1.2575)
    run_simulation(cfg, console=False)

    run = open_run(out)
    s = run.series()
    m = bm.mechanics_summary(s)
    assert m["peak_t_act_kpa"] == pytest.approx(np.max(s["t_act_mech_max"]))
    assert m["sigma_rise_kpa"] > 0.5 * m["peak_t_act_kpa"]
    assert m["min_lambda_f"] < 1.2575 and m["max_shortening"] > 0
    assert m["t_act_time_integral_kpa_mm2_ms"] > 0


def test_sweep_table_from_real_sweep(tmp_path):
    from cardiac_em.analysis import biomarkers as bm
    from cardiac_em.analysis import open_sweep
    from cardiac_em.control import SweepSpec, run_sweep

    root = _shared(tmp_path)
    base = _electric_config(root / "unused",
                            spec=RectangleMeshSpec(nx=24, ny=4, lx_mm=6.0, ly_mm=1.0),
                            coarsening=4, times=(1.0,), t_end=40.0)
    spec = SweepSpec(base=base, out_root=root / "sw",
                     parameters={"tissue_base.conduction.d_long": [0.15, 0.6],
                                 "tissue_base.conduction.d_trans": [0.15, 0.6]},
                     mode="zip")
    run_sweep(spec, console=False)

    def metrics(run):
        maps = run.activation()
        return bm.activation_summary(maps.act[:, 0])

    rows = open_sweep(root / "sw").table(metrics)
    assert [r["tissue_base.conduction.d_long"] for r in rows] == [0.15, 0.6]
    # вчетверо больше D — вдвое быстрее проведение: раньше последняя активация
    assert rows[1]["last_ms"] < rows[0]["last_ms"]
