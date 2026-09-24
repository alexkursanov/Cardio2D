"""
Тесты записи полей, чекпоинтов и продолжения расчёта (Шаг 9).
==============================================================

ТРЕБУЮТ DOLFINx.

    pytest tests/test_io.py -v
    mpirun -n 2 python -m pytest tests/test_io.py -q

Главная проверка шага — ПРОДОЛЖЕНИЕ РАВНО НЕПРЕРЫВНОМУ СЧЁТУ: прогон
0 → 12 мс и прогон 0 → 5 мс + продолжение 5 → 12 мс с чекпоинта дают
одно и то же. Электрика от механики не зависит, поэтому её состояние
обязано совпасть до последнего бита; механика — с точностью до допуска
Ньютона (продолжение заново решает равновесие в стартовой точке).

О папках при MPI: у каждого ранга pytest создаёт свой tmp_path. Вывод
же обязан идти в ОДНУ папку (запись XDMF — коллективная операция),
поэтому путь берётся с нулевого ранга и рассылается остальным.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.config import (  # noqa: E402
    DualMeshConfig,
    OutputConfig,
    PreloadProtocol,
    RectangleMeshSpec,
    RectRegion,
    RestartConfig,
    SimulationConfig,
    StimulusProtocol,
    TimeStepping,
)

pytestmark = pytest.mark.fem

NX, NY = 24, 8            # электрическая сетка; механическая — 6×2
T_END = 12.0


def _shared(path: Path) -> Path:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    return Path(comm.bcast(str(path) if comm.rank == 0 else None, root=0))


def _barrier():
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()


def _config(out_dir, *, t_end=T_END, nx=NX, ny=NY, coarsening=4, lx=6.0,
            ckpt_every=5, keep_all=True, snapshots=(0.0, 5.0, 7.5, 100.0),
            restart=None) -> SimulationConfig:
    return SimulationConfig(
        mesh=DualMeshConfig.nested(
            RectangleMeshSpec(nx=nx, ny=ny, lx_mm=lx, ly_mm=2.0),
            coarsening=coarsening),
        regions=[RectRegion(name="left", overrides={"T_MAX": 90.0},
                            x0=0.0, x1=3.1, y0=0.0, y1=2.0)],
        stimulus=StimulusProtocol(times_ms=(1.0,), duration_ms=2.0,
                                  amplitude=20.0, rise_time_ms=0.5,
                                  decay_length_mm=0.5, x_max_mm=2.0),
        time=TimeStepping(dt_electric_ms=0.05, n_electric_per_mech=20,
                          t_end_ms=t_end),
        preload=PreloadProtocol(stretch=1.2575, n_steps=4),
        output=OutputConfig(out_dir=out_dir, save_every_mech_steps=2,
                            snapshot_times_ms=snapshots,
                            ckpt_every_mech_steps=ckpt_every,
                            ckpt_keep_all=keep_all),
        restart=restart or RestartConfig(),
    )


def _ckpt_name(t):
    return f"ckpt_t{t:012.3f}ms.npz"


def _snap_name(t):
    return f"snap_t{t:012.3f}ms.npz"


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ═══════════════════════════════════════════════════════════════════════
#  ЭТАЛОННЫЙ ПРОГОН С ПОЛНЫМ ВЫВОДОМ
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def ref_run(tmp_path_factory):
    from cardiac_em.io import attach_outputs
    from cardiac_em.runtime import Simulation

    out = _shared(tmp_path_factory.mktemp("ref"))
    cfg = _config(out)
    sim = Simulation(cfg)
    observers = attach_outputs(sim)
    sim.preload()
    sim.run()
    _barrier()
    return {"out": out, "sim": sim, "cfg": cfg, "observers": observers}


def test_expected_files_exist(ref_run):
    out = ref_run["out"]
    expected = [
        "run.json", "series.csv", "preload.csv",
        "electrics.xdmf", "electrics.h5", "mechanics.xdmf", "mechanics.h5",
        "regions_electric.xdmf", "regions_mechanical.xdmf",
        "ckpt_preloaded.npz", "ckpt_last.npz",
        _ckpt_name(5.0), _ckpt_name(10.0), _ckpt_name(12.0),
        f"snapshots/{_snap_name(0.0)}", f"snapshots/{_snap_name(5.0)}",
        f"snapshots/{_snap_name(8.0)}",
    ]
    missing = [f for f in expected if not (out / f).exists()]
    assert not missing, f"нет файлов: {missing}"
    assert not list(out.rglob("*.tmp")), "остались недописанные временные файлы"


def test_manifest_describes_the_run(ref_run):
    m = json.loads((ref_run["out"] / "run.json").read_text(encoding="utf-8"))

    assert m["status"] == "finished" and m["error"] is None
    assert m["progress"]["t_ms"] == T_END
    assert m["schedule"]["n_mech_steps"] == 12
    assert m["schedule"]["unreachable_snapshots_ms"] == [100.0]
    assert m["models"]["transfer_exact"] is True
    assert m["meshes"]["electric"]["n_cells"] == NX * NY
    assert m["meshes"]["mechanical"]["n_cells"] == 12
    assert m["restart"] is None
    assert any("100.0" in w for w in m["warnings"]), "предупреждение о снимке"
    assert "ckpt_last.npz" in m["outputs"] and "series.csv" in m["outputs"]
    assert m["environment"]["dolfinx"]
    assert SimulationConfig.from_dict(m["config"]).to_dict() == ref_run["cfg"].to_dict()


def test_series_csv_matches_live_diagnostics(ref_run):
    data = np.genfromtxt(ref_run["out"] / "series.csv", delimiter=",", names=True)
    pre = np.genfromtxt(ref_run["out"] / "preload.csv", delimiter=",", names=True)

    np.testing.assert_allclose(data["t_ms"], np.arange(0.0, T_END + 0.5, 1.0))
    np.testing.assert_allclose(data["t_act_electric_integral"],
                               data["t_act_mech_integral"], rtol=1e-10, atol=1e-10)
    assert data["t_act_electric_max"].max() > 10.0, "волна должна была пройти"

    live = ref_run["sim"].diagnostics()
    assert data["lambda_f_max"][-1] == pytest.approx(live["lambda_f_max"], rel=1e-12)
    assert data["sigma_xx_max"][-1] == pytest.approx(live["sigma_xx_max"], rel=1e-12)

    assert len(pre) == 4
    assert pre["lambda_f_max"][-1] == pytest.approx(1.2575, rel=1e-8)


def test_xdmf_frames_and_names(ref_run):
    """Кадры — в t = 0 и каждые 2 мех. шага; имена полей — узнаваемые."""
    def times_and_names(path):
        root = ET.parse(path).getroot()
        times = {float(t.get("Value")) for t in root.iter("Time")}
        names = {g.get("Name") for g in root.iter("Grid")}
        return times, names

    t_m, n_m = times_and_names(ref_run["out"] / "mechanics.xdmf")
    t_e, n_e = times_and_names(ref_run["out"] / "electrics.xdmf")

    frames = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0}
    assert t_m == frames and t_e == frames
    assert {"u", "T_act_kPa", "lambda_f", "J", "sigma_00"} <= n_m
    assert {"u", "v", "T_act_kPa"} <= n_e


def test_snapshot_layout_and_content(ref_run):
    """
    Снимок упорядочен по (y, x): на прямоугольной сетке он складывается
    в двумерную карту обычным reshape — независимо от числа рангов.
    """
    snap = _load(ref_run["out"] / "snapshots" / _snap_name(8.0))

    assert float(snap["t_ms"]) == 8.0
    np.testing.assert_array_equal(snap["requested_ms"], [7.5])
    assert tuple(snap["e_state_names"]) == ("u", "v")

    xy = snap["e_coords"].reshape(NY + 1, NX + 1, 2)
    np.testing.assert_allclose(xy[0, :, 0], np.linspace(0, 6.0, NX + 1), atol=1e-12)
    np.testing.assert_allclose(xy[:, 0, 1], np.linspace(0, 2.0, NY + 1), atol=1e-12)
    assert snap["e_state"].shape == ((NX + 1) * (NY + 1), 2)
    assert snap["m_u"].shape == (7 * 3, 2)
    assert snap["m_t_act"].shape == (12,)

    # регионы: узлы и ячейки левее x = 3.1 мм — регион 1, ВКЛЮЧАЯ узлы
    # на краю области y = 2 (см. test_tissue: шум округления координат)
    np.testing.assert_array_equal(snap["e_region"], (snap["e_coords"][:, 0] < 3.1).astype(int))
    np.testing.assert_array_equal(snap["m_region"], (snap["m_cell_coords"][:, 0] < 3.1).astype(int))

    # осреднение сохраняет силу и в сохранённых данных
    area_e = (6.0 / NX) * (2.0 / NY)
    area_m = 1.0
    assert snap["e_t_act"].sum() * area_e == pytest.approx(
        snap["m_t_act"].sum() * area_m, rel=1e-10, abs=1e-10)
    assert snap["e_t_act"].max() > 0.0


def test_last_checkpoint_equals_final_state(ref_run):
    from cardiac_em.io import read_checkpoint

    c = read_checkpoint(ref_run["out"] / "ckpt_last.npz")
    final = read_checkpoint(ref_run["out"] / _ckpt_name(12.0))

    assert c.stage == "running" and c.t_ms == T_END and c.mech_index == 12
    np.testing.assert_array_equal(c.e_state, final.e_state)
    np.testing.assert_array_equal(c.m_u, final.m_u)

    pre = read_checkpoint(ref_run["out"] / "ckpt_preloaded.npz")
    assert pre.stage == "preloaded" and pre.t_ms == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  ПРОДОЛЖЕНИЕ
# ═══════════════════════════════════════════════════════════════════════

def _continue(ref_run, tmp, source, *, t_end=T_END, **cfg_kw):
    from cardiac_em.io import attach_outputs, read_checkpoint, restore_checkpoint
    from cardiac_em.runtime import Simulation

    out = _shared(tmp)
    sim = Simulation(_config(out, t_end=t_end, **cfg_kw))
    attach_outputs(sim)
    info = restore_checkpoint(sim, ref_run["out"] / source)
    sim.run(t_start_ms=info.t_start_ms)
    _barrier()
    return out, info, read_checkpoint(out / "ckpt_last.npz")


def test_continuation_equals_uninterrupted_run(ref_run, tmp_path):
    """0 → 12 мс подряд == 0 → 5 мс, чекпоинт, 5 → 12 мс."""
    from cardiac_em.io import read_checkpoint

    out, info, cont = _continue(ref_run, tmp_path, _ckpt_name(5.0))
    ref = read_checkpoint(ref_run["out"] / "ckpt_last.npz")

    assert info.t_checkpoint_ms == 5.0 and info.t_start_ms == 5.0
    assert info.nearest_nodes == {} and info.warnings == []
    assert cont.t_ms == T_END and cont.mech_index == 12, \
        "номера механических шагов продолжают глобальную нумерацию"

    np.testing.assert_array_equal(cont.e_coords, ref.e_coords)
    np.testing.assert_allclose(cont.e_state, ref.e_state, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cont.m_u, ref.m_u, rtol=0, atol=1e-7)

    series = np.genfromtxt(out / "series.csv", delimiter=",", names=True)
    assert series["t_ms"][0] == 5.0 and series["t_ms"][-1] == T_END

    m = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert m["restart"]["t_checkpoint_ms"] == 5.0
    assert m["restart"]["source"].endswith(_ckpt_name(5.0))


def test_start_from_preloaded_state_reproduces_full_run(ref_run, tmp_path):
    """
    ckpt_preloaded.npz — общая стартовая точка для серии: счёт с него
    обязан совпасть со счётом, включавшим преднагрузку.
    """
    from cardiac_em.io import read_checkpoint

    _, info, cont = _continue(ref_run, tmp_path, "ckpt_preloaded.npz")
    ref = read_checkpoint(ref_run["out"] / "ckpt_last.npz")

    assert info.stage == "preloaded" and info.t_start_ms == 0.0
    np.testing.assert_allclose(cont.e_state, ref.e_state, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cont.m_u, ref.m_u, rtol=0, atol=1e-7)


def test_finer_electric_mesh_needs_explicit_permission(ref_run, tmp_path):
    """
    Та же механическая сетка 6×2, электрическая вдвое мельче. Без
    разрешения — ошибка; с разрешением — счёт идёт, перенос отмечен.
    """
    from cardiac_em.io import restore_checkpoint
    from cardiac_em.runtime import Simulation

    out = _shared(tmp_path)
    cfg = _config(out, nx=2 * NX, ny=2 * NY, coarsening=8, t_end=8.0)
    src = ref_run["out"] / _ckpt_name(5.0)

    with pytest.raises(ValueError, match="allow_interp"):
        restore_checkpoint(Simulation(cfg), src)

    sim = Simulation(cfg)
    info = restore_checkpoint(sim, src, allow_interp=True)
    assert info.nearest_nodes.get("electric", 0) > 0
    assert "mechanical" not in info.nearest_nodes, "механическая сетка та же — перенос точный"
    assert any("ближайшему соседу" in w for w in sim.warnings)
    assert any("mesh.electric.nx" in w for w in sim.warnings), \
        "расхождение конфигураций названо поимённо"
    sim.run(t_start_ms=info.t_start_ms)
    assert sim.t_ms == 8.0


def test_other_geometry_is_rejected_even_with_interp(ref_run, tmp_path):
    from cardiac_em.io import restore_checkpoint
    from cardiac_em.runtime import Simulation

    sim = Simulation(_config(_shared(tmp_path), lx=8.0))
    with pytest.raises(ValueError, match="другой геометрии"):
        restore_checkpoint(sim, ref_run["out"] / _ckpt_name(5.0), allow_interp=True)


def test_other_cell_model_is_rejected(ref_run, tmp_path):
    from cardiac_em.io import read_checkpoint, restore_checkpoint
    from cardiac_em.runtime import Simulation

    data = read_checkpoint(ref_run["out"] / _ckpt_name(5.0))
    alien = dataclasses.replace(data, cell_model="tnnpm",
                                state_names=("V", "Nai", "Ki"))
    sim = Simulation(_config(_shared(tmp_path)))
    with pytest.raises(ValueError, match="модели клетки"):
        restore_checkpoint(sim, alien)


def test_reset_time_starts_from_zero(ref_run, tmp_path):
    from cardiac_em.io import restore_checkpoint
    from cardiac_em.runtime import Simulation

    sim = Simulation(_config(_shared(tmp_path),
                             restart=RestartConfig(reset_time=True)))
    info = restore_checkpoint(sim, ref_run["out"] / _ckpt_name(10.0))
    assert info.t_checkpoint_ms == 10.0 and info.t_start_ms == 0.0
    assert sim.t_ms == 0.0 and sim.is_preloaded


def test_restore_after_preload_is_refused(ref_run, tmp_path):
    from cardiac_em.io import restore_checkpoint
    from cardiac_em.runtime import Simulation

    sim = Simulation(_config(_shared(tmp_path)))
    sim.preload()
    u_before = sim.mechanics.u.x.array.copy()
    with pytest.raises(RuntimeError, match="свежесобранную"):
        restore_checkpoint(sim, ref_run["out"] / _ckpt_name(5.0))
    # отказ не должен портить состояние
    np.testing.assert_array_equal(sim.mechanics.u.x.array, u_before)


def test_forgotten_t_end_on_restart_is_explained(ref_run, tmp_path):
    from cardiac_em.io import restore_checkpoint
    from cardiac_em.runtime import Simulation

    sim = Simulation(_config(_shared(tmp_path), t_end=10.0))
    info = restore_checkpoint(sim, ref_run["out"] / _ckpt_name(10.0))
    with pytest.raises(ValueError, match="АБСОЛЮТНОЕ"):
        sim.run(t_start_ms=info.t_start_ms)


def test_legacy_checkpoint_continues(ref_run, tmp_path):
    """
    Чекпоинт в формате монолитной версии (собран из финального
    состояния эталона): состояние восстанавливается точно, время — с
    поправкой на dt, счёт продолжается.
    """
    from cardiac_em.io import read_checkpoint, restore_checkpoint
    from cardiac_em.runtime import Simulation

    out = _shared(tmp_path)
    ref = read_checkpoint(ref_run["out"] / "ckpt_last.npz")
    legacy = out / "legacy_ckpt.npz"
    from mpi4py import MPI
    if MPI.COMM_WORLD.rank == 0:
        z = np.zeros((len(ref.e_coords), 1))
        np.savez_compressed(
            legacy,
            coords_p1=np.hstack([ref.e_coords, z]),
            u_ap=ref.e_state[:, 0], v_ap=ref.e_state[:, 1],
            coords_u=np.hstack([ref.m_coords, np.zeros((len(ref.m_coords), 1))]),
            u=ref.m_u.ravel(),
            t=T_END - 0.05, step=int(round((T_END - 0.05) / 0.05)),
            mech_count=12, dt_e=0.05)
    _barrier()

    sim = Simulation(_config(out, t_end=T_END + 2.0))
    info = restore_checkpoint(sim, legacy)
    assert info.t_checkpoint_ms == pytest.approx(T_END, abs=1e-12)
    assert any("монолитной" in w for w in info.warnings)
    sim.run(t_start_ms=info.t_start_ms)
    assert sim.t_ms == T_END + 2.0


# ═══════════════════════════════════════════════════════════════════════
#  АВАРИЙНОЕ ЗАВЕРШЕНИЕ И ВЫБОР МОДЕЛЕЙ
# ═══════════════════════════════════════════════════════════════════════

def test_failure_is_recorded_in_manifest(tmp_path):
    """
    Сбой посреди счёта: run.json получает status=failed и текст ошибки,
    история в series.csv сохраняется до момента сбоя.

    Только на одном ранге: при сбое под MPI файлы XDMF намеренно не
    закрываются (закрытие коллективное), и уборка мусора в разном
    порядке на разных рангах в тестовом процессе могла бы зависнуть.
    """
    from mpi4py import MPI

    from cardiac_em.io import attach_outputs
    from cardiac_em.runtime import Observer, Simulation

    if MPI.COMM_WORLD.size > 1:
        pytest.skip("проверяется на одном ранге")

    class Boom(Observer):
        def on_mech_tick(self, sim, tick):
            if tick.mech_index == 3:
                raise RuntimeError("искусственный сбой")

    out = tmp_path
    sim = Simulation(_config(out, ckpt_every=0))
    attach_outputs(sim)
    sim.add_observer(Boom())
    sim.preload()
    with pytest.raises(RuntimeError, match="искусственный"):
        sim.run()

    m = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert m["status"] == "failed" and "искусственный сбой" in m["error"]
    series = np.genfromtxt(out / "series.csv", delimiter=",", names=True)
    np.testing.assert_allclose(series["t_ms"], [0.0, 1.0, 2.0, 3.0])


def test_unknown_model_name_fails_early(tmp_path):
    from cardiac_em.runtime import Simulation

    cfg = _config(_shared(tmp_path))
    cfg.cell_model = "no_such_model"
    with pytest.raises(KeyError, match="no_such_model"):
        Simulation(cfg)


def test_model_object_disagreeing_with_config_warns(tmp_path):
    from cardiac_em.models.cell.rogers_mcculloch import RogersMcCullochModel
    from cardiac_em.runtime import Simulation

    class Variant(RogersMcCullochModel):
        name = "rm_variant"

    sim = Simulation(_config(_shared(tmp_path)), cell_model=Variant())
    assert any("rm_variant" in w and "не воспроизведёт" in w for w in sim.warnings)
