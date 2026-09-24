"""
Интеграционные тесты связанного расчёта (Шаг 8).
=================================================

ТРЕБУЮТ DOLFINx.

    pytest tests/test_simulation.py -v
    mpirun -n 4 python -m pytest tests/test_simulation.py -q

Здесь впервые работает всё вместе: конфигурация → пара сеток → ткань →
монодомен → перенос → механика → наблюдатели. Отдельные звенья уже
проверены на шагах 1–7; эти тесты проверяют СОЕДИНЕНИЯ: что T_act
действительно доходит до механики, что перенос сохраняет силу на
каждом шаге, что зажатая граница не сдвигается, что две конфигурации
в одном процессе не мешают друг другу.

Полный прогон дорогой, поэтому он выполняется один раз на модуль
(фикстура `completed_run`), а тесты разбирают его историю.

О выборе преднагрузки
----------------------
Используется штатное λ_f = 1.2575, а не облегчённое 1.1. После зажима
активное напряжение неоднородно: активированная часть ткани
сокращается и растягивает неактивированную. Оценка одномерной
последовательной моделью при T_act = 30 кПа:

    λ₀ = 1.1     →  0.875 против 1.325   (разница 0.45)
    λ₀ = 1.2575  →  1.171 против 1.344   (разница 0.17)

Экспоненциальный член вдоль волокна в растянутом состоянии гораздо
жёстче, поэтому внутреннее перераспределение умеренное. При слабой
преднагрузке оно почти вдвое больше — это уже вопрос устойчивости
Ньютона, а не проверка связи.
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.config import (  # noqa: E402
    ActiveStressParams,
    DualMeshConfig,
    PreloadProtocol,
    RectangleMeshSpec,
    SimulationConfig,
    StimulusProtocol,
    TimeStepping,
    TissueBaseParams,
)

pytestmark = pytest.mark.fem


def _config(t_end_ms=40.0, stretch=1.2575, n_preload=6, t_max=120.0,
            mesh=None) -> SimulationConfig:
    """
    Малая, но физически осмысленная конфигурация: электрика 24×8
    (h = 0.25 мм), механика 6×2 (вложенное огрубление в 4 раза).
    Стимул — как в монолитной версии (амплитуда 20), заведомо
    надкритический.
    """
    return SimulationConfig(
        mesh=mesh or DualMeshConfig.nested(
            RectangleMeshSpec(nx=24, ny=8, lx_mm=6.0, ly_mm=2.0), coarsening=4),
        tissue_base=TissueBaseParams(active=ActiveStressParams(t_max=t_max)),
        stimulus=StimulusProtocol(times_ms=(1.0,), duration_ms=2.0,
                                  amplitude=20.0, rise_time_ms=0.5,
                                  decay_length_mm=0.5, x_max_mm=2.0),
        time=TimeStepping(dt_electric_ms=0.05, n_electric_per_mech=20,
                          t_end_ms=t_end_ms),
        preload=PreloadProtocol(stretch=stretch, n_steps=n_preload),
    )


def _boundary_mask(solver, spec):
    from cardiac_em.fem import dof_coordinates

    n = solver.V.dofmap.index_map.size_local
    xy = dof_coordinates(solver.V)[:n, :2]
    tol = 1e-9
    return ((np.abs(xy[:, 0]) < tol) | (np.abs(xy[:, 0] - spec.lx_mm) < tol)
            | (np.abs(xy[:, 1]) < tol) | (np.abs(xy[:, 1] - spec.ly_mm) < tol))


# ═══════════════════════════════════════════════════════════════════════
#  ОДИН ПОЛНЫЙ ПРОГОН НА ВЕСЬ МОДУЛЬ
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def completed_run():
    from cardiac_em.runtime import Simulation, TraceObserver

    cfg = _config()
    trace = TraceObserver()
    sim = Simulation(cfg, observers=[trace])

    sim.preload()
    n_owned = sim.mechanics.V.dofmap.index_map.size_local
    u_after_preload = sim.mechanics.u.x.array[:n_owned * 2].copy()

    schedule = sim.run()
    return {"sim": sim, "trace": trace, "schedule": schedule,
            "cfg": cfg, "u_after_preload": u_after_preload}


def test_run_reaches_the_end(completed_run):
    sim, cfg = completed_run["sim"], completed_run["cfg"]
    assert math.isclose(sim.t_ms, cfg.time.t_end_ms)


def test_observer_sees_every_mechanical_tick(completed_run):
    trace, schedule = completed_run["trace"], completed_run["schedule"]
    # начальная запись + по одной на каждый механический тик
    assert len(trace.records) == 1 + schedule.n_mech_ticks
    times = [r["t_ms"] for r in trace.records]
    assert times == sorted(times)
    assert times[-1] == completed_run["cfg"].time.t_end_ms


def test_preload_reached_uniform_stretch(completed_run):
    """Условия симметрии → однородное растяжение до заданного λ_f."""
    trace = completed_run["trace"]
    last = trace.preload[-1]
    assert len(trace.preload) == 6
    assert math.isclose(last["lambda_f_min"], 1.2575, rel_tol=1e-8)
    assert math.isclose(last["lambda_f_max"], 1.2575, rel_tol=1e-8)


def test_active_tension_develops_on_both_meshes(completed_run):
    """
    Волна запустилась, T_act появился на электрической сетке И дошёл до
    механической. Пик T_max/4 = 30 кПа; на крупной сетке он ниже из-за
    осреднения по ячейке, но заметно больше нуля.
    """
    recs = completed_run["trace"].records
    peak_e = max(r["t_act_electric_max"] for r in recs)
    peak_m = max(r["t_act_mech_max"] for r in recs)

    assert peak_e > 20.0, f"пик T_act на электрической сетке {peak_e:.2f} кПа"
    assert peak_m > 0.05 * peak_e, (
        f"T_act не дошёл до механики: пик {peak_m:.3f} кПа при {peak_e:.2f} на эл. сетке")
    assert peak_m <= peak_e + 1e-9, "осреднение не может превысить максимум источника"


def test_transfer_conserves_active_force_every_tick(completed_run):
    """
    Главная проверка СОЕДИНЕНИЯ: на каждом механическом шаге интеграл
    T_act по области одинаков на обеих сетках. Это означает, что перенос
    не просто работает в изоляции (шаг 4), а правильно встроен в цикл:
    передаётся именно текущее состояние, без сдвига на шаг и без потерь.
    """
    for r in completed_run["trace"].records:
        ie, im = r["t_act_electric_integral"], r["t_act_mech_integral"]
        assert math.isclose(ie, im, rel_tol=1e-10, abs_tol=1e-10), (
            f"t = {r['t_ms']} мс: ∫T_act эл. {ie:.6e}, мех. {im:.6e}")


def test_active_stress_rises_above_preload_level(completed_run):
    """
    Волокна вдоль x, ткань зажата: активное напряжение почти целиком
    добавляется к σ_xx. Требуем прирост хотя бы в половину пика T_act на
    механической сетке — с запасом на перераспределение внутри ткани.
    """
    recs = completed_run["trace"].records
    s0 = recs[0]["sigma_xx_max"]
    s_peak = max(r["sigma_xx_max"] for r in recs)
    peak_m = max(r["t_act_mech_max"] for r in recs)
    assert s_peak - s0 > 0.5 * peak_m, (
        f"активация почти не подняла напряжение: {s0:.2f} → {s_peak:.2f} кПа "
        f"при пике T_act {peak_m:.2f} кПа")


def test_clamped_boundary_does_not_move(completed_run):
    """
    Граничные условия зажима ставятся ОДИН раз после преднагрузки и не
    пересоздаются на каждом шаге (в отличие от монолитной версии). Раз
    так, граница обязана остаться ровно там, где её зажали.
    """
    sim, cfg = completed_run["sim"], completed_run["cfg"]
    spec = cfg.mesh.mechanical
    mask = _boundary_mask(sim.mechanics, spec)

    n = sim.mechanics.V.dofmap.index_map.size_local
    u_now = sim.mechanics.u.x.array[:n * 2].reshape(-1, 2)
    u_pre = completed_run["u_after_preload"].reshape(-1, 2)

    np.testing.assert_allclose(u_now[mask], u_pre[mask], atol=1e-12)


def test_interior_deformation_stays_physical(completed_run):
    """
    Внутри ткань перераспределяется (активированная часть сокращается,
    остальная растягивается), но в разумных пределах и без потери
    объёма сверх приближённой несжимаемости.
    """
    recs = completed_run["trace"].records
    lam_min = min(r["lambda_f_min"] for r in recs)
    lam_max = max(r["lambda_f_max"] for r in recs)
    J_min = min(r["J_min"] for r in recs)
    J_max = max(r["J_max"] for r in recs)

    assert 1.0 < lam_min <= 1.2575 + 1e-9, f"λ_f min = {lam_min:.4f}"
    assert 1.2575 - 1e-9 <= lam_max < 1.5, f"λ_f max = {lam_max:.4f}"
    assert 0.9 < J_min and J_max < 1.1, f"J ∈ [{J_min:.4f}, {J_max:.4f}]"


def test_newton_converged_with_few_iterations(completed_run):
    """
    0 итераций допустимо: пока волна не пришла, T_act = 0 и состояние уже
    равновесное. Важно, что Ньютон не раскачивается.
    """
    iters = [r["newton_iterations"] for r in completed_run["trace"].records[1:]]
    assert all(0 <= it < 20 for it in iters), f"итерации Ньютона: {iters}"
    assert max(iters) > 0, "Ньютон ни разу не работал — T_act не менялся?"


def test_potential_stays_bounded(completed_run):
    for r in completed_run["trace"].records:
        assert r["u_min"] >= -1e-9 and r["u_max"] <= 1.6


# ═══════════════════════════════════════════════════════════════════════
#  СБОРКА И КОНТРАКТ
# ═══════════════════════════════════════════════════════════════════════

def test_builds_without_warnings_for_clean_config():
    from cardiac_em.runtime import Simulation

    sim = Simulation(_config())
    assert sim.transfer.exact
    assert sim.warnings == []
    text = sim.summary()
    assert "Перенос T_act" in text and "Монодомен" in text


def test_run_without_preload_is_rejected():
    from cardiac_em.runtime import Simulation

    sim = Simulation(_config(t_end_ms=5.0))
    with pytest.raises(RuntimeError, match="преднагрузка"):
        sim.run()


def test_preload_cannot_run_twice():
    from cardiac_em.runtime import Simulation

    sim = Simulation(_config(stretch=1.0))
    sim.preload()
    with pytest.raises(RuntimeError, match="уже выполнена"):
        sim.preload()


def test_trivial_preload_skips_stretching():
    from cardiac_em.runtime import Simulation, TraceObserver

    trace = TraceObserver()
    sim = Simulation(_config(stretch=1.0), observers=[trace])
    sim.preload()

    assert trace.preload == [], "при λ_f = 1 шагов растяжения быть не должно"
    lo, hi = sim.mechanics.value_range(sim.mechanics.fiber_stretch())
    assert math.isclose(lo, 1.0, abs_tol=1e-10) and math.isclose(hi, 1.0, abs_tol=1e-10)


def test_non_nested_meshes_produce_a_warning():
    from cardiac_em.runtime import Simulation

    mesh = DualMeshConfig(
        electric=RectangleMeshSpec(nx=24, ny=8, lx_mm=6.0, ly_mm=2.0),
        mechanical=RectangleMeshSpec(nx=5, ny=2, lx_mm=6.0, ly_mm=2.0))
    sim = Simulation(_config(mesh=mesh))

    assert not sim.transfer.exact
    assert any("приближённый" in w for w in sim.warnings)


def test_two_configurations_in_one_process_are_independent():
    """
    Следствие отказа от глобальных параметров: две конфигурации подряд
    в одном процессе не влияют друг на друга.

    Электрика от механики не зависит (обратной связи нет), поэтому при
    вдвое меньшем T_max электрическое T_act обязано быть ровно вдвое
    меньше. Если бы вторая конфигурация унаследовала что-то от первой,
    отношение бы поплыло.
    """
    from cardiac_em.runtime import Simulation, TraceObserver

    peaks = {}
    for t_max in (120.0, 60.0):
        trace = TraceObserver()
        sim = Simulation(_config(t_end_ms=10.0, t_max=t_max, stretch=1.0),
                         observers=[trace])
        sim.preload()
        sim.run()
        peaks[t_max] = max(r["t_act_electric_max"] for r in trace.records)

    assert peaks[120.0] > 1.0, "за 10 мс волна должна была запуститься"
    assert math.isclose(peaks[120.0] / peaks[60.0], 2.0, rel_tol=1e-10), (
        f"отношение пиков {peaks[120.0] / peaks[60.0]:.12f}, ожидалось ровно 2")


def test_console_observer_prints_progress():
    from cardiac_em.runtime import ConsoleObserver, Simulation

    stream = io.StringIO()
    sim = Simulation(_config(t_end_ms=4.0, stretch=1.0),
                     observers=[ConsoleObserver(every_mech_steps=2, stream=stream)])
    sim.preload()
    sim.run()

    if sim.comm.rank == 0:
        out = stream.getvalue()
        assert "Расписание" in out
        assert "готово за" in out
        assert out.count("\n") >= 5
