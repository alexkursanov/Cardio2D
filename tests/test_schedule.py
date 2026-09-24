"""
Тесты расписания шагов (Шаг 8).
================================

DOLFINx НЕ ТРЕБУЕТСЯ.

    python tests/test_schedule.py
    pytest tests/test_schedule.py -v

Расписание — место классических ошибок на единицу: какой шаг первый
механический, где кончается счёт, не теряется ли фаза при продолжении
с чекпоинта, не срабатывает ли снимок дважды. Здесь всё это проверяется
перебором тиков, без солверов.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.config import OutputConfig, TimeStepping  # noqa: E402
from cardiac_em.runtime.schedule import Schedule  # noqa: E402


def _raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            f"ожидалось {exc_type.__name__}, получено {type(e).__name__}: {e}") from e
    raise AssertionError(f"ожидалось {exc_type.__name__}, но исключения не было")


def _ts(t_end=10.0, dt=0.05, n=20):
    return TimeStepping(dt_electric_ms=dt, n_electric_per_mech=n, t_end_ms=t_end)


# ═══════════════════════════════════════════════════════════════════════
#  ШАГИ И МЕХАНИКА
# ═══════════════════════════════════════════════════════════════════════

def test_tick_count_and_endpoints():
    sched = Schedule(_ts(t_end=10.0))
    ticks = list(sched)

    assert len(ticks) == len(sched) == 200
    assert ticks[0].t_ms == 0.0 and ticks[0].t_end_ms == 0.05
    assert ticks[-1].t_end_ms == 10.0, "счёт обязан кончаться ровно в t_end"
    assert ticks[-1].is_last and not any(t.is_last for t in ticks[:-1])


def test_mechanics_at_exact_multiples_of_dt_mech():
    """
    Механика — в моменты dt_мех, 2·dt_мех, … а не со сдвигом на один
    электрический шаг, как в монолитной версии.
    """
    ticks = [t for t in Schedule(_ts(t_end=10.0)) if t.is_mech]

    assert [t.t_end_ms for t in ticks] == [float(i) for i in range(1, 11)]
    assert [t.mech_index for t in ticks] == list(range(1, 11))


def test_last_tick_is_always_mechanical():
    """Если t_end не кратно dt_мех, финальное состояние всё равно согласовано."""
    ticks = list(Schedule(_ts(t_end=10.5)))
    mech = [t for t in ticks if t.is_mech]

    assert ticks[-1].is_mech
    assert ticks[-1].t_end_ms == 10.5
    assert len(mech) == 11
    assert mech[-1].mech_index == 11


def test_non_mech_ticks_have_no_mech_index():
    for t in Schedule(_ts(t_end=2.0)):
        assert (t.mech_index is None) == (not t.is_mech)


def test_no_float_drift_on_long_run():
    """
    Время считается как k·dt, а не накоплением: иначе за 30 000 шагов
    конец поплыл бы, и последний механический шаг мог бы не совпасть
    с t_end.
    """
    sched = Schedule(_ts(t_end=1500.0, dt=0.05, n=20))
    ticks = list(sched)
    mech = [t for t in ticks if t.is_mech]

    assert len(ticks) == 30000
    assert len(mech) == 1500
    assert ticks[-1].t_end_ms == 1500.0
    assert all(t.t_end_ms == float(t.mech_index) for t in mech)


# ═══════════════════════════════════════════════════════════════════════
#  ЗАПИСЬ И ЧЕКПОИНТЫ
# ═══════════════════════════════════════════════════════════════════════

def test_write_cadence_includes_final_state():
    out = OutputConfig(save_every_mech_steps=3, ckpt_every_mech_steps=0)
    writes = [t.mech_index for t in Schedule(_ts(t_end=10.0), out) if t.write_fields]
    assert writes == [3, 6, 9, 10], "запись каждые 3 мех. шага + финальная"


def test_writes_only_on_mechanical_ticks():
    out = OutputConfig(save_every_mech_steps=1, ckpt_every_mech_steps=1)
    for t in Schedule(_ts(t_end=3.0), out):
        if t.write_fields or t.checkpoint:
            assert t.is_mech, "запись на немеханическом тике — состояние рассогласовано"


def test_checkpoint_cadence_and_final():
    out = OutputConfig(ckpt_every_mech_steps=4)
    ckpts = [t.mech_index for t in Schedule(_ts(t_end=10.0), out) if t.checkpoint]
    assert ckpts == [4, 8, 10]


def test_checkpoints_disabled_means_none_at_all():
    out = OutputConfig(ckpt_every_mech_steps=0)
    assert not any(t.checkpoint for t in Schedule(_ts(t_end=10.0), out))


def test_without_output_config_nothing_is_written():
    ticks = list(Schedule(_ts(t_end=5.0)))
    assert not any(t.write_fields or t.checkpoint or t.snapshots for t in ticks)


# ═══════════════════════════════════════════════════════════════════════
#  СНИМКИ
# ═══════════════════════════════════════════════════════════════════════

def test_snapshot_placement():
    """
    Снимок берётся с первого согласованного состояния не раньше
    заказанного момента.
    """
    out = OutputConfig(snapshot_times_ms=(0.0, 2.5, 5.0, 100.0))
    sched = Schedule(_ts(t_end=10.0), out)

    placed = {ts: t.t_end_ms for t in sched for ts in t.snapshots}

    assert sched.initial_snapshots == (0.0,), "t = 0 — начальное состояние"
    assert placed == {2.5: 3.0, 5.0: 5.0}
    assert sched.unreachable_snapshots == (100.0,)


def test_each_snapshot_fires_once():
    out = OutputConfig(snapshot_times_ms=(1.0, 1.0001, 4.0))
    fired = [ts for t in Schedule(_ts(t_end=10.0), out) for ts in t.snapshots]
    assert sorted(fired) == [1.0, 1.0001, 4.0]


def test_snapshots_land_on_mechanical_ticks():
    out = OutputConfig(snapshot_times_ms=(0.3, 1.7, 6.66))
    for t in Schedule(_ts(t_end=10.0), out):
        if t.snapshots:
            assert t.is_mech


# ═══════════════════════════════════════════════════════════════════════
#  ПРОДОЛЖЕНИЕ С ЧЕКПОИНТА
# ═══════════════════════════════════════════════════════════════════════

def test_restart_preserves_global_phase():
    """
    При старте с 5 мс номера механических шагов и частоты записи должны
    совпадать с теми, что были бы при счёте с нуля.
    """
    out = OutputConfig(save_every_mech_steps=3, ckpt_every_mech_steps=4)
    full = {t.index: t for t in Schedule(_ts(t_end=10.0), out)}
    resumed = list(Schedule(_ts(t_end=10.0), out, t_start_ms=5.0))

    assert resumed[0].t_ms == 5.0
    assert len(resumed) == 100
    for t in resumed:
        ref = full[t.index]
        assert (t.is_mech, t.mech_index, t.write_fields, t.checkpoint) == \
               (ref.is_mech, ref.mech_index, ref.write_fields, ref.checkpoint), (
            f"расхождение на шаге {t.index}")


def test_restart_initial_snapshot():
    out = OutputConfig(snapshot_times_ms=(3.0, 5.0, 7.0))
    sched = Schedule(_ts(t_end=10.0), out, t_start_ms=5.0)
    assert sched.initial_snapshots == (3.0, 5.0)
    fired = [ts for t in sched for ts in t.snapshots]
    assert fired == [7.0]


def test_start_off_grid_rejected():
    _raises(ValueError, Schedule, _ts(), None, 0.0301)


def test_nothing_to_compute_rejected():
    """Классическая ошибка при рестарте: забыли поднять t_end."""
    _raises(ValueError, Schedule, _ts(t_end=10.0), None, 10.0)
    _raises(ValueError, Schedule, _ts(t_end=10.0), None, 15.0)


def test_summary_renders():
    text = Schedule(_ts(t_end=10.0)).summary()
    assert "200" in text and "10" in text


# ═══════════════════════════════════════════════════════════════════════

def _main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:  # noqa: BLE001
            failed.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} тестов прошло")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
