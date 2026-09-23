"""
Тесты слоя конфигурации.
=================================

DOLFINx не требуется — можно запускать где угодно.

    python -m pytest tests/test_config.py -v      # если есть pytest
    python tests/test_config.py                    # без pytest

Тесты проверяют три вещи:
  1. производные величины считаются правильно (шаги сетки, вложенность);
  2. НЕВЕРНЫЕ конфигурации падают с понятной ошибкой, а не молча
     работают неправильно (перекрывающиеся стимулы, опечатки в ключах
     переопределений, несогласованные области сеток);
  3. сериализация делает полный круг без потери данных.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.config import (  # noqa: E402
    CircleRegion,
    ConductionParams,
    CustomRegion,
    DualMeshConfig,
    OutputConfig,
    PassiveMechParams,
    PreloadProtocol,
    RectRegion,
    RectangleMeshSpec,
    RestartConfig,
    SimulationConfig,
    StimulusProtocol,
    TimeStepping,
    TissueBaseParams,
    load_regions_json,
)


def _raises(exc_type, fn, *args, **kwargs):
    """Мини-замена pytest.raises, чтобы файл работал и без pytest."""
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            f"ожидалось {exc_type.__name__}, получено "
            f"{type(e).__name__}: {e}") from e
    raise AssertionError(f"ожидалось {exc_type.__name__}, но исключения не было")


# ═══════════════════════════════════════════════════════════════════════
#  СЕТКИ
# ═══════════════════════════════════════════════════════════════════════

def test_rectangle_derived_quantities():
    spec = RectangleMeshSpec(nx=40, ny=20, lx_mm=10.0, ly_mm=10.0)
    assert spec.n_cells == 800
    assert spec.n_vertices == 41 * 21
    assert math.isclose(spec.hx_mm, 0.25)
    assert math.isclose(spec.hy_mm, 0.5)
    assert math.isclose(spec.h_min_mm, 0.25)


def test_rectangle_triangle_cell_count():
    quad = RectangleMeshSpec(nx=10, ny=10, cell_type="quadrilateral")
    tri = RectangleMeshSpec(nx=10, ny=10, cell_type="triangle")
    assert tri.n_cells == 2 * quad.n_cells


def test_rectangle_rejects_bad_input():
    _raises(ValueError, RectangleMeshSpec, nx=0, ny=10)
    _raises(ValueError, RectangleMeshSpec, nx=10, ny=10, lx_mm=-1.0)
    _raises(ValueError, RectangleMeshSpec, nx=10, ny=10, cell_type="hexagon")


def test_coarsening_requires_integer_ratio():
    spec = RectangleMeshSpec(nx=40, ny=40)
    coarse = spec.coarsened(4)
    assert (coarse.nx, coarse.ny) == (10, 10)
    assert coarse.domain_matches(spec)
    # 40 не делится на 3 — должно быть сказано прямо, а не округлено молча
    _raises(ValueError, spec.coarsened, 3)


def test_nested_pair_is_detected():
    cfg = DualMeshConfig.nested(RectangleMeshSpec(nx=80, ny=80), coarsening=4)
    assert (cfg.mechanical.nx, cfg.mechanical.ny) == (20, 20)

    nest = cfg.nesting()
    assert nest.is_nested
    assert (nest.kx, nest.ky) == (4, 4)
    assert nest.cells_per_mech_cell == 16
    assert math.isclose(cfg.refinement_ratio, 4.0)


def test_non_nested_pair_reports_reason():
    cfg = DualMeshConfig(
        electric=RectangleMeshSpec(nx=80, ny=80),
        mechanical=RectangleMeshSpec(nx=25, ny=25))
    nest = cfg.nesting()
    assert not nest.is_nested
    assert nest.cells_per_mech_cell is None
    assert "не кратны" in nest.reason


def test_uniform_pair_is_nested_with_factor_one():
    spec = RectangleMeshSpec(nx=40, ny=40)
    nest = DualMeshConfig.uniform(spec).nesting()
    assert nest.is_nested and (nest.kx, nest.ky) == (1, 1)


def test_mismatched_domains_rejected():
    _raises(
        ValueError, DualMeshConfig,
        electric=RectangleMeshSpec(nx=40, ny=40, lx_mm=10.0, ly_mm=10.0),
        mechanical=RectangleMeshSpec(nx=20, ny=20, lx_mm=12.0, ly_mm=10.0))


def test_anisotropic_nesting():
    """Разные коэффициенты по x и y — законный случай."""
    cfg = DualMeshConfig(
        electric=RectangleMeshSpec(nx=80, ny=40),
        mechanical=RectangleMeshSpec(nx=20, ny=20))
    nest = cfg.nesting()
    assert nest.is_nested and (nest.kx, nest.ky) == (4, 2)


# ═══════════════════════════════════════════════════════════════════════
#  ПАРАМЕТРЫ ТКАНИ
# ═══════════════════════════════════════════════════════════════════════

def test_flat_dict_has_historical_key_names():
    """Имена ключей должны совпадать со старым форматом JSON областей."""
    flat = TissueBaseParams().to_flat_dict()
    expected = {"AP_c1", "AP_c2", "AP_a", "AP_b", "AP_d",
                "D_LONG", "D_TRANS", "FIBER_ANGLE_DEG", "T_MAX",
                "MU", "B_ISO", "MU_F", "B_F", "KAPPA"}
    assert set(flat) == expected
    assert math.isclose(flat["AP_c1"], 0.26)
    assert math.isclose(flat["T_MAX"], 120.0)
    assert math.isclose(flat["KAPPA"], 100.0)


def test_electric_and_mechanical_key_split_covers_everything():
    """
    Разделение ключей на электрические и механические используется при
    построении полей параметров на двух разных сетках — ни один ключ не
    должен потеряться.
    """
    all_keys = set(TissueBaseParams.flat_keys())
    split = set(TissueBaseParams.ELECTRIC_KEYS) | set(TissueBaseParams.MECHANICAL_KEYS)
    assert split == all_keys
    # угол волокон нужен обеим задачам
    assert "FIBER_ANGLE_DEG" in TissueBaseParams.ELECTRIC_KEYS
    assert "FIBER_ANGLE_DEG" in TissueBaseParams.MECHANICAL_KEYS


def test_negative_diffusion_rejected():
    _raises(ValueError, ConductionParams, d_long=-0.1)


def test_zero_kappa_rejected():
    _raises(ValueError, PassiveMechParams, kappa=0.0)


# ═══════════════════════════════════════════════════════════════════════
#  ОБЛАСТИ
# ═══════════════════════════════════════════════════════════════════════

def test_rect_region_contains():
    reg = RectRegion(x0=6.0, x1=10.0, y0=0.0, y1=10.0,
                     overrides={"T_MAX": 20.0}, name="ischemia")
    x = np.array([5.9, 6.0, 8.0, 10.0, 10.1])
    y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    assert list(reg.contains(x, y)) == [False, True, True, True, False]


def test_circle_region_contains():
    reg = CircleRegion(cx=5.0, cy=5.0, r=1.0, name="scar")
    x = np.array([5.0, 5.9, 6.0, 6.1])
    y = np.array([5.0, 5.0, 5.0, 5.0])
    assert list(reg.contains(x, y)) == [True, True, True, False]


def test_typo_in_override_key_is_rejected():
    """
    Главная причина, по которой ключи валидируются: раньше опечатка
    молча не делала ничего, и «ишемическая зона» оказывалась здоровой.
    """
    _raises(ValueError, RectRegion, x0=0, x1=1, y0=0, y1=1,
            overrides={"AP_C1": 0.1})          # регистр
    _raises(ValueError, RectRegion, x0=0, x1=1, y0=0, y1=1,
            overrides={"D_Long": 0.1})          # регистр
    _raises(ValueError, RectRegion, x0=0, x1=1, y0=0, y1=1,
            overrides={"TMAX": 0.0})            # опечатка


def test_valid_override_keys_accepted():
    reg = RectRegion(x0=0, x1=1, y0=0, y1=1,
                     overrides={"AP_c1": 0.13, "D_LONG": 0.05,
                                "T_MAX": 20.0, "MU": 5.0})
    assert reg.overrides["T_MAX"] == 20.0


def test_bad_geometry_rejected():
    _raises(ValueError, RectRegion, x0=5.0, x1=1.0, y0=0.0, y1=1.0)
    _raises(ValueError, CircleRegion, cx=0.0, cy=0.0, r=0.0)


def test_custom_region_works_but_does_not_serialize():
    reg = CustomRegion(predicate=lambda x, y: x + y > 10.0, name="diag")
    x = np.array([1.0, 9.0])
    y = np.array([1.0, 9.0])
    assert list(reg.contains(x, y)) == [False, True]
    assert reg.is_serializable is False
    _raises(TypeError, reg.to_dict)


def test_custom_region_requires_predicate():
    _raises(ValueError, CustomRegion)


def test_region_json_roundtrip():
    regions = [
        RectRegion(x0=6, x1=10, y0=0, y1=10, name="ischemia",
                   overrides={"AP_c1": 0.13, "D_LONG": 0.05}),
        CircleRegion(cx=8, cy=5, r=1.0, name="scar",
                     overrides={"T_MAX": 0.0, "MU": 5.0}),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "regions.json"
        path.write_text(json.dumps([r.to_dict() for r in regions]),
                        encoding="utf-8")
        loaded = load_regions_json(path)

    assert len(loaded) == 2
    assert isinstance(loaded[0], RectRegion) and loaded[0].name == "ischemia"
    assert isinstance(loaded[1], CircleRegion)
    assert math.isclose(loaded[1].r, 1.0)
    assert loaded[1].overrides["T_MAX"] == 0.0


def test_region_json_reports_bad_entry_position():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.json"
        path.write_text(json.dumps([
            {"shape": "rect", "params": {"x0": 0, "x1": 1, "y0": 0, "y1": 1}},
            {"shape": "trapezoid", "params": {}},
        ]), encoding="utf-8")
        try:
            load_regions_json(path)
            raise AssertionError("ожидалась ошибка на второй записи")
        except ValueError as e:
            assert "запись 1" in str(e)


# ═══════════════════════════════════════════════════════════════════════
#  ПРОТОКОЛЫ
# ═══════════════════════════════════════════════════════════════════════

def test_stimulus_multiple_times():
    stim = StimulusProtocol(times_ms=(0.0, 1000.0, 2000.0), duration_ms=2.0)
    assert stim.active_onset(0.5) == 0.0
    assert stim.active_onset(500.0) is None
    assert stim.active_onset(1001.0) == 1000.0
    assert stim.active_onset(2002.0) == 2000.0
    assert stim.active_onset(2002.1) is None


def test_stimulus_envelope_shape():
    stim = StimulusProtocol(times_ms=(0.0,), duration_ms=2.0, rise_time_ms=0.5)
    assert math.isclose(stim.envelope(0.0), 0.0)
    assert math.isclose(stim.envelope(0.25), 0.5)     # середина фронта
    assert math.isclose(stim.envelope(1.0), 1.0)      # плато
    assert math.isclose(stim.envelope(1.75), 0.5)     # середина спада
    assert math.isclose(stim.envelope(2.0), 0.0)
    assert math.isclose(stim.envelope(5.0), 0.0)      # вне импульса


def test_overlapping_stimuli_rejected():
    """Раньше перекрытие молча приводило к срабатыванию только первого."""
    _raises(ValueError, StimulusProtocol, times_ms=(0.0, 1.0), duration_ms=2.0)


def test_unsorted_stimuli_rejected():
    _raises(ValueError, StimulusProtocol, times_ms=(1000.0, 0.0))


def test_stimulus_rise_must_fit_in_pulse():
    _raises(ValueError, StimulusProtocol, duration_ms=1.0, rise_time_ms=0.6)


def test_time_stepping_derived():
    ts = TimeStepping(dt_electric_ms=0.05, n_electric_per_mech=20, t_end_ms=1000.0)
    assert math.isclose(ts.dt_mech_ms, 1.0)
    assert ts.n_electric_steps == 20001
    assert ts.step_index(500.0) == 10000


def test_time_stepping_rejects_bad_input():
    _raises(ValueError, TimeStepping, dt_electric_ms=0.0)
    _raises(ValueError, TimeStepping, n_electric_per_mech=0)
    _raises(ValueError, TimeStepping, t_end_ms=-1.0)


def test_preload_trivial_detection():
    assert PreloadProtocol(stretch=1.0).is_trivial
    assert not PreloadProtocol(stretch=1.2575).is_trivial
    _raises(ValueError, PreloadProtocol, stretch=0.0)
    _raises(ValueError, PreloadProtocol, n_steps=0)


# ═══════════════════════════════════════════════════════════════════════
#  КОРНЕВОЙ КОНФИГ
# ═══════════════════════════════════════════════════════════════════════

def test_default_config_builds_and_is_clean():
    cfg = SimulationConfig.default(nx_electric=40, coarsening=2)
    assert cfg.mesh.nesting().is_nested
    assert cfg.mesh.mechanical.nx == 20
    # снимки по умолчанию не заданы, стимул в пределах счёта — тревог быть не должно
    assert cfg.check() == []


def test_check_warns_on_swapped_meshes():
    cfg = SimulationConfig(mesh=DualMeshConfig(
        electric=RectangleMeshSpec(nx=20, ny=20),
        mechanical=RectangleMeshSpec(nx=40, ny=40)))
    warnings = cfg.check()
    assert any("МЕЛЬЧЕ" in w for w in warnings)


def test_check_warns_on_late_stimulus():
    cfg = SimulationConfig(
        mesh=DualMeshConfig.nested(RectangleMeshSpec(nx=40, ny=40)),
        stimulus=StimulusProtocol(times_ms=(50.0, 5000.0)),
        time=TimeStepping(t_end_ms=1000.0))
    assert any("позже конца счёта" in w for w in cfg.check())


def test_check_warns_on_non_nested_meshes():
    cfg = SimulationConfig(mesh=DualMeshConfig(
        electric=RectangleMeshSpec(nx=80, ny=80),
        mechanical=RectangleMeshSpec(nx=25, ny=25)))
    assert any("не вложены" in w for w in cfg.check())


def test_check_warns_on_too_narrow_stimulus_zone():
    cfg = SimulationConfig(
        mesh=DualMeshConfig.nested(RectangleMeshSpec(nx=10, ny=10, lx_mm=10.0)),
        stimulus=StimulusProtocol(x_max_mm=0.5))   # hx = 1 мм
    assert any("стимульная зона" in w for w in cfg.check())


def test_config_json_roundtrip():
    cfg = SimulationConfig(
        mesh=DualMeshConfig.nested(
            RectangleMeshSpec(nx=80, ny=80, lx_mm=12.0, ly_mm=8.0), coarsening=4),
        regions=[
            RectRegion(x0=6, x1=12, y0=0, y1=8, name="ischemia",
                       overrides={"AP_c1": 0.13, "D_LONG": 0.05, "T_MAX": 20.0}),
            CircleRegion(cx=8, cy=4, r=1.0, name="scar",
                         overrides={"T_MAX": 0.0, "MU": 5.0}),
        ],
        stimulus=StimulusProtocol(times_ms=(0.0, 1000.0), duration_ms=2.0),
        time=TimeStepping(dt_electric_ms=0.02, n_electric_per_mech=50,
                          t_end_ms=2000.0),
        preload=PreloadProtocol(stretch=1.1, n_steps=5),
        output=OutputConfig(out_dir=Path("out_x"), ckpt_every_mech_steps=50,
                            snapshot_times_ms=(0.0, 100.0)),
        restart=RestartConfig(checkpoint_path=Path("prev/ckpt_last.npz"),
                              reset_time=True),
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        cfg.to_json(path)
        back = SimulationConfig.from_json(path)

    assert back.mesh.electric.nx == 80
    assert back.mesh.mechanical.nx == 20
    assert math.isclose(back.mesh.electric.lx_mm, 12.0)
    assert len(back.regions) == 2
    assert back.regions[0].name == "ischemia"
    assert math.isclose(back.regions[1].r, 1.0)
    assert back.stimulus.times_ms == (0.0, 1000.0)
    assert math.isclose(back.time.dt_mech_ms, 1.0)
    assert math.isclose(back.preload.stretch, 1.1)
    assert back.output.ckpt_every_mech_steps == 50
    assert back.output.snapshot_times_ms == (0.0, 100.0)
    assert back.restart.reset_time is True
    assert back.restart.checkpoint_path == Path("prev/ckpt_last.npz")


def test_config_with_custom_region_refuses_to_serialize_silently():
    """
    Ключевая гарантия: неоднородность, которая не переживёт сохранение,
    не должна исчезать молча.
    """
    cfg = SimulationConfig(
        mesh=DualMeshConfig.nested(RectangleMeshSpec(nx=40, ny=40)),
        regions=[CustomRegion(predicate=lambda x, y: x > 5.0, name="half")])

    assert len(cfg.unserializable_regions()) == 1
    _raises(TypeError, cfg.to_dict)

    # ...но при явном согласии — пропускаем
    d = cfg.to_dict(skip_unserializable=True)
    assert d["regions"] == []


def test_tissue_params_roundtrip_preserves_values():
    base = TissueBaseParams(
        conduction=ConductionParams(d_long=0.5, d_trans=0.05,
                                    fiber_angle_deg=45.0),
        passive=PassiveMechParams(mu=2.0, kappa=200.0))
    back = TissueBaseParams.from_dict(base.to_dict())
    assert math.isclose(back.conduction.d_long, 0.5)
    assert math.isclose(back.conduction.fiber_angle_deg, 45.0)
    assert math.isclose(back.passive.mu, 2.0)
    assert math.isclose(back.passive.kappa, 200.0)
    assert back.to_flat_dict() == base.to_flat_dict()


def test_summary_renders():
    cfg = SimulationConfig.default()
    text = cfg.summary()
    assert "Сетка эл." in text and "Вложенность" in text
    assert "Стимулы" in text


# ═══════════════════════════════════════════════════════════════════════
#  ЗАПУСК БЕЗ PYTEST
# ═══════════════════════════════════════════════════════════════════════

def _main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:  # noqa: BLE001
            failed.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")

    print(f"\n{len(tests) - len(failed)}/{len(tests)} тестов прошло")
    if failed:
        print("\nНе прошли:")
        for name, e in failed:
            print(f"  {name}: {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
