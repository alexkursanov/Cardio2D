"""
Тесты переноса полей между сетками (Шаг 4).
============================================

ТРЕБУЮТ DOLFINx.

    pytest tests/test_transfer.py -v
    mpirun -n 4 python -m pytest tests/test_transfer.py -q

Здесь проверяется не только «код не падает», но и заявленное СВОЙСТВО
каждой стратегии:

  * осреднение ТОЧНО воспроизводит линейное поле в центрах крупных
    ячеек и ТОЧНО сохраняет интеграл по области — это и есть смысл
    слова «точный» применительно к нему;

  * ближайшая ячейка на фронте заметно хуже осреднения — тест меряет
    разницу численно, а не принимает её на веру. Если когда-нибудь
    разница исчезнет, значит либо фронт в тесте стал слишком пологим,
    либо в осреднении что-то сломалось;

  * результат не зависит от числа MPI-рангов — сравнение идёт с
    аналитически известным ответом, поэтому тест осмыслен и на одном
    ранге, и на восьми.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiac_em.config import DualMeshConfig, RectangleMeshSpec  # noqa: E402

pytestmark = pytest.mark.fem


# ═══════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНОЕ
# ═══════════════════════════════════════════════════════════════════════

def _spaces(cfg):
    """Построить пару сеток и DG0-пространства на них."""
    import dolfinx.fem as fem

    from cardiac_em.fem import build_mesh_pair

    pair = build_mesh_pair(cfg)
    DG0_e = fem.functionspace(pair.electric, ("DG", 0))
    DG0_m = fem.functionspace(pair.mechanical, ("DG", 0))
    return pair, DG0_e, DG0_m


def _sample(space, func):
    """Точечные значения func(x, y) в центрах владеемых ячеек."""
    from cardiac_em.fem import owned_dof_coordinates

    xy = owned_dof_coordinates(space)
    if xy.shape[0] == 0:
        return np.zeros(0)
    return func(xy[:, 0], xy[:, 1])


def _global_integral(space, values_owned, cell_area) -> float:
    """∫ f dΩ по всей области (кусочно-постоянное поле)."""
    from mpi4py import MPI

    local = float(np.sum(values_owned) * cell_area)
    return space.mesh.comm.allreduce(local, op=MPI.SUM)


def _global_max_abs(comm, values) -> float:
    from mpi4py import MPI

    local = float(np.max(np.abs(values))) if len(values) else 0.0
    return comm.allreduce(local, op=MPI.MAX)


# ═══════════════════════════════════════════════════════════════════════
#  ВЫБОР СТРАТЕГИИ
# ═══════════════════════════════════════════════════════════════════════

def test_auto_picks_averaging_for_nested_meshes():
    from cardiac_em.coupling import AveragingTransfer, make_transfer

    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=12, ny=12, lx_mm=6.0, ly_mm=6.0), coarsening=3)
    _, DG0_e, DG0_m = _spaces(cfg)

    tr = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical)
    assert isinstance(tr, AveragingTransfer)
    assert tr.exact
    assert tr.cells_per_target == 9
    assert "точный" in tr.describe()


def test_auto_falls_back_to_nearest_for_non_nested():
    from cardiac_em.coupling import NearestCellTransfer, make_transfer

    cfg = DualMeshConfig(
        electric=RectangleMeshSpec(nx=12, ny=12, lx_mm=6.0, ly_mm=6.0),
        mechanical=RectangleMeshSpec(nx=5, ny=5, lx_mm=6.0, ly_mm=6.0))
    _, DG0_e, DG0_m = _spaces(cfg)

    tr = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical)
    assert isinstance(tr, NearestCellTransfer)
    assert not tr.exact
    assert "приближённый" in tr.describe()
    # расстояние до ближайшей ячейки — порядка шага сетки, не больше
    assert tr.max_distance < max(cfg.mechanical.hx_mm, cfg.mechanical.hy_mm)


def test_averaging_refuses_non_nested_meshes():
    from cardiac_em.coupling import make_transfer

    cfg = DualMeshConfig(
        electric=RectangleMeshSpec(nx=12, ny=12),
        mechanical=RectangleMeshSpec(nx=5, ny=5))
    _, DG0_e, DG0_m = _spaces(cfg)

    with pytest.raises(ValueError, match="точное осреднение невозможно"):
        make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical,
                      strategy="averaging")


def test_unknown_strategy_rejected():
    from cardiac_em.coupling import make_transfer

    cfg = DualMeshConfig.nested(RectangleMeshSpec(nx=8, ny=8), coarsening=2)
    _, DG0_e, DG0_m = _spaces(cfg)

    with pytest.raises(ValueError, match="неизвестная стратегия"):
        make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical,
                      strategy="магия")


def test_anisotropic_nesting_supported():
    """kx ≠ ky — законный случай вложенности."""
    from cardiac_em.coupling import AveragingTransfer, make_transfer

    cfg = DualMeshConfig(
        electric=RectangleMeshSpec(nx=12, ny=8, lx_mm=6.0, ly_mm=4.0),
        mechanical=RectangleMeshSpec(nx=3, ny=4, lx_mm=6.0, ly_mm=4.0))
    _, DG0_e, DG0_m = _spaces(cfg)

    tr = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical)
    assert isinstance(tr, AveragingTransfer)
    assert tr.cells_per_target == (12 * 8) // (3 * 4)   # 4·2 = 8


# ═══════════════════════════════════════════════════════════════════════
#  СОГЛАСОВАННОСТЬ: КОНСТАНТА ПЕРЕНОСИТСЯ ТОЧНО
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("strategy", ["averaging", "nearest"])
def test_constant_field_is_preserved(strategy):
    """
    Минимальное требование к любому оператору переноса: константа
    обязана остаться той же константой. Нарушение означает ошибку в
    весах или в сопоставлении ячеек.
    """
    from cardiac_em.coupling import make_transfer

    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=16, ny=16, lx_mm=8.0, ly_mm=8.0), coarsening=4)
    _, DG0_e, DG0_m = _spaces(cfg)
    tr = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical,
                       strategy=strategy)

    src = np.full(DG0_e.dofmap.index_map.size_local, 42.0)
    dst = tr.apply(src)

    assert dst.shape == (DG0_m.dofmap.index_map.size_local,)
    np.testing.assert_allclose(dst, 42.0, rtol=1e-14)


# ═══════════════════════════════════════════════════════════════════════
#  ТОЧНОСТЬ ОСРЕДНЕНИЯ
# ═══════════════════════════════════════════════════════════════════════

def test_averaging_is_exact_on_linear_field():
    """
    Главное свойство осреднения. Среднее линейного поля по ячейке равно
    его значению в центре ячейки — точно, а не приближённо, потому что
    центры мелких ячеек расположены симметрично относительно центра
    крупной.

    Если этот тест покраснел — значит веса неравные либо ячейки
    сгруппированы неправильно.
    """
    from cardiac_em.coupling import make_transfer
    from cardiac_em.fem import owned_dof_coordinates

    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=24, ny=24, lx_mm=12.0, ly_mm=12.0), coarsening=4)
    _, DG0_e, DG0_m = _spaces(cfg)
    tr = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical)

    def f(x, y):
        return 3.0 + 2.5 * x - 1.25 * y

    got = tr.apply(_sample(DG0_e, f))
    expect = _sample(DG0_m, f)

    np.testing.assert_allclose(got, expect, atol=1e-12)


def test_averaging_conserves_the_integral():
    """
    Интеграл по области сохраняется точно. Для T_act это означает
    сохранение суммарной активной силы при переносе на механическую
    сетку — физически самое существенное свойство.
    """
    from cardiac_em.coupling import make_transfer

    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=20, ny=20, lx_mm=10.0, ly_mm=10.0), coarsening=5)
    _, DG0_e, DG0_m = _spaces(cfg)
    tr = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical)

    rng = np.random.default_rng(12345)
    src = rng.random(DG0_e.dofmap.index_map.size_local) * 120.0
    dst = tr.apply(src)

    area_e = cfg.electric.hx_mm * cfg.electric.hy_mm
    area_m = cfg.mechanical.hx_mm * cfg.mechanical.hy_mm

    int_src = _global_integral(DG0_e, src, area_e)
    int_dst = _global_integral(DG0_m, dst, area_m)

    assert math.isclose(int_src, int_dst, rel_tol=1e-12), (
        f"интеграл не сохранён: {int_src} → {int_dst}")


def test_averaging_equals_manual_cell_mean():
    """
    Прямая сверка: значение в крупной ячейке равно среднему тех мелких,
    что физически внутри неё. Считаем группировку независимо от кода
    переноса — по координатам.
    """
    from mpi4py import MPI

    from cardiac_em.coupling import make_transfer
    from cardiac_em.fem import owned_dof_coordinates

    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=12, ny=12, lx_mm=6.0, ly_mm=6.0), coarsening=3)
    _, DG0_e, DG0_m = _spaces(cfg)
    tr = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical)

    def f(x, y):
        return np.sin(1.7 * x) * np.cos(0.9 * y) + 2.0

    src = _sample(DG0_e, f)
    dst = tr.apply(src)

    comm = DG0_e.mesh.comm
    e_xy = np.vstack(comm.allgather(owned_dof_coordinates(DG0_e)))
    e_val = np.concatenate(comm.allgather(src))
    m_xy = owned_dof_coordinates(DG0_m)

    hx, hy = cfg.mechanical.hx_mm, cfg.mechanical.hy_mm
    tol = 1e-9
    for j, (cx, cy) in enumerate(m_xy):
        inside = (
            (e_xy[:, 0] > cx - hx / 2 - tol) & (e_xy[:, 0] < cx + hx / 2 + tol) &
            (e_xy[:, 1] > cy - hy / 2 - tol) & (e_xy[:, 1] < cy + hy / 2 + tol))
        assert inside.sum() == 9
        assert math.isclose(dst[j], float(e_val[inside].mean()), rel_tol=1e-12)


# ═══════════════════════════════════════════════════════════════════════
#  ПОЧЕМУ ОСРЕДНЕНИЕ, А НЕ БЛИЖАЙШАЯ ЯЧЕЙКА
# ═══════════════════════════════════════════════════════════════════════

def test_nearest_is_worse_than_averaging_on_a_front():
    """
    Количественное обоснование выбора стратегии.

    Берём фронт возбуждения — резкий переход шириной порядка мелкого
    шага. Эталон для крупной ячейки — ИСТИННОЕ среднее непрерывного поля
    по ней (плотная квадратура). Сравниваем, насколько от него
    отклоняются обе стратегии.

    Осреднение должно быть на порядок ближе к истине: оно суммирует
    вклад всех мелких ячеек, а ближайшая берёт одну, случайно оказавшуюся
    в центре.
    """
    from cardiac_em.coupling import make_transfer
    from cardiac_em.fem import owned_dof_coordinates

    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=40, ny=40, lx_mm=10.0, ly_mm=10.0), coarsening=8)
    _, DG0_e, DG0_m = _spaces(cfg)
    # мелкий шаг 0.25 мм, крупный 2.0 мм

    x_front, width = 5.0, 0.3

    def front(x, y):
        """Сглаженный фронт: 0 слева, 120 кПа справа."""
        return 120.0 * 0.5 * (1.0 + np.tanh((x - x_front) / width))

    src = _sample(DG0_e, front)

    tr_avg = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical,
                           strategy="averaging")
    tr_near = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical,
                            strategy="nearest")
    got_avg = tr_avg.apply(src)
    got_near = tr_near.apply(src)

    # истинное среднее по каждой крупной ячейке — плотной квадратурой
    m_xy = owned_dof_coordinates(DG0_m)
    hx, hy = cfg.mechanical.hx_mm, cfg.mechanical.hy_mm
    q = (np.arange(64) + 0.5) / 64.0 - 0.5
    truth = np.empty(len(m_xy))
    for j, (cx, cy) in enumerate(m_xy):
        xs = cx + q * hx
        ys = cy + q * hy
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        truth[j] = front(X, Y).mean()

    comm = DG0_m.mesh.comm
    err_avg = _global_max_abs(comm, got_avg - truth)
    err_near = _global_max_abs(comm, got_near - truth)

    # Осреднение должно быть заметно точнее. Порог с запасом: на практике
    # разрыв на порядок и более.
    assert err_avg < err_near / 3.0, (
        f"ожидалось, что осреднение существенно точнее: "
        f"ошибка осреднения {err_avg:.3f} кПа, ближайшей ячейки "
        f"{err_near:.3f} кПа")
    # и само осреднение должно быть близко к истине
    assert err_avg < 5.0, f"осреднение слишком далеко от истины: {err_avg:.3f} кПа"


def test_nearest_loses_total_force_on_a_front():
    """
    То же с точки зрения физики: суммарная активная сила. Осреднение
    сохраняет её с точностью дискретизации мелкой сетки, ближайшая
    ячейка — теряет или добавляет заметную долю.
    """
    from cardiac_em.coupling import make_transfer

    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=40, ny=40, lx_mm=10.0, ly_mm=10.0), coarsening=8)
    _, DG0_e, DG0_m = _spaces(cfg)

    def front(x, y):
        return 120.0 * 0.5 * (1.0 + np.tanh((x - 5.0) / 0.3))

    src = _sample(DG0_e, front)
    area_e = cfg.electric.hx_mm * cfg.electric.hy_mm
    area_m = cfg.mechanical.hx_mm * cfg.mechanical.hy_mm
    ref = _global_integral(DG0_e, src, area_e)

    got_avg = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical,
                            strategy="averaging").apply(src)
    got_near = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical,
                             strategy="nearest").apply(src)

    rel_avg = abs(_global_integral(DG0_m, got_avg, area_m) - ref) / ref
    rel_near = abs(_global_integral(DG0_m, got_near, area_m) - ref) / ref

    assert rel_avg < 1e-12, f"осреднение обязано сохранять интеграл, вышло {rel_avg:.2e}"
    assert rel_near > 1e-4, (
        f"ожидалась заметная потеря на ближайшей ячейке, вышло {rel_near:.2e} — "
        f"проверьте, что фронт в тесте действительно резкий")


# ═══════════════════════════════════════════════════════════════════════
#  УСТОЙЧИВОСТЬ К ЧИСЛУ РАНГОВ И ОШИБКАМ ВЫЗОВА
# ═══════════════════════════════════════════════════════════════════════

def test_result_matches_analytic_answer_regardless_of_ranks():
    """
    Ответ определён аналитически, поэтому тест одинаково осмыслен при
    любом числе рангов: результат не должен зависеть от того, как сетки
    разложены по процессам (а раскладки у двух сеток разные).
    """
    from cardiac_em.coupling import make_transfer

    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=32, ny=32, lx_mm=8.0, ly_mm=8.0), coarsening=4)
    _, DG0_e, DG0_m = _spaces(cfg)
    tr = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical)

    def f(x, y):
        return 1.0 + 0.5 * x - 0.25 * y      # линейное → перенос точен

    got = tr.apply(_sample(DG0_e, f))
    np.testing.assert_allclose(got, _sample(DG0_m, f), atol=1e-12)


def test_wrong_input_size_is_rejected():
    """
    Передали массив с гало вместо владеемой части — классическая
    ошибка. Должна ловиться сразу, а не давать сдвинутые значения.
    """
    from cardiac_em.coupling import make_transfer

    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=8, ny=8), coarsening=2)
    _, DG0_e, DG0_m = _spaces(cfg)
    tr = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical)

    n_owned = DG0_e.dofmap.index_map.size_local
    with pytest.raises(ValueError, match="владеемых ячейках"):
        tr.apply(np.zeros(n_owned - 1))


def test_transfer_reusable_across_steps():
    """
    Оператор строится один раз и применяется многократно — проверяем,
    что состояние не портится между вызовами.
    """
    from cardiac_em.coupling import make_transfer

    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=16, ny=16, lx_mm=8.0, ly_mm=8.0), coarsening=4)
    _, DG0_e, DG0_m = _spaces(cfg)
    tr = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical)

    n = DG0_e.dofmap.index_map.size_local
    for value in (0.0, 10.0, 120.0, 0.0):
        np.testing.assert_allclose(tr.apply(np.full(n, value)), value,
                                   rtol=1e-14)


def test_transfer_into_dolfinx_function_roundtrip():
    """
    Практический сценарий: результат переноса кладётся в Function на
    механической сетке и должен там корректно оказаться, включая гало
    после scatter_forward.
    """
    import dolfinx.fem as fem

    from cardiac_em.coupling import make_transfer

    cfg = DualMeshConfig.nested(
        RectangleMeshSpec(nx=16, ny=16, lx_mm=8.0, ly_mm=8.0), coarsening=4)
    _, DG0_e, DG0_m = _spaces(cfg)
    tr = make_transfer(DG0_e, DG0_m, cfg.electric, cfg.mechanical)

    def f(x, y):
        return 2.0 + x - 0.5 * y

    dst = tr.apply(_sample(DG0_e, f))

    T = fem.Function(DG0_m, name="T_act")
    n_owned = DG0_m.dofmap.index_map.size_local
    T.x.array[:n_owned] = dst
    T.x.scatter_forward()

    np.testing.assert_allclose(T.x.array[:n_owned], _sample(DG0_m, f), atol=1e-12)
