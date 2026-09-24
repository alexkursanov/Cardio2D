"""
Сбор полей по координатам и сопоставление точек.
=================================================

Внутренний модуль слоя io. Две операции:

* `gather_owned` — собрать на нулевом ранге значения ВЛАДЕЕМЫХ узлов
  (или ячеек) со всех рангов вместе с их координатами и упорядочить по
  (y, x). Порядок не зависит от числа рангов и от нумерации DOF в
  DOLFINx: файл, записанный на одном ядре, совпадает с записанным на
  восьми, а на структурированной сетке массив складывается в
  (ny+1, nx+1) обычным reshape — это удобно для анализа.

* `match_points` — для каждой точки текущей сетки найти её в сохранённом
  наборе. Точное совпадение — с допуском, привязанным к шагу сетки; при
  промахе — ближайший сосед, но только если это явно разрешено.

DOLFINx импортируется внутри функций, которым он нужен: сопоставление
точек и чтение чекпоинтов работают и проверяются без расчётного стека.
"""

from __future__ import annotations

import numpy as np

__all__ = ["gather_owned", "match_points", "local_coordinates", "yx_order"]


def local_coordinates(V) -> np.ndarray:
    """Координаты всех ЛОКАЛЬНЫХ узлов пространства (включая гало), (n, 2)."""
    from ..fem import dof_coordinates
    imap = V.dofmap.index_map
    n = imap.size_local + imap.num_ghosts
    return np.ascontiguousarray(dof_coordinates(V)[:n, :2], dtype=np.float64)


def gather_owned(V, values: np.ndarray, comm):
    """
    Собрать на ранге 0 (координаты, значения) владеемых узлов V.

    `values` — массив, строки которого соответствуют ЛОКАЛЬНЫМ узлам V
    (гало допускаются, они отрезаются), форма (n, k) или (n,).

    КОЛЛЕКТИВНАЯ операция. На ранге 0 возвращает (coords (N, 2),
    values (N, k)) в порядке (y, x); на остальных — (None, None).
    """
    from ..fem import dof_coordinates

    n = V.dofmap.index_map.size_local
    vals = np.asarray(values, dtype=np.float64)
    vals = vals.reshape(vals.shape[0], -1)[:n]
    coords = np.ascontiguousarray(dof_coordinates(V)[:n, :2], dtype=np.float64)

    all_c = comm.gather(coords, root=0)
    all_v = comm.gather(np.ascontiguousarray(vals), root=0)
    if comm.rank != 0:
        return None, None

    c = np.vstack(all_c)
    v = np.vstack(all_v)
    order = yx_order(c)
    return c[order], v[order]


def yx_order(coords: np.ndarray) -> np.ndarray:
    """
    Перестановка, упорядочивающая точки по строкам (y), внутри строки — по x.

    Сравниваются координаты, ОКРУГЛЁННЫЕ относительно размера области:
    у DOLFINx узлы одной строки сетки могут отличаться по y на ~1e-17, и
    сортировка по точным значениям разбрасывает строку (узел с
    y = −1e-17 оказывается раньше узла с y = 0). Сами координаты не
    меняются — округляется только ключ сортировки.
    """
    c = np.asarray(coords, dtype=np.float64)
    if len(c) == 0:
        return np.zeros(0, dtype=np.int64)
    scale = max(float(np.abs(c).max()), 1.0)
    key = np.round(c / scale * 1e9)
    return np.lexsort((key[:, 0], key[:, 1]))


def _nearest_bruteforce(ref: np.ndarray, pts: np.ndarray):
    """Ближайший сосед перебором (кусками) — запасной путь без scipy."""
    idx = np.empty(len(pts), dtype=np.int64)
    dist = np.empty(len(pts))
    chunk = max(1, 2_000_000 // max(1, len(ref)))
    for s in range(0, len(pts), chunk):
        d2 = ((pts[s:s + chunk, None, :] - ref[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(axis=1)
        idx[s:s + chunk] = j
        dist[s:s + chunk] = np.sqrt(d2[np.arange(len(j)), j])
    return idx, dist


def _nearest(ref: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ближайшая точка из ref для каждой из pts: (индексы, расстояния)."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return _nearest_bruteforce(ref, pts)
    dist, idx = cKDTree(ref).query(pts)
    return np.asarray(idx, dtype=np.int64), np.asarray(dist)


def match_points(ref: np.ndarray, pts: np.ndarray, tol: float,
                 allow_nearest: bool, label: str) -> tuple[np.ndarray, int]:
    """
    Для каждой точки `pts` — индекс совпадающей точки в `ref`.

    Совпадение: расстояние ≤ tol. Если какие-то точки не нашлись:
    при allow_nearest — берётся ближайшая, и возвращается их число; без
    него — ошибка с понятным сообщением (тихая порча полей хуже
    остановки).

    Возвращает (индексы, число точек, взятых по ближайшему соседу).
    """
    if len(ref) == 0:
        raise ValueError(f"поле '{label}': в сохранённом наборе нет точек")
    idx, dist = _nearest(ref, pts)
    n_miss = int((dist > tol).sum())
    if n_miss and not allow_nearest:
        raise ValueError(
            f"поле '{label}': {n_miss} из {len(pts)} узлов текущей сетки не "
            f"найдены в чекпоинте (наибольшее расхождение "
            f"{float(dist.max()):.3g} мм) — сетки не совпадают. Перенос по "
            f"ближайшему соседу включается явно: allow_interp=True")
    return idx, n_miss
