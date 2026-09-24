"""
Быстрые графики (matplotlib — по желанию: pip install -e ".[analysis]").
========================================================================

    from cardiac_em.analysis import open_run, plots
    run = open_run("runs/a")
    plots.series(run, ["t_act_mech_max", "sigma_xx_max"])
    maps = run.activation()
    plots.field_map(maps.grid(maps.act[:, 0]), run.meshes["electric"],
                    title="время активации, мс")

Функции возвращают (fig, ax) и ничего не показывают сами — для
Jupyter, скриптов (fig.savefig) и отчётов одинаково.
"""

from __future__ import annotations

import numpy as np

__all__ = ["series", "field_map"]


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("для графиков нужен matplotlib: "
                          "pip install -e \".[analysis]\"") from exc
    return plt


def series(run, keys, ax=None, t_from_ms=None, t_to_ms=None):
    """Временные ряды из series.csv на одних осях."""
    plt = _plt()
    s = run.series()
    t = s["t_ms"]
    mask = np.ones_like(t, dtype=bool)
    if t_from_ms is not None:
        mask &= t >= t_from_ms
    if t_to_ms is not None:
        mask &= t <= t_to_ms
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
    else:
        fig = ax.figure
    for k in ([keys] if isinstance(keys, str) else keys):
        ax.plot(t[mask], s[k][mask], label=k)
    ax.set_xlabel("t, мс")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig, ax


def field_map(grid, mesh: dict, ax=None, title: str = "", cmap: str = "viridis",
              nodes: bool = True):
    """
    Двумерная карта (строки — y, столбцы — x) на физических координатах.
    `mesh` — описание сетки из манифеста (lx_mm, ly_mm).
    """
    plt = _plt()
    if ax is None:
        lx, ly = float(mesh["lx_mm"]), float(mesh["ly_mm"])
        fig, ax = plt.subplots(figsize=(6, 6 * ly / lx + 0.8))
    else:
        fig = ax.figure
    extent = (0, float(mesh["lx_mm"]), 0, float(mesh["ly_mm"]))
    im = ax.imshow(np.asarray(grid), origin="lower", extent=extent, cmap=cmap,
                   interpolation="bilinear" if nodes else "nearest", aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("x, мм")
    ax.set_ylabel("y, мм")
    if title:
        ax.set_title(title)
    return fig, ax
