"""
Запись полей: XDMF для просмотра, .npz-снимки для анализа.
==========================================================

Два формата с разным назначением:

* XDMF (+ HDF5) — для ParaView. Пишется через `dolfinx.io.XDMFFile`
  каждые `output.save_every_mech_steps` механических шагов:

      electrics.xdmf  — все переменные состояния клетки (P1) и T_act (DG0)
      mechanics.xdmf  — u (P1, вектор), T_act, λ_f, J, σ_xx (DG0)
      regions_electric.xdmf, regions_mechanical.xdmf — карты регионов
                        (0 — базовая ткань, i — regions[i-1]); пишутся один
                        раз при старте, если `output.write_region_maps`

* Снимки .npz — для анализа БЕЗ DOLFINx (numpy достаточно). Пишутся в
  моменты `output.snapshot_times_ms` в snapshots/snap_t…ms.npz. Значения
  собраны со всех рангов вместе с координатами и упорядочены по (y, x):
  на прямоугольной сетке `e_state[:, 0].reshape(ny+1, nx+1)` даёт карту
  потенциала.

  Ключи снимка:
      t_ms, requested_ms (заказанные моменты, попавшие в этот снимок)
      e_coords, e_state, e_state_names, e_region      — узлы эл. сетки
      e_cell_coords, e_t_act                          — ячейки эл. сетки
      m_coords, m_u                                   — узлы мех. сетки
      m_cell_coords, m_t_act, m_lambda_f, m_J, m_sigma_xx, m_region
                                                      — ячейки мех. сетки
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dolfinx import fem
from dolfinx import io as dio

from ..runtime.observers import Observer
from ._points import gather_owned

__all__ = ["FieldWriter", "write_snapshot"]


def write_snapshot(sim, path, requested_ms=()) -> Path:
    """
    Снимок всех полей в .npz. КОЛЛЕКТИВНАЯ операция; пишет ранг 0.
    """
    path = Path(path)
    comm = sim.comm
    e, m = sim.electrics, sim.mechanics

    e_c, e_state = gather_owned(e.P1, e.state, comm)
    _, e_reg = gather_owned(e.P1, sim.tissue_e.region_id_p1 + 1, comm)
    e_cc, e_tact = gather_owned(e.DG0, e.t_act_function().x.array, comm)

    m_c, m_u = gather_owned(m.V, m.u.x.array.reshape(-1, 2), comm)
    cell_fields = np.column_stack([
        m.T_act.x.array,
        m.fiber_stretch().x.array,
        m.jacobian_determinant().x.array,
        m.cauchy_stress(0, 0).x.array,
        sim.tissue_m.region_id_dg0 + 1,
    ])
    m_cc, m_cells = gather_owned(m.DG0, cell_fields, comm)

    if comm.rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                t_ms=np.array(float(sim.t_ms)),
                requested_ms=np.asarray(requested_ms, dtype=np.float64),
                e_coords=e_c, e_state=e_state,
                e_state_names=np.array(sim.cell_model.state_names),
                e_region=e_reg[:, 0].astype(np.int32),
                e_cell_coords=e_cc, e_t_act=e_tact[:, 0],
                m_coords=m_c, m_u=m_u,
                m_cell_coords=m_cc,
                m_t_act=m_cells[:, 0], m_lambda_f=m_cells[:, 1],
                m_J=m_cells[:, 2], m_sigma_xx=m_cells[:, 3],
                m_region=m_cells[:, 4].astype(np.int32),
            )
        tmp.replace(path)
    comm.Barrier()
    return path


class FieldWriter(Observer):
    """
    Наблюдатель записи полей. Файлы открываются в `on_start` и
    закрываются в `on_finish`. Один экземпляр — на один `run()`.
    """

    def __init__(self, out_dir, write_region_maps: bool = True):
        self.out_dir = Path(out_dir)
        self.write_region_maps = write_region_maps
        self._xe = self._xm = None
        self._state_fns: list[fem.Function] = []
        self.frames: list[float] = []
        self.snapshots: list[tuple[float, Path]] = []

    # ── файлы ─────────────────────────────────────────────────────────
    def _open(self, sim) -> None:
        if self._xe is not None:
            raise RuntimeError("FieldWriter уже использован — создайте новый "
                               "для следующего прогона")
        if sim.comm.rank == 0:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        sim.comm.Barrier()

        mesh_e = sim.electrics.mesh
        mesh_m = sim.mechanics.mesh
        self._xe = dio.XDMFFile(sim.comm, str(self.out_dir / "electrics.xdmf"), "w")
        self._xe.write_mesh(mesh_e)
        self._xm = dio.XDMFFile(sim.comm, str(self.out_dir / "mechanics.xdmf"), "w")
        self._xm.write_mesh(mesh_m)

        # По функции на каждую переменную клетки: имя в ParaView — имя
        # переменной модели.
        self._state_fns = [fem.Function(sim.electrics.P1, name=name)
                           for name in sim.cell_model.state_names]

        if self.write_region_maps:
            for label, tissue in (("electric", sim.tissue_e),
                                  ("mechanical", sim.tissue_m)):
                path = self.out_dir / f"regions_{label}.xdmf"
                with dio.XDMFFile(sim.comm, str(path), "w") as xf:
                    xf.write_mesh(tissue.mesh)
                    xf.write_function(tissue.region_id_fn)

    def _close(self) -> None:
        for f in (self._xe, self._xm):
            if f is not None:
                f.close()

    def _write_frame(self, sim) -> None:
        e, m = sim.electrics, sim.mechanics
        t = float(sim.t_ms)

        n_local = e.n_local
        for j, fn in enumerate(self._state_fns):
            fn.x.array[:n_local] = e.state[:, j]
            fn.x.scatter_forward()
            self._xe.write_function(fn, t)
        self._xe.write_function(e.t_act_function(), t)

        self._xm.write_function(m.u, t)
        self._xm.write_function(m.T_act, t)
        self._xm.write_function(m.fiber_stretch(), t)
        self._xm.write_function(m.jacobian_determinant(), t)
        self._xm.write_function(m.cauchy_stress(0, 0), t)
        self.frames.append(t)

    def _snapshot(self, sim, requested) -> None:
        path = self.out_dir / "snapshots" / f"snap_t{sim.t_ms:012.3f}ms.npz"
        write_snapshot(sim, path, requested)
        self.snapshots.append((sim.t_ms, path))

    # ── события ───────────────────────────────────────────────────────
    def on_start(self, sim, schedule):
        self._open(sim)
        self._write_frame(sim)
        if schedule.initial_snapshots:
            self._snapshot(sim, schedule.initial_snapshots)

    def on_mech_tick(self, sim, tick):
        if tick.write_fields:
            self._write_frame(sim)
        if tick.snapshots:
            self._snapshot(sim, tick.snapshots)

    def on_finish(self, sim):
        self._close()

    def on_abort(self, sim, exc):
        # Закрытие XDMF/HDF5 при нескольких рангах — коллективная
        # операция, а ошибка могла случиться не везде. На одном ранге
        # закрываем, чтобы файл был читаемым; при MPI оставляем как есть.
        if sim.comm.size == 1:
            self._close()
