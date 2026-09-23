"""
Слой FEM: сетки, функциональные пространства, поля параметров.
===============================================================

Здесь конфигурация (чистые данные) превращается в объекты DOLFINx.
Всё в этом подпакете ТРЕБУЕТ установленного DOLFINx — в отличие от
`cardiac_em.config`, который работает и без него.

Состав:
    mesh.py    — построение пары сеток, координаты DOF
    tissue.py  — поля параметров ткани по регионам  (шаг 3)
"""

from __future__ import annotations

from .mesh import (
    MeshPair,
    build_mesh,
    build_mesh_pair,
    cell_midpoints,
    dof_coordinates,
    n_cells_global,
    n_vertices_global,
    owned_dof_coordinates,
)
from .tissue import Tissue

__all__ = [
    "build_mesh",
    "build_mesh_pair",
    "MeshPair",
    "cell_midpoints",
    "dof_coordinates",
    "owned_dof_coordinates",
    "n_cells_global",
    "n_vertices_global",
    "Tissue",
]
