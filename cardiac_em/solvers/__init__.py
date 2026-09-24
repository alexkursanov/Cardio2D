"""
Солверы: электрика и механика.
===============================

    monodomain.py  — реакция-диффузия на ЭЛЕКТРИЧЕСКОЙ сетке
    mechanics.py   — квазистатическая гиперупругость на МЕХАНИЧЕСКОЙ (шаг 7)

Каждый солвер работает на своей сетке и ничего не знает о другой.
Связь между ними — исключительно через `coupling/`: механика получает
готовый массив T_act на своих ячейках и не подозревает, что он пришёл с
более мелкой сетки.
"""

from __future__ import annotations

from .mechanics import (
    MechanicsSolver,
    bcs_clamped_at_current_state,
    bcs_free_contraction,
    bcs_prescribed_on_boundary,
    bcs_uniaxial_stretch,
    bcs_uniaxial_stretch_symmetric,
)
from .monodomain import MonodomainSolver

__all__ = [
    "MonodomainSolver",
    "MechanicsSolver",
    "bcs_uniaxial_stretch",
    "bcs_uniaxial_stretch_symmetric",
    "bcs_free_contraction",
    "bcs_clamped_at_current_state",
    "bcs_prescribed_on_boundary",
]
