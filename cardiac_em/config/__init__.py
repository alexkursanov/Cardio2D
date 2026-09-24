"""
Конфигурация расчёта — чистые данные, без DOLFINx.
===================================================

Этот подпакет намеренно зависит только от стандартной библиотеки и
numpy. Благодаря этому конфигурации серий экспериментов можно собирать
и проверять в лёгком окружении, отдельно от расчётной машины, а
`analysis/` может читать сохранённые конфиги, не таща за собой FEM-стек.

Быстрый старт
-------------
    from cardiac_em.config import SimulationConfig, RectangleMeshSpec, \\
        DualMeshConfig, CircleRegion

    cfg = SimulationConfig(
        mesh=DualMeshConfig.nested(
            RectangleMeshSpec(nx=80, ny=80, lx_mm=10, ly_mm=10),
            coarsening=4),                       # механика 20×20
        regions=[CircleRegion(cx=8, cy=5, r=1.0, name="scar",
                              overrides={"T_MAX": 0.0, "D_LONG": 0.001})],
    )
    for w in cfg.check():
        print("предупреждение:", w)
    print(cfg.summary())
"""

from __future__ import annotations

from .mesh_spec import (
    DualMeshConfig,
    MeshNesting,
    MeshSpec,
    RectangleMeshSpec,
)
from .protocol import (
    PreloadProtocol,
    StimulusProtocol,
    TimeStepping,
)
from .simulation import (
    OutputConfig,
    RestartConfig,
    SimulationConfig,
)
from .tissue_spec import (
    CELL_PREFIX,
    ActiveStressParams,
    CircleRegion,
    ConductionParams,
    CustomRegion,
    PassiveMechParams,
    ReactionParams,
    RectRegion,
    RegionSpec,
    TissueBaseParams,
    load_regions_json,
    region_from_dict,
)

__all__ = [
    # сетки
    "RectangleMeshSpec", "MeshSpec", "MeshNesting", "DualMeshConfig",
    # параметры ткани
    "ReactionParams", "ConductionParams", "ActiveStressParams",
    "PassiveMechParams", "TissueBaseParams",
    # области
    "RegionSpec", "RectRegion", "CircleRegion", "CustomRegion", "CELL_PREFIX",
    "region_from_dict", "load_regions_json",
    # протоколы
    "StimulusProtocol", "TimeStepping", "PreloadProtocol",
    # корневой конфиг
    "OutputConfig", "RestartConfig", "SimulationConfig",
]
