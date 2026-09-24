"""
Программный запуск прогона.
============================

    from cardiac_em.control import run_simulation
    result = run_simulation(config)

Одна функция делает то, что иначе пишется руками каждый раз: собирает
`Simulation`, подключает стандартный вывод, либо выполняет
преднагрузку, либо продолжает с чекпоинта из `config.restart`, и ведёт
счёт до `config.time.t_end_ms`.

Это та точка, через которую работают командная строка и серии; через неё
же будет работать будущий интерфейс управления.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config.simulation import SimulationConfig

__all__ = ["RunResult", "run_simulation"]


@dataclass
class RunResult:
    out_dir: Path
    manifest_path: Path
    t_start_ms: float
    t_end_ms: float
    restart: dict | None
    warnings: list[str] = field(default_factory=list)
    sim: object = None          # Simulation — для доступа к состоянию в процессе


def run_simulation(config: SimulationConfig, *, observers=(),
                   console: bool = True, console_every_mech_steps: int = 20,
                   overwrite: bool = False, comm=None) -> RunResult:
    """
    Выполнить прогон целиком. КОЛЛЕКТИВНАЯ операция (все ранги MPI).

    observers  : дополнительные наблюдатели (после консоли, до записи файлов)
    console    : печатать ход счёта (нулевой ранг)
    overwrite  : разрешить запись в папку, где уже есть run.json

    Ошибка счёта пробрасывается дальше; run.json к этому моменту уже
    помечен как failed. Ошибка ДО начала счёта (сборка, преднагрузка,
    чтение чекпоинта) манифеста не создаёт — прогон не начинался.
    """
    from ..io import MANIFEST_NAME, attach_outputs, restore_checkpoint
    from ..runtime import ConsoleObserver, Simulation

    obs = list(observers)
    if console:
        obs.insert(0, ConsoleObserver(every_mech_steps=console_every_mech_steps))

    sim = Simulation(config, observers=obs, comm=comm)
    if console and sim.comm.rank == 0:
        print(config.summary(), flush=True)
        print(sim.summary(), flush=True)

    attach_outputs(sim, overwrite=overwrite)

    if config.restart.is_restart:
        info = restore_checkpoint(sim, config.restart.checkpoint_path)
        t_start = info.t_start_ms
        if console and sim.comm.rank == 0:
            print(f"  продолжение с {info.source} (t = {info.t_checkpoint_ms:g} мс"
                  f"{', отсчёт с нуля' if t_start == 0 and info.t_checkpoint_ms else ''})",
                  flush=True)
    else:
        sim.preload()
        t_start = 0.0

    sim.run(t_start_ms=t_start)
    # run.json и карты пишет ранг 0; после возврата они должны быть
    # дописаны для всех рангов (иначе чтение сразу после — гонка)
    sim.comm.Barrier()

    out_dir = Path(config.output.out_dir)
    return RunResult(out_dir=out_dir, manifest_path=out_dir / MANIFEST_NAME,
                     t_start_ms=t_start, t_end_ms=sim.t_ms,
                     restart=sim.restart_info, warnings=list(sim.warnings),
                     sim=sim)
