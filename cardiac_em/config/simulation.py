"""
Корневой объект конфигурации — `SimulationConfig`.
==================================================

Один объект описывает прогон целиком. Расчётное ядро (`runtime/`)
принимает его на вход и не читает НИКАКИХ глобальных переменных.

Зачем это нужно, кроме аккуратности: пока параметры живут в модульных
глобалах, в одном процессе нельзя корректно прогнать две разные
конфигурации — второй прогон унаследует изменения первого. Именно это
блокировало бы будущие параметрические серии и интерфейс управления,
который собирает конфиг из формы и запускает счёт.

Сериализация
------------
`to_dict()` / `from_dict()` дают JSON-совместимое представление. Оно
попадает в манифест прогона и в чекпоинт — благодаря этому рестарт
восстанавливает в том числе неоднородность ткани, которая раньше
терялась.

Единственное исключение — области `CustomRegion` (произвольный
Python-предикат): они не сериализуются. `to_dict()` по умолчанию
поднимает ошибку, чтобы потеря неоднородности не прошла незаметно;
`to_dict(skip_unserializable=True)` пропускает их, вернув список
пропущенного в `unserializable_regions()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .mesh_spec import DualMeshConfig, RectangleMeshSpec
from .protocol import PreloadProtocol, StimulusProtocol, TimeStepping
from .tissue_spec import RegionSpec, TissueBaseParams, region_from_dict

__all__ = ["OutputConfig", "RestartConfig", "SimulationConfig"]


@dataclass
class OutputConfig:
    """
    Что и как часто писать на диск.

    Частоты заданы в МЕХАНИЧЕСКИХ шагах: именно на механическом шаге
    состояние обеих задач согласовано между собой.
    """

    out_dir: Path = Path("output_em")
    save_every_mech_steps: int = 10        # запись полей в XDMF
    snapshot_times_ms: tuple[float, ...] = ()   # моменты снимков VTK
    ckpt_every_mech_steps: int = 100       # 0 — не сохранять чекпоинты
    ckpt_keep_all: bool = False            # False — только ckpt_last.npz
    write_region_maps: bool = True         # карты регионов при старте
    record_activation: bool = True         # карты активации/реполяризации
    apd_level: float = 0.9                 # APD90: реполяризация на 90 %

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.snapshot_times_ms = tuple(sorted(float(t) for t in self.snapshot_times_ms))
        if self.save_every_mech_steps < 1:
            raise ValueError(
                f"save_every_mech_steps должен быть >= 1, получено "
                f"{self.save_every_mech_steps}")
        if self.ckpt_every_mech_steps < 0:
            raise ValueError(
                f"ckpt_every_mech_steps не может быть < 0, получено "
                f"{self.ckpt_every_mech_steps}")
        if not 0.0 < self.apd_level < 1.0:
            raise ValueError(
                f"apd_level — доля реполяризации в (0, 1), например 0.9 для "
                f"APD90; получено {self.apd_level}")

    @property
    def checkpoints_enabled(self) -> bool:
        return self.ckpt_every_mech_steps > 0

    def to_dict(self) -> dict:
        return {"out_dir": str(self.out_dir),
                "save_every_mech_steps": self.save_every_mech_steps,
                "snapshot_times_ms": list(self.snapshot_times_ms),
                "ckpt_every_mech_steps": self.ckpt_every_mech_steps,
                "ckpt_keep_all": self.ckpt_keep_all,
                "write_region_maps": self.write_region_maps,
                "record_activation": self.record_activation,
                "apd_level": self.apd_level}

    @staticmethod
    def from_dict(d: dict) -> "OutputConfig":
        return OutputConfig(
            out_dir=Path(d.get("out_dir", "output_em")),
            save_every_mech_steps=int(d.get("save_every_mech_steps", 10)),
            snapshot_times_ms=tuple(d.get("snapshot_times_ms", ())),
            ckpt_every_mech_steps=int(d.get("ckpt_every_mech_steps", 100)),
            ckpt_keep_all=bool(d.get("ckpt_keep_all", False)),
            write_region_maps=bool(d.get("write_region_maps", True)),
            record_activation=bool(d.get("record_activation", True)),
            apd_level=float(d.get("apd_level", 0.9)),
        )


@dataclass
class RestartConfig:
    """
    Продолжение расчёта с чекпоинта — в том числе снятого в другом
    эксперименте.

    Поля
    ----
    checkpoint_path : .npz с сохранённым состоянием; None — счёт с нуля
    reset_time      : начать отсчёт с нуля, сохранив поля (удобно, когда
                      к загруженному состоянию применяется новый протокол
                      стимуляции)
    allow_interp    : разрешить перенос состояния на сетку другого
                      разрешения по ближайшему соседу; без этого
                      несовпадение сеток — ошибка, а не тихая порча полей
    """

    checkpoint_path: Path | None = None
    reset_time: bool = False
    allow_interp: bool = False

    def __post_init__(self) -> None:
        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)

    @property
    def is_restart(self) -> bool:
        return self.checkpoint_path is not None

    def to_dict(self) -> dict:
        return {"checkpoint_path": (str(self.checkpoint_path)
                                    if self.checkpoint_path else None),
                "reset_time": self.reset_time,
                "allow_interp": self.allow_interp}

    @staticmethod
    def from_dict(d: dict) -> "RestartConfig":
        p = d.get("checkpoint_path")
        return RestartConfig(
            checkpoint_path=Path(p) if p else None,
            reset_time=bool(d.get("reset_time", False)),
            allow_interp=bool(d.get("allow_interp", False)),
        )


@dataclass
class SimulationConfig:
    """
    Полное описание прогона. Это единственный объект, который нужен
    расчётному ядру: `runtime.run(config)`.
    """

    mesh: DualMeshConfig
    tissue_base: TissueBaseParams = field(default_factory=TissueBaseParams)
    regions: list[RegionSpec] = field(default_factory=list)
    stimulus: StimulusProtocol = field(default_factory=StimulusProtocol)
    time: TimeStepping = field(default_factory=TimeStepping)
    preload: PreloadProtocol = field(default_factory=PreloadProtocol)
    output: OutputConfig = field(default_factory=OutputConfig)
    restart: RestartConfig = field(default_factory=RestartConfig)

    # Выбор моделей — по ИМЕНИ из реестров `models/cell` и
    # `models/passive`. Имена, а не объекты: конфигурация обязана
    # сериализоваться, и прогон по сохранённому run.json должен собрать
    # ту же физику. Проверка имени — при сборке `Simulation` (config/ не
    # знает о реестрах: он лежит ниже слоя моделей).
    cell_model: str = "rogers_mcculloch"
    passive_material: str = "transversely_isotropic_exponential"

    def __post_init__(self) -> None:
        for label, value in (("cell_model", self.cell_model),
                             ("passive_material", self.passive_material)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} должен быть непустым именем модели, "
                                 f"получено {value!r}")

    # ── удобные конструкторы ──────────────────────────────────────────
    @staticmethod
    def default(nx_electric: int = 40, coarsening: int = 2) -> "SimulationConfig":
        """
        Конфигурация «как в исходном монолитном скрипте», но с вложенной
        механической сеткой. Годится как отправная точка и для тестов.
        """
        return SimulationConfig(
            mesh=DualMeshConfig.nested(
                RectangleMeshSpec(nx=nx_electric, ny=nx_electric,
                                  lx_mm=10.0, ly_mm=10.0),
                coarsening=coarsening),
        )

    # ── проверки согласованности между группами ───────────────────────
    def check(self) -> list[str]:
        """
        Проверки, которые невозможно сделать внутри отдельной группы
        параметров, потому что они связывают разные части конфигурации.

        Возвращает список ПРЕДУПРЕЖДЕНИЙ (то, что подозрительно, но
        законно). Настоящие ошибки поднимаются исключениями здесь же.
        """
        warnings: list[str] = []

        # Стимулы позже конца счёта просто никогда не сработают —
        # почти всегда это забытый --t-end при смене протокола.
        late = [t for t in self.stimulus.times_ms if t > self.time.t_end_ms]
        if late:
            warnings.append(
                f"стимулы на t={late} мс позже конца счёта "
                f"t_end={self.time.t_end_ms} мс — они не сработают")

        # Шаг стимуляции не должен проскакивать между шагами по времени.
        if self.stimulus.duration_ms < 2 * self.time.dt_electric_ms:
            warnings.append(
                f"длительность импульса {self.stimulus.duration_ms} мс "
                f"меньше двух электрических шагов "
                f"({2 * self.time.dt_electric_ms} мс) — стимул может быть "
                f"разрешён по времени слишком грубо")

        # Пространственный профиль стимула должен покрывать хотя бы
        # несколько ячеек электрической сетки.
        hx = self.mesh.electric.hx_mm
        if self.stimulus.x_max_mm < 2 * hx:
            warnings.append(
                f"стимульная зона x_max={self.stimulus.x_max_mm} мм уже двух "
                f"ячеек электрической сетки (hx={hx:.3g} мм) — стимул может "
                f"не запустить волну")

        # Механическая сетка мельче электрической — почти наверняка
        # перепутаны местами.
        if self.mesh.mechanical.h_min_mm < self.mesh.electric.h_min_mm:
            warnings.append(
                f"механическая сетка МЕЛЬЧЕ электрической "
                f"(h_мех={self.mesh.mechanical.h_min_mm:.3g} мм < "
                f"h_эл={self.mesh.electric.h_min_mm:.3g} мм) — обычно нужно "
                f"наоборот; не перепутаны ли сетки местами?")

        # Неточный перенос T_act — стоит знать заранее.
        nest = self.mesh.nesting()
        if not nest.is_nested:
            warnings.append(
                f"сетки не вложены ({nest.reason}) — перенос T_act будет "
                f"приближённым (ближайшая ячейка вместо осреднения)")

        # Снимки вне интервала счёта.
        stray = [t for t in self.output.snapshot_times_ms if t > self.time.t_end_ms]
        if stray:
            warnings.append(
                f"моменты снимков {stray} мс позже конца счёта — не сработают")

        # Рестарт с t_end раньше точки старта — цикл окажется пустым.
        # Проверить точно можно только прочитав чекпоинт, поэтому здесь
        # только явно бессмысленный случай.
        if self.restart.is_restart and self.restart.reset_time is False:
            pass  # фактическое сравнение делает runtime после чтения чекпоинта

        return warnings

    def unserializable_regions(self) -> list[RegionSpec]:
        """Области, которые не переживут сохранение конфигурации."""
        return [r for r in self.regions if not r.is_serializable]

    # ── сериализация ──────────────────────────────────────────────────
    def to_dict(self, skip_unserializable: bool = False) -> dict:
        bad = self.unserializable_regions()
        if bad and not skip_unserializable:
            names = [r.name or type(r).__name__ for r in bad]
            raise TypeError(
                f"конфигурацию нельзя сериализовать: области {names} заданы "
                f"Python-предикатом. Выразите их через RectRegion/CircleRegion "
                f"либо вызовите to_dict(skip_unserializable=True), понимая, "
                f"что эта неоднородность в сохранённый конфиг не попадёт")

        return {
            "mesh": self.mesh.to_dict(),
            "tissue_base": self.tissue_base.to_dict(),
            "regions": [r.to_dict() for r in self.regions if r.is_serializable],
            "stimulus": self.stimulus.to_dict(),
            "time": self.time.to_dict(),
            "preload": self.preload.to_dict(),
            "output": self.output.to_dict(),
            "restart": self.restart.to_dict(),
            "cell_model": self.cell_model,
            "passive_material": self.passive_material,
        }

    @staticmethod
    def from_dict(d: dict) -> "SimulationConfig":
        return SimulationConfig(
            mesh=DualMeshConfig.from_dict(d["mesh"]),
            tissue_base=TissueBaseParams.from_dict(d.get("tissue_base", {})),
            regions=[region_from_dict(r) for r in d.get("regions", [])],
            stimulus=StimulusProtocol.from_dict(d.get("stimulus", {})),
            time=TimeStepping.from_dict(d.get("time", {})),
            preload=PreloadProtocol.from_dict(d.get("preload", {})),
            output=OutputConfig.from_dict(d.get("output", {})),
            restart=RestartConfig.from_dict(d.get("restart", {})),
            cell_model=d.get("cell_model", "rogers_mcculloch"),
            passive_material=d.get("passive_material",
                                   "transversely_isotropic_exponential"),
        )

    def to_json(self, path, skip_unserializable: bool = False) -> None:
        import json
        with open(Path(path), "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(skip_unserializable=skip_unserializable),
                      fh, ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(path) -> "SimulationConfig":
        import json
        with open(Path(path), encoding="utf-8") as fh:
            return SimulationConfig.from_dict(json.load(fh))

    # ── человекочитаемая сводка ───────────────────────────────────────
    def summary(self) -> str:
        flat = self.tissue_base.to_flat_dict()
        lines = [
            self.mesh.summary(),
            f"  Модели       : клетка {self.cell_model}, "
            f"материал {self.passive_material}",
            f"  Шаги         : dt_эл = {self.time.dt_electric_ms} мс, "
            f"dt_мех = {self.time.dt_mech_ms:g} мс, "
            f"t_end = {self.time.t_end_ms} мс",
            f"  Стимулы      : t = {list(self.stimulus.times_ms)} мс, "
            f"A = {self.stimulus.amplitude}, "
            f"длит. {self.stimulus.duration_ms} мс",
            f"  Преднагрузка : λ_f = {self.preload.stretch} "
            f"за {self.preload.n_steps} шагов",
            f"  Базовая ткань: T_max={flat['T_MAX']} кПа, "
            f"D={flat['D_LONG']}/{flat['D_TRANS']} мм²/мс, "
            f"волокна {flat['FIBER_ANGLE_DEG']}°",
            f"  Регионы      : {len(self.regions)} "
            f"(+ базовая ткань)",
            f"  Вывод        : {self.output.out_dir}",
        ]
        if self.restart.is_restart:
            lines.append(f"  Рестарт      : {self.restart.checkpoint_path}"
                         f"{' (время с нуля)' if self.restart.reset_time else ''}")
        return "\n".join(lines)
