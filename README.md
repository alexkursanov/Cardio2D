# cardiac-em

Электромеханика сердечной ткани: монодоменная электрика + квазистатическая
гиперупругость на **раздельных сетках**, DOLFINx 0.10 + MPI + PETSc.

Ключевые свойства:

* **двойная сетка** — мелкая для фронта возбуждения, крупная для
  механики, с точным переносом активного напряжения при вложенных
  сетках;
* **неоднородная ткань** — области с собственными параметрами клеток
  (возбудимость, проводимость, сократимость, жёсткость);
* **продолжение расчёта** с чекпоинта, в том числе из другого
  эксперимента и на другом числе MPI-рангов;
* **конфигурация как данные** — весь прогон описан одним
  сериализуемым объектом, без глобальных переменных.

Устройство пакета и проектные решения — в [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Состояние

Проект собирается по шагам; каждый шаг проверяется изолированно.

| Шаг | Что | Состояние | Нужен DOLFINx |
|---|---|---|---|
| 1 | `config/` — конфигурация, области, протоколы | готов, 43 теста | нет |
| 2 | `fem/mesh.py` — построение пары сеток | готов, 16 | да |
| 3 | `fem/tissue.py` — поля параметров по регионам | готов, 17 | да |
| 4 | `coupling/transfer.py` — перенос Э↔М | готов, 16 | да |
| 5 | `models/cell/` — абстракция CellModel | готов, 26 | нет |
| 6 | `solvers/monodomain.py` | готов, 17 | да |
| 7 | `solvers/mechanics.py` | готов, 22 | да |
| 8 | `runtime/` — расписание, наблюдатели, Simulation | готов, 18 + 17 | частично |
| 9 | `io/` — поля, ряды, чекпоинты, манифест | готов, 20 + 18 | частично |
| 10а | `control/` — CLI, API, параметрические серии | готов, 17 + 7 | частично |
| 10б | `analysis/` — чтение результатов, биомаркеры; карты активации | готов, 17 + 6 | нет (расчётные тесты — да) |
| **11** | `models/cell/tnnpm.py` — TNNPM (без 2-комп. СР и CaMKII, с I_K(ATP)); параметры клетки по регионам | **на проверке**, 11 + 7 | модель — нет |

---

## Как запустить расчёт

### Из командной строки

```bash
python -m cardiac_em init base.json --nx 80 --coarsening 4   # конфиг для правки
python -m cardiac_em show base.json                          # сводка и предупреждения
mpirun -n 4 python -m cardiac_em run base.json --out runs/a
python -m cardiac_em run base.json --out runs/b --t-end 3000 \
    --restart runs/a/ckpt_last.npz --set stimulus.times_ms=[2000,2400]
python -m cardiac_em status runs/a
```

`--set путь=значение` меняет любой параметр конфигурации без правки
файла (опечатка в пути — ошибка). Серия прогонов описывается JSON-файлом
и запускается `python -m cardiac_em sweep серия.json`; формат —
`cardiac_em/control/sweep.py`, пример — `examples/sweep_tmax.json`.
Прерванную серию достаточно запустить ещё раз: завершённые точки
пропускаются.

### Модель TNNPM и ишемия

```bash
python -m cardiac_em show examples/tnnpm_ischemia.json
mpirun -n 4 python -m cardiac_em run examples/tnnpm_ischemia.json
```

Модель выбирается `cell_model: "tnnpm"`. Параметры клетки по регионам —
ключами `cell:<имя>` в overrides региона (`cell:K_o`, `cell:ATP_i`,
`cell:KmATP`, `cell:g_Na`, … — список в `TNNPMModel.param_names`), для
всей ткани — `cell_params`. `T_MAX` у TNNPM — пиковое активное
напряжение здоровой ткани в кПа. При региональной ишемии задавайте
`preload.cell_relax_ms` ≈ 300: ткань приходит к своему покою до стимула.

### Анализ результатов (без DOLFINx — хоть на ноутбуке)

```python
from cardiac_em.analysis import open_run, open_sweep, biomarkers as bm, plots

run = open_run("runs/a")
maps = run.activation()                         # activation.npz: по узлам и ударам
cv = bm.conduction_velocity(maps.coords, maps.act[:, 0])      # мм/мс
apd = bm.apd_summary(maps.apd[:, 0], maps.region)             # APD90 по регионам
mech = bm.mechanics_summary(run.series())                     # пики T_act, σ_xx, укорочение
plots.field_map(maps.grid(maps.act[:, 0]), run.meshes["electric"], title="t_act, мс")

rows = open_sweep("runs/sweep_tmax").table(lambda r: bm.mechanics_summary(r.series()))
```

### Из Python

```python
from cardiac_em.config import SimulationConfig, OutputConfig
from cardiac_em.io import attach_outputs, restore_checkpoint
from cardiac_em.runtime import ConsoleObserver, Simulation

cfg = SimulationConfig.default()                 # или SimulationConfig.from_json(...)
cfg.output = OutputConfig(out_dir="runs/baseline", snapshot_times_ms=(50, 100))

sim = Simulation(cfg, observers=[ConsoleObserver()])
attach_outputs(sim)            # run.json, series.csv, XDMF, снимки, чекпоинты
sim.preload()
sim.run()
```

Продолжение с чекпоинта (в том числе другого эксперимента или
монолитной версии) — вместо `preload()`:

```python
cfg.output = OutputConfig(out_dir="runs/continued")
cfg.time.t_end_ms = 3000.0                       # время АБСОЛЮТНОЕ
sim = Simulation(cfg, observers=[ConsoleObserver()])
attach_outputs(sim)
info = restore_checkpoint(sim, "runs/baseline/ckpt_last.npz")
sim.run(t_start_ms=info.t_start_ms)
```

`ckpt_preloaded.npz` — общая стартовая точка для серии: преднагрузка
делается один раз. Что лежит в папке вывода — `cardiac_em/io/outputs.py`.

---

## Установка

DOLFINx, PETSc4py и MPI4py **не ставятся через pip** — они приходят из
conda/spack или докер-образа вместе с собранными PETSc и MPI. Поэтому
их нет в зависимостях пакета: попытка поставить их через pip сломает
окружение.

```bash
# в окружении, где уже есть DOLFINx 0.10
pip install -e ".[solver,dev]"

# только конфигурация и анализ, без FEM-стека (ноутбук, постобработка)
pip install -e ".[analysis,dev]"
```

Проверка, что DOLFINx на месте:

```bash
python -c "import dolfinx; print(dolfinx.__version__)"
```

---

## Быстрый старт

```python
from cardiac_em.config import (
    SimulationConfig, DualMeshConfig, RectangleMeshSpec,
    StimulusProtocol, TimeStepping, RectRegion, CircleRegion,
)

cfg = SimulationConfig(
    # электрика 80×80, механика — вложенное огрубление в 4 раза → 20×20
    mesh=DualMeshConfig.nested(
        RectangleMeshSpec(nx=80, ny=80, lx_mm=10.0, ly_mm=10.0),
        coarsening=4),

    # ишемическая зона справа + невозбудимый жёсткий рубец внутри неё
    regions=[
        RectRegion(x0=6, x1=10, y0=0, y1=10, name="ischemia",
                   overrides={"AP_c1": 0.13, "AP_a": 0.20,
                              "D_LONG": 0.05, "D_TRANS": 0.01,
                              "T_MAX": 20.0}),
        CircleRegion(cx=8, cy=5, r=1.0, name="scar",
                     overrides={"AP_c1": 0.0, "D_LONG": 0.001,
                                "T_MAX": 0.0, "MU": 5.0, "MU_F": 15.0}),
    ],

    # протокол S1-S1-S1 с периодом 1000 мс
    stimulus=StimulusProtocol(times_ms=(0.0, 1000.0, 2000.0)),
    time=TimeStepping(t_end_ms=3000.0),
)

print(cfg.summary())
for w in cfg.check():
    print("предупреждение:", w)

cfg.to_json("examples/my_run.json")     # конфиг — сериализуемые данные
```

Вывод `summary()`:

```
  Сетка эл.    : 80×80 quadrilateral на 10.0×10.0 мм (h = 0.125×0.125 мм)
  Сетка мех.   : 20×20 quadrilateral на 10.0×10.0 мм (h = 0.5×0.5 мм)
  Вложенность  : вложенные, 4×4 эл. ячеек на механическую → возможен точный перенос
  Шаги         : dt_эл = 0.05 мс, dt_мех = 1 мс, t_end = 3000.0 мс
  Стимулы      : t = [0.0, 1000.0, 2000.0] мс, A = 20.0, длит. 2.0 мс
  Преднагрузка : λ_f = 1.2575 за 10 шагов
  Базовая ткань: T_max=120.0 кПа, D=0.3/0.03 мм²/мс, волокна 0.0°
  Регионы      : 2 (+ базовая ткань)
  Вывод        : output_em
```

---

## Тесты

```bash
pytest                          # всё, что доступно в текущем окружении
pytest -m "not fem"             # только то, что не требует DOLFINx
pytest tests/test_config.py -v  # конкретный шаг

python tests/test_config.py     # без pytest вообще
```

Тесты, требующие DOLFINx, помечены `@pytest.mark.fem` и автоматически
пропускаются там, где его нет.

Под MPI:

```bash
mpirun -n 4 python -m pytest tests/ -m mpi -q
```

---

## Единицы

Везде **мм, мс, кПа**. Производные: диффузия мм²/мс, скорость
проведения мм/мс (= м/с). Имена полей конфигурации несут единицу в
суффиксе (`lx_mm`, `dt_electric_ms`, `decay_length_mm`), чтобы не
приходилось сверяться с комментарием.

---

## Структура

```
cardiac_em/
  config/      конфигурация: сетки, ткань, области, протоколы  (без DOLFINx)
  fem/         сетки, пространства, поля параметров
  models/      модели клетки, активного напряжения, пассивной механики
  solvers/     монодоменная электрика, квазистатическая механика
  coupling/    перенос полей между сетками
  runtime/     расписание событий, наблюдатели, основной цикл
  io/          чекпоинты, поля, манифест прогона
  control/     запуск: CLI, серии, будущий API
  analysis/    постобработка; читает только файлы             (без DOLFINx)
tests/         тесты по шагам
examples/      примеры конфигураций и областей
```

Правило зависимостей и обоснование — в
[ARCHITECTURE.md](ARCHITECTURE.md#1-принципы).

---

## Лицензия

MIT — см. [LICENSE](LICENSE).
