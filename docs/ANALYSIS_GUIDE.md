# Обработка результатов

Порядок работы с посчитанным прогоном или серией:

1. **Проверить**, что прогон закончился и посчитан правдоподобно (раздел 1).
2. **Стандартный анализ** — ноутбуки 03, 04, 05 (разделы 2–4).
3. **Свой анализ** — Python, модуль `cardiac_em.analysis` (раздел 5).
4. **Поля во времени** — ParaView (раздел 6).
5. **Сохранить**: таблицы, рисунки, конфигурации (раздел 8).

Файлы, из которых всё берётся, описаны в [OUTPUT_FILES.md](OUTPUT_FILES.md).
Для анализа DOLFINx не нужен — достаточно `pip install -e ".[analysis]"`,
в том числе на своём компьютере со скопированной папкой прогона.

---

## 1. Контроль качества прогона

```bash
python -m cardiac_em status runs/my_run      # finished?
python -m cardiac_em status runs/my_sweep    # все точки finished?
grep -A5 '"warnings"' runs/my_run/run.json   # предупреждения
tail -n 15 runs/my_run.log                   # конец счёта
```

Затем в ноутбуке 03 (разделы 1–2) или 04 (раздел 1) проверьте ряды.
Ориентиры для TNNPM по нашим прогонам (T_MAX = 60 кПа, D = 0.15/0.05
мм²/мс, преднагрузка 1.2575):

| Что | Норма | Тревожный признак |
|-----|-------|-------------------|
| `status` | `finished` | `failed` — смотреть `error` в run.json и лог |
| потенциал покоя (`u_min`) | ≈ −86 мВ в здоровой ткани; −72 мВ в ишемии «15 мин» | выше −60 мВ во всей ткани, NaN |
| пик потенциала (`u_max`) | +10…+30 мВ | волна не пошла: `u_max` не поднимается после стимула |
| APD90 (1 Гц) | ~240–250 мс в здоровой ткани; короче в ишемии | 0 или NaN во всей ткани |
| скорость проведения вдоль волокна | ~0.8–0.9 мм/мс при D_LONG = 0.15 | блок там, где его не ждали |
| пик T_act | близко к T_MAX (≈ 60 кПа) | ≫ T_MAX или отрицательный |
| интегралы T_act | `t_act_electric_integral` ≈ `t_act_mech_integral` | заметное расхождение — сломан перенос |
| `J_min…J_max` | ≈ 1 (0.98–1.02) | < 0.9 или > 1.1 — вырожденная деформация |
| `newton_iterations` | 2–4 | растёт до 10+ |
| удар к удару (серия ударов) | пик T_act и min λ_f последних ударов совпадают | дрейф до конца счёта — ритм не установился, нужно больше ударов |

---

## 2. Ноутбук 03 — таблицы и числа

`notebooks/03_analysis.ipynb`. В ячейке с параметрами:

```python
RUN_DIR = RUNS / "night_1hz" / "run_000"
SWEEP_DIR = RUNS / "night_1hz"
```

**Run All**. Что получится:

| Раздел | Результат |
|--------|-----------|
| паспорт | статус, время счёта, сетки, модели, версии, предупреждения |
| параметры | T_MAX, D, стимулы, таблица регионов с переопределениями |
| ряды | сводка `series.csv` (min/max/mean каждого столбца) |
| механика | `mechanics_summary`: пик T_act и время до пика, пик σ_xx, наибольшее укорочение, интеграл T_act |
| удары | по каждому удару: первая/последняя активация, общее время активации, CV вдоль x, APD mean/min/max, дисперсия APD |
| CV и APD по регионам | медиана локальной CV; APD (mean, std, min, max) в каждом регионе, `BEAT = -1` — последний удар |
| блок, реституция | доля узлов, не проведших следующий удар; число пар (DI, APD) |
| по регионам во времени | средние V, λ_f, T_act по регионам в каждом снимке |
| серия | таблица: параметры точки + метрики (`SWEEP_BEAT = -1` — по последнему удару) |
| экспорт | `runs/<прогон>/analysis/`: `series.csv`, `beats.csv`, `snapshots_by_region.csv`, `summary.json`; серия — `sweep_table.csv` |

CSV открываются в Excel/Origin — удобно для отчётов.

---

## 3. Ноутбук 04 — рисунки

`notebooks/04_visualization.ipynb`. Параметры:

| Переменная | Смысл |
|------------|-------|
| `RUN_DIR`, `SWEEP_DIR` | что рисовать |
| `BEAT` | удар для карт: 0 — первый, −1 — последний |
| `ISO_STEP_MS` | шаг изохрон (None — ~10 линий) |
| `SNAP_T_RANGE` | окно снимков, например `(9000, 9900)` — последний удар |
| `SNAP_T` | момент для механики (None — момент наибольшего T_act) |
| `METRIC` | метрика для графика серии (`"APD регион 1, мс"`, `"peak_t_act_kpa"`, …) |
| `SAVE` | сохранять PNG |

Рисунки (в `runs/<прогон>/figures/`):

* `series.png` — потенциал, T_act (переданное и действующее), λ_f, σ_xx во времени;
* `beat<k>_maps.png` — время активации с изохронами, APD, локальная CV;
* `snapshots_potential.png` — потенциал в моменты снимков (общая шкала);
* `mechanics_t<t>.png` — λ_f и T_act на деформированной сетке;
* `apd_regions.png` — APD по регионам, реституция;
* `<серия>/figures/<метрика>.png` — метрика по точкам серии.

Как читать карты:

* **изохроны** — линии одинакового времени активации; сгущение —
  замедление проведения, разрыв — блок;
* **локальная CV** у левого края (зона стимула) и у правого (волна
  ускоряется у непроводящей границы) — артефакты; шкала строится по
  внутренним узлам;
* **APD** в маленьком регионе сглажен электротоническим влиянием
  окружающей ткани — сравнивайте с одиночной клеткой осторожно.

---

## 4. Ноутбук 05 — деформация ткани

`notebooks/05_deformation.ipynb`. Параметры: `RUN_DIR`, `SWEEP_DIR`,
`AMPLIFY` (во сколько раз увеличить смещения от преднагрузки на
рисунках; цвет всегда показывает истинное растяжение).

Деформация сокращения отсчитывается **от состояния после преднагрузки**
(самый ранний снимок): `λ_f / λ_f,pre − 1` в процентах — синий
укорочение, красный удлинение.

| Рисунок | Что показывает |
|---------|----------------|
| `deformation_stages.png` | исходная сетка → преднагрузка → пик удара в натуральную величину |
| `displacement_peak.png` | стрелки смещения от преднагрузки на пике |
| `beat_sequence.png` | раскадровка последнего удара |
| `steady_state.png` | наибольшее укорочение и пик T_act по ударам; растяжение по регионам в последнем ударе |
| `beat_animation.gif` | анимация удара |
| `<серия>/figures/deformation_peak_by_stage.png`, `strain_by_region_stage.png` | стадии ишемии рядом; среднее растяжение в регионе и в базовой ткани |

Признак ишемии в механике: в систолу волокно в ишемическом регионе
**удлиняется** (пассивно растягивается сокращающейся здоровой тканью),
а не укорачивается.

Для раздела 1 нужен ранний снимок (≤ 5 мс после первого стимула) — он
служит кадром «после преднагрузки»; ноутбук предупредит, если в нём
уже заметна активная сила.

---

## 5. Свой анализ на Python

Всё ниже работает в ноутбуке (ядро fenicsx-env) или в любом Python с
установленным пакетом.

### 5.1. Открыть прогон

```python
from cardiac_em.analysis import open_run, open_sweep, biomarkers as bm
run = open_run("runs/night_1hz/run_000")

run.status                        # "finished"
run.config                        # конфигурация (dict)
run.param("regions.0.overrides.cell:K_o")   # 9.4
run.meshes["electric"]            # {"nx": 160, "ny": 160, "lx_mm": 8.0, …}
run.warnings                      # список предупреждений
```

### 5.2. Ряды

```python
s = run.series(as_frame=True)     # pandas.DataFrame из series.csv
s.plot(x="t_ms", y=["t_act_mech_max", "t_act_actual_max"])

last = max(run.config["stimulus"]["times_ms"])
bm.mechanics_summary(run.series(), t_from_ms=last)   # метрики последнего удара
```

### 5.3. Карты активации и APD

```python
maps = run.activation()
maps.n_beats                        # число ударов
act = maps.act[:, -1]               # время активации последнего удара, по узлам
apd = maps.apd[:, -1]               # APD последнего удара
grid = maps.grid(act)               # карта (ny+1, nx+1) для imshow

bm.activation_summary(act)          # first_ms, last_ms, total_activation_time_ms, fraction_activated
bm.conduction_velocity(maps.coords, act, axis=0)                  # CV вдоль x, мм/мс
bm.conduction_velocity(maps.coords, act, axis=0, band=(3, 5))     # только полоса 3 ≤ y ≤ 5 мм
g = run.meshes["electric"]
cv = bm.cv_field(grid, g["lx_mm"] / g["nx"], g["ly_mm"] / g["ny"])   # локальная CV
bm.apd_summary(apd, maps.region)    # mean_ms, dispersion_ms, by_region{0: {...}, 1: {...}}
bm.conduction_block(maps.act, beat=3)          # блок между ударами 4 и 5 (нумерация с 0)
rest = bm.restitution(maps.act, maps.repol)    # rest["di_ms"], rest["apd_ms"]
```

### 5.4. Снимки

```python
run.snapshot_times                  # список моментов снимков
sn = run.snapshot(9150.0)           # снимок ровно на этот момент (из списка выше)
sn.state_names                      # ('d', 'f2', …, 'V', …)
V = sn.grid("e_state", "V")         # карта потенциала (ny+1, nx+1)
Ca = sn.grid("e_state", "Ca_i")
lam = sn.grid("m_lambda_f")         # растяжение по ячейкам механики (ny, nx)
ux = sn.grid("m_u", 0)              # x-перемещение узлов
reg = sn["m_region"]                # регион ячеек (плоский массив)
lam_isch = sn["m_lambda_f"][reg == 1].mean()
```

### 5.5. Серия

```python
sweep = open_sweep("runs/night_1hz")
sweep.parameters                    # {"regions.0.overrides.cell:K_o": [9.4, 8.0, 5.4], …}

def metrics(r):
    maps = r.activation()
    a = bm.apd_summary(maps.apd[:, -1], maps.region)
    return {"APD_base": a["by_region"][0]["mean"],
            "APD_isch": a["by_region"][1]["mean"],
            "CV": bm.conduction_velocity(maps.coords, maps.act[:, -1]),
            **bm.mechanics_summary(r.series(), t_from_ms=max(r.config["stimulus"]["times_ms"]))}

import pandas as pd
table = pd.DataFrame(sweep.table(metrics))   # строка на точку: параметры + метрики
table.to_csv("runs/night_1hz/my_table.csv", index=False)
```

### 5.6. Биомаркеры — определения

| Функция | Что считает |
|---------|-------------|
| `activation_summary(act)` | первая и последняя активация, общее время активации ткани, доля возбуждённых узлов |
| `conduction_velocity(coords, act, axis, window, band)` | скорость плоского фронта: 1 / наклон прямой t(x) методом наименьших квадратов по узлам в окне (по умолчанию — средние 40 % области, вдали от стимула и края) |
| `cv_field(grid, hx, hy)` | локальная скорость 1/‖∇t_act‖ по карте; NaN, где градиент ≈ 0 |
| `apd_summary(apd, region)` | APD: среднее, min, max, дисперсия (max − min); по регионам |
| `conduction_block(act, beat)` | узлы, возбуждённые в удар `beat`, но не в `beat + 1` |
| `restitution(act, repol)` | пары DI_k = act_{k+1} − repol_k и APD_{k+1} по всем узлам и ударам |
| `mechanics_summary(series, t_from_ms, t_to_ms)` | в окне: пик T_act и время до пика, пик σ_xx, время до пика, прирост σ_xx, min λ_f, наибольшее укорочение 1 − λ_min/λ_начала, ∫∫T_act dA dt |
| `by_region(values, region)` | n, mean, std, min, max значений по регионам |

Новые биомаркеры — функциями в своём ноутбуке; если пригодятся всем —
обсудить с руководителем добавление в `cardiac_em/analysis/biomarkers.py`.

---

## 6. ParaView

Поля во времени (все переменные клетки, перемещения, напряжения)
смотрятся в ParaView (https://www.paraview.org, бесплатно). Удобнее
скопировать папку прогона к себе (раздел 7) и открыть локально.

1. **File → Open** → `electrics.xdmf` → читатель **Xdmf3 Reader T** →
   **Apply**. `.h5` должен лежать рядом.
2. Вверху выберите поле (`V`, `Ca_i`, `T_act_kPa`) и шкалу; кнопка
   **Rescale to data range over all timesteps** — одна шкала на всё время.
3. Проигрывание — кнопки времени вверху; **File → Save Animation** —
   видео или серия PNG.
4. Механика: откройте `mechanics.xdmf` → выделите его → **Filters →
   Warp By Vector** (вектор `u`, **Scale Factor** 1 — натуральная
   величина) → раскрасьте по `lambda_f` или `T_act_kPa`.
5. Регионы: `regions_electric.xdmf` → **Filters → Contour** по
   `region_id` на уровне 0.5 — граница региона поверх карты.
6. Профиль вдоль линии: **Filters → Plot Over Line**.

---

## 7. Скопировать результаты к себе

Без больших `*.h5`:

```bash
# на своём компьютере
rsync -av --exclude '*.h5' --exclude 'ckpt_*' \
    <логин>@<сервер>:~/cardio2d/runs/night_1hz/ ./night_1hz/
```

С полями для ParaView — без `--exclude '*.h5'` (гигабайты). Одна
картинка — `scp <логин>@<сервер>:~/cardio2d/runs/…/figures/x.png .`,
или в VS Code: правый клик по файлу → **Download…**.

---

## 8. Что сохранять для отчёта

Для каждого результата должно быть понятно, **чем** он получен:

* папка прогона/серии (её `run.json` хранит полную конфигурацию и
  версию кода);
* файл конфигурации/серии из `configs/`;
* таблицы из `analysis/` и `sweep_table.csv`;
* рисунки из `figures/`;
* запись в журнале экспериментов: дата, цель, папка, вывод.

На рисунках указывайте: модель, сетку (h), стадию/параметры региона,
номер удара, момент времени и, для деформации, масштаб `AMPLIFY`.
