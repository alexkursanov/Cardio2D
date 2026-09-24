"""
Биомаркеры: электрика, механика, ритм.
=======================================

Функции принимают массивы numpy (или объекты reader.py) и возвращают
числа и словари. Никаких файлов и расчётных слоёв — только numpy.
Новый биомаркер — новая функция здесь (раздел 9 ARCHITECTURE.md).

Единицы: время — мс, длина — мм, скорость — мм/мс (численно = м/с),
напряжение — кПа.

Электрика (по карте активации одного удара)
--------------------------------------------
    activation_summary     первая/последняя активация, общее время
                           активации, доля активированных узлов
    conduction_velocity    скорость плоского фронта вдоль оси: наклон
                           x(t) в окне, устойчиво к шуму
    cv_field               локальная скорость |∇t_act|⁻¹ по карте
    apd_summary            APD по узлам и по регионам
    conduction_block       узлы, возбуждённые в удар k, но не в k+1

Ритм (по нескольким ударам)
----------------------------
    restitution            пары (DI, APD) для кривой реституции

Механика (по series.csv)
-------------------------
    mechanics_summary      пики T_act и σ_xx и их моменты, наибольшее
                           укорочение, интеграл T_act по времени
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "activation_summary",
    "conduction_velocity",
    "cv_field",
    "apd_summary",
    "conduction_block",
    "restitution",
    "mechanics_summary",
    "by_region",
]


def by_region(values: np.ndarray, region: np.ndarray) -> dict[int, dict]:
    """
    Статистика по регионам: {номер: {n, mean, std, min, max}} без NaN.
    Номер 0 — базовая ткань, i — regions[i-1] конфигурации.
    """
    values = np.asarray(values, dtype=np.float64)
    region = np.asarray(region)
    out = {}
    for r in np.unique(region):
        v = values[(region == r) & np.isfinite(values)]
        out[int(r)] = ({"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
                        "min": float(v.min()), "max": float(v.max())}
                       if v.size else {"n": 0, "mean": np.nan, "std": np.nan,
                                       "min": np.nan, "max": np.nan})
    return out


# ═══════════════════════════════════════════════════════════════════════
#  ЭЛЕКТРИКА
# ═══════════════════════════════════════════════════════════════════════

def activation_summary(t_act: np.ndarray) -> dict:
    """Сводка по времени активации одного удара (NaN — не активирован)."""
    t = np.asarray(t_act, dtype=np.float64)
    ok = np.isfinite(t)
    if not ok.any():
        return {"first_ms": np.nan, "last_ms": np.nan,
                "total_activation_time_ms": np.nan, "fraction_activated": 0.0}
    return {"first_ms": float(t[ok].min()), "last_ms": float(t[ok].max()),
            "total_activation_time_ms": float(t[ok].max() - t[ok].min()),
            "fraction_activated": float(ok.mean())}


def conduction_velocity(coords: np.ndarray, t_act: np.ndarray, *, axis: int = 0,
                        window: tuple[float, float] | None = None,
                        band: tuple[float, float] | None = None) -> float:
    """
    Скорость плоского фронта вдоль оси `axis` (0 — x, 1 — y), мм/мс.

    Берутся узлы с координатой вдоль оси в `window` (по умолчанию —
    средние 40 % диапазона: вдали от зоны стимула и от края, где фронт
    ускоряется у границы с нулевым потоком) и, если задано, с
    поперечной координатой в `band`. Скорость — 1 / наклон прямой
    t(x) по методу наименьших квадратов.
    """
    xy = np.asarray(coords, dtype=np.float64)
    t = np.asarray(t_act, dtype=np.float64)
    s = xy[:, axis]
    if window is None:
        lo, hi = s.min(), s.max()
        window = (lo + 0.3 * (hi - lo), lo + 0.7 * (hi - lo))
    mask = np.isfinite(t) & (s >= window[0]) & (s <= window[1])
    if band is not None:
        q = xy[:, 1 - axis]
        mask &= (q >= band[0]) & (q <= band[1])
    if mask.sum() < 3 or np.ptp(s[mask]) == 0:
        return float("nan")
    slope = np.polyfit(s[mask], t[mask], 1)[0]
    return float(1.0 / slope) if slope != 0 else float("inf")


def cv_field(t_grid: np.ndarray, hx: float, hy: float,
             min_gradient: float = 1e-6) -> np.ndarray:
    """
    Локальная скорость |∇t|⁻¹ по карте времён активации (ny+1, nx+1),
    мм/мс. Центральные разности внутри, односторонние на краях. Где
    градиент почти нулевой (зона одновременной активации — стимул) или
    соседей нет — NaN.
    """
    t = np.asarray(t_grid, dtype=np.float64)
    gy, gx = np.gradient(t, hy, hx)
    g = np.hypot(gx, gy)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = 1.0 / g
    cv[~np.isfinite(g) | (g < min_gradient)] = np.nan
    return cv


def apd_summary(apd: np.ndarray, region: np.ndarray | None = None) -> dict:
    """APD одного удара: общая статистика и (если дан region) по регионам."""
    a = np.asarray(apd, dtype=np.float64)
    ok = np.isfinite(a)
    out = {"n": int(ok.sum()),
           "mean_ms": float(a[ok].mean()) if ok.any() else np.nan,
           "min_ms": float(a[ok].min()) if ok.any() else np.nan,
           "max_ms": float(a[ok].max()) if ok.any() else np.nan,
           "dispersion_ms": float(np.ptp(a[ok])) if ok.any() else np.nan}
    if region is not None:
        out["by_region"] = by_region(a, region)
    return out


def conduction_block(act: np.ndarray, beat: int) -> dict:
    """
    Блок проведения между ударом `beat` и следующим: узлы, возбуждённые
    в удар `beat`, но не возбуждённые в удар `beat + 1`.

    Внимание: удары считаются в каждом узле отдельно (k-я активация
    узла), поэтому при сложной динамике (reentry) k-й удар разных узлов
    может принадлежать разным волнам. Для периодической стимуляции с
    проведением или блоком интерпретация прямая.
    """
    act = np.asarray(act, dtype=np.float64)
    if act.shape[1] <= beat + 1:
        nxt = np.full(act.shape[0], np.nan)
    else:
        nxt = act[:, beat + 1]
    was = np.isfinite(act[:, beat])
    blocked = was & ~np.isfinite(nxt)
    return {"blocked_mask": blocked,
            "n_blocked": int(blocked.sum()),
            "fraction_blocked": float(blocked.sum() / max(was.sum(), 1))}


# ═══════════════════════════════════════════════════════════════════════
#  РИТМ
# ═══════════════════════════════════════════════════════════════════════

def restitution(act: np.ndarray, repol: np.ndarray) -> dict:
    """
    Пары для кривой реституции APD: DI_k = act_{k+1} − repol_k и
    APD_{k+1} = repol_{k+1} − act_{k+1}, по всем узлам и ударам.

    Возвращает {"di_ms", "apd_ms", "node", "beat"} — плоские массивы
    одной длины, только полные пары (без NaN).
    """
    act = np.asarray(act, dtype=np.float64)
    rep = np.asarray(repol, dtype=np.float64)
    if act.shape[1] < 2:
        empty = np.zeros(0)
        return {"di_ms": empty, "apd_ms": empty,
                "node": empty.astype(int), "beat": empty.astype(int)}
    di = act[:, 1:] - rep[:, :-1]
    apd_next = rep[:, 1:] - act[:, 1:]
    ok = np.isfinite(di) & np.isfinite(apd_next)
    node, beat = np.nonzero(ok)
    return {"di_ms": di[ok], "apd_ms": apd_next[ok], "node": node, "beat": beat + 1}


# ═══════════════════════════════════════════════════════════════════════
#  МЕХАНИКА
# ═══════════════════════════════════════════════════════════════════════

def _peak(t, y):
    i = int(np.nanargmax(y))
    return float(y[i]), float(t[i])


def mechanics_summary(series, t_from_ms: float | None = None,
                      t_to_ms: float | None = None) -> dict:
    """
    Механические биомаркеры по ряду series.csv (структурный массив или
    DataFrame) в окне [t_from, t_to] (по умолчанию — весь прогон).

        peak_t_act_kpa, time_to_peak_t_act_ms  — по механической сетке
        peak_sigma_xx_kpa, time_to_peak_sigma_ms, sigma_rise_kpa
                                               — прирост над началом окна
        min_lambda_f, max_shortening           — 1 − λ_min/λ_начала
        t_act_time_integral_kpa_mm2_ms         — ∫∫T_act dA dt
    """
    t = np.asarray(series["t_ms"], dtype=np.float64)
    mask = np.ones_like(t, dtype=bool)
    if t_from_ms is not None:
        mask &= t >= t_from_ms
    if t_to_ms is not None:
        mask &= t <= t_to_ms
    if mask.sum() < 1:
        raise ValueError("в окне нет ни одной точки ряда")
    t = t[mask]
    col = {k: np.asarray(series[k], dtype=np.float64)[mask]
           for k in ("t_act_mech_max", "t_act_mech_integral",
                     "sigma_xx_max", "lambda_f_min")}

    peak_t, t_peak_t = _peak(t, col["t_act_mech_max"])
    peak_s, t_peak_s = _peak(t, col["sigma_xx_max"])
    lam = col["lambda_f_min"]
    i_min = int(np.nanargmin(lam))
    integral = col["t_act_mech_integral"]
    area = float(np.sum(0.5 * (integral[1:] + integral[:-1]) * np.diff(t))) if t.size > 1 else 0.0
    return {
        "peak_t_act_kpa": peak_t,
        "time_to_peak_t_act_ms": t_peak_t - t[0],
        "peak_sigma_xx_kpa": peak_s,
        "time_to_peak_sigma_ms": t_peak_s - t[0],
        "sigma_rise_kpa": peak_s - float(col["sigma_xx_max"][0]),
        "min_lambda_f": float(lam[i_min]),
        "max_shortening": float(1.0 - lam[i_min] / lam[0]),
        "t_act_time_integral_kpa_mm2_ms": area,
    }
