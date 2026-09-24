"""
TNNPM — электромеханическая модель кардиомиоцита.
==================================================

Электрофизиология ten Tusscher–Noble–Noble–Panfilov (TP06) с
четырёхсостоянийной моделью RyR (Shannon), АТФ-зависимым калиевым
током I_K(ATP) и механикой «Екатеринбургской» модели сокращения:
кинетика Ca–тропонина C с кооперативностью, прикрепление поперечных
мостиков N, реологическая схема с последовательным (l₂ − l₁),
параллельным (l₂) и внешним последовательным (l₃) упругими элементами
и вязкостями. Перенос из model.py (А.Г. Курсанов), см. эталон в
tests/tnnpm_reference.py.

ТЕКУЩАЯ РЕДАКЦИЯ (по указанию автора)
-------------------------------------
* СР однокомпартментный — уравнения TP06 (Ca_SR, i_up, i_leak, i_rel).
  Двухкомпартментный СР (nSR/jSR, i_tr) удалён.
* Блок CaMKII (Семин) удалён: i_up — исходная формула TP06.
* I_K(ATP) сохранён. Ишемия — параметрами по регионам: ATP_i, KmATP
  (чувствительность канала к АТФ растёт при ишемии), K_o, K_mNa и др.
  Стадии из сценария автора (норма / 5 / 10 / 15 мин):
      K_o    5.4   6.2   8.0   9.4
      ATP_i  6.8   6.0   4.5   4.0
      KmATP  0.09  0.30  0.35  0.38
* Механика — в двух вариантах:

  "tnnpm" (TNNPMModel) — ДЛЯ ТКАНИ. В клетке только сократительный
      элемент: N, A, l₁, v. Вся пассивная механика — в материале ткани.
      Ткань задаёт клетке l₁ = 1.67·(λ_f − 1) и v = 1.67·dλ_f/dt; клетка
      отдаёт ткани λ·N (сила при нулевой скорости), а множитель p(v)
      ткань вычисляет неявно, внутри равновесия (models/active) — явная
      связь по скорости неустойчива (tests/test_tnnpm.py,
      test_velocity_coupling_must_be_implicit).

  "tnnpm_isometric" (TNNPMIsometricModel) — полная реологическая схема
      оригинала в изометрии (l₂ + l₃ = l₀). Совпадает с моделью автора;
      для одиночной клетки и сверки. В ткани деформацию не видит.

Единицы: время мс, потенциал мВ, концентрации мМ, токи пА/пФ, длины
мкм, силы мН (как в исходной модели).

Активное напряжение
-------------------
    tnnpm:            T_act = T_MAX · λ·p(v)·N / F_REF_CE,  F_REF_CE = 60.56 мН
    tnnpm_isometric:  T_act = T_MAX · (F_XSE − r0) / F_REF,  F_REF = 35.4 мН

Нормировки — пики изометрического удара с параметрами по умолчанию
(у tnnpm — при саркомере 2.1 мкм, это штатная преднагрузка 1.2575).
Так T_MAX — пиковое изометрическое активное напряжение здоровой ткани в
кПа (у модели Роджерса–МакКаллоха пик — T_MAX/4, другая нормировка).

Параметры по узлам
------------------
Параметры из `param_names` можно задавать по регионам (`cell:<имя>` в
overrides региона) — это ишемия: ATP_i, K_o, g_Na, g_CaL, … Остальные
константы общие для всей ткани; их можно изменить аргументом
`constants={...}` при создании модели.
"""

from __future__ import annotations

import math

import numpy as np

from .base import CellModel

__all__ = ["TNNPMModel", "TNNPMIsometricModel", "TNNPM_CONSTANTS", "TNNPM_INITIAL"]


# ═══════════════════════════════════════════════════════════════════════
#  КОНСТАНТЫ (значения из calculate_parameters.py)
# ═══════════════════════════════════════════════════════════════════════

TNNPM_CONSTANTS: dict[str, float] = {
    # клетка
    "T": 310.0, "F": 96485.3415, "R": 8314.472,
    "V_c": 0.016404, "V_sr": 0.001094, "V_ss": 5.47e-05, "Cm": 0.185,
    "Ca_o": 2.0, "Na_o": 140.0, "K_o": 5.4,
    # кальций
    "k_o_Ca": 2.1, "k_i_Ca": 0.025, "k_om": 0.06, "k_im": 0.005,
    "K_up": 0.00025, "max_sr": 2.5, "min_sr": 1.0, "EC": 1.5,
    "V_leak": 0.00036, "V_rel": 0.1224, "V_xfer": 0.00456, "Vmax_up": 0.00765,
    "Buf_c": 0.13, "Buf_sr": 10.0, "Buf_ss": 0.4,
    "K_buf_c": 0.00085, "K_buf_sr": 0.3, "K_buf_ss": 0.00025,
    "A_tot": 0.07, "a_on": 35.0, "a_off": 0.17, "k_A": 28.0,
    # механика
    "alpha_1": 14.6, "beta_1": 4.2, "alpha_2": 14.6, "beta_2": 0.009,
    "alpha_3": 55.0, "beta_3": 0.11, "llambda": 250.0,
    "q_1": 0.0173, "q_2": 0.259, "q_3": 0.0173, "q_4": 0.015,
    "v_max": 0.0055, "a": 0.25, "alpha_Q": 10.0, "beta_Q": 5.0, "x_st": 0.964285,
    "alpha_G": 1.0, "m_0": 0.9, "g_1": 0.6, "g_2": 0.52, "s_0": 1.14,
    "s046": 0.46, "s055": 0.55, "chi_0": 2.1, "chi_1": 0.55, "chi_2": 0.0,
    "pi_min": 0.02, "r0": 2.55248904424517, "d_h": 0.5, "v_1": 0.1,
    "alpha_P": 4.0, "s_c": 1.0, "k_mu": 0.6, "mu": 3.3,
    "n1_A": 0.5, "n1_B": 55.0, "n1_C": 1.0, "n1_K": 1.0, "n1_Q": 0.835, "n1_nu": 5.0,
    "beta_vp_l": 0.1, "alpha_vp_l": 16.0, "beta_vp_s": 10.0, "alpha_vp_s": 16.0,
    "beta_vs_l": 20.0, "alpha_vs_l": 46.0, "beta_vs_s": 60.0, "alpha_vs_s": 39.0,
    # токи
    "g_Kr": 0.153, "g_pK": 0.0146, "g_Ks": 0.392, "g_K1": 5.405,
    "P_NaK": 2.724, "P_kna": 0.03, "K_mk": 1.0, "K_mNa": 40.0,
    "g_to": 0.735, "g_Na": 14.838, "g_bna": 0.00029,
    "alpha": 1.0, "gamma": 0.35, "K_NaCa": 10000.0, "K_sat": 0.1,
    "Km_Ca": 1.38, "Km_Nai": 87.5,
    "g_bCa": 0.000592, "g_pCa": 0.2476, "K_pCa": 0.0005, "g_CaL": 5.0e-5,
    # ишемия (I_K(ATP)); в исходном файле — "[ATP]i", "KmATP", "gKATP"
    "ATP_i": 6.8, "KmATP": 0.0976, "g_KATP": 1.59294,
    # нормировка активного напряжения, мН
    "F_REF": 35.4,          # tnnpm_isometric: пик (F_XSE − r0)
    "F_REF_CE": 60.56,      # tnnpm: пик F_CE изометрического удара при SL 2.1 мкм
    # длины саркомера, мкм (tnnpm)
    "SL_slack": 1.67,       # длина провиса: l₁ = 0
    "SL_rest": 2.1,         # начальная длина одиночной клетки
}

#: Начальные условия (стационар при 1 Гц в диастоле); механика
#: пересчитывается из r0 при создании модели — как в исходном коде.
TNNPM_INITIAL: dict[str, float] = {
    "d": 3.07278916e-05, "f2": 9.99476773e-01, "fCass": 9.99970647e-01,
    "f": 9.82107417e-01, "Ca_SR": 9.28218670e-01, "Ca_i": 4.34295055e-05,
    "Ca_ss": 1.57457522e-04, "h": 7.63521043e-01, "j": 7.62863363e-01,
    "m": 1.47900595e-03, "V": -8.59273503e+01, "K_i": 1.35877968e+02,
    "Xr1": 1.92071800e-04, "Xr2": 4.78422683e-01, "Xs": 3.09815853e-03,
    "Na_i": 1.02371819e+01, "r": 2.15152212e-08, "s": 9.99998122e-01,
    "v": 0.0, "w": 0.0, "N": 1.34966357e-06, "A": 6.31929074e-04,
    "l_1": 3.86134451e-01, "l_2": 3.86910662e-01, "l_3": 5.80515391e-02,
    "R": 9.88267582e-01, "O": 4.23457956e-07, "I": 5.02698275e-09,
    "RI": 1.17319890e-02,
}

_EP = ("d", "f2", "fCass", "f", "Ca_SR", "Ca_i", "Ca_ss", "h", "j", "m",
       "V", "K_i", "Xr1", "Xr2", "Xs", "Na_i", "r", "s")
_RYR = ("R", "O", "I", "RI")
#: полная реологическая схема (изометрия) — как в исходной модели
_STATES_ISOMETRIC = _EP + ("v", "w", "N", "A", "l_1", "l_2", "l_3") + _RYR
#: только сократительный элемент; l_1 и v задаёт ткань
_STATES_CE = _EP + ("v", "N", "A", "l_1") + _RYR
_GATES = ("d", "f2", "fCass", "f", "h", "j", "m", "Xr1", "Xr2", "Xs", "r", "s")

#: Параметры, которые можно задавать по узлам (регионы: "cell:<имя>")
_REGIONAL = ("K_o", "Na_o", "Ca_o", "ATP_i", "KmATP", "g_KATP", "K_mNa",
             "g_Na", "g_CaL", "g_K1", "g_Kr", "g_Ks", "g_to", "g_pK",
             "g_bna", "g_bCa", "g_pCa", "K_NaCa", "P_NaK",
             "Vmax_up", "V_rel", "V_leak")


class _TNNPMCore(CellModel):
    """
    Общая часть обоих вариантов: мембрана, кальций (TP06 + RyR Shannon),
    тропонин C и кинетика мостиков (функции оригинала). Механика —
    в наследниках (`_initial_state`, `_mech_rhs`, `force`).
    """

    state_names: tuple[str, ...] = ()
    param_names = _REGIONAL
    tension_kind = "scaled"
    suggested_dt_ms = 0.02
    activation_threshold = -40.0

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.state_names:
            cls.v_index = cls.state_names.index("V")
            cls.gate_indices = tuple(cls.state_names.index(g) for g in _GATES)

    def __init__(self, constants: dict[str, float] | None = None):
        unknown = set(constants or {}) - set(TNNPM_CONSTANTS)
        if unknown:
            raise KeyError(f"неизвестные константы TNNPM: {sorted(unknown)}")
        self.c = {**TNNPM_CONSTANTS, **(constants or {})}
        self._i = {n: k for k, n in enumerate(self.state_names)}
        self._rest = self._initial_state()

    # ── параметры по умолчанию для тканевых полей ─────────────────────
    def default_params(self) -> dict[str, float]:
        return {k: self.c[k] for k in self.param_names}

    def resting_state(self) -> np.ndarray:
        return self._rest.copy()

    # ═══════════════════════════════════════════════════════════════════
    #  ФУНКЦИИ МЕХАНИКИ (векторные версии функций оригинала)
    # ═══════════════════════════════════════════════════════════════════

    def _chi(self, v):
        c = self.c
        return np.where(v <= 0.0, c["chi_1"] + c["chi_2"] * v / c["v_max"], c["chi_1"])

    def _q(self, v):
        c = self.c
        vm = c["v_max"]
        # в используемой ветке (v > x_st·v_max) основание > 1
        base = np.maximum(1.0 + c["beta_Q"] * (v / vm - c["x_st"]), 1.0)
        return np.where(v <= 0.0, c["q_1"] - c["q_2"] * v / vm,
                        np.where(v <= c["x_st"] * vm,
                                 (c["q_4"] - c["q_3"]) * v / (c["x_st"] * vm) + c["q_3"],
                                 c["q_4"] / base ** c["alpha_Q"]))

    def _P_star(self, x):
        c = self.c
        a, d, x1 = c["a"], c["d_h"], c["v_1"]
        gamma = a * d * x1 ** 2 / (3.0 * a * d - (a + 1.0) * x1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 + d - d ** 2 * a / ((a + 1) * x + d * a + a * d * x ** 2 / gamma)

    def _G_star(self, x):
        c = self.c
        a, x1 = c["a"], c["v_1"]
        den = (0.4 * a + 1.0) * x / a + 1.0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            P = self._P_star(x)
            tail = np.exp(-c["alpha_G"] * np.maximum(x - x1, 0.0) ** c["alpha_P"])
            return np.where(x <= 0, 1.0 + 0.6 * x,
                            np.where(x <= x1, P / den, P * tail / den))

    def _k_p(self, v):
        c = self.c
        return self._chi(v) * c["chi_0"] * self._q(v) * c["m_0"] * self._G_star(v / c["v_max"])

    def _k_m(self, v):
        c = self.c
        return c["chi_0"] * self._q(v) * (1.0 - self._chi(v) * c["m_0"] * self._G_star(v / c["v_max"]))

    def _M(self, A):
        c = self.c
        r = np.maximum(A / c["A_tot"], 0.0) ** c["mu"]
        km = c["k_mu"] ** c["mu"]
        return r * (1.0 + km) / (r + km)

    def _n1(self, l_1):
        c = self.c
        w1 = (c["g_1"] * l_1 + c["g_2"]) * (
            c["n1_A"] + (c["n1_K"] - c["n1_A"])
            / (c["n1_C"] + c["n1_Q"] * np.exp(-c["n1_B"] * l_1)) ** (1.0 / c["n1_nu"]))
        return np.clip(w1, 0.0, 1.0)

    def _L_oz(self, l_1):
        c = self.c
        return np.where(l_1 <= c["s055"], (l_1 + c["s_0"]) / (c["s046"] + c["s_0"]),
                        (c["s_0"] + c["s055"]) / (c["s046"] + c["s_0"]))

    def _pi_NA(self, N, A):
        c = self.c
        with np.errstate(divide="ignore", invalid="ignore"):
            NA = c["A_tot"] * c["s_c"] * N / A
        return np.where(NA <= 0.0, 1.0,
                        np.where(NA <= 1.0, c["pi_min"] ** np.clip(NA, 0.0, 1.0), c["pi_min"]))

    def _p(self, v):
        c = self.c
        a, vm, x1 = c["a"], c["v_max"], c["v_1"]
        x = v / vm
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            neg = a * (1.0 + x) / ((a - x) * (1.0 + 0.6 * x))
            lin = (0.4 * a + 1.0) * x / a + 1.0
            fast = lin * np.exp(c["alpha_G"] * np.maximum(x - x1, 0.0) ** c["alpha_P"])
        return np.where(v <= -vm, 0.0, np.where(v <= 0.0, neg, np.where(v <= x1 * vm, lin, fast)))

    def _p_prime(self, v):
        c = self.c
        a, vm, x1, aG, aP = c["a"], c["v_max"], c["v_1"], c["alpha_G"], c["alpha_P"]
        x = v / vm
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            below = np.full_like(v, a * (0.4 + 0.4 * a) / (vm * ((a + 1.0) * 0.4) ** 2))
            neg = a * (1.0 + 0.4 * a + 1.2 * x + 0.6 * x ** 2) / (vm * ((a - x) * (1.0 + 0.6 * x)) ** 2)
            lin = np.full_like(v, (0.4 * a + 1.0) / (a * vm))
            dx = np.maximum(x - x1, 0.0)
            fast = np.exp(aG * dx ** aP) * ((0.4 * a + 1.0) / a + aG * aP
                                            * (1.0 + (0.4 * a + 1.0) * x / a) * dx ** (aP - 1.0)) / vm
        return np.where(v <= -vm, below,
                        np.where(v <= 0.0, neg, np.where(v <= x1 * vm, lin, fast)))

    # ═══════════════════════════════════════════════════════════════════
    #  ВОРОТНЫЕ ПЕРЕМЕННЫЕ
    # ═══════════════════════════════════════════════════════════════════

    def gate_inf_tau(self, t, y, params):
        V = y[:, self._i["V"]]
        Ca_ss = y[:, self._i["Ca_ss"]]
        e = np.exp

        d_inf = 1.0 / (1.0 + e((-8.0 - V) / 7.5))
        alpha_d = 1.4 / (1.0 + e((-35.0 - V) / 13.0)) + 0.25
        beta_d = 1.4 / (1.0 + e((V + 5.0) / 5.0))
        gamma_d = 1.0 / (1.0 + e((50.0 - V) / 20.0))
        tau_d = alpha_d * beta_d + gamma_d

        f2_inf = 0.67 / (1.0 + e((V + 35.0) / 7.0)) + 0.33
        tau_f2 = (562.0 * e(-((V + 27.0) ** 2) / 240.0)
                  + 31.0 / (1.0 + e((25.0 - V) / 10.0))
                  + 80.0 / (1.0 + e((V + 30.0) / 10.0)))

        fCass_inf = 0.6 / (1.0 + (Ca_ss / 0.05) ** 2) + 0.4
        tau_fCass = 80.0 / (1.0 + (Ca_ss / 0.05) ** 2) + 2.0

        f_inf = 1.0 / (1.0 + e((V + 20.0) / 7.0))
        tau_f = (1102.5 * e(-((V + 27.0) ** 2) / 225.0)
                 + 200.0 / (1.0 + e((13.0 - V) / 10.0))
                 + 180.0 / (1.0 + e((V + 30.0) / 10.0)) + 20.0)

        low = V < -40.0
        h_inf = 1.0 / (1.0 + e((V + 71.55) / 7.43)) ** 2
        alpha_h = np.where(low, 0.057 * e(-(V + 80.0) / 6.8), 0.0)
        beta_h = np.where(low, 2.7 * e(0.079 * V) + 310000.0 * e(0.3485 * V),
                          0.77 / (0.13 * (1.0 + e((V + 10.66) / -11.1))))
        tau_h = 1.0 / (alpha_h + beta_h)

        j_inf = h_inf
        alpha_j = np.where(low, (-25428.0 * e(0.2444 * V) - 6.948e-6 * e(-0.04391 * V))
                           * (V + 37.78) / (1.0 + e(0.311 * (V + 79.23))), 0.0)
        beta_j = np.where(low, 0.02424 * e(-0.01052 * V) / (1.0 + e(-0.1378 * (V + 40.14))),
                          0.6 * e(0.057 * V) / (1.0 + e(-0.1 * (V + 32.0))))
        tau_j = 1.0 / (alpha_j + beta_j)

        m_inf = 1.0 / (1.0 + e((-56.86 - V) / 9.03)) ** 2
        alpha_m = 1.0 / (1.0 + e((-60.0 - V) / 5.0))
        beta_m = 0.1 / (1.0 + e((V + 35.0) / 5.0)) + 0.1 / (1.0 + e((V - 50.0) / 200.0))
        tau_m = alpha_m * beta_m

        xr1_inf = 1.0 / (1.0 + e((-26.0 - V) / 7.0))
        tau_xr1 = (450.0 / (1.0 + e((-45.0 - V) / 10.0))) * (6.0 / (1.0 + e((V + 30.0) / 11.5)))
        xr2_inf = 1.0 / (1.0 + e((V + 88.0) / 24.0))
        tau_xr2 = (3.0 / (1.0 + e((-60.0 - V) / 20.0))) * (1.12 / (1.0 + e((V - 60.0) / 20.0)))
        xs_inf = 1.0 / (1.0 + e((-5.0 - V) / 14.0))
        tau_xs = (1400.0 / np.sqrt(1.0 + e((5.0 - V) / 6.0))) * (1.0 / (1.0 + e((V - 35.0) / 15.0))) + 80.0

        r_inf = 1.0 / (1.0 + e((20.0 - V) / 6.0))
        tau_r = 9.5 * e(-((V + 40.0) ** 2) / 1800.0) + 0.8
        s_inf = 1.0 / (1.0 + e((V + 20.0) / 5.0))
        tau_s = 85.0 * e(-((V + 45.0) ** 2) / 320.0) + 5.0 / (1.0 + e((V - 20.0) / 5.0)) + 3.0

        # порядок — как в _GATES
        inf = np.stack([d_inf, f2_inf, fCass_inf, f_inf, h_inf, j_inf, m_inf,
                        xr1_inf, xr2_inf, xs_inf, r_inf, s_inf], axis=1)
        tau = np.stack([tau_d, tau_f2, tau_fCass, tau_f, tau_h, tau_j, tau_m,
                        tau_xr1, tau_xr2, tau_xs, tau_r, tau_s], axis=1)
        return inf, tau

    # ═══════════════════════════════════════════════════════════════════
    #  ТОКИ И ОСТАЛЬНЫЕ ПЕРЕМЕННЫЕ
    # ═══════════════════════════════════════════════════════════════════

    def currents(self, y, params) -> dict[str, np.ndarray]:
        """Ионные токи, пА/пФ — для анализа и для правых частей."""
        c = self.c
        p = params
        g = lambda n: y[:, self._i[n]]                               # noqa: E731
        V, Ca_i, Ca_ss, Na_i, K_i = g("V"), g("Ca_i"), g("Ca_ss"), g("Na_i"), g("K_i")
        FRT = c["F"] / (c["R"] * c["T"])
        RTF = 1.0 / FRT
        K_o, Na_o, Ca_o = p["K_o"], p["Na_o"], p["Ca_o"]
        e = np.exp

        E_Na = RTF * np.log(Na_o / Na_i)
        E_K = RTF * np.log(K_o / K_i)
        E_Ca = 0.5 * RTF * np.log(Ca_o / Ca_i)
        E_Ks = RTF * np.log((K_o + c["P_kna"] * Na_o) / (K_i + c["P_kna"] * Na_i))

        gam = c["gamma"]
        i_NaCa = (p["K_NaCa"]
                  * (e(gam * V * FRT) * Na_i ** 3 * Ca_o
                     - e((gam - 1.0) * V * FRT) * Na_o ** 3 * Ca_i * c["alpha"])
                  / ((c["Km_Nai"] ** 3 + Na_o ** 3) * (c["Km_Ca"] + Ca_o)
                     * (1.0 + c["K_sat"] * e((gam - 1.0) * V * FRT))))
        i_b_Ca = p["g_bCa"] * (V - E_Ca)
        i_p_Ca = p["g_pCa"] * Ca_i / (Ca_i + c["K_pCa"])

        ex = e(2.0 * (V - 15.0) * FRT)
        i_CaL = (p["g_CaL"] * g("d") * g("f") * g("f2") * g("fCass") * 4.0 * (V - 15.0)
                 * c["F"] * FRT * (0.25 * Ca_ss * ex - Ca_o) / (ex - 1.0))

        i_Na = p["g_Na"] * g("m") ** 3 * g("h") * g("j") * (V - E_Na)

        alpha_K1 = 0.1 / (1.0 + e(0.06 * (V - E_K - 200.0)))
        beta_K1 = ((3.0 * e(0.0002 * (V - E_K + 100.0)) + e(0.1 * (V - E_K - 10.0)))
                   / (1.0 + e(-0.5 * (V - E_K))))
        i_K1 = p["g_K1"] * alpha_K1 / (alpha_K1 + beta_K1) * np.sqrt(K_o / 5.4) * (V - E_K)
        i_to = p["g_to"] * g("r") * g("s") * (V - E_K)
        i_Kr = p["g_Kr"] * np.sqrt(K_o / 5.4) * g("Xr1") * g("Xr2") * (V - E_K)
        i_Ks = p["g_Ks"] * g("Xs") ** 2 * (V - E_Ks)

        P_ATP = 1.0 / (1.0 + (p["ATP_i"] / p["KmATP"]) ** 2.2)
        i_K_ATP = p["g_KATP"] * P_ATP * (K_o / 5.4) ** 0.24 * (V - E_K)

        i_NaK = (p["P_NaK"] * K_o * Na_i
                 / ((K_o + c["K_mk"]) * (Na_i + p["K_mNa"])
                    * (1.0 + 0.1245 * e(-0.1 * V * FRT) + 0.0353 * e(-V * FRT))))
        i_b_Na = p["g_bna"] * (V - E_Na)
        i_p_K = p["g_pK"] * (V - E_K) / (1.0 + e((25.0 - V) / 5.98))

        return {"i_Na": i_Na, "i_CaL": i_CaL, "i_NaCa": i_NaCa, "i_NaK": i_NaK,
                "i_K1": i_K1, "i_to": i_to, "i_Kr": i_Kr, "i_Ks": i_Ks,
                "i_K_ATP": i_K_ATP, "i_b_Na": i_b_Na, "i_b_Ca": i_b_Ca,
                "i_p_Ca": i_p_Ca, "i_p_K": i_p_K}

    def rhs_non_gate(self, t, y, stim, params):
        c = self.c
        p = params
        g = lambda n: y[:, self._i[n]]                               # noqa: E731
        Ca_SR, Ca_i, Ca_ss = g("Ca_SR"), g("Ca_i"), g("Ca_ss")
        N, A = g("N"), g("A")
        R, O, I, RI = g("R"), g("O"), g("I"), g("RI")
        e = np.exp

        cur = self.currents(y, params)
        i_Stim = -stim            # в оригинале стимул входит со знаком минус

        # ── тропонин C ────────────────────────────────────────────────
        A_off = c["a_off"] * self._pi_NA(N, A) * e(-c["k_A"] * A)
        dA = c["a_on"] * (c["A_tot"] - A) * Ca_i - A_off * A

        # ── СР (TP06) и RyR (Shannon) ─────────────────────────────────
        i_up = p["Vmax_up"] / (1.0 + c["K_up"] ** 2 / Ca_i ** 2)
        i_leak = p["V_leak"] * (Ca_SR - Ca_i)
        k_CaSR = c["max_sr"] - (c["max_sr"] - c["min_sr"]) / (1.0 + (c["EC"] / Ca_SR) ** 2)
        k_o = c["k_o_Ca"] / k_CaSR
        k_i = c["k_i_Ca"] * k_CaSR
        k_om, k_im = c["k_om"], c["k_im"]
        dR = k_im * RI - k_i * R * Ca_ss - k_o * R * Ca_ss ** 2 + k_om * O
        dO = k_o * R * Ca_ss ** 2 - k_om * O - k_i * O * Ca_ss + k_im * I
        dI = k_i * O * Ca_ss - k_im * I - k_om * I + k_o * RI * Ca_ss ** 2
        dRI = k_om * I - k_o * RI * Ca_ss ** 2 - k_im * RI + k_i * R * Ca_ss

        i_rel = p["V_rel"] * O * (Ca_SR - Ca_ss)
        i_xfer = c["V_xfer"] * (Ca_ss - Ca_i)
        dCa_SR = (1.0 / (1.0 + c["Buf_sr"] * c["K_buf_sr"] / (Ca_SR + c["K_buf_sr"]) ** 2)
                  * (i_up - (i_rel + i_leak)))

        Cm, Vc, F_ = c["Cm"], c["V_c"], c["F"]
        dCa_i = (1.0 / (1.0 + c["Buf_c"] * c["K_buf_c"] / (Ca_i + c["K_buf_c"]) ** 2)
                 * ((i_leak - i_up) * c["V_sr"] / Vc + i_xfer
                    - (cur["i_b_Ca"] + cur["i_p_Ca"] - 2.0 * cur["i_NaCa"]) * Cm / (2.0 * Vc * F_)
                    - dA))
        dCa_ss = (1.0 / (1.0 + c["Buf_ss"] * c["K_buf_ss"] / (Ca_ss + c["K_buf_ss"]) ** 2)
                  * (-cur["i_CaL"] * Cm / (2.0 * c["V_ss"] * F_)
                     + i_rel * c["V_sr"] / c["V_ss"]
                     - i_xfer * Vc / c["V_ss"]))

        # ── мембрана ──────────────────────────────────────────────────
        i_ion = sum(cur.values())
        dV = -(i_ion + i_Stim)
        dK_i = -(cur["i_K1"] + cur["i_to"] + cur["i_Kr"] + cur["i_Ks"] + cur["i_p_K"]
                 + i_Stim - 2.0 * cur["i_NaK"] + cur["i_K_ATP"]) / (Vc * F_) * Cm
        dNa_i = -(cur["i_Na"] + cur["i_b_Na"] + 3.0 * cur["i_NaK"] + 3.0 * cur["i_NaCa"]) / (Vc * F_) * Cm

        out = {"Ca_SR": dCa_SR, "Ca_i": dCa_i, "Ca_ss": dCa_ss, "V": dV, "K_i": dK_i,
               "Na_i": dNa_i, "A": dA, "R": dR, "O": dO, "I": dI, "RI": dRI}
        out.update(self._mech_rhs(y))
        # порядок — как в non_gate_indices (порядок state_names без ворот)
        return np.stack([out[self.state_names[i]] for i in self.non_gate_indices], axis=1)

    def calcium(self, y):
        return y[:, self._i["Ca_i"]]

    def _ix(self, name: str) -> int:
        return self._i[name]


# ═══════════════════════════════════════════════════════════════════════
#  ВАРИАНТ 1: ПОЛНАЯ РЕОЛОГИЧЕСКАЯ СХЕМА, ИЗОМЕТРИЯ (как в оригинале)
# ═══════════════════════════════════════════════════════════════════════

class TNNPMIsometricModel(_TNNPMCore):
    """
    TNNPM с полной реологической схемой (CE, SE, PE, XSE и вязкости) в
    изометрии: l₂ + l₃ = l₀. Совпадает с исходной моделью автора (см.
    tests/tnnpm_reference.py). Для одиночной клетки и сверки; в ткани
    длину клетки не видит — там нужен вариант `tnnpm`.

        T_act = T_MAX · (F_XSE − r0) / F_REF,  F_REF = 35.4 мН
    """

    name = "tnnpm_isometric"
    state_names = _STATES_ISOMETRIC

    # ── начальное состояние ───────────────────────────────────────────
    def _initial_state(self) -> np.ndarray:
        """
        Электрика и кальций — из TNNPM_INITIAL; механика — изометрическое
        равновесие при силе преднагрузки r0 (calculate_init_conditions
        оригинала: деление пополам по l₂).
        """
        c = self.c
        y = np.array([TNNPM_INITIAL[n] for n in self.state_names], dtype=np.float64)
        l_2 = self._bisect_l2()
        y[self._ix("l_2")] = l_2
        y[self._ix("l_1")] = l_2 + (math.log(c["beta_1"]) - math.log(
            c["r0"] + c["beta_1"] - c["beta_2"] * (math.exp(c["alpha_2"] * l_2) - 1))) / c["alpha_1"]
        y[self._ix("N")] = self._N0(l_2)
        y[self._ix("l_3")] = math.log((c["r0"] + c["beta_3"]) / c["beta_3"]) / c["alpha_3"]
        y[self._ix("v")] = 0.0
        y[self._ix("w")] = 0.0
        return y

    def _N0(self, l0: float) -> float:
        c = self.c
        return (c["r0"] - c["beta_2"] * (math.exp(c["alpha_2"] * l0) - 1)) / c["llambda"]

    def _fi(self, l_1: float) -> float:
        """fi(l) оригинала: dN/dt = 0 при v = 0 и N = N0(l)."""
        one = np.array([0.0])
        A = np.array([6.31929074e-04])            # как в оригинале
        l = np.array([l_1])
        N0 = self._N0(l_1)
        return float(self._k_p(one)[0] * self._M(A)[0] * self._n1(l)[0] * self._L_oz(l)[0]
                     * (1.0 - N0) - self._k_m(one)[0] * N0)

    def _bisect_l2(self) -> float:
        c = self.c
        l100 = math.log((c["r0"] + c["beta_2"]) / c["beta_2"]) / c["alpha_2"]
        a, b = 0.9 * l100, l100
        if self._fi(a) == 0.0:
            return a
        if self._fi(b) == 0.0:
            return b
        x = a + (b - a) / 2.0
        for _ in range(200):
            if abs(self._fi(x)) < 1e-7:
                break
            x = a + (b - a) / 2.0
            if self._fi(x) < 0:
                a = x
            else:
                b = x
        return x

    @property
    def l0(self) -> float:
        """Длина l₂ + l₃ изометрического препарата, мкм."""
        return float(self._rest[self._ix("l_2")] + self._rest[self._ix("l_3")])

    def _mech_rhs(self, y):
        c = self.c
        g = lambda n: y[:, self._i[n]]                           # noqa: E731
        v, w, N, A = g("v"), g("w"), g("N"), g("A")
        l_1, l_2, l_3 = g("l_1"), g("l_2"), g("l_3")
        e = np.exp
        K_chi = (self._k_p(v) * self._M(A) * self._n1(l_1) * self._L_oz(l_1) * (1.0 - N)
                 - self._k_m(v) * N)
        dN = K_chi

        neg_v = v <= 0.0
        alpha_p = np.where(neg_v, c["alpha_vp_l"], c["alpha_vp_s"])
        k_P_vis = np.where(neg_v, c["beta_vp_l"] * e(c["alpha_vp_l"] * l_1),
                           c["beta_vp_s"] * e(c["alpha_vp_s"] * l_1))
        slow_w = w <= v
        alpha_s = np.where(slow_w, c["alpha_vs_l"], c["alpha_vs_s"])
        k_S_vis = np.where(slow_w, c["beta_vs_l"] * e(c["alpha_vs_l"] * (l_2 - l_1)),
                           c["beta_vs_s"] * e(c["alpha_vs_s"] * (l_2 - l_1)))

        stiff_23 = (c["alpha_2"] * c["beta_2"] * e(c["alpha_2"] * l_2)
                    + c["alpha_3"] * c["beta_3"] * e(c["alpha_3"] * l_3))
        lam = c["llambda"]
        phi = -(lam * K_chi * self._p(v) + alpha_p * k_P_vis * v ** 2 + stiff_23 * w) \
            / (lam * N * self._p_prime(v) + k_P_vis)
        dv = phi
        dw = (phi - alpha_s * (w - v) ** 2
              - (c["alpha_1"] * c["beta_1"] * e(c["alpha_1"] * (l_2 - l_1)) * (w - v)
                 + stiff_23 * w) / k_S_vis)

        return {"v": dv, "w": dw, "N": dN, "l_1": v, "l_2": w, "l_3": -w}

    def force(self, y) -> np.ndarray:
        """Сила на внешнем последовательном элементе, мН."""
        c = self.c
        return c["beta_3"] * (np.exp(c["alpha_3"] * y[:, self._i["l_3"]]) - 1.0)

    def active_tension(self, y, params):
        """(F − r0) / F_REF — безразмерно; вызывающий умножает на T_MAX."""
        c = self.c
        return (self.force(y) - c["r0"]) / c["F_REF"]


# ═══════════════════════════════════════════════════════════════════════
#  ВАРИАНТ 2: ТОЛЬКО СОКРАТИТЕЛЬНЫЙ ЭЛЕМЕНТ — ДЛЯ ТКАНИ
# ═══════════════════════════════════════════════════════════════════════

class TNNPMModel(_TNNPMCore):
    """
    TNNPM для ткани: на уровне клетки остаётся только сократительный
    элемент (CE). Вся пассивная механика (PE, SE, XSE, вязкости) — на
    уровне ткани, в её гиперупругом материале. Решение автора модели.

    Механика клетки:
        l₁  — удлинение CE от длины провиса саркомера SL_slack = 1.67 мкм;
              задаёт ТКАНЬ: l₁ = SL_slack · (λ_f − 1)
        v   — скорость CE, dl₁/dt; задаёт ТКАНЬ: v = SL_slack · dλ_f/dt
        N   — доля прикреплённых мостиков, dN/dt = k₊(v)·M(A)·n₁(l₁)·L(l₁)·(1−N) − k₋(v)·N
        A   — Ca–тропонин C (как в полной модели, с кооперативностью π(N, A))

    Между механическими шагами l₁ продолжается по последней скорости
    (dl₁/dt = v, dv/dt = 0) и на каждом механическом шаге заменяется
    значением от ткани (`apply_stretch`).

    Сила и активное напряжение:
        F_CE  = λ · p(v) · N                          (мН)
        T_act = T_MAX · F_CE / F_REF_CE

    F_REF_CE — пик F_CE в изометрическом ударе при саркомере 2.1 мкм
    (l₁ = 0.43 мкм — это и есть штатная преднагрузка λ_f = 2.1/1.67 =
    1.2575), стационар при 1 Гц. Так T_MAX — пиковое активное напряжение
    здоровой ткани в кПа при этой длине; на других длинах и при
    укорочении сила меньше или больше — по n₁(l₁), L(l₁) и p(v).
    """

    name = "tnnpm"
    state_names = _STATES_CE
    stretch_sensitive = True
    tissue_active_law = "tnnpm_force_velocity"

    def _initial_state(self) -> np.ndarray:
        """
        Электрика и кальций — из TNNPM_INITIAL; l₁ — при саркомере
        SL_rest (по умолчанию 2.1 мкм), v = 0, N — стационар dN/dt = 0
        при этой длине и покойном A.
        """
        c = self.c
        y = np.array([TNNPM_INITIAL[n] for n in self.state_names], dtype=np.float64)
        y[self._i["l_1"]] = c["SL_rest"] - c["SL_slack"]
        y[self._i["v"]] = 0.0
        y[self._i["N"]] = self._N_steady(y[None, :])[0]
        return y

    def _N_steady(self, y) -> np.ndarray:
        g = lambda n: y[:, self._i[n]]                           # noqa: E731
        v, A, l_1 = g("v"), g("A"), g("l_1")
        on = self._k_p(v) * self._M(A) * self._n1(l_1) * self._L_oz(l_1)
        return on / (on + self._k_m(v))

    def _mech_rhs(self, y):
        g = lambda n: y[:, self._i[n]]                           # noqa: E731
        v, N, A, l_1 = g("v"), g("N"), g("A"), g("l_1")
        dN = (self._k_p(v) * self._M(A) * self._n1(l_1) * self._L_oz(l_1) * (1.0 - N)
              - self._k_m(v) * N)
        return {"N": dN, "l_1": v, "v": np.zeros_like(v)}

    # ── связь с тканью ────────────────────────────────────────────────
    def apply_stretch(self, y: np.ndarray, stretch: np.ndarray,
                      stretch_rate: np.ndarray) -> None:
        """
        Задать длину и скорость CE по растяжению волокна ткани λ_f и
        его скорости dλ_f/dt (на месте, в массиве состояния).
        """
        sl0 = self.c["SL_slack"]
        y[:, self._i["l_1"]] = sl0 * (np.asarray(stretch) - 1.0)
        y[:, self._i["v"]] = sl0 * np.asarray(stretch_rate)

    def sarcomere_length(self, y) -> np.ndarray:
        """Длина саркомера, мкм."""
        return self.c["SL_slack"] + y[:, self._i["l_1"]]

    def force(self, y) -> np.ndarray:
        """Сила сократительного элемента F_CE = λ·p(v)·N, мН."""
        c = self.c
        return c["llambda"] * self._p(y[:, self._i["v"]]) * y[:, self._i["N"]]

    def active_tension(self, y, params):
        """F_CE / F_REF_CE при текущей скорости — безразмерно (× T_MAX)."""
        return self.force(y) / self.c["F_REF_CE"]

    def isometric_tension(self, y, params):
        """
        λ·N / F_REF_CE — сила при v = 0. В ткань передаётся она, а
        множитель p(v) ткань вычисляет неявно (models/active).
        """
        return self.c["llambda"] * y[:, self._i["N"]] / self.c["F_REF_CE"]
