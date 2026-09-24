"""
ЭТАЛОН для проверки cardiac_em/models/cell/tnnpm.py — НЕ часть пакета.

Это model.py из кода TNNPM (А.Г. Курсанов) с минимальными правками по
указанию автора: удалены двухкомпартментный СР и блок CaMKII (Семин),
I_K(ATP) сохранён. СР возвращён к однокомпартментной схеме TP06 — теми
уравнениями, что в оригинале стояли закомментированными. Правки
помечены [ПРАВКА]; остальной текст — без изменений, чтобы расхождение
векторной реализации с ним указывало на ошибку переноса, а не на
разницу в физике. Параметры встроены (значения из calculate_parameters.py).

Переменные состояния (29):
    d f2 fCass f Ca_SR Ca_i Ca_ss h j m V K_i Xr1 Xr2 Xs Na_i r s
    v w N A l_1 l_2 l_3 R O I RI
"""

# from numpy import exp, log, sqrt, floor
from math import exp, floor, log, sqrt

import numpy as np

PARAM = {
 "stim_start": 10.0,
 "stim_period": 1000.0,
 "stim_duration": 1.0,
 "stim_amplitude": 52.0,
 "T": 310.0,
 "F": 96485.3415,
 "R": 8314.472,
 "V_c": 0.016404,
 "V_sr": 0.001094,
 "V_ss": 5.47e-05,
 "Ca_o": 2.0,
 "Na_o": 140.0,
 "K_o": 5.4,
 "Cm": 0.185,
 "k_o_Ca": 2.1,
 "k_i_Ca": 0.025,
 "k_om": 0.06,
 "k_im": 0.005,
 "K_up": 0.00025,
 "max_sr": 2.5,
 "min_sr": 1.0,
 "EC": 1.5,
 "V_leak": 0.00036,
 "V_rel": 0.1224,
 "V_xfer": 0.00456,
 "Vmax_up": 0.00765,
 "Buf_c": 0.13,
 "Buf_sr": 10.0,
 "Buf_ss": 0.4,
 "K_buf_c": 0.00085,
 "K_buf_sr": 0.3,
 "K_buf_ss": 0.00025,
 "A_tot": 0.07,
 "a_on": 35.0,
 "a_off": 0.17,
 "k_A": 28.0,
 "kdecay": 10.0,
 "K_inh": 1.0,
 "alpha_1": 14.6,
 "beta_1": 4.2,
 "alpha_2": 14.6,
 "beta_2": 0.009,
 "llambda": 250.0,
 "q_1": 0.0173,
 "q_2": 0.259,
 "q_3": 0.0173,
 "q_4": 0.015,
 "v_max": 0.0055,
 "a": 0.25,
 "alpha_Q": 10.0,
 "beta_Q": 5.0,
 "x_st": 0.964285,
 "alpha_G": 1.0,
 "m_0": 0.9,
 "g_1": 0.6,
 "g_2": 0.52,
 "s_0": 1.14,
 "s046": 0.46,
 "s055": 0.55,
 "chi_1": 0.55,
 "pi_min": 0.02,
 "r0": 2.55248904424517,
 "d_h": 0.5,
 "v_1": 0.1,
 "x0": 1.0,
 "m": 0.00147900595,
 "chi_0": 2.1,
 "alpha_P": 4.0,
 "q_st": 1000.0,
 "alpha_3": 55.0,
 "beta_3": 0.11,
 "alpha_tir": 30.0,
 "beta_tir": 0.0,
 "l_tir": -0.03,
 "beta_vp_l": 0.1,
 "alpha_vp_l": 16.0,
 "beta_vp_s": 10.0,
 "alpha_vp_s": 16.0,
 "beta_vs_l": 20.0,
 "alpha_vs_l": 46.0,
 "beta_vs_s": 60.0,
 "alpha_vs_s": 39.0,
 "s_c": 1.0,
 "k_mu": 0.6,
 "mu": 3.3,
 "chi_2": 0.0,
 "n1_B": 55.0,
 "n1_Q": 0.835,
 "n1_K": 1.0,
 "n1_A": 0.5,
 "n1_C": 1.0,
 "n1_nu": 5.0,
 "g_Kr": 0.153,
 "g_pK": 0.0146,
 "g_Ks": 0.392,
 "g_K1": 5.405,
 "P_NaK": 2.724,
 "P_kna": 0.03,
 "K_mk": 1.0,
 "K_mNa": 40.0,
 "g_to": 0.735,
 "g_Na": 14.838,
 "g_bna": 0.00029,
 "alpha": 1.0,
 "gamma": 0.35,
 "K_NaCa": 10000.0,
 "K_sat": 0.1,
 "Km_Ca": 1.38,
 "Km_Nai": 87.5,
 "g_bCa": 0.000592,
 "g_pCa": 0.2476,
 "K_pCa": 0.0005,
 "g_CaL": 5e-05,
 "Ca_sense": 0.0,
 "k_Ca_sense": 0.0069,
 "[ATP]i": 6.8,
 "KmATP": 0.0976,
 "gKATP": 1.59294,
 "Ko,norm": 4.0,
 "kp": 1000.0,
 "KP": 0.000325,
 "V_nSR": 0.0100648,
 "V_jSR": 0.0008752,
 "K_buf_jsr": 0.8,
 "Buf_jsr": 10.0,
 "Cai": 4.34295055e-05,
 "CaTn": 0.000631929074,
 "CaSS": 0.000157457522,
 "CaSR": 0.92821867,
 "l1": 0.386134451,
 "l2": 0.386910662,
 "N": 1.34966357e-06,
 "l3": 0.0580515391,
 "v": 3.39882644e-06,
 "w": 8.52399973e-07,
 "xr1": 0.0001920718,
 "xr2": 0.478422683,
 "xs": 0.00309815853,
 "r": 2.15152212e-08,
 "s": 0.999998122,
 "h": 0.763521043,
 "j": 0.762863363,
 "d": 3.07278916e-05,
 "f": 0.982107417,
 "f2": 0.999476773,
 "fCaSS": 0.999970647,
 "Nai": 10.2371819,
 "Ki": 135.877968,
 "E": -85.9273503,
 "R_RyR": 0.988267582,
 "O_RyR": 4.23457956e-07,
 "I_RyR": 5.02698275e-09,
 "RI_RyR": 0.011731989,
 "p_iup": 0.488823882,
 "CaMKt": 0.0110752904836162,
 "l0": 0.4449622011,
 "Ca_nSR": 0.42821867,
 "Ca_jSR": 0.48821867
}

param = {k: {'value': v} for k, v in PARAM.items()}


class TNNPM(object):
    def __init__(self) -> object:

        self.currents = np.zeros(11, dtype=np.float64)
        self.forces = np.zeros(6, dtype=np.float64)
        self.F_afterload = 0.0
        self.l0 = param["l0"]["value"]
        self.r0 = param["r0"]["value"]
        # meh
        self.llambda = param["llambda"]["value"]
        self.alpha_vp_l = param["alpha_vp_l"]["value"]
        self.beta_vp_l = param["beta_vp_l"]["value"]
        self.alpha_vp_s = param["alpha_vp_s"]["value"]
        self.beta_vp_s = param["beta_vp_s"]["value"]

        self.alpha_vs_l = param["alpha_vs_l"]["value"]
        self.beta_vs_l = param["beta_vs_l"]["value"]
        self.alpha_vs_s = param["alpha_vs_s"]["value"]
        self.beta_vs_s = param["beta_vs_s"]["value"]

        self.alpha_1 = param["alpha_1"]["value"]
        self.beta_1 = param["beta_1"]["value"]
        self.alpha_2 = param["alpha_2"]["value"]
        self.beta_2 = param["beta_2"]["value"]
        self.alpha_3 = param["alpha_3"]["value"]
        self.beta_3 = param["beta_3"]["value"]

        # N
        self.m_0 = param["m_0"]["value"]
        self.chi_0 = param["chi_0"]["value"]
        self.chi_1 = param["chi_1"]["value"]
        self.chi_2 = param["chi_2"]["value"]
        self.v_max = param["v_max"]["value"]  #
        # q
        self.q_1 = param["q_1"]["value"]
        self.q_2 = param["q_2"]["value"]
        self.q_3 = param["q_3"]["value"]
        self.q_4 = param["q_4"]["value"]
        self.x_st = param["x_st"]["value"]
        self.alpha_Q = param["alpha_Q"]["value"]
        self.beta_Q = param["beta_Q"]["value"]
        # Gstar
        self.a = param["a"]["value"]
        self.v_1 = param["v_1"]["value"]  # Rst
        self.alpha_G = param["alpha_G"]["value"]
        self.alpha_P = param["alpha_P"]["value"]
        # Pstar
        self.d_h = param["d_h"]["value"]
        # M
        self.A_tot = param["A_tot"]["value"]
        self.mu = param["mu"]["value"]
        self.k_mu = param["k_mu"]["value"]
        # n1
        self.g_1 = param["g_1"]["value"]
        self.g_2 = param["g_2"]["value"]
        self.n1_A = param["n1_A"]["value"]
        self.n1_B = param["n1_B"]["value"]
        self.n1_C = param["n1_C"]["value"]
        self.n1_K = param["n1_K"]["value"]
        self.n1_Q = param["n1_Q"]["value"]
        self.n1_nu = param["n1_nu"]["value"]
        # L_oz
        self.s_0 = param["s_0"]["value"]
        self.s046 = param["s046"]["value"]
        self.s055 = param["s055"]["value"]
        # dA
        self.a_on = param["a_on"]["value"]
        self.a_off = param["a_off"]["value"]
        self.k_A = param["k_A"]["value"]
        # pi_NA
        self.s_c = param["s_c"]["value"]
        self.pi_min = param["pi_min"]["value"]

        self.stim_start = param["stim_start"]["value"]
        self.stim_period = param["stim_period"]["value"]
        self.stim_duration = param["stim_duration"]["value"]
        self.stim_amplitude = param["stim_amplitude"]["value"]

        self.Ca_o = param["Ca_o"]["value"]
        self.K_o = param["K_o"]["value"]
        self.Na_o = param["Na_o"]["value"]

        self.Cm = param["Cm"]["value"]
        self.F = param["F"]["value"]
        self.T = param["T"]["value"]
        self.R = param["R"]["value"]
        # Ca_SR
        self.Buf_sr = param["Buf_sr"]["value"]
        self.K_buf_sr = param["K_buf_sr"]["value"]
        # Semin
        self.k_sm_p = param["kp"]["value"]
        self.K_lrg_p = param["KP"]["value"]
        # i_up
        self.Vmax_up = param["Vmax_up"]["value"]
        self.K_up = param["K_up"]["value"]
        # i_leak
        self.V_leak = param["V_leak"]["value"]
        # Shenon
        self.max_sr = param["max_sr"]["value"]
        self.min_sr = param["min_sr"]["value"]
        self.EC = param["EC"]["value"]

        self.k_o_Ca = param["k_o_Ca"]["value"]
        self.k_i_Ca = param["k_i_Ca"]["value"]
        self.k_om = param["k_om"]["value"]
        self.k_im = param["k_im"]["value"]

        # i_rel
        self.V_rel = param["V_rel"]["value"]
        # i_xfer
        self.V_xfer = param["V_xfer"]["value"]
        # Ca_i
        self.Buf_c = param["Buf_c"]["value"]
        self.K_buf_c = param["K_buf_c"]["value"]
        self.V_sr = param["V_sr"]["value"]
        self.V_c = param["V_c"]["value"]
        # i_bCa
        self.g_bCa = param["g_bCa"]["value"]
        # i_pCa
        self.g_pCa = param["g_pCa"]["value"]
        self.K_pCa = param["K_pCa"]["value"]
        # i_NaCa
        self.gamma = param["gamma"]["value"]
        self.alpha = param["alpha"]["value"]
        self.K_sat = param["K_sat"]["value"]
        self.Km_Ca = param["Km_Ca"]["value"]
        self.Km_Nai = param["Km_Nai"]["value"]
        self.K_NaCa = param["K_NaCa"]["value"]
        # dCa_ss
        self.Buf_ss = param["Buf_ss"]["value"]
        self.K_buf_ss = param["K_buf_ss"]["value"]
        self.V_ss = param["V_ss"]["value"]
        # i_CaL
        self.g_CaL = param["g_CaL"]["value"]
        # i_Na
        self.g_Na = param["g_Na"]["value"]
        # i_K1
        self.g_K1 = param["g_K1"]["value"]
        # i_to
        self.g_to = param["g_to"]["value"]
        # i_Kr
        self.g_Kr = param["g_Kr"]["value"]
        # i_Ks
        self.g_Ks = param["g_Ks"]["value"]
        self.P_kna = param["P_kna"]["value"]
        # i_K_ATP
        self.g_K_ATP = param["gKATP"]["value"]
        self.ATPi = param["[ATP]i"]["value"]
        self.KmATP = param["KmATP"]["value"]
        # i_NaK
        self.P_NaK = param["P_NaK"]["value"]
        self.K_mk = param["K_mk"]["value"]
        self.K_mNa = param["K_mNa"]["value"]
        # i_b_Na
        self.g_bna = param["g_bna"]["value"]
        #
        self.g_pK = param["g_pK"]["value"]
        # Ca_SR
        self.V_nSR = param["V_nSR"]["value"]
        self.V_jSR = param["V_jSR"]["value"]
        self.K_buf_jsr =param["K_buf_jsr"]["value"]
        self.Buf_jsr = param["Buf_jsr"]["value"]


        # init conditions

        self.d_init = param["d"]["value"]
        self.f2_init = param["f2"]["value"]
        self.fCass_init = param["fCaSS"]["value"]
        self.f_init = param["f"]["value"]
        self.Ca_SR_init = param["CaSR"]["value"]
        self.Ca_i_init = param["Cai"]["value"]
        self.Ca_ss_init = param["CaSS"]["value"]
        self.p_iup = param["p_iup"]["value"]
        self.h_init = param["h"]["value"]
        self.j_init = param["j"]["value"]
        self.m_init = param["m"]["value"]
        self.V_init = param["E"]["value"]
        self.K_i_init = param["Ki"]["value"]
        self.Xr1_init = param["xr1"]["value"]
        self.Xr2_init = param["xr2"]["value"]
        self.Xs_init = param["xs"]["value"]
        self.Na_i_init = param["Nai"]["value"]
        self.r_init = param["r"]["value"]
        self.s_init = param["s"]["value"]

        # Ekb variables
        self.v_init = param["v"]["value"]
        self.w_init = param["w"]["value"]
        self.N_init = param["N"]["value"]
        self.A_init = param["CaTn"]["value"]
        self.l_1_init = param["l1"]["value"]
        self.l_2_init = param["l2"]["value"]
        self.l_3_init = param["l3"]["value"]

        # Shenon Ca_rel variables
        self.R_init = param["R_RyR"]["value"]
        self.O_init = param["O_RyR"]["value"]
        self.I_init = param["I_RyR"]["value"]
        self.RI_init = param["RI_RyR"]["value"]

        self.CaMKt_init = param["CaMKt"]["value"]

        self.Ca_nSR_init = param["Ca_nSR"]["value"]
        self.Ca_jSR_init = param["Ca_jSR"]["value"]


    def diff_equations(self, time, state_variables):

        # TNNP variables
        (d, f2, fCass, f, Ca_SR, Ca_i, Ca_ss, h, j, m, V, K_i, Xr1, Xr2, Xs,
         Na_i, r, s, v, w, N, A, l_1, l_2, l_3, R, O, I, RI) = state_variables

        machine_zero = 1e-15

        # dA
        """
         if  x[1] < 1.0E-8 then    x[1]:= 1.0E-8;
            if x[2]<>0 then
            begin
            //if t<=0.260 then

            if Form1.RadioGroup4.ItemIndex<>2
            then PiPi:= Pi(x[5],x[7],x[2])
        
            {для блебестатина} //PiPi:=1;
            if Form1.RadioGroup4.ItemIndex<>4 then
            {CaTnC}   f[2] :=bindingconsttnc*(GeneralTnC-x[2])*x[1]-c20*exp(-qa*x[2])*PiPi*x[2]//;Pi(x[8],x[10],x[2])
                

        """

        A_off = self.a_off * self.pi_N_A(N, A) * exp(-self.k_A * A)

        dA = self.a_on * (self.A_tot - A) * Ca_i - A_off * A

        # dd
        d_inf = 1.0 / (1.0 + exp((-8.0 - V) / 7.5))
        alpha_d = 1.4 / (1.0 + exp((-35.0 - V) / 13.0)) + 0.25
        beta_d = 1.4 / (1.0 + exp((V + 5.0) / 5.0))
        gamma_d = 1.0 / (1.0 + exp((50.0 - V) / 20.0))
        tau_d = 1.0 * alpha_d * beta_d + gamma_d
        dd = (d_inf - d) / tau_d

        ## df2
        f2_inf = 0.67 / (1.0 + exp((V + 35.0) / 7.0)) + 0.33
        tau_f2 = (
            562.0 * exp(-((V + 27.0) ** 2.0) / 240.0)
            + 31.0 / (1.0 + exp((25.0 - V) / 10.0))
            + 80.0 / (1.0 + exp((V + 30.0) / 10.0))
        )
        df2 = (f2_inf - f2) / tau_f2

        # dfCass
        fCass_inf = 0.6 / (1.0 + (Ca_ss / 0.05) ** 2.0) + 0.4
        tau_fCass = 80.0 / (1.0 + (Ca_ss / 0.05) ** 2.0) + 2.0
        dfCass = (fCass_inf - fCass) / tau_fCass

        # df
        f_inf = 1.0 / (1.0 + exp((V + 20.0) / 7.0))
        tau_f = (
            1102.5 * exp(-((V + 27.0) ** 2.0) / 225.0)
            + 200.0 / (1.0 + exp((13.0 - V) / 10.0))
            + 180.0 / (1.0 + exp((V + 30.0) / 10.0))
            + 20.0
        )
        df = (f_inf - f) / tau_f

        # [ПРАВКА] без CaMKII (Семин) и без двухкомпартментного СР:
        # исходные уравнения TP06 (в оригинале — закомментированные строки)
        i_up = self.Vmax_up / (1.0 + self.K_up**2.0 / Ca_i**2.0)
        i_leak = self.V_leak * (Ca_SR - Ca_i)

        # Shenon не влияет на следующий кусок кода
        k_CaSR = self.max_sr - (self.max_sr - self.min_sr) / (1.0 + (self.EC / Ca_SR) ** 2.0)
        k_o_SR_Ca = self.k_o_Ca / k_CaSR
        k_i_SR_Ca = self.k_i_Ca * k_CaSR

        dR = self.k_im * RI - k_i_SR_Ca * R * Ca_ss - k_o_SR_Ca * R * Ca_ss**2.0 + self.k_om * O
        dO = k_o_SR_Ca * R * Ca_ss**2.0 - self.k_om * O - k_i_SR_Ca * O * Ca_ss + self.k_im * I
        dI = k_i_SR_Ca * O * Ca_ss - self.k_im * I - self.k_om * I + k_o_SR_Ca * RI * Ca_ss**2.0
        dRI = self.k_om * I - k_o_SR_Ca * RI * Ca_ss**2.0 - self.k_im * RI + k_i_SR_Ca * R * Ca_ss

        # i_rel
        i_rel = self.V_rel * O * (Ca_SR - Ca_ss)
        # i_up
        # i_up = self.Vmax_up / (1.0 + self.K_up**2.0 / Ca_i**2.0)
        # i_leak
        # i_leak = self.V_leak * (Ca_SR - Ca_i)
        # i_xfer
        i_xfer = self.V_xfer * (Ca_ss - Ca_i)

        # dCa_SR  [ПРАВКА: однокомпартментный СР TP06]
        Ca_sr_bufsr = 1.0 / (1.0 + self.Buf_sr * self.K_buf_sr / (Ca_SR + self.K_buf_sr) ** 2.0)
        dCa_SR = Ca_sr_bufsr * (i_up - (i_rel + i_leak))

        # i_NaCa
        i_NaCa_first = exp(self.gamma * V * self.F / (self.R * self.T)) * (Na_i**3.0) * self.Ca_o
        i_NaCa_second = (
            exp((self.gamma - 1.0) * V * self.F / (self.R * self.T))
            * (self.Na_o**3.0)
            * Ca_i
            * self.alpha
        )
        i_NaCa = (
            self.K_NaCa
            * (i_NaCa_first - i_NaCa_second)
            / (
                ((self.Km_Nai**3.0) + (self.Na_o**3.0))
                * (self.Km_Ca + self.Ca_o)
                * (1.0 + self.K_sat * exp((self.gamma - 1.0) * V * self.F / (self.R * self.T)))
            )
        )

        E_Ca = 0.5 * self.R * self.T / self.F * log(self.Ca_o / Ca_i)

        # i_b_Ca
        i_b_Ca = self.g_bCa * (V - E_Ca)

        # i_p_Ca
        i_p_Ca = self.g_pCa * Ca_i / (Ca_i + self.K_pCa)

        # dCa_i
        Ca_i_bufc = 1.0 / (1.0 + self.Buf_c * self.K_buf_c / (Ca_i + self.K_buf_c) ** 2.0)
        dCa_i = Ca_i_bufc * (
            (i_leak - i_up) * self.V_sr / self.V_c
            + i_xfer
            - 1.0 * (i_b_Ca + i_p_Ca - 2.0 * i_NaCa) * self.Cm / (2.0 * 1.0 * self.V_c * self.F)
            - dA
        )
        # i_CaL
        i_CaL = (
            self.g_CaL
            * d
            * f
            * f2
            * fCass
            * 4.0
            * (V - 15.0)
            * self.F**2.0
            / (self.R * self.T)
            * (0.25 * Ca_ss * exp(2.0 * (V - 15.0) * self.F / (self.R * self.T)) - self.Ca_o)
            / (exp(2.0 * (V - 15.0) * self.F / (self.R * self.T)) - 1.0)
        )
        # dCa_ss
        Ca_ss_bufss = 1.0 / (1.0 + self.Buf_ss * self.K_buf_ss / (Ca_ss + self.K_buf_ss) ** 2.0)
        dCa_ss = Ca_ss_bufss * (
            -1.0 * i_CaL * self.Cm / (2.0 * 1.0 * self.V_ss * self.F)
            + i_rel * self.V_sr / self.V_ss
            - i_xfer * self.V_c / self.V_ss
        )



        # dh
        h_inf = 1.0 / (1.0 + exp((V + 71.55) / 7.43)) ** 2.0
        if V < -40.0:
            alpha_h = 0.057 * exp(-(V + 80.0) / 6.8)
        else:
            alpha_h = 0.0
        if V < -40.0:
            beta_h = 2.7 * exp(0.079 * V) + 310000.0 * exp(0.3485 * V)
        else:
            beta_h = 0.77 / (0.13 * (1.0 + exp((V + 10.66) / -11.1)))

        tau_h = 1.0 / (alpha_h + beta_h)
        dh = (h_inf - h) / tau_h

        # dj
        j_inf = 1.0 / (1.0 + exp((V + 71.55) / 7.43)) ** 2.0
        if V < -40.0:
            alpha_j = (
                (-25428.0 * exp(0.2444 * V) - 6.948e-6 * exp(-0.04391 * V))
                * (V + 37.78)
                / 1.0
                / (1.0 + exp(0.311 * (V + 79.23)))
            )
        else:
            alpha_j = 0.0

        if V < -40.0:
            beta_j = 0.02424 * exp(-0.01052 * V) / (1.0 + exp(-0.1378 * (V + 40.14)))
        else:
            beta_j = 0.6 * exp(0.057 * V) / (1.0 + exp(-0.1 * (V + 32.0)))

        tau_j = 1.0 / (alpha_j + beta_j)
        dj = (j_inf - j) / tau_j

        # dm
        m_inf = 1.0 / (1.0 + exp((-56.86 - V) / 9.03)) ** 2.0
        alpha_m = 1.0 / (1.0 + exp((-60.0 - V) / 5.0))
        beta_m = 0.1 / (1.0 + exp((V + 35.0) / 5.0)) + 0.1 / (1.0 + exp((V - 50.0) / 200.0))
        tau_m = 1.0 * alpha_m * beta_m
        dm = (m_inf - m) / tau_m

        # i_Na
        E_Na = self.R * self.T / self.F * log(self.Na_o / Na_i)
        i_Na = self.g_Na * (m**3.0) * h * j * (V - E_Na)

        E_K = self.R * self.T / self.F * log(self.K_o / K_i)

        # i_K1
        alpha_K1 = 0.1 / (1.0 + exp(0.06 * (V - E_K - 200.0)))
        beta_K1 = (3.0 * exp(0.0002 * (V - E_K + 100.0)) + exp(0.1 * (V - E_K - 10.0))) / (
            1.0 + exp(-0.5 * (V - E_K))
        )
        xK1_inf = alpha_K1 / (alpha_K1 + beta_K1)
        i_K1 = self.g_K1 * xK1_inf * sqrt(self.K_o / 5.4) * (V - E_K)

        # i_to
        i_to = self.g_to * r * s * (V - E_K)

        # i_Kr
        i_Kr = self.g_Kr * sqrt(self.K_o / 5.4) * Xr1 * Xr2 * (V - E_K)

        # i_Ks
        E_Ks = (
            self.R
            * self.T
            / self.F
            * log((self.K_o + self.P_kna * self.Na_o) / (K_i + self.P_kna * Na_i))
        )
        i_Ks = self.g_Ks * (Xs**2.0) * (V - E_Ks)

        # i_K_ATP

        PATP = 1.0 / (1.0 + (self.ATPi / self.KmATP)** 2.2)
        i_K_ATP = self.g_K_ATP * PATP * ((self.K_o / 5.4) ** 0.24) * (V - E_K)

        # i_NaK
        i_NaK = (
            self.P_NaK
            * self.K_o
            * Na_i
            / (
                (self.K_o + self.K_mk)
                * (Na_i + self.K_mNa)
                * (
                    1.0
                    + 0.1245 * exp(-0.1 * V * self.F / (self.R * self.T))
                    + 0.0353 * exp(-V * self.F / (self.R * self.T))
                )
            )
        )

        # i_b_Na
        i_b_Na = self.g_bna * (V - E_Na)

        # i_p_K
        i_p_K = self.g_pK * (V - E_K) / (1.0 + exp((25.0 - V) / 5.98))

        # dV   проверь что с Cm
        if (time - floor(time / self.stim_period) * self.stim_period >= self.stim_start) and (
            time - floor(time / self.stim_period) * self.stim_period
            <= self.stim_start + self.stim_duration
        ):
            i_Stim = -self.stim_amplitude
        else:
            i_Stim = 0.0
        dV = (
            -1.0
            / 1.0
            * (
                i_K1
                + i_to
                + i_Kr
                + i_Ks
                + i_CaL
                + i_NaK
                + i_Na
                + i_b_Na
                + i_NaCa
                + i_b_Ca
                + i_p_K
                + i_p_Ca
                + i_K_ATP
                + i_Stim
            )
        )

        # dK_i = 0
        dK_i = (
            -1.0
            * (i_K1 + i_to + i_Kr + i_Ks + i_p_K + i_Stim - 2.0 * i_NaK+i_K_ATP)
            / (1.0 * self.V_c * self.F)
            * self.Cm
        )

        # dXr1
        xr1_inf = 1.0 / (1.0 + exp((-26.0 - V) / 7.0))
        alpha_xr1 = 450.0 / (1.0 + exp((-45.0 - V) / 10.0))
        beta_xr1 = 6.0 / (1.0 + exp((V + 30.0) / 11.5))
        tau_xr1 = 1.0 * alpha_xr1 * beta_xr1
        dXr1 = (xr1_inf - Xr1) / tau_xr1

        # dXr2
        xr2_inf = 1.0 / (1.0 + exp((V + 88.0) / 24.0))
        alpha_xr2 = 3.0 / (1.0 + exp((-60.0 - V) / 20.0))
        beta_xr2 = 1.12 / (1.0 + exp((V - 60.0) / 20.0))
        tau_xr2 = 1.0 * alpha_xr2 * beta_xr2
        dXr2 = (xr2_inf - Xr2) / tau_xr2

        # dXs
        xs_inf = 1.0 / (1.0 + exp((-5.0 - V) / 14.0))
        alpha_xs = 1400.0 / sqrt(1.0 + exp((5.0 - V) / 6.0))
        beta_xs = 1.0 / (1.0 + exp((V - 35.0) / 15.0))
        tau_xs = 1.0 * alpha_xs * beta_xs + 80.0
        dXs = (xs_inf - Xs) / tau_xs

        # dNa_i
        dNa_i = (
            -1.0
            * (i_Na + i_b_Na + 3.0 * i_NaK + 3.0 * i_NaCa)
            / (1.0 * self.V_c * self.F)
            * self.Cm
        )

        # dr
        r_inf = 1.0 / (1.0 + exp((20.0 - V) / 6.0))
        tau_r = 9.5 * exp(-((V + 40.0) ** 2.0) / 1800.0) + 0.8
        dr = (r_inf - r) / tau_r

        # ds
        s_inf = 1.0 / (1.0 + exp((V + 20.0) / 5.0))
        tau_s = (
            85.0 * exp(-((V + 45.0) ** 2.0) / 320.0) + 5.0 / (1.0 + exp((V - 20.0) / 5.0)) + 3.0
        )
        ds = (s_inf - s) / tau_s

        # Механика

        K_chi = (
            self.k_p_v(v) * self.M(A) * self.n_1(l_1) * self.L_oz(l_1) * (1.0 - N)
            - self.k_m_v(v) * N
        )
        dN = K_chi

        F_muscle = self.beta_3 * (exp(self.alpha_3 * l_3) - 1.0)
        l = l_2 + l_3

        if self.F_afterload <= machine_zero:
            isotonic = False
        else:
            isotonic = True

        if (
            (isotonic)
            and (F_muscle >= self.F_afterload)
            and (l <= self.l0 * (1.0 + 1.0e-4))
            and (F_muscle > self.r0)
        ):
            isotonic_mode = True
        else:
            isotonic_mode = False

        """
        {dv}     f[9]:=-(lambda*f[7]*p(x[9])+alpha_vp*kp_vis*x[9]*x[9]+alpha2*beta2*exp(alpha2*x[6])*x[10])/
              (lambda*x[7]*dp(x[9])+kp_vis);
        {dw}     f[10]:=(f[9]-alpha_vs*sqr(x[10]-x[9]))-(alpha1*beta1*exp(alpha1*(x[6]-x[5]))*(x[10]-x[9])
              +alpha2*beta2*exp(alpha2*x[6])*x[10])/ks_vis;
        {l3}     f[8]:=0;
        """

        # dv
        if v <= 0.0:
            alpha_p = self.alpha_vp_l
        else:
            alpha_p = self.alpha_vp_s

        if v <= 0.0:
            k_P_vis = self.beta_vp_l * exp(self.alpha_vp_l * l_1)
        else:
            k_P_vis = self.beta_vp_s * exp(self.alpha_vp_s * l_1)

        if w <= v:
            alpha_s = self.alpha_vs_l
        else:
            alpha_s = self.alpha_vs_s

        if w <= v:
            k_S_vis = self.beta_vs_l * exp(self.alpha_vs_l * (l_2 - l_1))
        else:
            k_S_vis = self.beta_vs_s * exp(self.alpha_vs_s * (l_2 - l_1))
        #
        # izot f[9]:=-(lambda*f[7]*p(x[9])+alpha_vp*kp_vis*x[9]*x[9]+alpha2*beta2*exp(alpha2*x[6])*x[10])/
        #               (lambda*x[7]*dp(x[9])+kp_vis);
        #
        # izom f[9]:=-(lambda*f[7]*p(x[9])+alpha_vp*kp_vis*x[9]*x[9]+(alpha2*beta2*exp(alpha2*x[6])+alpha3*beta3*exp(alpha3*x[8]))*x[10])/
        #                 (lambda*x[7]*dp(x[9])+kp_vis)

        if isotonic_mode:
            phi_chi = -(
                self.llambda * K_chi * self.p_v(v)
                + alpha_p * k_P_vis * v**2.0
                + self.alpha_2 * self.beta_2 * exp(self.alpha_2 * l_2) * w
            ) / (self.llambda * N * self.p_prime_v(v) + k_P_vis)
        else:
            phi_chi = -(
                self.llambda * K_chi * self.p_v(v)
                + alpha_p * k_P_vis * v**2.0
                + (
                    self.alpha_2 * self.beta_2 * exp(self.alpha_2 * l_2)
                    + self.alpha_3 * self.beta_3 * exp(self.alpha_3 * l_3)
                )
                * w
            ) / (self.llambda * N * self.p_prime_v(v) + k_P_vis)
        dv = phi_chi

        # dw

        # izot  f[10] := (f[9] - alpha_vs * sqr(x[10] - x[9])) - (alpha1 * beta1 * exp(alpha1 * (x[6] - x[5])) * (x[10] - x[9])
        #                                                   + alpha2 * beta2 * exp(alpha2 * x[6]) * x[10]) / ks_vis;
        # izom f[10]:=f[9]-alpha_vs*sqr(x[10]-x[9])-(alpha1*beta1*exp(alpha1*(x[6]-x[5]))*(x[10]-x[9])
        #                                  +(alpha2*beta2*exp(alpha2*x[6])+alpha3*beta3*exp(alpha3*x[8]))*x[10])/ks_vis;

        if isotonic_mode:
            dw = (
                phi_chi
                - alpha_s * (w - v) ** 2.0
                - (
                    self.alpha_1 * self.beta_1 * exp(self.alpha_1 * (l_2 - l_1)) * (w - v)
                    + self.alpha_2 * self.beta_2 * exp(self.alpha_2 * l_2) * w
                )
                / k_S_vis
            )
        else:
            dw = (
                phi_chi
                - alpha_s * (w - v) ** 2.0
                - (
                    self.alpha_1 * self.beta_1 * exp(self.alpha_1 * (l_2 - l_1)) * (w - v)
                    + (
                        self.alpha_2 * self.beta_2 * exp(self.alpha_2 * l_2)
                        + self.alpha_3 * self.beta_3 * exp(self.alpha_3 * l_3)
                    )
                    * w
                )
                / k_S_vis
            )

        # dl_1
        dl_1 = v

        # dl_2
        dl_2 = w

        # dl_3
        # izot f[8]:=0;
        #
        # izom f[8]:= -w

        if isotonic_mode:
            dl_3 = 0.0
        else:
            dl_3 = -w

        F_CE = self.llambda * self.p_v(v) * N
        F_SE = self.beta_1 * (exp(self.alpha_1 * (l_2 - l_1)) - 1.0)
        F_PE = self.beta_2 * (exp(self.alpha_2 * l_2) - 1.0)
        F_VS1 = k_P_vis * v
        F_VS2 = k_S_vis * (w - v)
        F_XSE = self.beta_3 * (exp(self.alpha_3 * l_3) - 1.0)


        self.forces = np.array([F_CE, F_SE, F_PE, F_VS1, F_VS2, F_XSE])
        self.currents = np.array([i_Na, i_CaL, i_NaCa, i_NaK, i_K1, i_Kr,
                                  i_Ks, i_K_ATP, i_to, i_rel, i_up, i_leak], dtype=np.float64)

        return np.array(
            [
                dd,
                df2,
                dfCass,
                df,
                dCa_SR,
                dCa_i,
                dCa_ss,
                dh,
                dj,
                dm,
                dV,
                dK_i,
                dXr1,
                dXr2,
                dXs,
                dNa_i,
                dr,
                ds,
                dv,
                dw,
                dN,
                dA,
                dl_1,
                dl_2,
                dl_3,
                dR,
                dO,
                dI,
                dRI,
            ]
        )

    def k_p_v(self, v):
        # Result:=kappa(v)*kappa0*q(v)*m0*Gst(v/vmax);
        return self.chi(v) * self.chi_0 * self.q_v(v) * self.m_0 * self.G_star(v / self.v_max)

    def k_m_v(self, v):
        # k_:=kappa0*q(v)*(1-kappa(v)*m0*Gst(v/vmax))
        return (
            self.chi_0 * self.q_v(v) * (1.0 - self.chi(v) * self.m_0 * self.G_star(v / self.v_max))
        )

    def M(self, A):
        # Result := power((Aa / GeneralTnC), mu1) * (1 + power(k_mu, mu1)) / (power((Aa / GeneralTnC), mu1) + power(k_mu, mu1))
        return (
            ((A / self.A_tot) ** self.mu)
            * (1.0 + (self.k_mu**self.mu))
            / ((A / self.A_tot) ** self.mu + (self.k_mu**self.mu))
        )

    def n_1(self, l_1):
        # w1 := (g1 * l1 + g2) * (Al1 + ((Kl1 - Al1) / power(Cl1 + Ql1 * exp(-Bl1 * l1), 1 / nul1{nu})));
        # if w1 < 0 then Result := 0
        # else
        # if w1 < 1 then Result := w1
        # else
        # Result := 1;
        w1 = (self.g_1 * l_1 + self.g_2) * (
            self.n1_A
            + (self.n1_K - self.n1_A)
            / (self.n1_C + self.n1_Q * exp(-self.n1_B * l_1)) ** (1.0 / self.n1_nu)
        )
        if w1 < 0.0:
            return 0.0
        elif w1 < 1.0:
            return w1
        else:
            return 1.0

    def L_oz(self, l_1):
        # if l1<=s055 then Result:=(s0+l1)/(s046+s0)
        # else
        # Result:=(s0+s055)/(s046+s0);
        if l_1 <= self.s055:
            return (l_1 + self.s_0) / (self.s046 + self.s_0)
        else:
            return (self.s_0 + self.s055) / (self.s046 + self.s_0)

    def chi(self, v):
        # if v<=0 then Result:=kappa1+kappa2*v/vmax
        # else Result:=kappa1;
        if v <= 0.0:
            return self.chi_1 + self.chi_2 * v / self.v_max
        else:
            return self.chi_1

    def q_v(self, v):
        # if v<=0 then Result:=q1-q2*v/vmax
        # else
        # if v<=xst*vmax then Result:=(q4-q3)*v/(xst*vmax)+q3
        # else Result:=q4/power(1+betaQ*(v/vmax-xst),alphaQ);
        if v <= 0.0:
            return self.q_1 - self.q_2 * v / self.v_max
        elif v <= self.x_st * self.v_max:
            return (self.q_4 - self.q_3) * v / (self.x_st * self.v_max) + self.q_3
        else:
            return self.q_4 / (1.0 + self.beta_Q * (v / self.v_max - self.x_st)) ** self.alpha_Q

    def G_star(self, v):
        # begin
        # den:=(0.4*a+1)*x/a+1;
        # if  x<=0 then Result:=1+0.6*x
        # else
        # if  x<=x1 then Result:=Pst(x)/den
        # else Result:=Pst(x)*exp(-alphaG*power((x-x1),alphaP))/den;
        # end;
        den = (0.4 * self.a + 1.0) * v / self.a + 1.0
        if v <= 0:
            return 1 + 0.6 * v
        elif v <= self.v_1:
            return self.P_star(v) / den
        else:
            return self.P_star(v) * exp(-self.alpha_G * ((v - self.v_1) ** self.alpha_P)) / den

    def P_star(self, v):
        # begin
        # gamma:=a*d*sqr(x1)/( 3*a*d-(a+1)*x1 );
        # Result:=1+d-sqr(d)*a/( (a+1)*x+d*a+a*d*sqr(x)/gamma );
        # end;
        gamma = (
            self.a
            * self.d_h
            * (self.v_1**2.0)
            / (3.0 * self.a * self.d_h - (self.a + 1.0) * self.v_1)
        )
        return (
            1.0
            + self.d_h
            - (self.d_h**2.0)
            * self.a
            / ((self.a + 1) * v + self.d_h * self.a + self.a * self.d_h * (v**2.0) / gamma)
        )

    def pi_N_A(self, N, A):
        """
        if form1.CheckBox1.Checked then
        n1:=GeneralTnC*sc*Nn/Aa
        else
        n1:=GeneralTnC*sc*Nn/(s(l1)*Aa);
        if n1<0 then Result:=1
        else
        if n1<=1 then Result:=power(pimin,n1)
        else
        Result:=pimin;
        // result:=power(pimin,N);
        end;
        """
        N_A = self.A_tot * self.s_c * N / A

        if N_A <= 0.0:
            return 1.0
        elif N_A <= 1.0:
            return self.pi_min**N_A
        else:
            return self.pi_min

    def p_v(self, v):
        """
        if v<=-vmax then Result:=0
        else
        if v<=0 then Result:=a*(1+v/vmax)/((a-v/vmax)*(1+0.6*v/vmax))
        else
        if v<=x1*vmax then Result:=(0.4*a+1)*v/(a*vmax)+1
        else
        Result:=( (0.4*a+1)*v/(a*vmax)+1)*exp(alphaG*power((v/vmax-x1),alphaP));
        """
        if v <= -self.v_max:
            return 0.0
        elif v <= 0.0:
            return (
                self.a
                * (1.0 + v / self.v_max)
                / ((self.a - v / self.v_max) * (1.0 + 0.6 * v / self.v_max))
            )
        elif v <= self.v_1 * self.v_max:
            return (0.4 * self.a + 1.0) * v / (self.a * self.v_max) + 1.0
        else:
            return ((0.4 * self.a + 1.0) * v / (self.a * self.v_max) + 1.0) * exp(
                self.alpha_G * (v / self.v_max - self.v_1) ** self.alpha_P
            )

    def p_prime_v(self, v):
        """
        if v<=-vmax then Result:=a*(0.4+0.4*a)/
                         ( vmax*sqr((a+1)*0.4) )
        else
        if v<=0 then Result:=a*(1+0.4*a+1.2*v/vmax+0.6*sqr(v/vmax))/
                         ( vmax*sqr((a-v/vmax)*(1+0.6*v/vmax)) )
        else
        if  v<=x1*vmax then Result:=(0.4*a+1)/(a*vmax)
        else
        Result:=exp(alphaG*power(v/vmax-x1,alphaP))*
                         ( (0.4*a+1)/a+alphaG*alphaP*(1+(0.4*a+1)*v/(a*vmax))*
                         power(v/vmax-x1,alphaP-1) )/vmax;
        """
        if v <= -self.v_max:
            return self.a * (0.4 + 0.4 * self.a) / (self.v_max * ((self.a + 1.0) * 0.4) ** 2.0)
        elif (-self.v_max < v) and (v <= 0.0):  # !!!!!!!!!!!!1
            return (
                self.a
                * (1.0 + 0.4 * self.a + 1.2 * v / self.v_max + 0.6 * (v / self.v_max) ** 2.0)
                / (self.v_max * ((self.a - v / self.v_max) * (1.0 + 0.6 * v / self.v_max)) ** 2.0)
            )
        elif (0.0 < v) and (v <= self.v_1 * self.v_max):  # !!!!!!!!!!2
            return (0.4 * self.a + 1.0) / (self.a * self.v_max)
        else:
            return (
                exp(self.alpha_G * (v / self.v_max - self.v_1) ** self.alpha_P)
                * (
                    (0.4 * self.a + 1.0) / self.a
                    + self.alpha_G
                    * self.alpha_P
                    * (1.0 + (0.4 * self.a + 1.0) * v / (self.a * self.v_max))
                    * (v / self.v_max - self.v_1) ** (self.alpha_P - 1.0)
                )
                / self.v_max
            )

    def N0(self, l_0):
        return (self.r0 - self.beta_2 * (exp(self.alpha_2 * l_0) - 1)) / self.llambda

    def fi(self,l_1):
        return self.k_p_v(0) * self.M(6.31929074e-04) * self.n_1(l_1) * self.L_oz(l_1) * (1.0 - self.N0(l_1)) - self.k_m_v(0) * self.N0(l_1)

    def L(self,l_0):
        return l_0 + (log(self.beta_1) - log(self.r0 + self.beta_1 - self.beta_2 * (exp(self.alpha_2 * l_0) - 1))) / self.alpha_1


    def delenie(self):
        l200 = log((self.r0 + self.beta_2) / self.beta_2) / self.alpha_2
        l100 = l200
        a = 0.9 * l100
        b = l100
        if self.fi(a) == 0.0:
            return a
        if self.fi(b) == 0.0:
            return b

        x = a + (b - a) / 2.0
        while abs(self.fi(x)) >= 0.0000001:
            x = a + (b - a) / 2.0
            if self.fi(x) < 0:
                a = x
            if self.fi(x) >= 0:
                b = x
        return x

    def calculate_init_conditions(self):
        l_2 = self.delenie()
        l_1 = self.L(l_2)
        N = self.N0(l_2)
        l_3 = log((self.r0 + self.beta_3) / self.beta_3) / self.alpha_3

        self.v_init = 0.0
        self.w_init = 0.0
        self.N_init = N

        self.l_1_init = l_1
        self.l_2_init = l_2
        self.l_3_init = l_3

        self.l0 = self.l_2_init + self.l_3_init
        self.F_afterload = 0.0

        return np.array([self.d_init,
        self.f2_init,
        self.fCass_init,
        self.f_init,
        self.Ca_SR_init,
        self.Ca_i_init,
        self.Ca_ss_init,
        self.h_init,
        self.j_init,
        self.m_init,
        self.V_init,
        self.K_i_init,
        self.Xr1_init,
        self.Xr2_init,
        self.Xs_init,
        self.Na_i_init,
        self.r_init,
        self.s_init,
        self.v_init,
        self.w_init,
        self.N_init,
        self.A_init,
        self.l_1_init,
        self.l_2_init,
        self.l_3_init,
        self.R_init,
        self.O_init,
        self.I_init,
        self.RI_init,
        ])
