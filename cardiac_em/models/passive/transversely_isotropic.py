"""
Трансверсально-изотропный экспоненциальный материал.
=====================================================

    Ψ = μ/b·exp(b·(I₁−2)) + μ_f/(2b_f)·exp(b_f·(I₄−1)²) + κ/2·(J−1)²

Три слагаемых: изотропная матрица, усиление вдоль волокна и штраф за
изменение объёма (почти несжимаемый материал). Второй тензор
Пиолы–Кирхгофа получается дифференцированием, S = 2·∂Ψ/∂C:

    S = 2[ μ·e^{b(I₁−2)}·I
         + μ_f·(I₄−1)·e^{b_f(I₄−1)²}·f₀⊗f₀
         + κ·(J−1)·J/2·C⁻¹ ]

Производные выписаны аналитически, а не через `ufl.diff`: выражение
короче, читается как формула из статьи, и якобиан всё равно строится
автоматическим дифференцированием слабой формы.

Параметры (μ, b, μ_f, b_f, κ) — поля на DG0, поэтому могут различаться
по регионам: например, фиброзный рубец задаётся повышенными μ и μ_f.
"""

from __future__ import annotations

from ufl import Identity, exp, inv, outer

from .base import PassiveMaterial, kinematics

__all__ = ["TransverselyIsotropicExponential"]


class TransverselyIsotropicExponential(PassiveMaterial):
    """Экспоненциальный трансверсально-изотропный материал миокарда."""

    name = "transversely_isotropic_exponential"
    param_names = ("MU", "B_ISO", "MU_F", "B_F", "KAPPA")

    def second_piola(self, u, tissue):
        f0 = tissue.fiber_vector()
        F, C, J, I1, I4 = kinematics(u, tissue)
        C_inv = inv(C)

        dPsi_dI1 = tissue.mu * exp(tissue.b_iso * (I1 - 2))
        dPsi_dI4 = tissue.mu_f * (I4 - 1) * exp(tissue.b_f * (I4 - 1) ** 2)
        dPsi_dJ = tissue.kappa * (J - 1)

        return 2 * (
            dPsi_dI1 * Identity(u.ufl_shape[0])
            + dPsi_dI4 * outer(f0, f0)
            + dPsi_dJ * J / 2 * C_inv
        )
