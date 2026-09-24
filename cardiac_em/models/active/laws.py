"""Законы φ(v) — см. докстринг пакета."""

from __future__ import annotations

from ufl import conditional, exp, le, max_value

__all__ = ["ActiveTensionLaw", "IsometricLaw", "TNNPMForceVelocity",
           "ACTIVE_LAWS", "make_active_law"]


class ActiveTensionLaw:
    """Базовый закон: без зависимости от скорости (φ ≡ 1)."""

    name = "isometric"
    uses_velocity = False

    def factor(self, stretch, stretch_old, dt):
        """UFL-выражение φ; stretch — λ_f(u), stretch_old — λ_f прошлого шага."""
        return 1.0

    def describe(self) -> str:
        return "T_act = T_iso (без зависимости от скорости)"


class IsometricLaw(ActiveTensionLaw):
    pass


class TNNPMForceVelocity(ActiveTensionLaw):
    """
    φ(v) = p(v) модели TNNPM (функция p оригинала), v = SL_slack · dλ_f/dt,
    dλ_f/dt = (λ_f(u) − λ_f,пред) / Δt.

        x = v / v_max
        x ≤ −1        : 0
        −1 < x ≤ 0    : a(1 + x) / ((a − x)(1 + 0.6x))             укорочение
        0 < x ≤ x₁    : (0.4a + 1)x/a + 1                         удлинение
        x > x₁        : ((0.4a + 1)x/a + 1)·exp(α_G (x − x₁)^α_P)

    Знаменатели защищены от нуля так, что в используемых ветвях значения
    не меняются (UFL вычисляет все ветви).
    """

    name = "tnnpm_force_velocity"
    uses_velocity = True

    def __init__(self, constants: dict):
        self.a = float(constants["a"])
        self.v_max = float(constants["v_max"])
        self.x1 = float(constants["v_1"])
        self.alpha_G = float(constants["alpha_G"])
        self.alpha_P = float(constants["alpha_P"])
        self.sl0 = float(constants["SL_slack"])

    def velocity(self, stretch, stretch_old, dt):
        """Скорость сократительного элемента, мкм/мс."""
        return self.sl0 * (stretch - stretch_old) / dt

    def factor(self, stretch, stretch_old, dt):
        a, x1 = self.a, self.x1
        x = self.velocity(stretch, stretch_old, dt) / self.v_max
        shorten = a * (1.0 + x) / (max_value(a - x, a) * max_value(1.0 + 0.6 * x, 0.4))
        lin = (0.4 * a + 1.0) * x / a + 1.0
        fast = lin * exp(self.alpha_G * max_value(x - x1, 0.0) ** self.alpha_P)
        return conditional(le(x, -1.0), 0.0,
                           conditional(le(x, 0.0), shorten,
                                       conditional(le(x, x1), lin, fast)))

    def describe(self) -> str:
        return (f"T_act = T_iso · p(v), v = {self.sl0:g}·dλ_f/dt "
                f"(сила–скорость TNNPM, неявно по скорости)")


ACTIVE_LAWS = {IsometricLaw.name: IsometricLaw, TNNPMForceVelocity.name: TNNPMForceVelocity}


def make_active_law(cell_model) -> ActiveTensionLaw:
    """Закон, который объявила модель клетки (`tissue_active_law`)."""
    name = getattr(cell_model, "tissue_active_law", None) or "isometric"
    if name == "isometric":
        return IsometricLaw()
    try:
        cls = ACTIVE_LAWS[name]
    except KeyError:
        raise KeyError(f"модель клетки {cell_model.name!r} требует закон {name!r}, "
                       f"он не зарегистрирован; есть {sorted(ACTIVE_LAWS)}") from None
    return cls(cell_model.c)
