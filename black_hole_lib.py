"""Physical formulas for Schwarzschild/Kerr black holes in SI units.

The module contains two layers:
- Exact/standard formulas requested by the user (SI units by default).
- Backward-compatible helpers used by the current 3D scene code.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# 0) Constants (SI)
# ---------------------------------------------------------------------------
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 2.99792458e8  # m s^-1
hbar = 1.054571817e-34  # J s
k_B = 1.380649e-23  # J K^-1
sigma = 5.670374419e-8  # W m^-2 K^-4
pi = math.pi
m_p = 1.67262192369e-27  # kg
sigma_T = 6.6524587321e-29  # m^2
M_sun = 1.98847e30  # kg

# Compatibility aliases
G_SI = G
C_SI = c
SIM_G = 1.0
SIM_C = 1.0
EPS = 1e-6  # simulation-friendly epsilon
STABILITY_EPS = 1e-12  # strict numerical-stability epsilon


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _require_positive(name: str, value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


def _to_array(value: float | np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _match_type(reference: float | np.ndarray, value: np.ndarray) -> float | np.ndarray:
    if np.isscalar(reference):
        return float(np.asarray(value))
    return value


def _validate_radius_outside_horizon(
    M: float,
    r: float | np.ndarray,
    eps: float = STABILITY_EPS,
    g_const: float = G,
    c_const: float = c,
) -> tuple[float, np.ndarray]:
    rs = schwarzschild_radius(M, g_const=g_const, c_const=c_const)
    r_arr = _to_array(r)
    min_valid = rs * (1.0 + eps)
    if np.any(r_arr <= min_valid):
        raise ValueError("r must be > r_s * (1 + eps)")
    return rs, r_arr


def _validate_astar(astar: float) -> float:
    astar = float(astar)
    if abs(astar) > 1.0 + 1e-12:
        raise ValueError("|a_*| must be <= 1")
    return float(np.clip(astar, -1.0, 1.0))


# ---------------------------------------------------------------------------
# 1) Schwarzschild black hole
# ---------------------------------------------------------------------------
def schwarzschild_radius(M: float, g_const: float = G, c_const: float = c) -> float:
    """r_s = 2 G M / c^2."""
    M = _require_positive("M", M)
    return 2.0 * g_const * M / (c_const ** 2)


def horizon_area(M: float, g_const: float = G, c_const: float = c) -> float:
    """A = 4 pi r_s^2."""
    rs = schwarzschild_radius(M, g_const=g_const, c_const=c_const)
    return 4.0 * pi * rs ** 2


def time_dilation_factor(
    M: float,
    r: float | np.ndarray,
    eps: float = STABILITY_EPS,
    g_const: float = G,
    c_const: float = c,
) -> float | np.ndarray:
    """gamma_t(r) = sqrt(1 - r_s/r), valid only for r > r_s."""
    rs, r_arr = _validate_radius_outside_horizon(M, r, eps=eps, g_const=g_const, c_const=c_const)
    gamma = np.sqrt(np.clip(1.0 - rs / r_arr, 0.0, 1.0))
    return _match_type(r, gamma)


def escape_velocity(M: float, r: float | np.ndarray, g_const: float = G) -> float | np.ndarray:
    """v_esc(r) = sqrt(2 G M / r)."""
    M = _require_positive("M", M)
    r_arr = np.maximum(_to_array(r), STABILITY_EPS)
    v = np.sqrt(np.maximum(2.0 * g_const * M / r_arr, 0.0))
    return _match_type(r, v)


# ---------------------------------------------------------------------------
# 2) Orbital mechanics outside the hole
# ---------------------------------------------------------------------------
def orbital_velocity_newton(M: float, r: float | np.ndarray, g_const: float = G) -> float | np.ndarray:
    """v_orb(r) = sqrt(G M / r)."""
    M = _require_positive("M", M)
    r_arr = np.maximum(_to_array(r), STABILITY_EPS)
    v = np.sqrt(np.maximum(g_const * M / r_arr, 0.0))
    return _match_type(r, v)


def orbital_omega(M: float, r: float | np.ndarray, g_const: float = G) -> float | np.ndarray:
    """omega(r) = sqrt(G M / r^3)."""
    M = _require_positive("M", M)
    r_arr = np.maximum(_to_array(r), STABILITY_EPS)
    omega = np.sqrt(np.maximum(g_const * M / np.maximum(r_arr ** 3, STABILITY_EPS), 0.0))
    return _match_type(r, omega)


def isco_schwarzschild(M: float, g_const: float = G, c_const: float = c) -> float:
    """r_ISCO = 6 G M / c^2 = 3 r_s."""
    M = _require_positive("M", M)
    return 6.0 * g_const * M / (c_const ** 2)


# Backward-compatible name used by main.py

def isco_radius(mass: float, g_const: float = G, c_const: float = c) -> float:
    return isco_schwarzschild(mass, g_const=g_const, c_const=c_const)


# Backward-compatible helper used by the simulation.
# Signature preserved; formula now follows omega = sqrt(GM/r^3).
def orbital_angular_velocity(
    mass: float,
    radius: float | np.ndarray,
    r_s: float | None = None,
    g_const: float = G,
) -> float | np.ndarray:
    radius_arr = _to_array(radius)
    if r_s is None:
        safe_radius = np.maximum(radius_arr, STABILITY_EPS)
    else:
        safe_radius = np.maximum(radius_arr, float(r_s) * (1.0 + 1e-9))

    omega = np.sqrt(np.maximum(g_const * float(mass) / np.maximum(safe_radius ** 3, STABILITY_EPS), 0.0))
    return _match_type(radius, omega)


def paczynski_wiita_acceleration(
    mass: float,
    radius: float | np.ndarray,
    r_s: float | None = None,
    g_const: float = G,
) -> float | np.ndarray:
    """a_r = - G M / (r - r_s)^2 (pseudo-Newtonian potential)."""
    if r_s is None:
        r_s = schwarzschild_radius(mass, g_const=g_const, c_const=1.0)

    radius_arr = _to_array(radius)
    effective_radius = np.maximum(radius_arr - float(r_s), EPS)
    acc = -(g_const * float(mass)) / (effective_radius ** 2)
    return _match_type(radius, acc)


def radial_inflow_velocity(
    mass: float,
    radius: float | np.ndarray,
    r_s: float | None = None,
    alpha: float = 0.03,
    g_const: float = G,
) -> float | np.ndarray:
    """Approximate thin-disk inflow speed used by the visual simulation."""
    radius_arr = _to_array(radius)
    if r_s is None:
        r_s = schwarzschild_radius(mass, g_const=g_const, c_const=1.0)

    safe_radius = np.maximum(radius_arr, float(r_s) * (1.0 + 1e-9))
    omega = np.asarray(orbital_angular_velocity(mass, safe_radius, r_s=r_s, g_const=g_const), dtype=float)
    v_orbital = omega * safe_radius

    proximity = np.clip(float(r_s) / safe_radius, 0.0, 0.99)
    radial_speed = -float(alpha) * (proximity ** 1.5) * v_orbital
    max_inflow = 0.25 * v_orbital
    clipped = np.clip(radial_speed, -max_inflow, -STABILITY_EPS)
    return _match_type(radius, clipped)


def normalized_disk_temperature(radius: np.ndarray, inner_radius: float) -> np.ndarray:
    """T(r) ~ r^(-3/4) * (1 - sqrt(r_in/r))^(1/4), normalized to [0, 1]."""
    inner_radius = _require_positive("inner_radius", inner_radius)
    radius_arr = _to_array(radius)
    safe_radius = np.maximum(radius_arr, inner_radius * (1.0 + 1e-9))

    profile = (safe_radius / inner_radius) ** (-0.75)
    profile *= np.maximum(1.0 - np.sqrt(inner_radius / safe_radius), 0.0) ** 0.25

    pmin = float(np.min(profile))
    pmax = float(np.max(profile))
    return (profile - pmin) / (pmax - pmin + STABILITY_EPS)


# ---------------------------------------------------------------------------
# 3) Hawking radiation
# ---------------------------------------------------------------------------
def hawking_temperature(M: float) -> float:
    """T_H = hbar c^3 / (8 pi G M k_B)."""
    M = _require_positive("M", M)
    return hbar * c ** 3 / (8.0 * pi * G * M * k_B)


def bh_entropy(M: float) -> float:
    """S = k_B c^3 A / (4 G hbar)."""
    A = horizon_area(M)
    return k_B * c ** 3 * A / (4.0 * G * hbar)


def hawking_power_blackbody(M: float) -> float:
    """P ~= A sigma T_H^4 (blackbody approximation)."""
    A = horizon_area(M)
    T = hawking_temperature(M)
    return A * sigma * T ** 4


def mass_loss_rate_blackbody(M: float) -> float:
    """dM/dt = -P/c^2 (blackbody approximation)."""
    return -hawking_power_blackbody(M) / (c ** 2)


def hawking_lifetime(M: float) -> float:
    """tau ~ 5120 pi G^2 M^3 / (hbar c^4)."""
    M = _require_positive("M", M)
    return 5120.0 * pi * G ** 2 * M ** 3 / (hbar * c ** 4)


# ---------------------------------------------------------------------------
# 4) Tidal forces
# ---------------------------------------------------------------------------
def tidal_acceleration(M: float, r: float | np.ndarray, L: float) -> float | np.ndarray:
    """Delta a ~= 2 G M L / r^3."""
    M = _require_positive("M", M)
    L = _require_positive("L", L)
    r_arr = np.maximum(_to_array(r), STABILITY_EPS)
    da = 2.0 * G * M * L / np.maximum(r_arr ** 3, STABILITY_EPS)
    return _match_type(r, da)


# ---------------------------------------------------------------------------
# 5) Gravitational redshift
# ---------------------------------------------------------------------------
def gravitational_redshift(
    M: float,
    r: float | np.ndarray,
    eps: float = STABILITY_EPS,
    g_const: float = G,
    c_const: float = c,
) -> float | np.ndarray:
    """z = (1 - r_s/r)^(-1/2) - 1, valid only for r > r_s."""
    rs, r_arr = _validate_radius_outside_horizon(M, r, eps=eps, g_const=g_const, c_const=c_const)
    z = np.power(np.clip(1.0 - rs / r_arr, STABILITY_EPS, None), -0.5) - 1.0
    return _match_type(r, z)


# ---------------------------------------------------------------------------
# 6) Kerr black hole
# ---------------------------------------------------------------------------
def kerr_a(J: float, M: float) -> float:
    """a = J / (M c)."""
    M = _require_positive("M", M)
    return float(J) / (M * c)


def kerr_astar(J: float, M: float) -> float:
    """a_* = c J / (G M^2)."""
    M = _require_positive("M", M)
    return c * float(J) / (G * M ** 2)


def kerr_horizon_radius(M: float, astar: float) -> float:
    """r_+ = (G M / c^2) * (1 + sqrt(1 - a_*^2))."""
    M = _require_positive("M", M)
    astar = _validate_astar(astar)
    root_term = math.sqrt(max(0.0, 1.0 - astar ** 2))
    return (G * M / c ** 2) * (1.0 + root_term)


def ergo_radius(M: float, astar: float, theta: float) -> float:
    """r_ergo(theta) = (G M / c^2) * (1 + sqrt(1 - a_*^2 cos^2(theta)))."""
    M = _require_positive("M", M)
    astar = _validate_astar(astar)
    cos2 = math.cos(float(theta)) ** 2
    root_term = math.sqrt(max(0.0, 1.0 - astar ** 2 * cos2))
    return (G * M / c ** 2) * (1.0 + root_term)


def horizon_angular_velocity(M: float, astar: float) -> float:
    """Omega_H = a_* c^3 / (2 G M (1 + sqrt(1 - a_*^2)))."""
    M = _require_positive("M", M)
    astar = _validate_astar(astar)
    denom = 2.0 * G * M * (1.0 + math.sqrt(max(0.0, 1.0 - astar ** 2)))
    return astar * c ** 3 / denom


# ---------------------------------------------------------------------------
# 7) Kerr ISCO
# ---------------------------------------------------------------------------
def kerr_isco(M: float, astar: float, prograde: bool = True) -> float:
    """ISCO radius for Kerr black hole (Bardeen et al. form)."""
    M = _require_positive("M", M)
    astar = _validate_astar(astar)

    one_minus_a2 = max(0.0, 1.0 - astar ** 2)
    z1 = 1.0 + one_minus_a2 ** (1.0 / 3.0) * ((1.0 + astar) ** (1.0 / 3.0) + (1.0 - astar) ** (1.0 / 3.0))
    z2 = math.sqrt(max(0.0, 3.0 * astar ** 2 + z1 ** 2))
    root = math.sqrt(max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2)))

    factor = 3.0 + z2 - root if prograde else 3.0 + z2 + root
    return (G * M / c ** 2) * factor


# ---------------------------------------------------------------------------
# 8) Accretion efficiency and luminosity
# ---------------------------------------------------------------------------
def accretion_luminosity(mdot: float, eta: float = 0.057) -> float:
    """L = eta * mdot * c^2."""
    if eta < 0.0:
        raise ValueError("eta must be >= 0")
    return float(eta) * float(mdot) * c ** 2


# ---------------------------------------------------------------------------
# 9) Eddington limit
# ---------------------------------------------------------------------------
def eddington_luminosity(M: float) -> float:
    """L_E = 4 pi G M m_p c / sigma_T."""
    M = _require_positive("M", M)
    return 4.0 * pi * G * M * m_p * c / sigma_T


def eddington_accretion_rate(M: float, eta: float = 0.1) -> float:
    """mdot_E = L_E / (eta c^2)."""
    if eta <= 0.0:
        raise ValueError("eta must be > 0")
    return eddington_luminosity(M) / (float(eta) * c ** 2)


# ---------------------------------------------------------------------------
# 10) Unit helpers
# ---------------------------------------------------------------------------
def solar_masses_to_kg(msun: float) -> float:
    return float(msun) * M_sun


def kg_to_solar_masses(kg: float) -> float:
    return float(kg) / M_sun


# ---------------------------------------------------------------------------
# 11) Minimal implementation skeleton (extended)
# ---------------------------------------------------------------------------
class BlackHole:
    def __init__(self, mass_kg: float, spin_astar: float = 0.0):
        self.M = _require_positive("mass_kg", mass_kg)
        self.astar = _validate_astar(spin_astar)

    def rs(self) -> float:
        return schwarzschild_radius(self.M)

    def r_plus(self) -> float:
        return kerr_horizon_radius(self.M, self.astar)

    def area(self) -> float:
        if abs(self.astar) < 1e-15:
            return horizon_area(self.M)
        r_plus = self.r_plus()
        a = self.astar * G * self.M / c ** 2
        return 4.0 * pi * (r_plus ** 2 + a ** 2)

    def temperature(self) -> float:
        # Schwarzschild Hawking temperature approximation
        return hawking_temperature(self.M)

    def entropy(self) -> float:
        return k_B * c ** 3 * self.area() / (4.0 * G * hbar)

    def isco(self, prograde: bool = True) -> float:
        if abs(self.astar) < 1e-15:
            return isco_schwarzschild(self.M)
        return kerr_isco(self.M, self.astar, prograde=prograde)

    def eddington_luminosity(self) -> float:
        return eddington_luminosity(self.M)


# ---------------------------------------------------------------------------
# 12) Numeric stability helpers
# ---------------------------------------------------------------------------
def adaptive_step_size_near_horizon(M: float, r: float, base_step: float, min_factor: float = 0.05) -> float:
    """Reduce integration step close to the horizon."""
    M = _require_positive("M", M)
    base_step = _require_positive("base_step", base_step)
    min_factor = float(np.clip(min_factor, 1e-4, 1.0))

    rs = schwarzschild_radius(M)
    if r <= rs:
        return base_step * min_factor

    proximity = (r - rs) / max(rs, STABILITY_EPS)
    scale = float(np.clip(proximity / 10.0, min_factor, 1.0))
    return base_step * scale


# ---------------------------------------------------------------------------
# 13) Ray tracing / geodesics helpers (Schwarzschild equatorial, first-order)
# ---------------------------------------------------------------------------
def schwarzschild_equatorial_geodesic_rhs(
    _tau: float,
    y: np.ndarray,
    M: float,
    E: float,
    Lz: float,
) -> np.ndarray:
    """First-order ODE system for y = [r, phi, t, p_r] in proper time."""
    r, _phi, _t, p_r = map(float, y)
    rs = schwarzschild_radius(M)
    r_safe = max(r, rs * (1.0 + 1e-9))

    f = 1.0 - rs / r_safe
    dt_dtau = float(E) / max(f, STABILITY_EPS)
    dphi_dtau = float(Lz) / (r_safe ** 2)

    # From p_r^2 + V_eff(r) = E^2, with V_eff = f * (c^2 + L^2/r^2):
    # dp_r/dtau = -0.5 * dV_eff/dr
    dV_dr = (rs / (r_safe ** 2)) * (c ** 2 + (Lz ** 2) / (r_safe ** 2))
    dV_dr += f * (-2.0 * Lz ** 2 / (r_safe ** 3))
    dp_r_dtau = -0.5 * dV_dr

    dr_dtau = p_r
    return np.array([dr_dtau, dphi_dtau, dt_dtau, dp_r_dtau], dtype=float)


def rk4_step(
    rhs: Callable[[float, np.ndarray, float, float, float], np.ndarray],
    tau: float,
    y: np.ndarray,
    h: float,
    M: float,
    E: float,
    Lz: float,
) -> np.ndarray:
    """Classic RK4 step for the first-order geodesic system."""
    k1 = rhs(tau, y, M, E, Lz)
    k2 = rhs(tau + 0.5 * h, y + 0.5 * h * k1, M, E, Lz)
    k3 = rhs(tau + 0.5 * h, y + 0.5 * h * k2, M, E, Lz)
    k4 = rhs(tau + h, y + h * k3, M, E, Lz)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_schwarzschild_equatorial_geodesic(
    M: float,
    E: float,
    Lz: float,
    y0: np.ndarray,
    tau_end: float,
    base_step: float = 1e-3,
    max_steps: int = 200_000,
) -> np.ndarray:
    """Integrate y=[r,phi,t,p_r] using RK4 with adaptive step near horizon."""
    y = np.asarray(y0, dtype=float)
    if y.shape != (4,):
        raise ValueError("y0 must have shape (4,) for [r, phi, t, p_r]")

    tau = 0.0
    rs = schwarzschild_radius(M)
    history = [y.copy()]

    for _ in range(int(max_steps)):
        if tau >= tau_end:
            break

        h = adaptive_step_size_near_horizon(M, y[0], base_step)
        if tau + h > tau_end:
            h = tau_end - tau

        y = rk4_step(schwarzschild_equatorial_geodesic_rhs, tau, y, h, M, E, Lz)
        tau += h
        history.append(y.copy())

        if y[0] <= rs * (1.0 + 1e-9):
            break

    return np.asarray(history, dtype=float)


# ---------------------------------------------------------------------------
# 14) Minimum useful output set
# ---------------------------------------------------------------------------
def minimum_useful_output(
    M: float,
    astar: float = 0.0,
    r: float | None = None,
    mdot: float | None = None,
    L: float = 1.0,
    eta: float = 0.057,
    prograde: bool = True,
) -> dict[str, float | None]:
    """Return the compact output set requested in the specification."""
    bh = BlackHole(mass_kg=M, spin_astar=astar)

    out: dict[str, float | None] = {
        "r_s": bh.rs(),
        "r_+": bh.r_plus(),
        "A": bh.area(),
        "T_H": bh.temperature(),
        "S": bh.entropy(),
        "r_ISCO": bh.isco(prograde=prograde),
        "L_Edd": bh.eddington_luminosity(),
        "z(r)": None,
        "tidal_acceleration(r, L)": None,
        "accretion_luminosity(mdot, eta)": None,
    }

    if r is not None:
        out["z(r)"] = gravitational_redshift(M, r)
        out["tidal_acceleration(r, L)"] = tidal_acceleration(M, r, L)

    if mdot is not None:
        out["accretion_luminosity(mdot, eta)"] = accretion_luminosity(mdot, eta=eta)

    return out


# Backward-compatible typo alias

def standard_gravity(mass1: float, mass2: float, radius: float, g_const: float = G) -> float:
    radius = max(float(radius), STABILITY_EPS)
    return g_const * float(mass1) * float(mass2) / (radius ** 2)


def standart_gravity(M: float, m: float, r: float) -> float:
    return standard_gravity(M, m, r)


__all__ = [
    "G",
    "c",
    "hbar",
    "k_B",
    "sigma",
    "pi",
    "m_p",
    "sigma_T",
    "M_sun",
    "G_SI",
    "C_SI",
    "SIM_G",
    "SIM_C",
    "EPS",
    "STABILITY_EPS",
    "standard_gravity",
    "standart_gravity",
    "schwarzschild_radius",
    "horizon_area",
    "time_dilation_factor",
    "escape_velocity",
    "orbital_velocity_newton",
    "orbital_omega",
    "orbital_angular_velocity",
    "isco_schwarzschild",
    "isco_radius",
    "paczynski_wiita_acceleration",
    "radial_inflow_velocity",
    "normalized_disk_temperature",
    "hawking_temperature",
    "bh_entropy",
    "hawking_power_blackbody",
    "mass_loss_rate_blackbody",
    "hawking_lifetime",
    "tidal_acceleration",
    "gravitational_redshift",
    "kerr_a",
    "kerr_astar",
    "kerr_horizon_radius",
    "ergo_radius",
    "horizon_angular_velocity",
    "kerr_isco",
    "accretion_luminosity",
    "eddington_luminosity",
    "eddington_accretion_rate",
    "solar_masses_to_kg",
    "kg_to_solar_masses",
    "BlackHole",
    "adaptive_step_size_near_horizon",
    "schwarzschild_equatorial_geodesic_rhs",
    "rk4_step",
    "integrate_schwarzschild_equatorial_geodesic",
    "minimum_useful_output",
]
