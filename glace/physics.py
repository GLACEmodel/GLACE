"""
glace/physics.py — Physical processes

Functions
---------
velocity_linear(z, Z, Umax)
    Linear velocity profile (simple test case).

velocity_lliboutry(z, H_ice, b_acc, p)
    Lliboutry (1979) velocity profile for a frozen-bed ice sheet.

gas_loss_profile(z, lambda_max, z_char, z_gl_limit)
    Spatially varying gas-loss rate (exponential decay near the bed).
"""

import numpy as np


def velocity_linear(z, Z, Umax):
    """Linear vertical velocity profile.

    Parameters
    ----------
    z    : (nz,) array — height above bed [m]
    Z    : float — domain top [m]
    Umax : float — velocity at z=Z [m/yr], negative = downward

    Returns
    -------
    u : (nz,) array — velocity [m/yr]
    """
    return Umax * (z / Z)


def velocity_lliboutry(z, H_ice, b_acc, p):
    """Lliboutry (1979) vertical velocity for a frozen-bed ice sheet.

    w(ζ) = −ḃ · [(p+2)ζ − (p+2)ζ^(p+1) + ζ^(p+2)],  ζ = z / H_ice

    Parameters
    ----------
    z      : (nz,) array — height above bed [m]
    H_ice  : float — total ice thickness [m]
    b_acc  : float — accumulation rate [m/yr ice equiv], positive
    p      : float — Lliboutry shape parameter (p=3 typical for polar ice)

    Returns
    -------
    u : (nz,) array — velocity [m/yr], negative = downward
    """
    zeta = z / H_ice
    u = -b_acc * ((p + 2) * zeta
                  - (p + 2) * zeta ** (p + 1)
                  + zeta ** (p + 2))
    return u


def gas_loss_profile(z, lambda_max, z_char, z_gl_limit):
    """Spatially varying gas-loss rate λ(z)  [yr⁻¹].

    Exponential decay active only below z_gl_limit:
        λ(z) = lambda_max · exp(−z / z_char)   if z < z_gl_limit
        λ(z) = 0                                otherwise

    Parameters
    ----------
    z           : (nz,) array — height above bed [m]
    lambda_max  : float — maximum gas-loss rate [yr⁻¹]
    z_char      : float — e-folding depth [m]
    z_gl_limit  : float — depth above which gas loss is zero [m]

    Returns
    -------
    lam : (nz,) array — gas-loss rate [yr⁻¹]
    """
    lam = np.zeros_like(z, dtype=float)
    mask = z < z_gl_limit
    lam[mask] = lambda_max * np.exp(-z[mask] / z_char)
    return lam


def velocity_dansgaard_johnsen(z, H_ice, b_acc, h_kink):
    """Dansgaard-Johnsen (1969) vertical velocity adapted for Camp Century basal zone.

    Below the kink height h_kink, the vertical velocity is approximately constant:
        w_basal = -b_acc * h_kink / (2 * H_ice)

    In our 30-m basal domain the velocity goes from 0 at the bed to w_basal at the top,
    following a quadratic profile (physically reasonable in the basal shear layer):
        u(z) = w_basal * (z / Z_domain)^2

    Parameters
    ----------
    z      : (nz,) array -- height above bed [m]
    H_ice  : float -- total ice thickness [m]
    b_acc  : float -- accumulation rate [m/yr ice equiv], positive
    h_kink : float -- kink height above bed [m]  (Camp Century ~400 m, Hammer et al. 1978)

    Returns
    -------
    u : (nz,) array -- velocity [m/yr], negative = downward

    Reference
    ---------
    Dansgaard & Johnsen (1969), J. Glaciol., 8, 215-223.
    Hammer et al. (1978), J. Glaciol., 20, 3-26.
    """
    u_basal  = -b_acc * h_kink / (2.0 * H_ice)   # constant below kink [m/yr]
    Z_domain = z[-1]                              # top of modelled domain
    u = u_basal * (z / Z_domain) ** 2
    return u
