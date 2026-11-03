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
