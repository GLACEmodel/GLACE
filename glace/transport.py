"""
glace/transport.py — Advection schemes

Available schemes
-----------------
advect_upwind(C, u, dz, dt, bc_surf, bc_bed)
    First-order upwind scheme (current default). Simple and stable (CFL ≤ 1),
    but introduces numerical diffusion ∝ |u|·dz/2.

advect_tvd(C, u, dz, dt, bc_surf, bc_bed)
    TVD van Leer scheme (TODO). Higher order, much less numerical diffusion.
    Needed for large-scale runs where upwind smears gradients too much.

Boundary condition dictionaries
--------------------------------
Each BC dict has:
    'type'  : 'dirichlet' | 'flux' | 'zeroflux'
    'C'     : Dirichlet value for C_bulk [mol m⁻³]  (dirichlet only)
    'C_ice' : Dirichlet value for C_ice  [mol m⁻³]  (dirichlet only)
    'F'     : Neumann flux [mol m⁻² yr⁻¹]           (flux only)
"""

import numpy as np


def advect_upwind(C, u, dz, dt, bc_surf, bc_bed):
    """First-order upwind advection of a scalar field C.

    "Upwind" means we look UPSTREAM (against the flow) to estimate the gradient.

    Ice compacts downward (u < 0 everywhere in the Eulerian frame fixed to the bed):
        u < 0  →  flow is downward  →  upstream is ABOVE  →  forward difference
                  ∂C/∂z ≈ (C_{i+1} − C_i) / dz

    If ever u > 0 (e.g. a sensitivity test with upward flow):
        u ≥ 0  →  upstream is BELOW  →  backward difference
                  ∂C/∂z ≈ (C_i − C_{i-1}) / dz

    The update is: C_adv = C − dt · u · ∂C/∂z

    Parameters
    ----------
    C       : (nz,) array — scalar field (C_bulk) at current timestep
    u       : (nz,) array — vertical velocity [m/yr], negative = downward
    dz      : float — grid spacing [m]
    dt      : float — timestep [yr]
    bc_surf : dict — surface boundary condition (z = Z, top)
    bc_bed  : dict — bed boundary condition    (z = 0, bottom)

    Returns
    -------
    C_adv : (nz,) array — field after one advection step
    """
    nz = len(C)
    dCdz = np.zeros(nz)

    # Interior nodes — upwind direction from sign of u
    u_pos = u[1:-1] >= 0
    dCdz[1:-1] = (u_pos  * (C[1:-1] - C[:-2])    # u ≥ 0: backward diff (upstream = below)
                + ~u_pos * (C[2:]   - C[1:-1])) / dz  # u < 0: forward diff (upstream = above)

    C_adv = C - dt * u * dCdz

    # ── Bottom BC (z = 0, index 0) ──────────────────────────────────────────
    if bc_bed['type'] == 'dirichlet':
        C_adv[0] = bc_bed['C']
    else:                                          # flux or zeroflux
        C_adv[0] = C[0]

    # ── Surface BC (z = Z, index -1) ────────────────────────────────────────
    if bc_surf['type'] == 'dirichlet':
        C_adv[-1] = bc_surf['C']
    else:
        C_adv[-1] = C[-1] - dt * u[-1] * (C[-1] - C[-2]) / dz

    return C_adv


def advect_tvd(C, u, dz, dt, bc_surf, bc_bed):
    """TVD van Leer advection scheme (higher order, less numerical diffusion).

    TODO: implement van Leer flux limiter.
    Reference: van Leer (1974), LeVeque (2002) §6.

    Currently raises NotImplementedError — use advect_upwind in the meantime.
    """
    raise NotImplementedError(
        "TVD scheme not yet implemented.\n"
        "Use advect_upwind, or implement the van Leer limiter here."
    )
