"""
glace/core.py — Core numerical routines

Operator splitting per timestep (A → F):
    A  advect_upwind / advect_tvd  (transport.py)
    B  partition()
    C  diffuse_implicit()
    D  rebuild_Cbulk()
    E  partition()          (re-equilibrate)
    F  apply_gas_loss()

Main entry point: run_simulation(params, advect_fn, return_full)
"""

import numpy as np
from .transport import advect_upwind
from .physics   import gas_loss_profile


# ── Thermodynamics ────────────────────────────────────────────────────────────

def partition(C_bulk, pS, n_air):
    """Instantaneous thermodynamic partition of C_bulk.

    C_bulk = x · (pS + n_air)
    C_ice  = x · pS

    Parameters
    ----------
    C_bulk : (nz,) array — total Ar concentration  [mol m⁻³ ice]
    pS     : (nz,) array — P_diss · S_Ar            [mol m⁻³ ice Pa⁻¹ × Pa]
    n_air  : float | (nz,) — mol air / m³ ice       [mol m⁻³ ice]

    Returns
    -------
    x     : (nz,) — Ar mixing ratio in clathrate gas  [mol/mol]
    C_ice : (nz,) — Ar dissolved in ice lattice        [mol m⁻³ ice]
    """
    x     = C_bulk / (pS + n_air)
    C_ice = x * pS
    return x, C_ice


def rebuild_Cbulk(C_bulk_adv, Cice_new, Cice_adv):
    """Reconstruct C_bulk after the diffusion step (operator D).

    Only the lattice increment is added:
        C_bulk_new = C_bulk_adv + (Cice_new − Cice_adv)

    Parameters
    ----------
    C_bulk_adv : (nz,) — C_bulk after advection step A
    Cice_new   : (nz,) — C_ice after diffusion step C
    Cice_adv   : (nz,) — C_ice just before diffusion (after partition B)

    Returns
    -------
    C_bulk_new : (nz,)
    """
    return C_bulk_adv + (Cice_new - Cice_adv)


# ── Diffusion ─────────────────────────────────────────────────────────────────

def build_diff_matrix(nz, r, bc_surf, bc_bed):
    """Build the implicit backward-Euler tridiagonal matrix for lattice diffusion.

    Interior: −r · C_{i-1} + (1+2r) · C_i − r · C_{i+1} = B_i
    Boundaries: Dirichlet → identity row; Neumann/flux → ghost-point formula.

    Parameters
    ----------
    nz             : int   — number of grid points
    r              : float — D · dt / dz²
    bc_surf, bc_bed: dict  — boundary condition dicts (key: 'type')

    Returns
    -------
    A : (nz, nz) ndarray — tridiagonal matrix
    """
    A = np.zeros((nz, nz))

    # Interior
    for i in range(1, nz - 1):
        A[i, i - 1] = -r
        A[i, i]     = 1 + 2 * r
        A[i, i + 1] = -r

    # Bottom BC (z = 0, index 0)
    if bc_bed['type'] == 'dirichlet':
        A[0, 0] = 1.0
    else:                             # flux or zeroflux — ghost point
        A[0, 0] = 1 + 2 * r
        A[0, 1] = -2 * r

    # Surface BC (z = Z, index -1)
    if bc_surf['type'] == 'dirichlet':
        A[-1, -1] = 1.0
    else:
        A[-1, -1] = 1 + 2 * r
        A[-1, -2] = -2 * r

    return A


def diffuse_implicit(Cice_adv, A, dz, dt, bc_surf, bc_bed):
    """One implicit diffusion step for C_ice (Fick's law, operator C).

    Solves A · Cice_new = B  with appropriate BC modifications.

    Parameters
    ----------
    Cice_adv       : (nz,) — C_ice after advection + partition
    A              : (nz, nz) ndarray — diffusion matrix from build_diff_matrix
    dz, dt         : float
    bc_surf, bc_bed: dict

    Returns
    -------
    Cice_new : (nz,) — C_ice after diffusion
    """
    B = Cice_adv.copy()

    # Bottom BC modification
    if bc_bed['type'] == 'dirichlet':
        B[0] = bc_bed['C_ice']
    elif bc_bed['type'] == 'flux':
        B[0] = Cice_adv[0] + 2 * bc_bed['F'] * dt / dz

    # Surface BC modification
    if bc_surf['type'] == 'dirichlet':
        B[-1] = bc_surf['C_ice']
    elif bc_surf['type'] == 'flux':
        B[-1] = Cice_adv[-1] - 2 * bc_surf['F'] * dt / dz

    return np.linalg.solve(A, B)


def apply_gas_loss(C_bulk, C_ice, decay_factor):
    """Gas loss step (operator F): clathrate phase decays, lattice unchanged.

    The clathrate content is C_clath = C_bulk − C_ice.
    After decay: C_clath_new = C_clath · decay_factor
    So:          C_bulk_new  = C_ice + C_clath_new

    Parameters
    ----------
    C_bulk       : (nz,) — total Ar before gas loss
    C_ice        : (nz,) — lattice Ar (unchanged by gas loss)
    decay_factor : (nz,) — exp(−λ · dt)

    Returns
    -------
    C_bulk_new : (nz,)
    """
    C_clath    = C_bulk - C_ice
    C_bulk_new = C_ice + C_clath * decay_factor
    return C_bulk_new


# ── Main simulation loop ──────────────────────────────────────────────────────

def run_simulation(params, advect_fn=None, return_full=False):
    """Run the full GLACE simulation.

    Parameters
    ----------
    params      : dict — assembled parameter dict (from glace/main.py::build_params)
    advect_fn   : callable, optional — advection function (default: advect_upwind)
    return_full : bool — if True, store the full time series (memory-heavy)
                         if False (default), return only the final state

    Returns
    -------
    result : dict with keys
        'C_bulk' : (nz,) or (nz, nt+1) depending on return_full
        'C_ice'  : same shape
        'X'      : same shape
        'z'      : (nz,) height above bed
        't'      : (nt+1,) or scalar T — time axis
    """
    if advect_fn is None:
        advect_fn = advect_upwind

    # Unpack
    z            = params['z']
    dz           = params['dz']
    dt           = params['dt']
    nt           = params['nt']
    u            = params['u']
    D            = params['D']           # [m² yr⁻¹]
    pS           = params['pS']
    n_air        = params['n_air']
    bc_surf      = params['bc_surf']
    bc_bed       = params['bc_bed']
    lambda_z     = params['lambda_z']
    Cbulk0       = params['Cbulk0']

    nz           = len(z)
    decay_factor = np.exp(-lambda_z * dt)
    t_axis       = np.arange(nt + 1) * dt

    # Build diffusion matrix once
    r = D * dt / dz ** 2
    A = build_diff_matrix(nz, r, bc_surf, bc_bed)

    # Initial condition
    x0, Cice0 = partition(Cbulk0, pS, n_air)
    C_bulk = Cbulk0.copy()
    C_ice  = Cice0.copy()

    if return_full:
        C_bulk_all = np.zeros((nz, nt + 1))
        C_ice_all  = np.zeros((nz, nt + 1))
        X_all      = np.zeros((nz, nt + 1))
        C_bulk_all[:, 0] = C_bulk
        C_ice_all[:, 0]  = C_ice
        X_all[:, 0]      = x0

    # ── Time loop — operator splitting A → F ─────────────────────────────────
    #
    #  Each timestep decomposes the full PDE into sequential sub-steps.
    #  The state variable transported is C_bulk; C_ice is diagnosed at each
    #  step via the Henry's-law equilibrium (partition).
    #
    #  A  Advect C_bulk          (transport.py: upwind or TVD)
    #  B  Partition C_bulk       → x, C_ice via Henry's law
    #  C  Diffuse C_ice          (implicit Euler, Fick's law)
    #  D  Rebuild C_bulk         C_bulk += ΔC_ice from diffusion
    #  E  Re-equilibrate         re-partition after diffusion step
    #  F  Gas loss               clathrate phase decays: C_clath × exp(−λ·dt)
    #
    for k in range(nt):

        # ── A — Advection (upwind) ───────────────────────────────────────────
        # Ice moves downward (u < 0); C_bulk is carried with the ice.
        # Stability: CFL = |u_max| · dt / dz < 1 (checked in build_params).
        C_bulk_adv = advect_fn(C_bulk, u, dz, dt, bc_surf, bc_bed)

        # ── B — Partition (Henry's law) ──────────────────────────────────────
        # Instantaneous equilibrium: x = C_bulk/(pS+n_air), C_ice = x·pS
        x_adv, Cice_adv = partition(C_bulk_adv, pS, n_air)

        # ── C — Diffusion of C_ice (implicit Euler) ──────────────────────────
        # Fick's law: ∂C_ice/∂t = D · ∂²C_ice/∂z²
        # Implicit scheme: unconditionally stable (no diffusion CFL constraint).
        # Solved as a tridiagonal linear system built once before the loop.
        Cice_new = diffuse_implicit(Cice_adv, A, dz, dt, bc_surf, bc_bed)

        # ── D — Reconstruct C_bulk ───────────────────────────────────────────
        # Only the lattice increment changes C_bulk (clathrathes don't diffuse).
        # C_bulk_new = C_bulk_adv + (C_ice_new − C_ice_adv)
        C_bulk = rebuild_Cbulk(C_bulk_adv, Cice_new, Cice_adv)

        # ── E — Re-equilibrate ───────────────────────────────────────────────
        # Diffusion has changed C_ice → re-partition to get consistent x, C_ice.
        X_k, C_ice = partition(C_bulk, pS, n_air)

        # ── F — Gas loss (clathrate phase only) ──────────────────────────────
        # C_clath = C_bulk − C_ice decays exponentially near the bed.
        # C_ice is not affected by gas loss (Henry's law re-establishes after).
        C_bulk = apply_gas_loss(C_bulk, C_ice, decay_factor)
        X_k, C_ice = partition(C_bulk, pS, n_air)

        if return_full:
            C_bulk_all[:, k + 1] = C_bulk
            C_ice_all[:, k + 1]  = C_ice
            X_all[:, k + 1]      = X_k

    # ── Output ───────────────────────────────────────────────────────────────
    X_fin, Cice_fin = partition(C_bulk, pS, n_air)

    if return_full:
        return {'C_bulk': C_bulk_all, 'C_ice': C_ice_all, 'X': X_all,
                'z': z, 't': t_axis}
    else:
        return {'C_bulk': C_bulk, 'C_ice': Cice_fin, 'X': X_fin,
                'z': z, 't': t_axis[-1]}
