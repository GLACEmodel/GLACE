"""
glace/main.py — GLACE main simulation script

Usage
-----
From the GLACE/ root directory:

    python -m glace.main              # run with default params (Camp Century)
    python glace/main.py              # same

The script:
  1. Assembles all parameters into a dict via build_params()
  2. Runs the simulation via run_simulation()
  3. Runs three attribution cases (diffusion / +advection / +gas loss)
  4. Saves figures to figures/
"""

import sys
import os
import numpy as np

# Allow running as a script from GLACE/ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glace import params as P
from glace.core    import run_simulation, partition
from glace.physics import velocity_lliboutry, gas_loss_profile
from glace.figures import (plot_profiles, plot_snapshots,
                            plot_partition_fractions, plot_attribution,
                            plot_data_vs_model)


# ── Parameter assembly ────────────────────────────────────────────────────────

def build_params(p=P):
    """Assemble all model parameters into a dict for run_simulation.

    Override any value by passing a modified params module or a Namespace:
        from glace import params as p; p.D_m2s = 5e-12
        sim_params = build_params(p)

    Returns
    -------
    params_dict : dict
    """
    # Grid
    z  = np.linspace(0, p.Z, p.nz)
    dz = z[1] - z[0]

    # Time
    nt = int(round(p.T / p.dt))

    # Thermodynamics
    pS    = p.P_diss * p.S_Ar * np.ones(p.nz)   # [mol m⁻³ ice]
    n_air = p.TAC * p.rho_ice / p.Vm_STP          # [mol m⁻³ ice]

    # Velocity (Lliboutry)
    u    = velocity_lliboutry(z, p.H_ice, p.b_acc, p.p_llib)
    Umax = u[-1]                                   # velocity at surface

    # Convert D to m²/yr
    D = p.D_m2s * 3.1536e7

    # CFL / stability check
    CFL  = abs(Umax) * p.dt / dz
    diff_num = D * p.dt / dz**2
    print(f"  Umax          = {Umax:.4e} m/yr")
    print(f"  CFL           = {CFL:.3f}  {'⚠ > 1 !' if CFL > 1 else '✓'}")
    print(f"  Diffusion no. = {diff_num:.4f}")

    # Gas-loss profile — two modes selectable via params.GAS_LOSS_MODE
    #   'none' : λ = 0 everywhere (Ar, noble gas, conservative)
    #   'data' : exponential profile calibrated from lab data (N₂/O₂ or sensitivity)
    if p.GAS_LOSS_MODE == 'none':
        lambda_z = np.zeros(p.nz)
        print(f"  Gas loss mode   = none  (λ = 0 everywhere)")
    else:  # 'data'
        lambda_z = gas_loss_profile(z, p.gas_loss_data, p.z_char, p.z_gl_limit)
        print(f"  Gas loss mode   = data  (λ_max = {p.gas_loss_data:.2e} yr⁻¹, z_char = {p.z_char} m)")

    # Surface mixing ratio — topmost measured Ar fraction from lab Excel file
    # (z = 26.55 m, col L of Table_S1_122025_for manuscript.xlsx)
    from glace.data import load_campcentury
    _data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'data', 'Table_S1_122025_for manuscript.xlsx')
    _obs = load_campcentury(_data_path)
    _idx_top = np.argmax(_obs['z'])
    x_surf = float(_obs['f_Ar'][_idx_top])
    print(f'  Surface BC/IC   = {x_surf*100:.4f} % Ar  (topmost data point, z = {_obs["z"][_idx_top]:.2f} m)')

    # Boundary conditions
    bc_surf = {
        'type' : 'dirichlet',
        'C'    : x_surf * (pS[-1] + n_air),   # total Ar at surface mixing ratio
        'C_ice': x_surf * pS[-1],              # lattice Ar via Henry's law
        'F'    : 0.0,
    }
    bc_bed = {
        'type' : 'flux',
        'C'    : 0.0,
        'C_ice': 0.0,
        'F'    : p.flux_bottom,                # [mol m⁻² yr⁻¹] — already in yr units
    }

    # Initial condition — uniform field at surface mixing ratio
    Cbulk0 = x_surf * (pS + n_air)

    return dict(
        z=z, dz=dz, dt=p.dt, nt=nt,
        u=u, D=D,
        pS=pS, n_air=n_air,
        bc_surf=bc_surf, bc_bed=bc_bed,
        lambda_z=lambda_z, Cbulk0=Cbulk0,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')    # headless — save to file, no display needed
    import matplotlib.pyplot as plt

    os.makedirs('figures', exist_ok=True)

    print("=" * 55)
    print("  GLACE — Gas Lattice And Clathrate Evolution")
    print("  Camp Century, Greenland")
    print("=" * 55)

    # ── 1. Build params ───────────────────────────────────────────────────────
    print("\n[ Parameters ]")
    sim_params = build_params()
    print(f"  nz = {P.nz},  T = {P.T:.2g} yr,  dt = {P.dt} yr"
          f"  ({sim_params['nt']} timesteps)")

    # ── 2. Main simulation ────────────────────────────────────────────────────
    print("\n[ Running main simulation ]")
    result = run_simulation(sim_params, return_full=True)
    print("  Done.")

    # Diagnostics
    C_fin = result['C_bulk'][:, -1]
    inventory = np.trapezoid(C_fin, result['z'])
    print(f"  Final Ar inventory : {inventory:.4e} mol m⁻²")
    err = result['C_bulk'] - (result['C_ice'] + result['X'] * sim_params['n_air'])
    print(f"  Partition max error: {np.abs(err).max():.2e} mol m⁻³")

    # ── 3. Figures ────────────────────────────────────────────────────────────
    print("\n[ Saving figures ]")

    fig1 = plot_profiles(result,
                         title=f"GLACE — final state  (T = {P.T:.2g} yr)")
    fig1.savefig('figures/fig1_profiles.png', dpi=150, bbox_inches='tight')
    print("  figures/fig1_profiles.png")

    fig2 = plot_snapshots(result, n_snap=6,
                          title="GLACE — time evolution")
    fig2.savefig('figures/fig2_snapshots.png', dpi=150, bbox_inches='tight')
    print("  figures/fig2_snapshots.png")

    fig3 = plot_partition_fractions(result)
    fig3.savefig('figures/fig3_partition.png', dpi=150, bbox_inches='tight')
    print("  figures/fig3_partition.png")

    # ── 4. Attribution ────────────────────────────────────────────────────────
    print("\n[ Attribution cases ]")
    z = sim_params['z']

    # Case 1 — diffusion + basal flux only (u = 0, no gas loss)
    p1 = dict(sim_params)
    p1['u']        = np.zeros_like(sim_params['u'])
    p1['lambda_z'] = np.zeros_like(sim_params['lambda_z'])
    print("  Case 1 : diffusion + basal flux ...", end=' ', flush=True)
    r1 = run_simulation(p1)
    print("OK")

    # Case 2 — + advection (no gas loss)
    p2 = dict(sim_params)
    p2['lambda_z'] = np.zeros_like(sim_params['lambda_z'])
    print("  Case 2 : + advection ...", end=' ', flush=True)
    r2 = run_simulation(p2)
    print("OK")

    # Case 3 — full model
    print("  Case 3 : full model ...", end=' ', flush=True)
    r3 = run_simulation(sim_params)
    print("OK")

    fig4 = plot_attribution(
        {
            'Diffusion + basal flux': r1['C_bulk'],
            '+ Advection':            r2['C_bulk'],
            '+ Gas loss  (full)':     r3['C_bulk'],
        },
        z,
        title=f"Process attribution  (T = {P.T:.2g} yr)",
    )
    fig4.savefig('figures/fig4_attribution.png', dpi=150, bbox_inches='tight')
    print("  figures/fig4_attribution.png")

    fig5 = plot_data_vs_model(result,
                             title=f"Model vs observations  (T = {P.T:.2g} yr)")
    fig5.savefig('figures/fig5_data_vs_model.png', dpi=150, bbox_inches='tight')
    print("  figures/fig5_data_vs_model.png")

    plt.close('all')
    print("\nAll done! ✓")
