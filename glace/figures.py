"""
glace/figures.py — Plotting functions

All figure generation lives here. Each function takes a result dict
(output of run_simulation) and returns a matplotlib Figure.

Functions
---------
plot_profiles(result, title)
    Final-state thermodynamic partition: x, C_ice, C_bulk vs z.

plot_snapshots(result, n_snap, title)
    Time evolution at n evenly-spaced snapshots.

plot_tendencies(ADV, DIFF, t, z, title)
    Heatmap of advective and diffusive tendencies over time.

plot_partition_fractions(result, title)
    Fraction of total Ar in lattice vs clathrates.

plot_attribution(results_dict, z, title)
    Compare multiple simulations (diffusion-only, +advection, +gas loss).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── Colour helpers ────────────────────────────────────────────────────────────

BLUE_RAMP  = lambda n: plt.cm.Blues(np.linspace(0.3, 1.0, n))
ATTRIBUTION_COLORS = ['#144E9A', '#D72637', '#1E8A3F']
ATTRIBUTION_LINES  = ['-', '--', ':']


def _get_final(result, key):
    """Return final state from either a full time-series or a final-state result."""
    arr = result[key]
    return arr[:, -1] if arr.ndim == 2 else arr


# ── Figure functions ──────────────────────────────────────────────────────────

def plot_profiles(result, title="Final state — thermodynamic partition"):
    """Plot final x, C_ice, C_bulk profiles vs height above bed.

    Parameters
    ----------
    result : dict — output of run_simulation
    title  : str

    Returns
    -------
    fig : matplotlib Figure
    """
    z      = result['z']
    X      = _get_final(result, 'X')
    C_ice  = _get_final(result, 'C_ice')
    C_bulk = _get_final(result, 'C_bulk')

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    axes[0].plot(X * 100, z, lw=2.5, color='tab:blue')
    axes[0].set_xlabel('x  (%)',              fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Height above bed (m)', fontsize=12, fontweight='bold')
    axes[0].set_title('Ar mixing ratio  x',   fontsize=11)

    axes[1].plot(C_ice, z, lw=2.5, color='tab:orange')
    axes[1].set_xlabel('C_ice  (mol m⁻³)', fontsize=12, fontweight='bold')
    axes[1].set_title('Ar in ice lattice  C_ice', fontsize=11)

    axes[2].plot(C_bulk, z, lw=2.5, color='tab:green')
    axes[2].set_xlabel('C_bulk  (mol m⁻³)', fontsize=12, fontweight='bold')
    axes[2].set_title('Total Ar  C_bulk', fontsize=11)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=10)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_snapshots(result, n_snap=6, title="Time evolution"):
    """Plot C_bulk, C_ice, x at n_snap evenly-spaced snapshots.

    Parameters
    ----------
    result : dict — output of run_simulation with return_full=True
    n_snap : int
    title  : str

    Returns
    -------
    fig : matplotlib Figure
    """
    z  = result['z']
    t  = result['t']
    nt = result['C_bulk'].shape[1] - 1
    idx = np.round(np.linspace(0, nt, n_snap)).astype(int)
    colors = BLUE_RAMP(n_snap)
    labels = [f't = {t[i]:.2g} yr' for i in idx]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    for ii, k in enumerate(idx):
        kw = dict(lw=1.8, color=colors[ii])
        axes[0].plot(result['X'][:, k] * 100, z, **kw, label=labels[ii])
        axes[1].plot(result['C_ice'][:, k],   z, **kw)
        axes[2].plot(result['C_bulk'][:, k],  z, **kw)

    axes[0].set_xlabel('x  (%)',               fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Height above bed (m)',  fontsize=12, fontweight='bold')
    axes[0].set_title('Mixing ratio  x',        fontsize=11)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].set_xlabel('C_ice  (mol m⁻³)',  fontsize=12, fontweight='bold')
    axes[1].set_title('Ar in lattice',       fontsize=11)

    axes[2].set_xlabel('C_bulk  (mol m⁻³)', fontsize=12, fontweight='bold')
    axes[2].set_title('Total Ar',            fontsize=11)

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_partition_fractions(result, title="Ar partition — lattice vs clathrates"):
    """Fraction of total Ar in the lattice vs clathrates (final state).

    Returns
    -------
    fig : matplotlib Figure
    """
    z      = result['z']
    C_ice  = _get_final(result, 'C_ice')
    C_bulk = _get_final(result, 'C_bulk')

    frac_lattice = C_ice  / C_bulk
    frac_clath   = 1.0 - frac_lattice

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(frac_lattice * 100, z, lw=2.5, label='Ice lattice')
    ax.plot(frac_clath   * 100, z, lw=2.5, label='Clathrates', linestyle='--')
    ax.set_xlabel('Fraction of total Ar  (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Height above bed  z  (m)',  fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, frameon=False)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


def plot_attribution(results_dict, z, title="Process attribution"):
    """Compare C_bulk profiles from multiple simulations.

    Parameters
    ----------
    results_dict : dict  label (str) → C_bulk (nz,) array — final states
    z            : (nz,) array
    title        : str

    Returns
    -------
    fig : matplotlib Figure

    Example
    -------
    >>> plot_attribution({
    ...     'Diffusion + basal flux': r1['C_bulk'],
    ...     '+ Advection':            r2['C_bulk'],
    ...     '+ Gas loss (full)':      r3['C_bulk'],
    ... }, z)
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    for (label, C), color, ls in zip(results_dict.items(),
                                      ATTRIBUTION_COLORS,
                                      ATTRIBUTION_LINES):
        ax.plot(C, z, lw=3.5, color=color, linestyle=ls, label=label)

    ax.set_xlabel('C_bulk  (mol m⁻³)',        fontsize=14, fontweight='bold')
    ax.set_ylabel('Height above bed  z  (m)',  fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, frameon=False)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig
