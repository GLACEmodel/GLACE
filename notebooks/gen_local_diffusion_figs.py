import sys, os
sys.path.insert(0, '/sessions/rcw-01xadggl2d8opx7pxng8xpmm/mnt/MODEL_local')
os.chdir('/sessions/rcw-01xadggl2d8opx7pxng8xpmm/mnt/MODEL_local')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

from glace import params as P
from glace.main import build_params
from glace.core import run_simulation

os.makedirs('figures', exist_ok=True)

# Parametres de base
pS_val  = P.P_diss * P.S_Ar
n_air   = P.TAC * P.rho_ice / P.Vm_STP
alpha   = pS_val / (pS_val + n_air)
D_eff   = P.D_m2s * 3.1536e7 * alpha
print(f"D_eff = {D_eff:.3e} m2/yr  |  alpha = {alpha*100:.3f} %")

# ── Figure 1 : front de diffusion dz(T) ─────────────────────────────────────
T_vals = np.logspace(2, 8, 500)   # 100 yr -> 1e8 yr
dz_vals = np.sqrt(D_eff * T_vals)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.loglog(T_vals, dz_vals, color='#2166AC', lw=2.5)
ax.axhline(30.0, color='gray', lw=1.2, linestyle='--', label='Domaine complet (30 m)')
ax.axhline(1.0,  color='#D6604D', lw=1.2, linestyle=':', label='1 m (zone critique)')

# Marquer T cles
for T_mark, label, col in [(1e4, '10 000 yr', '#4DAC26'),
                             (1e5, '100 000 yr', '#762A83'),
                             (8e5, '800 000 yr', '#E08214'),
                             (1.16e8, '116 Myr (tau_diff)', '#1A1A1A')]:
    dz_m = np.sqrt(D_eff * T_mark)
    ax.plot(T_mark, dz_m, 'o', color=col, ms=7, zorder=5)
    ax.annotate(f'{label}\ndz~{dz_m:.1f} m', xy=(T_mark, dz_m),
                xytext=(T_mark*1.6, dz_m*0.55), fontsize=8, color=col,
                arrowprops=dict(arrowstyle='->', color=col, lw=0.8))

ax.set_xlabel('Temps T  [yr]', fontsize=11)
ax.set_ylabel('Profondeur de penetration  dz = sqrt(D_eff * T)  [m]', fontsize=11)
ax.set_title('Front de diffusion depuis le fond — Ar dissous dans le reseau cristallin', fontsize=12)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(1e2, 5e8)
ax.set_ylim(0.01, 200)

info = (f"D_eff = alpha * D  =  {alpha*100:.3f}% * {P.D_m2s:.0e} m2/s  =  {D_eff:.2e} m2/yr\n"
        f"tau_diff (30 m) = Z2/D_eff = {30**2/D_eff/1e6:.0f} Myr")
ax.text(0.98, 0.04, info, transform=ax.transAxes, ha='right', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

fig.tight_layout()
fig.savefig('figures/fig_diffusion_front.png', dpi=150, bbox_inches='tight')
print("Saved: figures/fig_diffusion_front.png")
plt.close(fig)

# ── Figure 2 : partition C_ice vs X*n_air pres du fond ──────────────────────
print("Running full simulation for partition plot...")
sim = build_params(P)
result = run_simulation(sim, return_full=True)

z_grid = result['z']
C_ice  = result['C_ice'][:, -1]      # Ar dissous dans reseau [mol/m3]
X      = result['X'][:, -1]          # fraction clathrate
n_air_arr = sim['n_air']
pS_arr    = sim['pS']
C_cla  = X * n_air_arr               # Ar dans clathrates [mol/m3]
C_tot  = result['C_bulk'][:, -1]

# fraction de chaque phase
frac_ice = C_ice / np.where(C_tot > 0, C_tot, 1e-30) * 100
frac_cla = C_cla / np.where(C_tot > 0, C_tot, 1e-30) * 100

# zoom sur les 5 premiers metres
mask = z_grid <= 5.0

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5.5))

ax = axes2[0]
ax.plot(C_ice[mask]*1e6,  z_grid[mask], color='#D6604D', lw=2.0, label='Ar dissous (C_ice) [x1e-6]')
ax.plot(C_cla[mask]*1e6,  z_grid[mask], color='#2166AC', lw=2.0, linestyle='--', label='Ar clathrates (X*n_air) [x1e-6]')
ax.set_xlabel('Concentration  [1e-6 mol/m3]', fontsize=11)
ax.set_ylabel('z  [m]', fontsize=11)
ax.set_title('Partition Ar : dissous vs clathrates\n(5 m basaux)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes2[1]
ax2.plot(frac_ice[mask], z_grid[mask], color='#D6604D', lw=2.0, label='% en phase dissoute')
ax2.plot(frac_cla[mask], z_grid[mask], color='#2166AC', lw=2.0, linestyle='--', label='% en clathrates')
ax2.axvline(50, color='k', lw=0.8, linestyle=':')
ax2.set_xlabel('Part de chaque phase  [%]', fontsize=11)
ax2.set_title('Fraction de chaque phase\n(5 m basaux)', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig2.suptitle("L'Ar basal entre en phase DISSOUTE — c'est la diffusion qui le transporte localement",
              fontsize=12, fontweight='bold')
fig2.tight_layout()
fig2.savefig('figures/fig_basal_partition.png', dpi=150, bbox_inches='tight')
print("Saved: figures/fig_basal_partition.png")
plt.close(fig2)

# ── Figure 3 : schema conceptuel ────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(9, 6))
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_facecolor('#F8F8F8')
fig3.patch.set_facecolor('#F8F8F8')

# Zone glace basale
ax3.add_patch(plt.Rectangle((1, 1.5), 8, 7, facecolor='#DEEBF7', edgecolor='#2166AC', lw=1.5, alpha=0.5))
ax3.text(5, 8.7, 'Glace basale (30 m)', ha='center', fontsize=11, color='#2166AC', fontweight='bold')

# Zone critique fond
ax3.add_patch(plt.Rectangle((1, 1.5), 8, 1.8, facecolor='#FEE0D2', edgecolor='#D6604D', lw=1.5, alpha=0.7))
ax3.text(5, 2.0, 'Zone critique : 0-2 m  (signal Ar dans les donnees)', ha='center', fontsize=9.5, color='#D6604D', fontweight='bold')

# Flux basal
ax3.annotate('', xy=(5, 1.7), xytext=(5, 0.5),
             arrowprops=dict(arrowstyle='->', color='#984EA3', lw=2.5))
ax3.text(5.2, 1.0, 'Flux basal Ar\n(phase DISSOUTE)', fontsize=9, color='#984EA3')

# Fleche diffusion locale
ax3.annotate('', xy=(3.2, 3.8), xytext=(3.2, 1.9),
             arrowprops=dict(arrowstyle='->', color='#D6604D', lw=2.0))
ax3.text(1.1, 2.9, 'Diffusion locale\n(dz~sqrt(D_eff*T))\ndominante ici', fontsize=8.5, color='#D6604D',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Fleche advection
ax3.annotate('', xy=(7, 7.5), xytext=(7, 4.0),
             arrowprops=dict(arrowstyle='->', color='#2166AC', lw=2.0))
ax3.text(7.2, 5.8, 'Advection\n(Lliboutry)\ndominante\nau-dessus', fontsize=8.5, color='#2166AC',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Transition dissolved -> clathrate
ax3.text(4.5, 3.5, 'Ar dissous\n--> clathrate', fontsize=8.5, color='#4D9221',
         bbox=dict(boxstyle='round', facecolor='#E6F5D0', alpha=0.9))

# Pe local
ax3.text(1.1, 7.5,
         'Pe global = 150 000\n(advection domine)\n\nPe local (z~0) -> 0\n(u -> 0, diffusion domine)',
         fontsize=8.5, color='#525252',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

ax3.set_title("Mecanismes de transport de l'Ar basal — deux zones distinctes",
              fontsize=12, fontweight='bold', pad=12)
fig3.tight_layout()
fig3.savefig('figures/fig_transport_schema.png', dpi=150, bbox_inches='tight')
print("Saved: figures/fig_transport_schema.png")
plt.close(fig3)
print("All done!")
