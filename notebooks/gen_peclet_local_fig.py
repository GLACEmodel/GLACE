"""Génère fig_peclet_local.png — Pe local vs z, avec encadré zoom et contexte de la grille."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from glace import params as P
from glace.physics import velocity_lliboutry

z  = np.linspace(0, P.Z, P.nz)
u  = velocity_lliboutry(z, P.H_ice, P.b_acc, P.p_llib)
dz = z[1] - z[0]

pS    = P.P_diss * P.S_Ar
n_air = P.TAC * P.rho_ice / P.Vm_STP
alpha = pS / (pS + n_air)
D_eff = P.D_m2s * 3.1536e7 * alpha   # m²/yr

# Pe_local(z) = |u(z)| * z / D_eff
Pe_local = np.abs(u) * z / D_eff
Pe_local[0] = 1e-10  # éviter log(0)

# Pe_cfl(z) = |u(z)| * dz / D_eff  (pertinent pour la stabilité numérique)
Pe_cfl = np.abs(u) * dz / D_eff
Pe_cfl[0] = 1e-10

# Trouver z* (Pe_local = 1)
mask = Pe_local >= 1
z_star = z[np.argmax(mask)]   # première valeur >= 1
# interpolation linéaire entre le point avant et après
i0 = np.argmax(mask) - 1
if i0 >= 0:
    z_star = z[i0] + (1 - Pe_local[i0]) / (Pe_local[i0+1] - Pe_local[i0]) * dz

Pe_global = float(np.max(np.abs(u))) * P.Z / D_eff

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

# ── Panneau gauche : Pe_local(z), axe z horizontal ───────────────────────────
ax = axes[0]
ax.semilogy(z, Pe_local, color='#1a6faf', lw=2.0, label=r'$Pe_{\rm local}(z) = |u(z)|\cdot z / D_{\rm eff}$')
ax.semilogy(z, Pe_cfl,   color='#e07b3f', lw=1.8, linestyle='--',
            label=r'$Pe_{\rm CFL}(z) = |u(z)|\cdot \Delta z / D_{\rm eff}$')

# Ligne Pe = 1
ax.axhline(1, color='crimson', lw=1.5, linestyle=':', label='$Pe = 1$ (transition)')

# Résolution de la grille
ax.axvline(dz, color='grey', lw=1.2, linestyle='--', alpha=0.7, label=f'$\\Delta z$ = {dz:.2f} m (grille)')

# Zone diffusion locale (z < z*)
ax.axvspan(0, z_star, alpha=0.12, color='green', label=f'$z < z^*$ = {z_star*100:.1f} cm\n(diff. domine)')

ax.set_xlabel('Hauteur au-dessus du fond  $z$  [m]', fontsize=11)
ax.set_ylabel('$Pe_{\mathrm{local}}(z)$  [—]', fontsize=11)
ax.set_title('Péclet local — profil vertical', fontsize=12)
ax.set_xlim(0, P.Z)
ax.set_ylim(1e-3, 2e5)
ax.legend(fontsize=8.5, loc='lower right')
ax.grid(True, which='both', alpha=0.3)

# Annotations
ax.annotate(f'$Pe_{{\\rm global}}$ = {Pe_global:.1e}', xy=(P.Z, Pe_global),
            xytext=(18, 2e4), fontsize=9, color='#1a6faf',
            arrowprops=dict(arrowstyle='->', color='#1a6faf', lw=1))
ax.annotate(f'$z^*$ ≈ {z_star*100:.0f} cm\n(sous la grille !)', xy=(z_star, 1),
            xytext=(2, 5e-2), fontsize=8.5, color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=1))

# ── Panneau droit : zoom 0-2 m ────────────────────────────────────────────────
ax2 = axes[1]
mask_zoom = z <= 2.0
ax2.semilogy(z[mask_zoom], Pe_local[mask_zoom], color='#1a6faf', lw=2.5,
             label=r'$Pe_{\rm local}(z)$')
ax2.semilogy(z[mask_zoom], Pe_cfl[mask_zoom],   color='#e07b3f', lw=2.0, linestyle='--',
             label=r'$Pe_{\rm CFL}(z)$')
ax2.axhline(1,    color='crimson', lw=1.5, linestyle=':', label='$Pe = 1$')
ax2.axvline(dz,   color='grey',    lw=1.5, linestyle='--', alpha=0.8,
            label=f'$\\Delta z$ = {dz:.2f} m')
ax2.axvspan(0, z_star, alpha=0.15, color='green')
ax2.axvspan(0, dz, alpha=0.08, color='grey', label='Zone sous-grille')

ax2.set_xlabel('Hauteur au-dessus du fond  $z$  [m]', fontsize=11)
ax2.set_ylabel('$Pe_{\mathrm{local}}(z)$  [—]', fontsize=11)
ax2.set_title('Zoom — 2 m basaux', fontsize=12)
ax2.set_xlim(0, 2.0)
ax2.set_ylim(5e-4, 1e4)
ax2.legend(fontsize=8.5, loc='lower right')
ax2.grid(True, which='both', alpha=0.3)

ax2.annotate(f'$z^*$ ≈ {z_star*100:.0f} cm\n(diff. domine en dessous)', xy=(z_star, 1),
             xytext=(0.4, 0.05), fontsize=9, color='green',
             arrowprops=dict(arrowstyle='->', color='green', lw=1.2))
ax2.annotate(f'$\\Delta z$ = {dz:.2f} m\n(limite grille)', xy=(dz, 10),
             xytext=(0.6, 500), fontsize=8.5, color='grey',
             arrowprops=dict(arrowstyle='->', color='grey', lw=1.2))

info = (f"$Pe_{{\\rm global}}$ = {Pe_global:.1e}  (Z={P.Z:.0f}m)  |  "
        f"$z^*$ = {z_star*100:.0f} cm  |  $\\Delta z$ = {dz:.2f} m  |  "
        f"$D_{{\\rm eff}}$ = {D_eff:.2e} m²/yr")
fig.text(0.5, 0.01, info, ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
fig.suptitle('Nombre de Péclet local  —  Camp Century\n'
             r'$Pe_{\rm local}(z) = |u(z)|\cdot z / D_{\rm eff}$   vs   '
             r'$Pe_{\rm CFL}(z) = |u(z)|\cdot \Delta z / D_{\rm eff}$',
             fontsize=12, fontweight='bold', y=1.01)
fig.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig('figures/fig_peclet_local.png', dpi=150, bbox_inches='tight')
print(f"Saved: figures/fig_peclet_local.png")
print(f"z* = {z_star*100:.1f} cm  (Pe_local = 1  crossover)")
print(f"Pe_global = {Pe_global:.2e}")
