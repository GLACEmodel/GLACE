"""Génère fig_peclet_time.png — évolution de Pe au front diffusif avec le temps."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glace import params as P

# ── Paramètres ────────────────────────────────────────────────────────────────
pS    = P.P_diss * P.S_Ar
n_air = P.TAC * P.rho_ice / P.Vm_STP
alpha = pS / (pS + n_air)
D_eff = P.D_m2s * 3.1536e7 * alpha   # [m²/yr]

# ── Formule analytique ────────────────────────────────────────────────────────
# u(z) ≈ b_acc*(p+2)*z/H_ice  (Lliboutry linéarisé près du fond)
# δz(T) = sqrt(D_eff * T)
# Pe_front(T) = |u(δz)| * δz / D_eff = b_acc*(p+2)*T/H_ice   (D_eff s'annule !)
k = P.b_acc * (P.p_llib + 2) / P.H_ice   # [yr⁻¹]
T_star = 1.0 / k                          # T où Pe_front = 1
print(f"Coefficient  k = b_acc*(p+2)/H_ice = {k:.4e} yr⁻¹")
print(f"T* (Pe=1)       = {T_star:.0f} yr")
print(f"Pe_front linéaire en T  (D_eff s'annule !)")

T_arr = np.logspace(1, 7, 500)   # 10 → 10 Myr
Pe_front = k * T_arr
dz_front = np.sqrt(D_eff * T_arr)

# T notables Camp Century
T_marks = [792, 1e4, 1e5, 8e5, 1.16e8]
labels  = ['$T^*$=793 yr\n(Pe=1)', '10 kyr', '100 kyr', '800 kyr', '116 Myr\n(τ_diff)']
colors  = ['green', '#4393c3', '#2166ac', '#d6604d', 'purple']

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

# ── Panneau gauche : Pe_front(T) ──────────────────────────────────────────────
ax = axes[0]
ax.loglog(T_arr, Pe_front, color='#2166ac', lw=2.5,
          label=r'$Pe_{\rm front}(T) = \frac{b_{\rm acc}(p+2)}{H_{\rm ice}} \cdot T$')

# Zones de régime
ax.axhspan(0.001, 1,    alpha=0.10, color='green',  label='Diffusion domine (Pe < 1)')
ax.axhspan(1,     10,   alpha=0.08, color='orange',  label='Transition (1 < Pe < 10)')
ax.axhspan(10,    2e6,  alpha=0.07, color='crimson', label='Advection domine (Pe > 10)')

ax.axhline(1,  color='green',  lw=1.2, linestyle='--', alpha=0.8)
ax.axhline(10, color='orange', lw=1.2, linestyle='--', alpha=0.8)

for Tv, lbl, col in zip(T_marks, labels, colors):
    if 10 <= Tv <= 1e7:
        Pe_v = k * Tv
        ax.axvline(Tv, color=col, lw=1.3, linestyle=':', alpha=0.9)
        ax.scatter([Tv], [Pe_v], color=col, s=60, zorder=5)
        offset_y = 2.5 if Pe_v < 500 else 0.4
        ax.annotate(lbl, xy=(Tv, Pe_v), xytext=(Tv*1.4, Pe_v*offset_y),
                    fontsize=8, color=col, va='center')

ax.set_xlabel('Temps  $T$  [yr]', fontsize=11)
ax.set_ylabel('$Pe_{\\rm front}(T)$  [—]', fontsize=11)
ax.set_title('Péclet au front diffusif\n' r'$Pe_{\rm front} \propto T$  (D_eff s\'annule)', fontsize=11)
ax.set_xlim(10, 1e7)
ax.set_ylim(0.001, 2e4)
ax.legend(fontsize=8.5, loc='upper left')
ax.grid(True, which='both', alpha=0.3)

# ── Panneau droit : δz(T) + Pe_front coloré ──────────────────────────────────
ax2 = axes[1]

# Colorier la courbe δz(T) selon le régime Pe
from matplotlib.collections import LineCollection
points  = np.array([T_arr, dz_front]).T.reshape(-1, 1, 2)
segs    = np.concatenate([points[:-1], points[1:]], axis=1)
Pe_norm = Pe_front / 10   # normalisé : Pe=10 → 1
cmap    = plt.get_cmap('RdYlGn_r')
lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, 3))
lc.set_array(np.log10(np.maximum(Pe_front, 0.001)))
lc.set_linewidth(2.5)
ax2.add_collection(lc)
fig.colorbar(lc, ax=ax2, label='$\\log_{10}(Pe_{\\rm front})$', shrink=0.8)

# Marquer les temps notables
for Tv, lbl, col in zip(T_marks[1:4], labels[1:4], colors[1:4]):
    dz_v = np.sqrt(D_eff * Tv)
    ax2.scatter([Tv], [dz_v], color=col, s=70, zorder=5)
    ax2.annotate(f'{lbl}\nδz={dz_v:.2f} m', xy=(Tv, dz_v),
                 xytext=(Tv*2, dz_v+0.15), fontsize=8.5, color=col,
                 arrowprops=dict(arrowstyle='->', color=col, lw=1))

ax2.set_xscale('log')
ax2.set_xlabel('Temps  $T$  [yr]', fontsize=11)
ax2.set_ylabel('Front diffusif  $\\delta z(T) = \\sqrt{D_{\\rm eff} \\cdot T}$  [m]', fontsize=11)
ax2.set_title('Position du front  colorée par régime Pe', fontsize=11)
ax2.set_xlim(10, 1e7)
ax2.set_ylim(0, 32)
ax2.axhline(P.Z, color='grey', lw=1.2, linestyle='--', alpha=0.7, label='Domaine (30 m)')
ax2.axhline(1,   color='grey', lw=0.8, linestyle=':', alpha=0.6, label='1 m')
ax2.legend(fontsize=8.5)
ax2.grid(True, which='both', alpha=0.3)

info = (f"k = b_acc(p+2)/H_ice = {k:.3e} yr⁻¹  |  "
        f"T* = {T_star:.0f} yr  |  "
        f"D_eff = {D_eff:.2e} m²/yr  |  alpha = {alpha*100:.2f}%")
fig.text(0.5, 0.01, info, ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
fig.suptitle('Évolution temporelle du Péclet au front d\'enrichissement basal\n'
             r'Camp Century  —  $Pe_{\rm front}(T) = k \cdot T$,  $k = b_{\rm acc}(p+2)/H_{\rm ice}$',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()

os.makedirs('figures', exist_ok=True)
fig.savefig('figures/fig_peclet_time.png', dpi=150, bbox_inches='tight')
print("Saved: figures/fig_peclet_time.png")
print(f"T* = {T_star:.0f} yr  |  Pe(10kyr)={k*1e4:.0f}  |  Pe(100kyr)={k*1e5:.0f}  |  Pe(800kyr)={k*8e5:.0f}")
