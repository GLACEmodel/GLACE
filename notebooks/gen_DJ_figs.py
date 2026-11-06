import sys, os
sys.path.insert(0, os.path.expanduser('~/mnt/MODEL_local'))
os.chdir(os.path.expanduser('~/mnt/MODEL_local'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import types

from glace import params as P
from glace.physics import velocity_lliboutry, velocity_dansgaard_johnsen
from glace.main import build_params
from glace.core import run_simulation
from glace.data import load_campcentury

os.makedirs('figures', exist_ok=True)

z = np.linspace(0, P.Z, P.nz)
u_llib = velocity_lliboutry(z, P.H_ice, P.b_acc, P.p_llib)
u_dj   = velocity_dansgaard_johnsen(z, P.H_ice, P.b_acc, P.h_kink)

Umax_llib = u_llib[-1]
Umax_dj   = u_dj[-1]
print(f"Umax Lliboutry = {Umax_llib*100:.4f} cm/yr")
print(f"Umax DJ        = {Umax_dj*100:.4f} cm/yr")

# Figure 1 : profils de vitesse
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

ax = axes[0]
ax.plot(u_llib * 100, z, color='#2166AC', lw=2.0, label='Lliboutry (1979)')
ax.plot(u_dj   * 100, z, color='#D6604D', lw=2.0, linestyle='--', label='Dansgaard-Johnsen (1969)')
ax.set_xlabel('Vitesse verticale u(z)  [cm/yr]', fontsize=11)
ax.set_ylabel('Hauteur au-dessus du fond z  [m]', fontsize=11)
ax.set_title("Profils de vitesse d'advection", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

ax2 = axes[1]
diff_pct = (u_dj - u_llib) / np.abs(np.where(u_llib == 0, 1e-30, u_llib)) * 100
ax2.plot(diff_pct, z, color='#762A83', lw=2.0)
ax2.axvline(0, color='k', lw=0.8, linestyle=':')
ax2.set_xlabel('(DJ - Llib) / |Llib|  [%]', fontsize=11)
ax2.set_title('Ecart relatif DJ vs Lliboutry', fontsize=12)
ax2.grid(True, alpha=0.3)

info = (f"Lliboutry: Umax = {Umax_llib*100:.4f} cm/yr  |  DJ: Umax = {Umax_dj*100:.4f} cm/yr\n"
        f"h_kink = {P.h_kink:.0f} m  |  H_ice = {P.H_ice:.0f} m  |  b_acc = {P.b_acc} m/yr")
fig.text(0.5, 0.01, info, ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
fig.suptitle('Comparaison profils de vitesse verticale — Camp Century, 30 m basaux',
             fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig('figures/fig_velocity_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: figures/fig_velocity_comparison.png")
plt.close(fig)

# Figure 2 : effet sur profil Ar
print("Running Lliboutry simulation...")
sim_llib = build_params(P)
r_llib   = run_simulation(sim_llib, return_full=True)

print("Running DJ simulation...")
sim_dj       = build_params(P)
sim_dj['u']  = u_dj
r_dj         = run_simulation(sim_dj, return_full=True)

z_grid = r_llib['z']
pS     = sim_llib['pS']
n_air  = sim_llib['n_air']
f_llib = r_llib['C_bulk'][:, -1] / (pS + n_air) * 100
f_dj   = r_dj['C_bulk'][:, -1]  / (pS + n_air) * 100

obs = load_campcentury('data/Table_S1_122025_for manuscript.xlsx')

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)

ax = axes2[0]
ax.plot(f_llib, z_grid, color='#2166AC', lw=2.0, label='Lliboutry (1979)')
ax.plot(f_dj,   z_grid, color='#D6604D', lw=2.0, linestyle='--', label='Dansgaard-Johnsen (1969)')
obs_err = obs.get('f_Ar_u', np.zeros(len(obs['z'])))
ax.errorbar(obs['f_Ar']*100, obs['z'], xerr=obs_err*100,
            fmt='o', color='k', ms=5, lw=1, label='Observations', zorder=5)
ax.set_xlabel('Fraction Ar  [%]', fontsize=11)
ax.set_ylabel('z  [m]', fontsize=11)
ax.set_title('Profil Ar final — Llib vs DJ', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes2[1]
diff_Ar = f_dj - f_llib
ax2.plot(diff_Ar, z_grid, color='#762A83', lw=2.0)
ax2.axvline(0, color='k', lw=0.8, linestyle=':')
ax2.set_xlabel('Delta Ar  (DJ - Llib)  [% Ar]', fontsize=11)
ax2.set_title('Ecart profil Ar  DJ - Llib', fontsize=12)
ax2.grid(True, alpha=0.3)

max_diff = np.max(np.abs(diff_Ar))
info2 = (f"T = {P.T:.2g} yr  |  nz = {P.nz}  |  dt = {P.dt} yr\n"
         f"Ecart max |DJ - Llib| = {max_diff:.4f} % Ar")
fig2.text(0.5, 0.01, info2, ha='center', fontsize=9,
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
fig2.suptitle('Effet de la parametrisation de vitesse sur le profil Ar — Camp Century',
              fontsize=12, fontweight='bold')
fig2.tight_layout()
fig2.savefig('figures/fig_DJ_vs_llib_Ar.png', dpi=150, bbox_inches='tight')
print(f"Saved: figures/fig_DJ_vs_llib_Ar.png  |  ecart max = {max_diff:.4f} % Ar")
plt.close(fig2)
print("All done!")
