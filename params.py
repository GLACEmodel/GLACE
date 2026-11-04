"""
GLACE — Model parameters

All physical, numerical and boundary parameters live here.
To adapt to a different gas or site, copy this file (e.g. params_Ne_GRIP.py)
and override the relevant groups:

    GAS-DEPENDENT   : P_diss, S_Ar, D_m2s, X_atm
    SITE-DEPENDENT  : H_ice, b_acc, p_llib, TAC, flux_bottom, gas_loss, ...
    NUMERICAL       : nz, Z, T, dt

Default: Argon (Ar), Camp Century, Greenland.
"""

# ── Physical constants ────────────────────────────────────────────────────────
rho_ice = 917.0       # ice density                      [kg m⁻³]
Vm_STP  = 22.414e-3   # molar volume ideal gas at STP    [m³ mol⁻¹]

# ── Gas-specific thermodynamics ───────────────────────────────────────────────
# Henry's law: C_ice = x · P_diss · S_Ar  (Ar dissolved in ice lattice)
# Change P_diss and S_Ar to switch to another gas (Ne, Kr, Xe, N2 …).
P_diss  = 5e6         # clathrate dissociation pressure     [Pa]
S_Ar    = 10e-9       # gas solubility in ice lattice       [mol m⁻³ Pa⁻¹]
X_atm   = 0.934e-2    # atmospheric mixing ratio (Ar: 0.934 %; adapt for other gases)

# ── Site-specific parameters ──────────────────────────────────────────────────
TAC     = 0.091e-3    # total air content (Sailer et al.)   [m³_STP kg⁻¹ ice]

# ── Transport ─────────────────────────────────────────────────────────────────
D_m2s       = 1e-11   # gas diffusivity in ice lattice      [m² s⁻¹]
flux_bottom = 1.8e-9  # basal gas flux from bedrock         [mol m⁻² s⁻¹]

# ── Velocity — Lliboutry (1979), frozen bed ───────────────────────────────────
# Eulerian frame fixed to bed; z positive upward.
# w(ζ) = −b_acc · [(p+2)ζ − (p+2)ζ^(p+1) + ζ^(p+2)],  ζ = z/H_ice
# w < 0 everywhere: ice compacts DOWNWARD as snow accumulates above.
# w(0) = 0  (frozen bed, no basal sliding)
# w(H) = −b_acc  (surface descends at the accumulation rate)
b_acc  = 0.35         # accumulation rate                   [m yr⁻¹ ice equiv]
H_ice  = 1387.0       # ice thickness                       [m]
p_llib = 3            # Lliboutry shape parameter           [-]

# ── Gas loss ──────────────────────────────────────────────────────────────────
# λ(z) = gas_loss · exp(−z / z_char)  for z < z_gl_limit, else 0
# Only the clathrate phase is affected (C_ice unchanged by gas loss).
#
# Calibration source : data/Table_S1_122025_for manuscript.xlsx
#   Sheet CampCentury, col D = Gas Content [mL_STP / kg_ice].
#   The total gas content drops sharply in the bottom ~2 m (from ~91 mL/kg
#   at depth to 10–25 mL/kg near the bed), indicating gas loss dominated by
#   clathrate dissociation. The exponential profile is fitted to this gradient.
#   See glace/data.py :: load_campcentury() to reload the observations.
gas_loss   = 1.8e-6   # maximum gas-loss rate               [yr⁻¹]
z_char     = 8.0      # e-folding height above bed          [m]
z_gl_limit = 20.0     # upper limit of gas-loss zone        [m]

# ── Grid ──────────────────────────────────────────────────────────────────────
nz = 101              # number of vertical nodes
Z  = 30.0             # domain height above bed             [m]

# ── Time ──────────────────────────────────────────────────────────────────────
T  = 1.25e5           # total simulation time               [yr]
dt = 3.0              # timestep                            [yr]
# CFL check is performed in build_params(); target CFL < 1 for upwind stability.
