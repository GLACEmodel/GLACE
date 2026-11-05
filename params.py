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
# Surface BC / IC mode — controls what mixing ratio is used at z = Z (top of domain)
# and as initial condition throughout the column:
#   'atm'  — use X_atm (atmospheric value, 0.934 % for Ar)
#   'data' — use the topmost measured Ar fraction from the lab Excel file
#             (z = 26.55 m, Ar = 0.9658 % from Table_S1_122025_for manuscript.xlsx)
BC_IC_MODE = 'data'   # ← change to 'atm' to use the atmospheric value instead

X_atm   = 0.934e-2    # atmospheric mixing ratio (Ar: 0.934 %; adapt for other gases)
#   Used when BC_IC_MODE = 'atm', or as fallback if data file is unavailable.

# ── Site-specific parameters ──────────────────────────────────────────────────
TAC     = 4.856e-5    # total air content                   [m³_STP kg⁻¹ ice]
#   Source : data/Table_S1_122025_for manuscript.xlsx, sheet CampCentury, col D
#   (Gas Content [mL_STP / kg_ice]).
#   TAC = mean of the 10 measurements at z > 20 m (top of domain, above the
#   gas-loss zone): z = 20.70 → 26.55 m, values 36.7–56.0 mL/kg, mean = 48.56 mL/kg.
#   Conversion: 48.56 mL/kg × 1e-6 = 4.856e-5 m³_STP/kg.
#   These are the least gas-loss-affected samples and therefore best represent
#   the undisturbed basal-ice air content at Camp Century.
#   (Previous value: 0.091e-3 from Sailer et al., measured on the main ice column
#    above the basal zone — ~2× higher than observed here in the basal ice.)

# ── Transport ─────────────────────────────────────────────────────────────────
D_m2s       = 1e-11   # gas diffusivity in ice lattice      [m² s⁻¹]
flux_bottom = 6e-7    # basal Ar flux from bedrock          [mol m⁻² yr⁻¹]
#   Calibrated to match observed Ar% enrichment near bed (~1.6% at z<1m);
#   gas_loss = 0 (Ar is a noble gas, not lost like N2/O2).

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
# λ(z) = gas_loss_data · exp(−z / z_char)  for z < z_gl_limit, else 0  (mode data)
# λ(z) = 0 everywhere                                                   (mode none)
# Only the clathrate phase is affected (C_ice unchanged by gas loss).
#
# ► Switch between modes by changing GAS_LOSS_MODE below:
#     'none' — Ar is a noble gas, conserved (default for Ar runs)
#     'data' — exponential decay calibrated from lab data (for N₂/O₂ or sensitivity tests)
#
GAS_LOSS_MODE = 'none'   # ← change to 'data' to activate lab-calibrated gas loss

# Mode 'data' — calibration source: data/Table_S1_122025_for manuscript.xlsx
#   Sheet CampCentury, col D (Gas Content [mL_STP/kg_ice]).
#   The total gas content drops sharply in the bottom ~2 m (from ~50 mL/kg
#   to <15 mL/kg near the bed), indicating gas loss near the bed.
#   The exponential rate gas_loss_data is fitted to this gradient.
#   See glace/data.py :: load_campcentury() to reload the observations.
gas_loss_data = 1.8e-6   # calibrated gas-loss rate (mode 'data' only) [yr⁻¹]
z_char        = 8.0      # e-folding height above bed                   [m]
z_gl_limit    = 20.0     # upper limit of gas-loss zone                 [m]

# ── Grid ──────────────────────────────────────────────────────────────────────
nz = 101              # number of vertical nodes
Z  = 30.0             # domain height above bed             [m]

# ── Time ──────────────────────────────────────────────────────────────────────
T  = 8e5              # total simulation time               [yr]
dt = 3.0              # timestep                            [yr]
# CFL check is performed in build_params(); target CFL < 1 for upwind stability.
