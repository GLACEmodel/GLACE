"""
GLACE — Model parameters (Camp Century, Greenland)

All physical, numerical and boundary parameters live here.
To run with different settings, import this module and override values,
or create a new params_*.py for another ice core.
"""

# ── Physical constants ────────────────────────────────────────────────────────
rho_ice = 917.0       # ice density                      [kg m⁻³]
Vm_STP  = 22.414e-3   # molar volume ideal gas at STP    [m³ mol⁻¹]

# ── Thermodynamics ────────────────────────────────────────────────────────────
P_diss  = 5e6         # Ar clathrate dissociation pressure  [Pa]
S_Ar    = 10e-9       # Ar solubility in ice lattice        [mol m⁻³ Pa⁻¹]
TAC     = 0.091e-3    # total air content (Sailer)          [m³_STP kg⁻¹]

# ── Transport ─────────────────────────────────────────────────────────────────
D_m2s       = 1e-11   # Ar lattice diffusivity              [m² s⁻¹]
flux_bottom = 1.8e-9  # basal Ar flux                       [mol m⁻² yr⁻¹]

# ── Velocity — Lliboutry (1979), frozen bed ───────────────────────────────────
# w(ζ) = −ḃ · [(p+2)ζ − (p+2)ζ^(p+1) + ζ^(p+2)],  ζ = z/H_ice
b_acc  = 0.35         # accumulation rate                   [m yr⁻¹ ice equiv]
H_ice  = 1387.0       # Camp Century ice thickness          [m]
p_llib = 3            # Lliboutry shape parameter           [-]

# ── Gas loss ──────────────────────────────────────────────────────────────────
# Exponential decay active only in the bottom z_gl_limit metres.
# Relates to observed gradient: gas_loss ≈ |dG/dz| * |Umax| / G_ref
gas_loss   = 1.8e-6   # maximum gas-loss rate               [yr⁻¹]
z_char     = 8.0      # e-folding depth                     [m]
z_gl_limit = 20.0     # depth below which gas loss is active [m]

# ── Grid ──────────────────────────────────────────────────────────────────────
nz = 101              # number of vertical nodes
Z  = 30.0             # domain height above bed             [m]

# ── Time ──────────────────────────────────────────────────────────────────────
T  = 1.25e5           # total simulation time               [yr]
dt = 3.0              # timestep                            [yr]
# CFL note: with Umax ~ −0.05 m/yr and dz ~ 0.3 m → dt_max ~ 6 yr → dt=3 gives CFL~0.5
