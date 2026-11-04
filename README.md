# GLACE — Gas Lattice And Clathrate Evolution

A 1D Eulerian model for noble gas transport in basal ice sheets, combining downward ice advection, lattice diffusion (Fick's law), thermodynamic partitioning between clathrathes and the ice lattice (Henry's law), and clathrate gas loss near the bed.

Developed as part of a PhD thesis at the Université Libre de Bruxelles (ULB), Sandra group.  
Default application: **Argon (Ar), Camp Century ice core, Greenland**.

---

## Physical variables

| Variable | Symbol | Units | Description |
|---|---|---|---|
| Ar mixing ratio | `x` | mol/mol | Fraction of Ar in clathrate gas phase |
| Lattice Ar | `C_ice` | mol m⁻³ | Ar dissolved in ice crystal (Henry's law) |
| Total Ar | `C_bulk` | mol m⁻³ | C_ice + clathrate Ar |

**Thermodynamic equilibrium (Henry's law):**
```
x       = C_bulk / (P_diss · S_gas + n_air)
C_ice   = x · P_diss · S_gas
```
`P_diss` is the clathrate dissociation pressure, `S_gas` is the gas solubility in the ice lattice, and `n_air` is the total air content. This relationship is instantaneous: given `C_bulk`, `x` and `C_ice` are fully determined.

---

## Coordinate system

- `z` = height above bed [m], from `z = 0` (bedrock) to `z = Z` (top of domain)
- Vertical ice velocity `w(z) < 0` everywhere: ice compacts **downward** as snow accumulates above (Eulerian frame fixed to bed)

---

## Operator splitting — one timestep

Each timestep `dt` decomposes into six sequential sub-steps (A → F):

| Step | Operation | Method |
|---|---|---|
| A | Advect `C_bulk` downward with ice | 1st-order upwind (TVD planned) |
| B | Partition → `x`, `C_ice` | Henry's law, instantaneous |
| C | Diffuse `C_ice` via Fick's law | Implicit backward Euler |
| D | Rebuild `C_bulk` from diffusion increment | `C_bulk += ΔC_ice` |
| E | Re-equilibrate `x`, `C_ice` | Henry's law |
| F | Clathrate gas loss (near bed) | Exponential decay |

---

## Numerical schemes and stability

### A — Advection: first-order upwind
The ice velocity `w(z)` is downward (`w < 0`), so the upstream direction is **above** each node. The upwind scheme uses a one-sided finite difference in that direction:

```
∂C/∂z ≈ (C_{i+1} − C_i) / dz    when w < 0  (upstream = above)
∂C/∂z ≈ (C_i − C_{i-1}) / dz    when w > 0  (upstream = below)
```

**Stability condition (CFL):** `|w_max| · dt / dz < 1`

The first-order upwind scheme introduces **numerical diffusion** proportional to `|w| · dz / 2`. This is acceptable for the current resolution but can smear sharp gradients. A TVD van Leer scheme (higher order, less diffusion) is planned in `transport.py`.

### C — Diffusion: implicit backward Euler
Fick's law for lattice Ar:
```
∂C_ice/∂t = D · ∂²C_ice/∂z²
```
Discretised as a tridiagonal system `A · C_ice^{n+1} = B` (backward Euler). Solved with `numpy.linalg.solve`.

**Why implicit?** The explicit (forward Euler) scheme requires `D · dt / dz² < 0.5` for stability — with `D ~ 3×10⁻⁴ m²/yr` and `dz = 0.3 m`, this would impose `dt < 0.15 yr`, far smaller than the physically meaningful timestep. The implicit scheme is **unconditionally stable**: no diffusion CFL constraint.

**Boundary conditions:**
- Surface (`z = Z`): Dirichlet — `C_ice` fixed at atmospheric equilibrium `x_atm · P_diss · S_gas`
- Bed (`z = 0`): Neumann flux — prescribed basal gas flux `F_bed` [mol m⁻² yr⁻¹]

### F — Gas loss: exponential decay
Near the bed, clathrate-hosted gas is lost (dissociation). Only `C_clath = C_bulk − C_ice` decays; the lattice component is unaffected:
```
C_clath^{n+1} = C_clath^n · exp(−λ(z) · dt)
```
The rate profile:
```
λ(z) = λ_max · exp(−z / z_char)    for z < z_gl_limit
λ(z) = 0                            for z ≥ z_gl_limit
```

**TODO: document the calibration source for λ_max, z_char, z_gl_limit.**

---

## Repository structure

```
GLACE/
├── params.py               # ALL parameters (gas, site, numerical) — edit here
├── glace/
│   ├── __init__.py
│   ├── physics.py          # velocity_lliboutry(), gas_loss_profile()
│   ├── transport.py        # advect_upwind(), advect_tvd() [TODO]
│   ├── core.py             # partition(), run_simulation() — main time loop
│   ├── figures.py          # plot_profiles(), plot_snapshots(), plot_attribution()
│   └── main.py             # build_params(), entry point
├── notebooks/
│   └── exploration.ipynb   # interactive exploration (Jupyter)
├── data/                   # Camp Century ice core data
├── main_GLACE.m            # MATLAB reference (archived)
├── main_CampCentury.m      # MATLAB reference — full Camp Century (archived)
└── archive/                # Old versions (not active)
```

---

## Adapting to another gas or site

All parameters are in `params.py`, grouped by what to change:

- **Gas-dependent:** `P_diss`, `S_Ar` (rename to `S_gas`), `D_m2s`, `X_atm`
- **Site-dependent:** `H_ice`, `b_acc`, `p_llib`, `TAC`, `flux_bottom`, `gas_loss`, `z_char`, `z_gl_limit`, `Z`
- **Numerical:** `nz`, `T`, `dt`

Create a new `params_SITE_GAS.py` and pass it to `build_params()`:
```python
import params_GRIP_Ne as p
sim_params = build_params(p)
```

---

## Running the model

```bash
# From the MODEL/ directory:
python -m glace.main              # run with Camp Century defaults, saves figures/

# Or interactively:
jupyter notebook notebooks/exploration.ipynb
```

---

## Velocity profile

Lliboutry (1979) profile for a frozen-bed ice sheet:
```
w(ζ) = −b_acc · [(p+2)ζ − (p+2)ζ^(p+1) + ζ^(p+2)],   ζ = z / H_ice
```
- `w(0) = 0` — frozen bed, no basal sliding
- `w(H) = −b_acc` — surface sinks at the accumulation rate
- `p` controls the shape of the profile (p = 3 for Camp Century)

---

## References

- Lliboutry, L. (1979). Local friction laws for glaciers: a critical review and new openings. *Journal of Glaciology*, 23(89), 67–95.
- Dansgaard, W., & Johnsen, S. J. (1969). A flow model and a time scale for the ice core from Camp Century, Greenland. *Journal of Glaciology*, 8(53), 215–223.
- TODO: add gas-loss calibration reference.

---

## Author

Charlotte Maistriau — ULB Sandra group  
Contact: charlotte.maistriau@ulb.be
