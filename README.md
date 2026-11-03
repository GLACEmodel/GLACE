# GLACE — Gas Lattice And Clathrate Evolution

A 1D Eulerian model for Argon transport in ice sheets, combining advection, lattice diffusion (Fick's law), and thermodynamic partitioning between clathrates and the ice lattice.

Developed as part of a PhD thesis at Université Libre de Bruxelles (ULB).

## Model physics

The model tracks three state variables per depth node:
- `x` — Ar mixing ratio in clathrate gas [mol/mol]
- `C_ice` — Ar dissolved in the ice lattice [mol/m³ ice]
- `C_bulk` — total Ar (clathrate + lattice) [mol/m³ ice]

Each timestep applies operator splitting:
1. **A** — Advect `C_bulk` (upwind scheme)
2. **B** — Thermodynamic partition → `x`, `C_ice`
3. **C** — Diffuse `C_ice` (implicit backward Euler, Fick's law)
4. **D** — Rebuild `C_bulk` from diffusion increment
5. **E** — Re-equilibrate `x`, `C_ice`
6. **F** — Gas loss (exponential decay, clathrate phase only)

## Repository structure

```
GLACE/
├── main_GLACE.m            # Main script — simplified test version
├── main_CampCentury.m      # Main script — full Camp Century application
├── core/                   # Core numerical routines
├── transport/              # Advection schemes (upwind, TVD...)
├── physics/                # Velocity profiles, gas loss
├── figures/                # Plotting functions
├── data/                   # Input data (Camp Century ice core)
└── archive/                # Previous versions (not active)
```

## Application

Applied to the **Camp Century** ice core (Greenland, ~1387 m depth).  
Reference: Dansgaard et al. (1969), Lliboutry (1979).

## Author

Charlotte Maistriau — ULB Sandra group
