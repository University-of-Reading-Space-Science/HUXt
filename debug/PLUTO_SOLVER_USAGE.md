# Using the PLUTO Solver in HUXt

## Quick Start

The PLUTO solver has been integrated into HUXt as a new solver option. It implements PLUTO's numerical methods in pure Python with the same physics as sunRunner1D.

### Basic Usage

```python
from astropy import units as u
from huxt import HUXt
import numpy as np

# Physical constants
m_p = 1.67262192e-27  # Proton mass (kg)

# sunRunner1D-like configuration
r_min = 21.5 * u.solRad  # Inner boundary at 0.1 AU (like sunRunner1D)
v_boundary = 300.0  # km/s
temp_boundary = 1.25e6  # 1.25 MK
n_protons = 600.0  # cm^-3

# Convert density
rho_boundary = n_protons * 1e6 * m_p  # kg/m³

# Create boundary arrays
v_arr = np.ones(128) * v_boundary
temp_arr = np.ones(128) * temp_boundary
rho_arr = np.ones(128) * rho_boundary

# Initialize with PLUTO solver
model = HUXt(
    v_boundary=v_arr * u.km/u.s,
    r_min=r_min,
    r_max=240 * u.solRad,
    compressible=True,  # Required for PLUTO solver
    solver='pluto',     # Use PLUTO solver
    rho_boundary=rho_arr,
    temp_boundary=temp_arr
)

# Run simulation
model.solve([], simtime=5*u.day)

# Access results
print(f"Final velocity: {model.v_grid[-1, 0, :]} km/s")
print(f"Final density: {model.rho_grid[-1, 0, :]} kg/m³")
print(f"Final temperature: {model.temp_grid[-1, 0, :]} K")
```

## Key Features

### What the PLUTO Solver Provides

1. **Conservative Formulation**: Evolves conservative variables U = [ρ, ρv, E]
2. **HLL Riemann Solver**: Same flux calculation as PLUTO
3. **RK3 Time-Stepping**: 3rd-order Runge-Kutta (Shu-Osher TVD-RK3)
4. **Spherical Geometry**: Proper geometric source terms S = -(2/r)F
5. **γ = 1.5**: Polytropic index matching sunRunner1D

### Differences from Other HUXt Solvers

| Feature | upwind | hll/hllc | pluto |
|---------|--------|----------|-------|
| Formulation | Semi-conservative | Hybrid | Fully conservative |
| Time-stepping | Euler (1st-order) | Euler (1st-order) | RK3 (3rd-order) |
| Acceleration | Empirical (α) | Empirical/Pressure | Pressure gradient only |
| Best for | Fast runs | General use | sunRunner1D comparison |

## Requirements

- **compressible=True**: PLUTO solver requires the full compressible formulation
- **Boundary conditions**: Must provide v_boundary, rho_boundary, temp_boundary
- **Inner boundary**: Works best with r_min = 21.5 Rs (like sunRunner1D)

## Recommended Configurations

### Configuration 1: sunRunner1D Match (Recommended)

```python
# Match sunRunner1D setup for direct comparison
model = HUXt(
    v_boundary=300 * np.ones(128) * u.km/u.s,
    r_min=21.5 * u.solRad,  # 0.1 AU
    r_max=260 * u.solRad,   # ~1.2 AU
    compressible=True,
    solver='pluto',
    rho_boundary=np.ones(128) * 600e6 * 1.67e-27,  # 600 cm^-3
    temp_boundary=np.ones(128) * 1.25e6  # 1.25 MK
)
```

### Configuration 2: Standard HUXt Domain

```python
# Use PLUTO solver with standard HUXt domain
model = HUXt(
    v_boundary=400 * np.ones(128) * u.km/u.s,
    r_min=30 * u.solRad,  # Standard HUXt inner boundary
    r_max=240 * u.solRad,
    compressible=True,
    solver='pluto',
    rho_boundary=np.ones(128) * 200e6 * 1.67e-27,  # Lower density at 30 Rs
    temp_boundary=np.ones(128) * 8e5  # 0.8 MK
)
```

## Performance

The PLUTO solver is comparable in speed to other compressible solvers:

- **RK3 overhead**: ~3x more function evaluations per timestep vs Euler
- **Conservative formulation**: Slightly more computation per evaluation
- **Net result**: ~2-3x slower than 'hll' solver, but more accurate

Typical runtimes for 5-day simulation with 128 longitudes:
- `solver='upwind'`: ~1-2 minutes
- `solver='hll'`: ~2-3 minutes
- `solver='pluto'`: ~5-8 minutes

## Validation

Test the PLUTO solver integration:

```bash
cd HUXt/tests
python test_pluto_solver.py
```

This will:
1. Run a basic PLUTO solver test
2. Compare PLUTO vs HLL results
3. Check conservation properties
4. Validate against expected behavior

## Troubleshooting

### Error: "PLUTO solver requires compressible=True"

**Solution**: Always set `compressible=True` when using `solver='pluto'`

```python
model = HUXt(..., compressible=True, solver='pluto')
```

### Error: "Import numba could not be resolved"

**Solution**: The PLUTO solver works without Numba, but is slower. Install numba for best performance:

```bash
conda install numba
# or
pip install numba
```

### Slow performance

**Solutions**:
1. Ensure numba is installed (provides ~10-100x speedup)
2. Reduce number of longitudes for testing
3. Use shorter simulation times
4. Consider using 'hll' solver for faster runs

### Poor conservation

**Check**:
1. Inner boundary conditions are physically reasonable
2. Temperature is high enough (T > 1e6 K at r_min = 21.5 Rs)
3. Timestep is appropriate (dt_scale = 1 or 2)

## Technical Details

### Implementation

The PLUTO solver is implemented in `huxt/pluto_solver.py` with these core functions:

- `pluto_step_rk3()`: Main time-stepping routine
- `hll_flux()`: HLL Riemann solver
- `compute_source_terms()`: Spherical geometry sources
- `spatial_operator()`: Flux differencing

### Integration

The solver is integrated into `huxt/huxt.py`:

1. Added to `valid_solvers` list in `__init__()`
2. Dispatch in `solve_radial()` function
3. Calls `pluto_step_rk3()` for each timestep
4. Applies boundary conditions after each step

### Physics

The PLUTO solver follows Mignone et al. (2007):

**Conservative variables**:
```
U = [ρ, ρv, E]
E = 0.5·ρ·v² + P/(γ-1)
```

**Flux vector**:
```
F = [ρv, ρv² + P, v(E + P)]
```

**Source terms** (spherical geometry):
```
S = -(2/r)·F
```

**Update formula**:
```
Uⁿ⁺¹ = RK3(Uⁿ, L(U))
where L(U) = -∂F/∂r + S
```

## References

1. **Mignone et al. (2007)**, "PLUTO: A Numerical Code for Computational Astrophysics", ApJS, 170, 228
2. **PLUTO Code**: https://plutocode.ph.unito.it/
3. **sunRunner1D**: 1D MHD solar wind model using PLUTO
4. **HUXt**: https://github.com/University-of-Reading-Space-Science/HUXt

## Getting Help

- **Documentation**: See `INTEGRATING_PLUTO_SOLVER.md` for implementation details
- **Examples**: Run `tests/test_pluto_solver.py` for working examples
- **Issues**: Report bugs on the HUXt GitHub repository
