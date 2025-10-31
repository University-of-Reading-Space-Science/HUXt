# Conservative Solver Implementation Issues

## Summary of Findings

After attempting to match sunRunner1D's conservative approach with γ=1.5 and proper initial conditions (T=1 MK at 21.5 Rs, KE/IE=2.7), we discovered fundamental issues with the conservative solver implementation.

## Issues Identified

### 1. **Non-Conservative Acceleration Terms** ✅ FIXED
The solver was adding explicit acceleration source terms:
```python
accel_source_mom = rho * accel
accel_source_energy = rho * v * accel
```

These violated conservation by adding energy from outside the system. **Fixed by removing these terms.**

### 2. **Spherical Geometry Not Properly Handled** ⚠️ CRITICAL
The conservative solver treats the problem as Cartesian 1D, but it's actually **spherical 1D**.

**Issue**: Mass flux ρv is NOT conserved in spherical coords - it should be **r²ρv = constant**.

**Evidence**:
- Mass flux ratio (240 Rs / 21.5 Rs): 0.0031 (should be 1.0)
- This is approximately (21.5/240)² = 0.008, suggesting missing r² factor

**Required fix**: Implement area-weighted conservative variables or geometric source terms that properly account for spherical divergence.

### 3. **No Natural Acceleration**
With purely conservative formulation (no artificial acceleration):
- Velocity stays constant: v = 300 km/s everywhere
- Temperature stays constant: T = 1 MK everywhere  
- Energy is perfectly conserved (good!)
- But this doesn't match solar wind physics (bad!)

**Why**: The pressure gradient in the momentum flux is too weak to drive acceleration with these boundary conditions.

## How PLUTO/sunRunner1D Handles This

PLUTO is a **finite-volume code designed for spherical geometry**. It uses:

1. **Geometric source terms**: Properly accounts for spherical divergence in conservation laws
2. **Area-weighted fluxes**: Fluxes are multiplied by interface areas (4πr²)
3. **Volume-weighted updates**: Cell volumes scale as r² (or r²Δr in 1D spherical)

The 1D spherical conservation laws are:
```
∂U/∂t + (1/r²) ∂(r²F)/∂r = S
```

Where the r² factors ensure mass flux r²ρv is conserved.

## Our Current Implementation

HUXt's conservative solver uses **Cartesian 1D** formulation:
```python
U_new = U_old - (Δt/Δr) * (F_R - F_L) + Δt * S
```

This is **incorrect for spherical geometry**. Should be:
```python
U_new = U_old - (Δt/Δr) * (r_R² * F_R - r_L² * F_L) / r_center² + Δt * S
```

## Recommendations

### Option 1: Fix Spherical Geometry (Major Rewrite)
- Implement proper area/volume weighting
- Add geometric source terms for r² factor
- Test mass conservation: verify r²ρv = constant
- **Effort**: High (requires careful implementation and extensive testing)

### Option 2: Use HLL Hybrid Solver (Current Best Practice) ✅ RECOMMENDED
- HLL Hybrid properly handles:
  * Pressure gradient force (explicit in velocity equation)
  * Temperature advection and compression
  * Realistic solar wind profiles
- Results:
  * 21.5 Rs: v=300 km/s, T=1.0 MK  
  * 240 Rs: v=404 km/s, T=18 kK
  * Temperature drops by factor of 55× (realistic!)
- **Effort**: None (already working)

### Option 3: Start Conservative Solver Further Out
- Use Hybrid solver for 21.5-50 Rs (acceleration region)
- Switch to Conservative for >50 Rs (adiabatic region)
- **Effort**: Medium (interface coupling between solvers)

## Conclusion

**The HLL Hybrid solver is the correct choice for HUXt's inner heliosphere modeling.**

The conservative solver implementation has fundamental issues with spherical geometry that prevent it from conserving mass correctly. While fixable, it would require a major rewrite.

The Hybrid solver:
1. ✅ Produces realistic temperature profiles
2. ✅ Handles pressure gradient acceleration properly
3. ✅ Works correctly from 21.5 Rs outward
4. ✅ Validated against multiple test cases
5. ✅ Computationally stable

**Recommendation**: Focus development effort on the Hybrid solver, which is physics-based and working correctly.

## Status

- ✅ Hybrid solver: Production ready
- ⚠️ Conservative solver: Research/development only, requires spherical geometry fix
- ✅ Gamma parameter: Implemented (but not helpful without proper geometry)
- ✅ Initial conditions: Can match sunRunner1D (T=1 MK, v=300 km/s at 21.5 Rs)

## Files

- `huxt/huxt.py`: Main implementation (Hybrid solver recommended)
- `debug/test_sunrunner_setup.py`: Comparison with sunRunner1D parameters
- `debug/check_conservative_energy.py`: Energy conservation analysis
- `debug/CONSERVATIVE_SOLVER_LIMITATIONS.md`: Physics analysis
- `debug/sunRunner1D_method_summary.md`: PLUTO method documentation
