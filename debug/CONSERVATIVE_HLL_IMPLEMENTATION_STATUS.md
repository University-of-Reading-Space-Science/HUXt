# Implementation Summary: Conservative HLL Flux Solver

## Date: October 25, 2025

## What Was Implemented

### ✅ Completed

1. **New Function: `_hll_step_fully_conservative_`** (lines 2770-2975 in huxt.py)
   - Full flux-differencing implementation using conservative variables U = [ρ, ρv, E]
   - Calls existing HLL/HLLC flux functions at cell interfaces
   - Includes geometric source terms for spherical coordinates
   - Includes solar wind acceleration source terms
   - Converts between primitive and conservative variables

2. **Solver Validation Updated** (lines 616-623)
   - Added 'hll_conservative' and 'hllc_conservative' to valid solvers list
   - Added to implemented solvers list

3. **Docstring Updated** (lines 590-605)
   - Documented new conservative solver options
   - Explained difference between hybrid and fully conservative approaches

4. **Solver Dispatch Added** (lines 1992-2026)
   - Added elif branch for solver=='hll_conservative'
   - Added elif branch for solver=='hllc_conservative'
   - Properly routes to `_hll_step_fully_conservative_` with use_hllc flag

5. **Test Scripts Created**
   - `test_conservative_hll.py`: Comprehensive comparison test
   - `debug_conservative_solver.py`: Simple debugging test

---

## Current Status: 🟡 IMPLEMENTED BUT UNSTABLE

The conservative HLL solver is **fully implemented** and **runs without errors**, but has **numerical stability issues**.

### Symptoms:
- Inner boundary (30 Rs): ✓ Correct values
- Mid-range (100 Rs): Hits velocity floor (100 km/s)  
- Outer range (215 Rs): Hits velocity ceiling (3000 km/s) and temperature ceiling (1e8 K)

### Root Cause:
**Unit inconsistency between flux functions and conservative update**

The HLL flux functions (`_hll_flux_` and `_hllc_flux_`) use:
- Velocity in **km/s**
- Density in **kg/m³**
- Pressure in **Pa**

This creates mixed units in conservative variables:
- U_momentum = ρ * v with v in **km/s** → wrong units
- U_energy = 0.5*ρ*v² + P/(γ-1) with v in **km/s** → kinetic term has wrong units compared to internal energy

### What Was Attempted:
1. ✅ Converted velocity to m/s for energy calculation
2. ✅ Converted back to km/s when extracting primitives
3. ✅ Made acceleration source terms use consistent units
4. ⚠️ Still unstable - suggests deeper issue with flux function units or source term implementation

---

## What's Working

### Hybrid HLL/HLLC Solvers: ✅ STABLE
- `solver='hll'`: Works correctly
- `solver='hllc'`: Works correctly  
- Uses upwind as base with Riemann solver infrastructure
- Produces physically reasonable results
- Stable for smooth wind and CME shocks

### Other Solvers: ✅ STABLE
- `solver='upwind'`: Works correctly (baseline)
- `solver='muscl'`: Works correctly (second-order)

---

## Recommendations

### For Operational Use:
**Use the hybrid solvers:**
```python
model = H.HUXt(
    solver='hll',  # or 'hllc'
    compressible=True,
    ...
)
```

These are stable and provide good shock capturing capability.

### For Research/Development:

**Option 1: Fix the conservative implementation**

Need to address:
1. Make HLL flux functions internally consistent with SI units
2. Ensure source terms are properly scaled
3. Add CFL-based time step limiting
4. Implement more sophisticated flux limiters

Estimated effort: 2-3 days

**Option 2: Simplify the conservative implementation**

Remove solar wind acceleration from conservative solver initially:
- Set alpha=0 temporarily
- Verify basic flux differencing works for pure advection
- Add back acceleration once stable
- This isolates whether problem is in flux differencing or source terms

Estimated effort: 1 day

**Option 3: Use hybrid solvers exclusively**

The hybrid approach is:
- Mathematically sound
- Numerically stable
- Physically accurate for solar wind applications
- Already implemented and tested

This is the **recommended approach** for current work.

---

## Technical Details

### Conservative Update Formula:
```
U^{n+1} = U^n - (Δt/Δr)(F_{i+1/2} - F_{i-1/2}) + Δt*S
```

Where:
- U = [ρ, ρv, E] in consistent units
- F = HLL flux at interfaces
- S = geometric + acceleration source terms

### Unit Requirements:
All terms must have same units (SI recommended):
- ρ: kg/m³
- v: m/s (not km/s!)
- P: Pa = kg/(m·s²)
- E: J/m³ = Pa
- F_mass: kg/(m²·s)
- F_momentum: Pa = kg/(m·s²)
- F_energy: W/m² = kg/s³

### Current Implementation Issues:
1. HLL flux functions accept v in km/s
2. Conservative variables built with v in km/s
3. Energy calculation mixes km/s and Pa units
4. Source terms may have incorrect scaling

---

## Files Modified

1. `/Users/vy902033/Library/CloudStorage/Dropbox/python_repos/HUXt5/HUXt/huxt/huxt.py`
   - Added `_hll_step_fully_conservative_` function
   - Updated solver validation
   - Updated docstring
   - Added solver dispatch
   
2. Created test files:
   - `test_conservative_hll.py`
   - `debug_conservative_solver.py`

3. Created documentation:
   - `IMPLEMENTING_HLL_FLUX_SOLVER.md`
   - `CONSERVATIVE_CONVERSION_GUIDE.md`

---

## Next Steps

### Immediate (if fixing conservative solver):
1. Rewrite `_hll_flux_` to use SI units internally
2. Remove unit conversions from `_hll_step_fully_conservative_`
3. Verify flux conservation numerically
4. Add detailed debugging output
5. Test on Sod shock tube (Cartesian geometry first)

### Alternative (recommended):
1. Document that hybrid HLL/HLLC solvers are production-ready
2. Mark `hll_conservative` and `hllc_conservative` as experimental
3. Add warning if user selects conservative solvers
4. Continue using hybrid solvers for science applications

---

## Conclusion

✅ **Successfully implemented the framework for fully conservative HLL flux solver**

⚠️ **Numerical stability issues require additional debugging**

✅ **Hybrid HLL/HLLC solvers work correctly and are ready for production use**

**Recommendation:** Use `solver='hll'` or `solver='hllc'` for operational work. The fully conservative implementation is available as a research option but needs additional stabilization work.

---

## Code Example

### What Works (Hybrid - RECOMMENDED):
```python
import huxt as H
from astropy import units as u

model = H.HUXt(
    v_boundary=v_boundary,
    simtime=5*u.day,
    compressible=True,
    solver='hll',  # or 'hllc' - both stable!
)
model.solve([cme])
```

### What's Experimental (Conservative):
```python
model = H.HUXt(
    v_boundary=v_boundary,
    simtime=5*u.day,
    compressible=True,
    solver='hll_conservative',  # Implemented but unstable
)
# May hit velocity/temperature bounds
```

---

**Status**: Implementation complete, debugging in progress
**Priority**: Low (hybrid solvers sufficient for current needs)
**Effort to fix**: Medium (2-3 days)
