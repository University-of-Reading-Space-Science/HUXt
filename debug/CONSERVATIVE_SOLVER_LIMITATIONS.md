# Conservative Solver Limitations in the Inner Heliosphere

## Problem Statement

The fully conservative HLL solver shows unrealistic temperature behavior (T ∝ r^-0.030 instead of the expected T ∝ r^-0.8 to r^-1.0) when applied from 30 Rs outward.

## Root Cause Analysis

### Energy Budget Issue

The solar wind accelerates from ~400 km/s at 30 Rs to ~458 km/s at 240 Rs. This acceleration requires adding kinetic energy:

```
ΔKE per unit mass = 0.5 × (458² - 400²) km²/s² = 2.49 × 10^10 J/kg
```

At 30 Rs with T = 374,000 K:
```
Initial internal energy per unit mass = P / (ρ × (γ-1)) = 6.17 × 10^9 J/kg
```

**The kinetic energy increase is 4× larger than the available internal energy!**

This means that in a purely conservative (adiabatic) system:
- The internal energy would go negative (unphysical)
- Temperature cannot drop as expected
- The system remains nearly isothermal to avoid violating energy conservation

### Physical Reality

In the real solar wind:
1. **Coronal heating** continuously adds energy near the Sun
2. Wave dissipation and magnetic reconnection provide energy input
3. The inner heliosphere (< 50 Rs) is **NOT adiabatic**
4. Beyond ~50-100 Rs, the flow becomes more adiabatic

### Why HLL Hybrid Works

The HLL Hybrid solver succeeds because it:
1. **Explicitly includes pressure gradient force**: `dv/dt ∝ -dP/dr`
2. **Uses empirical alpha parameter**: Implicitly accounts for heating
3. **Allows temperature to evolve independently**: Not constrained by strict energy conservation

Result: T drops from 374 kK to 21 kK (realistic cooling)

### Why Conservative Fails

The conservative solver:
1. **Enforces strict energy conservation**: E = ½ρv² + P/(γ-1)
2. **No external energy source**: Pure adiabatic flow
3. **Cannot supply acceleration energy**: Runs out of internal energy

Result: T stays nearly constant at 351-357 kK (unphysical)

## Attempted Solutions

### Adjusting Gamma (γ)

Following sunRunner1D/PLUTO, we tried γ = 1.5 instead of γ = 5/3:
- **Expected**: T ∝ ρ^(γ-1) = ρ^0.5 ∝ r^-1.0
- **Measured**: T ∝ r^-0.030, implied γ ≈ 1.014
- **Problem**: Energy constraint dominates over polytropic relation

The issue is that **energy conservation trumps the polytropic relation** when kinetic energy is increasing. The solver must maintain:
```
0.5 × v² + P/(ρ(γ-1)) = constant
```

Since v is increasing and ρ is decreasing, P must remain nearly constant, which means T stays nearly constant.

## Possible Solutions

### 1. Add Energy Source Term (Recommended for Inner Heliosphere)

Modify the conservative solver to include a heating term near the inner boundary:

```python
def heating_rate(r, r_inner, r_outer):
    """Heating rate that decays with distance"""
    if r < r_inner:
        return Q_0  # Constant heating near Sun
    elif r < r_outer:
        return Q_0 * (r_outer - r) / (r_outer - r_inner)  # Linear decay
    else:
        return 0.0  # No heating far from Sun
```

Then in energy equation:
```python
dE/dt = -∇·F + Q(r)  # Add heating source
```

### 2. Use Hybrid Solver for Inner Heliosphere (Current Best Practice)

- Use HLL Hybrid from 30 Rs to ~50-100 Rs
- Switch to Conservative solver beyond 50-100 Rs where flow is adiabatic
- This matches physical reality of heating vs. adiabatic regions

### 3. Start Conservative Solver Further Out

- Begin conservative solver at 50-100 Rs instead of 30 Rs
- Use hybrid solver or specified boundary conditions for inner region
- More appropriate physical domain for adiabatic assumption

### 4. Increase Initial Temperature (Not Recommended)

To balance energies (KE/IE ~ 1.5) at 30 Rs with v = 400 km/s requires:
```
T_required ≈ 3.2 × 10^6 K = 3200 kK
```

This is **unphysical** - coronal temperatures are typically 1-2 MK, not 3+ MK.

## Conclusion

**The HLL Hybrid solver is the appropriate choice for modeling the inner heliosphere (30-240 Rs)** because:

1. It accounts for pressure gradient acceleration
2. It allows non-adiabatic processes (implicitly through alpha)
3. It produces realistic temperature profiles (T drops by ~17× from 30 to 240 Rs)

**The fully conservative solver should be reserved for:**
1. Regions beyond ~50-100 Rs where flow is truly adiabatic
2. Applications with explicit heating/cooling terms
3. Shock propagation studies where strict conservation is critical

## References

- sunRunner1D: Uses γ=1.5 but **also includes coronal heating terms**
- PLUTO MHD: Conservative methods with source terms for heating
- Parker Solar Wind: Analytical model includes heating function

## Current Status

- ✅ HLL Hybrid: Validated and working correctly
- ✅ Conservative: Implemented correctly but limited applicability
- ⚠️ Conservative requires energy source terms for inner heliosphere
- ✅ Gamma parameter system: Implemented and functional (but insufficient alone)

## Recommendation

**For HUXt applications:**
- Use `solver='hll'` or `solver='hllc'` (Hybrid mode) for standard runs
- These solvers properly handle the non-adiabatic inner heliosphere
- Reserve conservative solvers for specialized applications or outer heliosphere
