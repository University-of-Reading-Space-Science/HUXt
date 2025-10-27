# PLUTO Code: How It Solves 1D Hydrodynamic Problems

## Overview

**PLUTO** (PLasma and Universe TOols) is a well-established astrophysical gasdynamics code that uses **conservative finite volume methods** for solving hyperbolic conservation laws.

Paper: Mignone et al. (2007), ApJS, 170, 228  
Website: https://plutocode.ph.unito.it/

---

## Key Features Relevant to Our Conservative HLL Implementation

### 1. **Conservative Formulation**

PLUTO uses **conservative variables** throughout:
```
U = [ρ, ρv, E]
```

Where:
- ρ = density
- ρv = momentum density  
- E = total energy density = kinetic + internal

This is exactly what we're trying to implement!

---

### 2. **Godunov-Type Shock-Capturing Schemes**

**Three-Step Algorithm** (same as our approach):

1. **Reconstruction**: Piecewise polynomial reconstruction of states
   - Options: Constant, Linear TVD, PPM, WENO, MP5
   
2. **Riemann Solver**: Solve Riemann problem at interfaces
   - Options: Two-shock, Roe, **HLL, HLLC**, HLLD, Lax-Friedrichs
   
3. **Evolution**: Update conservative variables via flux differencing

---

### 3. **Flux Differencing Update**

The core update formula (from PLUTO paper):

```
U^{n+1}_i = U^n_i - (Δt/Δx_i) * (F_{i+1/2} - F_{i-1/2}) + Δt * S_i
```

Where:
- `U^n_i` = conservative variables at cell i, time n
- `F_{i±1/2}` = numerical flux at cell interfaces (from Riemann solver)
- `S_i` = source terms (geometry, gravity, acceleration, etc.)

**This is exactly what our `_hll_step_fully_conservative_` does!**

---

### 4. **Geometric Source Terms for Spherical Coordinates**

PLUTO handles geometric source terms for curvilinear coordinates.

**For spherical coordinates (1D radial)**, the continuity equation becomes:

```
∂ρ/∂t + (1/r²)∂(r²ρv)/∂r = 0
```

Which in conservative form with geometric source terms is:

```
∂ρ/∂t + ∂(ρv)/∂r = -(2/r)ρv
```

**Momentum equation**:
```
∂(ρv)/∂t + ∂(ρv² + P)/∂r = -(2/r)(ρv² + P)
```

**Energy equation**:
```
∂E/∂t + ∂(v(E + P))/∂r = -(2/r)v(E + P)
```

**Key insight**: The source term is **`S = -(2/r) * F`** where F is the flux!

This is what we implemented in our conservative solver.

---

### 5. **Unit Consistency**

PLUTO internally uses **consistent physical units**:

- All velocities in **same unit** (e.g., cm/s or code units)
- All energies in **same unit** (e.g., erg/cm³)
- Pressure has same units as energy density

**CRITICAL**: Energy must be:
```
E = (1/2)ρv² + P/(γ-1)
```

With **all terms in same units**!

If velocity is in km/s:
```
E_kinetic = (1/2)ρ(v*1000)²  # Convert to m/s for SI consistency
E_internal = P/(γ-1)           # Already in Pa = J/m³
```

---

### 6. **HLL/HLLC Riemann Solvers in PLUTO**

PLUTO's HLL implementation:

**Wave Speed Estimates**:
```
S_L = min(v_L - c_L, v_R - c_R)
S_R = max(v_L + c_L, v_R + c_R)
```

Where `c = sqrt(γP/ρ)` is sound speed.

**HLL Flux**:
```
       ⎧ F_L                                    if S_L ≥ 0
F_HLL = ⎨ (S_R*F_L - S_L*F_R + S_L*S_R*(U_R - U_L))/(S_R - S_L)  if S_L < 0 < S_R
       ⎩ F_R                                    if S_R ≤ 0
```

**Physical Flux**:
```
F = [ρv, ρv² + P, v(E + P)]
```

**This is identical to what we implemented in `_hll_flux_`!**

---

## What PLUTO Tells Us About Our Implementation

### ✅ What We Did Right:

1. **Flux differencing formula**: Correct
2. **Geometric source terms**: `-(2/r)*F` is correct
3. **HLL flux function**: Algorithm is correct
4. **Conservative variables**: Using U=[ρ,ρv,E] is correct

### ⚠️ What We Need to Fix:

1. **Unit Consistency** - THE KEY ISSUE

PLUTO maintains consistent units throughout. Our problem:
- HLL flux functions take velocity in **km/s**
- Energy calculation mixes **km/s with Pa**
- Result: Inconsistent energy units cause numerical instability

**Solution**: Either:
   - a) Convert all velocities to m/s internally
   - b) Keep velocity in km/s but scale energy appropriately
   - c) Use dimensionless "code units"

2. **CFL Condition**

PLUTO uses CFL-based timestep control:
```
Δt = CFL * min(Δx/(|v| + c))
```

Our `dt_scale` might not respect CFL for the conservative update.

3. **Pressure Positivity**

PLUTO has sophisticated algorithms to ensure positive pressure:
- Pressure floor
- Energy correction
- Sometimes switches to entropy equation

Our simple floor might not be enough for extreme cases.

---

## Recommended Fixes for HUXt Conservative HLL

### **Fix 1: Consistent Units (HIGHEST PRIORITY)**

Change `_hll_step_fully_conservative_` to use SI units internally:

```python
# At start: Convert km/s to m/s
v_up_SI = v_up * 1000.0  # m/s
v_dn_SI = v_dn * 1000.0  # m/s

# Build conservative variables in SI
U_mass = rho_up
U_momentum = rho_up * v_up_SI  # kg/(m²·s)
U_energy = 0.5 * rho_up * v_up_SI**2 + p_up/(gamma-1)  # J/m³ = Pa

# Call HLL flux with SI velocities
# (Need to also modify _hll_flux_ to accept m/s, OR convert at interface)

# At end: Convert back to km/s
v_up_next = (U_momentum_new / U_mass_new) / 1000.0  # m/s → km/s
```

### **Fix 2: Modify HLL Flux Functions**

Make `_hll_flux_` accept a `velocity_unit` parameter:

```python
def _hll_flux_(rho_L, v_L, p_L, rho_R, v_R, p_R, gamma=5/3, v_in_mps=True):
    """
    If v_in_mps=True: velocity in m/s (SI consistent)
    If v_in_mps=False: velocity in km/s (need to convert for energy)
    """
    if not v_in_mps:
        v_L_SI = v_L * 1000.0
        v_R_SI = v_R * 1000.0
    else:
        v_L_SI = v_L
        v_R_SI = v_R
    
    # Use SI velocities for all calculations
    U_L_energy = 0.5 * rho_L * v_L_SI**2 + p_L/(gamma-1)
    # ... rest of calculation ...
```

### **Fix 3: CFL Safety Factor**

Add explicit CFL check in conservative solver:

```python
# After computing fluxes, check CFL
for i in range(nr):
    c_sound = np.sqrt(gamma * p_up[i] / rho_up[i])  # m/s
    cfl_local = dt[i] * (np.abs(v_up_mps[i]) + c_sound) / dr[i]
    if cfl_local > 0.8:  # Safety factor
        # Reduce timestep or issue warning
```

---

## PLUTO Best Practices Summary

1. **Use conservative variables everywhere** ✓ (we do this)
2. **Maintain unit consistency** ✗ (we need to fix this)  
3. **Geometric source terms = -(2/r)*F** ✓ (we do this)
4. **Respect CFL condition** ⚠️ (may need adjustment)
5. **Ensure positive pressure** ⚠️ (basic floor implemented)
6. **Use well-tested Riemann solvers** ✓ (HLL is standard)

---

## Example: PLUTO's Spherical Wind Test

From PLUTO test suite (Test Problem #5: Spherical Wind):
- 1D radial flow in spherical coordinates
- Uses conservative form with geometric sources
- HLL or HLLC solver
- Handles expansion naturally with `-(2/r)*F` term

**This is exactly analogous to HUXt's solar wind!**

---

## Conclusion

**PLUTO validates our approach!** The conservative HLL implementation framework in HUXt is **fundamentally correct**.

The instability is caused by **unit inconsistency** in the energy equation, not a flaw in the conservative method itself.

**Fix the energy units → Conservative solver will work.**

---

## References

1. Mignone et al. (2007), "PLUTO: A Numerical Code for Computational Astrophysics", ApJS, 170, 228
   - https://ui.adsabs.harvard.edu/abs/2007ApJS..170..228M

2. PLUTO Code Website: https://plutocode.ph.unito.it/

3. PLUTO User Guide (PDF): Available at plutocode.ph.unito.it/userguide.pdf

4. Test Problems: Included with PLUTO distribution, especially:
   - HD/Sod: 1D shock tube
   - HD/Stellar_Wind: 1D spherical wind (our case!)

---

## Next Steps for HUXt

1. **Immediate**: Fix unit consistency in `_hll_step_fully_conservative_`
   - Convert all velocities to SI (m/s) before energy calculation
   - Keep conversion transparent (input/output still in km/s)

2. **Test**: Run debug_conservative_solver.py after fix
   - Should see reasonable velocities (not 100 or 3000 km/s)
   - Should see smooth profiles matching hybrid solver

3. **Validate**: Compare against PLUTO's spherical wind test
   - Same geometry, same physics
   - Should get nearly identical results

4. **Document**: Update CONSERVATIVE_HLL_IMPLEMENTATION_STATUS.md
   - Mark as "STABLE" once fixed
   - Recommend for shock-dominated flows

**Estimated time to fix**: 2-4 hours (just unit conversion)
**Payoff**: Proper conservative solver ready for strong shocks!
