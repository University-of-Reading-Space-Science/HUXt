# Implementing a Fully Conservative HLL Flux Solver in HUXt

## What You Currently Have

HUXt already has the **HLL infrastructure** in place:
- ✓ Wave speed estimation: `_estimate_wave_speeds_`
- ✓ HLL flux function: `_hll_flux_`
- ✓ HLLC flux function: `_hllc_flux_`

But the current `_hll_step_compressible_` is a **hybrid approach** - it uses standard upwind as the base solver, not the HLL flux functions.

---

## What "Fully Conservative HLL" Means

A **fully conservative HLL solver** uses:

1. **Conservative variables**: U = [ρ, ρv, E]
2. **Flux differencing**: ΔU/Δt = -(F_{i+1/2} - F_{i-1/2})/Δr + Source
3. **HLL flux at interfaces**: F_{i+1/2} = _hll_flux_(U_i, U_{i+1})
4. **Geometric source terms**: For spherical coordinates

---

## Step-by-Step Implementation

### **Step 1: Create the Conservative HLL Step Function**

```python
@jit(nopython=True)
def _hll_step_fully_conservative_(v_up, v_dn, rho_up, rho_dn, temp_up, temp_dn,
                                   dtdr, alpha, r_accel, rrel, r_boundary):
    """
    Fully conservative HLL solver using conservative variables and flux differencing.
    
    This is the "textbook" implementation of HLL for comparison purposes.
    """
    gamma = 5.0 / 3.0
    k_B = 1.38064852e-23
    m_p = 1.67262192e-27
    nr = len(v_up)
    
    # Compute radial grid
    r_dn = rrel[:-1] * 695700.0 + r_boundary
    r_up = rrel[1:] * 695700.0 + r_boundary
    dr = r_up - r_dn
    dt = dtdr * dr
    r_center = 0.5 * (r_dn + r_up)
    
    # Initialize output arrays
    v_up_next = np.zeros(nr)
    rho_up_next = np.zeros(nr)
    temp_up_next = np.zeros(nr)
    
    # Step 1: Convert primitives to conservatives
    U_mass = rho_up.copy()
    U_momentum = rho_up * v_up
    p_up = (rho_up / m_p) * k_B * temp_up
    U_energy = 0.5 * rho_up * v_up**2 + p_up / (gamma - 1.0)
    
    # Also need downwind states for flux calculation
    U_mass_dn = rho_dn.copy()
    U_momentum_dn = rho_dn * v_dn
    p_dn = (rho_dn / m_p) * k_B * temp_dn
    U_energy_dn = 0.5 * rho_dn * v_dn**2 + p_dn / (gamma - 1.0)
    
    # Step 2: Loop over cells and compute fluxes at interfaces
    for i in range(nr):
        # Interface i+1/2: between cell i and cell i+1
        # Left state = cell i (upwind), Right state = cell i+1 (downwind)
        
        if i < nr - 1:
            # Interior interface
            rho_L = rho_up[i]
            v_L = v_up[i]
            p_L = p_up[i]
            
            rho_R = rho_up[i+1]
            v_R = v_up[i+1]
            p_R = p_up[i+1]
        else:
            # Right boundary - use extrapolation
            rho_L = rho_up[i]
            v_L = v_up[i]
            p_L = p_up[i]
            
            rho_R = rho_L
            v_R = v_L
            p_R = p_L
        
        # Compute HLL flux at this interface
        F_mass_R, F_mom_R, F_energy_R = _hll_flux_(rho_L, v_L, p_L, 
                                                     rho_R, v_R, p_R, gamma)
        
        # Interface i-1/2: between cell i-1 and cell i
        if i > 0:
            rho_L = rho_up[i-1]
            v_L = v_up[i-1]
            p_L = p_up[i-1]
            
            rho_R = rho_up[i]
            v_R = v_up[i]
            p_R = p_up[i]
        else:
            # Left boundary - use downwind state
            rho_L = rho_dn[i]
            v_L = v_dn[i]
            p_L = p_dn[i]
            
            rho_R = rho_up[i]
            v_R = v_up[i]
            p_R = p_up[i]
        
        F_mass_L, F_mom_L, F_energy_L = _hll_flux_(rho_L, v_L, p_L,
                                                     rho_R, v_R, p_R, gamma)
        
        # Step 3: Compute geometric source terms
        # For spherical coordinates: S = -(2/r) * F
        geom_source_mass = -(2.0 / r_center[i]) * 0.5 * (F_mass_L + F_mass_R)
        geom_source_mom = -(2.0 / r_center[i]) * 0.5 * (F_mom_L + F_mom_R)
        geom_source_energy = -(2.0 / r_center[i]) * 0.5 * (F_energy_L + F_energy_R)
        
        # Step 4: Add solar wind acceleration source term
        accel_arg = -rrel[i] / r_accel
        if i < nr - 1:
            accel_arg_p = -rrel[i+1] / r_accel
        else:
            accel_arg_p = accel_arg
        
        v_source = v_dn[i] / (1.0 + alpha * (1.0 - np.exp(accel_arg)))
        v_diff = alpha * v_source * (np.exp(accel_arg) - np.exp(accel_arg_p))
        
        # Acceleration adds momentum: S_momentum = ρ * a
        accel_source_mom = rho_up[i] * v_diff / dt[i]
        # Acceleration adds energy: S_energy = ρ * v * a
        accel_source_energy = rho_up[i] * v_up[i] * v_diff / dt[i]
        
        # Step 5: Flux differencing update
        # U^{n+1} = U^n - (Δt/Δr) * (F_R - F_L) + Δt * S
        U_mass_new = U_mass[i] - (dt[i] / dr[i]) * (F_mass_R - F_mass_L) + dt[i] * geom_source_mass
        U_momentum_new = U_momentum[i] - (dt[i] / dr[i]) * (F_mom_R - F_mom_L) + dt[i] * (geom_source_mom + accel_source_mom)
        U_energy_new = U_energy[i] - (dt[i] / dr[i]) * (F_energy_R - F_energy_R) + dt[i] * (geom_source_energy + accel_source_energy)
        
        # Step 6: Convert back to primitives
        rho_up_next[i] = U_mass_new
        v_up_next[i] = U_momentum_new / U_mass_new
        p_new = (gamma - 1.0) * (U_energy_new - 0.5 * U_mass_new * v_up_next[i]**2)
        temp_up_next[i] = p_new * m_p / (rho_up_next[i] * k_B)
        
        # Step 7: Apply physical bounds
        if v_up_next[i] < 100.0:
            v_up_next[i] = 100.0
        if v_up_next[i] > 3000.0:
            v_up_next[i] = 3000.0
        if rho_up_next[i] > 1e-17:
            rho_up_next[i] = 1e-17
        if rho_up_next[i] < 1e-30:
            rho_up_next[i] = 1e-30
        if temp_up_next[i] > 1e8:
            temp_up_next[i] = 1e8
        if temp_up_next[i] < 1e3:
            temp_up_next[i] = 1e3
    
    return v_up_next, rho_up_next, temp_up_next
```

---

### **Step 2: Add to Solver Validation**

Update the solver lists (around line 613):

```python
valid = ['upwind', 'muscl', 'hll', 'hllc', 'hll_conservative', 'hllc_conservative', 'tvd', 'weno']
implemented = ['upwind', 'muscl', 'hll', 'hllc', 'hll_conservative', 'hllc_conservative']
```

---

### **Step 3: Add Dispatch in solve_radial**

Add new branches around line 1950:

```python
elif self.solver == 'hll_conservative':
    if self.frame == 'sidereal':
        v_grid[nv, :] = _hll_step_fully_conservative_(
            v_grid[nv - 1, :], v_grid_relative[nv - 1, :],
            rho_grid[nv - 1, :], rho_grid_relative[nv - 1, :],
            temp_grid[nv - 1, :], temp_grid_relative[nv - 1, :],
            dtdr, alpha, r_accel[0].value, rrel[:], self.r[0].value)
    else:
        v_grid[nv, :] = _hll_step_fully_conservative_(
            v_grid[nv - 1, :], v_grid_lon[0, :],
            rho_grid[nv - 1, :], rho_grid_lon[0, :],
            temp_grid[nv - 1, :], temp_grid_lon[0, :],
            dtdr, alpha, r_accel[0].value, rrel[:], self.r[0].value)

elif self.solver == 'hllc_conservative':
    # Similar to above but with use_hllc=True flag
    # (Would need to modify function to accept this flag)
    pass
```

---

## Key Differences from Current Implementation

### **Current Hybrid Approach:**
```python
# Uses upwind as base
v_up_next = v_up - dtdr * v_up * (v_up - v_dn)  # Upwind advection
# + acceleration
# + compression effects
# (HLL flux functions exist but aren't used in the step)
```

### **Fully Conservative Approach:**
```python
# Build conservative variables
U = [ρ, ρv, E]

# Compute fluxes at interfaces using HLL
F_{i+1/2} = _hll_flux_(U_i, U_{i+1})
F_{i-1/2} = _hll_flux_(U_{i-1}, U_i)

# Update via flux differencing
U_new = U - (Δt/Δr) * (F_{i+1/2} - F_{i-1/2}) + Δt * Source

# Extract primitives
ρ, v, T = extract_primitives(U_new)
```

---

## Critical Implementation Details

### **1. Unit Consistency**

Your HLL flux functions mix units:
- Density: kg/m³
- Velocity: km/s
- Pressure: Pa
- Energy: Mixed!

**Fix needed:**
```python
# In _hll_flux_, convert everything to SI:
v_L_SI = v_L * 1000.0  # km/s → m/s
v_R_SI = v_R * 1000.0

U_L_energy = 0.5 * rho_L * v_L_SI**2 + p_L / (gamma - 1.0)  # Now in J/m³

# OR convert at boundaries when calling _hll_flux_
```

### **2. Geometric Source Terms**

For spherical coordinates (1D radial):
```python
∂U/∂t + ∂F/∂r = S_geom + S_accel

# Geometric source (from ∇·(A*F) with A = r²):
S_geom = -(2/r) * F

# Must be applied AFTER computing fluxes
```

### **3. Boundary Conditions**

- **Left (inner) boundary**: Use downwind values as left state
- **Right (outer) boundary**: Extrapolate or use last cell values

### **4. Pressure Recovery**

After flux update, recover pressure:
```python
p_new = (γ-1) * (U_energy - 0.5 * ρ * v²)

# Check for negative pressure!
if p_new < 0:
    # Apply floor or use different limiter
    p_new = p_min
    # Recompute energy to maintain consistency
    U_energy = 0.5 * ρ * v² + p_new / (γ-1)
```

---

## Testing Strategy

### **1. Sod Shock Tube** (already implemented in `sod_shock_tube_reference.py`)
```python
# Known exact solution - should match well
```

### **2. Smooth Solar Wind**
```python
# No shocks - should match upwind closely
model = HUXt(solver='hll_conservative')
```

### **3. Strong CME Shock**
```python
# Test shock capturing ability
model.solve([ConeCME(..., v=2000)])
# Compare density/temperature compression ratios
```

### **4. Conservation Check**
```python
# Total mass should be conserved:
total_mass_before = np.sum(rho * volume)
total_mass_after = np.sum(rho_new * volume)
assert np.abs(total_mass_after - total_mass_before) < tolerance
```

---

## Expected Behavior Differences

| Scenario | Upwind | Hybrid HLL | Conservative HLL |
|----------|--------|------------|------------------|
| **Smooth wind** | Good | Good | Good |
| **Weak shocks** | OK | OK | Better |
| **Strong shocks** | Oscillations | OK | Best |
| **Conservation** | ~Good | ~Good | Exact |
| **Speed** | Fast | Fast | Slower |
| **Complexity** | Low | Medium | High |

---

## Debugging Tips

### **1. Check Flux Units**
```python
# All fluxes should have consistent dimensions
print(f"Mass flux: {F_mass}")  # Should be ~ ρv
print(f"Mom flux: {F_mom}")    # Should be ~ ρv² + P
print(f"Energy flux: {F_energy}")  # Should be ~ v*(E+P)
```

### **2. Monitor for NaN/Inf**
```python
if np.any(np.isnan(U_mass_new)) or np.any(np.isinf(U_mass_new)):
    print(f"NaN detected at step {nv}, cell {i}")
    print(f"F_L = {F_mass_L}, F_R = {F_mass_R}")
    print(f"U_old = {U_mass[i]}, dt = {dt[i]}, dr = {dr[i]}")
```

### **3. Check Negative Pressure**
```python
p_new = (gamma - 1.0) * (U_energy_new - 0.5 * U_mass_new * v_new**2)
if p_new < 0:
    print(f"Negative pressure at cell {i}: p = {p_new}")
    print(f"U_energy = {U_energy_new}, kinetic = {0.5 * U_mass_new * v_new**2}")
    # Apply pressure floor and correct energy
```

### **4. Compare Against Hybrid**
```python
# Run same setup with both solvers
model_hybrid = HUXt(solver='hll')
model_conservative = HUXt(solver='hll_conservative')
# Should give similar results for smooth flows
```

---

## Recommended Approach

**For now: Keep the hybrid implementation**

The current hybrid approach works well because:
1. Solar wind is mostly smooth (not full of shocks)
2. Upwind base is stable and well-tested
3. HLL infrastructure exists for future use

**Add fully conservative as research option:**

```python
if self.solver == 'hll_conservative':
    # Use fully conservative method
elif self.solver == 'hll':
    # Use current hybrid (production default)
```

This gives you:
- ✓ Stable production solver (hybrid)
- ✓ Research capability (conservative)
- ✓ Method comparison ability
- ✓ Future extension path

---

## Summary

**To implement fully conservative HLL:**

1. ✓ Create `_hll_step_fully_conservative_` function (shown above)
2. ✓ Fix unit consistency in energy calculations
3. ✓ Add geometric source terms properly
4. ✓ Handle boundary conditions carefully
5. ✓ Add pressure floor for stability
6. ✓ Update solver validation and dispatch
7. ✓ Test against Sod shock tube
8. ✓ Compare with hybrid version
9. ✓ Document performance differences

**The key insight:** Your HLL flux functions are correct - you just need to wire them into a flux-differencing framework instead of using them with an upwind base.
