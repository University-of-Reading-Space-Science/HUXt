# Full HLL Implementation Guide for HUXt

## Current vs. Fully Conservative Implementation

### **Current Implementation (Hybrid Approach)**

The current HLL/HLLC solvers in HUXt use a **hybrid approach**:

```python
# Uses upwind advection for density/temperature
rho_advection = - dtdr * v_up * (rho_up - rho_dn)
rho_compression = - rho_up * div_v * dt
rho_up_next = rho_up + rho_advection + rho_compression

# Temperature via adiabatic relation
compression_factor = 1.0 - dt * div_v
temp_up_next = temp_up * (compression_factor ** (gamma - 1.0))
```

**Characteristics:**
- ✓ Simple and stable
- ✓ Works well with existing HUXt structure
- ✓ Includes pressure gradient force (recently added)
- ✗ Not fully conservative
- ✗ Doesn't use HLL flux functions
- ✗ May not capture strong shocks accurately

---

### **Fully Conservative HLL (Proper Finite Volume)**

A textbook HLL implementation would use:

```python
# 1. Work with conservative variables
U_mass = rho
U_momentum = rho * v
U_energy = 0.5 * rho * v² + P/(γ-1)

# 2. Compute HLL fluxes at cell interfaces
for each interface i+1/2:
    F[i+1/2] = HLL_flux(U[i], U[i+1])

# 3. Update via flux differencing
U_new = U_old - (Δt/Δr) * (F[i+1/2] - F[i-1/2]) + Δt * Source

# 4. Extract primitive variables
rho = U_mass
v = U_momentum / U_mass
P = (γ-1) * (U_energy - 0.5 * U_mass * v²)
T = P * m_p / (rho * k_B)
```

**Characteristics:**
- ✓ Fully conservative (guarantees mass/momentum/energy conservation)
- ✓ Properly captures shocks via Rankine-Hugoniot jump conditions
- ✓ Uses HLL flux functions as intended
- ✓ No spurious oscillations at discontinuities
- ✗ More complex to implement
- ✗ Requires reformulation of HUXt's structure
- ✗ Source terms need careful handling

---

## Key Differences

### **1. Mathematical Formulation**

**Current (Primitive Variables):**
```
∂ρ/∂t + v·∂ρ/∂r + ρ·∇·v = 0           [density]
∂v/∂t + v·∂v/∂r = -(1/ρ)·∂P/∂r + g    [momentum]
∂T/∂t + v·∂T/∂r + (γ-1)T·∇·v = 0      [temperature]
```
- Works directly with ρ, v, T
- Uses upwind advection + pressure force
- Three separate equations

**Fully Conservative:**
```
∂U/∂t + ∂F(U)/∂r = S(U)

where:
U = [ρ, ρv, E]                    (conservative variables)
F = [ρv, ρv²+P, v(E+P)]           (fluxes)
S = [0, ρg, ρvg]                  (sources)
```
- Works with conservative variables U
- Single unified system
- Fluxes computed at interfaces

### **2. Flux Calculation**

**Current:**
```python
# Simple upwind: no flux function
rho_advection = - dtdr * v_up * (rho_up - rho_dn)
```

**Fully Conservative:**
```python
# HLL flux at interface
F_hll = HLL_flux(U_left, U_right)
dU = -(Δt/Δr) * (F_right - F_left)
```

### **3. Shock Capturing**

**Current:**
- Shocks captured via divergence limiting and pressure force
- May smear strong shocks over multiple cells
- Depends on pressure gradient calculation

**Fully Conservative:**
- Shocks captured automatically via Riemann solver
- Rankine-Hugoniot conditions satisfied exactly
- Sharp resolution of discontinuities

---

## Implementation Options for HUXt

### **Option 1: Keep Current Hybrid (RECOMMENDED for now)**

**Pros:**
- Minimal code changes
- Stable and tested
- Works with existing infrastructure
- Good enough for most solar wind applications

**Cons:**
- Not "textbook" HLL
- May struggle with very strong shocks

**When to use:** Current CME propagation, space weather forecasting

---

### **Option 2: Full Conservative Rewrite**

**Pros:**
- Proper conservation guarantees
- Better shock capturing
- More accurate for extreme events

**Cons:**
- Major rewrite of solve_radial
- Risk of introducing bugs
- Need to revalidate all results
- Boundary conditions need redesign

**When to use:** Research on strong shocks, method comparison studies

---

### **Option 3: Hybrid Approach (SUGGESTED)**

Create **two separate code paths**:

```python
if solver == 'hll_conservative':
    # Use full conservative formulation
    U_new = conservative_hll_step(U_old, ...)
elif solver == 'hll':
    # Use current hybrid approach
    v_new, rho_new, T_new = hll_step_compressible(...)
```

**Pros:**
- Allows comparison between methods
- Keeps existing code working
- Can validate conservative version gradually

**Cons:**
- Code duplication
- Two implementations to maintain

---

## What Needs to Change for Full Conservation

### **1. Data Structure**
```python
# Current: primitive variables
v_grid[nt, nr, nlon]
rho_grid[nt, nr, nlon]
temp_grid[nt, nr, nlon]

# Need: conservative variables
U_mass[nt, nr, nlon]
U_momentum[nt, nr, nlon]
U_energy[nt, nr, nlon]
```

### **2. Time Stepping Loop**
```python
# Current: update primitives directly
v_new, rho_new, temp_new = solver_step(v, rho, temp, ...)

# Need: update conservatives, then extract primitives
U_new = conservative_step(U_old, ...)
v, rho, temp = extract_primitives(U_new)
```

### **3. Boundary Conditions**
```python
# Current: set v, rho, temp at boundaries
v[0, :, :] = v_boundary
rho[0, :, :] = rho_boundary
temp[0, :, :] = temp_boundary

# Need: set conservative variables or use ghost cells
U_mass[0] = rho_boundary
U_momentum[0] = rho_boundary * v_boundary
U_energy[0] = compute_total_energy(rho, v, P)
```

### **4. Source Terms**
```python
# Need to add geometric source terms for spherical coordinates
# These don't appear in Cartesian geometry!

# Geometric terms in spherical 1D:
S_momentum += -(2/r) * P              # Pressure × area change
S_energy += -(2/r) * P * v            # P·dV work term

# Plus solar wind acceleration
S_momentum += rho * g_solar
S_energy += rho * v * g_solar
```

---

## Bottom Line

### **For Production Use:**
Keep the current hybrid approach. It's:
- Stable
- Accurate enough for CME propagation
- Already includes pressure gradient force
- Works with HUXt's infrastructure

### **For Research/Validation:**
Implement the fully conservative version as a separate option to:
- Compare methods
- Test on extreme cases
- Validate against textbook solutions
- Publish methodology papers

### **Current Status:**
- ✅ HLL flux functions exist
- ✅ Wave speed estimation works
- ✅ Pressure gradient force added
- ⚠️ Flux functions not actually used in evolution
- ⚠️ Not fully conservative (by design, for simplicity)

The current implementation is a **pragmatic hybrid** that gives you most of the benefits of HLL (stability, shock handling) without the complexity of full conservative formulation. For solar wind modeling, this is often the right trade-off!
