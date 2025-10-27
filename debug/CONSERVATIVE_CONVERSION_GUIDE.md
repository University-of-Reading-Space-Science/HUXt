# Converting Compressible Upwind to Conservative Variables

## Quick Answer: YES! It's straightforward.

---

## The Conversion

### **Primitive Variables → Conservative Variables**

```python
# Given: ρ, v, T (what HUXt currently uses)

# Step 1: Compute pressure from ideal gas law
P = (ρ / m_p) * k_B * T

# Step 2: Build conservative variables
U_mass = ρ                              # Density (same!)
U_momentum = ρ * v                      # Momentum density
U_energy = 0.5 * ρ * v² + P/(γ-1)      # Total energy density
```

### **Conservative Variables → Primitive Variables**

```python
# Given: U_mass, U_momentum, U_energy (conservative form)

# Extract primitives
ρ = U_mass
v = U_momentum / U_mass
P = (γ-1) * (U_energy - 0.5 * U_mass * v²)
T = P * m_p / (ρ * k_B)
```

---

## Side-by-Side: Current vs. Conservative

### **Current Primitive Formulation**

```python
def upwind_step_primitive(v_up, v_dn, rho_up, rho_dn, temp_up, temp_dn, ...):
    """Current implementation - works with ρ, v, T directly"""
    
    # Density evolution
    rho_advection = - dtdr * v_up * (rho_up - rho_dn)
    rho_compression = - rho_up * div_v * dt
    rho_up_next = rho_up + rho_advection + rho_compression
    
    # Velocity evolution  
    v_up_next = v_up - dtdr * v_up * (v_up - v_dn)
    pressure_accel = -(dp_dr / rho_avg) / 1e6
    v_up_next += pressure_accel * dt + v_diff
    
    # Temperature evolution
    temp_compression = -(gamma - 1.0) * temp_dn * div_v * dt
    temp_up_next = temp_up + temp_advection + temp_compression
    
    return v_up_next, rho_up_next, temp_up_next
```

### **Conservative Formulation**

```python
def upwind_step_conservative(v_up, v_dn, rho_up, rho_dn, temp_up, temp_dn, ...):
    """Conservative form - works with U = [ρ, ρv, E] internally"""
    
    # Step 1: Convert to conservative variables
    U_mass = rho_up
    U_momentum = rho_up * v_up
    P = (rho_up / m_p) * k_B * temp_up
    U_energy = 0.5 * rho_up * v_up**2 + P/(gamma-1)
    
    # Step 2: Compute fluxes
    F_mass = rho * v
    F_momentum = rho * v**2 + P
    F_energy = v * (E + P)
    
    # Step 3: Update conservatives with flux differencing
    U_mass_new = U_mass - (dt/dr) * (F_mass[i+1] - F_mass[i])
    U_momentum_new = U_momentum - (dt/dr) * (F_momentum[i+1] - F_momentum[i]) + dt*Source
    U_energy_new = U_energy - (dt/dr) * (F_energy[i+1] - F_energy[i]) + dt*Source
    
    # Step 4: Extract primitives
    rho_up_next = U_mass_new
    v_up_next = U_momentum_new / U_mass_new
    P_new = (gamma-1) * (U_energy_new - 0.5 * U_mass_new * v_up_next**2)
    temp_up_next = P_new * m_p / (rho_up_next * k_B)
    
    return v_up_next, rho_up_next, temp_up_next
```

---

## Key Equations in Both Forms

### **1. Mass Conservation**

**Primitive:**
```
∂ρ/∂t + v·∂ρ/∂r + ρ·∇·v = 0
```

**Conservative:**
```
∂ρ/∂t + ∂(ρv)/∂r = 0  (+ geometric terms)
```

*These are identical!* Just different ways to write the same physics.

---

### **2. Momentum Conservation**

**Primitive:**
```
∂v/∂t + v·∂v/∂r = -(1/ρ)·∂P/∂r + g
```

**Conservative:**
```
∂(ρv)/∂t + ∂(ρv² + P)/∂r = ρg
```

*Multiply primitive by ρ to get conservative form.*

---

### **3. Energy Conservation**

**Primitive:**
```
∂T/∂t + v·∂T/∂r + (γ-1)T·∇·v = 0
```

**Conservative:**
```
∂E/∂t + ∂(v(E+P))/∂r = ρvg
where E = ½ρv² + P/(γ-1)
```

*The conservative form combines kinetic + internal energy.*

---

## Why Both Formulations Give Same Answer

If you implement them correctly:

```
Primitive:  ∂ρ/∂t = -v·∂ρ/∂r - ρ·∇·v
           ∂v/∂t = -v·∂v/∂r - (1/ρ)·∂P/∂r + g

Conservative: ∂ρ/∂t = -∂(ρv)/∂r
              ∂(ρv)/∂t = -∂(ρv²+P)/∂r + ρg
```

Expand the conservative form:
```
∂ρ/∂t = -ρ·∂v/∂r - v·∂ρ/∂r  ← chain rule

∂(ρv)/∂t = ρ·∂v/∂t + v·∂ρ/∂t
         = -∂(ρv²)/∂r - ∂P/∂r + ρg
         = -ρv·∂v/∂r - v²·∂ρ/∂r - ∂P/∂r + ρg
```

Substitute ∂ρ/∂t:
```
ρ·∂v/∂t = -ρv·∂v/∂r - ∂P/∂r + ρg
∂v/∂t = -v·∂v/∂r - (1/ρ)·∂P/∂r + g  ← Same as primitive!
```

**They're mathematically equivalent!**

---

## Practical Differences

### **Numerical Properties**

| Property | Primitive | Conservative |
|----------|-----------|--------------|
| **Conservation** | Approximate | Exact |
| **Shock capture** | Good | Better |
| **Oscillations** | Possible | Less likely |
| **Negative P** | Rare | Possible |
| **Intuition** | High | Lower |

### **Implementation**

| Aspect | Primitive | Conservative |
|--------|-----------|--------------|
| **Complexity** | Lower | Higher |
| **BC setup** | Easy | Harder |
| **Output** | Direct | Need extraction |
| **Debugging** | Easier | Harder |

---

## Recommendation for HUXt

### **Current Approach (Primitive): KEEP IT** ✓

Reasons:
1. **Works well** for solar wind (smooth flows + weak shocks)
2. **Intuitive** - directly see v, ρ, T
3. **Stable** - proven in operation
4. **Easy BCs** - just set v, ρ, T at boundaries
5. **Includes pressure force** - recently added

### **When to Use Conservative:**

Only if you need:
- Formal proof of conservation
- Very strong shocks (Mach > 5)
- Method comparison papers
- Textbook correctness

For operational space weather forecasting, the current primitive formulation with pressure gradient force is **perfect**.

---

## Bottom Line

**Q: Can compressible upwind be converted to conservative variables?**

**A: YES - trivially!**

The conversion is just:
```python
U = [ρ, ρv, 0.5*ρ*v² + P/(γ-1)]
```

And back:
```python
ρ = U[0]
v = U[1] / U[0]  
P = (γ-1) * (U[2] - 0.5*U[0]*v²)
```

The current implementation is already doing conservative physics - it just uses primitive variables as the working variables, which is perfectly valid and often preferable for practical applications!
