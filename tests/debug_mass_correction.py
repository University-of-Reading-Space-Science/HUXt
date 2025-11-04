"""
Debug the mass flux correction to understand why it's not working
"""

import numpy as np

# Simulate what happens in _upwind_step_compressible_

# Input arrays (simplified)
nr = 10
rrel = np.linspace(0.0, 1.0, nr + 1)  # nr+1 points for downwind/upwind staggering
r_boundary = 20.0 * 695700.0  # km

# Downwind (dn) arrays have nr points
v_dn = np.ones(nr) * 400.0  # km/s
rho_dn = np.ones(nr) * 1e-18  # kg/m³
temp_dn = np.ones(nr) * 1e6  # K

# Upwind (up) arrays also have nr points  
v_up = np.ones(nr) * 400.0
rho_up = np.ones(nr) * 1e-18
temp_up = np.ones(nr) * 1e6

# After velocity update (assume no change for this test)
v_up_next = v_up.copy()

# Radial positions for density update
r_dn_density = rrel[:-1] * 695700.0 + r_boundary  # nr points
r_up_density = rrel[1:] * 695700.0 + r_boundary   # nr points

print(f"Number of points:")
print(f"  v_dn: {len(v_dn)}")
print(f"  v_up: {len(v_up)}")  
print(f"  v_up_next: {len(v_up_next)}")
print(f"  r_dn_density: {len(r_dn_density)}")
print(f"  r_up_density: {len(r_up_density)}")

# After density update (assume no change)
rho_up_next = rho_up.copy()

# Now apply mass flux correction
F_ref = rho_dn[0] * v_dn[0] * (r_dn_density[0] ** 2)
print(f"\nReference flux at boundary:")
print(f"  F_ref = {F_ref:.6e} kg·km/s")
print(f"  rho_dn[0] = {rho_dn[0]:.6e}")
print(f"  v_dn[0] = {v_dn[0]:.6f}")
print(f"  r_dn_density[0] = {r_dn_density[0]:.6f} km")

print(f"\nBefore correction:")
for i in range(min(5, nr)):
    flux_i = rho_up_next[i] * v_up_next[i] * (r_up_density[i] ** 2)
    print(f"  i={i}: r={r_up_density[i]:.2f} km, flux={flux_i:.6e}, ratio={flux_i/F_ref:.6f}")

# Apply correction
for i in range(len(rho_up_next)):
    rho_conserved = F_ref / (v_up_next[i] * (r_up_density[i] ** 2))
    rho_up_next[i] = rho_conserved

print(f"\nAfter correction:")
for i in range(min(5, nr)):
    flux_i = rho_up_next[i] * v_up_next[i] * (r_up_density[i] ** 2)
    print(f"  i={i}: r={r_up_density[i]:.2f} km, flux={flux_i:.6e}, ratio={flux_i/F_ref:.6f}")
