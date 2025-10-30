"""
Test script to run 1D compressible HUXt with v=400 km/s and alpha=0 for a range of gamma values.
Plot velocity, density, and temperature as functions of radius for each gamma.
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time

import huxt as H

# Test parameters
cr_num = 2063
r_min = 30
r_max = 230
lon_start = 0
lon_stop = 360
v_boundary = 400  # km/s
alpha = 0.0  # No acceleration

# Range of gamma values to test
gamma_values = [1.1, 1.3, 1.5, 1.67, 2.0]

# Store results
results = {}

print("Running HUXt for different gamma values (alpha=0)...")
print(f"Boundary velocity: {v_boundary} km/s")
print(f"Alpha: {alpha}")
print(f"Domain: {r_min} - {r_max} solar radii")
print("")

# Create model with alpha=0
v_bound = np.ones(128) * v_boundary

print("first an incompressible run...")
model_incomp = H.HUXt(v_boundary=v_bound * u.km/u.s, 
                   cr_num=cr_num, 
                   r_min=r_min * u.solRad,
                   r_max=r_max * u.solRad,
                   lon_start=lon_start * u.deg,
                   lon_stop=lon_stop * u.deg,
                   simtime=1 * u.day,
                   dt_scale=4,
                   compressible=False, solver='upwind')
model_incomp.solve([])

for gamma in gamma_values:
    print(f"Testing gamma = {gamma}...")
    

    
    model = H.HUXt(v_boundary=v_bound * u.km/u.s, 
                   cr_num=cr_num, 
                   r_min=r_min * u.solRad,
                   r_max=r_max * u.solRad,
                   lon_start=lon_start * u.deg,
                   lon_stop=lon_stop * u.deg,
                   simtime=1 * u.day,
                   dt_scale=4,
                   compressible=True, solver='upwind')
    
    # Set gamma and alpha parameters
    model.set_gamma(gamma)  # Use set_gamma() to properly update temperature boundary
    model.alpha = alpha
    

    model.solve([])

    print(f"  Model gamma set to: {model.gamma}")
    print(f"  Model alpha set to: {model.alpha}")
    
    # Extract final profiles
    # Grid shapes are [nt, nr, nlon]
    v_final = model.v_grid.value[-1, :, 0]  # Last time, all radii, first longitude
    rho_final = model.rho_grid.value[-1, :, 0]
    temp_final = model.temp_grid.value[-1, :, 0]
    r_grid = model.r.to(u.solRad).value
    
    # Convert density to number density (protons/cm^3)
    m_p = 1.6726e-27  # kg
    n_final = rho_final / m_p / 1e6  # protons/cm^3
    
    results[gamma] = {
        'r': r_grid,
        'v': v_final,
        'n': n_final,
        'T': temp_final
    }
    
    print(f"  Final velocity at outer boundary: {v_final[-1]:.2f} km/s")
    print(f"  Final temperature at outer boundary: {temp_final[-1]:.2e} K")

print("\nGenerating plots...")

# Create 3-panel plot
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Color map for different gamma values
colors = plt.cm.viridis(np.linspace(0, 1, len(gamma_values)))

# Plot 1: Velocity
ax = axes[0]

ax.plot(model_incomp.r.to(u.solRad).value, 
        model_incomp.v_grid.value[-1, :, 0], 
        '-', color='k', linewidth=2, label='Incompressible')
for i, gamma in enumerate(gamma_values):
    r = results[gamma]['r']
    v = results[gamma]['v']
    ax.plot(r, v, '-', color=colors[i], linewidth=2, label=f'γ = {gamma}')
ax.axhline(y=v_boundary, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('Velocity [km/s]', fontsize=14)
ax.set_title(f'1D Compressible HUXt (α={alpha}): Boundary = {v_boundary} km/s', 
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11, ncol=2)
ax.set_xlim([r_min, r_max])

# Plot 2: Density
ax = axes[1]
for i, gamma in enumerate(gamma_values):
    r = results[gamma]['r']
    n = results[gamma]['n']
    ax.plot(r, n, '-', color=colors[i], linewidth=2, label=f'γ = {gamma}')
ax.set_ylabel('Number Density [protons/cm³]', fontsize=14)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11, ncol=2)
ax.set_xlim([r_min, r_max])

# Plot 3: Temperature
ax = axes[2]
for i, gamma in enumerate(gamma_values):
    r = results[gamma]['r']
    T = results[gamma]['T']
    ax.plot(r, T, '-', color=colors[i], linewidth=2, label=f'γ = {gamma}')
ax.set_xlabel('Radius [R$_☉$]', fontsize=14)
ax.set_ylabel('Temperature [K]', fontsize=14)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11, ncol=2)
ax.set_xlim([r_min, r_max])

plt.tight_layout()
plt.show()
plt.savefig('debug/gamma_alpha0_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved plot to debug/gamma_alpha0_comparison.png")



print("\nDone!")
