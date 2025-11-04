"""
Debug script to check what's happening inside the upwind step function.
We'll instrument the function to see intermediate values.
"""
import numpy as np
import astropy.units as u
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from huxt import HUXt
import huxt.huxt as huxt_module

# Create a simple model
print("Creating model...")
model = HUXt(
    cr_num=2010,
    r_min=21.5 * u.solRad,
    r_max=300 * u.solRad,
    lon_start=0 * u.deg,
    lon_stop=360 * u.deg,
    latitude=0 * u.deg,
    simtime=1 * u.day,
    dt_scale=1.0,
    compressible=True
)

# Set initial conditions similar to test_mass_conservation.py
v_boundary = np.ones(model.lon.size) * 400.0 * (u.km / u.s)
rho_boundary = np.ones(model.lon.size) * 3e-21 * (u.kg / u.m**3)
temp_boundary = np.ones(model.lon.size) * 1.5e6 * u.K

model.v_boundary = v_boundary
model.rho_boundary = rho_boundary
model.temp_boundary = temp_boundary

# Take one time step
print("Running single time step...")
model.solve([0])

# Now check the mass flux in the result
r_m = model.r.to(u.m).value
v_ms = model.v_grid[:, :, 0].to(u.m / u.s).value
rho_kgm3 = model.rho_grid[:, :, 0].to(u.kg / u.m**3).value

# Compute mass flux
mass_flux = rho_kgm3 * v_ms * (r_m[np.newaxis, :] ** 2)

# Normalize to inner boundary average
flux_ref = np.mean(mass_flux[:, 0])
flux_normalized = mass_flux / flux_ref

print("\n" + "=" * 70)
print("MASS FLUX CHECK AFTER ONE TIME STEP")
print("=" * 70)
print(f"Reference flux (inner boundary average): {flux_ref:.6e} kg·m/s")
print(f"\nFlux at each radial shell (normalized):")
for i in range(0, len(model.r), len(model.r) // 10):
    r_rs = model.r[i].to(u.solRad).value
    flux_avg = np.mean(flux_normalized[:, i])
    flux_std = np.std(flux_normalized[:, i])
    print(f"  r = {r_rs:6.1f} Rs: avg = {flux_avg:.6f}, std = {flux_std:.6f}")

print(f"\nOverall statistics:")
print(f"  Min flux (normalized): {np.min(flux_normalized):.6f}")
print(f"  Max flux (normalized): {np.max(flux_normalized):.6f}")
print(f"  Mean flux: {np.mean(flux_normalized):.6f}")
print(f"  Std dev: {np.std(flux_normalized):.6f}")

deviation = (np.max(flux_normalized) - np.min(flux_normalized)) / np.max(flux_normalized) * 100
print(f"  Max deviation: {deviation:.2f}%")

if deviation < 1.0:
    print("\n✓ PASS: Mass flux is conserved within 1%")
else:
    print(f"\n✗ FAIL: Mass flux varies by {deviation:.2f}%")
