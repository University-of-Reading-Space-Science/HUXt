"""
Check if the correction is actually being applied by examining
values right after one time step.
"""
import numpy as np
import astropy.units as u
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import huxt as H

# Simple uniform conditions
v0 = 400.0  # km/s
rho0 = 1e-18  # kg/m³
T0 = 1e6  # K

# Create model
print("Creating model...")
model = H.HUXt(
    v_boundary=np.ones(64) * v0 * u.km/u.s,
    rho_boundary=np.ones(64) * rho0 * (u.kg/u.m**3),
    temp_boundary=np.ones(64) * T0 * u.K,
    cr_num=2063,
    r_min=20 * u.solRad,
    r_max=220 * u.solRad,
    lon_start=0 * u.deg,
    lon_stop=360 * u.deg,
    simtime=0.001 * u.day,  # Just ONE time step
    dt_scale=10000,  # Force exactly 1 time step
    compressible=True,
    solver='upwind'
)

model.alpha = 0.0  # No residual acceleration
print("Running solver for 1 time step...")
model.solve([])  # Run the simulation

# Check mass flux conservation at last output
r = model.r.to(u.solRad).value
v = model.v_grid.value[:, :, -1]  # All longitudes, all radii, last time
rho = model.rho_grid.value[:, :, -1]

# Compute flux at each point
flux = rho * v * (r[np.newaxis, :] ** 2)

# Check first longitude
flux_lon0 = flux[0, :]
flux_norm = flux_lon0 / flux_lon0[0]

print("\n" + "=" * 70)
print("MASS FLUX AFTER 1 TIME STEP (longitude 0)")
print("=" * 70)
print(f"Boundary flux (r[0]): {flux_lon0[0]:.6e}")
print(f"\nNormalized flux at each radius:")
for i in range(0, len(r), len(r) // 10):
    print(f"  r[{i:3d}] = {r[i]:6.1f} Rs: flux = {flux_norm[i]:.8f}")

print(f"\nOverall statistics:")
print(f"  Min: {np.min(flux_norm):.8f}")
print(f"  Max: {np.max(flux_norm):.8f}")
deviation = (np.max(flux_norm) - np.min(flux_norm)) / np.max(flux_norm) * 100
print(f"  Deviation: {deviation:.2f}%")

if deviation < 0.01:
    print("\n✓ PASS: Mass flux conserved within 0.01%")
else:
    print(f"\n✗ FAIL: Mass flux varies by {deviation:.2f}%")
