"""
Check mass conservation excluding the boundary point.
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
    simtime=0.5 * u.day,
    dt_scale=4,
    compressible=True,
    solver='upwind'
)

model.alpha = 0.0
print("Running solver...")
model.solve([])

# Check mass flux conservation
r = model.r.to(u.solRad).value
v = model.v_grid.value[-1, :, 0]
rho = model.rho_grid.value[-1, :, 0]

flux = rho * v * r**2

# Normalize to FIRST INTERIOR POINT (index 1), not boundary
flux_norm_boundary = flux / flux[0]
flux_norm_interior = flux / flux[1]

print("\n" + "="*70)
print("MASS FLUX CONSERVATION CHECK")
print("="*70)

print(f"\nNormalized to BOUNDARY (index 0):")
print(f"  Min: {np.min(flux_norm_boundary):.6f}")
print(f"  Max: {np.max(flux_norm_boundary):.6f}")
dev_boundary = (np.max(flux_norm_boundary) - np.min(flux_norm_boundary)) / np.max(flux_norm_boundary) * 100
print(f"  Deviation: {dev_boundary:.2f}%")

print(f"\nNormalized to FIRST INTERIOR POINT (index 1):")
print(f"  Min: {np.min(flux_norm_interior):.6f}")
print(f"  Max: {np.max(flux_norm_interior):.6f}")
dev_interior = (np.max(flux_norm_interior) - np.min(flux_norm_interior)) / np.max(flux_norm_interior) * 100
print(f"  Deviation: {dev_interior:.2f}%")

# Check interior only (excluding boundary)
flux_interior_only = flux[1:]
flux_int_norm = flux_interior_only / flux_interior_only[0]
dev_int_only = (np.max(flux_int_norm) - np.min(flux_int_norm)) / np.max(flux_int_norm) * 100

print(f"\nINTERIOR ONLY (indices 1 to end):")
print(f"  Min: {np.min(flux_int_norm):.6f}")
print(f"  Max: {np.max(flux_int_norm):.6f}")
print(f"  Deviation: {dev_int_only:.2f}%")

if dev_int_only < 0.1:
    print("\n✓ PASS: Interior mass flux conserved!")
else:
    print(f"\n✗ FAIL: Interior mass flux varies by {dev_int_only:.2f}%")
