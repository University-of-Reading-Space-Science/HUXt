"""
Quick test to verify mass flux conservation in upwind solver.
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
    simtime=0.5 * u.day,  # Shorter simulation
    dt_scale=4,
    compressible=True,
    solver='upwind'
)

model.alpha = 0.0  # No residual acceleration
print("Running solver...")
model.solve([])

# Check mass flux conservation
r = model.r.to(u.solRad).value
v = model.v_grid.value[-1, :, 0]
rho = model.rho_grid.value[-1, :, 0]

flux = rho * v * r**2
flux_norm = flux / flux[0]

print("\n" + "="*70)
print("MASS FLUX CONSERVATION CHECK")
print("="*70)
print(f"Mass flux ρvr² (normalized to inner boundary):")
print(f"  Minimum: {np.min(flux_norm):.6f}")
print(f"  Maximum: {np.max(flux_norm):.6f}")
print(f"  Std dev: {np.std(flux_norm):.2e}")
print(f"  Max deviation: {np.max(np.abs(flux_norm - 1.0)) * 100:.2f}%")

if np.max(np.abs(flux_norm - 1.0)) < 1e-6:
    print("\n✓ PASS: Mass flux is conserved to machine precision!")
else:
    print(f"\n✗ FAIL: Mass flux varies by {np.max(np.abs(flux_norm - 1.0)) * 100:.2f}%")
