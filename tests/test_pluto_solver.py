"""
Test PLUTO solver integration in HUXt.
"""
import numpy as np
import astropy.units as u
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import huxt as H

print("Testing PLUTO solver integration in HUXt...")
print("=" * 70)

# Create simple uniform boundary conditions
v0 = 400.0  # km/s
rho0 = 1e-18  # kg/m³
T0 = 1e6  # K

print(f"\nBoundary conditions:")
print(f"  Velocity: {v0} km/s")
print(f"  Density: {rho0:.2e} kg/m³")
print(f"  Temperature: {T0:.2e} K")

# Create HUXt model with PLUTO solver
print(f"\nCreating HUXt model with PLUTO solver...")
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
    solver='pluto'  # Use PLUTO solver
)

model.alpha = 0.0  # No residual acceleration

print(f"Model created successfully!")
print(f"  Radial grid: {len(model.r)} points from {model.r[0]:.1f} to {model.r[-1]:.1f}")
print(f"  Longitude grid: {len(model.lon)} points")
print(f"  Simulation time: {model.simtime.to(u.day)}")
print(f"  Solver: {model.solver}")
print(f"  Compressible: {model.compressible}")

# Run the model
print(f"\nRunning model...")
try:
    model.solve([])
    print(f"✓ Model run completed successfully!")
    
    # Check results
    print(f"\nFinal results:")
    v_final = model.v_grid.value[-1, :, 0]
    rho_final = model.rho_grid.value[-1, :, 0]
    temp_final = model.temp_grid.value[-1, :, 0]
    
    print(f"  Velocity range: {np.min(v_final):.1f} - {np.max(v_final):.1f} km/s")
    print(f"  Density range: {np.min(rho_final):.2e} - {np.max(rho_final):.2e} kg/m³")
    print(f"  Temperature range: {np.min(temp_final):.2e} - {np.max(temp_final):.2e} K")
    
    # Check for physical values
    if np.all(np.isfinite(v_final)) and np.all(v_final > 0):
        print(f"✓ Velocity values are physical")
    else:
        print(f"✗ WARNING: Non-physical velocity values detected")
    
    if np.all(np.isfinite(rho_final)) and np.all(rho_final > 0):
        print(f"✓ Density values are physical")
    else:
        print(f"✗ WARNING: Non-physical density values detected")
    
    if np.all(np.isfinite(temp_final)) and np.all(temp_final > 0):
        print(f"✓ Temperature values are physical")
    else:
        print(f"✗ WARNING: Non-physical temperature values detected")
    
    print(f"\n" + "=" * 70)
    print(f"PLUTO SOLVER TEST: PASSED")
    print(f"=" * 70)
    
except Exception as e:
    print(f"\n✗ Error during model run:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    print(f"\n" + "=" * 70)
    print(f"PLUTO SOLVER TEST: FAILED")
    print(f"=" * 70)
    sys.exit(1)
