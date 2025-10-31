"""
Simple test to debug CGF solver initialization.
"""

import numpy as np
import astropy.units as u
from huxt import HUXt

# Create boundary conditions
v_boundary = np.ones(128) * 400.0 * (u.km / u.s)

print("Creating HUXt model with CGF solver...")
model = HUXt(
    simtime=5 * u.day,
    dt_scale=4,
    v_boundary=v_boundary,
    r_min=21.5 * u.solRad,
    r_max=240 * u.solRad,
    lon_start=330 * u.deg,
    lon_stop=30 * u.deg,
    frame='sidereal',
    solver='cgf',
    parallel=False
)

print(f"Model created successfully")
print(f"  compressible: {model.compressible}")
print(f"  solver: {model.solver}")
print(f"  n_longitudes: {model.lon.size}")
print(f"  input_rho_ts shape: {model.input_rho_ts.shape if hasattr(model.input_rho_ts, 'shape') else 'scalar'}")
print(f"  input_temp_ts shape: {model.input_temp_ts.shape if hasattr(model.input_temp_ts, 'shape') else 'scalar'}")
print(f"  input_v_ts shape: {model.input_v_ts.shape if hasattr(model.input_v_ts, 'shape') else 'scalar'}")

print("\nCalling solve()...")
try:
    model.solve([])
    print("✓ Solve completed successfully!")
except Exception as e:
    print(f"✗ Error during solve: {e}")
    import traceback
    traceback.print_exc()
