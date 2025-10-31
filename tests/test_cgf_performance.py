"""
Test CGF solver performance with serial vs parallel execution.
"""

import numpy as np
import astropy.units as u
from huxt import HUXt
import time

# Test parameters
simtime = 5 * u.day
dt_scale = 4

print("="*70)
print("CGF SOLVER PERFORMANCE TEST")
print("="*70)
print(f"Simulation time: {simtime}")
print("Number of longitudes: 22")
print(f"dt_scale: {dt_scale}")
print()

# Create boundary conditions (use real data for one longitude, replicate)
v_boundary = np.ones(128) * 400.0 * (u.km / u.s)

# ============================================================================
# TEST 1: Serial execution
# ============================================================================
print("\n" + "="*70)
print("TEST 1: SERIAL EXECUTION")
print("="*70)

model_serial = HUXt(
    simtime=simtime,
    dt_scale=dt_scale,
    v_boundary=v_boundary,
    r_min=21.5 * u.solRad,
    r_max=240 * u.solRad,
    lon_start=330 * u.deg,
    lon_stop=30 * u.deg,
    frame='sidereal',
    solver='cgf',
    parallel=False  # Serial
)

print("\nRunning serial CGF solver...")
start_serial = time.time()
model_serial.solve([])
time_serial = time.time() - start_serial

n_longitudes = model_serial.lon.size
print(f"\n✓ Serial execution completed")
print(f"  Total time: {time_serial:.2f} seconds ({time_serial/60:.2f} minutes)")
print(f"  Time per longitude: {time_serial/n_longitudes:.2f} seconds")

# ============================================================================
# TEST 2: Parallel execution
# ============================================================================
print("\n" + "="*70)
print("TEST 2: PARALLEL EXECUTION")
print("="*70)

model_parallel = HUXt(
    simtime=simtime,
    dt_scale=dt_scale,
    v_boundary=v_boundary,
    r_min=21.5 * u.solRad,
    r_max=240 * u.solRad,
    lon_start=330 * u.deg,
    lon_stop=30 * u.deg,
    frame='sidereal',
    solver='cgf',
    parallel=True  # Parallel
)

print("\nRunning parallel CGF solver...")
start_parallel = time.time()
model_parallel.solve([])
time_parallel = time.time() - start_parallel

print(f"\n✓ Parallel execution completed")
print(f"  Total time: {time_parallel:.2f} seconds ({time_parallel/60:.2f} minutes)")
print(f"  Time per longitude: {time_parallel/n_longitudes:.2f} seconds")

# ============================================================================
# PERFORMANCE COMPARISON
# ============================================================================
print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"Serial time:   {time_serial:.2f} s")
print(f"Parallel time: {time_parallel:.2f} s")
print(f"Speedup:       {time_serial/time_parallel:.2f}x")
print(f"Efficiency:    {100*(time_serial/time_parallel)/n_longitudes:.1f}%")

# Verify results match
v_diff = np.abs(model_serial.v_grid.value - model_parallel.v_grid.value).max()
rho_diff = np.abs(model_serial.rho_grid.value - model_parallel.rho_grid.value).max()
temp_diff = np.abs(model_serial.temp_grid.value - model_parallel.temp_grid.value).max()

print("\nResults verification:")
print(f"  Max velocity difference:     {v_diff:.2e} km/s")
print(f"  Max density difference:      {rho_diff:.2e} kg/m³")
print(f"  Max temperature difference:  {temp_diff:.2e} K")

if v_diff < 1e-10 and rho_diff < 1e-10 and temp_diff < 1e-10:
    print("\n✓ Serial and parallel results match!")
else:
    print("\n⚠ Warning: Serial and parallel results differ")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
