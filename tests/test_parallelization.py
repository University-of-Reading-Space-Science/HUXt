"""
Test parallelization feature in HUXt master branch.
Compares serial vs parallel execution to ensure they produce identical results.
"""
import numpy as np
import astropy.units as u
import sys
import time

sys.path.insert(0, 'c:/Users/vy902033/Dropbox/python_repos/HUXt5/HUXt')

from huxt.huxt import HUXt

# Create a simple test case with spatial structure
print("="*70)
print("Testing HUXt Parallelization")
print("="*70)

# Create boundary condition with structure
v_boundary = 400 * np.ones(128) * (u.km/u.s)
v_boundary[30:50] = 600 * (u.km/u.s)  # Fast stream
v_boundary[80:100] = 350 * (u.km/u.s)  # Slow stream

print("\n1. Running pre-JIT execution...")
t_start = time.time()
model_serial = HUXt(
    v_boundary=v_boundary,
    cr_num=2050,
    cr_lon_init=0 * u.deg,
    simtime=27 * u.day,
    dt_scale=4,
    frame='synodic',
    parallel=False  # Serial
)
model_serial.solve(cme_list=[])
t_serial = time.time() - t_start
print(f"   Pre-JIT execution time: {t_serial:.2f} seconds")

# Test 1: Serial execution
print("\n1. Running SERIAL execution...")
t_start = time.time()
model_serial = HUXt(
    v_boundary=v_boundary,
    cr_num=2050,
    cr_lon_init=0 * u.deg,
    simtime=27 * u.day,
    dt_scale=4,
    frame='synodic',
    parallel=False  # Serial
)
model_serial.solve(cme_list=[])
t_serial = time.time() - t_start


print(f"   Serial execution time: {t_serial:.2f} seconds")
print(f"   nlon: {model_serial.nlon:.2f}")

# Test 2: Parallel execution
print("\n2. Running PARALLEL execution...")
t_start = time.time()
model_parallel = HUXt(
    v_boundary=v_boundary,
    cr_num=2050,
    cr_lon_init=0 * u.deg,
    simtime=27 * u.day,
    dt_scale=4,
    frame='synodic',
    parallel=True  # Parallel
)
model_parallel.solve(cme_list=[])
t_parallel = time.time() - t_start
print(f"   Parallel execution time: {t_parallel:.2f} seconds")
print(f"   nlon: {model_parallel.nlon:.2f}")
print(f"   Speedup factor: {t_serial/t_parallel:.2f}x")

# Test 3: Compare results
print("\n3. Comparing results...")
v_diff = np.abs(model_serial.v_grid.value - model_parallel.v_grid.value)
max_diff = np.max(v_diff)
mean_diff = np.mean(v_diff)

print(f"   Max difference in v_grid: {max_diff:.6e} km/s")
print(f"   Mean difference in v_grid: {mean_diff:.6e} km/s")

if max_diff < 1e-10:
    print("\n✓ SUCCESS: Serial and parallel results are identical!")
else:
    print(f"\n✗ WARNING: Serial and parallel results differ by up to {max_diff:.6e} km/s")

# Test 4: Check CME tracking particles
cme_r_diff = np.abs(model_serial.cme_particles_r - model_parallel.cme_particles_r)
cme_v_diff = np.abs(model_serial.cme_particles_v - model_parallel.cme_particles_v)
print(f"\n4. CME particle tracking:")
if np.any(np.isfinite(cme_r_diff)):
    print(f"   Max difference in cme_particles_r: {np.nanmax(cme_r_diff):.6e}")
    print(f"   Max difference in cme_particles_v: {np.nanmax(cme_v_diff):.6e}")
else:
    print(f"   No CMEs in this test run (as expected)")

print("\n" + "="*70)
print("Test Complete!")
print("="*70)
