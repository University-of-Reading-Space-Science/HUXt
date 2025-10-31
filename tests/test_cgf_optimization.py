"""
Quick test to verify CGF solver optimizations and measure speedup.
"""
import numpy as np
import astropy.units as u
import time
from huxt import HUXt

print("\n" + "="*70)
print("TESTING CGF SOLVER OPTIMIZATIONS")
print("="*70)

# Test 1: Single longitude (baseline)
print("\n1. Single longitude (1D run) - BASELINE")
print("-"*70)
model_1d = HUXt(
    v_boundary=400.0 * u.km / u.s * np.ones(128),
    simtime=1.0 * u.day,
    lon_out=0.0 * u.deg,
    compressible=True,
    solver='cgf',
    parallel=False  # Serial for 1D
)

start = time.time()
model_1d.solve([])
time_1d = time.time() - start

print(f"\nBaseline (1 longitude): {time_1d:.2f} seconds")
print(f"This is the minimum time per longitude")

# Test 2: 8 longitudes serial
print("\n\n2. 8 longitudes - SERIAL")
print("-"*70)
model_8_serial = HUXt(
    v_boundary=400.0 * u.km / u.s * np.ones(128),
    simtime=1.0 * u.day,
    lon_start=0.0 * u.deg,
    lon_stop=22.5 * u.deg,  # 8 longitudes
    compressible=True,
    solver='cgf',
    parallel=False
)

start = time.time()
model_8_serial.solve([])
time_8_serial = time.time() - start

print(f"\n8 longitudes (serial): {time_8_serial:.2f} seconds")
print(f"Time per longitude: {time_8_serial/8:.2f} seconds")
print(f"Overhead vs baseline: {(time_8_serial/8)/time_1d:.2f}x")

# Test 3: 8 longitudes parallel
print("\n\n3. 8 longitudes - PARALLEL (adaptive workers)")
print("-"*70)
model_8_parallel = HUXt(
    v_boundary=400.0 * u.km / u.s * np.ones(128),
    simtime=1.0 * u.day,
    lon_start=0.0 * u.deg,
    lon_stop=22.5 * u.deg,  # 8 longitudes
    compressible=True,
    solver='cgf',
    parallel=True  # Will use 4 workers (adaptive)
)

start = time.time()
model_8_parallel.solve([])
time_8_parallel = time.time() - start

print(f"\n8 longitudes (parallel): {time_8_parallel:.2f} seconds")
print(f"Time per longitude: {time_8_parallel/8:.2f} seconds")
print(f"Speedup vs serial: {time_8_serial/time_8_parallel:.2f}x")

# Summary
print("\n\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Baseline (1 lon):      {time_1d:>8.2f}s")
print(f"Serial (8 lon):        {time_8_serial:>8.2f}s  ({time_8_serial/time_1d:>5.2f}x baseline)")
print(f"Parallel (8 lon):      {time_8_parallel:>8.2f}s  ({time_8_parallel/time_1d:>5.2f}x baseline)")
print(f"\nParallel speedup:      {time_8_serial/time_8_parallel:>8.2f}x")
print(f"Parallel efficiency:   {(time_8_serial/time_8_parallel)/4*100:>7.1f}% (relative to 4 workers)")

if time_8_parallel < time_8_serial:
    print("\n✓ SUCCESS: Parallel is faster than serial!")
    print(f"  Estimated time for 128 longitudes (parallel): {time_8_parallel/8 * 128:.1f}s ({time_8_parallel/8 * 128/60:.1f} min)")
else:
    print("\n⚠ WARNING: Serial is still faster than parallel")
    print("  Possible causes:")
    print("  - Problem size too small (parallel overhead dominates)")
    print("  - System has limited cores")
    print("  - Pyro initialization overhead is very high")

print("="*70 + "\n")
