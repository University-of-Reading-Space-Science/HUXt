"""
Simple test comparing incompressible vs CGF solver with time-varying boundary.
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from huxt import HUXt
import huxt.huxt_inputs as Hin
import huxt.huxt_analysis as HA

print("Setting up boundary conditions from MAS...")
cr = 2120
vr_in = Hin.get_MAS_long_profile(cr, 0.0 * u.deg)

print(f"v_boundary: {vr_in.shape}, range [{vr_in.min():.1f}, {vr_in.max():.1f}]")

simtime = 5 * u.day  # Shorter simulation for quick test

# Incompressible solver
print("\n" + "=" * 70)
print("Running INCOMPRESSIBLE solver...")
print("=" * 70)
model_incomp = HUXt(
    v_boundary=vr_in,
    lon_out=0.0 * u.rad,
    simtime=simtime,
    dt_scale=4,
    compressible=False,
    solver='upwind'
)
model_incomp.solve([])
print("✓ Incompressible solver complete")

# CGF solver
print("\n" + "=" * 70)
print("Running CGF solver...")
print("=" * 70)
model_cgf = HUXt(
    v_boundary=vr_in,
    lon_out=0.0 * u.rad,
    simtime=simtime,
    dt_scale=4,
    compressible=True,
    solver='cgf'
)
model_cgf.solve([])
print("✓ CGF solver complete")

# Compare results at Earth
print("\n" + "=" * 70)
print("Comparing results at Earth...")
print("=" * 70)

# Extract velocity at Earth
r_earth = 215 * u.solRad
idx_earth_incomp = np.argmin(np.abs(model_incomp.r.value - r_earth.to(u.solRad).value))
idx_earth_cgf = np.argmin(np.abs(model_cgf.r.value - r_earth.to(u.solRad).value))

v_earth_incomp = model_incomp.v_grid.value[:, idx_earth_incomp, 0]
v_earth_cgf = model_cgf.v_grid.value[:, idx_earth_cgf, 0]

print(f"Incompressible v at Earth: range [{v_earth_incomp.min():.1f}, {v_earth_incomp.max():.1f}] km/s")
print(f"CGF v at Earth: range [{v_earth_cgf.min():.1f}, {v_earth_cgf.max():.1f}] km/s")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

time_hours = model_incomp.time_out.to(u.h).value

ax1.plot(time_hours, v_earth_incomp, 'b-', label='Incompressible', linewidth=2)
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Velocity at Earth (km/s)')
ax1.set_title('Incompressible Solver')
ax1.grid(True, alpha=0.3)

ax2.plot(time_hours, v_earth_cgf, 'r-', label='CGF', linewidth=2)
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Velocity at Earth (km/s)')
ax2.set_title('CGF Solver')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/vy902033/Library/CloudStorage/Dropbox/python_repos/HUXt5/HUXt/tests/cgf_comparison.png', dpi=150)
print(f"\n✓ Plot saved to tests/cgf_comparison.png")

# Check if CGF has structure
v_std_incomp = np.std(v_earth_incomp)
v_std_cgf = np.std(v_earth_cgf)
print(f"\nVelocity std dev:")
print(f"  Incompressible: {v_std_incomp:.2f} km/s")
print(f"  CGF: {v_std_cgf:.2f} km/s")

if v_std_cgf < 1.0:
    print("\n⚠️  WARNING: CGF solution appears flat (std dev < 1 km/s)")
else:
    print(f"\n✓ CGF solution has structure")

plt.show(block=True)
