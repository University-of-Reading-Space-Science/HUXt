"""
Comparison script for HUXt solvers: upwind-incompressible, CGF, and PLUTO

Runs all three solvers on the same CME scenario and compares:
- Velocity at 1 AU over time
- Density at 1 AU over time
- Temperature at 1 AU over time

Includes execution timing (with JIT compilation dummy run for upwind-incompressible)
"""
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import time as time_module

import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin

print("\n" + "="*80)
print("HUXt SOLVER COMPARISON: Upwind-Incompressible vs CGF vs PLUTO")
print("="*80)

# Setup parameters
cr = 1920
simtime = 10 * u.day
dt_scale = 4

# Get background wind profile
print("\nLoading MAS wind profile for CR", cr, "...")
vr_in = Hin.get_MAS_long_profile(cr, 0.0*u.deg)

# Define CME
print("Setting up CME...")
cme = H.ConeCME(
    t_launch=0*u.day,
    longitude=0*u.deg,
    width=60*u.deg,
    v=1500*u.km/u.s,
    thickness=5*u.solRad,
    density_fraction=0.05,
    temperature_fraction=0.0001,
    profile_type='sinusoidal'
)
cme_list = [cme]

# ============================================================================
# 1. UPWIND-INCOMPRESSIBLE WITH JIT COMPILATION DUMMY RUN
# ============================================================================
print("\n" + "="*80)
print("1. UPWIND-INCOMPRESSIBLE SOLVER (with JIT compilation dummy run)")
print("="*80)

# Dummy run for JIT compilation (minimal parameters)
print("\n  Running dummy JIT compilation warm-up...")
dummy_v = np.ones(128) * 400 * (u.km/u.s)
model_dummy = H.HUXt(
    v_boundary=dummy_v,
    lon_out=0.0*u.rad,
    simtime=simtime,
    dt_scale=4,
    compressible=False,
    solver='upwind'
)
dummy_start = time_module.time()
model_dummy.solve(cme_list)
dummy_time = time_module.time() - dummy_start
print(f"  Dummy run completed in {dummy_time:.2f} seconds")

# Actual run
print("\n  Running actual upwind-incompressible solver...")
model_incomp = H.HUXt(
    cr_num=cr,
    v_boundary=vr_in,
    lon_out=0.0*u.rad,
    simtime=simtime,
    dt_scale=dt_scale,
    compressible=False,
    solver='upwind'
)
incomp_start = time_module.time()
model_incomp.solve(cme_list)
incomp_time = time_module.time() - incomp_start
print(f"✓ Upwind-incompressible completed in {incomp_time:.2f} seconds")

# Extract Earth timeseries
ts_incomp = HA.get_observer_timeseries(model_incomp, suppress_warning=True)
print(f"  Data points at Earth: {len(ts_incomp)}")

# ============================================================================
# 2. CGF (Conservative Generalized Flux) SOLVER
# ============================================================================
print("\n" + "="*80)
print("2. CGF (Conservative) SOLVER")
print("="*80)

print("\n  Running CGF solver...")
model_cgf = H.HUXt(
    cr_num=cr,
    v_boundary=vr_in,
    lon_out=0.0*u.rad,
    simtime=simtime,
    dt_scale=dt_scale,
    compressible=True,
    solver='cgf'
)
cgf_start = time_module.time()
model_cgf.solve(cme_list)
cgf_time = time_module.time() - cgf_start
print(f"✓ CGF completed in {cgf_time:.2f} seconds")

# Extract Earth timeseries
ts_cgf = HA.get_observer_timeseries(model_cgf, suppress_warning=True)
print(f"  Data points at Earth: {len(ts_cgf)}")

# ============================================================================
# 3. PLUTO SOLVER
# ============================================================================
print("\n" + "="*80)
print("3. PLUTO HYDRODYNAMICS SOLVER")
print("="*80)

print("\n  Running PLUTO solver...")
model_pluto = H.HUXt(
    cr_num=cr,
    v_boundary=vr_in,
    lon_out=0.0*u.rad,
    simtime=simtime,
    dt_scale=dt_scale,
    compressible=True,
    solver='pluto'
)
pluto_start = time_module.time()
model_pluto.solve(cme_list)
pluto_time = time_module.time() - pluto_start
print(f"✓ PLUTO completed in {pluto_time:.2f} seconds")

# Extract Earth timeseries
ts_pluto = HA.get_observer_timeseries(model_pluto, suppress_warning=True)
print(f"  Data points at Earth: {len(ts_pluto)}")

# ============================================================================
# TIMING SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EXECUTION TIMING SUMMARY")
print("="*80)
print(f"  Upwind-incompressible: {incomp_time:8.2f} s (+ {dummy_time:.2f} s JIT)")
print(f"  CGF (conservative):     {cgf_time:8.2f} s")
print(f"  PLUTO:                  {pluto_time:8.2f} s")
print(f"  Total time:             {incomp_time + cgf_time + pluto_time:.2f} s")
print()
print(f"  Speed-up (PLUTO vs CGF):          {cgf_time/pluto_time:.2f}x")
print(f"  Speed-up (PLUTO vs Upwind):       {incomp_time/pluto_time:.2f}x")
print(f"  Speed-up (CGF vs Upwind):         {incomp_time/cgf_time:.2f}x")
print("="*80)

# ============================================================================
# PLOTTING
# ============================================================================
print("\nGenerating comparison plots...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle(f'HUXt Solver Comparison: 1 AU Earth Timeseries\nCR{cr} with 1000 km/s CME', 
             fontsize=14, fontweight='bold')

# Color scheme
colors = {
    'incomp': '#1f77b4',  # blue
    'cgf': '#ff7f0e',      # orange
    'pluto': '#2ca02c'     # green
}

# Plot 1: Velocity
ax = axes[0]
ax.plot(ts_incomp['time'], ts_incomp['vsw'], 'o-', color=colors['incomp'], 
        label=f'Upwind-incomp ({incomp_time:.1f}s)', markersize=4, linewidth=2)
ax.plot(ts_cgf['time'], ts_cgf['vsw'], 's-', color=colors['cgf'], 
        label=f'CGF ({cgf_time:.1f}s)', markersize=4, linewidth=2)
ax.plot(ts_pluto['time'], ts_pluto['vsw'], '^-', color=colors['pluto'], 
        label=f'PLUTO ({pluto_time:.1f}s)', markersize=4, linewidth=2)
ax.set_ylabel('Velocity (km/s)', fontsize=11, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(ts_incomp['time'].min(), ts_incomp['time'].max())

# Plot 2: Density
ax = axes[1]
ax.plot(ts_cgf['time'], ts_cgf['n'], 's-', color=colors['cgf'], 
        label='CGF', markersize=4, linewidth=2)
ax.plot(ts_pluto['time'], ts_pluto['n'], '^-', color=colors['pluto'], 
        label='PLUTO', markersize=4, linewidth=2)
ax.set_ylabel('Density (p/cm³)', fontsize=11, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(ts_incomp['time'].min(), ts_incomp['time'].max())

# Plot 3: Temperature
ax = axes[2]
ax.plot(ts_cgf['time'], ts_cgf['T'], 's-', color=colors['cgf'], 
        label='CGF', markersize=4, linewidth=2)
ax.plot(ts_pluto['time'], ts_pluto['T'], '^-', color=colors['pluto'], 
        label='PLUTO', markersize=4, linewidth=2)
ax.set_ylabel('Temperature (K)', fontsize=11, fontweight='bold')
ax.set_xlabel('Time', fontsize=11, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(ts_incomp['time'].min(), ts_incomp['time'].max())

plt.tight_layout()
plt.savefig('solver_comparison_1au.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved as 'solver_comparison_1au.png'")

# ============================================================================
# STATISTICS
# ============================================================================
print("\n" + "="*80)
print("STATISTICS AT 1 AU")
print("="*80)

print("\nVELOCITY (km/s):")
print(f"  Upwind-incomp: min={ts_incomp['vsw'].min():.1f}, max={ts_incomp['vsw'].max():.1f}, "
      f"mean={ts_incomp['vsw'].mean():.1f}")
print(f"  CGF:           min={ts_cgf['vsw'].min():.1f}, max={ts_cgf['vsw'].max():.1f}, "
      f"mean={ts_cgf['vsw'].mean():.1f}")
print(f"  PLUTO:         min={ts_pluto['vsw'].min():.1f}, max={ts_pluto['vsw'].max():.1f}, "
      f"mean={ts_pluto['vsw'].mean():.1f}")

print("\nDENSITY (p/cm³):")
print(f"  CGF:           min={ts_cgf['n'].min():.2e}, max={ts_cgf['n'].max():.2e}, "
      f"mean={ts_cgf['n'].mean():.2e}")
print(f"  PLUTO:         min={ts_pluto['n'].min():.2e}, max={ts_pluto['n'].max():.2e}, "
      f"mean={ts_pluto['n'].mean():.2e}")

print("\nTEMPERATURE (K):")
print(f"  CGF:           min={ts_cgf['T'].min():.2e}, max={ts_cgf['T'].max():.2e}, "
      f"mean={ts_cgf['T'].mean():.2e}")
print(f"  PLUTO:         min={ts_pluto['T'].min():.2e}, max={ts_pluto['T'].max():.2e}, "
      f"mean={ts_pluto['T'].mean():.2e}")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80 + "\n")

# Display plot
plt.show(block=True)
