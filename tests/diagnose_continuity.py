"""
Diagnose what's happening in the upwind solver during evolution.

Check if the density continuity equation is being solved correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import k_B, m_p
from scipy.optimize import bisect
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import huxt as H

# ==============================================================================
# Setup
# ==============================================================================
U0 = 400  # km/s
n_rho0 = 20  # cm^-3
m_rho0 = (m_p * n_rho0 / (u.cm**3)).to(u.kg / u.m**3).value
T0 = 1e6  # K
gamma = 1.5
R_gas = (k_B / m_p).value

r_min = 20
r_max = 220

# ==============================================================================
# Analytical Parker solution
# ==============================================================================
x = np.arange(r_min, r_max, 1.0)
M0 = U0 * 1000 / np.sqrt(gamma * R_gas * T0)
T_t = T0 * (1 + ((gamma - 1)/2) * M0**2)
p_t = m_rho0 * R_gas * T0 * ((1 + ((gamma - 1)/2) * M0**2) ** (gamma/(gamma - 1)))
rho_t = m_rho0 * ((1 + ((gamma - 1)/2) * M0**2) ** (1/(gamma - 1)))

def A_norm_calc(M, gamma):
    a = 2 / (gamma + 1)
    b = (gamma - 1) / 2
    c = (gamma + 1) / (2*(gamma - 1))
    return (1/M) * (a * (1 + b*M**2))**c

A0 = x[0]**2
A0_norm = A_norm_calc(M0, gamma)
A_star = A0 / A0_norm
A = x**2
A_norm = A / A_star

M = np.zeros(len(A_norm))
for i, a_n in enumerate(A_norm):
    M[i] = bisect(lambda M, g, a: A_norm_calc(M, g) - a, 1 + 1e-12, 1e4, args=(gamma, a_n))

T_ana = T_t / (1 + ((gamma - 1)/2) * M**2)
rho_ana = rho_t * (T_ana/T_t) ** (1/(gamma - 1))
c = np.sqrt(gamma * R_gas * T_ana)
U_ana = M * c / 1000
n_ana = rho_ana / (m_p.value) / 1e6

# ==============================================================================
# Run HUXt upwind solver
# ==============================================================================
print("Running HUXt upwind solver...")

v_bound = np.ones(128) * U0
rho_bound = np.ones(128) * m_rho0
temp_bound = np.ones(128) * T0

model = H.HUXt(
    v_boundary=v_bound * u.km/u.s,
    rho_boundary=rho_bound * (u.kg/u.m**3),
    temp_boundary=temp_bound * u.K,
    cr_num=2063,
    r_min=r_min * u.solRad,
    r_max=r_max * u.solRad,
    lon_start=0 * u.deg,
    lon_stop=360 * u.deg,
    simtime=1 * u.day,
    dt_scale=4,
    compressible=True,
    solver='upwind'
)

model.gamma = gamma
model.alpha = 0.0
model.solve([])

r_huxt = model.r.to(u.solRad).value
v_huxt = model.v_grid.value[-1, :, 0]
rho_huxt = model.rho_grid.value[-1, :, 0]
T_huxt = model.temp_grid.value[-1, :, 0]
n_huxt = rho_huxt / (m_p.value) / 1e6

print("Done!")

# ==============================================================================
# Check if continuity is satisfied: ∂ρ/∂t + ∂(ρv)/∂r + 2ρv/r = 0
# At steady state: ∂(ρv)/∂r + 2ρv/r = 0
# Or equivalently: d(ρvr²)/dr = 0, i.e., ρvr² = const
# ==============================================================================

print("\n" + "="*70)
print("Checking if continuity equation is satisfied")
print("="*70)

# For analytical solution
flux_ana = rho_ana * U_ana * 1000 * x**2  # kg/s (convert km/s to m/s)
flux_ana_normalized = flux_ana / flux_ana[0]

# For HUXt solution  
flux_huxt = rho_huxt * v_huxt * 1000 * r_huxt**2
flux_huxt_normalized = flux_huxt / flux_huxt[0]

print(f"\nMass flux ρvr² variation:")
print(f"  Analytical: {np.min(flux_ana_normalized):.6f} - {np.max(flux_ana_normalized):.6f}")
print(f"  HUXt:       {np.min(flux_huxt_normalized):.6f} - {np.max(flux_huxt_normalized):.6f}")
print(f"  Analytical std dev: {np.std(flux_ana_normalized):.2e}")
print(f"  HUXt std dev:       {np.std(flux_huxt_normalized):.2e}")

# ==============================================================================
# Plot diagnostics
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Density comparison
ax = axes[0, 0]
ax.plot(x, n_ana, 'k-', linewidth=3, label='Parker analytical')
ax.plot(r_huxt, n_huxt, 'r--', linewidth=2, label='HUXt upwind')
ax.set_ylabel('Number Density (cm⁻³)', fontsize=12)
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_title('Density Evolution', fontsize=14, fontweight='bold')

# Density error
ax = axes[0, 1]
n_ana_interp = np.interp(r_huxt, x, n_ana)
n_err = (n_huxt - n_ana_interp) / n_ana_interp * 100
ax.plot(r_huxt, n_err, 'r-', linewidth=2)
ax.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
ax.set_ylabel('Density Error (%)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_title(f'RMS Error: {np.sqrt(np.mean(n_err**2)):.1f}%', fontsize=14)

# Mass flux ρvr²
ax = axes[1, 0]
ax.plot(x, flux_ana_normalized, 'k-', linewidth=3, label='Parker analytical')
ax.plot(r_huxt, flux_huxt_normalized, 'r--', linewidth=2, label='HUXt upwind')
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('Mass Flux ρvr² (normalized)', fontsize=12)
ax.set_xlabel('Radius (R☉)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_title('Continuity Check: ρvr² = const?', fontsize=14, fontweight='bold')

# Mass flux error
ax = axes[1, 1]
flux_err = (flux_huxt_normalized - 1.0) * 100
ax.plot(r_huxt, flux_err, 'r-', linewidth=2)
ax.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
ax.set_ylabel('Mass Flux Deviation (%)', fontsize=12)
ax.set_xlabel('Radius (R☉)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_title(f'Max Deviation: {np.max(np.abs(flux_err)):.2f}%', fontsize=14)

plt.tight_layout()
plt.savefig('continuity_diagnostic.png', dpi=150)
print(f"\nPlot saved to: continuity_diagnostic.png")

plt.show()
