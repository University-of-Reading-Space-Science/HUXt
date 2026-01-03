"""
Script to analyze OMNI data and determine empirical relationships between
temperature, density, and velocity for Parker solar wind model.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
import sys
import os

# Add parent directory to path to import huxt modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import huxt.huxt_insitu as HI

# Get all OMNI data from 1994 to now
print("Fetching OMNI data from 1994 to present...")
start_time = Time('1994-01-01')
end_time = Time.now()

omni_data = HI.get_omni(start_time, end_time)

print(f"Loaded {len(omni_data)} OMNI data points")

# Load Richardson & Cane ICME list
print("Loading Richardson & Cane ICME list...")
icmes = HI.ICMElist()
print(f"Loaded {len(icmes)} ICMEs")

# Remove ICMEs from OMNI data (don't interpolate, just remove)
print("Removing ICMEs from OMNI data...")
omni_data = HI.remove_ICMEs(omni_data, icmes, interpolate=False)
print(f"Data points after removing ICMEs: {len(omni_data)}")

print(f"Columns: {omni_data.columns.tolist()}")

# Extract relevant quantities
v = omni_data['V'].values  # km/s
T = omni_data['T'].values  # K
n = omni_data['N'].values  # cm^-3
B = omni_data['ABS_B'].values  # nT

# Remove NaN values
valid = np.isfinite(v) & np.isfinite(T) & np.isfinite(n) & np.isfinite(B)
v_clean = v[valid]
T_clean = T[valid]
n_clean = n[valid]
B_clean = B[valid]

print(f"Valid data points after removing NaNs: {len(v_clean)}")

# Use all non-ICME data (no quasi-steady wind filter)
v_filtered = v_clean
T_filtered = T_clean
n_filtered = n_clean
B_filtered = B_clean

print(f"Using all {len(v_filtered)} non-ICME data points")

# ==============================================================================
# Map to 0.1 AU using Parker nozzle solution FIRST
# ==============================================================================

print("\n=== Mapping to 0.1 AU using Parker nozzle solution ===")

from astropy.constants import k_B, m_p as m_proton

def compute_parker_nozzle_ratio(r1_AU, r2_AU, v1, T1, n1, gamma=1.5):
    """
    Compute the ratio of quantities at two radii using Parker solution scaling.
    
    For consistency with HUXt's compressible solver (adiabatic evolution):
    - Density: n ∝ 1/r² (mass conservation)
    - Temperature: T ∝ r^(-2(γ-1)) = r^(-1) for γ=1.5 (adiabatic)
    - Velocity: approximately constant in supersonic region
    
    Args:
        r1_AU: Initial radius (in AU)
        r2_AU: Final radius (in AU)
        v1: Velocity at r1 (km/s)
        T1: Temperature at r1 (K)
        n1: Number density at r1 (cm^-3)
        gamma: Adiabatic index (default 1.5)
        
    Returns:
        v2, T2, n2: Quantities at r2
    """
    r_ratio = r1_AU / r2_AU
    
    # Adiabatic scaling for consistency with HUXt
    v2 = v1  # Velocity approximately constant
    T2 = T1 * r_ratio**(2*(gamma-1))  # T ∝ r^(-1) for γ=1.5
    n2 = n1 * r_ratio**2  # n ∝ 1/r²
    
    return v2, T2, n2

gamma = 1.5
print(f"Using gamma = {gamma}")
print("Using adiabatic scaling: T ∝ r^(-1), n ∝ r^(-2)")

v_at_01AU = []
T_at_01AU = []
n_at_01AU = []

print("Mapping data points to 0.1 AU...")
for i in range(len(v_filtered)):
    v2, T2, n2 = compute_parker_nozzle_ratio(
        1.0, 0.1, v_filtered[i], T_filtered[i], n_filtered[i], gamma
    )
    v_at_01AU.append(v2)
    T_at_01AU.append(T2)
    n_at_01AU.append(n2)

v_at_01AU = np.array(v_at_01AU)
T_at_01AU = np.array(T_at_01AU)
n_at_01AU = np.array(n_at_01AU)

print(f"Completed mapping {len(v_at_01AU)} data points")

# ==============================================================================
# NOW bin the 0.1 AU data by velocity to avoid weighting bias
# ==============================================================================

print("\n=== Binning 0.1 AU data by velocity ===")

n_bins = 20
v_min, v_max = 250, 850  # km/s
v_bin_edges = np.linspace(v_min, v_max, n_bins + 1)
v_bin_centers = (v_bin_edges[:-1] + v_bin_edges[1:]) / 2

# Compute mean values in each bin for 0.1 AU data
T_binned_01AU = []
n_binned_01AU = []
v_binned_01AU = []
T_std_binned_01AU = []
n_std_binned_01AU = []
bin_counts_01AU = []

for i in range(n_bins):
    mask = (v_at_01AU >= v_bin_edges[i]) & (v_at_01AU < v_bin_edges[i+1])
    if np.sum(mask) > 10:  # Require at least 10 points per bin
        T_binned_01AU.append(np.median(T_at_01AU[mask]))
        n_binned_01AU.append(np.median(n_at_01AU[mask]))
        v_binned_01AU.append(v_bin_centers[i])
        T_std_binned_01AU.append(np.std(T_at_01AU[mask]))
        n_std_binned_01AU.append(np.std(n_at_01AU[mask]))
        bin_counts_01AU.append(np.sum(mask))

T_binned_01AU = np.array(T_binned_01AU)
n_binned_01AU = np.array(n_binned_01AU)
v_binned_01AU = np.array(v_binned_01AU)
T_std_binned_01AU = np.array(T_std_binned_01AU)
n_std_binned_01AU = np.array(n_std_binned_01AU)
bin_counts_01AU = np.array(bin_counts_01AU)

print(f"Created {len(v_binned_01AU)} velocity bins with data at 0.1 AU")
print(f"Velocity range: {v_binned_01AU[0]:.0f} - {v_binned_01AU[-1]:.0f} km/s")

# ==============================================================================
# Create plots
# ==============================================================================

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel (a): All 1 AU data - T vs v (scatter)
ax = axes[0, 0]
ax.scatter(v_filtered, T_filtered, s=1, alpha=0.05, c='lightblue')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Temperature (K)')
ax.set_title('(a) All OMNI Data at 1 AU: T vs v')
ax.set_xlim([250, 900])
ax.set_ylim([0, 5e5])
ax.grid(True, alpha=0.3)

# Panel (b): Binned 0.1 AU data - T vs v with error bars
ax = axes[0, 1]
ax.errorbar(v_binned_01AU, T_binned_01AU, yerr=T_std_binned_01AU, fmt='o', color='red', 
            markersize=6, capsize=3, label='Binned mean ± std (0.1 AU)')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Temperature (K)')
ax.set_title('(b) Binned Data at 0.1 AU (20 bins): T vs v')
ax.set_xlim([250, 900])
ax.set_ylim([0, 3e6])
ax.grid(True, alpha=0.3)

# Try multiple functional forms for temperature
from scipy.optimize import curve_fit

# Define functional forms
def linear(v, a, b):
    return a * v + b

def quadratic(v, a, b, c):
    return a * v**2 + b * v + c

def power_law(v, a, b, n):
    return a * v**n + b

# Fit binned 0.1 AU data
v_fit = np.linspace(250, 900, 100)

# Fit binned data at 0.1 AU
coeffs_binned_01AU = np.polyfit(v_binned_01AU, T_binned_01AU, 1)
T_fit_binned = np.polyval(coeffs_binned_01AU, v_fit)

# Panel (c): All 1 AU data - n vs v (scatter)
ax = axes[1, 0]
ax.scatter(v_filtered, n_filtered, s=1, alpha=0.05, c='lightblue')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Density (cm$^{-3}$)')
ax.set_title('(c) All OMNI Data at 1 AU: n vs v')
ax.set_xlim([250, 900])
ax.set_ylim([0, 50])
ax.grid(True, alpha=0.3)

# Panel (d): Binned 0.1 AU data - n vs v with error bars
ax = axes[1, 1]
ax.errorbar(v_binned_01AU, n_binned_01AU, yerr=n_std_binned_01AU, fmt='o', color='red',
            markersize=6, capsize=3, label='Binned mean ± std (0.1 AU)')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Density (cm$^{-3}$)')
ax.set_title('(d) Binned Data at 0.1 AU (20 bins): n vs v')
ax.set_xlim([250, 900])
ax.set_ylim([0, 5000])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('omni_empirical_relations.png', dpi=150)
print("Saved plot to omni_empirical_relations.png")

# Print statistics and compare fits
print("\n=== Statistics (Binned 0.1 AU Data) ===")
print(f"Number of points in each bin: {bin_counts_01AU.min():.0f} - {bin_counts_01AU.max():.0f}")
print(f"Linear fit: T = {coeffs_binned_01AU[0]:.2f}*v + {coeffs_binned_01AU[1]:.0f}")

# Try additional fits on binned 0.1 AU data to find best form
try:
    # Quadratic fit
    popt_quad, _ = curve_fit(quadratic, v_binned_01AU, T_binned_01AU, p0=[0.1, 100, 0])
    T_quad = quadratic(v_binned_01AU, *popt_quad)
    residuals_quad = T_binned_01AU - T_quad
    rmse_quad = np.sqrt(np.mean(residuals_quad**2))
    
    # Power law fit (with bounds to ensure physical behavior)
    popt_pow, _ = curve_fit(power_law, v_binned_01AU, T_binned_01AU, 
                            p0=[1000, 0, 2], bounds=([0, -1e6, 1.5], [1e6, 1e6, 3]))
    T_pow = power_law(v_binned_01AU, *popt_pow)
    residuals_pow = T_binned_01AU - T_pow
    rmse_pow = np.sqrt(np.mean(residuals_pow**2))
    
    # Linear RMSE
    T_lin = coeffs_binned_01AU[0] * v_binned_01AU + coeffs_binned_01AU[1]
    residuals_lin = T_binned_01AU - T_lin
    rmse_lin = np.sqrt(np.mean(residuals_lin**2))
    
    print(f"\n=== Fit Comparison (Binned 0.1 AU Data) ===")
    print(f"Linear:    T = {coeffs_binned_01AU[0]:.2f}*v + {coeffs_binned_01AU[1]:.0f}")
    print(f"           RMSE = {rmse_lin:.0f} K")
    print(f"           At 300 km/s: {coeffs_binned_01AU[0]*300 + coeffs_binned_01AU[1]:.0f} K (0.1 AU)")
    print(f"           At 300 km/s: {(coeffs_binned_01AU[0]*300 + coeffs_binned_01AU[1])/10:.0f} K (1 AU)")
    print(f"           At 600 km/s: {coeffs_binned_01AU[0]*600 + coeffs_binned_01AU[1]:.0f} K (0.1 AU)")
    print(f"           At 600 km/s: {(coeffs_binned_01AU[0]*600 + coeffs_binned_01AU[1])/10:.0f} K (1 AU)")
    
    print(f"\nQuadratic: T = {popt_quad[0]:.4f}*v^2 + {popt_quad[1]:.2f}*v + {popt_quad[2]:.0f}")
    print(f"           RMSE = {rmse_quad:.0f} K")
    print(f"           At 300 km/s: {quadratic(300, *popt_quad):.0f} K (0.1 AU), {quadratic(300, *popt_quad)/10:.0f} K (1 AU)")
    print(f"           At 600 km/s: {quadratic(600, *popt_quad):.0f} K (0.1 AU), {quadratic(600, *popt_quad)/10:.0f} K (1 AU)")
    
    print(f"\nPower law: T = {popt_pow[0]:.2f}*v^{popt_pow[2]:.3f} + {popt_pow[1]:.0f}")
    print(f"           RMSE = {rmse_pow:.0f} K")
    print(f"           At 300 km/s: {power_law(300, *popt_pow):.0f} K (0.1 AU), {power_law(300, *popt_pow)/10:.0f} K (1 AU)")
    print(f"           At 600 km/s: {power_law(600, *popt_pow):.0f} K (0.1 AU), {power_law(600, *popt_pow)/10:.0f} K (1 AU)")
    
    print(f"\n=== Best Fit: {'Linear' if rmse_lin <= min(rmse_quad, rmse_pow) else 'Quadratic' if rmse_quad <= rmse_pow else 'Power law'} (lowest RMSE) ===")
    
    # Store best fit coefficients for later use
    best_fit_coeffs = {'linear': coeffs_binned_01AU, 'quadratic': popt_quad, 'power': popt_pow}
    
    # Add fits to plots
    ax = axes[0, 1]
    ax.plot(v_fit, coeffs_binned_01AU[0]*v_fit + coeffs_binned_01AU[1], 'b-', linewidth=2, alpha=0.7, label='Linear')
    ax.plot(v_fit, quadratic(v_fit, *popt_quad), 'g--', linewidth=2, alpha=0.7, label='Quadratic')
    ax.plot(v_fit, power_law(v_fit, *popt_pow), 'r-.', linewidth=2, alpha=0.7, label='Power law')
    ax.legend()
    
except Exception as e:
    print(f"Warning: Could not fit alternative forms: {e}")
    best_fit_coeffs = None

# Fit density at 0.1 AU with multiple functional forms
print(f"\n=== Density Fit at 0.1 AU ===")

try:
    # Inverse square (assumes constant mass flux)
    def inv_sq(v, A):
        return A / v**2
    popt_inv_sq, _ = curve_fit(inv_sq, v_binned_01AU, n_binned_01AU)
    n_inv_sq = inv_sq(v_binned_01AU, *popt_inv_sq)
    residuals_inv_sq = n_binned_01AU - n_inv_sq
    rmse_inv_sq = np.sqrt(np.mean(residuals_inv_sq**2))
    
    # Linear fit
    def linear_n(v, a, b):
        return a * v + b
    popt_lin_n, _ = curve_fit(linear_n, v_binned_01AU, n_binned_01AU)
    n_lin = linear_n(v_binned_01AU, *popt_lin_n)
    residuals_lin_n = n_binned_01AU - n_lin
    rmse_lin_n = np.sqrt(np.mean(residuals_lin_n**2))
    
    # Quadratic fit
    def quadratic_n(v, a, b, c):
        return a * v**2 + b * v + c
    popt_quad_n, _ = curve_fit(quadratic_n, v_binned_01AU, n_binned_01AU)
    n_quad = quadratic_n(v_binned_01AU, *popt_quad_n)
    residuals_quad_n = n_binned_01AU - n_quad
    rmse_quad_n = np.sqrt(np.mean(residuals_quad_n**2))
    
    # Power law fit
    def power_law_n(v, a, n, b):
        return a * v**n + b
    popt_pow_n, _ = curve_fit(power_law_n, v_binned_01AU, n_binned_01AU, 
                              p0=[1e5, -2, 0], bounds=([0, -5, -1000], [1e8, 0, 1000]))
    n_pow = power_law_n(v_binned_01AU, *popt_pow_n)
    residuals_pow_n = n_binned_01AU - n_pow
    rmse_pow_n = np.sqrt(np.mean(residuals_pow_n**2))
    
    print(f"Inverse square: n = {popt_inv_sq[0]:.2e} / v^2")
    print(f"                RMSE = {rmse_inv_sq:.1f} cm^-3")
    print(f"                At 300 km/s: {inv_sq(300, *popt_inv_sq):.0f} cm^-3")
    print(f"                At 600 km/s: {inv_sq(600, *popt_inv_sq):.0f} cm^-3")
    
    print(f"\nLinear:         n = {popt_lin_n[0]:.4f}*v + {popt_lin_n[1]:.2f}")
    print(f"                RMSE = {rmse_lin_n:.1f} cm^-3")
    print(f"                At 300 km/s: {linear_n(300, *popt_lin_n):.0f} cm^-3")
    print(f"                At 600 km/s: {linear_n(600, *popt_lin_n):.0f} cm^-3")
    
    print(f"\nQuadratic:      n = {popt_quad_n[0]:.6f}*v^2 + {popt_quad_n[1]:.4f}*v + {popt_quad_n[2]:.2f}")
    print(f"                RMSE = {rmse_quad_n:.1f} cm^-3")
    print(f"                At 300 km/s: {quadratic_n(300, *popt_quad_n):.0f} cm^-3")
    print(f"                At 600 km/s: {quadratic_n(600, *popt_quad_n):.0f} cm^-3")
    
    print(f"\nPower law:      n = {popt_pow_n[0]:.2e}*v^{popt_pow_n[1]:.3f} + {popt_pow_n[2]:.2f}")
    print(f"                RMSE = {rmse_pow_n:.1f} cm^-3")
    print(f"                At 300 km/s: {power_law_n(300, *popt_pow_n):.0f} cm^-3")
    print(f"                At 600 km/s: {power_law_n(600, *popt_pow_n):.0f} cm^-3")
    
    print(f"\n=== Best Density Fit: ", end="")
    min_rmse = min(rmse_inv_sq, rmse_lin_n, rmse_quad_n, rmse_pow_n)
    if min_rmse == rmse_inv_sq:
        print(f"Inverse square (RMSE = {rmse_inv_sq:.1f}) ===")
        best_n_fit = ('inv_sq', popt_inv_sq)
    elif min_rmse == rmse_lin_n:
        print(f"Linear (RMSE = {rmse_lin_n:.1f}) ===")
        best_n_fit = ('linear', popt_lin_n)
    elif min_rmse == rmse_quad_n:
        print(f"Quadratic (RMSE = {rmse_quad_n:.1f}) ===")
        best_n_fit = ('quadratic', popt_quad_n)
    else:
        print(f"Power law (RMSE = {rmse_pow_n:.1f}) ===")
        best_n_fit = ('power', popt_pow_n)
    
    # Add density fits to plot
    ax = axes[1, 1]
    v_plot = np.linspace(250, 900, 100)
    ax.plot(v_plot, inv_sq(v_plot, *popt_inv_sq), 'b:', linewidth=2, alpha=0.7, label='Inverse square')
    ax.plot(v_plot, linear_n(v_plot, *popt_lin_n), 'g--', linewidth=2, alpha=0.7, label='Linear')
    ax.plot(v_plot, quadratic_n(v_plot, *popt_quad_n), 'm-.', linewidth=2, alpha=0.7, label='Quadratic')
    ax.plot(v_plot, power_law_n(v_plot, *popt_pow_n), 'r-', linewidth=2, alpha=0.7, label='Power law')
    ax.legend()
    
except Exception as e:
    print(f"Warning: Could not fit density forms: {e}")
    best_n_fit = None

print(f"\nMean values (all non-ICME data at 1 AU):")
print(f"  v = {np.mean(v_filtered):.1f} ± {np.std(v_filtered):.1f} km/s")
print(f"  T = {np.mean(T_filtered):.1e} ± {np.std(T_filtered):.1e} K")
print(f"  n = {np.mean(n_filtered):.2f} ± {np.std(n_filtered):.2f} cm^-3")
print(f"  B = {np.mean(B_filtered):.2f} ± {np.std(B_filtered):.2f} nT")

print(f"\nMean values (binned data at 0.1 AU):")
print(f"  v = {np.mean(v_binned_01AU):.1f} ± {np.std(v_binned_01AU):.1f} km/s")
print(f"  T = {np.mean(T_binned_01AU):.1e} ± {np.std(T_binned_01AU):.1e} K")
print(f"  n = {np.mean(n_binned_01AU):.2f} ± {np.std(n_binned_01AU):.2f} cm^-3")

# Create combined summary plot with both temperature and density fits at 0.1 AU
fig2, (ax_T, ax_n) = plt.subplots(2, 1, figsize=(10, 10))

# Temperature panel
ax_T.errorbar(v_binned_01AU, T_binned_01AU, yerr=T_std_binned_01AU, fmt='o', color='red',
             markersize=8, capsize=4, label='Binned data (0.1 AU)', zorder=3)
if best_fit_coeffs is not None:
    ax_T.plot(v_fit, power_law(v_fit, *popt_pow), 'r-', linewidth=3, label=f'Power law (best fit)', zorder=2)
    ax_T.plot(v_fit, quadratic(v_fit, *popt_quad), 'g--', linewidth=2, alpha=0.7, label='Quadratic', zorder=1)
    
    # Add equation to plot
    a, n, b = popt_pow
    eq_text = f'T = {a:.2f}$v^{{{n:.3f}}}$ + {b:.0f} K'
    ax_T.text(0.05, 0.95, eq_text, transform=ax_T.transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax_T.set_xlabel('Velocity (km/s)', fontsize=12)
ax_T.set_ylabel('Temperature at 0.1 AU (K)', fontsize=12)
ax_T.set_title('Temperature vs Velocity at 0.1 AU (21.5 Rs)', fontsize=14)
ax_T.grid(True, alpha=0.3)
ax_T.legend(fontsize=11)

# Density panel
ax_n.errorbar(v_binned_01AU, n_binned_01AU, yerr=n_std_binned_01AU, fmt='o', color='blue',
             markersize=8, capsize=4, label='Binned data (0.1 AU)', zorder=3)
if best_n_fit is not None:
    v_plot = np.linspace(250, 900, 100)
    if best_n_fit[0] == 'quadratic':
        ax_n.plot(v_plot, quadratic_n(v_plot, *best_n_fit[1]), 'm-', linewidth=3, 
                 label='Quadratic (best fit)', zorder=2)
        # Add equation
        a_n, b_n, c_n = best_n_fit[1]
        eq_text = f'n = {a_n:.6f}$v^2$ + {b_n:.4f}$v$ + {c_n:.2f} cm$^{{-3}}$'
        ax_n.text(0.05, 0.95, eq_text, transform=ax_n.transAxes, 
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    elif best_n_fit[0] == 'power':
        ax_n.plot(v_plot, power_law_n(v_plot, *best_n_fit[1]), 'r-', linewidth=3, 
                 label='Power law (best fit)', zorder=2)
    elif best_n_fit[0] == 'linear':
        ax_n.plot(v_plot, linear_n(v_plot, *best_n_fit[1]), 'g-', linewidth=3, 
                 label='Linear (best fit)', zorder=2)
    elif best_n_fit[0] == 'inv_sq':
        ax_n.plot(v_plot, inv_sq(v_plot, *best_n_fit[1]), 'b-', linewidth=3, 
                 label='Inverse square (best fit)', zorder=2)
    
    # Show all other fits as dashed lines
    ax_n.plot(v_plot, inv_sq(v_plot, *popt_inv_sq), 'b:', linewidth=1.5, alpha=0.5, label='Inverse square')
    ax_n.plot(v_plot, linear_n(v_plot, *popt_lin_n), 'g--', linewidth=1.5, alpha=0.5, label='Linear')
    ax_n.plot(v_plot, quadratic_n(v_plot, *popt_quad_n), 'm-.', linewidth=1.5, alpha=0.5, label='Quadratic')
    ax_n.plot(v_plot, power_law_n(v_plot, *popt_pow_n), 'r:', linewidth=1.5, alpha=0.5, label='Power law')

ax_n.set_xlabel('Velocity (km/s)', fontsize=12)
ax_n.set_ylabel('Density at 0.1 AU (cm$^{-3}$)', fontsize=12)
ax_n.set_title('Density vs Velocity at 0.1 AU (21.5 Rs)', fontsize=14)
ax_n.grid(True, alpha=0.3)
ax_n.legend(fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig('omni_empirical_relations_01AU.png', dpi=150)
print("\nSaved 0.1 AU plot to omni_empirical_relations_01AU.png")

# ==============================================================================
# Save lookup table for HUXt
# ==============================================================================
print("\n=== Saving lookup table for HUXt ===")

# Save the binned median values as a lookup table
lookup_table = np.column_stack((v_binned_01AU, n_binned_01AU, T_binned_01AU))
np.savetxt('omni_lookup_table_01AU.txt', lookup_table, 
           header='Velocity (km/s), Density (cm^-3), Temperature (K) at 0.1 AU\n'
                  'Median values from OMNI 1994-present, mapped to 0.1 AU using adiabatic Parker solution\n'
                  'Use linear interpolation between values',
           fmt='%.6f')

print(f"Saved lookup table with {len(v_binned_01AU)} velocity bins")
print(f"Velocity range: {v_binned_01AU[0]:.0f} - {v_binned_01AU[-1]:.0f} km/s")
print(f"File: omni_lookup_table_01AU.txt")
