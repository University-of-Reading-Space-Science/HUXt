"""
Script to analyze OMNI data and create lookup tables for T(v) and rho(v) at 1 AU.
Includes helium contribution to mass density.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
import sys
import os
# Fallback for this repository layout where modules live under surf/.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
surf_dir = os.path.join(project_root, 'surf')
if surf_dir not in sys.path:
    sys.path.insert(0, surf_dir)

import surf as H
import surf_insitu as HI

# Parse command line arguments
use_local_omni_file = True
# Load OMNI data
omni_data = None

if use_local_omni_file:
    # Try to load from local HDF5 file
    dbox = os.environ.get('DBOX')
    if dbox is None:
        print("Warning: DBOX environment variable not set, cannot load local file")
        use_local_omni_file = False
    else:
        local_file = os.path.join(dbox, 'Data_hdf5', 'omni_1hour.h5')
        if os.path.exists(local_file):
            try:
                import pandas as pd
                import h5py
                print(f"Loading OMNI data from {local_file}...")
                
                # First, list available keys in the HDF5 file
                with h5py.File(local_file, 'r') as f:
                    keys = list(f.keys())
                    print(f"Available keys in HDF5 file: {keys}")
                
                # Try to load with auto-detected key
                if 'omni' in keys:
                    omni_data = pd.read_hdf(local_file, key='omni')
                elif 'OMNI' in keys:
                    omni_data = pd.read_hdf(local_file, key='OMNI')
                elif len(keys) > 0:
                    # Use the first key if standard names don't exist
                    omni_data = pd.read_hdf(local_file, key=keys[0])
                    print(f"Using key '{keys[0]}' from HDF5 file")
                else:
                    raise KeyError("No data keys found in HDF5 file")
                
                print(f"Loaded {len(omni_data)} OMNI data points from local file")
            except Exception as e:
                print(f"Error loading local file: {e}")
                print("Falling back to downloading...")
                omni_data = None
        else:
            print(f"Local file not found: {local_file}")
            print("Falling back to downloading...")
            omni_data = None

if omni_data is None:
    # Download OMNI data from 1994 to now (with fallback to recent years if download fails)
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

# Extract relevant quantities - check for alternative column names
# Velocity: Vr or V
v_col = 'Vr' if 'Vr' in omni_data.columns else 'V'
v = omni_data[v_col].values  # km/s

# Temperature: T_p or T
T_col = 'T_p' if 'T_p' in omni_data.columns else 'T'
T = omni_data[T_col].values  # K

# Number density: n_p or N
n_col = 'n_p' if 'n_p' in omni_data.columns else 'N'
n = omni_data[n_col].values  # proton number density, cm^-3

# Magnetic field: Bmag or ABS_B
B_col = 'Bmag' if 'Bmag' in omni_data.columns else 'ABS_B'
B = omni_data[B_col].values  # nT

print(f"Using columns: v={v_col}, T={T_col}, n={n_col}, B={B_col}")

# Remove NaN values
valid = np.isfinite(v) & np.isfinite(T) & np.isfinite(n) & np.isfinite(B)
v_clean = v[valid]
T_clean = T[valid]
n_clean = n[valid]
B_clean = B[valid]

print(f"Valid data points after removing NaNs: {len(v_clean)}")

# Use all non-ICME data
v_filtered = v_clean
T_filtered = T_clean
n_filtered = n_clean



# ==============================================================================
# Bin the 1 AU data by velocity to avoid weighting bias
# ==============================================================================

print("\n=== Binning 1 AU data by velocity ===")

n_bins = 15
v_min, v_max = 200, 800  # km/s
v_bin_edges = np.linspace(v_min, v_max, n_bins + 1)
v_bin_centers = (v_bin_edges[:-1] + v_bin_edges[1:]) / 2

# Compute median values in each bin for 1 AU data
T_binned = []
n_binned = []
v_binned = []
T_lower_binned = []
T_upper_binned = []
n_lower_binned = []
n_upper_binned = []
bin_counts = []

for i in range(n_bins):
    mask = (v_filtered >= v_bin_edges[i]) & (v_filtered < v_bin_edges[i+1])
    if np.sum(mask) > 10:  # Require at least 10 points per bin
        T_binned.append(np.median(T_filtered[mask]))
        n_binned.append(np.median(n_filtered[mask]))
        v_binned.append(v_bin_centers[i])
        T_lower_binned.append(np.percentile(T_filtered[mask], 5))
        T_upper_binned.append(np.percentile(T_filtered[mask], 95))
        n_lower_binned.append(np.percentile(n_filtered[mask], 5))
        n_upper_binned.append(np.percentile(n_filtered[mask], 95))
        bin_counts.append(np.sum(mask))

T_binned = np.array(T_binned)
n_binned = np.array(n_binned)
v_binned = np.array(v_binned)
T_lower_binned = np.array(T_lower_binned)
T_upper_binned = np.array(T_upper_binned)
n_lower_binned = np.array(n_lower_binned)
n_upper_binned = np.array(n_upper_binned)
bin_counts = np.array(bin_counts)

T_binned = np.array(T_binned)
n_binned = np.array(n_binned)
v_binned = np.array(v_binned)
T_lower_binned = np.array(T_lower_binned)
T_upper_binned = np.array(T_upper_binned)
n_lower_binned = np.array(n_lower_binned)
n_upper_binned = np.array(n_upper_binned)
bin_counts = np.array(bin_counts)

print(f"Created {len(v_binned)} velocity bins with data at 1 AU")
print(f"Velocity range: {v_binned[0]:.0f} - {v_binned[-1]:.0f} km/s")

# ==============================================================================
# Create plots
# ==============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): All 1 AU data - T vs v (2D histogram) with binned values overlaid
ax = axes[0]
h1 = ax.hist2d(v_filtered, T_filtered/1e6, bins=[50, 50], range=[[250, 900], [0, 0.5]], cmap='Blues', norm=LogNorm(vmin=1))
divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("top", size="5%", pad=0.1)
cbar1 = plt.colorbar(h1[3], cax=cax1, orientation='horizontal')
cbar1.ax.xaxis.tick_top()
cbar1.ax.xaxis.set_label_position('top')
cbar1.set_label('Occurrence', fontsize=14)
cbar1.ax.tick_params(labelsize=14)
ax.text(0.03, 0.97, '(a)', transform=ax.transAxes, fontsize=16, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
T_err = [(T_binned - T_lower_binned)/1e6, (T_upper_binned - T_binned)/1e6]
ax.errorbar(v_binned, T_binned/1e6, yerr=T_err, fmt='o', color='red', 
            markersize=6, capsize=3, label='Median (5%, 95% percentiles)')
ax.set_xlabel('Velocity [km/s]', fontsize=14)
ax.set_ylabel('Temperature [MK]', fontsize=14)
ax.set_xlim([250, 900])
ax.set_ylim([0, 0.5])
ax.legend(fontsize=14)
ax.tick_params(labelsize=14)

# Panel (b): All 1 AU data - n vs v (2D histogram) with binned values overlaid
ax = axes[1]
h2 = ax.hist2d(v_filtered, n_filtered, bins=[50, 50], range=[[250, 900], [0, 25]], cmap='Blues', norm=LogNorm(vmin=1))
divider2 = make_axes_locatable(ax)
cax2 = divider2.append_axes("top", size="5%", pad=0.1)
cbar2 = plt.colorbar(h2[3], cax=cax2, orientation='horizontal')
cbar2.ax.xaxis.tick_top()
cbar2.ax.xaxis.set_label_position('top')
cbar2.set_label('Occurrence', fontsize=14)
cbar2.ax.tick_params(labelsize=14)
ax.text(0.03, 0.97, '(b)', transform=ax.transAxes, fontsize=16, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
n_err = [n_binned - n_lower_binned, n_upper_binned - n_binned]
ax.errorbar(v_binned, n_binned, yerr=n_err, fmt='o', color='red', 
            markersize=6, capsize=3, label='Median (5%, 95% percentiles)')
ax.set_xlabel('Velocity [km/s]', fontsize=14)
ax.set_ylabel('Number Density [cm$^{-3}$]', fontsize=14)
ax.set_xlim([250, 900])
ax.set_ylim([0, 25])
ax.legend(fontsize=14)
ax.tick_params(labelsize=14)

# Set consistent formatting for both panels
for ax in axes:
    ax.tick_params(labelsize=14)

plt.tight_layout()

# Save figure to both local directory and Overleaf directory
dbox = os.environ.get('DBOX')
if dbox is None:
    dbox = 'C:\\Users\\mathe\\Dropbox'

# Save PNG locally
plt.savefig('omni_empirical_relations_1AU.png', dpi=150)
print(f"Saved plot to omni_empirical_relations_1AU.png")

# Save PDF to Overleaf directory
overleaf_dir = os.path.join(dbox, 'Apps', 'Overleaf', 'SHUXt')
os.makedirs(overleaf_dir, exist_ok=True)
pdf_path = os.path.join(overleaf_dir, 'omni_empirical_relations_1AU.pdf')
plt.savefig(pdf_path, dpi=150)
print(f"Saved plot to {pdf_path}")

print(f"\nMean values (all non-ICME data at 1 AU):")
print(f"  v = {np.mean(v_filtered):.1f} ± {np.std(v_filtered):.1f} km/s")
print(f"  T = {np.mean(T_filtered):.1e} ± {np.std(T_filtered):.1e} K")
print(f"  n = {np.mean(n_filtered):.2f} ± {np.std(n_filtered):.2f} cm^-3")


print(f"\nMedian and percentile values (binned data at 1 AU):")
print(f"  v = {np.mean(v_binned):.1f} ± {np.std(v_binned):.1f} km/s")
print(f"  T = {np.median(T_binned):.1e} K (5%, 95%: {np.median(T_lower_binned):.1e}, {np.median(T_upper_binned):.1e})")
print(f"  n = {np.median(n_binned):.2f} cm^-3 (5%, 95%: {np.median(n_lower_binned):.2f}, {np.median(n_upper_binned):.2f})")

# ==============================================================================
# Save lookup table for HUXt
# ==============================================================================
print("\n=== Saving lookup table for HUXt ===")

# Save the binned median values as a lookup table
# Columns: velocity, number_density, temperature
lookup_table = np.column_stack((v_binned, n_binned, T_binned))

# Save to huxt/data/insitu directory
huxt_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'surf', 'data', 'insitu')
os.makedirs(huxt_data_dir, exist_ok=True)
lookup_table_path = os.path.join(huxt_data_dir, 'omni_lookup_table_1AU.txt')

np.savetxt(lookup_table_path, lookup_table, 
           header='Velocity (km/s), Number Density (cm^-3), Temperature (K) at 1 AU\n'
                  'Median values from OMNI 1994-present (non-ICME periods)\n'
                  'Use linear interpolation between values',
           fmt='%.6e')

print(f"Saved lookup table with {len(v_binned)} velocity bins")
print(f"Velocity range: {v_binned[0]:.0f} - {v_binned[-1]:.0f} km/s")
print(f"File: {lookup_table_path}")

# Generate LaTeX table
print("\n=== Generating LaTeX table ===")
latex_lines = []
latex_lines.append(r"\begin{table}[h]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{OMNI 1 AU empirical relations: velocity, number density, and temperature}")
latex_lines.append(r"\begin{tabular}{ccc}")
latex_lines.append(r"\hline")
latex_lines.append(r"Velocity (km/s) & Number Density (cm$^{-3}$) & Temperature (K) \\")
latex_lines.append(r"\hline")

for i in range(len(v_binned)):
    latex_lines.append(f"{v_binned[i]:.0f} & {n_binned[i]:.2f} & {T_binned[i]:.2e} \\\\")

latex_lines.append(r"\hline")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\label{tab:omni_1au_relations}")
latex_lines.append(r"\end{table}")

latex_table = "\n".join(latex_lines)

# Save LaTeX table to Overleaf directory
latex_path = os.path.join(overleaf_dir, 'omni_lookup_table_1AU.tex')
with open(latex_path, 'w') as f:
    f.write(latex_table)

print(f"Saved LaTeX table to {latex_path}")

# ==============================================================================
# Map 1 AU values to 0.1 AU using Parker mapping
# ==============================================================================
print("\n=== Mapping 1 AU values to 0.1 AU ===")

# Map each binned value from 1 AU to 0.1 AU
r_outer = 0.1 * u.au
r_inner = 1.0 * u.au

v_01au = []
n_01au = []
T_01au = []

for i in range(len(v_binned)):
    # Convert to astropy units
    v_1au = v_binned[i] * u.km / u.s
    n_1au = n_binned[i] * u.cm**-3
    T_1au = T_binned[i] * u.K
    
    # Map from 1 AU to 0.1 AU using Parker nozzle solution
    v_out, n_out, T_out = H.map_properties_parker(v_1au, r_inner, r_outer, n_1au, T_1au)
    
    # Store values (convert back to base units)
    v_01au.append(v_out.to(u.km / u.s).value)
    n_01au.append(n_out.to(u.cm**-3).value)
    T_01au.append(T_out.to(u.K).value)

v_01au = np.array(v_01au)
n_01au = np.array(n_01au)
T_01au = np.array(T_01au)

print(f"Mapped {len(v_01au)} velocity bins to 0.1 AU")
print(f"Velocity range at 0.1 AU: {v_01au[0]:.0f} - {v_01au[-1]:.0f} km/s")

# Save 0.1 AU lookup table as text file
# Columns: v_1au, velocity at 0.1 AU, Parker-mapped n, Parker-mapped T
lookup_table_01au = np.column_stack((v_binned, v_01au, n_01au, T_01au))
lookup_table_01au_path = os.path.join(overleaf_dir, 'omni_lookup_table_01AU.txt')
np.savetxt(lookup_table_01au_path, lookup_table_01au,
        header='Velocity at 1 AU (km/s), Velocity at 0.1 AU (km/s), Number Density (cm^-3), Temperature (K) at 0.1 AU\n'
            'Parker mapping from 1 AU median OMNI values\n'
            'Use linear interpolation between values',
        fmt='%.6e')

print(f"Saved 0.1 AU lookup table to {lookup_table_01au_path}")

# Generate LaTeX table for 0.1 AU
print("\n=== Generating 0.1 AU LaTeX table ===")
latex_lines_01au = []
latex_lines_01au.append(r"\begin{table}[h]")
latex_lines_01au.append(r"\centering")
latex_lines_01au.append(r"\caption{OMNI 0.1 AU empirical relations: Parker-mapped number density/temperature versus velocity}")
latex_lines_01au.append(r"\begin{tabular}{cccc}")
latex_lines_01au.append(r"\hline")
latex_lines_01au.append(r"$v_{1\,\mathrm{AU}}$ (km/s) & $v_{0.1\,\mathrm{AU}}$ (km/s) & $n$ Parker (cm$^{-3}$) & $T$ Parker (K) \\")
latex_lines_01au.append(r"\hline")

for i in range(len(v_01au)):
    latex_lines_01au.append(f"{v_binned[i]:.0f} & {v_01au[i]:.0f} & {n_01au[i]:.2f} & {T_01au[i]:.2e} \\\\")

latex_lines_01au.append(r"\hline")
latex_lines_01au.append(r"\end{tabular}")
latex_lines_01au.append(r"\label{tab:omni_01au_relations}")
latex_lines_01au.append(r"\end{table}")

latex_table_01au = "\n".join(latex_lines_01au)

# Save 0.1 AU LaTeX table to Overleaf directory
latex_path_01au = os.path.join(overleaf_dir, 'omni_lookup_table_01AU.tex')
with open(latex_path_01au, 'w') as f:
    f.write(latex_table_01au)

print(f"Saved 0.1 AU LaTeX table to {latex_path_01au}")

# ==============================================================================
# New figure: 0.1 AU relations
# ==============================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

# Left panel: v_1au vs v_0.1au with y=x line
ax = axes2[0]
ax.plot(v_01au, v_binned, color='red', linewidth=2, label='Parker mapping')
v_range = [min(v_01au.min(), v_binned.min()), max(v_01au.max(), v_binned.max())]
ax.plot(v_range, v_range, 'k', linewidth=1, label='y = x')
ax.set_xlabel('Velocity at 0.1 AU [km/s]', fontsize=14)
ax.set_ylabel('Velocity at 1 AU [km/s]', fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14)
ax.text(0.03, 0.97, '(a)', transform=ax.transAxes, fontsize=16, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Centre panel: density at 0.1 AU vs v at 0.1 AU
ax = axes2[1]
ax.plot(v_01au, n_01au, color='red', linewidth=2, label='Parker-mapped 1au relations')

# Constant mass flux: n * v = C  =>  n = C / v
C_mf = np.median(n_01au * v_01au)
v_range_01au = np.linspace(v_01au.min(), v_01au.max(), 200)
ax.plot(v_range_01au, C_mf / v_range_01au, color='blue', linewidth=2,
        linestyle='-', label='Const. mass flux ($n v = C$)')

# Constant momentum flux: n * v^2 = C  =>  n = C / v^2
C_mom = np.median(n_01au * v_01au**2)
ax.plot(v_range_01au, C_mom / v_range_01au**2, color='blue', linewidth=2,
        linestyle=':', label='Const. momentum flux ($n v^2 = C$)')

# Constant kinetic energy flux: 0.5 * n * v^3 = C  =>  n = 2C / v^3
C_ke = np.median(0.5 * n_01au * v_01au**3)
ax.plot(v_range_01au, 2 * C_ke / v_range_01au**3, color='blue', linewidth=2,
        linestyle='--', label='Const. KE flux ($\\frac{1}{2}nv^3 = C$)')

ax.set_xlabel('Velocity at 0.1 AU [km/s]', fontsize=14)
ax.set_ylabel('Number Density at 0.1 AU [cm$^{-3}$]', fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14)
ax.text(0.03, 0.97, '(b)', transform=ax.transAxes, fontsize=16, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Right panel: temperature at 0.1 AU vs v at 0.1 AU (in MK)
T_01au_MK = T_01au / 1e6
ax = axes2[2]
ax.plot(v_01au, T_01au_MK, color='red', linewidth=2, label='Parker-mapped 1au relations')

# Constant thermal pressure lines: P = n * T = const  =>  T = C_P / n
# Use median(n * T) as the pressure constant so lines pass through the data
C_P = np.median(n_01au * T_01au_MK)

# Using density from constant mass flux: n_mf = C_mf / v  =>  T = C_P * v / C_mf
ax.plot(v_range_01au, C_P * v_range_01au / C_mf, color='blue', linewidth=2,
        linestyle='-', label='Const. pressure, mass-flux')

# Using density from constant momentum flux: n_mom = C_mom / v^2  =>  T = C_P * v^2 / C_mom
ax.plot(v_range_01au, C_P * v_range_01au**2 / C_mom, color='blue', linewidth=2,
        linestyle=':', label='Const. pressure, momentum-flux')

# Using density from constant KE flux: n_ke = 2*C_ke / v^3  =>  T = C_P * v^3 / (2*C_ke)
ax.plot(v_range_01au, C_P * v_range_01au**3 / (2 * C_ke), color='blue', linewidth=2,
        linestyle='--', label='Const. pressure, KE-flux')

ax.set_xlabel('Velocity at 0.1 AU [km/s]', fontsize=14)
ax.set_ylabel('Temperature at 0.1 AU [MK]', fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14, loc='upper right')
ax.text(0.03, 0.97, '(c)', transform=ax.transAxes, fontsize=16, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

plt.savefig('omni_empirical_relations_01AU.png', dpi=150)
print(f"Saved plot to omni_empirical_relations_01AU.png")

pdf_path_01au_fig = os.path.join(overleaf_dir, 'omni_empirical_relations_01AU.pdf')
plt.savefig(pdf_path_01au_fig, dpi=150)
print(f"Saved plot to {pdf_path_01au_fig}")

plt.show()



