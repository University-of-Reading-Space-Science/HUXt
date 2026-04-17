import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.time import Time as AstroTime
import astropy.units as u
import astropy.constants as const
import os
import pandas as pd
from datetime import datetime
from glob import glob
import xarray as xr
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SURF_DIR = os.path.join(PROJECT_ROOT, 'surf')
if SURF_DIR not in sys.path:
    sys.path.insert(0, SURF_DIR)

import surf_analysis as surfA
import surf_inputs as surfIN
import surf_insitu as surfIS

WSA_ENLIL_dir = os.path.join(os.getenv('DBOX'), 'Data', 'Enlil', 'WSA_ENLIL_LUKE')

# Read in WSA-ENLIL data, file by file, averaging to 1 hr 
output_file = 'enlil_hourly_data.pkl'


def read_enlil_data(wsa_enlil_dir):
    """
    Read WSA-ENLIL netCDF files from year directories, extract speed, density, 
    and temperature, and average to 1-hour resolution.
    
    The WSA-ENLIL files contain:
    - TIME: seconds relative to refdate_mjd
    - D: density in kg/m³
    - T: temperature in K
    - V1: radial velocity in m/s
    
    Parameters:
    -----------
    wsa_enlil_dir : str
        Path to the WSA-ENLIL directory containing year subdirectories
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with datetime index and columns for:
        - speed (km/s) - radial velocity V1
        - density (cm⁻³) - number density
        - temperature (K)
    """
    all_data = []
    
    # Get all year directories
    year_dirs = sorted([d for d in os.listdir(wsa_enlil_dir) 
                       if os.path.isdir(os.path.join(wsa_enlil_dir, d))])
    
    print(f"Found {len(year_dirs)} year directories: {year_dirs}")
    
    file_count = 0
    for year_dir in year_dirs:
        year_path = os.path.join(wsa_enlil_dir, year_dir)
        
        # Get all netCDF files in this year directory
        nc_files = sorted(glob(os.path.join(year_path, '*.nc')))
        
        print(f"\nProcessing {year_dir}: {len(nc_files)} files")
        
        for i, nc_file in enumerate(nc_files):
            try:
                # Read the netCDF file
                ds = xr.open_dataset(nc_file)
                
                # Get reference date from attributes
                if 'refdate_mjd' not in ds.attrs:
                    print(f"Warning: No refdate_mjd in {nc_file}")
                    ds.close()
                    continue
                
                ref_mjd = ds.attrs['refdate_mjd']
                ref_time = AstroTime(ref_mjd, format='mjd')
                ref_datetime = ref_time.datetime
                
                # Extract time in seconds and convert to datetime (vectorized)
                time_seconds = ds['TIME'].values
                # Convert to pandas datetime using TimedeltaIndex for speed
                time_datetime = pd.to_datetime(ref_datetime) + pd.to_timedelta(time_seconds, unit='s')
                
                # Extract radial velocity (V1) and convert m/s to km/s
                v1 = ds['V1'].values / 1000.0  # m/s to km/s
                
                # Extract density and convert kg/m³ to particles/cm³
                # Assuming proton mass for conversion
                proton_mass_kg = const.m_p.value  # kg
                density_kg_m3 = ds['D'].values
                density_cm3 = (density_kg_m3 / proton_mass_kg) / 1e6  # particles/cm³
                
                # Extract temperature (already in K)
                temperature = ds['T'].values
                
                # Create a DataFrame for this file
                file_data = pd.DataFrame({
                    'datetime': time_datetime,
                    'speed': v1,
                    'density': density_cm3,
                    'temperature': temperature
                })
                
                all_data.append(file_data)
                
                ds.close()
                
                file_count += 1
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(nc_files)} files")
                
            except Exception as e:
                print(f"Error processing {os.path.basename(nc_file)}: {e}")
                continue
    
    print(f"\nSuccessfully read {file_count} files")
    
    # Combine all data
    if len(all_data) == 0:
        print("No data was read!")
        return pd.DataFrame()
    
    print("Combining data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Set datetime as index and sort
    print("Sorting by time...")
    combined_df.set_index('datetime', inplace=True)
    combined_df.sort_index(inplace=True)
    
    # Remove duplicates (keep first)
    n_before = len(combined_df)
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    n_after = len(combined_df)
    if n_before != n_after:
        print(f"Removed {n_before - n_after} duplicate time entries")
    
    # Resample to 1-hour average
    print("Resampling to 1-hour average...")
    df_hourly = combined_df.resample('1h').mean()
    
    # Remove rows with all NaN
    df_hourly = df_hourly.dropna(how='all')
    
    print(f"\n=== Summary ===")
    print(f"Total data points: {len(combined_df):,}")
    print(f"Hourly averaged data points: {len(df_hourly):,}")
    print(f"Date range: {df_hourly.index.min()} to {df_hourly.index.max()}")
    print(f"\nData statistics:")
    print(f"  Speed (km/s): mean={df_hourly['speed'].mean():.1f}, "
          f"std={df_hourly['speed'].std():.1f}, "
          f"min={df_hourly['speed'].min():.1f}, "
          f"max={df_hourly['speed'].max():.1f}")
    print(f"  Density (cm⁻³): mean={df_hourly['density'].mean():.2f}, "
          f"std={df_hourly['density'].std():.2f}, "
          f"min={df_hourly['density'].min():.2f}, "
          f"max={df_hourly['density'].max():.2f}")
    print(f"  Temperature (K): mean={df_hourly['temperature'].mean():.0f}, "
          f"std={df_hourly['temperature'].std():.0f}, "
          f"min={df_hourly['temperature'].min():.0f}, "
          f"max={df_hourly['temperature'].max():.0f}")
    
    return df_hourly


if os.path.exists(output_file):
    print(f"Loading processed ENLIL data from {output_file}...")
    enlil_data = pd.read_pickle(output_file)
    print(f"Loaded {len(enlil_data)} hourly data points")
else:
    print("Reading WSA-ENLIL data...")
    enlil_data = read_enlil_data(WSA_ENLIL_dir)
    
    # Save to pickle for faster loading in the future
    enlil_data.to_pickle(output_file)
    print(f"\nSaved processed data to {output_file}")
    
    # Also save to CSV for external use
    csv_file = 'enlil_hourly_data.csv'
    enlil_data.to_csv(csv_file)
    print(f"Saved processed data to {csv_file}")

# Get OMNI data for the same period
print("\n" + "="*60)
print(f"Downloading OMNI data for 2019-2022...")
print("="*60)

# Define time range
starttime = datetime(2019, 1, 1)
endtime = datetime(2023, 1, 1)  # Through end of 2022

omni_data = surfIS.get_omni(starttime, endtime)
#remove ICMEs
icmes = surfIS.ICMElist()
omni_data = surfIS.remove_ICMEs(omni_data, icmes, interpolate=True, icme_buffer=0.1 * u.day, interp_buffer=1 * u.day,
                 params=['V', 'BX_GSE', 'N', 'T'], fill_vals=None)

print(f"\nOMNI data: {len(omni_data)} data points")
print(f"Date range: {omni_data['datetime'].min()} to {omni_data['datetime'].max()}")

# Prepare OMNI data for comparison
# OMNI has columns: V (km/s), N (cm^-3), T (K)
omni_clean = omni_data[['datetime', 'V', 'N', 'T']].copy()
omni_clean.columns = ['datetime', 'speed', 'density', 'temperature']
omni_clean = omni_clean.set_index('datetime')

# Remove NaN values for plotting
enlil_clean = enlil_data.dropna()
omni_clean = omni_clean.dropna()

print(f"\nENLIL clean data: {len(enlil_clean)} points")
print(f"OMNI clean data: {len(omni_clean)} points")

# Use 2019-2022 period for comparison
enlil_period = enlil_clean[(enlil_clean.index >= '2019-01-01') & (enlil_clean.index < '2023-01-01')]
omni_period = omni_clean[(omni_clean.index >= '2019-01-01') & (omni_clean.index < '2023-01-01')]

print(f"\nENLIL 2019-2022 data: {len(enlil_period)} points")
print(f"OMNI 2019-2022 data: {len(omni_period)} points")

# Create comparison histograms
print("\nCreating comparison histograms...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Speed histogram
ax = axes[0]
bins = np.linspace(200, 800, 40)
ax.hist(enlil_clean['speed'], bins=bins, alpha=0.6, label='WSA-ENLIL', 
        density=True, color='blue', edgecolor='black', linewidth=0.5)
ax.hist(omni_clean['speed'], bins=bins, alpha=0.6, label='OMNI', 
        density=True, color='red', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Solar Wind Speed (km/s)', fontsize=12)
ax.set_ylabel('Normalized Frequency', fontsize=12)
ax.set_title('Speed Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, f'ENLIL: {enlil_clean["speed"].mean():.1f}±{enlil_clean["speed"].std():.1f} km/s\n' +
                     f'OMNI: {omni_clean["speed"].mean():.1f}±{omni_clean["speed"].std():.1f} km/s',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Density histogram
ax = axes[1]
bins = np.linspace(0, 20, 40)
ax.hist(enlil_clean['density'], bins=bins, alpha=0.6, label='WSA-ENLIL', 
        density=True, color='blue', edgecolor='black', linewidth=0.5)
ax.hist(omni_clean['density'], bins=bins, alpha=0.6, label='OMNI', 
        density=True, color='red', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Density (cm$^{-3}$)', fontsize=12)
ax.set_ylabel('Normalized Frequency', fontsize=12)
ax.set_title('Density Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.text(0.55, 0.95, f'ENLIL: {enlil_clean["density"].mean():.2f}±{enlil_clean["density"].std():.2f} cm$^{{-3}}$\n' +
                     f'OMNI: {omni_clean["density"].mean():.2f}±{omni_clean["density"].std():.2f} cm$^{{-3}}$',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Temperature histogram
ax = axes[2]
bins = np.linspace(0, 500000, 40)
ax.hist(enlil_clean['temperature'], bins=bins, alpha=0.6, label='WSA-ENLIL', 
        density=True, color='blue', edgecolor='black', linewidth=0.5)
ax.hist(omni_clean['temperature'], bins=bins, alpha=0.6, label='OMNI', 
        density=True, color='red', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Temperature (K)', fontsize=12)
ax.set_ylabel('Normalized Frequency', fontsize=12)
ax.set_title('Temperature Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
ax.text(0.55, 0.95, f'ENLIL: {enlil_clean["temperature"].mean():.0f}±{enlil_clean["temperature"].std():.0f} K\n' +
                     f'OMNI: {omni_clean["temperature"].mean():.0f}±{omni_clean["temperature"].std():.0f} K',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('WSA_ENLIL_vs_OMNI_histograms.png', dpi=300, bbox_inches='tight')
print("Saved figure: WSA_ENLIL_vs_OMNI_histograms.png")
plt.show()



# =============================================================================
# Run Insitu -HUXt for 2019-2022 period using omniHUXt_forecast in 27-day chunks
# =============================================================================
print("\n" + "="*60)
print(f"Running SURF in-situ forecasts for 2019-2022 in 27-day chunks...")
print("="*60)

# Check if pickled SURF data exists
surf_pickle_file = f'surf_2019_2022_data.pkl'
if os.path.exists(surf_pickle_file):
    print(f"Loading SURF 2019-2022 data from {surf_pickle_file}...")
    surf_clean = pd.read_pickle(surf_pickle_file)
    print(f"Loaded {len(surf_clean)} data points")
    
    # Average to 1-hour resolution
    surf_clean = surf_clean.resample('1h').mean()
    surf_clean = surf_clean.dropna()
    print(f"SURF resampled to 1-hour resolution: {len(surf_clean)} points")
else:
    print("No pickle file found. Running SURF forecasts...")
    start_time = datetime(2019, 2, 1)
    end_time = datetime(2023, 1, 1)

    # Run 27-day forecasts throughout 2019-2022
    forecast_duration = 27*u.day
    all_surf_data = []

    current_time = start_time
    chunk_num = 0

    while current_time < end_time:
        chunk_num += 1
        print(f"\nChunk {chunk_num}: Forecast from {current_time.strftime('%Y-%m-%d')}")
        
        # Run forecast
        try:
            is_model = surfIS.omniSURF_forecast(current_time, simtime=forecast_duration,
                                    rmin=21.5*u.solRad, rmax=230*u.solRad, 
                                    dt_scale=4,
                                    omni_input=None,
                                    run_2d=False,
                                    solver='hydro')
            
            is_model.solve([])
            ts_surf_chunk = surfA.get_observer_timeseries(is_model)
            
            # Prepare data in same format as ENLIL/OMNI
            surf_chunk = pd.DataFrame({
                'speed': ts_surf_chunk['vsw'].values,
                'density': ts_surf_chunk['n'].values,
                'temperature': ts_surf_chunk['T'].values
            }, index=pd.to_datetime(ts_surf_chunk['time']))
            
            all_surf_data.append(surf_chunk)
            print(f"  Generated {len(surf_chunk)} data points")
            
        except Exception as e:
            print(f"  Warning: Error in chunk {chunk_num}: {e}")
        
        # Advance to next chunk
        current_time = current_time + pd.Timedelta(days=27)

    # Concatenate all chunks and remove duplicates
    print(f"\nCombining {len(all_surf_data)} forecast chunks...")
    surf_clean = pd.concat(all_surf_data)
    surf_clean = surf_clean.sort_index()

    # Remove duplicate timestamps (keep first)
    surf_clean = surf_clean[~surf_clean.index.duplicated(keep='first')]

    # Filter to 2019-2022 period
    surf_clean = surf_clean[(surf_clean.index >= '2019-02-01') & (surf_clean.index < '2023-01-01')]

    surf_clean = surf_clean.dropna()
    print(f"SURF 2019-2022 data: {len(surf_clean)} points after cleanup")
    
    # Average to 1-hour resolution
    surf_clean = surf_clean.resample('1h').mean()
    surf_clean = surf_clean.dropna()
    print(f"SURF resampled to 1-hour resolution: {len(surf_clean)} points")
    
    # Save to pickle for future use
    surf_clean.to_pickle(surf_pickle_file)
    print(f"Saved SURF 2019-2022 data to {surf_pickle_file}")




# =============================================================================
# =============================================================================
# Create 2D histogram plots: n vs v with T as color
# =============================================================================
print("\nCreating 2D histogram plots (n vs v, colored by T)...")

# Create three-panel side-by-side comparison (only combined plot)
print("\nCreating three-panel comparison...")
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Reduced grid resolution by 2x: 50 -> 25 bins
v_bins = np.linspace(250, 750, 25)
n_bins = np.linspace(0, 20, 20)

# Store all pcolormesh objects to create a single colorbar
pcms = []

for idx, (ax, data, title) in enumerate(zip(axes, [enlil_period, omni_period, surf_clean], 
                            [f'WSA-ENLIL\n(equilibrium BCs)', f'OMNI\n(no ICMEs)', f'InSitu-SURF-hydro\n(non-equilibrium BCs)'])):
    # Calculate occurrence counts
    counts, v_edges, n_edges = np.histogram2d(
        data['speed'], data['density'], bins=[v_bins, n_bins]
    )
    
    # Calculate mean temperature in each bin
    v_idx = np.digitize(data['speed'], v_bins)
    n_idx = np.digitize(data['density'], n_bins)
    
    temp_mean = np.full((len(v_bins)-1, len(n_bins)-1), np.nan)
    for i in range(1, len(v_bins)):
        for j in range(1, len(n_bins)):
            mask = (v_idx == i) & (n_idx == j)
            count = mask.sum()
            # Only include bins with at least 10 data points
            if count >= 10:
                temp_mean[i-1, j-1] = data.loc[mask, 'temperature'].mean()
    
    v_centers = (v_edges[:-1] + v_edges[1:]) / 2
    n_centers = (n_edges[:-1] + n_edges[1:]) / 2
    
    # Plot temperature
    pcm = ax.pcolormesh(v_centers, n_centers, temp_mean.T,
                        cmap='plasma', shading='auto',
                        vmin=0, vmax=3e5)
    pcms.append(pcm)
    
    # Add contours (filter counts for bins with >=10 points)
    from scipy.ndimage import gaussian_filter
    # Mask out bins with <10 counts
    counts_filtered = np.where(counts >= 10, counts, 0)
    counts_smooth = gaussian_filter(counts_filtered.T, sigma=1.0)
    max_count = counts_smooth.max()
    if max_count > 10:
        levels = np.logspace(np.log10(max(10, max_count/1000)),
                             np.log10(max_count), 8)
        
        contours = ax.contour(v_centers, n_centers, counts_smooth,
                              levels=levels, colors='white',
                              linewidths=1.0, alpha=0.6)
        ax.clabel(contours, inline=True, fontsize=14, fmt='%d')
    
    ax.set_xlabel('Solar Wind Speed (km/s)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.2, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Only show y-axis label and ticks for leftmost panel
    if idx == 0:
        ax.set_ylabel('Density (cm$^{-3}$)', fontsize=14, fontweight='bold')
    else:
        ax.set_yticklabels([])
        ax.set_ylabel('')

# Add single colorbar to the right of the rightmost panel without squeezing panels
divider = make_axes_locatable(axes[-1])
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcms[-1], cax=cax, label='Temperature (K)')
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Temperature (K)', fontsize=14, fontweight='bold')

plt.tight_layout()

# Save as PNG
plt.savefig(f'WSA_ENLIL_vs_OMNI_vs_SURF_2019_2022_2D_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved figure: WSA_ENLIL_vs_OMNI_vs_SURF_2019_2022_2D_comparison.png")

# Save as PDF to Overleaf directory
dbox = os.getenv('DBOX')
save_dir = os.path.join(dbox, 'Apps', 'Overleaf', 'SHUXt')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'WSA_ENLIL_vs_OMNI_vs_SURF_2019_2022.pdf')
fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"Saved PDF to: {save_path}")

plt.show()


# # =============================================================================
# # Create time series comparison plots
# # =============================================================================

# def plot_timeseries_comparison(enlil_data, omni_data, start_date, end_date, 
#                                title_suffix="", filename_suffix=""):
#     """
#     Plot 3-panel time series comparison of ENLIL and OMNI data.
    
#     Parameters:
#     -----------
#     enlil_data : DataFrame with datetime index
#     omni_data : DataFrame with datetime index  
#     start_date : datetime
#     end_date : datetime
#     title_suffix : str, optional suffix for title
#     filename_suffix : str, optional suffix for filename
#     """
#     # Extract data for the time period
#     mask_enlil = (enlil_data.index >= start_date) & (enlil_data.index <= end_date)
#     mask_omni = (omni_data.index >= start_date) & (omni_data.index <= end_date)
    
#     enlil_period = enlil_data[mask_enlil]
#     omni_period = omni_data[mask_omni]
    
#     if len(enlil_period) == 0 or len(omni_period) == 0:
#         print(f"Warning: No data for period {start_date} to {end_date}")
#         return
    
#     # Create figure with 3 vertical panels
#     fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
#     # Speed panel
#     ax = axes[0]
#     ax.plot(enlil_period.index, enlil_period['speed'], 'b-', 
#             label='WSA-ENLIL', linewidth=1.5, alpha=0.8)
#     ax.plot(omni_period.index, omni_period['speed'], 'r-', 
#             label='OMNI', linewidth=1.0, alpha=0.7)
#     ax.set_ylabel('Speed (km/s)', fontsize=12, fontweight='bold')
#     ax.legend(loc='upper right', fontsize=10)
#     ax.grid(alpha=0.3)
#     ax.set_ylim([200, 800])
    
#     # Density panel
#     ax = axes[1]
#     ax.plot(enlil_period.index, enlil_period['density'], 'b-', 
#             label='WSA-ENLIL', linewidth=1.5, alpha=0.8)
#     ax.plot(omni_period.index, omni_period['density'], 'r-', 
#             label='OMNI', linewidth=1.0, alpha=0.7)
#     ax.set_ylabel('Density (cm$^{-3}$)', fontsize=12, fontweight='bold')
#     ax.legend(loc='upper right', fontsize=10)
#     ax.grid(alpha=0.3)
#     ax.set_ylim([0, 30])
    
#     # Temperature panel
#     ax = axes[2]
#     ax.plot(enlil_period.index, enlil_period['temperature'], 'b-', 
#             label='WSA-ENLIL', linewidth=1.5, alpha=0.8)
#     ax.plot(omni_period.index, omni_period['temperature'], 'r-', 
#             label='OMNI', linewidth=1.0, alpha=0.7)
#     ax.set_ylabel('Temperature (K)', fontsize=12, fontweight='bold')
#     ax.set_xlabel('Date', fontsize=12, fontweight='bold')
#     ax.legend(loc='upper right', fontsize=10)
#     ax.grid(alpha=0.3)
#     ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
#     ax.set_ylim([0, 5e5])
    
#     # Format x-axis
#     import matplotlib.dates as mdates
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     fig.autofmt_xdate(rotation=45, ha='right')
    
#     # Add title
#     fig.suptitle(f'WSA-ENLIL vs OMNI Time Series Comparison{title_suffix}', 
#                  fontsize=14, fontweight='bold', y=0.995)
    
#     plt.tight_layout()
    
#     # Save figure
#     filename = f'WSA_ENLIL_vs_OMNI_timeseries{filename_suffix}.png'
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     print(f"Saved figure: {filename}")
#     plt.show()
#     plt.close()


# print("\n" + "="*60)
# print("Creating time series comparison plots...")
# print("="*60)

# # Define several interesting time periods across the years
# time_periods = [
#     # Period 1: Late 2019 (from the ENLIL data)
#     {
#         'start': datetime(2019, 9, 1),
#         'end': datetime(2019, 10, 15),
#         'title': ' (Sep-Oct 2019)',
#         'suffix': '_2019_sep_oct'
#     },
#     # Period 2: Early 2020
#     {
#         'start': datetime(2020, 3, 1),
#         'end': datetime(2020, 4, 15),
#         'title': ' (Mar-Apr 2020)',
#         'suffix': '_2020_mar_apr'
#     },
#     # Period 3: Mid 2020 - Summer
#     {
#         'start': datetime(2020, 7, 1),
#         'end': datetime(2020, 8, 15),
#         'title': ' (Jul-Aug 2020)',
#         'suffix': '_2020_jul_aug'
#     },
#     # Period 4: Late 2020 - Fall
#     {
#         'start': datetime(2020, 11, 1),
#         'end': datetime(2020, 12, 15),
#         'title': ' (Nov-Dec 2020)',
#         'suffix': '_2020_nov_dec'
#     },
#     # Period 5: Spring 2021
#     {
#         'start': datetime(2021, 4, 1),
#         'end': datetime(2021, 5, 15),
#         'title': ' (Apr-May 2021)',
#         'suffix': '_2021_apr_may'
#     },
#     # Period 6: Fall 2021
#     {
#         'start': datetime(2021, 10, 1),
#         'end': datetime(2021, 11, 15),
#         'title': ' (Oct-Nov 2021)',
#         'suffix': '_2021_oct_nov'
#     },
#     # Period 7: Spring 2022
#     {
#         'start': datetime(2022, 3, 15),
#         'end': datetime(2022, 5, 1),
#         'title': ' (Mar-May 2022)',
#         'suffix': '_2022_mar_may'
#     },
#     # Period 8: Late 2022
#     {
#         'start': datetime(2022, 10, 1),
#         'end': datetime(2022, 11, 15),
#         'title': ' (Oct-Nov 2022)',
#         'suffix': '_2022_oct_nov'
#     },
# ]

# # Create plots for each period
# for i, period in enumerate(time_periods, 1):
#     print(f"\nPlotting period {i}/{len(time_periods)}: {period['title']}")
#     plot_timeseries_comparison(
#         enlil_data, 
#         omni_clean,
#         period['start'],
#         period['end'],
#         title_suffix=period['title'],
#         filename_suffix=period['suffix']
#     )

# print("\n" + "="*60)
# print("All figures completed!")
# print("="*60)
