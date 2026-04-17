#<codecell> Imports

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

import numpy as np
import astropy.units as u
from astropy.constants import m_p
import matplotlib
matplotlib.use('TkAgg')  # Set backend explicitly for Windows
import matplotlib.pyplot as plt
import datetime

import xarray as xr
import pandas as pd
from astropy.time import Time as AstroTime
from glob import glob

ftime = datetime.datetime(2019,9,15)

WSA_ENLIL_dir = os.path.join(os.getenv('DBOX'), 'Data', 'Enlil', 'WSA_ENLIL_LUKE')
simtime = 27*u.day

dbox = os.environ.get('DBOX')
overleaf_dir = os.path.join(dbox, 'Apps', 'Overleaf', 'SHUXt')

# <codecell> Enlil reader

def read_enlil_timeseries(wsa_enlil_dir, ftime, end_time=None, time_tolerance_days=1):
    """
    Read WSA-ENLIL netCDF file closest to ftime.
    Loads the entire file if it's within tolerance.
    If end_time is provided and not covered, loads additional files to fill the gap.
    """
    import datetime as dt
    
    # Helper function to load a single ENLIL file
    def load_enlil_file(nc_file):
        try:
            ds = xr.open_dataset(nc_file)
            
            ref_mjd = ds.attrs['refdate_mjd']
            ref_time = AstroTime(ref_mjd, format='mjd').datetime
            
            time_seconds = ds['TIME'].values
            time_datetime = pd.to_datetime(ref_time) + pd.to_timedelta(time_seconds, unit='s')
            
            # Extract data
            v1 = ds['V1'].values / 1000.0  # km/s
            proton_mass_kg = 1.67262192e-27
            density_cm3 = (ds['D'].values / proton_mass_kg) / 1e6
            temperature = ds['T'].values
            
            df = pd.DataFrame({
                'datetime': time_datetime,
                'speed': v1,
                'density': density_cm3,
                'temperature': temperature
            })
            ds.close()
            
            # Set index
            df.set_index('datetime', inplace=True)
            # Ensure index is timezone naive
            df.index = df.index.tz_localize(None)
            
            return df
            
        except Exception as e:
            print(f"Error loading file {os.path.basename(nc_file)}: {e}")
            return None
    
    # Get all candidate files
    years = sorted(list(set([ftime.year, ftime.year - 1, ftime.year + 1])))
    all_files = []
    
    print(f"DEBUG: Looking for ENLIL file closest to {ftime}")
    
    for year in years:
        year_str = str(year)
        year_path = os.path.join(wsa_enlil_dir, year_str)
        
        if not os.path.exists(year_path):
            continue
        
        nc_files = sorted(glob(os.path.join(year_path, '*.nc')))
        
        for nc_file in nc_files:
            try:
                with xr.open_dataset(nc_file) as ds:
                     if 'refdate_mjd' in ds.attrs:
                        ref_mjd = ds.attrs['refdate_mjd']
                        ref_time = AstroTime(ref_mjd, format='mjd').datetime
                        # Ensure timezone-naive
                        if hasattr(ref_time, 'tz_localize'):
                            ref_time = ref_time.replace(tzinfo=None)
                        
                        all_files.append({'path': nc_file, 'ref_time': ref_time})
            except:
                pass

    if not all_files:
        print("No ENLIL files found.")
        return pd.DataFrame()
    
    # Sort by reference time
    all_files.sort(key=lambda x: x['ref_time'])
    
    # Find the closest file to ftime
    all_files.sort(key=lambda x: abs((x['ref_time'] - ftime).total_seconds()))
    best_file = all_files[0]
    
    print(f"Closest ENLIL file: {os.path.basename(best_file['path'])}")
    print(f"  Ref time: {best_file['ref_time']}")
    print(f"  Time diff: {abs((best_file['ref_time'] - ftime).total_seconds())/3600:.2f} hours")
    
    # Check tolerance
    if abs((best_file['ref_time'] - ftime).total_seconds()) > time_tolerance_days * 86400:
        print(f"  -> File is outside tolerance of {time_tolerance_days} days. Skipping.")
        return pd.DataFrame()
    
    print("  -> Loading file.")
    
    # Load the best file
    df_combined = load_enlil_file(best_file['path'])
    if df_combined is None:
        return pd.DataFrame()
    
    # Check if we need to load additional files to cover the full time range
    if end_time is not None and len(df_combined) > 0:
        current_end = df_combined.index.max()
        
        if current_end < end_time:
            print(f"First file ends at {current_end}, but need data until {end_time}")
            print(f"  Searching for continuation files...")
            
            # Re-sort all files by reference time
            all_files.sort(key=lambda x: x['ref_time'])
            
            # Find files that come after the best_file
            remaining_files = [f for f in all_files if f['ref_time'] > best_file['ref_time']]
            
            for next_file in remaining_files:
                if current_end >= end_time:
                    break
                
                print(f"  Trying {os.path.basename(next_file['path'])}...")
                df_next = load_enlil_file(next_file['path'])
                
                if df_next is not None and len(df_next) > 0:
                    # Concatenate and remove duplicates
                    df_combined = pd.concat([df_combined, df_next])
                    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
                    df_combined = df_combined.sort_index()
                    
                    current_end = df_combined.index.max()
                    print(f"    Loaded, now have data until {current_end}")
    
    # Resample to 1-hour average
    df_hourly = df_combined.resample('1h').mean().dropna()
    
    print(f"Loaded {len(df_hourly)} data points total.")
    return df_hourly





# <codecell> Upwind and compressible tests



is_model = surfIS.omniSURF_forecast(ftime, simtime=27.27*u.day, 
                        rmin=21.5*u.solRad, rmax=230*u.solRad, 
                        dt_scale=4,
                        omni_input=None, buffertime=5*u.day,
                        run_2d=False)

is_model.solve([])
#HA.plot_earth_timeseries(is_model)

ts_incomp = surfA.get_observer_timeseries(is_model)

is_model = surfIS.omniSURF_forecast(ftime, simtime=27.27*u.day, 
                        rmin=21.5*u.solRad, rmax=230*u.solRad, 
                        dt_scale=4,
                        omni_input=None, buffertime=5*u.day,
                        run_2d=False, solver='hydro')

is_model.solve([])

# Get timeseries data for custom plot
ts_comp = surfA.get_observer_timeseries(is_model)

# Get OMNI data for comparison
starttime = ts_comp['time'][0]
endtime = ts_comp['time'][len(ts_comp) - 1]
omni_data = surfIS.get_omni(starttime, endtime)
mask = (omni_data['datetime'] >= starttime) & (omni_data['datetime'] <= endtime)
omni_plotdata = omni_data[mask]

# Get WSA-ENLIL data for comparison
print(f"\nReading WSA-ENLIL data for {ftime} to {endtime}...")

enlil_data = read_enlil_timeseries(WSA_ENLIL_dir, ftime, end_time=endtime)
print(f"Successfully loaded {len(enlil_data)} ENLIL data points")





# Create custom 1 AU plot without B polarity panel
fig, axs = plt.subplots(3, 1, figsize=(14, 12))

# Velocity panel
axs[0].plot(ts_comp['time'], ts_comp['vsw'], 'r-', label='SURF-hydro')
axs[0].plot(ts_incomp['time'], ts_incomp['vsw'], 'r--', label='SURF-HUXt')
if len(enlil_data) > 0:
    axs[0].plot(enlil_data.index, enlil_data['speed'], 'b-', label='WSA-ENLIL')
axs[0].plot(omni_plotdata['datetime'], omni_plotdata['V'], 'k', label='OMNI')
axs[0].set_ylim(250, 800)
axs[0].set_ylabel(r'$v$ [km/s]')
axs[0].legend(loc='upper right')
axs[0].set_xlim(starttime, endtime)
title_str = f'Solar wind forecast at 1 AU: {starttime.strftime("%Y-%m-%d")} to {endtime.strftime("%Y-%m-%d")}'
axs[0].set_title(title_str, fontsize=14)

# Density panel (linear scale)
axs[1].plot(ts_comp['time'], ts_comp['n'], 'r-', label='SURF-hydro')
if 'n' in ts_incomp:
    axs[1].plot(ts_incomp['time'], ts_incomp['n'], 'r--', label='SURF-HUXt')
if len(enlil_data) > 0:
    axs[1].plot(enlil_data.index, enlil_data['density'], 'b-', label='WSA-ENLIL')
# Plot OMNI density, filtering out invalid values
omni_n = omni_plotdata['N'].copy()
omni_n[omni_n >= 999.0] = np.nan
axs[1].plot(omni_plotdata['datetime'], omni_n, 'k-', label='OMNI')
axs[1].set_ylabel(r'$n_P$ [cm$^{-3}$]')
axs[1].legend(loc='upper right')
axs[1].set_xlim(starttime, endtime)

# Temperature panel (linear scale, in units of 10^5 K)
axs[2].plot(ts_comp['time'], ts_comp['T'] / 1e5, 'r-', label='SURF-hydro')
if 'T' in ts_incomp:
    axs[2].plot(ts_incomp['time'], ts_incomp['T'] / 1e5, 'r--', label='SURF-HUXt')
if len(enlil_data) > 0:
    axs[2].plot(enlil_data.index, enlil_data['temperature'] / 1e5, 'b-', label='WSA-ENLIL')
# Plot OMNI temperature, filtering out invalid values
omni_t = omni_plotdata['T'].copy()
omni_t[omni_t >= 999999.0] = np.nan
axs[2].plot(omni_plotdata['datetime'], omni_t / 1e5, 'k-', label='OMNI')
axs[2].set_ylabel(r'T [$10^5$ K]')
axs[2].set_xlabel('Date')
axs[2].legend(loc='upper right')
axs[2].set_xlim(starttime, endtime)

# Remove x-tick labels from top panels
axs[0].set_xticklabels([])
axs[1].set_xticklabels([])

# Add panel labels and grids to 1 AU plot
for i, ax in enumerate(axs):
    ax.text(0.02, 0.95, f'({chr(97+i)})', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.grid(True, alpha=0.3)

# Set regular date tick spacing on all axes
import matplotlib.dates as mdates
time_span = (endtime - starttime).total_seconds() / 86400  # days
if time_span <= 10:
    locator = mdates.DayLocator(interval=1)
elif time_span <= 30:
    locator = mdates.DayLocator(interval=5)
else:
    locator = mdates.DayLocator(interval=10)
for ax in axs:
    ax.xaxis.set_major_locator(locator)


pdf_path = os.path.join(overleaf_dir, 'SHUXt_1AU.pdf')
plt.savefig(pdf_path, dpi=150)


#also plot the inputs.

fig, axes = plt.subplots(3, figsize=(14, 9))
# Convert Carrington longitude to degrees
carr_lon_deg = np.rad2deg(is_model.v_boundary_lons)

axes[0].plot(carr_lon_deg, is_model.v_boundary.value, 'k-')
axes[0].set_ylabel(r'$v$ [km/s]')
axes[0].text(0.02, 0.95, '(a)', transform=axes[0].transAxes, 
            fontsize=14,  va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(labelbottom=False)
axes[0].set_xlim(0, 360)
axes[0].set_xticks([0, 90, 180, 270, 360])
# Calculate input period: 27.27 days back from ftime
input_start = ftime - datetime.timedelta(days=27.27)
input_title = f'Solar wind at 0.1 AU: {input_start.strftime("%Y-%m-%d")} to {ftime.strftime("%Y-%m-%d")}'
axes[0].set_title(input_title, fontsize=14)

# Convert mass density to proton density (in cm^-3)
proton_density = (is_model.rho_boundary / m_p).to(u.cm**-3)
axes[1].plot(carr_lon_deg, proton_density.value, 'k-')
axes[1].set_ylabel(r'$n_p$ [cm$^{-3}$]')
axes[1].text(0.02, 0.95, '(b)', transform=axes[1].transAxes, 
            fontsize=14,  va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelbottom=False)
axes[1].set_xlim(0, 360)
axes[1].set_xticks([0, 90, 180, 270, 360])

axes[2].plot(carr_lon_deg, is_model.temp_boundary.value/1e6, 'k-')
axes[2].set_ylabel(r'T [$10^6$ K]')
axes[2].set_xlabel('Carrington Longitude [deg]')
axes[2].text(0.02, 0.95, '(c)', transform=axes[2].transAxes, 
            fontsize=14,  va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, 360)
axes[2].set_xticks([0, 90, 180, 270, 360])

pdf_path = os.path.join(overleaf_dir, 'SHUXt_0p1AU.pdf')
plt.savefig(pdf_path, dpi=150)

# model = Hinsitu.omniHUXt_reconstruction(ftime, ftime +datetime.timedelta(days=27), 
#                         rmin=21.5*u.solRad, rmax=250*u.solRad, 
#                         dt_scale=4, dt=1*u.day,
#                         run_2d=False,
#                         solver='hllc',
#                         rho_source='speed',
#                         temp_source='speed')

# model.solve([])

# # Plot Earth timeseries
# HA.plot_earth_timeseries(model)

 

# <codecell> End of script

plt.show(block=True)
print("\nPlots displayed. Press Enter to close all plots and exit...")
input()