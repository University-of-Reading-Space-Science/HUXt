#<codecell> Imports

import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use('TkAgg')  # Set backend explicitly for Windows
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import os
from copy import deepcopy
import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin
import huxt.huxt_insitu as Hinsitu



simtime = 10*u.day

# <codecell> Upwind and compressible tests


print("="*60)
print("SCRIPT STARTED - huxt_quickrun_test.py")
print("="*60)
print(f"Matplotlib backend: {matplotlib.get_backend()}")
plt.ion()  # Turn on interactive mode




# Set up HUXt over a limited longitude range.
dirs = H._setup_dirs_()
data_path=dirs['example_inputs']
print(data_path)
filepath = os.path.join(data_path, 'wsa_gong_2024050906.fits')
vr_in = Hin.get_WSA_long_profile(filepath, lat=0.0 * u.deg)
br_in = Hin.get_WSA_br_long_profile(filepath, lat=0.0 * u.deg)

dirs = H._setup_dirs_()
print(dirs)



# Set up HUXt
#cr=1920
#vr_in = np.ones(128)*400*u.km/u.s #Hin.get_MAS_long_profile(cr, 0.0*u.deg)



# Set up to trace a set of field lines from a range of evenly spaced Carrington longitudes
dlon = (20*u.deg).to(u.rad).value
lon_grid = np.arange(dlon/2, 2*np.pi-dlon/2 + 0.0001, dlon)*u.rad



# Get a list of two ConeCMEs
daysec = 86400
times = [0*u.day]
speeds = [1000]
lons = [0]
widths = [60]
thickness = [5]
cme_list = []
for t, l, w, v, thick in zip(times, lons, widths, speeds, thickness):
    cme = H.ConeCME(t_launch=t, longitude=l*u.deg, width=w*u.deg, v=v*u.km/u.s, 
                    thickness=thick*u.solRad, 
                    density_fraction=0.1, temperature_fraction=0.1,
                    profile_type='sinusoidal')
    cme_list.append(cme)


# Set up to trace a set of field lines from a range of evenly spaced Carrington longitudes
dlon = (20*u.deg).to(u.rad).value
lon_grid = np.arange(dlon/2, 2*np.pi-dlon/2 + 0.0001, dlon)*u.rad

# Give the streakline footpoints (in Carr long) to the solve method
# model_comp = H.HUXt(v_boundary=vr_in, lon_start=330*u.deg, lon_stop = 30*u.deg,
#                      simtime=5*u.day, dt_scale=4, 
#                     compressible=True, solver ='upwind')
# model_comp.solve(cme_list, streak_carr=lon_grid)
# HA.plot_earth_timeseries(model_comp)

t_interest = 1*u.day

print("\n" + "="*60)
print("Starting incompressible model (upwind solver)...")
print("="*60)
model_incomp = H.HUXt(v_boundary=vr_in, b_boundary=br_in, #lon_start=350*u.deg, lon_stop = 10*u.deg, 
                    #lon_out=0.0*u.rad,
                    simtime=simtime, dt_scale=4, 
                    solver ='upwind', lon_out=0.0*u.rad)
model_incomp_ambient = deepcopy(model_incomp)  # Make a copy to run without CMEs for comparison
model_incomp_ambient.solve([])
print("Model initialized. Starting solve...")
t0 = datetime.datetime.now()
model_incomp.solve(cme_list, streak_carr=lon_grid)
dt_pcm = (datetime.datetime.now() - t0).total_seconds()
print(f"Solve complete in {dt_pcm:.2f}s!")




# ---------------------------------------------------------
# Test 1: First-order HLLC + PCM
# ---------------------------------------------------------
print("\n" + "="*60)
print("Starting compressible model (HLLC + PCM 1st Order)...")
print("="*60)
model_pcm = H.HUXt(v_boundary=vr_in, b_boundary=br_in,
                        simtime=simtime, dt_scale=4, 
                    solver ='hllc-pcm', lon_out=0.0*u.rad) # Explicitly request PCM
model_pcm_ambient = deepcopy(model_pcm)  # Make a copy to run without CMEs for comparison
model_pcm_ambient.solve([])
print("Model initialized. Starting solve...")
t0 = datetime.datetime.now()
model_pcm.solve(cme_list, streak_carr=lon_grid)
dt_pcm = (datetime.datetime.now() - t0).total_seconds()
print(f"Solve complete in {dt_pcm:.2f}s!")



# ---------------------------------------------------------
# Test 2: Second-order HLLC + PLM + RK2
# ---------------------------------------------------------
print("\n" + "="*60)
print("Starting compressible model (HLLC + PLM + RK2 2nd Order)...")
print("="*60)
model_plm = H.HUXt(v_boundary=vr_in, b_boundary=br_in,
                        simtime=simtime, dt_scale=4, 
                    solver ='hllc-plm-rk2', lon_out=0.0*u.rad) # Explicitly request PLM+RK2
model_plm_ambient = deepcopy(model_plm)  # Make a copy to run without CMEs for comparison
model_plm_ambient.solve([])
print("Model initialized. Starting solve...")
t0 = datetime.datetime.now()
model_plm.solve(cme_list, streak_carr=lon_grid)
dt_plm = (datetime.datetime.now() - t0).total_seconds()
print(f"Solve complete in {dt_plm:.2f}s!")



# # ---------------------------------------------------------
# # Comparison Plot
# # ---------------------------------------------------------
print("\nCreating plot 7: Direct comparison of Earth Profiles...")
ts_pcm = HA.get_observer_timeseries(model_pcm)
ts_plm = HA.get_observer_timeseries(model_plm)
ts_huxt = HA.get_observer_timeseries(model_incomp)

# Get ambient timeseries (without CMEs)
ts_pcm_ambient = HA.get_observer_timeseries(model_pcm_ambient)
ts_plm_ambient = HA.get_observer_timeseries(model_plm_ambient)
ts_huxt_ambient = HA.get_observer_timeseries(model_incomp_ambient)

fig, ax = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

# Determine common y-axis range for velocity plots
v_min = min(ts_huxt_ambient['vsw'].min(), ts_huxt['vsw'].min(), 
            ts_pcm_ambient['vsw'].min(), ts_pcm['vsw'].min(),
            ts_plm_ambient['vsw'].min(), ts_plm['vsw'].min())
v_max = max(ts_huxt_ambient['vsw'].max(), ts_huxt['vsw'].max(),
            ts_pcm_ambient['vsw'].max(), ts_pcm['vsw'].max(),
            ts_plm_ambient['vsw'].max(), ts_plm['vsw'].max())
v_range = v_max - v_min
v_lim = [v_min - 0.05*v_range, v_max + 0.05*v_range]

# Panel 0: HUXt velocity
ax[0].plot(ts_huxt_ambient['time'], ts_huxt_ambient['vsw'], 'gray', linewidth=1.5, label='SURF-HUXt (ambient)')
ax[0].plot(ts_huxt['time'], ts_huxt['vsw'], 'k', linewidth=1.5, label='SURF-HUXt (CME)')
ax[0].set_ylabel('V [km/s]')
ax[0].legend(loc='upper right', fontsize=11)
ax[0].set_ylim(v_lim)
ax[0].grid(True, alpha=0.3)

# Add shading where CME and ambient differ by more than 10 km/s
v_diff_huxt = np.abs(ts_huxt['vsw'].values - ts_huxt_ambient['vsw'].values)
mask_huxt = v_diff_huxt > 10
if np.any(mask_huxt):
    ax[0].fill_between(ts_huxt['time'], v_lim[0], v_lim[1], where=mask_huxt, 
                       color='lightgray', alpha=0.3, zorder=0)

# Panel 1: PCM velocity
ax[1].plot(ts_pcm_ambient['time'], ts_pcm_ambient['vsw'], 'gray', linewidth=1.5, label='SURF-hydro-PCM (ambient)')
ax[1].plot(ts_pcm['time'], ts_pcm['vsw'], 'b', linewidth=1.5, label='SURF-hydro-PCM (CME)')
ax[1].set_ylabel('V [km/s]')
ax[1].legend(loc='upper right', fontsize=11)
ax[1].set_ylim(v_lim)
ax[1].grid(True, alpha=0.3)

# Add shading where CME and ambient differ by more than 10 km/s
v_diff_pcm = np.abs(ts_pcm['vsw'].values - ts_pcm_ambient['vsw'].values)
mask_pcm = v_diff_pcm > 10
if np.any(mask_pcm):
    ax[1].fill_between(ts_pcm['time'], v_lim[0], v_lim[1], where=mask_pcm, 
                       color='lightblue', alpha=0.3, zorder=0)

# Panel 2: PLM velocity
ax[2].plot(ts_plm_ambient['time'], ts_plm_ambient['vsw'], 'gray', linewidth=1.5, label='SURF-hydro-PLM (ambient)')
ax[2].plot(ts_plm['time'], ts_plm['vsw'], 'r', linewidth=1.5, label='SURF-hydro-PLM (CME)')
ax[2].set_ylabel('V [km/s]')
ax[2].legend(loc='upper right', fontsize=11)
ax[2].set_ylim(v_lim)
ax[2].grid(True, alpha=0.3)

# Add shading where CME and ambient differ by more than 10 km/s
v_diff_plm = np.abs(ts_plm['vsw'].values - ts_plm_ambient['vsw'].values)
mask_plm = v_diff_plm > 10
if np.any(mask_plm):
    ax[2].fill_between(ts_plm['time'], v_lim[0], v_lim[1], where=mask_plm, 
                       color='lightcoral', alpha=0.3, zorder=0)

# Panel 3: Density comparison
ax[3].plot(ts_pcm['time'], ts_pcm['n'], 'b', linewidth=1.5, label='SURF-hydro-PCM (CME)') 
ax[3].plot(ts_plm['time'], ts_plm['n'], 'r', linewidth=1.5, label='SURF-hydro-PLM (CME)')
ax[3].set_ylabel(r'n [cm$^{-3}$]')
ax[3].legend(loc='upper right', fontsize=11)
ax[3].grid(True, alpha=0.3)

# Add shading where velocity differs by more than 10 km/s
n_lim = ax[3].get_ylim()
n_mid = (n_lim[0] + n_lim[1]) / 2
if np.any(mask_pcm):
    ax[3].fill_between(ts_pcm['time'], n_mid, n_lim[1], where=mask_pcm, 
                       color='lightblue', alpha=0.3, zorder=0)
if np.any(mask_plm):
    ax[3].fill_between(ts_plm['time'], n_lim[0], n_mid, where=mask_plm, 
                       color='lightcoral', alpha=0.3, zorder=0)
ax[3].set_ylim(n_lim)

# Panel 4: Temperature comparison
ax[4].plot(ts_pcm['time'], ts_pcm['T']/10**5, 'b', linewidth=1.5, label='SURF-hydro-PCM (CME)')
ax[4].plot(ts_plm['time'], ts_plm['T']/10**5, 'r', linewidth=1.5, label='SURF-hydro-PLM (CME)')
ax[4].set_ylabel(r'T [$10^5$ K]')
ax[4].legend(loc='upper right', fontsize=11)
ax[4].grid(True, alpha=0.3)

# Add shading where velocity differs by more than 10 km/s
t_lim = ax[4].get_ylim()
t_mid = (t_lim[0] + t_lim[1]) / 2
if np.any(mask_pcm):
    ax[4].fill_between(ts_pcm['time'], t_mid, t_lim[1], where=mask_pcm, 
                       color='lightblue', alpha=0.3, zorder=0)
if np.any(mask_plm):
    ax[4].fill_between(ts_plm['time'], t_lim[0], t_mid, where=mask_plm, 
                       color='lightcoral', alpha=0.3, zorder=0)
ax[4].set_ylim(t_lim)

# Format x-axis with DD-MM format
date_format = mdates.DateFormatter('%d-%m')
ax[4].xaxis.set_major_formatter(date_format)
ax[4].xaxis.set_major_locator(mdates.DayLocator(interval=2))

# Get the year from the first time point
year = ts_pcm['time'][0].year
ax[4].set_xlabel(f'Date of {year}')

# Set x-axis limits to the data range
ax[4].set_xlim(ts_pcm['time'].iloc[0], ts_pcm['time'].iloc[-1])

# Add CME arrival and end times as vertical lines
if len(cme_list) > 0:
    # Get CME particle data for each model (assuming single longitude, nlon=1)
    for cme_id in range(len(cme_list)):
        r_threshold_rs = 215  # Solar radii
        r_threshold = r_threshold_rs * 6.96e5  # Convert to km
        
        # Check how many particles the CME has (dimension 2 = particle index)
        n_particles = model_incomp.cme_particles_r.shape[2]
        
        # HUXt/upwind (black lines)
        # cme_particles_r[cme_id, time_idx, particle_idx, lon_idx]
        # particle_idx: 0=front, 1=back
        cme_r_lead_incomp = model_incomp.cme_particles_r[cme_id, :, 0, 0].value
        if n_particles > 1:
            cme_r_trail_incomp = model_incomp.cme_particles_r[cme_id, :, 1, 0].value
        else:
            cme_r_trail_incomp = cme_r_lead_incomp
        
        times_incomp = np.array(ts_huxt['time'])
        
        # Find crossing times for leading edge
        valid_lead_incomp = np.isfinite(cme_r_lead_incomp) & (cme_r_lead_incomp >= r_threshold)
        if np.any(valid_lead_incomp):
            idx_lead = np.where(valid_lead_incomp)[0][0]
            t_arrival_incomp = times_incomp[idx_lead]
        else:
            t_arrival_incomp = None
        
        # Find crossing times for trailing edge
        valid_trail_incomp = np.isfinite(cme_r_trail_incomp) & (cme_r_trail_incomp >= r_threshold)
        if np.any(valid_trail_incomp):
            idx_trail = np.where(valid_trail_incomp)[0][0]
            t_end_incomp = times_incomp[idx_trail]
        else:
            t_end_incomp = None
        
        if t_arrival_incomp is not None and t_end_incomp is not None:
            # Plot on HUXt velocity panel only
            ax[0].axvline(t_arrival_incomp, color='k', linestyle='-', linewidth=4, alpha=0.5)
            ax[0].axvline(t_end_incomp, color='k', linestyle='-', linewidth=4, alpha=0.5)
        
        # PCM (blue lines)
        cme_r_lead_pcm = model_pcm.cme_particles_r[cme_id, :, 0, 0].value
        if n_particles > 1:
            cme_r_trail_pcm = model_pcm.cme_particles_r[cme_id, :, 1, 0].value
        else:
            cme_r_trail_pcm = cme_r_lead_pcm
        
        times_pcm = np.array(ts_pcm['time'])
        
        valid_lead_pcm = np.isfinite(cme_r_lead_pcm) & (cme_r_lead_pcm >= r_threshold)
        if np.any(valid_lead_pcm):
            idx_lead = np.where(valid_lead_pcm)[0][0]
            t_arrival_pcm = times_pcm[idx_lead]
        else:
            t_arrival_pcm = None
        
        valid_trail_pcm = np.isfinite(cme_r_trail_pcm) & (cme_r_trail_pcm >= r_threshold)
        if np.any(valid_trail_pcm):
            idx_trail = np.where(valid_trail_pcm)[0][0]
            t_end_pcm = times_pcm[idx_trail]
        else:
            t_end_pcm = None
        
        if t_arrival_pcm is not None and t_end_pcm is not None:
            # Plot on PCM velocity panel and bottom two panels
            ax[1].axvline(t_arrival_pcm, color='b', linestyle='-', linewidth=4, alpha=0.5)
            ax[1].axvline(t_end_pcm, color='b', linestyle='-', linewidth=4, alpha=0.5)
            ax[3].axvline(t_arrival_pcm, color='b', linestyle='-', linewidth=4, alpha=0.5, ymin=0.5, ymax=1.0)
            ax[3].axvline(t_end_pcm, color='b', linestyle='-', linewidth=4, alpha=0.5, ymin=0.5, ymax=1.0)
            ax[4].axvline(t_arrival_pcm, color='b', linestyle='-', linewidth=4, alpha=0.5, ymin=0.5, ymax=1.0)
            ax[4].axvline(t_end_pcm, color='b', linestyle='-', linewidth=4, alpha=0.5, ymin=0.5, ymax=1.0)
        
        # PLM (red lines)
        cme_r_lead_plm = model_plm.cme_particles_r[cme_id, :, 0, 0].value
        if n_particles > 1:
            cme_r_trail_plm = model_plm.cme_particles_r[cme_id, :, 1, 0].value
        else:
            cme_r_trail_plm = cme_r_lead_plm
        
        times_plm = np.array(ts_plm['time'])
        
        valid_lead_plm = np.isfinite(cme_r_lead_plm) & (cme_r_lead_plm >= r_threshold)
        if np.any(valid_lead_plm):
            idx_lead = np.where(valid_lead_plm)[0][0]
            t_arrival_plm = times_plm[idx_lead]
        else:
            t_arrival_plm = None
        
        valid_trail_plm = np.isfinite(cme_r_trail_plm) & (cme_r_trail_plm >= r_threshold)
        if np.any(valid_trail_plm):
            idx_trail = np.where(valid_trail_plm)[0][0]
            t_end_plm = times_plm[idx_trail]
        else:
            t_end_plm = None
  
        if t_arrival_plm is not None and t_end_plm is not None:
            # Plot on PLM velocity panel and bottom two panels
            ax[2].axvline(t_arrival_plm, color='r', linestyle='-', linewidth=4, alpha=0.5)
            ax[2].axvline(t_end_plm, color='r', linestyle='-', linewidth=4, alpha=0.5)
            ax[3].axvline(t_arrival_plm, color='r', linestyle='-', linewidth=4, alpha=0.5, ymin=0.0, ymax=0.5)
            ax[3].axvline(t_end_plm, color='r', linestyle='-', linewidth=4, alpha=0.5, ymin=0.0, ymax=0.5)
            ax[4].axvline(t_arrival_plm, color='r', linestyle='-', linewidth=4, alpha=0.5, ymin=0.0, ymax=0.5)
            ax[4].axvline(t_end_plm, color='r', linestyle='-', linewidth=4, alpha=0.5, ymin=0.0, ymax=0.5)

# Reduce whitespace between panels
plt.subplots_adjust(hspace=0.15)

# Save figure as PDF
dbox = os.getenv('DBOX')
if dbox:
    save_dir = os.path.join(dbox, 'Apps', 'Overleaf', 'SHUXt')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'solver_comparison.pdf')
    fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
else:
    print("\nWarning: DBOX environment variable not set. Figure not saved.")





plt.show(block=True)
print("\nPlots displayed. Press Enter to close all plots and exit...")
input()