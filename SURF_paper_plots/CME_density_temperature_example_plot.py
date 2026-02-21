#<codecell> Imports

import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use('TkAgg')  # Set backend explicitly for Windows
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import datetime
import os
from copy import deepcopy
import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin
import huxt.huxt_insitu as Hinsitu



simtime = 2.0*u.day
t_interest = 1.7*u.day
cme_speed = 800 * u.km/u.s
cr_lon_init = (2*np.pi)*u.rad

# <codecell> Upwind and compressible tests



plt.ion()  # Turn on interactive mode



# Set up HUXt over a limited longitude range.
dirs = H._setup_dirs_()
data_path=dirs['example_inputs']
#print(data_path)
filepath = os.path.join(data_path, 'wsa_gong_2024050906.fits')
cr = 2000

vr_in = Hin.get_WSA_long_profile(filepath, lat=0.0 * u.deg)
br_in = Hin.get_WSA_br_long_profile(filepath, lat=0.0 * u.deg)

dirs = H._setup_dirs_()
print(dirs)



# Set up to trace a set of field lines from a range of evenly spaced Carrington longitudes
dlon = (20*u.deg).to(u.rad).value
lon_grid = np.arange(dlon/2, 2*np.pi-dlon/2 + 0.0001, dlon)*u.rad


#hot, dense CME
cme = H.ConeCME(t_launch=0*u.day, longitude=0*u.deg, width=60*u.deg, v=cme_speed, 
                density_fraction=5, temperature_fraction=5,
                profile_type='sinusoidal')

print("\n" + "="*60)
print("Hot, dense CME")
print("="*60)
model_hot_dense = H.HUXt(v_boundary=vr_in, #lon_start=350*u.deg, lon_stop = 10*u.deg, 
                    #lon_out=0.0*u.rad,
                    simtime=simtime, dt_scale=4, cr_num=cr, cr_lon_init=cr_lon_init, r_min=21.5*u.solRad,
                    solver ='hllc')
model_hot_dense.solve([cme])
print("Model initialized. Starting solve...")
t0 = datetime.datetime.now()
model_hot_dense.solve([cme], streak_carr=lon_grid)
dt = (datetime.datetime.now() - t0).total_seconds()
print(f"Solve complete in {dt:.2f}s!")


#tenuous, cool CME
cme = H.ConeCME(t_launch=0*u.day, longitude=0*u.deg, width=60*u.deg, v=cme_speed, 
                density_fraction=0.5, temperature_fraction=0.5,
                profile_type='sinusoidal')

print("\n" + "="*60)
print("Cool, tenuous CME")
print("="*60)
model_tenuous_cool = H.HUXt(v_boundary=vr_in,  #lon_start=350*u.deg, lon_stop = 10*u.deg, 
                    #lon_out=0.0*u.rad,
                    simtime=simtime, dt_scale=4, cr_num=cr, cr_lon_init=cr_lon_init, r_min=21.5*u.solRad,
                    solver ='hllc')
model_tenuous_cool.solve([cme])
print("Model initialized. Starting solve...")
t0 = datetime.datetime.now()
model_tenuous_cool.solve([cme], streak_carr=lon_grid)
dt = (datetime.datetime.now() - t0).total_seconds()
print(f"Solve complete in {dt:.2f}s!")


# now do the parameter space study
#==================================================================

n_0p1au = np.arange(300, 2501, 300)*u.cm**-3
T_0p1au = np.arange(7e5, 2.8e6, 3e5)*u.K

# Create arrays to store results
n_vals = []
T_vals = []
v_outer_vals = []

mp = 1.6726219e-27*u.kg  # Proton mass
simtime = 3*u.day

# Create arrays to store results
n_cme_vals = []
T_cme_vals = []
tt_vals = []
v_max_vals = []  # Store maximum CME speed at 1 AU over entire run
vcme_time_series = []  # Store velocity time series for each run
time_series = []  # Store corresponding time arrays

#ambient run 
model_ambient = H.HUXt(v_boundary=vr_in,  simtime=simtime, dt_scale=4, r_max=215*u.solRad, r_min=21.5*u.solRad,
                            solver ='hllc-plm-rk2', lon_out=0.0*u.rad, cr_num=cr, cr_lon_init=cr_lon_init)
model_ambient.solve([])
vambient = model_ambient.v_grid[:, -1, 0]

# Loop over all combinations of n and T
for n in n_0p1au:
    for T in T_0p1au:
        cme = H.ConeCME(t_launch=0*u.day, longitude=0*u.deg, width=60*u.deg,
                        v=cme_speed, 
                cme_density= n.to(u.m**-3) * mp * 1000, cme_temperature=T,  
                profile_type='sinusoidal')
        
        model_plm = H.HUXt(v_boundary=vr_in,  
                                simtime=simtime, dt_scale=4, r_max=215*u.solRad, r_min=21.5*u.solRad,
                            solver ='hllc-plm-rk2', lon_out=0.0*u.rad, cr_num=cr, cr_lon_init=cr_lon_init)
        model_plm.solve([cme])
        
        # find time when speed is above the ambient value at 1 AU
        vcme = model_plm.v_grid[:, -1, 0]
        time_grid = model_plm.time_out

        # Find the first time when the speed exceeds the ambient speed at 1 AU
        times_above_ambient = time_grid[vcme > vambient + 20*u.km/u.s]
        if len(times_above_ambient) > 0:
            arrival_time = times_above_ambient[0]
            print(f"  CME transit time to 1 AU: {arrival_time.to(u.day):.2f}")
        else:
            print("  CME did not arrive at 1 AU within simulation time.")
            arrival_time = np.nan * u.day
        
        # Find maximum speed at 1 AU within 0.2 days of the arrival time
        if not np.isnan(arrival_time.value):
            time_window = np.abs(time_grid - arrival_time) <= 0.2 * u.day
            v_max_at_1au = np.max(vcme[time_window])
        else:
            v_max_at_1au = np.nan * u.km / u.s
        print(f"  Maximum CME speed at 1 AU: {v_max_at_1au:.1f}")
        
        # Store values
        n_cme_vals.append(n.value)
        T_cme_vals.append(T.value)
        tt_vals.append(arrival_time.to(u.day).value)
        v_max_vals.append(v_max_at_1au.value)
        vcme_time_series.append(vcme.value)
        time_series.append(time_grid.to(u.day).value)
    

# Convert to numpy arrays and reshape for plotting
n_cme_vals = np.array(n_cme_vals)
T_cme_vals = np.array(T_cme_vals)
tt_vals = np.array(tt_vals)
v_max_vals = np.array(v_max_vals)

# Reshape into 2D grid
n_unique = np.unique(n_cme_vals)
T_unique = np.unique(T_cme_vals)
tt_grid = tt_vals.reshape(len(n_unique), len(T_unique))
v_max_grid = v_max_vals.reshape(len(n_unique), len(T_unique))

# Create figure with 3 horizontal panels, right panel split into 2 vertical panels
fig = plt.figure(figsize=(20, 7))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.5], 
                       hspace=0.15, wspace=0.01, left=0.05, right=0.9)


# Left panel: hot, dense CME velocity snapshot
ax1 = plt.subplot(gs[:, 0], projection='polar')
HA.plot(model_hot_dense, t_interest, fighandle=fig, axhandle=ax1, minimalplot=True, 
        plotHCS=False, annotateplot=False)
# Add Earth
earth_pos_1 = model_hot_dense.get_observer('EARTH')
id_t_1 = np.argmin(np.abs(model_hot_dense.time_out - t_interest))
ax1.plot(earth_pos_1.lon[id_t_1], earth_pos_1.r[id_t_1], 'o', color='darkblue', markersize=10, markeredgecolor='w', markeredgewidth=1.5)
ax1.set_title(f'Hot, Dense CME\n({t_interest.to(u.day).value:.1f} days)', fontsize=14, pad=20)

# Middle panel: cool, tenuous CME velocity snapshot
ax2 = plt.subplot(gs[:, 1], projection='polar')
HA.plot(model_tenuous_cool, t_interest, fighandle=fig, axhandle=ax2, minimalplot=True,
        plotHCS=False, annotateplot=False)
# Add Earth
earth_pos_2 = model_tenuous_cool.get_observer('EARTH')
id_t_2 = np.argmin(np.abs(model_tenuous_cool.time_out - t_interest))
ax2.plot(earth_pos_2.lon[id_t_2], earth_pos_2.r[id_t_2], 'o', color='darkblue', markersize=10, markeredgecolor='w', markeredgewidth=1.5)
ax2.set_title(f'Cool, Tenuous CME\n({t_interest.to(u.day).value:.1f} days)', fontsize=14, pad=20)

# Shift middle panel left by ~5% of figure width
pos2 = ax2.get_position()
ax2.set_position([pos2.x0 - 0.05, pos2.y0, pos2.width, pos2.height])

# Extract contour object from first plot for shared colorbar
contour_obj = None
for child in ax1.get_children():
    if hasattr(child, 'get_array'):
        contour_obj = child
        break

# Add single colorbar for the two polar plots below them
cbar_ax = fig.add_axes([0.145, 0.04, 0.38, 0.03])  # [left, bottom, width, height] - moved right by 10%
if contour_obj is not None:
    cbar_polar = fig.colorbar(contour_obj, cax=cbar_ax, orientation='horizontal', ticks=[200, 400, 600, 800])
    # Remove default label and add it to the right side
    cbar_polar.ax.tick_params(labelsize=14)
    # Add label text to the right of the colorbar
    fig.text(0.535, 0.055, r'$V_{SW}$ [km/s]', fontsize=14, va='center', ha='left')

# Right top panel: transit time parameter space study
ax3 = plt.subplot(gs[0, 2])

# Create meshgrid for plotting
N_mesh, T_mesh = np.meshgrid(n_unique, T_unique, indexing='ij')

# Plot as pcolormesh with colorblind-friendly colormap
# Set sensible colorbar limits based on data range
vmin = np.floor(tt_grid.min() * 10) / 10  # Round down to nearest 0.1
vmax = np.ceil(tt_grid.max() * 10) / 10   # Round up to nearest 0.1
im = ax3.pcolormesh(N_mesh, T_mesh/1e6, tt_grid, shading='auto', cmap='cividis', vmin=vmin, vmax=vmax)

# Add colorbar with sensible ticks - use aspect parameter to make colorbar WIDER (thicker)
cbar = plt.colorbar(im, ax=ax3, fraction=0.15, pad=0.04, aspect=8)
cbar.set_label('CME transit time\nto 1 AU [days]', fontsize=14, rotation=270, labelpad=30)
# Create ticks at 0.1 day intervals
tick_values = np.arange(np.ceil(vmin * 10) / 10, vmax + 0.05, 0.1)
cbar.set_ticks(tick_values)
cbar.ax.tick_params(labelsize=14)
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

# Labels and formatting - remove title, x-label, x-tick labels, and y-label for top plot
ax3.tick_params(labelsize=14, labelbottom=False)  # Remove x-tick labels
ax3.grid(False)

# Right bottom panel: Maximum CME speed at 1 AU
ax4 = plt.subplot(gs[1, 2])

# Plot maximum CME speed at 1 AU over entire run
vmin_speed = np.floor(np.nanmin(v_max_grid) / 50) * 50  # Round to nearest 50 km/s
vmax_speed = np.ceil(np.nanmax(v_max_grid) / 50) * 50
im_speed = ax4.pcolormesh(N_mesh, T_mesh/1e6, v_max_grid, shading='auto', cmap='plasma', vmin=vmin_speed, vmax=vmax_speed)

# Add colorbar - use aspect parameter to make colorbar WIDER (thicker)
cbar_speed = plt.colorbar(im_speed, ax=ax4, fraction=0.15, pad=0.04, aspect=8)
cbar_speed.set_label('CME Speed\nat 1 AU [km/s]', fontsize=14, rotation=270, labelpad=30)
cbar_speed.ax.tick_params(labelsize=14)

# Labels and formatting - remove title, single common y-label centered between both plots
ax4.set_xlabel(r'CME Density at 0.1 AU [cm$^{-3}$]', fontsize=14)
ax4.tick_params(labelsize=14)
ax4.grid(False)

# Add single y-label for both histogram plots, positioned near the y-axis
fig.text(0.69, 0.5, r'CME Temperature at 0.1 AU [$10^6$ K]', fontsize=14, 
         rotation=90, va='center', ha='center')

# Save figure
dbox = os.getenv('DBOX')
if dbox:
    save_dir = os.path.join(dbox, 'Apps', 'Overleaf', 'SHUXt')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'CME_tt_vs_n_T_V2.pdf')
    fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
else:
    print("\nWarning: DBOX environment variable not set. Figure not saved.")

# Create additional figure showing velocity vs time for each parameter combination
fig_vcme = plt.figure(figsize=(14, 10))

# Determine subplot layout based on number of parameter combinations
n_n = len(n_unique)
n_T = len(T_unique)

# Create color maps for density and temperature variations
colors_n = plt.cm.plasma(np.linspace(0, 1, n_n))
colors_T = plt.cm.viridis(np.linspace(0, 1, n_T))

# Plot velocity time series grouped by density
ax_vcme1 = plt.subplot(2, 1, 1)
for i, n_val in enumerate(n_unique):
    for j, T_val in enumerate(T_unique):
        idx = i * n_T + j
        if idx < len(vcme_time_series):
            label = f'n={n_val:.0f} cm$^{{-3}}$' if j == 0 else None
            ax_vcme1.plot(time_series[idx], vcme_time_series[idx], 
                         color=colors_n[i], alpha=0.7, linewidth=1.5, label=label)

ax_vcme1.axhline(y=vambient.value.mean(), color='k', linestyle='--', linewidth=2, label='Ambient')
ax_vcme1.set_xlabel('Time [days]', fontsize=14)
ax_vcme1.set_ylabel('Velocity at 1 AU [km/s]', fontsize=14)
ax_vcme1.set_title('CME Velocity Evolution at 1 AU\n(Colors by Density)', fontsize=14)
ax_vcme1.grid(True, alpha=0.3)
ax_vcme1.tick_params(labelsize=14)
ax_vcme1.legend(loc='best', fontsize=10, ncol=2)

# Plot velocity time series grouped by temperature
ax_vcme2 = plt.subplot(2, 1, 2)
for i, n_val in enumerate(n_unique):
    for j, T_val in enumerate(T_unique):
        idx = i * n_T + j
        if idx < len(vcme_time_series):
            label = f'T={T_val/1e6:.1f} MK' if i == 0 else None
            ax_vcme2.plot(time_series[idx], vcme_time_series[idx], 
                         color=colors_T[j], alpha=0.7, linewidth=1.5, label=label)

ax_vcme2.axhline(y=vambient.value.mean(), color='k', linestyle='--', linewidth=2, label='Ambient')
ax_vcme2.set_xlabel('Time [days]', fontsize=14)
ax_vcme2.set_ylabel('Velocity at 1 AU [km/s]', fontsize=14)
ax_vcme2.set_title('CME Velocity Evolution at 1 AU\n(Colors by Temperature)', fontsize=14)
ax_vcme2.grid(True, alpha=0.3)
ax_vcme2.tick_params(labelsize=14)
ax_vcme2.legend(loc='best', fontsize=10, ncol=2)

plt.tight_layout()

# Save the velocity time series figure
if dbox:
    save_path_vcme = os.path.join(save_dir, 'CME_vcme_vs_time.pdf')
    fig_vcme.savefig(save_path_vcme, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Velocity time series figure saved to: {save_path_vcme}")

plt.show()
