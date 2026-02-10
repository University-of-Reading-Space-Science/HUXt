#<codecell> Imports

import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use('TkAgg')  # Set backend explicitly for Windows
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import os
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

fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax[0].plot(ts_huxt['time'], ts_huxt['vsw'], 'k', label='HUXt')
ax[0].plot(ts_pcm['time'], ts_pcm['vsw'], 'b', label='SURF-hydro-PCM')
ax[0].plot(ts_plm['time'], ts_plm['vsw'], 'r', label='SURF-hydro-PLM')
ax[0].set_ylabel('Velocity [km/s]')
ax[0].legend()
ax[0].set_title('Earth (0 deg) Timeseries Comparison')

ax[1].plot(ts_pcm['time'], ts_pcm['n'], 'b') 
ax[1].plot(ts_plm['time'], ts_plm['n'], 'r')
ax[1].set_ylabel(r'Density [cm$^-3$]')

ax[2].plot(ts_pcm['time'], ts_pcm['T']/10**5, 'b')
ax[2].plot(ts_plm['time'], ts_plm['T']/10**5, 'r')
ax[2].set_ylabel(r'Temperature [$10^5$ K]')

# Format x-axis with DD-MM format
date_format = mdates.DateFormatter('%d-%m')
ax[2].xaxis.set_major_formatter(date_format)
ax[2].xaxis.set_major_locator(mdates.DayLocator(interval=2))

# Get the year from the first time point
year = ts_pcm['time'][0].year
ax[2].set_xlabel(f'Date of {year}')

plt.tight_layout()




plt.show(block=True)
print("\nPlots displayed. Press Enter to close all plots and exit...")
input()