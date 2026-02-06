#<codecell> Imports

import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use('TkAgg')  # Set backend explicitly for Windows
import matplotlib.pyplot as plt
import datetime
import os
import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin
import huxt.huxt_insitu as Hinsitu

simtime = 5*u.day

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
                    #density_fraction=0.1, temperature_fraction=0.1,
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

t_interest = 2*u.day


print("\n" + "="*60)
print("Starting compressible model (HLLC + PLM + RK2 2nd Order)...")
print("="*60)
model_plm = H.HUXt(v_boundary=vr_in, b_boundary=br_in,
                        simtime=5*u.day, dt_scale=4, 
                    solver ='hllc-plm-rk2') # Explicitly request PLM+RK2
print("Model initialized. Starting solve...")
t0 = datetime.datetime.now()
model_plm.solve(cme_list, streak_carr=lon_grid)
dt_plm = (datetime.datetime.now() - t0).total_seconds()
print(f"Solve complete in {dt_plm:.2f}s!")


print("Creating plot 6: Compressible spatial plot (PLM)...")
HA.plot_compressible_with_ts(model_plm, time=t_interest)

HA.plot_compressible(model_plm, time=t_interest)
HA.animate_compressible_with_ts(model_plm, tag='test', duration=10, fps=20)

plt.show(block=True)
print("\nPlots displayed. Press Enter to close all plots and exit...")
input()
