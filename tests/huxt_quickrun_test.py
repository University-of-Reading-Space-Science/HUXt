import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import datetime
import os

import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin

"""
# Form longitudinal boundary conditions - background wind of 400 km/s with two fast streams.
v_boundary = np.ones(128) * 400 * (u.km/u.s)
v_boundary[30:50] = 600 * (u.km/u.s)
v_boundary[95:125] = 700 * (u.km/u.s)

 # This boundary condition looks like
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(v_boundary,'k-')
ax.set_xlabel('Longitude bin')
ax.set_ylabel('Input Wind Speed (km/s)')

# Setup HUXt to do a 5-day simulation, with model output every 4 timesteps (roughly half and hour time step), looking at 0 longitude
model = H.HUXt(v_boundary=v_boundary, lon_out=0.0*u.deg, simtime=10*u.day, dt_scale=4, compressible=True)

# Solve these conditions, with no ConeCMEs added.
cme_list = []
model.solve(cme_list)



# Plot the time series of the ambient wind profile at a fixed radius. 
r = 1.0*u.AU
#HA.plot_timeseries(model, r, lon=0.0)



# Save the data
out_path = model.save(tag='cone_cme_test')

# And load it back in with
model2, cme_list2 = H.load_HUXt_run(out_path)

# Plot the time series of the ambient wind profile at a fixed radius. 
r = 1.0*u.AU
HA.plot_timeseries(model2, r, lon=0.0)
 """



# Set up HUXt over a limited longitude range.
# dirs = H._setup_dirs_()
# data_path=dirs['example_inputs']
# print(data_path)
# filepath = os.path.join(data_path, 'wsa_gong_2024050906.fits')
# vr_in = Hin.get_WSA_long_profile(filepath, lat=0.0 * u.deg)

dirs = H._setup_dirs_()
print(dirs)


simtime = 10 *u.day
# Set up HUXt
cr=1920
vr_in = np.ones(128)*400*u.km/u.s #Hin.get_MAS_long_profile(cr, 0.0*u.deg)



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
                    density_fraction=0.1, temperature_fraction=0.01, profile_type='sinusoidal')
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

t_interest = 2.01*u.day


model_incomp = H.HUXt(cr_num=cr,
                      v_boundary=vr_in, #lon_start=350*u.deg, lon_stop = 10*u.deg, 
                      #lon_out=0.0*u.rad,
                      simtime=simtime, dt_scale=4, 
                    compressible=False, solver ='upwind')
model_incomp.solve(cme_list)#, streak_carr=lon_grid)
HA.plot_earth_timeseries(model_incomp, plot_omni=False)
ts_upwind = HA.get_observer_timeseries(model_incomp)
HA.plot(model_incomp, time=t_interest)

model_comp_cgf = H.HUXt(cr_num=cr,
                        v_boundary=vr_in, #lon_start=350*u.deg, lon_stop = 10*u.deg,
                        #lon_out=0.0*u.rad,
                        simtime=simtime, dt_scale=4, 
                    compressible=True, solver ='cgf')
model_comp_cgf.solve(cme_list)#, streak_carr=lon_grid)
HA.plot_earth_timeseries(model_comp_cgf, plot_omni=False)
ts_cgf = HA.get_observer_timeseries(model_comp_cgf)

HA.plot_compressible(model_comp_cgf, time=t_interest)

# model_comp_pluto = H.HUXt(cr_num=cr,
#                         v_boundary=vr_in, #lon_start=350*u.deg, lon_stop = 10*u.deg,
#                         lon_out=0.0*u.rad,
#                         simtime=simtime, dt_scale=4, 
#                     compressible=True, solver ='pluto')
# model_comp_pluto.solve(cme_list)#, streak_carr=lon_grid)
# HA.plot_earth_timeseries(model_comp_pluto)
#HA.plot_timeseries(model_comp_cgf, 0.2*u.AU, lon=0.0)
#HA.plot(model_comp_cgf, time=t_interest)
# HA.animate(model_comp, tag = 'compressible_with_CME')

# Check CME particles at 2 days
print("\nChecking CME particles at t=2 days:")
t_idx = np.argmin(np.abs(model_comp_cgf.time_out - 2*u.day))
print(f"Time index: {t_idx}, Time: {model_comp_cgf.time_out[t_idx]}")

# cme_particles_r shape: (n_cme, nt_out, 2, nlon)
# Check CME 0, Leading edge (0)
cme_r = model_comp_cgf.cme_particles_r[0, t_idx, 0, :]
print("CME 0 Leading Edge Radial Positions (km):")
# Print a subset or summary to avoid spamming, but user asked to output them.
# Let's print them all if it's 128 longitudes.
print(cme_r)

# Check for zeros or small values (e.g. < 1 solar radius ~ 700000 km, or just < 1 km if it's really 0)
# The boundary is at 30rs ~ 2e7 km. So anything small is suspicious.
mask_small = (cme_r < 1000.0) & (~np.isnan(cme_r))
if np.any(mask_small):
    print("WARNING: Found small/zero values in CME particles:")
    print(cme_r[mask_small])
    print("Indices:", np.where(mask_small)[0])
else:
    print("No small/zero values found.")

# Check for NaNs
print(f"Number of NaNs: {np.isnan(cme_r).sum()}")


# Use block=True to ensure plot windows stay open until closed by user
plt.show(block=True)


#HA.plot(model_incomp, t_interest)
#HA.animate(model_incomp, tag =  'incompressible_with_CME')
