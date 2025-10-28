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



# Set up HUXt
cr=2120
vr_in = Hin.get_MAS_long_profile(cr, 0.0*u.deg)

model = H.HUXt(v_boundary=vr_in, cr_num = cr, cr_lon_init = 50*u.deg, simtime=5*u.day, dt_scale=1, frame='sidereal', lon_start=300*u.deg, lon_stop=60*u.deg)
# Add a CME
cme = H.ConeCME(t_launch=2.5*u.day, longitude=0.0*u.deg, width=40*u.deg, v=1000*(u.km/u.s), thickness=5*u.solRad)
cme_list = [cme]

# Set up to trace a set of field lines from a range of evenly spaced Carrington longitudes
dlon = (20*u.deg).to(u.rad).value
lon_grid = np.arange(dlon/2, 2*np.pi-dlon/2 + 0.0001, dlon)*u.rad

# Give the streakline footpoints (in Carr long) to the solve method
model.solve(cme_list, streak_carr=lon_grid)

# Plot these streaklines
time = 4.5*u.day
fig, ax = HA.plot(model,time, save = 'True', tag = 'CR2254_streaklines')


model_comp = H.HUXt(v_boundary=vr_in, lon_start=300*u.deg, lon_stop=60*u.deg, simtime=5*u.day, dt_scale=4, 
                    compressible=True)


# Get a list of two ConeCMEs
daysec = 86400
times = [0.5*daysec, 2*daysec]
speeds = [1000, 850]
lons = [-20, 20]
widths = [60, 60]
thickness = [8, 4]
cme_list = []
for t, l, w, v, thick in zip(times, lons, widths, speeds, thickness):
    cme = H.ConeCME(t_launch=t*u.s, longitude=l*u.deg, width=w*u.deg, v=v*model_comp.kms, 
                    thickness=thick*u.solRad, density_fraction=1.0)
    cme_list.append(cme)


# Set up to trace a set of field lines from a range of evenly spaced Carrington longitudes
dlon = (20*u.deg).to(u.rad).value
lon_grid = np.arange(dlon/2, 2*np.pi-dlon/2 + 0.0001, dlon)*u.rad

# Give the streakline footpoints (in Carr long) to the solve method
model_comp.solve(cme_list, streak_carr=lon_grid)

t_interest = 3*u.day
HA.plot(model_comp, t_interest)
HA.plot_compressible(model_comp, t_interest, tag = 'compressible_with_CME')

HA.plot_earth_timeseries(model_comp)

#HA.animate(model_comp, tag = 'compressible_with_CME')

plt.show()
# #Repeat with incompressible model
# model_incomp = H.HUXt(v_boundary=vr_in, lon_start=300*u.deg, lon_stop=60*u.deg, simtime=5*u.day, dt_scale=4,
#                         compressible=False)
# model_incomp.solve(cme_list)

#HA.plot(model_incomp, t_interest)
#HA.animate(model_incomp, tag =  'incompressible_with_CME')
