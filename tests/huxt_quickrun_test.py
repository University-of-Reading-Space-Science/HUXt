import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import datetime
import os

import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin


# Form longitudinal boundary conditions - background wind of 400 km/s with two fast streams.
v_boundary = np.ones(128) * 400 * (u.km/u.s)
v_boundary[30:50] = 600 * (u.km/u.s)
v_boundary[95:125] = 700 * (u.km/u.s)

""" # This boundary condition looks like
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
dirs = H._setup_dirs_()
data_path=dirs['example_inputs']
print(data_path)
filepath = os.path.join(data_path, 'wsa_gong_2024050906.fits')
vr_in = Hin.get_WSA_long_profile(filepath, lat=0.0 * u.deg)
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
    cme = H.ConeCME(t_launch=t*u.s, longitude=l*u.deg, width=w*u.deg, v=v*model_comp.kms, thickness=thick*u.solRad,
                     cme_density=20e-18*(u.kg/u.m**3), cme_temperature=3e6*u.K)
    cme_list.append(cme)

model_comp.solve(cme_list)

t_interest = 1.6*u.day
#HA.plot(model_comp, t_interest)
HA.plot_compressible(model_comp, t_interest, tag = 'compressible_with_CME')

HA.plot_earth_timeseries(model_comp)

HA.animate(model_comp, tag = 
           'compressible_with_CME')


#Repeat with incompressible model
model_incomp = H.HUXt(v_boundary=vr_in, lon_start=300*u.deg, lon_stop=60*u.deg, simtime=5*u.day, dt_scale=4,
                        compressible=False)
model_incomp.solve(cme_list)

#HA.plot(model_incomp, t_interest)

