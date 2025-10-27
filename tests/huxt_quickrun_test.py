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

t_interest = 2*u.day
HA.plot(model_comp, t_interest)
HA.plot_compressible(model_comp, t_interest)

HA.plot_earth_timeseries(model_comp)

HA.animate(model_comp, tag = 
           'compressible_with_CME')


#Repeat with incompressible model
model_incomp = H.HUXt(v_boundary=vr_in, lon_start=300*u.deg, lon_stop=60*u.deg, simtime=5*u.day, dt_scale=4,
                        compressible=False)
model_incomp.solve(cme_list)

HA.plot(model_incomp, t_interest)

# ============================================================================
# Compressible run with CME
# ============================================================================
print('\nRunning compressible model with CME...')

# Create a compressible model with simple boundary condition

model_comp = H.HUXt(v_boundary=v_boundary, lon_out=0.0*u.deg, 
                    simtime=27*u.day, dt_scale=4, compressible=True)

# Add a CME
cme = H.ConeCME(t_launch=2*u.day, longitude=0*u.deg, width=30*u.deg, 
                v=800*(u.km/u.s), thickness=5*u.solRad)

# Solve with the CME
model_comp.solve([cme])

# Print diagnostics
print(f'input_v_ts shape: {model_comp.input_v_ts.shape}')
print(f'input_v_ts min/max at lon=0: {model_comp.input_v_ts[:, 0].min():.1f} / {model_comp.input_v_ts[:, 0].max():.1f}')
print(f'input_v_ts mean/std at lon=0: {model_comp.input_v_ts[:, 0].mean():.1f} ± {model_comp.input_v_ts[:, 0].std():.1f}')

# Plot the time series at the inner boundary
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# Convert model time to hours
time_hours = model_comp.model_time.to(u.hour).value

# Get the inner boundary time series for longitude 0
lon_idx = 0

# Velocity
axes[0].plot(time_hours, model_comp.input_v_ts[:, lon_idx].value, 'b-', linewidth=1.5)
axes[0].set_ylabel('Velocity (km/s)')
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Inner Boundary Time Series (Compressible Run with CME)')

# Density
axes[1].plot(time_hours, model_comp.input_rho_ts[:, lon_idx].to(u.kg/u.m**3).value, 'r-', linewidth=1.5)
axes[1].set_ylabel('Density (kg/m³)')
axes[1].grid(True, alpha=0.3)
axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Temperature
axes[2].plot(time_hours, model_comp.input_temp_ts[:, lon_idx].to(u.K).value, 'g-', linewidth=1.5)
axes[2].set_ylabel('Temperature (K)')
axes[2].set_xlabel('Time (hours)')
axes[2].grid(True, alpha=0.3)
axes[2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()

print(f'CME launch time: {cme.t_launch}')
print(f'CME velocity: {cme.v}')
print(f'CME density: {cme.cme_density:.2e}')
print(f'CME temperature: {cme.cme_temperature:.2e}')

# ============================================================================
# Plot time series at 1 AU (215 Rs)
# ============================================================================
print('\nExtracting time series at 1 AU...')

# Find the radial index closest to 215 Rs (1 AU)
r_target = 215 * u.solRad
r_idx = np.argmin(np.abs(model_comp.r - r_target))
r_actual = model_comp.r[r_idx]
print(f'Target radius: {r_target:.1f}, Actual radius: {r_actual:.1f} (index {r_idx})')

# Get the output time (not including spin-up buffer)
time_out = model_comp.time_out.to(u.hour).value

# Extract time series at 1 AU for longitude 0
lon_idx = 0  # longitude 0 degrees
v_1au = model_comp.v_grid[:, r_idx, lon_idx].to(u.km/u.s).value

# For density and temperature, we need to check if they exist in the model
if hasattr(model_comp, 'rho_grid'):
    rho_1au = model_comp.rho_grid[:, r_idx, lon_idx].to(u.kg/u.m**3).value
    temp_1au = model_comp.temp_grid[:, r_idx, lon_idx].to(u.K).value
    has_rho_temp = True
else:
    print('Note: rho_grid and temp_grid not found - compressible solver may need to populate these')
    has_rho_temp = False

# Create the plot
fig2, axes2 = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# Velocity at 1 AU
axes2[0].plot(time_out, v_1au, 'b-', linewidth=1.5)
axes2[0].set_ylabel('Velocity (km/s)')
axes2[0].grid(True, alpha=0.3)
axes2[0].set_title(f'Time Series at 1 AU ({r_actual:.1f})')

if has_rho_temp:
    # Density at 1 AU
    axes2[1].plot(time_out, rho_1au, 'r-', linewidth=1.5)
    axes2[1].set_ylabel('Density (kg/m³)')
    axes2[1].grid(True, alpha=0.3)
    axes2[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Temperature at 1 AU
    axes2[2].plot(time_out, temp_1au, 'g-', linewidth=1.5)
    axes2[2].set_ylabel('Temperature (K)')
    axes2[2].set_xlabel('Time (hours)')
    axes2[2].grid(True, alpha=0.3)
    axes2[2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
else:
    axes2[1].text(0.5, 0.5, 'Density data not available', 
                  ha='center', va='center', transform=axes2[1].transAxes)
    axes2[2].text(0.5, 0.5, 'Temperature data not available', 
                  ha='center', va='center', transform=axes2[2].transAxes)
    axes2[1].set_ylabel('Density (kg/m³)')
    axes2[2].set_ylabel('Temperature (K)')
    axes2[2].set_xlabel('Time (hours)')

plt.tight_layout()

plt.show()