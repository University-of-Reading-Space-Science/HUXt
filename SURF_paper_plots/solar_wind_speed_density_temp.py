
# a script to produce sensitivity plots of soalr wind speed at 1 AU and CME transit time 
# to density and temperature at 0.1 AU.

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

simtime = 0.5*u.day
v_0p1au = 400*u.km/u.s


n_0p1au = np.arange(100, 3001, 300)*u.cm**-3
T_0p1au = np.arange(1e5, 3.1e6, 3e5)*u.K

# Create arrays to store results
n_vals = []
T_vals = []
v_outer_vals = []

mp = 1.6726219e-27*u.kg  # Proton mass

# Loop over all combinations of n and T
for n in n_0p1au:
    for T in T_0p1au:
        print(f"Running model with n={n.value:.0f} cm^-3, T={T.value:.1e} K")
        
        vr_in = np.ones(128) * v_0p1au 
        rho_in = n.to(u.m**-3) * mp * 1000 * np.ones(128) 
        T_in = T * np.ones(128)

        model_plm = H.HUXt(v_boundary=vr_in, rho_boundary=rho_in, temp_boundary=T_in,  
                                simtime=simtime, dt_scale=4, r_max=215*u.solRad, r_min=21.5*u.solRad,
                            solver ='hllc-plm-rk2', lon_out=0.0*u.rad)
        model_plm.solve([])
        
        # Extract velocity at outer boundary at final time
        v_outer = model_plm.v_grid[-1, -1, :].mean()  # Final time, outer radius, mean over longitude
        v_inner_final = model_plm.v_grid[-1, 0, :].mean()  # Final time, inner radius
        
        # Also get final density at outer boundary
        n_outer_final = model_plm.rho_grid[-1, -1, :].mean() / (mp * 1000)  # Convert back to number density
        
        # Store values
        n_vals.append(n.value)
        T_vals.append(T.value)
        v_outer_vals.append(v_outer.value)
        
        print(f"  Inner boundary (final): v={v_inner_final.value:.2f} km/s")
        print(f"  Outer boundary (final): v={v_outer.value:.2f} km/s, n={n_outer_final.value:.1f} cm^-3")
        print(f"  Delta V = {(v_outer-v_inner_final).value:.2f} km/s")




# Convert to numpy arrays and reshape for plotting
n_vals = np.array(n_vals)
T_vals = np.array(T_vals)
v_outer_vals = np.array(v_outer_vals)

# Reshape into 2D grid
n_unique = np.unique(n_vals)
T_unique = np.unique(T_vals)
V_grid = v_outer_vals.reshape(len(n_unique), len(T_unique))

# Create 2D plot
fig, ax = plt.subplots(figsize=(6, 6))

# Create meshgrid for plotting
N_mesh, T_mesh = np.meshgrid(n_unique, T_unique, indexing='ij')

# Plot as pcolormesh
im = ax.pcolormesh(N_mesh, T_mesh/1e6, V_grid, shading='auto', cmap='viridis')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Solar Wind Speed at 1 AU [km/s]', fontsize=14)

# Labels and formatting
ax.set_xlabel(r'Number Density at 0.1 AU [cm$^{-3}$]', fontsize=14)
ax.set_ylabel(r'Temperature at 0.1 AU [$10^6$ K]', fontsize=14)
ax.set_title(f'Solar Wind Speed at 0.1 AU: {v_0p1au.value:.0f} km/s', fontsize=14)

# Add grid
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
dbox = os.getenv('DBOX')
if dbox:
    save_dir = os.path.join(dbox, 'Apps', 'Overleaf', 'SHUXt')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'v_vs_n_T.pdf')
    fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
else:
    print("\nWarning: DBOX environment variable not set. Figure not saved.")







#additionally test CME density and temperature effect on CME arrival time
simtime = 4*u.day
cme_speed = 1000 * u.km/u.s
# Create arrays to store results
n_cme_vals = []
T_cme_vals = []
tt_vals = []

#compute 1 AU speed for 400 km/s inner boundary speed
model_plm = H.HUXt(v_boundary=vr_in, 
                    simtime=simtime, dt_scale=4, r_max=215*u.solRad, r_min=21.5*u.solRad,
                solver ='hllc-plm-rk2', lon_out=0.0*u.rad)
model_plm.solve([])
v_1au = model_plm.v_grid[-1, -1, :].mean()

print('1au speed: ', v_1au)

# Loop over all combinations of n and T
for n in n_0p1au:
    for T in T_0p1au:
        cme = H.ConeCME(t_launch=0*u.day, longitude=0*u.deg, width=60*u.deg,
                        v=cme_speed, 
                cme_density= n.to(u.m**-3) * mp * 1000,
                cme_temperature=T,  profile_type='sinusoidal')
        
        model_plm = H.HUXt(v_boundary=vr_in,  
                                simtime=simtime, dt_scale=4, r_max=215*u.solRad, r_min=21.5*u.solRad,
                            solver ='hllc-plm-rk2', lon_out=0.0*u.rad)
        model_plm.solve([cme])
        
        # find time when speed is above the ambient value at 1 AU
        vcme = model_plm.v_grid[:, -1, 0]
        time_grid = model_plm.time_out

        # Find the first time when the speed exceeds the ambient speed at 1 AU
        times_above_ambient = time_grid[vcme > v_1au + 20*u.km/u.s]
        if len(times_above_ambient) > 0:
            arrival_time = times_above_ambient[0]
            print(f"  CME arrival time at 1 AU: {arrival_time.to(u.day):.2f}")
        else:
            print("  CME did not arrive at 1 AU within simulation time.")
        
        # Store values
        n_cme_vals.append(n.value)
        T_cme_vals.append(T.value)
        tt_vals.append(arrival_time.to(u.day).value)
    

# Convert to numpy arrays and reshape for plotting
n_cme_vals = np.array(n_cme_vals)
T_cme_vals = np.array(T_cme_vals)
tt_vals = np.array(tt_vals)

# Reshape into 2D grid
n_unique = np.unique(n_cme_vals)
T_unique = np.unique(T_cme_vals)
tt_grid = tt_vals.reshape(len(n_unique), len(T_unique))

# Create 2D plot
fig, ax = plt.subplots(figsize=(6, 6))

# Create meshgrid for plotting
N_mesh, T_mesh = np.meshgrid(n_unique, T_unique, indexing='ij')

# Plot as pcolormesh
im = ax.pcolormesh(N_mesh, T_mesh/1e6, tt_grid, shading='auto', cmap='viridis')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('CME Arrival Time at 1 AU [days]', fontsize=14)

# Labels and formatting
ax.set_xlabel(r'CME Density at 0.1 AU [cm$^{-3}$]', fontsize=14)
ax.set_ylabel(r'CME Temperature at 0.1 AU [$10^6$ K]', fontsize=14)
ax.set_title(f'CME speed at 0.1 AU: {cme_speed.value:.0f} km/s', fontsize=14)

# Add grid
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
dbox = os.getenv('DBOX')
if dbox:
    save_dir = os.path.join(dbox, 'Apps', 'Overleaf', 'SHUXt')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'CME_tt_vs_n_T.pdf')
    fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
else:
    print("\nWarning: DBOX environment variable not set. Figure not saved.")












plt.show()

