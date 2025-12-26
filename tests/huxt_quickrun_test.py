#<codecell> Imports

import os

import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use('TkAgg')  # Set backend explicitly for Windows
import matplotlib.pyplot as plt
import datetime
import os
import sys

for path in ['', '.']:
    while path in sys.path:
        sys.path.remove(path)

# Clear the cached (wrong) huxt import
if 'huxt' in sys.modules:
    del sys.modules['huxt']

import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin
import huxt.huxt_insitu as HI

standard_tests = True
compressible_tests = True
insitu_tests = False
insitu_compressible_tests = False

# <codecell> Upwind and compressible tests

if standard_tests:
    print("="*60)
    print("SCRIPT STARTED - huxt_quickrun_test.py")
    print("="*60)
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    plt.ion()  # Turn on interactive mode
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
    dirs = H._setup_dirs_()
    data_path=dirs['example_inputs']
    print(data_path)
    filepath = os.path.join(data_path, 'wsa_gong_2024050906.fits')
    vr_in = Hin.get_WSA_long_profile(filepath, lat=0.0 * u.deg)
    br_in = Hin.get_WSA_br_long_profile(filepath, lat=0.0 * u.deg)

    dirs = H._setup_dirs_()
    print(dirs)


    simtime = 1.1*u.day
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

    t_interest = 1*u.day

    print("\n" + "="*60)
    print("Starting incompressible model (upwind solver)...")
    print("="*60)
    model_incomp = H.HUXt(v_boundary=vr_in, b_boundary=br_in, #lon_start=350*u.deg, lon_stop = 10*u.deg, 
                        #lon_out=0.0*u.rad,
                        simtime=simtime, dt_scale=4, 
                        solver ='upwind')
    print("Model initialized. Starting solve...")
    model_incomp.solve(cme_list, streak_carr=lon_grid)
    print("Solve complete!")
    #model_incomp.solve([])
    print("Creating plot 1: Earth timeseries (incompressible)...")
    HA.plot_earth_timeseries(model_incomp, plot_omni=False)
    print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")
    ts_upwind = HA.get_observer_timeseries(model_incomp)
    print("Creating plot 2: Spatial plot (incompressible)...")
    HA.plot(model_incomp, time=t_interest)
    print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")


if compressible_tests:
    print("\n" + "="*60)
    print("Starting compressible model (HLLC solver)...")
    print("="*60)
    model_comp_cgf = H.HUXt(v_boundary=vr_in, b_boundary=br_in,#lon_start=350*u.deg, lon_stop = 10*u.deg,
                            #lon_out=0.0*u.rad,
                            simtime=simtime, dt_scale=4, 
                        solver ='hllc')
    print("Model initialized. Starting solve...")
    model_comp_cgf.solve(cme_list, streak_carr=lon_grid)
    print("Solve complete!")
    #model_comp_cgf.solve([])
    print("Creating plot 3: Earth timeseries (compressible)...")
    HA.plot_earth_timeseries(model_comp_cgf, plot_omni=False)
    print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")
    ts_cgf = HA.get_observer_timeseries(model_comp_cgf)

    print("Creating plot 4: Compressible spatial plot...")
    HA.plot_compressible(model_comp_cgf, time=t_interest)
    print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")

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



# Display all plots
# print(f"\nTotal figures created: {len(plt.get_fignums())}")
# print("Figure numbers:", plt.get_fignums())
# print("\nAttempting to display plots...")
# plt.show(block=True)
# print("\nPlots displayed. Press Enter to close all plots and exit...")
# input()

#HA.plot(model_incomp, t_interest)
#HA.animate(model_incomp, tag =  'incompressible_with_CME')



#<codecell> InSitu-HUXt integration test

if insitu_tests:

    print("="*60)
    print("InSitu-HUXt integration test")
    print("="*60)


    ftime = datetime.datetime(2022,12,1)
    is_model = HI.omniHUXt_forecast(ftime, simtime=27.27*u.day, 
                            rmin=21.5*u.solRad, rmax=230*u.solRad, 
                            dt_scale=4,
                            omni_input=None, buffertime=5*u.day,
                            run_2d=False)

    is_model.solve([])
    HA.plot_earth_timeseries(is_model)



# <cocdecell> InSitu-HUXt compressible test

if insitu_compressible_tests:
    ftime = datetime.datetime(2022,12,1)
    cr, cr_lon_init = HI.datetime2huxtinputs(ftime)

    # time_grid, vgrid_carr, bgrid_carr, rhogrid_carr, tgrid_carr = HI.generate_vCarr_from_OMNI(ftime, 
    #                                                                                     ftime +datetime.timedelta(days=27), 
    #                                                                                     compressible=True)
    
    # # Set up the model, with (optional) time-dependent bpol boundary conditions
    # model = Hin.set_time_dependent_boundary(vgrid_carr, time_grid, ftime, 10*u.day, r_min=210*u.solRad, r_max=240*u.solRad, 
    #                                         dt_scale=4, solver = 'hllc',
    #                                     latitude=0*u.deg, bgrid_Carr = bgrid_carr, 
    #                                     rhogrid_Carr = rhogrid_carr, tempgrid_Carr= tgrid_carr, 
    #                                     lon_start=230*u.deg, lon_stop=60*u.deg,)
    # model.solve([])
    # HA.plot_earth_timeseries(model)



    model = HI.omniHUXt_reconstruction(ftime, ftime +datetime.timedelta(days=27), 
                            rmin=21.5*u.solRad, rmax=250*u.solRad, 
                            dt_scale=4, dt=1*u.day,
                            run_2d=True,
                            solver='hllc',
                            rho_source='omni',
                            temp_source='omni')
    
    model.solve([])
    
    # Plot Earth timeseries
    HA.plot_earth_timeseries(model)
    
    # Function to plot observer timeseries in the same format as Earth plot
    def plot_observer_timeseries(model, observer_name, observer_label):
        """Plot timeseries for a specific observer in same format as Earth plot"""
        obs_ts = HA.get_observer_timeseries(model, observer=observer_name)
        
        is_compressible = hasattr(model, 'compressible') and model.compressible
        
        # Determine number of panels
        n_panels = 1  # Velocity
        if hasattr(model, 'b_grid'):
            n_panels += 1
        if is_compressible:
            n_panels += 2  # Density and temperature
        
        fig, axs = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels))
        if n_panels == 1:
            axs = np.array([axs])
        
        # Velocity panel
        panel_idx = 0
        axs[panel_idx].plot(obs_ts['time'], obs_ts['vsw'], 'r', label='HUXt')
        axs[panel_idx].set_ylim(250, 1000)
        axs[panel_idx].set_ylabel('Solar Wind Speed (km/s)')
        
        # B polarity panel
        if hasattr(model, 'b_grid'):
            panel_idx += 1
            axs[panel_idx].plot(obs_ts['time'], np.sign(obs_ts['bpol']), 'r.', label='HUXt')
            axs[panel_idx].set_ylabel('B polarity')
            axs[panel_idx].set_ylim(-1.1, 1.1)
        
        # Density panel
        if is_compressible and 'n' in obs_ts.columns:
            panel_idx += 1
            axs[panel_idx].semilogy(obs_ts['time'], obs_ts['n'], 'r-', label='HUXt')
            axs[panel_idx].set_ylabel('n (protons/cm³)')
            axs[panel_idx].set_ylim(0.1, 1000)
            axs[panel_idx].grid(True, alpha=0.3)
        
        # Temperature panel
        if is_compressible and 'T' in obs_ts.columns:
            panel_idx += 1
            axs[panel_idx].semilogy(obs_ts['time'], obs_ts['T'], 'r-', label='HUXt')
            axs[panel_idx].set_ylabel('T (K)')
            axs[panel_idx].set_ylim(1e4, 1e7)
            axs[panel_idx].grid(True, alpha=0.3)
        
        starttime = obs_ts['time'].iloc[0]
        endtime = obs_ts['time'].iloc[-1]
        
        for a in axs:
            a.set_xlim(starttime, endtime)
            a.legend()
        
        # Only last panel gets x-label
        for i in range(len(axs) - 1):
            axs[i].set_xticklabels([])
        axs[-1].set_xlabel('Date')
        
        fig.suptitle(f'{observer_label} - HUXt Reconstruction', fontsize=14, y=0.995)
        plt.tight_layout()
        
        return fig, axs, obs_ts
    
    # Plot STEREO-A timeseries
    print("\n" + "="*60)
    print("Plotting STEREO-A timeseries")
    print("="*60)
    fig_sta, axs_sta, sta_ts = plot_observer_timeseries(model, 'STA', 'STEREO-A')
    
    print("\nSTEREO-A Statistics:")
    print(f"  Velocity: {sta_ts['vsw'].mean():.1f} ± {sta_ts['vsw'].std():.1f} km/s")
    if 'n' in sta_ts.columns:
        print(f"  Density: {sta_ts['n'].mean():.2f} ± {sta_ts['n'].std():.2f} cm^-3")
    if 'T' in sta_ts.columns:
        print(f"  Temperature: {sta_ts['T'].mean():.1e} ± {sta_ts['T'].std():.1e} K")
    
    # Plot Solar Orbiter timeseries
    print("\n" + "="*60)
    print("Plotting Solar Orbiter timeseries")
    print("="*60)
    fig_solo, axs_solo, solo_ts = plot_observer_timeseries(model, 'SOLO', 'Solar Orbiter')
    
    print("\nSolar Orbiter Statistics:")
    print(f"  Velocity: {solo_ts['vsw'].mean():.1f} ± {solo_ts['vsw'].std():.1f} km/s")
    if 'n' in solo_ts.columns:
        print(f"  Density: {solo_ts['n'].mean():.2f} ± {solo_ts['n'].std():.2f} cm^-3")
    if 'T' in solo_ts.columns:
        print(f"  Temperature: {solo_ts['T'].mean():.1e} ± {solo_ts['T'].std():.1e} K")

# <codecell> End of script

plt.show(block=True)
print("\nPlots displayed. Press Enter to close all plots and exit...")
input()