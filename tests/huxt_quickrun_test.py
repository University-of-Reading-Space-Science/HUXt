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

standard_tests = False
compressible_tests = False
insitu_tests = False
insitu_compressible_tests = True

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

    time_grid, vgrid_carr, bgrid_carr, ngrid_carr, tgrid_carr = HI.generate_vCarr_from_OMNI(ftime, 
                                                                                        ftime +datetime.timedelta(days=27), 
                                                                                        compressible=True)
    
    # Set up the model, with (optional) time-dependent bpol boundary conditions
    model = Hin.set_time_dependent_boundary(vgrid_carr, time_grid, ftime, 10*u.day, r_min=210*u.solRad, r_max=240*u.solRad, 
                                            dt_scale=4, solver = 'hllc',
                                        latitude=0*u.deg, bgrid_Carr = bgrid_carr, lon_start=230*u.deg, lon_stop=60*u.deg,)
    model.solve([])
    HA.plot_earth_timeseries(model)

# <codecell> End of script

plt.show(block=True)
print("\nPlots displayed. Press Enter to close all plots and exit...")
input()