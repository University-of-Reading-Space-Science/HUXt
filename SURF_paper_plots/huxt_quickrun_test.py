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


save_figs = False

standard_tests = True
compressible_tests = True
insitu_compressible_tests = False

simtime = 7*u.day

# <codecell> Upwind and compressible tests

if standard_tests:
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
    cr, cr_lon_init = Hin.datetime2huxtinputs(datetime.datetime(2024,5,9,6))
    cr=1800
    cr_lon_init = 2*u.rad
    vr_in = Hin.get_WSA_long_profile(filepath, lat=0.0 * u.deg)
    br_in = Hin.get_WSA_br_long_profile(filepath, lat=0.0 * u.deg)
    dirs = H._setup_dirs_()
    print(dirs)

    t_interest = 2*u.day

    # Set up HUXt
    #cr=1920
    #vr_in = np.ones(128)*400*u.km/u.s #Hin.get_MAS_long_profile(cr, 0.0*u.deg)



    # Set up to trace a set of field lines from a range of evenly spaced Carrington longitudes
    dlon = (20*u.deg).to(u.rad).value
    lon_grid = np.arange(dlon/2, 2*np.pi-dlon/2 + 0.0001, dlon)*u.rad



    # Get a list of two ConeCMEs
    daysec = 86400
    times = [0*u.day]
    speeds = [800]
    lons = [0]
    widths = [60]
    thickness = [5]
    cme_list = []
    for t, l, w, v, thick in zip(times, lons, widths, speeds, thickness):
        cme = H.ConeCME(t_launch=t, longitude=l*u.deg, width=w*u.deg, v=v*u.km/u.s, 
                        thickness=thick*u.solRad, 
                        density_fraction=1, temperature_fraction=1,
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

   

    print("\n" + "="*60)
    print("Starting incompressible model (upwind solver)...")
    print("="*60)
    model_incomp = H.HUXt(v_boundary=vr_in, b_boundary=br_in, #lon_start=350*u.deg, lon_stop = 10*u.deg, 
                        #lon_out=0.0*u.rad,
                        r_min=21.5*u.solRad, simtime=simtime, dt_scale=4, 
                        cr_num = cr, cr_lon_init=cr_lon_init,
                        solver ='huxt')
    print("Model initialized. Starting solve...")
    t0 = datetime.datetime.now()
    model_incomp.solve(cme_list, streak_carr=lon_grid)
    dt_pcm = (datetime.datetime.now() - t0).total_seconds()
    print(f"Solve complete in {dt_pcm:.2f}s!")
    #model_incomp.solve([])
    print("Creating plot 1: Earth timeseries (incompressible)...")
    HA.plot_earth_timeseries(model_incomp, plot_omni=False, timefromrunstart = 'True')
    print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")
    ts_upwind = HA.get_observer_timeseries(model_incomp)
    print("Creating plot 2: Spatial plot (incompressible)...")
    HA.plot(model_incomp, time=t_interest)
    print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")

    #HA.animate(model_incomp, tag =  'incompressible_with_CME')


if compressible_tests:
    # # ---------------------------------------------------------
    # # Test 1: First-order HLLC + PCM
    # # ---------------------------------------------------------
    # print("\n" + "="*60)
    # print("Starting compressible model (HLLC + PCM 1st Order)...")
    # print("="*60)
    # model_pcm = H.HUXt(v_boundary=vr_in, b_boundary=br_in,
    #                         simtime=simtime, dt_scale=4, r_min=21.5*u.solRad, 
    #                     solver ='hllc-pcm') # Explicitly request PCM
    # print("Model initialized. Starting solve...")
    # t0 = datetime.datetime.now()
    # model_pcm.solve(cme_list, streak_carr=lon_grid)
    # dt_pcm = (datetime.datetime.now() - t0).total_seconds()
    # print(f"Solve complete in {dt_pcm:.2f}s!")

    # print("Creating plot 3: Earth timeseries (PCM)...")
    # HA.plot_earth_timeseries(model_pcm, plot_omni=False)
    # plt.title("HLLC + PCM (1st Order)")
    # print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")

    # print("Creating plot 4: Compressible spatial plot (PCM)...")
    # HA.plot_compressible(model_pcm, time=t_interest)
    # plt.suptitle("HLLC + PCM (1st Order)")
    # print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")

    # ---------------------------------------------------------
    # Test 2: Second-order HLLC + PLM + RK2
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Starting compressible model (HLLC + PLM + RK2 2nd Order)...")
    print("="*60)
    model_plm = H.HUXt(v_boundary=vr_in, b_boundary=br_in,
                            simtime=simtime, dt_scale=4, r_min=21.5*u.solRad, 
                            cr_num = cr, cr_lon_init=cr_lon_init,
                        solver ='hydro') # Explicitly request PLM+RK2
    print("Model initialized. Starting solve...")
    t0 = datetime.datetime.now()
    model_plm.solve(cme_list, streak_carr=lon_grid)
    dt_plm = (datetime.datetime.now() - t0).total_seconds()
    print(f"Solve complete in {dt_plm:.2f}s!")
    
    #add the ambient series
    model_ambient = H.HUXt(v_boundary=vr_in, b_boundary=br_in,
                        simtime=simtime, dt_scale=4, r_min=21.5*u.solRad, 
                        cr_num = cr, cr_lon_init=cr_lon_init,
                    solver ='hydro') # Explicitly request PLM+RK2
    model_ambient.solve([])
    ts_ambient = HA.get_observer_timeseries(model_ambient)
    amb_times = model_ambient.time_out.to(u.day).value

    print("Creating plot 5: Earth timeseries (PLM)...")
    fig, axes = HA.plot_earth_timeseries(model_plm, plot_omni=False, timefromrunstart='True')
    # Overlay ambient timeseries as black dashed lines
    panel_idx = 0
    axes[panel_idx].plot(amb_times, ts_ambient['vsw'], 'k--', label='SURF-hydro (ambient)')
    axes[panel_idx].legend()
    if hasattr(model_plm, 'b_grid') and 'bpol' in ts_ambient.columns:
        panel_idx += 1
        axes[panel_idx].plot(amb_times, np.sign(ts_ambient['bpol']), 'k.', label='SURF-hydro (ambient)')
        axes[panel_idx].legend()
    if 'n' in ts_ambient.columns:
        panel_idx += 1
        axes[panel_idx].semilogy(amb_times, ts_ambient['n'], 'k--', label='SURF-hydro (ambient)')
        axes[panel_idx].legend()
    if 'T' in ts_ambient.columns:
        panel_idx += 1
        axes[panel_idx].semilogy(amb_times, ts_ambient['T'], 'k--', label='SURF-hydro (ambient)')
        axes[panel_idx].legend()
    #plt.title("HLLC + PLM + RK2 (2nd Order)")
    print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")

    # Add CME leading/trailing edge arrival times as vertical lines
    r_threshold = 215 * 6.96e5  # 1 AU in km
    plot_times = model_plm.time_out.to(u.day).value
    n_particles = model_plm.cme_particles_r.shape[2]
    n_lons = model_plm.cme_particles_r.shape[3]
    lon_idx = 0  # use first tracked longitude

    if n_lons > 0:
        for cme_id in range(len(cme_list)):
            cme_r_lead = model_plm.cme_particles_r[cme_id, :, 0, lon_idx].value
            cme_r_trail = model_plm.cme_particles_r[cme_id, :, 1, lon_idx].value if n_particles > 1 else cme_r_lead

            valid_lead = np.isfinite(cme_r_lead) & (cme_r_lead >= r_threshold)
            t_arrival = plot_times[np.where(valid_lead)[0][0]] if np.any(valid_lead) else None

            valid_trail = np.isfinite(cme_r_trail) & (cme_r_trail >= r_threshold)
            t_end = plot_times[np.where(valid_trail)[0][0]] if np.any(valid_trail) else None

            if t_arrival is not None:
                for a in axes:
                    a.axvline(t_arrival, color='r', linestyle='-', linewidth=2, alpha=0.6)
            if t_end is not None:
                for a in axes:
                    a.axvline(t_end, color='r', linestyle='--', linewidth=2, alpha=0.6)
    if save_figs:
        dbox = os.environ.get('DBOX')
        overleaf_dir = os.path.join(dbox, 'Apps', 'Overleaf', 'SHUXt')
        pdf_path = os.path.join(overleaf_dir, 'CME_example_ts.pdf')
        plt.savefig(pdf_path, dpi=150)


    print("Creating plot 6: Compressible spatial plot (PLM)...")
    HA.plot_compressible(model_plm, time=t_interest)
    #plt.suptitle("HLLC + PLM + RK2 (2nd Order)")
    print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")
    if save_figs:

        pdf_path = os.path.join(overleaf_dir, 'CME_example.pdf')
        plt.savefig(pdf_path, dpi=150)

    # # ---------------------------------------------------------
    # # Comparison Plot
    # # ---------------------------------------------------------
    # print("\nCreating plot 7: Direct comparison of Earth Profiles...")
    # ts_pcm = HA.get_observer_timeseries(model_pcm)
    # ts_plm = HA.get_observer_timeseries(model_plm)
    
    # fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # ax[0].plot(ts_pcm['time'], ts_pcm['vsw'], 'k--', label='PCM (1st Order)')
    # ax[0].plot(ts_plm['time'], ts_plm['vsw'], 'r-', label='PLM (2nd Order)')
    # ax[0].set_ylabel('Velocity (km/s)')
    # ax[0].legend()
    # ax[0].set_title('Earth (0 deg) Timeseries Comparison')
    
    # ax[1].plot(ts_pcm['time'], ts_pcm['n'], 'k--') 
    # ax[1].plot(ts_plm['time'], ts_plm['n'], 'r-')
    # ax[1].set_ylabel('Density (cm^-3)')
    
    # ax[2].plot(ts_pcm['time'], ts_pcm['T'], 'k--')
    # ax[2].plot(ts_plm['time'], ts_plm['T'], 'r-')
    # ax[2].set_ylabel('Temperature (K)')
    # ax[2].set_xlabel('Time (days)')
    
    # plt.tight_layout()

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



# <cocdecell> InSitu-HUXt compressible test

if insitu_compressible_tests:
    ftime = datetime.datetime(2019,8,5)
    is_model = Hinsitu.omniHUXt_forecast(ftime, simtime=27.27*u.day, 
                            rmin=21.5*u.solRad, rmax=230*u.solRad, 
                            dt_scale=4,
                            omni_input=None, buffertime=5*u.day,
                            run_2d=False)

    is_model.solve([])
    #HA.plot_earth_timeseries(is_model)

    ts_incomp = HA.get_observer_timeseries(is_model)

    is_model = Hinsitu.omniHUXt_forecast(ftime, simtime=27.27*u.day, 
                            rmin=21.5*u.solRad, rmax=230*u.solRad, 
                            dt_scale=4,
                            omni_input=None, buffertime=5*u.day,
                            run_2d=False, solver='hllc-plm-rk2')

    is_model.solve([])
    fig, axs = HA.plot_earth_timeseries(is_model)
    axs[0].plot(ts_incomp['time'], ts_incomp['vsw'], 'b', label='HUXt')
    axs[0].legend()

    dbox = os.environ.get('DBOX')
    overleaf_dir = os.path.join(dbox, 'Apps', 'Overleaf', 'SHUXt')
    pdf_path = os.path.join(overleaf_dir, 'SHUXt_1AU.pdf')
    plt.savefig(pdf_path, dpi=150)


    #also plot the inputs.
    #omni = Hinsitu.get_omni(ftime-datetime.timedelta(days=27.27),ftime)
    fig, axes = plt.subplots(4, figsize=(14, 10))
    axes[0].plot(is_model.v_boundary_lons, is_model.v_boundary.value, 'r-')
    axes[0].set_ylabel(r'$v$ [km/s]')

    axes[1].plot(is_model.v_boundary_lons, np.sign(is_model.b_boundary), 'r-')
    axes[1].set_ylabel(r'B$_{POL}$')

    axes[2].plot(is_model.v_boundary_lons, is_model.rho_boundary.value, 'r-')
    axes[2].set_ylabel(r'$\rho$ [kg m$^{-3}$]')

    axes[3].plot(is_model.v_boundary_lons, is_model.temp_boundary.value, 'r-')
    axes[3].set_ylabel(r'T [K]')
    axes[3].set_xlabel('Carrington Longitude [rad]')

    pdf_path = os.path.join(overleaf_dir, 'SHUXt_0p1AU.pdf')
    plt.savefig(pdf_path, dpi=150)

    # model = Hinsitu.omniHUXt_reconstruction(ftime, ftime +datetime.timedelta(days=27), 
    #                         rmin=21.5*u.solRad, rmax=250*u.solRad, 
    #                         dt_scale=4, dt=1*u.day,
    #                         run_2d=False,
    #                         solver='hllc',
    #                         rho_source='speed',
    #                         temp_source='speed')
    
    # model.solve([])
    
    # # Plot Earth timeseries
    # HA.plot_earth_timeseries(model)
    
 

# <codecell> End of script

plt.show(block=True)
print("\nPlots displayed. Press Enter to close all plots and exit...")
input()