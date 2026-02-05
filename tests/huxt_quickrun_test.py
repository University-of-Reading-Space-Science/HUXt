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

standard_tests = True
compressible_tests = True
insitu_tests = False
insitu_compressible_tests = False

simtime = 5*u.day

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
                        density_fraction=0.1, temperature_fraction=0.1, profile_type='sinusoidal')
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
    t0 = datetime.datetime.now()
    model_incomp.solve(cme_list, streak_carr=lon_grid)
    dt_pcm = (datetime.datetime.now() - t0).total_seconds()
    print(f"Solve complete in {dt_pcm:.2f}s!")
    #model_incomp.solve([])
    print("Creating plot 1: Earth timeseries (incompressible)...")
    HA.plot_earth_timeseries(model_incomp, plot_omni=False)
    print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")
    ts_upwind = HA.get_observer_timeseries(model_incomp)
    print("Creating plot 2: Spatial plot (incompressible)...")
    HA.plot(model_incomp, time=t_interest)
    print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")

    #HA.animate(model_incomp, tag =  'incompressible_with_CME')


if compressible_tests:
    # ---------------------------------------------------------
    # Test 1: First-order HLLC + PCM
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Starting compressible model (HLLC + PCM 1st Order)...")
    print("="*60)
    model_pcm = H.HUXt(v_boundary=vr_in, b_boundary=br_in,
                            simtime=simtime, dt_scale=4, 
                        solver ='hllc-pcm') # Explicitly request PCM
    print("Model initialized. Starting solve...")
    t0 = datetime.datetime.now()
    model_pcm.solve(cme_list, streak_carr=lon_grid)
    dt_pcm = (datetime.datetime.now() - t0).total_seconds()
    print(f"Solve complete in {dt_pcm:.2f}s!")

    print("Creating plot 3: Earth timeseries (PCM)...")
    HA.plot_earth_timeseries(model_pcm, plot_omni=False)
    plt.title("HLLC + PCM (1st Order)")
    print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")

    print("Creating plot 4: Compressible spatial plot (PCM)...")
    HA.plot_compressible(model_pcm, time=t_interest)
    plt.suptitle("HLLC + PCM (1st Order)")
    print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")

    # ---------------------------------------------------------
    # Test 2: Second-order HLLC + PLM + RK2
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Starting compressible model (HLLC + PLM + RK2 2nd Order)...")
    print("="*60)
    model_plm = H.HUXt(v_boundary=vr_in, b_boundary=br_in,
                            simtime=simtime, dt_scale=4, 
                        solver ='hllc-plm-rk2') # Explicitly request PLM+RK2
    print("Model initialized. Starting solve...")
    t0 = datetime.datetime.now()
    model_plm.solve(cme_list, streak_carr=lon_grid)
    dt_plm = (datetime.datetime.now() - t0).total_seconds()
    print(f"Solve complete in {dt_plm:.2f}s!")
    
    print(f"\nPerformance Comparison: PCM={dt_pcm:.2f}s vs PLM={dt_plm:.2f}s (Ratio: {dt_plm/dt_pcm:.2f}x)")

    print("Creating plot 5: Earth timeseries (PLM)...")
    HA.plot_earth_timeseries(model_plm, plot_omni=False)
    plt.title("HLLC + PLM + RK2 (2nd Order)")
    print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")

    print("Creating plot 6: Compressible spatial plot (PLM)...")
    HA.plot_compressible(model_plm, time=t_interest)
    plt.suptitle("HLLC + PLM + RK2 (2nd Order)")
    print(f"  -> Created figure(s). Current figures: {plt.get_fignums()}")

    # ---------------------------------------------------------
    # Comparison Plot
    # ---------------------------------------------------------
    print("\nCreating plot 7: Direct comparison of Earth Profiles...")
    ts_pcm = HA.get_observer_timeseries(model_pcm)
    ts_plm = HA.get_observer_timeseries(model_plm)
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(ts_pcm['time'], ts_pcm['vsw'], 'k--', label='PCM (1st Order)')
    ax[0].plot(ts_plm['time'], ts_plm['vsw'], 'r-', label='PLM (2nd Order)')
    ax[0].set_ylabel('Velocity (km/s)')
    ax[0].legend()
    ax[0].set_title('Earth (0 deg) Timeseries Comparison')
    
    ax[1].plot(ts_pcm['time'], ts_pcm['n'], 'k--') 
    ax[1].plot(ts_plm['time'], ts_plm['n'], 'r-')
    ax[1].set_ylabel('Density (cm^-3)')
    
    ax[2].plot(ts_pcm['time'], ts_pcm['T'], 'k--')
    ax[2].plot(ts_plm['time'], ts_plm['T'], 'r-')
    ax[2].set_ylabel('Temperature (K)')
    ax[2].set_xlabel('Time (days)')
    
    plt.tight_layout()

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




#<codecell> InSitu-HUXt integration test

if insitu_tests:

    print("="*60)
    print("InSitu-HUXt integration test")
    print("="*60)

    ftime = datetime.datetime(2022,12,1)
    is_model = Hinsitu.omniHUXt_forecast(ftime, simtime=27.27*u.day, 
                            rmin=21.5*u.solRad, rmax=230*u.solRad, 
                            dt_scale=4,
                            omni_input=None, buffertime=5*u.day,
                            run_2d=False)

    is_model.solve([])
    HA.plot_earth_timeseries(is_model)

# <cocdecell> InSitu-HUXt compressible test

if insitu_compressible_tests:
    ftime = datetime.datetime(2022,12,1)
    cr, cr_lon_init = Hin.datetime2huxtinputs(ftime)

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



    model = Hinsitu.omniHUXt_reconstruction(ftime, ftime +datetime.timedelta(days=27), 
                            rmin=21.5*u.solRad, rmax=250*u.solRad, 
                            dt_scale=4, dt=1*u.day,
                            run_2d=False,
                            solver='hllc',
                            rho_source='speed',
                            temp_source='speed')
    
    model.solve([])
    
    # Plot Earth timeseries
    HA.plot_earth_timeseries(model)
    
    # # Function to plot observer timeseries in the same format as Earth plot
    # def plot_observer_timeseries(model, observer_name, observer_label):
    #     """Plot timeseries for a specific observer in same format as Earth plot"""
    #     obs_ts = HA.get_observer_timeseries(model, observer=observer_name)
        
    #     is_compressible = hasattr(model, 'compressible') and model.compressible
        
    #     # Determine number of panels
    #     n_panels = 1  # Velocity
    #     if hasattr(model, 'b_grid'):
    #         n_panels += 1
    #     if is_compressible:
    #         n_panels += 2  # Density and temperature
        
    #     fig, axs = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels))
    #     if n_panels == 1:
    #         axs = np.array([axs])
        
    #     # Velocity panel
    #     panel_idx = 0
    #     axs[panel_idx].plot(obs_ts['time'], obs_ts['vsw'], 'r', label='HUXt')
    #     axs[panel_idx].set_ylim(250, 1000)
    #     axs[panel_idx].set_ylabel('Solar Wind Speed (km/s)')
        
    #     # B polarity panel
    #     if hasattr(model, 'b_grid'):
    #         panel_idx += 1
    #         axs[panel_idx].plot(obs_ts['time'], np.sign(obs_ts['bpol']), 'r.', label='HUXt')
    #         axs[panel_idx].set_ylabel('B polarity')
    #         axs[panel_idx].set_ylim(-1.1, 1.1)
        
    #     # Density panel
    #     if is_compressible and 'n' in obs_ts.columns:
    #         panel_idx += 1
    #         axs[panel_idx].semilogy(obs_ts['time'], obs_ts['n'], 'r-', label='HUXt')
    #         axs[panel_idx].set_ylabel('n (protons/cm³)')
    #         axs[panel_idx].set_ylim(0.1, 1000)
    #         axs[panel_idx].grid(True, alpha=0.3)
        
    #     # Temperature panel
    #     if is_compressible and 'T' in obs_ts.columns:
    #         panel_idx += 1
    #         axs[panel_idx].semilogy(obs_ts['time'], obs_ts['T'], 'r-', label='HUXt')
    #         axs[panel_idx].set_ylabel('T (K)')
    #         axs[panel_idx].set_ylim(1e4, 1e7)
    #         axs[panel_idx].grid(True, alpha=0.3)
        
    #     starttime = obs_ts['time'].iloc[0]
    #     endtime = obs_ts['time'].iloc[-1]
        
    #     for a in axs:
    #         a.set_xlim(starttime, endtime)
    #         a.legend()
        
    #     # Only last panel gets x-label
    #     for i in range(len(axs) - 1):
    #         axs[i].set_xticklabels([])
    #     axs[-1].set_xlabel('Date')
        
    #     fig.suptitle(f'{observer_label} - HUXt Reconstruction', fontsize=14, y=0.995)
    #     plt.tight_layout()
        
    #     return fig, axs, obs_ts
    
    # # Plot STEREO-A timeseries
    # print("\n" + "="*60)
    # print("Plotting STEREO-A timeseries")
    # print("="*60)
    # fig_sta, axs_sta, sta_ts = plot_observer_timeseries(model, 'STA', 'STEREO-A')
    
    # print("\nSTEREO-A Statistics:")
    # print(f"  Velocity: {sta_ts['vsw'].mean():.1f} ± {sta_ts['vsw'].std():.1f} km/s")
    # if 'n' in sta_ts.columns:
    #     print(f"  Density: {sta_ts['n'].mean():.2f} ± {sta_ts['n'].std():.2f} cm^-3")
    # if 'T' in sta_ts.columns:
    #     print(f"  Temperature: {sta_ts['T'].mean():.1e} ± {sta_ts['T'].std():.1e} K")
    
    # # Plot Solar Orbiter timeseries
    # print("\n" + "="*60)
    # print("Plotting Solar Orbiter timeseries")
    # print("="*60)
    # fig_solo, axs_solo, solo_ts = plot_observer_timeseries(model, 'SOLO', 'Solar Orbiter')
    
    # print("\nSolar Orbiter Statistics:")
    # print(f"  Velocity: {solo_ts['vsw'].mean():.1f} ± {solo_ts['vsw'].std():.1f} km/s")
    # if 'n' in solo_ts.columns:
    #     print(f"  Density: {solo_ts['n'].mean():.2f} ± {solo_ts['n'].std():.2f} cm^-3")
    # if 'T' in solo_ts.columns:
    #     print(f"  Temperature: {solo_ts['T'].mean():.1e} ± {solo_ts['T'].std():.1e} K")

# <codecell> End of script

plt.show(block=True)
print("\nPlots displayed. Press Enter to close all plots and exit...")
input()