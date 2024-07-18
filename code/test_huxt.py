import os

import astropy.units as u
from astropy.time import Time
import h5py
import numpy as np

import huxt as H
import huxt_inputs as Hin


def test_analytic_solution():
    """
    Test HUXt against the analytic solution for a stationary and uniform inner boundary condition
    """

    # Form longitudinal boundary conditions - background wind of 350 km/s.
    v_boundary = np.ones(128) * 350 * (u.km / u.s)

    # Setup HUXt to do a 1 day simulation, with model output every 8 timesteps (roughly an hour time step),
    # looking at 0 longitude
    model = H.HUXt(v_boundary=v_boundary, lon_out=0.0 * u.deg, simtime=5 * u.day, dt_scale=8)

    # Solve these conditions, with no ConeCMEs added.
    cme_list = []
    model.solve(cme_list)

    # Compute analytical solution from equation 5 in Owens et al 2020
    const = H.huxt_constants()
    alpha = const['alpha']
    rh = const['r_accel']

    v0 = model.v_boundary[0]
    r = model.r
    r0 = r[0]
    v = v0 * (1 + alpha * (1 - np.exp((r0 - r) / rh)))

    # Compute fractional differnce of model solution with analytical 
    dv = np.abs((model.v_grid.squeeze().value - v.value) / v.value)

    # These differences should be less than 1e-3 in an absolute sense.
    assert np.allclose(dv, np.zeros(dv.shape), atol=1e-3)

    return


def test_time_dependent():
    """
    Test HUXt against a reference HUXt solution for a structured solar wind inner boundary with a ConeCME addded.
    Checks consistency of the HUXt flow field, as well as the CME tracer particle coordinates, and CME arrival 
    time calculation.
    """

    v_boundary = np.ones(128) * 400 * (u.km / u.s)
    v_boundary[30:50] = 600 * (u.km / u.s)
    v_boundary[95:125] = 700 * (u.km / u.s)

    #  Add a CME
    cme = H.ConeCME(t_launch=0.5 * u.day, longitude=0.0 * u.deg, width=30 * u.deg, v=1000 * (u.km / u.s),
                    thickness=5 * u.solRad)
    cme_list = [cme]

    #  Setup HUXt to do a 5-day simulation, with model output every 4 timesteps (roughly half and hour time step)
    model_test = H.HUXt(v_boundary=v_boundary, cr_num=2080, cr_lon_init=180 * u.deg, simtime=5 * u.day, dt_scale=4)

    model_test.solve(cme_list)
    cme_test = model_test.cmes[0]

    # Load the reference model output
    dirs = H._setup_dirs_()
    test_case_path = os.path.join(dirs['test_data'], 'HUXt_CR2080_time_dependent_test_case.hdf5')
    model_ref, cme_list_ref = H.load_HUXt_run(test_case_path)
    cme_ref = model_ref.cmes[0]

    # Now compare the test and reference model outputs and ConeCMEs

    # Solar wind solution match
    assert np.allclose(model_ref.v_grid, model_test.v_grid)

    # CME tracking particles match
    test_cme_coords = []
    for (kt, vt), (kr, vr) in zip(cme_test.coords.items(), cme_ref.coords.items()):
        test_cme_coords.append(np.allclose(vt['time'].jd, vr['time'].jd))
        test_cme_coords.append(np.allclose(vt['model_time'].value, vr['model_time'].value))
        test_cme_coords.append(np.allclose(vt['lon'].value, vr['lon'].value))
        test_cme_coords.append(np.allclose(vt['lat'].value, vr['lat'].value))
        test_cme_coords.append(np.allclose(vt['r'].value, vr['r'].value))

    # CME test particle coordinates match
    assert np.all(test_cme_coords)

    # Check CME arrival time calculation

    # Get arrival stats of test CME
    arrival_stats_test = cme_test.compute_arrival_at_body('Earth')

    # Load in reference CME arrival stats
    dirs = H._setup_dirs_()
    test_cme_path = os.path.join(dirs['test_data'], 'cme_arrival_calc_test.hdf5')
    cme_ref = h5py.File(test_cme_path, 'r')

    hit_ref = cme_ref['hit'][()]
    hit_id_ref = cme_ref['hit_id'][()]
    t_arrive_ref = Time(cme_ref['t_arrive'][()].decode("utf-8"), format='isot')
    t_transit_ref = cme_ref['t_transit'][()] * u.Unit(cme_ref['t_transit'].attrs['unit'])
    lon_ref = cme_ref['lon'][()] * u.Unit(cme_ref['lon'].attrs['unit'])
    r_ref = cme_ref['r'][()] * u.Unit(cme_ref['r'].attrs['unit'])
    v_ref = cme_ref['v'][()] * u.Unit(cme_ref['v'].attrs['unit'])

    cme_ref.close()

    # Compare test and ref CME arrival stats
    assert arrival_stats_test['hit'] == hit_ref
    assert arrival_stats_test['hit_id'] == hit_id_ref
    assert np.allclose(arrival_stats_test['t_arrive'].jd, t_arrive_ref.jd)
    assert np.allclose(arrival_stats_test['t_transit'], t_transit_ref)
    assert np.allclose(arrival_stats_test['lon'], lon_ref)
    assert np.allclose(arrival_stats_test['r'], r_ref)
    assert np.allclose(arrival_stats_test['v'], v_ref)

    return


def test_streaklines():
    """
    Test HUXt against a reference HUXt solution for a structured solar wind inner boundary with a ConeCME addded.
    Checks consistency of the HUXt flow field, and the streakline positions
    """

    v_boundary = np.ones(128) * 400 * (u.km / u.s)
    v_boundary[30:50] = 600 * (u.km / u.s)
    v_boundary[95:125] = 700 * (u.km / u.s)

    #  Add a CME
    cme = H.ConeCME(t_launch=0.5 * u.day, longitude=0.0 * u.deg, width=30 * u.deg, v=1000 * (u.km / u.s),
                    thickness=5 * u.solRad)
    cme_list = [cme]

    #  Setup HUXt to do a 5-day simulation, with model output every 4 timesteps (roughly half and hour time step)
    model_test = H.HUXt(v_boundary=v_boundary, cr_num=2080, cr_lon_init=180 * u.deg, simtime=5 * u.day, dt_scale=4)

    # trace a bunch of field lines from a range of evenly spaced Carrington longitudes
    dlon = (20 * u.deg).to(u.rad).value
    lon_grid = np.arange(dlon / 2, 2 * np.pi - dlon / 2 + 0.0001, dlon) * u.rad

    # give the streakline footpoints (in Carr long) to the solve method
    model_test.solve(cme_list, streak_carr=lon_grid)

    # load in the test data
    dirs = H._setup_dirs_()
    test_case_path = os.path.join(dirs['test_data'], 'HUXt_CR2080_streaklines_test_case.hdf5')
    h5f = h5py.File(test_case_path, 'r')
    vgrid = np.array(h5f['v_grid'])
    streakline_particles_r = np.array(h5f['streak_particles_r'])
    h5f.close()

    # check the data agree - first check the vgrid is the same to machine tol
    assert np.allclose(model_test.v_grid.value, vgrid)

    # only compare non-nan values of streakline positions
    mask = np.isfinite(streakline_particles_r)
    assert np.allclose(streakline_particles_r[mask],
                       model_test.streak_particles_r[mask].value)

    return


def test_input_mapping():
    """
    Tests the mapping of v and b to smaller inner boundaries. There are expected
    to be small differences in the 1-AU solutions, therefore acceptabel tolerances
    are used. 
    
    Tests vlong profile mapping and vmap mapping independently
    
    """

    r_orig = 30 * u.solRad
    r_new = 10 * u.solRad

    vtol = 1  # % tolerance for agreement in 1-AU values of V [1]
    btol = 5  # % tolerance for agreement in 1-AU values of bpolarity [5]

    v_orig = np.ones(128) * 400 * (u.km / u.s)
    v_orig[30:50] = 600 * (u.km / u.s)
    v_orig[95:125] = 500 * (u.km / u.s)

    b_orig = np.ones(128)
    b_orig[15:75] = -1

    # long profile mapping
    # =====================

    # map the v_boundary inwards
    v_new, b_new = Hin.map_v_boundary_inwards(v_orig, r_orig, r_new, b_orig=b_orig)

    # run the models out
    model_orig = H.HUXt(v_boundary=v_orig, b_boundary=b_orig,
                        simtime=27 * u.day, dt_scale=4, r_min=r_orig, frame='sidereal')
    model_orig.solve([])

    model_new = H.HUXt(v_boundary=v_new, b_boundary=b_new,
                       simtime=27 * u.day, dt_scale=4, r_min=r_new, frame='sidereal')
    model_new.solve([])

    # compute the fractional difference of 1-AU values
    v_frac = np.nanmean(abs(model_orig.v_grid[:, -1, :] - model_new.v_grid[:, -1, :])
                        / np.nanmean(model_orig.v_grid[:, -1, :]))
    b_frac = np.nanmean(abs(model_orig.b_grid[:, -1, :] - model_new.b_grid[:, -1, :])
                        / np.nanmean(abs(model_orig.b_grid[:, -1, :])))

    assert (v_frac * 100 < vtol)
    assert (b_frac * 100 < btol)

    # check the map mapping
    # ======================

    demo_dir = H._setup_dirs_()['example_inputs']
    wsafilepath = os.path.join(demo_dir, '2022-02-24T22Z.wsa.gong.fits')

    wsa_vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
        = Hin.get_WSA_maps(wsafilepath)

    # map the map inwards
    v_map_new, b_map_new = Hin.map_vmap_inwards(wsa_vr_map, vr_lats, vr_longs,
                                                r_orig, r_new, b_map=br_map)

    # extract a 128-length long profile near the equator
    neq = int(np.floor(len(vr_lats) / 2))
    lon, dlon, nlon = H.longitude_grid()
    v_orig = np.interp(lon, vr_longs, wsa_vr_map[neq, :], period=2 * np.pi)
    v_new = np.interp(lon, vr_longs, v_map_new[neq, :], period=2 * np.pi)
    b_orig = np.interp(lon, vr_longs, br_map[neq, :], period=2 * np.pi)
    b_new = np.interp(lon, vr_longs, b_map_new[neq, :], period=2 * np.pi)

    # run the model out
    model_orig = H.HUXt(v_boundary=v_orig, b_boundary=b_orig,
                        simtime=27 * u.day, dt_scale=4, r_min=r_orig, frame='sidereal')
    model_orig.solve([])

    model_new = H.HUXt(v_boundary=v_new, b_boundary=b_new,
                       simtime=27 * u.day, dt_scale=4, r_min=r_new, frame='sidereal')
    model_new.solve([])

    # compute the fractional difference of 1-AU values
    v_frac = np.nanmean(abs(model_orig.v_grid[:, -1, :] - model_new.v_grid[:, -1, :])
                        / np.nanmean(model_orig.v_grid[:, -1, :]))
    b_frac = np.nanmean(abs(model_orig.b_grid[:, -1, :] - model_new.b_grid[:, -1, :])
                        / np.nanmean(abs(model_orig.b_grid[:, -1, :])))

    assert (v_frac * 100 < vtol)
    assert (b_frac * 100 < btol)

    return
