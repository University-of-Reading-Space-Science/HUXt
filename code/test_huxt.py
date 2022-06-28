import os

import astropy.units as u
from astropy.time import Time
import h5py
import numpy as np

import huxt as H


def test_analytic_solution():
    """
    Test HUXt against the analytic solution for a stationary and uniform inner boundary condition
    """

    # Form longitudinal boundary conditions - background wind of 350 km/s.
    v_boundary = np.ones(128) * 350 * (u.km/u.s)

    # Setup HUXt to do a 1 day simulation, with model output every 8 timesteps (roughly an hour time step),
    # looking at 0 longitude
    model = H.HUXt(v_boundary=v_boundary, lon_out=0.0*u.deg, simtime=5*u.day, dt_scale=8)

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
    v = v0 * (1 + alpha*(1 - np.exp((r0 - r)/rh)))
    
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
    
    v_boundary = np.ones(128) * 400 * (u.km/u.s)
    v_boundary[30:50] = 600 * (u.km/u.s)
    v_boundary[95:125] = 700 * (u.km/u.s)

    #  Add a CME
    cme = H.ConeCME(t_launch=0.5*u.day, longitude=0.0*u.deg, width=30*u.deg, v=1000*(u.km/u.s), thickness=5*u.solRad)
    cme_list = [cme]

    #  Setup HUXt to do a 5-day simulation, with model output every 4 timesteps (roughly half and hour time step)
    model_test = H.HUXt(v_boundary=v_boundary, cr_num=2080, cr_lon_init=180*u.deg, simtime=5*u.day, dt_scale=4)

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
    