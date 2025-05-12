import h5py
import os
import shutil
import sys

import numpy as np
import astropy.units as u

# Add HUXt code dir to the path
# this dir:
cwd = os.path.abspath(os.path.dirname(__file__))
# Get root dir, two steps up
huxt_root = os.path.split(os.path.split(cwd)[0])[0]
code_dir = os.path.join(huxt_root, 'code')
# Append code dir to path
sys.path.append(code_dir)

import huxt as H


def make_time_dependent_test_data():
    """
    Generate a HUXt simulation with time dependent inner boundary and a Cone CME. Save these data (and CME arrival
    statistics) into data/test_data to serve as reference cases in the test suite.
    Returns:
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
    tag = "time_dependent_test_case"
    src_file_path = model_test.save(tag)
    dst_file_path = os.path.join(dirs['test_data'], os.path.basename(src_file_path))

    # Copy the file over to the test data
    shutil.copy(src_file_path, dst_file_path)

    # Now also compute CME arrival statistics and save to HDF5
    arrival_stats = cme_test.compute_arrival_at_body('Earth')

    # Load in reference CME arrival stats
    test_cme_stats_path = os.path.join(dirs['test_data'], 'cme_arrival_calc_test.hdf5')
    cme_out = h5py.File(test_cme_stats_path, 'w')

    cme_out.create_dataset('hit', data=arrival_stats['hit'])
    cme_out.create_dataset('hit_id', data=arrival_stats['hit_id'])
    cme_out.create_dataset('t_arrive', data=arrival_stats['t_arrive'].isot)
    for key in ['t_transit', 'lon', 'r', 'v']:
        dset = cme_out.create_dataset(key, data=arrival_stats[key].value)
        dset.attrs['unit'] = arrival_stats[key].unit.to_string()

    cme_out.close()

    return


def make_streakline_test_data():
    """
    Generate a HUXt simulation with time dependent inner boundary and a Cone CME, and a set of streaklines.
    Save these data into data/test_data to serve as reference cases in the test suite.
    Returns:
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

    # Load the reference model output
    dirs = H._setup_dirs_()
    tag = "streaklines_test_case"
    src_file_path = model_test.save(tag)
    dst_file_path = os.path.join(dirs['test_data'], os.path.basename(src_file_path))

    # Copy the file over to the test data
    shutil.copy(src_file_path, dst_file_path)

    return


if __name__ == '__main__':

    # WARNING RUNNING THIS CODE WILL OVERRIDE THE CURRENT TEST DATA.
    # ONLY DO IF ABSOLUTELY CERTAIN TEST DATA NEEDS REGENERATING.
    #make_time_dependent_test_data()
    #make_streakline_test_data()
