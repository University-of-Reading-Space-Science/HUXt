"""
Functions for handling in-situ solar wind observations for SURF.

This module contains functions for downloading, processing, and using
in-situ solar wind measurements (primarily from OMNI) to create
time-dependent boundary conditions for SURF simulations.
"""

import datetime
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
import tqdm
import os
import re
import h5py

from sunpy.net import Fido
from sunpy.net import attrs
from sunpy.timeseries import TimeSeries
from sunpy.coordinates import sun
from dtaidistance import dtw

from urllib.request import urlopen
import json

import joblib
try:
    import onnxruntime as ort
except ImportError:
    ort = None

import surf as surf
import surf_inputs as surfIN


VALID_SOLVERS = ("huxt", "hydro", "hydro-pcm")


def _validate_solver_name(solver):
    """Validate supported solver names for in-situ workflows."""
    if solver not in VALID_SOLVERS:
        valid = ", ".join([f"'{name}'" for name in VALID_SOLVERS])
        raise ValueError(f"Invalid solver '{solver}'. Valid options are: {valid}.")


def _is_compressible_solver(solver):
    return solver in ("hydro", "hydro-pcm")


def get_omni(starttime, endtime):
    """
    A function to grab and process the OMNI COHO1HR data using FIDO
    
    Args:
        starttime : datetime for start of requested interval
        endtime : datetime for start of requested interval

    Returns:
        omni: Dataframe of the OMNI timeseries
    """
    trange = attrs.Time(starttime, endtime)
    dataset = attrs.cdaweb.Dataset('OMNI_COHO1HR_MERGED_MAG_PLASMA')
    result = Fido.search(trange, dataset)
    downloaded_files = Fido.fetch(result)

    # Import the OMNI data
    data = TimeSeries(downloaded_files, concatenate=True)

    omni = data.to_dataframe()
    del data

    # Set invalid data points to NaN
    id_bad = omni['V'] == 9999.0
    omni.loc[id_bad, 'V'] = np.nan
    
    id_bad = omni['N'] == 999.9
    omni.loc[id_bad, 'N'] = np.nan
    
    id_bad = omni['T'] == 9999999.0
    omni.loc[id_bad, 'T'] = np.nan

    # create a BX_GSE field that is expected by some SURF fucntions
    omni['BX_GSE'] = -omni['BR']

    # create a datetime column
    omni['datetime'] = omni.index
    # add a mjd column too
    omni['mjd'] = Time(omni['datetime']).mjd
    # reset the index
    omni = omni.reset_index()

    return omni


def generate_vCarr_from_OMNI(runstart, runend, nlon_grid=None, omni_input=None, dt=1 * u.day, ref_r=215 * u.solRad,
                             corot_type='both', compressible=False):
    """
    A function to download OMNI data and generate V_carr and time_grid for use with set_time_dependent_boundary

    Args:
        runstart: Start time as a datetime
        runend: End time as a datetime
        nlon_grid: Int. If none specified, will be set to the current SURF value (usually 128)
        omni_input: Optional input for supplying the OMNI data. If left as None, it will be downloaded at runtime.
        dt: time resolution, in days is 1*u.day.
        ref_r: radial distance to produce v at, 215*u.solRad by default.
        corot_type: String that determines corot type (both, back, forward)
        compressible: Boolean. If True, also return density and temperature arrays. Default is False.
    Returns:
        Time: Array of times as modified Julian days
        Vcarr: Array of solar wind speeds (km/s) mapped as a function of Carr long and time
        bcarr: Array of Br mapped as a function of Carr long and time
        rhocarr: (if compressible=True) Array of mass density (kg/m³) mapped as a function of Carr long and time
        tcarr: (if compressible=True) Array of temperature (K) mapped as a function of Carr long and time
    """

    # check the coro_type is one of the accepted values
    assert corot_type == 'both' or corot_type == 'back' or corot_type == 'forward'

    # set the default longitude grid, check specified value
    all_lons, dlon, nlon = surf.longitude_grid()
    if nlon_grid is None:
        nlon_grid = nlon
    if not (nlon_grid == nlon):
        print('Warning: vCarr generated for different longitude resolution than current SURF default')

    # if omni data is not supplied, download it
    if omni_input is None:

        # download an additional 28 days either side
        starttime = runstart - datetime.timedelta(days=28)
        endtime = runend + datetime.timedelta(days=28)
        data = get_omni(starttime, endtime)

        # find the period of interest
        mask = ((data['datetime'] > starttime) & (data['datetime'] < endtime))
        omni = data[mask]
        omni = omni.reset_index()
    else:
        # create a copy of the input data, so the original data is unchanged.
        omni = omni_input.copy()

    # interpolate through OMNI V data gaps
    omni_int = omni.interpolate(method='linear', axis=0).ffill().bfill()
    del omni

    omni_int['Time'] = Time(omni_int['datetime'])

    smjd = omni_int['Time'][0].mjd
    fmjd = omni_int['Time'][len(omni_int) - 1].mjd

    # compute the syndoic rotation period
    daysec = 24 * 60 * 60 * u.s
    synodic_period = 27.2753 * daysec  # Solar Synodic rotation period from Earth.
    omega_synodic = 2 * np.pi * u.rad / synodic_period

    # compute carrington longitudes
    cr = np.ones(len(omni_int))
    cr_lon_init = np.ones(len(omni_int)) * u.rad
    for i in range(0, len(omni_int)):
        cr[i], cr_lon_init[i] = surfIN.datetime2surfinputs(omni_int['datetime'][i])

    omni_int['Carr_lon'] = cr_lon_init.value  # remove unit as this confuses pd.DataFrame.copy() needed later
    omni_int['Carr_lon_unwrap'] = np.unwrap(omni_int['Carr_lon'].to_numpy())

    omni_int['mjd'] = [t.mjd for t in omni_int['Time'].array]

    # get the Earth radial distance info.
    dirs = surf._setup_dirs_()
    ephem = h5py.File(dirs['ephemeris'], 'r')
    # convert ephemeric to mjd and interpolate to required times
    all_time = Time(ephem['EARTH']['HEEQ']['time'], format='jd').value - 2400000.5
    omni_int['R'] = np.interp(omni_int['mjd'], all_time, ephem['EARTH']['HEEQ']['radius'][:])  # no unit as L1164

    # map each point back/forward to the reference radial distance
    omni_int['mjd_ref'] = omni_int['mjd']
    omni_int['Carr_lon_ref'] = omni_int['Carr_lon_unwrap']

    for t in range(0, len(omni_int)):
        # time lag to reference radius
        delta_r = ref_r.to(u.km).value - omni_int['R'][t]
        delta_t = delta_r / omni_int['V'][t] / daysec.value
        omni_int.loc[t, 'mjd_ref'] = omni_int.loc[t, 'mjd_ref'] + delta_t
        # change in Carr long of the measurement
        cr_lon_shift = delta_t * daysec.value * 2 * np.pi / synodic_period.value
        omni_int.loc[t, 'Carr_lon_ref'] = omni_int.loc[t, 'Carr_lon_ref'] - cr_lon_shift

    # sort the omni data by Carr_lon_ref for interpolation
    omni_temp = omni_int.copy()
    omni_temp = omni_temp.sort_values(by=['Carr_lon_ref'])

    # now remap these speeds back on to the original time steps
    omni_int['V_ref'] = np.interp(omni_int['Carr_lon_unwrap'], omni_temp['Carr_lon_ref'], omni_temp['V'])
    omni_int['Br_ref'] = np.interp(omni_int['Carr_lon_unwrap'], omni_temp['Carr_lon_ref'], -omni_temp['BX_GSE'])
    
    if compressible:
        omni_int['N_ref'] = np.interp(omni_int['Carr_lon_unwrap'], omni_temp['Carr_lon_ref'], omni_temp['N'])
        omni_int['T_ref'] = np.interp(omni_int['Carr_lon_unwrap'], omni_temp['Carr_lon_ref'], omni_temp['T'])

    # compute the longitudinal and time grids
    dphi_grid = 360 / nlon_grid
    lon_grid = np.arange(dphi_grid / 2, 360.1 - dphi_grid / 2, dphi_grid) * np.pi / 180 * u.rad
    dt = dt.to(u.day).value
    time_grid = np.arange(smjd, fmjd + dt / 2, dt)

    vgrid_carr_recon_back = np.ones((nlon_grid, len(time_grid))) * np.nan
    vgrid_carr_recon_forward = np.ones((nlon_grid, len(time_grid))) * np.nan
    vgrid_carr_recon_both = np.ones((nlon_grid, len(time_grid))) * np.nan

    bgrid_carr_recon_back = np.ones((nlon_grid, len(time_grid))) * np.nan
    bgrid_carr_recon_forward = np.ones((nlon_grid, len(time_grid))) * np.nan
    bgrid_carr_recon_both = np.ones((nlon_grid, len(time_grid))) * np.nan
    
    if compressible:
        rhogrid_carr_recon_back = np.ones((nlon_grid, len(time_grid))) * np.nan
        rhogrid_carr_recon_forward = np.ones((nlon_grid, len(time_grid))) * np.nan
        rhogrid_carr_recon_both = np.ones((nlon_grid, len(time_grid))) * np.nan
        
        tgrid_carr_recon_back = np.ones((nlon_grid, len(time_grid))) * np.nan
        tgrid_carr_recon_forward = np.ones((nlon_grid, len(time_grid))) * np.nan
        tgrid_carr_recon_both = np.ones((nlon_grid, len(time_grid))) * np.nan

    for t in range(0, len(time_grid)):
        # find nearest time and current Carrington longitude
        t_id = np.argmin(np.abs(omni_int['mjd'] - time_grid[t]))
        Elong = omni_int['Carr_lon'][t_id] * u.rad

        # get the Carrington longitude difference from current Earth pos
        dlong_back = surfIN.zerototwopi(lon_grid.value - Elong.value) * u.rad
        dlong_forward = surfIN.zerototwopi(Elong.value - lon_grid.value) * u.rad

        dt_back = (dlong_back / omega_synodic).to(u.day)
        dt_forward = (dlong_forward / omega_synodic).to(u.day)

        vgrid_carr_recon_back[:, t] = np.interp(time_grid[t] - dt_back.value, omni_int['mjd'], omni_int['V_ref'],
                                                left=np.nan, right=np.nan)
        bgrid_carr_recon_back[:, t] = np.interp(time_grid[t] - dt_back.value, omni_int['mjd'], omni_int['Br_ref'],
                                                left=np.nan, right=np.nan)

        vgrid_carr_recon_forward[:, t] = np.interp(time_grid[t] + dt_forward.value, omni_int['mjd'], omni_int['V_ref'],
                                                   left=np.nan, right=np.nan)
        bgrid_carr_recon_forward[:, t] = np.interp(time_grid[t] + dt_forward.value, omni_int['mjd'], omni_int['Br_ref'],
                                                   left=np.nan, right=np.nan)

        numerator = (dt_forward * vgrid_carr_recon_back[:, t] + dt_back * vgrid_carr_recon_forward[:, t])
        denominator = dt_forward + dt_back
        vgrid_carr_recon_both[:, t] = numerator / denominator

        numerator = (dt_forward * bgrid_carr_recon_back[:, t] + dt_back * bgrid_carr_recon_forward[:, t])
        bgrid_carr_recon_both[:, t] = numerator / denominator
        
        if compressible:
            rhogrid_carr_recon_back[:, t] = np.interp(time_grid[t] - dt_back.value, omni_int['mjd'], omni_int['N_ref'],
                                                    left=np.nan, right=np.nan)
            rhogrid_carr_recon_forward[:, t] = np.interp(time_grid[t] + dt_forward.value, omni_int['mjd'], omni_int['N_ref'],
                                                       left=np.nan, right=np.nan)
            numerator = (dt_forward * rhogrid_carr_recon_back[:, t] + dt_back * rhogrid_carr_recon_forward[:, t])
            rhogrid_carr_recon_both[:, t] = numerator / denominator
            
            tgrid_carr_recon_back[:, t] = np.interp(time_grid[t] - dt_back.value, omni_int['mjd'], omni_int['T_ref'],
                                                    left=np.nan, right=np.nan)
            tgrid_carr_recon_forward[:, t] = np.interp(time_grid[t] + dt_forward.value, omni_int['mjd'], omni_int['T_ref'],
                                                       left=np.nan, right=np.nan)
            numerator = (dt_forward * tgrid_carr_recon_back[:, t] + dt_back * tgrid_carr_recon_forward[:, t])
            tgrid_carr_recon_both[:, t] = numerator / denominator

    # cut out the requested time
    mask = ((time_grid >= Time(runstart).mjd) & (time_grid <= Time(runend).mjd))

    if compressible:
        # Convert number density (cm⁻³) to mass density (kg/m³) and add units
        m_p = 1.6726e-27  # proton mass in kg
        
        if corot_type == 'both':
            return time_grid[mask], \
                   vgrid_carr_recon_both[:, mask] * (u.km / u.s), \
                   bgrid_carr_recon_both[:, mask], \
                   rhogrid_carr_recon_both[:, mask] * m_p * 1e6 * (u.kg / u.m**3), \
                   tgrid_carr_recon_both[:, mask] * u.K
        elif corot_type == 'back':
            return time_grid[mask], \
                   vgrid_carr_recon_back[:, mask] * (u.km / u.s), \
                   bgrid_carr_recon_back[:, mask], \
                   rhogrid_carr_recon_back[:, mask] * m_p * 1e6 * (u.kg / u.m**3), \
                   tgrid_carr_recon_back[:, mask] * u.K
        elif corot_type == 'forward':
            return time_grid[mask], \
                   vgrid_carr_recon_forward[:, mask] * (u.km / u.s), \
                   bgrid_carr_recon_forward[:, mask], \
                   rhogrid_carr_recon_forward[:, mask] * m_p * 1e6 * (u.kg / u.m**3), \
                   tgrid_carr_recon_forward[:, mask] * u.K
    else:
        if corot_type == 'both':
            return time_grid[mask], \
                   vgrid_carr_recon_both[:, mask] * (u.km / u.s), \
                   bgrid_carr_recon_both[:, mask]
        elif corot_type == 'back':
            return time_grid[mask], \
                   vgrid_carr_recon_back[:, mask] * (u.km / u.s), \
                   bgrid_carr_recon_back[:, mask]
        elif corot_type == 'forward':
            return time_grid[mask], \
                   vgrid_carr_recon_forward[:, mask] * (u.km / u.s), \
                   bgrid_carr_recon_forward[:, mask]


def generate_vCarr_from_OMNI_DTW(runstart, runend, nlon=None, omni_input=None, res='24h', psi_days=7 * u.day,
                                 max_warp_days=3 * u.day, dtw_on='V'):
    """
    A function to download OMNI data and generate V_carr and time_grid for 
    use with set_time_dependent_boundary. Uses dynamic time warping, rather than
    corotation

    Args:
        runstart: Datetime object. Start of the interval
        runend: Datetime object. End of the interval
        nlon: Int. If none specified, will be set to the current SURF value (usually 128)
        omni_input: Optional input of OMNI data. If left as None is downloaded at runtime.
        res: String. Time averaging of OMNI prior to DTW. match to longitude (for nlon = 128, use '5h')
        psi_days: Float, in units of days. DTW parameter, determines window to ignore at the start and end of the fit.
        max_warp_days: Float, in units of days. DTW parameter, determining maximum warp allowed.
        dtw_on: String. Name of the omni dataframe column to be used to determine the DTW paths
    Returns:
        Time: Array of times as modified Julian days
        Vcarr: Array of solar wind speeds mapped as a function of Carr long and time
        bcarr: Array of Br mapped as a function of Carr long and time
    """

    # set the default longitude grid, check specified value
    all_lons_surf, dlon_surf, nlon_surf = surf.longitude_grid()
    if nlon is None:
        nlon = nlon_surf
    if not (nlon == nlon_surf):
        print('Warning: vCarr generated for different longitude resolution than current SURF default')

    # Download and process OMNI if not provided

    # download an additional 33 days previous and after (27 + 5 buffer)
    starttime = runstart - datetime.timedelta(days=28 + psi_days.value)
    endtime = runend + datetime.timedelta(days=28 + psi_days.value)

    if omni_input is None:
        # Download the 1hr OMNI data from CDAweb
        omni = get_omni(starttime, endtime)
    else:
        # do some check on onmi_input?
        if ((omni_input.loc[0, 'datetime'] > starttime) |
                (omni_input.loc[0, 'datetime'] > starttime)):
            print('Warning: supplied OMNI data does not completely cover required interval (allow +/- 28 days)')
        omni = omni_input.copy()

    # extra processing

    # interpolate through the datagaps
    omni[['V', 'BX_GSE']] = omni[['V', 'BX_GSE']].interpolate(method='linear', axis=0).ffill().bfill()
    omni[[dtw_on]] = omni[[dtw_on]].interpolate(method='linear', axis=0).ffill().bfill()

    # get the carrington longitude
    temp = surfIN.datetime2surfinputs(omni['datetime'].to_numpy())
    omni['carr_lon'] = temp[1].value
    # unwrap this.
    omni['clon_unwrap'] = np.unwrap(omni['carr_lon'].to_numpy())

    # interpolate to the required longitude grid 

    # average up to a given res for a clearer plot
    omni_res = omni.resample(res, on='datetime').mean()
    omni_res['datetime'] = Time(omni_res['mjd'], format='mjd').to_datetime(leap_second_strict='silent')
    omni_res.reset_index(drop=True, inplace=True)

    # compute carrington longitude of earth for each point
    temp = surfIN.datetime2surfinputs(omni_res['datetime'].to_numpy())
    omni_res['carr_lon'] = temp[1].value
    # unwrap this.
    omni_res['clon_unwrap'] = np.unwrap(omni_res['carr_lon'].to_numpy())

    clon_min = omni_res['clon_unwrap'].min()

    # now interpolate this time series onto the required Carr long grid
    # ==================================================================
    dlon = 2 * np.pi / nlon
    clon_unwrap_grid = - np.arange(-2 * np.pi - dlon, -clon_min + 2 * np.pi, dlon)

    v_clon = np.interp(-clon_unwrap_grid, -omni_res['clon_unwrap'].to_numpy(), omni_res['V'].to_numpy(),
                       left=np.nan, right=np.nan)
    mjd_clon = np.interp(-clon_unwrap_grid, -omni_res['clon_unwrap'].to_numpy(), omni_res['mjd'].to_numpy(),
                         left=np.nan, right=np.nan)
    bx_clon = np.interp(-clon_unwrap_grid, -omni_res['clon_unwrap'].to_numpy(), omni_res['BX_GSE'].to_numpy(),
                        left=np.nan, right=np.nan)
    dtwon_clon = np.interp(-clon_unwrap_grid, -omni_res['clon_unwrap'].to_numpy(), omni_res[dtw_on].to_numpy(),
                           left=np.nan, right=np.nan)

    del omni_res
    # bung this in a dataframe
    data = {'mjd': mjd_clon, 'V': v_clon, 'BX_GSE': bx_clon, dtw_on: dtwon_clon, 'clon_unwrap': clon_unwrap_grid}

    omni_res = pd.DataFrame(data)
    omni_res['carr_lon'] = np.mod(clon_unwrap_grid, 2 * np.pi)

    # drop any times which are outside the original data, and therefore have nan mjds
    omni_res = omni_res.dropna(subset=['mjd'])
    omni_res.reset_index(drop=True, inplace=True)
    # recompute the datetimes
    omni_res['datetime'] = Time(omni_res['mjd'], format='mjd').datetime

    del omni

    # get the resulting time resolution and convert DTW params from days to steps
    res_days = omni_res.loc[1, 'mjd'] - omni_res.loc[0, 'mjd']
    psi_steps = int(psi_days.value / res_days)
    max_warp_steps = int(max_warp_days.value / res_days)

    # from this longitude-interpolated time series, create a current
    # and lagged series of equal lengths and corresponding to same longitudes
    L = len(omni_res)

    # find the index of the previous time of the final longitude
    min_clon = omni_res.loc[L - 1, 'clon_unwrap']
    t_lagged_end = np.argmin(np.abs(omni_res['clon_unwrap'] - (min_clon + 2 * np.pi)))

    omni_lagged = omni_res.iloc[:t_lagged_end + 1]
    omni_lagged.reset_index(drop=True, inplace=True)

    # find the index of the previous time of initial longitude
    max_clon = omni_res.loc[0, 'clon_unwrap']
    t_unlagged_start = np.argmin(np.abs(omni_res['clon_unwrap'] - (max_clon - 2 * np.pi)))

    omni_unlagged = omni_res.iloc[t_unlagged_start:]
    omni_unlagged.reset_index(drop=True, inplace=True)

    # now do the actual DTW
    # =====================

    dtw2 = omni_unlagged[dtw_on].to_numpy()
    dtw1 = omni_lagged[dtw_on].to_numpy()

    # compute the DTW betweeen the behind and ahead using various parameters
    path_v = dtw.warping_path(dtw1, dtw2, psi_neg=psi_steps,
                              window=max_warp_steps)
    path_v_arr = np.array(path_v)

    # Now convert paths to a speeds on a regular grid
    def find_y(x1, y1, x2, y2, x):
        """
        Simple gradient calcualtion to compute y from straight line fit to x
        Args:
            x1: Lower x limit
            y1: Lower y limit
            x2: Upper x limit
            y2: Upper y limit
            x: The x value to compute y at.

        Returns:
            y: The computed y value
        """
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        y = m * x + b
        return y

    def gridpaths(paths, v1, v2, startpath, npoints):
        """
        A function to grid data using DTW paths between two time series
        along the startingpath and for npoints evenaly spaced from 0 to 1 inclusive
        Args:
            paths:
            v1:
            v2:
            startpath:
            npoints:

        Returns:
            vgrid_t:
        """

        t = startpath
        y_list = []
        v_list = []

        # put in the starting values
        v_list.append(v1[t])
        y_list.append(0)

        # find all the paths that start at this time
        mask = (paths[:, 0] == t)

        # check if any of these paths are a straight connection
        if any(paths[mask, 1] == t):
            v_list.append(v2[t])
            y_list.append(1)
        else:
            # find all the paths that cross this time
            mask = (paths[:, 0] < t) & (paths[:, 1] > t) | (paths[:, 0] > t) & (paths[:, 1] < t)
            crossing_paths = np.flipud(paths[mask, :])
            # find the y value at which each path crosses this time and the speed
            for path in crossing_paths:
                y = find_y(path[0], 0, path[1], 1, t)
                y_list.append(y)

                # the fractional distance along the path is also y?
                v_at_y = v1[path[0]] * (1 - y) + v2[path[1]] * y
                v_list.append(v_at_y)

            # make sure y is ascending
            zipped_lists = zip(y_list, v_list)

            # Sort the zipped list based on the first list (y_list)
            sorted_zipped_lists = sorted(zipped_lists, key=lambda x: x[0])

            # Unzip the sorted list back into two separate lists
            y_list_sorted, v_list_sorted = zip(*sorted_zipped_lists)

            # Convert the zipped objects back to lists (if needed)
            y_list = list(y_list_sorted)
            v_list = list(v_list_sorted)

            # put the last point in
            v_list.append(v2[t])
            y_list.append(1)

        # now grid this data
        yvals = np.arange(0, 1 + 0.000001, 1 / (npoints - 1))
        vgrid_t = np.ones(npoints) * np.nan
        for n in range(0, len(y_list) - 1):
            mask = (yvals >= y_list[n]) & (yvals < y_list[n + 1])
            vgrid_t[mask] = v_list[n]
        vgrid_t[-1] = v_list[-1]

        return vgrid_t

    # set up the grid info
    clon_grid = np.arange(dlon / 2, 2 * np.pi, dlon)
    t_grid = omni_unlagged['mjd'].to_numpy()
    # and the full grid
    nt = len(t_grid)
    t_grid_full = omni_res['mjd'].to_numpy()
    nt_full = len(t_grid_full)

    paths = path_v_arr

    # now map all this onto the Carrlon-t grid
    # =========================================
    vcarr_grid = np.ones((nlon, nt_full)) * np.nan
    bcarr_grid = np.ones((nlon, nt_full)) * np.nan

    v2 = omni_unlagged['V'].to_numpy()
    v1 = omni_lagged['V'].to_numpy()
    b2 = omni_unlagged['BX_GSE'].to_numpy()
    b1 = omni_lagged['BX_GSE'].to_numpy()

    for t in range(0, nt):
        # find the carrington longitude index this equates to
        lon_id = np.argmin(np.abs(clon_grid - omni_unlagged.loc[t, 'carr_lon']))

        # find the time ranges that are covered by the current longitude
        tmax = omni_unlagged.loc[t, 'mjd']
        tmin = omni_lagged.loc[t, 'mjd']
        mask_t = ((t_grid_full >= tmin) & (t_grid_full < tmax))

        ntimes = np.nansum(mask_t)

        # as a test, just linearly intprolate between the two speeds at this lon
        vgrid_t = gridpaths(paths, v1, v2, t, ntimes)
        bgrid_t = gridpaths(paths, b1, b2, t, ntimes)

        # find where to put this in the full sequence
        mask_t = ((t_grid_full >= tmin) & (t_grid_full < tmax))
        vcarr_grid[lon_id, mask_t] = vgrid_t
        bcarr_grid[lon_id, mask_t] = bgrid_t

    # trim to the required interval
    mask = ((t_grid_full >= Time(runstart).mjd) &
            (t_grid_full <= Time(runend).mjd))

    time_trim = t_grid_full[mask]
    vcarr_grid_trim = vcarr_grid[:, mask]
    bcarr_grid_trim = bcarr_grid[:, mask]

    return time_trim, clon_grid, vcarr_grid_trim, -bcarr_grid_trim


def remove_ICMEs(data_df, icmes, interpolate=True, icme_buffer=0.1 * u.day, interp_buffer=1 * u.day,
                 params=['V', 'BX_GSE'], fill_vals=None):
    """
    A function to remove ICMEs from a given time series

    Args:
        data_df: Pandas dataframe of time series with 'mjd' and reset index, such as provided by get_omni
        icmes: list
        interpolate: boolean. Whether to interpolate through ICMEs NaNs. The default is True.
        icme_buffer: Astropy Quantity with units of day. How much additional data to remove about the ICME boundaries.
        interp_buffer: Astropy Quantity, with units of day. How much of an average to take up and downstream.
        params: list of strings. Which parameters to remove. The default is ['V', 'BX_GSE'].
        fill_vals: list of floats, possibly with units. The fill values to use for interpolation if the upstream or
                   downstream data are all nans.
    Returns:
        data: pd.dataframe with ICMEs removed from required params
    """

    # create a copy of the dataframe, rather than alter the original
    data = data_df.copy()

    # convert the ICME shock and end times to MJD
    icmes['shock_mjd'] = Time(icmes['Shock_time'].to_numpy()).mjd
    icmes['end_mjd'] = Time(icmes['ICME_end'].to_numpy()).mjd

    # go throught the ICME list and remove/interpolate through any that are in the OMNI data

    icme_buffer_d = icme_buffer.to(u.day).value
    interp_buffer_d = interp_buffer.to(u.day).value

    # first remove all ICMEs and add NaNs to the required parameters
    for i in tqdm.trange(0, len(icmes), desc='Removing ICMEs'):

        icme_start = icmes['shock_mjd'][i] - icme_buffer_d
        icme_stop = icmes['end_mjd'][i] + icme_buffer_d

        mask_icme = ((data['mjd'] >= icme_start) &
                     (data['mjd'] <= icme_stop))

        if any(mask_icme):
            for param in params:
                data.loc[mask_icme, param] = np.nan

    # then interpolate through these gaps
    if interpolate:

        # check the fill vals
        if fill_vals is None:
            fill_vals = []
            for i in range(0, len(params)):
                fill_vals.append(np.nan)
        else:
            assert (len(params) == len(fill_vals))

        # loop through each ICME, determine the up and downstream conditions 
        # and interpolate through
        for i in range(0, len(icmes)):

            icme_start = icmes['shock_mjd'][i] - icme_buffer_d
            icme_stop = icmes['end_mjd'][i] + icme_buffer_d

            mask_icme = ((data['mjd'] >= icme_start) &
                         (data['mjd'] <= icme_stop))

            mask_upstream = ((data['mjd'] >= icmes['shock_mjd'][i] - interp_buffer_d) &
                             (data['mjd'] <= icmes['shock_mjd'][i]))

            mask_downstream = ((data['mjd'] >= icmes['end_mjd'][i]) &
                               (data['mjd'] <= icmes['end_mjd'][i] + interp_buffer_d))

            for param, fill_val in zip(params, fill_vals):
                # compute the up and down stream average values
                if any(mask_upstream):
                    v_up = data.loc[mask_upstream, param].mean()
                else:
                    v_up = fill_val

                if any(mask_downstream):
                    v_down = data.loc[mask_downstream, param].mean()
                else:
                    v_down = fill_val

                # if the average values are nans, use the fill values
                if np.isnan(v_up):
                    v_up = fill_val
                if np.isnan(v_down):
                    v_down = fill_val

                dv = v_down - v_up

                # linearly interpolate between the up and down stream values
                if any(mask_icme):
                    icme_duration = icme_stop - icme_start

                    # time through ICME, from start
                    time_through_icme = data.loc[mask_icme, 'mjd'] - icme_start
                    time_through_icme_frac = time_through_icme / icme_duration

                    # linearly interpolate
                    vseries = (v_up + dv * time_through_icme_frac).astype(np.float32)

                    data.loc[mask_icme, param] = vseries
    return data


def get_DONKI_ICMEs(startdate, enddate, location='Earth', ICME_duration=1.5 * u.day):
    """
    Scrape the DONKI database of interplanetary shocks at Earth or STEREO, to create a pseudo-ICME list in the same
    format as the Cane and Richardson list.
    Args:
        startdate: Datetime of the start of the window
        enddate: Datetime of the end of the window
        location: Earth or STEREO A/B
        ICME_duration: Timespan of the assumed ICME duration. Should have units of days.

    Returns:
        icmes: A dataframe of ICMEs
    """
    # scrape the DONKI database of interplanetary shocks at Earth or STEREO. Create
    # a pseudo-ICME list in the same format as Cane and Richardson

    # construct the url
    startdate_str = startdate.strftime('%Y-%m-%d')
    stopdate_str = enddate.strftime('%Y-%m-%d')
    url_head = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/IPS?startDate="
    url = url_head + startdate_str + '&endDate=' + stopdate_str

    # read teh json file
    response = urlopen(url)

    if response.status == 200:
        data = json.loads(response.read().decode("utf-8"))

        # convert to DataFrame
        df = pd.DataFrame(data)

        # only include ICMEs at given location
        mask = df['location'] == location
        icmes = df[mask]
        icmes = icmes.reset_index()

        # put it in the same format as the Cane&Richardson ICME list
        L = len(icmes)
        for i in range(0, L):
            icmes.loc[i, 'Shock_time'] = datetime.datetime.strptime(icmes.loc[i, 'eventTime'], '%Y-%m-%dT%H:%MZ')

        # add a guess at the ICME end time
        icmes['ICME_end'] = icmes['Shock_time'] + datetime.timedelta(days=ICME_duration.value)
    else:
        print("No repsonse for " + url)
        icmes = None

    return icmes


def ICMElist(filepath=None):
    """
    Read and process the Richardson & Cane ICME list.
    
    Reads the pre-processed CSV file of ICMEs detected at Earth from the 
    Richardson & Cane catalog. The original data is from:
    http://www.srl.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm
    
    Pre-processing required for raw HTML:
        1. Download the webpage as an HTML file
        2. Open in Excel, remove year header rows
        3. Delete last column (S) which is empty
        4. Cut out the data table only (delete header and footer)
        5. Save as CSV
    
    Parameters
    ----------
    filepath : str, optional
        Path to the processed CSV file. If None, uses the default path
        in the Data directory ('Richardson_Cane_Porcessed_ICME_list.csv').
    
    Returns
    -------
    icmes : pandas.DataFrame
        DataFrame containing ICME events with columns:
        - 'Shock_time': datetime of shock arrival
        - 'ICME_start': datetime of ICME start
        - 'ICME_end': datetime of ICME end
        - 'dV': velocity change [km/s]
        - 'V_mean': mean velocity [km/s]
        - 'V_max': maximum velocity [km/s]
        - 'Bmag': magnetic field magnitude [nT]
        - 'MCflag': magnetic cloud flag
        - 'Dst': Dst index [nT]
        - 'V_transit': transit velocity [km/s]
    """
    
    if filepath is None:
        datapath = surf._setup_dirs_()['insitu']
        filepath = os.path.join(datapath,
                                'Richardson_Cane_Porcessed_ICME_list.csv')
    
    icmes = pd.read_csv(filepath, header=None)
    # delete the first row
    icmes.drop(icmes.index[0], inplace=True)
    icmes.index = range(len(icmes))
    
    for rownum in range(0, len(icmes)):
        for colnum in range(0, 3):
            # convert the three date stamps
            datestr = icmes[colnum][rownum]
            year = int(datestr[:4])
            month = int(datestr[5:7])
            day = int(datestr[8:10])
            hour = int(datestr[11:13])
            minute = int(datestr[13:15])
            icmes.at[rownum, colnum] = datetime.datetime(year, month, day, hour, minute, 0)
            
        # tidy up the plasma properties
        for paramno in range(10, 17):
            dv = str(icmes[paramno][rownum])
            
            if dv == '...' or dv == 'dg' or dv == 'nan' or dv == '... P' or dv == '... Q':
                icmes.at[rownum, paramno] = np.nan
            else:
                # remove any remaining non-numeric characters
                dv = re.sub('[^0-9]', '', dv)
                icmes.at[rownum, paramno] = float(dv)
    
    # change the column headings
    icmes = icmes.rename(columns={0: 'Shock_time',
                                  1: 'ICME_start',
                                  2: 'ICME_end',
                                  10: 'dV',
                                  11: 'V_mean',
                                  12: 'V_max',
                                  13: 'Bmag',
                                  14: 'MCflag',
                                  15: 'Dst',
                                  16: 'V_transit'})
    return icmes

def removeICMEs(omni, 
                icme_list='CaneRichardson',
                pre_icme_buffer=0.2,  # days
                post_icme_buffer=1,  # days
                interp_gaps=True):
    """
    Remove ICME periods from OMNI solar wind data.
    
    Identifies ICME events in the input OMNI data using a specified catalog
    and replaces the affected time periods with NaN values. Optionally 
    interpolates through the resulting data gaps.
    
    Parameters
    ----------
    omni : pandas.DataFrame
        OMNI solar wind data with columns 'datetime', 'mjd', 'V', and 'BX_GSE'.
    icme_list : str, optional
        Which ICME catalog to use. Options are:
        - 'CaneRichardson': Richardson & Cane near-Earth ICME list (default)
        - 'DONKI': NASA DONKI ICME database
    pre_icme_buffer : float, optional
        Time buffer before ICME shock arrival to also remove, in days.
        Default is 0.2 days.
    post_icme_buffer : float, optional
        Time buffer after ICME end to also remove, in days.
        Default is 1 day.
    interp_gaps : bool, optional
        If True, interpolate through the data gaps created by ICME removal
        using time-weighted interpolation with forward/backward fill for edges.
        Default is True.
    
    Returns
    -------
    omni_noicmes : pandas.DataFrame
        Copy of input OMNI data with ICME periods removed (set to NaN or 
        interpolated depending on interp_gaps setting).
    
    Notes
    -----
    Only the 'V' (velocity) and 'BX_GSE' (radial magnetic field) columns
    are modified. Other columns remain unchanged.
    """
    # create a copy of the OMNI data for ICME removal
    omni_noicmes = omni.copy()
    
    dl_starttime = omni.loc[0]['datetime'] - datetime.timedelta(days=27)
    dl_endtime = omni.loc[len(omni)-1]['datetime'] + datetime.timedelta(days=27)
    
    # load the ICME list
    if icme_list == 'DONKI':
        icmes = get_DONKI_ICMEs(dl_starttime, dl_endtime)
    elif icme_list == 'CaneRichardson':
        icmes = ICMElist()
    
    params = ['V', 'BX_GSE']
    # first remove all ICMEs and add NaNs to the required parameters
    icmes['shock_mjd'] = Time(icmes['Shock_time'].to_numpy()).mjd
    icmes['end_mjd'] = Time(icmes['ICME_end'].to_numpy()).mjd
        
    for i in range(0, len(icmes)):
        icme_start = icmes['shock_mjd'][i] - pre_icme_buffer
        icme_stop = icmes['end_mjd'][i] + post_icme_buffer 
    
        mask_icme = ((omni_noicmes['mjd'] >= icme_start) &
                     (omni_noicmes['mjd'] <= icme_stop))
    
        if any(mask_icme):
            print('removing ICME #' + str(i))
            for param in params:
                omni_noicmes.loc[mask_icme, param] = np.nan
    
    if interp_gaps:
        # now interp through all datagaps
        omni_noicmes = omni_noicmes.set_index('datetime')
        omni_noicmes[['V', 'BX_GSE']] = omni_noicmes[['V', 'BX_GSE']].interpolate(method='time').ffill().bfill()
        omni_noicmes = omni_noicmes.reset_index()

    return omni_noicmes


def correct_inner_vlon_cnn_onnx(v_inner_array,
                                data_dir=None):
    """
    Corrects solar wind speed as a function of longitude using a 1D CNN model
    trained to account for stream interactions during backmapping from 1 AU 
    to 0.1 AU. Uses ONNX, rather than pytorch.

    Parameters
    ----------
    v_inner_array : np.ndarray
        Array of shape (128, N) [speed vs. longitude & samples]
    data_dir : str, optional
        Directory containing saved scalers and ONNX model. If None, uses
        the default data directory in the SURF package.

    Returns
    -------
    Y_pred : np.ndarray
        Array of shape (128, N), CNN-corrected speed
    """
    
    if data_dir is None:
        data_dir = surf._setup_dirs_()['insitu']

    # Load scalers
    y_scaler = joblib.load(os.path.join(data_dir, 'y_scaler_torch.save'))
    x_scaler = joblib.load(os.path.join(data_dir, 'x_scaler_torch.save'))

    # Transpose input to shape (N, 128) so each row is a sample
    vcarr_scaled = x_scaler.transform(v_inner_array.T)  # (N, 128)

    # Reshape to ONNX expected input: (batch_size, channels=1, length=128)
    X_input = vcarr_scaled[:, np.newaxis, :].astype(np.float32)  # (N, 1, 128)

    # Load ONNX model
    onnx_path = os.path.join(data_dir, 'CNN_model.onnx')
    ort_session = ort.InferenceSession(onnx_path)

    # Run inference
    input_name = ort_session.get_inputs()[0].name
    output = ort_session.run(None, {input_name: X_input})
    Y_pred_scaled = output[0]  # (N, 1, 128)

    # Postprocess: squeeze to (N, 128)
    Y_pred_scaled = Y_pred_scaled.squeeze(1)

    # Inverse transform
    Y_pred = y_scaler.inverse_transform(Y_pred_scaled)  # (N, 128)

    # Transpose back to (128, N) to match input shape
    return Y_pred.T


def omniSURF_forecast(ftime, simtime=27.27*u.day, 
                        rmin=21.5*u.solRad, rmax=230*u.solRad, 
                        dt_scale=4,
                        omni_input=None, buffertime=5*u.day,
                        run_2d=False, solver='huxt',
                        rho_source='speed', temp_source='speed'):
    """
    Create a SURF solar wind forecast initialized from in-situ OMNI observations.
    
    Uses the previous solar rotation of OMNI data (mapped back to the inner 
    boundary) to create a Carrington map of solar wind speed, applies a CNN
    correction for stream interaction effects, and initializes a SURF model
    to forecast solar wind conditions at Earth.

    Can be used for reconstructions too. Just set buffer time equal to simtime.
    
    Parameters
    ----------
    ftime : datetime.datetime
        Forecast initialization time. The model uses OMNI data from the
        previous ~27 days to construct the inner boundary condition.
    simtime : astropy.units.Quantity, optional
        Total simulation duration, inc buffer. Default is 27.27 days (one Carrington rotation).
    rmin : astropy.units.Quantity, optional
        Inner boundary radius for the SURF model. Default is 21.5 solar radii.
    rmax : astropy.units.Quantity, optional  
        Outer boundary radius for the SURF model. Default is 230 solar radii.
    dt_scale : int, optional
        Time step scaling factor for SURF. Higher values = faster but less
        accurate. Default is 4.
    omni_input : pandas.DataFrame, optional
        Pre-loaded OMNI data with ICMEs already removed. If None, the function
        will download OMNI data and remove ICMEs automatically. Should contain
        columns 'datetime', 'mjd', 'V', 'BX_GSE', and optionally 'N', 'T'.
    buffertime : astropy.units.Quantity, optional
        Buffer time before ftime to start the simulation, allowing transients
        to propagate through the domain. Default is 5 days.
    run_2d : bool, optional
        If False (default), runs a 1D radial simulation at Earth's longitude
        (lon_out=0). If True, runs a full 2D simulation across all longitudes.
    solver : str, optional
        Solver type. Valid options are:
        - 'huxt' (default): first-order HUXt advection solver
        - 'hydro': second-order compressible HLLC+PLM solver
        - 'hydro-pcm': compressible HLLC+PCM solver

    
    Returns
    -------
    model : surf.SURF
        Initialized (but not yet solved) SURF model object. Call model.solve([])
        to run the simulation.
    
    Notes
    -----
    The CNN correction (correct_inner_vlon_cnn_onnx) accounts for stream 
    interaction effects that occur during the backmapping process from 1 AU
    to the inner boundary at ~21.5 solar radii.
    
    Earth's orbital position is obtained from the built-in SURF ephemeris data.
    
    For compressible solvers:
    - 'speed' source uses empirical relations from solar wind observations
    - 'omni' source scales OMNI measurements: density by (r_ref/r)^2, 
      temperature by approximate Parker solution scaling
    
    Examples
    --------
    >>> import datetime
    >>> ftime = datetime.datetime(2022, 5, 1)
    >>> model = omniSURF_forecast(ftime, simtime=27*u.day)
    >>> model.solve([])
    >>> # For compressible solver
    >>> model = omniSURF_forecast(ftime, solver='hydro', 
    ...                           rho_source='speed', temp_source='speed')
    >>> # Extract Earth time series
    >>> import surf.surf_analysis as SA
    >>> ts = SA.get_observer_timeseries(model, observer='Earth')
    """
    _validate_solver_name(solver)
    
    # if no omni data provided, download it and remove ICMEs
    if omni_input is None:
        dl_starttime = ftime - datetime.timedelta(days=28)
        dl_endtime = ftime + datetime.timedelta(days=28)
    
        omni = get_omni(dl_starttime, dl_endtime)
        
        omni_input = removeICMEs(omni)
    
    # cut out the precise bit of the OMNI data that is required
    mask = (omni_input['datetime'] <= ftime) 
    omni_input = omni_input.loc[mask]
    
    
    # add the carrington longitude to the omni data
    def remainder(cr_frac):
        if np.isscalar(cr_frac):
            return int(np.floor(cr_frac))
        else:
            return np.floor(cr_frac).astype(int)
    
    cr_frac = sun.carrington_rotation_number(omni_input['datetime'])
    cr = remainder(cr_frac)
    omni_input['lon_carr'] = 2 * np.pi * (1 - (cr_frac - cr)) 
    
    # create vCarr with the omni time series at 1 AU
    # unwrap the carr long
    unwrapped = np.unwrap(omni_input['lon_carr'], discont=np.pi)
    # find the current value
    idx = np.argmin(np.abs(omni_input['datetime'] - ftime))
    curr_lon = unwrapped[idx] 
    # find the data up to 2 pi previously 
    mask = ((unwrapped < curr_lon + 2*np.pi) & (unwrapped >= curr_lon))
    omni_chunk = omni_input.loc[mask].reset_index(drop=True)
    
    # sort by carrington lon
    omni_lon = omni_chunk.sort_values(by='lon_carr').reset_index(drop=True)
    
    # now map back to the inner boundary
    # Get Earth's radial distance from ephemeris data
    dirs = surf._setup_dirs_()
    ephem = h5py.File(dirs['ephemeris'], 'r')
    # convert ephemeris to mjd and interpolate to required time
    all_time = Time(ephem['EARTH']['HEEQ']['time'], format='jd').value - 2400000.5
    Earth_R_km = np.interp(Time(ftime).mjd, all_time, ephem['EARTH']['HEEQ']['radius'][:]) * u.km
    ephem.close()
    
    # Backmap to the inner boundary with solver-dependent acceleration profile.
    if solver == 'huxt':
        vcarr_rmin_back, bcarr_rmin_back = surfIN.map_v_boundary_inwards(omni_lon['V'].to_numpy()*u.km/u.s, 
                                    Earth_R_km.to(u.solRad), rmin,
                                    b_orig=-omni_lon['BX_GSE'].to_numpy())
    else:
        vcarr_rmin_back, bcarr_rmin_back = surfIN.map_v_boundary_inwards(omni_lon['V'].to_numpy()*u.km/u.s,
                                    Earth_R_km.to(u.solRad), rmin,
                                    b_orig=-omni_lon['BX_GSE'].to_numpy(), acc_profile='huxt', gamma=1.5)
    
    
    # interp to typical SURF resolution
    dphi = 2*np.pi/surf.surf_constants()['nlong']
    longs = np.arange(dphi/2, 2*np.pi, dphi)
    vlon = np.interp(longs, omni_lon['lon_carr'], vcarr_rmin_back)
    blon = np.interp(longs, omni_lon['lon_carr'], bcarr_rmin_back) if bcarr_rmin_back is not None else None
    
    # apply the CNN to the backmapped data
    vcarr_rmin_back_cnn = correct_inner_vlon_cnn_onnx(vlon.reshape(-1, 1))

    # #teh HUXXt equation has stronger acceleration than Parker nozzle. so increase speeds slightly 
    # #to compensate
    if _is_compressible_solver(solver):
        vcarr_rmin_back_cnn = vcarr_rmin_back_cnn * 1.0
        #smooth the series, periodic at the edges
        vcarr_rmin_back_cnn = np.convolve(vcarr_rmin_back_cnn.flatten(), np.ones(5)/5, mode='same')
        
        #ensure no speeds below 250
        vcarr_rmin_back_cnn[vcarr_rmin_back_cnn <250] = 250
    
    # set up the model run to start 5 days before the forecast time, to allow for CMEs
    cr, cr_lon_init = surfIN.datetime2surfinputs(ftime - datetime.timedelta(days=buffertime.value))
    
    # Get Earth latitude - using get_earth_lat if available, otherwise default to 0
    Elat = surfIN.get_earth_lat(ftime)

    
    if run_2d:
        model = surf.SURF(v_boundary=vcarr_rmin_back_cnn.flatten() * u.km/u.s, 
                      b_boundary=blon, 
                      cr_num=cr, cr_lon_init=cr_lon_init,
                      simtime=simtime, r_min=rmin, r_max=rmax, 
                      dt_scale=dt_scale, latitude=Elat, frame='synodic', 
                      track_cmes=False, solver=solver)
    else:
        model = surf.SURF(v_boundary=vcarr_rmin_back_cnn.flatten() * u.km/u.s, 
                      b_boundary=blon, 
                      cr_num=cr, cr_lon_init=cr_lon_init,
                      simtime=simtime, r_min=rmin, r_max=rmax, 
                      dt_scale=dt_scale, latitude=Elat, frame='synodic', 
                      track_cmes=False, lon_out=0*u.rad, solver=solver)
    return model


def omniSURF_reconstruction(start_time, end_time, 
                            rmin=21.5*u.solRad, rmax=230*u.solRad, 
                            dt_scale=4, dt=1*u.day,
                            omni_input=None,
                            run_2d=False,
                            solver='huxt',
                            rho_source='speed',
                            temp_source='speed'):
    """
    Create a SURF solar wind reconstruction using OMNI observations over a time interval.
    
    Uses OMNI data mapped to Carrington coordinates via corotation (with 'both' forward
    and backward mapping), backmaps to the inner boundary, applies CNN correction for
    stream interaction effects, and creates a time-dependent SURF simulation.
    
    Parameters
    ----------
    start_time : datetime.datetime
        Start time of the reconstruction interval.
    end_time : datetime.datetime
        End time of the reconstruction interval.
    rmin : astropy.units.Quantity, optional
        Inner boundary radius for the SURF model. Default is 21.5 solar radii.
    rmax : astropy.units.Quantity, optional  
        Outer boundary radius for the SURF model. Default is 230 solar radii.
    dt_scale : int, optional
        Time step scaling factor for SURF. Higher values = faster but less
        accurate. Default is 4.
    dt : astropy.units.Quantity, optional
        Time resolution for the Carrington map, in days. Default is 1 day.
    omni_input : pandas.DataFrame, optional
        Pre-loaded OMNI data. If None, the function will download OMNI data
        and remove ICMEs (Cane & Richardson list) automatically. Should contain
        columns 'datetime', 'mjd', 'V', 'BX_GSE', 'N', 'T'.
    run_2d : bool, optional
        If False (default), runs a 1D radial simulation at Earth's longitude.
        If True, runs a full 2D simulation across all longitudes.
    solver : str, optional
        Solver type. Valid options are:
        - 'huxt' (default): first-order HUXt advection solver
        - 'hydro': second-order compressible HLLC+PLM solver
        - 'hydro-pcm': compressible HLLC+PCM solver
    rho_source : str, optional
        Source for density when solver is 'hydro' or 'hydro-pcm'. Options:
        - 'speed': Derive from speed using SURF input functions (default)
        - 'omni': Use OMNI data with 1/r^2 scaling from reference radius to rmin
    temp_source : str, optional
        Source for temperature when solver is 'hydro' or 'hydro-pcm'. Options:
        - 'speed': Derive from speed using SURF input functions (default)
        - 'omni': Use OMNI data with Parker-like radial scaling
    
    Returns
    -------
    model : surf.SURF
        Initialized (but not yet solved) SURF model object with time-dependent
        boundary conditions. Call model.solve([]) to run the simulation.
    
    Notes
    -----
    The function:
    1. Downloads OMNI data from start_time-28 days to end_time+28 days
    2. Removes ICMEs using the Richardson & Cane catalog
    3. Calls generate_vCarr_from_OMNI with corot_type='both'
    4. Backmaps velocity from reference radius (215 Rsun) to rmin
    5. Applies CNN correction to account for stream interactions during backmapping
    6. Creates a SURF model with time-dependent boundary conditions
    
    The CNN correction (correct_inner_vlon_cnn_onnx) accounts for stream 
    interaction effects that occur during the backmapping process from the
    reference radius to the inner boundary.
    
    For compressible solvers:
    - 'speed' source uses empirical relations from solar wind observations
    - 'omni' source scales OMNI measurements: density by (r_ref/r)^2, 
      temperature by approximate Parker solution scaling
    
    Examples
    --------
    >>> import datetime
    >>> start = datetime.datetime(2022, 5, 1)
    >>> end = datetime.datetime(2022, 5, 28)
    >>> model = omniSURF_reconstruction(start, end)
    >>> model.solve([])
    >>> # For compressible solver
    >>> model = omniSURF_reconstruction(start, end, solver='hydro', 
    ...                                 rho_source='omni', temp_source='omni')
    >>> # Extract Earth time series
    >>> import surf.surf_analysis as SA
    >>> ts = SA.get_observer_timeseries(model, observer='Earth')
    """
    _validate_solver_name(solver)
    
    # If no OMNI data provided, download it and remove ICMEs
    if omni_input is None:
        dl_starttime = start_time - datetime.timedelta(days=28)
        dl_endtime = end_time + datetime.timedelta(days=28)
        
        omni = get_omni(dl_starttime, dl_endtime)
        omni_input = removeICMEs(omni, icme_list='CaneRichardson')
    
    # Determine if we need density and temperature from OMNI
    need_compressible = _is_compressible_solver(solver)
    compressible = need_compressible and (rho_source == 'omni' or temp_source == 'omni')
    
    # Generate Carrington map using corotation (both forward and backward)
    if compressible:
        time_grid, vcarr_215, bcarr_215, rhocarr_215, tcarr_215 = generate_vCarr_from_OMNI(
            start_time, end_time, 
            omni_input=omni_input, 
            dt=dt,
            corot_type='both',
            compressible=True
        )
    else:
        time_grid, vcarr_215, bcarr_215 = generate_vCarr_from_OMNI(
            start_time, end_time, 
            omni_input=omni_input, 
            dt=dt,
            corot_type='both',
            compressible=False
        )
    
    # Get reference radius from generate_vCarr_from_OMNI (215 Rsun by default)
    ref_r = 215 * u.solRad
    
    # Backmap from 215 Rsun to rmin for each time step
    # Use different acceleration profile for compressible solvers (same as omniSURF_forecast)
    nlon, nt = vcarr_215.shape
    vcarr_rmin = np.zeros_like(vcarr_215.value)
    bcarr_rmin = np.zeros_like(bcarr_215)
    
    if solver == 'huxt':
        for t in range(nt):
            vcarr_rmin[:, t], bcarr_rmin[:, t] = surfIN.map_v_boundary_inwards(
                vcarr_215[:, t], 
                ref_r, 
                rmin,
                b_orig=bcarr_215[:, t]
            )
    else:
        for t in range(nt):
            vcarr_rmin[:, t], bcarr_rmin[:, t] = surfIN.map_v_boundary_inwards(
                vcarr_215[:, t], 
                ref_r, 
                rmin,
                b_orig=bcarr_215[:, t],
                acc_profile='huxt', gamma=1.5
            )
    
    # Apply CNN correction to backmapped data
    vcarr_rmin_cnn = correct_inner_vlon_cnn_onnx(vcarr_rmin)
    
    # For compressible solvers, apply post-processing (same as omniSURF_forecast)
    if _is_compressible_solver(solver):
        for t in range(vcarr_rmin_cnn.shape[1]):
            col = vcarr_rmin_cnn[:, t]
            # smooth the series, periodic at the edges
            col = np.convolve(col.flatten(), np.ones(5)/5, mode='same')
            # ensure no speeds below 250
            col[col < 250] = 250
            vcarr_rmin_cnn[:, t] = col
    
    # Handle density and temperature for compressible solvers
    rhogrid_carr = np.nan
    tempgrid_carr = np.nan
    
    if need_compressible:
        m_p = 1.67262192e-27  # proton mass in kg
        r_inner = rmin.to(u.solRad).value
        
        # Density
        if rho_source == 'speed':
            # Use empirical velocity-density relation via Parker nozzle lookup table
            # (same approach SURF uses internally). Pass np.nan to let SURF auto-derive.
            rhogrid_carr = np.nan
        elif rho_source == 'omni':
            # Map OMNI density from ref_r (~1 AU) to rmin using Parker nozzle solution
            rhogrid_carr_rmin = np.zeros((nlon, nt))
            for t in range(nt):
                v_col = vcarr_215[:, t]
                rho_col = rhocarr_215[:, t]  # kg/m^3
                # Convert mass density to number density (cm^-3) for Parker mapping
                n_col = (rho_col.value / m_p / 1e6) * u.cm**-3
                # Get temperature at ref_r from empirical relation
                _, T_col = S.get_density_temperature_from_velocity(
                    v_col.to(u.km/u.s).value, ref_r.to(u.solRad).value, gamma=1.5
                )
                # Map all properties from ref_r to rmin using Parker nozzle
                _, n_new, _ = S.map_properties_parker(
                    v_col, ref_r, rmin,
                    n_col, T_col * u.K, gamma=1.5
                )
                # Convert number density back to mass density (kg/m^3)
                rhogrid_carr_rmin[:, t] = (n_new.value * m_p * 1e6)
            rhogrid_carr = rhogrid_carr_rmin * (u.kg / u.m**3)
        else:
            raise ValueError(f"Unknown rho_source: {rho_source}. Use 'speed' or 'omni'.")
        
        # Temperature
        if temp_source == 'speed':
            # Use empirical velocity-temperature relation via Parker nozzle lookup table
            # (same approach SURF uses internally). Pass np.nan to let SURF auto-derive.
            tempgrid_carr = np.nan
        elif temp_source == 'omni':
            # Map OMNI temperature from ref_r (~1 AU) to rmin using Parker nozzle solution
            tempgrid_carr_rmin = np.zeros((nlon, nt))
            for t in range(nt):
                v_col = vcarr_215[:, t]
                T_col = tcarr_215[:, t]  # K
                # Get density at ref_r from empirical relation
                n_col, _ = S.get_density_temperature_from_velocity(
                    v_col.to(u.km/u.s).value, ref_r.to(u.solRad).value, gamma=1.5
                )
                # Map all properties from ref_r to rmin using Parker nozzle
                _, _, T_new = S.map_properties_parker(
                    v_col, ref_r, rmin,
                    n_col * u.cm**-3, T_col, gamma=1.5
                )
                tempgrid_carr_rmin[:, t] = T_new.value
            tempgrid_carr = tempgrid_carr_rmin * u.K
        else:
            raise ValueError(f"Unknown temp_source: {temp_source}. Use 'speed' or 'omni'.")
    
    # Calculate simulation time from start to end
    simtime = (Time(end_time).mjd - Time(start_time).mjd) * u.day
    
    # Get Earth latitude
    Elat = surfIN.get_earth_lat(start_time)
    
    # Create SURF model with time-dependent boundary
    if run_2d:
        model = surfIN.set_time_dependent_boundary(
            vgrid_Carr=vcarr_rmin_cnn * u.km/u.s,
            time_grid=time_grid,
            starttime=start_time,
            simtime=simtime,
            bgrid_Carr=bcarr_rmin,
            rhogrid_Carr=rhogrid_carr,
            tempgrid_Carr=tempgrid_carr,
            r_min=rmin,
            r_max=rmax,
            dt_scale=dt_scale,
            latitude=Elat,
            frame='synodic',
            solver=solver, track_cmes=False
        )
    else:
        model = surfIN.set_time_dependent_boundary(
            vgrid_Carr=vcarr_rmin_cnn * u.km/u.s,
            time_grid=time_grid,
            starttime=start_time,
            simtime=simtime,
            bgrid_Carr=bcarr_rmin,
            rhogrid_Carr=rhogrid_carr,
            tempgrid_Carr=tempgrid_carr,
            r_min=rmin,
            r_max=rmax,
            dt_scale=dt_scale,
            latitude=Elat,
            frame='synodic',
            lon_out=0*u.rad,
            solver=solver, track_cmes=False
        )
    
    return model
