import datetime
import os
import urllib
from urllib.request import urlopen
import json
import ssl
import copy
import pickle

from appdirs import user_data_dir
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import numpy as np
from pathlib import Path
import h5py
from scipy.io import netcdf_file, readsav
from scipy import interpolate

from sunpy.coordinates import sun

import requests
import pandas as pd
from dtaidistance import dtw


def convert_hdf4_to_hdf5(hdf4_path, hdf5_path):
    """
    Convert HDF4 file to HDF5 format using netCDF4 (with HDF4 support from conda) 
    to read and h5py to write.
    
    Args:
        hdf4_path: Path to input HDF4 file
        hdf5_path: Path to output HDF5 file
    """
    try:
        # Try using netCDF4 with HDF4 support (conda version)
        from netCDF4 import Dataset as NC4Dataset
        
        with NC4Dataset(str(hdf4_path), 'r') as hdf4:
            with h5py.File(str(hdf5_path), 'w') as hdf5:
                # Copy all variables
                for var_name in hdf4.variables:
                    data = hdf4.variables[var_name][:]
                    hdf5.create_dataset(var_name, data=data)
        return True
    except (ImportError, OSError):
        # netCDF4 with HDF4 support not available. Inform the user and return False.
        print("Warning: netCDF4 with HDF4 support not available for conversion.")
        print("Please install via conda: conda install -c conda-forge netcdf4")
        return False


from . import huxt as h


def get_data_dir():
    """Get path to output directory for figures and animations"""
    data_dir = Path(user_data_dir("huxt", "")) / "data" / 'boundary_conditions'
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_MAS_boundary_conditions(cr=np.nan, observatory='', runtype='', runnumber='', masres=''):
    """
    A function to grab the  solar wind speed (Vr) and radial magnetic field (Br) boundary conditions from MHDweb.
    An order of preference for observatories is given in the function.
    Checks first if the data already exists in the HUXt boundary condition folder.

    Args:
        cr: Integer Carrington rotation number
        observatory: String name of preferred observatory (e.g., 'hmi','mdi','solis',
            'gong','mwo','wso','kpo'). Empty if no preference and automatically selected.
        runtype: String name of preferred MAS run type (e.g., 'mas','mast','masp').
            Empty if no preference and automatically selected
        runnumber: String Name of preferred MAS run number (e.g., '0101','0201').
            Empty if no preference and automatically selected
        masres: String, specify the resolution of the MAS model run through 'high' or 'medium'.

    Returns:
    flag: Integer, 1 = successful download. 0 = files exist, -1 = no file found.
    """

    assert not np.isnan(cr)

    # The order of preference for different MAS run results
    overwrite = False
    if not masres:
        masres_order = ['high', 'medium']
    else:
        masres_order = [str(masres)]
        overwrite = True  # If the user wants a specific observatory, overwrite what's already downloaded

    if not observatory:
        observatories_order = ['hmi', 'mdi', 'solis', 'gong', 'kpo', 'mwo', 'wso']
    else:
        observatories_order = [str(observatory)]
        overwrite = True  # If the user wants a specific observatory, overwrite what's already downloaded

    if not runtype:
        runtype_order = ['mast', 'masp', 'mas']
    else:
        runtype_order = [str(runtype)]
        overwrite = True

    if not runnumber:
        runnumber_order = ['0201', '0101']
    else:
        runnumber_order = [str(runnumber)]
        overwrite = True

    # Get the local HUXt boundary condition directory
    boundary_dir = get_data_dir()

    # Example URL: http://www.predsci.com/data/runs/cr2010-medium/mdi_mas_mas_std_0101/helio/br_r0.hdf
    heliomas_url_front = 'http://www.predsci.com/data/runs/cr'
    heliomas_url_end = '_r0.hdf'

    vrfilename = 'HelioMAS_CR' + str(int(cr)) + '_vr' + heliomas_url_end
    brfilename = 'HelioMAS_CR' + str(int(cr)) + '_br' + heliomas_url_end
    vrfilename_h5 = 'HelioMAS_CR' + str(int(cr)) + '_vr_r0.h5'
    brfilename_h5 = 'HelioMAS_CR' + str(int(cr)) + '_br_r0.h5'

    brfilepath = boundary_dir.joinpath(brfilename)
    vrfilepath = boundary_dir.joinpath(vrfilename)
    brfilepath_h5 = boundary_dir.joinpath(brfilename_h5)
    vrfilepath_h5 = boundary_dir.joinpath(vrfilename_h5)
    
    # Check if HDF5 files already exist
    if brfilepath_h5.exists() and vrfilepath_h5.exists() and not overwrite:
        print('HDF5 files already exist for CR' + str(int(cr)))
        return 0
    
    # Check if HDF4 files exist and convert them
    if brfilepath.exists() and vrfilepath.exists() and not overwrite:
        print('Converting existing HDF4 files to HDF5 for CR' + str(int(cr)))
        convert_hdf4_to_hdf5(brfilepath, brfilepath_h5)
        convert_hdf4_to_hdf5(vrfilepath, vrfilepath_h5)
        return 0
    
    # Need to download files
    if brfilepath_h5.exists() is False or vrfilepath_h5.exists() is False or overwrite is True:

        # Search MHDweb for a HelioMAS run, in order of preference
        foundfile = False
        urlbase = None
        for res in masres_order:
            for masob in observatories_order:
                for masrun in runtype_order:
                    for masnum in runnumber_order:
                        urlbase = (heliomas_url_front + str(int(cr)) + '-' +
                                   res + '/' + masob + '_' +
                                   masrun + '_mas_std_' + masnum + '/helio/')
                        url = urlbase + 'br' + heliomas_url_end

                        # See if this br file exists using requests
                        try:
                            resp = requests.head(url, verify=False, timeout=10)
                            if resp.status_code < 400:
                                foundfile = True
                        except requests.RequestException:
                            pass

                        # Exit all the loops - clumsy, but works
                        if foundfile:
                            break
                    if foundfile:
                        break
                if foundfile:
                    break
            if foundfile:
                break

        if foundfile is False:
            print('No data available for given CR and observatory preferences')
            return -1

        # Download the vr and br files
        ssl._create_default_https_context = ssl._create_unverified_context

        print('Downloading from: ', urlbase)
        urllib.request.urlretrieve(urlbase + 'br' + heliomas_url_end, brfilepath)
        urllib.request.urlretrieve(urlbase + 'vr' + heliomas_url_end, vrfilepath)

        # Convert HDF4 files to HDF5
        print('Converting HDF4 files to HDF5...')
        if convert_hdf4_to_hdf5(brfilepath, brfilepath_h5):
            print(f'  Converted {brfilename} to {brfilename_h5}')
        if convert_hdf4_to_hdf5(vrfilepath, vrfilepath_h5):
            print(f'  Converted {vrfilename} to {vrfilename_h5}')

        return 1
    else:
        print('Files already exist for CR' + str(int(cr)))
        return 0


def read_MAS_vr_br(cr):
    """
    A function to read in the MAS boundary conditions for a given CR

    Args:
        cr: Integer Carrington rotation number

    Returns:
        MAS_vr: Solar wind speed at 30rS, numpy array in units of km/s.
        MAS_vr_Xa: Carrington longitude of Vr map, numpy array in units of rad.
        MAS_vr_Xm: Latitude of Vr as angle down from N pole, numpy array in units of rad.
        MAS_br: Radial magnetic field at 30rS, dimensionless numpy array.
        MAS_br_Xa: Carrington longitude of Br map, numpy array in units of rad.
        MAS_br_Xm: Latitude of Br as angle down from N pole, numpy array in units of rad.
    """
    # Get the boundary condition directory
    boundary_dir = get_data_dir()
    # Create the filenames (HDF5 format)
    vrfilename = 'HelioMAS_CR' + str(int(cr)) + '_vr_r0.h5'
    brfilename = 'HelioMAS_CR' + str(int(cr)) + '_br_r0.h5'

    filepath = boundary_dir.joinpath(vrfilename)
    assert filepath.exists(), f"HDF5 file not found: {filepath}. Run get_MAS_boundary_conditions() first."

    with h5py.File(str(filepath), 'r') as file:
        MAS_vr_Xa = file['fakeDim0'][:].copy()
        MAS_vr_Xm = file['fakeDim1'][:].copy()
        MAS_vr = file['Data-Set-2'][:].copy()

    # Convert from model to physicsal units
    MAS_vr = MAS_vr * 481.0 * u.km / u.s
    MAS_vr_Xa = MAS_vr_Xa * u.rad
    MAS_vr_Xm = MAS_vr_Xm * u.rad

    filepath = boundary_dir.joinpath(brfilename)
    assert filepath.exists(), f"HDF5 file not found: {filepath}. Run get_MAS_boundary_conditions() first."
    
    with h5py.File(str(filepath), 'r') as file:
        MAS_br_Xa = file['fakeDim0'][:].copy()
        MAS_br_Xm = file['fakeDim1'][:].copy()
        MAS_br = file['Data-Set-2'][:].copy()

    MAS_br_Xa = MAS_br_Xa * u.rad
    MAS_br_Xm = MAS_br_Xm * u.rad

    return MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm


def get_MAS_long_profile(cr, lat=0.0 * u.deg):
    """
    Function to download, read and process MAS output to provide a longitude profile at a specified latitude of the
    solar wind speed for use as boundary conditions in HUXt.

    Args:
        cr: Integer Carrington rotation number
        lat: Latitude at which to extract the longitudinal profile, measure up from the equator. Float with units of deg

    Returns:
        vr_in: Solar wind speed as a function of Carrington longitude at solar equator.
               Interpolated to HUXt longitudinal resolution. np.array (NDIM = 1) in units of km/s
    """
    assert (cr > 0 and not np.isnan(cr))
    assert (lat >= -90.0 * u.deg)
    assert (lat <= 90.0 * u.deg)

    # Convert angle from equator to angle down from N pole
    ang_from_N_pole = np.pi / 2 - (lat.to(u.rad)).value

    # Check the data exist, if not, download them
    flag = get_MAS_boundary_conditions(cr)
    assert (flag > -1)

    # Read the HelioMAS data
    MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = read_MAS_vr_br(cr)

    # Extract the value at the given latitude
    vr = np.ones(len(MAS_vr_Xa))
    for i in range(0, len(MAS_vr_Xa)):
        vr[i] = np.interp(ang_from_N_pole, MAS_vr_Xm.value, MAS_vr[i][:].value)

    return vr * u.km / u.s


def get_MAS_br_long_profile(cr, lat=0.0 * u.deg):
    """
    Function to download, read and process MAS output to provide a longitude profile at a specified latitude of the Br
    for use as boundary conditions in HUXt.

    Args:
        cr: Integer Carrington rotation number
        lat: Latitude at which to extract the longitudinal profile, measure up from the equator. Float with units of deg

    Returns:
        br_in: Br as a function of Carrington longitude at solar equator.
               Interpolated to HUXt longitudinal resolution. np.array (NDIM = 1)
    """
    assert ((not np.isnan(cr)) and cr > 0)
    assert (lat >= -90.0 * u.deg)
    assert (lat <= 90.0 * u.deg)

    # Convert angle from equator to angle down from N pole
    ang_from_N_pole = np.pi / 2 - (lat.to(u.rad)).value

    # Check the data exist, if not, download them
    flag = get_MAS_boundary_conditions(cr)
    assert (flag > -1)

    # Read the HelioMAS data
    MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = read_MAS_vr_br(cr)

    # Extract the value at the given latitude
    br = np.ones(len(MAS_br_Xa))
    for i in range(0, len(MAS_br_Xa)):
        br[i] = np.interp(ang_from_N_pole, MAS_br_Xm.value, MAS_br[i][:])

    return br


def get_MAS_vr_map(cr):
    """
    A function to download, read and process MAS output to provide HUXt boundary conditions as lat-long maps, along with
     angle from the equator for the maps.
    Maps returned in native resolution, not HUXt resolution.

    Args:
        cr: Integer, Carrington rotation number

    Returns:
        vr_map: Solar wind speed as a Carrington longitude-latitude map. numpy array with units of km/s
        vr_lats: The latitudes for the Vr map, relative to the equator. numpy array with units of radians
        vr_longs: The Carrington longitudes for the Vr map, numpy array with units of radians
    """

    assert ((not np.isnan(cr)) and cr > 0)

    # Check the data exist, if not, download them
    flag = get_MAS_boundary_conditions(cr)
    if flag < 0:
        return -1, -1, -1

    # Read the HelioMAS data
    MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = read_MAS_vr_br(cr)

    vr_map = MAS_vr

    # Convert the lat angles from N-pole to the equator centred
    vr_lats = (np.pi / 2) * u.rad - MAS_vr_Xm

    # Flip lats, so they're increasing in value
    vr_lats = np.flipud(vr_lats)
    vr_map = np.fliplr(vr_map)
    vr_longs = MAS_vr_Xa

    return vr_map.T, vr_longs, vr_lats


def get_MAS_br_map(cr):
    """
    A function to download, read and process MAS output to provide HUXt boundary conditions as lat-long maps,
    along with angle from the equator for the maps.
    Maps returned in native resolution, not HUXt resolution.

    Args:
        cr: Integer, Carrington rotation number

    Returns:
        vr_map: Solar wind speed as a Carrington longitude-latitude map. numpy array with units of km/s
        vr_lats: The latitudes for the Vr map, relative to the equator. numpy array with units of radians
        vr_longs: The Carrington longitudes for the Vr map, numpy array with units of radians
    """

    assert ((not np.isnan(cr)) and cr > 0)

    # Check the data exist, if not, download them
    flag = get_MAS_boundary_conditions(cr)
    if flag < 0:
        return -1, -1, -1

    # Read the HelioMAS data
    MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = read_MAS_vr_br(cr)

    br_map = MAS_br

    # Convert the lat angles from N-pole to the equator centred
    br_lats = (np.pi / 2) * u.rad - MAS_br_Xm

    # Flip lats, so they're increasing in value
    br_lats = np.flipud(br_lats)
    br_map = np.fliplr(br_map)
    br_longs = MAS_br_Xa

    return br_map.T, br_longs, br_lats


@u.quantity_input(v_outer=u.km / u.s)
@u.quantity_input(r_outer=u.solRad)
@u.quantity_input(lon_outer=u.rad)
@u.quantity_input(r_inner=u.solRad)
def map_v_inwards(v_orig, r_orig, lon_orig, r_new):
    """
    Function to map v from r_orig (in rs) to r_inner (in rs) accounting for residual acceleration, but neglecting
    stream interactions. Simply recomputes speed, doesn't longitudinally shift data

    Args:
        v_orig: Solar wind speed at original radial distance. Units of km/s.
        r_orig: Radial distance at original radial distance. Units of km.
        lon_orig: Carrington longitude at original distance. Units of rad
        r_new: Radial distance at new radial distance. Units of km.

    Returns:
        v_new: Solar wind speed mapped from r_orig to r_new. Units of km/s.
        lon_new: Carrington longitude at r_new. Units of rad.
    """

    # Get the acceleration parameters
    constants = h.huxt_constants()
    alpha = constants['alpha']  # Scale parameter for residual SW acceleration
    rH = constants['r_accel'].to(u.kilometer).value  # Spatial scale parameter for residual SW acceleration
    Tsyn = constants['synodic_period'].to(u.s).value
    r_orig = r_orig.to(u.km).value
    r_new = r_new.to(u.km).value
    r_0 = (30 * u.solRad).to(u.km).value

    # Compute the 30 rS speed
    v0 = v_orig.value / (1 + alpha * (1 - np.exp(-(r_orig - r_0) / rH)))

    # comppute new speed
    vnew = v0 * (1 + alpha * (1 - np.exp(-(r_new - r_0) / rH)))

    # Compute the transit time from the new to old inner boundary heights (i.e., integrate equations 3 and 4 wrt to r)
    A = v0 + alpha * v0
    term1 = rH * np.log(A * np.exp(r_orig / rH) - alpha * v0 * np.exp(r_new / rH)) / A
    term2 = rH * np.log(A * np.exp(r_new / rH) - alpha * v0 * np.exp(r_new / rH)) / A
    T_integral = term1 - term2

    # Work out the longitudinal shift
    phi_new = zerototwopi(lon_orig.value + (T_integral / Tsyn).value * 2 * np.pi)

    return vnew * u.km / u.s, phi_new * u.rad


@u.quantity_input(v_orig=u.km / u.s)
@u.quantity_input(r_orig=u.solRad)
@u.quantity_input(r_inner=u.solRad)
def map_v_boundary_inwards(v_orig, r_orig, r_new, b_orig=np.nan):
    """
    Function to map a longitudinal V series from r_outer (in rs) to r_inner (in rs) accounting for residual
    acceleration, but neglecting stream interactions. Produces the required longitude shift and remaps the data
    Series returned on input grid

    Args:
        v_orig: Solar wind speed as function of long at outer radial boundary. Units of km/s.
        r_orig: Radial distance at original radial boundary. Units of km.
        r_new: Radial distance at new radial boundary. Units of km.
        b_orig: b_r to be optionally mapped using the same time/long delay as v

    Returns:
        v_new: Solar wind speed as function of long mapped from r_orig to r_new. Units of km/s.
        b_new: (if b_orig input). B_r as a function of long.
    """

    # Compute the longitude grid from the length of the v_orig input variable
    nv = len(v_orig)
    dlon = 2 * np.pi / nv
    lon = np.arange(dlon / 2, 2 * np.pi - dlon / 2 + dlon / 10, dlon) * u.rad

    # Map each point in to a new speed and longitude
    v0, phis_new = map_v_inwards(v_orig, r_orig, lon, r_new)

    # Interpolate the mapped speeds back onto the regular Carr long grid,
    # making boundaries periodic
    v_new = np.interp(lon, phis_new, v0, period=2 * np.pi)

    if np.isfinite(b_orig).any():
        b_new = np.interp(lon, phis_new, b_orig, period=2 * np.pi)
        return v_new, b_new
    else:
        return v_new


@u.quantity_input(v_map=u.km / u.s)
@u.quantity_input(v_map_lat=u.rad)
@u.quantity_input(v_map_long=u.rad)
@u.quantity_input(r_outer=u.solRad)
@u.quantity_input(r_inner=u.solRad)
def map_vmap_inwards(v_map, v_map_lat, v_map_long, r_orig, r_new, b_map=np.nan):
    """
    Function to map a V Carrington map from r_orig (in rs) to r_new (in rs), accounting for acceleration, but ignoring
    stream interaction. Produces the required longitude shift and remaps the data
    Map returned on input coord system, not HUXT resolution.

    Args:
        v_map: Solar wind speed Carrington map at original radial boundary. np.array with units of km/s.
        v_map_lat: Latitude (from the equator) of v_map positions. np.array with units of radians
        v_map_long: Carrington longitude of v_map positions. np.array with units of radians
        r_orig: Radial distance at original radial boundary. np.array with units of km.
        r_new: Radial distance at new radial boundary. np.array with units of km.
        b_map: b_r to be optionally mapped using the same time/long delay as v. assumed to be on same grid

    Returns:
        v_map_new: Solar wind speed map at r_inner. np.array with units of km/s.
        b_map_new: (if b_orig input). B_r as a function of long.
    """

    # Check the dimensions
    assert (len(v_map_lat) == len(v_map[:, 1]))
    assert (len(v_map_long) == len(v_map[1, :]))

    v_map_new = np.ones((len(v_map_lat), len(v_map_long)))
    b_map_new = np.ones((len(v_map_lat), len(v_map_long)))
    for ilat in range(0, len(v_map_lat)):
        # Map each point in to a new speed and longitude
        v0, phis_new = map_v_inwards(v_map[ilat, :], r_orig, v_map_long, r_new)

        # Interpolate the mapped speeds back onto the regular Carr long grid,
        # making boundaries periodic
        v_map_new[ilat, :] = np.interp(v_map_long.value, phis_new.value, v0.value, period=2 * np.pi)

        # check if b_pol needs mapping
        if np.isfinite(b_map).any():
            # check teh b abd v maps are the same dimensions
            assert (v_map.shape == b_map.shape)
            b_map_new[ilat, :] = np.interp(v_map_long.value, phis_new.value, b_map[ilat, :], period=2 * np.pi)

    if np.isfinite(b_map).any():
        return v_map_new * u.km / u.s, b_map_new
    else:
        return v_map_new * u.km / u.s


def get_PFSS_maps(filepath):
    """
    A function to load, read and process PFSSpy output to provide HUXt boundary conditions as lat-long maps, along with
    angle from the equator for the maps.
    Maps returned in native resolution, not HUXt resolution.
    Maps are not transformed - make sure the PFSS maps are Carrington maps

    Args:
        filepath: String, The filepath for the PFSSpy .nc file

    Returns:
        vr_map: numpy array of solar wind speed as a Carrington longitude-latitude map. In km/s
        vr_lats: numpy array of the latitudes for the Vr map, in radians from the equator
        vr_longs: numpy array of the Carrington longitudes for the Vr map, in radians
        br_map:  numpy array of Br as a Carrington longitude-latitude map. Dimensionless
        br_lats: numpy array of the latitudes for the Br map, in radians from the equator
        br_longs: numpy array of the Carrington longitudes for the Br map, in radians
    """
    filepath = Path(filepath)
    assert filepath.exists()
    with netcdf_file(str(filepath), 'r', mmap=False) as nc:
        br_map = nc.variables['br'][:].copy()
        vr_map = nc.variables['vr'][:].copy() * u.km / u.s
        phi = nc.variables['ph'][:].copy()
        cotheta = nc.variables['cos(th)'][:].copy()

    phi = phi * u.rad
    theta = (np.pi / 2 - np.arccos(cotheta)) * u.rad
    vr_lats = theta[:, 0]
    br_lats = vr_lats
    vr_longs = phi[0, :]
    br_longs = vr_longs

    return vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats


def get_WSA_maps(filepath):
    """
    A function to load, read and process WSA FITS maps from the UK Met Office to provide HUXt boundary conditions as
    lat-long maps, along with angle from the equator for the maps.
    Maps returned in native resolution, not HUXt resolution.
    Maps are transformed to Carrington maps

    Args:
        filepath: String, The filepath for the WSA file

    Returns:
        vr_map: Solar wind speed as a Carrington longitude-latitude map. np.array in units of km/s.
        vr_lats: The latitudes for the Vr map, in radians from the equator. np.array in units of radians.
        vr_longs: The Carrington longitudes for the Vr map. np.array in units of radians.
        br_map: Br as a Carrington longitude-latitude map. Dimensionless np.array.
        br_lats: The latitudes for the Br map, in radians from the equator. np.array in units of radians.
        br_longs: The Carrington longitudes for the Br map, in radians. np.array in units of radians.
        cr: Integer, Carrington rotation number
    """
    filepath = Path(filepath)
    assert filepath.exists()
    hdul = fits.open(filepath)

    keys = hdul[0].header
    assert 'CARROT' in keys
    cr_num = hdul[0].header['CARROT']

    # different versions of WSA data have different keywords?
    if 'GRID' in keys:
        dgrid = hdul[0].header['GRID'] * np.pi / 180
    else:
        assert 'LONSTEP' in keys
        dgrid = hdul[0].header['LONSTEP'] * np.pi / 180

    # The map edge longitude is given by the CARRLONG variable.
    # This is 60 degrees from Central meridian (i.e. Earth Carrington longitude)
    carrlong = zerototwopi((hdul[0].header['CARRLONG']) * np.pi / 180)

    data = hdul[0].data
    br_map_fits = data[0, :, :]
    vr_map_fits = data[1, :, :]

    hdul.close()

    # compute the Carrington map grids
    vr_long_edges = np.arange(0, 2 * np.pi + 0.00001, dgrid)
    vr_long_centres = (vr_long_edges[1:] + vr_long_edges[:-1]) / 2

    vr_lat_edges = np.arange(-np.pi / 2, np.pi / 2 + 0.00001, dgrid)
    vr_lat_centres = (vr_lat_edges[1:] + vr_lat_edges[:-1]) / 2

    br_long_edges = np.arange(0, 2 * np.pi + 0.00001, dgrid)
    br_long_centres = (br_long_edges[1:] + br_long_edges[:-1]) / 2

    br_lat_edges = np.arange(-np.pi / 2, np.pi / 2 + 0.00001, dgrid)
    br_lat_centres = (br_lat_edges[1:] + br_lat_edges[:-1]) / 2

    vr_longs = vr_long_centres * u.rad
    vr_lats = vr_lat_centres * u.rad

    br_longs = br_long_centres * u.rad
    br_lats = br_lat_centres * u.rad

    # rotate the maps so they are in the Carrington frame
    vr_map = np.empty(vr_map_fits.shape)
    br_map = np.empty(br_map_fits.shape)

    for nlat in range(0, len(vr_lat_centres)):
        interp = interpolate.interp1d(zerototwopi(vr_long_centres + carrlong),
                                      vr_map_fits[nlat, :], kind="nearest",
                                      fill_value="extrapolate")
        vr_map[nlat, :] = interp(vr_long_centres)

    for nlat in range(0, len(br_lat_centres)):
        interp = interpolate.interp1d(zerototwopi(br_long_centres + carrlong),
                                      br_map_fits[nlat, :], kind="nearest",
                                      fill_value="extrapolate")
        br_map[nlat, :] = interp(br_long_centres)

    vr_map = vr_map * u.km / u.s

    return vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_num


def get_WSA_long_profile(filepath, lat=0.0 * u.deg):
    """
    Function to read and process WSA output to provide a longitude profile at a specified latitude
    of the solar wind speed for use as boundary conditions in HUXt.

    Args:
        filepath: A complete path to the WSA data file
        lat: Latitude to extract the longitudinal profile at, measure up from the equator. Float with units of deg

    Returns:
        vr_in: Solar wind speed as a function of Carrington longitude at solar equator.
               Interpolated to the default HUXt longitudinal grid. np.array (NDIM = 1) in units of km/s
    """

    filepath = Path(filepath)
    assert (lat >= -90.0 * u.deg)
    assert (lat <= 90.0 * u.deg)
    assert (filepath.is_file())

    vr_map, lon_map, lat_map, br_map, br_lon, br_lat, cr_num = get_WSA_maps(filepath)

    # Extract the value at the given latitude
    vr = np.zeros(lon_map.shape)
    for i in range(lon_map.size):
        vr[i] = np.interp(lat.to(u.rad).value, lat_map.to(u.rad).value, vr_map[:, i].value)

    return vr * u.km / u.s


def get_WSA_br_long_profile(filepath, lat=0.0 * u.deg):
    """
    Function to read and process WSA output to provide a longitude profile at a specified latitude
    of the HMF polarity for use as boundary conditions in HUXt.

    Args:
        filepath: A complete path to the WSA data file
        lat: Latitude to extract the longitudinal profile at, measure up from the equator. Float with units of deg

    Returns:
        vr_in: Solar wind speed as a function of Carrington longitude at solar equator.
               Interpolated to the default HUXt longitudinal grid. np.array (NDIM = 1) in units of km/s
    """

    filepath = Path(filepath)
    assert (lat >= -90.0 * u.deg)
    assert (lat <= 90.0 * u.deg)
    assert (filepath.is_file())

    vr_map, lon_map, lat_map, br_map, br_lon, br_lat, cr_num = get_WSA_maps(filepath)

    # Extract the value at the given latitude
    br = np.zeros(lon_map.shape)
    for i in range(lon_map.size):
        br[i] = np.interp(lat.to(u.rad).value, lat_map.to(u.rad).value, br_map[:, i])

    return br


def get_PFSS_long_profile(filepath, lat=0.0 * u.deg):
    """
    Function to read and process PFSS output to provide a longitude profile at a specified latitude
    of the solar wind speed for use as boundary conditions in HUXt.

    Args:
        filepath: A complete path to the PFSS data file
        lat: Latitude to extract the longitudinal profile at, measure up from the equator. Float with units of deg

    Returns:
        vr_in: Solar wind speed as a function of Carrington longitude at solar equator.
               Interpolated to the default HUXt longitudinal grid. np.array (NDIM = 1) in units of km/s
    """

    filepath = Path(filepath)
    assert (lat >= -90.0 * u.deg)
    assert (lat <= 90.0 * u.deg)
    assert (filepath.is_file())

    vr_map, lon_map, lat_map, br_map, br_lon, br_lat = get_PFSS_maps(filepath)

    # Extract the value at the given latitude
    vr = np.zeros(lon_map.shape)
    for i in range(lon_map.size):
        vr[i] = np.interp(lat.to(u.rad).value, lat_map.to(u.rad).value, vr_map[:, i].value)

    return vr * u.km / u.s


def get_CorTom_vr_map(filepath):
    """
    A function to load, read and process CorTom output to provide HUXt V boundary conditions as lat-long maps.
    Maps returned in native resolution, not HUXt resolution.
    Maps are not transformed - make sure the CorTom maps are Carrington maps

    Args:
        filepath: String, The filepath for the CorTom data. Accepts either the CorTom pickle files or the IDL save .dat
                 files. File must end in either .pkl or .dat.
    Returns:
        vr_map: numpy array of solar wind speed as a Carrington longitude-latitude map. In km/s
        vr_lats: numpy array of the latitudes for the Vr map, in radians from trhe equator
        vr_longs: numpy array of the Carrington longitudes for the Vr map, in radians
        phi: meshgrid of longitudes
        theta: mesh grid of latitudes
    """

    filepath = Path(filepath)

    assert (filepath.is_file())

    if filepath.suffix == '.dat':
        # IDL save file from Aber repository
        cortom_data = readsav(filepath)
        vr_map = copy.copy(cortom_data['velocity'])
        vr_colat = copy.copy(cortom_data['colat_rad'])
        vr_longs = copy.copy(cortom_data['lon_rad'])
        
        # Convert colatitude to latitude and flip so south pole at bottom
        vr_lats = (np.pi / 2 - vr_colat)
        vr_lats = np.flipud(vr_lats)
        vr_map = np.flipud(vr_map)

    elif filepath.suffix == '.pkl':
        # Pickled Cortom output from local or UKMO API
        with open(filepath, "rb") as file:
            data = pickle.load(file)

        vr_map = data['velocity']
        vr_colat = data['colat']
        vr_longs = data['lon']
        vr_map = np.swapaxes(vr_map, 0, 1)
        
        # Convert colatitude to latitude (no flip needed for pkl files)
        vr_lats = (np.pi / 2 - vr_colat)
    else:
        raise ValueError(f"Filename must have extension of either dat or pkl: {filepath}")

    # now rotate onto a 0 to 360 grid
    Nlon = len(vr_longs)
    vr_longs_out = np.linspace(np.pi / Nlon, 2 * np.pi - np.pi / Nlon, num=Nlon)
    vr_map_out = vr_map * np.nan
    for nlat in range(0, len(vr_lats)):
        vr_map_out[nlat, :] = np.interp(vr_longs_out, vr_longs, vr_map[nlat, :], period=2 * np.pi)

    return vr_map_out * u.km / u.s, vr_longs_out * u.rad, vr_lats * u.rad
    

def get_CorTom_long_profile(filepath, lat=0.0 * u.deg):
    """
    Function to read and process CorTom (Coronal Tomography) output to provide a longitude profile at a specified
    latitude of the solar wind speed for use as boundary conditions in HUXt.

    Args:
        filepath: A complete path to the CorTom data file
        lat: Latitude to extract the longitudinal profile at, measure up from the equator. Float with units of deg

    Returns:
        vr_in: Solar wind speed as a function of Carrington longitude at solar equator.
               Interpolated to the default HUXt longitudinal grid. np.array (NDIM = 1) in units of km/s
    """
    filepath = Path(filepath)
    assert (lat >= -90.0 * u.deg)
    assert (lat <= 90.0 * u.deg)
    assert (filepath.is_file())

    vr_map, lon_map, lat_map = get_CorTom_vr_map(filepath)

    # Extract the value at the given latitude
    vr = np.zeros(lon_map.shape)
    for i in range(lon_map.size):
        vr[i] = np.interp(lat.to(u.rad).value, lat_map.value, vr_map[:, i].value)

    return vr * u.km / u.s
    

def getMetOfficeWSAandCone(startdate, enddate, datadir=None):
    """Downloads the most recent WSA output and coneCME files for a given time window from the Met Office system.
    Requires an API key to be set as a system environment variable saves wsa and cone files to datadir, which defaults
    to the current directory. UTC date format is "%Y-%m-%dT%H:%M:%S". Outputs the filepaths to the WSA and cone files.
    
    Args:
        startdate : A DATETIME object representing the start of the download window 
        enddate : A DATETIME object representing the end of the download window,
                    normally the current forecast date
        datadir : Optional argument if a non-default download location is needed

    Returns:
       success :   True if both cone and wsa files were successfullly downloaded
       wsafilepath: filepath for the WSA output
       conefilepath: filepath for the cone CME file
       model_time : time-stamp of the associated enlil run
    """
    if datadir is None:
        boundary_dir = get_data_dir()
    else:
        boundary_dir = Path(datadir)

    version = 'v1'
    api_key = os.getenv("UKMO_API")
    url_base = "https://gateway.api-management.metoffice.cloud/swx_swimmr_s4/1.0"

    startdatestr = startdate.strftime("%Y-%m-%dT%H:%M:%S")
    enddatestr = enddate.strftime("%Y-%m-%dT%H:%M:%S")

    request_url = url_base + "/" + version + "/data/swc-enlil-wsa?from=" + startdatestr + "&to=" + enddatestr
    response = requests.get(request_url, headers={"accept": "application/json", "apikey": api_key})

    success = False
    wsafilepath = ''
    conefilepath = ''
    model_time = ''
    if response.status_code == 200:

        # Convert to json
        js = response.json()
        nfiles = len(js['data'])

        # get the latest file
        i = nfiles - 1
        found_wsa = False
        found_cone = False

        # start with the most recent file and work back in time
        while i > 0:
            model_time = js['data'][i]['model_run_time']
            wsa_file_name = js['data'][i]['gong_file']
            cone_file_name = js['data'][i]['cone_file']

            wsa_file_url = url_base + "/" + version + "/" + wsa_file_name
            cone_file_url = url_base + "/" + version + "/" + cone_file_name

            if not found_wsa:
                response_wsa = requests.get(wsa_file_url, headers={"apikey": api_key})
                if response_wsa.status_code == 200:
                    wsafilepath = boundary_dir.joinpath(wsa_file_name)
                    open(wsafilepath, "wb").write(response_wsa.content)
                    found_wsa = True
            if not found_cone:
                response_cone = requests.get(cone_file_url, headers={"apikey": api_key})
                if response_cone.status_code == 200:
                    conefilepath = boundary_dir.joinpath(cone_file_name)
                    open(conefilepath, "wb").write(response_cone.content)
                    found_cone = True
            i = i - 1
            if found_wsa and found_cone:
                success = True
                break

    return success, wsafilepath, conefilepath, model_time


def datetime2huxtinputs(dt):
    """
    A function to convert a datetime into the relevant Carrington rotation number and longitude
    for initialising a HUXt run.

    Args:
        dt : A DATETIME object representing the time of interest.

    Returns:
        cr : The Carrington rotation number as an Integer
        cr_lon_init : The Carrington longitude of Earth at the given datetime, as a float, with units of u.rad
    """

    def remainder(cr_frac):
        if np.isscalar(cr_frac):
            return int(np.floor(cr_frac))
        else:
            return np.floor(cr_frac).astype(int)

    cr_frac = sun.carrington_rotation_number(dt)
    cr = remainder(cr_frac)
    cr_lon_init = 2 * np.pi * (1 - (cr_frac - cr)) * u.rad

    return cr, cr_lon_init


def import_cone2bc_parameters(filename):
    """
    Convert a cone2bc.in file (for inserting cone cmes into ENLIL) into a dictionary of CME parameters.
    Assumes all cone2bc.in files have the same structure, except for the number of cone cmes.
    Args:
        filename: Path to the cone2bc.in file to convert.
    
    Returns:
         cmes: A dictionary of the cone cme parameters.
    """

    with open(filename, 'r') as file:
        data = file.readlines()

    # Get the number of cmes.
    n_cme = int(data[13].split('=')[1].split(',')[0])

    if n_cme == 0:
        print('Warning: No CMEs in conefile: ' + filename)
        return {}

    # Pull out the rows corresponding to the CME parameters
    cme_sub = data[14:-3].copy()

    # Extract the unique keys describing the CME parameters, excluding CME number.
    keys = []
    for i, d in enumerate(cme_sub):
        k = d.split('=')[0].split('(')[0].strip()

        if k not in keys:
            keys.append(k)

    # Build an empty dictionary to store the parameters of each CME. Set the CME key to be the 
    # number of the CME in the cone2bc.in file (counting from 1 to N).
    cmes = {i + 1: {k: {} for k in keys} for i in range(n_cme)}

    # Loop the CME parameters and bin into the dictionary
    for i, d in enumerate(cme_sub):

        parts = d.strip().split('=')
        param_name = parts[0].split('(')[0]
        cme_id = int(parts[0].split('(')[1].split(')')[0])
        param_val = parts[1].split(',')[0]

        if param_name == 'ldates':
            param_val = param_val.strip("'")
        else:
            if param_val == '.':  # I presume this is shorthand for zero?
                param_val = '0.0'
            param_val = float(param_val)

        cmes[cme_id][param_name] = param_val

    return cmes


def cone_dict_to_cme_list(model, cme_params):
    """
    Function to tranlsate a dictionary of cone parameters into a cme list that can be used with model.solve(cme_list).
    Assumes an initial height of 21.5 rS
    Args:
        model: A HUXt instance.
        cme_params: the cone CME parameter dictionary produced by import_cone2bc_parameters.
    returns:
        cme_list: A list of ConeCME instances.
    """

    cme_list = []
    for cme_id, cme_val in cme_params.items():
        # CME initialisation date
        t_cme = Time(cme_val['ldates'])
        # CME initialisation relative to model initialisation, in days
        dt_cme = (t_cme - model.time_init).jd * u.day

        # Get lon, lat and speed
        lon = cme_val['lon'] * u.deg
        lat = cme_val['lat'] * u.deg
        speed = cme_val['vcld'] * u.km / u.s

        # Get full angular width, cone2bc.in specifies angular half width under rmajor
        wid = 2 * cme_val['rmajor'] * u.deg

        # Set the initial height to be 21.5 rS, the default for WSA
        iheight = 21.5 * u.solRad

        thick = 0 * u.solRad

        cme = h.ConeCME(t_launch=dt_cme, longitude=lon, latitude=lat, width=wid, v=speed, thickness=thick,
                        initial_height=iheight, label=f"CME_{cme_id:02d}")
        cme_list.append(cme)

    # sort the CME list into chronological order
    launch_times = np.ones(len(cme_list))
    for i, cme in enumerate(cme_list):
        launch_times[i] = cme.t_launch.value
    id_sort = np.argsort(launch_times)
    cme_list = [cme_list[i] for i in id_sort]

    return cme_list


def ConeFile_to_ConeCME_list(model, filepath):
    """
    A function to produce a list of ConeCMEs for input to HUXt derived from a cone2bc.in file, as is used with  to input
    Cone CMEs into Enlil. Assumes CME height of 21.5 rS
    Args:
        model: A HUXt instance.
        filepath: The path to the relevant cone2bc.in file.
    returns:
        cme_list: A list of ConeCME instances.
    """
    filepath = Path(filepath)
    assert filepath.is_file()

    cme_params = import_cone2bc_parameters(filepath)
    cme_list = cone_dict_to_cme_list(model, cme_params)

    return cme_list


def ConeFile_to_ConeCME_list_time(filepath, time):
    """
    Simple wrapper for ConeFile_to_ConeCME_list so that dummy model is not needed
    Args:
        filepath: Full filepath to a ConeFile of Cone CME parameters
        time: The UTC time to initialise HUXt with.
    Returns:
        cme_list: A list of ConeCME objects that correspond CMEs in a ConeFile
    """
    filepath = Path(filepath)
    assert filepath.is_file()

    cr, cr_lon_init = datetime2huxtinputs(time)
    dummymodel = h.HUXt(v_boundary=np.ones(128) * 400 * (u.km / u.s), simtime=1 * u.day, cr_num=cr,
                        cr_lon_init=cr_lon_init, lon_out=0.0 * u.deg, r_min=21.5 * u.solRad)

    cme_list = ConeFile_to_ConeCME_list(dummymodel, filepath)
    return cme_list


def consolidate_cme_lists(cmelist_list, t_thresh=0.1 * u.day, lon_thresh=10 * u.deg, lat_thresh=10 * u.deg):
    """
    A function which takes a list of CME lists, as produced by multiple Hin.ConeFile_to_ConeCME_list_time outputs, and
    produces a consolidated list. The list of cme lists should be in order from oldest to newest. Threshold parameters
    can be passed to define what counts as the same CME in multiple lists. Also removes duplicate CMEs within a single
    list, which are sometimes present.

    Args:
        cmelist_list: A list of lists of ConeCME instances.
        t_thresh: The time threshold used to identify overlapping CME launches. An astropy quantity.
        lon_thresh: The longitude threshold used to identify overlapping CME launches. An astropy quantity.
        lat_thresh: The latitude threshold used to identify overlapping CME launches. An astropy quantity.

    Returns:
        cmelist_master: A single consolidated list of ConeCME instances.
    """

    # now go through each CME list and add in any new events
    cmelist_master = [cmelist_list[0][0]]
    for cmelist in cmelist_list:
        for cme in cmelist:
            # get properties of the CME
            this_time = cme.t_launch
            this_long = cme.longitude_huxt
            this_lat = cme.latitude
            same_cme = False

            # compare with properties of CMEs currently in the list
            for idx, existingcme in enumerate(cmelist_master):
                existing_time = existingcme.t_launch
                existing_long = existingcme.longitude_huxt
                existing_lat = existingcme.latitude

                # require a similar time, lat and long to be the same CME
                if ((abs(this_time - existing_time) < t_thresh) &
                        (zerototwopi(this_long - existing_long) < lon_thresh) &
                        (abs(this_lat - existing_lat) < lat_thresh)):
                    same_cme = True

                    # use the values from the newer cone file
                    cmelist_master[idx] = cme

            # if it's a new CME, add it to the list
            if not same_cme:
                cmelist_master.append(cme)

    return cmelist_master


def set_time_dependent_boundary(vgrid_Carr, time_grid, starttime, simtime, r_min=215 * u.solRad, r_max=1290 * u.solRad,
                                dt_scale=50, latitude=0 * u.deg, frame='sidereal', lon_start=0 * u.rad,
                                lon_stop=2 * np.pi * u.rad, lon_out=np.nan, bgrid_Carr=np.nan, 
                                rhogrid_Carr=np.nan, tempgrid_Carr=np.nan, track_cmes=True,
                                accel_limit=True, solver='upwind'):
    """
    A function to compute an explicitly time dependent inner boundary condition for HUXt, rather than due to
    synodic/sidereal rotation of static coronal structure.

    Args:
        vgrid_Carr: input solar wind speed as a function of Carrington longitude and time
        time_grid: time steps (in MJD) of vgrid_Carr
        starttime: The datetime object giving the start of the HUXt run
        simtime: The duration fo the HUXt run (in u.day)
        r_min: Specify the inner boundary radius of HUXt, defaults to 1 AU
        r_max: Specify the outer boundary radius of HUXt, defaults to 6 AU
        dt_scale: The output timestep of HUXt in terms of multiples of the intrinsic timestep.
        latitude: The latitude (from the equator) to run HUXt along
        frame: String, "synodic" or "sidereal", specifying the rotating frame of reference
        lon_out: Longitude for single 1-d run. Frame will be synodic
        lon_start: Longitude of one edge of the longitudinal domain of HUXt
        lon_stop: Longitude of the other edge of the longitudinal domain of HUXt
        bgrid_Carr: input magnetic polarity as a function of Carrington longitude and time
        rhogrid_Carr: input density (kg/m³) as a function of Carrington longitude and time
        tempgrid_Carr: input temperature (K) as a function of Carrington longitude and time
        track_cmes: Bool, whether to track CMEs through the simulation.
        accel_limit: Bool, whether to turn off the acceleration for fluid elements with speeds >650 km/s
    returns:
        model: A HUXt instance initialised with the fully time dependent boundary conditions.
    """
    all_lons, dlon, nlon = h.longitude_grid()
    assert (len(vgrid_Carr[:, 0]) == nlon)

    # see if br boundary conditions are supplied
    do_b = False
    if np.isfinite(bgrid_Carr).any():
        do_b = True
    
    # see if density boundary conditions are supplied
    do_rho = False
    if np.isfinite(rhogrid_Carr).any():
        do_rho = True
    
    # see if temperature boundary conditions are supplied
    do_temp = False
    if np.isfinite(tempgrid_Carr).any():
        do_temp = True

    # work out the start time in terms of cr number and cr_lon_init
    cr, cr_lon_init = datetime2huxtinputs(starttime)

    # set up the dummy model class
    if np.isfinite(lon_out):
        model = h.HUXt(v_boundary=np.ones(nlon) * 400 * u.km / u.s,
                       lon_out=lon_out,
                       latitude=latitude,
                       r_min=r_min, r_max=r_max,
                       simtime=simtime, dt_scale=dt_scale,
                       cr_num=cr, cr_lon_init=cr_lon_init,
                       frame='synodic', track_cmes=track_cmes,
                       accel_limit=accel_limit, solver=solver)
    else:
        model = h.HUXt(v_boundary=np.ones(nlon) * 400 * u.km / u.s,
                       lon_start=lon_start, lon_stop=lon_stop,
                       latitude=latitude,
                       r_min=r_min, r_max=r_max,
                       simtime=simtime, dt_scale=dt_scale,
                       cr_num=cr, cr_lon_init=cr_lon_init,
                       frame=frame, track_cmes=track_cmes,
                       accel_limit=accel_limit, solver=solver)

    # extract the values from the model class
    buffertime = model.buffertime  # standard buffer time seems insufficient
    simtime = model.simtime
    frame = model.frame
    dt = model.dt
    cr_lon_init = model.cr_lon_init

    latitude = model.latitude
    time_init = model.time_init

    constants = h.huxt_constants()
    rotation_period = None
    if frame == 'synodic':
        rotation_period = constants['synodic_period']
    elif frame == 'sidereal':
        rotation_period = constants['sidereal_period']

    # compute the model time step
    buffersteps = np.fix(buffertime.to(u.s) / dt)
    buffertime = buffersteps * dt
    model_time = np.arange(-buffertime.value, (simtime.to('s') + dt).value, dt.value) * dt.unit

    # interpolate the solar wind speed onto the model grid 
    # variables to store the input conditions.
    input_ambient_ts = np.nan * np.ones((model_time.size, nlon)) * (u.km / u.s)
    if do_b:
        input_ambient_ts_b = np.nan * np.ones((model_time.size, nlon))
    if do_rho:
        input_ambient_ts_rho = np.nan * np.ones((model_time.size, nlon)) * (u.kg / u.m**3)
    if do_temp:
        input_ambient_ts_temp = np.nan * np.ones((model_time.size, nlon)) * u.K

    for t in range(0, len(model_time)):

        mjd = time_init.mjd + model_time[t].to(u.day).value

        # find the nearest time to the current model time
        t_input = np.argmin(abs(time_grid - mjd))
        if model_time[t] < 0:  # spin up, use initiation time
            t_input = np.argmin(abs(time_grid - time_init.mjd))

        # shift the longitude to match the initial model time
        dlon_from_start = 2 * np.pi * u.rad * model_time[t] / rotation_period

        lon_shifted = zerototwopi((all_lons - cr_lon_init + dlon_from_start).value)
        # put longitudes in ascending order for np.interp
        id_sort = np.argsort(lon_shifted)
        lon_shifted = lon_shifted[id_sort]

        # take the vlong slice at this value
        v_boundary = vgrid_Carr[:, t_input]
        # Handle units if present
        if hasattr(v_boundary, 'unit'):
            v_boundary = v_boundary.value
        v_b_shifted = v_boundary[id_sort]
        # interpolate back to the original grid
        v_boundary = np.interp(all_lons.value, lon_shifted, v_b_shifted, period=2 * np.pi)
        input_ambient_ts[t, :] = v_boundary * (u.km / u.s)

        if do_b:
            b_boundary = bgrid_Carr[:, t_input]
            b_b_shifted = b_boundary[id_sort]
            # interpolate back to the original grid
            b_boundary = np.interp(all_lons.value, lon_shifted, b_b_shifted, period=2 * np.pi)
            input_ambient_ts_b[t, :] = b_boundary
        
        if do_rho:
            rho_boundary = rhogrid_Carr[:, t_input]
            # Handle units if present
            if hasattr(rho_boundary, 'unit'):
                rho_boundary = rho_boundary.value
            rho_b_shifted = rho_boundary[id_sort]
            # interpolate back to the original grid
            rho_boundary = np.interp(all_lons.value, lon_shifted, rho_b_shifted, period=2 * np.pi)
            input_ambient_ts_rho[t, :] = rho_boundary * (u.kg / u.m**3)
        
        if do_temp:
            temp_boundary = tempgrid_Carr[:, t_input]
            # Handle units if present
            if hasattr(temp_boundary, 'unit'):
                temp_boundary = temp_boundary.value
            temp_b_shifted = temp_boundary[id_sort]
            # interpolate back to the original grid
            temp_boundary = np.interp(all_lons.value, lon_shifted, temp_b_shifted, period=2 * np.pi)
            input_ambient_ts_temp[t, :] = temp_boundary * u.K

    # fill the nan values
    mask = np.isnan(input_ambient_ts)
    input_ambient_ts[mask] = 400 * (u.km / u.s)
    if do_b:
        mask = np.isnan(input_ambient_ts_b)
        input_ambient_ts_b[mask] = 0
    if do_rho:
        mask = np.isnan(input_ambient_ts_rho)
        input_ambient_ts_rho[mask] = 0 * (u.kg / u.m**3)
    if do_temp:
        mask = np.isnan(input_ambient_ts_temp)
        input_ambient_ts_temp[mask] = 0 * u.K

    # Build kwargs for HUXt model instantiation
    huxt_kwargs = {
        'v_boundary': np.ones(128) * 400 * u.km / u.s,
        'simtime': simtime,
        'cr_num': cr,
        'cr_lon_init': cr_lon_init,
        'r_min': r_min,
        'r_max': r_max,
        'dt_scale': dt_scale,
        'latitude': latitude,
        'input_v_ts': input_ambient_ts,
        'input_t_ts': model_time,
        'track_cmes': track_cmes,
        'accel_limit': accel_limit,
        'solver': solver
    }
    
    # Add optional boundary condition time series
    if do_b:
        huxt_kwargs['input_b_ts'] = input_ambient_ts_b
    if do_rho:
        huxt_kwargs['input_rho_ts'] = input_ambient_ts_rho
    if do_temp:
        huxt_kwargs['input_temp_ts'] = input_ambient_ts_temp
    
    # Set frame and longitude parameters
    if np.isfinite(lon_out):  # single longitude
        huxt_kwargs['frame'] = 'synodic'
        huxt_kwargs['lon_out'] = lon_out
    else:  # multiple longitudes
        huxt_kwargs['frame'] = frame
        huxt_kwargs['lon_start'] = lon_start
        huxt_kwargs['lon_stop'] = lon_stop
    
    model = h.HUXt(**huxt_kwargs)

    return model


def zerototwopi(angles):
    """
    Function to constrain angles to the 0 - 2pi domain.
    Args:
        angles: a numpy array of angles.
    Returns:
        angles_out: a numpy array of angles constrained to 0 - 2pi domain.
    """
    # Check if angles has astropy unit.
    if isinstance(angles, u.Quantity):
        angles_out = angles.to(u.rad).value
    else:
        angles_out = angles

    twopi = 2.0 * np.pi
    a = -np.floor_divide(angles_out, twopi)
    angles_out = angles_out + (a * twopi)

    # If it came in with units, restore them
    if isinstance(angles, u.Quantity):
        angles_out = angles_out * u.rad

    return angles_out


def get_DONKI_coneCMEs(startdate, enddate, mostAccOnly='true', catalog='ALL', feature='LE'):
    """
    Function to retrieve Cone CME paramters from the DONKI catalogue over a given time window.
    Args:
        startdate: Datetime of the start of the window
        enddate: Datetime of the end of the window
        mostAccOnly: Only download the best fit per feature (could be LE or shock)
        catalog: Which catalogue to search. Default to all.
        feature: LE or shock. Default to LE.

    Returns:
        cme_params: A dictionary of Cone CME parameters
    """

    startdate_str = startdate.strftime('%Y-%m-%d')
    stopdate_str = enddate.strftime('%Y-%m-%d')
    url_head = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CMEAnalysis?startDate="
    url_1 = url_head + startdate_str + '&endDate=' + stopdate_str
    url_2 = '&mostAccurateOnly=' + mostAccOnly + '&feature=' + feature
    url_3 = '&catalog=' + catalog
    url = url_1 + url_2 + url_3

    print(url)
    # read the json file
    response = urlopen(url)

    if response.status == 200:
        data = json.loads(response.read().decode("utf-8"))

        # convert to DataFrame
        df = pd.DataFrame(data)

        # standardise the headers
        df_renamed = df.rename(columns={'time21_5': 'ldates',
                                        'latitude': 'lat',
                                        'longitude': 'lon',
                                        'halfAngle': 'rmajor',
                                        'speed': 'vcld'})
        # convert to a dictionary
        cme_params = df_renamed.to_dict(orient='index')

    else:
        print("No repsonse for " + url)
        cme_params = None

    return cme_params


def get_DONKI_cme_list(model, startdate, enddate, mostAccOnly='true', catalog='ALL', feature='LE'):
    """
    Retrieves a list of Cone CME parameters from the DONKI catalogue and produces a list of coneCME objects for use in
    HUXt.
    Args:
        model: A HUXt model instance
        startdate: Datetime object of the start of the window to retrieve CME paramters.
        enddate:  Datetime object of the end of the window to retrieve CME paramters.
        mostAccOnly: Only download the best fit per feature (could be LE or shock)
        catalog: Which catalogue to search. Default to all.
        feature: LE or Shock. default to LE

    Returns:
        cme_list: A list of ConeCME objects.
    """
    cme_params = get_DONKI_coneCMEs(startdate, enddate,
                                    mostAccOnly=mostAccOnly,
                                    catalog=catalog, feature=feature)
    cme_list = cone_dict_to_cme_list(model, cme_params)

    return cme_list


def get_earth_lat(dt):
    """
    A function to return Earth latitude for a given date, in radians

    Args:
        dt : datetime

    Returns:
        E_lat: Earth latitude, with astropy units of radians

    """

    cr, cr_lon_init = datetime2huxtinputs(dt)
    # Use the HUXt ephemeris data to get Earth lat over the CR
    # ========================================================
    dummymodel = h.HUXt(v_boundary=np.ones(128)*400*(u.km/u.s), simtime=0.1*u.day, cr_num=cr, cr_lon_init=cr_lon_init,
                        lon_out=0.0*u.deg)
    # retrieve a bodies position at each model timestep:
    earth = dummymodel.get_observer('earth')
    # get average Earth lat
    E_lat = np.nanmean(earth.lat_c)
    
    return E_lat


def huxt_td_input_from_WSA_runs(datadir, start_dt, stop_dt, latitude, deacc=True, input_res_days=0.1, nlon=128,
                                format_template='models%2Fenlil%2FYYYY%2FMM%2FDD%2FHH%2Fwsa.gong.fits'):
    """
    Produces intput data for a time-dependent HUXt run from a collections of pre-downloaded WSA solutions.

    Args:
        datadir: string, path of WSA data files
        start_dt: datetime, start of the window to query
        stop_dt: datetime, end of the window to query
        latitude: float*u.rad, latitude at which to extract WSA V and Br
        deacc: bool, deaccelerate WSA speeds from 1 AU to 0.1 AU
        input_res_days: float, resolution (in days) at which HUXt input is generated
        nlon: int, number of longitude grid cells (should match HUXt model)
        format_template: str, file format with YYYY, MM, DD, HH, mm and ss used to identify the timestamp
    
    Returns:
        vlongs: 2d array, solar wind speed as fucntion of lon and time
        brlongs, 2d array, Br as a function of lon and time
        lon, 1d array, units of rad, longitudes
        mjds, 1d array, MJD
        times, 1 array, datetimes

    """

    datadir = Path(datadir)
    assert datadir.is_dir()
    
    def parse_format(filename, format_template):
        """
        Attempt to extract a datetime from filename using the given format_template.
        Returns a datetime object or None.
        """
        
        PLACEHOLDERS = {"YYYY": 4, "MM": 2, "DD": 2, "HH": 2, "mm": 2, "SS": 2}
        idx = 0
        date_parts = {}
        i = 0
  
        while i < len(format_template):
            matched = False
            for key, length in PLACEHOLDERS.items():
                if format_template[i:i+len(key)] == key:
                    # Extract the part from filename
                    date_str = filename[idx:idx+length]
                    if not date_str.isdigit():
                        return None  # Invalid number, fail early
                    date_parts[key] = int(date_str)
                    i += len(key)
                    idx += length
                    matched = True
                    break
            if not matched:
                # Literal character — must match exactly
                if i >= len(format_template) or idx >= len(filename):
                    return None
                if format_template[i] != filename[idx]:
                    return None
                i += 1
                idx += 1
  
        try:
            return datetime.datetime(
                date_parts.get("YYYY", 1900),
                date_parts.get("MM", 1),
                date_parts.get("DD", 1),
                date_parts.get("HH", 0),
                date_parts.get("mm", 0),
                date_parts.get("SS", 0)
            )
        except ValueError:
            return None

    def get_files_in_date_range(datadir, start_dt, end_dt, format_template):
        files_with_dates = []
  
        for filename in datadir.iterdir():
            file_date = parse_format(filename.name, format_template)
            if file_date and start_dt <= file_date <= end_dt:
                files_with_dates.append((file_date, filename))
  
        files_with_dates.sort(key=lambda x: x[0])
        return files_with_dates

    # get all the files in a given directory that are within the data range
    files_with_dates = get_files_in_date_range(datadir, start_dt, stop_dt, format_template)

    # read in each file and extract the speeds and Br at a given lat
    vlong_list = []
    brlong_list = []
    mjd_list = []
    
    # get the required longitude grid
    dlon = 2*np.pi / nlon
    lon_min_full = dlon / 2.0
    lon_max_full = 2*np.pi - (dlon / 2.0)
    lon, dlon = np.linspace(lon_min_full, lon_max_full, nlon, retstep=True)
    lon = lon*u.rad

    for filenum in range(0, len(files_with_dates)):
    
        filepath = files_with_dates[filenum][1]
        dt = files_with_dates[filenum][0]
        
        # get the longitude grid of the map
        if filepath.exists():
            vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits = get_WSA_maps(filepath)
                
        # get the Earth lat slice
        v_in = get_WSA_long_profile(filepath, lat=latitude)
        if deacc:
            # deaccelerate them?
            v_in, lon_temp = map_v_inwards(v_in, 215 * u.solRad, vr_longs,  21.5 * u.solRad)

        br_in = get_WSA_br_long_profile(filepath, lat=latitude)
         
        # store the data, on the required longitude grid
        vlong_list.append(np.interp(lon, vr_longs, v_in))
        brlong_list.append(np.interp(lon, br_longs, br_in))
        mjd_list.append(Time(dt).mjd)

    # convert to arrays
    vlongs_1d = np.array(vlong_list).T
    brlongs_1d = np.array(brlong_list).T
    mjds_1d = np.array(mjd_list)
    
    n_longs = len(vlongs_1d[:, 0])

    # increase the time resolution of the vlongs for the time-dependent runs
    mjds = np.arange(mjds_1d[0], mjds_1d[-1], input_res_days)
    times = Time(mjds + 2400000.5, format='jd').datetime
    vlongs = np.ones((n_longs, len(mjds)))
    brlongs = np.ones((n_longs, len(mjds)))
    for n in range(0, n_longs):
        vlongs[n, :] = np.interp(mjds, mjds_1d, vlongs_1d[n, :])
        brlongs[n, :] = np.interp(mjds, mjds_1d, brlongs_1d[n, :])
    
    return vlongs, brlongs, lon, mjds, times
