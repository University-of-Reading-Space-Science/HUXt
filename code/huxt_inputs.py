import datetime
import os
import urllib
from urllib.request import urlopen
import json
import ssl
import copy

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import httplib2
import numpy as np
from pyhdf.SD import SD, SDC
import h5py
from scipy.io import netcdf, readsav
from scipy import interpolate
from sunpy.coordinates import sun
from sunpy.net import Fido
from sunpy.net import attrs
from sunpy.timeseries import TimeSeries
import requests
import pandas as pd
from dtaidistance import dtw

import huxt as H


def get_MAS_boundary_conditions(cr=np.NaN, observatory='', runtype='', runnumber='', masres=''):
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

    assert (np.isnan(cr) == False)

    # The order of preference for different MAS run results
    overwrite = False
    if not masres:
        masres_order = ['high', 'medium']
    else:
        masres_order = [str(masres)]
        overwrite = True  # If the user wants a specific observatory, overwrite what's already downloaded

    if not observatory:
        observatories_order = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']
    else:
        observatories_order = [str(observatory)]
        overwrite = True  # If the user wants a specific observatory, overwrite what's already downloaded

    if not runtype:
        runtype_order = ['masp', 'mas', 'mast']
    else:
        runtype_order = [str(runtype)]
        overwrite = True

    if not runnumber:
        runnumber_order = ['0201', '0101']
    else:
        runnumber_order = [str(runnumber)]
        overwrite = True

    # Get the HUXt boundary condition directory
    dirs = H._setup_dirs_()
    _boundary_dir_ = dirs['boundary_conditions']

    # Example URL: http://www.predsci.com/data/runs/cr2010-medium/mdi_mas_mas_std_0101/helio/br_r0.hdf
    heliomas_url_front = 'http://www.predsci.com/data/runs/cr'
    heliomas_url_end = '_r0.hdf'

    vrfilename = 'HelioMAS_CR' + str(int(cr)) + '_vr' + heliomas_url_end
    brfilename = 'HelioMAS_CR' + str(int(cr)) + '_br' + heliomas_url_end

    if (os.path.exists(os.path.join(_boundary_dir_, brfilename)) is False or
            os.path.exists(os.path.join(_boundary_dir_, vrfilename)) is False or
            overwrite is True):  # Check if the files already exist

        # Search MHDweb for a HelioMAS run, in order of preference
        h = httplib2.Http(disable_ssl_certificate_validation=False)
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

                        # See if this br file exists
                        resp = h.request(url, 'HEAD')
                        if int(resp[0]['status']) < 400:
                            foundfile = True

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
        urllib.request.urlretrieve(urlbase + 'br' + heliomas_url_end,
                                   os.path.join(_boundary_dir_, brfilename))
        urllib.request.urlretrieve(urlbase + 'vr' + heliomas_url_end,
                                   os.path.join(_boundary_dir_, vrfilename))

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
        MAS_vr: Solar wind speed at 30rS, np.array in units of km/s.
        MAS_vr_Xa: Carrington longitude of Vr map, np.array in units of rad.
        MAS_vr_Xm: Latitude of Vr as angle down from N pole, np.array in units of rad.
        MAS_br: Radial magnetic field at 30rS, dimensionless np.array.
        MAS_br_Xa: Carrington longitude of Br map, np.array in units of rad.
        MAS_br_Xm: Latitude of Br as angle down from N pole, np.array in units of rad.
    """
    # Get the boundary condition directory
    dirs = H._setup_dirs_()
    _boundary_dir_ = dirs['boundary_conditions']
    # Create the filenames
    heliomas_url_end = '_r0.hdf'
    vrfilename = 'HelioMAS_CR' + str(int(cr)) + '_vr' + heliomas_url_end
    brfilename = 'HelioMAS_CR' + str(int(cr)) + '_br' + heliomas_url_end

    filepath = os.path.join(_boundary_dir_, vrfilename)
    assert os.path.exists(filepath)

    file = SD(filepath, SDC.READ)

    sds_obj = file.select('fakeDim0')  # select sds
    MAS_vr_Xa = sds_obj.get()  # get sds data
    sds_obj = file.select('fakeDim1')  # select sds
    MAS_vr_Xm = sds_obj.get()  # get sds data
    sds_obj = file.select('Data-Set-2')  # select sds
    MAS_vr = sds_obj.get()  # get sds data

    # Convert from model to physicsal units
    MAS_vr = MAS_vr * 481.0 * u.km / u.s
    MAS_vr_Xa = MAS_vr_Xa * u.rad
    MAS_vr_Xm = MAS_vr_Xm * u.rad

    filepath = os.path.join(_boundary_dir_, brfilename)
    assert os.path.exists(filepath)
    file = SD(filepath, SDC.READ)

    sds_obj = file.select('fakeDim0')  # select sds
    MAS_br_Xa = sds_obj.get()  # get sds data
    sds_obj = file.select('fakeDim1')  # select sds
    MAS_br_Xm = sds_obj.get()  # get sds data
    sds_obj = file.select('Data-Set-2')  # select sds
    MAS_br = sds_obj.get()  # get sds data

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
    assert (np.isnan(cr) == False and cr > 0)
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
        lat: Latitude at which to extract the longitudinal profile, measure up from equator. Float with units of deg

    Returns:
        br_in: Br as a function of Carrington longitude at solar equator.
               Interpolated to HUXt longitudinal resolution. np.array (NDIM = 1)
    """
    assert (np.isnan(cr) == False and cr > 0)
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
        vr_map: Solar wind speed as a Carrington longitude-latitude map. np.array with units of km/s
        vr_lats: The latitudes for the Vr map, relative to the equator. np.array with units of radians
        vr_longs: The Carrington longitudes for the Vr map, np.array with units of radians
    """

    assert (np.isnan(cr) == False and cr > 0)

    # Check the data exist, if not, download them
    flag = get_MAS_boundary_conditions(cr)
    if flag < 0:
        return -1, -1, -1

    # Read the HelioMAS data
    MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = read_MAS_vr_br(cr)

    vr_map = MAS_vr

    # Convert the lat angles from N-pole to equator centred
    vr_lats = (np.pi / 2) * u.rad - MAS_vr_Xm

    # Flip lats, so they're increasing in value
    vr_lats = np.flipud(vr_lats)
    vr_map = np.fliplr(vr_map)
    vr_longs = MAS_vr_Xa

    return vr_map.T, vr_longs, vr_lats


def get_MAS_br_map(cr):
    """
    A function to download, read and process MAS output to provide HUXt boundary conditions as lat-long maps,
    along with angle from equator for the maps.
    Maps returned in native resolution, not HUXt resolution.

    Args:
        cr: Integer, Carrington rotation number

    Returns:
        vr_map: Solar wind speed as a Carrington longitude-latitude map. np.array with units of km/s
        vr_lats: The latitudes for the Vr map, relative to the equator. np.array with units of radians
        vr_longs: The Carrington longitudes for the Vr map, np.array with units of radians
    """

    assert (np.isnan(cr) == False and cr > 0)

    # Check the data exist, if not, download them
    flag = get_MAS_boundary_conditions(cr)
    if flag < 0:
        return -1, -1, -1

    # Read the HelioMAS data
    MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = read_MAS_vr_br(cr)

    br_map = MAS_br

    # Convert the lat angles from N-pole to equator centred
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
    stream interactions.

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
    constants = H.huxt_constants()
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
    phi_new = H._zerototwopi_(lon_orig.value + (T_integral / Tsyn) * 2 * np.pi)

    return vnew * u.km / u.s, phi_new * u.rad


@u.quantity_input(v_orig=u.km / u.s)
@u.quantity_input(r_orig=u.solRad)
@u.quantity_input(r_inner=u.solRad)
def map_v_boundary_inwards(v_orig, r_orig, r_new, b_orig=np.nan):
    """
    Function to map a longitudinal V series from r_outer (in rs) to r_inner (in rs) accounting for residual
    acceleration, but neglecting stream interactions.
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
    stream interaction.
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
        # making boundaries periodic * u.km/u.s
        v_map_new[ilat, :] = np.interp(v_map_long.value,
                                       phis_new.value, v0.value, period=2 * np.pi)

        # check if b_pol needs mapping
        if np.isfinite(b_map).any():
            # check teh b abd v maps are the same dimensions
            assert (v_map.shape == b_map.shape)
            b_map_new[ilat, :] = np.interp(v_map_long.value,
                                           phis_new.value, b_map[ilat, :], period=2 * np.pi)

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
        vr_map: np.array, Solar wind speed as a Carrington longitude-latitude map. In km/s
        vr_lats: np.array, The latitudes for the Vr map, in radians from the equator
        vr_longs: np.array, The Carrington longitudes for the Vr map, in radians
        br_map:  np.array, Br as a Carrington longitude-latitude map. Dimensionless
        br_lats: np.array, The latitudes for the Br map, in radians from the equator
        br_longs: np.array, The Carrington longitudes for the Br map, in radians

    """

    assert os.path.exists(filepath)
    nc = netcdf.netcdf_file(filepath, 'r', mmap=False)
    br_map = nc.variables['br'][:]
    vr_map = nc.variables['vr'][:] * u.km / u.s
    phi = nc.variables['ph'][:]
    cotheta = nc.variables['cos(th)'][:]

    nc.close()

    phi = phi * u.rad
    theta = (np.pi / 2 - np.arccos(cotheta)) * u.rad
    vr_lats = theta[:, 0]
    br_lats = vr_lats
    vr_longs = phi[0, :]
    br_longs = vr_longs

    return vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats


def get_CorTom_vr_map(filepath):
    """
    A function to load, read and process CorTom density output to provide HUXt V boundary conditions as lat-long maps.
    Maps returned in native resolution, not HUXt resolution.
    Maps are not transformed - make sure the CorTom maps are Carrington maps

    Args:
        filepath: String, The filepath for the CorTom.txt file
    Returns:
        vr_map: np.array, Solar wind speed as a Carrington longitude-latitude map. In km/s
        vr_lats: np.array, The latitudes for the Vr map, in radians from trhe equator
        vr_longs: np.array, The Carrington longitudes for the Vr map, in radians
        phi: meshgrid og longitudes
        theta: mesh grid of latitudes

    """

    cortom_data = readsav(filepath)
    vr_map = copy.copy(cortom_data['velocity'])
    vr_colat = copy.copy(cortom_data['colat_rad'])
    vr_longs = copy.copy(cortom_data['lon_rad'])

    vr_lats = (np.pi / 2 - vr_colat) * u.rad
    # Flip so south pole at bottom
    vr_lats = np.flipud(vr_lats)
    vr_map = np.flipud(vr_map)

    return vr_map * u.km / u.s, vr_longs * u.rad, vr_lats * u.rad


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

    assert os.path.exists(filepath)
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
    carrlong = _zerototwopi_((hdul[0].header['CARRLONG']) * np.pi / 180)

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
        interp = interpolate.interp1d(_zerototwopi_(vr_long_centres + carrlong),
                                      vr_map_fits[nlat, :], kind="nearest",
                                      fill_value="extrapolate")
        vr_map[nlat, :] = interp(vr_long_centres)

    for nlat in range(0, len(br_lat_centres)):
        interp = interpolate.interp1d(_zerototwopi_(br_long_centres + carrlong),
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
        lat: Latitude at which to extract the longitudinal profile, measure up from equator. Float with units of deg

    Returns:
        vr_in: Solar wind speed as a function of Carrington longitude at solar equator.
               Interpolated to the default HUXt longitudinal grid. np.array (NDIM = 1) in units of km/s
    """
    assert (lat >= -90.0 * u.deg)
    assert (lat <= 90.0 * u.deg)
    assert (os.path.isfile(filepath))

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
        lat: Latitude at which to extract the longitudinal profile, measure up from equator. Float with units of deg

    Returns:
        vr_in: Solar wind speed as a function of Carrington longitude at solar equator.
               Interpolated to the default HUXt longitudinal grid. np.array (NDIM = 1) in units of km/s
    """
    assert (lat >= -90.0 * u.deg)
    assert (lat <= 90.0 * u.deg)
    assert (os.path.isfile(filepath))

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
        lat: Latitude at which to extract the longitudinal profile, measure up from equator. Float with units of deg

    Returns:
        vr_in: Solar wind speed as a function of Carrington longitude at solar equator.
               Interpolated to the default HUXt longitudinal grid. np.array (NDIM = 1) in units of km/s
    """
    assert (lat >= -90.0 * u.deg)
    assert (lat <= 90.0 * u.deg)
    assert (os.path.isfile(filepath))

    vr_map, lon_map, lat_map, br_map, br_lon, br_lat = get_PFSS_maps(filepath)

    # Extract the value at the given latitude
    vr = np.zeros(lon_map.shape)
    for i in range(lon_map.size):
        vr[i] = np.interp(lat.to(u.rad).value, lat_map.to(u.rad).value, vr_map[:, i].value)

    return vr * u.km / u.s


def get_CorTom_long_profile(filepath, lat=0.0 * u.deg):
    """
    Function to read and process CorTom (Coronal Tomography) output to provide a longitude profile at a specified
    latitude of the solar wind speed for use as boundary conditions in HUXt.

    Args:
        filepath: A complete path to the CorTom data file
        lat: Latitude at which to extract the longitudinal profile, measure up from equator. Float with units of deg

    Returns:
        vr_in: Solar wind speed as a function of Carrington longitude at solar equator.
               Interpolated to the default HUXt longitudinal grid. np.array (NDIM = 1) in units of km/s
    """
    assert (lat >= -90.0 * u.deg)
    assert (lat <= 90.0 * u.deg)
    assert (os.path.isfile(filepath))

    vr_map, lon_map, lat_map = get_CorTom_vr_map(filepath)

    # Extract the value at the given latitude
    vr = np.zeros(lon_map.shape)
    for i in range(lon_map.size):
        vr[i] = np.interp(lat.to(u.rad).value, lat_map.value, vr_map[:, i].value)

    return vr * u.km / u.s


def getMetOfficeWSAandCone(startdate, enddate, datadir=''):
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
    version = 'v1'
    api_key = os.getenv("API_KEY")
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
                    wsafilepath = os.path.join(datadir, wsa_file_name)
                    open(wsafilepath, "wb").write(response_wsa.content)
                    found_wsa = True
            if not found_cone:
                response_cone = requests.get(cone_file_url, headers={"apikey": api_key})
                if response_cone.status_code == 200:
                    conefilepath = os.path.join(datadir, cone_file_name)
                    open(conefilepath, "wb").write(response_cone.content)
                    found_cone = True
            i = i - 1
            if found_wsa and found_cone:
                success = True
                break

    return success, wsafilepath, conefilepath, model_time


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
    omni.loc[id_bad, 'V'] = np.NaN

    # create a BX_GSE field that is expected by some HUXt fucntions
    omni['BX_GSE'] = -omni['BR']

    # create a datetime column
    omni['datetime'] = omni.index
    # add an mjd column too
    omni['mjd'] = Time(omni['datetime']).mjd
    # reset the index
    omni = omni.reset_index()

    return omni


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
    Assumes all cone2bc files have the same structure, except for the number of cone cmes.
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
    # number of the CME in the cone2bc file (counting from 1 to N).
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

        # Get full angular width, cone2bc specifies angular half width under rmajor
        wid = 2 * cme_val['rmajor'] * u.deg

        # Set the initial height to be 21.5 rS, the default for WSA
        iheight = 21.5 * u.solRad

        # Thickness must be computed from CME cone initial radius and the xcld parameter,
        # which specifies the relative elongation of the cloud, 1=spherical,
        # 2=middle twice as long as cone radius e.g.
        # compute initial radius of the cone
        radius = np.abs(model.r[0] * np.tan(wid / 2.0))  # eqn on line 162 in ConeCME class
        # Thickness determined from xcld and radius
        thick = 5 * u.solRad  # (1.0 - cme_val['xcld']) * radius

        cme = H.ConeCME(t_launch=dt_cme, longitude=lon, latitude=lat,
                        width=wid, v=speed, thickness=thick, initial_height=iheight)
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
    cr, cr_lon_init = datetime2huxtinputs(time)
    dummymodel = H.HUXt(v_boundary=np.ones(128) * 400 * (u.km / u.s), simtime=1 * u.day, cr_num=cr,
                        cr_lon_init=cr_lon_init,
                        lon_out=0.0 * u.deg, r_min=21.5 * u.solRad)

    cme_list = ConeFile_to_ConeCME_list(dummymodel, filepath)
    return cme_list


def consolidate_cme_lists(cmelist_list, t_thresh=0.1 * u.day, lon_thresh=10 * u.deg, lat_thresh=10 * u.deg):
    """
    a function which takes a list of CME lists, as produced by multiple
    Hin.ConeFile_to_ConeCME_list_time outputs, and produces a consolidated list
    
    The list of cme lists should be in order from oldest to newest
    
    threshold parameters can be passed to define what counts as the same CME
    in multiple lists
    
    also removes duplicate CMEs within a single list, which are sometimes present
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
                        (H._zerototwopi_(this_long - existing_long) < lon_thresh) &
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
                                lon_stop=2 * np.pi * u.rad, lon_out=np.nan, bgrid_Carr=np.nan, track_cmes=True):
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
        latitude: The latitude (from equator) to run HUXt along
        frame: String, "synodic" or "sidereal", specifying the rotating frame of reference
        lon_out: Longitude for single 1-d run. Frame will be synodic
        lon_start: Longitude of one edge of the longitudinal domain of HUXt
        lon_stop: Longitude of the other edge of the longitudinal domain of HUXt
        bgrid_Carr: input magnetic polarity as a function of Carrington longitude and time
        track_cmes: Bool, whether to track CMEs through the simulation.
    returns:
        model: A HUXt instance initialised with the fully time dependent boundary conditions.
    """
    all_lons, dlon, nlon = H.longitude_grid()
    assert (len(vgrid_Carr[:, 0]) == nlon)

    # see if br boundary conditions are supplied
    do_b = False
    if np.isfinite(bgrid_Carr).any():
        do_b = True

    # work out the start time in terms of cr number and cr_lon_init
    cr, cr_lon_init = datetime2huxtinputs(starttime)

    # set up the dummy model class
    if np.isfinite(lon_out):
        model = H.HUXt(v_boundary=np.ones(nlon) * 400 * u.km / u.s,
                       lon_out=lon_out,
                       latitude=latitude,
                       r_min=r_min, r_max=r_max,
                       simtime=simtime, dt_scale=dt_scale,
                       cr_num=cr, cr_lon_init=cr_lon_init,
                       frame='synodic', track_cmes=track_cmes)
    else:
        model = H.HUXt(v_boundary=np.ones(nlon) * 400 * u.km / u.s,
                       lon_start=lon_start, lon_stop=lon_stop,
                       latitude=latitude,
                       r_min=r_min, r_max=r_max,
                       simtime=simtime, dt_scale=dt_scale,
                       cr_num=cr, cr_lon_init=cr_lon_init,
                       frame=frame, track_cmes=track_cmes)

    # extract the values from the model class
    buffertime = model.buffertime  # standard buffer time seems insufficient
    simtime = model.simtime
    frame = model.frame
    dt = model.dt
    cr_lon_init = model.cr_lon_init

    latitude = model.latitude
    time_init = model.time_init

    constants = H.huxt_constants()
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
    input_ambient_ts = np.nan * np.ones((model_time.size, nlon))
    if do_b:
        input_ambient_ts_b = np.nan * np.ones((model_time.size, nlon))

    for t in range(0, len(model_time)):

        mjd = time_init.mjd + model_time[t].to(u.day).value

        # find the nearest time to the current model time
        t_input = np.argmin(abs(time_grid - mjd))
        if model_time[t] < 0:  # spin up, use initiation time
            t_input = np.argmin(abs(time_grid - time_init.mjd))

        # shift the longitude to match the initial model time
        dlon_from_start = 2 * np.pi * u.rad * model_time[t] / rotation_period

        lon_shifted = H._zerototwopi_((all_lons - cr_lon_init + dlon_from_start).value)
        # put longitudes in ascending order for np.interp
        id_sort = np.argsort(lon_shifted)
        lon_shifted = lon_shifted[id_sort]

        # take the vlong slice at this value
        v_boundary = vgrid_Carr[:, t_input]
        v_b_shifted = v_boundary[id_sort]
        # interpolate back to the original grid
        v_boundary = np.interp(all_lons.value, lon_shifted, v_b_shifted, period=2 * np.pi)
        input_ambient_ts[t, :] = v_boundary

        if do_b:
            b_boundary = bgrid_Carr[:, t_input]
            b_b_shifted = b_boundary[id_sort]
            # interpolate back to the original grid
            b_boundary = np.interp(all_lons.value, lon_shifted, b_b_shifted, period=2 * np.pi)
            input_ambient_ts_b[t, :] = b_boundary

    # fill the nan values
    mask = np.isnan(input_ambient_ts)
    input_ambient_ts[mask] = 400
    if do_b:
        mask = np.isnan(input_ambient_ts_b)
        input_ambient_ts_b[mask] = 0

    if do_b:
        # set up the model class with these data initialised

        if np.isfinite(lon_out):  # single longitude
            model = H.HUXt(v_boundary=np.ones(128) * 400 * u.km / u.s,
                           simtime=simtime,
                           cr_num=cr, cr_lon_init=cr_lon_init,
                           r_min=r_min, r_max=r_max,
                           dt_scale=dt_scale, latitude=latitude,
                           frame='synodic',
                           lon_out=lon_out,
                           input_v_ts=input_ambient_ts,
                           input_b_ts=input_ambient_ts_b,
                           input_t_ts=model_time,
                           track_cmes=track_cmes)
        else:  # multiple longitudes
            model = H.HUXt(v_boundary=np.ones(128) * 400 * u.km / u.s,
                           simtime=simtime,
                           cr_num=cr, cr_lon_init=cr_lon_init,
                           r_min=r_min, r_max=r_max,
                           dt_scale=dt_scale, latitude=latitude,
                           frame=frame,
                           lon_start=lon_start, lon_stop=lon_stop,
                           input_v_ts=input_ambient_ts,
                           input_b_ts=input_ambient_ts_b,
                           input_t_ts=model_time,
                           track_cmes=track_cmes)

    else:
        # set up the model class without B
        if np.isfinite(lon_out):  # single longitude
            model = H.HUXt(v_boundary=np.ones(128) * 400 * u.km / u.s,
                           simtime=simtime,
                           cr_num=cr, cr_lon_init=cr_lon_init,
                           r_min=r_min, r_max=r_max,
                           dt_scale=dt_scale, latitude=latitude,
                           frame='synodic',
                           lon_out=lon_out,
                           input_v_ts=input_ambient_ts,
                           input_t_ts=model_time,
                           track_cmes=track_cmes)

        else:  # multiple longitudes
            model = H.HUXt(v_boundary=np.ones(128) * 400 * u.km / u.s,
                           simtime=simtime,
                           cr_num=cr, cr_lon_init=cr_lon_init,
                           r_min=r_min, r_max=r_max,
                           dt_scale=dt_scale, latitude=latitude,
                           frame=frame,
                           lon_start=lon_start, lon_stop=lon_stop,
                           input_v_ts=input_ambient_ts,
                           input_t_ts=model_time,
                           track_cmes=track_cmes)

    return model


def _zerototwopi_(angles):
    """
    Function to constrain angles to the 0 - 2pi domain.

    Args:
        angles: a numpy array of angles.
    Returns:
        angles_out: a numpy array of angles constrained to 0 - 2pi domain.
    """
    twopi = 2.0 * np.pi
    angles_out = angles
    a = -np.floor_divide(angles_out, twopi)
    angles_out = angles_out + (a * twopi)
    return angles_out


def generate_vCarr_from_OMNI(runstart, runend, nlon_grid=None, omni_input=None, dt=1 * u.day, ref_r=215 * u.solRad,
                             corot_type='both'):
    """
    A function to download OMNI data and generate V_carr and time_grid for use with set_time_dependent_boundary

    Args:
        runstart: Start time as a datetime
        runend: End time as a datetime
        nlon_grid: Int. If none specified, will be set to the current HUXt value (usually 128)
        omni_input: Optional input for supplying the OMNI data. If left as None, it will be downloaded at runtime.
        dt: time resolution, in days is 1*u.day.
        ref_r: radial distance to produce v at, 215*u.solRad by default.
        corot_type: String that determines corot type (both, back, forward)
    Returns:
        Time: Array of times as modified Julian days
        Vcarr: Array of solar wind speeds mapped as a function of Carr long and time
        bcarr: Array of Br mapped as a function of Carr long and time
    """

    # check the coro_type is one of the accepted values
    assert corot_type == 'both' or corot_type == 'back' or corot_type == 'forward'

    # set the default longitude grid, check specified value
    all_lons, dlon, nlon = H.longitude_grid()
    if nlon_grid is None:
        nlon_grid = nlon
    if not (nlon_grid == nlon):
        print('Warning: vCarr generated for different longitude resolution than current HUXt default')

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
        cr[i], cr_lon_init[i] = datetime2huxtinputs(omni_int['datetime'][i])

    omni_int['Carr_lon'] = cr_lon_init.value  # remove unit as this confuses pd.DataFrame.copy() needed later
    omni_int['Carr_lon_unwrap'] = np.unwrap(omni_int['Carr_lon'].to_numpy())

    omni_int['mjd'] = [t.mjd for t in omni_int['Time'].array]

    # get the Earth radial distance info.
    dirs = H._setup_dirs_()
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
        omni_int.loc[t, 'Carr_lon_ref'] = omni_int.loc[
                                              t, 'Carr_lon_ref'] - delta_t * daysec.value * 2 * np.pi / synodic_period.value

    # sort the omni data by Carr_lon_ref for interpolation
    omni_temp = omni_int.copy()
    omni_temp = omni_temp.sort_values(by=['Carr_lon_ref'])

    # now remap these speeds back on to the original time steps
    omni_int['V_ref'] = np.interp(omni_int['Carr_lon_unwrap'],
                                  omni_temp['Carr_lon_ref'], omni_temp['V'])
    omni_int['Br_ref'] = np.interp(omni_int['Carr_lon_unwrap'],
                                   omni_temp['Carr_lon_ref'], -omni_temp['BX_GSE'])

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

    for t in range(0, len(time_grid)):
        # find nearest time and current Carrington longitude
        t_id = np.argmin(np.abs(omni_int['mjd'] - time_grid[t]))
        Elong = omni_int['Carr_lon'][t_id] * u.rad

        # get the Carrington longitude difference from current Earth pos
        dlong_back = _zerototwopi_(lon_grid.value - Elong.value) * u.rad
        dlong_forward = _zerototwopi_(Elong.value - lon_grid.value) * u.rad

        dt_back = (dlong_back / omega_synodic).to(u.day)
        dt_forward = (dlong_forward / omega_synodic).to(u.day)

        vgrid_carr_recon_back[:, t] = np.interp(time_grid[t] - dt_back.value,
                                                omni_int['mjd'], omni_int['V_ref'],
                                                left=np.nan, right=np.nan)
        bgrid_carr_recon_back[:, t] = np.interp(time_grid[t] - dt_back.value,
                                                omni_int['mjd'], omni_int['Br_ref'],
                                                left=np.nan, right=np.nan)

        vgrid_carr_recon_forward[:, t] = np.interp(time_grid[t] + dt_forward.value,
                                                   omni_int['mjd'], omni_int['V_ref'],
                                                   left=np.nan, right=np.nan)
        bgrid_carr_recon_forward[:, t] = np.interp(time_grid[t] + dt_forward.value,
                                                   omni_int['mjd'], omni_int['Br_ref'],
                                                   left=np.nan, right=np.nan)

        numerator = (dt_forward * vgrid_carr_recon_back[:, t] + dt_back * vgrid_carr_recon_forward[:, t])
        denominator = dt_forward + dt_back
        vgrid_carr_recon_both[:, t] = numerator / denominator

        numerator = (dt_forward * bgrid_carr_recon_back[:, t] + dt_back * bgrid_carr_recon_forward[:, t])
        bgrid_carr_recon_both[:, t] = numerator / denominator
    # cut out the requested time
    mask = ((time_grid >= Time(runstart).mjd) & (time_grid <= Time(runend).mjd))

    if corot_type == 'both':
        return time_grid[mask], vgrid_carr_recon_both[:, mask], bgrid_carr_recon_both[:, mask]
    elif corot_type == 'back':
        return time_grid[mask], vgrid_carr_recon_back[:, mask], bgrid_carr_recon_back[:, mask]
    elif corot_type == 'forward':
        return time_grid[mask], vgrid_carr_recon_forward[:, mask], bgrid_carr_recon_forward[:, mask]


def generate_vCarr_from_OMNI_DTW(runstart, runend, nlon=None, omni_input=None, res='24h', psi_days=7 * u.day,
                                 max_warp_days=3 * u.day, dtw_on='V'):
    """
    A function to download OMNI data and generate V_carr and time_grid for 
    use with set_time_dependent_boundary. Uses dynamic time warping, rather than
    corotation

    Args:
        runstart: Datetime object. Start of the interval
        runend: Datetime object. End of the interval
        nlon: Int. If none specified, will be set to the current HUXt value (usually 128)
        omni_input: Optional input of OMNI data. If left as None is downloaded at runtime.
        res: String. Time averaging of OMNI prior to DTW. match to longitude (for nlon = 128, use '5h')
        psi_days: Float, in units of days. DTW parameter, determining how many days can be ignored at the start/end of the fit.
        max_warp_days: Float, in units of days. DTW parameter, determining maximum warp allowed.
        dtw_on. String. Name of the omni dataframe column to be used to determine the DTW paths
    Returns:
        Time: Array of times as modified Julian days
        Vcarr: Array of solar wind speeds mapped as a function of Carr long and time
        bcarr: Array of Br mapped as a function of Carr long and time
    """

    # set the default longitude grid, check specified value
    all_lons_huxt, dlon_huxt, nlon_huxt = H.longitude_grid()
    if nlon is None:
        nlon = nlon_huxt
    if not (nlon == nlon_huxt):
        print('Warning: vCarr generated for different longitude resolution than current HUXt default')

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
    temp = datetime2huxtinputs(omni['datetime'].to_numpy())
    omni['carr_lon'] = temp[1].value
    # unwrap this.
    omni['clon_unwrap'] = np.unwrap(omni['carr_lon'].to_numpy())

    # interpolate to the required longitude grid 

    # average up to a given res for a clearer plot
    omni_res = omni.resample(res, on='datetime').mean()
    omni_res['datetime'] = Time(omni_res['mjd'], format='mjd').to_datetime(leap_second_strict='silent')
    omni_res.reset_index(drop=True, inplace=True)

    # compute carrington longitude of earth for each point
    temp = datetime2huxtinputs(omni_res['datetime'].to_numpy())
    omni_res['carr_lon'] = temp[1].value
    # unwrap this.
    omni_res['clon_unwrap'] = np.unwrap(omni_res['carr_lon'].to_numpy())

    clon_min = omni_res['clon_unwrap'].min()

    # now interpolate this time series onto the required Carr long grid
    # ==================================================================
    dlon = 2 * np.pi / nlon
    clon_unwrap_grid = - np.arange(-2 * np.pi - dlon, -clon_min + 2 * np.pi, dlon)

    v_clon = np.interp(-clon_unwrap_grid, -omni_res['clon_unwrap'].to_numpy(),
                       omni_res['V'].to_numpy(),
                       left=np.nan, right=np.nan)
    mjd_clon = np.interp(-clon_unwrap_grid, -omni_res['clon_unwrap'].to_numpy(),
                         omni_res['mjd'].to_numpy(),
                         left=np.nan, right=np.nan)
    bx_clon = np.interp(-clon_unwrap_grid, -omni_res['clon_unwrap'].to_numpy(),
                        omni_res['BX_GSE'].to_numpy(),
                        left=np.nan, right=np.nan)
    dtwon_clon = np.interp(-clon_unwrap_grid, -omni_res['clon_unwrap'].to_numpy(),
                           omni_res[dtw_on].to_numpy(),
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


def remove_ICMEs(data_df, icmes, interpolate=True, icme_buffer=0.1 * u.day, interp_buffer=1 * u.day,
                 params=['V', 'BX_GSE'], fill_vals=None):
    """
    A function to remove ICMEs from a given time series

    Parameters
    ----------
    data_df : pd.dataframe
        Time series with 'mjd' and reset index, such as provided by get_omni
    icmes : list
        
    interpolate : bool
        Whether to interpolate through ICMEs NaNs. The default is True.
    icme_buffer : float with units of day
        How much additional data to remove about the ICME boundaries. The default is 0.1*u.day.
    interp_buffer : float, with units of day
        How much of an average to take up and downstream. The default is 1*u.day.
    params : list of strings
        which parameters to remove. The default is ['V', 'BX_GSE'].
    fill_vals : list of floats, possibly with units
        the fill values to use for interpolation if the upstream or downstream 
        data are all nans
    Returns
    -------
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
    for i in range(0, len(icmes)):

        icme_start = icmes['shock_mjd'][i] - icme_buffer_d
        icme_stop = icmes['end_mjd'][i] + icme_buffer_d

        mask_icme = ((data['mjd'] >= icme_start) &
                     (data['mjd'] <= icme_stop))

        if any(mask_icme):
            print('removing ICME #' + str(i))
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
