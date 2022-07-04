import datetime
import os
import urllib
import ssl

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import httplib2
import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC
from scipy.io import netcdf
from scipy import interpolate
from sunpy.coordinates import sun
from sunpy.net import Fido
from sunpy.net import attrs
from sunpy.timeseries import TimeSeries

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
    # heliomas_url_front = 'https://shadow.predsci.com/data/runs/cr'
    heliomas_url_front = 'http://www.predsci.com/data/runs/cr'
    heliomas_url_end = '_r0.hdf'

    vrfilename = 'HelioMAS_CR' + str(int(cr)) + '_vr' + heliomas_url_end
    brfilename = 'HelioMAS_CR' + str(int(cr)) + '_br' + heliomas_url_end

    if (os.path.exists(os.path.join(_boundary_dir_, brfilename)) == False or
       os.path.exists(os.path.join(_boundary_dir_, vrfilename)) == False or
       overwrite == True):  # Check if the files already exist

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

        if foundfile == False:
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
    Function to download, read and process MAS output to provide a longitude profile at a specified latitude
    of the solar wind speed for use as boundary conditions in HUXt.

    Args:
        cr: Integer Carrington rotation number
        lat: Latitude at which to extract the longitudinal profile, measure up from equator. Float with units of deg

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

    # Now interpolate on to the HUXt longitudinal grid
    longs, dlon, nlon = H.longitude_grid(lon_start=0.0 * u.rad, lon_stop=2 * np.pi * u.rad)
    vr_in = np.interp(longs.value, MAS_vr_Xa.value, vr) * u.km / u.s

    return vr_in


def get_MAS_br_long_profile(cr, lat=0.0 * u.deg):
    """
    Function to download, read and process MAS output to provide a longitude profile at a specified latitude
    of the Br for use as boundary conditions in HUXt.

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

    # Now interpolate on to the HUXt longitudinal grid
    longs, dlon, nlon = H.longitude_grid(lon_start=0.0 * u.rad, lon_stop=2 * np.pi * u.rad)
    br_in = np.interp(longs.value, MAS_br_Xa.value, br) 

    return br_in


def get_MAS_vr_map(cr):
    """
    A function to download, read and process MAS output to provide HUXt boundary
    conditions as lat-long maps, along with angle from equator for the maps.
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


@u.quantity_input(v_outer=u.km / u.s)
@u.quantity_input(r_outer=u.solRad)
@u.quantity_input(lon_outer=u.rad)
@u.quantity_input(r_inner=u.solRad)
def map_v_inwards(v_orig, r_orig, lon_orig, r_new):
    """
    Function to map v from r_orig (in rs) to r_inner (in rs) accounting for 
    residual acceleration, but neglecting stream interactions.

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

    # Compute the 30 rS speed
    v0 = v_orig.value * (1 + alpha * (1 - np.exp((r_orig - r_new) / rH)))

    # Compute the transit time from the new to old inner boundary heights (i.e., integrate equations 3 and 4 wrt to r)
    A = v0 + alpha * v0
    term1 = rH * np.log(A * np.exp(r_orig / rH) - alpha * v0 * np.exp(r_new / rH)) / A
    term2 = rH * np.log(A * np.exp(r_new / rH) - alpha * v0 * np.exp(r_new / rH)) / A
    T_integral = term1 - term2

    # Work out the longitudinal shift
    phi_new = H._zerototwopi_(lon_orig.value + (T_integral / Tsyn) * 2 * np.pi)

    return v0 * u.km / u.s, phi_new * u.rad


@u.quantity_input(v_orig=u.km / u.s)
@u.quantity_input(r_orig=u.solRad)
@u.quantity_input(r_inner=u.solRad)
def map_v_boundary_inwards(v_orig, r_orig, r_new):
    """
    Function to map a longitudinal V series from r_outer (in rs) to r_inner (in rs)
    accounting for residual acceleration, but neglecting stream interactions.
    Series return on HUXt longitudinal grid, not input grid

    Args:
        v_orig: Solar wind speed as function of long at outer radial boundary. Units of km/s.
        r_orig: Radial distance at original radial boundary. Units of km.
        r_new: Radial distance at new radial boundary. Units of km.

    Returns:
        v_new: Solar wind speed as funciton of long mapped from r_orig to r_new. Units of km/s.
    """

    # Compute the longitude grid from the length of the vouter input variable
    lon, dlon, nlon = H.longitude_grid()

    # Map each point in to a new speed and longitude
    v0, phis_new = map_v_inwards(v_orig, r_orig, lon, r_new)

    # Interpolate the mapped speeds back onto the regular Carr long grid,
    # making boundaries periodic
    v_new = np.interp(lon, phis_new, v0, period=2 * np.pi)

    return v_new


@u.quantity_input(v_map=u.km / u.s)
@u.quantity_input(v_map_lat=u.rad)
@u.quantity_input(v_map_long=u.rad)
@u.quantity_input(r_outer=u.solRad)
@u.quantity_input(r_inner=u.solRad)
def map_vmap_inwards(v_map, v_map_lat, v_map_long, r_orig, r_new):
    """
    Function to map a V Carrington map from r_orig (in rs) to r_new (in rs),
    accounting for acceleration, but ignoring stream interaction
    Map returned on input coord system, not HUXT resolution.

    Args:
        v_map: Solar wind speed Carrington map at original radial boundary. np.array with units of km/s.
        v_map_lat: Latitude (from equator) of v_map positions. np.array with units of radians
        v_map_long: Carrington longitude of v_map positions. np.array with units of radians
        r_orig: Radial distance at original radial boundary. np.array with units of km.
        r_new: Radial distance at new radial boundary. np.array with units of km.

    Returns:
        v_map_new: Solar wind speed map at r_inner. np.array with units of km/s.
    """

    # Check the dimensions
    assert (len(v_map_lat) == len(v_map[:, 1]))
    assert (len(v_map_long) == len(v_map[1, :]))

    v_map_new = np.ones((len(v_map_lat), len(v_map_long)))
    for ilat in range(0, len(v_map_lat)):
        # Map each point in to a new speed and longitude
        v0, phis_new = map_v_inwards(v_map[ilat, :], r_orig, v_map_long, r_new)

        # Interpolate the mapped speeds back onto the regular Carr long grid,
        # making boundaries periodic * u.km/u.s
        v_map_new[ilat, :] = np.interp(v_map_long.value,
                                       phis_new.value, v0.value, period=2 * np.pi)

    return v_map_new * u.km / u.s


def get_PFSS_maps(filepath):
    """
    A function to load, read and process PFSSpy output to provide HUXt boundary
    conditions as lat-long maps, along with angle from equator for the maps.
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

    return vr_map, vr_longs, vr_lats,  br_map, br_longs, br_lats


def get_CorTom_vr_map(filepath, convert_from_density=False):
    """
    A function to load, read and process CorTom density output to 
    provide HUXt V boundary conditions as lat-long maps, 
    Maps returned in native resolution, not HUXt resolution.
    Maps are not transformed - make sure the CorTom maps are Carrington maps

    Args:
        filepath: String, The filepath for the CorTom.txt file
        convert_from_density : Old files were density, not speed. Use this flag to convert

    Returns:
        vr_map: np.array, Solar wind speed as a Carrington longitude-latitude map. In km/s
        vr_lats: np.array, The latitudes for the Vr map, in radians from trhe equator
        vr_longs: np.array, The Carrington longitudes for the Vr map, in radians
        phi: meshgrid og longitudes
        theta: mesh grid of latitudes

    """
    
    columns = ['carrlong', 'carrlat', 'vsw']
    den_df = pd.read_csv(filepath,  skiprows=2, names=columns)
    
    # apply a 180-degree long shift
    #den_df['carrlong'] = den_df['carrlong'] #+ 180.0
    #den_df.loc[den_df['carrlong'] > 360.0, 'carrlong'] = den_df['carrlong'] - 360.0
        
    # create a regular grid
    xvals = np.linspace(180.0/128, 360.0-180.0/128, num=360)
    yvals = np.linspace(-90+180.0/128, 90-180.0/128, num=180)
    
    # create a mesh using these new positions
    X, Y = np.meshgrid(xvals, yvals)
    
    # interpolate the data. probably easiest to just use 2d arrays here. Set ML as forecast and OP as reference
    griddata = interpolate.griddata((den_df['carrlong'], den_df['carrlat']), den_df['vsw'], (X, Y), method='linear')
    vgrid = griddata.copy()
    
    if convert_from_density:
        # convert to V
        dmax = 20000.0
        dmin = 4000.0
        
        vmin = 300.0
        vmax = 680.0

        vgrid[np.where(vgrid < dmin)] = dmin
        vgrid[np.where(vgrid > dmax)] = dmax
        vgrid = np.abs(vgrid - dmax)
        
        vgrid = vmin + (vmax-vmin) * (vgrid/(dmax-dmin))
        
    vgrid = vgrid * u.km/u.s
    phi = X * np.pi/180 * u.rad
    theta = Y * np.pi/180 * u.rad

    lons = phi[0, :]
    lats = theta[:, 0]
    return vgrid, lons, lats


def get_WSA_maps(filepath):
    """
    A function to load, read and process WSA FITS maps from the UK Met Office
    to provide HUXt boundary conditions as lat-long maps, along with angle from
    equator for the maps.
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
        phi: Mesh grid of vr_longs. np.array in units of radians.
        theta: Mesh grid of vr_lats. np.array in units of radians.
        cr: Integer, Carrington rotation number

    """

    assert os.path.exists(filepath)
    hdul = fits.open(filepath)

    cr_num = hdul[0].header['CARROT']
    dgrid = hdul[0].header['GRID'] * np.pi / 180
    carrlong = hdul[0].header['CARRLONG'] * np.pi / 180

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
    assert(os.path.isfile(filepath))

    vr_map, lon_map, lat_map, br_map, br_lon, br_lat, cr_num = get_WSA_maps(filepath)

    # Extract the value at the given latitude
    vr = np.zeros(lon_map.shape)
    for i in range(lon_map.size):
        vr[i] = np.interp(lat.to(u.rad).value, lat_map.to(u.rad).value, vr_map[:, i].value)

    # Now interpolate on to the HUXt longitudinal grid
    lon, dlon, nlon = H.longitude_grid(lon_start=0.0 * u.rad, lon_stop=2 * np.pi * u.rad)
    vr_in = np.interp(lon.value, lon_map.value, vr) * u.km / u.s

    return vr_in

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
    assert(os.path.isfile(filepath))

    vr_map, lon_map, lat_map, br_map, br_lon, br_lat = get_PFSS_maps(filepath)


    # Extract the value at the given latitude
    vr = np.zeros(lon_map.shape)
    for i in range(lon_map.size):
        vr[i] = np.interp(lat.to(u.rad).value, lat_map.to(u.rad).value, vr_map[:, i].value)

    # Now interpolate on to the HUXt longitudinal grid
    lon, dlon, nlon = H.longitude_grid(lon_start=0.0 * u.rad, lon_stop=2 * np.pi * u.rad)
    vr_in = np.interp(lon.value, lon_map.value, vr) * u.km / u.s

    return vr_in

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
    assert(os.path.isfile(filepath))

    vr_map, lon_map, lat_map = get_CorTom_vr_map(filepath)

    # Extract the value at the given latitude
    vr = np.zeros(lon_map.shape)
    for i in range(lon_map.size):
        vr[i] = np.interp(lat.to(u.rad).value, lat_map.to(u.rad).value, vr_map[:, i].value)

    # Now interpolate on to the HUXt longitudinal grid
    lon, dlon, nlon = H.longitude_grid(lon_start=0.0 * u.rad, lon_stop=2 * np.pi * u.rad)
    vr_in = np.interp(lon.value, lon_map.value, vr) * u.km / u.s

    return vr_in

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

    cr_frac = sun.carrington_rotation_number(dt)
    cr = int(np.floor(cr_frac))
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
            param_val = float(param_val)

        cmes[cme_id][param_name] = param_val

    return cmes


def ConeFile_to_ConeCME_list(model, filepath):
    """
    A function to produce a list of ConeCMEs for input to HUXt derived from a cone2bc.in file, as is used with
    to input Cone CMEs into Enlil. Assumes CME height of 21.5 rS
    Args:
        model: A HUXt instance.
        filepath: The path to the relevant cone2bc.in file.
    returns:
        cme_list: A list of ConeCME instances.
    """

    cme_params = import_cone2bc_parameters(filepath)

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
        iheight = 21.5*u.solRad

        # Thickness must be computed from CME cone initial radius and the xcld parameter,
        # which specifies the relative elongation of the cloud, 1=spherical,
        # 2=middle twice as long as cone radius e.g.
        # compute initial radius of the cone
        radius = np.abs(model.r[0] * np.tan(wid / 2.0))  # eqn on line 162 in ConeCME class
        # Thickness determined from xcld and radius
        thick = (1.0 - cme_val['xcld']) * radius

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
    dummymodel = H.HUXt(v_boundary=np.ones(128) * 400 * (u.km/u.s), simtime=1*u.day, cr_num=cr, cr_lon_init=cr_lon_init,
                        lon_out=0.0*u.deg, r_min=21.5*u.solRad)

    cme_list = ConeFile_to_ConeCME_list(dummymodel, filepath)
    return cme_list


def set_time_dependent_boundary(vgrid_Carr, time_grid, starttime, simtime, r_min=215*u.solRad, r_max=1290*u.solRad,
                                dt_scale=50, latitude=0*u.deg, frame='sidereal', lon_start=0*u.rad,
                                lon_stop=2*np.pi * u.rad):
    
    """
    A function to compute an explicitly time dependent inner boundary condition for HUXt,
    rather than due to synodic/sidereal rotation of static coronal strucutre.    

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
        lon_start: Longitude of one edge of the longitudinal domain of HUXt
        lon_stop: Longitude of the other edge of the longitudinal domain of HUXt
    returns:
        model: A HUXt instance initialised with the fully time dependent boundary conditions.
    """
    
    # work out the start time in terms of cr number and cr_lon_init
    cr, cr_lon_init = datetime2huxtinputs(starttime)
    
    # set up the dummy model class
    model = H.HUXt(v_boundary=np.ones(128)*400*u.km/u.s,
                   lon_start=lon_start, lon_stop=lon_stop,
                   latitude=latitude,
                   r_min=r_min, r_max=r_max,
                   simtime=simtime, dt_scale=dt_scale,
                   cr_num=cr, cr_lon_init=cr_lon_init,
                   frame=frame)
    
    # extract the values from the model class
    buffertime = model.buffertime  # standard buffer time seems insufficient
    simtime = model.simtime
    frame = model.frame
    dt = model.dt
    cr_lon_init = model.cr_lon_init
    all_lons, dlon, nlon = H.longitude_grid()
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
    
    for t in range(0, len(model_time)):
        
        mjd = time_init.mjd + model_time[t].to(u.day).value
        
        # find the nearest time to the current model time
        t_input = np.argmin(abs(time_grid - mjd))
        if model_time[t] < 0:  # spin up, use initiation time
            t_input = np.argmin(abs(time_grid - time_init.mjd))
        
        # shift the longitude to match the initial model time
        dlon_from_start = 2*np.pi * u.rad * model_time[t] / rotation_period
        
        lon_shifted = H._zerototwopi_((all_lons - cr_lon_init + dlon_from_start).value)
        # put longitudes in ascending order for np.interp
        id_sort = np.argsort(lon_shifted)
        lon_shifted = lon_shifted[id_sort]
        
        # take the vlong slice at this value
        v_boundary = vgrid_Carr[:, t_input]
        v_b_shifted = v_boundary[id_sort]
        # interpolate back to the original grid
        v_boundary = np.interp(all_lons.value, lon_shifted, v_b_shifted, period=2*np.pi)
        input_ambient_ts[t, :] = v_boundary
        
    # fill the nan values
    mask = np.isnan(input_ambient_ts)
    input_ambient_ts[mask] = 400   
    
    # set up the model class with these data initialised
    model = H.HUXt(v_boundary=np.ones(128)*400*u.km/u.s,
                   simtime=simtime,
                   cr_num=cr, cr_lon_init=cr_lon_init,
                   r_min=r_min, r_max=r_max,
                   dt_scale=dt_scale, latitude=latitude,
                   frame=frame,
                   lon_start=lon_start, lon_stop=lon_stop,
                   input_v_ts=input_ambient_ts,
                   input_t_ts=model_time)
     
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


def generate_vCarr_from_OMNI(runstart, runend, nlon_grid=128, dt=1*u.day):
    """
    A function to download OMNI data and generate V_carr and time_grid
    for use with set_time_dependent_boundary

    Args:
        runstart: Start time as a datetime
        runend: End time as a datetime
        nlon_grid: Int, 128 by default
        dt: time resolution, in days is 1*u.day.
    Returns:
        Time: Array of times as Julian dates
        Vcarr: Array of solar wind speeds mapped as a function of Carr long and time
    """

    # download an additional 28 days either side
    starttime = runstart - datetime.timedelta(days=28)
    endtime = runend + datetime.timedelta(days=28)
    
    # Download the 1hr OMNI data from CDAweb
    trange = attrs.Time(starttime, endtime)
    dataset = attrs.cdaweb.Dataset('OMNI2_H0_MRG1HR')
    result = Fido.search(trange, dataset)
    downloaded_files = Fido.fetch(result)

    # Import the OMNI data
    omni = TimeSeries(downloaded_files, concatenate=True)
    data = omni.to_dataframe()
    
    # # Set invalid data points to NaN
    id_bad = data['V'] == 9999.0
    data.loc[id_bad, 'V'] = np.NaN
    
    # create a datetime column
    data['datetime'] = data.index.to_pydatetime()

    # compute the syndoic rotation period
    daysec = 24 * 60 * 60 * u.s
    synodic_period = 27.2753 * daysec  # Solar Synodic rotation period from Earth.
    omega_synodic = 2*np.pi * u.rad / synodic_period

    # find the period of interest
    mask = ((data['datetime'] > starttime) &
            (data['datetime'] < endtime))
    omni = data[mask]
    omni = omni.reset_index()
    omni['Time'] = Time(omni['datetime'])
    
    smjd = omni['Time'][0].mjd
    fmjd = omni['Time'][len(omni) - 1].mjd

    # interpolate through OMNI V data gaps
    omni_int = omni.interpolate(method='linear', axis=0).ffill().bfill()
    del omni
    
    # compute carrington longitudes
    cr = np.ones(len(omni_int))
    cr_lon_init = np.ones(len(omni_int))*u.rad
    for i in range(0, len(omni_int)):
        cr[i], cr_lon_init[i] = datetime2huxtinputs(omni_int['datetime'][i])

    omni_int['Carr_lon'] = cr_lon_init

    omni_int['mjd'] = [t.mjd for t in omni_int['Time'].array]

    # compute the longitudinal and time grids
    dphi_grid = 360/nlon_grid
    lon_grid = np.arange(dphi_grid/2, 360.1-dphi_grid/2, dphi_grid) * np.pi/180 * u.rad
    dt = dt.to(u.day).value
    time_grid = np.arange(smjd, fmjd + dt/2, dt)

    vgrid_carr_recon_back = np.ones((nlon_grid, len(time_grid))) * np.nan
    vgrid_carr_recon_forward = np.ones((nlon_grid, len(time_grid))) * np.nan
    vgrid_carr_recon_both = np.ones((nlon_grid, len(time_grid))) * np.nan

    for t in range(0, len(time_grid)):
        # find nearest time and current Carrington longitude
        t_id = np.argmin(np.abs(omni_int['mjd'] - time_grid[t]))
        Elong = omni_int['Carr_lon'][t_id] * u.rad
        
        # get the Carrington longitude difference from current Earth pos
        dlong_back = _zerototwopi_(lon_grid.value - Elong.value) * u.rad
        dlong_forward = _zerototwopi_(Elong.value - lon_grid.value) * u.rad
        
        dt_back = (dlong_back / omega_synodic).to(u.day)
        dt_forward = (dlong_forward / omega_synodic).to(u.day)
        
        vgrid_carr_recon_back[:, t] = np.interp(time_grid[t] - dt_back.value, omni_int['mjd'], omni_int['V'],
                                                left=np.nan, right=np.nan)

        vgrid_carr_recon_forward[:, t] = np.interp(time_grid[t] + dt_forward.value, omni_int['mjd'], omni_int['V'],
                                                   left=np.nan, right=np.nan)

        numerator = (dt_forward * vgrid_carr_recon_back[:, t] + dt_back * vgrid_carr_recon_forward[:, t])
        denominator = dt_forward + dt_back
        vgrid_carr_recon_both[:, t] = numerator / denominator
        
    # cut out the requested time
    mask = ((time_grid >= Time(runstart).mjd) & (time_grid <= Time(runend).mjd))

    return time_grid[mask], vgrid_carr_recon_both[:, mask]
