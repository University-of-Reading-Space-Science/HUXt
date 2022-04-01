import httplib2
import urllib
import huxt as H
import os
import ssl
from pyhdf.SD import SD, SDC
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from scipy.io import netcdf
from scipy import interpolate
from sunpy.coordinates import sun


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

    return vr_map.T, vr_lats, vr_longs


@u.quantity_input(v_outer=u.km / u.s)
@u.quantity_input(r_outer=u.solRad)
@u.quantity_input(lon_outer=u.rad)
@u.quantity_input(r_inner=u.solRad)
def map_v_inwards(v_outer, r_outer, lon_outer, r_inner):
    """
    Function to map v from r_outer (in rs) to r_inner (in rs) accounting for 
    residual acceleration, but neglecting stream interactions.

    Args:
        v_outer: Solar wind speed at outer radial distance. Units of km/s.
        r_outer: Radial distance at outer radial distance. Units of km.
        lon_outer: Carrington longitude at outer distance. Units of rad
        r_inner: Radial distance at inner radial distance. Units of km.

    Returns:
        v_inner: Solar wind speed mapped from r_outer to r_inner. Units of km/s.
        lon_inner: Carrington longitude at r_inner. Units of rad.
    """

    # Get the acceleration parameters
    constants = H.huxt_constants()
    alpha = constants['alpha']  # Scale parameter for residual SW acceleration
    rH = constants['r_accel'].to(u.kilometer).value  # Spatial scale parameter for residual SW acceleration
    Tsyn = constants['synodic_period'].to(u.s).value
    r_outer = r_outer.to(u.km).value
    r_inner = r_inner.to(u.km).value
    r_30 = 30 * u.solRad
    r_30 = r_30.to(u.km).value

    # Compute the 30 rS speed
    v30 = v_outer.value * (1 + alpha * (1 - np.exp((r_30 - r_outer) / rH)))

    # Compute the speed at the new inner boundary height (using Vacc term, equation 5 in the paper)
    v0 = v30 * (1 + alpha * (1 - np.exp((r_30 - r_inner) / rH)))

    # Compute the transit time from the new to old inner boundary heights (i.e., integrate equations 3 and 4 wrt to r)
    A = v0 + alpha * v0
    term1 = rH * np.log(A * np.exp(r_outer / rH) - alpha * v0 * np.exp(r_inner / rH)) / A
    term2 = rH * np.log(A * np.exp(r_inner / rH) - alpha * v0 * np.exp(r_inner / rH)) / A
    T_integral = term1 - term2

    # Work out the longitudinal shift
    phi_new = H._zerototwopi_(lon_outer.value + (T_integral / Tsyn) * 2 * np.pi)

    return v0 * u.km / u.s, phi_new * u.rad


@u.quantity_input(v_outer=u.km / u.s)
@u.quantity_input(r_outer=u.solRad)
@u.quantity_input(r_inner=u.solRad)
def map_v_boundary_inwards(v_outer, r_outer, r_inner):
    """
    Function to map a longitudinal V series from r_outer (in rs) to r_inner (in rs)
    accounting for residual acceleration, but neglecting stream interactions.
    Series return on HUXt longitudinal grid, not input grid

    Args:
        v_outer: Solar wind speed at outer radial boundary. Units of km/s.
        r_outer: Radial distance at outer radial boundary. Units of km.
        r_inner: Radial distance at inner radial boundary. Units of km.

    Returns:
        v_inner: Solar wind speed mapped from r_outer to r_inner. Units of km/s.
    """

    if r_outer < r_inner:
        raise ValueError("Warning: r_outer < r_inner. Mapping will not work.")

    # Compute the longitude grid from the length of the vouter input variable
    lon, dlon, nlon = H.longitude_grid()

    # Map each point in to a new speed and longitude
    v0, phis_new = map_v_inwards(v_outer, r_outer, lon, r_inner)

    # Interpolate the mapped speeds back onto the regular Carr long grid,
    # making boundaries periodic
    v_inner = np.interp(lon, phis_new, v0, period=2 * np.pi)

    return v_inner


@u.quantity_input(v_map=u.km / u.s)
@u.quantity_input(v_map_lat=u.rad)
@u.quantity_input(v_map_long=u.rad)
@u.quantity_input(r_outer=u.solRad)
@u.quantity_input(r_inner=u.solRad)
def map_vmap_inwards(v_map, v_map_lat, v_map_long, r_outer, r_inner):
    """
    Function to map a V Carrington map from r_outer (in rs) to r_inner (in rs),
    accounting for acceleration, but ignoring stream interaction
    Map returned on input coord system, not HUXT resolution.

    Args:
        v_map: Solar wind speed Carrington map at outer radial boundary. np.array with units of km/s.
        v_map_lat: Latitude (from equator) of v_map positions. np.array with units of radians
        v_map_long: Carrington longitude of v_map positions. np.array with units of radians
        r_outer: Radial distance at outer radial boundary. np.array with units of km.
        r_inner: Radial distance at inner radial boundary. np.array with units of km.

    Returns:
        v_map_inner: Solar wind speed map at r_inner. np.array with units of km/s.
    """

    if r_outer < r_inner:
        raise ValueError("Warning: r_outer < r_inner. Mapping will not work.")

    # Check the dimensions
    assert (len(v_map_lat) == len(v_map[:, 1]))
    assert (len(v_map_long) == len(v_map[1, :]))

    v_map_inner = np.ones((len(v_map_lat), len(v_map_long)))
    for ilat in range(0, len(v_map_lat)):
        # Map each point in to a new speed and longitude
        v0, phis_new = map_v_inwards(v_map[ilat, :], r_outer, v_map_long, r_inner)

        # Interpolate the mapped speeds back onto the regular Carr long grid,
        # making boundaries periodic * u.km/u.s
        v_map_inner[ilat, :] = np.interp(v_map_long.value, phis_new.value, v0.value, period=2 * np.pi)

    return v_map_inner * u.km / u.s


def get_PFSS_maps(filepath):
    """
    A function to load, read and process PFSSpy output to provide HUXt boundary
    conditions as lat-long maps, along with angle from equator for the maps.
    Maps returned in native resolution, not HUXt resolution.

    Args:
        filepath: String, The filepath for the PFSSpy .nc file

    Returns:
        vr_map: np.array, Solar wind speed as a Carrington longitude-latitude map. In km/s
        vr_lats: np.array, The latitudes for the Vr map, in radians from trhe equator
        vr_longs: np.array, The Carrington longitudes for the Vr map, in radians
        br_map:  np.array, Br as a Carrington longitude-latitude map. Dimensionless
        br_lats: np.array, The latitudes for the Br map, in radians from trhe equator
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

    return vr_map, vr_lats, vr_longs, br_map, br_lats, br_longs, phi, theta


def get_WSA_maps(filepath):
    """
    A function to load, read and process WSA FITS maps from the UK Met Office
    to provide HUXt boundary conditions as lat-long maps, along with angle from
    equator for the maps.
    Maps returned in native resolution, not HUXt resolution.

    Args:
        filepath: String, The filepath for the PFSSpy .nc file

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

    br_long_edges = np.arange(0, -2 * np.pi + 0.00001, dgrid)
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

    # create the mesh grid
    phi = np.empty(vr_map_fits.shape)
    theta = np.empty(vr_map_fits.shape)
    for nlat in range(0, len(vr_lat_centres)):
        theta[nlat, :] = vr_lats[nlat]
        phi[nlat, :] = vr_longs
    phi = phi * u.rad
    theta = theta * u.rad

    return vr_map, vr_lats, vr_longs, br_map, br_lats, br_longs, phi, theta, cr_num


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
    to input Cone CMEs into Enlil
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

        # Thickness must be computed from CME cone initial radius and the xcld parameter,
        # which specifies the relative elongation of the cloud, 1=spherical,
        # 2=middle twice as long as cone radius e.g.
        # compute initial radius of the cone
        radius = np.abs(model.r[0] * np.tan(wid / 2.0))  # eqn on line 162 in ConeCME class
        # Thickness determined from xcld and radius
        thick = (1.0 - cme_val['xcld']) * radius

        cme = H.ConeCME(t_launch=dt_cme, longitude=lon, latitude=lat, width=wid, v=speed, thickness=thick)
        cme_list.append(cme)

    return cme_list


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
