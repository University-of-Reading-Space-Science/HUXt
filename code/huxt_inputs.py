import httplib2
import urllib
import huxt as H
import os
from pyhdf.SD import SD, SDC  
import numpy as np
import astropy.units as u
from scipy.io import netcdf


def get_MAS_boundary_conditions(cr=np.NaN, observatory='', runtype='', runnumber='', masres=''):
    """
    A function to grab the  Vr and Br boundary conditions from MHDweb. An order
    of preference for observatories is given in the function. Checks first if
    the data already exists in the HUXt boundary condition folder

    Parameters
    ----------
    cr : INT
        Carrington rotation number 
    observatory : STRING
        Name of preferred observatory (e.g., 'hmi','mdi','solis',
        'gong','mwo','wso','kpo'). Empty if no preference and automatically selected 
    runtype : STRING
        Name of preferred MAS run type (e.g., 'mas','mast','masp').
        Empty if no preference and automatically selected 
    runnumber : STRING
        Name of preferred MAS run number (e.g., '0101','0201').
        Empty if no preference and automatically selected    

    Returns
    -------
    flag : INT
        1 = successful download. 0 = files exist, -1 = no file found.

    """
    
    assert(np.isnan(cr) == False)
    
    # The order of preference for different MAS run results
    overwrite = False
    if not masres:
        masres_order = ['high','medium']
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
    
    vrfilename = 'HelioMAS_CR'+str(int(cr)) + '_vr'+heliomas_url_end
    brfilename = 'HelioMAS_CR'+str(int(cr)) + '_br'+heliomas_url_end
    
    if (os.path.exists(os.path.join(_boundary_dir_, brfilename)) == False or
        os.path.exists(os.path.join(_boundary_dir_, vrfilename)) == False or
        overwrite == True):  # Check if the files already exist

        # Search MHDweb for a HelioMAS run, in order of preference
        h = httplib2.Http()
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
        
        # Download teh vr and br files
        print('Downloading from: ', urlbase)
        urllib.request.urlretrieve(urlbase + 'br' + heliomas_url_end,
                                   os.path.join(_boundary_dir_, brfilename))
        urllib.request.urlretrieve(urlbase+'vr'+heliomas_url_end,
                                   os.path.join(_boundary_dir_, vrfilename))
        
        return 1
    else:
        print('Files already exist for CR' + str(int(cr)))
        return 0

    
def read_MAS_vr_br(cr):
    """
    A function to read in the MAS coundary conditions for a given CR

    Parameters
    ----------
    cr : INT
        Carrington rotation number

    Returns
    -------
    MAS_vr : NP ARRAY (NDIM = 2)
        Solar wind speed at 30rS, in km/s
    MAS_vr_Xa : NP ARRAY (NDIM = 1)
        Carrington longitude of Vr map, in rad
    MAS_vr_Xm : NP ARRAY (NDIM = 1)
        Latitude of Vr as angle down from N pole, in rad
    MAS_br : NP ARRAY (NDIM = 2)
        Radial magnetic field at 30rS, in model units
    MAS_br_Xa : NP ARRAY (NDIM = 1)
        Carrington longitude of Br map, in rad
    MAS_br_Xm : NP ARRAY (NDIM = 1)
       Latitude of Br as angle down from N pole, in rad

    """
    # Get the boundary condition directory
    dirs = H._setup_dirs_()
    _boundary_dir_ = dirs['boundary_conditions'] 
    # Create the filenames
    heliomas_url_end = '_r0.hdf'
    vrfilename = 'HelioMAS_CR'+str(int(cr)) + '_vr' + heliomas_url_end
    brfilename = 'HelioMAS_CR'+str(int(cr)) + '_br' + heliomas_url_end

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
    MAS_vr = MAS_vr*481.0 * u.km/u.s
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


def get_MAS_long_profile(cr, lat=0.0*u.deg):
    """
    a function to download, read and process MAS output to provide HUXt boundary
    conditions at a given latitude

    Parameters
    ----------
    cr : INT
        Carrington rotation number
    lat : FLOAT
        Latitude at which to extract the longitudinal profile, measure up from equator

    Returns
    -------
    vr_in : NP ARRAY (NDIM = 1)
        Solar wind speed as a function of Carrington longitude at solar equator.
        Interpolated to HUXt longitudinal resolution. In km/s
    br_in : NP ARRAY(NDIM = 1)
        Radial magnetic field as a function of Carrington longitude at solar equator.
        Interpolated to HUXt longitudinal resolution. Dimensionless

    """
    assert(np.isnan(cr) == False and cr > 0)
    assert(lat >= -90.0*u.deg)
    assert(lat <= 90.0*u.deg)
    
    # Convert angle from equator to angle down from N pole
    ang_from_N_pole = np.pi/2 - (lat.to(u.rad)).value
    
    # Check the data exist, if not, download them
    flag = get_MAS_boundary_conditions(cr)
    assert(flag > -1)
    
    # Read the HelioMAS data
    MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = read_MAS_vr_br(cr)
    
    # Extract the value at the given latitude
    vr = np.ones(len(MAS_vr_Xa))
    for i in range(0, len(MAS_vr_Xa)):
        vr[i] = np.interp(ang_from_N_pole, MAS_vr_Xm.value, MAS_vr[i][:].value)
    
    br = np.ones(len(MAS_br_Xa))
    for i in range(0, len(MAS_br_Xa)):
        br[i] = np.interp(ang_from_N_pole, MAS_br_Xm.value, MAS_br[i][:])
        
    # Now interpolate on to the HUXt longitudinal grid
    longs, dlon, nlon = H.longitude_grid(lon_start=0.0 * u.rad, lon_stop=2*np.pi * u.rad)
    vr_in = np.interp(longs.value, MAS_vr_Xa.value, vr)*u.km/u.s
    br_in = np.interp(longs.value, MAS_br_Xa.value, br)
    
    return vr_in


def get_MAS_maps(cr):
    """
    a function to download, read and process MAS output to provide HUXt boundary
    conditions as lat-long maps, along with angle from equator for the maps
    maps returned in native resolution, not HUXt resolution

    Parameters
    ----------
    cr : INT
        Carrington rotation number


    Returns
    -------
    vr_map : NP ARRAY 
        Solar wind speed as a Carrington longitude-latitude map. In km/s   
    vr_lats :
        The latitudes for the Vr map, in radians from trhe equator   
    vr_longs :
        The Carrington longitudes for the Vr map, in radians
    br_map : NP ARRAY
        Br as a Carrington longitude-latitude map. Dimensionless
    br_lats :
        The latitudes for the Br map, in radians from trhe equator
    br_longs :
        The Carrington longitudes for the Br map, in radians 
    """
    
    assert(np.isnan(cr) == False and cr > 0)
    
    # Check the data exist, if not, download them
    flag = get_MAS_boundary_conditions(cr)
    assert(flag > -1)
    
    # Read the HelioMAS data
    MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = read_MAS_vr_br(cr)
    
    vr_map = MAS_vr
    br_map = MAS_br
    
    # Convert the lat angles from N-pole to equator centred
    vr_lats = (np.pi/2)*u.rad - MAS_vr_Xm
    br_lats = (np.pi/2)*u.rad - MAS_br_Xm
    
    # Flip lats, so they're increasing in value
    vr_lats = np.flipud(vr_lats)
    br_lats = np.flipud(br_lats)
    vr_map = np.fliplr(vr_map)
    br_map = np.fliplr(br_map)
    
    vr_longs = MAS_vr_Xa
    br_longs = MAS_br_Xa

    return vr_map, vr_lats, vr_longs

def get_MAS_vrmap(cr):
    """
    a function to download, read and process MAS output to provide HUXt boundary
    conditions as lat-long maps, along with angle from equator for the maps
    maps returned in native resolution, not HUXt resolution
    
    THIS VERSION RETURNS A CORRECTlY TRANSPOSED MAP. IN FUTURE, 
    get_MAS_maps AND read_MAS_vr_br SHOULD BE UPDATED TO BEHAVE THE
    SAME

    Parameters
    ----------
    cr : INT
        Carrington rotation number


    Returns
    -------
    vr_map : NP ARRAY 
        Solar wind speed as a Carrington longitude-latitude map. In km/s   
    vr_lats :
        The latitudes for the Vr map, in radians from trhe equator   
    vr_longs :
        The Carrington longitudes for the Vr map, in radians

    """
    
    assert(np.isnan(cr) == False and cr > 0)
    
    # Check the data exist, if not, download them
    flag = get_MAS_boundary_conditions(cr)
    if flag < 0:
         return -1, -1, -1
    
    # Read the HelioMAS data
    MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = read_MAS_vr_br(cr)
    
    vr_map = MAS_vr
    
    # Convert the lat angles from N-pole to equator centred
    vr_lats = (np.pi/2)*u.rad - MAS_vr_Xm

    
    # Flip lats, so they're increasing in value
    vr_lats = np.flipud(vr_lats)
    vr_map = np.fliplr(vr_map)
    vr_longs = MAS_vr_Xa

    return vr_map.T, vr_lats, vr_longs

def get_MAS_brmap(cr):
    """
    a function to download, read and process MAS output to provide HUXt boundary
    conditions as lat-long maps, along with angle from equator for the maps
    maps returned in native resolution, not HUXt resolution
    
    THIS VERSION RETURNS A CORRECTlY TRANSPOSED MAP. IN FUTURE, 
    get_MAS_maps AND read_MAS_vr_br SHOULD BE UPDATED TO BEHAVE THE
    SAME

    Parameters
    ----------
    cr : INT
        Carrington rotation number


    Returns
    -------
    br_map : NP ARRAY 
        Solar wind speed as a Carrington longitude-latitude map. In km/s   
    br_lats :
        The latitudes for the Vr map, in radians from trhe equator   
    br_longs :
        The Carrington longitudes for the Vr map, in radians

    """
    
    assert(np.isnan(cr) == False and cr > 0)
    
    # Check the data exist, if not, download them
    flag = get_MAS_boundary_conditions(cr)
    if flag < 0:
         return -1, -1, -1
    
    # Read the HelioMAS data
    MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = read_MAS_vr_br(cr)
    
    br_map = MAS_br
    
    # Convert the lat angles from N-pole to equator centred
    br_lats = (np.pi/2)*u.rad - MAS_br_Xm

    
    # Flip lats, so they're increasing in value
    br_lats = np.flipud(br_lats)
    br_map = np.fliplr(br_map)
    br_longs = MAS_br_Xa

    return br_map.T, br_lats, br_longs

@u.quantity_input(v_outer=u.km / u.s)
@u.quantity_input(r_outer=u.solRad)
@u.quantity_input(lon_outer=u.rad)
@u.quantity_input(r_inner=u.solRad)
def map_v_inwards(v_outer, r_outer, lon_outer, r_inner):
    """
    Function to map v from r_outer (in rs) to r_inner (in rs) accounting for 
    residual acceleration, but neglecting stream interactions.
    
    :param v_outer: Solar wind speed at outer radial distance. Units of km/s.
    :param r_outer: Radial distance at outer radial distance. Units of km.  
    :param lon_outer: Carrington longitude at outer distance. Units of rad
    :param r_inner: Radial distance at inner radial distance. Units of km.
    :return v_inner: Solar wind speed mapped from r_outer to r_inner. Units of km/s.
    :return lon_inner: Carrington longitude at r_inner. Units of rad.
    """

    # Get the acceleration parameters
    constants = H.huxt_constants()
    alpha = constants['alpha']  # Scale parameter for residual SW acceleration
    rH = constants['r_accel'].to(u.kilometer).value  # Spatial scale parameter for residual SW acceleration
    Tsyn = constants['synodic_period'].to(u.s).value
    r_outer = r_outer.to(u.km).value
    r_inner = r_inner.to(u.km).value
    r_30 = 30*u.solRad
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

    return v0*u.km/u.s, phi_new*u.rad


@u.quantity_input(v_outer=u.km / u.s)
@u.quantity_input(r_outer=u.solRad)
@u.quantity_input(r_inner=u.solRad)
def map_v_boundary_inwards(v_outer, r_outer, r_inner):
    """
    Function to map a longitudinal V series from r_outer (in rs) to r_inner (in rs)
    accounting for residual acceleration, but neglecting stream interactions.
    Series return on HUXt longitudinal grid, not input grid
    
    :param v_outer: Solar wind speed at outer radial boundary. Units of km/s.
    :param r_outer: Radial distance at outer radial boundary. Units of km.
    :param r_inner: Radial distance at inner radial boundary. Units of km.
    :return v_inner: Solar wind speed mapped from r_outer to r_inner. Units of km/s.
    """

    if r_outer < r_inner:
        raise ValueError("Warning: r_outer < r_inner. Mapping will not work.")

    # Compute the longitude grid from the length of the vouter input variable
    lon, dlon, nlon = H.longitude_grid()  
    
    # Map each point in to a new speed and longitude
    v0, phis_new = map_v_inwards(v_outer, r_outer, lon, r_inner)

    # Interpolate the mapped speeds back onto the regular Carr long grid,
    # making boundaries periodic
    v_inner = np.interp(lon, phis_new, v0, period=2*np.pi) 

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
    Map returned on input coord system, not HUXT resolution
    :param v_map: Solar wind speed Carrington map at outer radial boundary. Units of km/s.
    :param v_map_lat: Latitude (from equator) of v_map positions. Units of radians
    :param v_map_long: Carrington longitude of v_map positions. Units of radians
    :param r_outer: Radial distance at outer radial boundary. Units of km.
    :param r_inner: Radial distance at inner radial boundary. Units of km.
    :return v_map_inner: Solar wind speed map at r_inner. Units of km/s.
    """
    #updated to use correctly inverted maps

    if r_outer < r_inner:
        raise ValueError("Warning: r_outer < r_inner. Mapping will not work.")

    # Check the dimensions
    assert(len(v_map_lat) == len(v_map[:, 1]))
    assert(len(v_map_long) == len(v_map[1, :]))

    v_map_inner = np.ones((len(v_map_lat), len(v_map_long)))
    for ilat in range(0, len(v_map_lat)):
        # Map each point in to a new speed and longitude
        v0, phis_new = map_v_inwards(v_map[ilat, :], r_outer, v_map_long, r_inner)

        # Interpolate the mapped speeds back onto the regular Carr long grid,
        # making boundaries periodic * u.km/u.s
        v_map_inner[ilat, :] = np.interp(v_map_long.value, phis_new.value, v0.value, period=2*np.pi)

    return v_map_inner * u.km / u.s


def get_PFSS_maps(filepath):
    """
    a function to load, read and process PFSSpy output to provide HUXt boundary
    conditions as lat-long maps, along with angle from equator for the maps
    maps returned in native resolution, not HUXt resolution

    Parameters
    ----------
    filepath : STR 
        The filepath for the PFSSpy .nc file

    Returns
    -------
    vr_map : NP ARRAY 
        Solar wind speed as a Carrington longitude-latitude map. In km/s   
    vr_lats :
        The latitudes for the Vr map, in radians from trhe equator   
    vr_longs :
        The Carrington longitudes for the Vr map, in radians
    br_map : NP ARRAY
        Br as a Carrington longitude-latitude map. Dimensionless
    br_lats :
        The latitudes for the Br map, in radians from trhe equator
    br_longs :
        The Carrington longitudes for the Br map, in radians 

    """
    
    assert os.path.exists(filepath)
    #nc = netcdf.netcdf_file(filepath, 'r')
    
    nc = netcdf.netcdf_file(filepath,'r',mmap=False)
    br_map=nc.variables['br'][:]
    vr_map=nc.variables['vr'][:]* u.km / u.s
    phi=nc.variables['ph'][:]
    cotheta=nc.variables['cos(th)'][:]
    
    nc.close()
    
    phi = phi *u.rad
    theta = (np.pi/2 - np.arccos(cotheta) ) *u.rad
    vr_lats = theta[:, 0]
    br_lats = vr_lats
    vr_longs = phi[0, :] 
    br_longs = vr_longs
    
    
#    #theta is angle from north pole. convert to angle from equator
#    cotheta = nc.variables['cos(th)'].data
#    vr_lats = (np.pi/2 - np.arccos(cotheta[:, 0]) )*u.rad
#    br_lats = vr_lats
#    
#    phi = nc.variables['ph'].data
#    vr_longs = phi[0, :] * u.rad
#    br_longs = vr_longs
#    
#    br_map = np.rot90(nc.variables['br'].data)
#    vr_map = np.rot90(nc.variables['vr'].data) * u.km / u.s

    return vr_map, vr_lats, vr_longs, br_map, br_lats, br_longs, phi, theta
