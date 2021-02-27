import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
from sunpy.coordinates import sun
import os
import glob
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
from skimage import measure
import scipy.ndimage as ndi
from numba import jit
import copy

import HUXtinput as Hin
import HUXtlat as Hlat

from packaging import version
#check the numpy version, as this can cause all manner of difficult-to-diagnose problems
assert(version.parse(np.version.version) >= version.parse("1.18"))


class Observer:
    """
    A class returning the HEEQ and Carrington coordinates of a specified Planet or spacecraft, for a given set of times.
    The positions are linearly interpolated from a 2-hour resolution ephemeris that spans 1974-01-01 until 2020-01-01.
    Allowed bodies are Earth, Venus, Mercury, STEREO-A and STEREO-B.
    Attributes:
        body: String name of the planet or spacecraft.
        lat: HEEQ latitude of body at all values of time.
        lat_c: Carrington latitude of body at all values of time.
        lon: HEEQ longitude of body at all values of time.
        lon_c: Carrington longitude of body at all values of time.
        r: HEEQ radius of body at all values of time.
        r_c: Carrington radius of body at all values of time.
        time: Array of Astropy Times
    """

    def __init__(self, body, times):
        """
        :param body: String indicating which body to look up the positions of .
        :param times: A list/array of Astropy Times to interpolate the coordinate of the selected body.
        """
        bodies = ["EARTH", "VENUS", "MERCURY", "STA", "STB"]
        if body.upper() in bodies:
            self.body = body.upper()
        else:
            print("Warning, body {} not recognised.".format(body))
            print("Only {} are valid.".format(bodies))
            print("Defaulting to Earth")
            self.body = "EARTH"

        # Get path to ephemeris file and open
        dirs = _setup_dirs_()
        ephem = h5py.File(dirs['ephemeris'], 'r')

        # Now get observers coordinates
        all_time = Time(ephem[self.body]['HEEQ']['time'], format='jd')
        # Pad out the window to account for single values being passed. 
        dt = TimeDelta(2 * 60 * 60, format='sec')
        id_epoch = (all_time >= (times.min() - dt)) & (all_time <= (times.max() + dt))
        epoch_time = all_time[id_epoch]
        
        self.time = times
        if len(epoch_time.jd) ==0:
            self.r = np.ones(len(self.time)) * np.nan 
            self.lon = np.ones(len(self.time)) * np.nan 
            self.lat = np.ones(len(self.time)) * np.nan 
            
            self.r_hae = np.ones(len(self.time)) * np.nan 
            self.lon_hae = np.ones(len(self.time)) * np.nan 
            self.lat_hae = np.ones(len(self.time)) * np.nan 
            
            self.r_c = np.ones(len(self.time)) * np.nan 
            self.lon_c = np.ones(len(self.time)) * np.nan 
            self.lat_c = np.ones(len(self.time)) * np.nan 
            
            
        else:
            r = ephem[self.body]['HEEQ']['radius'][id_epoch]
            self.r = np.interp(times.jd, epoch_time.jd, r)
            self.r = (self.r * u.km).to(u.solRad)

            lon = np.deg2rad(ephem[self.body]['HEEQ']['longitude'][id_epoch])
            lon = np.unwrap(lon)
            self.lon = np.interp(times.jd, epoch_time.jd, lon)
            self.lon = _zerototwopi_(self.lon)
            self.lon = self.lon * u.rad

            lat = np.deg2rad(ephem[self.body]['HEEQ']['latitude'][id_epoch])
            self.lat = np.interp(times.jd, epoch_time.jd, lat)
            self.lat = self.lat * u.rad
      
        
            r = ephem[self.body]['HAE']['radius'][id_epoch]
            self.r_hae = np.interp(times.jd, epoch_time.jd, r)
            self.r_hae = (self.r_hae * u.km).to(u.solRad)
    
            lon = np.deg2rad(ephem[self.body]['HAE']['longitude'][id_epoch])
            lon = np.unwrap(lon)
            self.lon_hae = np.interp(times.jd, epoch_time.jd, lon)
            self.lon_hae = _zerototwopi_(self.lon_hae)
            self.lon_hae = self.lon_hae * u.rad
    
            lat = np.deg2rad(ephem[self.body]['HAE']['latitude'][id_epoch])
            self.lat_hae = np.interp(times.jd, epoch_time.jd, lat)
            self.lat_hae = self.lat_hae * u.rad
            
    
            r = ephem[self.body]['CARR']['radius'][id_epoch]
            self.r_c = np.interp(times.jd, epoch_time.jd, r)
            self.r_c = (self.r_c * u.km).to(u.solRad)
    
            lon = np.deg2rad(ephem[self.body]['CARR']['longitude'][id_epoch])
            lon = np.unwrap(lon)
            self.lon_c = np.interp(times.jd, epoch_time.jd, lon)
            self.lon_c = _zerototwopi_(self.lon_c)
            self.lon_c = self.lon_c * u.rad
    
            lat = np.deg2rad(ephem[self.body]['CARR']['latitude'][id_epoch])
            self.lat_c = np.interp(times.jd, epoch_time.jd, lat)
            self.lat_c = self.lat_c * u.rad

        ephem.close()
        return


class ConeCME:
    """
    A class containing the parameters of a cone model cme.
    Attributes:
        t_launch: Time of Cone CME launch, in seconds after the start of the simulation.
        longitude: HEEQ Longitude of the CME launch direction, in radians.
        v: CME nose speed in km/s.
        width: Angular width of the CME, in radians.
        initial_height: Initiation height of the CME, in km. Defaults to HUXt inner boundary at 30 solar radii.
        radius: Initial radius of the CME, in km.
        thickness: Thickness of the CME cone, in km.
        coords: Dictionary containing the radial and longitudinal (for HUXT2D) coordinates of the of Cone CME for each
                model time step.
    """

    # Some decorators for checking the units of input arguments
    @u.quantity_input(t_launch=u.s)
    @u.quantity_input(longitude=u.deg)
    @u.quantity_input(v=(u.km / u.s))
    @u.quantity_input(width=u.deg)
    @u.quantity_input(thickness=u.solRad)
    def __init__(self, t_launch=0.0 * u.s, longitude=0.0 * u.deg, latitude=0.0 * u.deg, v=1000.0 * (u.km / u.s),
                 width=30.0 * u.deg,
                 thickness=5.0 * u.solRad):
        """
        Set up a Cone CME with specified parameters.
        :param t_launch: Time of Cone CME launch, in seconds after the start of the simulation.
        :param longitude: HEEQ Longitude of the CME launch direction, in radians.
        :param latitude: HEEQ latitude of the CME launch direction, in radians.
        :param v: CME nose speed in km/s.
        :param width: Angular width of the CME, in degrees.
        :param thickness: Thickness of the CME cone, in solar radii
        """
        self.t_launch = t_launch  # Time of CME launch, after the start of the simulation
        lon = _zerototwopi_(longitude.to(u.rad).value) * u.rad
        self.longitude = lon  # Longitudinal launch direction of the CME
        self.latitude = latitude.to(u.rad)  # Latitude launch direction of the CME
        self.v = v  # CME nose speed
        self.width = width  # Angular width
        self.initial_height = 30.0 * u.solRad  # Initial height of CME (should match inner boundary of HUXt)
        self.radius = self.initial_height * np.tan(self.width / 2.0)  # Initial radius of CME
        self.thickness = thickness  # Extra CME thickness
        self.coords = {}
        return

    def parameter_array(self):
        """
        Returns a numpy array of CME parameters. This is used in the numba optimised solvers that don't play nicely
        with classes.
        """
        cme_parameters = [self.t_launch.to('s').value, self.longitude.to('rad').value, self.latitude.to('rad').value,
                          self.width.to('rad').value, self.v.value, self.initial_height.to('km').value,
                          self.radius.to('km').value, self.thickness.to('km').value]
        return cme_parameters

        
    def _track_(self, model, cme_id):
        """
        Tracks the perimeter of each ConeCME through the HUXt solution in model.
        :param model: An HUXt instance, solving for multiple longitudes, with solutions for the CME and ambient fields.
        :return: updates the ConeCME.coords dictionary of CME coordinates.
        """
        
        #  Pull out the particle field for this CME
        cme_field = model.cme_particles[cme_id, :, :, :]
        
        #  Setup dictionary to track this CME
        self.coords = {j: {'time':np.array([]), 'model_time':np.array([]) * u.s,
              'front_id': np.array([]) * u.dimensionless_unscaled,
              'lon': np.array([]) * model.lon.unit, 'r': np.array([]) * model.r.unit,
              'lat': np.array([]) * model.latitude.unit} for j in range(model.nt_out)}
        
        #  Loop through timesteps, save out coords to coords dict
        for j, t in enumerate(model.time_out):

            self.coords[j]['model_time'] =  t
            self.coords[j]['time'] = model.time_init + t

            cme_r_front = cme_field[j, 0, :]
            cme_r_back = cme_field[j, 1, :]

            if np.any(np.isfinite(cme_r_front)) | np.any(np.isfinite(cme_r_back)):       
                #  Get longitudes and center on CME
                lon = model.lon - self.longitude
                lon[lon>np.pi*u.rad] -= 2*np.pi*u.rad
                #  Find indices that sort the longitudes, to make a wraparound of lons
                id_sort_inc = np.argsort(lon)
                id_sort_dec = np.flipud(id_sort_inc)
                
                #  Get one array of longitudes and radii from the front and back particles
                lons = np.hstack([lon[id_sort_inc], lon[id_sort_dec]])
                cme_r = np.hstack([cme_r_front[id_sort_inc], cme_r_back[id_sort_dec]])
                front_id = np.hstack([np.ones(cme_r_front.shape), np.zeros(cme_r_back.shape)])
                
                # Find valid points and save to dict
                id_good = np.isfinite(cme_r)
                self.coords[j]['r'] = cme_r[id_good] * u.km
                self.coords[j]['lon'] = lons[id_good] + self.longitude
                self.coords[j]['front_id'] = front_id[id_good]*u.dimensionless_unscaled
                self.coords[j]['lat'] = model.latitude.copy()
        return
    
    
    def compute_earth_arrival(self):
        """
        Function to compute arrival time at a set longitude and radius of the CME front.

        Tracks radial distance of front along a given longitude out past specified longitude.
        Then interpolates the r-t profile to find t_arr at arr_rad. 
        """

        times = Time([coord['time'] for i, coord in self.coords.items()])
        ert = Observer('EARTH', times)
        
        # Need to force units to be the same to make interpolations work 
        arr_lon = 0*u.rad
        arr_rad = np.mean(ert.r)
        
        # Check if hit or miss.
        # Put longitude between -180 - 180, centered on CME lon.
        lon_diff = arr_lon - self.longitude
        if lon_diff < -180*u.deg:
            lon_diff += 360*u.deg
        elif lon_diff > 180*u.deg:
            lon_diff -= 360*u.deg
            
        cme_hw = self.width/2.0
        if (lon_diff >= -cme_hw) & (lon_diff <= cme_hw):
            # HIT, so get t-r profile along lon of interest.
            t_front = []
            r_front = []

            for i, coord in self.coords.items():

                if len(coord['r'])==0:
                    continue

                t_front.append(coord['model_time'].to('d').value)

                # Lookup radial coord at earth lon
                r = coord['r'].value
                lon = coord['lon'].value

                # Only keep front of cme
                id_front = r > np.mean(r)
                r = r[id_front]
                lon = lon[id_front]

                r_ans = np.interp(arr_lon.value, lon, r, period=2*np.pi)
                r_front.append(r_ans)
                # Stop when max r 
                if r_ans > arr_rad.value:
                    break

            t_front = np.array(t_front)
            r_front = np.array(r_front)
            try:
                t_transit = np.interp(arr_rad.value, r_front, t_front)
                self.earth_transit_time = t_transit * u.d
                self.earth_arrival_time = times[0] + self.earth_transit_time
            except:
                self.earth_transit_time = np.NaN*u.d
                self.earth_arrival_time = Time('0000-01-01T00:00:00')
        else:
            self.earth_transit_time = np.NaN*u.d
            self.earth_arrival_time = Time('0000-01-01T00:00:00')

        return        
        
class HUXt:
    """
    A class containing the HUXt model described in Owens et al. (2020, DOI: 10.1007/s11207-020-01605-3)

    Users must specify the solar wind speed boundary condition through either the v_boundary, or cr_num keyword
    arguments. Failure to do so defaults to a 400 km/s boundary. v_boundary takes precedence over cr_num, so specifying
    both results in only v_boundary being used.
    
    Model coordinate system is HEEQ radius and longitude.
    
    Attributes:
        cmes: A list of ConeCME instances used in the model solution.
        cr_num: If provided, this gives the Carrington rotation number of the selected period, else 9999.
        cr_lon_init: The initial Carrington longitude of Earth at the models initial timestep.
        daysec: seconds in a day.
        dlon: Longitudinal grid spacing (in radians)
        dr: Radial grid spacing (in km).
        dt: Model time step (in seconds), set by the CFL condition with v_max and dr.
        dt_out: Output model time step (in seconds).
        dt_scale: Integer scaling number to set the model output time step relative to the models CFL time step.
        dtdr: Ratio of the model time step and radial grid step (in seconds/km).
        frame : either synodic or sidereal
        kms: astropy.unit instance of km/s.       
        lon: Array of model longtidues (in radians).
        r_grid: Array of longitudinal coordinates meshed with the radial coordinates (in radians).
        nlon: Number of longitudinal grid points.
        nr: Number of radial grid points.
        Nt: Total number of model time steps, including spin up.
        nt_out: Number of output model time steps.
        r_accel: Scale parameter determining the residual solar wind acceleration.
        r: Radial grid (in km).
        r_grid: Array of radial coordinates meshed with the longitudinal coordinates (in km).
        rrel: Radial grid relative to first grid point (in km).
        rotation_period:  rotation period (in seconds), either synodic or sidereal
        simtime: Simulation time (in seconds).
        time: Array of model time steps, including spin up (in seconds).
        time_init: The UTC time corresonding to the initial Carrington rotation number and longitude. Else, NaN. 
        time_out: Array of output model time steps (in seconds).
        twopi: two pi radians
        v_boundary: Inner boundary solar wind speed profile (in km/s).
        v_max: Maximum model speed (in km/s), used with the CFL condition to set the model time step. 
        
        v_grid: Array of model speed inlcuding ConeCMEs for each time, radius, and longitude (in km/s).
        br_grid: Array of model Br inlcuding ConeCMEs for each time, radius, and longitude (dimensionless).
        
    """

    # Decorators to check units on input arguments
    @u.quantity_input(v_boundary=(u.km / u.s))
    @u.quantity_input(simtime=u.day)
    @u.quantity_input(cr_lon_init=u.deg)
    def __init__(self, 
                 v_boundary = np.NaN * (u.km / u.s),  
                 br_boundary = np.NaN * u.dimensionless_unscaled,
                 rho_boundary = np.NAN *u.dimensionless_unscaled,
                 cr_num=np.NaN, cr_lon_init=360.0 * u.deg, latitude = 0*u.deg,
                 r_min=30 * u.solRad, r_max=240 * u.solRad,
                 lon_out=np.NaN * u.rad, lon_start=np.NaN * u.rad, lon_stop=np.NaN * u.rad,
                 simtime=5.0 * u.day, dt_scale=1.0, frame='synodic'):
        """
        Initialise the HUXt model instance.

        :param v_boundary: Inner solar wind speed boundary condition. Must be an array of size 128 with units of km/s.
        :param br_boundary: Inner passive tracer boundary condition used for Br. Must be an array of size 128 with no units
        :param rho_boundary: Inner passive tracer boundary condition used for density. Must be an array of size 128 with no units
        :param cr_num: Integer Carrington rotation number. Used to determine the planetary and spacecraft positions
        :param cr_lon_init: Carrington longitude of Earth at model initialisation, in degrees.
        :param latitude: Helio latitude (from equator) of HUXt plane, in degrees
        :param lon_out: A specific single longitude (relative to Earth_ to compute HUXt solution along, in degrees
        :param lon_start: The first longitude (in a clockwise sense) of the longitude range to solve HUXt over.
        :param lon_stop: The last longitude (in a clockwise sense) of the longitude range to solve HUXt over.
        :param r_min: The radial inner boundary distance of HUXt.
        :param r_max: The radial outer boundary distance of HUXt.
        :param simtime: Duration of the simulation window, in days.
        :param dt_scale: Integer scaling number to set the model output time step relative to the models CFL time.
        :param frame: string determining the rotation frame for the model
        """

        # some constants and units
        constants = huxt_constants()
        self.twopi = constants['twopi']
        self.daysec = constants['daysec']
        self.kms = constants['kms']
        self.alpha = constants['alpha']  # Scale parameter for residual SW acceleration
        self.r_accel = constants['r_accel']  # Spatial scale parameter for residual SW acceleration
        
        #set the frame fo reference. Synodic keeps ES line at 0 longitude. 
        #sidereal means Earth moves to increasing longitude with time
        assert(frame == 'synodic' or frame == 'sidereal')
        self.frame = frame
        if (frame == 'synodic'):
            self.rotation_period = constants['synodic_period']  # Solar Synodic rotation period from Earth.
        elif (frame == 'sidereal'):
            self.rotation_period = constants['sidereal_period'] 

            
        self.v_max = constants['v_max']
        self.nlong = constants['nlong']
        del constants

        # Extract paths of figure and data directories
        dirs = _setup_dirs_()
        self._boundary_dir_ = dirs['boundary_conditions']
        self._data_dir_ = dirs['HUXt_data']
        self._figure_dir_ = dirs['HUXt_figures']
        self._ephemeris_file = dirs['ephemeris']

        # Setup radial coordinates - in solar radius
        self.r, self.dr, self.rrel, self.nr = radial_grid(r_min=r_min, r_max=r_max)
        self.buffertime = ((8.0 * u.day) / (210 * u.solRad)) * self.rrel[-1]

        # Setup longitude coordinates - in radians.
        self.lon, self.dlon, self.nlon = longitude_grid(lon_out=lon_out, lon_start=lon_start, lon_stop=lon_stop)
        
        #set up the latitude
        self.latitude=latitude.to(u.rad)

        # Setup time coords - in seconds
        self.simtime = simtime.to('s')  # number of days to simulate (in seconds)
        self.dt_scale = dt_scale * u.dimensionless_unscaled
        time_grid_dict = time_grid(self.simtime, self.dt_scale)
        self.dtdr = time_grid_dict['dtdr']
        self.Nt = time_grid_dict['Nt']
        self.dt = time_grid_dict['dt']
        self.time = time_grid_dict['time']
        self.nt_out = time_grid_dict['nt_out']
        self.dt_out = time_grid_dict['dt_out']
        self.time_out = time_grid_dict['time_out']
        del time_grid_dict
         
        # Establish the speed boundary condition
        if np.all(np.isnan(v_boundary)):
            print("Warning: No V boundary conditions supplied. Using default")
            self.v_boundary = 400 * np.ones(self.nlong) * self.kms
        elif not np.all(np.isnan(v_boundary)):
            assert v_boundary.size == self.nlong
            self.v_boundary = v_boundary
        # Keep a protected version that isn't processed for use in saving/loading model runs
        self._v_boundary_init_ = self.v_boundary.copy()
        
        # Determine CR number, used for spacecraft/planetary positions
        if np.isnan(cr_num):
            print('No initiation time specified. Defaulting to 1977-9-27')
            self.cr_num = 1659 * u.dimensionless_unscaled
            cr_lon_init = 0.8*u.rad
        else:
            self.cr_num = cr_num * u.dimensionless_unscaled 
            
        # Check cr_lon_init, make sure in 0-2pi range.
        self.cr_lon_init = cr_lon_init.to('rad')
        if (self.cr_lon_init < 0.0 * u.rad) | (self.cr_lon_init > self.twopi * u.rad):
            print("Warning: cr_lon_init={}, outside expected range. Rectifying to 0-2pi.".format(self.cr_lon_init))
            self.cr_lon_init = _zerototwopi_(self.cr_lon_init.value) * u.rad 
                     
        # Compute model UTC initalisation time
        cr_frac = self.cr_num.value + ((self.twopi - self.cr_lon_init.value) / self.twopi)
        self.time_init = sun.carrington_rotation_time(cr_frac)
  
        # Rotate the boundary condition as required by cr_lon_init.
        lon_boundary, dlon, nlon = longitude_grid()
        lon_shifted = _zerototwopi_((lon_boundary - self.cr_lon_init).value)
        id_sort = np.argsort(lon_shifted)
        lon_shifted = lon_shifted[id_sort]
        
        v_b_shifted = self.v_boundary[id_sort]
        self.v_boundary = np.interp(lon_boundary.value, lon_shifted, v_b_shifted, period=self.twopi)
        # Preallocate space for the output for the solar wind fields for the cme and ambient solution.
        self.v_grid = np.zeros((self.nt_out, self.nr, self.nlon)) * self.kms
        
        # Mesh the spatial coordinates.
        self.lon_grid, self.r_grid = np.meshgrid(self.lon, self.r)

        # Empty dictionary for storing the coordinates of CME boundaries.
        self.cmes = []

        # Numpy array of model parameters for parsing to external functions that use numba
        self.model_params = np.array([self.dtdr.value, self.alpha.value, self.r_accel.value,
                                      self.dt_scale.value, self.nt_out, self.nr, self.nlon,
                                      self.r[0].to('km').value])
        return

    
    def solve(self, cme_list, save=False, tag=''):
        """
        Solve HUXt for the provided boundary conditions and cme list

        :param cme_list: A list of ConeCME instances to use in solving HUXt
        :param save: Boolean, if True saves model output to HDF5 file
        :param tag: String, appended to the filename of saved soltuion.

        Returns:

        """
        #make a copy of the CME list objects so that the originals are not modified
        mycme_list = copy.deepcopy(cme_list)

        # Check only cone cmes in cme list, adjust for sidereal frame if necessary
        cme_list_checked = []
        for cme in mycme_list:
            if isinstance(cme, ConeCME):
                if self.frame == 'sidereal':
                    #if the solution is in the sideral frame, adjust CME longitudes
                    print('Adjusting CME HEEQ longitude for sidereal frame')
                    Earthpos = self.get_observer('EARTH')
                    #time and longitude from start of run
                    dt_t0 = (Earthpos.time - self.time_init).to(u.s)
                    dlon_t0 = Earthpos.lon_hae -  Earthpos.lon_hae[0]
                    #find the CME hae longitude relative to the run start
                    cme_hae = np.interp(cme.t_launch.value, dt_t0.value, dlon_t0)
                    #adjust the CME HEEQ longitude accordingly
                    cme.longitude = _zerototwopi_(cme.longitude  + cme_hae) * u.rad
                #add the CME to the list  
                cme_list_checked.append(cme)

            else:
                print("Warning: cme_list contained objects other than ConeCME instances. These will be excluded")

        self.cmes = cme_list_checked

        # If CMEs parsed, get an array of their parameters for using with the solver (which doesn't do classes)
        if len(self.cmes) > 0:
            cme_params = [cme.parameter_array() for cme in self.cmes]
            cme_params = np.array(cme_params)
            # Sort the CMEs in launch order.
            id_sort = np.argsort(cme_params[:, 0])
            cme_params = cme_params[id_sort]
            # Also sort the list of ConeCMEs so it corresponds ot cme_params
            self.cmes = [self.cmes[i] for i in id_sort]
            do_cme = True
        else:
            do_cme = False
            cme_params = np.NaN * np.zeros((1, 9))
            
        #set up the test particle position field
        self.cme_particles = np.zeros((len(self.cmes), self.nt_out, 2, self.nlon)) * u.dimensionless_unscaled

        buffersteps = np.fix(self.buffertime.to(u.s) / self.dt)
        buffertime = buffersteps * self.dt
        model_time = np.arange(-buffertime.value, (self.simtime.to('s') + self.dt).value, self.dt.value) * self.dt.unit
        dlondt = self.twopi * self.dt / self.rotation_period
        all_lons, dlon, nlon = longitude_grid()

        # How many radians of Carrington rotation in this simulation length
        simlon = self.twopi * self.simtime / self.rotation_period
        # How many radians of Carrington rotation in the spin up period
        bufferlon = self.twopi * buffertime / self.rotation_period

        # Loop through model longitudes and solve each radial profile.
        for i in range(self.lon.size):

            if self.lon.size == 1:
                lon_out = self.lon.value
            else:
                lon_out = self.lon[i].value

            # Find the Carrigton longitude range spanned by the spin up and simulation period,
            # centered on simulation longitude
            lon_start = (lon_out - simlon - dlondt)
            lon_stop = (lon_out + bufferlon)
            lonint = np.arange(lon_start, lon_stop, dlondt)
            # Rectify so that it is between 0 - 2pi
            loninit = _zerototwopi_(lonint)
            # Interpolate the inner boundary speed to this higher resolution
            vinit = np.interp(loninit, all_lons.value, self.v_boundary.value, period=2 * np.pi)
            # convert from cr longitude to timesolve
            vinput = np.flipud(vinit)
            
            #solve for these inputs, adding in CMEs where necessary
            v, CMErbounds  = solve_radial(vinput, model_time, self.rrel.value, lon_out,
                                          self.model_params, do_cme, cme_params,
                                          self.latitude.value)
            #save the outputs
            self.v_grid[:, :, i] = v * self.kms
            
            self.cme_particles[:, :, :, i] = CMErbounds * u.dimensionless_unscaled
            
        #  Update CMEs positions by tracking through the solution.
        updated_cmes = []
        for cme_num, cme in enumerate(self.cmes):
            cme._track_(self, cme_num)
            updated_cmes.append(cme)

        self.cmes = updated_cmes

        if save:
            if tag == '':
                print("Warning, blank tag means file likely to be overwritten")
            self.save(tag=tag)
        return

   
    def save(self, tag=''):
        """
        Save all model fields output to a HDF5 file.

        :param tag: identifying string to append to the filename
        :return out_filepath: Full path to the saved file.
        """
        # Open up hdf5 data file for the HI flow stats
        filename = "HUXt_CR{:03d}_{}.hdf5".format(np.int32(self.cr_num.value), tag)
        out_filepath = os.path.join(self._data_dir_, filename)

        if os.path.isfile(out_filepath):
            # File exists, so delete and start new.
            print("Warning: {} already exists. Overwriting".format(out_filepath))
            os.remove(out_filepath)

        out_file = h5py.File(out_filepath, 'w')

        # Save the Cone CME parameters to a new group.
        allcmes = out_file.create_group('ConeCMEs')
        for i, cme in enumerate(self.cmes):
            cme_name = "ConeCME_{:02d}".format(i)
            cmegrp = allcmes.create_group(cme_name)
            for k, v in cme.__dict__.items():
                if k != "coords":
                    dset = cmegrp.create_dataset(k, data=v.value)
                    dset.attrs['unit'] = v.unit.to_string()
                    out_file.flush()
                # Now handle the dictionary of CME boundary coordinates coords > time_out > position
                if k == "coords":
                    coordgrp = cmegrp.create_group(k)
                    for time, position in v.items():
                        time_label = "t_out_{:03d}".format(time)
                        timegrp = coordgrp.create_group(time_label)
                        for pos_label, pos_data in position.items():
                            dset = timegrp.create_dataset(pos_label, data=pos_data.value)
                            dset.attrs['unit'] = pos_data.unit.to_string()
                            out_file.flush()                           

        # Loop over the attributes of model instance and save select keys/attributes.
        keys = ['cr_num', 'cr_lon_init', 'simtime', 'dt', 'v_max', 'r_accel', 'alpha',
                'dt_scale', 'time_out', 'dt_out', 'r', 'dr', 'lon', 'dlon', 'r_grid', 'lon_grid',
                'v_grid', 'latitude', 'v_boundary', '_v_boundary_init_', 'cme_particles', 'frame']
        
        for k, v in self.__dict__.items():

            if k in keys:
                if isinstance(v,str):
                    dset = out_file.create_dataset(k, data=v)
                else:
                    dset = out_file.create_dataset(k, data=v.value)
                    dset.attrs['unit'] = v.unit.to_string()

                # Add on the dimensions of the spatial grids
                if k in ['r_grid', 'lon_grid']:
                    dset.dims[0].label = 'radius'
                    dset.dims[1].label = 'longitude'

                # Add on the dimensions of the output speed fields.
                if k == 'v_grid':
                    dset.dims[0].label = 'time'
                    dset.dims[1].label = 'radius'
                    dset.dims[2].label = 'longitude'

                out_file.flush()

        out_file.close()
        return out_filepath
    
    def get_observer(self, body):
        """
        Returns an instance of the Observer class, giving the HEEQ and Carrington coordinates at each model timestep.
        This is only well defined if the model was initialised with a Carrington rotation number.
        :param body: String specifying which body to look up. Valid bodies are Earth, Venus, Mercury, STA, and STB.
        """
        times = self.time_init + self.time_out
        obs = Observer(body, times)
        return obs

def huxt_constants():
    """
    Return some constants used in all HUXt model classes
    """
    nlong=128  #number of longitude bins for a full longitude grid [128]
    dr = 1.5 * u.solRad  # Radial grid step. With v_max, this sets the model time step [1.5*u.solRad]
    nlat=45    #number of ltitude bins for a full ltitude grid [45]
    v_max = 2000 * u.km/u.s  #maximum expected solar wind speed. Sets timestep [2000*u.km / u.s]
    
    #CONSTANTS - DON'T CHANGE
    twopi = 2.0 * np.pi
    daysec = 24 * 60 * 60 * u.s
    kms = u.km / u.s
    alpha = 0.15 * u.dimensionless_unscaled  # Scale parameter for residual SW acceleration
    r_accel = 50 * u.solRad  # Spatial scale parameter for residual SW acceleration
    synodic_period = 27.2753 * daysec  # Solar Synodic rotation period from Earth.
    sidereal_period = 25.38 *daysec # Solar sidereal rotation period 
    
    
    constants = {'twopi': twopi, 'daysec': daysec, 'kms': kms, 'alpha': alpha,
                 'r_accel': r_accel, 'synodic_period': synodic_period, 
                 'sidereal_period': sidereal_period, 'v_max': v_max,
                 'dr': dr, 'nlong' : nlong, 'nlat' : nlat}
            
    return constants


@u.quantity_input(r_min=u.solRad)
@u.quantity_input(r_max=u.solRad)
def radial_grid(r_min=30.0 * u.solRad, r_max=240. * u.solRad):
    """
    Define the radial grid of the HUXt model. Step size is fixed, but inner and outer boundary may be specified.

    :param r_min: The heliocentric distance of the inner radial boundary
    :param r_max: The heliocentric distance of the outer radial boundary
    """
    if r_min >= r_max:
        print("Warning, r_min cannot be less than r_max. Defaulting to r_min=30rs and r_max=240rs")
        r_min = 30 * u.solRad
        r_max = 240 * u.solRad

    if r_min < 2.0 * u.solRad:
        print("Warning, r_min should not be less than 5.0rs. Defaulting to 5.0rs")
        r_min = 5.0 * u.solRad

    if r_max > 3000 * u.solRad:
        print("Warning, r_max should not be more than 400rs. Defaulting to 400rs")
        r_max = 400 * u.solRad

    constants = huxt_constants()
    dr = constants['dr']
    r = np.arange(r_min.value, r_max.value + dr.value, dr.value)
    r = r * dr.unit
    nr = r.size
    rrel = r - 30*u.solRad
    return r, dr, rrel, nr


@u.quantity_input(lon_out=u.rad)
@u.quantity_input(lon_start=u.rad)
@u.quantity_input(lon_stop=u.rad)
def longitude_grid(lon_out=np.NaN * u.rad, lon_start=np.NaN * u.rad, lon_stop=np.NaN * u.rad):
    """
    Define the longitude grid of the HUXt model.

    :param lon_out:
    :param lon_start: The first longitude (in a clockwise sense) of a longitude range
    :param lon_stop: The last longitude (in a clockwise sense) of a longitude range
    """
    # Check the inputs.
    twopi = 2.0 * np.pi
    single_longitude = False
    longitude_range = False
    if np.isfinite(lon_out):
        # Select single longitude only. Check in range
        if (lon_out < 0 * u.rad) | (lon_out > twopi * u.rad):
            lon_out = _zerototwopi_(lon_out.to('rad').value)
            lon_out = lon_out * u.rad

        single_longitude = True
    elif np.isfinite(lon_start) & np.isfinite(lon_stop):
        # Select a range of longitudes. Check limits in range.
        if (lon_start < 0 * u.rad) | (lon_start > twopi * u.rad):
            lon_start = _zerototwopi_(lon_start.to('rad').value)
            lon_start = lon_start * u.rad

        if (lon_stop < 0 * u.rad) | (lon_stop > twopi * u.rad):
            lon_stop = _zerototwopi_(lon_stop.to('rad').value)
            lon_stop = lon_stop * u.rad

        longitude_range = True

    # Form the full longitude grid.
    nlon = huxt_constants()['nlong']
    dlon = twopi / nlon
    lon_min_full = dlon / 2.0
    lon_max_full = twopi - (dlon / 2.0)
    lon, dlon = np.linspace(lon_min_full, lon_max_full, nlon, retstep=True)
    lon = lon * u.rad
    dlon = dlon * u.rad

    # Now get only the selected longitude or range of longitudes
    if single_longitude:
        # Lon out takes precedence over lon_min and lon_max
        id_match = np.argmin(np.abs(lon - lon_out))
        lon = lon[id_match]
        nlon = lon.size

    elif longitude_range:
        # How to do the logic of this?
        # Want clockwise between lon start and lon stop
        if lon_start < lon_stop:
            id_match = (lon >= lon_start) & (lon <= lon_stop)
        elif lon_start > lon_stop:
            id_match = (lon >= lon_start) | (lon <= lon_stop)

        lon = lon[id_match]
        nlon = lon.size

    return lon, dlon, nlon


def time_grid(simtime, dt_scale):
    """
    Define the model timestep and time grid based on CFL condition and specified simulation time.

    :param simtime: The length of the simulation
    :param dt_scale: An integer specifying how frequently model timesteps should be saved to output.
    """
    constants = huxt_constants()
    v_max = constants['v_max']
    dr = constants['dr']
    dr = dr.to('km')
    dt = (dr / v_max).to('s')
    dtdr = dt / dr

    nt = np.int32(np.floor(simtime.to(dt.unit) / dt))  # number of time steps in the simulation
    time = np.arange(0, nt) * dt  # Model time steps

    dt_out = dt_scale * dt  # time step of the output
    nt_out = np.int32(nt / dt_scale)  # number of time steps in the output
    time_out = np.arange(0, nt_out) * dt_out  # Output time steps

    time_grid_dict = {'dt': dt, 'dtdr': dtdr, 'Nt': nt, 'time': time,
                      'dt_out': dt_out, 'nt_out': nt_out, 'time_out': time_out}
    return time_grid_dict


def _setup_dirs_():
    """
    Function to pull out the directories of boundary conditions, ephemeris, and to save figures and output data.
    """
    # Find the config.dat file path
    files = glob.glob('config.dat')

    if len(files) != 1:
        # If wrong number of config files, guess directories
        print('Error: Cannot find correct config file with project directories. Check config.dat exists')
        print('Defaulting to current directory')
        dirs = {'root': os.getcwd()}
        for rel_path in ['boundary_conditions', 'ephemeris', 'HUXt_data', 'HUXt_figures']:
            if rel_path == 'ephemeris':
                dirs[rel_path] = os.path.join(os.getcwd(), "ephemeris.hdf5")
            else:
                dirs[rel_path] = os.getcwd()
    else:
        # Extract data and figure directories from config.dat
        with open(files[0], 'r') as file:
            lines = file.read().splitlines()
            root = lines[0].split(',')[1]
            dirs = {line.split(',')[0]: os.path.join(root, line.split(',')[1]) for line in lines[1:]}

        # Just check the directories exist.
        for val in dirs.values():
            if not os.path.exists(val):
                print('Error, invalid path, check config.dat: ' + val)

    return dirs


@jit(nopython=True)
def _zerototwopi_(angles):
    """
    Function to constrain angles to the 0 - 2pi domain.

    :param angles: a numpy array of angles
    :return: a numpy array of angles
    """
    twopi = 2.0 * np.pi
    angles_out = angles
    a = -np.floor_divide(angles_out, twopi)
    angles_out = angles_out + (a * twopi)
    return angles_out


@jit(nopython=True)
def solve_radial(vinput, model_time, rrel, lon, params, 
                 do_cme, cme_params, latitude):
    """
    Solve the radial profile as a function of time (including spinup), and return radial profile at specified
    output timesteps.
    
    :param vinput: Timeseries of inner boundary solar wind speeds
    :param model_time: Array of model timesteps
    :param rrel: Array of model radial coordinates relative to inner boundary coordinate
    :param lon: The longitude of this radial
    :param params: Array of HUXt parameters
    :param do_cme: Boolean, if True any provided ConeCMEs are included in the solution.
    :param cme_params: Array of ConeCME parameters to include in the solution. 1 Row for each CME, with columns as
                       required by _is_in_cone_cme_boundary_
    :param latitude: Latitude (from equator) of the HUXt plane

    Returns:

    """
    
    # Main model loop
    dtdr = params[0]
    alpha = params[1]
    r_accel = params[2]
    dt_scale = np.int32(params[3])
    nt_out = np.int32(params[4])
    nr = np.int32(params[5])
    r_boundary = params[7]
    n_cme = cme_params.shape[0]
    
    #compute the radial grid for the test particles
    rgrid = rrel*695700.0 + r_boundary #  can't use astropy.untis because numba
    dr = rgrid[1]-rgrid[0]
    dt = dtdr*dr
   
    # Preallocate space for solutions
    v_grid = np.zeros((nt_out, nr)) 
    cme_particles = np.zeros((n_cme, nt_out, 2))*np.nan
    
    iter_count = 0
    t_out = 0
    
    for t, time in enumerate(model_time):
        # Get the initial condition, which will update in the loop,
        # and snapshots saved to output at right steps.
        if t == 0:
            v = np.ones(nr) * 400
            r_cmeparticles = np.ones((n_cme, 2))*np.nan
            
        # Update the inner boundary conditions
        v[0] = vinput[t]
        
        # Compute boundary speed of each CME at this time. 
        # Set boundary to the maximum CME speed at this time.
        if time > 0:
            if do_cme == 1:
                
                v_update_cme = np.zeros(n_cme)*np.nan
                #CMEtracer_update = np.zeros(n_cme)
                for n in range(n_cme):
                    cme = cme_params[n, :]
                    #check if this point is within the cone CME
                    if _is_in_cme_boundary_(r_boundary, lon, latitude, time, cme):                
                        v_update_cme[n] = cme[4]
                        #CMEtracer_update[n] = 1.0 #the CME number
                        
                        #  If the leading edge test particle doesn't exist, add it
                        if np.isnan(r_cmeparticles[n,0]):
                            r_cmeparticles[n, 0] = r_boundary
                            
                        #  Hold the CME trailing edge test particle at the inner boundary
                        #  Until if condition breaks
                        r_cmeparticles[n, 1] = r_boundary
                        
                #see if there are any CMEs
                if not np.all(np.isnan(v_update_cme)):
                    v[0] = np.nanmax(v_update_cme)
                    
        
        #  update all fields for the given longitude
        #  =====================================
        u_up = v[1:].copy()
        u_dn = v[:-1].copy()
        
        #  do a single model time step
        u_up_next = _upwind_step_(u_up, u_dn, dtdr, alpha, r_accel, rrel)
        #  Save the updated time step
        v[1:] = u_up_next.copy()
        
        #  move the test particles forward
        if t > 0:        
            for n in range(0,n_cme):  #  loop over each CME
                for bound in range(0,2): #  loop over front and rear boundaries
                     if (np.isnan(r_cmeparticles[n,bound]) == False):
                         #  linearly interpolate the speed
                         v_test = np.interp(r_cmeparticles[n, bound] - dr/2, rgrid, v)
                         #  advance the test particle
                         r_cmeparticles[n ,bound] = (r_cmeparticles[n, bound] + v_test * dt)
                            
                if r_cmeparticles[n, 0] > rgrid[-1]:
                    #  if the leading edge is past the outer boundary, put it at the outer boundary
                    r_cmeparticles[n, 0] = rgrid[-1]
                    
                if r_cmeparticles[n, 1] > rgrid[-1]:
                    #  if the trailing edge is past the outer boundary,delete
                    r_cmeparticles[n,:] = np.nan

        # Save this frame to output if it is an output timestep
        if time >= 0:
            iter_count = iter_count + 1
            if iter_count == dt_scale:
                if t_out <= nt_out - 1:
                    v_grid[t_out, :] = v.copy()
                    cme_particles[:, t_out, :] = r_cmeparticles.copy()
                    t_out = t_out + 1
                    iter_count = 0
    
    return v_grid, cme_particles


@jit(nopython=True)
def _upwind_step_(v_up, v_dn, dtdr, alpha, r_accel, rrel):
    """
    Compute the next step in the upwind scheme of Burgers equation with added acceleration of the solar wind.
    :param v_up: A numpy array of the upwind radial values. Units of km/s.
    :param v_dn: A numpy array of the downwind radial values. Units of km/s.
    :param br_up: A numpy array of the upwind passive tracer  values. 
    :param br_dn: A numpy array of the downwind passive tracer values. 
    :param dtdr: Ratio of HUXts time step and radial grid step. Units of s/km.
    :param alpha: Scale parameter for residual Solar wind acceleration.
    :param r_accel: Spatial scale parameter of residual solar wind acceleration. Units of km.
    :param rrel: The model radial grid relative to the radial inner boundary coordinate. Units of km.
    :return: The upwind values at the next time step, numpy array with units of km/s.
    """
   
    # Arguments for computing the acceleration factor
    accel_arg = -rrel[:-1] / r_accel
    accel_arg_p = -rrel[1:] / r_accel

    # Get estimate of next time step
    v_up_next = v_up - dtdr * v_up * (v_up - v_dn)
    # Compute the probable speed at 30rS from the observed speed at r
    v_source = v_dn / (1.0 + alpha * (1.0 - np.exp(accel_arg)))
    # Then compute the speed gain between r and r+dr
    v_diff = alpha * v_source * (np.exp(accel_arg) - np.exp(accel_arg_p))
    # Add the residual acceleration over this grid cell
    v_up_next = v_up_next + (v_dn * dtdr * v_diff)
    
    return v_up_next


@jit(nopython=True)
def _is_in_cme_boundary_(r_boundary, lon, lat, time, cme_params):
    """
    Check whether a given lat, lon point on the inner boundary is within a given CME.
    :param r_boundary: Height of model inner boundary.
    :param lon: A HEEQ latitude, in radians.
    :param lat: A HEEQ longitude, in radians.
    :param time: Model time step, in seconds
    :param cme_params: An array containing the cme parameters
    :return: True/False
    """
    isincme=False

    cme_t_launch = cme_params[0]
    cme_lon = cme_params[1]
    cme_lat = cme_params[2]
    cme_width = cme_params[3]
    cme_v = cme_params[4]
    # cme_initial_height = cme_params[5]
    cme_radius = cme_params[6]
    cme_thickness = cme_params[7]

    # Center the longitude array on CME nose, running from -pi to pi, to avoid dealing with any 0/2pi crossings
    lon_cent = lon - cme_lon
    if lon_cent > np.pi:
        lon_cent = 2.0 * np.pi - lon_cent
    if lon_cent < -np.pi:
        lon_cent = lon_cent + 2.0 * np.pi

    lat_cent = lat - cme_lat
    if lat_cent > np.pi:
        lat_cent = 2.0 * np.pi - lat_cent
    if lat_cent < -np.pi:
        lat_cent = lat_cent + 2.0 * np.pi

    # Compute great circle distance from nose to input latitude
    sigma = np.arccos( np.cos(lat_cent) * np.cos(lon_cent))  # simplified version for the frame centered on the CME
   
    x = np.NaN
    if (lon_cent >= -cme_width / 2) & (lon_cent <= cme_width / 2):
        # Longitude inside CME span.
        #  Compute y, the height of CME nose above the 30rS surface
        y = cme_v * (time - cme_t_launch)

        if (y >= 0) & (y < cme_radius):
            # this is the front hemisphere of the spherical CME
            x = np.sqrt(y * (2 * cme_radius - y))  # compute x, the distance of the current longitude from the nose
        elif (y >= (cme_radius + cme_thickness)) & (y <= (2 * cme_radius + cme_thickness)):
            # this is the back hemisphere of the spherical CME
            y = y - cme_thickness
            x = np.sqrt(y * (2 * cme_radius - y))
        elif (cme_thickness > 0) & (y >= cme_radius) & (y <= (cme_radius + cme_thickness)):
            # this is the "mass" between the hemispheres
            x = cme_radius

        theta = np.arctan(x / r_boundary)
        if sigma <= theta:
            isincme=True

    return isincme


def load_HUXt_run(filepath):
    """
    Load in data from a saved HUXt run. If Br fields are not saved, pads with
    NaN to avoid conflicts with other HUXt routines.

    :param filepath: The full path to a HDF5 file containing the output from HUXt.save()
    :return: cme_list: A list of instances of ConeCME
    :return: model: An instance of HUXt containing loaded results.
    """
    if os.path.isfile(filepath):

        data = h5py.File(filepath, 'r')

        cr_num = np.int32(data['cr_num'])
        cr_lon_init = data['cr_lon_init'][()] * u.Unit(data['cr_lon_init'].attrs['unit'])
        simtime = data['simtime'][()] * u.Unit(data['simtime'].attrs['unit'])
        simtime = simtime.to(u.day)
        dt_scale = data['dt_scale'][()]
        v_boundary = data['_v_boundary_init_'][()] * u.Unit(data['_v_boundary_init_'].attrs['unit'])
        r = data['r'][()] * u.Unit(data['r'].attrs['unit'])
        lon = data['lon'][()] * u.Unit(data['lon'].attrs['unit'])
        lat = data['latitude'][()] * u.Unit(data['latitude'].attrs['unit'])
        nlon = lon.size
        frame = data['frame'][()]
        
         #check if br data was saved
        if '_br_boundary_init_' in data.keys():
            br_boundary = data['_br_boundary_init_'][()] * u.Unit(data['_br_boundary_init_'].attrs['unit'])
        if '_rho_boundary_init_' in data.keys():
            rho_boundary = data['_rho_boundary_init_'][()] * u.Unit(data['_rho_boundary_init_'].attrs['unit'])


        #create the model class
        if nlon == 1:
            args='v_boundary=v_boundary,  cr_num=cr_num, cr_lon_init=cr_lon_init, r_min=r.min(), r_max=r.max(),lon_out=lon, simtime=simtime, dt_scale=dt_scale, latitude=lat, frame=frame' 
        elif nlon > 1:
            args = 'v_boundary=v_boundary, cr_num=cr_num, cr_lon_init=cr_lon_init, r_min=r.min(), r_max=r.max(), lon_start=lon.min(), lon_stop=lon.max(), simtime=simtime, dt_scale=dt_scale, latitude=lat, frame=frame'
        if '_br_boundary_init_' in data.keys():
            args = args +', br_boundary=br_boundary'
        if '_rho_boundary_init_' in data.keys():
            args = args +', rho_boundary=rho_boundary'  
        
        model = eval('HUXt(' + args +')')
        #reset the longitudes, as when onlyt a wedge is simulated, it gets confused.
        model.lon=lon
        model.nlon = nlon

        model.v_grid = data['v_grid'][()] * u.Unit(data['v_boundary'].attrs['unit'])
        model.CMEtracer_grid = data['CMEtracer_grid'][()] * u.dimensionless_unscaled
        model.cme_particles = data['cme_particles'][()] 
        #check if br data was saved
        if 'br_grid' in data.keys():
            model.br_grid = data['br_grid'][()] * u.Unit(data['br_boundary'].attrs['unit'])
        #check if rho data was saved
        if 'rho_grid' in data.keys():
            model.rho_grid = data['rho_grid'][()] * u.Unit(data['rho_boundary'].attrs['unit'])

        # Create list of the ConeCMEs
        cme_list = []
        all_cmes = data['ConeCMEs']
        for k in all_cmes.keys():
            cme_data = all_cmes[k]
            t_launch = cme_data['t_launch'][()] * u.Unit(cme_data['t_launch'].attrs['unit'])
            lon = cme_data['longitude'][()] * u.Unit(cme_data['longitude'].attrs['unit'])
            lat = cme_data['latitude'][()] * u.Unit(cme_data['latitude'].attrs['unit'])
            width = cme_data['width'][()] * u.Unit(cme_data['width'].attrs['unit'])
            thickness = cme_data['thickness'][()] * u.Unit(cme_data['thickness'].attrs['unit'])
            thickness = thickness.to('solRad')
            v = cme_data['v'][()] * u.Unit(cme_data['v'].attrs['unit'])
            cme = ConeCME(t_launch=t_launch, longitude=lon, latitude=lat, v=v, width=width, thickness=thickness)

            # Now sort out coordinates.
            # Use the same dictionary structure as defined in ConeCME._track_2d_
            coords_group = cme_data['coords']
            coords_data = {j: {'lon_pix': np.array([]) * u.pix, 'r_pix': np.array([]) * u.pix,
                               'lon': np.array([]) * model.lon.unit, 'r': np.array([]) * model.r.unit}
                           for j in range(len(coords_group))}

            for time_key, pos in coords_group.items():
                t = np.int(time_key.split("_")[2])
                coords_data[t]['lon_pix'] = pos['lon_pix'][()] * u.Unit(pos['lon_pix'].attrs['unit'])
                coords_data[t]['r_pix'] = pos['r_pix'][()] * u.Unit(pos['r_pix'].attrs['unit'])
                coords_data[t]['lon'] = pos['lon'][()] * u.Unit(pos['lon'].attrs['unit'])
                coords_data[t]['r'] = pos['r'][()] * u.Unit(pos['r'].attrs['unit'])

            cme.coords = coords_data
            cme_list.append(cme)

        # Update CMEs in model output
        model.cmes = cme_list

    else:
        # File doesnt exist return nothing
        print("Warning: {} doesnt exist.".format(filepath))
        cme_list = []
        model = []

    return model, cme_list