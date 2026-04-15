
import os
import sys

import copy
import errno
from appdirs import user_data_dir
import astropy.units as u
from astropy.time import Time, TimeDelta
import h5py
from joblib import Parallel, delayed
import numpy as np
from numba import jit
from pathlib import Path
from sunpy.coordinates import sun

from surf_solvers import create_solver as create_compressible_solver


VALID_SOLVERS = ("huxt", "hydro", "hydro-pcm")



class Observer:
    """
    A class returning the HEEQ and Carrington coordinates of a specified Planet or spacecraft, for a given set of times.
    Planets include the inner planets out to Saturn. The ephemeris data for Jupiter and Saturn correspond to the
    Jupiter system barycenter and Saturn system barycenter. Spacecraft include ACE, STEREO-A, STEREO-B, Parker Solar
    Probe and Solar Orbiter. The ephemeris data are downloaded from JPL Horizons. Planetary positions are linearly
    interpolated from a 6-hour resolution ephemeris that spans 1974-01-01 until 2029-01-01. Spacecraft positions are
    linearly interpolated from a 3-hour resolution ephemeris that spans the available duration on JPL Horizons for each
    mission. JPL Horizons only provides ACE and STEREO-A data for short windows into the future (~70 and ~100 days,
    respectively). And so the ephemeris file may need to be periodically updated. The ephemeris data can be updated
    using the SURF/scripts/make_ephemeris.py script.

    Attributes:
        body: String name of the planet or spacecraft.
        lat: HEEQ latitude of body at all values of time.
        lat_c: Carrington latitude of body at all values of time.
        lat_hae: HAE latitude of body at all values of time
        lon: HEEQ longitude of body at all values of time.
        lon_c: Carrington longitude of body at all values of time.
        lon_hae: HAE longitude of body at all values of time
        r: HEEQ radius of body at all values of time.
        r_c: Carrington radius of body at all values of time.
        r_hae: HAE radius of body at all values of time.
        time: Array of Astropy Times
    """

    def __init__(self, body, times):
        """
            body: String indicating which body to look up the positions of .
            times: A list/array of Astropy Times to interpolate the coordinate of the selected body.
        """
        craft = ["ACE", "STA", "STB", "PSP", "SOLO", "ULYSSES"]
        planets = ["MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN"]
        bodies = planets + craft
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

        # STEREO-A and ACE have shorter lengths of ephemeris data. Check requested times not outside those available.
        if np.any(times > all_time[-1]):
            raise ValueError(f"{body} ephemeris extends to {all_time[-1].isot}. Requested times are outside this limit."
                             f" Updating the SURF ephemeris file may resolve this issue.")

        # Pad out the window to account for single values being passed.
        if self.body in craft:
            dt = TimeDelta(2 * 60 * 60, format='sec')  # craft ephem is 4 hourly, so dt=2
        elif self.body in planets:
            dt = TimeDelta(6 * 60 * 60, format='sec')  # planet ephem is 12 hourly, so dt=6

        id_epoch = (all_time >= (times.min() - dt)) & (all_time <= (times.max() + dt))
        epoch_time = all_time[id_epoch]

        self.time = times
        if len(epoch_time.jd) == 0:
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
            self.lon = zerototwopi(self.lon)
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
            self.lon_hae = zerototwopi(self.lon_hae)
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
            self.lon_c = zerototwopi(self.lon_c)
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
        initial_height: Initiation height of the CME, in km. Defaults to SURF inner boundary at 21.5 solar radii.
        radius: Initial radius of the CME, in km.
        thickness: Thickness of the CME cone, in km.
        cme_density: Mass density of the CME in kg/m³. Defaults to x0.1 the solar wind density at initial_height.
        cme_temperature: Temperature of the CME in Kelvin. Defaults x0.1 the solar wind temperature at initial_height.
        profile_type: Temporal profile shape ('square' or 'sinusoidal'). 
                     'square': step function from ambient to CME values
                     'sinusoidal': smooth sinusoidal pulse from ambient to CME values and back
        coords: Dictionary containing the radial and longitudinal (for SURF2D) coordinates of the of Cone CME for each
                model time step.
    """

    def __init__(self, t_launch=0.0 * u.s, longitude=0.0 * u.deg, latitude=0.0 * u.deg, v=1000.0 * (u.km / u.s),
                 width=30.0 * u.deg, thickness=0.0 * u.solRad, initial_height=21.5 * u.solRad, cme_expansion=False,
                 cme_fixed_duration=True, fixed_duration=12 * 60 * 60 * u.s, 
                 cme_density=np.nan * (u.kg / u.m**3), cme_temperature=np.nan * u.K, 
                 density_fraction=1, temperature_fraction=1,
                 profile_type='square',
                 label=None):

        """
        Set up a Cone CME with specified parameters.
        Args:
            t_launch: Time of Cone CME launch, in seconds after the start of the simulation.
            longitude: HEEQ Longitude of the CME launch direction, in radians.
            latitude: HEEQ latitude of the CME launch direction, in radians.
            v: CME nose speed in km/s.
            width: Angular width of the CME, in degrees.
            thickness: Thickness of the CME cone, in solar radii.
            initial_height: Height (in solRad) that corresponds to the launch time.
            cme_expansion : Whether to insert a declining speed profile at the inner boundary
            cme_fixed_duration : Whether to fix the CME duration, or do a standard cone CME
            fixed_duration : If fixed duration, the value to use
            cme_density: CME mass density in kg/m³. If provided, density_fraction is ignored.
            cme_temperature: CME temperature in Kelvin. If provided, temperature_fraction is ignored.
            density_fraction: Fraction of ambient solar wind density.  Only used if cme_density is not provided.
            temperature_fraction: Fraction of ambient solar wind temperature. Only used if cme_temperature is not provided.
            profile_type: Type of temporal profile for CME perturbation. Options:
                         'square' (default): Step function from ambient to CME values
                         'sinusoidal': Smooth sinusoidal pulse from ambient to CME values and back
        Returns:
            None
        """
 
        self.t_launch = t_launch  # Time of CME launch, after the start of the simulation
        lon = zerototwopi(longitude.to(u.rad).value) * u.rad
        self.longitude = lon  # User-supplied Longitudinal launch direction of the CME
        self.latitude = latitude.to(u.rad)  # Latitude launch direction of the CME
        self.v = v  # CME nose speed
        self.width = width  # Angular width
        self.initial_height = initial_height  # Initial height of CME (should match inner boundary of SURF)
        self.radius = self.initial_height * np.tan(self.width / 2.0)  # Initial radius of CME
        self.thickness = thickness  # Extra CME thickness
        self.coords = {}
        self.frame = 'NA'
        self.longitude_surf = -1 * u.rad  # the SURF longitude, adjusted for sidereal frame if necessary
        self.cme_expansion = cme_expansion
        self.cme_fixed_duration = cme_fixed_duration
        self.fixed_duration = fixed_duration
        self.cme_density = cme_density   
        self.cme_temperature = cme_temperature
        self.density_fraction = density_fraction
        self.cme_temperature_fraction = temperature_fraction
        
        # Validate and store profile type
        if profile_type not in ['square', 'sinusoidal']:
            raise ValueError(f"profile_type must be 'square' or 'sinusoidal', not '{profile_type}'")
        self.profile_type = profile_type     
          
        if isinstance(label, str) | (label is None):
            self.label = label
        else:
            raise ValueError(f'Label must be an instance of str or None, not {type(label)}')

        self.__version__ = get_version()
        return

    def parameter_array(self, model):
        """
        Returns a numpy array of CME parameters. This is used in the numba optimised solvers that don't play nicely
        with classes.
        Returns:
            None
        """
        if model.compressible == True:
            # Check if CME density was provided (not NaN)
            if np.isnan(self.cme_density.value):
                # If CME density not provided, set to density fraction of ambient solar wind density at initial height
                self.cme_density = self.density_fraction * model.rho_sw_inner
            # Check if CME temperature was provided (not NaN)
            if np.isnan(self.cme_temperature.value):
                # If CME temperature not provided, set to temperature fraction of ambient solar wind temperature at initial height
                self.cme_temperature = self.cme_temperature_fraction * model.T_sw_inner

        # Convert profile_type to numeric flag: 0 = square, 1 = sinusoidal
        profile_flag = 1.0 if self.profile_type == 'sinusoidal' else 0.0
        
        cme_parameters = [self.t_launch.to('s').value, 
                          self.longitude_surf.to('rad').value,
                          self.latitude.to('rad').value,
                          self.width.to('rad').value, 
                          self.v.value, 
                          self.initial_height.to('km').value,
                          self.radius.to('km').value, 
                          self.thickness.to('km').value, 
                          self.longitude.to('rad').value,
                          self.cme_expansion,
                          self.cme_fixed_duration,
                          self.fixed_duration.to('s').value,
                          self.cme_density.value,
                          self.cme_temperature.value,
                          profile_flag]
        return cme_parameters

    def _track_(self, model, cme_id):
        """
        Tracks the perimeter of each ConeCME through the SURF solution in model. Updates the ConeCME.coords dictionary
        of CME coordinates.
        Args:
            model: An SURF instance with solution containing ConeCMEs
            cme_id: ID number of the CME to link the ConeCME object with the CME tracer particle fields.
        Returns:
             None
        """
        # Keep track of synodic or sidereal
        self.frame = copy.copy(model.frame)

        # Pull out the particle field for this CME
        cme_r_field = model.cme_particles_r[cme_id, :, :, :]
        cme_v_field = model.cme_particles_v[cme_id, :, :, :]

        # Setup dictionary to track this CME
        self.coords = {j: {'time': np.array([]), 'model_time': np.array([]) * u.s,
                           'front_id': np.array([]) * u.dimensionless_unscaled,
                           'lon': np.array([]) * model.lon.unit,
                           'r': np.array([]) * model.r.unit,
                           'lat': np.array([]) * model.latitude.unit,
                           'v': np.array([]) * model.v_grid.unit} for j in range(model.nt_out)}

        # Loop through timesteps, save out coords to coords dict
        for j, t in enumerate(model.time_out):

            self.coords[j]['model_time'] = t
            self.coords[j]['time'] = model.time_init + t

            cme_r_front = cme_r_field[j, 0, :]
            cme_r_back = cme_r_field[j, 1, :]

            cme_v_front = cme_v_field[j, 0, :]
            cme_v_back = cme_v_field[j, 1, :]

            if np.any(np.isfinite(cme_r_front)) | np.any(np.isfinite(cme_r_back)):
                # Get longitudes and center on CME
                lon = model.lon - self.longitude

                # single longitude runs need different treatment to multi lon runs.
                if lon.size == 1:
                    if lon > np.pi * u.rad:
                        lon -= 2 * np.pi * u.rad
                    elif lon < -np.pi * u.rad:
                        lon += 2 * np.pi * u.rad

                    lons = np.hstack([lon, lon])
                    cme_r = np.hstack([cme_r_front, cme_r_back])
                    cme_v = np.hstack([cme_v_front, cme_v_back])
                    front_id = np.hstack([1.0, 0.0])
                else:
                    # Correct the wrap arounds at +/-pi
                    lon[lon > np.pi * u.rad] -= 2 * np.pi * u.rad
                    lon[lon < -np.pi * u.rad] += 2 * np.pi * u.rad

                    # Find indices that sort the longitudes, to make a wraparound of lons
                    id_sort_inc = np.argsort(lon)
                    id_sort_dec = np.flipud(id_sort_inc)

                    cme_r_front = cme_r_front[id_sort_inc]
                    cme_r_back = cme_r_back[id_sort_dec]

                    cme_v_front = cme_v_front[id_sort_inc]
                    cme_v_back = cme_v_back[id_sort_dec]

                    lon_front = lon[id_sort_inc]
                    lon_back = lon[id_sort_dec]

                    # Only keep good values
                    id_good = np.isfinite(cme_r_front)
                    cme_r_front = cme_r_front[id_good]
                    cme_v_front = cme_v_front[id_good]
                    lon_front = lon_front[id_good]

                    id_good = np.isfinite(cme_r_back)
                    cme_r_back = cme_r_back[id_good]
                    cme_v_back = cme_v_back[id_good]
                    lon_back = lon_back[id_good]

                    # Get one array of longitudes and radii from the front and back particles
                    lons = np.hstack([lon_front, lon_back])
                    cme_r = np.hstack([cme_r_front, cme_r_back])
                    cme_v = np.hstack([cme_v_front, cme_v_back])
                    front_id = np.hstack([np.ones(cme_r_front.shape), np.zeros(cme_r_back.shape)])

                # Save to dict
                self.coords[j]['r'] = (cme_r * u.km).to(u.solRad)
                self.coords[j]['v'] = cme_v * (u.km / u.s)
                self.coords[j]['lon'] = lons + self.longitude
                self.coords[j]['front_id'] = front_id * u.dimensionless_unscaled
                self.coords[j]['lat'] = model.latitude.copy()
        return

    def compute_arrival_at_body(self, body_name):
        """
        Compute the arrival of the CME at a solar system body. Available bodies are those accepted by the 
        observer class, Mercury, Venus, Earth, STA, and STB. Takes account of differences between synodic 
        and sidereal frames
        Args:
            body_name: String body name as accepted by the Observer class, including Mercury, Venus, Earth, STA and STB.
        Returns:
             arrival_stats: A dictionary of the arrival stats of the CME, with keys hit, hit_id, t_arrive, t_transit,
                            lon, r and v.
        """

        # Get body ephemeris
        times = Time([coord['time'] for i, coord in self.coords.items()])
        body = Observer(body_name, times)
        arrival_stats = self.compute_arrival_at_location(body.lon, body.r)

        return arrival_stats

    def compute_arrival_at_location(self, longitude, radius):
        """
        Compute the arrival of the CME at a location specified with a longitude and radius. Takes account of differences
        between synodic and sidereal frames.
        Args:
            longitude: location longitude at t=0 of the model run. Should be in rads and be single value or have same
                       size as time.
            radius: location radius at t=0 of the model run. Should be in units of solRad and be single value or have
                    same size as time.
        Returns:
             arrival_stats: A dictionary of the arrival stats of the CME, with keys hit, hit_id, t_arrive, t_transit,
                            lon, r and v.
        """
        if not isinstance(longitude, u.Quantity):
            raise TypeError('longitude must be Quantity')

        if not isinstance(radius, u.Quantity):
            raise TypeError('radius must be Quantity')

        # Get body ephemeris
        times = Time([coord['time'] for i, coord in self.coords.items()])

        is_scalar = np.ndim(longitude) & np.ndim(radius)  # np.iscalar doesnt work on quantities
        if not is_scalar:
            # If not scalar, must have coords for each time step.
            match_len = (longitude.size == radius.size) & (longitude.size == times.size)
            if not match_len:
                raise ValueError('longitude and radius must be single values or be arrays of length equal to the number'
                                 ' of time steps')

        if is_scalar == 1:
            arrive_rad = np.ones(times.size) * radius
            arrive_lon = np.ones(times.size) * longitude

        # Correct longitudes if in sidereal frame
        if self.frame == 'sidereal':
            earth = Observer('EARTH', times)
            delta_lon = earth.lon_hae - earth.lon_hae[0]
            arrive_lon = zerototwopi(longitude + delta_lon)
            arrive_lon = arrive_lon * earth.lon.unit

        # Center longitudes on CME nose, between -180:180
        arrive_lon = arrive_lon - self.longitude
        id_low = arrive_lon < -180 * u.deg
        id_high = arrive_lon > 180 * u.deg
        if np.any(id_low):
            arrive_lon[id_low] += 360 * u.deg
        elif np.any(id_high):
            arrive_lon[id_high] -= 360 * u.deg

        hit = False
        t_front = []
        r_front = []
        v_front = []
        # Loop through coords at each timestep
        for i, coord in self.coords.items():

            if len(coord['r']) == 0:
                continue

            # Get lon and radial coords of the CME front only.
            r_cme = coord['r']
            v_cme = coord['v']
            lon_cme = coord['lon']
            front_id = coord['front_id'] == 1.0
            r_cme = r_cme[front_id]
            v_cme = v_cme[front_id]
            lon_cme = lon_cme[front_id] - self.longitude

            # If there are any CME front coords, then work out pos.
            if np.any(front_id):

                # Handle case for SURF run on multiple longitudes first
                if len(lon_cme) > 1:
                    # Lookup cme front radial coord along body longitude
                    r_interp = np.interp(arrive_lon[i], lon_cme, r_cme, left=np.nan, right=np.nan)
                    v_interp = np.interp(arrive_lon[i], lon_cme, v_cme, left=np.nan, right=np.nan)
                    if np.isfinite(r_interp):
                        t_front.append(coord['time'].jd)
                        r_front.append(r_interp)
                        v_front.append(v_interp.value)
                    else:
                        continue

                elif len(lon_cme) == 1:
                    # SURF run on a single longitude, so don't interpolate front to body longitude
                    # Instead, check when cme lon within tolerance lon of body

                    # If body and cme within 1.5 deg of each other, assume close enough for hit.
                    if np.isclose(arrive_lon[i], lon_cme, atol=1.5 * u.deg):
                        t_front.append(coord['time'].jd)
                        r_front.append(r_cme[0])
                        v_front.append(v_cme[0].value)
                    else:
                        continue

                # Has CME front crossed body radius
                if r_front[-1] > arrive_rad[i]:
                    hit = True
                    hit_id = i
                    hit_lon = arrive_lon[i] + self.longitude
                    hit_lon = zerototwopi(hit_lon) * u.rad
                    hit_rad = arrive_rad[i]

                    # Interpolate the arrival time and transit time
                    # from radial coords before and after body radius
                    t_arrive = np.interp(arrive_rad[i], r_front, t_front)
                    t_transit = (t_arrive - t_front[0]) * u.d
                    t_arrive = Time(t_arrive, format='jd')

                    v_arrive = np.interp(arrive_rad[i], r_front, v_front)
                    hit_v = v_arrive * u.km / u.s
                    break

        if not hit:
            hit_id = False
            t_arrive = Time("0000-01-01T00:00:00")
            t_transit = np.nan * u.d
            hit_lon = np.nan * u.deg
            hit_rad = np.nan * u.solRad
            hit_v = np.nan * u.km / u.s

        arrival_stats = {'hit': hit, 'hit_id': hit_id, 't_arrive': t_arrive, 't_transit': t_transit,
                         'lon': hit_lon, 'r': hit_rad, 'v': hit_v}

        return arrival_stats


class SURF:
    """
    A class containing the SURF model described in Owens et al. (2020, DOI: 10.1007/s11207-020-01605-3)

    Users must specify the solar wind speed boundary condition through the v_boundary keyword
    argument. Failure to do so defaults to a 400 km/s boundary.
    
    Model coordinate system is HEEQ radius and longitude.
    
    Attributes:
        cmes: A list of ConeCME instances used in the model solution.
        cr_num: If provided, this gives the Carrington rotation number of the selected period, else 9999.
        cr_lon_init: The initial Carrington longitude of Earth at the models initial timestep (2 pi at the start of the
                     CR, 0 at the end).
        daysec: seconds in a day.
        dlon: Longitudinal grid spacing (in radians)
        dr: Radial grid spacing (in km).
        dt: Model time step (in seconds), set by the CFL condition with v_max and dr.
        dt_out: Output model time step (in seconds).
        dt_scale: Integer scaling number to set the model output time step relative to the models CFL time step.
        dtdr: Ratio of the model time step and radial grid step (in seconds/km).
        frame : either synodic or sidereal
        kms: An astropy unit instance of km/s.
        lon: Array of model longtidues (in radians).
        model_time: time in seconds from the model start time. Includes spin up
        nlon: Number of longitudinal grid points.
        nr: Number of radial grid points.
        Nt: Total number of model time steps, including spin up.
        nt_out: Number of output model time steps.
        r_accel: Scale parameter determining the residual solar wind acceleration.
        r: Radial grid (in km).
        r_grid: Array of radial coordinates meshed with the longitudinal coordinates (in km).
        rrel: Radial grid relative to 30rS (in km).
        rotation_period:  rotation period (in seconds), either synodic or sidereal
        simtime: Simulation time (in seconds).
        time: Array of model time steps, including spin up (in seconds).
        time_init: The UTC time corresponding to the initial Carrington rotation number and longitude. Else, NaN.
        time_out: Array of output model time steps (in seconds).
        twopi: two pi radians
        v_boundary: Inner boundary solar wind speed profile (in km/s).
        v_max: Maximum model speed (in km/s), used with the CFL condition to set the model time step.
        v_grid: Array of model speed including ConeCMEs for each time, radius, and longitude (in km/s).
    """

    def __init__(self, v_boundary=np.nan * (u.km / u.s), b_boundary=np.nan, 
                 rho_boundary=np.nan, temp_boundary=np.nan,
                 cr_num=np.nan, cr_lon_init=360.0 * u.deg,
                 latitude=0 * u.deg, r_min=21.5 * u.solRad, r_max=240 * u.solRad, lon_out=np.nan * u.rad,
                 lon_start=np.nan * u.rad, lon_stop=np.nan * u.rad, simtime=5.0 * u.day, dt_scale=1.0, frame='synodic',
                 input_v_ts=np.nan * (u.km / u.s), input_b_ts=np.nan, 
                 input_rho_ts=np.nan * (u.kg / u.m**3), input_temp_ts=np.nan * u.K, 
                 input_iscme_ts=np.nan, input_t_ts=np.nan * u.s,
                 track_cmes=True, accel_limit=True, solver='huxt', parallel=False):
        """
        Initialise the SURF model instance.

            v_boundary: Inner solar wind speed boundary condition. An array of size nlon (default 128). Units of km/s.
            b_boundary: Inner B polarity boundary condition. An array of size nlon (default 128). Units of km/s.
            cr_num: Integer Carrington rotation number. Used to determine the planetary and spacecraft positions
            cr_lon_init: Carrington longitude of Earth at model initialisation, in degrees.
            latitude: Helio latitude (from the equator) of SURF plane, in degrees
            lon_out: A specific single longitude (relative to Earth) to compute SURF solution along, in degrees
            lon_start: The first longitude (in a clockwise sense) of the longitude range to solve SURF over.
            lon_stop: The last longitude (in a clockwise sense) of the longitude range to solve SURF over.
            r_min: The radial inner boundary distance of SURF.
            r_max: The radial outer boundary distance of SURF.
            simtime: Duration of the simulation window, in days.
            dt_scale: Integer scaling number to set the model output time step relative to the models CFL time.
            frame: string determining the rotation frame for the model
            input_v_ts: Time series of inner boundary V conditions. For initialising SURF with, for example, 
                           in-situ observations from L1. If used as keyword input argument, overrides v_boundary input.
            input_bv_ts: Time series of inner boundary B conditions. For initialising SURF with, for example, 
                            in-situ observations from L1. If used as keyword input argument, overrides b_boundary input.
            input_rho_ts: Time series of inner boundary density conditions in kg/m³. For initialising SURF with, for example,
                             in-situ observations from L1. If used as keyword input argument, overrides rho_boundary input.
                             Only used if compressible=True.
            input_temp_ts: Time series of inner boundary temperature conditions in Kelvin. For initialising SURF with, for example,
                              in-situ observations from L1. If used as keyword input argument, overrides temp_boundary input.
                              Only used if compressible=True.             
            input_t_ts: Times of input_v_ts in seconds, including spin up.
            input_iscme_ts: Boolean mask time series indicating what time steps correspond to CMEs in input_v_ts.
                               If used as keyword input argument, overrides ConeCMEs past to surf.sovle().
            save_full_v: Boolean flag to determine if full v field (including spin up) is saved for post-processing.
            track_cmes: Boolean flag to determine if CMEs are tracked at run time (small speed reduction).
            accel_limit: Boolean flag to determine if acceleration is switched for speeds above 650 km/s
            solver: String specifying the numerical solver to use. Options:
                     'huxt' (default): First-order HUXt advection scheme (incompressible)
                     'hydro': Second-order compressible HLLC+PLM solver
                     'hydro-pcm': Compressible HLLC+PCM solver
            parallel: Boolean flag to enable parallel computation across longitude slices (default True).
                     Uses joblib threading backend for parallelization. Set to False for debugging
                     or if running on a single-core system.
            rho_boundary: Inner density boundary condition in kg/m³. An array of size nlon. Only used if compressible=True.
                         If not provided, defaults to realistic solar wind density scaled from 1 AU (5 protons/cm³) 
                         using r⁻² scaling to r_min.
            temp_boundary: Inner temperature boundary condition in Kelvin. An array of size nlon. Only used if compressible=True.
                          If not provided, defaults to realistic solar wind temperature scaled from 1 AU (10⁵ K)
                          using r⁻⁰·⁶⁷ scaling to r_min.
        """

        # some constants and units
        constants = surf_constants()
        self.twopi = constants['twopi']
        self.daysec = constants['daysec']
        self.kms = constants['kms']
        self.alpha = constants['alpha']  # Scale parameter for residual SW acceleration (incompressible)
        self.r_accel = constants['r_accel']  # Spatial scale parameter for residual SW acceleration (incompressible)
        self.gamma = constants['gamma']  # Adiabatic index for compressible solver
        # Use a per-instance cache namespace so changes in constants are picked up
        # when a new SURF instance is created, while still reusing lookups within
        # the same instance.
        self._density_temp_cache_id = f"surf-{id(self)}"
        clear_density_temperature_cache(cache_id=self._density_temp_cache_id)
        self.__version__ = get_version()
        
        # Validate and store solver choice
        if solver not in VALID_SOLVERS:
            raise ValueError(f"Invalid solver '{solver}'. Valid options are: {list(VALID_SOLVERS)}")
        if solver == 'hydro':
            print("[OK] Compressible solver (hydro: HLLC+PLM) available")
        elif solver == 'hydro-pcm':
            print("[OK] Compressible solver (hydro-pcm: HLLC+PCM) available")
        
        self.solver = solver
        
        # Auto-determine compressible mode based on solver choice
        compressible = solver in ('hydro', 'hydro-pcm')
        
        # Store parallel computation flag
        self.parallel = parallel

        # set the frame of reference. Synodic keeps ES line at 0 longitude.
        # sidereal means Earth moves to increasing longitude with time
        assert (frame == 'synodic' or frame == 'sidereal')
        self.frame = frame
        if frame == 'synodic':
            self.rotation_period = constants['synodic_period']  # Solar Synodic rotation period from Earth.
        elif frame == 'sidereal':
            self.rotation_period = constants['sidereal_period']

        self.v_max = constants['v_max']
        self.nlong = constants['nlong']
        del constants

        # Extract paths of figure and data directories
        dirs = _setup_dirs_()
        self._boundary_dir_ = dirs['boundary_conditions']
        self._data_dir_ = dirs['SURF_data']
        self._figure_dir_ = dirs['SURF_figures']
        self._ephemeris_file = dirs['ephemeris']

        # Setup radial coordinates - in solar radius
        self.r, self.dr, self.rrel, self.nr = radial_grid(r_min=r_min, r_max=r_max)

        # Setup longitude coordinates - in radians.
        self.lon, self.dlon, self.nlon = longitude_grid(lon_out=lon_out, lon_start=lon_start, lon_stop=lon_stop)

        if (self.frame == 'sidereal') & (self.nlon == 1):
            print("Warning: SURF configured for a 1-D run in the sidereal frame. This simulation will not work"
                  "correctly with functions like surf_analysis.get_observer_time_series()")

        # Set up the latitude
        self.latitude = latitude.to(u.rad)

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

        # Establish the Carrington (time stationary) speed boundary condition 
        if np.all(np.isnan(v_boundary)):
            print("Warning: No V boundary conditions supplied. Using default")
            self.v_boundary = 400 * np.ones(self.nlong) * self.kms
            lon_boundary, dlon, nlon = longitude_grid()
            self.v_boundary_lons = lon_boundary * u.rad
        elif not np.all(np.isnan(v_boundary)):
            # check that the implicit time step from vlong is not comparable to the SURF timestep
            assert v_boundary.size < 4600  # this equates to about 9 mins

            self.v_boundary = v_boundary
            # generate the long grid for this v profile
            nv = len(v_boundary)
            dlon = 2 * np.pi / nv
            self.v_boundary_lons = np.arange(dlon / 2, 2 * np.pi - dlon / 2 + dlon / 10, dlon) * u.rad

        # Keep a protected version that isn't processed for use in saving/loading model runs
        self._v_boundary_init_ = self.v_boundary.copy()

        self.track_b = False
        if np.all(np.isnan(b_boundary)):
            self.b_boundary = np.ones(len(self.v_boundary_lons))
            self.b_boundary_lons = self.v_boundary_lons
        elif not np.all(np.isnan(b_boundary)):
            # check that the implicit time step from vlong is not comparable to the SURF timestep
            assert b_boundary.size < 4600  # this equates to about 9 mins

            self.b_boundary = b_boundary
            self.track_b = True
            # Keep a protected version that isn't processed for use in saving/loading model runs
            self._b_boundary_init_ = self.b_boundary.copy()
            # generate the long grid for this b profile
            nb = len(b_boundary)
            dlon = 2 * np.pi / nb
            self.b_boundary_lons = np.arange(dlon / 2, 2 * np.pi - dlon / 2 + dlon / 10, dlon) * u.rad

        # Handle rho and temp boundaries for compressible solver (rotation done later after cr_lon_init is set)
        if compressible:
            if np.all(np.isnan(rho_boundary)):
                # Calculate density using empirical velocity-density relation derived from 
                # OMNI data mapped via Parker nozzle solution
                v_kms = self.v_boundary.to(u.km/u.s).value
                r_inner = r_min.to(u.solRad).value
                n_sw, _ = get_density_temperature_from_velocity(
                    v_kms, r_inner, gamma=self.gamma,
                    cache_id=self._density_temp_cache_id
                )
                
                # Convert density from cm^-3 to kg/m^3
                # n [cm^-3] * m_p [kg] * 1e6 [cm^3/m^3] = rho [kg/m^3]
                m_p = 1.67262192e-27  # proton mass in kg
                self.rho_boundary = n_sw * m_p * 1e6 * (u.kg / u.m**3)
                self.rho_boundary_lons = self.v_boundary_lons
            elif not np.all(np.isnan(rho_boundary)):
                assert rho_boundary.size < 4600  # this equates to about 9 mins
                self.rho_boundary = rho_boundary
                self._rho_boundary_init_ = self.rho_boundary.copy()
                nrho = len(rho_boundary)
                dlon = 2 * np.pi / nrho
                self.rho_boundary_lons = np.arange(dlon / 2, 2 * np.pi - dlon / 2 + dlon / 10, dlon) * u.rad

            if np.all(np.isnan(temp_boundary)):
                # Calculate temperature using empirical velocity-temperature relation
                # derived from OMNI data mapped via Parker nozzle solution
                v_kms = self.v_boundary.to(u.km/u.s).value
                r_inner = self.r[0].to(u.solRad).value
                _, temp_from_velocity = get_density_temperature_from_velocity(
                    v_kms, r_inner, gamma=self.gamma,
                    cache_id=self._density_temp_cache_id)
                self.temp_boundary = temp_from_velocity * u.K
                self.temp_boundary_lons = self.v_boundary_lons
            elif not np.all(np.isnan(temp_boundary)):
                assert temp_boundary.size < 4600  # this equates to about 9 mins
                self.temp_boundary = temp_boundary
                self._temp_boundary_init_ = self.temp_boundary.copy()
                ntemp = len(temp_boundary)
                dlon = 2 * np.pi / ntemp
                self.temp_boundary_lons = np.arange(dlon / 2, 2 * np.pi - dlon / 2 + dlon / 10, dlon) * u.rad

        # add a flag for tracking streaklines
        self.track_streak = False

        # Determine CR number, used for spacecraft/planetary positions
        if np.isnan(cr_num):
            print('No initiation time specified. Defaulting to start of CR2000, 20/2/2003')
            self.cr_num = 2000 * u.dimensionless_unscaled
            cr_lon_init = 0 * u.rad
        else:
            self.cr_num = cr_num * u.dimensionless_unscaled

        # Check cr_lon_init, make sure in 0-2pi range.
        self.cr_lon_init = cr_lon_init.to('rad')
        if (self.cr_lon_init < 0.0 * u.rad) | (self.cr_lon_init > self.twopi * u.rad):
            print("Warning: cr_lon_init={}, outside expected range. Rectifying to 0-2pi.".format(self.cr_lon_init))
            self.cr_lon_init = zerototwopi(self.cr_lon_init.value) * u.rad

            # Compute model UTC initalisation time
        cr_frac = self.cr_num.value + ((self.twopi - self.cr_lon_init.value) / self.twopi)
        self.time_init = sun.carrington_rotation_time(cr_frac)

        # Rotate the boundary condition as required by cr_lon_init.
        lon_shifted = zerototwopi((self.v_boundary_lons - self.cr_lon_init).value)
        id_sort = np.argsort(lon_shifted)
        lon_shifted = lon_shifted[id_sort]
        v_b_shifted = self.v_boundary[id_sort]
        self.v_boundary = np.interp(self.v_boundary_lons.value, lon_shifted, v_b_shifted, period=self.twopi)

        lon_shifted = zerototwopi((self.b_boundary_lons - self.cr_lon_init).value)
        id_sort = np.argsort(lon_shifted)
        lon_shifted = lon_shifted[id_sort]
        b_b_shifted = self.b_boundary[id_sort]
        self.b_boundary = np.interp(self.b_boundary_lons.value, lon_shifted, b_b_shifted, period=self.twopi)

        # Rotate rho and temp boundaries if compressible
        if compressible:
            lon_shifted = zerototwopi((self.rho_boundary_lons - self.cr_lon_init).value)
            id_sort = np.argsort(lon_shifted)
            lon_shifted = lon_shifted[id_sort]
            rho_b_shifted = self.rho_boundary[id_sort]
            rho_unit = self.rho_boundary.unit
            self.rho_boundary = np.interp(self.rho_boundary_lons.value, lon_shifted, rho_b_shifted.value, period=self.twopi) * rho_unit

            lon_shifted = zerototwopi((self.temp_boundary_lons - self.cr_lon_init).value)
            id_sort = np.argsort(lon_shifted)
            lon_shifted = lon_shifted[id_sort]
            temp_b_shifted = self.temp_boundary[id_sort]
            temp_unit = self.temp_boundary.unit
            self.temp_boundary = np.interp(self.temp_boundary_lons.value, lon_shifted, temp_b_shifted.value, period=self.twopi) * temp_unit

        # Compute the buffertime required to spin up SURF, based on minimum speed on the inner boundary
        # and span of radial grid
        self.buffertime = 1.05 * (self.rrel[-1] / self.v_boundary.min()).to(u.day)

        # Preallocate space for the output for the solar wind fields for the cme and ambient solution.
        # Use Fortran order (column-major) for better cache efficiency during solve:
        # - We iterate over longitude (last dimension)
        # - For each longitude, we fill the entire (time, radius) slice
        # - Fortran order makes these slices contiguous in memory
        self.v_grid = np.zeros((self.nt_out, self.nr, self.nlon), order='F') * self.kms
        if self.track_b:
            self.b_grid = np.zeros((self.nt_out, self.nr, self.nlon), order='F')

            # Mesh the spatial coordinates.
        self.lon_grid, self.r_grid = np.meshgrid(self.lon, self.r)

        # Empty list for storing ConeCME objects
        self.cmes = []

        self.track_cmes = track_cmes  # If true, cmes are tracked, which costs a little extra computation time
        self.accel_limit = accel_limit  # If true, no acceleration is applied to speeds >650km/s
        self.compressible = compressible  # If true, use compressible solver instead of incompressible
        
        # Initialize density and temperature grids for compressible solver
        if self.compressible:
            # Use Fortran order for cache-efficient memory access during solve
            self.rho_grid = np.zeros((self.nt_out, self.nr, self.nlon), order='F') * (u.kg / u.m**3)
            self.temp_grid = np.zeros((self.nt_out, self.nr, self.nlon), order='F') * u.K
            # Note: Grids are initialized to zero here and will be populated during solve()

            # Compute typical solar wind values at the inner boundary for use in 
            # setting default CME values. Map typical 1 AU values to the model inner boundary
            # using Parker nozzle relations.
            const_1au = surf_constants()
            r_1au = 215 * u.solRad
            self.v_sw_inner, self.n_sw_inner, self.T_sw_inner = map_properties_parker(
                const_1au['v_sw_1au'], r_1au, self.r[0], 
                const_1au['n_sw_1au'], const_1au['T_sw_1au'], 
                gamma=self.gamma
            )
            self.T_sw_inner = self.T_sw_inner * u.K
            # Convert number density to mass density
            m_p = 1.67262192e-27  # proton mass in kg
            self.rho_sw_inner = self.n_sw_inner * m_p * 1e6 * (u.kg / u.m**3)
            


        # Numpy array of model parameters for parsing to external functions that use numba
        
        self.model_params = np.array([self.dtdr.value, self.alpha, self.r_accel.value,
                                      self.dt_scale.value, self.nt_out, self.nr, self.nlon,
                                      self.r[0].to('km').value,
                                     self.rotation_period.to(u.s).value, int(self.accel_limit),
                                     self.gamma])

        # Process inputs for time dependent boundary conditions, e.g., from in-situ data
        self.input_b_ts = np.nan
        self.input_b_ts_flag = False
        self.input_iscme_ts = np.nan
        self.input_iscme_ts_flag = False
        self.input_v_ts = np.nan * (u.km / u.s)
        self.input_v_ts_flag = False
        self.input_rho_ts = np.nan * (u.kg / u.m**3)
        self.input_rho_ts_flag = False
        self.input_temp_ts = np.nan * u.K
        self.input_temp_ts_flag = False
        if not np.all(np.isnan(input_v_ts)):

            # find the required longitudes
            full_lon_grid, dlon, nlon = longitude_grid()
            xy, x_ind, y_ind = np.intersect1d(self.lon.value, full_lon_grid.value,
                                              return_indices=True)

            self.input_v_ts = input_v_ts[:, y_ind]
            self.model_time = input_t_ts
            self.input_v_ts_flag = True

            # B polarity
            if not np.all(np.isnan(input_b_ts)):
                self.input_b_ts = input_b_ts[:, y_ind]
                self.input_b_ts_flag = True
                self.track_b = True

            # CME flag boundary
            if not np.all(np.isnan(input_iscme_ts)):
                self.input_iscme_ts = input_iscme_ts[:, y_ind]
                self.input_iscme_ts_flag = True

            # Density time series
            if not np.all(np.isnan(input_rho_ts)):
                self.input_rho_ts = input_rho_ts[:, y_ind]
                self.input_rho_ts_flag = True
            elif compressible:
                # Create default density values based on V to maintain constant mass flux
                # Mass flux ρv should be constant, so ρ ∝ 1/v
                # Use the typical rho_sw_inner computed for this model's inner boundary
                # Scale inversely with velocity to maintain constant mass flux
                v_ref = self.v_sw_inner
                rho_ref = self.rho_sw_inner
                # Scale density inversely with velocity to maintain constant mass flux
                self.input_rho_ts = rho_ref * (v_ref / self.input_v_ts)
                self.input_rho_ts_flag = True

            # Temperature time series
            if not np.all(np.isnan(input_temp_ts)):
                self.input_temp_ts = input_temp_ts[:, y_ind]
                self.input_temp_ts_flag = True
            elif compressible:
                # Create default temperature values based on V, same as for 1D case
                # Calculate temperature using empirical velocity-temperature relation
                v_kms = self.input_v_ts.to(u.km/u.s).value
                r_inner = self.r[0].to(u.solRad).value
                _, temp_from_velocity = get_density_temperature_from_velocity(
                    v_kms, r_inner, gamma=self.gamma,
                    cache_id=self._density_temp_cache_id
                )
                self.input_temp_ts = temp_from_velocity * u.K
                self.input_temp_ts_flag = True

        return

    def ts_from_vlong(self):
        """
        Generate the input ambient time series from the v_boundary (lon) values
        Returns:
            None
        """

        buffersteps = np.fix(self.buffertime.to(u.s) / self.dt)
        buffertime = buffersteps * self.dt
        model_time = np.arange(-buffertime.value, (self.simtime.to('s') + self.dt).value, self.dt.value) * self.dt.unit
        dlondt = self.twopi * self.dt / self.rotation_period
        # OPTIMIZATION: Use actual nlon instead of always generating 128
        nlon = self.nlon  # Use the actual model longitude count
        self.model_time = model_time

        # How many radians of Carrington rotation in this simulation length
        simlon = self.twopi * self.simtime / self.rotation_period
        # How many radians of Carrington rotation in the spin up period
        bufferlon = self.twopi * buffertime / self.rotation_period

        # Variables to store the input conditions.
        self.input_v_ts = np.nan * np.ones((model_time.size, nlon)) * (u.km / u.s)
        if self.track_b:
            self.input_b_ts = np.nan * np.ones((model_time.size, nlon))
        if self.compressible:
            self.input_rho_ts = np.nan * np.ones((model_time.size, nlon)) * (u.kg / u.m**3)
            self.input_temp_ts = np.nan * np.ones((model_time.size, nlon)) * u.K

        # Loop through model longitudes and compute boundary conditions at each radial profile.
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
            loninit = zerototwopi(lonint)
            
            # Interpolate the inner boundary speed to this higher resolution
            vinit = np.interp(loninit, self.v_boundary_lons.value, self.v_boundary.value, period=2 * np.pi)
            
            # convert from cr longitude to timesolve
            vinput = np.flipud(vinit) * (u.km / u.s)
            # Store the input series
            self.input_v_ts[:, i] = vinput

            if self.track_b:
                binit = np.interp(loninit, self.b_boundary_lons.value, self.b_boundary, period=2 * np.pi)
                # convert from cr longitude to timesolve
                binput = np.flipud(binit)
                # Store the input series
                self.input_b_ts[:, i] = binput

            if self.compressible:
                # Interpolate density boundary condition
                rhoinit = np.interp(loninit, self.rho_boundary_lons.value, self.rho_boundary.value, period=2 * np.pi)
                # convert from cr longitude to timesolve
                rhoinput = np.flipud(rhoinit) * self.rho_boundary.unit
                # Store the input series
                self.input_rho_ts[:, i] = rhoinput

                # Interpolate temperature boundary condition
                tempinit = np.interp(loninit, self.temp_boundary_lons.value, self.temp_boundary.value, period=2 * np.pi)
                # convert from cr longitude to timesolve
                tempinput = np.flipud(tempinit) * self.temp_boundary.unit
                # Store the input series
                self.input_temp_ts[:, i] = tempinput

        return
    
    def process_longitude(self, i, n_cme, n_hcs_max, streak_times):
        """
        Process a single longitude slice in the SURF simulation.
        This helper function is used for parallel execution across longitudes.
        Routes to the appropriate solver based on self.solver setting.
        
        Args:
            i: Longitude index to process
            n_cme: Number of CMEs
            n_hcs_max: Maximum number of HCS particles
            streak_times: Streakline timing data
            
        Returns:
            Tuple of (i, v, cme_r_bounds, cme_v_bounds, hcs_r, streak_r, rho_out, temp_out)
        """
        # Route based on configured solver family.
        
        if self.compressible:
            # Use solve_radial_compressible with selected Riemann solver
            return self._process_longitude_compressible(i, n_cme, n_hcs_max, streak_times)
        else:
            # HUXt solver uses solve_radial
            return self._process_longitude_builtin(i, n_cme, n_hcs_max, streak_times)
    
    def _process_longitude_builtin(self, i, n_cme, n_hcs_max, streak_times):
        """
        Process a longitude using the built-in solve_radial (huxt solver).
        
        Args:
            i: Longitude index to process
            n_cme: Number of CMEs
            n_hcs_max: Maximum number of HCS particles
            streak_times: Streakline timing data
            
        Returns:
            Tuple of (i, v, cme_r_bounds, cme_v_bounds, hcs_r, streak_r, rho_out, temp_out)
        """
        # check if there is b polarity data
        if self.track_b:
            bslice = self.input_b_ts[:, i]
        else:
            bslice = self.input_v_ts[:, i] * np.nan

        # Prepare density and temperature inputs for compressible solver
        if self.compressible:
            rhoslice = self.input_rho_ts[:, i].value
            tempslice = self.input_temp_ts[:, i].value
        else:
            # Pass dummy arrays to avoid None issues in Numba
            rhoslice = np.zeros(len(self.model_time))
            tempslice = np.zeros(len(self.model_time))

        # actually run the solver
        v, cme_r_bounds, cme_v_bounds, hcs_r, streak_r, rho_out, temp_out = solve_radial(
                                                                      self.input_v_ts[:, i].value,
                                                                      bslice,
                                                                      self.input_iscme_ts[:, i],
                                                                      self.model_time,
                                                                      self.rrel.value,
                                                                      self.model_params,
                                                                      n_cme, n_hcs_max,
                                                                      streak_times[i, :, :, :],
                                                                      rhoinput=rhoslice,
                                                                      tempinput=tempslice)
        
        return (i, v, cme_r_bounds, cme_v_bounds, hcs_r, streak_r, rho_out, temp_out)
    
    def _process_longitude_compressible(self, i, n_cme, n_hcs_max, streak_times):
        """
        Process a longitude using the compressible solver.
        
        Uses self.solver to select the internal method string.
        
        Args:
            i: Longitude index to process
            n_cme: Number of CMEs
            n_hcs_max: Maximum number of HCS particles
            streak_times: Streakline timing data
            
        Returns:
            Tuple of (i, v, cme_r_bounds, cme_v_bounds, hcs_r, streak_r, rho_out, temp_out)
        """
        # Compute radial grid in km for compressible solver
        rgrid_km = (self.rrel.value - self.rrel.value[0]) * 695700.0 + self.r[0].to('km').value
        # Get boundary conditions for this longitude (avoid .value for speed)
        v_bc_kms = self.input_v_ts[:, i].value  # km/s
        rho_bc_kgm3 = self.input_rho_ts[:, i].value  # kg/m³
        T_bc_K = self.input_temp_ts[:, i].value  # K
        
        # OPTIMIZATION: Skip particle tracking setup if nothing to track
        num_particles = 0
        particle_injection_rate = None
        particle_release_rate = None
        hcs_polarities = []  # Initialize HCS polarities list
        
        # Only set up tracking if actually needed
        if (self.track_cmes and n_cme > 0) or (self.track_b and n_hcs_max > 0) or self.track_streak:
            num_particles = {}
            particle_injection_rate = {}
            particle_release_rate = {}
            
            # CME particles: track leading and trailing edges
            if self.track_cmes and n_cme > 0:
                for cme_id in range(n_cme):
                    # Find times when this CME crosses this longitude
                    cme_mask = (self.input_iscme_ts[:, i] == cme_id + 1)
                    if np.any(cme_mask):
                        # Leading edge injected at start of CME
                        leading_idx = np.where(cme_mask)[0][0]
                        t_leading = self.model_time[leading_idx].value
                        
                        # Trailing edge injected at end of CME
                        trailing_idx = np.where(cme_mask)[0][-1]
                        t_trailing = self.model_time[trailing_idx].value
                        
                        num_particles[f'cme_{cme_id}_leading'] = 1
                        num_particles[f'cme_{cme_id}_trailing'] = 1
                        
                        particle_injection_rate[f'cme_{cme_id}_leading'] = [t_leading]
                        particle_release_rate[f'cme_{cme_id}_leading'] = [t_leading]
                        
                        # Fix for CME inner boundary: inject trailing edge at START of CME
                        # but hold it at the boundary until the END of the CME (release time)
                        particle_injection_rate[f'cme_{cme_id}_trailing'] = [t_leading]
                        particle_release_rate[f'cme_{cme_id}_trailing'] = [t_trailing]
            
            # HCS particles: inject at each polarity change
            # Also track the polarity direction for each crossing
            if self.track_b and n_hcs_max > 0:
                hcs_times = []
                b_input = self.input_b_ts[:, i]
                for t in range(1, len(b_input)):
                    diff = b_input[t] - b_input[t-1]
                    if diff != 0:  # Polarity change
                        t_hcs = self.model_time[t].value
                        # Track HCS crossings including during spin-up period
                        hcs_times.append(t_hcs)
                        # Store polarity direction: +1 if B increases, -1 if B decreases
                        if diff > 0:
                            hcs_polarities.append(1.0)
                        else:
                            hcs_polarities.append(-1.0)
                
                if len(hcs_times) > 0:
                    num_particles['hcs'] = len(hcs_times)
                    particle_injection_rate['hcs'] = hcs_times
            
            # Streakline particles: inject according to streak_times
            if self.track_streak:
                # streak_times has shape (nlon, n_streaks, n_rots, 2)
                # where last dim is [time_index, rotation_number]
                streak_data = streak_times[i, :, :, 0]  # Get time indices for this longitude
                n_streaks = streak_data.shape[0]
                n_rots = streak_data.shape[1]
                
                for istreak in range(n_streaks):
                    for irot in range(n_rots):
                        time_idx = streak_data[istreak, irot]
                        if not np.isnan(time_idx):
                            streak_name = f'streak_{istreak}_rot_{irot}'
                            t_inject = self.model_time[int(time_idx)]
                            t_inject = t_inject.value
                            num_particles[streak_name] = 1
                            particle_injection_rate[streak_name] = [t_inject]
            
            # If no particles were actually added, revert to no tracking
            if len(num_particles) == 0:
                num_particles = 0
                particle_injection_rate = None
        
        # Call compressible solver function with selected Riemann solver and particle tracking
        # Strip units from time arrays for solver
        model_time_sec = self.model_time.value
        time_out_sec = self.time_out.value
        
        v_out_kms, rho_out_kgm3, temp_out_K, particle_data = solve_radial_compressible(
            v_bc_kms=v_bc_kms,
            rho_bc_kgm3=rho_bc_kgm3,
            T_bc_K=T_bc_K,
            model_time=model_time_sec,
            time_out=time_out_sec,
            r_grid=rgrid_km,
            gamma=self.gamma,
            nt_out=self.nt_out,
            nr=self.nr,
            riemann=_compressible_method_from_solver(self.solver),
            verbose=False,  # Suppress detailed solver output in parallel mode
            num_particles=num_particles,
            particle_injection_rate=particle_injection_rate,
            particle_release_rate=particle_release_rate
        )
        
        # Extract particle positions at output times
        cme_particles_r_out = np.full((n_cme, self.nt_out, 2), np.nan)
        cme_particles_v_out = np.full((n_cme, self.nt_out, 2), np.nan)  # Compressible solver doesn't track velocity, fill with NaN
        hcs_particles_r_out = np.full((n_hcs_max, self.nt_out, 2), np.nan)
        
        # Initialize streakline array
        if self.track_streak:
            streak_data = streak_times[i, :, :, 0]
            n_streaks = streak_data.shape[0]
            n_rots = streak_data.shape[1]
            streak_particles_r_out = np.full((self.nt_out, n_streaks, n_rots), np.nan)
        else:
            streak_particles_r_out = np.full((self.nt_out, 1, 1), np.nan)
        
        if particle_data is not None and 'groups' in particle_data:
            groups = particle_data['groups']
            
            # Get time_out as plain array
            time_out_sec = self.time_out.value
            
            # Process CME particles
            if self.track_cmes:
                for cme_id in range(n_cme):
                    leading_key = f'cme_{cme_id}_leading'
                    trailing_key = f'cme_{cme_id}_trailing'
                    
                    if leading_key in groups:
                        # Compressible solver returns 1D trajectory arrays
                        r_traj = groups[leading_key]['r']
                        t_traj = groups[leading_key]['t']
                        valid_mask = ~np.isnan(r_traj)
                        if np.any(valid_mask):
                            r_valid = r_traj[valid_mask]
                            t_valid = t_traj[valid_mask]
                            r_out = np.interp(time_out_sec, t_valid, r_valid, 
                                             left=np.nan, right=np.nan)
                            cme_particles_r_out[cme_id, :, 0] = r_out
                    
                    if trailing_key in groups:
                        # Compressible solver returns 1D trajectory arrays
                        r_traj = groups[trailing_key]['r']
                        t_traj = groups[trailing_key]['t']
                        valid_mask = ~np.isnan(r_traj)
                        if np.any(valid_mask):
                            r_valid = r_traj[valid_mask]
                            t_valid = t_traj[valid_mask]
                            r_out = np.interp(time_out_sec, t_valid, r_valid,
                                             left=np.nan, right=np.nan)
                            cme_particles_r_out[cme_id, :, 1] = r_out
            
            # Process HCS particles
            if self.track_b:
                # Try both storage formats: single 'hcs' group or individual 'hcs_X' groups
                if 'hcs' in groups:
                    # Original format: single 'hcs' group with 2D arrays
                    hcs_group = groups['hcs']
                    n_hcs_this_lon = hcs_group['n_particles']
                    
                    for ihcs in range(min(n_hcs_this_lon, n_hcs_max)):
                        # Try 2D indexing first (built-in solver format)
                        try:
                            r_traj = hcs_group['r'][ihcs, :]
                            t_traj = hcs_group['t'][ihcs, :]
                        except (IndexError, TypeError):
                            # If 2D indexing fails, try 1D (compressible solver format)
                            r_traj = hcs_group['r']
                            t_traj = hcs_group['t']
                        
                        valid_mask = ~np.isnan(r_traj)
                        if np.any(valid_mask):
                            r_valid = r_traj[valid_mask]
                            t_valid = t_traj[valid_mask]
                            r_out = np.interp(time_out_sec, t_valid, r_valid,
                                             left=np.nan, right=np.nan)
                            hcs_particles_r_out[ihcs, :, 0] = r_out
                            
                            # Store polarity sign in second component
                            if ihcs < len(hcs_polarities):
                                hcs_particles_r_out[ihcs, :, 1] = hcs_polarities[ihcs]
                            else:
                                hcs_particles_r_out[ihcs, :, 1] = np.nan
                else:
                    # New format: individual 'hcs_X' groups (compressible solver)
                    for ihcs in range(n_hcs_max):
                        hcs_key = f'hcs_{ihcs}'
                        if hcs_key in groups:
                            # Compressible solver returns 1D trajectory arrays
                            r_traj = groups[hcs_key]['r']
                            t_traj = groups[hcs_key]['t']
                            valid_mask = ~np.isnan(r_traj)
                            if np.any(valid_mask):
                                r_valid = r_traj[valid_mask]
                                t_valid = t_traj[valid_mask]
                                r_out = np.interp(time_out_sec, t_valid, r_valid,
                                                 left=np.nan, right=np.nan)
                                hcs_particles_r_out[ihcs, :, 0] = r_out
                                
                                # Store polarity sign in second component
                                if ihcs < len(hcs_polarities):
                                    hcs_particles_r_out[ihcs, :, 1] = hcs_polarities[ihcs]
                                else:
                                    hcs_particles_r_out[ihcs, :, 1] = np.nan
            
            # Process streakline particles
            if self.track_streak:
                for istreak in range(n_streaks):
                    for irot in range(n_rots):
                        streak_name = f'streak_{istreak}_rot_{irot}'
                        if streak_name in groups:
                            # Compressible solver returns 1D trajectory arrays (already converted in solve_radial_compressible)
                            r_traj = groups[streak_name]['r']
                            t_traj = groups[streak_name]['t']
                            valid_mask = ~np.isnan(r_traj)
                            if np.any(valid_mask):
                                r_valid = r_traj[valid_mask]
                                t_valid = t_traj[valid_mask]
                                r_out = np.interp(time_out_sec, t_valid, r_valid,
                                                 left=np.nan, right=np.nan)
                                streak_particles_r_out[:, istreak, irot] = r_out
        
        # Return tuple consistent with builtin solver format
        return (i, v_out_kms, cme_particles_r_out, cme_particles_v_out, 
                hcs_particles_r_out, streak_particles_r_out, rho_out_kgm3, temp_out_K)
    
    def set_gamma(self, new_gamma):
        """
        Update the adiabatic index gamma and recalculate temperature boundary conditions.
        
        This method properly updates gamma throughout the model:
        1. Updates self.gamma
        2. Updates model_params array used by solvers  
        3. Recalculates temperature boundary using Lopez & Freeman (1986) relation
        
        Args:
            new_gamma: New adiabatic index value (typically 5/3 for monoatomic gas)
        """
        self.gamma = new_gamma
        
        # Update gamma in model_params (index 10)
        if hasattr(self, 'model_params'):
            self.model_params[10] = new_gamma
        
        # Recalculate temperature boundary with new gamma if compressible
        if hasattr(self, 'compressible') and self.compressible:
            if hasattr(self, 'v_boundary') and hasattr(self, 'temp_boundary'):
                # Recalculate temperature using new gamma value
                v_kms = self.v_boundary.to(u.km/u.s).value
                r_inner = self.r[0].to(u.solRad).value
                _, temp_from_velocity = get_density_temperature_from_velocity(
                    v_kms, r_inner, gamma=new_gamma,
                    cache_id=self._density_temp_cache_id
                )
                self.temp_boundary = temp_from_velocity * u.K
    
    def solve(self, cme_list, streak_carr=np.array([])*u.rad, save=False, tag=''):
        """
        Solve SURF for the provided longitudinal boundary conditions and cme list. Updates the SURF.v_grid
        Args:
            cme_list: A list of ConeCME instances to use in solving SURF
            streak_carr: An numpy array of Carrington longitudes from which to trace streaklines, units of radians.
            save: Boolean, if True saves model output to HDF5 file
            tag: String, appended to the filename of saved solution.
        Returns:
            None
        """
        
        # Update model_params with current alpha and gamma values
        # (in case user changed them after initialization)
        self.model_params = np.array([self.dtdr.value, self.alpha, self.r_accel.value,
                                      self.dt_scale.value, self.nt_out, self.nr, self.nlon,
                                      self.r[0].to('km').value,
                                      self.rotation_period.to(u.s).value, int(self.accel_limit),
                                      self.gamma])

        # ======================================================================
        # Generate ambient solar wind time series
        # ======================================================================
        # If the input time series has not been prescribed,
        # Generate it from v(long)
        if not self.input_v_ts_flag:
            self.ts_from_vlong()

        # ======================================================================
        # Process CME list
        # ======================================================================
        # Make a copy of the CME list objects so that the originals are not modified
        input_cme_list = copy.deepcopy(cme_list)

        # Quality control the CME list. Check:
        # Only ConeCMEs in list
        # Make sidereal correction if necessary
        # Check launch time is not in spin up period before simulation time.
        cme_list_checked = []
        for cme in input_cme_list:

            if isinstance(cme, ConeCME):

                if self.frame == 'sidereal':
                    # if the solution is in the sideral frame, adjust CME longitudes
                    earthpos = self.get_observer('EARTH')
                    # time and longitude from start of run
                    dt_t0 = (earthpos.time - self.time_init).to(u.s)
                    dlon_t0 = earthpos.lon_hae - earthpos.lon_hae[0]
                    # find the CME hae longitude relative to the run start
                    cme_hae = np.interp(cme.t_launch.to(u.s).value,
                                        dt_t0.value, dlon_t0)
                    # adjust the CME HEEQ longitude accordingly
                    cme.longitude_surf = zerototwopi(cme.longitude + cme_hae) * u.rad
                else:
                    cme.longitude_surf = cme.longitude

                if cme.t_launch >= 0*u.s:
                    # add the CME to the list
                    cme_list_checked.append(cme)
                else:
                    print(f"Warning: ConeCME had negative t_launch ({cme.t_launch}), which is not allowed.")
                    print("Warning: This ConeCME object was not passed into the SURF solver")
            else:
                print("Warning: cme_list contained objects other than ConeCME instances. These were excluded")

        self.cmes = cme_list_checked

        # If CMEs parsed, get an array of their parameters for using with the solver (which doesn't do classes)
        if len(self.cmes) > 0:
            cme_params = [cme.parameter_array(self) for cme in self.cmes]
            cme_params = np.array(cme_params)
            # Sort the CMEs in launch order.
            id_sort = np.argsort(cme_params[:, 0])
            cme_params = cme_params[id_sort]
            # Also sort the list of ConeCMEs so that it corresponds ot cme_params
            self.cmes = [self.cmes[i] for i in id_sort]
        else:
            # create dummy cme
            dummy_cme = ConeCME()
            cme_params = np.nan * np.zeros((1, len(dummy_cme.parameter_array(self))))

        # sanity check the CME initial height is the same as the model inner boundary
        if len(self.cmes) > 0:
            for cme in self.cmes:
                assert (self.r[0] == cme.initial_height)

        # check CME speeds aren't so fast they will butt up agains the CFL condition.
        if len(self.cmes) > 0:
            constants = surf_constants()
            v_max = constants['v_max']
            for cme in self.cmes:

                if cme.v >= v_max:
                    raise ValueError(f'CME speed {cme.v} is larger than allowed for CFL limit of {v_max}')
                elif cme.v >= 0.8 * v_max:
                    print(f'Warning: CME speed of {cme.v} is close to CFL limit of {v_max}. Simulation may be unstable')

        # ======================================================================
        # Create ambient (pre-CME) density and temperature time series
        # ======================================================================
        # Store copies of the ambient conditions before CMEs are added
        # This ensures CME perturbations are multiples of ambient values
        if self.compressible:
            self.ambient_rho_ts = self.input_rho_ts.copy()
            self.ambient_temp_ts = self.input_temp_ts.copy()
        
        # ======================================================================
        # Add CMEs
        # ======================================================================
        # See if the cmes-flag input time series has been prescribed
        if self.input_iscme_ts_flag:
            # CME input has been parsed as input - set up some dummy coneCME's
            # to hold the CME tracking data
            n_cme = np.nanmax(self.input_iscme_ts)
            # Create dummy CME list to sort the boundaries
            self.cmes = []
            for n in range(0, n_cme):
                cme = ConeCME(t_launch=0 * u.s, longitude=0 * u.deg,
                              width=0 * u.deg, v=0 * self.kms, thickness=0 * u.solRad)
                self.cmes.append(cme)
        else:
            # CME input has not been specified, but ConeCME's may have been input
            # So configure input_iscme_ts from the ConeCMES.
            self.input_iscme_ts = 0 * np.ones((self.model_time.size,
                                               self.nlon), dtype='int')

            n_cme = len(self.cmes)
            if n_cme > 0:
                # Loop through model longitudes and add the CMEs
                for i in range(self.lon.size):
                    if self.lon.size == 1:
                        lon_out = self.lon.value
                    else:
                        lon_out = self.lon[i].value

                    # Add the CMEs to the input series
                    if self.compressible:
                        v, isincme, rho, temp = add_cmes_to_input_series(
                           
                            self.input_v_ts[:, i].value,
                            self.model_time, lon_out,
                            self.r[0].to('km').value, cme_params,
                            self.latitude.value,
                            rhoinput=self.input_rho_ts[:, i].value,
                            tempinput=self.input_temp_ts[:, i].value,
                            rho_ambient=self.ambient_rho_ts[:, i].value,
                            temp_ambient=self.ambient_temp_ts[:, i].value,
                            compressible=True)
                        self.input_v_ts[:, i] = v * (u.km / u.s)
                        self.input_rho_ts[:, i] = rho * (u.kg / u.m**3)
                        self.input_temp_ts[:, i] = temp * u.K
                    else:
                        v, isincme, rho, temp = add_cmes_to_input_series(
                            self.input_v_ts[:, i].value,
                            self.model_time, lon_out,
                            self.r[0].to('km').value, cme_params,
                            self.latitude.value,
                            rhoinput=None,
                            tempinput=None,
                            rho_ambient=None,
                            temp_ambient=None,
                            compressible=False)
                        self.input_v_ts[:, i] = v * (u.km / u.s)
                    self.input_iscme_ts[:, i] = isincme

        # Set up the CME test particle position field
        self.cme_particles_r = np.full((n_cme, self.nt_out, 2, self.nlon), np.nan) * u.dimensionless_unscaled
        self.cme_particles_v = np.full((n_cme, self.nt_out, 2, self.nlon), np.nan) * u.dimensionless_unscaled

        # ======================================================================    
        # Set up the HCS test particle position field
        # ======================================================================
        n_hcs_max = 0
        if self.track_b:
            # convert Br to polarity
            self.input_b_ts = np.sign(self.input_b_ts)

            # find the number of HCS crossings at long
            n_hcs = np.zeros(self.lon.size)
            for i in range(self.lon.size):
                db = self.input_b_ts[:-1, i] - self.input_b_ts[1:, i]
                n_hcs[i] = (abs(db) > 0.01).sum()

            # create variables to store the HCS positions
            n_hcs_max = int(max(n_hcs)) + 1
            self.hcs_particles_r = np.full((n_hcs_max, self.nt_out, 2, self.nlon), np.nan) * u.dimensionless_unscaled

        # ======================================================================
        # Set up the streak lines
        # ======================================================================
        if isinstance(streak_carr, u.Quantity) & (streak_carr.size > 0):
            self.track_streak = True
            self.streak_lon_r0 = np.ones((len(self.time_out), len(streak_carr)))

            # compute the number of intersections with each longitude
            time_from_start = self.model_time - self.model_time[0]
            nrot = int(np.ceil(time_from_start[-1] / self.rotation_period) + 1)

            streak_times = np.ones((self.nlon, len(streak_carr), nrot, 2)) * np.nan
            # work out the source longitude at start of spin up 
            for istreak, carr in enumerate(streak_carr):

                # convert from Carrington longitude to model lon at t=0
                lon_src = zerototwopi((carr - self.cr_lon_init)) * u.rad

                # adjust the source longitude for the spin-up time
                dl_spinup = 2 * np.pi * self.buffertime.to(u.s) / self.rotation_period
                lon_0 = zerototwopi(lon_src.to(u.rad).value - dl_spinup)

                # compute the model longitude of the streak line footpoint with time
                streak_lon_t = (lon_0 + time_from_start * 2 * np.pi / self.rotation_period) * u.rad

                # save the streakline footpoint longitude on the model time step
                self.streak_lon_r0[:, istreak] = np.interp(self.time_out, self.model_time,
                                                           streak_lon_t, period=2 * np.pi)

                # Handle both scalar and array longitude cases
                lon_array = [self.lon] if self.lon.size == 1 else self.lon
                for ilon, lon in enumerate(lon_array):
                    for irot in range(0, nrot):
                        # find the time index of the streakline at the given lon
                        id_in = np.argmin(abs(streak_lon_t -
                                              (lon - self.dlon / 2 + 2 * np.pi * irot * u.rad)))
                        id_out = np.argmin(abs(streak_lon_t -
                                               (lon + self.dlon / 2 + 2 * np.pi * irot * u.rad)))

                        # if id_out ==0, then the streakline hasn't made it to that lon
                        if id_in > 0 and id_in < len(time_from_start) - 1:
                            streak_times[ilon, istreak, irot, 0] = id_in
                            streak_times[ilon, istreak, irot, 1] = id_out

            # self.streak_times = streak_times
            self.streak_particles_r = np.zeros((self.nt_out,
                                                len(streak_carr), nrot,
                                                self.nlon)) * np.nan * u.dimensionless_unscaled

        else:
            streak_times = np.ones((self.nlon, 1, 1, 1)) * np.nan

        # ======================================================================
        # Print solver information
        # ======================================================================
        if self.compressible:
            import time
            solve_start = time.time()
            compressible_method = _compressible_method_from_solver(self.solver)
            
            print("\n" + "="*70)
            print(f"USING COMPRESSIBLE SOLVER: {self.solver.upper()}")
            print("="*70)
            print(f"Method: {compressible_method}")
            print(f"Frame: {self.frame}")
            print(f"Parallel: {self.parallel}")
            if self.parallel:
                print(f"\n⚠ WARNING: Parallel execution for compressible solver is typically SLOWER than serial")
                print(f"  Recommended: Set parallel=False for better performance")
            print("="*70 + "\n")
        
        # ======================================================================
        # Solve the time series at each longitude (HUXT and COMPRESSIBLE SOLVERS)
        # ======================================================================
        # ======================================================================
        # Solve the time series at each longitude (ALL SOLVERS)
        # ======================================================================
        # Note: Grids are already in Fortran order from initialization, which makes
        # the [:, :, i] slices contiguous in memory for cache-efficient access
        # This section handles all solvers using unified process_longitude

        if True:  # Always execute
            if self.parallel:
                # Parallel execution using joblib
                results = Parallel(n_jobs=-1, backend='threading')(
                    delayed(self.process_longitude)(i, n_cme, n_hcs_max, streak_times) 
                    for i in range(self.lon.size)
                )
                
                # Unpack results into grids
                for i, v, cme_r_bounds, cme_v_bounds, hcs_r, streak_r, rho_out, temp_out in results:
                    self.v_grid[:, :, i] = v * self.kms
                    self.cme_particles_r[:, :, :, i] = cme_r_bounds * u.dimensionless_unscaled
                    self.cme_particles_v[:, :, :, i] = cme_v_bounds * u.dimensionless_unscaled
                    if self.track_b:
                        self.hcs_particles_r[:, :, :, i] = hcs_r * u.dimensionless_unscaled
                    if self.track_streak:
                        self.streak_particles_r[:, :, :, i] = streak_r * u.dimensionless_unscaled
                    
                    # Save density and temperature output for compressible solver
                    if self.compressible:
                        self.rho_grid[:, :, i] = rho_out * (u.kg / u.m**3)
                        self.temp_grid[:, :, i] = temp_out * u.K
            else:
                # Serial execution (original loop)
                for i in range(self.lon.size):
                    i, v, cme_r_bounds, cme_v_bounds, hcs_r, streak_r, rho_out, temp_out = self.process_longitude(
                        i, n_cme, n_hcs_max, streak_times
                    )
                    
                    # Save the output at each longitude
                    self.v_grid[:, :, i] = v * self.kms
                    self.cme_particles_r[:, :, :, i] = cme_r_bounds * u.dimensionless_unscaled
                    self.cme_particles_v[:, :, :, i] = cme_v_bounds * u.dimensionless_unscaled
                    if self.track_b:
                        self.hcs_particles_r[:, :, :, i] = hcs_r * u.dimensionless_unscaled
                    if self.track_streak:
                        self.streak_particles_r[:, :, :, i] = streak_r * u.dimensionless_unscaled
                    
                    # Save density and temperature output for compressible solver
                    if self.compressible:
                        self.rho_grid[:, :, i] = rho_out * (u.kg / u.m**3)
                        self.temp_grid[:, :, i] = temp_out * u.K

        # Update CMEs positions by tracking through the solution.
        if self.track_cmes:
            updated_cmes = []
            for cme_num, cme in enumerate(self.cmes):
                cme._track_(self, cme_num)
                updated_cmes.append(cme)

            self.cmes = updated_cmes

        # Create the bgrid
        if self.track_b:

            if self.nlon == 1:
                lons = np.array([self.lon.value])
            else:
                lons = self.lon.value

            self.b_grid = bgrid_from_hcs(self.hcs_particles_r, self.input_b_ts,
                                         self.model_time.value,
                                         self.time_out.value,
                                         self.r.to(u.km).value, lons)

        # Convert grids back to C order (row-major) for compatibility with rest of codebase
        # This ensures all downstream analysis and plotting code works as expected
        self.v_grid = np.ascontiguousarray(self.v_grid.value) * self.kms
        if self.compressible:
            self.rho_grid = np.ascontiguousarray(self.rho_grid.value) * (u.kg / u.m**3)
            self.temp_grid = np.ascontiguousarray(self.temp_grid.value) * u.K

        if save:
            if tag == '':
                print("Warning, blank tag means file likely to be overwritten")
            self.save(tag=tag)
        return

    def save(self, tag=''):
        """
        Save all model fields output to a HDF5 file.
        Args:
            tag: identifying string to append to the filename
        Returns:
             out_filepath: Full path to the saved file.
        """
        # Open up hdf5 data file for the HI flow stats
        filename = "SURF_CR{:03d}_{}.hdf5".format(np.int32(self.cr_num.value), tag)
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

                if k == "frame":
                    cmegrp.create_dataset(k, data=v)

                if k == 'label':
                    if isinstance(v, str):
                        cmegrp.create_dataset(k, data=v)
                    else:
                        cmegrp.create_dataset(k, data='None')


                if k not in ["coords", "frame", "label"]:
                    # check if the CME property has a value (new BOOLs do not)
                    if hasattr(v, 'value'):
                        dset = cmegrp.create_dataset(k, data=v.value)
                        dset.attrs['unit'] = v.unit.to_string()
                    else:
                        dset = cmegrp.create_dataset(k, data=v)
                        dset.attrs['unit'] = 'None'

                out_file.flush()
                # Now handle the dictionary of CME boundary coordinates coords > time_out > position
                if k == "coords":
                    coordgrp = cmegrp.create_group(k)
                    for time, position in v.items():
                        time_label = "t_out_{:03d}".format(time)
                        timegrp = coordgrp.create_group(time_label)
                        for pos_label, pos_data in position.items():
                            if pos_label == 'time':
                                timegrp.create_dataset(pos_label, data=pos_data.isot)
                            else:
                                dset = timegrp.create_dataset(pos_label, data=pos_data.value)
                                dset.attrs['unit'] = pos_data.unit.to_string()

                            out_file.flush()

        # Loop over the attributes of model instance and save select keys/attributes.
        keys = ['cr_num', 'cr_lon_init', 'simtime', 'dt', 'v_max', 'r_accel', 'alpha',
                'dt_scale', 'time_out', 'dt_out', 'r', 'dr', 'lon', 'dlon', 'r_grid', 'lon_grid',
                'v_grid', 'latitude', 'v_boundary', '_v_boundary_init_', 'cme_particles_r', 'cme_particles_v',
                'streak_particles_r', 'streak_lon_r0', 'hcs_particles_r', 'frame', 'track_cmes', 'accel_limit',
                'track_b', 'track_streak', 'compressible', 'solver']

        # Handle keys to magnetic field arrays seperately
        mag_keys = ['_b_boundary_init_', 'b_boundary_lons', 'b_boundary', 'b_grid']

        # Handle keys to compressible solver arrays separately
        compressible_keys = ['_rho_boundary_init_', 'rho_boundary_lons', 'rho_boundary', 
                             '_temp_boundary_init_', 'temp_boundary_lons', 'temp_boundary',
                             'rho_grid', 'temp_grid']

        for k, v in self.__dict__.items():

            if k in keys:
                if isinstance(v, str):
                    dset = out_file.create_dataset(k, data=v)
                elif isinstance(v, u.Quantity):
                    dset = out_file.create_dataset(k, data=v.value)
                    dset.attrs['unit'] = v.unit.to_string()
                elif isinstance(v, np.ndarray):
                    dset = out_file.create_dataset(k, data=v)
                    dset.attrs['unit'] = u.dimensionless_unscaled.to_string()
                elif isinstance(v, bool):
                    dset = out_file.create_dataset(k, data=int(v))
                    dset.attrs['unit'] = "bool"

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

            # Only save the magnetic field arrays if they exist.
            if self.track_b:
                if k in mag_keys:
                    if isinstance(v, np.ndarray):
                        dset = out_file.create_dataset(k, data=v)
                        dset.attrs['unit'] = u.dimensionless_unscaled.to_string()
                    elif isinstance(v, u.Quantity):
                        dset = out_file.create_dataset(k, data=v.value)
                        dset.attrs['unit'] = v.unit.to_string()

                    if k == 'b_grid':
                        dset.dims[0].label = 'time'
                        dset.dims[1].label = 'radius'
                        dset.dims[2].label = 'longitude'

                    out_file.flush()

            # Only save the compressible solver arrays if compressible is True
            if self.compressible:
                if k in compressible_keys:
                    if isinstance(v, np.ndarray):
                        dset = out_file.create_dataset(k, data=v)
                        dset.attrs['unit'] = u.dimensionless_unscaled.to_string()
                    elif isinstance(v, u.Quantity):
                        dset = out_file.create_dataset(k, data=v.value)
                        dset.attrs['unit'] = v.unit.to_string()
                    
                    # Add dimension labels for grid arrays
                    if k in ['rho_grid', 'temp_grid']:
                        dset.dims[0].label = 'time'
                        dset.dims[1].label = 'radius'
                        dset.dims[2].label = 'longitude'

                    out_file.flush()

        out_file.close()
        return out_filepath

    def get_observer(self, body):
        """
        Returns an instance of the Observer class, giving the HEEQ and Carrington coordinates at each model timestep.
        This is only well-defined if the model was initialised with a Carrington rotation number.
        Args:
            body: String specifying which body to look up. Valid bodies are Earth, Venus, Mercury, STA, and STB.
        Returns:
            obs: An Observer instance for body at times from SURF.time_init + SURF.time_out
        """
        times = self.time_init + self.time_out
        obs = Observer(body, times)
        return obs


class SURF3d:
    """
    A class containing a list of SURF classes, to enable mutliple latitudes to
    be simulated, plotted, animated, etc. together
    
    Attributes inherited from SURF. Additional:
        lat: The list of latitudes of individual SURF runs, in radians from the equator
        nlat: The number of latitudes simulated
        SURFlat: List of individual SURF model classes at each latitude
        v_in: a list of Carrington longitude solar wind profiles at each simulated latitude
        br_in: a list of Carrington longitude Br profiles at each simulated latitude
        
    
    """

    def __init__(self, v_map=np.nan * (u.km / u.s), v_map_lat=np.nan * u.rad, v_map_long=np.nan * u.rad,
                 cr_num=np.nan, cr_lon_init=360.0 * u.deg, latitude_max=30 * u.deg, latitude_min=-30 * u.deg,
                 r_min=30 * u.solRad, r_max=240 * u.solRad, lon_out=np.nan * u.rad, lon_start=np.nan * u.rad,
                 lon_stop=np.nan * u.rad, simtime=5.0 * u.day, dt_scale=1.0):
        """
        Initialise the SURF3D instance.

            v_map: Inner solar wind speed boundary Carrington map. Must have units of km/s.
            v_map_lat: List of latitude positions for v_map, in radians
            v_map_long: List of Carrington longitudes for v_map, in radians
            br_map: Inner Br boundary Carrington map. Must have no units.
            br_map_lat: List of latitude positions for br_map, in radians
            br_map_long: List of Carrington longitudes for br_map, in radians
            latitude_max: Maximum helio latitude (from the equator) of SURF plane, in degrees
            latitude_min: Maximum helio latitude (from the equator) of SURF plane, in degrees
            cr_num: Integer Carrington rotation number. Used to determine the planetary and spacecraft positions
            cr_lon_init: Carrington longitude of Earth at model initialisation, in degrees.
            lon_out: A specific single longitude (relative to Earth) to compute SURF solution along, in degrees
            lon_start: The first longitude (in a clockwise sense) of the longitude range to solve SURF over.
            lon_stop: The last longitude (in a clockwise sense) of the longitude range to solve SURF over.
            r_min: The radial inner boundary distance of SURF.
            r_max: The radial outer boundary distance of SURF.
            simtime: Duration of the simulation window, in days.
            dt_scale: Integer scaling number to set the model output time step relative to the models CFL time.
            cme_expansion: Boolean, whether CMEs have a declining velocity profile at the inner boundary
        """

        # Define latitude grid
        self.latitude_min = latitude_min.to(u.rad)
        self.latitude_max = latitude_max.to(u.rad)
        self.lat, self.nlat = latitude_grid(self.latitude_min, self.latitude_max)

        assert (len(v_map_lat) == len(v_map[:, 1]))
        assert (len(v_map_long) == len(v_map[1, :]))

        # Get the SURF longitudinal grid
        longs, dlon, nlon = longitude_grid(lon_start=0.0 * u.rad, lon_stop=2 * np.pi * u.rad)

        # Extract the vr value at the given latitudes
        self.v_in = []
        vlong = np.ones(len(v_map_long))
        for thislat in self.lat:
            for ilong in range(0, len(v_map_long)):
                vlong[ilong] = np.interp(thislat.value, v_map_lat.value, v_map[:, ilong].value)

            # Interpolate this longitudinal profile to the SURF resolution
            self.v_in.append(np.interp(longs.value, v_map_long.value, vlong) * u.km / u.s)

        # Set up the model at each latitude
        self.SURFlat = []
        for i in range(0, self.nlat):
            self.SURFlat.append(SURF(v_boundary=self.v_in[i],
                                     latitude=self.lat[i],
                                     cr_num=cr_num, cr_lon_init=cr_lon_init,
                                     r_min=r_min, r_max=r_max,
                                     lon_out=lon_out, lon_start=lon_start, lon_stop=lon_stop,
                                     simtime=simtime, dt_scale=dt_scale))
        return

    def solve(self, cme_list):
        """
        Compute solution of SURF3d instance.
        Args:
            cme_list: A list of ConeCME objects to solve
        Returns:
            None
        """
        for model in self.SURFlat:
            model.solve(cme_list)

        return


def _compressible_method_from_solver(solver_name):
    """Map public solver names to internal compressible method strings."""
    method_map = {
        'hydro': 'hllc-plm-rk2',
        'hydro-pcm': 'hllc-pcm',
    }
    if solver_name not in method_map:
        raise ValueError(f"Solver '{solver_name}' is not a compressible solver.")
    return method_map[solver_name]


def clear_density_temperature_cache(cache_id=None):
    """
    Clear cached lookup data used by get_density_temperature_from_velocity.

    Args:
        cache_id: Optional cache namespace to clear. If None, clears all namespaces.
    """
    if not hasattr(get_density_temperature_from_velocity, '_cache'):
        return

    # Lazily initialise _bounds_cache in case it was never set
    if not hasattr(get_density_temperature_from_velocity, '_bounds_cache'):
        get_density_temperature_from_velocity._bounds_cache = {}

    if cache_id is None:
        get_density_temperature_from_velocity._cache = {}
        get_density_temperature_from_velocity._bounds_cache = {}
    else:
        get_density_temperature_from_velocity._cache.pop(cache_id, None)
        get_density_temperature_from_velocity._bounds_cache.pop(cache_id, None)



def surf_constants():
    """
    Function to generate a dictionary of useful constants.
    Returns:
        constants: A dictionary of constants that configure SURF
    """
    nlong = 128  # Number of longitude bins for a full longitude grid [128]
    dr = 1.5 * u.solRad  # Radial grid step. With v_max, this sets the model time step [1.5 Rs]
    nlat = 45  # Number of latitude bins for a full latitude grid [45]
    v_max = 1000 * u.km / u.s  # Maximum expected solar wind speed. Sets timestep [3000 km/s]

    # CONSTANTS - DON'T CHANGE
    twopi = 2.0 * np.pi
    daysec = 24 * 60 * 60 * u.s
    kms = u.km / u.s
    alpha = 0.15 * u.dimensionless_unscaled  # Scale parameter for residual SW acceleration (for incompressible)
    r_accel = 50 * u.solRad  # Spatial scale parameter for residual SW acceleration
    gamma = 5 # Adiabatic index for compressible solver (1.5 for solar wind?)
    synodic_period = 27.2753 * daysec  # Solar Synodic rotation period from Earth.
    sidereal_period = 25.38 * daysec  # Solar sidereal rotation period
    
    # Typical 1 AU solar wind conditions. Used for CME perturbations and reference values
    v_sw_1au = 400 * u.km / u.s  # Typical solar wind speed at 1 AU
    n_sw_1au = 5 * u.cm**-3  # Typical solar wind density at 1 AU (~5 protons/cm³)
    T_sw_1au = 1e5 * u.K  # Typical solar wind temperature at 1 AU (~100,000 K)
    empirical_n_adjust_amp = 0.2  # Max fractional density remap amplitude applied at 0.1 AU
    empirical_T_adjust_amp = 0.1  # Max fractional temperature remap amplitude applied at 0.1 AU

    constants = {'twopi': twopi, 'daysec': daysec, 'kms': kms, 'alpha': alpha,
                 'gamma': gamma,
                 'r_accel': r_accel, 'synodic_period': synodic_period,
                 'sidereal_period': sidereal_period, 'v_max': v_max,
                 'dr': dr, 'nlong': nlong, 'nlat': nlat,
                 'v_sw_1au': v_sw_1au, 'n_sw_1au': n_sw_1au, 'T_sw_1au': T_sw_1au,
                 'empirical_n_adjust_amp': empirical_n_adjust_amp,
                 'empirical_T_adjust_amp': empirical_T_adjust_amp}

    return constants


# ==============================================================================
# Parker Solution Radial Scaling Functions
# ==============================================================================

# JIT-compiled core computation function (defined at module level for caching)
@jit(nopython=True, cache=True)
def _compute_parker_mapping(v_from_kms, T_from, n_from, r_from_km, r_to_km, gamma, max_iter=50, tol=1e-12):
    """
    Core Parker mapping computation (fully vectorized and JIT-compiled).
    
    All inputs/outputs in SURF standard units:
        v in km/s, T in K, n in cm^-3, r in km
    
    Physical constants in SI used internally.
    """
    # Physical constants (SI)
    k_B = 1.380649e-23   # J/K
    m_p = 1.67262192e-27  # kg
    
    # Convert to SI for consistent computation
    v_from_ms = v_from_kms * 1e3  # m/s
    r_from_m = r_from_km * 1e3    # m
    r_to_m = r_to_km * 1e3        # m
    n_from_m3 = n_from * 1e6      # cm^-3 -> m^-3
    rho_from = n_from_m3 * m_p    # kg/m^3
    
    # Compute Mach number and stagnation conditions at r_from
    p_from = n_from_m3 * k_B * T_from  # Pa
    c_from = np.sqrt(gamma * p_from / rho_from)  # m/s
    M_from = v_from_ms / c_from
    
    # Stagnation temperature - CONSERVED
    T_t = T_from * (1.0 + (gamma - 1.0)/2.0 * M_from**2)
    
    # Get reference area A* from conditions at r_from
    # Using area-Mach relation: A/A* = (1/M) * [(2/(γ+1)) * (1 + (γ-1)/2 * M²)]^((γ+1)/(2(γ-1)))
    a = 2.0 / (gamma + 1.0)
    b = (gamma - 1.0) / 2.0
    c = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    
    A_from = r_from_m**2
    A_norm_from = (1.0/M_from) * (a * (1.0 + b * M_from**2))**c
    A_star = A_from / A_norm_from
    
    # Compute area and normalized area at r_to
    A_to = r_to_m**2
    A_norm_to = A_to / A_star
    
    # Solve for Mach number at r_to using vectorized Newton's method
    n = len(A_norm_to)
    M_to = np.ones(n) * 2.0  # Initial guess
    
    for _ in range(max_iter):
        # Compute area_mach and its derivative
        term = a * (1.0 + b * M_to**2)
        area = (1.0/M_to) * term**c
        f = area - A_norm_to
        
        # Derivative: d/dM[(1/M) * term^c]
        df = -area/M_to + (1.0/M_to) * c * term**(c-1.0) * a * b * 2.0 * M_to
        
        # Newton update
        M_new = M_to - f / df
        
        # Check convergence
        if np.max(np.abs(M_new - M_to)) < tol:
            break
        M_to = M_new
    
    # Compute temperature and velocity at r_to
    T_to = T_t / (1.0 + (gamma - 1.0)/2.0 * M_to**2)
    c_to = np.sqrt(gamma * k_B * T_to / m_p)  # m/s
    v_to_ms = M_to * c_to
    v_to_kms = v_to_ms / 1e3  # back to km/s
    
    # Compute density using mass conservation (n * v * r^2 = const)
    n_to = n_from * (r_from_km / r_to_km)**2 * (v_from_kms / v_to_kms)  # cm^-3
    
    return v_to_kms, n_to, T_to


def map_properties_parker(velocity, r_from, r_to, density_from, temperature_from, gamma=1.5):
    """
    Map solar wind parameters between two heliocentric distances using Parker nozzle equations.
    
    Uses the full Parker nozzle (area-Mach) relation for adiabatic isentropic flow.
    
    Args:
        velocity: Velocity at r_from. Astropy Quantity in km/s.
        r_from: Initial heliocentric distance. Astropy Quantity with length units.
        r_to: Final heliocentric distance. Astropy Quantity with length units.
        density_from: Number density at r_from. Astropy Quantity in cm^-3.
        temperature_from: Temperature at r_from. Astropy Quantity in K.
        gamma: Adiabatic index (default 1.5 for solar wind)
    
    Returns:
        tuple: (velocity at r_to [km/s], density at r_to [cm^-3], temperature at r_to [K])
               All returned as astropy Quantities.
    
    Example:
        >>> v_1AU = 400 * u.km / u.s
        >>> n_1AU = 5 * u.cm**-3
        >>> T_1AU = 1e5 * u.K
        >>> v_01AU, n_01AU, T_01AU = map_properties_parker(v_1AU, 215*u.solRad, 21.5*u.solRad, n_1AU, T_1AU)
    """
    
    # Extract values in standard units
    v_from_kms = velocity.to(u.km/u.s).value
    T_from = temperature_from.to(u.K).value
    n_from = density_from.to(u.cm**-3).value
    
    # Convert radii to km for the JIT function
    r_from_km = r_from.to(u.km).value
    r_to_km = r_to.to(u.km).value
    
    # Check if we have scalar or array input
    is_scalar = np.isscalar(v_from_kms) or (hasattr(v_from_kms, 'shape') and v_from_kms.shape == ())
    
    # Convert scalar to array for unified processing
    if is_scalar:
        v_from_kms = np.array([v_from_kms])
    else:
        v_from_kms = np.asarray(v_from_kms)
    
    # Call the JIT-compiled core function (takes km, km/s, cm^-3, K)
    v_to_kms, n_to, T_to = _compute_parker_mapping(
        v_from_kms, T_from, n_from, r_from_km, r_to_km, gamma
    )
    
    # Convert back to scalar if input was scalar
    if is_scalar:
        v_to_kms = v_to_kms[0]
        n_to = n_to[0]
        T_to = T_to[0]
    
    # Return with units
    return v_to_kms * u.km / u.s, n_to * u.cm**-3, T_to * u.K


def get_omni_lookup_table_at_distance(r_target, lookup_table_path=None):
    """
    Map the OMNI 1 AU lookup table to a specified heliocentric distance using Parker nozzle equations.
    
    This function loads a lookup table of velocity, number density, and temperature at 1 AU
    (typically derived from OMNI data) and maps each entry to the requested distance using
    the Parker nozzle solution (map_properties_parker).
    
    Args:
        r_target: Target heliocentric distance (astropy Quantity with length units, e.g., 0.1*u.au)
        lookup_table_path: Path to the 1 AU lookup table file. If None, looks for 
                          'omni_lookup_table_1AU.txt' in the tests directory.
    
    Returns:
        tuple: (v_array, n_array, T_array) at r_target
               - v_array: Velocities in km/s
               - n_array: Number densities in cm^-3
               - T_array: Temperatures in K
    
    Example:
        >>> v_01au, n_01au, T_01au = get_omni_lookup_table_at_distance(0.1*u.au)
        >>> v_30Rs, n_30Rs, T_30Rs = get_omni_lookup_table_at_distance(30*u.solRad)
    """
    # Default path to lookup table
    if lookup_table_path is None:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        lookup_table_path = os.path.join(module_dir, 'data', 'insitu', 'omni_lookup_table_1AU.txt')
    
    if not os.path.exists(lookup_table_path):
        raise FileNotFoundError(f"Lookup table not found at {lookup_table_path}. "
                              "Please run tests/analyze_omni_relations.py to generate it.")
    
    # Load the 1 AU lookup table
    data = np.loadtxt(lookup_table_path)
    v_1au = data[:, 0]  # km/s
    n_1au = data[:, 1]  # cm^-3
    T_1au = data[:, 2]  # K
    
    # Reference distance (1 AU)
    r_1au = 1.0 * u.au
    
    # Map each entry to the target distance
    v_target = np.zeros_like(v_1au)
    n_target = np.zeros_like(n_1au)
    T_target = np.zeros_like(T_1au)
    
    for i in range(len(v_1au)):
        v_out, n_out, T_out = map_properties_parker(
            v_1au[i] * u.km / u.s,
            r_1au,
            r_target,
            n_1au[i] * u.cm**-3,
            T_1au[i] * u.K
        )
        
        v_target[i] = v_out.to(u.km / u.s).value
        n_target[i] = n_out.to(u.cm**-3).value
        T_target[i] = T_out.to(u.K).value
    
    return v_target, n_target, T_target

def get_density_temperature_from_velocity(v_value, r_target, gamma=1.5, cache_id='global'):
    """
    Get density and temperature for a given velocity at a specified radial distance.
    
    This function generates a lookup table at the target radial distance by mapping
    OMNI empirical relations from 1 AU via the Parker nozzle solution, then interpolates
    the given velocity to compute number density and temperature.
    
    Uses caching to avoid recomputing the lookup table for the same r_target value.
    
    Args:
        v_value: Solar wind velocity in km/s (scalar or array)
        r_target: Target heliocentric distance in solar radii (scalar)
        gamma: Adiabatic index (default 1.5 for solar wind)
        cache_id: Cache namespace key. Use a unique value per SURF instance to
                  recompute on new instances while still reusing within each one.
    
    Returns:
        n: Number density in cm^-3 (scalar or array matching v_value)
        T: Temperature in K (scalar or array matching v_value)
    
    Examples:
        >>> n, T = get_density_temperature_from_velocity(400, 30.0)
        >>> n, T = get_density_temperature_from_velocity([350, 400, 450], 30.0)
    """
    # Cache structure:
    #   _cache: {cache_id: {lookup_key: (v_lookup, n_lookup, T_lookup)}}
    if not hasattr(get_density_temperature_from_velocity, '_cache'):
        get_density_temperature_from_velocity._cache = {}

    all_cache = get_density_temperature_from_velocity._cache
    cache = all_cache.setdefault(cache_id, {})

    # Accept either astropy Quantity (distance) or scalar in solar radii.
    if isinstance(r_target, u.Quantity):
        r_target_q = r_target.to(u.solRad)
    else:
        r_target_q = r_target * u.solRad
    
    lookup_key = (round(float(r_target_q.value), 12), round(float(gamma), 12))

    # Check if we have cached data for this r_target in this cache namespace
    if lookup_key not in cache:
        # Generate lookup table at target distance and cache it
        v_lookup, n_lookup, T_lookup = get_omni_lookup_table_at_distance(r_target_q)
        cache[lookup_key] = (v_lookup, n_lookup, T_lookup)
    else:
        v_lookup, n_lookup, T_lookup = cache[lookup_key]
    
    # Interpolate to get density and temperature at given velocity
    # Use extrapolation for values outside the lookup table range
    n_interp = np.interp(v_value, v_lookup, n_lookup, 
                         left=n_lookup[0], right=n_lookup[-1])
    T_interp = np.interp(v_value, v_lookup, T_lookup,
                         left=T_lookup[0], right=T_lookup[-1])
    
    return n_interp, T_interp


def radial_grid(r_min=30.0 * u.solRad, r_max=240. * u.solRad):
    """
    Define the radial grid of the SURF model. Step size is fixed, but inner and outer boundary may be specified.
    Args:
        r_min: The heliocentric distance of the inner radial boundary.
        r_max: The heliocentric distance of the outer radial boundary.
    Returns:
        r: An array of radial coordinates, relative to Sun center, in solar radii.
        dr: The radial grid step value, in solar radii.
        rrel: An array of radial coordinates, relative to the model inner boundary.
        nr: The number of radial grid steps.
    """
    if r_min >= r_max:
        print("Warning, r_min cannot be less than r_max. Defaulting to r_min=30rs and r_max=240rs")
        r_min = 30 * u.solRad
        r_max = 240 * u.solRad

    if r_min < 2.0 * u.solRad:
        print("Warning, r_min should not be less than 5.0rs. Defaulting to 5.0rs")
        r_min = 5.0 * u.solRad

    if r_max > 6000 * u.solRad:
        print("Warning, r_max should not be more than 400rs. Defaulting to 400rs")
        r_max = 400 * u.solRad

    constants = surf_constants()
    dr = constants['dr']
    r = np.arange(r_min.value, r_max.value + dr.value, dr.value)
    r = r * dr.unit
    nr = r.size
    # acceleration is scaled relative to 30rS
    rrel = r - 30 * u.solRad
    return r, dr, rrel, nr


def longitude_grid(lon_out=np.nan * u.rad, lon_start=np.nan * u.rad, lon_stop=np.nan * u.rad):
    """
    Define the longitude grid of the SURF model.
    Args:
        lon_out: A single output longitude.
        lon_start: The first longitude (in a clockwise sense) of a longitude range, in radians.
        lon_stop: The last longitude (in a clockwise sense) of a longitude range, in radians.
    Returns:
        lon: An array of longitude values, or single longitude value, in radians.
        dlon: Spacing of the (regular) longitude grid.
        nlon: Number of longitude values returned.
    """
    # Check the inputs.
    twopi = 2.0 * np.pi
    single_longitude = False
    longitude_range = False
    if np.isfinite(lon_out):
        # Select single longitude only. Check in range
        if (lon_out < 0 * u.rad) | (lon_out > twopi * u.rad):
            lon_out = zerototwopi(lon_out.to('rad').value)
            lon_out = lon_out * u.rad

        single_longitude = True
    elif np.isfinite(lon_start) & np.isfinite(lon_stop):
        # Select a range of longitudes. Check limits in range.
        if (lon_start < 0 * u.rad) | (lon_start > twopi * u.rad):
            lon_start = zerototwopi(lon_start.to('rad').value)
            lon_start = lon_start * u.rad

        if (lon_stop < 0 * u.rad) | (lon_stop > twopi * u.rad):
            lon_stop = zerototwopi(lon_stop.to('rad').value)
            lon_stop = lon_stop * u.rad

        longitude_range = True

    # Form the full longitude grid.
    nlon = surf_constants()['nlong']
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


def latitude_grid(latitude_min=np.nan, latitude_max=np.nan):
    """
    Define the latitude grid of the SURF model. This is constant in sine latitude
    Args:
        latitude_min: The maximum latitude above the equator, in radians
        latitude_max: The minimum latitude below the equator, in radians
    Returns:
        lat: Array of latitude positions between given limits, in radians
        nlat: Number of latitude positions between given limits
    """
    # Check the inputs.
    assert (latitude_max > latitude_min)
    assert (np.absolute(latitude_max) <= (np.pi / 2) * u.rad)
    assert (np.absolute(latitude_min) <= (np.pi / 2) * u.rad)

    # Form the full longitude grid.
    nlat = surf_constants()['nlat']

    dsinlat = 2 / nlat
    sinlat_min_full = - 1 + dsinlat / 2.0
    sinlat_max_full = 1 - dsinlat / 2.0
    sinlat, dsinlat = np.linspace(sinlat_min_full, sinlat_max_full, nlat, retstep=True)
    lat = np.arcsin(sinlat) * u.rad

    # Now get only the selected range of latitudes
    id_match = (lat >= latitude_min) & (lat <= latitude_max)
    lat = lat[id_match]
    nlat = lat.size

    return lat, nlat


def time_grid(simtime, dt_scale):
    """
    Define the model timestep and time grid based on CFL condition and specified simulation time.
    Args:
        simtime: The length of the simulation
        dt_scale: An integer specifying how frequently model timesteps should be saved to output.
    Returns:
        time_grid_dict: A dictionary containing arrays of the models intrinsic time steps and the requsted output
                        timesteps.
    """
    constants = surf_constants()
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
    Returns:
        dirs: A dictionary of full paths to SURF directories of code, data, figures, and relevant files.
    """

    # Get path of surf.py
    cwd = os.path.abspath(os.path.dirname(__file__))

    dirs = {'ephemeris': os.path.join(cwd, 'data', 'ephemeris', 'ephemeris.hdf5'),
            'example_inputs': os.path.join(cwd, 'data', 'example_inputs'),
            'insitu': os.path.join(cwd, 'data', 'insitu')}

    # Use appdirs to get platform-specific user data directory
    base_dir = Path(user_data_dir("surf", ""))
    
    bc_dir = base_dir / "data" / 'boundary_conditions'
    bc_dir.mkdir(parents=True, exist_ok=True)
    dirs['boundary_conditions'] = str(bc_dir)

    sim_dir = base_dir / "data" / 'surf'
    sim_dir.mkdir(parents=True, exist_ok=True)
    dirs['SURF_data'] = str(sim_dir)

    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dirs['SURF_figures'] = str(fig_dir)

    # Just check the directories exist.
    for key, val in dirs.items():
        if key == 'ephemeris':
            if not os.path.isfile(val):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), val)
        else:
            if not os.path.isdir(val):
                raise NotADirectoryError(errno.ENOENT, os.strerror(errno.ENOENT), val)

    return dirs


@jit(nopython=True)
def zerototwopi(angles):
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


# ============================================================================
# Compressible Solver Integration
# ============================================================================



def solve_radial_compressible(v_bc_kms, rho_bc_kgm3, T_bc_K, model_time, time_out, 
                               r_grid, gamma, nt_out, nr, riemann='hllc-plm-rk2', verbose=False,
                               num_particles=0, particle_injection_rate=None, particle_release_rate=None,
                               solver_instance=None):
    """
    Solve 1D radial solar wind expansion using a compressible solver with selectable Riemann solver.
    
    This function wraps the CompressibleSolver class with proper unit conversions and 
    time handling for integration with SURF.
    
    Parameters
    ----------
    v_bc_kms : array_like
        Time series of inner boundary velocity (km/s), shape (nt_model,)
    rho_bc_kgm3 : array_like
        Time series of inner boundary density (kg/m³), shape (nt_model,)
    T_bc_K : array_like
        Time series of inner boundary temperature (K), shape (nt_model,)
    model_time : array_like
        Full model time grid including spin-up (seconds), shape (nt_model,)
    time_out : array_like
        Output time grid (seconds), shape (nt_out,)
    r_grid : array_like
        Radial grid positions (km), shape (nr,)
    gamma : float
        Adiabatic index
    nt_out : int
        Number of output time steps
    nr : int
        Number of radial grid points
    riemann : str, optional
        Compressible method string, e.g. 'hllc-plm-rk2' or 'hllc-pcm'.
        Default is 'hllc-plm-rk2'.
    verbose : bool, optional
        If True, print detailed diagnostics. Default False.
    num_particles : int or dict, optional
        Number of test particles to track. If 0 (default), no tracking.
        Dict with keys 'cme_leading', 'cme_trailing', 'hcs', 'streak_*' supported.
    particle_injection_rate : array-like or dict, optional
        Injection times (seconds) for particles in model_time coordinates.
        If num_particles is dict, this must also be dict with matching keys.
    particle_release_rate : array-like or dict, optional
        Release times (seconds) for particles.
        If None, default to injection times.
    solver_instance : CompressibleSolver, optional
        Pre-initialized solver instance to reuse. If None, a new one is created.
    
    Returns
    -------
    v_out : ndarray
        Velocity at output times (km/s), shape (nt_out, nr)
    rho_out : ndarray
        Density at output times (kg/m³), shape (nt_out, nr)
    temp_out : ndarray
        Temperature at output times (K), shape (nt_out, nr)
    particle_data : dict or None
        Particle trajectory data if num_particles > 0, otherwise None.
        Contains 'groups' dict with particle positions in km at all times
    
    Notes
    -----
    - Uses SURF compressible methods based on HLLC with PLM or PCM reconstruction
    - Initializes with Parker nozzle solution for smooth startup
    """
    KM_TO_M = 1e3  # m/km
    
    # Times are already plain arrays (units stripped by caller)
    model_time_seconds = model_time
    time_out_seconds = time_out
    
    # Convert to SI (m, m/s, kg/m³)
    v_bc_si = v_bc_kms * KM_TO_M  # m/s
    rho_bc_si = rho_bc_kgm3  # already kg/m³
    
    # Convert SURF radial grid to m
    r_grid_m = r_grid * KM_TO_M
    
    # Create output time grid for solver - include spin-up snapshots
    spinup_time_seconds = time_out_seconds[0] - model_time_seconds[0]
    n_spinup_snaps = max(5, int(spinup_time_seconds / 86400))  # At least 5, or ~1 per day
    spinup_sampled = np.linspace(model_time_seconds[0], time_out_seconds[0], n_spinup_snaps, endpoint=False)
    t_grid_combined = np.concatenate([spinup_sampled, time_out_seconds])
    
    # Boundary condition functions (MUST return plain floats, no units)
    def v_bc_func(t):
        return float(np.interp(t, model_time_seconds, v_bc_si))
    
    def rho_bc_func(t):
        return float(np.interp(t, model_time_seconds, rho_bc_si))
    
    def T_bc_func(t):
        return float(np.interp(t, model_time_seconds, T_bc_K))
    
    # Initialize solver with selected method string.
    if solver_instance is None:
        method = riemann
            
        solver = create_compressible_solver(
            r_grid=r_grid_m,
            gamma=gamma,
            method=method,
            cfl=0.7 if 'plm' in method else 0.8, # Lower CFL for PLM
            verbose=verbose
        )
    else:
        solver = solver_instance
    
    # Run simulation
    results = solver.solve(
        t_grid=t_grid_combined,
        v_bc_func=v_bc_func,
        rho_bc_func=rho_bc_func,
        T_bc_func=T_bc_func,
        num_particles=num_particles,
        particle_injection_rate=particle_injection_rate,
        particle_release_rate=particle_release_rate
    )
    
    # Extract output times (skip spin-up)
    n_spinup = len(spinup_sampled)
    
    # Skip spin-up snapshots, extract only output times
    v_out_si = results['v'][n_spinup:]
    rho_out_si = results['rho'][n_spinup:]
    temp_out_K = results['T'][n_spinup:]
    
    # If solver returned fewer points than expected, interpolate
    if v_out_si.shape[0] != nt_out:
        if verbose:
            print(f"  Warning: solver returned {v_out_si.shape[0]} points, expected {nt_out}, interpolating...")
        
        v_interp = np.zeros((nt_out, nr))
        rho_interp = np.zeros((nt_out, nr))
        temp_interp = np.zeros((nt_out, nr))
        
        # Skip spin-up times to get only output times
        solver_out_times = results['t'][n_spinup:]
        
        for ir in range(nr):
            v_interp[:, ir] = np.interp(time_out_seconds, solver_out_times, v_out_si[:, ir])
            rho_interp[:, ir] = np.interp(time_out_seconds, solver_out_times, rho_out_si[:, ir])
            temp_interp[:, ir] = np.interp(time_out_seconds, solver_out_times, temp_out_K[:, ir])
        
        v_out_si = v_interp
        rho_out_si = rho_interp
        temp_out_K = temp_interp
    
    # Convert to SURF units
    v_out_kms = v_out_si / KM_TO_M  # m/s -> km/s
    rho_out_kgm3 = rho_out_si  # already kg/m³
    temp_out = temp_out_K  # K (no conversion)
    
    # Extract and convert particle data if present
    particle_data = None
    if 'particles' in results:
        particle_data_si = results['particles']
        
        if isinstance(num_particles, dict):
            # Multi-group mode - convert positions to km
            particle_data = {'groups': {}}
            for group_name, group_data in particle_data_si['groups'].items():
                # Convert positions from m to km
                r_si = np.asarray(group_data['r'])
                v_si = np.asarray(group_data['v'])
                t_sec = np.asarray(group_data['t'])
                
                if r_si.size == 0:
                    r_km = np.array([])
                    v_km = np.array([])
                    t_sec = np.array([])
                else:
                    r_km = r_si / KM_TO_M
                    v_km = v_si / KM_TO_M
                
                particle_data['groups'][group_name] = {
                    'r': r_km,  # km
                    'v': v_km,  # km/s
                    't': t_sec,  # seconds
                    't_inject': group_data['t_inject'],  # seconds
                    'active': group_data['active'],
                    'n_particles': group_data['n_particles'],
                }
        else:
            # Single group mode - convert positions to km
            r_si = np.asarray(particle_data_si['r'])
            v_si = np.asarray(particle_data_si['v'])
            t_sec = np.asarray(particle_data_si['t'])
            
            if r_si.size == 0:
                r_km = np.array([])
                v_km = np.array([])
                t_sec = np.array([])
            else:
                r_km = r_si / KM_TO_M
                v_km = v_si / KM_TO_M
            
            particle_data = {
                'r': r_km,  # km
                'v': v_km,  # km/s
                't': t_sec,  # seconds
                't_inject': particle_data_si['t_inject'],  # seconds
                'active': particle_data_si['active'],
            }
    
    return v_out_kms, rho_out_kgm3, temp_out, particle_data


@jit(nopython=True, nogil=True)
def solve_radial(vinput, binput, iscmeinput, model_time, rrel, params,
                 n_cme, n_hcs_max, streak_times, rhoinput=None, tempinput=None):
    """
    Solve the radial profile as a function of time (including spinup), and
    return radial profile at specified output timesteps.
    Tracks CME frotns as test particles
    Args:
        vinput: Timeseries of inner boundary solar wind speeds.
        binput: Timeseries of inner boundary radial magnetic field.
        iscmeinput: Timeseries of in/out of a CME at the inner boundary.
        model_time: Array of model timesteps.
        rrel: Array of model radial coordinates relative to 30rS.
        params: Array of SURF parameters.
        n_cme: Number of CMEs in the whole model run (not nec this longitude).
        n_hcs_max: Maximum number of HCS crossings at any longitude
        streak_times: time indices of streak foot points to track
        rhoinput: Timeseries of inner boundary density (optional, for compressible solver). Plain array without units.
        tempinput: Timeseries of inner boundary temperature (optional, for compressible solver). Plain array without units.
        compressible: Boolean flag indicating if compressible solver is being used
        solver: String specifying which numerical solver to use
    Returns:
        v_grid: Array of radial solar wind speed profile as function of time.
        cme_particles_r: Array of CME tracer particle positions as function of time.
        cme_particles_v: Array of CME tracer particle speeds as a function of time.
        rho_grid: Array of radial density profile as function of time (only if compressible=True, else None).
        temp_grid: Array of radial temperature profile as function of time (only if compressible=True, else None).
    """

    # unpack the SURF params
    dtdr = params[0]
    alpha = params[1]
    r_accel = params[2]
    dt_scale = np.int32(params[3])
    nt_out = np.int32(params[4])
    nr = np.int32(params[5])
    r_boundary = params[7]
    accel_limit = bool(params[9])  # switch used to determine if speed limit is applied to acceleration.
    gamma = params[10]  # Adiabatic index for compressible solver
    solver = 'huxt'  # This function is only called for huxt solver
    compressible = False  # huxt solver is incompressible by default
    
    # Compute the radial grid for the test particles
    rgrid = (rrel - rrel[0]) * 695700.0 + r_boundary  # Can't use astropy.units because numba
    dr = rgrid[1] - rgrid[0]
    dt = dtdr * dr

    # Preallocate space for solutions
    v_grid = np.zeros((nt_out, nr))
    cme_particles_r = np.zeros((n_cme, nt_out, 2)) * np.nan
    cme_particles_v = np.zeros((n_cme, nt_out, 2)) * np.nan
    
    # Preallocate space for density and temperature 
    # Use dummy arrays if not compressible to avoid None issues in Numba
    if compressible:
        rho_grid = np.zeros((nt_out, nr))
        temp_grid = np.zeros((nt_out, nr))
    else:
        rho_grid = np.zeros((1, 1))  # Dummy array
        temp_grid = np.zeros((1, 1))  # Dummy array

    # Check if CMEs need to be tracked.
    do_cme = 0
    if np.any(iscmeinput) > 0:
        do_cme = 1

    # Check if HCS needs to be tracked.
    hcs_particles = np.zeros((n_hcs_max, nt_out, 2)) * np.nan

    # see if there are any streaklines to be traced
    do_streak = False
    if not (np.isnan(streak_times)).all():
        do_streak = True
        n_streaks = len(streak_times[:, 0, 0])
        n_rots = len(streak_times[0, :, 0])

        streak_particles = np.ones((nt_out, n_streaks, n_rots)) * np.nan
        r_streakparticles = np.ones((n_streaks, n_rots)) * np.nan
    else:
        streak_particles = np.ones((1, 1, 1)) * np.nan

    iter_count = 0
    t_out = 0
    hcs_count = 0

    for t, time in enumerate(model_time):
        # Get the initial condition, which will update in the loop,
        # and snapshots saved to output at right steps.
        if t == 0:
            v = np.ones(nr) * 400
            r_cmeparticles = np.ones((n_cme, 2)) * np.nan
            v_cmeparticles = np.ones((n_cme, 2)) * np.nan
            r_hcsparticles = np.ones((n_hcs_max, 2)) * np.nan
            
            # Initialize density and temperature arrays for compressible solver
            if compressible:
                # Initialize with proper continuity relation: ρ·v·r² = const
                # This accounts for velocity evolution and gives better initial guess
                
                r_inner = rrel[0] * 695700.0 + r_boundary  # km
                rho_inner = rhoinput[0]
                temp_inner = tempinput[0]
                v_inner = vinput[0]
                
                rho = np.zeros(nr)
                temp = np.zeros(nr)
                for ir in range(nr):
                    r_this = rrel[ir] * 695700.0 + r_boundary  # km
                    v_this = vinput[ir] if ir < len(vinput) else v_inner
                    # Use Parker solution density scaling with velocity correction
                    # Continuity: ρ·v·r² = const → ρ(r) = ρ₀·(v₀/v)·(r₀/r)²
                    # Density scales as 1/r²: rho(r_to) = rho(r_from) * (r_from/r_to)²
                    r_ratio = r_inner / r_this
                    rho_base = rho_inner * r_ratio**2
                    rho[ir] = rho_base * (v_inner / v_this)
                    # Temperature initialization: use constant value from boundary
                    # Full temperature evolution handled by compressible solver
                    # which accounts for adiabatic expansion and velocity changes
                    temp[ir] = temp_inner
            else:
                rho = np.zeros(nr)  # Dummy array
                temp = np.zeros(nr)  # Dummy array

        # Update the inner boundary conditions
        v[0] = vinput[t]
        
        # Update density and temperature boundary conditions for compressible solver
        if compressible:
            rho[0] = rhoinput[t]
            temp[0] = tempinput[t]

        # see if there's an HCS crossing to be inserted at the boundary
        if t > 0:
            if binput[t] - binput[t - 1] > 0:
                r_hcsparticles[hcs_count, 0] = r_boundary
                r_hcsparticles[hcs_count, 1] = 1
                hcs_count = hcs_count + 1
            elif binput[t] - binput[t - 1] < 0:
                r_hcsparticles[hcs_count, 0] = r_boundary
                r_hcsparticles[hcs_count, 1] = -1
                hcs_count = hcs_count + 1

        # see if there's a new streakline to track and if so, insert at boundary
        if do_streak:
            for istreak in range(0, n_streaks):
                for irot in range(0, n_rots):
                    if t == streak_times[istreak, irot, 0]:
                        # add a particle in
                        r_streakparticles[istreak, irot] = r_boundary

        # Compute boundary speed of each CME at this time. 
        if time > 0:
            if do_cme == 1:
                for n in range(n_cme):
                    # Check if this point is within the cone CME
                    if iscmeinput[t] > 0:
                        thiscme = iscmeinput[t] - 1

                        # If the leading edge test particle doesn't exist, add it
                        if np.isnan(r_cmeparticles[thiscme, 0]):
                            r_cmeparticles[thiscme, 0] = r_boundary
                            v_cmeparticles[thiscme, 0] = v[0]

                        # Hold the CME trailing edge test particle at the inner boundary
                        # Until if condition breaks
                        r_cmeparticles[thiscme, 1] = r_boundary
                        v_cmeparticles[thiscme, 1] = v[0]

        # Update all fields for the given longitude
        u_up = v[1:].copy()
        u_dn = v[:-1].copy()

        # Do a single model time step
        # Solver dispatch: select numerical method based on solver parameter
        
        if solver == 'huxt':
            # HUXt advection scheme (implemented with first-order upwind differencing)
            if compressible:
                # Evolve velocity, density, and temperature together for compressible runs
                # NOTE: accel_limit is ignored for compressible solver (no residual acceleration)
                rho_up = rho[1:].copy()
                rho_dn = rho[:-1].copy()
                temp_up = temp[1:].copy()
                temp_dn = temp[:-1].copy()
                
                u_up_next, rho_up_next, temp_up_next = _upwind_step_compressible_(
                    u_up, u_dn, rho_up, rho_dn, temp_up, temp_dn, dtdr, alpha, r_accel, rrel, r_boundary, gamma)
                
                # Save the updated time steps (direct assignment, no copy needed)
                v[1:] = u_up_next
                rho[1:] = rho_up_next
                temp[1:] = temp_up_next
            else:
                # Incompressible HUXt update (velocity only)
                if accel_limit:
                    u_up_next = _upwind_step_accel_limit(u_up, u_dn, dtdr, alpha, r_accel, rrel)
                else:
                    u_up_next = _upwind_step_(u_up, u_dn, dtdr, alpha, r_accel, rrel)
                
                # Save the updated time step (direct assignment, no copy needed)
                v[1:] = u_up_next
        
        else:
            raise ValueError(f"Unknown solver: {solver}. Supported solver: 'huxt'")

        # Move the CME test particles forward
        if t > 0 and do_cme:
            for n in range(0, n_cme):  # loop over each CME
                for bound in range(0, 2):  # loop over front and rear boundaries
                    if not np.isnan(r_cmeparticles[n, bound]):
                        # Linearly interpolate the speed
                        v_test = np.interp(r_cmeparticles[n, bound] - dr / 2, rgrid, v)

                        # Advance the test particle
                        r_cmeparticles[n, bound] = (r_cmeparticles[n, bound] + v_test * dt)
                        v_cmeparticles[n, bound] = v_test

                if r_cmeparticles[n, 0] > rgrid[-1]:
                    # If the leading edge is past the outer boundary, clamp it at the outer boundary
                    r_cmeparticles[n, 0] = rgrid[-1]

                if r_cmeparticles[n, 1] > rgrid[-1]:
                    # If the trailing edge is past the outer boundary, clamp it at the outer boundary
                    # This prevents CME contours from being corrupted in plots
                    r_cmeparticles[n, 1] = rgrid[-1]

        # Move the HCS test particles forward
        if t > 0:
            for n in range(0, n_hcs_max):  # loop over each HCS

                if not np.isnan(r_hcsparticles[n, 0]):
                    # Linearly interpolate the speed
                    v_test = np.interp(r_hcsparticles[n, 0] - dr / 2, rgrid, v)

                    # Advance the test particle
                    r_hcsparticles[n, 0] = (r_hcsparticles[n, 0] + v_test * dt)

                if r_hcsparticles[n, 0] > rgrid[-1]:
                    # If the leading edge is past the outer boundary, delete
                    r_hcsparticles[n, 0] = np.nan

        # move the streak line particles forward
        if t > 0 and do_streak:
            for istreak in range(0, n_streaks):
                for irot in range(0, n_rots):
                    if not np.isnan(r_streakparticles[istreak, irot]):
                        # Linearly interpolate the speed
                        v_test = np.interp(r_streakparticles[istreak, irot] - dr / 2, rgrid, v)

                        # Advance the test particle
                        r_streakparticles[istreak, irot] = (r_streakparticles[istreak, irot]
                                                            + v_test * dt)

                    if r_streakparticles[istreak, irot] > rgrid[-1]:
                        # If the leading edge is past the outer boundary, delete
                        r_streakparticles[istreak, irot] = np.nan

        # Save this frame to output if it is an output time step
        if time >= 0:
            iter_count = iter_count + 1
            if iter_count == dt_scale:
                if t_out <= nt_out - 1:
                    v_grid[t_out, :] = v.copy()
                    cme_particles_r[:, t_out, :] = r_cmeparticles.copy()
                    cme_particles_v[:, t_out, :] = v_cmeparticles.copy()
                    hcs_particles[:, t_out, :] = r_hcsparticles.copy()
                    if do_streak:
                        streak_particles[t_out, :, :] = r_streakparticles.copy()
                    
                    # Save density and temperature for compressible solver
                    if compressible:
                        rho_grid[t_out, :] = rho.copy()
                        temp_grid[t_out, :] = temp.copy()
                    
                    t_out = t_out + 1
                    iter_count = 0

    return v_grid, cme_particles_r, cme_particles_v, hcs_particles, streak_particles, rho_grid, temp_grid


@jit(nopython=True)
def add_cmes_to_input_series(vinput, model_time, lon, r_boundary, cme_params, latitude,
                             rhoinput=None, tempinput=None, rho_ambient=None, temp_ambient=None, compressible=False):
    """
    Add CMEs to the model input time series
    Args:
        vinput: Timeseries of inner boundary solar wind speeds.
        model_time: Array of model timesteps
        lon: The longitude of this radial
        r_boundary: The SURF inner boundary in rS
        cme_params: Array of ConeCME parameters to include in the solution. One row for each CME, with columns as
                    required by _is_in_cone_cme_boundary_expanding_
        latitude: Latitude (from the equator) of the SURF plane
        rhoinput: Timeseries of inner boundary density (optional, for compressible solver)
        tempinput: Timeseries of inner boundary temperature (optional, for compressible solver)
        rho_ambient: Timeseries of ambient (pre-CME) density (optional, for compressible solver)
        temp_ambient: Timeseries of ambient (pre-CME) temperature (optional, for compressible solver)
        compressible: Boolean flag indicating if compressible solver is being used
    Returns: 
        v: vinput with CME speeds added
        isincme: Boolean time series of CME occurrence at inner boundary
        rho: rhoinput with CME densities added (if compressible=True)
        temp: tempinput with CME temperatures added (if compressible=True)
    """

    n_cme = cme_params.shape[0]
    v = vinput
    isincme = v * 0
    
    # Initialize density and temperature outputs (plain arrays without units)
    if compressible and rhoinput is not None:
        rho = rhoinput.copy()
    else:
        rho = None
        
    if compressible and tempinput is not None:
        temp = tempinput.copy()
    else:
        temp = None

    for t, time in enumerate(model_time):

        # Compute boundary speed of each CME at this time. 
        # Set boundary to the maximum CME speed at this time.
        if time > 0:
            v_update_cme = np.zeros(n_cme) * np.nan
            rho_update_cme = np.zeros(n_cme) * np.nan
            temp_update_cme = np.zeros(n_cme) * np.nan
            
            for n in range(n_cme):
                cme = cme_params[n, :]
                cme_expansion = cme[9]
                profile_flag = cme[14]  # 0 = square, 1 = sinusoidal
                
                # Check if this point is within the cone CME
                iscme, dist_from_nose = _is_in_cme_boundary_(r_boundary, lon, latitude, time, cme)
                if iscme:
                    # Get ambient values at this time (use ambient arrays if provided, else use input arrays)
                    v_ambient = vinput[t]
                    rho_ambient_val = 0.0
                    temp_ambient_val = 0.0
                    
                    if compressible:
                        # Use ambient arrays (pre-CME) if provided, otherwise fall back to input arrays
                        if rho_ambient is not None:
                            rho_ambient_val = rho_ambient[t]
                        elif rhoinput is not None:
                            rho_ambient_val = rhoinput[t]
                            
                        if temp_ambient is not None:
                            temp_ambient_val = temp_ambient[t]
                        elif tempinput is not None:
                            temp_ambient_val = tempinput[t]
                    
                    # Compute modulation factor based on profile type
                    if profile_flag > 0.5:  # sinusoidal profile
                        # Use sine function: 0 at edges (dist=0 or dist=1), 1 at center (dist=0.5)
                        # sin(pi * dist_from_nose) gives: 0 at dist=0, 1 at dist=0.5, 0 at dist=1
                        modulation = np.sin(np.pi * dist_from_nose)
                    else:  # square profile (default)
                        modulation = 1.0
                    
                    # Apply velocity profile
                    if cme_expansion:
                        # use Owens2005 empirical relations
                        v_cme = cme[4]*(1-dist_from_nose) + 200*dist_from_nose
                    else:
                        v_cme = cme[4]
                    
                    # Interpolate between ambient and CME value using modulation
                    v_update_cme[n] = v_ambient + modulation * (v_cme - v_ambient)
                    
                    # Add CME density and temperature if compressible
                    if compressible:
                        # CME density is at index 12, temperature at index 13 in cme_params
                        rho_cme = cme[12]
                        temp_cme = cme[13]
                        
                        # Apply modulation to density and temperature as well
                        rho_update_cme[n] = rho_ambient_val + modulation * (rho_cme - rho_ambient_val)
                        temp_update_cme[n] = temp_ambient_val + modulation * (temp_cme - temp_ambient_val)

                    # record the CME number
                    isincme[t] = n + 1

            # See if there are any CMEs
            if not np.all(np.isnan(v_update_cme)):
                v[t] = np.nanmax(v_update_cme)
                
                # Update density and temperature for compressible case (plain values, units added by caller)
                if compressible and rho is not None:
                    rho[t] = np.nanmax(rho_update_cme)
                if compressible and temp is not None:
                    temp[t] = np.nanmax(temp_update_cme)

    return v, isincme, rho, temp



@jit(nopython=True)
def _upwind_step_(v_up, v_dn, dtdr, alpha, r_accel, rrel):
    """
    Compute the next step in the upwind scheme of Burgers equation with added acceleration of the solar wind.
    Args:
        v_up: A numpy array of the upwind radial values. Units of km/s.
        v_dn: A numpy array of the downwind radial values. Units of km/s.
        dtdr: Ratio of SURF time step and radial grid step. Units of s/km.
        alpha: Scale parameter for residual Solar wind acceleration.
        r_accel: Spatial scale parameter of residual solar wind acceleration. Units of km.
        rrel: The model radial grid relative to the radial inner boundary coordinate. Units of km.
    Returns:
         v_up_next: The upwind values at the next time step, numpy array with units of km/s.
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
def _upwind_step_accel_limit(v_up, v_dn, dtdr, alpha, r_accel, rrel):
    """
    Compute the next step in the upwind scheme of Burgers equation with added acceleration of the solar wind. Here, no
    acceleration is applied to speeds above 650km/s
    Args:
        v_up: A numpy array of the upwind radial values. Units of km/s.
        v_dn: A numpy array of the downwind radial values. Units of km/s.
        dtdr: Ratio of SURF time step and radial grid step. Units of s/km.
        alpha: Scale parameter for residual Solar wind acceleration.
        r_accel: Spatial scale parameter of residual solar wind acceleration. Units of km.
        rrel: The model radial grid relative to the radial inner boundary coordinate. Units of km.
    Returns:
         v_up_next: The upwind values at the next time step, numpy array with units of km/s.
    """

    n = len(v_dn)
    v_up_next = np.empty(n, dtype=np.float64)

    for i in range(n):
        # compute indices for accel arguments safely
        if i >= len(rrel) - 1:
            continue  # skip last point to avoid out-of-bounds

        accel_arg = -rrel[i] / r_accel
        accel_arg_p = -rrel[i + 1] / r_accel

        # Upwind scheme
        v_up_next[i] = v_up[i] - dtdr * v_up[i] * (v_up[i] - v_dn[i])

        # Acceleration factor
        denom = 1.0 + alpha * (1.0 - np.exp(accel_arg))
        v_source = v_dn[i] / denom

        # Residual acceleration
        v_diff = 0.0
        if v_source < 650.0:
            v_diff = alpha * v_source * (np.exp(accel_arg) - np.exp(accel_arg_p))

        # Add residual acceleration to upwind step
        v_up_next[i] += v_dn[i] * dtdr * v_diff

    return v_up_next

@jit(nopython=True)
def _upwind_step_compressible_(v_up, v_dn, rho_up, rho_dn, temp_up, temp_dn, 
                                dtdr, alpha, r_accel, rrel, r_boundary, gamma):
    """
    Compute the next step in the upwind scheme for the compressible solver.
    This includes velocity, density, and temperature evolution with compression/heating physics.
    Residual acceleration is NOT included - pressure gradient drives the flow.
    
    Args:
        v_up: A numpy array of the upwind velocity values. Units of km/s.
        v_dn: A numpy array of the downwind velocity values. Units of km/s.
        rho_up: A numpy array of the upwind density values. Units of kg/m^3.
        rho_dn: A numpy array of the downwind density values. Units of kg/m^3.
        temp_up: A numpy array of the upwind temperature values. Units of K.
        temp_dn: A numpy array of the downwind temperature values. Units of K.
        dtdr: Ratio of SURF time step and radial grid step. Units of s/km.
        alpha: Scale parameter for residual Solar wind acceleration (NOT USED in compressible solver).
        r_accel: Acceleration scale (NOT USED in compressible solver).
        rrel: The model radial grid relative to the radial inner boundary coordinate. Units of km.
        r_boundary: The inner boundary radius in km.
        gamma: Adiabatic index for compressible solver (typically 1.5 for solar wind).
        
    Returns:
        v_up_next: The upwind velocity values at the next time step. Units of km/s.
        rho_up_next: The upwind density values at the next time step. Units of kg/m^3.
        temp_up_next: The upwind temperature values at the next time step. Units of K.
    """

    # ====================================================================
    # Velocity evolution with pressure gradient force
    # Momentum equation: ∂v/∂t + v·∂v/∂r = -(1/ρ)·∂P/∂r
    # ====================================================================
    
    # Constants
    k_B = 1.38064852e-23  # J/K
    m_p = 1.67262192e-27  # kg
    
    # Radial coordinates
    r_up_km = rrel[:-1] * 695700.0 + r_boundary
    
    # Gradients (forward difference for upwind scheme)
    dv = v_dn - v_up
    drho = rho_dn - rho_up
    dtemp = temp_dn - temp_up
    
    # Pressure gradient term (1/rho * dP/dr)
    # P = rho * k_B * T / m_p
    # (1/rho) * dP = (k_B/m_p) * (dT + (T/rho) * drho)
    # Units: (J/K / kg) * (K + K) = J/kg = (m/s)^2
    term_P_SI = (k_B / m_p) * (dtemp + (temp_up / rho_up) * drho) # (m/s)^2
    
    # Convert to km/s^2 equivalent for update
    # 1 m^2/s^2 = 1e-6 km^2/s^2
    term_P_kms2 = 1e-6 * term_P_SI
    
    # Update velocity
    # v_new = v - dt * v * dv/dr - dt * 1/rho * dP/dr
    #       = v - v * (dt/dr) * dv - (dt/dr) * (1/rho * dP)
    v_up_next = v_up - v_up * dtdr * dv - dtdr * term_P_kms2
    
    # ====================================================================
    # Density evolution
    # Continuity equation: ∂ρ/∂t + v·∂ρ/∂r + ρ·∂v/∂r + 2ρv/r = 0
    # ====================================================================
    
    # Calculate dt from dtdr
    dr_km = (rrel[1] - rrel[0]) * 695700.0
    dt = dtdr * dr_km
    
    spherical_term = 2.0 * rho_up * v_up / r_up_km
    rho_up_next = rho_up - dtdr * (v_up * drho + rho_up * dv) - dt * spherical_term
    
    # ====================================================================
    # Temperature evolution
    # Energy equation: ∂T/∂t + v·∂T/∂r + (γ-1)T(∂v/∂r + 2v/r) = 0
    # ====================================================================
    
    div_v_dt = dtdr * dv + dt * 2.0 * v_up / r_up_km
    temp_up_next = temp_up - dtdr * v_up * dtemp - (gamma - 1.0) * temp_up * div_v_dt
    
    # Apply physical bounds to prevent instability
    v_up_next = np.maximum(100.0, np.minimum(v_up_next, 3000.0))
    temp_up_next = np.maximum(1e3, np.minimum(temp_up_next, 1e8))
    rho_up_next = np.maximum(1e-30, rho_up_next)
    
    return v_up_next, rho_up_next, temp_up_next


@jit(nopython=True)
def _is_in_cme_boundary_(r_boundary, lon, lat, time, cme_params):
    """
    Check whether a given lat, lon point on the inner boundary is within a given CME.
    Returns the fraction of the distance between nose and tail, which can be used to 
    produce a declining speed profile mimicking expansion
    Args:
        r_boundary: Height of model inner boundary.
        lon: A HEEQ latitude, in radians.
        lat: A HEEQ longitude, in radians.
        time: Model time step, in seconds
        cme_params: An array containing the cme parameters
    Returns:
         isincme: Boolean, True if coordinate is on or inside the CME domain.
         dist_from_nose: Float, fractional distance from nose to tail.
    """

    cme_t_launch = cme_params[0]
    cme_lon = cme_params[1]
    cme_lat = cme_params[2]
    cme_v = cme_params[4]
    cme_radius = cme_params[6]  # the physical width at the inner boundary
    cme_thickness = cme_params[7]
    cme_fixed_duration = cme_params[10]
    fixed_duration = cme_params[11]
    
    # change the cme speed so that it produces the correct pulse duration
    if cme_fixed_duration:
        cme_v = (cme_radius*2 + cme_thickness)/fixed_duration

    isincme = False
    dist_from_nose = 0.0

    # Compute y, the height of CME nose above the surface
    y = cme_v * (time - cme_t_launch)
    
    # compute x, the radius of the cme currently threading the inner boundary
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
    else:
        # the CME is not threading the inner boundary
        return False, dist_from_nose
            
    # convert x back to an angular width
    ang_width = np.arctan(x / r_boundary)
    
    # compute the angle between the given point and the CME centre
    delta_long = lon - cme_lon
    # Calculate the central angle between the reference point and the cme centroid
    # using the spherical law of cosines
    central_angle = np.arccos(np.sin(lat) * np.sin(cme_lat) + 
                              np.cos(lat) * np.cos(cme_lat) * np.cos(delta_long))

    if central_angle <= ang_width:
        isincme = True
        # also compute the fractional distance from nose to tail
        r_of_nose = cme_v * (time - cme_t_launch)
        nose_to_tail_r = (2 * cme_radius + cme_thickness)
        dist_from_nose = r_of_nose / nose_to_tail_r

    return isincme, dist_from_nose


def load_SURF_run(filepath):
    """
    Load in data from a saved SURF run. If Br fields are not saved, pads with
    NaN to avoid conflicts with other SURF routines.
    Args:
        filepath: The full path to a HDF5 file containing the output from SURF.save()
    Returns:
        cme_list: A list of instances of ConeCME
        model: An instance of SURF containing loaded results.
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
        frame = data['frame'][()].decode("utf-8")
        track_cmes = bool(data['track_cmes'][()])
        accel_limit = bool(data['accel_limit'][()])
        track_b = bool(data['track_b'][()])
        track_streak = bool(data['track_streak'][()])
        
        # Load compressible flag if it exists (for backward compatibility with older saved files)
        compressible = bool(data['compressible'][()]) if 'compressible' in data else False
        
        # Load solver if it exists (for backward compatibility with older saved files)
        loaded_solver = data['solver'][()].decode("utf-8") if 'solver' in data else 'huxt'
        legacy_solver_map = {
            'upwind': 'huxt',
            'hllc': 'hydro',
            'hllc-plm-rk2': 'hydro',
            'hllc-pcm': 'hydro-pcm',
            'cgf': 'hydro',
        }
        solver = legacy_solver_map.get(loaded_solver, loaded_solver)

        if track_b:
            b_boundary = data['_b_boundary_init_'][()]
            b_boundary_lons = data['b_boundary_lons'][()] * u.Unit(data['b_boundary_lons'].attrs['unit'])

        # Load compressible boundaries if they exist (for backward compatibility with older saved files)
        if compressible:
            if '_rho_boundary_init_' in data:
                rho_boundary = data['_rho_boundary_init_'][()]
                rho_boundary_lons = data['rho_boundary_lons'][()] * u.Unit(data['rho_boundary_lons'].attrs['unit'])
            else:
                rho_boundary = np.nan
                
            if '_temp_boundary_init_' in data:
                temp_boundary = data['_temp_boundary_init_'][()]
                temp_boundary_lons = data['temp_boundary_lons'][()] * u.Unit(data['temp_boundary_lons'].attrs['unit'])
            else:
                temp_boundary = np.nan

        # Create the model class
        # Build kwargs for constructor based on what's available
        constructor_kwargs = {
            'v_boundary': v_boundary,
            'cr_num': cr_num,
            'cr_lon_init': cr_lon_init,
            'r_min': r.min(),
            'r_max': r.max(),
            'simtime': simtime,
            'dt_scale': dt_scale,
            'latitude': lat,
            'frame': frame,
            'track_cmes': track_cmes,
            'accel_limit': accel_limit,
            'solver': solver
        }
        
        if track_b:
            constructor_kwargs['b_boundary'] = b_boundary
            
        if compressible:
            # Note: compressible boundaries are provided but model doesn't have compressible flag
            # They will be handled after model creation
            if 'rho_boundary' in locals():
                constructor_kwargs['rho_boundary'] = rho_boundary
            if 'temp_boundary' in locals():
                constructor_kwargs['temp_boundary'] = temp_boundary
        
        if nlon == 1:
            constructor_kwargs['lon_out'] = lon
            model = SURF(**constructor_kwargs)
        elif nlon > 1:
            constructor_kwargs['lon_start'] = lon.min()
            constructor_kwargs['lon_stop'] = lon.max()
            model = SURF(**constructor_kwargs)

        # Reset the longitudes, as when onlyt a wedge is simulated, it gets confused.
        model.lon = lon
        model.nlon = nlon

        model.v_grid = data['v_grid'][()] * u.Unit(data['v_boundary'].attrs['unit'])
        model.cme_particles_r = data['cme_particles_r'][()]
        model.cme_particles_v = data['cme_particles_v'][()]

        if track_b:
            model.b_boundary_lons = b_boundary_lons
            model.b_grid = data['b_grid'][()]
            model.hcs_particles_r = data['hcs_particles_r'][()] * u.Unit(data['hcs_particles_r'].attrs['unit'])
        
        if compressible:
            if '_rho_boundary_init_' in data:
                model.rho_boundary_lons = rho_boundary_lons
            if '_temp_boundary_init_' in data:
                model.temp_boundary_lons = temp_boundary_lons
            
            # Load rho_grid and temp_grid if they exist
            if 'rho_grid' in data:
                model.rho_grid = data['rho_grid'][()] * u.Unit(data['rho_grid'].attrs['unit'])
            if 'temp_grid' in data:
                model.temp_grid = data['temp_grid'][()] * u.Unit(data['temp_grid'].attrs['unit'])
        
        if track_streak:
            model.track_streak = track_streak
            model.streak_particles_r = data['streak_particles_r'][()] * u.Unit(data['streak_particles_r'].attrs['unit'])
            model.streak_lon_r0 = data['streak_lon_r0'][()]

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
            initial_height = cme_data['initial_height'][()] * u.Unit(cme_data['initial_height'].attrs['unit'])
            v = cme_data['v'][()] * u.Unit(cme_data['v'].attrs['unit'])
            
            # check for the new (post 4.2.1) cone CME parameters
            if 'cme_expansion' in cme_data:
                cme_expansion = cme_data['cme_expansion'][()]
                cme_fixed_duration = cme_data['cme_fixed_duration'][()]
                fixed_duration = cme_data['fixed_duration'][()] * u.Unit(cme_data['fixed_duration'].attrs['unit'])
                
                cme = ConeCME(t_launch=t_launch, longitude=lon, latitude=lat, v=v, width=width, thickness=thickness,
                              initial_height=initial_height, cme_expansion=cme_expansion,
                              cme_fixed_duration=cme_fixed_duration, fixed_duration=fixed_duration)
            else:
                cme = ConeCME(t_launch=t_launch, longitude=lon, latitude=lat, v=v, width=width, thickness=thickness,
                              initial_height=initial_height)
            
            cme.frame = cme_data['frame'][()].decode("utf-8")

            label = cme_data['frame'][()].decode("utf-8")
            if label == 'None':
                cme.label = None
            else:
                cme.label = label

            # Now sort out coordinates.
            # Use the same dictionary structure as defined in ConeCME._track_
            coords_group = cme_data['coords']
            coords_data = {j: {'time': np.array([]),
                               'model_time': np.array([]) * u.s,
                               'lon': np.array([]) * model.lon.unit,
                               'r': np.array([]) * u.km,
                               'lat': np.array([]) * u.rad,
                               'v': np.array([]) * u.km / u.s}
                           for j in range(len(coords_group))}

            for time_key, pos in coords_group.items():
                t = int(time_key.split("_")[2])
                time_out = Time(pos['time'][()], format="isot")
                time_out.format = 'jd'
                coords_data[t]['time'] = time_out
                coords_data[t]['model_time'] = pos['model_time'][()] * u.Unit(pos['model_time'].attrs['unit'])
                coords_data[t]['lon'] = pos['lon'][()] * u.Unit(pos['lon'].attrs['unit'])
                coords_data[t]['r'] = pos['r'][()] * u.Unit(pos['r'].attrs['unit'])
                coords_data[t]['v'] = pos['v'][()] * u.Unit(pos['v'].attrs['unit'])
                coords_data[t]['lat'] = pos['lat'][()] * u.Unit(pos['lat'].attrs['unit'])
                coords_data[t]['front_id'] = pos['front_id'][()] * u.Unit(pos['front_id'].attrs['unit'])

            cme.coords = coords_data
            cme_list.append(cme)

        # Update CMEs in model output
        model.cmes = cme_list

    else:
        # File doesn't exist return nothing
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

    return model, cme_list


@jit(nopython=True)
def bgrid_from_hcs(hcs_particles_r, input_b_ts, model_time, time_out, r_grid, lons):
    """
    Create the b polarity grid from the tracked HCS positions
    Args:
        hcs_particles_r: HCS r valus as a function of long and time, in km (model.hcs_particles_r)
        input_b_ts : Inner boundary b polarity time series (model.input_b_ts)
        model_time: Time (rel to model start) of inner boundary values (model.model_time)
        time_out : model output time steps
        r_grid: Radial grid in km
        lons : model output longitudes, in radians
    Returns:
        bgrid : b polarity grid as function of long, time, r
    """
    nt = len(time_out)
    nr = len(r_grid)
    nlon = len(lons)
    dr = r_grid[1] - r_grid[0]

    bgrid = np.ones((nt, nr, nlon)) * np.nan
    for ilon in range(0, nlon):

        # find the closest input longitude to the current longitude
        id_lon = ilon
        for t in range(0, nt):

            # find the closest input time to the current snapshot time
            thistime = time_out[t]
            id_t = np.argmin(np.abs(model_time - thistime))

            # make all the bgrid at this longitude equal to the inner boundary polarity
            bgrid[t, :, ilon] = input_b_ts[id_t, id_lon]

            # step through each HCS inversion and flip everything beyond each one
            hcs_crossings_r = hcs_particles_r[:, t, 0, ilon]
            hcs_crossings_p = hcs_particles_r[:, t, 1, ilon]

            # hcs crossings are ordered so the further is the first. flip them
            hcs_crossings_r = np.flip(hcs_crossings_r)
            hcs_crossings_p = np.flip(hcs_crossings_p)

            for ihcs in range(0, len(hcs_crossings_r)):
                if np.isfinite(hcs_crossings_r[ihcs]):
                    # find the r index corresponding to this HCS crossing
                    r_i = np.argmin(np.abs(hcs_crossings_r[ihcs] - r_grid + dr / 2))
                    # flip everything at and beyond this radius
                    if hcs_crossings_p[ihcs] > 0:
                        bgrid[t, r_i:, ilon] = -1.0
                    else:
                        bgrid[t, r_i:, ilon] = 1.0
    return bgrid


def get_version():
    return "1.0.0"


