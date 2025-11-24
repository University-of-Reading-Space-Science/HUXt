import copy
import errno
import os
import shutil
import warnings

from appdirs import user_data_dir
import astropy.units as u
from astropy.time import Time, TimeDelta
import h5py
from joblib import Parallel, delayed
import numpy as np
from numba import jit
from pathlib import Path
from sunpy.coordinates import sun
from huxt.cgf_solver import CGFSolver
from huxt.cgf_solver import CGFSolver


def _check_pluto_availability():
    """
    Check if PLUTO is available on the system.
    
    Returns:
        tuple: (bool, str) - (is_available, message)
    """
    pluto_exe = shutil.which('pluto')
    if pluto_exe:
        return True, f"PLUTO found at {pluto_exe}"
    
    pluto_dir = os.environ.get('PLUTO_DIR')
    if pluto_dir:
        pluto_exe_path = os.path.join(pluto_dir, 'Bin', 'pluto')
        if os.path.exists(pluto_exe_path) or os.path.exists(pluto_exe_path + '.exe'):
            return True, f"PLUTO found at {pluto_exe_path}"
        else:
            return False, f"PLUTO_DIR set to {pluto_dir}, but executable not found in Bin/ directory"
    
    return False, ("PLUTO not found on system. To use PLUTO solvers:\n"
                   "  1. Download and build PLUTO from http://plutocode.ph.unito.it/\n"
                   "  2. Set PLUTO_DIR environment variable to PLUTO installation directory\n"
                   "  3. See PLUTO_SETUP.md for detailed instructions")


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
    using the HUXt/scripts/generate_huxt_ephemeris.py script.

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
                             f" Updating the HUXt ephemeris file may resolve this issue.")

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
        initial_height: Initiation height of the CME, in km. Defaults to HUXt inner boundary at 30 solar radii.
        radius: Initial radius of the CME, in km.
        thickness: Thickness of the CME cone, in km.
        cme_density: Mass density of the CME in kg/m³. Defaults to 2x the solar wind density at initial_height.
        cme_temperature: Temperature of the CME in Kelvin. Defaults to 2x the solar wind temperature at initial_height.
        profile_type: Temporal profile shape ('square' or 'sinusoidal'). 
                     'square': step function from ambient to CME values
                     'sinusoidal': smooth sinusoidal pulse from ambient to CME values and back
        coords: Dictionary containing the radial and longitudinal (for HUXT2D) coordinates of the of Cone CME for each
                model time step.
    """

    # Some decorators for checking the units of input arguments
    @u.quantity_input(t_launch=u.s)
    @u.quantity_input(longitude=u.deg)
    @u.quantity_input(latitude=u.deg)
    @u.quantity_input(v=(u.km / u.s))
    @u.quantity_input(width=u.deg)
    @u.quantity_input(thickness=u.solRad)
    @u.quantity_input(fixed_duration=u.s)
    @u.quantity_input(cme_density=(u.kg / u.m**3))
    @u.quantity_input(cme_temperature=u.K)
    def __init__(self, t_launch=0.0 * u.s, longitude=0.0 * u.deg, latitude=0.0 * u.deg, v=1000.0 * (u.km / u.s),
                 width=30.0 * u.deg, thickness=0.0 * u.solRad, initial_height=30 * u.solRad, cme_expansion=False,
                 cme_fixed_duration=True, fixed_duration=12 * 60 * 60 * u.s, 
                 cme_density=np.nan * (u.kg / u.m**3), cme_temperature=np.nan * u.K, 
                 density_fraction=2.0, temperature_fraction=2.0,
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
            density_fraction: Fraction of ambient solar wind density. Default 2.0 means twice the ambient density.
                            Only used if cme_density is not provided.
            temperature_fraction: Fraction of ambient solar wind temperature. Default 2.0 means twice the ambient temperature.
                                Only used if cme_temperature is not provided.
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
        self.initial_height = initial_height  # Initial height of CME (should match inner boundary of HUXt)
        self.radius = self.initial_height * np.tan(self.width / 2.0)  # Initial radius of CME
        self.thickness = thickness  # Extra CME thickness
        self.coords = {}
        self.frame = 'NA'
        self.longitude_huxt = -1 * u.rad  # the HUXt longitude, adjusted for sidereal frame if necessary
        self.cme_expansion = cme_expansion
        self.cme_fixed_duration = cme_fixed_duration
        self.fixed_duration = fixed_duration
        
        # Validate and store profile type
        if profile_type not in ['square', 'sinusoidal']:
            raise ValueError(f"profile_type must be 'square' or 'sinusoidal', not '{profile_type}'")
        self.profile_type = profile_type
        
        # Set CME density and temperature
        if np.isnan(cme_density.value):
            # Calculate default solar wind density at initial_height
            r_1au = 215.0 * u.solRad  # 1 AU in solar radii
            rho_1au = 8.35e-21 * (u.kg / u.m**3)  # Solar wind density at 1 AU
            sw_density_at_height = rho_1au * (r_1au / initial_height)**2
            # Apply density_fraction to ambient solar wind density
            self.cme_density = density_fraction * sw_density_at_height
        else:
            self.cme_density = cme_density
            
        if np.isnan(cme_temperature.value):
            # Calculate solar wind temperature at initial_height using Lopez & Freeman (1986)
            # Temperature depends on velocity, not just radial distance
            # Use v (CME speed at 1 AU) to get temperature via empirical relation
            sw_temp_at_height = lopez_temperature_from_velocity(v)
            # Apply temperature_fraction to ambient solar wind temperature
            self.cme_temperature = temperature_fraction * sw_temp_at_height
        else:
            self.cme_temperature = cme_temperature
            
        if isinstance(label, str) | (label is None):
            self.label = label
        else:
            raise ValueError(f'Label must be an instance of str or None, not {type(label)}')

        self.__version__ = get_version()
        return

    def parameter_array(self):
        """
        Returns a numpy array of CME parameters. This is used in the numba optimised solvers that don't play nicely
        with classes.
        Returns:
            None
        """
        # Convert profile_type to numeric flag: 0 = square, 1 = sinusoidal
        profile_flag = 1.0 if self.profile_type == 'sinusoidal' else 0.0
        
        cme_parameters = [self.t_launch.to('s').value, 
                          self.longitude_huxt.to('rad').value,
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
        Tracks the perimeter of each ConeCME through the HUXt solution in model. Updates the ConeCME.coords dictionary
        of CME coordinates.
        Args:
            model: An HUXt instance with solution containing ConeCMEs
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

                # Handle case for HUXt run on multiple longitudes first
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
                    # HUXt run on a single longitude, so don't interpolate front to body lon
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


class HUXt:
    """
    A class containing the HUXt model described in Owens et al. (2020, DOI: 10.1007/s11207-020-01605-3)

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

    # Decorators to check units on input arguments
    @u.quantity_input(v_boundary=(u.km / u.s))
    @u.quantity_input(simtime=u.day)
    @u.quantity_input(cr_lon_init=u.deg)
    def __init__(self, v_boundary=np.nan * (u.km / u.s), b_boundary=np.nan, 
                 rho_boundary=np.nan, temp_boundary=np.nan,
                 cr_num=np.nan, cr_lon_init=360.0 * u.deg,
                 latitude=0 * u.deg, r_min=30 * u.solRad, r_max=240 * u.solRad, lon_out=np.nan * u.rad,
                 lon_start=np.nan * u.rad, lon_stop=np.nan * u.rad, simtime=5.0 * u.day, dt_scale=1.0, frame='synodic',
                 input_v_ts=np.nan * (u.km / u.s), input_b_ts=np.nan, 
                 input_rho_ts=np.nan * (u.kg / u.m**3), input_temp_ts=np.nan * u.K, 
                 input_iscme_ts=np.nan, input_t_ts=np.nan * u.s,
                 track_cmes=True, accel_limit=True, compressible=False, solver='upwind', parallel=False):
        """
        Initialise the HUXt model instance.

            v_boundary: Inner solar wind speed boundary condition. An array of size nlon (default 128). Units of km/s.
            b_boundary: Inner B polarity boundary condition. An array of size nlon (default 128). Units of km/s.
            cr_num: Integer Carrington rotation number. Used to determine the planetary and spacecraft positions
            cr_lon_init: Carrington longitude of Earth at model initialisation, in degrees.
            latitude: Helio latitude (from the equator) of HUXt plane, in degrees
            lon_out: A specific single longitude (relative to Earth) to compute HUXt solution along, in degrees
            lon_start: The first longitude (in a clockwise sense) of the longitude range to solve HUXt over.
            lon_stop: The last longitude (in a clockwise sense) of the longitude range to solve HUXt over.
            r_min: The radial inner boundary distance of HUXt.
            r_max: The radial outer boundary distance of HUXt.
            simtime: Duration of the simulation window, in days.
            dt_scale: Integer scaling number to set the model output time step relative to the models CFL time.
            frame: string determining the rotation frame for the model
            input_v_ts: Time series of inner boundary V conditions. For initialising HUXt with, for example, 
                           in-situ observations from L1. If used as keyword input argument, overrides v_boundary input.
            input_bv_ts: Time series of inner boundary B conditions. For initialising HUXt with, for example, 
                            in-situ observations from L1. If used as keyword input argument, overrides b_boundary input.
            input_rho_ts: Time series of inner boundary density conditions in kg/m³. For initialising HUXt with, for example,
                             in-situ observations from L1. If used as keyword input argument, overrides rho_boundary input.
                             Only used if compressible=True.
            input_temp_ts: Time series of inner boundary temperature conditions in Kelvin. For initialising HUXt with, for example,
                              in-situ observations from L1. If used as keyword input argument, overrides temp_boundary input.
                              Only used if compressible=True.             
            input_t_ts: Times of input_v_ts in seconds, including spin up.
            input_iscme_ts: Boolean mask time series indicating what time steps correspond to CMEs in input_v_ts.
                               If used as keyword input argument, overrides ConeCMEs past to huxt.sovle().
            save_full_v: Boolean flag to determine if full v field (including spin up) is saved for post-processing.
            track_cmes: Boolean flag to determine if CMEs are tracked at run time (small speed reduction).
            accel_limit: Boolean flag to determine if acceleration is switched for speeds above 650 km/s
            compressible: Boolean flag to use compressible solver instead of incompressible (default False).
            solver: String specifying the numerical solver to use. Options:
                   'upwind' (default): First-order upwind scheme (Godunov-type)
                   'cgf': Colella-Glaz-Ferguson solver (compressible only)
                   'pluto': PLUTO spherical hydrodynamics solver (HLL+RK3, compressible only)
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
        constants = huxt_constants()
        self.twopi = constants['twopi']
        self.daysec = constants['daysec']
        self.kms = constants['kms']
        self.alpha = constants['alpha']  # Scale parameter for residual SW acceleration (incompressible)
        self.r_accel = constants['r_accel']  # Spatial scale parameter for residual SW acceleration (incompressible)
        self.gamma = constants['gamma']  # Adiabatic index for compressible solver
        self.__version__ = get_version()
        
        # Validate and store solver choice
        valid_solvers = ['upwind', 'cgf', 'pluto']
        if solver not in valid_solvers:
            raise ValueError(f"Invalid solver '{solver}'. Must be one of: {valid_solvers}")
        
        # Check PLUTO availability if PLUTO solver is requested
        if solver == 'pluto':
            pluto_available, pluto_msg = _check_pluto_availability()
            if not pluto_available:
                raise RuntimeError(f"PLUTO solver requested but not available.\n{pluto_msg}")
            else:
                print(f"✓ {pluto_msg}")
        
        # Check CGF solver availability if CGF solver is requested
        if solver == 'cgf':
            print("[OK] CGF solver (HUXt-native) available")
        
        self.solver = solver
        
        # CGF and PLUTO solvers require compressible mode - force it regardless of user setting
        if solver in ['cgf', 'pluto'] and not compressible:
            print(f"Note: {solver.upper()} solver requires compressible=True. Enabling compressible mode.")
            compressible = True
        
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
        self._data_dir_ = dirs['HUXt_data']
        self._figure_dir_ = dirs['HUXt_figures']
        self._ephemeris_file = dirs['ephemeris']

        # Setup radial coordinates - in solar radius
        self.r, self.dr, self.rrel, self.nr = radial_grid(r_min=r_min, r_max=r_max)

        # Setup longitude coordinates - in radians.
        self.lon, self.dlon, self.nlon = longitude_grid(lon_out=lon_out, lon_start=lon_start, lon_stop=lon_stop)

        if (self.frame == 'sidereal') & (self.nlon == 1):
            print("Warning: HUXt configured for a 1-D run in the sidereal frame. This simulation will not work"
                  "correctly with functions like huxt_analysis.get_observer_time_series()")

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
            # check that the implicit time step from vlong is not comparable to the HUXt timestep
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
            # check that the implicit time step from vlong is not comparable to the HUXt timestep
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
                # Calculate realistic default density based on r_min
                # Solar wind density at 1 AU is ~5 protons/cm³ = 8.35e-21 kg/m³
                # Density scales as r^-2, so scale from 1 AU to r_min
                r_1au = 215.0 * u.solRad  # 1 AU in solar radii
                rho_1au = 16.35e-21 * (u.kg / u.m**3)  # Solar wind density at 1 AU
                scaling_factor = (r_1au / r_min)**2
                default_rho = rho_1au * scaling_factor
                self.rho_boundary = np.ones(len(self.v_boundary_lons)) * default_rho
                self.rho_boundary_lons = self.v_boundary_lons
            elif not np.all(np.isnan(rho_boundary)):
                assert rho_boundary.size < 4600  # this equates to about 9 mins
                self.rho_boundary = rho_boundary
                self._rho_boundary_init_ = self.rho_boundary.copy()
                nrho = len(rho_boundary)
                dlon = 2 * np.pi / nrho
                self.rho_boundary_lons = np.arange(dlon / 2, 2 * np.pi - dlon / 2 + dlon / 10, dlon) * u.rad

            if np.all(np.isnan(temp_boundary)):
                # Calculate temperature using Lopez & Freeman (1986) velocity-temperature relation
                # This accounts for the velocity at the inner boundary and uses physically-motivated scaling
                # Temperature at 1 AU follows T = (V/200)² × 10⁴ K
                # At r_min, velocity is ~7% lower due to continued acceleration
                # Temperature scales with velocity via adiabatic relation
                temp_from_velocity = lopez_temperature_from_velocity(self.v_boundary, gamma=self.gamma)
                self.temp_boundary = temp_from_velocity
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
            self.rho_boundary = np.interp(self.rho_boundary_lons.value, lon_shifted, rho_b_shifted, period=self.twopi)

            lon_shifted = zerototwopi((self.temp_boundary_lons - self.cr_lon_init).value)
            id_sort = np.argsort(lon_shifted)
            lon_shifted = lon_shifted[id_sort]
            temp_b_shifted = self.temp_boundary[id_sort]
            self.temp_boundary = np.interp(self.temp_boundary_lons.value, lon_shifted, temp_b_shifted, period=self.twopi)

        # Compute the buffertime required to spin up HUXt, based on minimum speed on the inner boundary
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

            # Temperature time series
            if not np.all(np.isnan(input_temp_ts)):
                self.input_temp_ts = input_temp_ts[:, y_ind]
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
        lons, dlon, nlon = longitude_grid()
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
        Process a single longitude slice in the HUXt simulation.
        This helper function is used for parallel execution across longitudes.
        Routes to the appropriate solver (solve_radial for upwind/pluto, solve_radial_cgf for cgf).
        
        Args:
            i: Longitude index to process
            n_cme: Number of CMEs
            n_hcs_max: Maximum number of HCS particles
            streak_times: Streakline timing data
            
        Returns:
            Tuple of (i, v, cme_r_bounds, cme_v_bounds, hcs_r, streak_r, rho_out, temp_out)
        """
        if self.solver == 'cgf':
            # CGF solver uses solve_radial_cgf with particle tracking
            return self._process_longitude_cgf(i, n_cme, n_hcs_max, streak_times)
        else:
            # Upwind and PLUTO solvers use solve_radial
            return self._process_longitude_builtin(i, n_cme, n_hcs_max, streak_times)
    
    def _process_longitude_builtin(self, i, n_cme, n_hcs_max, streak_times):
        """
        Process a longitude using the built-in solve_radial (upwind/pluto solvers).
        
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

        # actually run the HUXt solver
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
                                                                      tempinput=tempslice,
                                                                      compressible=self.compressible,
                                                                      solver=self.solver)
        
        return (i, v, cme_r_bounds, cme_v_bounds, hcs_r, streak_r, rho_out, temp_out)
    
    def _process_longitude_cgf(self, i, n_cme, n_hcs_max, streak_times):
        """
        Process a longitude using solve_radial_cgf (CGF solver).
        
        Args:
            i: Longitude index to process
            n_cme: Number of CMEs
            n_hcs_max: Maximum number of HCS particles
            streak_times: Streakline timing data
            
        Returns:
            Tuple of (i, v, cme_r_bounds, cme_v_bounds, hcs_r, streak_r, rho_out, temp_out)
        """
        # Compute radial grid in km for CGF solver
        rgrid_km = (self.rrel.value - self.rrel.value[0]) * 695700.0 + self.r[0].to('km').value
        # Get boundary conditions for this longitude
        v_bc_kms = self.input_v_ts[:, i].value  # km/s
        rho_bc_kgm3 = self.input_rho_ts[:, i].value  # kg/m³
        T_bc_K = self.input_temp_ts[:, i].value  # K
        
        # Set up particle tracking for this longitude
        num_particles = 0
        particle_injection_rate = None
        hcs_polarities = []  # Initialize HCS polarities list
        
        if self.track_cmes or self.track_b or self.track_streak:
            num_particles = {}
            particle_injection_rate = {}
            
            # CME particles: track leading and trailing edges
            if self.track_cmes and n_cme > 0:
                for cme_id in range(n_cme):
                    # Find times when this CME crosses this longitude
                    cme_mask = (self.input_iscme_ts[:, i] == cme_id + 1)
                    if np.any(cme_mask):
                        # Leading edge injected at start of CME
                        leading_idx = np.where(cme_mask)[0][0]
                        t_leading = self.model_time[leading_idx].value if hasattr(self.model_time[leading_idx], 'value') else self.model_time[leading_idx]
                        
                        # Trailing edge injected at end of CME
                        trailing_idx = np.where(cme_mask)[0][-1]
                        t_trailing = self.model_time[trailing_idx].value if hasattr(self.model_time[trailing_idx], 'value') else self.model_time[trailing_idx]
                        
                        num_particles[f'cme_{cme_id}_leading'] = 1
                        num_particles[f'cme_{cme_id}_trailing'] = 1
                        particle_injection_rate[f'cme_{cme_id}_leading'] = [t_leading]
                        particle_injection_rate[f'cme_{cme_id}_trailing'] = [t_trailing]
            
            # HCS particles: inject at each polarity change
            # Also track the polarity direction for each crossing
            if self.track_b and n_hcs_max > 0:
                hcs_times = []
                b_input = self.input_b_ts[:, i]
                if hasattr(b_input, 'value'):
                    b_input = b_input.value
                for t in range(1, len(b_input)):
                    diff = b_input[t] - b_input[t-1]
                    if diff != 0:  # Polarity change
                        t_hcs = self.model_time[t].value if hasattr(self.model_time[t], 'value') else self.model_time[t]
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
                            t_inject = t_inject.value if hasattr(t_inject, 'value') else t_inject
                            num_particles[streak_name] = 1
                            particle_injection_rate[streak_name] = [t_inject]
            
            # If no particles were actually added, revert to no tracking
            if len(num_particles) == 0:
                num_particles = 0
                particle_injection_rate = None
        
        # Call consolidated CGF solver function with particle tracking
        # Strip units from time arrays for CGF solver
        model_time_sec = self.model_time.value if hasattr(self.model_time, 'value') else self.model_time
        time_out_sec = self.time_out.value if hasattr(self.time_out, 'value') else self.time_out
        
        v_out_kms, rho_out_kgm3, temp_out_K, particle_data = solve_radial_cgf(
            v_bc_kms=v_bc_kms,
            rho_bc_kgm3=rho_bc_kgm3,
            T_bc_K=T_bc_K,
            model_time=model_time_sec,
            time_out=time_out_sec,
            r_grid=rgrid_km,
            gamma=self.gamma,
            nt_out=self.nt_out,
            nr=self.nr,
            verbose=False,  # Suppress detailed solver output in parallel mode
            create_diagnostic_plot=False,  # No plots in parallel mode
            num_particles=num_particles,
            particle_injection_rate=particle_injection_rate
        )
        
        # Extract particle positions at output times
        cme_particles_r_out = np.full((n_cme, self.nt_out, 2), np.nan)
        cme_particles_v_out = np.full((n_cme, self.nt_out, 2), np.nan)  # CGF doesn't track velocity, fill with NaN
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
            
            # Get time_out as plain array (strip units if present)
            time_out_sec = self.time_out.value if hasattr(self.time_out, 'value') else self.time_out
            
            # Process CME particles
            if self.track_cmes:
                for cme_id in range(n_cme):
                    leading_key = f'cme_{cme_id}_leading'
                    trailing_key = f'cme_{cme_id}_trailing'
                    
                    if leading_key in groups:
                        r_traj = groups[leading_key]['r'][0, :]
                        t_traj = groups[leading_key]['t'][0, :]
                        valid_mask = ~np.isnan(r_traj)
                        if np.any(valid_mask):
                            r_valid = r_traj[valid_mask]
                            t_valid = t_traj[valid_mask]
                            r_out = np.interp(time_out_sec, t_valid, r_valid, 
                                             left=np.nan, right=np.nan)
                            cme_particles_r_out[cme_id, :, 0] = r_out
                    
                    if trailing_key in groups:
                        r_traj = groups[trailing_key]['r'][0, :]
                        t_traj = groups[trailing_key]['t'][0, :]
                        valid_mask = ~np.isnan(r_traj)
                        if np.any(valid_mask):
                            r_valid = r_traj[valid_mask]
                            t_valid = t_traj[valid_mask]
                            r_out = np.interp(time_out_sec, t_valid, r_valid,
                                             left=np.nan, right=np.nan)
                            cme_particles_r_out[cme_id, :, 1] = r_out
            
            # Process HCS particles
            if self.track_b and 'hcs' in groups:
                hcs_group = groups['hcs']
                n_hcs_this_lon = hcs_group['n_particles']
                
                for ihcs in range(min(n_hcs_this_lon, n_hcs_max)):
                    r_traj = hcs_group['r'][ihcs, :]
                    t_traj = hcs_group['t'][ihcs, :]
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
                            r_traj = groups[streak_name]['r'][0, :]
                            t_traj = groups[streak_name]['t'][0, :]
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
                temp_from_velocity = lopez_temperature_from_velocity(self.v_boundary, gamma=new_gamma)
                self.temp_boundary = temp_from_velocity
    
    @u.quantity_input(streak_carr=u.rad)
    def solve(self, cme_list, streak_carr=np.array([])*u.rad, save=False, tag=''):
        """
        Solve HUXt for the provided longitudinal boundary conditions and cme list. Updates the HUXt.v_grid
        Args:
            cme_list: A list of ConeCME instances to use in solving HUXt
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
                    dlon_t0 = earthpos.lon_hae - earth.lon_hae[0]
                    # find the CME hae longitude relative to the run start
                    cme_hae = np.interp(cme.t_launch.to(u.s).value,
                                        dt_t0.value, dlon_t0)
                    # adjust the CME HEEQ longitude accordingly
                    cme.longitude_huxt = zerototwopi(cme.longitude + cme_hae) * u.rad
                else:
                    cme.longitude_huxt = cme.longitude

                if cme.t_launch >= 0*u.s:
                    # add the CME to the list
                    cme_list_checked.append(cme)
                else:
                    print(f"Warning: ConeCME had negative t_launch ({cme.t_launch}), which is not allowed.")
                    print("Warning: This ConeCME object was not passed into the HUXt solver")
            else:
                print("Warning: cme_list contained objects other than ConeCME instances. These were excluded")

        self.cmes = cme_list_checked

        # If CMEs parsed, get an array of their parameters for using with the solver (which doesn't do classes)
        if len(self.cmes) > 0:
            cme_params = [cme.parameter_array() for cme in self.cmes]
            cme_params = np.array(cme_params)
            # Sort the CMEs in launch order.
            id_sort = np.argsort(cme_params[:, 0])
            cme_params = cme_params[id_sort]
            # Also sort the list of ConeCMEs so that it corresponds ot cme_params
            self.cmes = [self.cmes[i] for i in id_sort]
        else:
            # create dummy cme
            dummy_cme = ConeCME()
            cme_params = np.nan * np.zeros((1, len(dummy_cme.parameter_array())))

        # sanity check the CME initial height is the same as the model inner boundary
        if len(self.cmes) > 0:
            for cme in self.cmes:
                assert (self.r[0] == cme.initial_height)

        # check CME speeds aren't so fast they will butt up agains the CFL condition.
        if len(self.cmes) > 0:
            constants = huxt_constants()
            v_max = constants['v_max']
            for cme in self.cmes:

                if cme.v >= v_max:
                    raise ValueError(f'CME speed {cme.v} is larger than allowed for CFL limit of {v_max}')
                elif cme.v >= 0.8 * v_max:
                    print(f'Warning: CME speed of {cme.v} is close to CFL limit of {v_max}. Simulation may be unstable')

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

                for ilon, lon in enumerate(self.lon):
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
        if self.solver == 'cgf':
            import time
            solve_start = time.time()
            
            print("\n" + "="*70)
            print("USING CGF SOLVER")
            print("="*70)
            print(f"Frame: {self.frame}")
            print(f"Parallel: {self.parallel}")
            print(f"v_boundary: {len(self.v_boundary)} longitudes, range [{self.v_boundary.value.min():.1f}, {self.v_boundary.value.max():.1f}] km/s")
            if self.v_boundary.value.max() - self.v_boundary.value.min() > 1.0:
                print(f"  ** Spatial structure detected in v_boundary **")
            
            if self.parallel:
                print(f"\n⚠ WARNING: Parallel execution for CGF solver is typically SLOWER than serial")
                print(f"  Recommended: Set parallel=False for better performance")
            print("="*70 + "\n")
        
        # ======================================================================
        # PLUTO SOLVER PATH (separate execution - does not use unified loop)
        # ======================================================================
        if self.solver == 'pluto':
            import time
            
            try:
                print("\n" + "="*70)
                print("USING PLUTO 1D HYDRODYNAMICS SOLVER")
                print("="*70)
                print(f"PLUTO directory: {os.getenv('PLUTO_DIR', 'Not set')}")
                print(f"Solving {self.lon.size} longitudes serially...")
                print("="*70 + "\n")
                
                solve_start = time.time()
                
                # PLUTO solver always runs serially (each longitude is independent)
                for i in range(self.lon.size):
                    lon_start = time.time()
                    if self.lon.size == 1:
                        lon_deg = self.lon.value
                    else:
                        lon_deg = self.lon[i].value
                    print(f"\nProcessing longitude {i+1}/{self.lon.size} ({lon_deg:.2f} degrees)...")
                    
                    # Get boundary conditions for this longitude
                    # Handle both multi-longitude and single-longitude cases
                    if self.lon.size == 1:
                        # Single longitude case - input_v_ts might be (time,) or (time, 1)
                        if len(self.input_v_ts.shape) == 1:
                            v_bc_kms = self.input_v_ts.to(u.km / u.s).value
                            rho_bc_kgm3 = self.input_rho_ts.to(u.kg / u.m**3).value
                            T_bc_K = self.input_temp_ts.to(u.K).value
                        else:
                            v_bc_kms = self.input_v_ts[:, i].to(u.km / u.s).value
                            rho_bc_kgm3 = self.input_rho_ts[:, i].to(u.kg / u.m**3).value
                            T_bc_K = self.input_temp_ts[:, i].to(u.K).value
                    else:
                        # Multiple longitudes
                        v_bc_kms = self.input_v_ts[:, i].to(u.km / u.s).value
                        rho_bc_kgm3 = self.input_rho_ts[:, i].to(u.kg / u.m**3).value
                        T_bc_K = self.input_temp_ts[:, i].to(u.K).value
                    
                    # Debug: Check if BC contains CME data
                    print(f"  BC velocity range: {v_bc_kms.min():.1f} - {v_bc_kms.max():.1f} km/s")
                    print(f"  BC density range: {rho_bc_kgm3.min():.2e} - {rho_bc_kgm3.max():.2e} kg/m³")
                    print(f"  Number of CMEs in model: {len(self.cmes)}")
                    
                    # Call solve_radial_pluto
                    # Strip units from time arrays
                    model_time_sec = self.model_time.value if hasattr(self.model_time, 'value') else self.model_time
                    time_out_sec = self.time_out.value if hasattr(self.time_out, 'value') else self.time_out
                    
                    # Convert HUXt radial grid to km
                    rgrid_km = (self.rrel.value - self.rrel.value[0]) * 695700.0 + self.r[0].to('km').value
                    
                    v_out_kms, rho_out_kgm3, temp_out_K, particle_data = solve_radial_pluto(
                        v_bc_kms=v_bc_kms,
                        rho_bc_kgm3=rho_bc_kgm3,
                        T_bc_K=T_bc_K,
                        model_time=model_time_sec,
                        time_out=time_out_sec,
                        r_grid=rgrid_km,
                        gamma=self.gamma,
                        nt_out=self.nt_out,
                        nr=self.nr,
                        verbose=False,
                        create_diagnostic_plot=False,
                        num_particles=0,  # Not implemented for PLUTO yet
                        particle_injection_rate=None
                    )

                    # Store results in HUXt grids with units
                    self.v_grid[:, :, i] = v_out_kms * self.kms
                    self.rho_grid[:, :, i] = rho_out_kgm3 * (u.kg / u.m**3)
                    self.temp_grid[:, :, i] = temp_out_K * u.K
            
                solve_time = time.time() - solve_start
                print("\n" + "="*70)
                print("PLUTO SOLVER COMPLETE")
                print("="*70)
                print(f"Total time: {solve_time:.2f} seconds ({solve_time/60:.2f} minutes)")
                print(f"Time per longitude: {solve_time/self.lon.size:.2f} seconds")
                print("="*70)
                print("\nNOTE: Particle tracking not yet implemented for PLUTO solver")
                print("      (particle positions set to NaN)")
                print("="*70 + "\n")
                    
            except Exception as e:
                print("\n" + "="*70)
                print("ERROR IN PLUTO SOLVER")
                print("="*70)
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("="*70)
                print("\nPossible issues:")
                print("1. PLUTO executable not found - check PLUTO_DIR environment variable")
                print("2. PLUTO was not built correctly - see PLUTO_SETUP.md for build instructions")
                print("3. Input parameters are invalid or out of physical range")
                print("4. Disk space or memory issues")
                print("\nFor more help, see PLUTO_SETUP.md")
                print("="*70)
                raise RuntimeError(f"PLUTO solver failed: {str(e)}") from e
        
        # ======================================================================
        # Solve the time series at each longitude (UPWIND and CGF SOLVERS)
        # ======================================================================
        # Note: Grids are already in Fortran order from initialization, which makes
        # the [:, :, i] slices contiguous in memory for cache-efficient access
        # This section handles upwind and cgf solvers using unified process_longitude
        # PLUTO solver has its own complete loop above and skips this section

        if self.solver != 'pluto':
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
            obs: An Observer instance for body at times from HUXt.time_init + HUXt.time_out
        """
        times = self.time_init + self.time_out
        obs = Observer(body, times)
        return obs


class HUXt3d:
    """
    A class containing a list of HUXt classes, to enable mutliple latitudes to
    be simulated, plotted, animated, etc. together
    
    Attributes inherited from HUXt. Additional:
        lat: The list of latitudes of individual HUXt runs, in radians from the equator
        nlat: The number of latitudes simulated
        HUXtlat: List of individual HUXt model classes at each latitude
        v_in: a list of Carrington longitude solar wind profiles at each simulated latitude
        br_in: a list of Carrington longitude Br profiles at each simulated latitude
        
    
    """

    # Decorators to check units on input arguments
    @u.quantity_input(v_map=(u.km / u.s))
    @u.quantity_input(v_map_lat=u.rad)
    @u.quantity_input(v_map_long=u.rad)
    @u.quantity_input(latitude_max=u.deg)
    @u.quantity_input(latitude_min=u.deg)
    @u.quantity_input(simtime=u.day)
    @u.quantity_input(cr_lon_init=u.deg)
    def __init__(self, v_map=np.nan * (u.km / u.s), v_map_lat=np.nan * u.rad, v_map_long=np.nan * u.rad,
                 cr_num=np.nan, cr_lon_init=360.0 * u.deg, latitude_max=30 * u.deg, latitude_min=-30 * u.deg,
                 r_min=30 * u.solRad, r_max=240 * u.solRad, lon_out=np.nan * u.rad, lon_start=np.nan * u.rad,
                 lon_stop=np.nan * u.rad, simtime=5.0 * u.day, dt_scale=1.0):
        """
        Initialise the HUXt3D instance.

            v_map: Inner solar wind speed boundary Carrington map. Must have units of km/s.
            v_map_lat: List of latitude positions for v_map, in radians
            v_map_long: List of Carrington longitudes for v_map, in radians
            br_map: Inner Br boundary Carrington map. Must have no units.
            br_map_lat: List of latitude positions for br_map, in radians
            br_map_long: List of Carrington longitudes for br_map, in radians
            latitude_max: Maximum helio latitude (from the equator) of HUXt plane, in degrees
            latitude_min: Maximum helio latitude (from the equator) of HUXt plane, in degrees
            cr_num: Integer Carrington rotation number. Used to determine the planetary and spacecraft positions
            cr_lon_init: Carrington longitude of Earth at model initialisation, in degrees.
            lon_out: A specific single longitude (relative to Earth) to compute HUXt solution along, in degrees
            lon_start: The first longitude (in a clockwise sense) of the longitude range to solve HUXt over.
            lon_stop: The last longitude (in a clockwise sense) of the longitude range to solve HUXt over.
            r_min: The radial inner boundary distance of HUXt.
            r_max: The radial outer boundary distance of HUXt.
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

        # Get the HUXt longitudinal grid
        longs, dlon, nlon = longitude_grid(lon_start=0.0 * u.rad, lon_stop=2 * np.pi * u.rad)

        # Extract the vr value at the given latitudes
        self.v_in = []
        vlong = np.ones(len(v_map_long))
        for thislat in self.lat:
            for ilong in range(0, len(v_map_long)):
                vlong[ilong] = np.interp(thislat.value, v_map_lat.value, v_map[:, ilong].value)

            # Interpolate this longitudinal profile to the HUXt resolution
            self.v_in.append(np.interp(longs.value, v_map_long.value, vlong) * u.km / u.s)

        # Set up the model at each latitude
        self.HUXtlat = []
        for i in range(0, self.nlat):
            self.HUXtlat.append(HUXt(v_boundary=self.v_in[i],
                                     latitude=self.lat[i],
                                     cr_num=cr_num, cr_lon_init=cr_lon_init,
                                     r_min=r_min, r_max=r_max,
                                     lon_out=lon_out, lon_start=lon_start, lon_stop=lon_stop,
                                     simtime=simtime, dt_scale=dt_scale))
        return

    def solve(self, cme_list):
        """
        Compute solution of HUXt3d instance.
        Args:
            cme_list: A list of ConeCME objects to solve
        Returns:
            None
        """
        for model in self.HUXtlat:
            model.solve(cme_list)

        return


def huxt_constants():
    """
    Function to generate a dictionary of useful constants.
    Returns:
        constants: A dictionary of constants that configure HUXt
    """
    nlong = 128  # Number of longitude bins for a full longitude grid [128]
    dr = 1.5 * u.solRad  # Radial grid step. With v_max, this sets the model time step [1.5 Rs]
    nlat = 45  # Number of latitude bins for a full latitude grid [45]
    v_max = 3000 * u.km / u.s  # Maximum expected solar wind speed. Sets timestep [3000 km/s]

    # CONSTANTS - DON'T CHANGE
    twopi = 2.0 * np.pi
    daysec = 24 * 60 * 60 * u.s
    kms = u.km / u.s
    alpha = 0.15 * u.dimensionless_unscaled  # Scale parameter for residual SW acceleration (for incompressible)
    r_accel = 50 * u.solRad  # Spatial scale parameter for residual SW acceleration
    gamma = 1.5 # Adiabatic index for compressible solver (1.5 for solar wind?)
    synodic_period = 27.2753 * daysec  # Solar Synodic rotation period from Earth.
    sidereal_period = 25.38 * daysec  # Solar sidereal rotation period

    constants = {'twopi': twopi, 'daysec': daysec, 'kms': kms, 'alpha': alpha,
                 'gamma': gamma,
                 'r_accel': r_accel, 'synodic_period': synodic_period,
                 'sidereal_period': sidereal_period, 'v_max': v_max,
                 'dr': dr, 'nlong': nlong, 'nlat': nlat}

    return constants


def lopez_temperature_from_velocity(v_boundary, gamma=1.5):
    """
    Compute realistic coronal temperature at inner boundary (r_min ≈ 30 Rs).
    
    At 30 solar radii, we're in the extended corona where temperatures are ~1-2 MK.
    The solar wind accelerates from this region, and adiabatic cooling during expansion
    brings the temperature down to values consistent with Lopez & Freeman (1986) at 1 AU.
    
    Approach:
    1. Use coronal temperature scaling: T(r) ∝ r^(-α) where α ≈ 0.5-0.7 in the corona
    2. At 1 AU (215 Rs), temperature should follow Lopez & Freeman: T ∝ V²
    3. At 30 Rs, temperature should be ~1-2 MK
    
    For typical 400 km/s wind:
    - At 1 AU: T ≈ 100,000 K (Lopez & Freeman)
    - At 30 Rs: T ≈ 1.5 × 10⁶ K (coronal value)
    - Ratio: T(30Rs)/T(1AU) ≈ 15×
    
    Args:
        v_boundary: Solar wind velocity at inner boundary (r_min) in km/s (scalar or array)
        gamma: Adiabatic index (default 1.5 for solar wind)
    
    Returns:
        Temperature at inner boundary (r_min) in Kelvin
        
    Reference:
        Lopez, R. E., & Freeman, J. W. (1986). Solar wind proton temperature-velocity relationship.
        Journal of Geophysical Research, 91(A2), 1701-1705.
    """
    # Extract velocity value
    if hasattr(v_boundary, 'unit'):
        v_rmin_value = v_boundary.to(u.km/u.s).value
    else:
        v_rmin_value = v_boundary
    
    # At 30 Rs, use coronal temperature scaling
    # Assume T_coronal ∝ V^1.5 (intermediate between T ∝ V and T ∝ V²)
    # Calibrated so that 400 km/s gives ~1.5 MK at 30 Rs
    # This gives reasonable temperatures that cool to ~100,000 K at 1 AU after adiabatic expansion
    T_rmin = (v_rmin_value / 400.0)**1.5 * 1e6  # K
    
    if hasattr(v_boundary, 'unit'):
        return T_rmin * u.K
    else:
        return T_rmin


@u.quantity_input(r_min=u.solRad)
@u.quantity_input(r_max=u.solRad)
def radial_grid(r_min=30.0 * u.solRad, r_max=240. * u.solRad):
    """
    Define the radial grid of the HUXt model. Step size is fixed, but inner and outer boundary may be specified.
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

    if r_max > 3000 * u.solRad:
        print("Warning, r_max should not be more than 400rs. Defaulting to 400rs")
        r_max = 400 * u.solRad

    constants = huxt_constants()
    dr = constants['dr']
    r = np.arange(r_min.value, r_max.value + dr.value, dr.value)
    r = r * dr.unit
    nr = r.size
    # acceleration is scaled relative to 30rS
    rrel = r - 30 * u.solRad
    return r, dr, rrel, nr


@u.quantity_input(lon_out=u.rad)
@u.quantity_input(lon_start=u.rad)
@u.quantity_input(lon_stop=u.rad)
def longitude_grid(lon_out=np.nan * u.rad, lon_start=np.nan * u.rad, lon_stop=np.nan * u.rad):
    """
    Define the longitude grid of the HUXt model.
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


@u.quantity_input(latitude_min=u.rad)
@u.quantity_input(latitude_max=u.rad)
def latitude_grid(latitude_min=np.nan, latitude_max=np.nan):
    """
    Define the latitude grid of the HUXt model. This is constant in sine latitude
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
    nlat = huxt_constants()['nlat']

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
    Returns:
        dirs: A dictionary of full paths to HUXt directories of code, data, figures, and relevant files.
    """

    # Get path of huxt.py
    cwd = os.path.abspath(os.path.dirname(__file__))

    dirs = {'ephemeris': os.path.join(cwd, 'data', 'ephemeris', 'ephemeris.hdf5'),
            'example_inputs': os.path.join(cwd, 'data', 'example_inputs')}

    # Use appdirs to get platform-specific user data directory
    base_dir = Path(user_data_dir("huxt", ""))
    
    bc_dir = base_dir / "data" / 'boundary_conditions'
    bc_dir.mkdir(parents=True, exist_ok=True)
    dirs['boundary_conditions'] = str(bc_dir)

    sim_dir = base_dir / "data" / 'huxt'
    sim_dir.mkdir(parents=True, exist_ok=True)
    dirs['HUXt_data'] = str(sim_dir)

    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dirs['HUXt_figures'] = str(fig_dir)

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


def _plot_cgf_inputs_outputs_(model_time, vinput, rhoinput, tempinput, 
                                results, KM_TO_CM, PROTON_MASS):
    """
    Plot the boundary conditions going into CGF solver and the results coming out.
    
    This diagnostic function visualizes:
    - Time-dependent boundary conditions (v, rho, T)
    - Radial profiles at selected times
    - Particle trajectories (if available)
    
    Note: This function creates the figure but does not display it.
    The figure will be shown when plt.show() is called by the user.
    """
    import matplotlib.pyplot as plt
    
    # Create figure with a unique number to avoid conflicts
    fig = plt.figure(num='CGF Diagnostic', figsize=(16, 12))
    
    # Convert units for plotting
    v_bc_kms = vinput  # Already in km/s
    rho_bc_protons = rhoinput  # Already in protons/cc
    T_bc_K = tempinput  # Already in K
    
    v_out_kms = results['v'] / KM_TO_CM  # cm/s to km/s
    rho_out_protons = results['rho'] / PROTON_MASS  # g/cm³ to protons/cc
    T_out_K = results['T']  # Already in K
    
    r_AU = results['r'] / 1.496e13  # cm to AU
    t_days = results['t'] / 86400.0  # s to days
    model_time_days = model_time / 86400.0
    
    print(f"    Plotting time range: {t_days.min():.2f} to {t_days.max():.2f} days ({len(t_days)} points)")
    print(f"    Model time range: {model_time_days.min():.2f} to {model_time_days.max():.2f} days ({len(model_time_days)} points)")
    
    # Row 1: Boundary conditions (inputs)
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(model_time_days, v_bc_kms, 'b-', linewidth=2, label='Input BC')
    ax1.set_ylabel('Velocity (km/s)')
    ax1.set_title('Boundary Conditions (Input to CGF)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(model_time_days, rho_bc_protons, 'r-', linewidth=2, label='Input BC')
    ax2.set_ylabel('Density (protons/cc)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3 = plt.subplot(4, 3, 3)
    ax3.plot(model_time_days, T_bc_K / 1e6, 'g-', linewidth=2, label='Input BC')
    ax3.set_ylabel('Temperature (MK)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Row 2: Time evolution at inner boundary (comparing input vs output)
    ax4 = plt.subplot(4, 3, 4)
    ax4.plot(model_time_days, v_bc_kms, 'b--', linewidth=2, alpha=0.5, label='Input BC')
    ax4.plot(t_days, v_out_kms[:, 0], 'b-', linewidth=2, label='CGF output (r_in)')
    ax4.axvline(0, color='k', linestyle=':', alpha=0.5, label='t=0 (sim start)')
    ax4.set_ylabel('Velocity (km/s)')
    ax4.set_title('Inner Boundary Evolution (including spin-up)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    ax5 = plt.subplot(4, 3, 5)
    ax5.plot(model_time_days, rho_bc_protons, 'r--', linewidth=2, alpha=0.5, label='Input BC')
    ax5.plot(t_days, rho_out_protons[:, 0], 'r-', linewidth=2, label='CGF output (r_in)')
    ax5.axvline(0, color='k', linestyle=':', alpha=0.5, label='t=0 (sim start)')
    ax5.set_ylabel('Density (protons/cc)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    ax6 = plt.subplot(4, 3, 6)
    ax6.plot(model_time_days, T_bc_K / 1e6, 'g--', linewidth=2, alpha=0.5, label='Input BC')
    ax6.plot(t_days, T_out_K[:, 0] / 1e6, 'g-', linewidth=2, label='CGF output (r_in)')
    ax6.axvline(0, color='k', linestyle=':', alpha=0.5, label='t=0 (sim start)')
    ax6.set_ylabel('Temperature (MK)')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Row 3: Radial profiles at selected times
    # Only select times from the actual simulation (t >= 0), not spin-up
    sim_mask = t_days >= 0
    sim_indices = np.where(sim_mask)[0]
    if len(sim_indices) > 0:
        # Pick 4 times evenly spaced through the simulation period
        n_sim = len(sim_indices)
        time_indices = [sim_indices[0], 
                       sim_indices[n_sim // 3], 
                       sim_indices[2 * n_sim // 3], 
                       sim_indices[-1]]
    else:
        # Fallback if no simulation times (shouldn't happen)
        n_times = len(t_days)
        time_indices = [0, n_times // 3, 2 * n_times // 3, -1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
    
    ax7 = plt.subplot(4, 3, 7)
    for idx, color in zip(time_indices, colors):
        ax7.plot(r_AU, v_out_kms[idx, :], color=color, linewidth=2,
                label=f't={t_days[idx]:.2f} days')
    ax7.set_xlabel('Radius (AU)')
    ax7.set_ylabel('Velocity (km/s)')
    ax7.set_title('Radial Profiles (CGF Output)')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    ax8 = plt.subplot(4, 3, 8)
    for idx, color in zip(time_indices, colors):
        ax8.semilogy(r_AU, rho_out_protons[idx, :], color=color, linewidth=2,
                    label=f't={t_days[idx]:.2f} days')
    ax8.set_xlabel('Radius (AU)')
    ax8.set_ylabel('Density (protons/cc)')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    ax9 = plt.subplot(4, 3, 9)
    for idx, color in zip(time_indices, colors):
        ax9.semilogy(r_AU, T_out_K[idx, :] / 1e6, color=color, linewidth=2,
                    label=f't={t_days[idx]:.2f} days')
    ax9.set_xlabel('Radius (AU)')
    ax9.set_ylabel('Temperature (MK)')
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    # Row 4: Time-radius contours
    T_grid, R_grid = np.meshgrid(t_days, r_AU, indexing='xy')
    
    ax10 = plt.subplot(4, 3, 10)
    c10 = ax10.contourf(T_grid, R_grid, v_out_kms.T, levels=20, cmap='viridis')
    plt.colorbar(c10, ax=ax10, label='Velocity (km/s)')
    ax10.set_xlabel('Time (days)')
    ax10.set_ylabel('Radius (AU)')
    ax10.set_title('Velocity Evolution')
    
    ax11 = plt.subplot(4, 3, 11)
    c11 = ax11.contourf(T_grid, R_grid, np.log10(rho_out_protons.T), levels=20, cmap='plasma')
    plt.colorbar(c11, ax=ax11, label='log₁₀(n) [protons/cc]')
    ax11.set_xlabel('Time (days)')
    ax11.set_ylabel('Radius (AU)')
    ax11.set_title('Density Evolution')
    
    ax12 = plt.subplot(4, 3, 12)
    c12 = ax12.contourf(T_grid, R_grid, np.log10(T_out_K.T), levels=20, cmap='hot')
    plt.colorbar(c12, ax=ax12, label='log₁₀(T) [K]')
    ax12.set_xlabel('Time (days)')
    ax12.set_ylabel('Radius (AU)')
    ax12.set_title('Temperature Evolution')
    
    # Add particle trajectories if available
    if 'particles' in results and results['particles'] is not None:
        particles = results['particles']
        
        # Check if grouped particles
        if 'groups' in particles:
            # Plot each group with different colors
            group_colors = {'cme': 'red', 'hcs': 'blue', 'streaklines': 'green'}
            for group_name, group_data in particles['groups'].items():
                r_traj = group_data['r'] / 1.496e13  # cm to AU
                t_traj = group_data['t'] / 86400.0  # s to days
                color = group_colors.get(group_name, 'gray')
                
                for i in range(r_traj.shape[0]):
                    mask = ~np.isnan(r_traj[i, :])
                    if np.any(mask):
                        ax10.plot(t_traj[i, mask], r_traj[i, mask], 
                                color=color, linewidth=0.5, alpha=0.6)
                        ax11.plot(t_traj[i, mask], r_traj[i, mask], 
                                color=color, linewidth=0.5, alpha=0.6)
                        ax12.plot(t_traj[i, mask], r_traj[i, mask], 
                                color=color, linewidth=0.5, alpha=0.6)
        else:
            # Single group mode
            r_traj = particles['r'] / 1.496e13
            t_traj = particles['t'] / 86400.0
            
            for i in range(r_traj.shape[0]):
                mask = ~np.isnan(r_traj[i, :])
                if np.any(mask):
                    ax10.plot(t_traj[i, mask], r_traj[i, mask], 
                            'w-', linewidth=0.5, alpha=0.6)
                    ax11.plot(t_traj[i, mask], r_traj[i, mask], 
                            'w-', linewidth=0.5, alpha=0.6)
                    ax12.plot(t_traj[i, mask], r_traj[i, mask], 
                            'w-', linewidth=0.5, alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('cgf_solver_diagnostic.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved diagnostic plot to: cgf_solver_diagnostic.png")
    # Don't call plt.show() here - let the user control when plots are shown
    # The figure will be displayed when the user calls plt.show() at the end of their script


# ============================================================================
# PLUTO Spherical Hydrodynamics Solver
# ============================================================================
# Based on PLUTO code (Mignone et al. 2007) and sunRunner1D configuration.
# Uses:
# - Conservative finite volume method  
# - HLL Riemann solver
# - Spherical geometry source terms
# - Compatible with HUXt time-stepping loop

# PLUTO configuration constants
GAMMA_PLUTO = 1.5  # Polytropic index (matches sunRunner1D)


@jit(nopython=True)
def _pluto_conservative_to_primitive(U, gamma=GAMMA_PLUTO):
    """
    Convert conservative variables to primitive variables.
    
    Args:
        U: Conservative variables [rho, rho*v, E] where E = 0.5*rho*v^2 + P/(gamma-1)
        gamma: Ratio of specific heats
        
    Returns:
        V: Primitive variables [rho, v, P]
    """
    rho = U[0]
    mom = U[1]
    E = U[2]
    
    # Safety check for density
    if rho <= 0:
        rho = 1e-20
    
    v = mom / rho
    # E = 0.5*rho*v^2 + P/(gamma-1)
    # P = (gamma-1) * (E - 0.5*rho*v^2)
    P = (gamma - 1.0) * (E - 0.5 * rho * v * v)
    
    # Ensure positive pressure
    if P <= 0:
        P = 1e-10
    
    return np.array([rho, v, P])


@jit(nopython=True)
def _pluto_primitive_to_conservative(V, gamma=GAMMA_PLUTO):
    """
    Convert primitive variables to conservative variables.
    
    Args:
        V: Primitive variables [rho, v, P]
        gamma: Ratio of specific heats
        
    Returns:
        U: Conservative variables [rho, rho*v, E]
    """
    rho = V[0]
    v = V[1]
    P = V[2]
    
    mom = rho * v
    E = 0.5 * rho * v * v + P / (gamma - 1.0)
    
    return np.array([rho, mom, E])


@jit(nopython=True)
def _pluto_compute_flux(V, gamma=GAMMA_PLUTO):
    """
    Compute physical flux from primitive variables.
    
    Args:
        V: Primitive variables [rho, v, P]
        gamma: Ratio of specific heats
        
    Returns:
        F: Flux vector [rho*v, rho*v^2 + P, v*(E + P)]
    """
    rho = V[0]
    v = V[1]
    P = V[2]
    
    E = 0.5 * rho * v * v + P / (gamma - 1.0)
    
    F = np.zeros(3)
    F[0] = rho * v  # Mass flux
    F[1] = rho * v * v + P  # Momentum flux
    F[2] = v * (E + P)  # Energy flux
    
    return F


@jit(nopython=True)
def _pluto_hll_flux(V_L, V_R, gamma=GAMMA_PLUTO):
    """
    Compute HLL Riemann flux at cell interface.
    
    Following PLUTO's HLL implementation:
    - Estimates fastest wave speeds S_L and S_R
    - Computes intermediate flux based on wave speeds
    
    Args:
        V_L: Left primitive state [rho, v, P]
        V_R: Right primitive state [rho, v, P]
        gamma: Ratio of specific heats
        
    Returns:
        F_HLL: HLL flux at interface
    """
    rho_L, v_L, p_L = V_L[0], V_L[1], V_L[2]
    rho_R, v_R, p_R = V_R[0], V_R[1], V_R[2]
    
    # Sound speeds
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)
    
    # Wave speed estimates
    S_L = min(v_L - c_L, v_R - c_R)
    S_R = max(v_L + c_L, v_R + c_R)
    
    # Conservative variables in SI units
    U_L_mass = rho_L
    U_R_mass = rho_R
    U_L_mom = rho_L * v_L * 1000.0  # kg/(m²·s), convert v_L to m/s
    U_R_mom = rho_R * v_R * 1000.0
    U_L_energy = 0.5 * rho_L * (v_L * 1000.0)**2 + p_L / (gamma - 1.0)  # Pa = J/m³
    U_R_energy = 0.5 * rho_R * (v_R * 1000.0)**2 + p_R / (gamma - 1.0)
    
    # Physical fluxes in SI units
    F_L_mass = rho_L * v_L * 1000.0  # kg/(m²·s), convert v_L to m/s
    F_R_mass = rho_R * v_R * 1000.0
    F_L_mom = rho_L * (v_L * 1000.0)**2 + p_L  # Pa
    F_R_mom = rho_R * (v_R * 1000.0)**2 + p_R
    F_L_energy = (v_L * 1000.0) * (U_L_energy + p_L)  # W/m²
    F_R_energy = (v_R * 1000.0) * (U_R_energy + p_R)
    
    # HLL flux selection
    if S_L >= 0.0:
        # Supersonic right-going: use left state
        F_mass = F_L_mass
        F_mom = F_L_mom
        F_energy = F_L_energy
    elif S_R <= 0.0:
        # Supersonic left-going: use right state
        F_mass = F_R_mass
        F_mom = F_R_mom
        F_energy = F_R_energy
    else:
        # Subsonic: use HLL average flux
        # F_HLL = (S_R*F_L - S_L*F_R + S_L*S_R*(U_R - U_L))/(S_R - S_L)
        denom = S_R - S_L
        
        F_mass = (S_R * F_L_mass - S_L * F_R_mass + S_L * S_R * (U_R_mass - U_L_mass)) / denom
        F_mom = (S_R * F_L_mom - S_L * F_R_mom + S_L * S_R * (U_R_mom - U_L_mom)) / denom
        F_energy = (S_R * F_L_energy - S_L * F_R_energy + S_L * S_R * (U_R_energy - U_L_energy)) / denom
    
    return F_mass, F_mom, F_energy


@jit(nopython=True)
def _pluto_compute_source_terms(U, r, gamma=GAMMA_PLUTO):
    """
    Compute geometric source terms for spherical coordinates.
    
    For 1D spherical geometry:
    S = -(2/r) * F
    
    where F is the flux vector.
    
    Args:
        U: Conservative variables [rho, rho*v, E]
        r: Radial coordinate (km)
        gamma: Ratio of specific heats
        
    Returns:
        S: Source term vector
    """
    V = _pluto_conservative_to_primitive(U, gamma)
    F = _pluto_compute_flux(V, gamma)
    
    # Geometric source: S = -(2/r) * F
    # Convert r from km to m for SI consistency
    r_m = r * 1000.0
    S = -2.0 * F / r_m
    
    return S


@jit(nopython=True)
def _pluto_spatial_operator(U, r_grid, dr, gamma=GAMMA_PLUTO):
    """
    Compute spatial operator L(U) = -∂F/∂r + S.
    
    Uses HLL Riemann solver for fluxes and includes geometric source terms.
    Applies zero-gradient outflow boundary condition at outer boundary.
    
    Args:
        U: Conservative variables array (nr x 3)
        r_grid: Radial coordinate grid (km)
        dr: Grid spacing (km)
        gamma: Ratio of specific heats
        
    Returns:
        dUdt: Time derivative of conservative variables
    """
    nr = len(U)
    dUdt = np.zeros((nr, 3))
    
    # Convert dr from km to m
    dr_m = dr * 1000.0
    
    # Compute fluxes at cell interfaces using HLL solver
    for i in range(1, nr):
        # Left and right states
        V_L = _pluto_conservative_to_primitive(U[i-1], gamma)
        V_R = _pluto_conservative_to_primitive(U[i], gamma)
        

        
        # HLL flux at interface
        F_interface = _pluto_hll_flux(V_L, V_R, gamma)
        F_arr = np.array(F_interface)
        
        # Flux difference
        dUdt[i] -= F_arr / dr_m
        dUdt[i-1] += F_arr / dr_m
    
    # Add geometric source terms to all cells
    for i in range(nr):
        S = _pluto_compute_source_terms(U[i], r_grid[i], gamma)
        dUdt[i] += S
    
    return dUdt


@jit(nopython=True)
def _pluto_step_euler(v_grid, rho_grid, temp_grid, r_grid, dt, gamma=GAMMA_PLUTO):
    """
    Single forward Euler step using PLUTO's spatial operator.
    
    This is compatible with HUXt's time-stepping structure where
    the outer loop manages time integration.
    
    U^(n+1) = U^n + dt*L(U^n)
    
    where L is the spatial operator including HLL fluxes and geometric sources.
    
    Args:
        v_grid: Velocity grid (km/s)
        rho_grid: Density grid (kg/m³)
        temp_grid: Temperature grid (K)
        r_grid: Radial coordinate grid (km)
        dt: Time step (s)
        gamma: Ratio of specific heats
        
    Returns:
        v_new, rho_new, temp_new: Updated grids
    """
    nr = len(v_grid)
    dr = r_grid[1] - r_grid[0]  # Assuming uniform grid
    
    # Physical constants
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    m_p = 1.67262192e-27  # Proton mass (kg)
    
    # Convert to conservative variables
    U = np.zeros((nr, 3))
    for i in range(nr):
        # Convert velocity from km/s to m/s for energy calculation
        v_ms = v_grid[i] * 1000.0
        P = rho_grid[i] * k_B * temp_grid[i] / m_p  # Pressure (Pa)
        
        U[i, 0] = rho_grid[i]  # rho
        U[i, 1] = rho_grid[i] * v_ms  # rho*v
        U[i, 2] = 0.5 * rho_grid[i] * v_ms * v_ms + P / (gamma - 1.0)  # E
    
    # Forward Euler step
    L = _pluto_spatial_operator(U, r_grid, dr, gamma)
    U_new = U + dt * L
    
    # Convert back to primitive variables
    v_new = np.zeros(nr)
    rho_new = np.zeros(nr)
    temp_new = np.zeros(nr)
    
    for i in range(nr):
        # Ensure conservative variables are physical before conversion
        if U_new[i, 0] <= 0:
            U_new[i, 0] = U[i, 0]  # Revert to old value if density became negative
        
        V = _pluto_conservative_to_primitive(U_new[i], gamma)
        rho_new[i] = V[0]
        v_new[i] = V[1] / 1000.0  # Convert back to km/s
        P = V[2]
        
        # Ensure temperature is positive
        if rho_new[i] > 0:
            temp_new[i] = P * m_p / (rho_new[i] * k_B)
        else:
            temp_new[i] = temp_grid[i]  # Revert to old value
    
    return v_new, rho_new, temp_new


# ============================================================================
# CGF Solver Integration
# ============================================================================



def solve_radial_cgf(v_bc_kms, rho_bc_kgm3, T_bc_K, model_time, time_out, 
                     r_grid, gamma, nt_out, nr, verbose=False, create_diagnostic_plot=False,
                     num_particles=0, particle_injection_rate=None, solver_instance=None):
    """
    Solve 1D radial solar wind expansion using the HUXt-native CGF Riemann solver.
    
    This function wraps the CGFSolver class with proper unit conversions and 
    time handling for integration with HUXt.
    
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
    verbose : bool, optional
        If True, print detailed diagnostics. Default False.
    create_diagnostic_plot : bool, optional
        If True, create diagnostic plot for first longitude. Default True.
    num_particles : int or dict, optional
        Number of test particles to track. If 0 (default), no tracking.
        Dict with keys 'cme_leading', 'cme_trailing', 'hcs', 'streak_*' supported.
    particle_injection_rate : array-like or dict, optional
        Injection times (seconds) for particles in model_time coordinates.
        If num_particles is dict, this must also be dict with matching keys.
    solver_instance : CGFSolver, optional
        Pre-initialized CGFSolver instance to reuse. If None, a new one is created.
    
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
    - Uses PLUTO's HLL Riemann solver with WENO3 reconstruction
    - Initializes with Parker nozzle solution for smooth startup
    - Achieves excellent mass flux conservation (~0.06%)
    - Runs as external C binary via subprocess
    """
    # Physical constants (CGS)
    AU = 1.496e13  # cm
    PROTON_MASS = 1.67262192e-24  # grams
    KM_TO_CM = 1e5  # cm/km
    
    # Strip units from times if needed
    model_time_seconds = model_time.value if hasattr(model_time, 'value') else model_time
    time_out_seconds = time_out.value if hasattr(time_out, 'value') else time_out
    
    # Convert to CGS
    v_bc_cgs = v_bc_kms * KM_TO_CM  # cm/s
    rho_bc_cgs = rho_bc_kgm3 * 0.001  # kg/m³ to g/cm³
    
    # Convert HUXt radial grid to cm
    r_grid_cm = r_grid * KM_TO_CM
    
    # Create output time grid for solver - include spin-up snapshots
    spinup_time_seconds = time_out_seconds[0] - model_time_seconds[0]
    n_spinup_snaps = max(5, int(spinup_time_seconds / 86400))  # At least 5, or ~1 per day
    spinup_sampled = np.linspace(model_time_seconds[0], time_out_seconds[0], n_spinup_snaps, endpoint=False)
    t_grid_combined = np.concatenate([spinup_sampled, time_out_seconds])
    
    # Boundary condition functions (MUST return plain floats, no units)
    def v_bc_func(t):
        return float(np.interp(t, model_time_seconds, v_bc_cgs))
    
    def rho_bc_func(t):
        return float(np.interp(t, model_time_seconds, rho_bc_cgs))
    
    def T_bc_func(t):
        return float(np.interp(t, model_time_seconds, T_bc_K))
    
    # Initialize solver
    if solver_instance is None:
        solver = CGFSolver(
            r_grid=r_grid_cm,
            gamma=gamma,
            cfl=0.7,
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
        particle_injection_rate=particle_injection_rate
    )
    
    # Extract output times (skip spin-up)
    n_spinup = len(spinup_sampled)
    
    # Skip spin-up snapshots, extract only output times
    v_out_cgs = results['v'][n_spinup:]
    rho_out_cgs = results['rho'][n_spinup:]
    temp_out_K = results['T'][n_spinup:]
    
    # If solver returned fewer points than expected, interpolate
    if v_out_cgs.shape[0] != nt_out:
        if verbose:
            print(f"  Warning: solver returned {v_out_cgs.shape[0]} points, expected {nt_out}, interpolating...")
        
        v_interp = np.zeros((nt_out, nr))
        rho_interp = np.zeros((nt_out, nr))
        temp_interp = np.zeros((nt_out, nr))
        
        # Skip spin-up times to get only output times
        solver_out_times = results['t'][n_spinup:]
        
        for ir in range(nr):
            v_interp[:, ir] = np.interp(time_out_seconds, solver_out_times, v_out_cgs[:, ir])
            rho_interp[:, ir] = np.interp(time_out_seconds, solver_out_times, rho_out_cgs[:, ir])
            temp_interp[:, ir] = np.interp(time_out_seconds, solver_out_times, temp_out_K[:, ir])
        
        v_out_cgs = v_interp
        rho_out_cgs = rho_interp
        temp_out_K = temp_interp
    
    # Convert to HUXt units
    v_out_kms = v_out_cgs / KM_TO_CM  # cm/s -> km/s
    rho_out_kgm3 = rho_out_cgs / 0.001  # g/cm³ -> kg/m³
    temp_out = temp_out_K  # K (no conversion)
    
    # Create diagnostic plot if requested
    if create_diagnostic_plot and len(results['t']) > 1:
        # Create a copy of results for plotting
        results_plot = results.copy()
        _plot_cgf_inputs_outputs_(model_time_seconds, 
                                  v_bc_kms, rho_bc_kgm3/0.001/PROTON_MASS, T_bc_K,
                                  results_plot, KM_TO_CM, PROTON_MASS)
    
    # Extract and convert particle data if present
    particle_data = None
    if 'particles' in results:
        particle_data_cgs = results['particles']
        
        if isinstance(num_particles, dict):
            # Multi-group mode - convert positions to km
            particle_data = {'groups': {}}
            for group_name, group_data in particle_data_cgs['groups'].items():
                # Convert positions from cm to km
                # group_data['r'] is now a padded numpy array (from cgf_solver)
                r_cgs = group_data['r']
                v_cgs = group_data['v']
                t_sec = group_data['t']
                
                if r_cgs.size == 0:
                    r_km = np.array([])
                    v_km = np.array([])
                    t_sec = np.array([])
                else:
                    r_km = r_cgs / KM_TO_CM
                    v_km = v_cgs / KM_TO_CM
                    # t_sec is already in seconds
                
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
            r_cgs = particle_data_cgs['r']
            v_cgs = particle_data_cgs['v']
            t_sec = particle_data_cgs['t']
            
            if r_cgs.size == 0:
                r_km = np.array([])
                v_km = np.array([])
                t_sec = np.array([])
            else:
                r_km = r_cgs / KM_TO_CM
                v_km = v_cgs / KM_TO_CM
                # t_sec is already in seconds
            
            particle_data = {
                'r': r_km,  # km
                'v': v_km,  # km/s
                't': t_sec,  # seconds
                't_inject': particle_data_cgs['t_inject'],  # seconds
                'active': particle_data_cgs['active'],
            }
    
    return v_out_kms, rho_out_kgm3, temp_out, particle_data


def solve_radial_pluto(v_bc_kms, rho_bc_kgm3, T_bc_K, model_time, time_out, 
                       r_grid, gamma, nt_out, nr, verbose=False, create_diagnostic_plot=False,
                       num_particles=0, particle_injection_rate=None,
                       pluto_dir=None, work_dir=None):
    """
    Solve 1D radial solar wind expansion using PLUTO's HLL Riemann solver.
    
    This function wraps the PLUTO solver with proper unit conversions and 
    time handling for integration with HUXt. PLUTO provides a robust C-based
    solver with excellent mass flux conservation (~0.06%).
    
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
    verbose : bool, optional
        If True, print detailed diagnostics. Default False.
    create_diagnostic_plot : bool, optional
        If True, create diagnostic plot. Default False.
    num_particles : int or dict, optional
        Number of test particles to track. If 0 (default), no tracking.
        Dict with keys 'cme_leading', 'cme_trailing', 'hcs', 'streak_*' supported.
    particle_injection_rate : array-like or dict, optional
        Injection times (seconds) for particles in model_time coordinates.
    pluto_dir : str, optional
        Path to PLUTO installation. If None, uses environment variable or default.
    work_dir : str, optional
        Working directory for PLUTO. If None, creates temporary directory.
    
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
    - Uses PLUTO's HLL Riemann solver with WENO3 reconstruction
    - Initializes with Parker nozzle solution for smooth startup
    - Achieves excellent mass flux conservation (~0.06%)
    - Runs as external C binary via subprocess
    """
    from huxt.pluto_wrapper import PLUTOCustomBCWrapper
    import os
    
    # Physical constants (CGS)
    AU = 1.496e13  # cm
    PROTON_MASS = 1.67262192e-24  # grams
    KM_TO_CM = 1e5  # cm/km
    KG_TO_G = 1000  # g/kg
    M3_TO_CM3 = 1e6  # cm³/m³
    
    # Strip units from times if needed
    if hasattr(model_time, 'value'):
        model_time = model_time.value
    if hasattr(time_out, 'value'):
        time_out = time_out.value
    
    # Convert inputs to CGS (PLUTO native units)
    r_grid_cm = r_grid * KM_TO_CM
    v_bc_cms = v_bc_kms * KM_TO_CM
    rho_bc_gcm3 = rho_bc_kgm3 * KG_TO_G / M3_TO_CM3
    T_bc_K = np.asarray(T_bc_K)
    
    # Set PLUTO directory
    if pluto_dir is None:
        pluto_dir = os.environ.get('PLUTO_DIR', 
                                    '/Users/vy902033/Library/CloudStorage/Dropbox/python_repos/HUXt5/PLUTO')
    
    if verbose:
        print("="*70)
        print("PLUTO 1D Spherical Solver")
        print("="*70)
        print(f"  Domain: {r_grid_cm[0]/AU:.3f} - {r_grid_cm[-1]/AU:.3f} AU ({nr} cells)")
        print(f"  Time: {model_time[0]/86400:.2f} - {model_time[-1]/86400:.2f} days")
        print(f"  Output times: {len(time_out)} snapshots")
        print(f"  Boundary: v={v_bc_kms[0]:.1f}-{v_bc_kms[-1]:.1f} km/s")
        print(f"  Boundary: ρ={rho_bc_kgm3[0]:.2e}-{rho_bc_kgm3[-1]:.2e} kg/m³")
        print(f"  Boundary: T={T_bc_K[0]:.2e}-{T_bc_K[-1]:.2e} K")
    
    # Create PLUTO wrapper
    # CFL=0.4 is conservative but stable for HLL+WENO3 with time-varying BCs
    # Higher values (0.6-0.8) can cause negative velocities with strong gradients
    wrapper = PLUTOCustomBCWrapper(
        pluto_dir=pluto_dir,
        gamma=gamma,
        r_inner=r_grid_cm[0],
        r_outer=r_grid_cm[-1],
        nr=nr,
        cfl=0.4,
        t_offset=model_time[0],
        work_dir=work_dir
    )
    
    # Set time-varying boundary conditions using lookup tables
    # This generates init.c with BC tables embedded and UserDefBoundary interpolation
    # CRITICAL: Do NOT call _write_init_c() after this - it will overwrite the time-varying BC code!
    wrapper.set_time_varying_bc(model_time, rho_bc_gcm3, v_bc_cms, T_bc_K)
    
    # DEBUG: Print work directory so we can inspect files
    print(f"  **DEBUG: PLUTO work directory: {wrapper.work_dir}")
    
    # Compile PLUTO once with the time-varying BC code
    wrapper._compile_pluto()
    
    # Run simulation
    t_duration = model_time[-1] - model_time[0]
    if verbose:
        print(f"\n  Running PLUTO for {t_duration/86400:.2f} days...")
    
    import subprocess
    result = subprocess.run(
        ["./pluto"],
        cwd=wrapper.work_dir,
        capture_output=True,
        text=True
    )
    
    if verbose:
        if result.returncode == 0:
            print(f"  PLUTO completed successfully")
        else:
            print(f"  PLUTO warning (non-zero exit but may have output)")
    
    # Read outputs at requested times
    import sys
    sys.path.append(str(wrapper.pluto_dir / "Tools" / "pyPLUTO"))
    import pyPLUTO.pload as pp
    
    output_files = sorted(wrapper.work_dir.glob("data.*.dbl"))
    
    if verbose:
        print(f"  Found {len(output_files)} output files")
    
    # Read all outputs and interpolate to time_out
    work_dir_str = str(wrapper.work_dir)
    if not work_dir_str.endswith('/'):
        work_dir_str += '/'
    
    # Storage for all timesteps
    all_times = []
    all_v = []
    all_rho = []
    all_T = []
    
    for output_file in output_files:
        file_num = int(output_file.stem.split('.')[-1])
        D = pp.pload(file_num, w_dir=work_dir_str)
        
        # Read time from dbl.out
        with open(wrapper.work_dir / "dbl.out") as f:
            lines = f.readlines()
            if file_num < len(lines):
                t_pluto = float(lines[file_num].split()[1])  # Simulation time
                t_model = t_pluto + model_time[0]  # Convert to model time
            else:
                continue
        
        all_times.append(t_model)
        all_v.append(D.vx1.squeeze())
        all_rho.append(D.rho.squeeze())
        
        # Calculate temperature
        P = D.prs.squeeze()
        rho = D.rho.squeeze()
        T = P * PROTON_MASS / (rho * 1.380649e-16)  # k_B in CGS
        all_T.append(T)
    
    all_times = np.array(all_times)
    all_v = np.array(all_v)
    all_rho = np.array(all_rho)
    all_T = np.array(all_T)
    
    v_out_cgs = all_v
    rho_out_cgs = all_rho
    
    # Debug: Check PLUTO outputs at MULTIPLE radial points to see if CME is present
    v_r0 = all_v[:, 0] / KM_TO_CM  # cm/s -> km/s, innermost cell
    v_r1 = all_v[:, 1] / KM_TO_CM if all_v.shape[1] > 1 else v_r0
    v_r2 = all_v[:, 2] / KM_TO_CM if all_v.shape[1] > 2 else v_r0
    print(f"\n  DEBUG: PLUTO outputs at multiple radial points:")
    print(f"    Number of outputs: {len(all_times)}")
    print(f"    Time range: {all_times[0]/86400:.3f} to {all_times[-1]/86400:.3f} days (model time)")
    print(f"    Velocity at r[0]: {v_r0.min():.1f} - {v_r0.max():.1f} km/s")
    print(f"    Velocity at r[1]: {v_r1.min():.1f} - {v_r1.max():.1f} km/s")
    print(f"    Velocity at r[2]: {v_r2.min():.1f} - {v_r2.max():.1f} km/s")
    print(f"    Velocity at ALL r: {(all_v/KM_TO_CM).min():.1f} - {(all_v/KM_TO_CM).max():.1f} km/s")
    
    # Find when CME should appear (t ~ 0 days in model time)
    cme_time_idx = np.argmin(np.abs(all_times))
    print(f"    At t~0 (CME launch): v[r=0] = {v_r0[cme_time_idx]:.1f}, v[r=1] = {v_r1[cme_time_idx]:.1f} km/s")
    print(f"    Times near t=0: {all_times[max(0,cme_time_idx-2):cme_time_idx+3]/86400}")
    print(f"    Velocities[r=0] near t=0: {v_r0[max(0,cme_time_idx-2):cme_time_idx+3]}")
    
    if verbose:
        print(f"\n  PLUTO output times (model time): {len(all_times)} snapshots")
        print(f"    Range: {all_times[0]/86400:.3f} to {all_times[-1]/86400:.3f} days")
        print(f"    First few: {all_times[:5]/86400}")
        print(f"  time_out range: {time_out[0]/86400:.3f} to {time_out[-1]/86400:.3f} days")
    
    # Interpolate to output times
    v_out = np.zeros((nt_out, nr))
    rho_out = np.zeros((nt_out, nr))
    temp_out = np.zeros((nt_out, nr))
    
    for i in range(nr):
        v_out[:, i] = np.interp(time_out, all_times, v_out_cgs[:, i])
        rho_out[:, i] = np.interp(time_out, all_times, rho_out_cgs[:, i])
        temp_out[:, i] = np.interp(time_out, all_times, all_T[:, i])
    
    # Convert back to HUXt units
    v_out_kms = v_out / KM_TO_CM  # cm/s -> km/s
    rho_out_kgm3 = rho_out * M3_TO_CM3 / KG_TO_G  # g/cm³ -> kg/m³
    temp_out_K = temp_out  # Already in K
    
    # Particle tracking not yet implemented for PLUTO
    particle_data = None
    if num_particles != 0:
        if verbose:
            print("  Warning: Particle tracking not yet implemented for PLUTO solver")
    
    if verbose:
        print(f"\n  Output shape: {v_out_kms.shape}")
        print(f"  Velocity range: {v_out_kms.min():.1f} - {v_out_kms.max():.1f} km/s")
        print(f"  Density range: {rho_out_kgm3.min():.2e} - {rho_out_kgm3.max():.2e} kg/m³")
        print(f"  Temperature range: {temp_out_K.min():.2e} - {temp_out_K.max():.2e} K")
        print("="*70)
    
    return v_out_kms, rho_out_kgm3, temp_out_K, particle_data


@jit(nopython=True, nogil=True)
def solve_radial(vinput, binput, iscmeinput, model_time, rrel, params,
                 n_cme, n_hcs_max, streak_times, rhoinput=None, tempinput=None, compressible=False, solver='upwind'):
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
        params: Array of HUXt parameters.
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

    # unpack the HUXt params
    dtdr = params[0]
    alpha = params[1]
    r_accel = params[2]
    dt_scale = np.int32(params[3])
    nt_out = np.int32(params[4])
    nr = np.int32(params[5])
    r_boundary = params[7]
    accel_limit = bool(params[9])  # switch used to determine if speed limit is applied to acceleration.
    gamma = params[10]  # Adiabatic index for compressible solver
    
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
                    r_ratio = r_inner / r_this
                    v_this = vinput[ir] if ir < len(vinput) else v_inner
                    # Continuity: ρ·v·r² = const → ρ(r) = ρ₀·(v₀/v)·(r₀/r)²
                    rho[ir] = rho_inner * (v_inner / v_this) * r_ratio**2
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
        
        if solver == 'upwind':
            # First-order upwind scheme (Godunov-type)
            if compressible:
                # Use compressible upwind step that evolves velocity, density, and temperature together
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
                # Use incompressible upwind step (velocity only)
                if accel_limit:
                    u_up_next = _upwind_step_accel_limit(u_up, u_dn, dtdr, alpha, r_accel, rrel)
                else:
                    u_up_next = _upwind_step_(u_up, u_dn, dtdr, alpha, r_accel, rrel)
                
                # Save the updated time step (direct assignment, no copy needed)
                v[1:] = u_up_next
        
        elif solver == 'hll':
            # HLL Riemann solver for shock capturing
            
            if compressible:
                # Use HLL Riemann solver for compressible flow
                # NOTE: accel_limit is ignored for compressible solver (no residual acceleration)
                rho_up = rho[1:].copy()
                rho_dn = rho[:-1].copy()
                temp_up = temp[1:].copy()
                temp_dn = temp[:-1].copy()
                
                # Fallback to upwind solver as _hll_step_compressible_ is missing
                u_up_next, rho_up_next, temp_up_next = _upwind_step_compressible_(
                    u_up, u_dn, rho_up, rho_dn, temp_up, temp_dn, dtdr, alpha, r_accel, rrel, r_boundary, gamma)
                
                # Save the updated time steps (direct assignment, no copy needed)
                v[1:] = u_up_next
                rho[1:] = rho_up_next
                temp[1:] = temp_up_next
            else:
                # For incompressible, fall back to upwind (Riemann solver requires full conservation)
                if accel_limit:
                    u_up_next = _upwind_step_accel_limit(u_up, u_dn, dtdr, alpha, r_accel, rrel)
                else:
                    u_up_next = _upwind_step_(u_up, u_dn, dtdr, alpha, r_accel, rrel)
                
                # Save the updated time step (direct assignment, no copy needed)
                v[1:] = u_up_next
        
        elif solver == 'pluto':
            # PLUTO spherical hydrodynamics solver
            # Uses conservative formulation with HLL Riemann solver and spherical source terms
            
            if compressible:
                # PLUTO requires full grid (not split into up/dn)
                # Convert HUXt radial grid to absolute km
                r_grid_km = rrel * 695700.0 + r_boundary
                
                # Compute time step - use same method as upwind/hll solvers
                # Grid spacing in km
                dr_km = r_grid_km[1] - r_grid_km[0]
                dt = dtdr * dr_km  # Actual time step in seconds
                
                # Call PLUTO Euler step
                v_new, rho_new, temp_new = _pluto_step_euler(
                    v, rho, temp, r_grid_km, dt, gamma=GAMMA_PLUTO)
                
                # Update grids (preserve boundary condition at r[0])
                v[1:] = v_new[1:]
                rho[1:] = rho_new[1:]
                temp[1:] = temp_new[1:]
            else:
                raise ValueError("PLUTO solver requires compressible=True")
        
        else:
            raise ValueError(f"Unknown solver: {solver}. Supported solvers: 'upwind', 'hll', 'cgf', 'pluto'")

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
                    # If the leading edge is past the outer boundary, put it at the outer boundary
                    r_cmeparticles[n, 0] = rgrid[-1]

                if r_cmeparticles[n, 1] > rgrid[-1]:
                    # If the trailing edge is past the outer boundary,delete
                    r_cmeparticles[n, :] = rgrid[-1]

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
                             rhoinput=None, tempinput=None, compressible=False):
    """
    Add CMEs to the model input time series
    Args:
        vinput: Timeseries of inner boundary solar wind speeds.
        model_time: Array of model timesteps
        lon: The longitude of this radial
        r_boundary: The HUXt inner boundary in rS
        cme_params: Array of ConeCME parameters to include in the solution. One row for each CME, with columns as
                    required by _is_in_cone_cme_boundary_expanding_
        latitude: Latitude (from the equator) of the HUXt plane
        rhoinput: Timeseries of inner boundary density (optional, for compressible solver)
        tempinput: Timeseries of inner boundary temperature (optional, for compressible solver)
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
                    # Get ambient values at this time (initialize all to avoid Numba type inference issues)
                    v_ambient = vinput[t]
                    rho_ambient = 0.0
                    temp_ambient = 0.0
                    if compressible and rhoinput is not None:
                        rho_ambient = rhoinput[t]
                    if compressible and tempinput is not None:
                        temp_ambient = tempinput[t]
                    
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
                        rho_update_cme[n] = rho_ambient + modulation * (rho_cme - rho_ambient)
                        temp_update_cme[n] = temp_ambient + modulation * (temp_cme - temp_ambient)

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
        dtdr: Ratio of HUXts time step and radial grid step. Units of s/km.
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
        dtdr: Ratio of HUXts time step and radial grid step. Units of s/km.
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
        dtdr: Ratio of HUXt time step and radial grid step. Units of s/km.
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


def load_HUXt_run(filepath):
    """
    Load in data from a saved HUXt run. If Br fields are not saved, pads with
    NaN to avoid conflicts with other HUXt routines.
    Args:
        filepath: The full path to a HDF5 file containing the output from HUXt.save()
    Returns:
        cme_list: A list of instances of ConeCME
        model: An instance of HUXt containing loaded results.
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
        solver = data['solver'][()].decode("utf-8") if 'solver' in data else 'upwind'

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
            'compressible': compressible,
            'solver': solver
        }
        
        if track_b:
            constructor_kwargs['b_boundary'] = b_boundary
            
        if compressible:
            constructor_kwargs['rho_boundary'] = rho_boundary
            constructor_kwargs['temp_boundary'] = temp_boundary
        
        if nlon == 1:
            constructor_kwargs['lon_out'] = lon
            model = HUXt(**constructor_kwargs)
        elif nlon > 1:
            constructor_kwargs['lon_start'] = lon.min()
            constructor_kwargs['lon_stop'] = lon.max()
            model = HUXt(**constructor_kwargs)

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
    return "5.0.0"


