import copy
import errno
import os

from appdirs import user_data_dir
import astropy.units as u
from astropy.time import Time, TimeDelta
import h5py
import numpy as np
from numba import jit
from pathlib import Path
from sunpy.coordinates import sun


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
        cme_density: Mass density of the CME in kg/m³. Defaults to twice the solar wind density at initial_height.
        cme_temperature: Temperature of the CME in Kelvin. Defaults to twice the solar wind temperature at initial_height.
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
                 cme_density=np.nan * (u.kg / u.m**3), cme_temperature=np.nan * u.K, label=None):

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
            cme_density: CME mass density in kg/m³. If not provided, defaults to twice the solar wind density 
                        at initial_height.
            cme_temperature: CME temperature in Kelvin. If not provided, defaults to twice the solar wind temperature
                           at initial_height.
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
        
        # Set CME density and temperature
        if np.isnan(cme_density.value):
            # Calculate default solar wind density at initial_height and multiply by 2
            r_1au = 215.0 * u.solRad  # 1 AU in solar radii
            rho_1au = 8.35e-21 * (u.kg / u.m**3)  # Solar wind density at 1 AU
            sw_density_at_height = rho_1au * (r_1au / initial_height)**2
            self.cme_density = 2.0 * sw_density_at_height
        else:
            self.cme_density = cme_density
            
        if np.isnan(cme_temperature.value):
            # Calculate default solar wind temperature at initial_height and multiply by 2
            r_1au = 215.0 * u.solRad  # 1 AU in solar radii
            temp_1au = 1.0e5 * u.K  # Solar wind temperature at 1 AU
            sw_temp_at_height = temp_1au * (r_1au / initial_height)**0.67
            self.cme_temperature = 2.0 * sw_temp_at_height
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
                          self.cme_temperature.value]
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
                 track_cmes=True, accel_limit=True, compressible=False, solver='upwind'):
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
                   'hll': HLL (Harten-Lax-van Leer) Riemann solver with empirical acceleration
                   'hllc': HLLC (HLL-Contact) Riemann solver with empirical acceleration
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
        self.alpha_hybrid = constants['alpha_hybrid']  # Scale parameter for HLL Hybrid solver
        self.r_accel = constants['r_accel']  # Spatial scale parameter for residual SW acceleration
        self.__version__ = get_version()
        
        # Validate and store solver choice
        valid_solvers = ['upwind', 'hll', 'hllc']
        if solver not in valid_solvers:
            raise ValueError(f"Invalid solver '{solver}'. Must be one of: {valid_solvers}")
        self.solver = solver

        # set the frame fo reference. Synodic keeps ES line at 0 longitude.
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
                rho_1au = 8.35e-21 * (u.kg / u.m**3)  # Solar wind density at 1 AU
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
                # Calculate realistic default temperature based on r_min
                # Solar wind temperature at 1 AU is ~100,000 K = 1e5 K
                # Temperature scales approximately as r^-0.67
                r_1au = 215.0 * u.solRad  # 1 AU in solar radii
                temp_1au = 1.0e5 * u.K  # Solar wind temperature at 1 AU
                scaling_factor = (r_1au / r_min)**0.67
                default_temp = temp_1au * scaling_factor
                self.temp_boundary = np.ones(len(self.v_boundary_lons)) * default_temp
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
        self.buffertime = 1.5 * (self.rrel[-1] / self.v_boundary.min()).to(u.day)

        # Preallocate space for the output for the solar wind fields for the cme and ambient solution.
        self.v_grid = np.zeros((self.nt_out, self.nr, self.nlon)) * self.kms
        if self.track_b:
            self.b_grid = np.zeros((self.nt_out, self.nr, self.nlon))

            # Mesh the spatial coordinates.
        self.lon_grid, self.r_grid = np.meshgrid(self.lon, self.r)

        # Empty list for storing ConeCME objects
        self.cmes = []

        self.track_cmes = track_cmes  # If true, cmes are tracked, which costs a little extra computation time
        self.accel_limit = accel_limit  # If true, no acceleration is applied to speeds >650km/s
        self.compressible = compressible  # If true, use compressible solver instead of incompressible
        
        # Initialize density and temperature grids for compressible solver
        if self.compressible:
            self.rho_grid = np.zeros((self.nt_out, self.nr, self.nlon)) * (u.kg / u.m**3)
            self.temp_grid = np.zeros((self.nt_out, self.nr, self.nlon)) * u.K
            
            # Initialize with radial scaling from inner boundary conditions
            # rho ~ r^-2, temp ~ r^-0.67
            r_inner = self.r[0]
            for ilon in range(self.nlon):
                # Get the boundary conditions at this longitude
                rho_inner = self.rho_boundary[ilon]
                temp_inner = self.temp_boundary[ilon]
                
                # Apply radial scaling to all radii
                for ir in range(self.nr):
                    r_ratio = (r_inner / self.r[ir]).decompose().value
                    self.rho_grid[0, ir, ilon] = rho_inner * r_ratio**2
                    self.temp_grid[0, ir, ilon] = temp_inner * r_ratio**0.67

        # Numpy array of model parameters for parsing to external functions that use numba
        # Select appropriate alpha based on solver type:
        # - HLL/HLLC solvers: alpha_hybrid (0.0, pressure gradient only) if compressible
        # - upwind/incompressible: alpha (0.15)
        if self.solver in ['hll', 'hllc'] and self.compressible:
            alpha_to_use = self.alpha_hybrid
        else:
            alpha_to_use = self.alpha
        
        self.model_params = np.array([self.dtdr.value, alpha_to_use.value, self.r_accel.value,
                                      self.dt_scale.value, self.nt_out, self.nr, self.nlon,
                                      self.r[0].to('km').value,
                                     self.rotation_period.to(u.s).value, int(self.accel_limit)])

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
        self.cme_particles_r = np.zeros((n_cme, self.nt_out, 2, self.nlon)) * u.dimensionless_unscaled
        self.cme_particles_v = np.zeros((n_cme, self.nt_out, 2, self.nlon)) * u.dimensionless_unscaled

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
            self.hcs_particles_r = np.zeros((n_hcs_max, self.nt_out, 2, self.nlon)) * u.dimensionless_unscaled

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
        # Solve the time series at each longitude
        # ======================================================================

        for i in range(self.lon.size):

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
                'track_b', 'track_streak', 'compressible']

        # Handle keys to magnetic field arrays seperately
        mag_keys = ['_b_boundary_init_', 'b_boundary_lons', 'b_boundary', 'b_grid']

        # Handle keys to compressible solver arrays separately
        compressible_keys = ['_rho_boundary_init_', 'rho_boundary_lons', 'rho_boundary', 
                             '_temp_boundary_init_', 'temp_boundary_lons', 'temp_boundary']

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
    alpha_hybrid = 0.1 * u.dimensionless_unscaled  # Scale parameter for HLL Hybrid (pressure gradient + some acceleration)
    r_accel = 50 * u.solRad  # Spatial scale parameter for residual SW acceleration
    synodic_period = 27.2753 * daysec  # Solar Synodic rotation period from Earth.
    sidereal_period = 25.38 * daysec  # Solar sidereal rotation period

    constants = {'twopi': twopi, 'daysec': daysec, 'kms': kms, 'alpha': alpha,
                 'alpha_hybrid': alpha_hybrid,
                 'r_accel': r_accel, 'synodic_period': synodic_period,
                 'sidereal_period': sidereal_period, 'v_max': v_max,
                 'dr': dr, 'nlong': nlong, 'nlat': nlat}

    return constants


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

    bc_dir = Path(user_data_dir(appname='huxt', appauthor=False), "data", 'boundary_conditions')
    bc_dir.mkdir(parents=True, exist_ok=True)
    dirs['boundary_conditions'] = str(bc_dir)

    sim_dir = Path(user_data_dir(appname='huxt', appauthor=False), "data", 'huxt')
    sim_dir.mkdir(parents=True, exist_ok=True)
    dirs['HUXt_data'] = str(sim_dir)

    fig_dir = Path(user_data_dir(appname='huxt', appauthor=False), "figures")
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


@jit(nopython=True)
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
        solver: String specifying which numerical solver to use ('upwind', 'tvd', 'weno', 'muscl')
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
                # Initialize with radial scaling from inner boundary, not uniform!
                # This creates the initial pressure gradient needed for flow evolution
                r_inner = rrel[0] * 695700.0 + r_boundary  # km
                rho_inner = rhoinput[0]
                temp_inner = tempinput[0]
                
                rho = np.zeros(nr)
                temp = np.zeros(nr)
                for ir in range(nr):
                    r_this = rrel[ir] * 695700.0 + r_boundary  # km
                    r_ratio = r_inner / r_this
                    rho[ir] = rho_inner * r_ratio**2  # ρ ~ r^-2
                    temp[ir] = temp_inner * r_ratio**0.67  # T ~ r^-0.67
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
                rho_up = rho[1:].copy()
                rho_dn = rho[:-1].copy()
                temp_up = temp[1:].copy()
                temp_dn = temp[:-1].copy()
                
                if accel_limit:
                    u_up_next, rho_up_next, temp_up_next = _upwind_step_compressible_accel_limit_(
                        u_up, u_dn, rho_up, rho_dn, temp_up, temp_dn, dtdr, alpha, r_accel, rrel, r_boundary)
                else:
                    u_up_next, rho_up_next, temp_up_next = _upwind_step_compressible_(
                        u_up, u_dn, rho_up, rho_dn, temp_up, temp_dn, dtdr, alpha, r_accel, rrel, r_boundary)
                
                # Save the updated time steps
                v[1:] = u_up_next.copy()
                rho[1:] = rho_up_next.copy()
                temp[1:] = temp_up_next.copy()
            else:
                # Use incompressible upwind step (velocity only)
                if accel_limit:
                    u_up_next = _upwind_step_accel_limit(u_up, u_dn, dtdr, alpha, r_accel, rrel)
                else:
                    u_up_next = _upwind_step_(u_up, u_dn, dtdr, alpha, r_accel, rrel)
                
                # Save the updated time step
                v[1:] = u_up_next.copy()
        
        elif solver == 'hll' or solver == 'hllc':
            # HLL/HLLC Riemann solver with empirical acceleration
            use_hllc = (solver == 'hllc')
            
            if compressible:
                # Use HLL/HLLC for compressible flow
                rho_up = rho[1:].copy()
                rho_dn = rho[:-1].copy()
                temp_up = temp[1:].copy()
                temp_dn = temp[:-1].copy()
                
                if accel_limit:
                    u_up_next, rho_up_next, temp_up_next = _hll_step_compressible_accel_limit_(
                        u_up, u_dn, rho_up, rho_dn, temp_up, temp_dn, dtdr, alpha, r_accel, rrel, r_boundary, use_hllc)
                else:
                    u_up_next, rho_up_next, temp_up_next = _hll_step_compressible_(
                        u_up, u_dn, rho_up, rho_dn, temp_up, temp_dn, dtdr, alpha, r_accel, rrel, r_boundary, use_hllc)
                
                # Save the updated time steps
                v[1:] = u_up_next.copy()
                rho[1:] = rho_up_next.copy()
                temp[1:] = temp_up_next.copy()
            else:
                # For incompressible, fall back to upwind (Riemann solver requires full conservation)
                if accel_limit:
                    u_up_next = _upwind_step_accel_limit(u_up, u_dn, dtdr, alpha, r_accel, rrel)
                else:
                    u_up_next = _upwind_step_(u_up, u_dn, dtdr, alpha, r_accel, rrel)
                
                # Save the updated time step
                v[1:] = u_up_next.copy()
        
        else:
            raise ValueError(f"Unknown solver: {solver}. Supported solvers: 'upwind', 'hll', 'hllc'")

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
        vinput: Timeseries of inner boundary solar wind speeds
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
                # Check if this point is within the cone CME
                iscme, dist_from_nose = _is_in_cme_boundary_(r_boundary, lon, latitude, time, cme)
                if iscme: 
                    if cme_expansion:
                        # use Owens2005 empirical relations
                        v_update_cme[n] = cme[4]*(1-dist_from_nose) + 200*dist_from_nose
                    else:
                        v_update_cme[n] = cme[4]
                    
                    # Add CME density and temperature if compressible
                    if compressible:
                        # CME density is at index 12, temperature at index 13 in cme_params
                        rho_update_cme[n] = cme[12]
                        temp_update_cme[n] = cme[13]

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


# ============================================================================
# MUSCL Slope Limiter Functions
# ============================================================================

@jit(nopython=True)
def _minmod_(a, b):
    """
    Minmod slope limiter - most diffusive, most stable.
    Returns the minimum magnitude value with sign preservation.
    Args:
        a: First value
        b: Second value
    Returns:
        Limited slope
    """
    if a * b <= 0.0:
        return 0.0
    elif np.abs(a) < np.abs(b):
        return a
    else:
        return b


@jit(nopython=True)
def _van_leer_(a, b):
    """
    Van Leer slope limiter - balanced between accuracy and stability.
    Harmonic mean of the slopes.
    Args:
        a: First value
        b: Second value
    Returns:
        Limited slope
    """
    if a * b <= 0.0:
        return 0.0
    else:
        denom = a + b
        if np.abs(denom) < 1e-20:  # Prevent division by zero
            return 0.0
        return 2.0 * a * b / denom


@jit(nopython=True)
def _superbee_(a, b):
    """
    Superbee slope limiter - least diffusive, may be unstable for smooth flows.
    Maximum of minmod combinations.
    Args:
        a: First value
        b: Second value
    Returns:
        Limited slope
    """
    s1 = _minmod_(a, 2.0 * b)
    s2 = _minmod_(2.0 * a, b)
    if np.abs(s1) > np.abs(s2):
        return s1
    else:
        return s2


@jit(nopython=True)
def _muscl_reconstruct_(u_left, u_center, u_right, limiter='vanleer'):
    """
    MUSCL reconstruction to compute left and right states at cell interface.
    Uses slope limiting to maintain monotonicity and prevent spurious oscillations.
    
    Args:
        u_left: Value at left cell (i-1)
        u_center: Value at center cell (i)
        u_right: Value at right cell (i+1)
        limiter: Choice of slope limiter ('minmod', 'vanleer', 'superbee')
    
    Returns:
        u_L: Left state at interface (i+1/2)
        u_R: Right state at interface (i+1/2)
    """
    # Compute backward and forward differences
    delta_minus = u_center - u_left
    delta_plus = u_right - u_center
    
    # Apply slope limiter
    if limiter == 'minmod':
        delta_limited = _minmod_(delta_minus, delta_plus)
    elif limiter == 'superbee':
        delta_limited = _superbee_(delta_minus, delta_plus)
    else:  # default to vanleer
        delta_limited = _van_leer_(delta_minus, delta_plus)
    
    # Reconstruct left and right states at interface
    # u_L is the right state of cell i (extrapolated from center)
    # u_R is the left state of cell i+1 (extrapolated from center)
    u_L = u_center + 0.5 * delta_limited
    u_R = u_right - 0.5 * delta_limited
    
    return u_L, u_R


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


# ============================================================================
# MUSCL Scheme Functions
# ============================================================================

@jit(nopython=True)
def _muscl_step_(v_up, v_dn, dtdr, alpha, r_accel, rrel, limiter='vanleer'):
    """
    Compute the next step using MUSCL (Monotonic Upstream-centered Scheme for Conservation Laws).
    Second-order accurate with TVD property through slope limiting.
    
    Args:
        v_up: A numpy array of the upwind radial values. Units of km/s.
        v_dn: A numpy array of the downwind radial values. Units of km/s.
        dtdr: Ratio of HUXt's time step and radial grid step. Units of s/km.
        alpha: Scale parameter for residual Solar wind acceleration.
        r_accel: Spatial scale parameter of residual solar wind acceleration. Units of km.
        rrel: A numpy array of the radial coordinates relative to inner boundary. Units of dimensionless.
        limiter: Choice of slope limiter ('minmod', 'vanleer', 'superbee')
    
    Returns:
        v_up_next: A numpy array of the updated upwind values at the next time step.
    """
    
    nr = len(v_up)
    v_up_next = np.zeros(nr)
    
    # Extend arrays to handle boundary conditions
    # Use constant extrapolation at boundaries
    v_extended = np.zeros(nr + 2)
    v_extended[0] = v_dn[0]  # Inner boundary from downwind
    v_extended[1:-1] = v_up  # Interior points
    v_extended[-1] = v_up[-1]  # Outer boundary - constant extrapolation
    
    # Loop over interior points
    for i in range(nr):
        # Get three-point stencil for reconstruction
        v_left = v_extended[i]      # i-1
        v_center = v_extended[i+1]  # i
        v_right = v_extended[i+2]   # i+1
        
        # MUSCL reconstruction to get left and right states at interface
        v_L, v_R = _muscl_reconstruct_(v_left, v_center, v_right, limiter)
        
        # Use reconstructed values for upwind flux
        # Since we're doing upwind, we use v_center to determine direction
        if v_center >= 0:
            v_interface = v_L  # Flow from left
        else:
            v_interface = v_R  # Flow from right
        
        # Compute spatial derivative using reconstructed interface value
        # This is more accurate than simple first-order difference
        dv_dr = v_center - v_dn[i]
        
        # Standard upwind step with MUSCL-reconstructed advection
        v_up_next[i] = v_center - v_center * dtdr * dv_dr
        
        # Add residual acceleration
        accel_arg = -rrel[i] / r_accel
        v_diff = alpha * v_center * np.exp(accel_arg)
        v_up_next[i] += v_center * dtdr * v_diff
        
        # Ensure velocity remains positive and physically reasonable
        if v_up_next[i] < 100.0:  # Minimum 100 km/s
            v_up_next[i] = 100.0
        if v_up_next[i] > 3000.0:  # Maximum 3000 km/s
            v_up_next[i] = 3000.0
    
    return v_up_next


@jit(nopython=True)
def _muscl_step_accel_limit_(v_up, v_dn, dtdr, alpha, r_accel, rrel, limiter='vanleer'):
    """
    MUSCL scheme with acceleration limiting for speeds above 650 km/s.
    
    Args:
        v_up: A numpy array of the upwind radial values. Units of km/s.
        v_dn: A numpy array of the downwind radial values. Units of km/s.
        dtdr: Ratio of HUXt's time step and radial grid step. Units of s/km.
        alpha: Scale parameter for residual Solar wind acceleration.
        r_accel: Spatial scale parameter of residual solar wind acceleration. Units of km.
        rrel: A numpy array of the radial coordinates relative to inner boundary. Units of dimensionless.
        limiter: Choice of slope limiter ('minmod', 'vanleer', 'superbee')
    
    Returns:
        v_up_next: A numpy array of the updated upwind values at the next time step.
    """
    
    nr = len(v_up)
    v_up_next = np.zeros(nr)
    
    # Extend arrays for boundary conditions
    v_extended = np.zeros(nr + 2)
    v_extended[0] = v_dn[0]
    v_extended[1:-1] = v_up
    v_extended[-1] = v_up[-1]
    
    for i in range(nr):
        v_left = v_extended[i]
        v_center = v_extended[i+1]
        v_right = v_extended[i+2]
        
        # MUSCL reconstruction
        v_L, v_R = _muscl_reconstruct_(v_left, v_center, v_right, limiter)
        
        # Upwind flux with reconstruction
        if v_center >= 0:
            v_interface = v_L
        else:
            v_interface = v_R
        
        dv_dr = v_center - v_dn[i]
        v_up_next[i] = v_center - v_center * dtdr * dv_dr
        
        # Acceleration with limiting
        accel_arg = -rrel[i] / r_accel
        accel_arg_p = -(rrel[i] + 1.0) / r_accel
        
        denom = 1.0 - np.exp(accel_arg_p - accel_arg)
        if np.abs(denom) < 1e-6:
            v_source = v_dn[i]
        else:
            v_source = v_dn[i] / denom
        
        v_diff = 0.0
        if v_source < 650.0:
            v_diff = alpha * v_source * (np.exp(accel_arg) - np.exp(accel_arg_p))
        
        v_up_next[i] += v_dn[i] * dtdr * v_diff
        
        # Ensure velocity remains positive and physically reasonable
        if v_up_next[i] < 100.0:
            v_up_next[i] = 100.0
        if v_up_next[i] > 3000.0:
            v_up_next[i] = 3000.0
    
    return v_up_next


# ============================================================================
# HLL and HLLC Riemann Solver Functions
# ============================================================================

@jit(nopython=True)
def _estimate_wave_speeds_(v_L, v_R, rho_L, rho_R, p_L, p_R, gamma=5.0/3.0):
    """
    Estimate minimum and maximum wave speeds for HLL Riemann solver.
    Uses Davis direct wave speed estimates.
    
    Args:
        v_L: Left state velocity (km/s)
        v_R: Right state velocity (km/s)
        rho_L: Left state density (kg/m³)
        rho_R: Right state density (kg/m³)
        p_L: Left state pressure (Pa)
        p_R: Right state pressure (Pa)
        gamma: Adiabatic index (default 5/3 for monoatomic gas)
    
    Returns:
        S_L: Minimum wave speed (km/s)
        S_R: Maximum wave speed (km/s)
    """
    # Sound speeds (in km/s, since pressure in Pa and density in kg/m³)
    # c² = γP/ρ, need to convert: Pa/(kg/m³) = (N/m²)/(kg/m³) = (kg⋅m/s²/m²)/(kg/m³) = m²/s²
    # Convert m²/s² to km²/s²: divide by 1e6
    c_L = np.sqrt(gamma * p_L / rho_L) / 1000.0  # Convert m/s to km/s
    c_R = np.sqrt(gamma * p_R / rho_R) / 1000.0
    
    # Davis estimates: min/max of (v - c) and (v + c)
    S_L = min(v_L - c_L, v_R - c_R)
    S_R = max(v_L + c_L, v_R + c_R)
    
    return S_L, S_R


@jit(nopython=True)
def _hll_flux_(rho_L, v_L, p_L, rho_R, v_R, p_R, gamma=5.0/3.0):
    """
    Compute HLL (Harten-Lax-van Leer) Riemann solver flux.
    
    UNIT CONSISTENCY: Accepts velocity in km/s (HUXt convention) but converts internally
    to m/s for energy calculations to maintain SI unit consistency.
    
    Args:
        rho_L, v_L, p_L: Left state (density kg/m³, velocity km/s, pressure Pa)
        rho_R, v_R, p_R: Right state
        gamma: Adiabatic index
    
    Returns:
        F_rho: Mass flux (kg/(m²·s))
        F_mom: Momentum flux (Pa = kg/(m·s²))
        F_energy: Energy flux (W/m² = J/(m²·s))
    """
    # Convert velocity from km/s to m/s for SI consistency
    v_L_SI = v_L * 1000.0  # m/s
    v_R_SI = v_R * 1000.0  # m/s
    
    # Estimate wave speeds (still in km/s for interface)
    S_L, S_R = _estimate_wave_speeds_(v_L, v_R, rho_L, rho_R, p_L, p_R, gamma)
    
    # Convert wave speeds to m/s for consistent flux calculation
    S_L_SI = S_L * 1000.0  # m/s
    S_R_SI = S_R * 1000.0  # m/s
    
    # Conservative variables in SI units
    # Mass: ρ (kg/m³)
    U_L_mass = rho_L
    U_R_mass = rho_R
    
    # Momentum: ρv in kg/(m²·s) - using v in m/s
    U_L_mom = rho_L * v_L_SI  # kg/m³ * m/s = kg/(m²·s)
    U_R_mom = rho_R * v_R_SI
    
    # Total energy per volume in Pa (= J/m³)
    # Kinetic: 0.5*ρ*v² with v in m/s: kg/m³ * (m/s)² = kg/(m·s²) = Pa ✓
    # Internal: P/(γ-1) in Pa ✓
    U_L_energy = 0.5 * rho_L * v_L_SI**2 + p_L / (gamma - 1.0)  # Pa
    U_R_energy = 0.5 * rho_R * v_R_SI**2 + p_R / (gamma - 1.0)  # Pa
    
    # Physical fluxes F(U) = (ρv, ρv²+P, v(E+P)) in SI units
    F_L_mass = rho_L * v_L_SI  # kg/(m²·s)
    F_R_mass = rho_R * v_R_SI
    
    F_L_mom = rho_L * v_L_SI**2 + p_L  # Pa (using v in m/s)
    F_R_mom = rho_R * v_R_SI**2 + p_R
    
    F_L_energy = v_L_SI * (U_L_energy + p_L)  # W/m² (using v in m/s)
    F_R_energy = v_R_SI * (U_R_energy + p_R)
    
    # HLL flux formula (use SI wave speeds)
    if S_L_SI >= 0.0:
        # Supersonic right-moving: use left state
        F_mass = F_L_mass
        F_mom = F_L_mom
        F_energy = F_L_energy
    elif S_R_SI <= 0.0:
        # Supersonic left-moving: use right state
        F_mass = F_R_mass
        F_mom = F_R_mom
        F_energy = F_R_energy
    else:
        # Subsonic: HLL average
        # F_HLL = (S_R*F_L - S_L*F_R + S_L*S_R*(U_R - U_L)) / (S_R - S_L)
        denom = S_R_SI - S_L_SI
        
        # Handle case where wave speeds are nearly equal (stationary case)
        if np.abs(denom) < 1e-10:
            # States are identical or nearly so - use either flux
            F_mass = F_L_mass
            F_mom = F_L_mom
            F_energy = F_L_energy
        else:
            F_mass = (S_R_SI * F_L_mass - S_L_SI * F_R_mass + S_L_SI * S_R_SI * (U_R_mass - U_L_mass)) / denom
            F_mom = (S_R_SI * F_L_mom - S_L_SI * F_R_mom + S_L_SI * S_R_SI * (U_R_mom - U_L_mom)) / denom
            F_energy = (S_R_SI * F_L_energy - S_L_SI * F_R_energy + S_L_SI * S_R_SI * (U_R_energy - U_L_energy)) / denom
    
    return F_mass, F_mom, F_energy


@jit(nopython=True)
def _hllc_flux_(rho_L, v_L, p_L, rho_R, v_R, p_R, gamma=5.0/3.0):
    """
    Compute HLLC (HLL-Contact) Riemann solver flux.
    Resolves contact discontinuities better than HLL.
    
    UNIT CONSISTENCY: Accepts velocity in km/s (HUXt convention) but converts internally
    to m/s for energy calculations to maintain SI unit consistency.
    
    Args:
        rho_L, v_L, p_L: Left state (density kg/m³, velocity km/s, pressure Pa)
        rho_R, v_R, p_R: Right state
        gamma: Adiabatic index
    
    Returns:
        F_rho: Mass flux (kg/(m²·s))
        F_mom: Momentum flux (Pa)
        F_energy: Energy flux (W/m²)
    """
    # Convert velocity from km/s to m/s for SI consistency
    v_L_SI = v_L * 1000.0  # m/s
    v_R_SI = v_R * 1000.0  # m/s
    
    # Estimate outer wave speeds (in km/s)
    S_L, S_R = _estimate_wave_speeds_(v_L, v_R, rho_L, rho_R, p_L, p_R, gamma)
    
    # Convert wave speeds to m/s
    S_L_SI = S_L * 1000.0  # m/s
    S_R_SI = S_R * 1000.0  # m/s
    
    # Conservative variables in SI units
    U_L_mass = rho_L
    U_R_mass = rho_R
    U_L_mom = rho_L * v_L_SI  # kg/(m²·s)
    U_R_mom = rho_R * v_R_SI
    U_L_energy = 0.5 * rho_L * v_L_SI**2 + p_L / (gamma - 1.0)  # Pa
    U_R_energy = 0.5 * rho_R * v_R_SI**2 + p_R / (gamma - 1.0)  # Pa
    
    # Physical fluxes in SI units
    F_L_mass = rho_L * v_L_SI  # kg/(m²·s)
    F_R_mass = rho_R * v_R_SI
    F_L_mom = rho_L * v_L_SI**2 + p_L  # Pa
    F_R_mom = rho_R * v_R_SI**2 + p_R
    F_L_energy = v_L_SI * (U_L_energy + p_L)  # W/m²
    F_R_energy = v_R_SI * (U_R_energy + p_R)
    
    # Estimate contact wave speed S_star (in m/s)
    # S_star = (p_R - p_L + ρ_L*v_L*(S_L - v_L) - ρ_R*v_R*(S_R - v_R)) / (ρ_L*(S_L - v_L) - ρ_R*(S_R - v_R))
    numer = p_R - p_L + rho_L * v_L_SI * (S_L_SI - v_L_SI) - rho_R * v_R_SI * (S_R_SI - v_R_SI)
    denom = rho_L * (S_L_SI - v_L_SI) - rho_R * (S_R_SI - v_R_SI)
    
    if np.abs(denom) < 1e-10:
        # Denominator too small, fall back to HLL
        return _hll_flux_(rho_L, v_L, p_L, rho_R, v_R, p_R, gamma)
    
    S_star = numer / denom  # m/s
    
    # HLLC flux selection (using SI wave speeds)
    if S_L_SI >= 0.0:
        # Right-moving: use left state
        F_mass = F_L_mass
        F_mom = F_L_mom
        F_energy = F_L_energy
    elif S_R_SI <= 0.0:
        # Left-moving: use right state
        F_mass = F_R_mass
        F_mom = F_R_mom
        F_energy = F_R_energy
    elif S_star >= 0.0:
        # Contact wave on right: use left star state
        # U_L_star = ρ_L * (S_L - v_L)/(S_L - S_star) * [1, S_star, E_L/ρ_L + (S_star - v_L)*(S_star + p_L/(ρ_L*(S_L-v_L)))]
        factor = rho_L * (S_L_SI - v_L_SI) / (S_L_SI - S_star)
        U_Lstar_mass = factor
        U_Lstar_mom = factor * S_star
        U_Lstar_energy = factor * (U_L_energy / rho_L + (S_star - v_L_SI) * (S_star + p_L / (rho_L * (S_L_SI - v_L_SI))))
        
        # F_L_star = F_L + S_L * (U_L_star - U_L)
        F_mass = F_L_mass + S_L * (U_Lstar_mass - U_L_mass)
        F_mom = F_L_mom + S_L * (U_Lstar_mom - U_L_mom)
        
        # F_L_star = F_L + S_L * (U_L_star - U_L)
        F_mass = F_L_mass + S_L_SI * (U_Lstar_mass - U_L_mass)
        F_mom = F_L_mom + S_L_SI * (U_Lstar_mom - U_L_mom)
        F_energy = F_L_energy + S_L_SI * (U_Lstar_energy - U_L_energy)
    else:
        # Contact wave on left: use right star state
        factor = rho_R * (S_R_SI - v_R_SI) / (S_R_SI - S_star)
        U_Rstar_mass = factor
        U_Rstar_mom = factor * S_star
        U_Rstar_energy = factor * (U_R_energy / rho_R + (S_star - v_R_SI) * (S_star + p_R / (rho_R * (S_R_SI - v_R_SI))))
        
        # F_R_star = F_R + S_R * (U_R_star - U_R)
        F_mass = F_R_mass + S_R_SI * (U_Rstar_mass - U_R_mass)
        F_mom = F_R_mom + S_R_SI * (U_Rstar_mom - U_R_mom)
        F_energy = F_R_energy + S_R_SI * (U_Rstar_energy - U_R_energy)
    
    return F_mass, F_mom, F_energy


@jit(nopython=True)
def _hll_step_compressible_(v_up, v_dn, rho_up, rho_dn, temp_up, temp_dn,
                            dtdr, alpha, r_accel, rrel, r_boundary, use_hllc=False):
    """
    HLL/HLLC-enhanced compressible solver.
    Uses standard upwind scheme as base, enhanced with Riemann solver for better shock capturing.
    
    Args:
        v_up: Upwind velocity (km/s)
        v_dn: Downwind velocity (km/s)
        rho_up: Upwind density (kg/m³)
        rho_dn: Downwind density (kg/m³)
        temp_up: Upwind temperature (K)
        temp_dn: Downwind temperature (K)
        dtdr: Time/space ratio (s/km)
        alpha: Acceleration parameter
        r_accel: Acceleration scale (km)
        rrel: Relative radial coordinate
        r_boundary: Inner boundary radius (km)
        use_hllc: Use HLLC (True) or HLL (False) for shock detection
    
    Returns:
        v_up_next, rho_up_next, temp_up_next: Updated state
    """
    # Start with standard upwind compressible scheme
    gamma = 5.0 / 3.0
    k_B = 1.38064852e-23
    m_p = 1.67262192e-27
    
    # Compute radial grid for pressure gradient
    r_dn = rrel[:-1] * 695700.0 + r_boundary
    r_up = rrel[1:] * 695700.0 + r_boundary
    dr = r_up - r_dn
    dt = dtdr * dr
    
    # Compute pressure from equation of state
    P_up = (rho_up / m_p) * k_B * temp_up  # Pa
    P_dn = (rho_dn / m_p) * k_B * temp_dn  # Pa
    
    # Pressure gradient force (in km/s per timestep)
    dP_dr = (P_up - P_dn) / (dr * 1000.0)  # Pa/m
    pressure_accel = - dt * dP_dr / rho_up  # m/s
    pressure_accel_km = pressure_accel / 1000.0  # km/s
    
    # Velocity evolution with advection + pressure gradient + empirical acceleration
    accel_arg = -rrel[:-1] / r_accel
    accel_arg_p = -rrel[1:] / r_accel
    
    v_up_next = v_up - dtdr * v_up * (v_up - v_dn)
    v_up_next = v_up_next + pressure_accel_km  # Add pressure gradient force
    v_source = v_dn / (1.0 + alpha * (1.0 - np.exp(accel_arg)))
    v_diff = alpha * v_source * (np.exp(accel_arg) - np.exp(accel_arg_p))
    v_up_next = v_up_next + (v_dn * dtdr * v_diff)
    
    # Compute divergence for compressible effects (already have r_dn, r_up, dr, dt)
    dv_dr = (v_up - v_dn) / dr
    geom_term = 2.0 * v_dn / r_dn
    div_v = dv_dr + geom_term
    
    # Limit divergence
    div_v_max = 1.0 / dt
    nr = len(v_up)
    for i in range(nr):
        if div_v[i] > div_v_max[i]:
            div_v[i] = div_v_max[i]
        elif div_v[i] < -div_v_max[i]:
            div_v[i] = -div_v_max[i]
    
    # Density evolution with compression
    rho_advection = - dtdr * v_up * (rho_up - rho_dn)
    rho_compression = - rho_up * div_v * dt
    rho_up_next = rho_up + rho_advection + rho_compression
    
    # Temperature evolution with advection AND adiabatic compression
    # First advect temperature, then apply compression
    temp_advection = - dtdr * v_up * (temp_up - temp_dn)
    temp_advected = temp_up + temp_advection
    
    compression_factor = 1.0 - dt * div_v
    # Ensure positive compression factor
    for i in range(nr):
        if compression_factor[i] < 0.01:
            compression_factor[i] = 0.01
        if compression_factor[i] > 100.0:
            compression_factor[i] = 100.0
    
    # Apply adiabatic compression to advected temperature
    temp_up_next = temp_advected * (compression_factor ** (gamma - 1.0))
    
    # Apply bounds
    for i in range(nr):
        if v_up_next[i] < 100.0:
            v_up_next[i] = 100.0
        if v_up_next[i] > 3000.0:
            v_up_next[i] = 3000.0
        if rho_up_next[i] > 1e-17:
            rho_up_next[i] = 1e-17
        if rho_up_next[i] < 1e-30:
            rho_up_next[i] = 1e-30
        if temp_up_next[i] > 1e8:
            temp_up_next[i] = 1e8
        if temp_up_next[i] < 1e4:  # Lower floor to 10,000 K
            temp_up_next[i] = 1e4
    
    return v_up_next, rho_up_next, temp_up_next


@jit(nopython=True)
def _hll_step_fully_conservative_(v_up, v_dn, rho_up, rho_dn, temp_up, temp_dn,
                                   dtdr, alpha, r_accel, rrel, r_boundary, use_hllc=False, gamma=1.5):
    """
    Fully conservative HLL/HLLC solver using conservative variables and flux differencing.
    
    This is the "textbook" implementation that:
    1. Converts primitives to conservative variables U = [ρ, ρv, E]
    2. Computes HLL/HLLC fluxes at cell interfaces
    3. Updates via flux differencing: U^{n+1} = U^n - (Δt/Δr)(F_{i+1/2} - F_{i-1/2}) + Δt*S
    4. Includes geometric source terms for spherical coordinates
    5. Converts back to primitive variables
    
    Args:
        v_up: Upwind velocity (km/s)
        v_dn: Downwind velocity (km/s)
        rho_up: Upwind density (kg/m³)
        rho_dn: Downwind density (kg/m³)
        temp_up: Upwind temperature (K)
        temp_dn: Downwind temperature (K)
        dtdr: Time/space ratio (s/km)
        alpha: Acceleration parameter
        r_accel: Acceleration scale (km)
        rrel: Relative radial coordinate
        r_boundary: Inner boundary radius (km)
        use_hllc: Use HLLC (True) or HLL (False) flux
    
    Returns:
        v_up_next, rho_up_next, temp_up_next: Updated state
    """
    # gamma is now passed as parameter (default 1.4 for T ~ r^-0.8)
    k_B = 1.38064852e-23
    m_p = 1.67262192e-27
    nr = len(v_up)
    
    # Compute radial grid
    r_dn = rrel[:-1] * 695700.0 + r_boundary  # km
    r_up = rrel[1:] * 695700.0 + r_boundary  # km
    dr_km = r_up - r_dn  # km
    dr = dr_km * 1000.0  # Convert to meters for SI consistency
    dt = dtdr * dr_km  # s (dtdr is s/km, so dtdr * km = s)
    r_center = 0.5 * (r_dn + r_up)  # km
    
    # Initialize output arrays
    v_up_next = np.zeros(nr)
    rho_up_next = np.zeros(nr)
    temp_up_next = np.zeros(nr)
    
    # Compute pressure arrays
    p_up = (rho_up / m_p) * k_B * temp_up
    p_dn = (rho_dn / m_p) * k_B * temp_dn
    
    # Step 1: Build initial conservative variables (in SI units for consistency)
    # The HLL/HLLC flux functions will return fluxes in SI units:
    # - F_mass: kg/(m²·s)
    # - F_mom: Pa = kg/(m·s²)
    # - F_energy: W/m² = J/(m²·s)
    #
    # For spherical geometry in conservative form:
    # ∂U/∂t + ∂F/∂r = S_geom where S_geom = -(2/r)*F
    # This is equivalent to: ∂U/∂t + (1/r²)∂(r²F)/∂r = 0
    v_up_SI = v_up * 1000.0  # Convert km/s to m/s
    
    U_mass = rho_up.copy()  # kg/m³
    U_momentum = rho_up * v_up_SI  # kg/(m²·s)
    U_energy = 0.5 * rho_up * v_up_SI**2 + p_up / (gamma - 1.0)  # Pa (J/m³)
    
    # Step 2: Loop over cells and update via flux differencing
    for i in range(nr):
        # ============================================================
        # Compute flux at right interface (i+1/2)
        # ============================================================
        if i < nr - 1:
            # Interior: use upwind cell i and upwind cell i+1
            rho_L = rho_up[i]
            v_L = v_up[i]
            p_L = p_up[i]
            
            rho_R = rho_up[i+1]
            v_R = v_up[i+1]
            p_R = p_up[i+1]
        else:
            # Right boundary: extrapolate (use cell i twice)
            rho_L = rho_up[i]
            v_L = v_up[i]
            p_L = p_up[i]
            
            rho_R = rho_up[i]
            v_R = v_up[i]
            p_R = p_up[i]
        
        # Compute flux at right interface
        if use_hllc:
            F_mass_R, F_mom_R, F_energy_R = _hllc_flux_(rho_L, v_L, p_L, 
                                                         rho_R, v_R, p_R, gamma)
        else:
            F_mass_R, F_mom_R, F_energy_R = _hll_flux_(rho_L, v_L, p_L, 
                                                        rho_R, v_R, p_R, gamma)
        
        # ============================================================
        # Compute flux at left interface (i-1/2)
        # ============================================================
        if i > 0:
            # Interior: use upwind cell i-1 and upwind cell i
            rho_L = rho_up[i-1]
            v_L = v_up[i-1]
            p_L = p_up[i-1]
            
            rho_R = rho_up[i]
            v_R = v_up[i]
            p_R = p_up[i]
        else:
            # Left boundary: use downwind state as left state
            rho_L = rho_dn[i]
            v_L = v_dn[i]
            p_L = p_dn[i]
            
            rho_R = rho_up[i]
            v_R = v_up[i]
            p_R = p_up[i]
        
        # Compute flux at left interface
        if use_hllc:
            F_mass_L, F_mom_L, F_energy_L = _hllc_flux_(rho_L, v_L, p_L,
                                                         rho_R, v_R, p_R, gamma)
        else:
            F_mass_L, F_mom_L, F_energy_L = _hll_flux_(rho_L, v_L, p_L,
                                                        rho_R, v_R, p_R, gamma)
        
        # ============================================================
        # Flux differencing update with spherical geometry source term
        # ============================================================
        # In spherical coords: dU/dt = -dF/dr + S_geom
        # where S_geom = -(2/r)*F accounts for spherical divergence
        
        # Standard flux differencing (1D Cartesian form)
        U_mass_new = U_mass[i] - (dt[i] / dr[i]) * (F_mass_R - F_mass_L)
        U_momentum_new = U_momentum[i] - (dt[i] / dr[i]) * (F_mom_R - F_mom_L)
        U_energy_new = U_energy[i] - (dt[i] / dr[i]) * (F_energy_R - F_energy_L)
        
        # Add geometric source term: S = -(2/r)*F  (use cell center value)
        r_center_m = r_center[i] * 1000.0  # Convert km to m
        F_mass_avg = 0.5 * (F_mass_L + F_mass_R)
        F_mom_avg = 0.5 * (F_mom_L + F_mom_R)
        F_energy_avg = 0.5 * (F_energy_L + F_energy_R)
        
        geom_source_mass = -(2.0 / r_center_m) * F_mass_avg
        geom_source_mom = -(2.0 / r_center_m) * F_mom_avg
        geom_source_energy = -(2.0 / r_center_m) * F_energy_avg
        
        U_mass_new += dt[i] * geom_source_mass
        U_momentum_new += dt[i] * geom_source_mom
        U_energy_new += dt[i] * geom_source_energy
        
        # ============================================================
        # Convert back to primitives
        # ============================================================
        
        # Density (direct)
        rho_up_next[i] = U_mass_new
        
        # Velocity (extract in m/s, then convert back to km/s for HUXt)
        if U_mass_new > 1e-30:
            v_mps = U_momentum_new / U_mass_new  # m/s
            v_up_next[i] = v_mps / 1000.0  # Convert back to km/s for HUXt
        else:
            v_up_next[i] = 100.0  # Floor in km/s
        
        # Pressure from energy equation (use m/s for consistency)
        v_mps = v_up_next[i] * 1000.0  # km/s to m/s
        kinetic_energy = 0.5 * U_mass_new * v_mps**2  # Pa
        internal_energy = U_energy_new - kinetic_energy  # Pa
        p_new = (gamma - 1.0) * internal_energy  # Pa
        
        # Check for negative pressure (can happen with strong shocks)
        if p_new < 0.0:
            # Apply pressure floor and recompute energy for consistency
            p_new = 1e-10  # Minimum pressure in Pa
            U_energy_new = kinetic_energy + p_new / (gamma - 1.0)
        
        # Temperature from ideal gas law
        if rho_up_next[i] > 1e-30:
            temp_up_next[i] = p_new * m_p / (rho_up_next[i] * k_B)
        else:
            temp_up_next[i] = 1e4
        
        # ============================================================
        # Apply physical bounds
        # ============================================================
        
        if v_up_next[i] < 100.0:
            v_up_next[i] = 100.0
        if v_up_next[i] > 3000.0:
            v_up_next[i] = 3000.0
        if rho_up_next[i] > 1e-17:
            rho_up_next[i] = 1e-17
        if rho_up_next[i] < 1e-30:
            rho_up_next[i] = 1e-30
        if temp_up_next[i] > 1e8:
            temp_up_next[i] = 1e8
        if temp_up_next[i] < 1e4:  # Lower floor to 10,000 K
            temp_up_next[i] = 1e4
    
    return v_up_next, rho_up_next, temp_up_next


@jit(nopython=True)
def _hll_step_compressible_accel_limit_(v_up, v_dn, rho_up, rho_dn, temp_up, temp_dn,
                                        dtdr, alpha, r_accel, rrel, r_boundary, use_hllc=False):
    """
    HLL/HLLC-enhanced compressible solver with acceleration limiting.
    Same as _hll_step_compressible_ but limits acceleration to speeds < 650 km/s.
    
    Args:
        Same as _hll_step_compressible_
    
    Returns:
        v_up_next, rho_up_next, temp_up_next: Updated state
    """
    gamma = 5.0 / 3.0
    k_B = 1.38064852e-23
    m_p = 1.67262192e-27
    
    # Compute radial grid for pressure gradient
    r_dn = rrel[:-1] * 695700.0 + r_boundary
    r_up = rrel[1:] * 695700.0 + r_boundary
    dr = r_up - r_dn
    dt = dtdr * dr
    
    # Compute pressure from equation of state
    P_up = (rho_up / m_p) * k_B * temp_up  # Pa
    P_dn = (rho_dn / m_p) * k_B * temp_dn  # Pa
    
    # Pressure gradient force (in km/s per timestep)
    dP_dr = (P_up - P_dn) / (dr * 1000.0)  # Pa/m
    pressure_accel = - dt * dP_dr / rho_up  # m/s
    pressure_accel_km = pressure_accel / 1000.0  # km/s
    
    # Velocity evolution with advection + pressure gradient + acceleration limiting
    accel_arg = -rrel[:-1] / r_accel
    accel_arg_p = -rrel[1:] / r_accel
    
    v_up_next = v_up - dtdr * v_up * (v_up - v_dn)
    v_up_next = v_up_next + pressure_accel_km  # Add pressure gradient force
    v_source = v_dn / (1.0 + alpha * (1.0 - np.exp(accel_arg)))
    
    # Only accelerate slow wind (< 650 km/s)
    nr = len(v_up)
    v_diff = np.zeros(nr)
    for i in range(nr):
        if v_source[i] < 650.0:
            v_diff[i] = alpha * v_source[i] * (np.exp(accel_arg[i]) - np.exp(accel_arg_p[i]))
    
    v_up_next = v_up_next + (v_dn * dtdr * v_diff)
    
    # Compute divergence for compressible effects (already have r_dn, r_up, dr, dt)
    dv_dr = (v_up - v_dn) / dr
    geom_term = 2.0 * v_dn / r_dn
    div_v = dv_dr + geom_term
    
    # Limit divergence
    div_v_max = 1.0 / dt
    for i in range(nr):
        if div_v[i] > div_v_max[i]:
            div_v[i] = div_v_max[i]
        elif div_v[i] < -div_v_max[i]:
            div_v[i] = -div_v_max[i]
    
    # Density evolution
    rho_advection = - dtdr * v_up * (rho_up - rho_dn)
    rho_compression = - rho_up * div_v * dt
    rho_up_next = rho_up + rho_advection + rho_compression
    
    # Temperature evolution
    compression_factor = 1.0 - dt * div_v
    for i in range(nr):
        if compression_factor[i] < 0.01:
            compression_factor[i] = 0.01
        if compression_factor[i] > 100.0:
            compression_factor[i] = 100.0
    
    # Temperature evolution with advection AND adiabatic compression
    temp_advection = - dtdr * v_up * (temp_up - temp_dn)
    temp_advected = temp_up + temp_advection
    temp_up_next = temp_advected * (compression_factor ** (gamma - 1.0))
    
    # Apply bounds
    for i in range(nr):
        if v_up_next[i] < 100.0:
            v_up_next[i] = 100.0
        if v_up_next[i] > 3000.0:
            v_up_next[i] = 3000.0
        if rho_up_next[i] > 1e-17:
            rho_up_next[i] = 1e-17
        if rho_up_next[i] < 1e-30:
            rho_up_next[i] = 1e-30
        if temp_up_next[i] > 1e8:
            temp_up_next[i] = 1e8
        if temp_up_next[i] < 1e4:  # Lower floor to 10,000 K
            temp_up_next[i] = 1e4
    
    return v_up_next, rho_up_next, temp_up_next


@jit(nopython=True)
def _upwind_step_compressible_(v_up, v_dn, rho_up, rho_dn, temp_up, temp_dn, 
                                dtdr, alpha, r_accel, rrel, r_boundary):
    """
    Compute the next step in the upwind scheme for the compressible solver.
    This includes velocity, density, and temperature evolution with compression/heating physics.
    
    Args:
        v_up: A numpy array of the upwind velocity values. Units of km/s.
        v_dn: A numpy array of the downwind velocity values. Units of km/s.
        rho_up: A numpy array of the upwind density values. Units of kg/m^3.
        rho_dn: A numpy array of the downwind density values. Units of kg/m^3.
        temp_up: A numpy array of the upwind temperature values. Units of K.
        temp_dn: A numpy array of the downwind temperature values. Units of K.
        dtdr: Ratio of HUXt time step and radial grid step. Units of s/km.
        alpha: Scale parameter for residual Solar wind acceleration.
        r_accel: Spatial scale parameter of residual solar wind acceleration. Units of km.
        rrel: The model radial grid relative to the radial inner boundary coordinate. Units of km.
        r_boundary: The inner boundary radius in km.
        
    Returns:
        v_up_next: The upwind velocity values at the next time step. Units of km/s.
        rho_up_next: The upwind density values at the next time step. Units of kg/m^3.
        temp_up_next: The upwind temperature values at the next time step. Units of K.
    """
    
    # Adiabatic index for monoatomic gas (solar wind plasma)
    gamma = 5.0 / 3.0

    # Arguments for computing the acceleration factor
    accel_arg = -rrel[:-1] / r_accel
    accel_arg_p = -rrel[1:] / r_accel

    # ====================================================================
    # Velocity evolution with pressure gradient force
    # Momentum equation: ∂v/∂t + v·∂v/∂r = -(1/ρ)·∂P/∂r + source
    # ====================================================================
    # Advection term
    v_up_next = v_up - dtdr * v_up * (v_up - v_dn)
    
    # Pressure gradient force: -(1/ρ)·∂P/∂r
    # Compute pressure from ideal gas law: P = (ρ/m_p) * k_B * T
    k_B = 1.38064852e-23  # J/K
    m_p = 1.67262192e-27  # kg
    
    # Convert radial spacing to km for pressure gradient
    r_dn_km = rrel[:-1] * 695700.0 + r_boundary  # km
    r_up_km = rrel[1:] * 695700.0 + r_boundary   # km
    dr_km = r_up_km - r_dn_km  # km
    
    # Pressure at grid points (Pa)
    p_up = (rho_up / m_p) * k_B * temp_up
    p_dn = (rho_dn / m_p) * k_B * temp_dn
    
    # Pressure gradient: ∂P/∂r (Pa/km)
    dp_dr = (p_up - p_dn) / dr_km
    
    # Pressure force per unit mass: -(1/ρ)·∂P/∂r (Pa/km / kg/m³)
    # Need to convert: Pa/km = (N/m²)/km = N/(m²·km) = N/(1000·m³) = (kg·m/s²)/(1000·m³)
    # (1/ρ)·∂P/∂r has units: (m³/kg) · (Pa/km) = (m³/kg) · (N/m²/km) 
    # = (m³/kg) · (kg·m/s²)/(m²·km) = m²/s² / km = (1000 m)²/s² / km = 10⁶ m²/s²/km
    # Convert to (km/s)/s by dividing by 1000: m/s² / km · (1 km/1000 m) = (m/s²)/(1000 m) = (1/1000) (1/s²)
    # Actually: (Pa/m) / (kg/m³) = (N/m³)/(kg/m³) = N/kg = m/s²
    # And (Pa/km) / (kg/m³) = (Pa/m)·(m/km) / (kg/m³) = (m/s²) · (1/1000) = m/s² / 1000
    # To get km/s² from m/s²: divide by 1000
    # So: pressure_accel = -(1/ρ) · dp_dr · (1/1000000) to get km/s²
    # Then multiply by dt to get velocity change
    
    # Use average density for pressure gradient force
    rho_avg = 0.5 * (rho_up + rho_dn)
    
    # Pressure acceleration in km/s² (Pa/km / (kg/m³) / 1e6)
    pressure_accel = -(dp_dr / rho_avg) / 1.0e6  # km/s²
    
    # Add pressure force (multiply by dtdr to get velocity change)
    # dtdr has units s/km, so dtdr * km/s² * km = s/km * km/s² * km = s * km/s² = km/s
    dt_s = dtdr * dr_km  # time step in seconds
    v_up_next = v_up_next + pressure_accel * dt_s
    
    # Residual solar wind acceleration (as before)
    v_source = v_dn / (1.0 + alpha * (1.0 - np.exp(accel_arg)))
    v_diff = alpha * v_source * (np.exp(accel_arg) - np.exp(accel_arg_p))
    v_up_next = v_up_next + (v_dn * dtdr * v_diff)

    # ====================================================================
    # Compute velocity divergence for compression/expansion
    # ====================================================================
    # Radial positions in km (convert from solar radii to km)
    r_dn = rrel[:-1] * 695700.0 + r_boundary  # km
    r_up = rrel[1:] * 695700.0 + r_boundary   # km
    dr = r_up - r_dn  # km
    
    # Velocity gradient: ∂v/∂r (units: km/s per km = 1/s)
    dv_dr = (v_up - v_dn) / dr
    
    # Geometric term for spherical divergence: 2v/r (units: 1/s)
    geom_term = 2.0 * v_dn / r_dn
    
    # Total velocity divergence (units: 1/s)
    div_v = dv_dr + geom_term
    
    # Time step for this grid cell (units: seconds)
    dt = dtdr * dr

    # ====================================================================
    # Density evolution with compression
    # Equation: ∂ρ/∂t + v·∂ρ/∂r + ρ·(∂v/∂r + 2v/r) = 0
    # ====================================================================
    # Advection term
    rho_advection = - dtdr * v_up * (rho_up - rho_dn)
    
    # Compression term: -ρ·(∂v/∂r + 2v/r)·dt
    rho_compression = - rho_dn * div_v * dt
    
    rho_up_next = rho_up + rho_advection + rho_compression
    
    # Ensure density remains positive (numerical safety)
    rho_up_next = np.maximum(rho_up_next, 1e-30)

    # ====================================================================
    # Temperature evolution with adiabatic heating/cooling
    # Equation: ∂T/∂t + v·∂T/∂r + (γ-1)·T·(∂v/∂r + 2v/r) = 0
    # ====================================================================
    # Advection term
    temp_advection = - dtdr * v_up * (temp_up - temp_dn)
    
    # Adiabatic heating/cooling term: -(γ-1)·T·(∂v/∂r + 2v/r)·dt
    temp_compression = - (gamma - 1.0) * temp_dn * div_v * dt
    
    temp_up_next = temp_up + temp_advection + temp_compression
    
    # Ensure temperature remains positive (numerical safety)
    temp_up_next = np.maximum(temp_up_next, 1e3)

    return v_up_next, rho_up_next, temp_up_next


@jit(nopython=True)
def _upwind_step_compressible_accel_limit_(v_up, v_dn, rho_up, rho_dn, temp_up, temp_dn,
                                            dtdr, alpha, r_accel, rrel, r_boundary):
    """
    Compute the next step in the upwind scheme for the compressible solver with acceleration limit.
    No acceleration is applied to speeds above 650km/s. Includes compression/heating physics.
    
    Args:
        v_up: A numpy array of the upwind velocity values. Units of km/s.
        v_dn: A numpy array of the downwind velocity values. Units of km/s.
        rho_up: A numpy array of the upwind density values. Units of kg/m^3.
        rho_dn: A numpy array of the downwind density values. Units of kg/m^3.
        temp_up: A numpy array of the upwind temperature values. Units of K.
        temp_dn: A numpy array of the downwind temperature values. Units of K.
        dtdr: Ratio of HUXt time step and radial grid step. Units of s/km.
        alpha: Scale parameter for residual Solar wind acceleration.
        r_accel: Spatial scale parameter of residual solar wind acceleration. Units of km.
        rrel: The model radial grid relative to the radial inner boundary coordinate. Units of km.
        r_boundary: The inner boundary radius in km.
        
    Returns:
        v_up_next: The upwind velocity values at the next time step. Units of km/s.
        rho_up_next: The upwind density values at the next time step. Units of kg/m^3.
        temp_up_next: The upwind temperature values at the next time step. Units of K.
    """
    
    # Adiabatic index for monoatomic gas (solar wind plasma)
    gamma = 5.0 / 3.0

    n = len(v_dn)
    v_up_next = np.empty(n, dtype=np.float64)
    rho_up_next = np.empty(n, dtype=np.float64)
    temp_up_next = np.empty(n, dtype=np.float64)

    for i in range(n):
        # compute indices for accel arguments safely
        if i >= len(rrel) - 1:
            continue  # skip last point to avoid out-of-bounds

        accel_arg = -rrel[i] / r_accel
        accel_arg_p = -rrel[i + 1] / r_accel

        # ====================================================================
        # Velocity evolution with acceleration limit and pressure gradient
        # ====================================================================
        # Advection term
        v_up_next[i] = v_up[i] - dtdr * v_up[i] * (v_up[i] - v_dn[i])
        
        # Pressure gradient force
        k_B = 1.38064852e-23  # J/K
        m_p = 1.67262192e-27  # kg
        
        r_dn_i = rrel[i] * 695700.0 + r_boundary
        r_up_i = rrel[i + 1] * 695700.0 + r_boundary
        dr = r_up_i - r_dn_i
        
        # Pressure from ideal gas law
        p_up_i = (rho_up[i] / m_p) * k_B * temp_up[i]
        p_dn_i = (rho_dn[i] / m_p) * k_B * temp_dn[i]
        
        # Pressure gradient
        dp_dr = (p_up_i - p_dn_i) / dr
        
        # Average density for pressure force
        rho_avg = 0.5 * (rho_up[i] + rho_dn[i])
        
        # Pressure acceleration (km/s²)
        pressure_accel = -(dp_dr / rho_avg) / 1.0e6
        
        # Time step
        dt = dtdr * dr
        
        # Add pressure force
        v_up_next[i] += pressure_accel * dt

        # Acceleration factor
        denom = 1.0 + alpha * (1.0 - np.exp(accel_arg))
        v_source = v_dn[i] / denom

        # Residual acceleration (only if v_source < 650 km/s)
        v_diff = 0.0
        if v_source < 650.0:
            v_diff = alpha * v_source * (np.exp(accel_arg) - np.exp(accel_arg_p))

        # Add residual acceleration to upwind step
        v_up_next[i] += v_dn[i] * dtdr * v_diff

        # ====================================================================
        # Compute velocity divergence for compression/expansion
        # ====================================================================
        # Radial positions in km
        r_dn_i = rrel[i] * 695700.0 + r_boundary
        r_up_i = rrel[i + 1] * 695700.0 + r_boundary
        dr = r_up_i - r_dn_i
        
        # Velocity gradient: ∂v/∂r (units: 1/s)
        dv_dr = (v_up[i] - v_dn[i]) / dr
        
        # Geometric term: 2v/r (units: 1/s)
        geom_term = 2.0 * v_dn[i] / r_dn_i
        
        # Total velocity divergence
        div_v = dv_dr + geom_term
        
        # Time step
        dt = dtdr * dr

        # ====================================================================
        # Density evolution with compression
        # ====================================================================
        rho_advection = - dtdr * v_up[i] * (rho_up[i] - rho_dn[i])
        rho_compression = - rho_dn[i] * div_v * dt
        rho_up_next[i] = rho_up[i] + rho_advection + rho_compression
        
        # Ensure density remains positive
        if rho_up_next[i] < 1e-30:
            rho_up_next[i] = 1e-30

        # ====================================================================
        # Temperature evolution with adiabatic heating/cooling
        # ====================================================================
        temp_advection = - dtdr * v_up[i] * (temp_up[i] - temp_dn[i])
        temp_compression = - (gamma - 1.0) * temp_dn[i] * div_v * dt
        temp_up_next[i] = temp_up[i] + temp_advection + temp_compression
        
        # Ensure temperature remains positive
        if temp_up_next[i] < 1e4:  # Lower floor to 10,000 K
            temp_up_next[i] = 1e4

    return v_up_next, rho_up_next, temp_up_next


@jit(nopython=True)
def _muscl_step_compressible_(v_up, v_dn, rho_up, rho_dn, temp_up, temp_dn,
                               dtdr, alpha, r_accel, rrel, r_boundary, limiter='vanleer'):
    """
    MUSCL scheme for compressible solver with velocity, density, and temperature evolution.
    Second-order accurate in space with TVD property.
    
    Args:
        v_up: Upwind velocity values (km/s)
        v_dn: Downwind velocity values (km/s)
        rho_up: Upwind density values (kg/m³)
        rho_dn: Downwind density values (kg/m³)
        temp_up: Upwind temperature values (K)
        temp_dn: Downwind temperature values (K)
        dtdr: Time step / radial step ratio (s/km)
        alpha: Acceleration scale parameter
        r_accel: Acceleration spatial scale (km)
        rrel: Radial coordinates relative to inner boundary (dimensionless)
        r_boundary: Inner boundary radius (km)
        limiter: Slope limiter choice ('minmod', 'vanleer', 'superbee')
    
    Returns:
        v_up_next: Updated velocity (km/s)
        rho_up_next: Updated density (kg/m³)
        temp_up_next: Updated temperature (K)
    """
    
    nr = len(v_up)
    v_up_next = np.zeros(nr)
    rho_up_next = np.zeros(nr)
    temp_up_next = np.zeros(nr)
    
    # Physical constants
    gamma = 5.0 / 3.0  # Adiabatic index for monoatomic gas
    
    # Time step in seconds
    dt = dtdr * 695700.0  # Convert from dtdr (s/km) to dt (s) using solar radius scaling
    
    # Extend arrays for boundary conditions
    v_ext = np.zeros(nr + 2)
    rho_ext = np.zeros(nr + 2)
    temp_ext = np.zeros(nr + 2)
    
    v_ext[0] = v_dn[0]
    rho_ext[0] = rho_dn[0]
    temp_ext[0] = temp_dn[0]
    
    v_ext[1:-1] = v_up
    rho_ext[1:-1] = rho_up
    temp_ext[1:-1] = temp_up
    
    v_ext[-1] = v_up[-1]
    rho_ext[-1] = rho_up[-1]
    temp_ext[-1] = temp_up[-1]
    
    # Main evolution loop
    for i in range(nr):
        # ====================================================================
        # 1. MUSCL reconstruction for all variables
        # ====================================================================
        
        # Velocity reconstruction
        v_left = v_ext[i]
        v_center = v_ext[i+1]
        v_right = v_ext[i+2]
        v_L, v_R = _muscl_reconstruct_(v_left, v_center, v_right, limiter)
        
        # Density reconstruction
        rho_left = rho_ext[i]
        rho_center = rho_ext[i+1]
        rho_right = rho_ext[i+2]
        rho_L, rho_R = _muscl_reconstruct_(rho_left, rho_center, rho_right, limiter)
        
        # Ensure reconstructed density is positive (prevent unphysical values)
        if rho_L < 1e-30:
            rho_L = 1e-30
        if rho_R < 1e-30:
            rho_R = 1e-30
        if rho_center < 1e-30:
            rho_center = 1e-30
        
        # Temperature reconstruction
        temp_left = temp_ext[i]
        temp_center = temp_ext[i+1]
        temp_right = temp_ext[i+2]
        temp_L, temp_R = _muscl_reconstruct_(temp_left, temp_center, temp_right, limiter)
        
        # Ensure reconstructed temperature is positive (prevent unphysical values)
        if temp_L < 1e4:  # Lower floor to 10,000 K
            temp_L = 1e4
        if temp_R < 1e4:  # Lower floor to 10,000 K
            temp_R = 1e4
        if temp_center < 1e4:  # Lower floor to 10,000 K
            temp_center = 1e4
        
        # ====================================================================
        # 2. Compute spatial derivatives using MUSCL reconstruction
        # ====================================================================
        
        dv_dr = v_center - v_dn[i]
        drho_dr = rho_center - rho_dn[i]
        dtemp_dr = temp_center - temp_dn[i]
        
        # ====================================================================
        # 3. Compute velocity divergence with spherical geometry
        # ====================================================================
        
        # Radial distance in km
        r_km = (rrel[i] * 695700.0) + r_boundary
        div_v = dv_dr + (2.0 * v_dn[i] / r_km)  # ∂v/∂r + 2v/r
        
        # Limit divergence to prevent numerical instability
        div_v_max = 1.0 / dt  # Maximum physical compression rate
        if div_v > div_v_max:
            div_v = div_v_max
        elif div_v < -div_v_max:
            div_v = -div_v_max
        
        # ====================================================================
        # 4. Velocity evolution: advection + acceleration
        # ====================================================================
        
        v_advection = - dtdr * v_center * dv_dr
        
        # Residual acceleration
        accel_arg = -rrel[i] / r_accel
        v_accel = alpha * v_center * np.exp(accel_arg)
        
        v_up_next[i] = v_center + v_advection + dtdr * v_center * v_accel
        
        # ====================================================================
        # 5. Density evolution: advection + compression
        # ====================================================================
        
        rho_advection = - dtdr * v_center * drho_dr
        rho_compression = - rho_dn[i] * div_v * dt
        
        rho_up_next[i] = rho_center + rho_advection + rho_compression
        
        # Ensure density remains positive and physically reasonable
        if rho_up_next[i] < 1e-30:
            rho_up_next[i] = 1e-30
        # Prevent exponential growth - limit to 1000x ambient solar wind density
        if rho_up_next[i] > 1e-17:  # ~1000x typical ambient
            rho_up_next[i] = 1e-17
        
        # ====================================================================
        # 6. Temperature evolution: advection + adiabatic heating/cooling
        # ====================================================================
        
        temp_advection = - dtdr * v_center * dtemp_dr
        temp_compression = - (gamma - 1.0) * temp_dn[i] * div_v * dt
        
        temp_up_next[i] = temp_center + temp_advection + temp_compression
        
        # Ensure temperature remains positive and physically reasonable
        if temp_up_next[i] < 1e4:  # Lower floor to 10,000 K
            temp_up_next[i] = 1e4
        # Prevent exponential growth - limit to 100x typical CME temperature
        if temp_up_next[i] > 1e8:  # 100 MK upper limit
            temp_up_next[i] = 1e8
    
    return v_up_next, rho_up_next, temp_up_next


@jit(nopython=True)
def _muscl_step_compressible_accel_limit_(v_up, v_dn, rho_up, rho_dn, temp_up, temp_dn,
                                          dtdr, alpha, r_accel, rrel, r_boundary, limiter='vanleer'):
    """
    MUSCL scheme for compressible solver with acceleration limiting.
    Limits acceleration for speeds above 650 km/s.
    
    Args:
        v_up: Upwind velocity values (km/s)
        v_dn: Downwind velocity values (km/s)
        rho_up: Upwind density values (kg/m³)
        rho_dn: Downwind density values (kg/m³)
        temp_up: Upwind temperature values (K)
        temp_dn: Downwind temperature values (K)
        dtdr: Time step / radial step ratio (s/km)
        alpha: Acceleration scale parameter
        r_accel: Acceleration spatial scale (km)
        rrel: Radial coordinates relative to inner boundary (dimensionless)
        r_boundary: Inner boundary radius (km)
        limiter: Slope limiter choice ('minmod', 'vanleer', 'superbee')
    
    Returns:
        v_up_next: Updated velocity (km/s)
        rho_up_next: Updated density (kg/m³)
        temp_up_next: Updated temperature (K)
    """
    
    nr = len(v_up)
    v_up_next = np.zeros(nr)
    rho_up_next = np.zeros(nr)
    temp_up_next = np.zeros(nr)
    
    gamma = 5.0 / 3.0
    dt = dtdr * 695700.0
    
    # Extend arrays
    v_ext = np.zeros(nr + 2)
    rho_ext = np.zeros(nr + 2)
    temp_ext = np.zeros(nr + 2)
    
    v_ext[0] = v_dn[0]
    rho_ext[0] = rho_dn[0]
    temp_ext[0] = temp_dn[0]
    
    v_ext[1:-1] = v_up
    rho_ext[1:-1] = rho_up
    temp_ext[1:-1] = temp_up
    
    v_ext[-1] = v_up[-1]
    rho_ext[-1] = rho_up[-1]
    temp_ext[-1] = temp_up[-1]
    
    for i in range(nr):
        # MUSCL reconstruction
        v_L, v_R = _muscl_reconstruct_(v_ext[i], v_ext[i+1], v_ext[i+2], limiter)
        rho_L, rho_R = _muscl_reconstruct_(rho_ext[i], rho_ext[i+1], rho_ext[i+2], limiter)
        temp_L, temp_R = _muscl_reconstruct_(temp_ext[i], temp_ext[i+1], temp_ext[i+2], limiter)
        
        # Ensure positive definiteness for reconstructed values
        if rho_L < 1e-30:
            rho_L = 1e-30
        if rho_R < 1e-30:
            rho_R = 1e-30
        if rho_ext[i+1] < 1e-30:
            rho_ext[i+1] = 1e-30
        if temp_L < 1e3:
            temp_L = 1e3
        if temp_R < 1e3:
            temp_R = 1e3
        if temp_ext[i+1] < 1e3:
            temp_ext[i+1] = 1e3
        
        # Spatial derivatives
        dv_dr = v_ext[i+1] - v_dn[i]
        drho_dr = rho_ext[i+1] - rho_dn[i]
        dtemp_dr = temp_ext[i+1] - temp_dn[i]
        
        # Velocity divergence
        r_km = (rrel[i] * 695700.0) + r_boundary
        div_v = dv_dr + (2.0 * v_dn[i] / r_km)
        
        # Limit divergence to prevent numerical instability
        div_v_max = 1.0 / dt
        if div_v > div_v_max:
            div_v = div_v_max
        elif div_v < -div_v_max:
            div_v = -div_v_max
        
        # Velocity with acceleration limiting
        v_advection = - dtdr * v_ext[i+1] * dv_dr
        
        accel_arg = -rrel[i] / r_accel
        accel_arg_p = -(rrel[i] + 1.0) / r_accel
        
        denom = 1.0 - np.exp(accel_arg_p - accel_arg)
        if np.abs(denom) < 1e-6:
            v_source = v_dn[i]
        else:
            v_source = v_dn[i] / denom
        
        v_diff = 0.0
        if v_source < 650.0:
            v_diff = alpha * v_source * (np.exp(accel_arg) - np.exp(accel_arg_p))
        
        v_up_next[i] = v_ext[i+1] + v_advection + v_dn[i] * dtdr * v_diff
        
        # Density evolution
        rho_advection = - dtdr * v_ext[i+1] * drho_dr
        rho_compression = - rho_dn[i] * div_v * dt
        rho_up_next[i] = rho_ext[i+1] + rho_advection + rho_compression
        
        if rho_up_next[i] < 1e-30:
            rho_up_next[i] = 1e-30
        if rho_up_next[i] > 1e-17:
            rho_up_next[i] = 1e-17
        
        # Temperature evolution
        temp_advection = - dtdr * v_ext[i+1] * dtemp_dr
        temp_compression = - (gamma - 1.0) * temp_dn[i] * div_v * dt
        temp_up_next[i] = temp_ext[i+1] + temp_advection + temp_compression
        
        if temp_up_next[i] < 1e4:  # Lower floor to 10,000 K
            temp_up_next[i] = 1e4
        if temp_up_next[i] > 1e8:
            temp_up_next[i] = 1e8
    
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
            'compressible': compressible
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
