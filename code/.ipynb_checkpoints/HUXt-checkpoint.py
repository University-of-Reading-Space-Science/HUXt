import numpy as np
import astropy.units as u
import os
import glob
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage

mpl.rc("axes", labelsize=20)
mpl.rc("ytick", labelsize=20)
mpl.rc("xtick", labelsize=20)
mpl.rc("legend", fontsize=20)

class ConeCME:
    """
    A class to hold the parameters of a cone model cme.
    """

    def __init__(self, t_launch=0.0, longitude=0.0, v=1000.0, width=30.0, thickness=10.0):
        self.t_launch = t_launch*u.s  # Time of CME launch, after the start of the simulation
        self.longitude = np.deg2rad(longitude) * u.rad  # Longitudinal launch direction of the CME
        self.v = v * u.km / u.s  # CME nose speed
        self.width = np.deg2rad(width) * u.rad  # Angular width
        self.initial_height = (30.0*u.solRad).to(u.km)  # Initial height of CME (should match inner boundary of HUXt)
        self.radius = self.initial_height*np.tan(self.width)  # Initial radius of CME
        self.thickness = (thickness*u.solRad).to(u.km)  # Extra CME thickness

class HUXt2D:

    def __init__(self, simtime=5.0, dt_scale=1.0):

        # some constants and units
        constants = huxt_constants()
        self.twopi = constants['twopi']
        self.daysec = constants['daysec']
        self.kms = constants['kms']
        self.alpha = constants['alpha']  # Scale parameter for residual SW acceleration
        self.r_accel = constants['r_accel']  # Spatial scale parameter for residual SW acceleration
        self.synodic_period = constants['synodic_period']  # Solar Synodic rotation period from Earth.
        self.v_max = constants['v_max']
        del constants
        
        # Setup radial coordinates - in solar radius
        self.r, self.dr, self.rrel, self.Nr = radial_grid()
        
        # Setup longitude coordinates - in radians.
        self.lon, self.dlon, self.Nlon = longitude_grid()
        
        self.simtime = (simtime * u.day).to(u.s)  # number of days to simulate (in seconds)
        self.dt_scale = dt_scale * u.dimensionless_unscaled
        time_grid_dict = time_grid(self.v_max, self.dr, self.simtime, self.dt_scale)
        self.dtdr = time_grid_dict['dtdr']
        self.Nt = time_grid_dict['Nt']
        self.dt = time_grid_dict['dt']
        self.time = time_grid_dict['time']
        self.Nt_out = time_grid_dict['Nt_out']
        self.dt_out = time_grid_dict['dt_out']
        self.time_out = time_grid_dict['time_out']
        del time_grid_dict
        
        # Preallocate space for the output for the solar wind fields for the cme and ambient solution.
        self.v_grid_cme = np.zeros((self.Nt_out, self.Nr, self.Nlon)) * self.kms
        self.v_grid_amb = np.zeros((self.Nt_out, self.Nr, self.Nlon)) * self.kms

        # Mesh the spatial coordinates.
        self.lon_grid, self.r_grid = np.meshgrid(self.lon, self.r)

        # Extract paths of figure and data directories
        dirs = _setup_dirs_()
        self._boundary_dir_ = dirs['boundary_conditions']
        self._data_dir_ = dirs['HUXt2D_data']
        self._figure_dir_ = dirs['HUXt2D_figures']
        return

    def solve(self, v_boundary, cme_list, save=False, tag=''):
        """
        Solve HUXt2D for the specified inner boundary condition and list of cone cmes.
        Results are stored in the v_grid_cme and v_grid_amb attributes. 
        Save output to a HDF5 file if requested.
        """
        
        # Check only cone cmes in cme list
        cme_list_checked = []
        for cme in cme_list:
            if isinstance(cme, ConeCME):
                cme_list_checked.append(cme)
            else:
                print("Warning: cme_list contained objects other than ConeCME instances. These will be excluded")
                
        # Initialise v from the steady-state solution - no spin up required
        # ----------------------------------------------------------------------------------------
        # compute the steady-state solution, as function of time, convert to function of long
        v_cr = self._solve_carrington_rotation_(v_boundary)
        v_cr = np.fliplr(v_cr)

        # Initialise the output
        self.v_grid_cme[0, :, :] = v_cr.copy()
        self.v_grid_amb[0, :, :] = v_cr.copy()

        # compute longitude increment that matches the timestep dt
        dlondt = self.twopi * self.dt / self.synodic_period
        lon_tstep = np.arange(self.lon.value.min(), self.lon.value.max() + dlondt, dlondt) * u.rad
        # interpolate vin_long to this timestep matched resolution
        v_boundary_tstep = np.interp(lon_tstep.value, self.lon.value, v_boundary, period=self.twopi)

        # ----------------------------------------------------------------------------------------
        # Main model loop
        # ----------------------------------------------------------------------------------------
        for t in range(self.Nt):

            # Get the initial condition, which will update in the loop,
            # and snapshots saved to output at right steps.
            if t == 0:
                v_cme = self.v_grid_cme[t, :, :].copy()
                v_amb = self.v_grid_cme[t, :, :].copy()

            # loop through each longitude and compute the the 1-d radial solution
            for n in range(self.Nlon):
                # update cone cme v(r) for the given longitude
                # =====================================
                u_up = v_cme[1:, n].copy()
                u_dn = v_cme[:-1, n].copy()
                u_up_next = _upwind_step_(self, u_up, u_dn)
                # Save the updated timestep
                v_cme[1:, n] = u_up_next.copy()

                u_up = v_amb[1:, n].copy()
                u_dn = v_amb[:-1, n].copy()
                u_up_next = _upwind_step_(self, u_up, u_dn)
                # Save the updated timestep
                v_amb[1:, n] = u_up_next.copy()

            # Save this frame to output if output timestep is a factor of time elapsed 
            if np.mod(t, self.dt_scale) == 0:
                t_out = np.int32(t / self.dt_scale)  # index of timestep in output array
                if t_out < self.Nt_out - 1:  # Model can run one step longer than output steps, so check:
                    self.v_grid_cme[t_out, :, :] = v_cme.copy()
                    self.v_grid_amb[t_out, :, :] = v_amb.copy()

            # Update the boundary conditions for next timestep.
            if t < self.Nt - 1:
                #  Update ambient solution inner boundary
                # ==================================================================
                v_boundary_tstep = np.roll(v_boundary_tstep, 1)
                v_boundary_update = np.interp(self.lon.value, lon_tstep.value, v_boundary_tstep)
                v_amb[0, :] = v_boundary_update * self.kms

                #  Add cone CME to updated inner boundary
                # ==================================================================
                v_boundary_cone = v_boundary_tstep.copy()
                for cme in cme_list_checked:
                    r_boundary = self.r.min().to(u.km)
                    v_boundary_cone = _cone_cme_boundary_update_(r_boundary, lon_tstep, v_boundary_cone, self.time[t], cme)
                    
                v_boundary_update = np.interp(self.lon.value, lon_tstep.value, v_boundary_cone)
                v_cme[0, :] = v_boundary_update * self.kms

        if save:
            if tag == '':
                print("Warning, blank tag means file likely to be overwritten")
            self.save(v_boundary, cme_list_checked, tag=tag)
        return

    def save(self, v_boundary, cme_list, tag):
        """
        Function to save output to a HDF5 file. 
        """
        # Open up hdf5 data file for the HI flow stats
        filename = "HUXt2D_{}.hdf5".format(tag)
        out_filepath = os.path.join(self._data_dir_, filename)

        if os.path.isfile(out_filepath):
            # File exists, so delete and start new.
            print("Warning: {} already exists. Overwriting".format(out_filepath))
            os.remove(out_filepath)

        out_file = h5py.File(out_filepath, 'w')

        # Save the input boundary condition
        dset = out_file.create_dataset("v_boundary", data=v_boundary.value)
        dset.attrs['unit'] = v_boundary.unit.to_string()
        out_file.flush()

        # Save the Cone CME parameters to a new group.
        allcmes = out_file.create_group('ConeCMEs')
        for i, cme in enumerate(cme_list):
            cme_name = "ConeCME_{:02d}".format(i)
            cmegrp = allcmes.create_group(cme_name)
            for k, v in cme.__dict__.items():
                dset = cmegrp.create_dataset(k, data=v.value)
                dset.attrs['unit'] = v.unit.to_string()
                out_file.flush()

        # Loop over the attributes of model instance and save select keys/attributes.
        keys = ['simtime', 'dt_scale', 'time_out', 'dt_out', 'r', 'dr', 'lon', 'dlon', 'r_grid', 'lon_grid',
                'v_grid_cme', 'v_grid_amb']
        for k, v in self.__dict__.items():

            if k in keys:

                if k in ['time_out', 'dt_out']:
                    kn = k.split('_')[0]  # loose the "_out"
                    dset = out_file.create_dataset(kn, data=v.value)
                    dset.attrs['unit'] = v.unit.to_string()
                else:
                    dset = out_file.create_dataset(k, data=v.value)
                    dset.attrs['unit'] = v.unit.to_string()

                # Add on the dimensions of the spatial grids
                if k in ['r_grid', 'lon_grid']:
                    dset.dims[0].label = 'radius'
                    dset.dims[1].label = 'longtiude'

                # Add on the dimensions of the output speed fields.
                if k in ['v_grid_cme', 'v_grid_amb']:
                    dset.dims[0].label = 'time'
                    dset.dims[1].label = 'radius'
                    dset.dims[2].label = 'longtiude'

                out_file.flush()

        out_file.close()
        return

    def plot(self, t, field='cme', save=False, tag=''):
        """
        Make a contour plot on polar axis of the solar wind solution at a specific time
        :param t: Integer to index the time coordinate.
        :param field: String, either 'cme', or 'ambient', specifying which solution to plot.
        :param save: Boolean to determine if the figure is saved.
        :param tag: String to append to the filename if saving the figure.
        :return:
        """
        
        if field not in ['cme', 'ambient']:
            print("Error, field must be either 'cme', or 'ambient'. Default to CME")
            field = 'cme'
            
        if (t<0) | (t > (self.Nt_out-1)):
            print("Error, invalid time index t")
            
        # Get plotting data
        lon = self.lon_grid.value.copy()
        rad = self.r_grid.value.copy()
        if field == 'cme':
            v = self.v_grid_cme.value[t, :, :].copy()
        elif field == 'ambient':
            v = self.v_grid_amb.value[t, :, :].copy()
        
        # Pad out to fill the full 2pi of contouring
        pad = lon[:, 0].reshape((lon.shape[0], 1)) + self.twopi
        lon = np.concatenate((lon, pad), axis=1)
        pad = rad[:, 0].reshape((rad.shape[0], 1))
        rad = np.concatenate((rad, pad), axis=1)
        pad = v[:, 0].reshape((v.shape[0], 1))
        v = np.concatenate((v, pad), axis=1)

        levels = np.arange(350, 950, 25)
        fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={"projection": "polar"})
        cnt = ax.contourf(lon, rad, v, levels=levels)
        ax.set_ylim(0, 230)
        ax.set_yticklabels([])
        ax.tick_params(axis='x', which='both', pad=15)
        fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.95)

        # Add colorbar
        pos = ax.get_position()
        dw = 0.005
        dh = 0.075
        left = pos.x0 + dw
        bottom = pos.y0 - dh
        wid = pos.width - 2*dw
        cbaxes = fig.add_axes([left, bottom, wid, 0.03])
        cbar1 = fig.colorbar(cnt, cax=cbaxes, orientation='horizontal')
        cbar1.ax.set_xlabel("Solar Wind Speed (km/s)")

        # Add label
        time_label = "Time: {:3.2f} days".format(self.time_out[t].to(u.day).value)
        ax.set_title(time_label, position=(0.8, -0.05), fontsize=20)

        if save:
            filename = "HUXt2D_{}_frame_{:03d}.png".format(tag, t)
            filepath = os.path.join(self._figure_dir_, filename)
            fig.savefig(filepath)

        return fig, ax

    def animate(self, field, tag):
        """
        Animate the solution, and save as MP4.
        :param field: String, either 'cme', or 'ambient', specifying which solution to animate.
        :param tag: String to append to the filename of the animation.
        :return:
        """
        if field not in ['cme', 'ambient']:
            print("Error, field must be either 'cme', or 'ambient'. Default to CME")
            field = 'cme'
        
        # Set the duration of the movie
        # Scaled so a 5 day simultion with dt_scale=4 is a 10 second movie.
        duration = self.simtime.value * (10 / 432000)

        def make_frame(t):
            """
            Function to produce the frame required by MoviePy.VideoClip.
            """
            # Get the time index closest to this fraction of movie duration
            i = np.int32((self.Nt_out-1) * t / duration)
            fig, ax = self.plot(i, field)
            frame = mplfig_to_npimage(fig)
            plt.close('all')           
            return frame

        filename = "HUXt2D_{}_movie.mp4".format(tag)
        filepath = os.path.join(self._figure_dir_, filename)
        animation = mpy.VideoClip(make_frame, duration=duration)
        animation.write_videofile(filepath, fps=24, codec='libx264')
        return
    
    def _solve_carrington_rotation_(self, v_boundary):
        """
        """

        if v_boundary.size != 128:
            print('Warning HUXt2D.solve_carrington_rotation: v_boundary not expected size of 128.')

        simtime = self.synodic_period  # One CR from Earth.
        buffertime = (5.0 * u.day).to(u.s)  # spin up time
        tmax = (simtime + buffertime)  # full simulation time

        # compute the longitude increment corresponding to timestep dt
        dlondt = self.twopi * self.dt / self.synodic_period
        # work out the phi increment to allow for the spin up
        bufferlon = self.twopi * buffertime / self.synodic_period
        # create the input timeseries including the spin up series, periodic in phi
        lonint = np.arange(0, self.twopi + bufferlon + dlondt, dlondt)
        loninit = _zerototwopi_(lonint)
        vinit = np.interp(loninit, self.lon.value, v_boundary.value, period=self.twopi) * self.kms
        # convert from longitude to time
        vinput = np.flipud(vinit)
        times = np.arange(0.0, (tmax + self.dt).value, self.dt.value)
        times = times * u.s

        # compute the rout time series
        # ==========================
        ts_allR = solve_upwind(self, vinput)

        # remove the spin up.
        id_after_spinup = times >= buffertime
        ts_allR = ts_allR[:, id_after_spinup]
        times = times[id_after_spinup]
        times = times - buffertime

        # interpolate back to the original longitudinal grid
        times_orig = self.synodic_period * self.lon.value / self.twopi
        vout_allR = np.zeros((self.r.size, times_orig.size)) * ts_allR.unit
        for j in range(self.r.size):
            vout = np.interp(times_orig.value, times.value, ts_allR[j, :].value)
            vout_allR[j, :] = vout * self.kms

        return vout_allR

    
def huxt_constants():
    """
    Return some constants used in all HUXt model classes
    """
    twopi = 2.0*np.pi
    daysec = 24*60*60
    kms = u.km / u.s
    alpha = 0.15  # Scale parameter for residual SW acceleration
    r_accel = 50*u.solRad  # Spatial scale parameter for residual SW acceleration
    synodic_period = 27.2753*daysec*u.s  # Solar Synodic rotation period from Earth.
    v_max = 2000*kms
    constants = {'twopi': twopi, 'daysec': daysec, 'kms': kms, 'alpha': alpha,
                 'r_accel': r_accel, 'synodic_period': synodic_period, 'v_max': v_max}
    return constants
    
def radial_grid():
    """
    Define the radial grid of the HUXt model (1D and 2D)
    """
    r_min = 30.0
    r_max = 240.0
    Nr = 140
    r, dr = np.linspace(r_min, r_max, Nr, retstep=True)
    r = r*u.solRad
    dr = dr*u.solRad
    rrel = r - r[0]
    return r, dr, rrel, Nr

def longitude_grid():
    """
    Define the longitude grid of the HUXt2D model.
    """
    Nlon = 128
    twopi = 2.0*np.pi
    dlon = twopi / Nlon
    lon_min = dlon / 2.0
    lon_max = twopi - (dlon / 2.0)
    lon, dlon = np.linspace(lon_min, lon_max, Nlon, retstep=True)
    lon = lon*u.rad
    dlon = dlon*u.rad
    return lon, dlon, Nlon

def time_grid(v_max, dr, simtime, dt_scale):
    """
    Define the model timestep and time grid based on CFL condition and specified simulation time.
    """
    dt = dr.to(u.km) / v_max
    dtdr = dt / dr.to(u.km)

    Nt = np.int32(np.floor(simtime.value / dt.value))  # number of time steps in the simulation
    time = np.arange(0, Nt)*dt  # Model timesteps
    
    dt_out = dt_scale*dt  # time step of the output
    Nt_out = np.int32(Nt / dt_scale.value)  # number of time steps in the output   
    time_out = np.arange(0, Nt_out)*dt_out  # Output timesteps
    
    time_grid_dict = {'dt': dt, 'dtdr': dtdr, 'Nt': Nt, 'time': time,
                 'dt_out': dt_out, 'Nt_out': Nt_out, 'time_out': time_out}
    return time_grid_dict

def load_HUXt2D_run(filepath):
    """
    Load in data from a previous run.
    :param filepath: The full path to a HDF5 file containing the output from HUXt2D.save()
    :return: v_boundary: Numpy array of the solar wind boundary condition. Unit km/s
    :return: cme_list: A list of instances of ConeCME
    :return: model: An instance of HUXt2DCME containing loaded results.
    """
    if os.path.isfile(filepath):

        data = h5py.File(filepath, 'r')

        # Load in the inner boundary wind speed profile
        v_boundary = data['v_boundary'][()] * u.Unit(data['v_boundary'].attrs['unit'])

        # Load in the CME paramters
        cme_list = []

        all_cmes = data['ConeCMEs']
        for k in all_cmes.keys():
            cme_data = all_cmes[k]
            t_launch = cme_data['t_launch'][()]
            lon = np.rad2deg(cme_data['longitude'][()])
            width = np.rad2deg(cme_data['width'][()])
            thickness = cme_data['thickness'][()] * u.Unit(cme_data['thickness'].attrs['unit'])
            thickness = thickness.to('solRad').value
            v = cme_data['v'][()]
            cme = ConeCME(t_launch=t_launch, longitude=lon, v=v, width=width, thickness=thickness)
            cme_list.append(cme)

        # Initialise the model, and check it matches what the resolution and limits of the HDF5 file.
        simtime = data['simtime'][()] * u.Unit(data['simtime'].attrs['unit'])
        simtime = simtime.to(u.day).value
        dt_scale = data['dt_scale'][()]
        model = HUXt2DCME(simtime=simtime, dt_scale=dt_scale)
        model.v_grid_cme[:, :, :] = data['v_grid_cme'][()] * u.Unit(data['v_boundary'].attrs['unit'])
        model.v_grid_amb[:, :, :] = data['v_grid_cme'][()] * u.Unit(data['v_boundary'].attrs['unit'])

    else:
        # File doesnt exist return nothing
        print("Warning: {} doesnt exist.".format(filepath))
        v_boundary = []
        cme_list = []
        model = []

    return v_boundary, cme_list, model

def _cone_cme_boundary_update_(r_boundary, longitude, v_boundary, t, cme):
    """
    Function to update inner speed boundary condition with the time dependent cone cme speed
    """
    # Center the longitude array on CME nose, running from -pi to pi, to avoid dealing with any 0/2pi crossings
    lon_cent = longitude - cme.longitude
    id_high = lon_cent > np.pi*u.rad
    lon_cent[id_high] = 2.0*np.pi*u.rad - lon_cent[id_high]
    id_low = lon_cent < -np.pi*u.rad
    lon_cent[id_low] = lon_cent[id_low] + 2.0*np.pi*u.rad 
    
    #  Compute y, the height of CME nose above the 30rS surface
    y = cme.v*(t - cme.t_launch)
    if (y >= 0*u.km) & (y < cme.radius):  # this is the front hemisphere of the spherical CME
        x = np.sqrt(y*(2*cme.radius - y))  # compute x, the distance of the current longitude from the nose
        # convert x back to an angular separation
        theta = np.arctan(x / r_boundary)
        pos = (lon_cent > - theta) & (lon_cent <= theta)
        v_boundary[pos] = cme.v.value
    elif (y >= (cme.radius + cme.thickness)) & (
        y <= (2*cme.radius + cme.thickness)):  # this is the back hemisphere of the spherical CME
        y = y - cme.thickness
        x = np.sqrt(y*(2*cme.radius - y))
        # convert back to an angle
        theta = np.arctan(x / r_boundary)
        pos = (lon_cent > - theta) & (lon_cent <= theta)
        v_boundary[pos] = cme.v.value
    elif (cme.thickness > 0*u.km) & (y >= cme.radius) & (
        y <= (cme.radius + cme.thickness)):  # this is the "mass" between the hemispheres
        x = cme.radius
        # convert back to an angle
        theta = np.arctan(x / r_boundary)
        pos = (lon_cent > - theta) & (lon_cent <= theta)
        v_boundary[pos] = cme.v.value

    return v_boundary

def solve_upwind(model, v_boundary):
        """
        Functon to solve the upwind scheme for Burgers equation for the time evolution
        of the radial wind speed, with additional solar wind acceleration
        """
        # TODO - check input of v_boundary on size and unit.        
        Nt = v_boundary.size  # number of time steps
        # Initialise output speeds as 400kms everywhere
        v_out = np.ones((model.Nr, Nt)) * 400.0 * model.kms
        # Update inner boundary condition
        v_out[0, :] = v_boundary.copy()

        # loop through time and compute the updated 1-d radial solution
        for t in range(1, Nt):
            # Pull out the upwind and downwind slices at current time
            u_up = v_out[1:, t - 1].copy()
            u_dn = v_out[:-1, t - 1].copy()
            u_up_next = _upwind_step_(model, u_up, u_dn)
            # Save the updated time step
            v_out[1:, t] = u_up_next.copy()

        return v_out

def _upwind_step_(model, v_up, v_dn):
    """
    Function to compute the next step in the upwind scheme of Burgers equation with added residual acceleration of
    the solar wind. Model should be an instance of a HUXt model class, as it must contain the models domain and parameters.
    :param model: An instsance of the HUXt2DCME class.
    :param v_up: A numpy array of the upwind radial values. Units of km/s
    :param v_dn: A numpy array of the downwind radial values. Units of km/s
    :return: The upwind values at the next time step, numpy array with units of km/s
    """

    # Arguments for computing the acceleration factor
    accel_arg = -model.rrel[:-1] / model.r_accel
    accel_arg_p = -model.rrel[1:] / model.r_accel

    # Get estimate of next time step
    v_up_next = v_up - model.dtdr*v_up*(v_up - v_dn)
    # Compute the probable speed at 30rS from the observed speed at r
    v_source = v_dn / (1.0 + model.alpha*(1.0 - np.exp(accel_arg)))
    # Then compute the speed gain between r and r+dr
    v_diff = model.alpha*v_source*(np.exp(accel_arg) - np.exp(accel_arg_p))
    # Add the residual acceleration over this grid cell
    v_up_next = v_up_next + (v_dn*model.dtdr*v_diff)
    return v_up_next

def _zerototwopi_(angles):
    """
    Function to constrain angles to the 0 - 2pi domain.
    :param angles: a numpy array of angles
    :return: a numpy array of angles
    """
    twopi = 2.0*np.pi
    angles_out = angles.copy()
    a = -np.floor_divide(angles_out, twopi)
    angles_out = angles_out + (a*twopi)
    return angles_out

def _setup_dirs_():
    """
    Function to pull out the directories to save figures and output data.
    """
    # Find the config.dat file path
    files = glob.glob('config.dat')

    if len(files) != 1:
        # If too few or too many config files, guess directories
        print('Error: Cannot find correct config file with project directories. Check config.txt exists')
        print('Defaulting to current directory')
        dirs = {'root': os.getcwd()}
        for rel_path in ['boundary_conditions', 'HUXt1D_data', 'HUXt2D_data', 'HUXt1D_figures', 'HUXt2D_figures']:
            dirs[rel_path] = os.getcwd()
    else:
        # Extract data and figure directories from config.dat
        with open(files[0], 'r') as file:
            lines = file.read().splitlines()
            root = lines[0].split(',')[1]
            dirs = {l.split(',')[0]: os.path.join(root, l.split(',')[1]) for l in lines[1:]}

        # Just check the directories exist.
        for val in dirs.values():
            if not os.path.exists(val):
                print('Error, invalid path, check config.dat: ' + val)

    return dirs
