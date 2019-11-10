import numpy as np
import astropy.units as u
import os
import glob
import h5py
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage

           
class ConeCME:
    
    def __init__(self, t_launch=0.0, longitude=0.0, v=1000.0, width=30.0, thickness=10.0):
        
        self.t_launch = t_launch * u.s #  Time of CME launch, after the start of the simulation
        self.longitude = np.deg2rad(longitude) * u.rad  #  Longitudinal launch direction of the CME
        self.v = v * u.km / u.s #  CME nose speed
        self.width = np.deg2rad(width) * u.rad #  Angular width
        self.initial_height = (30.0 * u.solRad).to(u.km) # Initial height of CME (should match inner boundary of HUXt)
        self.radius = self.initial_height * np.tan(self.width) #  Initial radius of CME
        self.thickness = (thickness * u.solRad).to(u.km) #  Extra CME thickness
            
        
class HUXt2DCME:

    def __init__(self, simtime=5.0, dt_scale=1.0):

        # some constants and units
        self.twopi = 2.0 * np.pi
        daysec = 24 * 60 * 60
        self.kms = u.km / u.s
        self.alpha = 0.15  # Scale parameter for residual SW acceleration
        self.r_accel = 40 * u.solRad  # Spatial scale parameter for residual SW acceleration
        self.synodic_period = 27.2753 * daysec * u.s  # Solar Synodic rotation period from Earth.
        
        # Setup radial coordinates - in solar radius
        rmin = 30.0
        rmax = 240.0
        self.Nr = 140
        self.r, self.dr = np.linspace(rmin, rmax, self.Nr, retstep=True)
        self.r = self.r * u.solRad
        self.dr = self.dr * u.solRad
        self.rrel = self.r - self.r[0]

        # Set up longitudinal coordinates - in radians
        self.Nlon = 128
        dlon = self.twopi / self.Nlon
        lon_min = dlon / 2.0
        lon_max = self.twopi - (dlon / 2.0)
        self.lon, self.dlon = np.linspace(lon_min, lon_max, self.Nlon, retstep=True)
        self.lon = self.lon * u.rad
        self.dlon = self.dlon * u.rad
        
        # Set up time coordinates - in seconds
        # Get model timestep from the CFL condition, and gradient needed by upwind scheme
        self.vmax = 2000.0 * self.kms  # Maximum speed expected in model, used for CFL condition
        self.dt = self.dr.to(u.km) / self.vmax  
        self.dtdr = self.dt / self.dr.to(u.km)
        
        self.simtime = (simtime * u.day).to(u.s) #  number of days to simulate (in seconds)
        self.Nt = np.int32(np.floor(self.simtime.value / self.dt.value)); # number of time steps in the simulation
        self.dt_scale = dt_scale * u.dimensionless_unscaled
        self.dt_out = self.dt_scale * self.dt # time step of the output
        self.Nt_out = np.int32(self.Nt / self.dt_scale.value) # number of time steps in the output 
        self.longRef = self.twopi * (1.0 - 11.0/27.0) #  this sets the Carrington longitude of phi=180. So if this is the E-S line,
                                              #  this determines time through the Carringotn rotation
    
        self.time = np.arange(0, self.Nt) * self.dt #  Model timesteps
        self.time_out = np.arange(0, self.Nt_out) * self.dt_out #  Output timesteps 
        
        # Preallocate space for the output for the solar wind fields for the cme and ambient solution.
        self.v_grid_cme = np.zeros((self.Nt_out, self.Nr, self.Nlon)) * self.kms
        self.v_grid_amb = np.zeros((self.Nt_out, self.Nr, self.Nlon)) * self.kms
        
        # Mesh the spatial coordinates.
        self.lon_grid, self.r_grid = np.meshgrid(self.lon, self.r)
        
        # Extract paths of figure and data directories
        self._data_dir_, self._fig_dir_ = _setup_dirs_()
        
    
    def solve1D(self, v_boundary):
        """
        Functon to solve Burgers equation for the time evolution of the radial wind speed, given a variable input boundary condition.
        """
        # TODO - check input of v_boundary on size and unit.        
        Nt = v_boundary.size #number of time steps
        #Initialise output speeds as 400kms everywhere
        v_out = np.ones((self.Nr, Nt)) * 400.0 * self.kms
        # Update inner boundary condition
        v_out[0, :] = v_boundary.copy()
        
        #loop through time and compute the updated 1-d radial solution
        for t in range(1, Nt):
            # Pull out the upwind and downwind slices at current time
            u_up = v_out[1:, t-1].copy()
            u_dn = v_out[:-1, t-1].copy()
            u_up_next = _upwind_step_(self, u_up, u_dn)
            # Save the updated timestep
            v_out[1:, t] = u_up_next.copy()
            
        return v_out
    
        
    def solve_carrington_rotation(self, v_boundary):
                
        if v_boundary.size != 128:
            print('Warning from HUXt2D.solve_carrington_rotation: Longitudinal grid not as expected, radial grid may not be correct.')
        
        simtime = self.synodic_period #  One CR from Earth.
        buffertime = (5.0 * u.day).to(u.s) #  spin up time
        tmax = (simtime + buffertime) #  full simulation time
        
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
        ts_allR = self.solve1D(vinput)

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
    
    
    def solve_cone_cme(self, v_boundary, cme, save=False, tag=''):
        """
        Function to produce HDF5 file of output from HUXT2DCME run.
        """
        # Initialise v from the steady-state solution - no spin up required
        #----------------------------------------------------------------------------------------
        #compute the steady-state solution, as function of time, convert to function of long
        v_cr = self.solve_carrington_rotation(v_boundary)
        v_cr = np.fliplr(v_cr)
        
        # Initialise the output
        self.v_grid_cme[0, :, :] = v_cr.copy()
        self.v_grid_amb[0, :, :] = v_cr.copy()
        
        #compute longitude increment that matches the timestep dt
        dlondt = self.twopi * self.dt / self.synodic_period
        lon_tstep = np.arange(self.lon.value.min(), self.lon.value.max() + dlondt, dlondt) * u.rad
        #interpolate vin_long to this timestep matched resolution
        v_boundary_tstep = np.interp(lon_tstep.value, self.lon.value, v_boundary, period=self.twopi)
        
        #----------------------------------------------------------------------------------------
        # Main model loop
        #----------------------------------------------------------------------------------------
        for t in range(self.Nt):
            
            # Get the initial condition, which will update in the loop,
            # and snapshots saved to output at right steps.
            if t == 0:
                v_cme = self.v_grid_cme[t, :, :].copy()
                v_amb = self.v_grid_cme[t, :, :].copy()

            #loop through each longitude and compute the the 1-d radial solution
            for n in range(self.Nlon):
                
                #update cone cme v(r) for the given longitude
                #=====================================
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
                
            # Save this frame to output if is factor of output timestep
            if np.mod(t, self.dt_scale) == 0:
                t_out = np.int32(t / self.dt_scale) #  index of timestep in output array
                if t_out < self.Nt_out - 1: # Model can run one step longer than output steps, so check:
                    self.v_grid_cme[t_out, :, :] = v_cme.copy()
                    self.v_grid_amb[t_out, :, :] = v_amb.copy()
                
            # Update the boundary conditions for next timestep.
            if t < self.Nt-1:
                #  Update ambient solution inner boundary
                #==================================================================
                v_boundary_tstep = np.roll(v_boundary_tstep, 1)
                v_boundary_update = np.interp(self.lon.value, lon_tstep.value, v_boundary_tstep)
                v_amb[0, :] = v_boundary_update * self.kms

                #  Add cone CME to updated inner boundary
                #==================================================================
                v_boundary_cone = v_boundary_tstep.copy()
                v_boundary_cone = self._cone_cme_boundary_(lon_tstep, v_boundary_cone, self.time[t], cme)   
                v_boundary_update = np.interp(self.lon.value, lon_tstep.value, v_boundary_cone)
                v_cme[0, :] = v_boundary_update * self.kms
                  
        if save:
            if tag == '':
                print("Warning, blank tag means file likely to be overwritten")
                
            self.save_cone_cme_run(v_boundary, cme, tag=tag)

        return
    
    
    def _cone_cme_boundary_(self, longitude, v_boundary, t, cme):
        """
        Function to update inner boundary condition with the time dependent cone cme speed
        """
        # Center the longitude array on CME nose, and make it run from -pi to pi, to avoid dealing with any 0/2pi crossings
        lon_cent = longitude - cme.longitude
        id_high = lon_cent > np.pi*u.rad
        lon_cent[id_high] = 2.0*np.pi*u.rad - lon_cent[id_high]
        
        rin = self.r.min().to(u.km) 
        #  Compute y, the height of CME nose above the 30rS surface
        y = cme.v * (t - cme.t_launch) 
        if (y >= 0*u.km) & (y < cme.radius): # this is the front hemisphere of the spherical CME
            x = np.sqrt(y*(2*cme.radius - y)) #compute x, the distance of the current longitude from the nose
            #convert x back to an angular separation
            theta = np.arctan(x / rin)
            pos = (lon_cent > - theta) & (lon_cent <=  theta)
            v_boundary[pos] = cme.v.value
        elif (y >= (cme.radius + cme.thickness)) & (y <= (2*cme.radius + cme.thickness)):  # this is the back hemisphere of the spherical CME
            y = y - cme.thickness
            x = np.sqrt(y*(2*cme.radius - y))
            #convert back to an angle
            theta = np.arctan(x / rin)
            pos = (lon_cent > - theta) & (lon_cent <=  theta)
            v_boundary[pos] = cme.v.value
        elif (cme.thickness > 0*u.km) & (y >= cme.radius) & (y <= (cme.radius + cme.thickness)): #this is the "mass" between the hemispheres
            x = cme.radius
            #convert back to an angle
            theta = np.arctan(x / rin)
            pos = (lon_cent > - theta) & (lon_cent <=  theta)
            v_boundary[pos] = cme.v.value
            
        return v_boundary
    
    
    def save_cone_cme_run(self, v_boundary, cme, tag):
        """
        Function to save output to a HDF5 file. 
        """
        # Open up hdf5 data file for the HI flow stats
        filename = "HUXt2DCME_{}.hdf5".format(tag)
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
        cmegrp = out_file.create_group('ConeCME')
        for k, v in cme.__dict__.items():
            dset = cmegrp.create_dataset(k, data=v.value)
            dset.attrs['unit'] = v.unit.to_string()
            out_file.flush()
            
        # Loop over the attributes of model instance and save select keys/attributes.
        keys = ['simtime', 'dt_scale', 'time_out', 'dt_out', 'r', 'dr', 'lon', 'dlon', 'r_grid', 'lon_grid', 'v_grid_cme', 'v_grid_amb']
        for k, v in self.__dict__.items():
            
            if k in keys:
                
                if k in ['time_out', 'dt_out']:
                    kn = k.split('_')[0]#loose the "_out"
                    dset = out_file.create_dataset(k, data=v.value)
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
    
    
    def plot_frame(self, t, save=False, tag=''):
          
        # Get plotting data, and pad out to fill the full 2pi of contouring
        lon = self.lon_grid.value.copy()
        rad = self.r_grid.value.copy()
        v = self.v_grid_cme.value[t, :, :].copy()
        
        pad = lon[:,0].reshape((lon.shape[0],1)) + self.twopi
        lon = np.concatenate((lon, pad),axis=1)
        pad = rad[:,0].reshape((rad.shape[0],1))
        rad = np.concatenate((rad,pad),axis=1)
        pad = v[:,0].reshape((v.shape[0],1))
        v = np.concatenate((v,pad),axis=1)      
        
        levels = np.arange(250, 875, 25)
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection":"polar"})
        cnt = ax.contourf(lon, rad, v, levels=levels)
        ax.set_ylim(0, 230)
        ax.set_yticklabels([])
        fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.95)

        # Add colorbar
        pos = ax.get_position()
        dw = 0.005
        dh = 0.075
        left = pos.x0 + dw
        bottom = pos.y0 - dh
        wid = pos.width - 2 * dw
        cbaxes = fig.add_axes([left, bottom, wid, 0.03])
        cbar1 = fig.colorbar(cnt, cax=cbaxes, orientation='horizontal')
        cbar1.ax.set_xlabel("Solar Wind Speed (km/s)")

        # Add label
        time_label = "Time: {:3.2f} days".format(self.time_out[t].to(u.day).value)
        ax.set_title(time_label, position=(0.8, -0.05))
        
        if save:
            filename = "HUXt2DCME_{}_frame_{:03d}.png".format(tag, t)
            filepath = os.path.join(self._fig_dir_, filename)
            fig.savefig(filepath)
            
        return fig, ax
    
    
    def animate_plot(self, tag):
        duration = 10

        def make_frame(t):

            i = np.int32(self.Nt_out * t / duration)
            fig, ax = self.plot_frame(i)
            frame = mplfig_to_npimage(fig)
            plt.close('all')
            return frame
        
        filename = "HUXt2DCME_{}_movie.mp4".format(tag)
        filepath = os.path.join(self._fig_dir_, filename)
        animation = mpy.VideoClip(make_frame, duration=duration)
        animation.write_videofile(filepath, fps=24, codec='libx264')
        return
    
    
def load_cone_cme_run(filepath):
    """
    Function to load saved cone run output into the class
    """    
    if os.path.isfile(filepath):
        
        data = h5py.File(filepath, 'r')
        
        # Load in the inner boundary wind speed profile
        v_boundary = data['v_boundary'][()] * u.Unit(data['v_boundary'].attrs['unit'])
        
        # Load in the CME paramters
        cmedata = data['ConeCME']
        t_launch = cmedata['t_launch'][()]
        lon = np.rad2deg(cmedata['longitude'][()])
        width = np.rad2deg(cmedata['width'][()])
        thickness = cmedata['thickness'][()] * u.Unit(cmedata['thickness'].attrs['unit'])
        thickness = thickness.to('solRad').value
        v = cmedata['v'][()]
        cme = ConeCME(t_launch=t_launch, longitude=lon, v=v, width=width, thickness=thickness)
        
        # Initialise the model, and check it matches what the resolution and limits of the HDF5 file.
        simtime = data['simtime'][()] * u.Unit(data['simtime'].attrs['unit'])
        simtime = simtime.to(u.day).value                                           
        dt_scale = data['dt_scale'][()]
        model = HUXt2DCME(simtime=simtime, dt_scale=dt_scale)
        model.v_grid_cme[:,:,:] = data['v_grid_cme'][()] * u.Unit(data['v_boundary'].attrs['unit'])
        model.v_grid_amb[:,:,:] = data['v_grid_cme'][()] * u.Unit(data['v_boundary'].attrs['unit'])
           
    else:
        # File doesnt exist return nothing
        print("Warning: {} doesnt exist.".format(in_filepath))
            

    return v_boundary, cme, model


def _upwind_step_(model, v_up, v_dn):
    """
    Function to compute the next step in the upwind scheme of Burgers equation with added residual acceleration of solar wind.
    Model should be an instance of a HUXt model class, as it must contain the models domain and paramters.
    """
        
    # Arguments for computing the acceleration factor
    accel_arg = -model.rrel[:-1] / model.r_accel
    accel_arg_p = -model.rrel[1:] / model.r_accel
        
    # Get estimate of next timestep          
    v_up_next = v_up - model.dtdr * v_up * (v_up - v_dn)
    # Compute the probable speed at 30rS from the observed speed at r
    v_source = v_dn / (1.0 + model.alpha * (1.0 - np.exp(accel_arg)))
    # Then compute the speed gain between r and r+dr
    v_diff = model.alpha * v_source * (np.exp(accel_arg_p) - np.exp(accel_arg))
    # Add the residual acceleration over this grid cell
    v_up_next = v_up_next + (v_dn * model.dtdr * v_diff)
    return v_up_next
    
    
def _zerototwopi_(angles):
    """
    Constrain angles (in rad) to 0 - 2pi domain
    """
    twopi = 2.0 * np.pi
    angles_out = angles.copy()
    a = -np.floor_divide(angles_out, twopi)
    angles_out = angles_out + (a * twopi)
    return angles_out
    
    
def _setup_dirs_():
    """
    Function to pull out the directories to save figures and output data.
    """
    # Find the config.dat file path
    files = glob.glob('config.dat')

    if len(files) != 1:
        # If too few or too many config files, guess projdirs
        print('Error: Cannot find correct config file with project directories. Check config.txt exists')
        print('Defaulting to current directory')
        datadir = os.getcwd()
        figdir = os.getcwd()

    else:
        # Extract data and figure directories from config.dat
        with open(files[0], 'r') as file:
            lines = file.read().splitlines()
            dirs = {l.split(',')[0]: l.split(',')[1] for l in lines}

        # Just check the directories exist.
        for val in dirs.values():
            if not os.path.exists(val):
                print('Error, invalid path, check config.dat: ' + val)

        data_dir = dirs['data']
        figure_dir = dirs['figures']
        
    return dirs['data'], dirs['figures']
