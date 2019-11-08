import numpy as np
import astropy.units as u
import os
import glob
import h5py

           
class ConeCME:
    
    def __init__(self,t_launch=0.0, longitude=0.0, v=1000.0, width=30.0, thickness=10.0):
        
        self.t_launch = t_launch * u.s #  Time of CME launch, after the start of the simulation
        self.longitude = np.deg2rad(longitude) * u.rad  #  Longitudinal launch direction of the CME
        self.v = v * u.km / u.s #  CME nose speed
        self.width = np.deg2rad(width) * u.rad #  Angular width
        self.initial_height = (30.0 * u.solRad).to(u.km) # Initial height of CME (should match inner boundary of HUXt)
        self.radius = self.initial_height * np.tan(self.width) #  Initial radius of CME
        self.thickness = (thickness * u.solRad).to(u.km) #  Extra CME thickness
        
        
        
class HUXt2D:

    def __init__(self):

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
        
        # Set up time timestep from the CFL condition, and gradient needed by upwind scheme
        self.vmax = 2000.0 * self.kms  # Maximum speed expected in model, used for CFL condition
        self.dt = self.dr.to(u.km) / self.vmax  
        self.dtdr = self.dt / self.dr.to(u.km) 

        # Mesh the spatial coordinates.
        self.lon_grid, self.r_grid = np.meshgrid(self.lon, self.r)
        
        # Extract paths of figure and data directories
        self._setup_dirs_()
        
    
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
            u_up_next = self._upwind_step_(u_up, u_dn)
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
        loninit = self._zerototwopi_(lonint)
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
    
    
    def solve_cone_cme(self, v_boundary, cme, save=False):
                
        #----------------------------------------------------------------------------------------
        #  Setup some constants of the simulation.
        #----------------------------------------------------------------------------------------
        simtime = (5.0 * u.day).to(u.s) #  number of days to simulate (in seconds)
        Nt = np.int32(np.floor(simtime.value / self.dt.value)); # number of required time steps
        longRef = self.twopi * (1.0 - 11.0/27.0) #  this sets the Carrington longitude of phi=180. So if this is the E-S line,
                                              #  this determines time through the Carringotn rotation
    
        #----------------------------------------------------------------------------------------
        # Initialise v from the steady-state solution - no spin up required
        #----------------------------------------------------------------------------------------
        #compute the steady-state solution, as function of time, convert to function of long
        v_cr = self.solve_carrington_rotation(v_boundary)
        v_cr = np.fliplr(v_cr)
        
        #create matrices to store the whole t, r, phi data cubes
        v_t_r_lon_cone = np.zeros((Nt, self.Nr, self.Nlon)) * self.kms
        v_t_r_lon_ambient = np.zeros((Nt, self.Nr, self.Nlon)) * self.kms
        
        v_t_r_lon_cone[0, :, :] = v_cr.copy()
        v_t_r_lon_ambient[0, :, :] = v_cr.copy()
        
        #compute longitude increment that matches the timestep dt
        dlondt = self.twopi * self.dt / self.synodic_period
        lon_tstep = np.arange(self.lon.value.min(), self.lon.value.max() + dlondt, dlondt) * u.rad
        #interpolate vin_long to this timestep matched resolution
        v_boundary_tstep = np.interp(lon_tstep.value, self.lon.value, v_boundary, period=self.twopi)
        
        time = np.arange(0, Nt) * self.dt
        #----------------------------------------------------------------------------------------
        # Main model loop
        #----------------------------------------------------------------------------------------
        for t in range(Nt):

            #loop through each longitude and compute the the 1-d radial solution
            for n in range(self.Nlon):
                
                #update cone cme v(r) for the given longitude
                #=====================================
                u_up = v_t_r_lon_cone[t, 1:, n].copy()
                u_dn = v_t_r_lon_cone[t, :-1, n].copy()
                u_up_next = self._upwind_step_(u_up, u_dn)
                # Save the updated timestep
                v_t_r_lon_cone[t, 1:, n] = u_up_next.copy()
            
                u_up = v_t_r_lon_ambient[t, 1:, n].copy()
                u_dn = v_t_r_lon_ambient[t, :-1, n].copy()
                u_up_next = self._upwind_step_(u_up, u_dn)
                # Save the updated timestep
                v_t_r_lon_ambient[t, 1:, n] = u_up_next.copy()
                
            if t < Nt-1:
                # Prepare next step
                v_t_r_lon_cone[t+1, :, :] = v_t_r_lon_cone[t, :, :].copy()
                v_t_r_lon_ambient[t+1, :, :] = v_t_r_lon_ambient[t, :, :].copy()
                
                #  Update ambient solution inner boundary
                #==================================================================
                v_boundary_tstep = np.roll(v_boundary_tstep, 1)
                v_boundary_update = np.interp(self.lon.value, lon_tstep.value, v_boundary_tstep)
                v_t_r_lon_ambient[t+1, 0, :] = v_boundary_update * self.kms

                #  Add cone CME to updated inner boundary
                #==================================================================
                v_boundary_cone = v_boundary_tstep.copy()
                v_boundary_cone = self._cone_cme_boundary_(lon_tstep, v_boundary_cone, time[t], cme)
                v_boundary_update = np.interp(self.lon.value, lon_tstep.value, v_boundary_cone)
                v_t_r_lon_cone[t+1, 0, :] = v_boundary_update * self.kms
                
                
        if save_run:
            self.save_cone_cme_run(v_boundary, cme, time, v_t_r_lon_ambient, v_t_r_lon_cone)

        return time, v_t_r_lon_ambient, v_t_r_lon_cone
    
    
    def save_cone_cme_run(self, v_boundary, cme, time, v_ambient, v_cone):
        """
        Function to save output to a HDF5 file. 
        """
        # Open up hdf5 data file for the HI flow stats
        out_filepath = os.path.join(self.__datadir__,"HUXt_output.hdf5")
        
        if os.path.isfile(out_filepath):
            # File exists, so delete and start new.
            print("Warning: {} already exists. Overwriting".format(out_filepath))
            os.remove(out_filepath)

        out_file = h5py.File(out_filepath, 'w')
        
        dset = out_file.create_dataset("v_boundary", data=v_bounary.value)
        dset.attrs['unit'] = v_boundary.unit.to_string()
        
        # Create a new group to store the Cone CME parameters.
        cmegrp = out_file.create_group('ConeCME')
        for key, value in cme.__dict__.items():
            dset = cmegrp.create_dataset(key, data=value.value)
            dset.attrs['unit'] = value.unit
        
        dset = out_file.create_dataset("time", data=time.value)
        dset.attrs['unit'] = time.unit.to_string()
        
        dset = out_file.create_dataset("dt", data=self.dt.value)
        dset.attrs['unit'] = self.dt.unit.to_string()
        
        dset = out_file.create_dataset("radius", data=self.r.value)
        dset.attrs['unit'] = self.r.unit.to_string()
        
        dset = out_file.create_dataset("dr", data=self.dr.value)
        dset.attrs['unit'] = self.dr.unit.to_string()
        
        dset = out_file.create_dataset("longitude", data=self.lon.value)
        dset.attrs['unit'] = self.lon.unit.to_string()
        
        dset = out_file.create_dataset("dlon", data=self.dlon.value)
        dset.attrs['unit'] = self.dlon.unit.to_string()
        
        dset = out_file.create_dataset("radius_grid", data=self.r_grid.value)
        dset.attrs['unit'] = self.r_grid.unit.to_string()
        dset.dims[0].label = 'radius'
        dset.dims[1].label = 'longitude'
        
        dset = out_file.create_dataset("lon_grid", data=self.lon_grid.value)
        dset.attrs['unit'] = self.lon_grid.unit.to_string()
        dset.dims[0].label = 'radius'
        dset.dims[1].label = 'longitude'
        
        dset = out_file.create_dataset("v_ambient", data=v_ambient.value)
        dset.attrs['unit'] = v_ambient.unit.to_string()
        dset.dims[0].label = 'time'
        dset.dims[1].label = 'radius'
        dset.dims[2].label = 'longitude'
        
        dset = out_file.create_dataset("v_cone", data=v_cone.value)
        dset.attrs['unit'] = v_cone.unit.to_string()
        dset.dims[0].label = 'time'
        dset.dims[1].label = 'radius'
        dset.dims[2].label = 'longitude'
        
        out_file.flush()
        out_file.close()
        return
        

    def _upwind_step_(self, v_up, v_dn):
        """
        Function to compute the next step in the upwind scheme of Burgers equation with added residual acceleration of solar wind
        """
        
        # Arguments for computing the acceleration factor
        accel_arg = -self.rrel[:-1] / self.r_accel
        accel_arg_p = -self.rrel[1:] / self.r_accel
        
        # Get estimate of next timestep          
        v_up_next = v_up - self.dtdr * v_up * (v_up - v_dn)
        # Compute the probable speed at 30rS from the observed speed at r
        v_source = v_dn / (1.0 + self.alpha * (1.0 - np.exp(accel_arg)))
        # Then compute the speed gain between r and r+dr
        v_diff = self.alpha * v_source * (np.exp(accel_arg_p) - np.exp(accel_arg))
        # Add the residual acceleration over this grid cell
        v_up_next = v_up_next + (v_dn * self.dtdr * v_diff)
        return v_up_next
    
    
    def _cone_cme_boundary_(self, longitude, v_boundary, t, cme):
        """
        Function to update inner boundary condition with the time dependent cone cme speed
        """
        
        rin = self.r.min().to(u.km) 
        #  Compute y, the height of CME nose above the 30rS surface
        y = cme.v * (t - cme.t_launch)
        if (y >= 0*u.km) & (y < cme.radius): # this is the front hemisphere of the spherical CME
            x = np.sqrt(y*(2*cme.radius - y)) #compute x, the distance of the current longitude from the nose
            #convert x back to an angular separation
            theta = np.arctan(x / rin)
            pos = (longitude > (cme.longitude - theta)) & (longitude <= (cme.longitude + theta))
            v_boundary[pos] = cme.v.value
        elif (y >= (cme.radius + cme.thickness)) & (y <= (2*cme.radius + cme.thickness)):  # this is the back hemisphere of the spherical CME
            y = y - cme.thickness
            x = np.sqrt(y*(2*cme.radius - y))
            #convert back to an angle
            theta = np.arctan(x / rin)
            pos = (longitude > (cme.longitude - theta)) & (longitude <= (cme.longitude + theta))
            v_boundary[pos] = cme.v.value
        elif (cme.thickness > 0*u.km) & (y >= cme.radius) & (y <= (cme.radius + cme.thickness)): #this is the "mass" between the hemispheres
            x = cme.radius
            #convert back to an angle
            theta = np.arctan(x / rin)
            pos = (longitude > (cme.longitude - theta)) & (longitude <= (cme.longitude + theta))
            v_boundary[pos] = cme.v.value
            
        return v_boundary

        
    def _zerototwopi_(self, angles):
        """
        Constrain angles (in rad) to 0 - 2pi domain
        """
        angles_out = angles.copy()
        a = -np.floor_divide(angles_out, self.twopi)
        angles_out = angles_out + (a * self.twopi)
        return angles_out
    
    
    def _setup_dirs_(self):
        """
        Function to pull out the directories to save figures and output data.
        """
        # Find the config.dat file path
        files = glob.glob('config.dat')
        
        if len(files) != 1:
            # If too few or too many config files, guess projdirs
            print('Error: Cannot find correct config file with project directories. Check config.txt exists')
            print('Defaulting to current directory')
            self.__datadir__ = os.getcwd()
            self.__figdir__ = os.getcwd()

        else:
            # Extract data and figure directories from config.dat
            with open(files[0], 'r') as file:
                lines = file.read().splitlines()
                dirs = {l.split(',')[0]: l.split(',')[1] for l in lines}

            # Just check the directories exist.
            for val in dirs.values():
                if not os.path.exists(val):
                    print('Error, invalid path, check config.dat: ' + val)
                
            self._datadir_ = dirs['data']
            self._figuredir_ = dirs['figures']
         
        return
        