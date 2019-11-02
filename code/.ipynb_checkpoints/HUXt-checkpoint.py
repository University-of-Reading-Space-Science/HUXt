import numpy as np
import astropy.units as u

           
class HUXt2D:

    def __init__(self):

        # some constants
        twopi = 2.0 * np.pi
        daysec = 24 * 60 * 60
        self.alpha = 0.15  # Scale parameter for residual SW acceleration
        self.r_accel = 40 * u.solRad  # Spatial scale parameter for residual SW acceleration
        self.synodic_period = 27.2753 * daysec * u.s  # Solar Synodic rotation period from Earth.
        self.vmax = 2000.0 * u.km / u.s  # Maximum speed expected in model, used for CFL condition
        
        # Setup radial coordinates - in solar radius
        rmin = 30.0
        rmax = 240.0
        self.Nr = 140
        self.r, self.dr = np.linspace(rmin, rmax, self.Nr, retstep=True)
        self.r = self.r * u.solRad
        self.dr = self.dr * u.solRad

        # Set up longitudinal coordinates - in radians
        self.Nphi = 128
        dphi = twopi / self.Nphi
        phimin = dphi / 2.0
        phimax = twopi - (dphi / 2.0)
        self.phi, self.dphi = np.linspace(phimin, phimax, self.Nphi, retstep=True)
        self.phi = self.phi * u.rad
        self.dphi = self.dphi * u.rad
        
        # Set up time coordinates - in seconds.
        self.dt = self.dr.to(u.km) / self.vmax  # maximum timestep set by the CFL condition.

        # Mesh the spatial coordinates.
        self.phi_grid, self.r_grid = np.meshgrid(self.phi, self.r)
        
    
    def solve1D(self, v_boundary):
        """
        Functon to solve Burgers equation for the time evolution of the radial wind speed, given a variable input boundary condition.
        """
        # TODO - check input of v_boundary on size and unit.
        
        rrel = self.r - self.r[0] #positions relative to the inner boundary
        # Arguments for computing the acceleration factor
        accel_arg = -rrel[:-1] / self.r_accel
        accel_arg_p = -rrel[1:] / self.r_accel
        dtdr = self.dt / self.dr.to(u.km) # factor needed in the upind scheme.
        
        Nt = v_boundary.size #number of time steps
        #Initialise output speeds as 400kms everywhere
        vout = np.ones((self.Nr, Nt)) * 400.0 * u.km / u.s
        # Update inner boundary condition
        vout[0, :] = v_boundary.copy()
        
        #loop through each boundary condition and compute the updated 1-d radial solution
        for t in range(1, Nt):
            # Pull out the upwind and downwind slices at current time
            ur = vout[1:, t-1].copy()
            urp = vout[:-1, t-1].copy()
            # Get estimate of next timestep          
            urnext = ur - dtdr * ur * (ur - urp)
            # Compute the probable speed at 30rS from the observed speed at r
            vsource = urp / (1.0 + self.alpha*(1.0 - np.exp(accel_arg)))
            # Then compute the speed gain between r and r+dr
            vdiff = self.alpha * vsource * (np.exp(accel_arg_p) - np.exp(accel_arg))
            # Add the residual acceleration over this grid cell
            urnext = urnext + (urp * dtdr * vdiff)
            # Save the updated timestep
            vout[1:, t] = urnext.copy()
            
        return vout
        
        
    def solve_carrington_rotation(self, v_boundary):
                
        if v_boundary.size != 128:
            print('Warning from HUXt2D.solve_carrington_rotation: Longitudinal grid not as expected, radial grid may not be correct. ')
        
        simtime = self.synodic_period # One CR from Earth.
        buffertime = (5.0 * u.day).to(u.s) # spin up time in days
        tmax = (simtime + buffertime)# full simulation time, including spin up
        
        twopi = 2.0 * np.pi
        # compute the new dphi to match dt
        dphidt = twopi * self.dt / self.synodic_period
        # work out the phi increment to allow for the spin up
        bufferphi = twopi * buffertime / self.synodic_period
        # create the input timeseries including the spin up series, periodic in phi
        phiint = np.arange(0, twopi + bufferphi + dphidt, dphidt)
        phiinit = self.__zerototwopi__(phiint)
        vinit = np.interp(phiinit, self.phi.value, v_boundary.value, period=twopi) * u.km / u.s
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
        times_orig = self.synodic_period * self.phi.value / twopi
        vout_allR = np.zeros((self.r.size, times_orig.size)) * ts_allR.unit
        for j in range(self.r.size):
            
            vout = np.interp(times_orig.value, times.value, ts_allR[j, :].value)
            vout_allR[j, :] = vout * u.km / u.s
        
        return vout_allR
    
    
    def solve_cone_cme(self, v_boundary):
        
        #----------------------------------------------------------------------------------------
        #  Define the CME Cone
        #----------------------------------------------------------------------------------------
        v_cme = 1200.0 * u.km / u.s #  CME nose speed
        width_cme = 30.0*np.pi/180.0 * u.rad #  Angular width
        radius_cme = (30.0 * u.solRad).to(u.km) * np.tan(width_cme) # Initial radius of CME
        thickness_cme = 0.5 * radius_cme #extra CME thickness (in terms of radius) to increase momentum
        longitude_cme = np.pi/2.0 * u.rad  #longitudinal launch direction of the CME
        t_launch_cme = 0.5 * 60 * 60 * 24 * u.s #time of CME launch, after the start of the simulation
        #--------------------------------------
        
        #----------------------------------------------------------------------------------------
        #  Setup some constants of the simulation.
        #----------------------------------------------------------------------------------------
        twopi = 2.0 * np.pi
        simtime = (5.0 * u.day).to(u.s) #  number of days to simulate (in seconds)
        nsteps = np.int32(np.floor(simtime.value / self.dt.value)); # number of required time steps
        longRef = twopi * (1.0 - 11.0/27.0) #  this sets the Carrington longitude of phi=180. So if this is the E-S line,
                                              #  this determines time through the Carringotn rotation
    
        #----------------------------------------------------------------------------------------
        # Initialise v from the steady-state solution - no spin up required
        #----------------------------------------------------------------------------------------
        #compute the steady-state solution, as function of time, convert to function of long
        vout = self.solve_carrington_rotation(v_boundary)
        vgridt_eclip = np.fliplr(vout.copy())
        vgridt_eclip_ambient = np.fliplr(vout.copy())

        #create matrices to store the whole t, r, phi data cubes
        v_t_r_phi_cone = np.zeros((nsteps, self.Nr, self.Nphi))
        v_t_r_phi_ambient = np.zeros((nsteps, self.Nr, self.Nphi))
        
        #compute the high resolution dphi to match dt (in order to produce time series at inner boundary)
        dphidt = twopi * self.dt/self.synodic_period
        phiinit = np.arange(self.phi.value.min(), self.phi.value.max() + dphidt, dphidt)
        #interpolate vin_long to this new grid resolution
        vinit = np.interp(phiinit, self.phi.value, v_boundary, period=twopi)

        #----------------------------------------------------------------------------------------
        # Get parameters needed for upwind scheme
        #----------------------------------------------------------------------------------------
        dtdr = self.dt / self.dr.to(u.km)
        rrel = self.r - self.r[0] #positions relative to the inner boundary
        # Arguments for computing the acceleration factor
        accel_arg = -rrel[:-1] / self.r_accel
        accel_arg_p = -rrel[1:] / self.r_accel
        dtdr = self.dt / self.dr.to(u.km) # factor needed in the upind scheme.
        
        time = []
        #----------------------------------------------------------------------------------------
        # Main model loop
        #----------------------------------------------------------------------------------------
        for t in range(0, nsteps):

            #loop through each longitude and compute the the 1-d radial solution
            for n in range(self.Nphi):
                
                #update v(r) for the given longitude
                #=====================================
                ur = vgridt_eclip[1:, n].copy()
                urp = vgridt_eclip[:-1, n].copy()
                # Get estimate of next timestep          
                urnext = ur - dtdr * ur * (ur - urp)
                # Compute the probable speed at 30rS from the observed speed at r
                vsource = urp / (1.0 + self.alpha*(1.0 - np.exp(accel_arg)))
                # Then compute the speed gain between r and r+dr
                vdiff = self.alpha * vsource * (np.exp(accel_arg_p) - np.exp(accel_arg))
                # Add the residual acceleration over this grid cell
                urnext = urnext + (urp * dtdr * vdiff)
                # Save the updated timestep
                vgridt_eclip[1:, n] = urnext.copy()
                
                # Repeat for ambient solution
                ur = vgridt_eclip_ambient[1:, n].copy()
                urp = vgridt_eclip_ambient[:-1, n].copy()
                # Get estimate of next timestep          
                urnext = ur - dtdr * ur * (ur - urp)
                # Compute the probable speed at 30rS from the observed speed at r
                vsource = urp / (1.0 + self.alpha*(1.0 - np.exp(accel_arg)))
                # Then compute the speed gain between r and r+dr
                vdiff = self.alpha * vsource * (np.exp(accel_arg_p) - np.exp(accel_arg))
                # Add the residual acceleration over this grid cell
                urnext = urnext + (urp * dtdr * vdiff)
                # Save the updated timestep
                vgridt_eclip_ambient[1:, n] = urnext.copy()
                
            #save the data
            v_t_r_phi_cone[t,:,:] = vgridt_eclip
            v_t_r_phi_ambient[t,:,:] = vgridt_eclip_ambient

            #==================================================================
            #update the inner boundary value
            #==================================================================
            vinit = np.roll(vinit, 1)
            v_update = np.interp(self.phi.value, phiinit, vinit)
            vgridt_eclip_ambient[0, :] = v_update * u.km/u.s

            #add the CME
            #======================================================================
            vcone = vinit.copy()
            
            rin = (30.0 * u.solRad).to(u.km) # TODO - this can be set from the radial array?
            t0 = t * self.dt #time from spin-up end
            time.append(t0.value)
            iscme=True
            if iscme:
                #compute y, the height of CME nose above the 30rS surface
                y = v_cme * (t0 - t_launch_cme)
                if (y >= 0*u.km) & (y < radius_cme): # this is the front hemisphere of the spherical CME
                    x = np.sqrt(y*(2*radius_cme - y)) #compute x, the distance of the current longitude from the nose
                    #convert x back to an angular separation
                    thet = np.arctan(x / rin)
                    pos = (phiinit > (longitude_cme.value - thet.value)) & (phiinit <= (longitude_cme.value + thet.value))
                    vcone[pos] = v_cme.value
                elif (y >= (radius_cme + thickness_cme)) & (y <= (2*radius_cme + thickness_cme)):  # this is the back hemisphere of the spherical CME
                    y = y - thickness_cme
                    x = np.sqrt(y*(2*radius_cme - y))
                    #convert back to an angle
                    thet=np.arctan(x / rin)
                    pos = (phiinit > (longitude_cme.value - thet.value)) & (phiinit <= (longitude_cme.value + thet.value))
                    vcone[pos] = v_cme.value
                elif (thickness_cme > 0*u.km) & (y >= radius_cme) & (y <= (radius_cme + thickness_cme)): #this is the "mass" between the hemispheres
                    x = radius_cme
                    #convert back to an angle
                    thet = np.arctan(x / rin)
                    pos = (phiinit > (longitude_cme.value - thet.value)) & (phiinit <= (longitude_cme.value + thet.value))
                    vcone[pos] = v_cme.value
                
            v_update = np.interp(self.phi.value, phiinit, vcone)
            vgridt_eclip[0, :] = v_update * u.km/u.s

        return np.array(time), v_t_r_phi_ambient, v_t_r_phi_cone
        

    def __zerototwopi__(self, angles):
        """
        Constrain angles (in rad) to 0 - 2pi domain
        """
        twopi = 2.0 * np.pi
        angles_out = angles.copy()
        a = -np.floor_divide(angles_out, twopi)
        angles_out = angles_out + (a * twopi)
        return angles_out

