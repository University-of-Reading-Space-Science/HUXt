# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:01:24 2020

@author: mathewjowens
"""
import numpy as np
import HUXt as H
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import jit
import os


@jit(nopython=False)
def generate_v_boundary_ts(self, cme_list=[]):
    """
    Genreate the input time series for the given longitudinla solar wind and list of cone cmes.
    :param cme_list: A list of ConeCME instances to insert into the model.
    """
    
    # Check only cone cmes in cme list
    cme_list_checked = []
    for cme in cme_list:
        if isinstance(cme, ConeCME):
            cme_list_checked.append(cme)
        else:
            print("Warning: cme_list contained objects other than ConeCME instances. These will be excluded")
    
    self.cmes = cme_list_checked
            
    constants= huxt_constants()
    buffertime=constants['buffertime'].to(u.s)
    self.model_time = np.arange(-buffertime.value, (self.simtime.to('s') + self.dt).value, self.dt.value) * self.dt.unit

    dlondt = self.twopi * self.dt / self.synodic_period
    lon, dlon, Nlon = longitude_grid()

    # How many radians of Carrington rotation in this simulation length
    simlon = self.twopi * self.simtime / self.synodic_period
    # How many radians of Carrington rotation in the spin up period
    bufferlon = self.twopi * buffertime / self.synodic_period
    # Find the Carrigton longitude range spanned by the spin up and simulation period,
    # centered on simulation longitude
    #lonint = np.arange(self.lon.value-bufferlon, self.lon.value + simlon+dlondt, dlondt)
    lonint = np.arange(self.lon.value - simlon-dlondt, self.lon.value+bufferlon, dlondt)
    # Rectify so that it is between 0 - 2pi
    loninit = _zerototwopi_(lonint)
    # Interpolate the inner boundary speed to this higher resolution
    vinit = np.interp(loninit, lon.value, self.v_boundary.value, period=self.twopi) * self.kms
    # convert from cr longitude to time
    self.vinput_ts_amb = np.flipud(vinit).copy()
    self.vinput_ts_cme = np.flipud(vinit).copy()

    # Add the CME(s)
    # ----------------------------------------------------------------------------------------
    for t, time in enumerate(self.model_time):            
        # Compute boundary speed of each CME at this time. Set boundary to the maximum CME speed at this time.
        if time > 0:
            if len(cme_list_checked) != 0:
                v_update_cme = np.zeros(len(cme_list_checked)) * self.kms
                for i, c in enumerate(cme_list_checked):
                    r_boundary = self.r.min().to(u.km)
                    v_update_cme[i] = _cone_cme_boundary_1d_(r_boundary, self.lon, time, self.vinput_ts_cme[t], c)
                self.vinput_ts_cme[t] = v_update_cme.max()



@jit(nopython=False)
def solve_upwind_Rout(v_input, dt=500*u.s, tmax=10*u.day, rin = 30.0 *u.solRad, rout= 215.0*u.solRad, dr=1.4747*u.solRad):
    """
    Solve the upwind scheme for Burgers equation for the time evolution of the radial wind speed.
    This version does not require a model class and only returns the speed at rout
    
    :param v_input: Time series of inner boundary solar wind speeds, in km/s.
    :dt: The time step of v_input in seconds
    
    """
    #convert input units
    tmax=tmax.to('s')
    rin=rin.to('km')
    rout=rout.to('km')
    dr=dr.to('km')
    dtdr=dt/dr

    #get the HUXt constants
    constants= huxt_constants()
    alpha = constants['alpha']  # Scale parameter for residual SW acceleration
    r_accel = constants['r_accel']  # Spatial scale parameter for residual SW acceleration
    #v_max = constants['v_max'] 
    r_accel=r_accel.to('km')   
    
    #check the CFL condition is met
    if (np.nanmax(v_input)>dr.to('km')/dt):
        print('Warning: CFL condition not met, time step too short. Increase HUXt.constants.v_max')    

    #get the radial grid parameters
    Nr=int((rout-rin)/dr)
    r= np.linspace(rin, rout, Nr)
    rrel = r - r[0]
    
    voutindex=int(r.size)
    n_steps=v_input.size
    
    # Initialise the arrays
    vt = np.ones(Nr) * v_input[0]
    urnext = np.ones(Nr) * np.nan *u.km/u.s
    ur = np.ones(Nr) * np.nan *u.km/u.s
    vsource=np.ones(Nr) * np.nan *u.km/u.s
    vdiff=np.ones(Nr) * np.nan *u.km/u.s
    times=np.ones(n_steps)*np.nan*u.s
    speeds=np.ones(n_steps)*np.nan*u.km/u.s
    times[0]=0.0
    speeds[0]=v_input[0]   
   
    # loop through time and compute the updated 1-d radial solution
    for t in range(1, n_steps):
        #using the _upwind_step_opt_ function is slower by ~ x100
        # # Pull out the upwind and downwind slices at current time
        # u_up=vt[1:];  u_dn=vt[:-1]
        # u_up_next = _upwind_step_opt_(u_up.to(u.km/u.s).value, 
        #                                   u_dn.to(u.km/u.s).value,
        #                                   dtdr.to(u.s/u.km).value, 
        #                                   alpha, r_accel.to(u.km).value, rrel.to(u.km).value)
        # # Save the updated time step
        # vt[1:] = u_up_next.copy()*u.km/u.s
        # #update the inner boundary value
        # vt[0]=v_input[t]

        #update each radial position
        ur=vt
        urnext[1:]=ur[1:] - dt*ur[1:]*(ur[1:]-ur[:-1])/dr 
        #compute the source speed
        vsource=ur[:-1]/(1 + alpha*(1 - np.exp(-rrel[:-1]/r_accel)) )
        #compute the speed gain between r and r+dr
        vdiff=alpha*vsource*(1-np.exp(-rrel[1:]/r_accel)) - alpha*vsource*(1-np.exp(-rrel[:-1]/r_accel))
        #add the residual acceleration over this grid cell
        urnext[1:]=urnext[1:] + (ur[:-1]*dt/dr)*vdiff
        #update the grid
        vt=urnext
        #update the inner boundary value
        vt[0]=v_input[t]
        #save the output
        times[t]=t*dt
        speeds[t]=vt[voutindex-1]
    return times,speeds
        

@jit(nopython=False)
def solve_upwind_allR(v_input, dt=500*u.s, tmax=10*u.day, rin = 30.0 *u.solRad, rout= 215.0*u.solRad, dr=1.4747*u.solRad):
    """
    Solve the upwind scheme for Burgers equation for the time evolution of the radial wind speed.
    This version does not require a model class and returns the speed at all R values
    
    :param v_input: Time series of inner boundary solar wind speeds, in km/s.
    :dt: The time step of v_input in seconds
    
    """
    #convert input units
    tmax=tmax.to('s')
    rin=rin.to('km')
    rout=rout.to('km')
    dr=dr.to('km')
    
    #get the HUXt constants
    constants= huxt_constants()
    alpha = constants['alpha']  # Scale parameter for residual SW acceleration
    r_accel = constants['r_accel']  # Spatial scale parameter for residual SW acceleration
    #v_max = constants['v_max'] 
    r_accel=r_accel.to('km')   
    
    #check the CFL condition is met
    if (np.nanmax(v_input)>dr.to('km')/dt):
        print('Warning: CFL condition not met, time step too short. Increase HUXt.constants.v_max')
        
    #get the radial and temporal grid parameters
    Nr=int(np.ceil((rout-rin)/dr))+1
    r= np.linspace(rin, rout, Nr)
    rrel = r - r[0]
    n_steps=v_input.size
    
    # Initialise the arrays
    vt = np.ones(Nr) * v_input[0]
    urnext = np.ones(Nr) * np.nan *u.km/u.s
    ur = np.ones(Nr) * np.nan *u.km/u.s
    vsource=np.ones(Nr) * np.nan *u.km/u.s
    vdiff=np.ones(Nr) * np.nan *u.km/u.s
    times=np.ones(n_steps)*np.nan*u.s
    speeds=np.ones((n_steps,Nr))*np.nan*u.km/u.s
    times[0]=0.0
    speeds[0,:]=v_input[0]   
       
    # loop through time and compute the updated 1-d radial solution
    for t in range(1, n_steps):  
        #update each radial position
        ur=vt
        urnext[1:]=ur[1:] - dt*ur[1:]*(ur[1:]-ur[:-1])/dr
        #compute the source speed
        vsource=ur[:-1]/(1 + alpha*(1 - np.exp(-rrel[:-1]/r_accel)) )
        #compute the speed gain between r and r+dr
        vdiff=alpha*vsource*(1-np.exp(-rrel[1:]/r_accel)) - alpha*vsource*(1-np.exp(-rrel[:-1]/r_accel))
        #add the residual acceleration over this grid cell
        urnext[1:]=urnext[1:] + (ur[:-1]*dt/dr)*vdiff
        #update the grid
        vt=urnext
        #update the inner boundary value
        vt[0]=v_input[t]
        #save the output
        times[t]=t*dt
        speeds[t,:]=vt
    return times,r,speeds