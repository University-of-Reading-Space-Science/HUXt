# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 15:24:56 2021

@author: mathewjowens
"""

import huxt as H
import huxt_inputs as HI
import numpy as np
import astropy.units as u



def HUXt_timedependent(vgrid_Carr, time_grid, starttime, simtime,
                       r_min = 215*u.solRad, r_max = 1290*u.solRad,
                       dt_scale = 100, latitude = 0*u.deg, 
                       frame = 'sidereal',
                       lon_start = 0*u.rad, lon_stop = 2*np.pi *u.rad):
    
    """
    A fucntion to set up the HUXt class for use with time-dependent boundary conditions
    

    :param vgrid_Carr: input solar wind speed as a function of Carrington longitude and time
    :param time_grid: time steps (in MJD) of vgrid_Carr
    :param starttime: The datetime object giving the start of the HUXt run
    :param simtime: The duration fo the HUXt run (in u.day)
    """
    
    #work out the start time in terms of cr number and cr_lon_init
    cr, cr_lon_init = HI.datetime2huxtinputs(starttime)
    #bufferdays = 0 #the reconstructions have missing data prior to 28 days
    #cr_frac = sun.carrington_rotation_number(starttime + datetime.timedelta(days=bufferdays))
    #cr = int(np.floor(cr_frac))
    #cr_lon_init = 2*np.pi*(1 - (cr_frac-cr)) *u.rad
    
    
    #set up the dummy model class
    model = H.HUXt(v_boundary=np.ones((128))*400*u.km/u.s, 
                   lon_start= lon_start , lon_stop = lon_stop,
                   latitude = latitude, 
                   r_min = r_min, r_max = r_max,
                   simtime = simtime, dt_scale = dt_scale, cr_num = cr,
                   cr_lon_init  = cr_lon_init,
                   frame = frame)
    
    
    
    #extract the values from the model class
    buffertime = model.buffertime
    simtime = model.simtime
    frame = model.frame
    dt = model.dt
    cr_lon_init = model.cr_lon_init
    all_lons, dlon, nlon = H.longitude_grid()
    latitude = model.latitude
    time_init = model.time_init
    #model_time = model_new.time
    
    if frame == 'synodic':
        rotation_period = 27.2753 *  24 * 60 * 60 * u.s  # Solar Synodic rotation period from Earth.
    elif frame == 'sidereal':    
        rotation_period = 25.38 *  24 * 60 * 60 * u.s  # Solar sidereal rotation period
        
    
    
    # compute the model time step
    buffersteps = np.fix(buffertime.to(u.s) / dt)
    buffertime = buffersteps * dt
    model_time = np.arange(-buffertime.value, (simtime.to('s') + dt).value,dt.value) * dt.unit
    # dlondt = 2 * np.pi * dt / rotation_period
    
    # interpolate the solar wind speed onto the model grid
    
    #variables to store the input conditions.
    input_ambient_ts = np.nan * np.ones((model_time.size,nlon))
    
    
    for t in range(0,len(model_time)):
        mjd = time_init.mjd + model_time[t].to(u.day).value
        #find the nearest time to the current model time
        t_input = np.argmin(abs(time_grid - mjd))
        
         
        #shift the longitude to match the initial model time
        dlon_from_start = 2*np.pi * u.rad * model_time[t] / rotation_period
        
        lon_shifted = H._zerototwopi_((all_lons - cr_lon_init + dlon_from_start).value)
        #put longitudes in ascending order for np.interp
        id_sort = np.argsort(lon_shifted)
        lon_shifted = lon_shifted[id_sort]
        
        
        #take the vlong slice at this value
        v_boundary = vgrid_Carr[:,t_input]
        v_b_shifted = v_boundary[id_sort]
        #interpolate back to the original grid
        v_boundary = np.interp(all_lons.value, lon_shifted, v_b_shifted, period=2*np.pi)
        input_ambient_ts[t,:] = v_boundary
        
        
        
    #fill the nan values
    mask = np.isnan(input_ambient_ts)
    input_ambient_ts[mask] = 400
    
    
    #insert the data into the model instance
    model.model_time = model_time
    model.input_v_ts = input_ambient_ts
    
    return model



# <codecell> Example usage

# import h5py
# import datetime

# savedir =  os.environ['DBOX'] + 'python_repos\\SolarWindInputs_DTW\\output\\'
# data_dir = os.environ['DBOX'] + 'Data_hdf5\\'

# starttime = datetime.datetime(2020, 8, 1, 0, 0, 0)
# stoptime = datetime.datetime(2020, 10, 1, 12, 0, 0)

# filepath = savedir + 'Carr_OMNI_59001_to_59365_Corot_Back.h5' 

# h5f = h5py.File(filepath,'r')
# vgrid_Carr = np.array(h5f['vgrid_Carr_recon_back'])
# time_edges = np.array(h5f['time_edges'])
# lon_edges = np.array(h5f['lon_edges'])
# time_grid = np.array(h5f['time_grid'])
# lon_grid = np.array(h5f['lon_grid'])
# h5f.close()


# os.chdir(os.path.abspath(os.environ['DBOX'] + 'python_repos\\HUXt\\code'))


# #compute the required simulation time
# simtime = (stoptime-starttime).total_seconds()/24/60/60 *u.day

# model = HUXt_timedependent(vgrid_Carr, time_grid, starttime, simtime,
#                            lon_start = -30*u.deg, lon_stop = 90*u.deg)
# model.solve([])


# import huxt_analysis as HA
# HA.plot(model, 30*u.day)
