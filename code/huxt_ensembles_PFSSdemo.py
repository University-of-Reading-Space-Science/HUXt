# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:11:46 2021

@author: mathewjowens
"""

# <codecell> Demo script - load data, generate ensemble, run HUXt
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.environ['DBOX'] + 'python_repos\\HUXt\\code'))
import huxt_inputs as Hin
import huxt as H
import huxt_ensembles as Hens

#==============================================================================
N=100 #number of ensemble members
lat_rot_sigma = 2*np.pi/180 #The standard deviation of the Gaussain from which the rotational perturbation is drawn
lat_dev_sigma = 2*np.pi/180 #The standard deviation of the Gaussain from which the linear latitudinal perturbation is drawn                          
long_dev_sigma = 2*np.pi/180 #The standard deviation of the Gaussain from which the linear longitudinal perturbation is drawn
#filepath = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\AnthonyYeates\\windbound_b_pf720_20130816.12.nc'; cr = 999
filepath = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\AnthonyYeates\\windbound_b20181105.12.nc'; 
cr=2210.3
r_in = 21.5*u.solRad #the radial distance of the speed map
simtime=27*u.day     #HUXt simulation time
dt_scale = 4         #output every n-th HUXt time step
#==============================================================================
#load the solar wind speed map, determine Earth lat
#==============================================================================
vr_map, vr_lats, vr_longs, br_map, br_lats, br_longs, phi, theta = Hin.get_PFSS_maps(filepath)

#Use the HUXt ephemeris data to get Earth lat over the CR
model = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=27.27*u.day, 
                   dt_scale=4, cr_num= np.floor(cr), lon_out=0.0*u.deg, 
                   r_min=21.5*u.solRad, r_max=215*u.solRad)
#retrieve a bodies position at each model timestep:
earth = model.get_observer('earth')
#get Earth lat as a function of longitude (not time)
E_lat = np.interp(vr_longs,np.flipud(earth.lon_c),np.flipud(earth.lat_c))

#plot the speed map       
plt.figure()
plt.pcolor(vr_longs.value*180/np.pi, vr_lats.value*180/np.pi, vr_map.value, 
           shading='auto',vmin=250, vmax=700)
plt.plot(vr_longs*180/np.pi,E_lat*180/np.pi,'r',label = 'Earth')
plt.plot(vr_longs*180/np.pi,E_lat*0,'k--')
plt.xlabel('Carrington Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.title('CR' + str(cr))
plt.legend()
cbar = plt.colorbar()
cbar.set_label(r'V$_{SW}$')

#==============================================================================
#generate the input ensemble
#==============================================================================
vr_ensemble = Hens.generate_input_ensemble(phi, theta, vr_map, 
                                      reflats = E_lat, Nens = N,
                                      lat_rot_sigma = lat_rot_sigma, 
                                      lat_dev_sigma = lat_dev_sigma,
                                      long_dev_sigma = long_dev_sigma)
    
#resample the ensemble to 128 longitude bins
vr128_ensemble = np.ones((N,128))  
dphi = 2*np.pi/128
phi128 = np.linspace(dphi/2, 2*np.pi - dphi/2, 128)
for i in range(0, N):
    vr128_ensemble[i,:] = np.interp(phi128,
                  vr_longs.value,vr_ensemble[i,:])
    
#==============================================================================
#run huxt with the input ensemble
#==============================================================================
nsteps = int(np.floor(simtime.value*24*60*60/model.dt.value/dt_scale))
huxtoutput = np.ones((N,nsteps))

os.chdir(os.environ['DBOX'] + 'python_repos\\HUXt\\code')
for i in range(0,N):
    model = H.HUXt(v_boundary=vr128_ensemble[i]* (u.km/u.s), simtime=simtime, 
                   dt_scale=dt_scale, cr_num= cr, lon_out=0.0*u.deg, 
                   r_min=r_in, r_max=215*u.solRad)
    model.solve([]) 
    
    #find Earth location and extract the time series
    #huxtoutput.append(HA.get_earth_timeseries(model))
    
    #it's quicker to just run to 215 rS and take the outer grid cell
    huxtoutput[i,:] = model.v_grid[:,-1,0]
    
    print('HUXt run ' + str(i+1) + ' of ' + str(N))

model_time = model.time_out


# <codecell> Additional stuff - plots and ensemble saving

#==============================================================================
#plots
#==============================================================================    

    
#solar wind speed as a function of longitude         
endata = vr128_ensemble
tdata = phi128*180/np.pi
confid_intervals = [5, 10, 33]

plt.figure()
Hens.plotconfidbands(tdata,endata,confid_intervals)
plt.plot(tdata,vr128_ensemble[0,:],'k',label='Unperturbed')
plt.legend(facecolor='grey')
plt.xlabel('Carrington Longitude [deg]')
plt.ylabel(r'V$_{SW}$ [km/s]')
plt.title('HUXt input at ' + str(r_in) +' rS')

    
#HUXt output
endata = huxtoutput
tdata = model_time/(24*60*60)

plt.figure()
Hens.plotconfidbands(tdata,endata,confid_intervals)
plt.plot(tdata,huxtoutput[0,:],'k',label='Unperturbed')
plt.legend(facecolor='grey')
plt.xlabel('Time through CR [days]')
plt.ylabel(r'V$_{SW}$ [km/s]')
plt.title('HUXt output at Earth')

#==============================================================================
#save the ensemble, e,g for use with DA
#==============================================================================
import h5py
savedir =  os.path.abspath(os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\MattLang\\HelioMASEnsembles_python')

h5f = h5py.File(savedir + '\\CR' + str(cr) +'_vin_ensemble.h5', 'w')
h5f.create_dataset('Vin_ensemble', data=vr128_ensemble)
h5f.attrs['lat_rot_sigma'] = lat_rot_sigma
h5f.attrs['lat_dev_sigma'] = lat_dev_sigma
h5f.attrs['long_dev_sigma'] = long_dev_sigma
h5f.attrs['source_file'] = filepath
h5f.close()