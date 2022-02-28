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
sys.path.append('D:\\Dropbox\\python_repos\\HUX\\code')
import huxt_inputs as Hin
import huxt as H
import huxt_ensembles as Hens

#==============================================================================
# <codecell> input args

huxtpath = 'D:\\Dropbox\\python_repos\\HUXt\\' #sys.argv[1]
input_filepath = 'D:\\Dropbox\\Papers_WIP\\_coauthor\\AnthonyYeates\\windbound_b20181105.12.nc'# sys.argv[2]; 
t0_mjd = 58431.68 #sys.argv[3]

#This info would come from CME input files
n_cme = 0 #Read the CME input files, determine the number within the rHUXt run time frame
cme_mjds = [t0_mjd -2, t0_mjd + 1.5, t0_mjd+ 3]
cme_speeds = [850, 1000, 700]
cme_lons = [0, -10, 10]
cme_lats = [0, -10, 10]
cme_widths = [30, 40, 20]
cme_thicknesses = [5,5,5]

# <codecell> non-input args
#==============================================================================
N=100 #number of ensemble members
lat_rot_sigma = 5*np.pi/180 #The standard deviation of the Gaussain from which the rotational perturbation is drawn
lat_dev_sigma = 2*np.pi/180 #The standard deviation of the Gaussain from which the linear latitudinal perturbation is drawn                          
long_dev_sigma = 2*np.pi/180 #The standard deviation of the Gaussain from which the linear longitudinal perturbation is drawn

cme_v_sigma_frac = 0.1 #The standard deviation of the Gaussain from which the CME V [frac of value] perturbation is drawn
cme_width_sigma_frac = 0.1 #The standard deviation of the Gaussain from which the CME width [frac of value] perturbation is drawn
cme_thick_sigma_frac = 0.1 #The standard deviation of the Gaussain from which the CME thickness [frac of value] perturbation is drawn
cme_lon_sigma = 10 #The standard deviation of the Gaussain from which the CME long [deg] perturbation is drawn
cme_lat_sigma = 10 #The standard deviation of the Gaussain from which the CME lat [deg] perturbation is drawn

#filepath = os.environ['DBOX'] + 'Papers_WIP/_coauthor/AnthonyYeates/windbound_b_pf720_20130816.12.nc'; cr = 999

r_in = 21.5*u.solRad #the radial distance of the speed map
simtime=27*u.day     #HUXt simulation time
back_time = 10*u.day  #number of days previous to t0 to run, to account for earlier CMEs
dt_scale = 4         #output every n-th HUXt time step
#==============================================================================


# <codecell> Time conversion functions (from helio_time.py)

def crnum2mjd(crnum):
    """   
    Converts a Carrington Rotation number to MJD
    Mathew Owens, 16/10/20
    """
    return (crnum - 1750)*27.2753 + 45871.41

def mjd2crnum(mjd):
    """
    Converts MJD to Carrington Rotation number
    Mathew Owens, 16/10/20
    """ 
    return 1750 + ((mjd-45871.41)/27.2753)

# <codecell> process inputs
os.chdir(huxtpath + 'code')

tstart_mjd = t0_mjd - back_time.value

tstart_cr = mjd2crnum(tstart_mjd)
cr_num = np.floor(tstart_cr)
cr_frac = tstart_cr - cr_num
cr_lon_init = 2*np.pi * (1 -cr_frac) *u.rad


# <codecell> Load data
#load the solar wind speed map, determine Earth lat
#==============================================================================
vr_map, vr_lats, vr_longs, br_map, br_lats, br_longs, phi, theta = Hin.get_PFSS_maps(input_filepath)

#Use the HUXt ephemeris data to get Earth lat over the CR
model = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=27.27*u.day, 
                   dt_scale=4, cr_num= cr_num, lon_out=0.0*u.deg, 
                   r_min=21.5*u.solRad, r_max=215*u.solRad)
#retrieve a bodies position at each model timestep:
earth = model.get_observer('earth')
#get Earth lat as a function of longitude (not time)
E_lat = np.interp(vr_longs,np.flipud(earth.lon_c),np.flipud(earth.lat_c))

#compute the longitude and latitude of Earth at the forecast time
E_lat_init = np.interp(cr_lon_init,np.flipud(earth.lon_c),np.flipud(earth.lat_c))



#plot the speed map as a sanity check
#====================================      
# plt.figure()
# plt.pcolor(vr_longs.value*180/np.pi, vr_lats.value*180/np.pi, vr_map.value, 
#            shading='auto',vmin=250, vmax=700)
# plt.plot(vr_longs*180/np.pi,E_lat*180/np.pi,'r',label = 'Earth')
# plt.plot(vr_longs*180/np.pi,E_lat*0,'k--')
# plt.xlabel('Carrington Longitude [deg]')
# plt.ylabel('Latitude [deg]')
# plt.title('CR' + str(cr))
# plt.legend()
# cbar = plt.colorbar()
# cbar.set_label(r'V$_{SW}$')

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

#os.chdir(huxtpath + '/code')
for i in range(0,N):
    
    #perturb each CME
    cme_list_perturb = []
    for n in range(n_cme):
        v_perturb = cme_speeds[n] + np.random.normal(0.0, cme_v_sigma_frac* cme_speeds[n])
        width_perturb = cme_widths[n] + np.random.normal(0.0, cme_width_sigma_frac* cme_widths[n])
        thick_perturb = cme_thicknesses[n] + np.random.normal(0.0, cme_thick_sigma_frac* cme_thicknesses[n])
        lon_perturb = cme_lons[n] + np.random.normal(0.0, cme_lon_sigma)
        lat_perturb = cme_lats[n] + np.random.normal(0.0, cme_lat_sigma)
        
        cme = H.ConeCME(t_launch=(cme_mjds[n] - tstart_mjd)*u.day, 
                           longitude=lon_perturb*u.deg, latitude = lat_perturb*u.deg,
                           width=width_perturb*u.deg, 
                           v=v_perturb*u.km/u.s, thickness= thick_perturb*u.solRad)
        cme_list_perturb.append(cme)
    
    #run huxt
    model = H.HUXt(v_boundary=vr128_ensemble[i]* (u.km/u.s), simtime=simtime,
                   latitude = E_lat_init, cr_lon_init = cr_lon_init,
                   dt_scale=dt_scale, cr_num= cr_num, lon_out=0.0*u.deg, 
                   r_min=r_in, r_max=215*u.solRad)
    model.solve(cme_list_perturb) 
    
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

# plt.figure()
# Hens.plotconfidbands(tdata,endata,confid_intervals)
# plt.plot(tdata,vr128_ensemble[0,:],'k',label='Unperturbed')
# plt.legend(facecolor='grey')
# plt.xlabel('Carrington Longitude [deg]')
# plt.ylabel(r'V$_{SW}$ [km/s]')
# plt.title('HUXt input at ' + str(r_in) +' rS')

    
#HUXt output
endata = huxtoutput
tdata = model_time.to(u.day) - back_time

plt.figure()
Hens.plotconfidbands(tdata,endata,confid_intervals)
plt.plot(tdata,huxtoutput[0,:],'k',label='Unperturbed')
plt.legend(facecolor='grey')
plt.xlabel('Forecast lead time [days]')
plt.ylabel(r'V$_{SW}$ [km/s]')
plt.title('HUXt output at Earth')



