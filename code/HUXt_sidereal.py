# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:50:03 2020

@author: mathewjowens
"""

import numpy as np
import HUXt_OMNI as Homni
import HUXt as H
import astropy.units as u
import matplotlib.pyplot as plt
import os
import pandas as pd

os.chdir(os.path.abspath(os.environ['DBOX'] + 'python'))
import helio_time as htime

#change to the HUXt dir so that the config.dat is found
os.chdir(os.path.abspath(os.environ['DBOX'] + 'python_repos\\HUXt\\code'))

from scipy.spatial import cKDTree

def interp2d(xi, yi, V, x, y, n_neighbour = 4):
    """
    Fast 3d interpolation on an irregular grid. Uses the K-Dimensional Tree
    implementation in SciPy. Neighbours are weighted by 1/d^2, where d is the 
    distance from the required point.
    
    Based on Earthpy exmaple: http://earthpy.org/interpolation_between_grids_with_ckdtree.html
    
    Mathew Owens, 8/7/20
    
    Added check for infinite weights, resulting from the interpolated points 
    being identicial to original grid points. 26/11/20

    Parameters
    ----------
    xi, yi, zi :  Ni x Mi arrays of new positions at which to interpolate. 
    
    V : N x M array of the parameter field to be interpolated
    
    x, y, z: N x M arrays of the position of the parameter field, V
        
    n_neighbour : Number of neighbours to use in interpolation. The default is 4.

    Returns
    -------
    Vi : Ni x Mi array of the parameter at new positions.

    """
    
    #check if the interpolated points are singular
    if isinstance(xi,float):
        xi = np.array([xi])
        yi = np.array([yi])

    
    #check that the dimensions of the coords and V are the same
    assert len(V) == len(x)
    assert len(x) == len(y)   
    assert len(xi) == len(yi)
    
    z=x*0.0
    zi=xi*0.0
    
    #create a list of grid points
    gridpoints=np.ones((len(x.flatten()),3))
    gridpoints[:,0]=x.flatten()
    gridpoints[:,1]=y.flatten()
    gridpoints[:,2]=z.flatten()
    
    #create a list of densities
    V_list=V.flatten()
    
    #Create cKDTree object to represent source grid
    tree=cKDTree(gridpoints)
    
    #get the size of the new coords
    origsize=xi.shape

    newgridpoints=np.ones((len(xi.flatten()),3))
    newgridpoints[:,0]=xi.flatten()
    newgridpoints[:,1]=yi.flatten()
    newgridpoints[:,2]=zi.flatten()
           
    #nearest neighbour
    #d, inds = tree.query(newgridpoints, k = 1)
    #rho_ls[:,ie]=rholist[inds]
    
    #weighted sum of N nearest points
    distance, index = tree.query(newgridpoints, k = n_neighbour)
    #tree.query  will sometimes return an index past the end of the grid list?
    index[index>=len(gridpoints)]=len(gridpoints)-1
    
    #weight each point by 1/dist^2
    weights = 1.0 / distance**2
    
    #check for infinite weights
    areinf=np.isinf(weights[:,0])
    weights[areinf,0]=1.0
    weights[areinf,1:]=0.0
    
    #generate the new value as the weighted average of the neighbours
    Vi_list = np.sum(weights * V_list[index], axis=1) / np.sum(weights, axis=1)
    
    #revert to original size
    Vi=Vi_list.reshape(origsize)
    
    return Vi
# <codecell> 
#get the HUXt inputs
cr=1830
vr_in, br_in = Homni.Hin.get_MAS_long_profile(cr)

#convert longitude profiles to time series
input_vr = np.flipud(vr_in)
input_br = np.flipud(br_in)
input_rho = input_br * np.nan

tsyn = Homni.huxt_constants()['synodic_period']
tsyn_day = tsyn.to(u.day).value
dt = tsyn_day/(len(input_vr))
input_days = np.arange(dt/2, tsyn_day -dt/2 +0.0001, dt)*u.day
input_days = np.arange(0, tsyn_day, dt)*u.day
#dphi = 2*np.pi/(len(input_vr))
#input_CarrLong = np.flipud(np.arange(dphi/2 ,2*np.pi-dphi/2 +0.001 ,dphi))

longitude = 60*u.deg
#now run HUXt
modelOMNI = Homni.HUXt_OMNI(input_vr = input_vr, input_br =input_br,
               input_days = input_days, input_rho = input_rho,
               latitude=0*u.deg, #lon_out=longitude,
               simtime=25*u.day, dt_scale=4,
               r_min=30 * u.solRad, r_max=240 * u.solRad,
               frame='sidereal')

modelOMNI.solve([]) 

model = H.HUXt(v_boundary = vr_in, br_boundary = br_in,             
               latitude=0*u.deg, #lon_out=longitude,
               simtime=25*u.day, dt_scale=4,
               r_min=30 * u.solRad, r_max=240 * u.solRad,
               #r_max = 215 *10*u.solRad,
               frame='sidereal')
model.solve([]) 



t = 10*u.day
modelOMNI.plot_radial(t, lon=0.0*u.deg,field='v')
model.plot_radial(t, lon=0.0*u.deg,field='v')

# #model.plot_radial(t, lon=200.0*u.deg,field='ambient')
model.plot(t, field='v')
modelOMNI.plot(t, field='v')

# model.plot(t, field='br')



# <codecell> 
modelOMNI = Homni.HUXt_OMNI(input_vr = input_vr, input_br =input_br,
               input_days = input_days, input_rho = input_rho,
               latitude=0*u.deg, #lon_out=longitude,
               simtime=20*u.day, dt_scale=4,
               frame='sidereal')


modelOMNI.solve([]) 

t = 7*u.day
modelOMNI.plot(t, field='v')
modelOMNI.plot(t, field='br')
modelOMNI.plot(t, field='bpol')

# <codecell> Load in OMNI data

import heliopy.data.omni as omni

from datetime import datetime

os.chdir(os.path.abspath(os.environ['DBOX'] + 'python'))
#change to the HUXt dir so that the config.dat is found
os.chdir(os.path.abspath(os.environ['DBOX'] + 'python_repos\\HUXt\\code'))

starttime = datetime(2020, 8, 1, 0, 0, 0)
endtime =  datetime(2020, 11, 27, 0, 0, 0)

temp=omni.h0_mrg1hr(starttime, endtime)
data=temp.data

#remove nans
data.dropna(subset=['V'], inplace=True)

input_days = htime.datetime2mjd(data.index).to_numpy()*u.day
input_vr = data['V'].to_numpy()*u.km/u.s



# <codecell> Run HUXt_OMNI

#double up the last CR to extend the run

pos = np.where(input_days.value >= input_days.value[-1] - 27.27)
pad_vr=np.append(input_vr,input_vr[pos])
dt=np.nanmean(input_days[1:]-input_days[:-1])
pad_days = np.append(input_days,input_days[pos] + 27.27*u.day +dt)

pos = np.where(input_days.value >= input_days.value[-1] - 27.27)
pad_vr=np.append(pad_vr,input_vr[pos])
dt=np.nanmean(input_days[1:]-input_days[:-1])
pad_days = np.append(pad_days,input_days[pos] + 2*27.27*u.day +dt)


modelOMNI = Homni.HUXt_OMNI(input_vr = pad_vr, 
                            input_br = pad_vr.value*0*u.dimensionless_unscaled,
                            input_days = pad_days, 
                            input_rho = pad_vr.value*0.0*u.dimensionless_unscaled +1000.0*u.dimensionless_unscaled,
               latitude=0*u.deg, #lon_out=longitude,
               r_min=215*u.solRad, r_max = 2165*u.solRad,
               simtime=130*u.day, dt_scale=30,
               frame='sidereal')

modelOMNI.solve([]) 
#modelOMNI.animate(field='v',tag='omni_test')

t = 50*u.day
modelOMNI.plot(t, field='rho')

# <codecell> extract the Earth time series, assuming it is at phi = 0 at t=0

#generate the density profile
#modelOMNI.rho_post_process()


#load the planetary ephemeris data from  https://omniweb.gsfc.nasa.gov/coho/helios/heli.html
dirpath = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\JonnyNichols\\'

#Earth
filepath = dirpath + 'Earth_HGI.lst'
pos_Earth = pd.read_csv(filepath,
                     skiprows = 1, delim_whitespace=True,
                     names=['year','doy',
                            'rad_au','HGI_lat','HGI_lon'])
#convert to mjd
pos_Earth['mjd'] = htime.doyyr2mjd(pos_Earth['doy'],pos_Earth['year'])

#Saturn
filepath = dirpath + 'Saturn_HGI.lst'
pos_Saturn = pd.read_csv(filepath,
                     skiprows = 1, delim_whitespace=True,
                     names=['year','doy',
                            'rad_au','HGI_lat','HGI_lon'])
#convert to mjd
pos_Saturn['mjd'] = htime.doyyr2mjd(pos_Saturn['doy'],pos_Saturn['year'])


from astropy.coordinates import spherical_to_cartesian
#convert the HUXt grid to cartesean
gridx, gridy, gridz = spherical_to_cartesian(modelOMNI.r_grid.value, 
                                             modelOMNI.lon_grid.value*0.0, 
                                             modelOMNI.lon_grid.value)

#determine the HGI longitude at t = 0
HGI_lon_0 = np.interp(input_days[0].to(u.day).value,pos_Earth['mjd'],pos_Earth['HGI_lon'])*np.pi/180

v_Earth = np.ones((len(modelOMNI.time_out),5))*np.nan
v_Saturn = np.ones((len(modelOMNI.time_out),5))*np.nan

#time
v_Earth[:,0] = modelOMNI.time_out.to(u.day).value + input_days[0].to(u.day).value 
v_Saturn[:,0] = modelOMNI.time_out.to(u.day).value + input_days[0].to(u.day).value
#radius, in rs
v_Earth[:,1] = np.interp(v_Earth[:,0],pos_Earth['mjd'].to_numpy(),
                         pos_Earth['rad_au'].to_numpy(),left =np.nan, right =np.nan)*215.032
v_Saturn[:,1] = np.interp(v_Saturn[:,0],pos_Saturn['mjd'].to_numpy(),
                          pos_Saturn['rad_au'].to_numpy(),left =np.nan, right =np.nan)*215.032
#longitude relative to Earth at the start of the run
v_Earth[:,2] = Homni._zerototwopi_(np.interp(v_Earth[:,0],pos_Earth['mjd'].to_numpy(),
                                             pos_Earth['HGI_lon'].to_numpy(),
                                             left =np.nan, right =np.nan)*np.pi/180 -
                                   HGI_lon_0)
                                   
v_Saturn[:,2] = Homni._zerototwopi_(np.interp(v_Saturn[:,0],pos_Saturn['mjd'].to_numpy(),
                                              pos_Saturn['rad_au'].to_numpy(),
                                              left =np.nan, right =np.nan)*np.pi/180 -
                                    HGI_lon_0)

for t in range(0, len(modelOMNI.time_out)):
    #v_Earth[t,0] = modelOMNI.time_out[t].to(u.day).value + input_days[0].to(u.day).value  
    
    #tsyn_s = Homni.huxt_constants()['synodic_period'].to(u.s)
    #tsid_s = Homni.huxt_constants()['sidereal_period'].to(u.s)
    #v_Earth[t,1] = 215
    #v_Earth[t,2] = Homni._zerototwopi_(modelOMNI.time_out[t] 
    #                                   * 2 *np.pi *(1/tsid_s -1/tsyn_s)) 
    
    
    #convert to cartesian
    x, y, z = spherical_to_cartesian(v_Earth[t,1],  0, v_Earth[t,2])
    #now interpolate the V solutions to the Earth pos
    v_Earth[t,3] = interp2d(x.value, y.value, modelOMNI.v_grid[t,:,:].value, 
                        gridx, gridy,  4)
    v_Earth[t,4] = interp2d(x.value, y.value, modelOMNI.rho_grid[t,:,:].value, 
                        gridx, gridy,  4)
    #convert to cartesian
    x, y, z = spherical_to_cartesian(v_Saturn[t,1],  0, v_Saturn[t,2])
    #now interpolate the V solutions to the Earth pos
    v_Saturn[t,3] = interp2d(x.value, y.value, modelOMNI.v_grid[t,:,:].value, 
                        gridx, gridy,  4)
    v_Saturn[t,4] = interp2d(x.value, y.value, modelOMNI.rho_grid[t,:,:].value, 
                        gridx, gridy,  4)

from datetime import timedelta
#plt.plot(input_days,input_vr)
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(111)
ax1.plot(htime.mjd2datetime(v_Earth[:,0]),v_Earth[:,3],label = 'Earth')
ax1.plot(htime.mjd2datetime(v_Saturn[:,0]),v_Saturn[:,3],label = 'Saturn')
ax1.set_ylabel('V [km/s]')
ax1.legend()
ax1.set_xticks(np.arange(htime.mjd2datetime(np.floor(v_Earth[:,0]))[0], 
                     htime.mjd2datetime(np.floor(v_Earth[:,0])+1)[-1], 
                     timedelta(days=20)))
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.tick_params(which='minor', length=4, color='k')
ax1.tick_params(which='major', length=10, color='k')
plt.show()

# ax2 = fig.add_subplot(122)
# ax2.plot(htime.mjd2datetime(v_Earth[:,0]),v_Earth[:,4],label = 'Earth')
# ax2.plot(htime.mjd2datetime(v_Saturn[:,0]),v_Saturn[:,4],label = 'Saturn')
# ax2.set_ylabel('Density [unscaled units]')
# ax2.legend()