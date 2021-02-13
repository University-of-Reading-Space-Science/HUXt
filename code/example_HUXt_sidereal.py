# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:40:18 2021

Example scripts for running HUXt in sidereal frame

@author: mathewjowens
"""

import HUXt as H
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime as datetime

os.chdir(os.path.abspath(os.environ['DBOX'] + 'python'))
import helio_time as htime
import coords as hcoords

#change to the HUXt dir so that the config.dat is found
os.chdir(os.path.abspath(os.environ['DBOX'] + 'python_repos\\HUXt\\code'))






startdate=datetime.datetime(1977,9,27,0,0,0,999999)
runtime = 15*u.day

#create a bunch of CMEs
daysec = 86400
times = [2*daysec, 7*daysec, 11*daysec]
speeds = [850, 1000, 700]
lons = [0, 0, 0]
widths = [30, 40, 20]
thickness = [5, 4, 2]
cme_list = []
for t, l, w, v, thick in zip(times, lons, widths, speeds, thickness):
    cme = H.ConeCME(t_launch=t*u.s, longitude=l*u.deg, width=w*u.deg, 
                    v=v*u.km/u.s, thickness=thick*u.solRad)
    cme_list.append(cme)
    
    

startmjd = htime.datetime2mjd(startdate)
stopmjd = startmjd + runtime.to(u.day).value
#get the CR number
crnum = np.floor(htime.mjd2crnum(startmjd))

#fraction of CR gives the longitude
frac = htime.mjd2crnum(startmjd) - crnum
lon = 2*np.pi * (1 - frac) *180/np.pi

#get longitude grid resolution
nlong  = H.huxt_constants()['nlong']
dlongrid = 2*np.pi/nlong

#max longitude to be simulated
dlong = hcoords.zerototwopi((hcoords.earthecliplong(stopmjd) - 
                             hcoords.earthecliplong(startmjd))*np.pi/180)


#get the input data
vr_in, br_in = H.Hin.get_MAS_long_profile(crnum)

modelsid = H.HUXt(v_boundary=vr_in, cr_num=crnum, br_boundary=br_in, latitude=0*u.deg,
               simtime=runtime, cr_lon_init = lon *u.deg,
               #lon_start= - dlongrid/2 * u.rad , lon_stop = (dlong + dlongrid/2) * u.rad,
               dt_scale=4,frame='sidereal')
modelsid.solve(cme_list) 

# t_interest=11*u.day
# modelsid.plot(t_interest, field='v')

t_interest=13*u.day
modelsid.plot(t_interest, field='v')

r = 1.0*u.AU
modelsid.plot_timeseries(r, lon=0.0*u.deg,field='v')

modelsid.animate('v', tag='sidereal_test')

# <codecell>
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


#now interpolate the speed at Earth's position
mjd_model =  modelsid.time_out.to(u.day).value + startmjd
earth_lon = hcoords.earthecliplong(mjd_model) * np.pi/180
earth_dlon = earth_lon - earth_lon[0]
earth_R = hcoords.earth_R(mjd_model)


from astropy.coordinates import spherical_to_cartesian
#convert the HUXt grid to cartesean
gridx, gridy, gridz = spherical_to_cartesian(modelsid.r_grid.to(u.km).value, 
                                             modelsid.lon_grid.value*0.0, 
                                             modelsid.lon_grid.value)

earth_V_KDxy = earth_R*0.0
earth_V_KDrlon = earth_R*0.0
earth_V_grid = earth_R*0.0
earth_V_nearR = earth_R*0.0

from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator


r0 = earth_R[0]*u.km

for t in range(0,len(mjd_model)): 
    #convert to cartesian
    #x, y, z = spherical_to_cartesian(earth_R[t],  0, earth_dlon[t])
    x, y, z = spherical_to_cartesian(r0,  0, earth_dlon[t])
    
    #now interpolate the V solutions to the Earth pos
    earth_V_KDxy[t] = interp2d(x.value, y.value, modelsid.v_grid[t,:,:].value, 
                        gridx, gridy, 2)
    
    # earth_V_KDrlon[t] = interp2d(modelsid.r_grid.to(u.km).value, 
    #                              modelsid.lon_grid.value, 
    #                              modelsid.v_grid[t,:,:].value, 
    #                     r0.value,  earth_dlon[t], 2)
    
    #f = interpolate.interp2d(model.r_grid.value, model.lon_grid.value,  
    #                         model.v_grid[t,:,:].value)
    
    interp_function = RegularGridInterpolator((modelsid.r.to(u.km).value,
                                                         modelsid.lon.value), 
                                                        modelsid.v_grid[t,:,:].value,
                                                        bounds_error=False, fill_value=np.nan)
    #earth_V_grid[t] = interp_function((earth_R[t], earth_dlon[t]))
    earth_V_grid[t] = interp_function((r0.value, earth_dlon[t]))
    
    #fidn the nearest R coord
    #id_r = np.argmin(np.abs(modelsid.r.to(u.km).value - earth_R[t]))
    id_r = np.argmin(np.abs(modelsid.r.to(u.km).value - r0.value))
    id_lon = np.argmin(np.abs(modelsid.r.to(u.km).value - r0.value))
    #then interpolate the longitude
    earth_V_nearR[t] = np.interp(earth_dlon[t],
                                 modelsid.lon.value,modelsid.v_grid[t,id_r,:].value,
                                 period = 2*np.pi)
    
    
    

#r = r0.to(u.AU)
#modelsid.plot_timeseries(r, lon=0.0*u.deg,field='v')
    
# <codecell>  


 
    
#compare with synodic result
modelsyn = H.HUXt(v_boundary=vr_in, cr_num=crnum, br_boundary=br_in, latitude=0*u.deg,
               simtime=runtime, cr_lon_init = lon *u.deg,
               dt_scale=4,frame='synodic')
modelsyn.solve(cme_list) 

modelsyn.animate('v', tag='synodic_test')

t_interest=13*u.day
modelsyn.plot(t_interest, field='v')
modelsid.plot(t_interest, field='v')

# t_interest=13*u.day
# modelsyn.plot(t_interest, field='v')


r = r0.to(u.AU)
modelsyn.plot_timeseries(r, lon=0.0*u.deg,field='v')
plt.plot(mjd_model-mjd_model[0],earth_V_KDxy,label='KDTree (x,y)')   
#plt.plot(mjd_model-mjd_model[0],earth_V_KDrlon,label='KDTree (R,lon)')  
plt.plot(mjd_model-mjd_model[0],earth_V_grid,label='SciPy Grid (R, lon)') 
plt.plot(mjd_model-mjd_model[0],earth_V_nearR,label='Near R, linear Long') 
plt.legend()