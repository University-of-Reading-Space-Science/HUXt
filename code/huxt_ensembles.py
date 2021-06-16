# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 17:18:32 2021

@author: mathewjowens
"""
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# <codecell> Helper functions
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
    
    #check that the dimensions of the coords and V are the same
    assert len(V) == len(x)
    assert len(x) == len(y)
    assert len(xi) == len(yi)

    
    
    #create a list of grid points
    gridpoints=np.ones((len(x.flatten()),3))
    gridpoints[:,0]=x.flatten()
    gridpoints[:,1]=y.flatten()
    gridpoints[:,2]=x.flatten()*0
    
    #create a list of densities
    V_list=V.flatten()

    #Create cKDTree object to represent source grid
    tree=cKDTree(gridpoints)
    
    #get the size of the new coords
    origsize=xi.shape

    newgridpoints=np.ones((len(xi.flatten()),3))
    newgridpoints[:,0]=xi.flatten()
    newgridpoints[:,1]=yi.flatten()
    newgridpoints[:,2]=xi.flatten()*0
    
    #nearest neighbour
    #d, inds = tree.query(newgridpoints, k = 1)
    #rho_ls[:,ie]=rholist[inds]
    
    #weighted sum of N nearest points
    distance, index = tree.query(newgridpoints, k = n_neighbour)
    #tree.query  will sometimes return an index past the end of the grid list?
    index[index>=len(gridpoints)]=len(gridpoints)-1
    
    #weight each point by 1/dist^2
    weights = 1.0 / distance**2
    
    #check for infinite weights (i.e., interp points identical to original grid)
    areinf=np.isinf(weights[:,0])
    weights[areinf,0]=1.0
    weights[areinf,1:]=0.0
    
    #generate the new value as the weighted average of the neighbours
    Vi_list = np.sum(weights * V_list[index], axis=1) / np.sum(weights, axis=1)
    
    #revert to original size
    Vi=Vi_list.reshape(origsize)
    
    return Vi

def _zerototwopi_(angles):
    """
    Function to constrain angles to the 0 - 2pi domain.
    
    :param angles: a numpy array of angles
    :return: a numpy array of angles
    """
    twopi = 2.0 * np.pi
    angles_out = angles
    a = -np.floor_divide(angles_out, twopi)
    angles_out = angles_out + (a * twopi)
    return angles_out

# <codecell> generate ensembles
def generate_input_ensemble(phi, theta, vr_map, 
                            reflat = 0.0*u.rad, Nens = 100, 
                            lat_rot_sigma = 5*np.pi/180, lat_dev_sigma = 2*np.pi/180,
                            long_dev_sigma = 2*np.pi/180):
    """
    a function generate an ensemble of solar wind speed HUXt inputs from a 
    V map such as provided by PFSS, DUMFRIC, HelioMAS. The first ensemble 
    member is always the unperturbed value

    Parameters
    ----------
    vr_map : float array, dimensions (nlong, nlat)
         The solar wind speed map
    phi : Float array, dimensions (nlong, nlat)
        The Carrington longitude in radians
    theta : Float array, dimensions (nlong, nlat)
        The heliographic longitude in radians (from equator)
    reflat : float
        The Earth's latitude in radians (from equator)
    Nens : Integer
        The number of ensemble members to generate
    lat_rot_sigma : float
        The standard deviation of the Gaussain from which the rotational 
        perturbation is drawn. In radians. 
    lat_dev_sigma: float
        The standard deviation of the Gaussain from which the linear 
        latitudinal perturbation is drawn. In radians
    long_dev_sigma: float
        The standard deviation of the Gaussain from which the linear 
        longitudinal perturbation is drawn. In radians
        
    Returns
    -------
    vr_ensmeble : NP ARRAY, dimensions (Nens, nlong)
        Solar wind speed longitudinal series   
    

    """
    assert((reflat.value > -np.pi/2)  & (reflat.value < np.pi/2))
    assert(Nens > 0)
    
    vr_longs = phi[0, :] 
    
    lats_E = reflat *np.ones((len(vr_longs)))
    
    vr_E = interp2d(vr_longs, lats_E, vr_map, phi, theta)
    #br_E = interp2d(br_longs, lats_E, br_map, phi, theta)
    #vr_eq = interp2d(vr_longs, lats_E*0, vr_map, phi, theta)
    
    
    #rotation - latidunial amplitude - gaussian distribution
    lat_rots = np.random.normal(0.0, lat_rot_sigma, Nens)  
    #rotation - longitude of node of rotation -uniform distribution
    long_rots = np.random.random_sample(Nens)*2*np.pi
    #deviation - latitude - gaussian distribution
    lat_devs = np.random.normal(0.0, lat_dev_sigma, Nens)  
    #deviation - longitude - gaussian distribution
    long_devs = np.random.normal(0.0, lat_dev_sigma, Nens)  
    
    vr_ensemble = np.ones((Nens,len(vr_longs)))
    #br_ensemble = np.ones((Nens,len(br_longs)))
    #first ensemble member is the undeviated value
    vr_ensemble[0,:] = vr_E
    #br_ensemble[0,:] = br_E
    
    #for each set of random params, generate a V long series
    for i in range(1, Nens):
        this_lat = lats_E.value + lat_rots[i] * np.sin(vr_longs.value + long_rots[i]) +lat_devs[i]
        this_long = _zerototwopi_(vr_longs.value + long_devs[i])
        
        v = interp2d(this_long, this_lat, vr_map, phi, theta)
        vr_ensemble[i,:] = v
        
       # b = interp2d(this_long, this_lat, br_map, phi, theta)
       # br_ensemble[i,:] = b
    return vr_ensemble


# <codecell> Demo script
import sys
import os
import h5py
import scipy.signal
sys.path.append(os.path.abspath(os.environ['DBOX'] + 'python_repos\\HUXt\\code'))
import huxt_inputs as Hin
import huxt as H

#==============================================================================
reflat = 5*(np.pi/180)*u.rad # Earth lat
N=100 #number of ensemble members
#filepath = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\AnthonyYeates\\windbound_b_pf720_20130816.12.nc'; cr = 999
filepath = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\AnthonyYeates\\windbound_b20181105.12.nc'; cr=2210.3
savedir =  os.path.abspath(os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\MattLang\\HelioMASEnsembles_python')

#==============================================================================
#load the solar wind speed map
#==============================================================================
vr_map, vr_lats, vr_longs, br_map, br_lats, br_longs, phi, theta = Hin.get_PFSS_maps(filepath)

#plot the speed map
plt.figure()
plt.pcolor(vr_longs.value*180/np.pi, vr_lats.value*180/np.pi, vr_map.value)
plt.plot([0, 360],[1,1]*reflat*180/np.pi,'r')
plt.xlabel('Carrington Longitude')
plt.ylabel('Latitude')

#==============================================================================
#generate the input ensemble
#==============================================================================
vr_ensemble = generate_input_ensemble(phi, theta, vr_map, reflat = reflat, Nens = N)
    
#resample the ensemble to 128 longitude bins
vr128_ensemble = np.ones((N,128))    
for i in range(0, N):
    vr128_ensemble[i,:] = scipy.signal.resample(vr_ensemble[i,:],128)
    
#box plot of the input Vr ensemble
plt.figure()
plt.boxplot(vr128_ensemble)
plt.plot(vr128_ensemble[0,:])
plt.xlabel('Carrington Longitude')
plt.ylabel('V_{SW} [km/s]')

#==============================================================================
#save the ensemble, e,g for use with DA
#==============================================================================
h5f = h5py.File(savedir + '\\CR' + str(cr) +'_vin_ensemble.h5', 'w')
h5f.create_dataset('Vin_ensemble', data=vr128_ensemble)
h5f.close()


#==============================================================================
#run huxt with the input ensemble
#==============================================================================
huxtoutput = np.ones((N,1117))

os.chdir(os.environ['DBOX'] + 'python_repos\\HUXt\\code')
for i in range(0,N):
    model = H.HUXt(v_boundary=vr128_ensemble[i]* (u.km/u.s), simtime=27*u.day, 
                   dt_scale=4, cr_num= cr, lon_out=0.0*u.deg, r_max=215*u.solRad)
    model.solve([]) 
    
    #find Earth location and extract the time series
    #huxtoutput.append(HA.get_earth_timeseries(model))
    
    #it's quicker to just run to 215 rS and take the outer grid cell
    huxtoutput[i,:] = model.v_grid[:,-1,0]
    
    print('HUXt run ' + str(i+1) + ' of ' + str(N))

#box plot of the HUXt output
plt.figure()
plt.boxplot(huxtoutput)
plt.plot(huxtoutput[0,:])
plt.xlabel('Time')
plt.ylabel('V_{SW} [km/s]')
