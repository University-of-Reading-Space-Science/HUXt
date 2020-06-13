# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:50:49 2020

@author: mathewjowens
"""
import httplib2
import urllib
import HUXt as H
import os
from pyhdf.SD import SD, SDC  
#import matplotlib.pyplot as plt
#from astropy.time import Time
#import heliopy
#import sunpy
import numpy as np
import astropy.units as u
#import astropy
#from heliopy.data import psp as psp_data, spice as spice_data

# <codecell> Get MAS data from MHDweb



def getMASboundaryconditions(cr=np.NaN, observatory='', runtype='', runnumber=''):
    """
    A function to grab the  Vr and Br boundary conditions from MHDweb. An order
    of preference for observatories is given in the function. Checks first if
    the data already exists in the HUXt boundary condition folder

    Parameters
    ----------
    cr : INT
        Carrington rotation number 
    observatory : STRING
        Name of preferred observatory (e.g., 'hmi','mdi','solis',
        'gong','mwo','wso','kpo'). Empty if no preference and automatically selected 
    runtype : STRING
        Name of preferred MAS run type (e.g., 'mas','mast','masp').
        Empty if no preference and automatically selected 
    runnumber : STRING
        Name of preferred MAS run number (e.g., '0101','0201').
        Empty if no preference and automatically selected    

    Returns
    -------
    flag : INT
        1 = successful download. 0 = files exist, -1 = no file found.

    """
    
    assert(np.isnan(cr)==False)
    
    #the order of preference for different MAS run results
    overwrite=False
    if not observatory:
        observatories_order=['hmi','mdi','solis','gong','mwo','wso','kpo']
    else:
        observatories_order=[str(observatory)]
        overwrite=True #if the user wants a specific observatory, overwrite what's already downloaded
        
    if not runtype:
        runtype_order=['mas','mast','masp']
    else:
        runtype_order=[str(runtype)]
        overwrite=True
    
    if not runnumber:
        runnumber_order=['0101','0201']
    else:
        runnumber_order=[str(runnumber)]
        overwrite=True
    
    #get the HUXt boundary condition directory
    dirs = H._setup_dirs_()
    _boundary_dir_ = dirs['boundary_conditions'] 
      
    #example URL: http://www.predsci.com/data/runs/cr2010-medium/mdi_mas_mas_std_0101/helio/br_r0.hdf 
    heliomas_url_front='http://www.predsci.com/data/runs/cr'
    heliomas_url_end='_r0.hdf'
    
    vrfilename = 'HelioMAS_CR'+str(int(cr)) + '_vr'+heliomas_url_end
    brfilename = 'HelioMAS_CR'+str(int(cr)) + '_br'+heliomas_url_end
    
    if (os.path.exists(os.path.join( _boundary_dir_, brfilename)) == False or 
        os.path.exists(os.path.join( _boundary_dir_, vrfilename)) == False or
        overwrite==True): #check if the files already exist
        #Search MHDweb for a HelioMAS run, in order of preference 
        h = httplib2.Http()
        foundfile=False
        for masob in observatories_order:
            for masrun in runtype_order:
                for masnum in runnumber_order:
                    urlbase=(heliomas_url_front + str(int(cr)) + '-medium/' + masob +'_' +
                         masrun + '_mas_std_' + masnum + '/helio/')
                    url=urlbase + 'br' + heliomas_url_end
                    #print(url)
                    
                    #see if this br file exists
                    resp = h.request(url, 'HEAD')
                    if int(resp[0]['status']) < 400:
                        foundfile=True
                        #print(url)
                    
                    #exit all the loops - clumsy, but works
                    if foundfile: 
                        break
                if foundfile:
                    break
            if foundfile:
                break
            
        if foundfile==False:
            print('No data available for given CR and observatory preferences')
            return -1
        
        #download teh vr and br files            
        print('Downloading from: ',urlbase)
        urllib.request.urlretrieve(urlbase+'br'+heliomas_url_end,
                           os.path.join(_boundary_dir_, brfilename) )    
        urllib.request.urlretrieve(urlbase+'vr'+heliomas_url_end,
                           os.path.join(_boundary_dir_, vrfilename) )  
        
        return 1
    else:
         print('Files already exist for CR' + str(int(cr)))   
         return 0


   
def readMASvrbr(cr):
    """
    A function to read in the MAS coundary conditions for a given CR

    Parameters
    ----------
    cr : INT
        Carrington rotation number

    Returns
    -------
    MAS_vr : NP ARRAY (NDIM = 2)
        Solar wind speed at 30rS, in km/s
    MAS_vr_Xa : NP ARRAY (NDIM = 1)
        Carrington longitude of Vr map, in rad
    MAS_vr_Xm : NP ARRAY (NDIM = 1)
        Latitude of Vr as angle down from N pole, in rad
    MAS_br : NP ARRAY (NDIM = 2)
        Radial magnetic field at 30rS, in model units
    MAS_br_Xa : NP ARRAY (NDIM = 1)
        Carrington longitude of Br map, in rad
    MAS_br_Xm : NP ARRAY (NDIM = 1)
       Latitude of Br as angle down from N pole, in rad

    """
    #get the boundary condition directory
    dirs = H._setup_dirs_()
    _boundary_dir_ = dirs['boundary_conditions'] 
    #create the filenames 
    heliomas_url_end='_r0.hdf'
    vrfilename = 'HelioMAS_CR'+str(int(cr)) + '_vr'+heliomas_url_end
    brfilename = 'HelioMAS_CR'+str(int(cr)) + '_br'+heliomas_url_end

    filepath=os.path.join(_boundary_dir_, vrfilename)
    assert os.path.exists(filepath)
    #print(os.path.exists(filepath))

    file = SD(filepath, SDC.READ)
    # print(file.info())
    # datasets_dic = file.datasets()
    # for idx,sds in enumerate(datasets_dic.keys()):
    #     print(idx,sds)
        
    sds_obj = file.select('fakeDim0') # select sds
    MAS_vr_Xa = sds_obj.get() # get sds data
    sds_obj = file.select('fakeDim1') # select sds
    MAS_vr_Xm = sds_obj.get() # get sds data
    sds_obj = file.select('Data-Set-2') # select sds
    MAS_vr = sds_obj.get() # get sds data
    
    #convert from model to physicsal units
    MAS_vr = MAS_vr*481.0 * u.km/u.s
    MAS_vr_Xa=MAS_vr_Xa * u.rad
    MAS_vr_Xm=MAS_vr_Xm * u.rad
    
    
    filepath=os.path.join(_boundary_dir_, brfilename)
    assert os.path.exists(filepath)
    file = SD(filepath, SDC.READ)
   
    sds_obj = file.select('fakeDim0') # select sds
    MAS_br_Xa = sds_obj.get() # get sds data
    sds_obj = file.select('fakeDim1') # select sds
    MAS_br_Xm = sds_obj.get() # get sds data
    sds_obj = file.select('Data-Set-2') # select sds
    MAS_br = sds_obj.get() # get sds data
    
    MAS_br_Xa=MAS_br_Xa * u.rad
    MAS_br_Xm=MAS_br_Xm * u.rad
    
    return MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm


def get_MAS_equatorial_profiles(cr):
    """
    a function to download, read and process MAS output to provide HUXt boundary
    conditions at the helioequator

    Parameters
    ----------
    cr : INT
        Carrington rotation number

    Returns
    -------
    vr_in : NP ARRAY (NDIM = 1)
        Solar wind speed as a function of Carrington longitude at solar equator.
        Interpolated to HUXt longitudinal resolution. In km/s
    br_in : NP ARRAY(NDIM = 1)
        Radial magnetic field as a function of Carrington longitude at solar equator.
        Interpolated to HUXt longitudinal resolution. Dimensionless

    """
    
    assert(np.isnan(cr)==False and cr>0)
    
    #check the data exist, if not, download them
    getMASboundaryconditions(cr)    #getMASboundaryconditions(cr,observatory='mdi')
    
    #read the HelioMAS data
    MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = readMASvrbr(cr)
    
    #extract the value at the helioequator
    vr_eq=np.ones(len(MAS_vr_Xa))
    for i in range(0,len(MAS_vr_Xa)):
        vr_eq[i]=np.interp(np.pi/2,MAS_vr_Xm.value,MAS_vr[i][:].value)
    
    br_eq=np.ones(len(MAS_br_Xa))
    for i in range(0,len(MAS_br_Xa)):
        br_eq[i]=np.interp(np.pi/2,MAS_br_Xm.value,MAS_br[i][:])
        
    #now interpolate on to the HUXt longitudinal grid
    nlong=H.huxt_constants()['nlong']
    dphi=2*np.pi/nlong
    longs=np.linspace(dphi/2 , 2*np.pi -dphi/2,nlong)
    vr_in=np.interp(longs,MAS_vr_Xa.value,vr_eq)*u.km/u.s
    br_in=np.interp(longs,MAS_br_Xa.value,br_eq)

    #convert br into +/- 1
    #br_in[br_in>=0.0]=1.0*u.dimensionless_unscaled
    #br_in[br_in<0.0]=-1.0*u.dimensionless_unscaled
    
    return vr_in, br_in

# <codecell> Map MAS inputs to smaller radial distances, for starting HUXt below 30 rS

@u.quantity_input(v_outer=u.km / u.s)
@u.quantity_input(r_outer=u.solRad)
@u.quantity_input(lon_outer=u.rad)
@u.quantity_input(r_inner=u.solRad)
def map_v_inwards(v_outer, r_outer, lon_outer, r_inner):
    """
    Function to map v from r_outer (in rs) to r_inner (in rs)
    :param v_outer: Solar wind speed at outer radial distance. Units of km/s.
    :param r_outer: Radial distance at outer radial distance. Units of km.  
    :param lon_outer: Carrington longitude at outer distance. Units of rad
    :param r_inner: Radial distance at inner radial distance. Units of km.
    :return v_inner: Solar wind speed mapped from r_outer to r_inner. Units of km/s.
    :return lon_inner: Carrington longitude at r_inner. Units of rad.
    """

    if r_outer < r_inner:
        raise ValueError("Warning: r_outer < r_inner. Mapping will not work.")

    # get the acceleration parameters
    constants = H.huxt_constants()
    alpha = constants['alpha']  # Scale parameter for residual SW acceleration
    rH = constants['r_accel'].to(u.kilometer).value  # Spatial scale parameter for residual SW acceleration
    Tsyn = constants['synodic_period'].to(u.s).value
    r_outer = r_outer.to(u.km).value
    r_inner = r_inner.to(u.km).value

    # compute the speed at the new inner boundary height (using Vacc term, equation 5 in the paper)
    v0 = v_outer.value / (1 + alpha * (1 - np.exp((r_inner - r_outer) / rH)))

    # compute the transit time from the new to old inner boundary heights (i.e., integrate equations 3 and 4 wrt to r)
    A = v0 + alpha * v0
    term1 = rH * np.log(A * np.exp(r_outer / rH) - 
                      alpha * v0 * np.exp(r_inner / rH)) / A
    term2 = rH * np.log(A * np.exp(r_inner / rH) - 
                      alpha * v0 * np.exp(r_inner / rH)) / A                      
    T_integral = term1 - term2

    # work out the longitudinal shift
    phi_new = H._zerototwopi_(lon_outer.value + (T_integral / Tsyn) * 2 * np.pi)

    return v0*u.km/u.s, phi_new*u.rad


@u.quantity_input(v_outer=u.km / u.s)
@u.quantity_input(r_outer=u.solRad)
@u.quantity_input(r_inner=u.solRad)
def map_v_boundary_inwards(v_outer, r_outer, r_inner):
    """
    Function to map a longitudinal V series from r_outer (in rs) to r_inner (in rs)
    :param v_outer: Solar wind speed at outer radial boundary. Units of km/s.
    :param r_outer: Radial distance at outer radial boundary. Units of km.
    :param r_inner: Radial distance at inner radial boundary. Units of km.
    :return v_inner: Solar wind speed mapped from r_outer to r_inner. Units of km/s.
    """

    if r_outer < r_inner:
        raise ValueError("Warning: r_outer < r_inner. Mapping will not work.")

    # compute the longitude grid from the length of the vouter input variable
    lon, dlon, nlon = H.longitude_grid()   
    #map each point in to a new speed and longitude
    v0, phis_new = map_v_inwards(v_outer, r_outer, lon, r_inner)

    #interpolate the mapped speeds back onto the regular Carr long grid,
    #making boundaries periodic 
    v_inner = np.interp(lon, phis_new, v0, period=2*np.pi) 

    return v_inner

@u.quantity_input(v_outer=u.km / u.s)
@u.quantity_input(r_outer=u.solRad)
@u.quantity_input(r_inner=u.solRad)
def map_ptracer_boundary_inwards(v_outer, r_outer, r_inner, ptracer_outer):
    """
    Function to map a longitudinal V series from r_outer (in rs) to r_inner (in rs)
    :param v_outer: Solar wind speed at outer radial boundary. Units of km/s.
    :param r_outer: Radial distance at outer radial boundary. Units of km.
    :param r_inner: Radial distance at inner radial boundary. Units of km.
    :param p_tracer_outer:  Passive tracer at outer radial boundary. 
    :return ptracer_inner: Passive tracer mapped from r_outer to r_inner. 
    """

    if r_outer < r_inner:
        raise ValueError("Warning: r_outer < r_inner. Mapping will not work.")

    # compute the longitude grid from the length of the vouter input variable
    lon, dlon, nlon = H.longitude_grid()   
    #map each point in to a new speed and longitude
    v0, phis_new = map_v_inwards(v_outer, r_outer, lon, r_inner)

    #interpolate the mapped speeds back onto the regular Carr long grid,
    #making boundaries periodic 
    ptracer_inner = np.interp(lon, phis_new, ptracer_outer, period=2*np.pi) 

    return ptracer_inner

