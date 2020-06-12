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
import matplotlib.pyplot as plt
from astropy.time import Time
import heliopy
import sunpy

# <codecell> Get MAS data from MHDweb

cr=2010

def getMASboundaryconditions(cr):
    #get the HUXt boundary condition directory
    dirs = H._setup_dirs_()
    _boundary_dir_ = dirs['boundary_conditions'] 
      
    #example URL: http://www.predsci.com/data/runs/cr2010-medium/mdi_mas_mas_std_0101/helio/br_r0.hdf 
    observatories_order=['hmi','mdi','solis','gong','mwo','wso','kpo']
    runtype_order=['mas','mast','masp']
    runnumber_order=['0101','0201']
    
    heliomas_url_front='http://www.predsci.com/data/runs/cr'
    heliomas_url_end='_r0.hdf'
    
    vrfilename = 'HelioMAS_CR'+str(int(cr)) + '_vr'+heliomas_url_end
    brfilename = 'HelioMAS_CR'+str(int(cr)) + '_br'+heliomas_url_end
    
    if (os.path.exists(os.path.join( _boundary_dir_, brfilename)) == False & 
        os.path.exists(os.path.join( _boundary_dir_, vrfilename)) == False): #check if the files already exist
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
            
        #download teh vr and br files            
        print('Downloading from: ',urlbase)
        urllib.request.urlretrieve(urlbase+'br'+heliomas_url_end,
                           os.path.join(_boundary_dir_, brfilename) )    
        urllib.request.urlretrieve(urlbase+'vr'+heliomas_url_end,
                           os.path.join(_boundary_dir_, vrfilename) )  
    else:
         print('Files already exist for CR' + str(int(cr)))   


getMASboundaryconditions(cr)

# <codecell> Read the HDF files 

   
def readMASvrbr(cr):
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
    
    
    filepath=os.path.join(_boundary_dir_, brfilename)
    assert os.path.exists(filepath)
    file = SD(filepath, SDC.READ)
   
    sds_obj = file.select('fakeDim0') # select sds
    MAS_br_Xa = sds_obj.get() # get sds data
    sds_obj = file.select('fakeDim1') # select sds
    MAS_br_Xm = sds_obj.get() # get sds data
    sds_obj = file.select('Data-Set-2') # select sds
    MAS_br = sds_obj.get() # get sds data
    
    return MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm

MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_br, MAS_br_Xa, MAS_br_Xm = readMASvrbr(2010)


# <codecell> Extract the properties at Earth latitude

#create the time series


### Now get Earth's Carrington Longitude vs time and visualize
earthSpiceKernel = spice_data.get_kernel("planet_trajectories")
heliopy.spice.furnish(earthSpiceKernel)
earthTrajectory = heliopy.spice.Trajectory("Earth")
earthTrajectory.generate_positions(dt_hourly,'Sun','IAU_SUN')
earth = astropy.coordinates.SkyCoord(x=earthTrajectory.x,
                                     y=earthTrajectory.y,
                                     z=earthTrajectory.z,
                                     frame = sunpy.coordinates.frames.HeliographicCarrington,
                                     representation_type="cartesian"
                                     )
earth.representation_type="spherical"
