# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:29:37 2020

#Examples of HUXt runs and plots

@author: mathewjowens
"""

import numpy as np
import HUXt as H
import astropy.units as u
import matplotlib.pyplot as plt
import time
import os

#change to the HUXt dir so that the config.dat is found
os.chdir(os.path.abspath(os.environ['DBOX'] + 'python_repos\\HUXt\\code'))





# <codecell> HUXt1D, no CME
#==============================================================================
#Run HUXt1D with user specified boundary conditions and no CMEs
#==============================================================================


#Form longitudinal boundary conditions - background wind of 400 km/s with two fast streams.
v_boundary = np.ones(128) * 400 * (u.km/u.s)
v_boundary[30:50] = 600 * (u.km/u.s)
v_boundary[95:125] = 700 * (u.km/u.s)

# This boundary condition looks like
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(v_boundary,'k-')
ax.set_xlabel('Longitude bin')
ax.set_ylabel('Input Wind Speed (km/s)')
ax.set_xlim(0,128)

# Setup HUXt to do a 5 day simulation, with model output every 4 timesteps (roughly half and hour time step), looking at 0 longitude
model = H.HUXt(v_boundary=v_boundary, lon_out=0.0*u.deg, simtime=5*u.day, dt_scale=4)

# Solve these conditions, with no ConeCMEs added.
cme_list = []

#model.generate_v_boundary_ts(cme_list)
model.solve(cme_list)

# Plot the radial profile of the ambient wind profile at a fixed time (in days). 
t = 1.5*u.day
model.plot_radial(t, lon=0.0,field='v')

# Plot the time series of the ambient wind profile at a fixed radius. 
r = 1.0*u.AU
model.plot_timeseries(r,lon=0.0,field='v')

#now post-process the results to mimic compression at stream interactions and the 1/R^2 fall off.
model.rho_post_process()
model.plot_radial(t, lon=0.0*u.deg,field='rho')
model.plot_timeseries(r, lon=0.0*u.deg,field='rho')

# #test load/save
# out_path = model.save(tag='oneD_noCME')
# model2, cme_list2 = H.load_HUXt_run(out_path)

# # Plot the radial profile of the ambient wind profile at a fixed time (in days). 
# t = 1.5*u.day
# model2.plot_radial(t, lon=0.0,field='ambient')

# # Plot the time series of the ambient wind profile at a fixed radius. 
# r = 1.0*u.AU
# model2.plot_timeseries(r,lon=0.0,field='ambient')






# <codecell> HUXt1D, with CME
#==============================================================================
#Run HUXt1D with user specified boundary conditions and single CME
#==============================================================================

# Now lets run HUXt1D with the same background ambient wind and a cone cme.
# Launch the CME half a day after the simulation, at 0 longitude, 30 degree width, speed 850km/s and thickness=5 solar radii
cme = H.ConeCME(t_launch=0.5*u.day, longitude=0.0*u.deg, width=30*u.deg, v=1000*(u.km/u.s), thickness=10*u.solRad)
cme_list = [cme]

# Setup HUXt to do a 5 day simulation, with model output every 4 timesteps (roughly half and hour time step), looking at 0 longitude
model = H.HUXt(v_boundary=v_boundary, lon_out=0.0*u.deg, simtime=5*u.day, dt_scale=1)

# Run the model, and this time save the results to file.
#model.solve(cme_list, save=True, tag='1d_conecme_test')
model.solve(cme_list)

#test load/save
#out_path = model.save(tag='oneD_CME')
#model, cme_list = H.load_HUXt_run(out_path)


# Plot the radial profile and time series of both the ambient and ConeCME solutions at a fixed time (in days). 
# Save both to file as well. These are saved in HUXt>figures>HUXt1D
t = 1*u.day
model.plot_radial(t, lon=0.0*u.deg, field='v')
#model.plot_radial(t, lon=0.0*u.deg, field='cme')
#model.plot_radial(t, field='both', save=True, tag='1d_cone_test_radial')

#r = 1.0*u.AU
#model.plot_timeseries(r, lon=0.0*u.deg,field='v', tag='1d_cone_test_radial')

#plot the CME front
cmecoords=model.cmes[0].coords
rmax=np.ones(len(model.time_out))
for t in range(0,model.nt_out):
    rcme_t=cmecoords[t]['r'].to(u.km).value
    if len(rcme_t)>0:
        rmax[t]=rcme_t[-1][0]
    else:
        rmax[t]=np.nan
    
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(model.time_out.to(u.day),rmax)
ax.set_xlabel('Time [days]')
ax.set_ylabel('CME front r [km]')

vcme=(rmax[1:]-rmax[:-1])/model.dt_out

from scipy.ndimage import gaussian_filter1d

vsmooth = gaussian_filter1d(vcme, 10)

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(model.time_out[1:].to(u.day),vsmooth)
ax.set_xlabel('Time [days]')
ax.set_ylabel('CME front speed [km/s]')
# <codecell> HUXt1D, with CME. Trace CME front as a funciton of time

cme = H.ConeCME(t_launch=0.5*u.day, longitude=0.0*u.deg, width=30*u.deg, v=1000*(u.km/u.s), thickness=5*u.solRad)
cme_list = [cme]

v_boundary=450*np.ones(128) * u.km/u.s
# Setup HUXt to do a 5 day simulation, with model output every 4 timesteps (roughly half and hour time step), looking at 0 longitude
model = H.HUXt(v_boundary=v_boundary, lon_out=0.0*u.deg, simtime=5*u.day, dt_scale=4)
model.solve(cme_list)

model.plot_radial(1.0*u.day, lon=0.0,field='cme')
model.plot_radial(1.5*u.day, lon=0.0,field='cme')
model.plot_radial(2.0*u.day, lon=0.0,field='cme')

# CME_r=np.ones(model.time_out.size)
# for time, t in model.time_out:
    




# <codecell> HUXt1D, HelioMAS input
#==============================================================================
#Run HUXt1D with HelioMAS input
#==============================================================================
#HUXt1D and HUXt2D can both be initiated with archived output from HelioMAS

model = H.HUXt(cr_num=2124, cr_lon_init=0*u.rad, lon_out=0.0*u.deg, simtime=5*u.day, dt_scale=4)

# Solve these conditions with a ConeCME.
cme = H.ConeCME(t_launch=0.5*u.day, longitude=0.0*u.deg, width=30*u.deg, v=1000*(u.km/u.s), thickness=5*u.solRad)
cme_list = [cme]
#start=time.time(); model.solve(cme_list); end = time.time()
#print("Standard solve = %s" % (end - start))

model.solve(cme_list)
start=time.time(); model.solve(cme_list); end = time.time()
print("Fast solve (after compilation) = %s" % (end - start)) 

# Plot the time series of the ambient wind profile at a fixed radius. 
r = 1.0*u.AU
model.plot_timeseries(r, lon=0.0*u.deg,field='v')

#test load/save
out_path = model.save(tag='oneD_CME_MAS')
model, cme_list = H.load_HUXt_run(out_path)

# Plot the radial profile of the ambient wind profile at a fixed time (in days). 
t = 1.5*u.day
model.plot_radial(t, lon=0.0*u.deg,field='v')

# Plot the time series of the ambient wind profile at a fixed radius. 
r = 1.0*u.AU
model.plot_timeseries(r, lon=0.0*u.deg,field='v')




# <codecell> HUXt2D, with CME
#==============================================================================
#Run HUXt2D with user specified boundary conditions and single CME
#==============================================================================

# HUXt2D runs in a similar manner. 

#Form boundary conditions - background wind of 400 km/s with two fast streams.
vr_in = np.ones(128) * 400 * (u.km/u.s)
#v_boundary[30:50] = 600 * (u.km/u.s)
#v_boundary[95:125] = 700 * (u.km/u.s)
#add a CME
cme = H.ConeCME(t_launch=0.5*u.day, longitude=0.0*u.deg, width=40*u.deg, v=1000*(u.km/u.s), thickness=5*u.solRad)
cme_list = [cme]

# Setup HUXt to do a 5 day simulation, with model output every 4 timesteps (roughly half and hour time step), looking at 0 longitude

model = H.HUXt(v_boundary=vr_in, simtime=4*u.day, 
               cr_lon_init =  70*u.deg,
               lon_start= - np.pi/2 * u.rad , lon_stop = np.pi/2 * u.rad,
               dt_scale=1)
#model.solve([])
model.solve(cme_list) # This takes a minute or so to run.

#test load/save
#out_path = model.save(tag='twoD_CME')
#model, cme_list = H.load_HUXt_run(out_path)

t_interest = 3.5*u.day
model.plot(t_interest, field='v')
#model.plot(t_interest, field='br')
#model.plot(t_interest, field='cme')

model.animate('v', tag='pancaking_CR2209')

# <codecell> HUXt2D, animate output
#==============================================================================
#Animate a MP4 of the CME solution. These are saved in HUXt>figures>HUXt2D
#==============================================================================
model.animate('cme', tag='cone_cme_test') # This takes about a minute too.



# <codecell> HUXt2D, trace CME boundary
#==============================================================================
#Output the coordinates of the tracked CME boundary 
#==============================================================================
cme = model.cmes[0]
timestep = 50
rad = cme.coords[timestep]['r']
lon = cme.coords[timestep]['lon']
x = rad * np.cos(lon)
y = rad * np.sin(lon)

fig, ax = plt.subplots() # compare this with the boundary in the frame above.
ax.plot(x,y,'r.')
ax.set_xlabel('X ($R_{sun}$)')
ax.set_ylabel('Y ($R_{sun}$)')

# <codecell> HUXt2D, multiple CMEs
#==============================================================================
#Both HUXt1D and HUXt2D can be run with multiple ConeCMEs.
#==============================================================================
# Setup HUXt to do a 5 day simulation, with model output every 4 timesteps (roughly half and hour time step), looking at 0 longitude
cr=2209
vr_in, br_in = H.Hin.get_MAS_long_profile(cr)

model = H.HUXt(v_boundary=vr_in, cr_num=cr, br_boundary=br_in,
               simtime=5*u.day, cr_lon_init=60*u.deg, dt_scale=4)

daysec = 86400
times = [0.5*daysec, 1.5*daysec, 3*daysec]
speeds = [850, 1000, 700]
lons = [0, 90, 300]
widths = [30, 40, 20]
thickness = [15, 4, 2]
cme_list = []
for t, l, w, v, thick in zip(times, lons, widths, speeds, thickness):
    cme = H.ConeCME(t_launch=t*u.s, longitude=l*u.deg, width=w*u.deg, v=v*model.kms, thickness=thick*u.solRad)
    cme_list.append(cme)

start=time.time(); 
model.solve(cme_list) # This takes a minute or so to run.
end = time.time()
print("Fast solve (after compilation) = %s" % (end - start)) 


t_interest=4*u.day
model.plot(t_interest, field='v')
#model.plot(t_interest, field='br')
#model.plot(t_interest, field='rho')
#model.plot(t_interest, field='cme')

#model.animate('v', tag='testparticle')

# <codecell> Load and save

#save just the speed fields
out_path = model.save(tag='cone_cme_test')

# And loaded back in with
model2, cme_list2 = H.load_HUXt_run(out_path)
t_interest=4.5*u.day
model2.plot(t_interest, field='ambient')
model2.plot(t_interest, field='cme')
model2.plot(t_interest, field='br_cme')
model2.plot(t_interest, field='br_ambient')


#save the speed and Br fields
out_path = model.save_all(tag='cone_cme_test')

# And loaded back in with
model2, cme_list2 = H.load_HUXt_run(out_path)
t_interest=4.5*u.day
model2.plot(t_interest, field='ambient')
model2.plot(t_interest, field='cme')
model2.plot(t_interest, field='br_cme')
model2.plot(t_interest, field='br_ambient')

# #test the load/save function 
# out_path = model.save(tag='multi_cones')
# model2, cme_list2 = H.load_HUXt_run(out_path)

# t_interest=4.5*u.day
# model2.plot(t_interest, field='ambient')
# model2.plot(t_interest, field='cme')
# model2.plot(t_interest, field='ptracer_cme')
# model2.plot(t_interest, field='ptracer_ambient')

#model.animate('cme', tag='multi_cones')

# <codecell> Planetary and spacecraft positions
#==============================================================================
#There is also an ephemeris of the HEEQ and Carrington coordiates of Earth, Mercury, Venus, STEREO-A and STEREO-B
#==============================================================================

# These are automatically plotted on model solutions derived from a particular Carrington rotation.
model.plot(model.time_out[0])

# You can retrieve a bodies position at each model timestep like:
earth = model.get_observer('earth')
sta = model.get_observer('sta')
venus = model.get_observer('venus')

# The bodies HEEQ and Carrington coordinates are attributes.
help(earth)

# So to plot them:
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.plot(earth.lon, earth.r, 'co')
ax.plot(sta.lon, sta.r, 'rs')
ax.plot(venus.lon, venus.r, 'mo')

# <codecell> HUXt help

# Most of the model parameters are attributes of the HUXt1D(2D) classes, so can be accessed like:
# Model time step from the CFL condition:
print(model.dt)
# Output model time step
print(model.dt_out)
# A list of all attributes is in the documentation
help(model)



# <codecell> Get the MAS equatorial profiles and run HUXt (from the MAS boundary of 30rS)

#get the HUXt inputs
cr=2210
vr_in, br_in = H.Hin.get_MAS_long_profile(cr)
#now run HUXt
model = H.HUXt(v_boundary=vr_in, cr_num=cr, br_boundary=br_in, latitude=10*u.deg,
               simtime=5*u.day, dt_scale=4)
cme = H.ConeCME(t_launch=0.5*u.day, longitude=0.0*u.deg, latitude=0.0*u.deg,
                width=30*u.deg, v=1000*(u.km/u.s), thickness=5*u.solRad)
model.solve([cme]) 

t_interest=3*u.day
model.plot(t_interest, field='v')
model.plot(t_interest, field='cme')
model.plot(t_interest, field='br_cme')
model.plot(t_interest, field='br_ambient')


# <codecell> Run from HelioMAS with user-defined rho from constant mass flux

#get the HUXt inputs
cr=1833
vr_in, br_in = H.Hin.get_MAS_long_profile(cr)

#create a density map assuming constant mass flux
n_in=vr_in.value*0.0 + 4.0
n_in=n_in + ((650 - vr_in.value) /400)*5.0

#now run HUXt
model = H.HUXt(v_boundary=vr_in, cr_num=cr, br_boundary=br_in, rho_boundary=n_in,
               latitude=0*u.deg, simtime=2*u.day, dt_scale=4)
cme = H.ConeCME(t_launch=0.5*u.day, longitude=0.0*u.deg, latitude=0.0*u.deg,
                width=60*u.deg, v=1500*(u.km/u.s), thickness=5*u.solRad)
cme=[]
model.solve([cme]) 

t = 0*u.day
#model.plot_radial(t, lon=200.0*u.deg,field='br_ambient')
#model.plot_radial(t, lon=200.0*u.deg,field='ambient')
model.plot(t, field='v')
model.plot(t, field='br')

model.rho_post_process()
model.plot(t, field='rho')

# #now loop through each time step and change the density based upon the speed gradient
# for nt in range(0,model.nt_out):
#     for nlong in range(0,model.nlong):
#         vup=model.v_grid_amb[nt,1:,nlong]
#         vdown=model.v_grid_amb[nt,:-1,nlong]
#         vgrad=(vdown-vup).value
#         #modify the density according to the speed gradient
#         nr=model.br_grid_amb[nt,:,nlong]
#         nr[1:]=nr[1:] + nr[1:] * vgrad/10
#         nr[nr<1.0]=1.0
#         #introduce the 1/r^2 factor
#         nr=nr/(model.r*model.r)
#         model.br_grid_amb[nt,:,nlong]=nr.value
        
#         vup=model.v_grid_cme[nt,1:,nlong]
#         vdown=model.v_grid_cme[nt,:-1,nlong]
#         vgrad=(vdown-vup).value
#         #modify the density according to the speed gradient
#         nr=model.br_grid_cme[nt,:,nlong]
#         nr[1:]=nr[1:] + nr[1:] * vgrad/10
#         nr[nr<1.0]=1.0
#         #introduce the 1/r^2 factor
#         nr=nr/(model.r*model.r)
#         model.br_grid_cme[nt,:,nlong]=nr.value
        

# #model.plot_radial(t, lon=200.0*u.deg,field='br_ambient')
# model.plot(t, field='br_cme')
# <codecell> Get the MAS equatorial profiles and run HUXt from 5 rS

#get the HUXt inputs
cr=2054
lat=10*u.deg
vr_in, br_in = H.Hin.get_MAS_long_profile(cr,lat)

#map the MAS values inwards from 30 rS to 5 rS
vr_5rs=H.Hin.map_v_boundary_inwards(vr_in, 30*u.solRad, 5*u.solRad)
br_5rs=H.Hin.map_ptracer_boundary_inwards(vr_in, 30*u.solRad, 5*u.solRad,br_in)

#now run HUXt
model = H.HUXt(v_boundary=vr_5rs, cr_num=cr, br_boundary=br_5rs,simtime=5*u.day, 
               latitude=lat, dt_scale=4, r_min=5*u.solRad)
cme = H.ConeCME(t_launch=0.5*u.day, longitude=0.0*u.deg, latitude=0.0*u.deg,
                width=60*u.deg, v=1000*(u.km/u.s), thickness=5*u.solRad)
model.solve([cme]) 

t_interest=3*u.day
model.plot(t_interest, field='v')
model.plot(t_interest, field='br')
model.plot(t_interest, field='rho')




# <codecell> Meridional cut of HUXt

#create a list of latitudes at which to run HUXt
cr=2140
rin=10*u.solRad
vmap, vlats, vlongs, bmap, blats, blongs=H.Hin.get_MAS_maps(cr)

#map the speed inwards
vinnermap=H.Hin.map_vmap_inwards(v_map=vmap, v_map_lat=vlats, v_map_long=vlongs, 
                        r_outer=30*u.solRad, r_inner=rin)
#map brinwards
brinnermap=H.Hin.map_ptracer_map_inwards(v_map=vmap, v_map_lat=vlats, v_map_long=vlongs,
                            ptracer_map=bmap, ptracer_map_lat=blats, ptracer_map_long=blongs,
                            r_outer=30*u.solRad, r_inner=rin)

#set up the model
model=H.Hlat.HUXt3d(cr_num=cr,v_map=vinnermap, v_map_lat=vlats, v_map_long=vlongs,
                    br_map=brinnermap, br_map_lat=blats, br_map_long=blongs,
                    latitude_max=40*u.deg, latitude_min=-40*u.deg, lon_out=0.0*u.deg,
                    simtime=5*u.day, r_min=rin)

cme = H.ConeCME(t_launch=0.5*u.day, longitude=0.0*u.deg, latitude=0.0*u.deg,
                width=30*u.deg, v=1000*(u.km/u.s), thickness=5*u.solRad)

#run the model
cme_list=[]
model.solve(cme_list)

#plot and animate the output
model.plot(time=1*u.day, field='v')
#model.animate('cme', tag='mercut_cme_test') # This takes about a minute too.

# <codecell> Meridional cut of HUXt using PFSS.py input


#create a list of latitudes at which to run HUXt
cr=2140
filepath="D:\\Dropbox\\Papers_WIP\\_coauthor\\AnthonyYeates\\windbound_b_pf720_20130816.12.nc"
rin=21.5*u.solRad
vmap, vlats, vlongs, bmap, blats, blongs=H.Hin.get_PFSS_maps(filepath)

#set up the model
model=H.Hlat.HUXt3d(cr_num=cr,v_map=vmap, v_map_lat=vlats, v_map_long=vlongs,
                    br_map=bmap, br_map_lat=blats, br_map_long=blongs,
                    latitude_max=80*u.deg, latitude_min=-80*u.deg, lon_out=0.0*u.deg,
                    simtime=27*u.day, r_min=rin)

#run the model
cme_list=[]
model.solve(cme_list)

#plot and animate the output
model.plot(time=1*u.day, field='ambient')
model.animate('cme', tag='mercut_cme_test') # This takes about a minute too.



# <codecell> Extract the properties at Earth latitude - not currently used

#create the time series
nlong=H.huxt_constants()['nlong']
tstart=sunpy.coordinates.sun.carrington_rotation_time(cr)
tstop=sunpy.coordinates.sun.carrington_rotation_time(cr+1)
dt=(tstop-tstart)/nlong
t=tstart + dt/2 + (tstop-tstart-dt) * np.linspace(0.,1.0,nlong)

### Now get Earth's Carrington Longitude vs time and visualize
earthSpiceKernel = spice_data.get_kernel("planet_trajectories")
heliopy.spice.furnish(earthSpiceKernel)
earthTrajectory = heliopy.spice.Trajectory("Earth")
earthTrajectory.generate_positions(t,'Sun','IAU_SUN')
earth = astropy.coordinates.SkyCoord(x=earthTrajectory.x,
                                     y=earthTrajectory.y,
                                     z=earthTrajectory.z,
                                     frame = sunpy.coordinates.frames.HeliographicCarrington,
                                     representation_type="cartesian"
                                     )
earth.representation_type="spherical"


# <codecell> Test the HUXt inner boundary mapping
v_outer=np.ones(128)*400*u.km/u.s
r_outer=30*u.solRad
r_inner=10*u.solRad

v_inner=H.Hin.map_v_boundary_inwards(v_outer, r_outer, r_inner)

fig= plt.figure(figsize=(6, 10))
ax = fig.add_subplot(211)
ax.plot(v_outer)
ax.plot(v_inner)

ax = fig.add_subplot(212)


t=4*u.day
model = H.HUXt(v_boundary=v_inner, r_min=r_inner, lon_out=0.0*u.deg, simtime=5*u.day, dt_scale=4)
model.solve([])
#model.plot_radial(t, lon=0.0*u.deg, field='v')
ax.plot(model.r_grid,model.v_grid[-1,:,0])

model = H.HUXt(v_boundary=v_outer, r_min=r_outer, lon_out=0.0*u.deg, simtime=5*u.day, dt_scale=4)
model.solve([])
#model.plot_radial(t, lon=0.0*u.deg, field='v')
ax.plot(model.r_grid,model.v_grid[-1,:,0])



