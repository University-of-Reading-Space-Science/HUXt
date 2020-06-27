# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:02:28 2020

@author: mathewjowens
"""

import numpy as np
import HUXt as H
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
import os


class HUXt3d:
    """
    A class containing a list of HUXt classes, to enable mutliple latitudes to
    be simulated, plotted, animated, etc, together
    
    Attributes inherited from HUXt. Additional:
        lat: The list of latitudes of individual HUXt runs, in radians from equator
        nlat: The number of latitudes simulated
        HUXtlat: List of individual HUXt model classes at each latitude
        v_in: a list of Carrington longitude solar wind profiles at each simulated latitude
        br_in: a list of Carrington longitude Br profiles at each simulated latitude
        
    
    """

    # Decorators to check units on input arguments
    @u.quantity_input(v_map=(u.km / u.s))
    @u.quantity_input(v_map_lat=(u.rad))
    @u.quantity_input(v_map_long=(u.rad))
    @u.quantity_input(latitude_max=(u.deg))
    @u.quantity_input(latitude_min=(u.deg))
    @u.quantity_input(simtime=u.day)
    @u.quantity_input(cr_lon_init=u.deg)
    def __init__(self, v_map=np.NaN * (u.km / u.s), v_map_lat=np.NaN * u.rad, v_map_long=np.NaN * u.rad, 
                 br_map=np.NaN * u.dimensionless_unscaled, br_map_lat=np.NaN*u.rad, br_map_long=np.NaN * u.rad, 
                 cr_num=np.NaN, cr_lon_init=360.0 * u.deg, 
                 latitude_max = 30*u.deg, latitude_min=-30*u.deg,
                 r_min=30 * u.solRad, r_max=240 * u.solRad,
                 lon_out=np.NaN * u.rad, lon_start=np.NaN * u.rad, lon_stop=np.NaN * u.rad,
                 simtime=5.0 * u.day, dt_scale=1.0):
        """
        Initialise the HUXt instance.

        :param v_map: Inner solar wind speed boundary Carrington map. Must have units of km/s.
        :param v_map_lat: List of latitude positions for v_map, in radians
        :param v_map_long: List of Carrington longitudes for v_map, in radians
        :param br_map: Inner Br boundary Carrington map. Must have no units.
        :param br_map_lat: List of latitude positions for br_map, in radians
        :param br_map_long: List of Carrington longitudes for br_map, in radians
        :param latitude_max: Maximum helio latitude (from equator) of HUXt plane, in degrees
        :param latitude_min: Maximum helio latitude (from equator) of HUXt plane, in degrees
        :param cr_num: Integer Carrington rotation number. Used to determine the planetary and spacecraft positions
        :param cr_lon_init: Carrington longitude of Earth at model initialisation, in degrees.
        :param lon_out: A specific single longitude (relative to Earth_ to compute HUXt solution along, in degrees
        :param lon_start: The first longitude (in a clockwise sense) of the longitude range to solve HUXt over.
        :param lon_stop: The last longitude (in a clockwise sense) of the longitude range to solve HUXt over.
        :param r_min: The radial inner boundary distance of HUXt.
        :param r_max: The radial outer boundary distance of HUXt.
        :param simtime: Duration of the simulation window, in days.
        :param dt_scale: Integer scaling number to set the model output time step relative to the models CFL time.
        """
                 
        #latitude grid
        self.latitude_min=latitude_min.to(u.rad)
        self.latitude_max=latitude_max.to(u.rad)
        self.lat, self.nlat = latitude_grid(self.latitude_min,self.latitude_max)
        
        #check the dimensions 
        assert( len(v_map_lat) == len(v_map[1,:]) )
        assert( len(v_map_long) == len(v_map[:,1]) )
        
        if isnan(br_map):
            br_map_lat=v_map_lat
            br_map_long=v_map_long
            br_map=v_map.value*0.0
        else:
            assert( len(br_map_lat) == len(br_map[1,:]) )
            assert( len(br_map_long) == len(br_map[:,1]) )
        
        
        #get the HUXt longitunidal grid
        longs, dlon, nlon = H.longitude_grid(lon_start=0.0 * u.rad, lon_stop=2*np.pi * u.rad)
        #nlong=H.huxt_constants()['nlong']
        #dphi=2*np.pi/nlong
        #longs=np.linspace(dphi/2, 2*np.pi - dphi/2, nlong)
        
        #extract the vr value at the given latitudes
        self.v_in=[]
        vlong=np.ones(len(v_map_long)) 
        for thislat in self.lat:
            for ilong in range(0,len(v_map_long)):
                vlong[ilong]=np.interp(thislat.value, v_map_lat.value, v_map[ilong,:].value)
            #interpolate this longitudinal profile to the HUXt resolution
            self.v_in.append(np.interp(longs.value, v_map_long.value, vlong)*u.km/u.s)

        #extract the br value at the given latitudes
        self.br_in=[]
        blong=np.ones(len(br_map_long))
        for thislat in self.lat:
            for ilong in range(0,len(br_map_long)):
                blong[ilong]=np.interp(thislat.value, br_map_lat.value, br_map[ilong,:])
            #interpolate this longitudinal profile to the HUXt resolution
            self.br_in.append(np.interp(longs.value, br_map_long.value, blong)*u.dimensionless_unscaled) 
            
         
        #set up the model at each latitude
        self.HUXtlat=[]
        for i in range(0,self.nlat):
            self.HUXtlat.append(H.HUXt(v_boundary=self.v_in[i], br_boundary=self.br_in[i],
                                     latitude=self.lat[i],
                                     cr_num=cr_num, cr_lon_init=cr_lon_init, 
                                     r_min=r_min, r_max=r_max,
                                     lon_out=lon_out, lon_start=lon_start, lon_stop=lon_stop,
                                     simtime=simtime, dt_scale=dt_scale))
        return
    
    def solve(self, cme_list):
        for model in self.HUXtlat:
            model.solve(cme_list)
        
        return
    
    @u.quantity_input(time=u.day)
    def plot(self, time, field='cme', lon=np.NaN*u.deg):
        """
        Make a contour plot on polar axis of the solar wind solution at a specific time.
        :param time: Time to look up closet model time to (with an astropy.unit of time).
        :param long: the longitude at which to take the cut
        :param field: String, either 'cme', or 'ambient', specifying which solution to plot.        
        :return fig: Figure handle.
        :return ax: Axes handle.
        """
        
        if field not in ['cme', 'ambient','br_cme','br_ambient']:
            print("Error, field must be either 'cme', 'ambient','br_cme','br_ambient'. Default to CME")
            field = 'cme'
         
        #get the metadata from one of the individual HUXt elements
        model=self.HUXtlat[0]
        if (time < model.time_out.min()) | (time > (model.time_out.max())):
            print("Error, input time outside span of model times. Defaulting to closest time")

        id_t = np.argmin(np.abs(model.time_out - time))
        time_out = model.time_out[id_t].to(u.day).value
        
        #get the requested longitude
        if model.lon.size == 1:
            id_lon = 0
            lon_out = model.lon.value
        else:
            id_lon = np.argmin(np.abs(model.lon - lon))
            lon_out = model.lon[id_lon].to(u.deg).value
            
        #loop over latitudes and extract teh radial profiles
        mercut=np.ones((len(model.r),self.nlat))
        ymax=0.0
        for n in range(0,self.nlat):
            model=self.HUXtlat[n]
            if field== 'cme':
                ymin=200; ymax=810
                ylab='Solar Wind Speed (km/s)'
                label = 'CME Run'
                mercut[:,n]=model.v_grid_cme[id_t, :, id_lon]
                mymap = mpl.cm.viridis
            elif field == 'ambient':
                label = 'Ambient'
                ylab='Solar Wind Speed (km/s)'
                ymin=200; ymax=810
                mercut[:,n]=model.v_grid_amb[id_t, :, id_lon]
                mymap = mpl.cm.viridis
            elif field == 'br_cme':
                if np.all(np.isnan(model.br_grid_cme)):
                    return -1
                label = 'CME Run'
                ylab='Magnetic field polarity (code units)'
                mercut[:,n]=model.br_grid_cme[id_t, :, id_lon]
                brmax=np.absolute(model.br_grid_cme[id_t, :, id_lon]).max()
                if brmax>ymax:
                    ymax=brmax
                    ymin=-ymax
                mymap = mpl.cm.bwr
            elif field == 'br_ambient':
                if np.all(np.isnan(model.br_grid_amb)):
                    return -1
                label = 'Ambient'
                ylab='Magnetic field polarity (code units)'
                mercut[:,n]=model.br_grid_amb[id_t, :, id_lon]
                brmax=np.absolute(model.br_grid_amb[id_t, :, id_lon]).max()
                if brmax>ymax:
                    ymax=brmax
                    ymin=-ymax
                mymap = mpl.cm.bwr

        dv=((ymax-ymin)/100.0)
        mymap.set_over('lightgrey')
        mymap.set_under([0, 0, 0])
        levels = np.arange(ymin, ymax + dv, dv)
        
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
        cnt = ax.contourf(self.lat.to(u.rad), model.r, mercut, levels=levels, cmap=mymap, extend='both')
        
        ax.set_ylim(0, model.r.value.max())
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.patch.set_facecolor('slategrey')
        fig.subplots_adjust(left=0.05, bottom=0.16, right=0.95, top=0.99)

        # Add color bar
        pos = ax.get_position()
        dw = 0.005
        dh = 0.045
        left = pos.x0 + dw
        bottom = pos.y0 - dh
        wid = pos.width - 2 * dw
        cbaxes = fig.add_axes([left, bottom, wid, 0.03])
        cbar1 = fig.colorbar(cnt, cax=cbaxes, orientation='horizontal')
        cbar1.set_label(ylab)
        cbar1.set_ticks(np.arange(ymin, ymax, dv*20))

        # Add label
        label = "Time: {:3.2f} days".format(time_out)
        fig.text(0.675, pos.y0, label, fontsize=16)
        label = "HUXt2D"
        fig.text(0.175, pos.y0, label, fontsize=16)
  
            
        return fig, ax
    
    
    def animate(self, field, tag):
        """
        Animate the model solution, and save as an MP4.
        :param field: String, either 'cme', or 'ambient', specifying which solution to animate.
        :param tag: String to append to the filename of the animation.
        """

        if field not in ['cme', 'ambient','br_cme','br_ambient']:
            print("Error, field must be either 'cme', 'ambient','br_cme','br_amb'. Default to CME")
            field = 'cme'

        # Set the duration of the movie
        # Scaled so a 5 day simulation with dt_scale=4 is a 10 second movie.
        model=self.HUXtlat[0]
        duration = model.simtime.value * (10 / 432000)

        def make_frame(t):
            """
            Produce the frame required by MoviePy.VideoClip.
            :param t: time through the movie
            """
            # Get the time index closest to this fraction of movie duration
            i = np.int32((model.nt_out - 1) * t / duration)
            fig, ax = self.plot(model.time_out[i], field)
            frame = mplfig_to_npimage(fig)
            plt.close('all')
            return frame

        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_movie.mp4".format(cr_num, tag)
        filepath = os.path.join(model._figure_dir_, filename)
        animation = mpy.VideoClip(make_frame, duration=duration)
        animation.write_videofile(filepath, fps=24, codec='libx264')
        return
        
@u.quantity_input(latitude_min=u.rad)
@u.quantity_input(latitude_max=u.rad)
def latitude_grid(latitude_min = np.nan, latitude_max = np.nan):
    """
    Define the latitude grid of the HUXt model. This is constant in sine latitude
    

    :param latitude_min: The maximum latitude above the equator, in radians
    :param latitude_max: The minimum latitude below the equator, in radians
    
    return lat: List of latitude positions between given limits, in radians
    return lat: number of latitude positions between given limits
    """
    # Check the inputs.
    assert(latitude_max > latitude_min)
    assert(np.absolute(latitude_max) <= (np.pi/2)*u.rad)
    assert(np.absolute(latitude_min) <= (np.pi/2)*u.rad)

    # Form the full longitude grid.
    nlat = H.huxt_constants()['nlat'] 
    
    # dlat = np.pi / nlat
    # lat_min_full = - np.pi/2 + dlat / 2.0
    # lat_max_full =  np.pi/2 - dlat / 2.0
    # lat, dlat = np.linspace(lat_min_full, lat_max_full, nlat, retstep=True)
    # lat = lat * u.rad
    # dlat = dlat * u.rad
    
    dsinlat = 2 / nlat
    sinlat_min_full = - 1 + dsinlat / 2.0
    sinlat_max_full =  1 - dsinlat / 2.0
    sinlat, dsinlat = np.linspace(sinlat_min_full, sinlat_max_full, nlat, retstep=True)
    lat = np.arcsin(sinlat) * u.rad
    
    # Now get only the selected range of latitudes
    id_match = (lat >= latitude_min) & (lat <= latitude_max)
    lat = lat[id_match]
    nlat = lat.size

    return lat, nlat