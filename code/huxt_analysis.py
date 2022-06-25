import os

import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib as mpl
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
import pandas as pd
from numba import jit
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries

import huxt as H

mpl.rc("axes", labelsize=16)
mpl.rc("ytick", labelsize=16)
mpl.rc("xtick", labelsize=16)
mpl.rc("legend", fontsize=16)


@u.quantity_input(time=u.day)
def plot(model, time, save=False, tag='', fighandle=np.nan, axhandle=np.nan,
         minimalplot=False, streaklines = [], plotHCS=True):
    """
    Make a contour plot on polar axis of the solar wind solution at a specific time.
    :param model: An instance of the HUXt class with a completed solution.
    :param time: Time to look up closet model time to (with an astropy.unit of time).
    :param save: Boolean to determine if the figure is saved.
    :param tag: String to append to the filename if saving the figure.
    :param fighandle, axhandle: existing figure and axis handles if a subplot is to be made
    :param minimalplot: removes colorbar, planets/spacecraft and labels
    :return fig: Figure handle.
    :return ax: Axes handle.
    """

    if (time < model.time_out.min()) | (time > (model.time_out.max())):
        print("Error, input time outside span of model times. Defaulting to closest time")

    id_t = np.argmin(np.abs(model.time_out - time))

    # Get plotting data
    lon_arr, dlon, nlon = H.longitude_grid()
    lon, rad = np.meshgrid(lon_arr.value, model.r.value)
    mymap = mpl.cm.viridis
    v_sub = model.v_grid.value[id_t, :, :].copy()
    plotvmin = 200
    plotvmax = 810
    dv = 10
    ylab = "Solar Wind Speed (km/s)"
    
    # Insert into full array
    if lon_arr.size != model.lon.size:
        v = np.zeros((model.nr, nlon)) * np.NaN
        if model.lon.size != 1:
            for i, lo in enumerate(model.lon):
                id_match = np.argwhere(lon_arr == lo)[0][0]
                v[:, id_match] = v_sub[:, i]
        else:
            print('Warning: Trying to contour single radial solution will fail.')
    else:
        v = v_sub

    # Pad out to fill the full 2pi of contouring
    pad = lon[:, 0].reshape((lon.shape[0], 1)) + model.twopi
    lon = np.concatenate((lon, pad), axis=1)
    pad = rad[:, 0].reshape((rad.shape[0], 1))
    rad = np.concatenate((rad, pad), axis=1)
    pad = v[:, 0].reshape((v.shape[0], 1))
    v = np.concatenate((v, pad), axis=1)

    mymap.set_over('lightgrey')
    mymap.set_under([0, 0, 0])
    levels = np.arange(plotvmin, plotvmax + dv, dv)
    
    # if no fig and axis handles are given, create a new figure
    if isinstance(fighandle, float):
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    else:
        fig = fighandle
        ax = axhandle
        
    cnt = ax.contourf(lon, rad, v, levels=levels, cmap=mymap, extend='both')

    # Add on CME boundaries
    cme_colors = ['r', 'c', 'm', 'y', 'deeppink', 'darkorange']
    for j, cme in enumerate(model.cmes):
        cid = np.mod(j, len(cme_colors))
        cme_lons = cme.coords[id_t]['lon']
        cme_r = cme.coords[id_t]['r'].to(u.solRad)
        if np.any(np.isfinite(cme_r)):
            # Pad out to close the profile.
            cme_lons = np.append(cme_lons, cme_lons[0])
            cme_r = np.append(cme_r, cme_r[0])
            ax.plot(cme_lons, cme_r, '-', color=cme_colors[cid], linewidth=3)

    ax.set_ylim(0, model.r.value.max())
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    if not minimalplot:
        # Add on observers 
        for body, style in zip(['EARTH', 'VENUS', 'MERCURY', 'STA', 'STB'], ['co', 'mo', 'ko', 'rs', 'y^']):
            obs = model.get_observer(body)
            deltalon = 0.0*u.rad
            if model.frame == 'sidereal':
                earth_pos = model.get_observer('EARTH')
                deltalon = earth_pos.lon_hae[id_t] - earth_pos.lon_hae[0]
                
            obslon = H._zerototwopi_(obs.lon[id_t] + deltalon)
            ax.plot(obslon, obs.r[id_t], style, markersize=16, label=body)
    
        # Add on a legend.
        fig.legend(ncol=5, loc='lower center', frameon=False, handletextpad=0.2, columnspacing=1.0)
        
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
        cbar1.set_ticks(np.arange(plotvmin, plotvmax, dv*10))
    
        # Add label
        label = "   Time: {:3.2f} days".format(model.time_out[id_t].to(u.day).value)
        label = label + '\n ' + (model.time_init + time).strftime('%Y-%m-%d %H:%M')
        fig.text(0.70, pos.y0, label, fontsize=16)
        
        label = "HUXt2D"
        fig.text(0.175, pos.y0, label, fontsize=16)

    
        #plot any provided streaklines
        for i in range(0, len(streaklines)):
            r, lon = streaklines[i]
            ax.plot(lon, r[id_t, :], 'k') 
        
        #plot any HCS that have been traced
        if plotHCS and (hasattr(model, 'HCS_p2n') or hasattr(model, 'HCS_n2p')):
            for i in range(0, len(model.HCS_p2n)):
                r,l =  model.HCS_p2n[i]
                ax.plot(l, r[id_t, :], 'w')
            for i in range(0, len(model.HCS_n2p)):
                r,l =   model.HCS_n2p[i]
                ax.plot(l, r[id_t, :], 'k') 

    if save:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_frame_{:03d}.png".format(cr_num, tag, id_t)
        filepath = os.path.join(model._figure_dir_, filename)
        fig.savefig(filepath)

    return fig, ax


def animate(model, tag, streaklines=[], plotHCS=True):
    """
    Animate the model solution, and save as an MP4.
    :param model: An instance of the HUXt class with a completed solution.
    :param tag: String to append to the filename of the animation.
    """

    # Set the duration of the movie
    # Scaled so a 5 day simulation with dt_scale=4 is a 10 second movie.
    duration = model.simtime.value * (10 / 432000)

    def make_frame(t):
        """
        Produce the frame required by MoviePy.VideoClip.
        :param t: time through the movie
        """
        # Get the time index closest to this fraction of movie duration
        i = np.int32((model.nt_out - 1) * t / duration)
        fig, ax = plot(model, model.time_out[i],
                       streaklines=streaklines, plotHCS=plotHCS)
        frame = mplfig_to_npimage(fig)
        plt.close('all')
        return frame

    cr_num = np.int32(model.cr_num.value)
    filename = "HUXt_CR{:03d}_{}_movie.mp4".format(cr_num, tag)
    filepath = os.path.join(model._figure_dir_, filename)
    animation = mpy.VideoClip(make_frame, duration=duration)
    animation.write_videofile(filepath, fps=24, codec='libx264')
    return


def plot_radial(model, time, lon, save=False, tag=''):
    """
    Plot the radial solar wind profile at model time closest to specified time.
    :param model: An instance of the HUXt class with a completed solution.
    :param time: Time (in seconds) to find the closest model time step to.
    :param lon: The model longitude of the selected radial to plot.
    :param save: Boolean to determine if the figure is saved.
    :param tag: String to append to the filename if saving the figure.
    :return: fig: Figure handle
    :return: ax: Axes handle
    """

    if (time < model.time_out.min()) | (time > (model.time_out.max())):
        print("Error, input time outside span of model times. Defaulting to closest time")
        id_t = np.argmin(np.abs(model.time_out - time))
        time = model.time_out[id_t]

    if model.lon.size != 1:
        if (lon < model.lon.min()) | (lon > (model.lon.max())):
            print("Error, input lon outside range of model longitudes. Defaulting to closest longitude")
            id_lon = np.argmin(np.abs(model.lon - lon))
            lon = model.lon[id_lon]

    fig, ax = plt.subplots(figsize=(14, 7))
    # Get plotting data
    id_t = np.argmin(np.abs(model.time_out - time))
    time_out = model.time_out[id_t].to(u.day).value

    if model.lon.size == 1:
        id_lon = 0
        lon_out = model.lon.value
    else:
        id_lon = np.argmin(np.abs(model.lon - lon))
        lon_out = model.lon[id_lon].to(u.deg).value

    ylab = 'Solar Wind Speed (km/s)'
    ax.plot(model.r, model.v_grid[id_t, :, id_lon], 'k-')
    ymin = 200
    ymax = 1000
    
    # Plot the CME points on if needed
    cme_colors = ['r', 'c', 'm', 'y', 'deeppink', 'darkorange']
    for c, cme in enumerate(model.cmes):
        cc = np.mod(c, len(cme_colors))
        
        lon_cme = cme.coords[id_t]['lon']
        r_cme = cme.coords[id_t]['r'].to(u.solRad)
        
        id_front = cme.coords[id_t]['front_id'] == 1.0
        id_back = cme.coords[id_t]['front_id'] == 0.0 
        r_front = r_cme[id_front]
        lon_front = lon_cme[id_front]
        r_back = r_cme[id_back]
        lon_back = lon_cme[id_back]
        
        id_cme_lon = np.argmin(np.abs(lon_front - lon))
        r_front = r_front[id_cme_lon]
        id_cme_lon = np.argmin(np.abs(lon_back - lon))
        r_back = r_back[id_cme_lon]
        
        id_cme = (model.r >= r_back) & (model.r <= r_front)
        label = "CME {:02d}".format(c)
        ax.plot(model.r[id_cme], model.v_grid[id_t, id_cme, id_lon], '.', color=cme_colors[cc], label=label)

    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylab)
    ax.set_xlim(model.r.value.min(), model.r.value.max())
    ax.set_xlabel('Radial distance ($R_{sun}$)')

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)

    # Add label
    time_label = " Time: {:3.2f} days".format(time_out)
    lon_label = " Lon: {:3.2f}$^\circ$".format(lon_out)
    label = "HUXt" + time_label + lon_label
    ax.set_title(label, fontsize=20)
    
    if save:
        cr_num = np.int32(model.cr_num.value)
        lon_tag = "{}deg".format(lon.to(u.deg).value)
        filename = "HUXt_CR{:03d}_{}_radial_profile_lon_{}_frame_{:03d}.png".format(cr_num, tag, lon_tag, id_t)
        filepath = os.path.join(model._figure_dir_, filename)
        fig.savefig(filepath)

    return fig, ax


def plot_timeseries(model, radius, lon, save=False, tag=''):
    """
    Plot the solar wind model timeseries at model radius and longitude closest to those specified.
    :param model: An instance of the HUXt class with a completed solution.
    :param radius: Radius to find the closest model radius to.
    :param lon: Longitude to find the closest model longitude to.
    :param save: Boolean to determine if the figure is saved.
    :param tag: String to append to the filename if saving the figure.
    :return: fig: Figure handle
    :return: ax: Axes handle
    """

    if (radius < model.r.min()) | (radius > (model.r.max())):
        print("Error, specified radius outside of model radial grid")

    if model.lon.size != 1:
        if (lon < model.lon.min()) | (lon > (model.lon.max())):
            print("Error, input lon outside range of model longitudes. Defaulting to closest longitude")
            id_lon = np.argmin(np.abs(model.lon - lon))
            lon = model.lon[id_lon]

    fig, ax = plt.subplots(figsize=(14, 7))
    # Get plotting data
    id_r = np.argmin(np.abs(model.r - radius))
    r_out = model.r[id_r].value
    if model.lon.size == 1:
        id_lon = 0
        lon_out = model.lon.value
    else:
        id_lon = np.argmin(np.abs(model.lon - lon))
        lon_out = model.lon[id_lon].value

    t_day = model.time_out.to(u.day)
    
    ax.plot(t_day, model.v_grid[:, id_r, id_lon], 'k-')
    ylab = 'Solar Wind Speed (km/s)'
    ymin = 200
    ymax = 1000

    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylab)
    ax.set_xlim(t_day.value.min(), t_day.value.max())
    ax.set_xlabel('Time (days)')

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)

    # Add label
    radius_label = " Radius: {:3.2f}".format(r_out) + "$R_{sun}$ "
    lon_label = " Longitude: {:3.2f}".format(lon_out) + "$^\circ$"
    label = "HUXt" + radius_label + lon_label
    ax.set_title(label, fontsize=20)

    if save:
        cr_num = np.int32(model.cr_num.value)
        r_tag = np.int32(r_out)
        lon_tag = np.int32(lon_out)
        template_string = "HUXt1D_CR{:03d}_{}_time_series_radius_{:03d}_lon_{:03d}.png"
        filename = template_string.format(cr_num, tag, r_tag, lon_tag)
        filepath = os.path.join(model._figure_dir_, filename)
        fig.savefig(filepath)

    return fig, ax


def get_earth_timeseries(model):
    """
    Compute the solar wind time series at Earth. Returns a pandas dataframe with the 
    solar wind speed time series at Earth interpolated from the model solution using the
    Earth ephemeris. Nearest neighbour interpolation in r, linear interpolation in longitude.
    
    Returns bpolarity if that has been computed post run
    """
    earth_pos = model.get_observer('Earth')

    # adjust the HEEQ coordinates if the sidereal frame has been used
    if model.frame == 'sidereal':
        deltalon = earth_pos.lon_hae - earth_pos.lon_hae[0]
        lonheeq = H._zerototwopi_(earth_pos.lon.value + deltalon.value)
    elif model.frame == 'synodic':
        lonheeq = earth_pos.lon.value 

    if model.nlon == 1:
        print('Single longitude simulated. Extracting time series at Earth r')

    time = np.ones(model.nt_out)*np.nan
    lon = np.ones(model.nt_out)*np.nan
    rad = np.ones(model.nt_out)*np.nan
    speed = np.ones(model.nt_out)*np.nan
    bpol = np.ones(model.nt_out)*np.nan

    for t in range(model.nt_out):
        time[t] = (model.time_init + model.time_out[t]).jd

        #find the nearest longitude cell
        model_lons = model.lon.value
        if model.nlon == 1:
            model_lons = np.array([model_lons])
        id_lon = np.argmin(np.abs(model_lons - lonheeq[t]))
        
        #check whether the Earth is within the model domain
        if ( (earth_pos.r[t].value < model.r[0].value) or 
             (earth_pos.r[t].value > model.r[-1].value) or
             ( (abs(model_lons[id_lon] - lonheeq[t]) > model.dlon.value) and
               (abs(model_lons[id_lon] - lonheeq[t])- 2*np.pi > model.dlon.value) 
             ) ):
            
            bpol[t] = np.nan
            speed[t] = np.nan
            print('Outside model domain')
        else:
            # find the nearest R coord
            id_r = np.argmin(np.abs(model.r.value - earth_pos.r[t].value))
            rad[t] = model.r[id_r].value
            lon[t] = lonheeq[t]
            # then interpolate the values in longitude
            if model.nlon == 1:
                speed[t] = model.v_grid[t, id_r, 0].value
                if hasattr(model, 'b_grid'):
                    bpol[t] = model.b_grid[t, id_r, 0]
            else:
                speed[t] = np.interp(lonheeq[t], model.lon.value, model.v_grid[t, id_r, :].value, period=2*np.pi)
                if hasattr(model, 'b_grid'):
                    bpol[t] = np.interp(lonheeq[t], model.lon.value, model.b_grid[t, id_r, :], period=2*np.pi)

    time = Time(time, format='jd')

    earth_time_series = pd.DataFrame(data={'time':time.datetime, 'r':rad, 
                                           'lon':lon, 'vsw':speed, 'bpol':bpol})
    return earth_time_series

def plot_earth_timeseries(model, plot_OMNI = True):
    """
    A function to plot the HUXt Earth time series. With optino to download and
    plot OMNI data.

    Parameters
    ----------
    model : input model class
    plot_OMNI: Flag for downloading and plotting OMNI data

    Returns
    -------
    fig : figure handle
    axs : ax handles

    """
    
    huxt_ts = get_earth_timeseries(model)
    
    # 2-panel plot if the B polarity has been traced
    if hasattr(model, 'b_grid'):
        fig, axs = plt.subplots(2,1, figsize=(14, 7))
        axs[1].plot(huxt_ts['time'], np.sign(huxt_ts['bpol']), 'k.', label='HUXt')
        axs[1].set_ylabel('B polarity')
    else:
        fig, axs = plt.subplots(1,1, figsize=(14, 7))
        axs = np.array([axs])
        
    axs[0].plot(huxt_ts['time'], huxt_ts['vsw'], 'k', label='HUXt')
    axs[0].set_ylim(250,1000)
    
    
    if plot_OMNI:
        starttime = huxt_ts['time'][0]
        endtime =   huxt_ts['time'][len(huxt_ts)-1]
        
        # Download the 1hr OMNI data from CDAweb
        trange = a.Time(starttime, endtime)
        dataset = a.cdaweb.Dataset('OMNI2_H0_MRG1HR')
        result = Fido.search(trange, dataset)
        downloaded_files = Fido.fetch(result)
        # print(downloaded_files)
        
        # # Import the OMNI data
        omni = TimeSeries(downloaded_files, concatenate=True)
        data = omni.to_dataframe()
        
        # # Set invalid data points to NaN
        id_bad = data['V']==9999.0
        data.loc[id_bad, 'V']=np.NaN
        
        #create a datetime column
        data['datetime'] = data.index.to_pydatetime()
        
        mask = (data['datetime'] >= starttime) & (data['datetime'] <= endtime)
        plotdata = data[mask]
        axs[0].plot(plotdata['datetime'], plotdata['V'], 'r', label='OMNI')
        
        if hasattr(model, 'b_grid'):
            axs[1].plot(plotdata['datetime'], -np.sign(plotdata['BX_GSE'])*0.92, 
                        'r.', label='OMNI')
            axs[1].legend()
            axs[1].set_xlim(starttime, endtime)
            axs[1].set_ylim(-1.1, 1.1)
            
    axs[0].legend()
    axs[0].set_ylabel('Solar Wind Speed (km/s)')
    axs[0].set_xlim(starttime, endtime)
    
    return fig, axs

@u.quantity_input(time=u.day)
def plot_3d_meridional(model3d, time, lon=np.NaN*u.deg, save=False, tag=''):
    """
    Make a contour plot on polar axis of the solar wind solution at a specific time.
    :param model: An instance of the HUXt class with a completed solution.
    :param time: Time to look up closet model time to (with an astropy.unit of time).
    :param save: Boolean to determine if the figure is saved.
    :param tag: String to append to the filename if saving the figure.
    :return fig: Figure handle.
    :return ax: Axes handle.
    """
    # get the metadata from one of the individual HUXt elements
    model=model3d.HUXtlat[0]
    
    if (time < model.time_out.min()) | (time > (model.time_out.max())):
        print("Error, input time outside span of model times. Defaulting to closest time")

    id_t = np.argmin(np.abs(model.time_out - time))
    time_out = model.time_out[id_t].to(u.day).value
    
    # get the requested longitude
    if model.lon.size == 1:
        id_lon = 0
        lon_out = model.lon.value
    else:
        id_lon = np.argmin(np.abs(model.lon - lon))
        lon_out = model.lon[id_lon].to(u.deg).value
        
    # loop over latitudes and extract the radial profiles
    mercut = np.ones((len(model.r), model3d.nlat))
    ymax = 0.0
    for n in range(0, model3d.nlat):
        model = model3d.HUXtlat[n]
        ymin = 200
        ymax = 810
        dv = 19
        ylab = 'Solar Wind Speed (km/s)'
        mercut[:, n] = model.v_grid[id_t, :, id_lon]
        mymap = mpl.cm.viridis
        
    mymap.set_over('lightgrey')
    mymap.set_under([0, 0, 0])
    levels = np.arange(ymin, ymax + dv, dv)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    cnt = ax.contourf(model3d.lat.to(u.rad), model.r, mercut, levels=levels, cmap=mymap, extend='both')

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
    label = label + "; Lon: {:3.2f} degrees".format(lon.to(u.deg).value)
    fig.text(0.675, pos.y0, label, fontsize=16)
    label = "HUXt3D"
    fig.text(0.175, pos.y0, label, fontsize=16)       

    if save:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_frame_{:03d}.png".format(cr_num, tag, id_t)
        filepath = os.path.join(model._figure_dir_, filename)
        fig.savefig(filepath)

    return fig, ax


def animate_3d(model3d, lon=np.NaN*u.deg, tag=''):
    """
    Animate the model solution, and save as an MP4.
    :param field: String, either 'cme', or 'ambient', specifying which solution to animate.
    :param tag: String to append to the filename of the animation.
    """

    # Set the duration of the movie
    # Scaled so a 5 day simulation with dt_scale=4 is a 10 second movie.
    model = model3d.HUXtlat[0]
    duration = model.simtime.value * (10 / 432000)

    def make_frame_3d(t):
        """
        Produce the frame required by MoviePy.VideoClip.
        :param t: time through the movie
        """
        # Get the time index closest to this fraction of movie duration
        i = np.int32((model.nt_out - 1) * t / duration)
        fig, ax = plot_3d_meridional(model3d, model.time_out[i], lon)
        frame = mplfig_to_npimage(fig)
        plt.close('all')
        return frame

    cr_num = np.int32(model.cr_num.value)
    filename = "HUXt_CR{:03d}_{}_movie.mp4".format(cr_num, tag)
    filepath = os.path.join(model._figure_dir_, filename)
    animation = mpy.VideoClip(make_frame_3d, duration=duration)
    animation.write_videofile(filepath, fps=24, codec='libx264')
    return


def huxt_streakline(model, carr_lon_src):
    """
    A function to compute a streakline in the HUXt solution. 
    Requires that the model was run with the "save_full_v" flag
    
    The logic follows as:
    # Input an initial longitude to follow, lon
    # Release a partical at lon
    # Advect.
    # Release a new particle when Omega*Dt > longitude grid resolution
    # Advect all particles and update location.
    # Repeat til end of simulation
    
    model: a huxt instance with a computed solution in
    lon_src: the longitude you want to follow at t=0 in the model solution
    """
    
    #check that the full model output is available
    assert(model.save_full_v == True)
    
    #work out the source longitude at t=0 from the given Carrington longitude
    lon_src = H._zerototwopi_((carr_lon_src - model.cr_lon_init))*u.rad
    
    #adjust the source longitude for the spin-up time
    lon_src = H._zerototwopi_(lon_src.to(u.rad).value - 2*np.pi*model.buffertime.to(u.s) 
                                  / model.rotation_period)
    
    # Get grid values
    r_grid = model.r.copy().to(u.km).value
    v_grid = model.v_grid_full.copy().value
    lon_sim = model.lon.value
    time_grid = model.model_time.value
    dlon = model.dlon.value
    T_rot = model.rotation_period.value 
    
    #check if it's a 1d solution
    if model.nlon == 1:
        lon_sim = np.array([lon_sim])

    particle_r, particle_lons = trace_particles(r_grid, lon_sim, dlon,
                                                time_grid, v_grid, T_rot, lon_src)

    # Set all particles stuck on the outer boundary to NaN.        
    id_outer = particle_r > r_grid[-1] + 3*model.dr.to(u.km).value
    particle_r[id_outer] = np.NaN

    # Only return the longitudes with particles on
    lon_grid = np.arange(dlon/2, 2*np.pi-dlon/2+0.01, dlon)
    n_lon = len(particle_lons)
    particle_r = particle_r[:, :n_lon]
    
    # Only return time steps for the output time (not the full time grid)
    #find the common timesteps to within +/- 500 seconds
    common, x_ind, y_ind  = np.intersect1d(np.rint(model.model_time/500),
                                           np.rint(model.time_out/500),
                                           return_indices=True)
    particle_r = particle_r[x_ind,:]
    # If you're getting a kernel restart, check that the length of particle_r is
    # equal to nt. the tolerance of np.interect1d may need reducing if not.
    assert(len(particle_r) == len(model.time_out))
    
    #add units back in
    particle_r = (particle_r * u.km).to(u.solRad)
    lon = lon_grid[particle_lons]
    lon = lon *u.rad
    
    return particle_r, lon

@jit(nopython=True)
def trace_particles(r_grid, lon_sim, dlon, time_grid, v_grid, T_rot, lon_src):
    
    """
    optimised function to track a single longitude through the HUXt solution
    
    used by huxt_streakline
    """
    
    nt = len(time_grid)

    dt = time_grid[1]-time_grid[0]
    lon_grid = np.arange(dlon/2, 2*np.pi-dlon/2+0.01, dlon)
    nlon = len(lon_grid)
    
    id_lon = np.argmin(np.abs(lon_grid - lon_src))
    
    # work out how many rotations are present, including spin up
    t_total = nt*dt
    n_rots = int(np.floor(t_total/T_rot) + 1)
    particle_r = np.NaN * np.zeros((nt, (nlon+1)*n_rots)) 
    
    # Initialise starting position of first particle
    particle_r[0, 0] = r_grid[0]
    particle_lons = [id_lon]
    
    lon_acc = 0 
    omega = 2 * np.pi / T_rot
    
    # Loop through model time steps
    for t_counter, t_out in enumerate(time_grid):
        
        # Loop through longitudes with particles on
        for lon_counter, lon_p in enumerate(particle_lons):
           
            # Get the j index for creating a new particle if necessary.
            lon_next = lon_counter + 1
            
            this_lon_present = False
            if len(lon_sim)>1: #2d solution
                if (np.abs(lon_sim - lon_grid[lon_p]) < 0.01).any():
                    this_lon_present = True
            else: #1d solution, single longiutde
                if (np.abs(lon_sim - lon_grid[lon_p]) < 0.01):
                    this_lon_present = True
            
            # check whether the longitude of the particle was simualted by HUXt
            if this_lon_present: 

                # Get the particles position
                if t_counter == 0:
                    r_p = r_grid[0]
                else:
                    r_p = particle_r[t_counter-1, lon_counter]
                        
                # get the model longitude that matches the grid longitude
                id_mod = np.argmin(np.abs(lon_sim - lon_grid[lon_p]))
                    
                # interpolate the speed profile at the particles longitude and radius
                v_p = np.interp(r_p, r_grid, v_grid[t_counter, :, id_mod])
                # Advect particle position
                r_n = r_p + dt*v_p
                
                # If particle still in model domain, save updated position, else, set at boundary.
                if r_n <= r_grid[-1]:
                    particle_r[t_counter, lon_counter] = r_n
                else:
                    particle_r[t_counter, lon_counter] = r_grid[-1]
                
                # Get the j index for creating a new particle if necessary.
                lon_next = lon_counter + 1
            else:
                #print('uknown longitude')
                particle_r[t_counter, lon_counter] = np.nan
        
        # Has initial longitude rotated into next longitude bin?
        lon_acc = lon_acc + (omega * dt)
        if lon_acc >= dlon:
            # Add new particle 
            # Get the new longitude index
            id_lon = id_lon + 1
            # Reset to zero if cross the 360-0 boundary
            if id_lon > (nlon-1):
                id_lon = 0 
            # Update the longitude ids    
            particle_lons.append(id_lon)
            # Initialise the particle height at this time/longitude
            particle_r[t_counter, lon_next] = r_grid[0]
            # Reset the longitude accumulator
            lon_acc = 0
            
        #add a particle at the innter boundary for plotting purposes
        particle_r[t_counter,lon_counter+1] = r_grid[0]

    return particle_r, particle_lons


def trace_HCS(model, br_in):
    
    """
    Function to trace HCS from given B(carrLon) and add tracks to model class
    
    used by add_bgrid
    """
    
    #find the HCS crossings
    Nlon = len(br_in)
    HCS = np.zeros((Nlon,1))
    for i in range(0, Nlon-1):
        if (br_in[i] >= 0) & (br_in[i+1]<0):
            #place the HCS crossing at 
            HCS[i+1] = 1 #this will be neg to pos in time
        elif (br_in[i] <= 0) & (br_in[i+1]>0):   
            HCS[i+1] = -1 #this will be pos to neg in time
            
    #check the last value by wrapping arounf 0/2pi
    if (br_in[Nlon-1] >= 0) & (br_in[0]<0):
        #place the HCS crossing at 
        HCS[0] = 1
    elif (br_in[Nlon-1] <= 0) & (br_in[0]>0):   
        HCS[0] = -1
        
    #track the streaklines from the given Carrington lontidues    
    HCS_p2n_tracks = []
    HCS_n2p_tracks = []
    
    dlon = model.dlon.value
    lon_grid = np.arange(dlon/2, 2*np.pi-dlon/2+0.01, dlon)
    
    for i in range(0,Nlon):
        if HCS[i] == 1:
            carr_lon = lon_grid[i]*u.rad
            r, l = huxt_streakline(model, carr_lon)
            #HCS is placed at the r grid immedaitely before the polarity reversal. move it to the interface
            #l = l + dlon/2
            HCS_p2n_tracks.append((r,l))
        elif HCS[i] == -1:
            carr_lon = lon_grid[i]*u.rad
            r, l = huxt_streakline(model, carr_lon)
            #l = l + dlon/2
            HCS_n2p_tracks.append((r,l))
    
    #add the HCS crossings to the model class        
    model.HCS_n2p = HCS_n2p_tracks
    model.HCS_p2n = HCS_p2n_tracks
    
    return 1

def add_bgrid(model, br_in):
    """
    Input: Solved HUXt model class and Br at inner boundary in Carrington longitude
    
    First traces HCS postion and adds solutions to model class. Requires HUXt
    to output the full v field (i.e. solved with save_full_v == True)
    
    Then generates the associated b_grid
    
    """
    
    #trace the HCS positions
    trace_HCS(model, br_in)
    
    #create the full longitude grid
    dlon = model.dlon.value
    lon_grid = np.arange(dlon/2, 2*np.pi-dlon/2+0.01, dlon)*u.rad
    
    # Rotate the boundary condition, as required by cr_lon_init.    
    lon_boundary = lon_grid#model.lon   
    lon_shifted = H._zerototwopi_((lon_boundary - model.cr_lon_init).value)
    id_sort = np.argsort(lon_shifted)
    lon_shifted = lon_shifted[id_sort]
    br_shifted = br_in[id_sort]
    br_boundary = np.interp(lon_boundary.value, lon_shifted, br_shifted, period=np.pi*2)
    
    #combined list
    HCS_tracks = model.HCS_p2n + model.HCS_n2p
    HCS_r_list = []
    HCS_l_list = []
    HCS_p_list = []
    for i in range(0,len(HCS_tracks)):
        r,l =  HCS_tracks[i]
        HCS_r_list.append(r.value)
        HCS_l_list.append(l.value)
        # set the polarity flag
        if i<len(model.HCS_p2n):
            HCS_p_list.append(1)
        else:
            HCS_p_list.append(-1)
        
        
    r_values = model.r.value  
    time = model.time_out.value 
    all_lons = lon_grid.value
    T_rot = model.rotation_period.value
    
    lon_values = model.lon.value 
    if model.nlon == 1:
        lon_values = np.array([lon_values])
    
    # work out how many rotations are present, including spin up
    time_grid = model.model_time.value
    nt = len(time_grid)
    dt = time_grid[1]-time_grid[0]
    t_total = nt*dt
    n_rots = int(np.floor(t_total/T_rot) + 1)
    
    model.b_grid =  bgrid_from_hcs_tracks(time, HCS_r_list, HCS_l_list, HCS_p_list, 
                                           br_boundary, all_lons, lon_values, r_values,  
                                           T_rot, n_rots) 
    
    return 1
               

@jit(nopython=True)                 
def bgrid_from_hcs_tracks(time, HCS_r_list, HCS_l_list, HCS_p_list,
                          br_boundary, all_lons, lon_values, r_values, T_rot, n_rots):
    
    """
    optimised function to create a b-polarity grid from HCS tracks
    
    used by 
    """
   
    nHCS = len(HCS_r_list)
    nt = len(time)
    nr = len(r_values)
    nlon = len(lon_values)
    

    
    
    #produce b_grid from HCS crossings
    b_grid = np.ones((nt,nr,nlon))
    for t in range(0,nt):
        
        #rotate the Br_boundary
        rot_total = 2*np.pi * time[t] / T_rot
        
        #rectify between 0 and 2pi
        a = -np.floor_divide(rot_total, 2*np.pi)
        rot = rot_total + (a *  2*np.pi)
        
        #compute this rotated longitude grid
        lon_shifted = all_lons + rot
        #rectify between 0 and 2pi
        a = -np.floor_divide(lon_shifted, 2*np.pi)
        lon_shifted = lon_shifted + (a *  2*np.pi)
    
        id_sort = np.argsort(lon_shifted)
        lon_shifted = lon_shifted[id_sort]
        br_shifted = br_boundary[id_sort]
        br_rot = np.interp(lon_values, lon_shifted, br_shifted)#, period=np.pi*2)

        
        for lon in range(0,nlon):
            #use the footpoint polarity
            b_grid[t,:,lon] = np.sign(br_rot[lon])
            
            #find which HCS are present at this longitude
            HCS_thislon = np.ones((nHCS*n_rots,2)) * np.nan
            HCS_counter = 0
            for i in range(0,nHCS):
                r = HCS_r_list[i]
                l = HCS_l_list[i]
                p = HCS_p_list[i]
                #check if this HCS corsses the current longitude
                #mask = (l == lon_values[lon])
                mask = np.argwhere(l == lon_values[lon])
                for j in range(0, mask.size):
                    HCS_thislon[HCS_counter,0] = r[t,mask[j].item()]
                    HCS_thislon[HCS_counter,1] = p
                    #get rid of stuff at the outer bound
                    if HCS_thislon[HCS_counter,0] >= r_values[-1]:
                        HCS_thislon[HCS_counter,0] = np.nan
                    HCS_counter = HCS_counter + 1

    
            #sort the HCS crossings into radial distance order
            order = np.argsort(HCS_thislon[:,0])
            HCS_thislon = HCS_thislon[order,:]
            
    
            #if len(HCS_thislon_r) > 0:
            innerR_index = 0
            outerR_index = nr - 1
            for j in range(0,nHCS*n_rots):
                
                if ~np.isnan(HCS_thislon[j,0]):
                    #innerR = HCS_thislon[j]
                    r_index = np.argmin(np.abs(HCS_thislon[j,0] - r_values))
                    if HCS_thislon[j,1] > 0:
                        b_grid[t, innerR_index:r_index, lon] = 1
                        b_grid[t, r_index:outerR_index, lon] = -1
                    else:
                        b_grid[t, innerR_index:r_index, lon] = -1
                        b_grid[t, r_index:outerR_index, lon] = 1
                    #set this HCS as the new inner boundary
                    innerR_index = r_index
                    
            # if there's no HCS at this long, use the previous polarity
            # stops the polarity from flipping about due to small numerical errors at inner boundary
            if np.isnan(HCS_thislon[:,0]).all():
                if t > 0: #use the polarity from the previous timestep
                    b_grid[t,:,lon] = b_grid[t-1,:,lon]
                    
                    
                        
    return b_grid


# @u.quantity_input(time=u.day)
# def plot_v_HCS(model, time):#,  save=False, tag='', fighandle=np.nan, axhandle=np.nan,
#          #minimalplot=False):
#     """
#     A wrapper for ha.plot which adds the traced HCS crossings to the velocity 
#     field    

#     """
#     fig, ax = plot(model, time)#,  save=save, tag=tag, fighandle=fighandle, 
#                       #axhandle=axhandle,  minimalplot=minimalplot)   
    
#     id_t = np.argmin(np.abs(model.time_out - time))
#     #id_t_particle = np.argmin(np.abs(model.model_time -  model.time_out[id_t]))
#     for i in range(0, len(model.HCS_p2n)):
#         r,l =  model.HCS_p2n[i]
#         ax.plot(l, r[id_t, :], 'w')
#     for i in range(0, len(model.HCS_n2p)):
#         r,l =   model.HCS_n2p[i]
#         ax.plot(l, r[id_t, :], 'k') 
        
#     return fig, ax


@u.quantity_input(time=u.day)
def plot_bpol(model, time, save=False, tag='', fighandle=np.nan, axhandle=np.nan,
         minimalplot=False, streaklines=[], plotHCS = True):
    """
    Make a contour plot on polar axis of the solar wind solution at a specific time.
    :param model: An instance of the HUXt class with a completed solution.
    :param time: Time to look up closet model time to (with an astropy.unit of time).
    :param save: Boolean to determine if the figure is saved.
    :param tag: String to append to the filename if saving the figure.
    :param fighandle, axhandle: existing figure and axis handles if a subplot is to be made
    :param minimalplot: removes colorbar, planets/spacecraft and labels
    :return fig: Figure handle.
    :return ax: Axes handle.
    """

    if (time < model.time_out.min()) | (time > (model.time_out.max())):
        print("Error, input time outside span of model times. Defaulting to closest time")

    id_t = np.argmin(np.abs(model.time_out - time))

    # Get plotting data
    lon_arr, dlon, nlon = H.longitude_grid()
    lon, rad = np.meshgrid(lon_arr.value, model.r.value)
    mymap = mpl.cm.PuOr
    v_sub = model.b_grid[id_t, :, :].copy()
    plotvmin = -1.1
    plotvmax = 1.1
    dv = 1
    ylab = "Magnetic field polarity"
    
    # Insert into full array
    if lon_arr.size != model.lon.size:
        v = np.zeros((model.nr, nlon)) * np.NaN
        if model.lon.size != 1:
            for i, lo in enumerate(model.lon):
                id_match = np.argwhere(lon_arr == lo)[0][0]
                v[:, id_match] = v_sub[:, i]
        else:
            print('Warning: Trying to contour single radial solution will fail.')
    else:
        v = v_sub

    # Pad out to fill the full 2pi of contouring
    pad = lon[:, 0].reshape((lon.shape[0], 1)) + model.twopi
    lon = np.concatenate((lon, pad), axis=1)
    pad = rad[:, 0].reshape((rad.shape[0], 1))
    rad = np.concatenate((rad, pad), axis=1)
    pad = v[:, 0].reshape((v.shape[0], 1))
    v = np.concatenate((v, pad), axis=1)

    mymap.set_over('lightgrey')
    mymap.set_under([0, 0, 0])
    levels = np.arange(plotvmin, plotvmax + dv, dv)
    
    # if no fig and axis handles are given, create a new figure
    if isinstance(fighandle, float):
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    else:
        fig = fighandle
        ax = axhandle
        
    cnt = ax.contourf(lon, rad, v, levels=levels, cmap=mymap, extend='both')

    # Add on CME boundaries
    cme_colors = ['r', 'c', 'm', 'y', 'deeppink', 'darkorange']
    for j, cme in enumerate(model.cmes):
        cid = np.mod(j, len(cme_colors))
        cme_lons = cme.coords[id_t]['lon']
        cme_r = cme.coords[id_t]['r'].to(u.solRad)
        if np.any(np.isfinite(cme_r)):
            # Pad out to close the profile.
            cme_lons = np.append(cme_lons, cme_lons[0])
            cme_r = np.append(cme_r, cme_r[0])
            ax.plot(cme_lons, cme_r, '-', color=cme_colors[cid], linewidth=3)

    ax.set_ylim(0, model.r.value.max())
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    if not minimalplot:
        # Add on observers 
        for body, style in zip(['EARTH', 'VENUS', 'MERCURY', 'STA', 'STB'], ['co', 'mo', 'ko', 'rs', 'y^']):
            obs = model.get_observer(body)
            deltalon = 0.0*u.rad
            if model.frame == 'sidereal':
                earth_pos = model.get_observer('EARTH')
                deltalon = earth_pos.lon_hae[id_t] - earth_pos.lon_hae[0]
                
            obslon = H._zerototwopi_(obs.lon[id_t] + deltalon)
            ax.plot(obslon, obs.r[id_t], style, markersize=16, label=body)
    
        # Add on a legend.
        fig.legend(ncol=5, loc='lower center', frameon=False, handletextpad=0.2, columnspacing=1.0)
        
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
        cbar1.set_ticks(np.arange(plotvmin, plotvmax, 1))
    
        # Add label
        label = "   Time: {:3.2f} days".format(model.time_out[id_t].to(u.day).value)
        label = label + '\n ' + (model.time_init + time).strftime('%Y-%m-%d %H:%M')
        fig.text(0.70, pos.y0, label, fontsize=16)
        
        label = "HUXt2D"
        fig.text(0.175, pos.y0, label, fontsize=16)
        
        #plot any provided streak lines
        for i in range(0, len(streaklines)):
            r, lon = streaklines[i]
            ax.plot(lon, r[id_t, :], 'k') 
        
        #plot any HCS that have been traced
        if plotHCS and (hasattr(model, 'HCS_p2n') or hasattr(model, 'HCS_n2p')):
            for i in range(0, len(model.HCS_p2n)):
                r,l =  model.HCS_p2n[i]
                ax.plot(l, r[id_t, :], 'w')
            for i in range(0, len(model.HCS_n2p)):
                r,l =   model.HCS_n2p[i]
                ax.plot(l, r[id_t, :], 'k') 

    if save:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_frame_{:03d}.png".format(cr_num, tag, id_t)
        filepath = os.path.join(model._figure_dir_, filename)
        fig.savefig(filepath)

    return fig, ax

    
# def animate_hcs(model, tag):
#     """
#     Animate the model solution with HCS, and save as an MP4.
#     :param model: An instance of the HUXt class with a completed solution.
#     :param tag: String to append to the filename of the animation.
#     """

#     # Set the duration of the movie
#     # Scaled so a 5 day simulation with dt_scale=4 is a 10 second movie.
#     duration = model.simtime.value * (10 / 432000)
    
#     def make_frame(t):
#         """
#         Produce the frame required by MoviePy.VideoClip.
#         :param t: time through the movie
#         """
#         # Get the time index closest to this fraction of movie duration
#         i = np.int32((model.nt_out - 1) * t / duration)
        
#         fig, ax = plot_v_HCS(model, model.time_out[i])    
        
#         frame = mplfig_to_npimage(fig)
#         plt.close('all')
#         return frame
    
#     cr_num = np.int32(model.cr_num.value)
#     filename = "HUXt_CR{:03d}_{}_movie.mp4".format(cr_num, tag)
#     filepath = os.path.join(model._figure_dir_, filename)
#     animation = mpy.VideoClip(make_frame, duration=duration)
#     animation.write_videofile(filepath, fps=24, codec='libx264')