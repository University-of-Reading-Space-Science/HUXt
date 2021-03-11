import numpy as np
import astropy.units as u
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage

import HUXt as H

mpl.rc("axes", labelsize=16)
mpl.rc("ytick", labelsize=16)
mpl.rc("xtick", labelsize=16)
mpl.rc("legend", fontsize=16)


@u.quantity_input(time=u.day)
def plot(model, time, save=False, tag=''):
    """
    Make a contour plot on polar axis of the solar wind solution at a specific time.
    :param model: An instance of the HUXt class with a completed solution.
    :param time: Time to look up closet model time to (with an astropy.unit of time).
    :param save: Boolean to determine if the figure is saved.
    :param tag: String to append to the filename if saving the figure.
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
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
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
    cbar1.set_ticks(np.arange(plotvmin, plotvmax, dv*10))

    # Add label
    label = "Time: {:3.2f} days".format(model.time_out[id_t].to(u.day).value)
    fig.text(0.675, pos.y0, label, fontsize=16)
    label = "HUXt2D"
    fig.text(0.175, pos.y0, label, fontsize=16)

    if save:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_frame_{:03d}.png".format(cr_num, tag, id_t)
        filepath = os.path.join(model._figure_dir_, filename)
        fig.savefig(filepath)

    return fig, ax


def animate(model, tag):
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
        fig, ax = plot(model, model.time_out[i])
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
    #ax.legend(loc=1)
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
    Returns Earth time series. Columns are:
        0 - time (MJD)
        1 - speed (km/s)
        2 - CME tracer density 
        3 - Br (is available)
        4 - rho (if available)
    """
    Earthpos = model.get_observer('Earth')

    #adjust the HEEQ coordinates if the sidereal frame has been used
    if model.frame == 'sidereal':
        deltalon = Earthpos.lon_hae -  Earthpos.lon_hae[0]
        lonheeq = H._zerototwopi_(Earthpos.lon.value + deltalon.value)
    elif model.frame == 'synodic':
        lonheeq = Earthpos.lon.value 

    if model.nlon == 1:
        print('Single longitude simulated. Extracting time series at Earth r')


    earth_time_series = np.ones((model.nt_out, 2))*np.nan
    for t in range(0,model.nt_out):
        earth_time_series[t,0] = (model.time_init + model.time_out[t]).mjd

        #find the nearest R coord
        id_r = np.argmin(np.abs(model.r.value - Earthpos.r[t].value))

        #then interpolate the values in longitude
        if model.nlon == 1:
            earth_time_series[t, 1] = model.v_grid[t, id_r, 0].value
        else:
            earth_time_series[t, 1] = np.interp(lonheeq[t], model.lon.value,
                                                model.v_grid[t, id_r, :].value, period=2*np.pi)
    return earth_time_series



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
    #get the metadata from one of the individual HUXt elements
    model=model3d.HUXtlat[0]
    
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
        
    #loop over latitudes and extract the radial profiles
    mercut=np.ones((len(model.r),model3d.nlat))
    ymax=0.0
    for n in range(0,model3d.nlat):
        model=model3d.HUXtlat[n]
        ymin=200; ymax=810; dv=19;
        ylab='Solar Wind Speed (km/s)'
        mercut[:,n]=model.v_grid[id_t, :, id_lon]
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
    fig.text(0.675, pos.y0, label, fontsize=16)
    label = "HUXt2D"
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
    model=model3d.HUXtlat[0]
    duration = model.simtime.value * (10 / 432000)

    def make_frame_3d(t):
        """
        Produce the frame required by MoviePy.VideoClip.
        :param t: time through the movie
        """
        # Get the time index closest to this fraction of movie duration
        i = np.int32((model.nt_out - 1) * t / duration)
        fig, ax =  plot_3d_meridional(model3d, model.time_out[i], lon)
        frame = mplfig_to_npimage(fig)
        plt.close('all')
        return frame

    cr_num = np.int32(model.cr_num.value)
    filename = "HUXt_CR{:03d}_{}_movie.mp4".format(cr_num, tag)
    filepath = os.path.join(model._figure_dir_, filename)
    animation = mpy.VideoClip(make_frame_3d, duration=duration)
    animation.write_videofile(filepath, fps=24, codec='libx264')
    return