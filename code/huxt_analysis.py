import os

import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from numba import jit
from scipy.optimize import minimize

import huxt as H
from huxt_inputs import get_omni

mpl.rc("axes", labelsize=16)
mpl.rc("ytick", labelsize=16)
mpl.rc("xtick", labelsize=16)
mpl.rc("legend", fontsize=16)


@u.quantity_input(time=u.day)
def plot(model, time, save=False, tag='', fighandle=np.nan, axhandle=np.nan, minimalplot=False, plotHCS=True,
         annotateplot = True, trace_earth_connection =False):
    """
    Make a contour plot on polar axis of the solar wind solution at a specific time.
    Args:
        model: An instance of the HUXt class with a completed solution.
        time: Time to look up closet model time to (with an astropy.unit of time).
        save: Boolean to determine if the figure is saved.
        tag: String to append to the filename if saving the figure.
        fighandle: Figure handle for placing plot in existing figure.
        axhandle: Axes handle for placing plot in existing axes.
        minimalplot: Boolean, if True removes colorbar, planets, spacecraft, and labels.
        plotHCS: Boolean, if True plots heliospheric current sheet coordinates
    Returns:
        fig: Figure handle.
        ax: Axes handle.
    """

    if (time < model.time_out.min()) | (time > (model.time_out.max())):
        print("Error, input time outside span of model times. Defaulting to closest time")

    # if no fig and axis handles are given, create a new figure
    if isinstance(fighandle, float):
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    else:
        fig = fighandle
        ax = axhandle    

    id_t = np.argmin(np.abs(model.time_out - time))

    # Get plotting data
    lon_arr, dlon, nlon = H.longitude_grid()
    lon, rad = np.meshgrid(lon_arr.value, model.r.value)

    orig_cmap = mpl.cm.viridis
    # make a copy
    mymap = type(orig_cmap)(orig_cmap.colors)

    v_sub = model.v_grid.value[id_t, :, :].copy()
    plotvmin = 200
    plotvmax = 810
    dv = 10
    ylab = r"$V_{SW}$" + "\n[km/s]"

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

    cnt = ax.contourf(lon, rad, v, levels=levels, cmap=mymap, extend='both')

    cnt.set(edgecolor="face")

    # Add on CME boundaries
    if model.track_cmes:
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
    
    # plot the Sun
    ax.plot(0, 0, 'o', color=[1.0, 0.5, 0.25], markersize=16)

    if not minimalplot:
        
        # determine which bodies should be plotted
        plot_observers = list(zip(['EARTH'], ['ko']))
        plot_observers.append(('STA', 'c*'))
        if model.time_init < datetime.datetime(2016,8,21):
            plot_observers.append(('STB', 'y*'))
        if model.r[-1] < 350 * u.solRad:
            plot_observers.append(('VENUS', 'yo'))
            plot_observers.append(('MERCURY', 'mo'))
        if model.r[-1] > 350 * u.solRad:
            plot_observers.append(('MARS', 'ro'))
        if model.r[-1] > 1100 * u.solRad:
            plot_observers.append(('JUPITER', 'mo'))   
        if model.r[-1] > 2000 * u.solRad:
            plot_observers.append(('SATURN', 'yo'))      

        # Add on observers 
        for body, style in plot_observers:
            obs = model.get_observer(body)
            deltalon = 0.0 * u.rad
            if model.frame == 'sidereal':
                earth_pos = model.get_observer('EARTH')
                deltalon = earth_pos.lon_hae[id_t] - earth_pos.lon_hae[0]

            obslon = H._zerototwopi_(obs.lon[id_t] + deltalon)
            ax.plot(obslon, obs.r[id_t], style, markersize=14, label=body)

        # Add on a legend.
        if annotateplot:
            ax.legend(ncol=5, loc='lower center', frameon=False, fontsize=14, 
                      handletextpad=0.2, columnspacing=1.0,  bbox_to_anchor=(0.5, -0.22))
            
        ax.patch.set_facecolor('slategrey')
        pos = ax.get_position()
        new_pos = [pos.x0, pos.y0 + 0.1, pos.width, pos.height]
        ax.set_position(new_pos)

        # Add color bar
        pos = ax.get_position()
        dw = 0.004
        dh = 0.06
        left = pos.x0 + dw
        bottom = pos.y0 - dh
        wid = pos.width - 2 * dw
        cbaxes = fig.add_axes([left, bottom, wid*0.84, 0.03])
        cbar1 = fig.colorbar(cnt, cax=cbaxes, orientation='horizontal')
        cbar1.set_ticks(np.arange(plotvmin, plotvmax, dv * 10))
        cbaxes.text(1.15, -0.4, ylab, fontsize=15, transform=cbaxes.transAxes, horizontalalignment='center')

        if annotateplot:
            # Add label
            label = "{:3.2f} days".format(model.time_out[id_t].to(u.day).value)
            label = label + '\n ' + (model.time_init + time).strftime('%Y-%m-%d %H:%M')
            ax.text(0.98, -0.01, label, fontsize=15, transform=ax.transAxes, horizontalalignment='right')
    
            label = "HUXt2D \nLat: {:3.0f} deg".format(model.latitude.to(u.deg).value)
            ax.text(0.02, -0.01, label, fontsize=15, transform=ax.transAxes,)

        # plot any tracked streaklines
        if model.track_streak:
            nstreak = len(model.streak_particles_r[0, :, 0, 0])
            for istreak in range(0, nstreak):
                # construct the streakline from multiple rotations
                nrot = len(model.streak_particles_r[0, 0, :, 0])
                streak_r = []
                streak_lon = []
                for irot in range(0, nrot):
                    streak_lon = streak_lon + model.lon.value.tolist()
                    streak_r = streak_r + (
                            model.streak_particles_r[id_t, istreak, irot, :] * u.km.to(u.solRad)).value.tolist()
                    
                # get the real values for plotting
                mask = np.isfinite(streak_r)
                plotlon = np.array(streak_lon)[mask]
                plotr = np.array(streak_r)[mask]
                if len(plotr) > 0:
                    # for plotting only, fix the inner most point on the inner bounday.
                    r_min = model.r[0].to(u.solRad).value
                    dr = plotr[-1] - r_min
                    # compute the long of the footpoint assuming a constant solar wind speed
                    dt = (dr * u.solRad / (350 *u.km /u.s)).to(u.s)
                    dlon_streak = (2*np.pi)*(dt/model.rotation_period).value 
                    inner_lon = H._zerototwopi_(plotlon[-1] + dlon_streak)
                    # check that this new longitude was actually simulated
                    if (np.nanmin(abs(model.lon - inner_lon*u.rad)) < dlon):
                        plotr = np.append(plotr, r_min)
                        plotlon = np.append(plotlon, inner_lon)

                    # for plotting only, fix the outermost point on the outer boundary
                    r_max = model.r[-1].to(u.solRad).value
                    dr = r_max - plotr[0]
                    # compute the long of the outer footpoint assuming a constant solar wind speed
                    dt = (dr * u.solRad / (450 *u.km /u.s)).to(u.s)
                    dlon_streak = (2*np.pi)*(dt/model.rotation_period).value
                    outer_lon = H._zerototwopi_(plotlon[0] - dlon_streak)
                    # check that this new longitude was actually simulated
                    if (np.nanmin(abs(model.lon - outer_lon*u.rad)) < dlon):
                        plotr = np.append(r_max, plotr)
                        plotlon = np.append(outer_lon, plotlon)
                    
                # plot the streakline
                ax.plot(plotlon, plotr, 'k')

        # plot any HCS that have been traced
        if plotHCS and hasattr(model, 'b_grid'):
            for i in range(0, len(model.hcs_particles_r[:, 0, 0, 0])):
                r = model.hcs_particles_r[i, id_t, 0, :] * u.km.to(u.solRad)
                lons = model.lon
                ax.plot(lons, r, 'w.')
                
    if trace_earth_connection:
        plotlon, plotr, optimal_lon, optimal_t = find_Earth_connected_field_line(model, time)
        ax.plot(plotlon, plotr, 'w')

    if save:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_frame_{:03d}.png".format(cr_num, tag, id_t)
        filepath = os.path.join(model._figure_dir_, filename)
        fig.savefig(filepath)

    return fig, ax


def animate(model, tag, duration=10, fps=20, plotHCS=True, trace_earth_connection=False, outputfilepath=''):
    """
    Animate the model solution, and save as an MP4.
    Args:
        model: An instance of the HUXt class with a completed solution.
        tag: String to append to the filename of the animation.
        duration: the movie duration, in seconds
        fps: frames per second
        plotHCS: Boolean flag on whether to plot the heliospheric current sheet location.
        outputfilepath: full path, including filename if output is to be saved anywhere other than huxt/figures
    Returns:
        None
    """
    
    interval = (1/fps)*1000
    nframes = int(duration*1000/interval)
    
    exp_time = int(nframes*0.2)
    print('Rendering ' + str(nframes) + ' frames. Expected time: ' + str(exp_time) + ' secs')
    
    def make_frame(frame):
        """
        Produce the frame required by MoviePy.VideoClip.
        Args:
            t: time through the movie
        Returns:
            frame: An image array for rendering to movie clip.
        """
        plt.clf()  # Clear the previous frame
        ax = fig.add_subplot(111, projection='polar')
        
        # Get the time index closest to this fraction of movie duration
        i = np.int32((model.nt_out - 1) * frame / nframes)
        plot(model, model.time_out[i], fighandle=fig, axhandle=ax, 
             plotHCS=plotHCS, trace_earth_connection=trace_earth_connection)
        return frame
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    
    # Create the animation
    ani = FuncAnimation(fig, make_frame, frames=range(nframes), interval=interval)
    
    # set up the save path
    if outputfilepath:
        filepath = outputfilepath
    else:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_movie.mp4".format(cr_num, tag)
        filepath = os.path.join(model._figure_dir_, filename)
    
    # Save the animation as a movie file
    ani.save(filepath, writer='ffmpeg')
    print('mp4 file written to ' + filepath)

    return

def plot_radial(model, time, lon, save=False, tag=''):
    """
    Plot the radial solar wind profile at model time closest to specified time.
    Args:
        model: An instance of the HUXt class with a completed solution.
        time: Time (in seconds) to find the closest model time step to.
        lon: The model longitude of the selected radial to plot.
        save: Boolean to determine if the figure is saved.
        tag: String to append to the filename if saving the figure.
    Returns:
        fig: Figure handle
        ax: Axes handle
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
    lon_label = " Lon: {:3.2f}$^\\circ$".format(lon_out)
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
    Args:
        model: An instance of the HUXt class with a completed solution.
        radius: Radius to find the closest model radius to.
        lon: Longitude to find the closest model longitude to.
        save: Boolean to determine if the figure is saved.
        tag: String to append to the filename if saving the figure.
    Returns:
        fig: Figure handle
        ax: Axes handle
    """

    if (radius < model.r.min()) | (radius > (model.r.max())):
        print("Error, specified radius outside of model radial grid")

    if model.lon.size != 1:
        if (lon < model.lon.min() - model.dlon) | (lon > model.lon.max() + model.dlon):
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
    lon_label = " Longitude: {:3.2f}".format(lon_out) + "$^\\circ$"
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


def get_observer_timeseries(model, observer='Earth', suppress_warning=False):
    """
    Compute the solar wind time series at an observer location. Returns a pandas dataframe with the 
    solar wind speed time series interpolated from the model solution using the
    observer ephemeris. Nearest neighbour interpolation in r, linear interpolation in longitude.
    Args:
        model: A HUXt instance with a solution generated by HUXt.solve().
        observer: String name of the observer. Can be any permitted by Observer class.
        suppress_warning: Bool for stopping a warning printing.
    Returns:
         time_series: A pandas dataframe giving time series of solar wind speed, and if it exists in the HUXt
                            solution, the magnetic field polarity, at the observer.
    """
    earth_pos = model.get_observer('Earth')
    obs_pos = model.get_observer(observer)

    # find the model coords of Earth as a function of time
    if model.frame == 'sidereal':
        deltalon = earth_pos.lon_hae - earth_pos.lon_hae[0]
        model_lon_earth = H._zerototwopi_(earth_pos.lon.value + deltalon.value)
    elif model.frame == 'synodic':
        model_lon_earth = earth_pos.lon.value

        # find the model coords of the given osberver as a function of time
    deltalon = obs_pos.lon_hae - earth_pos.lon_hae
    model_lon_obs = H._zerototwopi_(model_lon_earth + deltalon.value)

    if model.nlon == 1 and not suppress_warning:
        print('Single longitude simulated. Extracting time series at Observer r')

    time = np.ones(model.nt_out) * np.nan
    mjd = np.ones(model.nt_out) * np.nan
    lon = np.ones(model.nt_out) * np.nan
    rad = np.ones(model.nt_out) * np.nan
    speed = np.ones(model.nt_out) * np.nan
    bpol = np.ones(model.nt_out) * np.nan

    for t in range(model.nt_out):
        time[t] = (model.time_init + model.time_out[t]).jd
        mjd[t] = (model.time_init + model.time_out[t]).mjd

        # find the nearest longitude cell
        model_lons = model.lon.value
        if model.nlon == 1:
            model_lons = np.array([model_lons])
        id_lon = np.argmin(np.abs(model_lons - model_lon_obs[t]))

        # check whether the observer is within the model domain
        if ((obs_pos.r[t].value < model.r[0].value) or
                (obs_pos.r[t].value > model.r[-1].value) or
                (
                        (abs(model_lons[id_lon] - model_lon_obs[t]) > model.dlon.value) and
                        (abs(model_lons[id_lon] + 2 * np.pi - model_lon_obs[t]) > model.dlon.value)
                )
        ):

            bpol[t] = np.nan
            speed[t] = np.nan
            print('Outside model domain')
        else:
            # find the nearest R coord
            id_r = np.argmin(np.abs(model.r.value - obs_pos.r[t].value))
            rad[t] = model.r[id_r].value
            lon[t] = model_lon_obs[t]
            # then interpolate the values in longitude
            if model.nlon == 1:
                speed[t] = model.v_grid[t, id_r, 0].value
                if hasattr(model, 'b_grid'):
                    bpol[t] = model.b_grid[t, id_r, 0]
            else:
                speed[t] = np.interp(model_lon_obs[t], model.lon.value, model.v_grid[t, id_r, :].value,
                                     period=2 * np.pi)
                if hasattr(model, 'b_grid'):
                    bpol[t] = np.interp(model_lon_obs[t], model.lon.value, model.b_grid[t, id_r, :], period=2 * np.pi)

    time = pd.to_datetime(time, unit='D', origin='julian')

    time_series = pd.DataFrame(data={'time': time, 'r': rad, 'lon': lon, 
                                     'vsw': speed, 'bpol': bpol, 'mjd': mjd})
    return time_series


def plot_earth_timeseries(model, plot_omni=True):
    """
    A function to plot the HUXt Earth time series. With option to download and plot OMNI data.
    Args:
        model : input model class
        plot_omni: Boolean, if True downloads and plots OMNI data

    Returns:
        fig : Figure handle
        axs : Axes handles

    """

    huxt_ts = get_observer_timeseries(model, observer='Earth')

    # 2-panel plot if the B polarity has been traced
    if hasattr(model, 'b_grid'):
        fig, axs = plt.subplots(2, 1, figsize=(14, 7))
        axs[1].plot(huxt_ts['time'], np.sign(huxt_ts['bpol']), 'k.', label='HUXt')
        axs[1].set_ylabel('B polarity')
    else:
        fig, axs = plt.subplots(1, 1, figsize=(14, 4))
        axs = np.array([axs])

    axs[0].plot(huxt_ts['time'], huxt_ts['vsw'], 'k', label='HUXt')
    axs[0].set_ylim(250, 1000)

    starttime = huxt_ts['time'][0]
    endtime = huxt_ts['time'][len(huxt_ts) - 1]

    if plot_omni:
        # grab the omni data
        data = get_omni(starttime, endtime)
        # plot the period of interest
        mask = (data['datetime'] >= starttime) & (data['datetime'] <= endtime)
        plotdata = data[mask]
        axs[0].plot(plotdata['datetime'], plotdata['V'], 'r', label='OMNI')

        if hasattr(model, 'b_grid'):
            axs[1].plot(plotdata['datetime'], -np.sign(plotdata['BX_GSE']) * 0.92, 'r.', label='OMNI')
            axs[1].set_ylim(-1.1, 1.1)

    for a in axs:
        a.set_xlim(starttime, endtime)
        a.legend()

    axs[0].set_ylabel('Solar Wind Speed (km/s)')

    if axs.size == 1:
        axs[0].set_xlabel('Date')
    elif axs.size == 2:
        axs[0].set_xticklabels([])
        axs[1].set_xlabel('Date')

    fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)

    return fig, axs


@u.quantity_input(time=u.day)
def plot3d_radial_lat_slice(model3d, time, lon=np.NaN * u.deg, save=False, tag='', fighandle=np.nan, axhandle=np.nan):
    """
    Make a contour plot on polar axis of a radial-latitudinal plane of the solar wind solution at a fixed time and
    longitude.
    Args:
        model3d: An instance of the HUXt3d class with a completed solution.
        time: Time to look up closet model time to (with an astropy.unit of time).
        lon: The longitude along which to render the radial-latitude plane.
        save: Boolean to determine if the figure is saved.
        tag: String to append to the filename if saving the figure.
    Returns:
        fig: Figure handle.
        ax: Axes handle.
    """
    plotvmin = 200
    plotvmax = 810
    dv = 10
    ylab = "Solar Wind Speed (km/s)"

    # get the metadata from one of the individual HUXt elements
    model = model3d.HUXtlat[0]

    if (time < model.time_out.min()) | (time > (model.time_out.max())):
        print("Error, input time outside span of model times. Defaulting to closest time")

    id_t = np.argmin(np.abs(model.time_out - time))

    # get the requested longitude
    if model.lon.size == 1:
        id_lon = 0
        lon_out = model.lon.to(u.deg)
    else:
        id_lon = np.argmin(np.abs(model.lon - lon))
        lon_out = model.lon[id_lon].to(u.deg)

    # loop over latitudes and extract the radial profiles
    mercut = np.ones((len(model.r), model3d.nlat))
    for n in range(0, model3d.nlat):
        model = model3d.HUXtlat[n]
        mercut[:, n] = model.v_grid[id_t, :, id_lon]

    orig_cmap = mpl.cm.viridis
    # make a copy
    mymap = type(orig_cmap)(orig_cmap.colors)

    mymap.set_over('lightgrey')
    mymap.set_under([0, 0, 0])
    levels = np.arange(plotvmin, plotvmax + dv, dv)
    
    # if no fig and axis handles are given, create a new figure
    if isinstance(fighandle, float):
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    else:
        fig = fighandle
        ax = axhandle

    cnt = ax.contourf(model3d.lat.to(u.rad), model.r, mercut, levels=levels, cmap=mymap, extend='both')

    # Set edge color of contours the same, for good rendering in PDFs
    for c in cnt.collections:
        c.set_edgecolor("face")

    # Trace the CME boundaries
    cme_colors = ['r', 'c', 'm', 'y', 'deeppink', 'darkorange']
    for n in range(0, len(model.cmes)):

        # Get latitudes 
        lats = model3d.lat

        cme_r_front = np.ones(model3d.nlat) * np.nan
        cme_r_back = np.ones(model3d.nlat) * np.nan
        for ilat in range(0, model3d.nlat):
            model = model3d.HUXtlat[ilat]

            cme_r_front[ilat] = model.cme_particles_r[n, id_t, 0, id_lon]
            cme_r_back[ilat] = model.cme_particles_r[n, id_t, 1, id_lon]

        # trim the nans
        # Find indices that sort the longitudes, to make a wraparound of lons
        id_sort_inc = np.argsort(lats)
        id_sort_dec = np.flipud(id_sort_inc)

        cme_r_front = cme_r_front[id_sort_inc]
        cme_r_back = cme_r_back[id_sort_dec]

        lat_front = lats[id_sort_inc]
        lat_back = lats[id_sort_dec]

        # Only keep good values
        id_good = np.isfinite(cme_r_front)
        if id_good.any():
            cme_r_front = cme_r_front[id_good]
            lat_front = lat_front[id_good]

            id_good = np.isfinite(cme_r_back)
            cme_r_back = cme_r_back[id_good]
            lat_back = lat_back[id_good]

            # Get one array of longitudes and radii from the front and back particles
            lats = np.hstack([lat_front, lat_back, lat_front[0]])
            cme_r = np.hstack([cme_r_front, cme_r_back, cme_r_front[0]])

            ax.plot(lats.to(u.rad), (cme_r * u.km).to(u.solRad), color=cme_colors[n], linewidth=3)

    # determine which bodies should be plotted
    plot_observers = zip(['EARTH', 'VENUS', 'MERCURY', 'STA', 'STB'], ['ko', 'mo', 'co', 'rs', 'y^'])
    if model.r[0] > 200 * u.solRad:
        plot_observers = zip(['EARTH', 'MARS', 'JUPITER', 'SATURN'], ['ko', 'mo', 'ro', 'cs'])

    # Add on observers 
    for body, style in plot_observers:
        obs = model.get_observer(body)
        deltalon = 0.0 * u.rad

        # adjust body longitude for the frame
        if model.frame == 'sidereal':
            earth_pos = model.get_observer('EARTH')
            deltalon = earth_pos.lon_hae[id_t] - earth_pos.lon_hae[0]

        bodylon = H._zerototwopi_(obs.lon[id_t] + deltalon) * u.rad
        # plot bodies that are close to being in the plane
        if abs(bodylon - lon_out) < model.dlon * 2:
            ax.plot(obs.lat[id_t], obs.r[id_t], style, markersize=16, label=body)

    # Add on a legend.
    fig.legend(ncol=5, loc='lower center', frameon=False, handletextpad=0.2, columnspacing=1.0)

    ax.patch.set_facecolor('slategrey')
    fig.subplots_adjust(left=0.05, bottom=0.16, right=0.95, top=0.99)

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
    cbar1.set_ticks(np.arange(plotvmin, plotvmax, dv * 10))

    # Add label
    label = "   Time: {:3.2f} days".format(model.time_out[id_t].to(u.day).value)
    label = label + '\n ' + (model.time_init + time).strftime('%Y-%m-%d %H:%M')
    fig.text(0.70, pos.y0, label, fontsize=16)

    label = "HUXt3D \nLong: {:3.1f} deg".format(lon_out.to(u.deg).value)
    fig.text(0.175, pos.y0, label, fontsize=16)

    if save:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_frame_{:03d}.png".format(cr_num, tag, id_t)
        filepath = os.path.join(model._figure_dir_, filename)
        fig.savefig(filepath)

    return fig, ax


def animate_3d(model3d, lon=0.0 * u.deg, tag='', duration=10, fps=20, outputfilepath=''):
    """
    Animate the model solution, and save as an MP4.
    Args:
        model3d: An instance of HUXt3d
        lon: The longitude along which to render the latitudinal slice.
        tag: String to append to filename when saving the animation.
        duration: the movie duration, in seconds
        fps: frames per second
        outputfilepath: full path, including filename if output is to be saved anywhere other than huxt/figures
    Returns:
        None
    """
    model = model3d.HUXtlat[0]
    
    interval = (1/fps)*1000
    nframes = int(duration*1000/interval)
    
    exp_time = int(nframes*0.2)
    print('Rendering ' + str(nframes) + ' frames. Expected time: ' + str(exp_time) + ' secs')
    
    def make_frame3d(frame):
        """v
        Produce the frame required by MoviePy.VideoClip.
        Args:
            frame: frame number of the movie
        Returns:
            frame: An image array for rendering to movie clip.
        """
        plt.clf()  # Clear the previous frame
        ax = fig.add_subplot(111, projection='polar')
        
        # Get the time index closest to this fraction of movie duration
        i = np.int32((model.nt_out - 1) * frame / nframes)
        plot3d_radial_lat_slice(model3d, model.time_out[i], lon, 
                                fighandle=fig, axhandle=ax)
        return frame
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    
    # Create the animation
    ani = FuncAnimation(fig, make_frame3d, frames=range(nframes), interval=interval)
    
    if outputfilepath:
        filepath = outputfilepath
    else:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_movie.mp4".format(cr_num, tag)
        filepath = os.path.join(model._figure_dir_, filename)
    
    # Save the animation as a movie file
    ani.save(filepath, writer='ffmpeg')
    print('mp4 file written to ' + filepath)
    
    return

@u.quantity_input(time=u.day)
def plot_bpol(model, time, save=False, tag='', fighandle=np.nan, axhandle=np.nan, minimalplot=False, plotHCS=True):
    """
    Make a contour plot on polar axis of the solar wind solution at a specific time.
    Args:
        model: An instance of the HUXt class with a completed solution.
        time: Time to look up closet model time to (with an astropy.unit of time).
        save: Boolean to determine if the figure is saved.
        tag: String to append to the filename if saving the figure.
        fighandle: Figure handle for placing plot in a figure that already exists.
        axhandle: Axes handle for placing plot in axes that already exists.
        minimalplot: removes colorbar, planets/spacecraft and labels
        plotHCS: Boolean to determine if the heliospheric current sheet locations are plotted.
    Returns:
        fig: Figure handle.
        ax: Axes handle.
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

    # Set edge color of contours the same, for good rendering in PDFs
    for c in cnt.collections:
        c.set_edgecolor("face")

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

        # determine which bodies should be plotted
        plot_observers = zip(['EARTH', 'VENUS', 'MERCURY', 'STA', 'STB'], ['ko', 'mo', 'co', 'rs', 'y^'])
        if model.r[0] > 200 * u.solRad:
            plot_observers = zip(['EARTH', 'MARS', 'JUPITER', 'SATURN'], ['ko', 'mo', 'ro', 'cs'])
        # Add on observers 
        for body, style in plot_observers:
            obs = model.get_observer(body)
            deltalon = 0.0 * u.rad
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

        # plot any tracked streaklines
        if model.track_streak:
            nstreak = len(model.streak_particles_r[0, :, 0, 0])
            for istreak in range(0, nstreak):
                # construct the streakline from multiple rotations
                nrot = len(model.streak_particles_r[0, 0, :, 0])
                streak_r = []
                streak_lon = []
                for irot in range(0, nrot):
                    streak_lon = streak_lon + model.lon.value.tolist()
                    streak_r = streak_r + (
                            model.streak_particles_r[id_t, istreak, irot, :] * u.km.to(u.solRad)).value.tolist()

                    # add the inner boundary postion too
                mask = np.isfinite(streak_r)
                plotlon = np.array(streak_lon)[mask]
                plotr = np.array(streak_r)[mask]
                # only add the inner boundary if it's in the HUXt longitude grid
                foot_lon = H._zerototwopi_(model.streak_lon_r0[id_t, istreak])
                dlon_foot = abs(model.lon.value - foot_lon)
                if dlon_foot.min() <= model.dlon.value:
                    plotlon = np.append(plotlon, foot_lon + model.dlon.value / 2)
                    plotr = np.append(plotr, model.r[0].to(u.solRad).value)

                ax.plot(plotlon, plotr, 'k')

        # plot any HCS that have been traced
        if plotHCS and hasattr(model, 'b_grid'):
            for i in range(0, len(model.hcs_particles_r[:, 0, 0, 0])):
                r = model.hcs_particles_r[i, id_t, 0, :] * u.km.to(u.solRad)
                lons = model.lon
                ax.plot(lons, r, 'k.')

    if save:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_frame_{:03d}.png".format(cr_num, tag, id_t)
        filepath = os.path.join(model._figure_dir_, filename)
        fig.savefig(filepath)

    return fig, ax


@jit(nopython=True)
def trace_field_line_out(v_trl_kms, longrid_rad, rgrid_km, tgrid_s, start_lon, time_start_s, time_stop_s, rot_period_s):
    """
    Trace a field line through an exixisting model run. 
    model must output with dt_scale = 1
    
    Args:
        v_trl_kms: model.v_grid.value - the speed as a funciton of time, radius and longitude
        longrid_rad: model.lon.to(u.rad).value - the longitude grid in radians
        rgrid_km: model.r.to(u.km).value - the radial grid in km
        tgrid_s: model.time_out.to(u.s).value - the time grid in seconds
        start_lon: The longitude, in HUXt coords, from which to start tracing
        time_start_s: The time from the start of the model run, in seconds, from which to start tracing
        time_stop_s: The time from thr start of the model run, in seconds, at which to stop tracing
        rot_period_s: the  HUXt rotation period, in seconds

    Returns:
        r_streak_km: An array of test particle distances for each longitude at given time step
    """

    # get the grid dimensions
    nt = len(tgrid_s)
    nlon = len(longrid_rad)
    nr = len(rgrid_km)
    
    # check the dimensions of the grid
    assert (len(v_trl_kms[:, 0, 0] == nt))
    assert (len(v_trl_kms[0, :, 0] == nr))
    assert (len(v_trl_kms[0, 0, :] == nlon))
    
    dt_phi_s = rot_period_s / nlon
    dt_s = tgrid_s[1] - tgrid_s[0]

    # create the main variable
    r_streak_km = np.ones((nt, nlon)) * np.nan
    
    # find the start time index
    id_t_start = np.argmin(np.abs(tgrid_s - time_start_s))
    id_t_stop = np.argmin(np.abs(tgrid_s - time_stop_s))
    # find the start lon index
    id_lon_start = np.argmin(np.abs(longrid_rad - start_lon))
    
    id_t = id_t_start
    id_lon = id_lon_start
    while id_t < nt:
        # stick a particle at the inner boundary
        r_streak_km[id_t, id_lon] = rgrid_km[0]
        
        # find the time when the next longitude will need a test particle
        id_t = id_t + int(dt_phi_s/dt_s)
        id_lon = id_lon + 1
        if id_lon >= nlon:
            id_lon = 0
    
    # move each particle  forward
    for it in range(id_t_start, id_t_stop):  # loop through the required time range
        for ilon in range(0, nlon):  # loop through each longitude and move the particles forward
            if ~np.isnan(r_streak_km[it, ilon]):
                v_test_kms = np.interp(r_streak_km[it, ilon], rgrid_km, v_trl_kms[it, :, ilon])
                r_streak_km[it+1, ilon] = r_streak_km[it, ilon] + v_test_kms * dt_s
                
                if  r_streak_km[it+1, ilon] > rgrid_km[-1]:
                    r_streak_km[it+1, ilon] = np.nan
     
    return r_streak_km[id_t_stop,:]
        

@jit(nopython=True)
def min_distance_streakline_point(streak_lon_rad, streak_r_km, point_lon_rad, point_r_km, d=5000):
    """
    Return the minimum distance between a given field line and a fixed point (e.g. Earth)
    
    Args:
        streak_lon_rad: Longitudes of fieldline
        streak_r_km: Radial distances of fiedline
        point_lon_rad: longitude of fixed point
        point_r_km: radial distance of fixed point
        d: resolution at which to interpolate the field line (in km)

    Returns:
        distance: minimum distance, in km
        r: radial distance of closest point on fieldline, in km
        theta: longitude of closest point on fieldine, in radians
    """
    
    # convert the fieldline points to cartesean
    # Convert polar coordinates to Cartesian coordinates
    x = streak_r_km * np.cos(streak_lon_rad)
    y = streak_r_km * np.sin(streak_lon_rad)
    
    # get Earth poisition in Cartesean
    Ex = point_r_km * np.cos(point_lon_rad)
    Ey = point_r_km * np.sin(point_lon_rad)
    
    # Calculate cumulative distances between consecutive points
    dist_along_line = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cumulative_distances = np.nancumsum(dist_along_line)
    
    # Pad cumulative_distances with a zero at the beginning
    padded_cumulative_distances = np.zeros(len(cumulative_distances) + 1)
    padded_cumulative_distances[1:] = cumulative_distances

    # Calculate total length of the line
    total_length = padded_cumulative_distances[-1]
    # Determine the number of interpolated points
    num_interpolated_points = int(total_length / d)
    
    # Interpolate points along the line
    intx = np.interp(np.linspace(0, total_length, num_interpolated_points + 1),
                               padded_cumulative_distances, x)
    inty = np.interp(np.linspace(0, total_length, num_interpolated_points + 1),
                               padded_cumulative_distances, y)

    
    # find closest point to Earth
    distances = np.sqrt((intx - Ex)**2 + (inty - Ey)**2)
    
    # Replace NaN values with a large value
    distances[np.isnan(distances)] = np.inf
    i = np.argmin(distances)
    
    # convert the closest point back to r, lon
    r = np.sqrt(intx[i]**2 + inty[i]**2)
    # theta = H._zerototwopi_(
    theta = np.arctan2(inty[i], intx[i])
    
    return distances[i], r, theta


@jit(nopython=True)
def respinup_model(v_trl_kms, tgrid_s, rgrid_km, longrid_rad, rot_period_s, buffer_time_s):
    """
    recreate steady-state solar wind conditions during the spin-up period 
    to enable field-line tracing near the start of a model run
    
    Args:
        v_trl_kms: model.v_grid.value - 
        longrid_rad: model.lon.to(u.rad).value - the longitude grid in radians
        rgrid_km: model.r.to(u.km).value - the radial grid in km
        tgrid_s: model.time_out.to(u.s).value - the time grid in seconds
        start_lon: The longitude, in HUXt coords, from which to start tracing
        rot_period_s: the  HUXt rotation period, in seconds
        buffer_time_s: How back to take the model before the start time, in seconds

    Returns:
        new_v_trl_kms: the speed as a funciton of time, radius and longitude, 
                    for both the spint-up and model run period
        new_tgrid_s: the new time grid. spin-up period has negative times.
    """

    dt_s = tgrid_s[1] - tgrid_s[0]
    nlon = len(longrid_rad)
    nr = len(rgrid_km)

    # get the exact starting time
    nsteps = int(np.ceil(buffer_time_s/dt_s))
    spinup_tgrid_s = np.arange(-nsteps*dt_s,0, dt_s)
    new_tgrid_s = np.append(spinup_tgrid_s, tgrid_s)

    new_v_trl_kms = np.ones((len(new_tgrid_s), nr, nlon))
    # put the existing data in
    new_v_trl_kms[nsteps:, :, :] = v_trl_kms[:, :, :]

    for t in range(0,len(spinup_tgrid_s)):
        dt = -spinup_tgrid_s[t]
        dlon = np.mod(2*np.pi * dt / rot_period_s, 2*np.pi)
        this_lons = np.mod(longrid_rad + dlon, 2*np.pi)
        for r in range(0,nr):
            new_v_trl_kms[t, r, :] = np.interp(this_lons, longrid_rad, v_trl_kms[0, r, :])

    return new_v_trl_kms, new_tgrid_s


@jit(nopython=True)
def _return_distance_for_given_t_(t, start_lon, v_trl_kms=np.nan, longrid_rad=np.nan, rgrid_km=np.nan,
                                  tgrid_s=np.nan, time_stop_s=np.nan, Earth_lon_rad=np.nan, Earth_r_km=np.nan,
                                  rot_period_s=np.nan):

    """
    Function to be minimised. finds the closest time step for a given longitude
    """
    
    # make sure the longitude is between 0 and 2 pi
    start_lon = np.mod(start_lon, 2*np.pi)

    # first trace the field line
    r_streak = trace_field_line_out(v_trl_kms, longrid_rad, rgrid_km, tgrid_s, start_lon, t, time_stop_s, rot_period_s)

    #  the longitude points starting at the initial lon
    rel_lons = np.mod( longrid_rad - start_lon, 2*np.pi)
    sort_indices = np.argsort(rel_lons)
    plotlon = longrid_rad[sort_indices]
    plotr= r_streak[sort_indices]

    # then compute the distance to Earth
    dist, r, theta = min_distance_streakline_point(plotlon, plotr, Earth_lon_rad, Earth_r_km)
    
    return dist


@jit(nopython=True)
def _return_distance_for_given_lon_(start_lon, t, v_trl_kms=np.nan, longrid_rad=np.nan, rgrid_km=np.nan, tgrid_s=np.nan,
                                    time_stop_s=np.nan, Earth_lon_rad=np.nan, Earth_r_km=np.nan, rot_period_s=np.nan):
    """
    Function to be minimised. finds the closest longitude for a given timestep
    """
    
    # make sure the longitude is between 0 and 2 pi
    start_lon = np.mod(start_lon, 2*np.pi)
    
    # first trace the field line
    r_streak = trace_field_line_out(v_trl_kms, longrid_rad, rgrid_km, tgrid_s, start_lon, t, time_stop_s, rot_period_s)

    # order the longitude points starting at the initial lon
    rel_lons = np.mod(longrid_rad - start_lon, 2*np.pi)
    sort_indices = np.argsort(rel_lons)
    plotlon = longrid_rad[sort_indices]
    plotr = r_streak[sort_indices]

    # then compute the distance to Earth
    dist, r, theta = min_distance_streakline_point(plotlon, plotr, Earth_lon_rad, Earth_r_km)
    
    return dist


def find_Earth_connected_field_line(model, time):
    """
    Locate the Earth connected field line for a completed model run at a given model time
    Re-spins the model if necessary
    
    Args:
        model: The HUXt model class for the solved run. Must be output at dt_scale = 1
        time: The model time at which to trace the field line
    Returns:
        plotlon: The longitudes of the Earth-connected field line (in radians)
        plotr: The radial distances of the Earth-connected field line (in solar radii, for plotting) 
        optimal_lon: The model longitude at the field line start, in radians 
        optimal_t: The time step at the field line start, in seconds from start of model run
    """

    assert (model.dt_scale == 1)

    buffertime = 6*u.day  # the tracing this time before the ballistic start time estimate
    buffer_time_s = buffertime.to(u.s).value
    time_s = time.to(u.s).value
    
    rgrid_km = model.r.to(u.km).value
    tgrid_s = model.time_out.to(u.s).value
    longrid_rad = model.lon.to(u.rad).value
    v_trl_kms = model.v_grid.to(u.km/u.s).value
    rot_period_s = model.rotation_period.to(u.s).value
    
    # get the current Earth position, ignore spin up if it's been added
    id_t = np.argmin(np.abs(tgrid_s - time_s))
    earth_pos = model.get_observer('Earth')
    r_Earth_km = earth_pos.r[id_t].to(u.km).value
    if model.frame == 'synodic':
        lon_Earth_rad = earth_pos.lon[id_t].to(u.rad).value
    elif model.frame == 'sidereal':
        lon_Earth_t = earth_pos.lon_hae[id_t]
        lon_Earth_0 = earth_pos.lon_hae[0]
        lon_Earth_rad = H._zerototwopi_(lon_Earth_t - lon_Earth_0)
    
    # check if fieldline tracing will need to start before the model run start
    if time < buffertime:
        # see if it's already spun up.
        if hasattr(model, 'v_grid_spunup'):
            v_trl_kms = model.v_grid_spunup
            tgrid_s = model.time_spunup
        else:
            v_trl_kms, tgrid_s = respinup_model(v_trl_kms, tgrid_s, rgrid_km, longrid_rad, rot_period_s, buffer_time_s)
            
            # store the data in the model class, so it doesn't have to be spun-up again
            model.v_grid_spunup = v_trl_kms
            model.time_spunup = tgrid_s

    time_start_s = (time - buffertime).to(u.s).value
    start_lon = H._zerototwopi_(lon_Earth_rad - 1)
    
    # first minimise the longitude for a fixed time
    result = minimize(_return_distance_for_given_lon_, x0=start_lon,
                      args=(time_start_s, v_trl_kms, longrid_rad, rgrid_km, tgrid_s, time_s, lon_Earth_rad, r_Earth_km,
                            rot_period_s),
                      method='Nelder-Mead')
    
    optimal_params = result.x
    optimal_lon = optimal_params[0]
    
    # then minimise t for the fixed lon
    result = minimize(_return_distance_for_given_t_, x0=time_start_s,
                      args=(optimal_lon, v_trl_kms, longrid_rad, rgrid_km, tgrid_s, time_s, lon_Earth_rad, r_Earth_km,
                            rot_period_s),
                      method='Nelder-Mead')

    optimal_params = result.x
    optimal_t = optimal_params[0]
    
    rstreak = trace_field_line_out(v_trl_kms, longrid_rad, rgrid_km, tgrid_s, optimal_lon, optimal_t, time_s,
                                   rot_period_s)
    
    # make the field line nice for plotting
    # ====================================
    
    # order  points in increasing angle from the initial value
    rel_lon = H._zerototwopi_(longrid_rad - start_lon)
    sort_indices = np.argsort(rel_lon)
    plotlon = longrid_rad[sort_indices]
    plotr = rstreak[sort_indices]
    
    # remove nans
    mask = np.isfinite(plotr)
    plotlon = plotlon[mask]
    plotr = plotr[mask]
    
    # #add in the connection to the inner boundary
    r_min = model.r[0].to(u.km).value
    dr = plotr[-1] - r_min
    plotr = np.append(plotr, r_min)
    # compute the long of the footpoint assuming a constant solar wind speed
    dt = (dr * u.km / (350 * u.km / u.s)).to(u.s)
    dlon = (2*np.pi)*(dt/model.rotation_period).value 
    plotlon = np.append(plotlon, H._zerototwopi_(plotlon[-1] + dlon))

    return plotlon, (plotr*u.km).to(u.solRad).value, optimal_lon, optimal_t
