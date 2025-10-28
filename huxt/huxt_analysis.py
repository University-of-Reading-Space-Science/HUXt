from appdirs import user_data_dir
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from pathlib import Path
from numba import jit
from scipy.optimize import minimize
import sunpy
from sunpy.coordinates import get_horizons_coord

from . import huxt as h
from . import huxt_inputs as hin

mpl.rc("axes", labelsize=16)
mpl.rc("ytick", labelsize=16)
mpl.rc("xtick", labelsize=16)
mpl.rc("legend", fontsize=16)


def get_figure_dir():
    """Get path to output directory for figures and animations"""
    figure_dir = Path(user_data_dir(appname='huxt', appauthor=False), "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


@u.quantity_input(time=u.day)
def plot(model, time, save=False, tag='', fighandle=np.nan, axhandle=np.nan, minimalplot=False, plotHCS=True,
         annotateplot=True, trace_earth_connection=False, plot_rmax=None):
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
        annotateplot: Boolean, whether to include text and legends
        trace_earth_connection: boolean, whether to plot Earth-connected field. Slow.
        plot_rmax: float (no units, but in rS). Limit outer boundary to help with field lines during CMEs
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
    lon_arr, dlon, nlon = h.longitude_grid()
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
        v = np.zeros((model.nr, nlon)) * np.nan
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
        cme_colors = get_cme_colors()
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
        planet_list = get_planets_to_plot(model)
        spacecraft_list = get_spacecraft_to_plot(model)
        observers_list = planet_list + spacecraft_list

        # Add on observers
        styles = observer_styles()
        for body in observers_list:
            obs = model.get_observer(body)
            deltalon = 0.0 * u.rad
            if model.frame == 'sidereal':
                earth_pos = model.get_observer('EARTH')
                deltalon = earth_pos.lon_hae[id_t] - earth_pos.lon_hae[0]

            obslon = zerototwopi(obs.lon[id_t] + deltalon)
            ax.plot(obslon, obs.r[id_t], markersize=14, color=styles[body]['color'], marker=styles[body]['marker'],
                    linestyle='', label=body)

        # Add on a legend.
        if annotateplot:
            ax.legend(ncol=len(observers_list), loc='lower center', frameon=False, fontsize=14,
                      handletextpad=0.1, columnspacing=0.5,  bbox_to_anchor=(0.5, -0.18))
            
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
            r_max = model.r[-1].to(u.solRad).value
            dr = (model.r[1] - model.r[0]).to(u.solRad).value  # Grid spacing
            
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
                streak_lon = np.array(streak_lon)
                streak_r = np.array(streak_r)
                
                # Filter: keep only finite values that are well within the domain
                # Exclude particles within 2*dr of the outer boundary to avoid awkward connections
                mask = np.isfinite(streak_r) & (streak_r < (r_max - 2*dr))
                plotlon = streak_lon[mask]
                plotr = streak_r[mask]
                
                if len(plotr) > 0:
                    # for plotting only, fix the inner most point on the inner bounday.
                    r_min = model.r[0].to(u.solRad).value
                    if plotr[-1] > r_min:
                        dr_inner = plotr[-1] - r_min
                        # compute the long of the footpoint assuming a constant solar wind speed
                        dt = (dr_inner * u.solRad / (350 * u.km / u.s)).to(u.s)
                        dlon_streak = (2*np.pi)*(dt/model.rotation_period).value 
                        inner_lon = zerototwopi(plotlon[-1] + dlon_streak)
                        # check that this new longitude was actually simulated
                        if np.nanmin(abs(model.lon - inner_lon*u.rad)) < dlon:
                            plotr = np.append(plotr, r_min)
                            plotlon = np.append(plotlon, inner_lon)
                    
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
        
    if plot_rmax:
        ax.set_rmax(plot_rmax)

    if save:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_frame_{:03d}.png".format(cr_num, tag, id_t)
        figure_dir = get_figure_dir()
        filepath = figure_dir.joinpath(filename)
        fig.savefig(filepath)

    return fig, ax


def animate(model, tag, duration=10, fps=20, plotHCS=True, trace_earth_connection=False, outputfilepath='',
            plot_rmax=None):
    """
    Animate the model solution, and save as an MP4.
    Args:
        model: An instance of the HUXt class with a completed solution.
        tag: String to append to the filename of the animation.
        duration: the movie duration, in seconds
        fps: frames per second
        plotHCS: Boolean flag on whether to plot the heliospheric current sheet location.
        trace_earth_connection: Boolean flag on whether to plot the earth connected streak line.
        outputfilepath: full path, including filename if output is to be saved anywhere other than huxt/figures
        plot_rmax: float (no units, but in rS). Limit outer boundary to help with field lines during CMEs
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
            frame: number of this frame
        Returns:
            frame: An image array for rendering to movie clip.
        """
        plt.clf()  # Clear the previous frame
        
        # Get the time index closest to this fraction of movie duration
        i = np.int32((model.nt_out - 1) * frame / nframes)
        
        # Use plot_compressible for compressible models, otherwise use standard plot
        if hasattr(model, 'compressible') and model.compressible:
            plot_compressible(model, model.time_out[i], fighandle=fig, minimalplot=False, 
                            annotateplot=True, plot_rmax=plot_rmax)
        else:
            ax = fig.add_subplot(111, projection='polar')
            plot(model, model.time_out[i], fighandle=fig, axhandle=ax, plotHCS=plotHCS,
                 trace_earth_connection=trace_earth_connection, plot_rmax=plot_rmax)
        return frame
    
    # Create a new figure - size depends on compressible mode
    if hasattr(model, 'compressible') and model.compressible:
        fig, ax = plt.subplots(figsize=(24, 8))
    else:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    
    # Create the animation
    ani = FuncAnimation(fig, make_frame, frames=range(nframes), interval=interval)
    
    # set up the save path
    if outputfilepath:
        filepath = outputfilepath
    else:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_movie.mp4".format(cr_num, tag)
        figure_dir = get_figure_dir()
        filepath = figure_dir.joinpath(filename)

    # Save the animation as a movie file
    ani.save(filepath, writer='ffmpeg')
    print('mp4 file written to ' + str(filepath))

    return


@u.quantity_input(time=u.day)
def plot_compressible(model, time, save=False, tag='', fighandle=np.nan, minimalplot=False, 
                      annotateplot=True, plot_rmax=None):
    """
    Make three contour plots on polar axes of the compressible solar wind solution at a specific time.
    Shows velocity, density, and temperature in separate subplots.
    
    Args:
        model: An instance of the HUXt class with a completed compressible solution.
        time: Time to look up closest model time to (with an astropy.unit of time).
        save: Boolean to determine if the figure is saved.
        tag: String to append to the filename if saving the figure.
        fighandle: Figure handle for placing plot in existing figure.
        minimalplot: Boolean, if True removes colorbar, planets, spacecraft, and labels.
        annotateplot: Boolean, whether to include text and legends
        plot_rmax: float (no units, but in Rs). Limit outer boundary to help with field lines during CMEs
    Returns:
        fig: Figure handle.
        axes: Array of axes handles [ax_v, ax_rho, ax_T].
    """
    
    if not hasattr(model, 'rho_grid') or not hasattr(model, 'temp_grid'):
        raise ValueError("Model must be run with compressible=True to use plot_compressible")

    if (time < model.time_out.min()) | (time > (model.time_out.max())):
        print("Error, input time outside span of model times. Defaulting to closest time")

    # Create figure with 3 subplots
    if isinstance(fighandle, float):
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={"projection": "polar"})
    else:
        fig = fighandle
        axes = [fig.add_subplot(1, 3, i+1, projection='polar') for i in range(3)]

    id_t = np.argmin(np.abs(model.time_out - time))

    # Get plotting data
    lon_arr, dlon, nlon = h.longitude_grid()
    lon, rad = np.meshgrid(lon_arr.value, model.r.value)

    # Prepare data arrays for velocity, density, and temperature
    v_sub = model.v_grid.value[id_t, :, :].copy()
    rho_sub = model.rho_grid.value[id_t, :, :].copy()
    temp_sub = model.temp_grid.value[id_t, :, :].copy()
    
    # Convert density to number density (protons/cm³)
    m_p = 1.6726e-27  # kg
    n_sub = rho_sub / m_p / 1e6
    
    # Insert into full array
    if lon_arr.size != model.lon.size:
        v = np.zeros((model.nr, nlon)) * np.nan
        n = np.zeros((model.nr, nlon)) * np.nan
        temp = np.zeros((model.nr, nlon)) * np.nan
        if model.lon.size != 1:
            for i, lo in enumerate(model.lon):
                id_match = np.argwhere(lon_arr == lo)[0][0]
                v[:, id_match] = v_sub[:, i]
                n[:, id_match] = n_sub[:, i]
                temp[:, id_match] = temp_sub[:, i]
        else:
            print('Warning: Trying to contour single radial solution will fail.')
    else:
        v = v_sub
        n = n_sub
        temp = temp_sub

    # Pad out to fill the full 2pi of contouring
    pad = lon[:, 0].reshape((lon.shape[0], 1)) + model.twopi
    lon = np.concatenate((lon, pad), axis=1)
    pad = rad[:, 0].reshape((rad.shape[0], 1))
    rad = np.concatenate((rad, pad), axis=1)
    
    pad_v = v[:, 0].reshape((v.shape[0], 1))
    v = np.concatenate((v, pad_v), axis=1)
    pad_n = n[:, 0].reshape((n.shape[0], 1))
    n = np.concatenate((n, pad_n), axis=1)
    pad_temp = temp[:, 0].reshape((temp.shape[0], 1))
    temp = np.concatenate((temp, pad_temp), axis=1)

    # Define colormaps and levels for each quantity
    # Velocity
    cmap_v = mpl.cm.viridis.copy()
    cmap_v.set_over('lightgrey')
    cmap_v.set_under([0, 0, 0])
    vmin, vmax, dv = 200, 810, 10
    levels_v = np.arange(vmin, vmax + dv, dv)
    
    # Density (log scale)
    cmap_n = mpl.cm.plasma.copy()
    cmap_n.set_over('white')
    cmap_n.set_under([0, 0, 0])
    # Use log10 of density for better visualization
    n_log = np.log10(n)
    nmin, nmax, dn = -1, 3, 0.1  # log10 scale: 0.1 to 1000 protons/cm³
    levels_n = np.arange(nmin, nmax + dn, dn)
    
    # Temperature (log scale)
    cmap_T = mpl.cm.inferno.copy()
    cmap_T.set_over('white')
    cmap_T.set_under([0, 0, 0])
    # Use log10 of temperature
    T_log = np.log10(temp)
    Tmin, Tmax, dT = 4, 6, 0.05  # log10 scale: 10^4 to 10^6 K
    levels_T = np.arange(Tmin, Tmax + dT, dT)

    # Plot 1: Velocity
    cnt_v = axes[0].contourf(lon, rad, v, levels=levels_v, cmap=cmap_v, extend='both')
    cnt_v.set(edgecolor="face")
    axes[0].set_ylim(0, model.r.value.max())
    axes[0].set_yticklabels([])
    axes[0].set_xticklabels([])
    axes[0].plot(0, 0, 'o', color=[1.0, 0.5, 0.25], markersize=16)
    # Title removed - info in colorbars instead
    
    # Plot 2: Density
    cnt_n = axes[1].contourf(lon, rad, n_log, levels=levels_n, cmap=cmap_n, extend='both')
    cnt_n.set(edgecolor="face")
    axes[1].set_ylim(0, model.r.value.max())
    axes[1].set_yticklabels([])
    axes[1].set_xticklabels([])
    axes[1].plot(0, 0, 'o', color=[1.0, 0.5, 0.25], markersize=16)
    # Title removed - info in colorbars instead
    
    # Plot 3: Temperature
    cnt_T = axes[2].contourf(lon, rad, T_log, levels=levels_T, cmap=cmap_T, extend='both')
    cnt_T.set(edgecolor="face")
    axes[2].set_ylim(0, model.r.value.max())
    axes[2].set_yticklabels([])
    axes[2].set_xticklabels([])
    axes[2].plot(0, 0, 'o', color=[1.0, 0.5, 0.25], markersize=16)
    # Title removed - info in colorbars instead

    # Add CME boundaries to all plots
    if model.track_cmes:
        cme_colors = get_cme_colors()
        for j, cme in enumerate(model.cmes):
            cid = np.mod(j, len(cme_colors))
            cme_lons = cme.coords[id_t]['lon']
            cme_r = cme.coords[id_t]['r'].to(u.solRad)
            if np.any(np.isfinite(cme_r)):
                cme_lons = np.append(cme_lons, cme_lons[0])
                cme_r = np.append(cme_r, cme_r[0])
                for ax in axes:
                    ax.plot(cme_lons, cme_r, '-', color=cme_colors[cid], linewidth=3)

    # Plot any tracked streaklines
    if model.track_streak:
        nstreak = len(model.streak_particles_r[0, :, 0, 0])
        r_max = model.r[-1].to(u.solRad).value
        dr = (model.r[1] - model.r[0]).to(u.solRad).value  # Grid spacing
        
        for istreak in range(0, nstreak):
            # Construct the streakline from multiple rotations
            nrot = len(model.streak_particles_r[0, 0, :, 0])
            streak_r = []
            streak_lon = []
            
            for irot in range(0, nrot):
                streak_lon = streak_lon + model.lon.value.tolist()
                streak_r = streak_r + (
                        model.streak_particles_r[id_t, istreak, irot, :] * u.km.to(u.solRad)).value.tolist()
                
            # Get the real values for plotting
            streak_lon = np.array(streak_lon)
            streak_r = np.array(streak_r)
            
            # Filter: keep only finite values that are well within the domain
            # Exclude particles within 2*dr of the outer boundary to avoid awkward connections
            mask = np.isfinite(streak_r) & (streak_r < (r_max - 2*dr))
            plotlon = streak_lon[mask]
            plotr = streak_r[mask]
            
            if len(plotr) > 0:
                # For plotting only, fix the innermost point on the inner boundary
                r_min = model.r[0].to(u.solRad).value
                if plotr[-1] > r_min:
                    dr_inner = plotr[-1] - r_min
                    # Compute the long of the footpoint assuming a constant solar wind speed
                    dt = (dr_inner * u.solRad / (350 * u.km / u.s)).to(u.s)
                    dlon_streak = (2*np.pi)*(dt/model.rotation_period).value 
                    inner_lon = zerototwopi(plotlon[-1] + dlon_streak)
                    # Check that this new longitude was actually simulated
                    if np.nanmin(abs(model.lon - inner_lon*u.rad)) < dlon:
                        plotr = np.append(plotr, r_min)
                        plotlon = np.append(plotlon, inner_lon)
                
            # Plot the streakline on all axes
            if len(plotr) > 0:
                for ax in axes:
                    ax.plot(plotlon, plotr, 'k')

    if not minimalplot:
        # Determine which bodies should be plotted
        planet_list = get_planets_to_plot(model)
        spacecraft_list = get_spacecraft_to_plot(model)
        observers_list = planet_list + spacecraft_list

        # Add observers to all plots
        styles = observer_styles()
        for body in observers_list:
            obs = model.get_observer(body)
            deltalon = 0.0 * u.rad
            if model.frame == 'sidereal':
                earth_pos = model.get_observer('EARTH')
                deltalon = earth_pos.lon_hae[id_t] - earth_pos.lon_hae[0]

            obslon = zerototwopi(obs.lon[id_t] + deltalon)
            # Plot on all axes, but only add label on first axis for legend
            for i, ax in enumerate(axes):
                label = body if i == 0 else None
                ax.plot(obslon, obs.r[id_t], markersize=14, color=styles[body]['color'], 
                       marker=styles[body]['marker'], linestyle='', label=label)
        
        # Set background color and adjust position
        # No titles now, so less space needed at top
        for ax in axes:
            ax.patch.set_facecolor('slategrey')
            pos = ax.get_position()
            # Adjust positioning with titles removed - can make plots bigger
            new_pos = [pos.x0, pos.y0 + 0.08, pos.width, pos.height * 0.90]
            ax.set_position(new_pos)

        # Add colorbars below each plot
        for i, (ax, cnt, label, ticks) in enumerate([
            (axes[0], cnt_v, r"$V_{SW}$ [km/s]", np.arange(vmin, vmax, dv * 10)),
            (axes[1], cnt_n, r"$\log_{10}(n)$ [protons/cm³]", np.arange(nmin, nmax, 1.0)),
            (axes[2], cnt_T, r"$\log_{10}(T)$ [K]", np.arange(Tmin, Tmax, 0.5))
        ]):
            pos = ax.get_position()
            dh = 0.035  # Vertical offset from plot
            cb_height = 0.02  # Colorbar height
            # Make colorbar same width as panel
            left = pos.x0
            bottom = pos.y0 - dh
            wid = pos.width
            cbaxes = fig.add_axes([left, bottom, wid, cb_height])
            cbar = fig.colorbar(cnt, cax=cbaxes, orientation='horizontal')
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(labelsize=16)
            # Position label below colorbar with larger font
            cbaxes.text(0.5, -2.2, label, fontsize=16, transform=cbaxes.transAxes, 
                       horizontalalignment='center', verticalalignment='top')

    # Add observer labels in a box below colorbars
    if annotateplot and not minimalplot and len(observers_list) > 0:
        # Position below the colorbar labels (closer to plots)
        label_y = 0.08
        
        styles = observer_styles()
        
        # Calculate approximate width per observer item (marker + text)
        # Each item: circle (0.015) + gap (0.008) + text (~0.007 per char)
        item_widths = []
        for body in observers_list:
            text_width = len(body) * 0.007  # Approximate width per character
            item_width = 0.015 + 0.008 + text_width  # marker + gap + text
            item_widths.append(item_width)
        
        # Space between items
        spacing = 0.02  # Small gap between items
        
        # Total width of all items plus spacing
        content_width = sum(item_widths) + spacing * (len(observers_list) - 1)
        
        # Add minimal padding
        box_padding = 0.015  # Small padding
        box_width = content_width + 2 * box_padding
        box_height = 0.025
        
        # Center everything
        box_x = 0.5 - box_width / 2
        content_start_x = box_x + box_padding
        
        # Draw a subtle box around the observer labels
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((box_x, label_y - 0.0125), box_width, box_height,
                            boxstyle="round,pad=0.003", 
                            edgecolor='gray', facecolor='white', alpha=0.8,
                            linewidth=1, transform=fig.transFigure, zorder=10)
        fig.patches.append(box)
        
        # Draw each observer label
        current_x = content_start_x
        for i, body in enumerate(observers_list):
            # Add colored circle marker
            fig.text(current_x + 0.0075, label_y, '●', fontsize=18, 
                    color=styles[body]['color'], 
                    horizontalalignment='center', verticalalignment='center', zorder=11)
            # Add body name - darker and bolder for visibility
            fig.text(current_x + 0.015 + 0.008, label_y, body.upper(), fontsize=13,
                    color='black', fontweight='bold',
                    horizontalalignment='left', verticalalignment='center', zorder=11)
            
            # Move to next position
            current_x += item_widths[i] + spacing
            
    if annotateplot:
        # Get positions of left and right panels for alignment
        pos_left = axes[0].get_position()
        pos_right = axes[2].get_position()
        
        # Add time label at top right, aligned with right edge of right panel
        # Reduced spacing from +0.10 to +0.05 to reduce whitespace
        time_label = "{:3.2f} days | ".format(model.time_out[id_t].to(u.day).value)
        time_label = time_label + (model.time_init + time).strftime('%Y-%m-%d %H:%M')
        fig.text(pos_right.x1, pos_right.y1 + 0.05, time_label, fontsize=15, fontweight='bold',
                horizontalalignment='right', verticalalignment='bottom')
        
        # Add model info at top left, aligned with left edge of left panel
        # Reduced spacing from +0.10 to +0.05 to reduce whitespace
        model_label = "HUXt2D Compressible | Lat: {:3.0f}°".format(model.latitude.to(u.deg).value)
        fig.text(pos_left.x0, pos_left.y1 + 0.05, model_label, fontsize=16, fontweight='bold',
                horizontalalignment='left', verticalalignment='bottom')

    if plot_rmax:
        for ax in axes:
            ax.set_rmax(plot_rmax)

    if save:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_compressible_CR{:03d}_{}_frame_{:03d}.png".format(cr_num, tag, id_t)
        figure_dir = get_figure_dir()
        filepath = figure_dir.joinpath(filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_radial(model, time, lon, save=False, tag=''):
    """
    Plot the radial solar wind profile at model time closest to specified time.
    For compressible models, shows velocity, density, and temperature.
    Args:
        model: An instance of the HUXt class with a completed solution.
        time: Time (in seconds) to find the closest model time step to.
        lon: The model longitude of the selected radial to plot.
        save: Boolean to determine if the figure is saved.
        tag: String to append to the filename if saving the figure.
    Returns:
        fig: Figure handle
        ax: Axes handle (or array of axes for compressible models)
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

    # Create figure with multiple panels for compressible models
    is_compressible = hasattr(model, 'compressible') and model.compressible
    if is_compressible:
        fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
        ax = axes[0]  # Velocity axis
    else:
        fig, ax = plt.subplots(figsize=(14, 7))
        axes = [ax]
        
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
    cme_colors = get_cme_colors()
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
    ax.set_ylabel(ylab if not is_compressible else 'V (km/s)')
    ax.set_xlim(model.r.value.min(), model.r.value.max())
    if not is_compressible:
        ax.set_xlabel('Radial distance ($R_{sun}$)')
    
    # Plot density if compressible
    if is_compressible:
        m_p = 1.6726e-27  # Proton mass in kg
        n_profile = model.rho_grid[id_t, :, id_lon].value / m_p / 1e6  # Convert to protons/cm³
        axes[1].semilogy(model.r, n_profile, 'b-')
        axes[1].set_ylabel('n (protons/cm³)', color='b')
        axes[1].tick_params(axis='y', labelcolor='b')
        axes[1].set_xlim(model.r.value.min(), model.r.value.max())
        axes[1].grid(True, alpha=0.3)
        
        # Plot temperature
        T_profile = model.temp_grid[id_t, :, id_lon].value
        axes[2].semilogy(model.r, T_profile, 'r-')
        axes[2].set_ylabel('T (K)', color='r')
        axes[2].tick_params(axis='y', labelcolor='r')
        axes[2].set_xlabel('Radial distance ($R_{sun}$)')
        axes[2].set_xlim(model.r.value.min(), model.r.value.max())
        axes[2].grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)

    # Add label
    time_label = " Time: {:3.2f} days".format(time_out)
    lon_label = " Lon: {:3.2f}$^\\circ$".format(lon_out)
    label = "HUXt" + time_label + lon_label
    if is_compressible:
        axes[0].set_title(label, fontsize=20)
    else:
        ax.set_title(label, fontsize=20)

    if save:
        cr_num = np.int32(model.cr_num.value)
        lon_tag = "{}deg".format(lon.to(u.deg).value)
        filename = "HUXt_CR{:03d}_{}_radial_profile_lon_{}_frame_{:03d}.png".format(cr_num, tag, lon_tag, id_t)
        figure_dir = get_figure_dir()
        filepath = figure_dir.joinpath(filename)
        fig.savefig(filepath)

    return fig, axes if is_compressible else ax


def plot_timeseries(model, radius, lon, save=False, tag=''):
    """
    Plot the solar wind model timeseries at model radius and longitude closest to those specified.
    For compressible models, shows velocity, density, and temperature.
    Args:
        model: An instance of the HUXt class with a completed solution.
        radius: Radius to find the closest model radius to.
        lon: Longitude to find the closest model longitude to.
        save: Boolean to determine if the figure is saved.
        tag: String to append to the filename if saving the figure.
    Returns:
        fig: Figure handle
        ax: Axes handle (or array of axes for compressible models)
    """

    if (radius < model.r.min()) | (radius > (model.r.max())):
        print("Error, specified radius outside of model radial grid")

    if model.lon.size != 1:
        if (lon < model.lon.min() - model.dlon) | (lon > model.lon.max() + model.dlon):
            print("Error, input lon outside range of model longitudes. Defaulting to closest longitude")
            id_lon = np.argmin(np.abs(model.lon - lon))
            lon = model.lon[id_lon]

    # Create figure with multiple panels for compressible models
    is_compressible = hasattr(model, 'compressible') and model.compressible
    if is_compressible:
        fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
        ax = axes[0]  # Velocity axis
    else:
        fig, ax = plt.subplots(figsize=(14, 7))
        axes = [ax]
        
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
    ax.set_ylabel(ylab if not is_compressible else 'V (km/s)')
    ax.set_xlim(t_day.value.min(), t_day.value.max())
    if not is_compressible:
        ax.set_xlabel('Time (days)')
    ax.grid(True, alpha=0.3)
    
    # Plot density if compressible
    if is_compressible:
        m_p = 1.6726e-27  # Proton mass in kg
        n_timeseries = model.rho_grid[:, id_r, id_lon].value / m_p / 1e6  # Convert to protons/cm³
        axes[1].semilogy(t_day, n_timeseries, 'b-')
        axes[1].set_ylabel('n (protons/cm³)', color='b')
        axes[1].tick_params(axis='y', labelcolor='b')
        axes[1].set_xlim(t_day.value.min(), t_day.value.max())
        axes[1].grid(True, alpha=0.3)
        
        # Plot temperature
        T_timeseries = model.temp_grid[:, id_r, id_lon].value
        axes[2].semilogy(t_day, T_timeseries, 'r-')
        axes[2].set_ylabel('T (K)', color='r')
        axes[2].tick_params(axis='y', labelcolor='r')
        axes[2].set_xlabel('Time (days)')
        axes[2].set_xlim(t_day.value.min(), t_day.value.max())
        axes[2].grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)

    # Add label
    radius_label = " Radius: {:3.2f}".format(r_out) + "$R_{sun}$ "
    lon_label = " Longitude: {:3.2f}".format(lon_out) + "$^\\circ$"
    label = "HUXt" + radius_label + lon_label
    if is_compressible:
        axes[0].set_title(label, fontsize=20)
    else:
        ax.set_title(label, fontsize=20)

    if save:
        cr_num = np.int32(model.cr_num.value)
        r_tag = np.int32(r_out)
        lon_tag = np.int32(lon_out)
        template_string = "HUXt1D_CR{:03d}_{}_time_series_radius_{:03d}_lon_{:03d}.png"
        filename = template_string.format(cr_num, tag, r_tag, lon_tag)
        figure_dir = get_figure_dir()
        filepath = figure_dir.joinpath(filename)
        fig.savefig(filepath)

    return fig, axes if is_compressible else ax


def get_observer_timeseries(model, observer='Earth', suppress_warning=False):
    """
    Compute the solar wind time series at an observer location. Returns a pandas dataframe with the 
    solar wind speed time series interpolated from the model solution using the
    observer ephemeris. Nearest neighbour interpolation in r, linear interpolation in longitude.
    For compressible models, also includes density and temperature.
    Args:
        model: A HUXt instance with a solution generated by HUXt.solve().
        observer: String name of the observer. Can be any permitted by Observer class.
        suppress_warning: Bool for stopping a warning printing.
    Returns:
         time_series: A pandas dataframe giving time series of solar wind speed, and if it exists in the HUXt
                            solution, the magnetic field polarity (and for compressible models, density and temperature), at the observer.
    """
    earth_pos = model.get_observer('Earth')
    obs_pos = model.get_observer(observer)

    # find the model coords of Earth as a function of time
    if model.frame == 'sidereal':
        deltalon = earth_pos.lon_hae - earth_pos.lon_hae[0]
        model_lon_earth = zerototwopi(earth_pos.lon.value + deltalon.value)
    elif model.frame == 'synodic':
        model_lon_earth = earth_pos.lon.value

    # find the model coords of the given osberver as a function of time
    deltalon = obs_pos.lon_hae - earth_pos.lon_hae
    model_lon_obs = zerototwopi(model_lon_earth + deltalon.value)

    if (model.frame == 'sidereal') & (model.nlon == 1) & (not suppress_warning):
        print("Warning: HUXt configured for a 1-D run in the sidereal frame. This simulation will not work correctly"
              "with functions like huxt_analysis.get_observer_time_series()")

    if model.nlon == 1 and not suppress_warning:
        print('Single longitude simulated. Extracting time series at Observer r')

    time = np.ones(model.nt_out) * np.nan
    mjd = np.ones(model.nt_out) * np.nan
    lon = np.ones(model.nt_out) * np.nan
    rad = np.ones(model.nt_out) * np.nan
    speed = np.ones(model.nt_out) * np.nan
    bpol = np.ones(model.nt_out) * np.nan
    
    # Add density and temperature arrays for compressible models
    is_compressible = hasattr(model, 'compressible') and model.compressible
    if is_compressible:
        density = np.ones(model.nt_out) * np.nan
        temperature = np.ones(model.nt_out) * np.nan
        m_p = 1.6726e-27  # Proton mass in kg

    for t in range(model.nt_out):
        time[t] = (model.time_init + model.time_out[t]).jd
        mjd[t] = (model.time_init + model.time_out[t]).mjd

        # find the nearest longitude cell
        model_lons = model.lon.value
        if model.nlon == 1:
            model_lons = np.array([model_lons])
        id_lon = np.argmin(np.abs(model_lons - model_lon_obs[t]))

        # check whether the observer is within the model domain
        if ((obs_pos.r[t].value < model.r[0].value) or (obs_pos.r[t].value > model.r[-1].value) or
           ((abs(model_lons[id_lon] - model_lon_obs[t]) > model.dlon.value) and
           (abs(model_lons[id_lon] + 2 * np.pi - model_lon_obs[t]) > model.dlon.value))):

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
                if is_compressible:
                    density[t] = model.rho_grid[t, id_r, 0].value / m_p / 1e6  # Convert to protons/cm³
                    temperature[t] = model.temp_grid[t, id_r, 0].value
            else:
                speed[t] = np.interp(model_lon_obs[t], model.lon.value, model.v_grid[t, id_r, :].value,
                                     period=2 * np.pi)
                if hasattr(model, 'b_grid'):
                    bpol[t] = np.interp(model_lon_obs[t], model.lon.value, model.b_grid[t, id_r, :], period=2 * np.pi)
                if is_compressible:
                    rho_interp = np.interp(model_lon_obs[t], model.lon.value, model.rho_grid[t, id_r, :].value,
                                          period=2 * np.pi)
                    density[t] = rho_interp / m_p / 1e6  # Convert to protons/cm³
                    temperature[t] = np.interp(model_lon_obs[t], model.lon.value, model.temp_grid[t, id_r, :].value,
                                              period=2 * np.pi)

    time = pd.to_datetime(time, unit='D', origin='julian')

    data_dict = {'time': time, 'r': rad, 'lon': lon, 'vsw': speed, 'bpol': bpol, 'mjd': mjd}
    if is_compressible:
        data_dict['n'] = density  # protons/cm³
        data_dict['T'] = temperature  # K
    
    time_series = pd.DataFrame(data=data_dict)
    return time_series


def get_HUXt_at_position_HEEQ(model, target_mjd, target_r, target_lon_heeq):
    """
    Extract and return HUXt fields at a generic set of locations. Assumes the 
    object is in the lat plane of the model. For compressible models, also includes
    density and temperature.
    
    Args:
        model: The HUXt model class for the solved run.
        target_mjd: Time at which values should be extracted, as MJD
        target_r: radial distance at which values should be extracted, in rS
        target_lon_heeq: HEEQ lon at which values should be extracted, in radians
        
    Returns:
       time_series: A pandas dataframe giving time series of solar wind speed, and if it exists in the HUXt
                           solution, the magnetic field polarity (and for compressible models, density and temperature), at the observer.
    """
    
    tim_mjd = model.time_init.mjd + model.time_out.value/(24*60*60)
    earth_pos = model.get_observer('Earth')
    
    # find the model coords of Earth as a function of time
    if model.frame == 'sidereal':
        e_deltalon = earth_pos.lon_hae - earth_pos.lon_hae[0]
        model_lon_earth = zerototwopi(earth_pos.lon.value + e_deltalon.value)
    elif model.frame == 'synodic':
        model_lon_earth = earth_pos.lon.value

    if model.nlon == 1:
        print('Single longitude simulated. Extracting time series at Observer r')

    # find only the time stamps that are within the model run
    nt = len(target_mjd)
    lon = np.ones(nt) * np.nan
    rad = np.ones(nt) * np.nan
    speed = np.ones(nt) * np.nan
    bpol = np.ones(nt) * np.nan
    
    # Add density and temperature arrays for compressible models
    is_compressible = hasattr(model, 'compressible') and model.compressible
    if is_compressible:
        density = np.ones(nt) * np.nan
        temperature = np.ones(nt) * np.nan
        m_p = 1.6726e-27  # Proton mass in kg
    
    for t in range(0, nt):

        # check the required time is within the model run
        if (target_mjd[t] >= tim_mjd[0]) & (target_mjd[t] <= tim_mjd[-1]):
            
            # find the nearest time index in the HUXt run
            id_t = np.argmin(np.abs(tim_mjd - target_mjd[t]))
            
            # compute the HUXT long associated with the given HEEQ lon
            model_lon_obs = zerototwopi(model_lon_earth[id_t] + target_lon_heeq[t])
    
            # find the nearest longitude cell
            model_lons = model.lon.value
            if model.nlon == 1:
                model_lons = np.array([model_lons])
                
            id_lon = np.argmin(np.abs(model_lons - model_lon_obs))

            # check whether the observer radius is within the model domain
            if ((target_r[t] < model.r[0].value) or (target_r[t] > model.r[-1].value) or
               ((abs(model_lons[id_lon] - model_lon_obs) > model.dlon.value) and
               (abs(model_lons[id_lon] + 2 * np.pi - model_lon_obs) > model.dlon.value))):
        
                bpol[t] = np.nan
                speed[t] = np.nan
                print('r or lon outside model domain')
            else:
                # find the nearest R coord
                id_r = np.argmin(np.abs(model.r.value - target_r[t]))
                rad[t] = model.r[id_r].value
                lon[t] = model_lons[id_lon]
                # then interpolate the values in longitude
                if model.nlon == 1:
                    speed[t] = model.v_grid[id_t, id_r, 0].value
                    if hasattr(model, 'b_grid'):
                        bpol[t] = model.b_grid[id_t, id_r, 0]
                    if is_compressible:
                        density[t] = model.rho_grid[id_t, id_r, 0].value / m_p / 1e6  # Convert to protons/cm³
                        temperature[t] = model.temp_grid[id_t, id_r, 0].value
                else:
                    speed[t] = np.interp(model_lon_obs, model.lon.value, model.v_grid[id_t, id_r, :].value,
                                         period=2 * np.pi)
                    if hasattr(model, 'b_grid'):
                        bpol[t] = np.interp(model_lon_obs, model.lon.value, model.b_grid[id_t, id_r, :],
                                            period=2 * np.pi)
                    if is_compressible:
                        rho_interp = np.interp(model_lon_obs, model.lon.value, model.rho_grid[id_t, id_r, :].value,
                                              period=2 * np.pi)
                        density[t] = rho_interp / m_p / 1e6  # Convert to protons/cm³
                        temperature[t] = np.interp(model_lon_obs, model.lon.value, model.temp_grid[id_t, id_r, :].value,
                                                  period=2 * np.pi)
        else:
            print('time outside model domain')

    base = pd.Timestamp("1858-11-17 00:00:00")
    datetimes = base + pd.to_timedelta(target_mjd, unit='D')

    data_dict = {'time': datetimes, 'r': rad, 'lon': lon, 'vsw': speed, 'bpol': bpol, 'mjd': target_mjd}
    if is_compressible:
        data_dict['n'] = density  # protons/cm³
        data_dict['T'] = temperature  # K
    
    time_series = pd.DataFrame(data=data_dict)
    
    return time_series


def plot_earth_timeseries(model, plot_omni=True, save=False, tag=''):
    """
    A function to plot the HUXt Earth time series. With option to download and plot OMNI data.
    For compressible models, also plots density and temperature.
    Args:
        model : input model class
        plot_omni: Boolean, if True downloads and plots OMNI data
        save: Boolean, if True saves plot
        tag: String, tag string to append to the plot title
    Returns:
        fig : Figure handle
        axs : Axes handles

    """

    huxt_ts = get_observer_timeseries(model, observer='Earth')
    
    is_compressible = hasattr(model, 'compressible') and model.compressible

    # Determine number of panels
    n_panels = 1
    if hasattr(model, 'b_grid'):
        n_panels += 1
    if is_compressible:
        n_panels += 2  # Add density and temperature panels
        
    fig, axs = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels))
    if n_panels == 1:
        axs = np.array([axs])
    
    # Velocity panel (always first)
    panel_idx = 0
    axs[panel_idx].plot(huxt_ts['time'], huxt_ts['vsw'], 'k', label='HUXt')
    axs[panel_idx].set_ylim(250, 1000)
    axs[panel_idx].set_ylabel('Solar Wind Speed (km/s)')
    
    # B polarity panel (if available)
    if hasattr(model, 'b_grid'):
        panel_idx += 1
        axs[panel_idx].plot(huxt_ts['time'], np.sign(huxt_ts['bpol']), 'k.', label='HUXt')
        axs[panel_idx].set_ylabel('B polarity')
    
    # Density panel (if compressible)
    if is_compressible and 'n' in huxt_ts.columns:
        panel_idx += 1
        axs[panel_idx].semilogy(huxt_ts['time'], huxt_ts['n'], 'b-', label='HUXt')
        axs[panel_idx].set_ylabel('n (protons/cm³)', color='b')
        axs[panel_idx].tick_params(axis='y', labelcolor='b')
        axs[panel_idx].grid(True, alpha=0.3)
    
    # Temperature panel (if compressible)
    if is_compressible and 'T' in huxt_ts.columns:
        panel_idx += 1
        axs[panel_idx].semilogy(huxt_ts['time'], huxt_ts['T'], 'r-', label='HUXt')
        axs[panel_idx].set_ylabel('T (K)', color='r')
        axs[panel_idx].tick_params(axis='y', labelcolor='r')
        axs[panel_idx].grid(True, alpha=0.3)

    starttime = huxt_ts['time'][0]
    endtime = huxt_ts['time'][len(huxt_ts) - 1]

    if plot_omni:
        # grab the omni data
        data = hin.get_omni(starttime, endtime)
        # plot the period of interest
        mask = (data['datetime'] >= starttime) & (data['datetime'] <= endtime)
        plotdata = data[mask]
        axs[0].plot(plotdata['datetime'], plotdata['V'], 'r', label='OMNI')

        if hasattr(model, 'b_grid'):
            axs[1].plot(plotdata['datetime'], -np.sign(plotdata['BX_GSE']) * 0.92, 'r.', label='OMNI')
            axs[1].set_ylim(-1.1, 1.1)
        
        # Plot OMNI density if compressible model
        if is_compressible and 'n' in huxt_ts.columns:
            # Find the density panel index
            density_panel = 1 if hasattr(model, 'b_grid') else 0
            density_panel += 1
            # OMNI density field is typically 'N' (protons/cm³)
            if 'N' in plotdata.columns:
                # Set invalid data points to NaN
                omni_n = plotdata['N'].copy()
                omni_n[omni_n == 999.9] = np.nan
                omni_n[omni_n == 9999.0] = np.nan
                axs[density_panel].semilogy(plotdata['datetime'], omni_n, 'r-', label='OMNI', alpha=0.7)
        
        # Plot OMNI temperature if compressible model
        if is_compressible and 'T' in huxt_ts.columns:
            # Find the temperature panel index
            temp_panel = 1 if hasattr(model, 'b_grid') else 0
            temp_panel += 2
            # OMNI temperature field is typically 'T' (K)
            if 'T' in plotdata.columns:
                # Set invalid data points to NaN
                omni_t = plotdata['T'].copy()
                omni_t[omni_t == 9999999.0] = np.nan
                omni_t[omni_t == 999999.0] = np.nan
                axs[temp_panel].semilogy(plotdata['datetime'], omni_t, 'r-', label='OMNI', alpha=0.7)

    for a in axs:
        a.set_xlim(starttime, endtime)
        a.legend()

    # Only last panel gets x-label
    for i in range(len(axs) - 1):
        axs[i].set_xticklabels([])
    axs[-1].set_xlabel('Date')

    fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)

    if save:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_earth_timeseries.png".format(cr_num, tag)
        figure_dir = get_figure_dir()
        filepath = figure_dir.joinpath(filename)
        fig.savefig(filepath)

    return fig, axs


@u.quantity_input(time=u.day)
def plot3d_radial_lat_slice(model3d, time, lon=np.nan * u.deg, save=False, tag='', fighandle=np.nan, axhandle=np.nan):
    """
    Make a contour plot on polar axis of a radial-latitudinal plane of the solar wind solution at a fixed time and
    longitude.
    Args:
        model3d: An instance of the HUXt3d class with a completed solution.
        time: Time to look up closet model time to (with an astropy.unit of time).
        lon: The longitude along which to render the radial-latitude plane.
        save: Boolean to determine if the figure is saved.
        tag: String to append to the filename if saving the figure.
        fighandle: Pass a figure handle to render the plot in that figure.
        axhandle: Pass an axes handle to renfer the plot in that axes.
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
    cnt.set_edgecolor("face")

    # Trace the CME boundaries
    cme_colors = get_cme_colors()
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
    planet_list = get_planets_to_plot(model)
    spacecraft_list = get_spacecraft_to_plot(model)
    observer_list = planet_list + spacecraft_list

    if model.r[0] > 200 * u.solRad:
        observer_list = ['EARTH', 'MARS', 'JUPITER', 'SATURN']

    styles = observer_styles()
    # Add on observers 
    for body in observer_list:
        obs = model.get_observer(body)
        deltalon = 0.0 * u.rad

        # adjust body longitude for the frame
        if model.frame == 'sidereal':
            earth_pos = model.get_observer('EARTH')
            deltalon = earth_pos.lon_hae[id_t] - earth_pos.lon_hae[0]

        bodylon = zerototwopi(obs.lon[id_t] + deltalon)
        # plot bodies that are close to being in the plane
        if abs(bodylon - lon_out) < model.dlon * 2:
            ax.plot(obs.lat[id_t], obs.r[id_t], markersize=16, label=body, linestyle='',
                    marker=styles[body]['marker'], color=styles[body]['color'])

    # Add on a legend.
    fig.legend(ncol=len(observer_list), loc='lower center', frameon=False, handletextpad=0.1, columnspacing=0.5)

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
        filename = "HUXt_CR{:03d}_{}_3D_frame_{:03d}.png".format(cr_num, tag, id_t)
        figure_dir = get_figure_dir()
        filepath = figure_dir.joinpath(filename)
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
        plot3d_radial_lat_slice(model3d, model.time_out[i], lon, fighandle=fig, axhandle=ax)
        return frame
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    
    # Create the animation
    ani = FuncAnimation(fig, make_frame3d, frames=range(nframes), interval=interval)
    
    if outputfilepath:
        filepath = outputfilepath
    else:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_3D_movie.mp4".format(cr_num, tag)
        figure_dir = get_figure_dir()
        filepath = figure_dir.joinpath(filename)
    
    # Save the animation as a movie file
    ani.save(filepath, writer='ffmpeg')
    print('mp4 file written to ' + str(filepath))
    
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
    lon_arr, dlon, nlon = h.longitude_grid()
    lon, rad = np.meshgrid(lon_arr.value, model.r.value)
    mymap = mpl.cm.PuOr
    v_sub = model.b_grid[id_t, :, :].copy()
    plotvmin = -1.1
    plotvmax = 1.1
    dv = 1
    ylab = "Magnetic field polarity"

    # Insert into full array
    if lon_arr.size != model.lon.size:
        v = np.zeros((model.nr, nlon)) * np.nan
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
    cnt.set_edgecolor("face")

    # Add on CME boundaries
    cme_colors = get_cme_colors()
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
        planet_list = get_planets_to_plot(model)
        spacecraft_list = get_spacecraft_to_plot(model)
        observer_list = planet_list + spacecraft_list

        if model.r[0] > 200 * u.solRad:
            observer_list = ['EARTH', 'MARS', 'JUPITER', 'SATURN']

        styles = observer_styles()
        # Add on observers 
        for body in observer_list:
            obs = model.get_observer(body)
            deltalon = 0.0 * u.rad
            if model.frame == 'sidereal':
                earth_pos = model.get_observer('EARTH')
                deltalon = earth_pos.lon_hae[id_t] - earth_pos.lon_hae[0]

            obslon = zerototwopi(obs.lon[id_t] + deltalon)
            ax.plot(obslon, obs.r[id_t],  markersize=16, label=body, linestyle='',
                    marker=styles[body]['marker'], color=styles[body]['color'])

        # Add on a legend.
        fig.legend(ncol=len(observer_list), loc='lower center', frameon=False, handletextpad=0.1, columnspacing=0.5)

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
                foot_lon = zerototwopi(model.streak_lon_r0[id_t, istreak])
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
        filename = "HUXt_CR{:03d}_{}_bpol_frame_{:03d}.png".format(cr_num, tag, id_t)
        figure_dir = get_figure_dir()
        filepath = figure_dir.joinpath(filename)
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
                
                if r_streak_km[it+1, ilon] > rgrid_km[-1]:
                    r_streak_km[it+1, ilon] = np.nan
     
    return r_streak_km[id_t_stop, :]
        

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
    intx = np.interp(np.linspace(0, total_length, num_interpolated_points + 1), padded_cumulative_distances, x)
    inty = np.interp(np.linspace(0, total_length, num_interpolated_points + 1), padded_cumulative_distances, y)

    # find closest point to Earth
    distances = np.sqrt((intx - Ex)**2 + (inty - Ey)**2)
    
    # Replace NaN values with a large value
    distances[np.isnan(distances)] = np.inf
    i = np.argmin(distances)
    
    # convert the closest point back to r, lon
    r = np.sqrt(intx[i]**2 + inty[i]**2)
    theta = np.arctan2(inty[i], intx[i])
    
    return distances[i], r, theta


@jit(nopython=True)
def respinup_model(v_trl_kms, tgrid_s, rgrid_km, longrid_rad, rot_period_s, buffer_time_s):
    """
    recreate steady-state solar wind conditions during the spin-up period 
    to enable field-line tracing near the start of a model run
    
    Args:
        v_trl_kms: model.v_grid.value - 
        tgrid_s: model.time_out.to(u.s).value - the time grid in seconds
        rgrid_km: model.r.to(u.km).value - the radial grid in km
        longrid_rad: model.lon.to(u.rad).value - the longitude grid in radians
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
    spinup_tgrid_s = np.arange(-nsteps * dt_s, 0, dt_s)
    new_tgrid_s = np.append(spinup_tgrid_s, tgrid_s)

    new_v_trl_kms = np.ones((len(new_tgrid_s), nr, nlon))
    # put the existing data in
    new_v_trl_kms[nsteps:, :, :] = v_trl_kms[:, :, :]

    for t in range(0, len(spinup_tgrid_s)):
        dt = -spinup_tgrid_s[t]
        dlon = np.mod(2*np.pi * dt / rot_period_s, 2*np.pi)
        this_lons = np.mod(longrid_rad + dlon, 2*np.pi)
        for r in range(0, nr):
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
    rel_lons = np.mod(longrid_rad - start_lon, 2*np.pi)
    sort_indices = np.argsort(rel_lons)
    plotlon = longrid_rad[sort_indices]
    plotr = r_streak[sort_indices]

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

    buffertime = 6 * u.day  # the tracing this time before the ballistic start time estimate
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
        lon_Earth_t = earth_pos.lon_hae[id_t].value
        lon_Earth_0 = earth_pos.lon_hae[0].value
        lon_Earth_rad = zerototwopi(lon_Earth_t - lon_Earth_0)
    
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
    start_lon = zerototwopi(lon_Earth_rad - 1)
    
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
    rel_lon = zerototwopi(longrid_rad - start_lon)
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
    plotlon = np.append(plotlon, zerototwopi(plotlon[-1] + dlon))

    return plotlon, (plotr*u.km).to(u.solRad).value, optimal_lon, optimal_t


def run_WSA_HUXt_td_wedge_about_observer(start_dt, stop_dt, vel_path, vel_format_template, obj='Earth',
                                         deacc=True, dlat=2*u.deg):
    """
    Parameters
    ----------
    start_dt : datetime
        Run start date
    stop_dt : datetime
        Run end date
    vel_path : string
        path to file directory of WSA solutions
    vel_format_template : string
        file format with YYYY, MM, DD, HH, mm and ss used to identify the timestamp
    obj : string, optional
        Name of object for JPL horizons. The default is 'Earth'.
    deacc : BOOL, optional
        Whether to deacccelerate the WSA speeds from 215 to 21.5 rS. The default is True.
    dlat : float, with units of deg, optional
        latitudinal spacing of HUXt runs. The default is 2*u.deg.

    Returns
    -------
    huxt_ts : pandas dataframe
        Time series of HUXt speeds and magnetic field polarities

    """
    
    r_min = 21.5 * u.solRad
    simtime = (stop_dt - start_dt).days * u.day

    if obj == 'Earth':
        obj = 'geocenter'
    
    coord = get_horizons_coord(obj, {'start': Time(start_dt + datetime.timedelta(days=2/24)), 
                                     'stop': Time(stop_dt - datetime.timedelta(days=2/24)),
                                     'step': '1H'})
    smjd = Time(start_dt).mjd
    fmjd = Time(stop_dt).mjd
    
    # make a dataframe from this
    coords = pd.DataFrame()
    coords['r_AU'] = coord.radius.value
    coords['lon_heeq'] = zerototwopi(coord.lon.value * np.pi/180)
    coords['lat_heeq'] = coord.lat.value * np.pi/180
    coords['mjd'] = np.linspace(smjd, fmjd, len(coords))
    coords['datetime'] = Time(coords['mjd'] + 2400000.5, format='jd').to_datetime()
    
    # convert to Carrington longitude for plotting purposes
    Carrington_coord = coord.transform_to(sunpy.coordinates.HeliographicCarrington(observer="self"))
    coords['lon_carr'] = Carrington_coord.lon.value * np.pi/180

    # get the required lat range for the HUXt runs
    obj_max_lat = np.nanmax(coords['lat_heeq'])*180/np.pi
    obj_min_lat = np.nanmin(coords['lat_heeq'])*180/np.pi
    
    obj_min_lon = 0
    obj_max_lon = 2*np.pi * simtime.value / 365.25
    
    obj_min_r = np.min(coords['r_AU']) * 215
    obj_max_r = np.max(coords['r_AU']) * 215  
    
    assert (obj_min_r >= r_min.to(u.solRad).value)

    minlat = np.floor(obj_min_lat)
    maxlat = np.ceil(obj_max_lat)
    lat_list = np.arange(minlat, maxlat + 0.0001, dlat.to(u.deg).value)
    
    # run HUXt at each latitude and extract the values at the object's long and r
    huxt_cuts = []
    for lat in lat_list:
        print('Runnig HUXt at lat = ' + str(lat) + ' degrees')
        thislat = (lat*np.pi/180)*u.rad
        # create the HUXt input from the WSA files
        vlongs, brlongs, lon, mjds, times = hin.huxt_td_input_from_WSA_runs(vel_path, start_dt, stop_dt,
                                                                            latitude=thislat, deacc=deacc,
                                                                            input_res_days=0.1,
                                                                            format_template=vel_format_template)

        # set up the model, with (optional) time-dependent bpol boundary conditions
        model = hin.set_time_dependent_boundary(vlongs, mjds, start_dt, simtime, lon_start=obj_min_lon*u.rad,
                                                lon_stop=obj_max_lon*u.rad, r_min=r_min, r_max=obj_max_r*u.solRad,
                                                bgrid_Carr=brlongs, dt_scale=4, latitude=thislat, frame='sidereal')
        model.solve([])
        
        # get values at Earth long
        cut = get_HUXt_at_position_HEEQ(model, coords['mjd'], coords['r_AU']*215, coords['lon_heeq'])
        huxt_cuts.append(cut)

    # now interpolate the extracted series to the object's latitude
    # copy the basic info across
    huxtvals = huxt_cuts[0]
    huxt_ts = huxtvals[['time', 'mjd', 'r']].copy()
    
    # loop through each time step and interpolate in lat
    for t in range(len(huxtvals)):
        # get this lat
        lat_t = np.interp(huxtvals.loc[t, 'mjd'], coords['mjd'], coords['lat_heeq'])
        b_lat_t = [df.loc[t, 'bpol'] if t in df.index else None for df in huxt_cuts]
        huxt_ts.loc[t, 'bpol'] = np.interp(lat_t, np.deg2rad(lat_list), b_lat_t)
        v_lat_t = [df.loc[t, 'vsw'] if t in df.index else None for df in huxt_cuts]
        huxt_ts.loc[t, 'vsw'] = np.interp(lat_t, np.deg2rad(lat_list), v_lat_t)
        
    return huxt_ts


def zerototwopi(angles):
    """
    Function to constrain angles to the 0 - 2pi domain.
    Args:
        angles: a numpy array of angles.
    Returns:
        angles_out: a numpy array of angles constrained to 0 - 2pi domain.
    """
    # Check if angles has astropy unit.
    if isinstance(angles, u.Quantity):
        angles_out = angles.to(u.rad).value
    else:
        angles_out = angles

    twopi = 2.0 * np.pi
    a = -np.floor_divide(angles_out, twopi)
    angles_out = angles_out + (a * twopi)

    # If it came in with units, restore them
    if isinstance(angles, u.Quantity):
        angles_out = angles_out * u.rad

    return angles_out


def observer_styles():
    """Returns a dictionary giving the colors and marker styles to use for each planet and spacecraft."""

    styles = {'MERCURY': {'marker': 'o', 'color': 'darkviolet'},
              'VENUS': {'marker': 'o', 'color': 'hotpink'},
              'EARTH': {'marker': 'o', 'color': 'black'},
              'MARS': {'marker': 'o', 'color': 'lightcoral'},
              'JUPITER': {'marker': 'o', 'color': 'darkorange'},
              'SATURN': {'marker': 'o', 'color': 'moccasin'},
              'ACE': {'marker': '^', 'color': 'tab:gray'},
              'STA': {'marker': '^', 'color': 'tab:red'},
              'STB': {'marker': '^', 'color': 'tab:cyan'},
              'PSP': {'marker': '^', 'color': 'tab:orange'},
              'SOLO': {'marker': '^', 'color': 'tab:pink'},
              'ULYSSES': {'marker': '^', 'color': 'tab:brown'}}

    return styles


def get_planets_to_plot(model):
    """
    Helper function for plotting - produces a list of planet names to be plotted.
    Args:
        model: A solved HUXt instance
    Returns:
        planet_list: A list of planets to plot.
    """
    planet_list = ['EARTH']
    if model.r[-1] < 350 * u.solRad:
        planet_list.append('VENUS')
        planet_list.append('MERCURY')
    if model.r[-1] > 350 * u.solRad:
        planet_list.append('MARS')
    if model.r[-1] > 1100 * u.solRad:
        planet_list.append('JUPITER')
    if model.r[-1] > 2000 * u.solRad:
        planet_list.append('SATURN')

    return planet_list


def get_spacecraft_to_plot(model):
    """
        Helper function for plotting - produces a list of spacecarft names to be plotted.
        Args:
            model: A solved HUXt instance
        Returns:
            spacecraft_list: A list of spacecraft to plot.
        """
    spacecraft_list = []
    if model.r[0] < 200 * u.solRad:
        if model.time_init > datetime.datetime(2007, 1, 1):
            spacecraft_list.append('STA')
            if model.time_init < datetime.datetime(2016, 8, 21):
                spacecraft_list.append('STB')

        if model.time_init > datetime.datetime(2018, 8, 13):
            spacecraft_list.append('PSP')

        if model.time_init > datetime.datetime(2020, 2, 11):
            spacecraft_list.append('SOLO')

        if model.time_init > datetime.datetime(2006, 7, 3):
            spacecraft_list.append('ACE')

    if (model.time_init > datetime.datetime(1990, 10, 7)) & \
       (model.time_init < datetime.datetime(2009, 6, 29)):

        if (model.r.min() > 280*u.solRad) | (model.r.max() > 280*u.solRad):
            spacecraft_list.append('ULYSSES')

    return spacecraft_list


def get_cme_colors():
    """
    Return a list of colors for plotting CME boundaries
    """
    cme_colors = ['r', 'c', 'm', 'y', 'deeppink', 'darkorange']
    return cme_colors
