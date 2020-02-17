import numpy as np
import astropy.units as u
import os
import glob
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
from skimage import measure
import scipy.ndimage as ndi

mpl.rc("axes", labelsize=20)
mpl.rc("ytick", labelsize=20)
mpl.rc("xtick", labelsize=20)
mpl.rc("legend", fontsize=20)


class ConeCME:
    """
    A class containing the parameters of a cone model cme.

    Attributes:
        t_launch: Time of Cone CME launch, in seconds after the start of the simulation.
        longitude: Longitude of the CME launch direction, in radians.
        v: CME nose speed in km/s.
        width: Angular width of the CME, in radians.
        initial_height: Initiation height of the CME, in km. Defaults to HUXt inner boundary at 30 solar radii.
        radius: Initial radius of the CME, in km.
        thickness: Thickness of the CME cone, in km.
        coords: Dictionary containing the radial and longitudinal (for HUXT2D) coordinates of the of Cone CME for each
                model time step.
    """

    def __init__(self, t_launch=0.0, longitude=0.0, v=1000.0, width=30.0, thickness=10.0):
        """
        Set up a Cone CME with specified parameters.

        :param t_launch: Time of Cone CME launch, in seconds after the start of the simulation.
        :param longitude: Longitude of the CME launch direction, in radians.
        :param v: CME nose speed in km/s.
        :param width: Angular width of the CME, in degrees.
        :param thickness: Thickness of the CME cone, in solar radii
        """
        self.t_launch = t_launch*u.s  # Time of CME launch, after the start of the simulation
        self.longitude = np.deg2rad(longitude) * u.rad  # Longitudinal launch direction of the CME
        self.v = v * u.km / u.s  # CME nose speed
        self.width = np.deg2rad(width) * u.rad  # Angular width
        self.initial_height = (30.0*u.solRad).to(u.km)  # Initial height of CME (should match inner boundary of HUXt)
        self.radius = self.initial_height*np.tan(self.width/2.0)  # Initial radius of CME
        self.thickness = (thickness*u.solRad).to(u.km)  # Extra CME thickness
        self.coords = {}
        
    def _track_1d_(self, model):
        """
        Tracks the length of each ConeCME through the HUXt1D solution in model.

        :param model: An instance of HUXt1D with completed solutions for the CME and ambient fields.
        :return: updates the ConeCME.coords dictionary of CME coordinates.
        """
        # Owens definition of CME in HUXt:
        diff = model.v_grid_cme - model.v_grid_amb
        cme_bool = diff >= 20*model.kms

        # Workflow: Loop over each CME, track CME through each time step,
        # find contours of boundary, save to dict.
        self.coords = {j: {'r_pix': np.array([])*u.pix, 'r': np.array([])*model.r.unit} for j in range(model.Nt_out)}
        first_frame = True
        for j, t in enumerate(model.time_out):

            if t < self.t_launch:
                continue

            cme_bool_t = cme_bool[j, :]
            # Center the solution on the CME longitude to avoid edge effects

            # measure separate CME regions.
            cme_label, n_cme = measure.label(cme_bool_t.astype(int), connectivity=1, background=0, return_num=True)
            cme_tags = [i for i in range(1, n_cme+1)]
            
            if n_cme != 0:
                if first_frame:
                    # Find only the label in the origin region of this CME
                    # Use a binary mask over the source region of the CME.
                    target = np.zeros(cme_bool_t.shape)
                    target[0] = 1
                    first_frame = False
                    # Find the CME label that intersects this region
            
                matches_id = []
                matches_level = []
                for label in cme_tags:
                    this_label = cme_label == label
                    overlap = np.sum(np.logical_and(target, this_label))
                    if overlap > 0:
                        matches_id.append(label)
                        matches_level.append(overlap)

                if len(matches_id) != 0:
                    # Check only one match, if not find closest match.
                    if len(matches_id) == 1:
                        match_id = matches_id[0]
                    else:
                        print("Warning, multiple matches found, selecting match with greatest target overlap")
                        match_id = matches_id[np.argmax(matches_level)]

                    cme_id = cme_label == match_id
                    r_pix = np.argwhere(cme_id)

                    self.coords[j]['r_pix'] = r_pix * u.pix
                    self.coords[j]['r'] = np.interp(r_pix, np.arange(0, model.Nr), model.r)

                    # Update the target, so next iteration finds CME that overlaps with this frame.
                    target = cme_id.copy()
        return
    
    def _track_2d_(self, model):
        """
        Tracks the perimeter of each ConeCME through the HUXt2D solution in model.

        :param model: An instance of HUXt2D with completed solutions for the CME and ambient fields.
        :return: updates the ConeCME.coords dictionary of CME coordinates.
        """

        # Owens definition of CME in HUXt:
        diff = model.v_grid_cme - model.v_grid_amb
        cme_bool = diff >= 20*model.kms

        # Find index of middle longitude for centering arrays on the CMEs
        id_mid_lon = np.argmin(np.abs(model.lon - np.median(model.lon)))

        # Workflow: Loop over each CME, center model solution on CME source lon,
        # track CME through each time step, find contours of boundary, save to dict.
        # Get index of CME longitude
        id_cme_lon = np.argmin(np.abs(model.lon - self.longitude))
        self.coords = {j: {'lon_pix': np.array([])*u.pix, 'r_pix': np.array([])*u.pix,
                       'lon': np.array([])*model.lon.unit, 'r': np.array([])*model.r.unit} for j in range(model.Nt_out)}

        first_frame = True
        for j, t in enumerate(model.time_out):
                               
            if t < self.t_launch:
                continue

            cme_bool_t = cme_bool[j, :, :]
            # Center the solution on the CME longitude to avoid edge effects
            center_shift = id_mid_lon - id_cme_lon
            cme_bool_t = np.roll(cme_bool_t, center_shift, axis=1)

            # measure separate CME regions.
            cme_label, n_cme = measure.label(cme_bool_t.astype(int), connectivity=1, background=0, return_num=True)
            cme_tags = [i for i in range(1, n_cme+1)]
            
            if n_cme != 0:
                if first_frame:
                    # Find only the label in the origin region of this CME
                    # Use a binary mask over the source region of the CME.
                    target = np.zeros(cme_bool_t.shape)
                    half_width = self.width / (2*model.dlon)
                    left_edge = np.int32(id_mid_lon - half_width)
                    right_edge = np.int32(id_mid_lon + half_width)
                    target[0, left_edge:right_edge] = 1
                    first_frame = False
                    # Find the CME label that intersects this region

                matches_id = []
                matches_level = []
                for label in cme_tags:
                    this_label = cme_label == label
                    overlap = np.sum(np.logical_and(target, this_label))
                    if overlap > 0:
                        matches_id.append(label)
                        matches_level.append(overlap)

                if len(matches_id) != 0:
                    # Check only one match, if not find closest match.
                    if len(matches_id) == 1:
                        match_id = matches_id[0]
                    else:
                        print("Warning, multiple matches found, selecting match with greatest target overlap")
                        match_id = matches_id[np.argmax(matches_level)]

                    # Find the coordinates of this region and store 
                    cme_id = cme_label == match_id
                    # Fill holes in the labelled region
                    cme_id_filled = ndi.binary_fill_holes(cme_id)
                    coords = measure.find_contours(cme_id_filled, 0.5)

                    # Contour can be broken at inner and outer boundary, so stack broken contours
                    if len(coords) == 1:
                        coord_array = coords[0]
                    elif len(coords) > 1:
                        coord_array = np.vstack(coords)   

                    r_pix = coord_array[:, 0]
                    # Remove centering and correct wraparound indices
                    lon_pix = coord_array[:, 1] - center_shift
                    lon_pix[lon_pix < 0] += model.Nlon
                    lon_pix[lon_pix > model.Nlon] -= model.Nlon
                    self.coords[j]['lon_pix'] = lon_pix * u.pix
                    self.coords[j]['r_pix'] = r_pix * u.pix
                    self.coords[j]['r'] = np.interp(r_pix, np.arange(0,model.Nr), model.r)
                    self.coords[j]['lon'] = np.interp(lon_pix, np.arange(0,model.Nlon), model.lon)
                    # Update the target, so next iteration finds CME that overlaps with this frame.
                    target = cme_id.copy()
        return


class HUXt1D:
    """
    A class containing the 1D HUXt model described in Owens et al. (2020).

    Users must specify the solar wind speed boundary condition through either the v_boundary, or cr_num keyword
    arguments. Failure to do so defaults to a 400 km/s boundary. v_boundary takes precedence over cr_num, so specifying
    both results in only v_boundary being used.
    
    Attributes:
        cmes: A list of ConeCME instances used in the model solution.
        cr_num: If provided, this gives the Carrington rotation number of the selected period, else 9999.
        daysec: seconds in a day.
        dr: Radial grid spacing (in km).
        dt: Model time step (in seconds), set by the CFL condition with v_max and dr.
        dt_out: Output model time step (in seconds).
        dt_scale: Integer scaling number to set the model output time step relative to the models CFL time step.
        dtdr: Ratio of the model time step and radial grid step (in seconds/km).
        kms: astropy.unit instance of km/s.       
        lon: Model source longtidue (in radians).
        Nr: Number of radial grid points.
        Nt: Total number of model time steps, including spin up.
        Nt_out: Number of output model time steps.
        r_accel: Scale parameter determining the residual solar wind acceleration.
        r: Radial grid (in km).
        rrel: Radial grid relative to first grid point (in km).
        simtime: Simulation time (in seconds).
        synodic_period: Solar Synodic rotation period from Earth (in seconds).
        time: Array of model time steps, including spin up (in seconds).
        time_out: Array of output model time steps (in seconds).
        twopi: two pi radians
        v_boundary: Inner boundary solar wind speed profile (in km/s).
        v_grid_amb: Array of ambient model solution excluding ConeCMEs for each time and radius (in km/s).
        v_grid_cme: Array of model solution inlcuding ConeCMEs for each time and radius (in km/s).
        v_max: Maximum model speed (in km/s), used with the CFL condition to set the model time step. 
    """

    def __init__(self, v_boundary=None, cr_num=None, lon=0.0, simtime=5.0, dt_scale=1.0):
        """
        Initialise the HUXt1D instance.

        :param v_boundary: Inner solar wind speed boundary condition. Must be an array of size 128 with units of km/s.
        :param cr_num: Integer Carrington rotation number. Used to lookup the longitudinal solar wind speed profile
                       at the solar equator from HelioMAS. This is then used as the inner boundary condition.
        :param simtime: Duration of the simulation window, in days.
        :param dt_scale: Integer scaling number to set the model output time step relative to the models CFL time step.
        """
        # some constants and units
        constants = huxt_constants()
        self.twopi = constants['twopi']
        self.daysec = constants['daysec']
        self.kms = constants['kms']
        self.alpha = constants['alpha']  # Scale parameter for residual SW acceleration
        self.r_accel = constants['r_accel']  # Spatial scale parameter for residual SW acceleration
        self.synodic_period = constants['synodic_period']  # Solar Synodic rotation period from Earth.
        self.v_max = constants['v_max']
        self.lon = np.deg2rad(lon) * u.rad
        del constants
        
        # Extract paths of figure and data directories
        dirs = _setup_dirs_()
        self._boundary_dir_ = dirs['boundary_conditions']
        self._data_dir_ = dirs['HUXt1D_data']
        self._figure_dir_ = dirs['HUXt1D_figures']
        
        # Determine the boundary conditions from input v_boundary and cr_num
        if (v_boundary is None) & (cr_num is None):
            print("Warning: No boudary conditions supplied. Defaulting to 400 km/s boundary")
            self.v_boundary = 400 * np.ones(128) * self.kms
        elif v_boundary is not None:
            assert v_boundary.unit == self.kms
            assert v_boundary.size == 128
            self.v_boundary = v_boundary
            # Set dummy number for cr_num
            self.cr_num = 9999 * u.dimensionless_unscaled
        elif cr_num is not None:
            # Find and load in the boundary condition file
            self.cr_num = cr_num * u.dimensionless_unscaled
            cr_tag = "CR{:03d}.hdf5".format(np.int32(self.cr_num.value))
            boundary_file = os.path.join(self._boundary_dir_, cr_tag)
            if os.path.exists(boundary_file):
                data = h5py.File(boundary_file, 'r')
                self.v_boundary = data['v_boundary'] * u.Unit(data['v_boundary'].attrs['unit'])
                data.close()
            else:
                print("Warning: {} not found. Defaulting to 400 km/s boundary".format(boundary_file))
                self.v_boundary = 400 * np.ones(128) * self.kms
        
        # Setup radial coordinates - in solar radius
        self.r, self.dr, self.rrel, self.Nr = radial_grid()
        
        self.simtime = (simtime * u.day).to(u.s)  # number of days to simulate (in seconds)
        self.dt_scale = dt_scale * u.dimensionless_unscaled
        time_grid_dict = time_grid(self.v_max, self.dr, self.simtime, self.dt_scale)
        self.dtdr = time_grid_dict['dtdr']
        self.Nt = time_grid_dict['Nt']
        self.dt = time_grid_dict['dt']
        self.time = time_grid_dict['time']
        self.Nt_out = time_grid_dict['Nt_out']
        self.dt_out = time_grid_dict['dt_out']
        self.time_out = time_grid_dict['time_out']
        del time_grid_dict
        
        # Preallocate space for the output for the solar wind fields for the cme and ambient solution.
        self.v_grid_cme = np.zeros((self.Nt_out, self.Nr)) * self.kms
        self.v_grid_amb = np.zeros((self.Nt_out, self.Nr)) * self.kms
        
        # Make an empty dictionary for storing ConeCMEs into
        self.cmes = []
        return
    
    def solve(self, cme_list, save=False, tag=''):
        """
        Solve HUXt1D for the specified inner boundary condition and list of cone cmes.

        Results are stored in the HUXt1D.v_grid_cme and HUXt1D.v_grid_amb attributes.

        :param cme_list: A list of ConeCME instances to insert into the model.
        :param save: Boolean. If True, saves model output to a HDF5 file.
        :param tag: Identifying string to append to the filename.
        """
        
        # Check only cone cmes in cme list
        cme_list_checked = []
        for cme in cme_list:
            if isinstance(cme, ConeCME):
                cme_list_checked.append(cme)
            else:
                print("Warning: cme_list contained objects other than ConeCME instances. These will be excluded")
        
        self.cmes = cme_list_checked
                
        buffersteps = np.fix((5.0*u.day).to(u.s) / self.dt)
        buffertime = buffersteps*self.dt
        model_time = np.arange(-buffertime.value, self.simtime.value + self.dt.value, self.dt.value) * self.dt.unit

        dlondt = self.twopi * self.dt / self.synodic_period
        lon, dlon, Nlon = longitude_grid()

        # How many radians of Carrington rotation in this simulation length
        simlon = self.twopi * self.simtime / self.synodic_period
        # How many radians of Carrington rotation in the spin up period
        bufferlon = self.twopi * buffertime / self.synodic_period
        # Find the Carrigton longitude range spanned by the spin up and simulation period,
        # centered on simulation longitude
        lonint = np.arange(self.lon.value-bufferlon, self.lon.value + simlon+dlondt, dlondt)
        # Rectify so that it is between 0 - 2pi
        loninit = _zerototwopi_(lonint)
        # Interpolate the inner boundary speed to this higher resolution
        vinit = np.interp(loninit, lon.value, self.v_boundary.value, period=self.twopi) * self.kms
        # convert from cr longitude to time
        vinput = np.flipud(vinit)

        # Main model loop
        # ----------------------------------------------------------------------------------------
        iter_count = 0
        t_out = 0
        for t, time in enumerate(model_time):

            # Get the initial condition, which will update in the loop,
            # and snapshots saved to output at right steps.
            if t == 0:
                v_cme = np.ones(self.Nr)*400*self.kms
                v_amb = np.ones(self.Nr)*400*self.kms

            # Update the inner boundary conditions
            v_amb[0] = vinput[t]
            v_cme[0] = vinput[t]
            
            # Compute boundary speed of each CME at this time. Set boundary to the maximum CME speed at this time.
            if time > 0:
                if len(cme_list_checked) != 0:
                    v_update_cme = np.zeros(len(cme_list_checked)) * self.kms
                    for i, c in enumerate(cme_list_checked):
                        r_boundary = self.r.min().to(u.km)
                        v_update_cme[i] = _cone_cme_boundary_1d_(r_boundary, self.lon, time, v_cme[0], c)

                    v_cme[0] = v_update_cme.max()

            # update cone cme v(r) for the given longitude
            # =====================================
            u_up = v_cme[1:].copy()
            u_dn = v_cme[:-1].copy()
            u_up_next = _upwind_step_(self, u_up, u_dn)
            # Save the updated time step
            v_cme[1:] = u_up_next.copy()

            u_up = v_amb[1:].copy()
            u_dn = v_amb[:-1].copy()
            u_up_next = _upwind_step_(self, u_up, u_dn)
            # Save the updated time step
            v_amb[1:] = u_up_next.copy()

            # Save this frame to output if output
            if time >= 0:
                iter_count = iter_count + 1
                if iter_count == self.dt_scale.value:
                    if t_out <= self.Nt_out - 1:
                        self.v_grid_cme[t_out, :] = v_cme.copy()
                        self.v_grid_amb[t_out, :] = v_amb.copy()
                        t_out = t_out + 1
                        iter_count = 0
        
        # Update CMEs positions by tracking through the solution.
        updated_cmes = []
        for cme in self.cmes:
            cme._track_1d_(self)
            updated_cmes.append(cme)
        
        self.cmes = updated_cmes
        
        if save:
            if tag == '':
                print("Warning, blank tag means file likely to be overwritten")
            self.save(tag=tag)
        return
    
    def save(self, tag=''):
        """
        Save model output to a HDF5 file.

        :param tag: Identifying string to append to the filename.
        """
        # Open up hdf5 data file for the HI flow stats
        filename = "HUXt1D_CR{:03d}_{}.hdf5".format(np.int32(self.cr_num.value), tag)
        out_filepath = os.path.join(self._data_dir_, filename)

        if os.path.isfile(out_filepath):
            # File exists, so delete and start new.
            print("Warning: {} already exists. Overwriting".format(out_filepath))
            os.remove(out_filepath)

        out_file = h5py.File(out_filepath, 'w')
        
        allcmes = out_file.create_group('ConeCMEs')
        for i, cme in enumerate(self.cmes):
            cme_name = "ConeCME_{:02d}".format(i)
            cmegrp = allcmes.create_group(cme_name)
            for k, v in cme.__dict__.items():
                if k != "coords":
                    dset = cmegrp.create_dataset(k, data=v.value)
                    dset.attrs['unit'] = v.unit.to_string()
                    out_file.flush()
                # Now handle the dictionary of CME boundary coordinates coords > time_out > position
                if k == "coords":
                    coordgrp = cmegrp.create_group(k)
                    for time, position in v.items():
                        time_label = "t_out_{:03d}".format(time)
                        timegrp = coordgrp.create_group(time_label)
                        for pos_label, pos_data in position.items():
                            dset = timegrp.create_dataset(pos_label, data=pos_data.value)
                            dset.attrs['unit'] = pos_data.unit.to_string()
                            out_file.flush()
                            
        # Loop over the attributes of model instance and save select keys/attributes.
        keys = ['cr_num', 'simtime', 'dt', 'v_max', 'r_accel', 'alpha',
                'dt_scale', 'time_out', 'dt_out', 'r', 'dr', 'lon', 'dlon',
                'v_grid_cme', 'v_grid_amb', 'v_boundary']
        for k, v in self.__dict__.items():
            if k in keys:
                dset = out_file.create_dataset(k, data=v.value)
                dset.attrs['unit'] = v.unit.to_string()
                # Add on the dimensions of the output speed fields.
                if k in ['v_grid_cme', 'v_grid_amb']:
                    dset.dims[0].label = 'time'
                    dset.dims[1].label = 'radius'

                out_file.flush()

        out_file.close()
        return out_filepath
    
    def plot_radial(self, time, field='cme', save=False, tag=''):
        """
        Plot the radial solar wind profile at model time closest to specified time.

        :param time: Time (in seconds) to find the closest model time step to.
        :param field: String, either 'cme', 'ambient', or 'both' specifying which solution to plot.
        :param save: Boolean to determine if the figure is saved.
        :param tag: String to append to the filename if saving the figure.
        :return: fig: Figure handle
        :return: ax: Axes handle
        """
        
        if field not in ['cme', 'ambient', 'both']:
            print("Error, field must be either 'cme', or 'ambient'. Default to cme")
            field = 'cme'
            
        if (time < self.time_out.min()) | (time > (self.time_out.max())):
            print("Error, input time outside span of model times. Defaulting to closest time")
            id_t = np.argmin(np.abs(self.time_out - time))
            time = self.time_out[id_t]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        # Get plotting data
        id_t = np.argmin(np.abs(self.time_out - time))
        if field == 'cme':
            label = 'Cone Run'
            ax.plot(self.r, self.v_grid_cme[id_t, :], 'k-', label=label)
        elif field == 'ambient':
            label = 'Ambient'
            ax.plot(self.r, self.v_grid_amb[id_t, :], '--', color='slategrey', label=label)
        elif field == 'both':
            label = 'Cone Run'
            ax.plot(self.r, self.v_grid_cme[id_t, :], 'k-', label=label)
            label = 'Ambient'
            ax.plot(self.r, self.v_grid_amb[id_t, :], '--', color='slategrey', label=label)
            
        # Plot the CME points on if needed
        if field in ['cme', 'both']:
            cme_colors = ['r', 'c', 'm', 'y', 'deeppink', 'darkorange']
            for c, cme in enumerate(self.cmes):
                cc = np.mod(c, len(cme_colors))
                id_r = np.int32(cme.coords[id_t]['r_pix'].value)
                label = "CME {:02d}".format(c)
                ax.plot(self.r[id_r], self.v_grid_cme[id_t, id_r], '.', color=cme_colors[cc], label=label)

        ax.set_ylim(250, 1500)
        ax.set_ylabel('Solar Wind Speed (km/s)')
        ax.set_xlim(self.r.value.min(), self.r.value.max())
        ax.set_xlabel('Radial distance ($R_{sun}$)')

        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)

        # Add label
        label = "HUXt1D    Time: {:3.2f} days".format(self.time_out[id_t].to(u.day).value)
        ax.set_title(label, fontsize=20)
        ax.legend(loc=1)
        if save:
            cr_num = np.int32(self.cr_num.value)
            filename = "HUXt1D_CR{:03d}_{}_{}_radial_profile_frame_{:03d}.png".format(cr_num, tag, field, id_t)
            filepath = os.path.join(self._figure_dir_, filename)
            fig.savefig(filepath)

        return fig, ax
    
    def plot_timeseries(self, radius, field='cme', save=False, tag=''):
        """
        Plot the solar wind model timeseries at model radius closest to specified radius.

        :param radius: Radius (in solar radii) to find the closest model radius to.
        :param field: String, either 'cme', 'ambient', or 'both' specifying which solution to plot.
        :param save: Boolean to determine if the figure is saved.
        :param tag: String to append to the filename if saving the figure.
        :return: fig: Figure handle
        :return: ax: Axes handle
        """
        
        if field not in ['cme', 'ambient', 'both']:
            print("Error, field must be either 'cme', or 'ambient'. Default to cme")
            field = 'cme'
            
        if (radius < self.r.min()) | (radius > (self.r.max())):
            print("Error, specified radius outside of model radial grid")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        # Get plotting data
        id_r = np.argmin(np.abs(self.r - radius))
        t_day = self.time_out.to(u.day)
        if field == 'cme':
            label = 'Cone Run'
            ax.plot(t_day, self.v_grid_cme[:, id_r], 'k-', label=label)
        elif field == 'ambient':
            label = 'Ambient'
            ax.plot(t_day, self.v_grid_amb[:, id_r], '--', color='slategrey', label=label)
        elif field == 'both':
            label = 'Cone Run'
            ax.plot(t_day, self.v_grid_cme[:, id_r], 'k-', label=label)
            label = 'Ambient'
            ax.plot(t_day, self.v_grid_amb[:, id_r], '--', color='slategrey', label=label)
            
        ax.set_ylim(250, 1500)
        ax.set_ylabel('Solar Wind Speed (km/s)')
        ax.set_xlim(t_day.value.min(), t_day.value.max())
        ax.set_xlabel('Time (days)')

        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)

        # Add label
        label = "HUXt1D    Radius: {:3.2f}".format(self.r[id_r].to(u.solRad).value) +  "$R_{sun}$"
        ax.set_title(label, fontsize=20)
        ax.legend(loc=1)
        if save:
            cr_num = np.int32(self.cr_num.value)
            filename = "HUXt1D_CR{:03d}_{}_{}_time_series_radius{:03d}.png".format(cr_num, tag, field, id_r)
            filepath = os.path.join(self._figure_dir_, filename)
            fig.savefig(filepath)

        return fig, ax

        
class HUXt2D:
    """
    A class containing the 2D HUXt model described in Owens et al. (2020).

    Users must specify the solar wind speed boundary condition through either the v_boundary, or cr_num keyword
    arguments. Failure to do so defaults to a 400 km/s boundary. v_boundary takes precedence over cr_num, so specifying
    both results in only v_boundary being used.
    
    Attributes:
        cmes: A list of ConeCME instances used in the model solution.
        cr_num: If provided, this gives the Carrington rotation number of the selected period, else 9999.
        daysec: seconds in a day.
        dlon: Longitudinal grid spacing (in radians)
        dr: Radial grid spacing (in km).
        dt: Model time step (in seconds), set by the CFL condition with v_max and dr.
        dt_out: Output model time step (in seconds).
        dt_scale: Integer scaling number to set the model output time step relative to the models CFL time step.
        dtdr: Ratio of the model time step and radial grid step (in seconds/km).
        kms: astropy.unit instance of km/s.       
        lon: Array of model longtidues (in radians).
        r_grid: Array of longitudinal coordinates meshed with the radial coordinates (in radians).
        Nlon: Number of longitudinal grid points.
        Nr: Number of radial grid points.
        Nt: Total number of model time steps, including spin up.
        Nt_out: Number of output model time steps.
        r_accel: Scale parameter determining the residual solar wind acceleration.
        r: Radial grid (in km).
        r_grid: Array of radial coordinates meshed with the longitudinal coordinates (in km).
        rrel: Radial grid relative to first grid point (in km).
        simtime: Simulation time (in seconds).
        synodic_period: Solar Synodic rotation period from Earth (in seconds).
        time: Array of model time steps, including spin up (in seconds).
        time_out: Array of output model time steps (in seconds).
        twopi: two pi radians
        v_boundary: Inner boundary solar wind speed profile (in km/s).
        v_grid_amb: Array of ambient model solution excluding ConeCMEs for each time, radius, and longitude (in km/s).
        v_grid_cme: Array of model solution inlcuding ConeCMEs for each time, radius, and longitude (in km/s).
        v_max: Maximum model speed (in km/s), used with the CFL condition to set the model time step. 
    """

    def __init__(self, v_boundary=None, cr_num=None, simtime=5.0, dt_scale=1.0):
        """
        Initialise the HUXt2D instance.

        :param v_boundary: Inner solar wind speed boundary condition. Must be an array of size 128 with units of km/s.
        :param cr_num: Integer Carrington rotation number. Used to lookup the longitudinal solar wind speed profile
                       at the solar equator from HelioMAS. This is then used as the inner boundary condition.
        :param simtime: Duration of the simulation window, in days.
        :param dt_scale: Integer scaling number to set the model output time step relative to the models CFL time.
        """

        # some constants and units
        constants = huxt_constants()
        self.twopi = constants['twopi']
        self.daysec = constants['daysec']
        self.kms = constants['kms']
        self.alpha = constants['alpha']  # Scale parameter for residual SW acceleration
        self.r_accel = constants['r_accel']  # Spatial scale parameter for residual SW acceleration
        self.synodic_period = constants['synodic_period']  # Solar Synodic rotation period from Earth.
        self.v_max = constants['v_max']
        del constants
        
        # Extract paths of figure and data directories
        dirs = _setup_dirs_()
        self._boundary_dir_ = dirs['boundary_conditions']
        self._data_dir_ = dirs['HUXt2D_data']
        self._figure_dir_ = dirs['HUXt2D_figures']
        
        # Determine the boundary conditions from input v_boundary and cr_num
        if (v_boundary is None) & (cr_num is None):
            print("Warning: No boudary conditions supplied. Defaulting to 400 km/s boundary")
            self.v_boundary = 400 * np.ones(128) * self.kms
        elif v_boundary is not None:
            assert v_boundary.unit == self.kms
            assert v_boundary.size == 128
            self.v_boundary = v_boundary
            # Set dummy number for cr_num
            self.cr_num = 9999 * u.dimensionless_unscaled
        elif cr_num is not None:
            # Find and load in the boundary condition file
            self.cr_num = cr_num * u.dimensionless_unscaled
            cr_tag = "CR{:03d}.hdf5".format(np.int32(self.cr_num.value))
            boundary_file = os.path.join(self._boundary_dir_, cr_tag)
            if os.path.exists(boundary_file):
                data = h5py.File(boundary_file, 'r')
                self.v_boundary = data['v_boundary'] * u.Unit(data['v_boundary'].attrs['unit'])
                data.close()
            else:
                print("Warning: {} not found. Defaulting to 400 km/s boundary".format(boundary_file))
                self.v_boundary = 400 * np.ones(128) * self.kms
        
        # Setup radial coordinates - in solar radius
        self.r, self.dr, self.rrel, self.Nr = radial_grid()
        
        # Setup longitude coordinates - in radians.
        self.lon, self.dlon, self.Nlon = longitude_grid()
        
        self.simtime = (simtime * u.day).to(u.s)  # number of days to simulate (in seconds)
        self.dt_scale = dt_scale * u.dimensionless_unscaled
        time_grid_dict = time_grid(self.v_max, self.dr, self.simtime, self.dt_scale)
        self.dtdr = time_grid_dict['dtdr']
        self.Nt = time_grid_dict['Nt']
        self.dt = time_grid_dict['dt']
        self.time = time_grid_dict['time']
        self.Nt_out = time_grid_dict['Nt_out']
        self.dt_out = time_grid_dict['dt_out']
        self.time_out = time_grid_dict['time_out']
        del time_grid_dict
        
        # Preallocate space for the output for the solar wind fields for the cme and ambient solution.
        self.v_grid_cme = np.zeros((self.Nt_out, self.Nr, self.Nlon)) * self.kms
        self.v_grid_amb = np.zeros((self.Nt_out, self.Nr, self.Nlon)) * self.kms

        # Mesh the spatial coordinates.
        self.lon_grid, self.r_grid = np.meshgrid(self.lon, self.r)
        
        # Empty dictionary for storing the coordinates of CME boundaries.
        self.cmes = []
        return

    def solve(self, cme_list, save=False, tag=''):
        """
        Solve HUXt2D for the specified inner boundary condition and list of cone cmes.

        Results are stored in the HUXt2D.v_grid_cme and HUXt2D.v_grid_amb attributes.

        :param cme_list: A list of ConeCME instances to insert into the model.
        :param save: Boolean. If True, saves model output to a HDF5 file.
        :param tag: Identifying string to append to the filename.
        """
        
        # Check only cone cmes in cme list
        cme_list_checked = []
        for cme in cme_list:
            if isinstance(cme, ConeCME):
                cme_list_checked.append(cme)
            else:
                print("Warning: cme_list contained objects other than ConeCME instances. These will be excluded")
        
        self.cmes = cme_list_checked

        # Initialise v from the steady-state solution - no spin up required
        # ----------------------------------------------------------------------------------------
        # compute the steady-state solution, as function of time, convert to function of long
        v_cr = self._solve_carrington_rotation_()
        v_cr = np.fliplr(v_cr)

        # Initialise the output
        self.v_grid_cme[0, :, :] = v_cr.copy()
        self.v_grid_amb[0, :, :] = v_cr.copy()

        # compute longitude increment that matches the time step dt
        dlondt = self.twopi * self.dt / self.synodic_period
        lon_tstep = np.arange(self.lon.value.min(), self.lon.value.max() + dlondt, dlondt) * u.rad
        # interpolate vin_long to this time step matched resolution
        v_boundary_tstep = np.interp(lon_tstep.value, self.lon.value, self.v_boundary, period=self.twopi)

        # Main model loop
        # ----------------------------------------------------------------------------------------
        iter_count = 0 
        t_out = 0
        for t in range(self.Nt):

            # Get the initial condition, which will update in the loop,
            # and snapshots saved to output at right steps.
            if t == 0:
                v_cme = self.v_grid_cme[t, :, :].copy()
                v_amb = self.v_grid_cme[t, :, :].copy()

            # loop through each longitude and compute the the 1-d radial solution
            for n in range(self.Nlon):
                # update cone cme v(r) for the given longitude
                # =====================================
                u_up = v_cme[1:, n].copy()
                u_dn = v_cme[:-1, n].copy()
                u_up_next = _upwind_step_(self, u_up, u_dn)
                # Save the updated time step
                v_cme[1:, n] = u_up_next.copy()

                u_up = v_amb[1:, n].copy()
                u_dn = v_amb[:-1, n].copy()
                u_up_next = _upwind_step_(self, u_up, u_dn)
                # Save the updated time step
                v_amb[1:, n] = u_up_next.copy()

            # Save this frame to output if output time step is a factor of time elapsed
            iter_count = iter_count + 1
            if iter_count == self.dt_scale.value:
                if t_out <= self.Nt_out - 1:  # Model can run one step longer than output steps, so check:
                    self.v_grid_cme[t_out, :, :] = v_cme.copy()
                    self.v_grid_amb[t_out, :, :] = v_amb.copy()
                    t_out = t_out + 1
                    iter_count = 0

            # Update boundary conditions for next time step
            # Ambient boundary
            v_boundary_tstep = np.roll(v_boundary_tstep, 1)
            v_boundary_update = np.interp(self.lon.value, lon_tstep.value, v_boundary_tstep)
            v_amb[0, :] = v_boundary_update
            
            #  Cone CME boundary
            if len(self.cmes) != 0:
                v_boundary_cone = v_boundary_tstep.copy()
                for cme in self.cmes:
                    r_boundary = self.r.min().to(u.km)
                    v_boundary_cone = _cone_cme_boundary_2d_(r_boundary, lon_tstep, v_boundary_cone, self.time[t], cme)
                    v_boundary_update = np.interp(self.lon.value, lon_tstep.value, v_boundary_cone)
            else:
                v_boundary_update = np.interp(self.lon.value, lon_tstep.value, v_boundary_tstep)
            
            v_cme[0, :] = v_boundary_update
            
        # Update CME positions
        updated_cmes = []
        for cme in self.cmes:
            cme._track_2d_(self)
            updated_cmes.append(cme)
        
        self.cmes = updated_cmes
        
        if save:
            if tag == '':
                print("Warning, blank tag means file likely to be overwritten")
            self.save(tag=tag)
        return

    def save(self, tag=''):
        """
        Save model output to a HDF5 file.

        :param tag: identifying string to append to the filename
        :return out_filepath: Full path to the saved file.
        """
        # Open up hdf5 data file for the HI flow stats
        filename = "HUXt2D_CR{:03d}_{}.hdf5".format(np.int32(self.cr_num.value), tag)
        out_filepath = os.path.join(self._data_dir_, filename)

        if os.path.isfile(out_filepath):
            # File exists, so delete and start new.
            print("Warning: {} already exists. Overwriting".format(out_filepath))
            os.remove(out_filepath)

        out_file = h5py.File(out_filepath, 'w')

        # Save the Cone CME parameters to a new group.
        allcmes = out_file.create_group('ConeCMEs')
        for i, cme in enumerate(self.cmes):
            cme_name = "ConeCME_{:02d}".format(i)
            cmegrp = allcmes.create_group(cme_name)
            for k, v in cme.__dict__.items():
                if k != "coords":
                    dset = cmegrp.create_dataset(k, data=v.value)
                    dset.attrs['unit'] = v.unit.to_string()
                    out_file.flush()
                # Now handle the dictionary of CME boundary coordinates coords > time_out > position
                if k == "coords":
                    coordgrp = cmegrp.create_group(k)
                    for time, position in v.items():
                        time_label = "t_out_{:03d}".format(time)
                        timegrp = coordgrp.create_group(time_label)
                        for pos_label, pos_data in position.items():
                            dset = timegrp.create_dataset(pos_label, data=pos_data.value)
                            dset.attrs['unit'] = pos_data.unit.to_string()
                            out_file.flush()

        # Loop over the attributes of model instance and save select keys/attributes.
        keys = ['cr_num', 'simtime', 'dt', 'v_max', 'r_accel', 'alpha',
                'dt_scale', 'time_out', 'dt_out', 'r', 'dr', 'lon', 'dlon', 'r_grid', 'lon_grid',
                'v_grid_cme', 'v_grid_amb', 'v_boundary']
        for k, v in self.__dict__.items():

            if k in keys:

                dset = out_file.create_dataset(k, data=v.value)
                dset.attrs['unit'] = v.unit.to_string()

                # Add on the dimensions of the spatial grids
                if k in ['r_grid', 'lon_grid']:
                    dset.dims[0].label = 'radius'
                    dset.dims[1].label = 'longtiude'

                # Add on the dimensions of the output speed fields.
                if k in ['v_grid_cme', 'v_grid_amb']:
                    dset.dims[0].label = 'time'
                    dset.dims[1].label = 'radius'
                    dset.dims[2].label = 'longtiude'

                out_file.flush()

        out_file.close()
        return out_filepath

    def plot(self, t, field='cme', save=False, tag=''):
        """
        Make a contour plot on polar axis of the solar wind solution at a specific time.

        :param t: Integer to index the time coordinate.
        :param field: String, either 'cme', or 'ambient', specifying which solution to plot.
        :param save: Boolean to determine if the figure is saved.
        :param tag: String to append to the filename if saving the figure.
        :return fig: Figure handle.
        :return ax: Axes handle.
        """
        
        if field not in ['cme', 'ambient']:
            print("Error, field must be either 'cme', or 'ambient'. Default to CME")
            field = 'cme'
            
        if (t < 0) | (t > (self.Nt_out-1)):
            print("Error, invalid time index t")
            
        # Get plotting data
        lon = self.lon_grid.value.copy()
        rad = self.r_grid.value.copy()
        if field == 'cme':
            v = self.v_grid_cme.value[t, :, :].copy()
        elif field == 'ambient':
            v = self.v_grid_amb.value[t, :, :].copy()
        
        # Pad out to fill the full 2pi of contouring
        pad = lon[:, 0].reshape((lon.shape[0], 1)) + self.twopi
        lon = np.concatenate((lon, pad), axis=1)
        pad = rad[:, 0].reshape((rad.shape[0], 1))
        rad = np.concatenate((rad, pad), axis=1)
        pad = v[:, 0].reshape((v.shape[0], 1))
        v = np.concatenate((v, pad), axis=1)
        
        mymap = mpl.cm.viridis
        mymap.set_over([1, 1, 1])
        mymap.set_under([0, 0, 0])
        dv = 10
        levels = np.arange(200, 800+dv, dv)
        fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={"projection": "polar"})
        cnt = ax.contourf(lon, rad, v, levels=levels, cmap=mymap, extend='both')
        
        # Add on CME boundaries
        if field == 'cme':
            cme_colors = ['r', 'c', 'm', 'y', 'deeppink', 'darkorange']
            for j, cme in enumerate(self.cmes):
                cid = np.mod(j, len(cme_colors))
                ax.plot(cme.coords[t]['lon'], cme.coords[t]['r'], '-', color=cme_colors[cid], linewidth=3)
            
        ax.set_ylim(0, 230)
        ax.set_yticklabels([])
        ax.tick_params(axis='x', which='both', pad=15)
        ax.patch.set_facecolor('slategrey')
        fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.95)

        # Add color bar
        pos = ax.get_position()
        dw = 0.005
        dh = 0.075
        left = pos.x0 + dw
        bottom = pos.y0 - dh
        wid = pos.width - 2*dw
        cbaxes = fig.add_axes([left, bottom, wid, 0.03])
        cbar1 = fig.colorbar(cnt, cax=cbaxes, orientation='horizontal')
        cbar1.set_label("Solar Wind Speed (km/s)")
        cbar1.set_ticks(np.arange(200, 900, 100))

        # Add label
        label = "Time: {:3.2f} days".format(self.time_out[t].to(u.day).value)
        fig.text(0.675, 0.17, label, fontsize=20)
        label = "HUXt2D"
        fig.text(0.175, 0.17, label, fontsize=20)
        if save:
            cr_num = np.int32(self.cr_num.value)
            filename = "HUXt2D_CR{:03d}_{}_frame_{:03d}.png".format(cr_num, tag, t)
            filepath = os.path.join(self._figure_dir_, filename)
            fig.savefig(filepath)

        return fig, ax

    def animate(self, field, tag):
        """
        Animate the model solution, and save as an MP4.

        :param field: String, either 'cme', or 'ambient', specifying which solution to animate.
        :param tag: String to append to the filename of the animation.
        """

        if field not in ['cme', 'ambient']:
            print("Error, field must be either 'cme', or 'ambient'. Default to CME")
            field = 'cme'
        
        # Set the duration of the movie
        # Scaled so a 5 day simulation with dt_scale=4 is a 10 second movie.
        duration = self.simtime.value * (10 / 432000)

        def make_frame(t):
            """
            Produce the frame required by MoviePy.VideoClip.

            :param t: time through the movie
            """
            # Get the time index closest to this fraction of movie duration
            i = np.int32((self.Nt_out-1) * t / duration)
            fig, ax = self.plot(i, field)
            frame = mplfig_to_npimage(fig)
            plt.close('all')           
            return frame

        cr_num = np.int32(self.cr_num.value)
        filename = "HUXt2D_CR{:03d}_{}_movie.mp4".format(cr_num, tag)
        filepath = os.path.join(self._figure_dir_, filename)
        animation = mpy.VideoClip(make_frame, duration=duration)
        animation.write_videofile(filepath, fps=24, codec='libx264')
        return

    def _solve_carrington_rotation_(self):
        """
        Solve the steady-state ambient solution.
        
        :return:
        """
        simtime = self.synodic_period  # One CR from Earth.
        buffertime = (5.0 * u.day).to(u.s)  # spin up time
        tmax = (simtime + buffertime)  # full simulation time

        # compute the longitude increment corresponding to timestep dt
        dlondt = self.twopi * self.dt / self.synodic_period
        # work out the phi increment to allow for the spin up
        bufferlon = self.twopi * buffertime / self.synodic_period
        # create the input timeseries including the spin up series, periodic in phi
        lonint = np.arange(0, self.twopi + bufferlon + dlondt, dlondt)
        loninit = _zerototwopi_(lonint)
        vinit = np.interp(loninit, self.lon.value, self.v_boundary.value, period=self.twopi) * self.kms
        # convert from longitude to time
        vinput = np.flipud(vinit)
        times = np.arange(0.0, (tmax + self.dt).value, self.dt.value)
        times = times * u.s

        # compute the rout time series
        # ==========================
        ts_allR = solve_upwind(self, vinput)

        # remove the spin up.
        id_after_spinup = times >= buffertime
        ts_allR = ts_allR[:, id_after_spinup]
        times = times[id_after_spinup]
        times = times - buffertime

        # interpolate back to the original longitudinal grid
        times_orig = self.synodic_period * self.lon.value / self.twopi
        vout_allR = np.zeros((self.r.size, times_orig.size)) * ts_allR.unit
        for j in range(self.r.size):
            vout = np.interp(times_orig.value, times.value, ts_allR[j, :].value)
            vout_allR[j, :] = vout * self.kms

        return vout_allR


def huxt_constants():
    """
    Return some constants used in all HUXt model classes
    """
    twopi = 2.0*np.pi
    daysec = 24*60*60
    kms = u.km / u.s
    alpha = 0.15*u.dimensionless_unscaled  # Scale parameter for residual SW acceleration
    r_accel = 50*u.solRad  # Spatial scale parameter for residual SW acceleration
    synodic_period = 27.2753*daysec*u.s  # Solar Synodic rotation period from Earth.
    v_max = 2000*kms
    constants = {'twopi': twopi, 'daysec': daysec, 'kms': kms, 'alpha': alpha,
                 'r_accel': r_accel, 'synodic_period': synodic_period, 'v_max': v_max}
    return constants


def radial_grid():
    """
    Define the radial grid of the HUXt model (1D and 2D)
    """
    r_min = 30.0
    r_max = 240.0
    Nr = 140
    r, dr = np.linspace(r_min, r_max, Nr, retstep=True)
    r = r*u.solRad
    dr = dr*u.solRad
    rrel = r - r[0]
    return r, dr, rrel, Nr


def longitude_grid():
    """
    Define the longitude grid of the HUXt2D model.
    """
    Nlon = 128
    twopi = 2.0*np.pi
    dlon = twopi / Nlon
    lon_min = dlon / 2.0
    lon_max = twopi - (dlon / 2.0)
    lon, dlon = np.linspace(lon_min, lon_max, Nlon, retstep=True)
    lon = lon*u.rad
    dlon = dlon*u.rad
    return lon, dlon, Nlon


def time_grid(v_max, dr, simtime, dt_scale):
    """
    Define the model timestep and time grid based on CFL condition and specified simulation time.
    """
    dt = dr.to(u.km) / v_max
    dtdr = dt / dr.to(u.km)

    Nt = np.int32(np.floor(simtime.value / dt.value))  # number of time steps in the simulation
    time = np.arange(0, Nt)*dt  # Model time steps
    
    dt_out = dt_scale*dt  # time step of the output
    Nt_out = np.int32(Nt / dt_scale.value)  # number of time steps in the output   
    time_out = np.arange(0, Nt_out)*dt_out  # Output time steps
    
    time_grid_dict = {'dt': dt, 'dtdr': dtdr, 'Nt': Nt, 'time': time,
                      'dt_out': dt_out, 'Nt_out': Nt_out, 'time_out': time_out}
    return time_grid_dict


def load_HUXt1D_run(filepath):
    """
    Load in data from a saved HUXt1D run.

    :param filepath: The full path to a HDF5 file containing the output from HUXt1D.save()
    :return: cme_list: A list of instances of ConeCME
    :return: model: An instance of HUXt1D containing the loaded results.
    """
    if os.path.isfile(filepath):

        data = h5py.File(filepath, 'r')
        
        # Initialise the model
        # TODO: check it matches what the resolution and limits of the HDF5 file??
        cr_num = np.int32(data['cr_num'])
        simtime = data['simtime'][()] * u.Unit(data['simtime'].attrs['unit'])
        simtime = simtime.to(u.day).value
        dt_scale = data['dt_scale'][()]
        lon = np.rad2deg(data['lon'])
        v_boundary = data['v_boundary'][()] * u.Unit(data['v_boundary'].attrs['unit'])
        if cr_num != 9999:
            model = HUXt1D(cr_num=cr_num, lon=lon, simtime=simtime, dt_scale=dt_scale)
        else:
            model = HUXt1D(v_boundary=v_boundary, lon=lon, simtime=simtime, dt_scale=dt_scale)
        
        model.v_grid_cme[:, :] = data['v_grid_cme'][()] * u.Unit(data['v_boundary'].attrs['unit'])
        model.v_grid_amb[:, :] = data['v_grid_amb'][()] * u.Unit(data['v_boundary'].attrs['unit'])
        
        # Load in the CME parameters
        cme_list = []

        all_cmes = data['ConeCMEs']
        for k in all_cmes.keys():
            cme_data = all_cmes[k]
            t_launch = cme_data['t_launch'][()]
            lon = np.rad2deg(cme_data['longitude'][()])
            width = np.rad2deg(cme_data['width'][()])
            thickness = cme_data['thickness'][()] * u.Unit(cme_data['thickness'].attrs['unit'])
            thickness = thickness.to('solRad').value
            v = cme_data['v'][()]
            cme = ConeCME(t_launch=t_launch, longitude=lon, v=v, width=width, thickness=thickness)
            
            coords_group = cme_data['coords']
            coords_data = {j: {'r_pix': np.array([])*u.pix, 'r': np.array([])*model.r.unit}
                           for j in range(len(coords_group))}

            for time_key, pos in coords_group.items():
                t = np.int(time_key.split("_")[2])
                coords_data[t]['r_pix'] = pos['r_pix'][()] * u.Unit(pos['r_pix'].attrs['unit'])
                coords_data[t]['r'] = pos['r'][()] * u.Unit(pos['r'].attrs['unit'])

            cme.coords = coords_data
            cme_list.append(cme)
            
        # Update CMEs in model output
        model.cmes = cme_list
    else:
        # File doesnt exist return nothing
        print("Warning: {} doesnt exist.".format(filepath))
        cme_list = []
        model = []

    return model, cme_list


def load_HUXt2D_run(filepath):
    """
    Load in data from a saved HUXt2D run.

    :param filepath: The full path to a HDF5 file containing the output from HUXt2D.save()
    :return: cme_list: A list of instances of ConeCME
    :return: model: An instance of HUXt2D containing loaded results.
    """
    if os.path.isfile(filepath):

        data = h5py.File(filepath, 'r')
        
        # Initialise the model
        # TODO: check it matches the resolution and limits of the HDF5 file??
        cr_num = np.int32(data['cr_num'])
        simtime = data['simtime'][()] * u.Unit(data['simtime'].attrs['unit'])
        simtime = simtime.to(u.day).value
        dt_scale = data['dt_scale'][()]
        v_boundary = data['v_boundary'][()] * u.Unit(data['v_boundary'].attrs['unit'])
        if cr_num != 9999:
            model = HUXt2D(cr_num=cr_num, simtime=simtime, dt_scale=dt_scale)
        else:
            model = HUXt2D(v_boundary=v_boundary, simtime=simtime, dt_scale=dt_scale)
        
        model.v_grid_cme[:, :, :] = data['v_grid_cme'][()] * u.Unit(data['v_boundary'].attrs['unit'])
        model.v_grid_amb[:, :, :] = data['v_grid_amb'][()] * u.Unit(data['v_boundary'].attrs['unit'])
        
        # Create list of the ConeCMEs
        cme_list = []
        all_cmes = data['ConeCMEs']
        for k in all_cmes.keys():
            cme_data = all_cmes[k]
            t_launch = cme_data['t_launch'][()]
            lon = np.rad2deg(cme_data['longitude'][()])
            width = np.rad2deg(cme_data['width'][()])
            thickness = cme_data['thickness'][()] * u.Unit(cme_data['thickness'].attrs['unit'])
            thickness = thickness.to('solRad').value
            v = cme_data['v'][()]
            cme = ConeCME(t_launch=t_launch, longitude=lon, v=v, width=width, thickness=thickness)

            # Now sort out coordinates.
            # Use the same dictionary structure as defined in ConeCME._track_2d_
            coords_group = cme_data['coords']
            coords_data = {j: {'lon_pix': np.array([])*u.pix, 'r_pix': np.array([])*u.pix,
                           'lon': np.array([])*model.lon.unit, 'r': np.array([])*model.r.unit}
                           for j in range(len(coords_group))}

            for time_key, pos in coords_group.items():
                t = np.int(time_key.split("_")[2])
                coords_data[t]['lon_pix'] = pos['lon_pix'][()] * u.Unit(pos['lon_pix'].attrs['unit'])
                coords_data[t]['r_pix'] = pos['r_pix'][()] * u.Unit(pos['r_pix'].attrs['unit'])
                coords_data[t]['lon'] = pos['lon'][()] * u.Unit(pos['lon'].attrs['unit'])
                coords_data[t]['r'] = pos['r'][()] * u.Unit(pos['r'].attrs['unit'])

            cme.coords = coords_data
            cme_list.append(cme)
            
        # Update CMEs in model output
        model.cmes = cme_list

    else:
        # File doesnt exist return nothing
        print("Warning: {} doesnt exist.".format(filepath))
        cme_list = []
        model = []

    return model, cme_list


def solve_upwind(model, v_input):
    """
    Solve the upwind scheme for Burgers equation for the time evolution of the radial wind speed.
    
    :param model: An instance of HUXt1D or HUXt2D.
    :param v_input: Time series of inner boundary solar wind speeds, in km/s.
    :return:
    """
    n_steps = v_input.size  # number of time steps
    # Initialise output speeds as 400kms everywhere
    v_out = np.ones((model.Nr, n_steps)) * 400.0 * model.kms
    # Update inner boundary condition
    v_out[0, :] = v_input.copy()

    # loop through time and compute the updated 1-d radial solution
    for t in range(1, n_steps):
        # Pull out the upwind and downwind slices at current time
        u_up = v_out[1:, t - 1].copy()
        u_dn = v_out[:-1, t - 1].copy()
        u_up_next = _upwind_step_(model, u_up, u_dn)
        # Save the updated time step
        v_out[1:, t] = u_up_next.copy()

    return v_out


def _upwind_step_(model, v_up, v_dn):
    """
    Compute the next step in the upwind scheme of Burgers equation with added acceleration of the solar wind.

    :param model: An instance of HUXt1D or HUXt2D.
    :param v_up: A numpy array of the upwind radial values. Units of km/s.
    :param v_dn: A numpy array of the downwind radial values. Units of km/s.
    :return: The upwind values at the next time step, numpy array with units of km/s.
    """
    # Arguments for computing the acceleration factor
    accel_arg = -model.rrel[:-1] / model.r_accel
    accel_arg_p = -model.rrel[1:] / model.r_accel

    # Get estimate of next time step
    v_up_next = v_up - model.dtdr*v_up*(v_up - v_dn)
    # Compute the probable speed at 30rS from the observed speed at r
    v_source = v_dn / (1.0 + model.alpha*(1.0 - np.exp(accel_arg)))
    # Then compute the speed gain between r and r+dr
    v_diff = model.alpha*v_source*(np.exp(accel_arg) - np.exp(accel_arg_p))
    # Add the residual acceleration over this grid cell
    v_up_next = v_up_next + (v_dn*model.dtdr*v_diff)
    return v_up_next


def _cone_cme_boundary_1d_(r_boundary, longitude, time, v_boundary, cme):
    """
    Update inner speed boundary condition with the time dependent cone cme speed, for HUXt1D.

    :param r_boundary: Height of model inner boundary.
    :param longitude: The modelled longitude in HUXt1D, in radians.
    :param v_boundary: Array of the ambient solar wind speed inner boundary condition, in km/s
    :param t: Model time step, in seconds
    :param cme: An instance of ConeCME.
    :return:
    """

    # Center the longitude array on CME nose, running from -pi to pi, to avoid dealing with any 0/2pi crossings
    lon_cent = longitude - cme.longitude
    id_high = lon_cent > np.pi*u.rad
    lon_cent[id_high] = 2.0*np.pi*u.rad - lon_cent[id_high]
    id_low = lon_cent < -np.pi*u.rad
    lon_cent[id_low] = lon_cent[id_low] + 2.0*np.pi*u.rad 
    
    if (lon_cent >= -cme.width/2) & (lon_cent <= cme.width/2):
        # Longitude inside CME span.
        #  Compute y, the height of CME nose above the 30rS surface
        y = cme.v*(time - cme.t_launch)
        x = np.NaN*y.unit
        if (y >= 0*u.km) & (y < cme.radius):
            # this is the front hemisphere of the spherical CME
            x = np.sqrt(y*(2*cme.radius - y))  # compute x, the distance of the current longitude from the nose
        elif (y >= (cme.radius + cme.thickness)) & (y <= (2*cme.radius + cme.thickness)):
            # this is the back hemisphere of the spherical CME
            y = y - cme.thickness
            x = np.sqrt(y*(2*cme.radius - y))
        elif (cme.thickness > 0*u.km) & (y >= cme.radius) & (y <= (cme.radius + cme.thickness)):
            # this is the "mass" between the hemispheres
            x = cme.radius
            
        theta = np.arctan(x / r_boundary)
        if (lon_cent >= - theta) & (lon_cent <= theta):
            v_boundary = cme.v
            
    return v_boundary


def _cone_cme_boundary_2d_(r_boundary, longitude, v_boundary, t, cme):
    """
    Update inner speed boundary condition with the time dependent cone cme speed, for HUXt2D.

    :param r_boundary: Height of model inner boundary.
    :param longitude: Array of model longitudes, in radians.
    :param v_boundary: Array of the ambient solar wind speed inner boundary condition, in km/s
    :param t: Model time step, in seconds
    :param cme: An instance of ConeCME.
    :return:
    """
    # Center the longitude array on CME nose, running from -pi to pi, to avoid dealing with any 0/2pi crossings
    lon_cent = longitude - cme.longitude
    id_high = lon_cent > np.pi*u.rad
    lon_cent[id_high] = 2.0*np.pi*u.rad - lon_cent[id_high]
    id_low = lon_cent < -np.pi*u.rad
    lon_cent[id_low] = lon_cent[id_low] + 2.0*np.pi*u.rad 
    
    #  Compute y, the height of CME nose above the 30rS surface
    y = cme.v*(t - cme.t_launch)
    if (y >= 0*u.km) & (y < cme.radius):  # this is the front hemisphere of the spherical CME
        x = np.sqrt(y*(2*cme.radius - y))  # compute x, the distance of the current longitude from the nose
        # convert x back to an angular separation
        theta = np.arctan(x / r_boundary)
        pos = (lon_cent > - theta) & (lon_cent <= theta)
        v_boundary[pos] = cme.v 
    elif (y >= (cme.radius + cme.thickness)) & (y <= (2*cme.radius + cme.thickness)):
        # this is the back hemisphere of the spherical CME
        y = y - cme.thickness
        x = np.sqrt(y*(2*cme.radius - y))
        # convert back to an angle
        theta = np.arctan(x / r_boundary)
        pos = (lon_cent > - theta) & (lon_cent <= theta)
        v_boundary[pos] = cme.v 
    elif (cme.thickness > 0*u.km) & (y >= cme.radius) & (y <= (cme.radius + cme.thickness)):
        # this is the "mass" between the hemispheres
        x = cme.radius
        # convert back to an angle
        theta = np.arctan(x / r_boundary)
        pos = (lon_cent > - theta) & (lon_cent <= theta)
        v_boundary[pos] = cme.v 

    return v_boundary


def _zerototwopi_(angles):
    """
    Function to constrain angles to the 0 - 2pi domain.

    :param angles: a numpy array of angles
    :return: a numpy array of angles
    """
    twopi = 2.0*np.pi
    angles_out = angles.copy()
    a = -np.floor_divide(angles_out, twopi)
    angles_out = angles_out + (a*twopi)
    return angles_out


def _setup_dirs_():
    """
    Function to pull out the directories to save figures and output data.
    """
    # Find the config.dat file path
    files = glob.glob('config.dat')

    if len(files) != 1:
        # If too few or too many config files, guess directories
        print('Error: Cannot find correct config file with project directories. Check config.txt exists')
        print('Defaulting to current directory')
        dirs = {'root': os.getcwd()}
        for rel_path in ['boundary_conditions', 'HUXt1D_data', 'HUXt2D_data', 'HUXt1D_figures', 'HUXt2D_figures']:
            dirs[rel_path] = os.getcwd()
    else:
        # Extract data and figure directories from config.dat
        with open(files[0], 'r') as file:
            lines = file.read().splitlines()
            root = lines[0].split(',')[1]
            dirs = {line.split(',')[0]: os.path.join(root, line.split(',')[1]) for line in lines[1:]}

        # Just check the directories exist.
        for val in dirs.values():
            if not os.path.exists(val):
                print('Error, invalid path, check config.dat: ' + val)

    return dirs

