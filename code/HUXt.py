import numpy as np
import astropy.units as u


class HUXt2D:

    def __init__(self):

        # some constants
        twopi = 2.0 * np.pi
        daysec = 24 * 60 * 60
        self.alpha = 0.15  # Scale parameter for residual SW acceleration
        self.r_acc = 40 * u.solRad  # Spatial scale parameter for residual SW acceleration
        self.synodic_period = 27.2753 * daysec * u.s  # Solar Synodic rotation period from Earth.
        self.vmax = 2000.0 * u.km / u.s  # Maximum speed expected in model, used for CFL condition

        # Setup radial coordinates
        rmin = 30.0
        rmax = 240.0
        self.Nr = 140
        self.r, self.dr = np.linspace(rmin, rmax, self.Nr, retstep=True)
        self.r = self.r * u.solRad
        self.dr = self.dr * u.solRad

        # Set up longitudinal coordinates
        self.Nphi = 128
        dphi = twopi / self.Nphi
        phimin = dphi / 2.0
        phimax = twopi - (dphi / 2.0)
        self.phi, self.dphi = np.linspace(phimin, phimax, self.Nphi, retstep=True)
        self.phi = self.phi * u.rad
        self.phi = self.dphi * u.rad

        # Mesh the coordinates.
        self.phi_grid, self.r_grid = np.meshgrid(self.phi, self.r)

        self.v_ambient = self.__ambient_solar_wind_


