"""
huxt_fluxrope.py - Flux Rope in 3D (FRiED) magnetic field model for CMEs.

Implements the FRiED methodology of Isavnin (2016, ApJ, 833, 267) to generate
synthetic in-situ magnetic field time series for a 3D flux-rope CME.

The model constructs a croissant-shaped flux rope attached to the Sun at both
footpoints, with:
  - Axis shape governed by a force-balance solution (Eq. 14 of Isavnin 2016)
  - Circular cross-section tapering to zero at the footpoints (Eq. 1)
  - Lundquist-type magnetic field distribution (Eq. 16)
  - Constant twist tau applied to field lines
  - Magnetic flux conservation along the axis (Eqs. 17-19)
  - Global deformations: front flattening (n), pancaking (delta), skewing (sigma)
  - Evolution via expanding toroidal height Rt and poloidal height Rp

References:
    Isavnin, A. 2016, ApJ, 833, 267 (FRiED model)
    doi:10.3847/1538-4357/833/2/267

    Lundquist, S. 1950, Ark. Fys., 2, 361
"""

import numpy as np
import astropy.units as u
from scipy.special import j0, j1
from copy import deepcopy


# Bessel function first zero: J0(alpha_0) = 0
_ALPHA0 = 2.4048255577


def _zerototwopi(x):
    """Map angle x into [0, 2pi)."""
    return x % (2.0 * np.pi)


# =============================================================================
# Axis geometry
# =============================================================================

def fri3d_axis(phi, half_width, n=1.0):
    """
    Compute the radial distance along the FRiED CME axis in polar coordinates.

    Approximate solution to the force-balance equation (Eq. 14 of Isavnin 2016):
        r(phi) = sin(phi / phi_hw * pi/2)^n

    The result is normalised so that r=1 at the apex (phi = phi_hw).

    Args:
        phi: Polar angle(s) along the axis, in radians. 0 <= phi <= 2*phi_hw.
        half_width: Angular half-width (lambda) of the CME axis, in radians.
        n: Front-flattening coefficient (n > 0). n=1 gives circular front,
           n<1 flattens the front, n>1 sharpens it.

    Returns:
        r_norm: Normalised radial distance (0 at footpoints, 1 at apex).
    """
    phi = np.asarray(phi, dtype=float)
    hw = float(half_width)
    # Map phi into [0, 2*hw]
    s = np.clip(phi / hw, 0.0, 2.0)
    # Symmetric about the apex at s=1
    r_norm = np.sin(np.minimum(s, 2.0 - s) * np.pi / 2.0) ** n
    return r_norm


def fri3d_axis_3d(phi, Rt, Rp, half_width, n=1.0, tilt=0.0, theta=0.0,
                  phi_dir=0.0, delta=0.0, sigma=0.0):
    """
    Compute the 3D axis of the FRiED CME in HEEQ Cartesian coordinates.

    The axis lies in a plane that is tilted by angle `tilt` (gamma) w.r.t. the
    solar equatorial plane and directed towards (theta, phi_dir).

    Args:
        phi: Array of polar angles along the axis (radians), 0 to 2*half_width.
        Rt: Toroidal height — heliocentric distance to the apex of the axis (solar radii or km).
        Rp: Poloidal height — radius of cross-section at the apex.
        half_width: Angular half-width of the axis (radians).
        n: Front-flattening coefficient.
        tilt: Tilt angle gamma of the flux-rope axis plane (radians).
        theta: HEEQ latitude of the propagation direction (radians).
        phi_dir: HEEQ longitude of the propagation direction (radians).
        delta: Pancaking angle (radians). 0 = no pancaking.
        sigma: Skewing angle (radians). 0 = no skew.

    Returns:
        xyz: (N, 3) array of HEEQ Cartesian positions along the axis.
        s_arc: (N,) array of cumulative arc length along the axis.
    """
    phi = np.asarray(phi, dtype=float)

    # Normalised radial distance along axis
    r_norm = fri3d_axis(phi, half_width, n=n)

    # Axis in the CME plane (u, v) where u is radial, v is lateral
    # phi runs from 0 (one footpoint, east) to 2*hw (other footpoint, west)
    # The apex is at phi = hw.
    hw = float(half_width)
    angle = (phi / hw - 1.0) * hw  # angle from apex, in [-hw, +hw]
    u = r_norm * np.cos(angle) * Rt  # radial
    v = r_norm * np.sin(angle) * Rt  # lateral

    # Apply skewing: rotate in the CME plane by sigma
    if sigma != 0.0:
        cs, ss = np.cos(sigma), np.sin(sigma)
        u2 = cs * u - ss * v
        v2 = ss * u + cs * v
        u, v = u2, v2

    # Out-of-plane (w) for pancaking — stretch in w direction
    w = np.zeros_like(u)

    # Now rotate into HEEQ frame
    # 1. Tilt about the radial direction (gamma)
    # 2. Rotate to (theta, phi_dir) propagation direction
    ct, st = np.cos(tilt), np.sin(tilt)
    v2 = ct * v - st * w
    w2 = st * v + ct * w
    v, w = v2, w2

    # Apply pancaking: stretch w by factor related to delta
    if delta != 0.0:
        pancake_factor = np.tan(delta) / np.tan(half_width) if half_width > 0 else 1.0
        w = w * pancake_factor

    # Rotate to propagation direction
    cth, sth = np.cos(theta), np.sin(theta)
    cph, sph = np.cos(phi_dir), np.sin(phi_dir)

    # u is along radial, v is lateral (in equatorial plane), w is vertical
    x = cph * cth * u - sph * v - cph * sth * w
    y = sph * cth * u + cph * v - sph * sth * w
    z = sth * u + cth * w

    xyz = np.column_stack([x, y, z])

    # Arc length
    dxyz = np.diff(xyz, axis=0)
    ds = np.sqrt(np.sum(dxyz ** 2, axis=1))
    s_arc = np.concatenate([[0.0], np.cumsum(ds)])

    return xyz, s_arc


# =============================================================================
# Cross-section
# =============================================================================

def cross_section_radius(phi, half_width, Rp, n=1.0):
    """
    Compute the cross-section radius at position phi along the axis.

    The cross-section diameter varies proportionally to the heliocentric 
    distance along the axis (Eq. 1 of Isavnin 2016):
        d(phi) = 2 * Rp * r(phi)

    where r(phi) is the normalised axis distance.  The radius is d/2.

    Args:
        phi: Polar angle(s) along the axis (radians).
        half_width: Angular half-width of the CME axis (radians).
        Rp: Poloidal height — cross-section radius at the apex.
        n: Front-flattening coefficient.

    Returns:
        rho_max: Cross-section radius at each phi.
    """
    r_norm = fri3d_axis(phi, half_width, n=n)
    return Rp * r_norm


# =============================================================================
# Magnetic field: Lundquist + constant twist
# =============================================================================

def lundquist_field(rho_norm, B0=1.0):
    """
    Lundquist force-free magnetic field components in cylindrical coordinates.

    B_axial  = B0 * J0(alpha_0 * rho/R)
    B_azimuthal = B0 * J1(alpha_0 * rho/R)

    (Eq. 16 of Isavnin 2016, cf. Lundquist 1950)

    Args:
        rho_norm: Normalised poloidal distance from the axis (0 = axis, 1 = edge).
        B0: Core magnetic field strength (at rho=0).

    Returns:
        B_ax: Axial (toroidal) component.
        B_az: Azimuthal (poloidal) component.
    """
    rho_norm = np.asarray(rho_norm, dtype=float)
    arg = _ALPHA0 * rho_norm
    B_ax = B0 * j0(arg)
    B_az = B0 * j1(arg)
    return B_ax, B_az


def magnetic_flux_integrand(rho_norm):
    """
    Integrand for axial magnetic flux through a circular cross-section.

    Phi = integral_0^R B_ax(rho) * 2*pi*rho drho
        = B0 * 2*pi*R^2 * integral_0^1 J0(alpha0*x) * x dx

    The integral of x*J0(alpha0*x) from 0 to 1 is J1(alpha0)/alpha0.
    So: Phi = B0 * 2*pi*R^2 * J1(alpha0)/alpha0  (Eqs. 18-19 of Isavnin 2016)

    Returns:
        Scalar value of the integral J1(alpha0)/alpha0.
    """
    return j1(_ALPHA0) / _ALPHA0


def B0_from_flux(Phi, R):
    """
    Compute the core field strength B0 needed to give total axial magnetic 
    flux Phi through a circular cross-section of radius R.

    From Phi = B0 * 2*pi*R^2 * J1(alpha0)/alpha0  (Eq. 18)

    Args:
        Phi: Total axial magnetic flux (Wb if SI, or any consistent unit).
        R: Cross-section radius (m if SI).

    Returns:
        B0: Core field strength (T if SI).
    """
    integral = magnetic_flux_integrand(None)
    return Phi / (2.0 * np.pi * R ** 2 * integral)


# =============================================================================
# FRiED flux-rope CME class
# =============================================================================

class FRiEDCME:
    """
    A Flux Rope in 3D (FRiED) CME model following Isavnin (2016).

    This class defines the 3D geometry and magnetic field of a flux-rope CME
    and can generate synthetic in-situ magnetic field time series for a
    spacecraft trajectory passing through the structure.

    The magnetic field follows the Lundquist force-free solution (Eq. 16 of
    Isavnin 2016):
        B_axial    = B0 * J0(alpha * rho/R)   -- along the flux rope axis
        B_azimuthal = B0 * J1(alpha * rho/R)  -- wrapping around the axis
        B_radial   = 0                         -- no radial component
    where alpha = 2.4048 (first zero of J0), rho is distance from axis, and R
    is the local cross-section radius.

    Note on twist parameter: In the Lundquist solution, the twist (number of
    turns field lines make around the axis) is intrinsically determined by the
    Bessel function ratio J1/J0 and is not a free parameter. The 'twist'
    parameter here is used for 3D field line visualization but does not modify
    the local magnetic field calculation. For variable twist, a different
    force-free model (e.g., Gold-Hoyle) would be needed.

    Parameters (all with astropy units):
        Rt: Toroidal height — heliocentric distance to apex (u.solRad).
        Rp: Poloidal height — cross-section radius at apex (u.solRad).
        half_width: Angular half-width of the CME axis (u.deg or u.rad).
        n: Front-flattening coefficient (dimensionless, > 0).
        tilt: Tilt angle of the flux-rope plane (u.deg or u.rad).
        theta: HEEQ latitude of propagation direction (u.deg or u.rad).
        phi_dir: HEEQ longitude of propagation direction (u.deg or u.rad).
        delta: Pancaking angle (u.deg or u.rad).
        sigma: Skewing angle (u.deg or u.rad).
        twist: Total twist in turns (for 3D visualization, not field calculation).
        Phi: Total axial magnetic flux (u.Wb or u.T * u.m**2).
        polarity: +1 (east-west core field) or -1 (west-east core field).
        chirality: +1 (right-handed) or -1 (left-handed).
        v_propagation: Speed of toroidal height growth (u.km/u.s). For evolving CME.
        v_expansion: Speed of poloidal height growth (u.km/u.s). For expanding CME.
    """

    def __init__(self,
                 Rt=100.0 * u.solRad,
                 Rp=15.0 * u.solRad,
                 half_width=60.0 * u.deg,
                 n=1.0,
                 tilt=0.0 * u.deg,
                 theta=0.0 * u.deg,
                 phi_dir=0.0 * u.deg,
                 delta=0.0 * u.deg,
                 sigma=0.0 * u.deg,
                 twist=2.0,
                 Phi=1e13 * u.Wb,
                 polarity=1,
                 chirality=1,
                 v_propagation=0.0 * u.km / u.s,
                 v_expansion=0.0 * u.km / u.s):
        """
        Initialise a FRiED CME with specified parameters.

        Args:
            Rt: Toroidal height (heliocentric distance to apex) in solar radii.
            Rp: Poloidal height (cross-section radius at apex) in solar radii.
            half_width: Angular half-width of CME axis.
            n: Front-flattening coefficient (>= 0.5).
            tilt: Tilt angle of flux-rope plane.
            theta: HEEQ latitude of propagation direction.
            phi_dir: HEEQ longitude of propagation direction.
            delta: Pancaking angle.
            sigma: Skewing angle.
            twist: Total twist in turns (for 3D visualization only; Lundquist
                   field twist is fixed by the Bessel function solution).
            Phi: Total axial magnetic flux.
            polarity: +1 or -1 for core field direction.
            chirality: +1 (right-handed) or -1 (left-handed).
            v_propagation: Radial propagation speed (for evolving CME).
            v_expansion: Poloidal expansion speed (for evolving CME).
        """
        self.Rt = Rt.to(u.solRad)
        self.Rp = Rp.to(u.solRad)
        self.half_width = half_width.to(u.rad)
        self.n = float(n)
        self.tilt = tilt.to(u.rad)
        self.theta = theta.to(u.rad)
        self.phi_dir = phi_dir.to(u.rad)
        self.delta = delta.to(u.rad)
        self.sigma = sigma.to(u.rad)
        self.twist = float(twist)  # total turns
        self.Phi = Phi.to(u.Wb)
        self.polarity = int(np.sign(polarity)) if polarity != 0 else 1
        self.chirality = int(np.sign(chirality)) if chirality != 0 else 1
        self.v_propagation = v_propagation.to(u.km / u.s)
        self.v_expansion = v_expansion.to(u.km / u.s)

        if self.n <= 0.0:
            raise ValueError(f"Front-flattening coefficient n must be > 0, got {self.n}")

    def __repr__(self):
        return (f"FRiEDCME(Rt={self.Rt:.1f}, Rp={self.Rp:.1f}, "
                f"half_width={self.half_width.to(u.deg):.1f}, n={self.n:.2f}, "
                f"tilt={self.tilt.to(u.deg):.1f}, twist={self.twist:.1f}, "
                f"Phi={self.Phi:.2e}, pol={self.polarity}, chi={self.chirality})")

    def get_axis(self, n_points=500):
        """
        Compute the 3D axis of the CME in HEEQ Cartesian coordinates (in solar radii).

        Args:
            n_points: Number of points along the axis.

        Returns:
            xyz: (N, 3) array of HEEQ Cartesian positions (solar radii).
            s_arc: (N,) array of cumulative arc length (solar radii).
            phi_arr: (N,) array of polar angles along the axis (radians).
        """
        hw = self.half_width.value
        phi_arr = np.linspace(0, 2.0 * hw, n_points)

        xyz, s_arc = fri3d_axis_3d(
            phi_arr,
            Rt=self.Rt.value,
            Rp=self.Rp.value,
            half_width=hw,
            n=self.n,
            tilt=self.tilt.value,
            theta=self.theta.value,
            phi_dir=self.phi_dir.value,
            delta=self.delta.value,
            sigma=self.sigma.value,
        )
        return xyz, s_arc, phi_arr

    def get_axis_length(self, n_points=500):
        """
        Compute the total arc length of the CME axis.

        Args:
            n_points: Number of points for numerical integration.

        Returns:
            L: Total arc length in solar radii (astropy Quantity).
        """
        _, s_arc, _ = self.get_axis(n_points=n_points)
        return s_arc[-1] * u.solRad

    def _evolve_params(self, t_seconds):
        """
        Return evolved (Rt, Rp) at time t_seconds after reference epoch.

        Rt(t) = Rt0 + v_propagation * t   (Eq. 20)
        Rp(t) = Rp0 + v_expansion * t     (Eq. 21)

        Args:
            t_seconds: Time in seconds relative to epoch when Rt, Rp are defined.

        Returns:
            Rt_t: Evolved toroidal height (solar radii, float).
            Rp_t: Evolved poloidal height (solar radii, float).
        """
        solrad_km = u.solRad.to(u.km)  # km per solar radius
        Rt_t = self.Rt.value + (self.v_propagation.value * t_seconds) / solrad_km
        Rp_t = self.Rp.value + (self.v_expansion.value * t_seconds) / solrad_km
        # Rp cannot be negative
        Rp_t = max(Rp_t, 0.01)
        Rt_t = max(Rt_t, 0.01)
        return Rt_t, Rp_t

    def magnetic_field_at_point(self, x, y, z, Rt=None, Rp=None, Rt_ref=None, n_axis=500):
        """
        Compute the magnetic field vector (Bx, By, Bz) in HEEQ at a point.

        The field is zero outside the flux rope boundary.

        When Rt_ref is provided (evolving CME), apply toroidal-stretching
        scaling to weaken only the poloidal (azimuthal) component as ~Rt_ref/Rt,
        keeping the axial component and geometry fixed unless Rp evolves. This
        produces the "magnetic erosion" asymmetry described in Isavnin (2016,
        Section 5): the field is stronger when the CME apex is closer to the
        Sun (leading edge) and weaker when it is farther out (trailing edge).

        The cross-section is elliptical due to pancaking (delta parameter):
          - Semi-axis a (in CME plane, radial from Sun direction)
          - Semi-axis b (perpendicular to CME plane) = a * aspect_ratio
          - aspect_ratio = 1/cos(delta) for pancaking (Isavnin 2016, Eq. 8)

        Args:
            x, y, z: HEEQ Cartesian coordinates of the point (solar radii, floats).
            Rt: Override toroidal height (solar radii, float). If None, use self.Rt.
            Rp: Override poloidal height (solar radii, float). If None, use self.Rp.
            Rt_ref: Reference toroidal height for self-similar expansion scaling
                    (solar radii, float). If None, no expansion scaling is applied.
            n_axis: Number of discretisation points along the axis.

        Returns:
            B_heeq: (3,) array of (Bx, By, Bz) in Tesla.
        """
        if Rt is None:
            Rt_val = self.Rt.value
        else:
            Rt_val = float(Rt)
        if Rp is None:
            Rp_val = self.Rp.value
        else:
            Rp_val = float(Rp)

        hw = self.half_width.value
        phi_arr = np.linspace(0, 2.0 * hw, n_axis)

        # Get the axis positions in HEEQ
        xyz_axis, s_arc, _ = self.get_axis(n_points=n_axis)

        # Scale axis for the current Rt
        scale_Rt = Rt_val / self.Rt.value if self.Rt.value > 0 else 1.0
        xyz_axis_scaled = xyz_axis * scale_Rt
        s_arc_scaled = s_arc * scale_Rt

        point = np.array([x, y, z])

        # Find the closest point on the polyline axis by segment projection
        best_dist = 1e30
        best_idx = 0
        best_t = 0.0
        for i in range(n_axis - 1):
            p0 = xyz_axis_scaled[i]
            p1 = xyz_axis_scaled[i + 1]
            v = p1 - p0
            vv = np.dot(v, v)
            if vv < 1e-30:
                continue
            t_seg = np.dot(point - p0, v) / vv
            t_clamped = np.clip(t_seg, 0.0, 1.0)
            proj = p0 + t_clamped * v
            dist = np.linalg.norm(point - proj)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
                best_t = t_clamped

        phi_star = phi_arr[best_idx] + best_t * (phi_arr[best_idx + 1] - phi_arr[best_idx])
        s_local = s_arc_scaled[best_idx] + best_t * (s_arc_scaled[best_idx + 1] - s_arc_scaled[best_idx])
        axis_star = xyz_axis_scaled[best_idx] + best_t * (xyz_axis_scaled[best_idx + 1] - xyz_axis_scaled[best_idx])

        # Magnetic expansion from toroidal stretching: in FRiED, conserving
        # toroidal flux primarily weakens the poloidal (azimuthal) component
        # as ~1/Rt. We keep the cross-section size fixed (unless Rp evolves)
        # and apply the Rt scaling only to the azimuthal field, leaving the
        # axial (toroidal) component unchanged.
        if Rt_ref is not None and float(Rt_ref) > 0:
            az_scale = float(Rt_ref) / Rt_val
        else:
            az_scale = 1.0

        # Base cross-section radius at this axis position (circular case)
        r_cs_base = cross_section_radius(phi_star, hw, Rp_val, n=self.n)
        if r_cs_base < 1e-10:
            return np.zeros(3)

        # Compute local coordinate frame for the cross-section
        # Tangent to axis (toroidal/axial direction)
        v_seg = xyz_axis_scaled[best_idx + 1] - xyz_axis_scaled[best_idx]
        tangent = v_seg / (np.linalg.norm(v_seg) + 1e-30)

        # Define the local cross-section coordinate system:
        # - e_a: in-plane direction (toward/away from Sun, in the CME plane)
        # - e_b: out-of-plane direction (perpendicular to CME plane)
        # 
        # For a CME propagating radially, the CME plane contains the axis and
        # the Sun. The "in-plane" direction at any point on the axis points
        # away from the Sun (major radius direction).
        
        # Major-radius direction (from Sun to axis point)
        major_dir = axis_star.copy()
        major_norm = np.linalg.norm(major_dir)
        if major_norm > 1e-15:
            major_dir = major_dir / major_norm
        else:
            major_dir = np.array([1.0, 0.0, 0.0])
        
        # Project out the tangent component to get the in-plane direction
        e_a = major_dir - np.dot(major_dir, tangent) * tangent
        e_a_norm = np.linalg.norm(e_a)
        if e_a_norm > 1e-15:
            e_a = e_a / e_a_norm
        else:
            # Axis is radial; pick arbitrary perpendicular
            tmp = np.cross(tangent, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(tmp) < 1e-15:
                tmp = np.cross(tangent, np.array([0.0, 1.0, 0.0]))
            e_a = tmp / (np.linalg.norm(tmp) + 1e-30)

        # Out-of-plane direction (perpendicular to both tangent and e_a)
        e_b = np.cross(tangent, e_a)
        e_b = e_b / (np.linalg.norm(e_b) + 1e-30)

        # Elliptical cross-section due to pancaking (delta)
        # Following Isavnin 2016: pancaking stretches the cross-section
        # perpendicular to the CME plane (the e_b direction).
        # Aspect ratio: b/a = 1/cos(delta) where delta is pancaking angle
        delta_val = self.delta.value
        if abs(delta_val) > 1e-10:
            aspect_ratio = 1.0 / np.cos(delta_val)
        else:
            aspect_ratio = 1.0
        
        # Semi-axes of elliptical cross-section
        semi_a = r_cs_base  # in-plane (radial) semi-axis
        semi_b = r_cs_base * aspect_ratio  # out-of-plane semi-axis (stretched)

        # Vector from axis to point
        d_vec = point - axis_star
        # Remove axial component (project onto cross-section plane)
        d_perp = d_vec - np.dot(d_vec, tangent) * tangent

        # Decompose into cross-section coordinates
        coord_a = np.dot(d_perp, e_a)  # in-plane coordinate
        coord_b = np.dot(d_perp, e_b)  # out-of-plane coordinate

        # Normalised elliptical distance: (coord_a/semi_a)^2 + (coord_b/semi_b)^2
        # If this is > 1, point is outside the flux rope
        rho_ellipse_sq = (coord_a / semi_a) ** 2 + (coord_b / semi_b) ** 2
        if rho_ellipse_sq > 1.0 + 1e-12:
            return np.zeros(3)

        # Normalised poloidal distance for Lundquist field
        # Use the elliptical distance as the normalized radius
        rho_norm = np.sqrt(rho_ellipse_sq)

        # Physical distance from axis and radial unit vector
        rho_abs = np.linalg.norm(d_perp)
        if rho_abs > 1e-15:
            radial = d_perp / rho_abs
        else:
            radial = e_a  # Default to in-plane direction at axis

        # Azimuthal direction: perpendicular to both radial and tangent
        # This is the direction B_azimuthal points (wraps around the axis)
        azimuthal = np.cross(tangent, radial)
        az_norm = np.linalg.norm(azimuthal)
        if az_norm > 1e-15:
            azimuthal = azimuthal / az_norm
        else:
            azimuthal = e_b

        # Compute B0 from flux conservation at the apex
        # For elliptical cross-section, area = pi * a * b
        # But we reference to the apex where we define Phi
        R_eff_apex_m = Rp_val * u.solRad.to(u.m)  # metres (circular reference)
        B0 = B0_from_flux(self.Phi.value, R_eff_apex_m)

        # Scale B0 for flux conservation along the axis
        # Area ratio: (pi * semi_a * semi_b) / (pi * Rp^2 * aspect_ratio)
        # = (r_cs_base^2 * aspect_ratio) / (Rp^2 * aspect_ratio) = (r_cs_base/Rp)^2
        r_cs_apex = Rp_val
        if r_cs_base > 1e-10:
            B0_local = B0 * (r_cs_apex / r_cs_base) ** 2
        else:
            return np.zeros(3)

        # Lundquist field components
        B_axial, B_azimuthal = lundquist_field(rho_norm, B0=B0_local)

        # Apply toroidal stretching scaling to azimuthal component
        B_azimuthal *= az_scale

        # Apply polarity and chirality
        B_axial_signed = self.polarity * B_axial
        B_azimuthal_signed = self.chirality * B_azimuthal

        # Construct field vector in HEEQ
        B_vec = B_axial_signed * tangent + B_azimuthal_signed * azimuthal

        return B_vec

    def synthetic_insitu(self, spacecraft_pos_func, t_start, t_end, dt=60.0 * u.s,
                         evolving=True, magnetic_expansion=True):
        """
        Generate a synthetic in-situ magnetic field time series.

        Simulates what a spacecraft would measure as a CME passes over it (or
        as the spacecraft passes through a static CME).

        Args:
            spacecraft_pos_func: Callable that takes time (seconds, float) and
                returns (x, y, z) in HEEQ Cartesian coordinates (solar radii).
                Signature: spacecraft_pos_func(t) -> (x, y, z)
            t_start: Start time relative to epoch (astropy time unit).
            t_end: End time relative to epoch (astropy time unit).
            dt: Time step (astropy time unit, default 60 s).
            evolving: If True, Rt and Rp evolve with time via v_propagation
                and v_expansion (Eqs. 20-21). If False, use static Rt, Rp.
            magnetic_expansion: If True and evolving is True, apply toroidal
                stretching scaling to the poloidal (azimuthal) component only
                (~Rt_ref/Rt), keeping the geometry fixed unless Rp evolves.
                This reproduces the "magnetic erosion" asymmetry shown in
                Figure 6 of Isavnin (2016). Default True.

        Returns:
            times: (N,) array of times in seconds.
            B_heeq: (N, 3) array of (Bx, By, Bz) in HEEQ (Tesla).
            B_mag: (N,) array of total magnetic field magnitude (Tesla).
        """
        t_start_s = t_start.to(u.s).value
        t_end_s = t_end.to(u.s).value
        dt_s = dt.to(u.s).value

        times = np.arange(t_start_s, t_end_s + dt_s * 0.5, dt_s)
        n_t = len(times)
        B_heeq = np.zeros((n_t, 3))

        # Reference Rt for magnetic expansion scaling
        Rt_ref = self.Rt.value if (evolving and magnetic_expansion) else None

        for i, t in enumerate(times):
            # Get spacecraft position
            x, y, z = spacecraft_pos_func(t)

            # Evolve CME parameters if requested
            if evolving:
                Rt_t, Rp_t = self._evolve_params(t)
            else:
                Rt_t = self.Rt.value
                Rp_t = self.Rp.value

            B_heeq[i, :] = self.magnetic_field_at_point(x, y, z, Rt=Rt_t, Rp=Rp_t,
                                                         Rt_ref=Rt_ref)

        B_mag = np.sqrt(np.sum(B_heeq ** 2, axis=1))

        return times, B_heeq, B_mag

    def synthetic_insitu_1au(self, v_cme, impact_y=0.0 * u.solRad,
                             impact_z=0.0 * u.solRad, dt=60.0 * u.s,
                             evolving=True, t_padding=0.5 * u.day):
        """
        Convenience method: generate synthetic in-situ at 1 AU along the 
        Sun-Earth line for a CME propagating radially.

        The spacecraft is placed at (x=215 Rs, y=impact_y, z=impact_z) and the
        CME propagates towards it. The time window is estimated automatically.

        Args:
            v_cme: CME propagation speed at 1 AU (astropy velocity unit).
            impact_y: Impact parameter in the y-direction (astropy length unit).
            impact_z: Impact parameter in the z-direction (astropy length unit).
            dt: Time step (default 60 s).
            evolving: Whether to evolve Rt, Rp with time.
            t_padding: Extra time on each side of estimated crossing.

        Returns:
            times: (N,) array of times in seconds.
            B_heeq: (N, 3) array of (Bx, By, Bz) in HEEQ (Tesla).
            B_mag: (N,) total field magnitude (Tesla).
        """
        r_1au = 215.0  # solar radii
        solrad_km = u.solRad.to(u.km)

        # Estimate transit time to 1 AU
        v_km_s = v_cme.to(u.km / u.s).value
        transit_time_s = (r_1au * solrad_km) / v_km_s

        # CME crossing time at 1 AU: approximate as 2*Rp / v_cme
        rp_km = self.Rp.to(u.km).value
        crossing_time_s = 2.0 * rp_km / v_km_s

        # Set v_propagation so the CME reaches 1 AU
        original_v_prop = self.v_propagation
        self.v_propagation = v_cme.to(u.km / u.s)

        # Time window
        t_pad_s = t_padding.to(u.s).value
        t_start = transit_time_s - crossing_time_s - t_pad_s
        t_end = transit_time_s + crossing_time_s + t_pad_s

        y0 = impact_y.to(u.solRad).value
        z0 = impact_z.to(u.solRad).value

        def sc_pos(t):
            return (r_1au, y0, z0)

        times, B_heeq, B_mag = self.synthetic_insitu(
            sc_pos,
            t_start=t_start * u.s,
            t_end=t_end * u.s,
            dt=dt,
            evolving=evolving,
        )

        # Restore original v_propagation
        self.v_propagation = original_v_prop

        return times, B_heeq, B_mag


# =============================================================================
# Convenience functions
# =============================================================================

def make_lundquist_timeseries(B0, R, v_transit, impact_param=0.0, chirality=1,
                              polarity=1, tilt=0.0, dt=60.0):
    """
    Generate a simple Lundquist force-free cylinder magnetic field time series.

    This is a simplified 2D cross-section model (no 3D deformations), useful
    for quick flux-rope fitting or testing. The spacecraft passes through a
    static cylindrical flux rope at constant speed.

    Args:
        B0: Core magnetic field strength (nT).
        R: Flux-rope radius (AU).
        v_transit: Relative speed of spacecraft w.r.t. flux rope (km/s).
        impact_param: Closest approach distance to axis, normalised by R.
                      0 = centre, 1 = edge. Default 0.
        chirality: +1 (right-handed) or -1 (left-handed).
        polarity: +1 or -1 for axial field direction.
        tilt: Tilt of flux-rope axis from ecliptic (radians). 0 = in ecliptic.
        dt: Time step in seconds.

    Returns:
        t: Time array (seconds), centred on closest approach.
        Bx: Radial component (nT), in RTN-like coordinates.
        By: Tangential component (nT).
        Bz: Normal component (nT).
        Bt: Total field magnitude (nT).
    """
    R_km = R * 1.496e8  # AU to km
    # Crossing half-duration
    y0 = abs(impact_param)
    if y0 >= 1.0:
        raise ValueError("impact_param must be < 1 (inside the flux rope)")
    chord_half = np.sqrt(1.0 - y0 ** 2) * R_km
    t_half = chord_half / v_transit  # seconds

    t = np.arange(-t_half - 300, t_half + 300, dt)

    # Position along the chord
    x_chord = v_transit * t  # km from closest approach point
    x_norm = x_chord / R_km  # normalised by R

    # Distance from axis
    rho_norm = np.sqrt(x_norm ** 2 + y0 ** 2)

    # Field components
    B_ax_raw, B_az_raw = lundquist_field(rho_norm, B0=B0)
    # Zero outside the rope
    inside = rho_norm < 1.0
    B_ax_raw[~inside] = 0.0
    B_az_raw[~inside] = 0.0

    # Direction from axis to point in cross-section plane
    angle = np.arctan2(y0, x_norm)

    # Decompose azimuthal field into x, y components
    B_az_x = -chirality * B_az_raw * np.sin(angle)
    B_az_y = chirality * B_az_raw * np.cos(angle)

    # Apply tilt rotation
    ct, st = np.cos(tilt), np.sin(tilt)
    # Axial direction is along y (tangential) before tilt
    # After tilt, axial -> mixed y,z
    Bx = B_az_x  # radial component (not affected by tilt)
    By = polarity * B_ax_raw * ct + B_az_y * ct
    Bz = polarity * B_ax_raw * st + B_az_y * st

    Bt = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

    return t, Bx, By, Bz, Bt


def plot_fluxrope_insitu(times, B_heeq, B_mag=None, units='nT', figsize=(10, 8)):
    """
    Plot synthetic in-situ magnetic field time series.

    Args:
        times: Time array (seconds).
        B_heeq: (N, 3) array of (Bx, By, Bz).
        B_mag: (N,) total field magnitude. If None, computed from B_heeq.
        units: String label for field units (default 'nT').
        figsize: Figure size tuple.

    Returns:
        fig, axes: Matplotlib figure and axes handles.
    """
    import matplotlib.pyplot as plt

    if B_mag is None:
        B_mag = np.sqrt(np.sum(B_heeq ** 2, axis=1))

    t_hours = (times - times[0]) / 3600.0

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    axes[0].plot(t_hours, B_mag, 'k', linewidth=1.5)
    axes[0].set_ylabel(f'|B| [{units}]')
    axes[0].set_ylim(bottom=0)

    labels = [f'Bx [{units}]', f'By [{units}]', f'Bz [{units}]']
    colors = ['C0', 'C1', 'C2']
    for i in range(3):
        axes[i + 1].plot(t_hours, B_heeq[:, i], color=colors[i], linewidth=1.2)
        axes[i + 1].axhline(0, color='grey', linewidth=0.5, linestyle='--')
        axes[i + 1].set_ylabel(labels[i])

    axes[-1].set_xlabel('Time [hours]')
    fig.suptitle('FRiED Synthetic In-situ Magnetic Field', fontsize=14)
    fig.tight_layout()

    return fig, axes


def plot_fluxrope_3d(cme, n_field_lines=20, n_points_axis=200, figsize=(10, 10)):
    """
    Plot a 3D visualisation of the FRiED flux-rope CME.

    Shows the CME axis and a set of magnetic field lines colour-coded by 
    field strength.

    Args:
        cme: A FRiEDCME instance.
        n_field_lines: Number of field lines to plot.
        n_points_axis: Number of points for the axis.
        figsize: Figure size.

    Returns:
        fig, ax: Matplotlib figure and 3D axes.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xyz_axis, s_arc, phi_arr = cme.get_axis(n_points=n_points_axis)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot axis
    ax.plot(xyz_axis[:, 0], xyz_axis[:, 1], xyz_axis[:, 2], 'k-', linewidth=2,
            label='CME axis')

    # Plot Sun
    ax.scatter([0], [0], [0], color='yellow', s=200, edgecolors='orange',
               linewidths=2, zorder=5, label='Sun')

    # Plot field lines at various poloidal distances
    hw = cme.half_width.value
    rho_fracs = np.linspace(0.1, 0.9, min(n_field_lines, 10))

    for rho_frac in rho_fracs:
        # At each axis point, offset by rho_frac * local cross-section radius
        n_angle = 1  # just plot one azimuthal position for simplicity
        for angle_idx in range(n_angle):
            theta_fl = 2 * np.pi * angle_idx / max(n_angle, 1)
            fl_x = []
            fl_y = []
            fl_z = []
            fl_B = []

            for j in range(len(phi_arr)):
                r_cs = cross_section_radius(phi_arr[j], hw, cme.Rp.value, n=cme.n)
                if r_cs < 1e-10:
                    continue
                rho_abs = rho_frac * r_cs

                # Get B0 at this position for colour
                R_apex = cme.Rp.value * u.solRad.to(u.m)
                B0_apex = B0_from_flux(cme.Phi.value, R_apex)
                r_cs_apex = cme.Rp.value
                B0_local = B0_apex * (r_cs_apex / r_cs) ** 2 if r_cs > 1e-10 else 0
                B_ax, B_az = lundquist_field(rho_frac, B0=B0_local)
                B_total = np.sqrt(B_ax ** 2 + B_az ** 2)
                fl_B.append(B_total)

                # Tangent vector
                if j == 0:
                    tang = xyz_axis[1] - xyz_axis[0]
                elif j == len(phi_arr) - 1:
                    tang = xyz_axis[-1] - xyz_axis[-2]
                else:
                    tang = xyz_axis[j + 1] - xyz_axis[j - 1]
                tang = tang / (np.linalg.norm(tang) + 1e-30)

                # Perpendicular directions
                up = np.array([0.0, 0.0, 1.0])
                perp1 = np.cross(tang, up)
                pn = np.linalg.norm(perp1)
                if pn < 1e-10:
                    up = np.array([0.0, 1.0, 0.0])
                    perp1 = np.cross(tang, up)
                    pn = np.linalg.norm(perp1)
                perp1 = perp1 / pn
                perp2 = np.cross(tang, perp1)

                # Apply twist
                L_total = s_arc[-1]
                s_local = s_arc[j]
                twist_angle = 2.0 * np.pi * cme.twist * (s_local / L_total) if L_total > 0 else 0
                tw = theta_fl + twist_angle

                offset = rho_abs * (np.cos(tw) * perp1 + np.sin(tw) * perp2)

                fl_x.append(xyz_axis[j, 0] + offset[0])
                fl_y.append(xyz_axis[j, 1] + offset[1])
                fl_z.append(xyz_axis[j, 2] + offset[2])

            if len(fl_x) > 2:
                ax.plot(fl_x, fl_y, fl_z, linewidth=0.8, alpha=0.7)

    ax.set_xlabel('X (solar radii)')
    ax.set_ylabel('Y (solar radii)')
    ax.set_zlabel('Z (solar radii)')
    ax.set_title('FRiED CME 3D Structure')
    ax.legend()

    return fig, ax
