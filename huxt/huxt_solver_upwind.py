import numpy as np
from numba import jit
from huxt.huxt_solver_pluto import _pluto_step_euler, GAMMA_PLUTO
from huxt.cgf_solver import _cgf_step_
from huxt.cgf_solver import _cgf_step_

@jit(nopython=True, nogil=True)
def solve_radial_upwind(vinput, binput, iscmeinput, model_time, rrel, params,
                 n_cme, n_hcs_max, streak_times, rhoinput=None, tempinput=None, compressible=False, solver='upwind'):
    """
    Solve the radial profile as a function of time (including spinup), and
    return radial profile at specified output timesteps.
    Tracks CME frotns as test particles
    Args:
        vinput: Timeseries of inner boundary solar wind speeds.
        binput: Timeseries of inner boundary radial magnetic field.
        iscmeinput: Timeseries of in/out of a CME at the inner boundary.
        model_time: Array of model timesteps.
        rrel: Array of model radial coordinates relative to 30rS.
        params: Array of HUXt parameters.
        n_cme: Number of CMEs in the whole model run (not nec this longitude).
        n_hcs_max: Maximum number of HCS crossings at any longitude
        streak_times: time indices of streak foot points to track
        rhoinput: Timeseries of inner boundary density (optional, for compressible solver). Plain array without units.
        tempinput: Timeseries of inner boundary temperature (optional, for compressible solver). Plain array without units.
        compressible: Boolean flag indicating if compressible solver is being used
        solver: String specifying which numerical solver to use
    Returns:
        v_grid: Array of radial solar wind speed profile as function of time.
        cme_particles_r: Array of CME tracer particle positions as function of time.
        cme_particles_v: Array of CME tracer particle speeds as a function of time.
        rho_grid: Array of radial density profile as function of time (only if compressible=True, else None).
        temp_grid: Array of radial temperature profile as function of time (only if compressible=True, else None).
    """

    # unpack the HUXt params
    dtdr = params[0]
    alpha = params[1]
    r_accel = params[2]
    dt_scale = np.int32(params[3])
    nt_out = np.int32(params[4])
    nr = np.int32(params[5])
    r_boundary = params[7]
    accel_limit = bool(params[9])  # switch used to determine if speed limit is applied to acceleration.
    gamma = params[10]  # Adiabatic index for compressible solver
    
    # Compute the radial grid for the test particles
    rgrid = (rrel - rrel[0]) * 695700.0 + r_boundary  # Can't use astropy.units because numba
    dr = rgrid[1] - rgrid[0]
    dt = dtdr * dr

    # Preallocate space for solutions
    v_grid = np.zeros((nt_out, nr))
    cme_particles_r = np.zeros((n_cme, nt_out, 2)) * np.nan
    cme_particles_v = np.zeros((n_cme, nt_out, 2)) * np.nan
    
    # Preallocate space for density and temperature 
    # Use dummy arrays if not compressible to avoid None issues in Numba
    if compressible:
        rho_grid = np.zeros((nt_out, nr))
        temp_grid = np.zeros((nt_out, nr))
    else:
        rho_grid = np.zeros((1, 1))  # Dummy array
        temp_grid = np.zeros((1, 1))  # Dummy array

    # Check if CMEs need to be tracked.
    do_cme = 0
    if np.any(iscmeinput) > 0:
        do_cme = 1

    # Check if HCS needs to be tracked.
    hcs_particles = np.zeros((n_hcs_max, nt_out, 2)) * np.nan

    # see if there are any streaklines to be traced
    do_streak = False
    if not (np.isnan(streak_times)).all():
        do_streak = True
        n_streaks = len(streak_times[:, 0, 0])
        n_rots = len(streak_times[0, :, 0])

        streak_particles = np.ones((nt_out, n_streaks, n_rots)) * np.nan
        r_streakparticles = np.ones((n_streaks, n_rots)) * np.nan
    else:
        streak_particles = np.ones((1, 1, 1)) * np.nan

    iter_count = 0
    t_out = 0
    hcs_count = 0

    for t, time in enumerate(model_time):
        # Get the initial condition, which will update in the loop,
        # and snapshots saved to output at right steps.
        if t == 0:
            v = np.ones(nr) * 400
            r_cmeparticles = np.ones((n_cme, 2)) * np.nan
            v_cmeparticles = np.ones((n_cme, 2)) * np.nan
            r_hcsparticles = np.ones((n_hcs_max, 2)) * np.nan
            
            # Initialize density and temperature arrays for compressible solver
            if compressible:
                # Initialize with proper continuity relation: ρ·v·r² = const
                # This accounts for velocity evolution and gives better initial guess
                r_inner = rrel[0] * 695700.0 + r_boundary  # km
                rho_inner = rhoinput[0]
                temp_inner = tempinput[0]
                v_inner = vinput[0]
                
                rho = np.zeros(nr)
                temp = np.zeros(nr)
                for ir in range(nr):
                    r_this = rrel[ir] * 695700.0 + r_boundary  # km
                    r_ratio = r_inner / r_this
                    v_this = vinput[ir] if ir < len(vinput) else v_inner
                    # Continuity: ρ·v·r² = const → ρ(r) = ρ₀·(v₀/v)·(r₀/r)²
                    rho[ir] = rho_inner * (v_inner / v_this) * r_ratio**2
                    # Temperature initialization: use constant value from boundary
                    # Full temperature evolution handled by compressible solver
                    # which accounts for adiabatic expansion and velocity changes
                    temp[ir] = temp_inner
            else:
                rho = np.zeros(nr)  # Dummy array
                temp = np.zeros(nr)  # Dummy array

        # Update the inner boundary conditions
        v[0] = vinput[t]
        
        # Update density and temperature boundary conditions for compressible solver
        if compressible:
            rho[0] = rhoinput[t]
            temp[0] = tempinput[t]

        # see if there's an HCS crossing to be inserted at the boundary
        if t > 0:
            if binput[t] - binput[t - 1] > 0:
                r_hcsparticles[hcs_count, 0] = r_boundary
                r_hcsparticles[hcs_count, 1] = 1
                hcs_count = hcs_count + 1
            elif binput[t] - binput[t - 1] < 0:
                r_hcsparticles[hcs_count, 0] = r_boundary
                r_hcsparticles[hcs_count, 1] = -1
                hcs_count = hcs_count + 1

        # see if there's a new streakline to track and if so, insert at boundary
        if do_streak:
            for istreak in range(0, n_streaks):
                for irot in range(0, n_rots):
                    if t == streak_times[istreak, irot, 0]:
                        # add a particle in
                        r_streakparticles[istreak, irot] = r_boundary

        # Compute boundary speed of each CME at this time. 
        if time > 0:
            if do_cme == 1:
                for n in range(n_cme):
                    # Check if this point is within the cone CME
                    if iscmeinput[t] > 0:
                        thiscme = iscmeinput[t] - 1

                        # If the leading edge test particle doesn't exist, add it
                        if np.isnan(r_cmeparticles[thiscme, 0]):
                            r_cmeparticles[thiscme, 0] = r_boundary
                            v_cmeparticles[thiscme, 0] = v[0]

                        # Hold the CME trailing edge test particle at the inner boundary
                        # Until if condition breaks
                        r_cmeparticles[thiscme, 1] = r_boundary
                        v_cmeparticles[thiscme, 1] = v[0]

        # Update all fields for the given longitude
        u_up = v[1:].copy()
        u_dn = v[:-1].copy()

        # Do a single model time step
        # Solver dispatch: select numerical method based on solver parameter
        
        if solver == 'upwind':
            # First-order upwind scheme (Godunov-type)
            if compressible:
                # Use compressible upwind step that evolves velocity, density, and temperature together
                # NOTE: accel_limit is ignored for compressible solver (no residual acceleration)
                rho_up = rho[1:].copy()
                rho_dn = rho[:-1].copy()
                temp_up = temp[1:].copy()
                temp_dn = temp[:-1].copy()
                
                u_up_next, rho_up_next, temp_up_next = _upwind_step_compressible_(
                    u_up, u_dn, rho_up, rho_dn, temp_up, temp_dn, dtdr, alpha, r_accel, rrel, r_boundary, gamma)
                
                # Save the updated time steps (direct assignment, no copy needed)
                v[1:] = u_up_next
                rho[1:] = rho_up_next
                temp[1:] = temp_up_next
            else:
                # Use incompressible upwind step (velocity only)
                if accel_limit:
                    u_up_next = _upwind_step_accel_limit(u_up, u_dn, dtdr, alpha, r_accel, rrel)
                else:
                    u_up_next = _upwind_step_(u_up, u_dn, dtdr, alpha, r_accel, rrel)
                
                # Save the updated time step (direct assignment, no copy needed)
                v[1:] = u_up_next
        
        elif solver == 'cgf':
            # CGF Riemann solver (compressible)
            if compressible:
                rho_up = rho[1:].copy()
                rho_dn = rho[:-1].copy()
                temp_up = temp[1:].copy()
                temp_dn = temp[:-1].copy()
                
                u_up_next, rho_up_next, temp_up_next = _cgf_step_(
                    u_up, u_dn, rho_up, rho_dn, temp_up, temp_dn, dtdr, rgrid[1:], gamma)
                
                v[1:] = u_up_next
                rho[1:] = rho_up_next
                temp[1:] = temp_up_next
            else:
                raise ValueError("CGF solver requires compressible=True")
        
        else:
            # For other solvers (pluto), they should be handled by their own functions
            raise ValueError(f"Unknown solver: {solver}. Supported solvers in this function: 'upwind', 'cgf'")

        # Move the CME test particles forward
        if t > 0 and do_cme:
            for n in range(0, n_cme):  # loop over each CME
                for bound in range(0, 2):  # loop over front and rear boundaries
                    if not np.isnan(r_cmeparticles[n, bound]):
                        # Linearly interpolate the speed
                        v_test = np.interp(r_cmeparticles[n, bound] - dr / 2, rgrid, v)

                        # Advance the test particle
                        r_cmeparticles[n, bound] = (r_cmeparticles[n, bound] + v_test * dt)
                        v_cmeparticles[n, bound] = v_test

                if r_cmeparticles[n, 0] > rgrid[-1]:
                    # If the leading edge is past the outer boundary, put it at the outer boundary
                    r_cmeparticles[n, 0] = rgrid[-1]

                if r_cmeparticles[n, 1] > rgrid[-1]:
                    # If the trailing edge is past the outer boundary,delete
                    r_cmeparticles[n, :] = rgrid[-1]

        # Move the HCS test particles forward
        if t > 0:
            for n in range(0, n_hcs_max):  # loop over each HCS

                if not np.isnan(r_hcsparticles[n, 0]):
                    # Linearly interpolate the speed
                    v_test = np.interp(r_hcsparticles[n, 0] - dr / 2, rgrid, v)

                    # Advance the test particle
                    r_hcsparticles[n, 0] = (r_hcsparticles[n, 0] + v_test * dt)

                if r_hcsparticles[n, 0] > rgrid[-1]:
                    # If the leading edge is past the outer boundary, delete
                    r_hcsparticles[n, 0] = np.nan

        # move the streak line particles forward
        if t > 0 and do_streak:
            for istreak in range(0, n_streaks):
                for irot in range(0, n_rots):
                    if not np.isnan(r_streakparticles[istreak, irot]):
                        # Linearly interpolate the speed
                        v_test = np.interp(r_streakparticles[istreak, irot] - dr / 2, rgrid, v)

                        # Advance the test particle
                        r_streakparticles[istreak, irot] = (r_streakparticles[istreak, irot]
                                                            + v_test * dt)

                    if r_streakparticles[istreak, irot] > rgrid[-1]:
                        # If the leading edge is past the outer boundary, delete
                        r_streakparticles[istreak, irot] = np.nan

        # Save this frame to output if it is an output time step
        if time >= 0:
            iter_count = iter_count + 1
            if iter_count == dt_scale:
                if t_out <= nt_out - 1:
                    v_grid[t_out, :] = v.copy()
                    cme_particles_r[:, t_out, :] = r_cmeparticles.copy()
                    cme_particles_v[:, t_out, :] = v_cmeparticles.copy()
                    hcs_particles[:, t_out, :] = r_hcsparticles.copy()
                    if do_streak:
                        streak_particles[t_out, :, :] = r_streakparticles.copy()
                    
                    # Save density and temperature for compressible solver
                    if compressible:
                        rho_grid[t_out, :] = rho.copy()
                        temp_grid[t_out, :] = temp.copy()
                    
                    t_out = t_out + 1
                    iter_count = 0

    return v_grid, cme_particles_r, cme_particles_v, hcs_particles, streak_particles, rho_grid, temp_grid


@jit(nopython=True)
def _upwind_step_(v_up, v_dn, dtdr, alpha, r_accel, rrel):
    """
    Compute the next step in the upwind scheme of Burgers equation with added acceleration of the solar wind.
    Args:
        v_up: A numpy array of the upwind radial values. Units of km/s.
        v_dn: A numpy array of the downwind radial values. Units of km/s.
        dtdr: Ratio of HUXts time step and radial grid step. Units of s/km.
        alpha: Scale parameter for residual Solar wind acceleration.
        r_accel: Spatial scale parameter of residual solar wind acceleration. Units of km.
        rrel: The model radial grid relative to the radial inner boundary coordinate. Units of km.
    Returns:
         v_up_next: The upwind values at the next time step, numpy array with units of km/s.
    """

    # Arguments for computing the acceleration factor
    accel_arg = -rrel[:-1] / r_accel
    accel_arg_p = -rrel[1:] / r_accel

    # Get estimate of next time step
    v_up_next = v_up - dtdr * v_up * (v_up - v_dn)
    # Compute the probable speed at 30rS from the observed speed at r
    v_source = v_dn / (1.0 + alpha * (1.0 - np.exp(accel_arg)))
    # Then compute the speed gain between r and r+dr
    v_diff = alpha * v_source * (np.exp(accel_arg) - np.exp(accel_arg_p))
    # Add the residual acceleration over this grid cell
    v_up_next = v_up_next + (v_dn * dtdr * v_diff)

    return v_up_next


@jit(nopython=True)
def _upwind_step_accel_limit(v_up, v_dn, dtdr, alpha, r_accel, rrel):
    """
    Compute the next step in the upwind scheme of Burgers equation with added acceleration of the solar wind. Here, no
    acceleration is applied to speeds above 650km/s
    Args:
        v_up: A numpy array of the upwind radial values. Units of km/s.
        v_dn: A numpy array of the downwind radial values. Units of km/s.
        dtdr: Ratio of HUXts time step and radial grid step. Units of s/km.
        alpha: Scale parameter for residual Solar wind acceleration.
        r_accel: Spatial scale parameter of residual solar wind acceleration. Units of km.
        rrel: The model radial grid relative to the radial inner boundary coordinate. Units of km.
    Returns:
         v_up_next: The upwind values at the next time step, numpy array with units of km/s.
    """

    n = len(v_dn)
    v_up_next = np.empty(n, dtype=np.float64)

    for i in range(n):
        # compute indices for accel arguments safely
        if i >= len(rrel) - 1:
            continue  # skip last point to avoid out-of-bounds

        accel_arg = -rrel[i] / r_accel
        accel_arg_p = -rrel[i + 1] / r_accel

        # Upwind scheme
        v_up_next[i] = v_up[i] - dtdr * v_up[i] * (v_up[i] - v_dn[i])

        # Acceleration factor
        denom = 1.0 + alpha * (1.0 - np.exp(accel_arg))
        v_source = v_dn[i] / denom

        # Residual acceleration
        v_diff = 0.0
        if v_source < 650.0:
            v_diff = alpha * v_source * (np.exp(accel_arg) - np.exp(accel_arg_p))

        # Add residual acceleration to upwind step
        v_up_next[i] += v_dn[i] * dtdr * v_diff

    return v_up_next

@jit(nopython=True)
def _upwind_step_compressible_(v_up, v_dn, rho_up, rho_dn, temp_up, temp_dn, 
                                dtdr, alpha, r_accel, rrel, r_boundary, gamma):
    """
    Compute the next step in the upwind scheme for the compressible solver.
    This includes velocity, density, and temperature evolution with compression/heating physics.
    Residual acceleration is NOT included - pressure gradient drives the flow.
    
    Args:
        v_up: A numpy array of the upwind velocity values. Units of km/s.
        v_dn: A numpy array of the downwind velocity values. Units of km/s.
        rho_up: A numpy array of the upwind density values. Units of kg/m^3.
        rho_dn: A numpy array of the downwind density values. Units of kg/m^3.
        temp_up: A numpy array of the upwind temperature values. Units of K.
        temp_dn: A numpy array of the downwind temperature values. Units of K.
        dtdr: Ratio of HUXt time step and radial grid step. Units of s/km.
        alpha: Scale parameter for residual Solar wind acceleration (NOT USED in compressible solver).
        r_accel: Acceleration scale (NOT USED in compressible solver).
        rrel: The model radial grid relative to the radial inner boundary coordinate. Units of km.
        r_boundary: The inner boundary radius in km.
        gamma: Adiabatic index for compressible solver (typically 1.5 for solar wind).
        
    Returns:
        v_up_next: The upwind velocity values at the next time step. Units of km/s.
        rho_up_next: The upwind density values at the next time step. Units of kg/m^3.
        temp_up_next: The upwind temperature values at the next time step. Units of K.
    """

    # ====================================================================
    # Velocity evolution with pressure gradient force
    # Momentum equation: ∂v/∂t + v·∂v/∂r = -(1/ρ)·∂P/∂r
    # ====================================================================
    
    # Constants
    k_B = 1.38064852e-23  # J/K
    m_p = 1.67262192e-27  # kg
    
    # Radial coordinates
    r_up_km = rrel[:-1] * 695700.0 + r_boundary
    
    # Gradients (forward difference for upwind scheme)
    dv = v_dn - v_up
    drho = rho_dn - rho_up
    dtemp = temp_dn - temp_up
    
    # Pressure gradient term (1/rho * dP/dr)
    # P = rho * k_B * T / m_p
    # (1/rho) * dP = (k_B/m_p) * (dT + (T/rho) * drho)
    # Units: (J/K / kg) * (K + K) = J/kg = (m/s)^2
    term_P_SI = (k_B / m_p) * (dtemp + (temp_up / rho_up) * drho) # (m/s)^2
    
    # Convert to km/s^2 equivalent for update
    # 1 m^2/s^2 = 1e-6 km^2/s^2
    term_P_kms2 = 1e-6 * term_P_SI
    
    # Update velocity
    # v_new = v - dt * v * dv/dr - dt * 1/rho * dP/dr
    #       = v - v * (dt/dr) * dv - (dt/dr) * (1/rho * dP)
    v_up_next = v_up - v_up * dtdr * dv - dtdr * term_P_kms2
    
    # ====================================================================
    # Density evolution
    # Continuity equation: ∂ρ/∂t + v·∂ρ/∂r + ρ·∂v/∂r + 2ρv/r = 0
    # ====================================================================
    
    # Calculate dt from dtdr
    dr_km = (rrel[1] - rrel[0]) * 695700.0
    dt = dtdr * dr_km
    
    spherical_term = 2.0 * rho_up * v_up / r_up_km
    rho_up_next = rho_up - dtdr * (v_up * drho + rho_up * dv) - dt * spherical_term
    
    # ====================================================================
    # Temperature evolution
    # Energy equation: ∂T/∂t + v·∂T/∂r + (γ-1)T(∂v/∂r + 2v/r) = 0
    # ====================================================================
    
    div_v_dt = dtdr * dv + dt * 2.0 * v_up / r_up_km
    temp_up_next = temp_up - dtdr * v_up * dtemp - (gamma - 1.0) * temp_up * div_v_dt
    
    # Apply physical bounds to prevent instability
    v_up_next = np.maximum(100.0, np.minimum(v_up_next, 3000.0))
    temp_up_next = np.maximum(1e3, np.minimum(temp_up_next, 1e8))
    rho_up_next = np.maximum(1e-30, rho_up_next)
    
    return v_up_next, rho_up_next, temp_up_next
