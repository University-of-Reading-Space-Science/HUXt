"""
Generalized solar wind simulation using pyro with custom time-dependent boundary conditions.

This module provides a function to run the pyro spherical solver with arbitrary
time-dependent inner boundary conditions, with custom particle tracking using
the midpoint method (RK2) for more accurate advection.
"""
import numpy as np
from pyro.pyro_sim import Pyro
import os
import sys
# Add current directory to path to import pyro_custom_bc_wrapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyro_custom_bc_wrapper import PyroCustomBCWrapper

# Physical constants (CGS)
AU = 1.496e13  # cm
PROTON_MASS = 1.67262192e-24  # grams
BOLTZMANN = 1.380649e-16  # erg/K
KM_TO_CM = 1e5  # cm/km


def compute_parker_nozzle_solution(r_grid, v0, n_rho0, T0, gamma=1.5):
    """
    Compute the analytical Parker nozzle solution for solar wind expansion.
    
    The Parker nozzle solution treats spherical expansion as flow through an expanding
    nozzle where A(r) = r². This gives an analytical solution for the supersonic solar wind
    assuming:
    - Fixed gamma (adiabatic index)
    - Steady-state isentropic flow
    - No heat addition or viscosity
    - Perfect gas equation of state
    
    Args:
        r_grid: Radial grid positions (cm)
        v0: Initial velocity (cm/s)
        n_rho0: Initial number density (cm^-3)
        T0: Initial temperature (K)
        gamma: Adiabatic index
    
    Returns:
        v, n_rho, T, rho: Velocity (cm/s), number density (cm^-3), temperature (K), 
                          mass density (g/cm^3) as functions of r
    """
    from scipy.optimize import bisect
    
    # Initial mass density
    rho0 = n_rho0 * PROTON_MASS
    
    # Gas constant per particle (erg/K)
    R_gas = BOLTZMANN
    
    # Compute Mach number at inner boundary
    c0 = np.sqrt(gamma * R_gas * T0 / PROTON_MASS)
    M0 = v0 / c0
    
    # Compute stagnation (total) conditions
    T_t = T0 * (1 + ((gamma - 1)/2)*(M0**2))
    p_t = (n_rho0 * BOLTZMANN * T0) * ((1 + ((gamma - 1)/2)*(M0**2)) ** (gamma/(gamma - 1)))
    rho_t = rho0 * ((1 + ((gamma - 1)/2)*(M0**2)) ** (1/(gamma - 1)))
    
    # Compute reference area at sonic point (A*)
    A0 = r_grid[0]**2  # Area at inner boundary
    
    def A_norm_calc(M, gamma):
        """Normalized area-Mach relation for quasi-1D nozzle flow"""
        a = 2 / (gamma + 1)
        b = (gamma - 1) / 2
        c = (gamma + 1) / (2*(gamma - 1))
        A_norm = (1/M) * (a*(1 + b*M*M))**c
        return A_norm
    
    A0_norm = A_norm_calc(M0, gamma)
    A_star = A0 / A0_norm  # Reference area at sonic point
    
    # For each radius, compute area ratio and solve for supersonic Mach number
    A = r_grid**2
    A_norm = A / A_star
    
    def invert_A_for_M(M, gamma, A_n):
        """Root finding function to invert A(M) relation"""
        return A_norm_calc(M, gamma) - A_n
    
    m_min = 1 + 1e-12  # Just above sonic
    m_max = 1e4  # High supersonic
    
    M = np.zeros(len(r_grid))
    for i, a_n in enumerate(A_norm):
        M[i] = bisect(invert_A_for_M, m_min, m_max, args=(gamma, a_n))
    
    # Compute static properties from isentropic relations
    T = T_t / (1 + ((gamma - 1)/2)*(M**2))
    p = p_t * (T/T_t) ** (gamma/(gamma - 1))
    rho = rho_t * (T/T_t) ** (1/(gamma - 1))
    
    # Number density
    n_rho = rho / PROTON_MASS
    
    # Compute velocity from Mach number
    c = np.sqrt(gamma * R_gas * T / PROTON_MASS)
    v = M * c
    
    return v, n_rho, T, rho


def run_solar_wind_pyro(
    r_grid,
    t_grid,
    v_bc_func,
    rho_bc_func,
    T_bc_func,
    gamma=5.0/3.0,
    ntheta=1,
    cfl=0.8,
    riemann_solver="CGF",
    verbose=False,
    initial_profile="steady_state",
    num_particles=0,
    particle_injection_rate=None
):
    """
    Run a solar wind simulation using pyro with time-dependent boundary conditions.
    
    Parameters
    ----------
    r_grid : array-like
        Radial positions for interior cell centers (cm). Should be uniformly spaced.
        These specify where you want the solution computed, NOT including ghost cells.
        The domain boundaries (xmin/xmax) will be set to r_grid[0] - dr/2 and 
        r_grid[-1] + dr/2, where dr is the grid spacing. This ensures pyro's
        interior cells are centered exactly at the specified r_grid positions.
        Ghost cells will extend beyond these boundaries as needed.
    t_grid : array-like
        Time grid for output snapshots (seconds). Solutions will be interpolated
        to these times.
    v_bc_func : callable
        Function v_bc_func(t) returning the radial velocity (cm/s) at the inner
        boundary as a function of time (s).
    rho_bc_func : callable
        Function rho_bc_func(t) returning the density (g/cm³) at the inner
        boundary as a function of time (s).
    T_bc_func : callable
        Function T_bc_func(t) returning the temperature (K) at the inner
        boundary as a function of time (s).
    gamma : float, optional
        Adiabatic index. Default is 5/3.
    ntheta : int, optional
        Number of angular grid points (currently should be 1 for 1D).
        Default is 1.
    cfl : float, optional
        CFL number for time-stepping. Default is 0.8.
    riemann_solver : str, optional
        Riemann solver to use. Default is "CGF" (Colella-Glaz-Ferguson).
    verbose : bool, optional
        If True, print detailed step-by-step diagnostics. Default is False.
    initial_profile : str or dict, optional
        How to initialize the solution. Options:
        - "steady_state": Simple empirical profile (rho ~ 1/r², T ~ r^-0.7, v=const)
        - "parker_nozzle": Analytical Parker nozzle solution (physically accurate 
          steady-state isentropic flow through expanding nozzle). Recommended.
        - dict with keys "rho", "v", "T": Custom initial profiles as functions of r
        Default is "steady_state".
    num_particles : int, optional
        Number of test particles to inject at the inner boundary.
        If 0 (default), no particles are tracked.
        Can also be a dict with keys as particle group names and values as 
        injection time arrays (see particle_injection_rate).
    particle_injection_rate : float, callable, array-like, or dict, optional
        Specifies when particles are injected:
        - None: all particles injected at t=0
        - float: constant injection rate (particles/second), uniformly distributed
        - callable: rate(t) function for time-varying injection
        - array-like: explicit injection times (seconds) for each particle
        - dict: keys are group names, values are arrays of injection times
          (num_particles must also be a dict with same keys)
        Examples:
          - CME boundaries: {'leading': [t1, t2, ...], 'trailing': [t1, t2, ...]}
          - HCS particles: {'hcs': [t1, t2, t3, ...]}
          - Streaklines: {'streak1': [t1, t1+dt, t1+2*dt, ...], 'streak2': [...]}
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - "r": radial grid positions (cm), shape (nr,)
        - "t": output times (s), shape (nt,)
        - "rho": density (g/cm³), shape (nt, nr)
        - "v": radial velocity (cm/s), shape (nt, nr)
        - "T": temperature (K), shape (nt, nr)
        - "P": pressure (dyne/cm²), shape (nt, nr)
        - "n": number density (particles/cm³), shape (nt, nr)
        - "step_count": total number of time steps taken
        - "solve_time": wall-clock time for the solve (s)
        - "particles": dict with particle trajectories (only if num_particles > 0)
            If num_particles is an int:
                - "r": radial positions, shape (num_particles, nt_particle)
                - "v": velocities, shape (num_particles, nt_particle)
                - "t": times, shape (num_particles, nt_particle)
                - "t_inject": injection times, shape (num_particles,)
                - "active": whether particle is still in domain, shape (num_particles,)
            If num_particles is a dict (grouped particles):
                - "groups": dict with group names as keys, each containing:
                    - "r", "v", "t", "t_inject", "active" as above
                - "all_r", "all_v", "all_t", "all_t_inject", "all_active": combined arrays
    
    Examples
    --------
    >>> # Define boundary conditions
    >>> def v_bc(t):
    ...     return 400e5  # 400 km/s in cm/s
    >>> def rho_bc(t):
    ...     return 200 * 1.67e-24  # 200 protons/cc
    >>> def T_bc(t):
    ...     return 1e6  # 1 MK
    >>> 
    >>> # Set up grids
    >>> r_grid = np.linspace(0.1 * AU, 1.0 * AU, 130)
    >>> t_grid = np.linspace(0, 10 * 86400, 21)  # 10 days, 21 snapshots
    >>> 
    >>> # Run simulation
    >>> results = run_solar_wind_pyro(r_grid, t_grid, v_bc, rho_bc, T_bc)
    >>> 
    >>> # Access results
    >>> print(results["v"][-1, :])  # Final velocity profile
    """
    import time as time_module
    
    # Validate inputs
    r_grid = np.asarray(r_grid)
    t_grid = np.asarray(t_grid)
    
    if len(r_grid) < 2:
        raise ValueError("r_grid must have at least 2 points")
    if len(t_grid) < 1:
        raise ValueError("t_grid must have at least 1 point")
    if not np.all(np.diff(t_grid) >= 0):
        raise ValueError("t_grid must be non-decreasing")
    
    # r_grid specifies the desired interior cell centers (NOT including ghost cells)
    # We need to compute xmin/xmax such that pyro's interior cells match r_grid
    nr = len(r_grid)
    
    # Compute cell spacing from user grid (assume uniform)
    dr = (r_grid[-1] - r_grid[0]) / (nr - 1)
    
    # Domain boundaries: half a cell beyond first/last interior cell centers
    # This ensures interior cells are centered at r_grid positions
    r_inner = r_grid[0] - 0.5 * dr  # xmin for pyro
    r_outer = r_grid[-1] + 0.5 * dr  # xmax for pyro
    t_max = t_grid[-1]
    
    # Get initial boundary conditions at the START of the pyro simulation
    # This is t_grid[0], not necessarily 0.0 (could be negative in model_time coordinates)
    rho_bc_init = rho_bc_func(t_grid[0])
    v_bc_init = v_bc_func(t_grid[0])
    T_bc_init = T_bc_func(t_grid[0])
    
    # Also check what the BC is at a few other times to ensure variation
    if verbose:
        v_bc_mid = v_bc_func(t_grid[len(t_grid)//2])
        v_bc_end = v_bc_func(t_grid[-1])
        if v_bc_init != v_bc_mid or v_bc_init != v_bc_end:
            print(f"Boundary condition check: v varies from {v_bc_init/KM_TO_CM:.2f} to {v_bc_mid/KM_TO_CM:.2f} to {v_bc_end/KM_TO_CM:.2f} km/s")
        else:
            print(f"Boundary condition is constant at {v_bc_init/KM_TO_CM:.2f} km/s")
    
    if verbose:
        print("=" * 70)
        print("Solar Wind Simulation with Time-Dependent Boundary Conditions")
        print("=" * 70)
        print(f"Interior cell centers: {r_grid[0]/AU:.6f} - {r_grid[-1]/AU:.6f} AU")
        print(f"Domain boundaries (xmin/xmax): {r_inner/AU:.6f} - {r_outer/AU:.6f} AU")
        print(f"Grid: {nr} interior cells x {ntheta} angular (+ {4} ghost cells per side)")
        print(f"Cell spacing: {dr/AU:.6f} AU")
        print(f"Time range: 0 - {t_max/86400.0:.2f} days")
        print(f"Output snapshots: {len(t_grid)}")
        print()
        print(f"Initial inner boundary conditions:")
        n_bc_init = rho_bc_init / PROTON_MASS
        print(f"  Velocity: {v_bc_init/KM_TO_CM:.1f} km/s")
        print(f"  Density: {n_bc_init:.1f} protons/cc")
        print(f"  Temperature: {T_bc_init:.2e} K")
        P_bc_init = rho_bc_init * BOLTZMANN * T_bc_init / PROTON_MASS
        print(f"  Pressure: {P_bc_init:.2e} dyne/cm²")
        print()
    
    # Create pyro simulation
    pyro = Pyro("compressible")
    
    pyro.initialize_problem(
        problem_name="sedov",
        inputs_dict={
            "mesh.nx": nr,
            "mesh.ny": ntheta,
            "mesh.xmin": r_inner,
            "mesh.xmax": r_outer,
            "mesh.ymin": 0.0,
            "mesh.ymax": np.pi,
            "driver.tmax": t_max,
            "driver.max_steps": 1000000,  # Large number
            "driver.cfl": cfl,
            "compressible.riemann": riemann_solver,
            "driver.verbose": 0,
            "eos.gamma": gamma,
            "mesh.grid_type": "SphericalPolar",
            "mesh.xlboundary": "outflow",
            "mesh.xrboundary": "outflow",
        }
    )
    
    # Initialize the solution
    myd = pyro.sim.cc_data
    myg = myd.grid
    
    dens = myd.get_var("density")
    xmom = myd.get_var("x-momentum")
    ymom = myd.get_var("y-momentum")
    ener = myd.get_var("energy")
    
    # Get interior cell coordinates only (2D grid)
    r_interior = myg.x2d[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1]
    
    if initial_profile == "steady_state":
        # Old steady-state wind profile: constant velocity, rho ~ 1/r^2, T ~ r^(-0.7)
        # This is a simple empirical profile, not physically accurate
        # Note: r_inner is now xmin (boundary), but we want the BC location (first interior cell)
        r_bc = r_grid[0]  # Use user-specified first interior cell as reference
        v_init = v_bc_init * np.ones_like(r_interior)
        rho_init = rho_bc_init * (r_bc / r_interior)**2
        T_init = T_bc_init * (r_bc / r_interior)**0.7
    elif initial_profile == "parker_nozzle":
        # Analytical Parker nozzle solution - physically accurate steady-state
        # Uses quasi-1D isentropic flow through expanding nozzle (A ~ r^2)
        if verbose:
            print("Computing Parker nozzle analytical solution for initial conditions...")
        
        # Get 1D radial profile from pyro's actual grid (INTERIOR CELLS ONLY)
        # IMPORTANT: pyro uses cell centers, not the edges we specify in r_grid input
        # Also need to use only interior cells since we'll set values with proper slicing
        r_1d = myg.x[myg.ilo:myg.ihi+1]
        
        # Initial number density
        n_bc_init = rho_bc_init / PROTON_MASS
        
        # Compute Parker solution on the ACTUAL grid points (cell centers, interior only)
        v_parker, n_parker, T_parker, rho_parker = compute_parker_nozzle_solution(
            r_1d, v_bc_init, n_bc_init, T_bc_init, gamma
        )
        
        # Broadcast to 2D grid (copy same profile to all angular positions)
        # Note: these arrays now have shape (n_interior_cells, n_angular_cells)
        v_init = np.tile(v_parker[:, np.newaxis], (1, myg.jhi - myg.jlo + 1))
        rho_init = np.tile(rho_parker[:, np.newaxis], (1, myg.jhi - myg.jlo + 1))
        T_init = np.tile(T_parker[:, np.newaxis], (1, myg.jhi - myg.jlo + 1))
        
        if verbose:
            print(f"  Parker solution computed on pyro grid:")
            print(f"    Grid: r_min={r_1d[0]/AU:.4f} AU, r_max={r_1d[-1]/AU:.4f} AU (cell centers)")
            print(f"    Inner cell: v={v_parker[0]/KM_TO_CM:.1f} km/s, n={n_parker[0]:.1f} cm^-3, T={T_parker[0]:.2e} K")
            print(f"    Outer cell: v={v_parker[-1]/KM_TO_CM:.1f} km/s, n={n_parker[-1]:.2e} cm^-3, T={T_parker[-1]:.2e} K")
            print(f"    Mach at inner: {v_parker[0]/np.sqrt(gamma*BOLTZMANN*T_parker[0]/PROTON_MASS):.2f}")
            print(f"    Mach at outer: {v_parker[-1]/np.sqrt(gamma*BOLTZMANN*T_parker[-1]/PROTON_MASS):.2f}")
    elif isinstance(initial_profile, dict):
        # Custom initial profile
        if not all(k in initial_profile for k in ["rho", "v", "T"]):
            raise ValueError("initial_profile dict must have keys 'rho', 'v', 'T'")
        rho_init = initial_profile["rho"](r_interior)
        v_init = initial_profile["v"](r_interior)
        T_init = initial_profile["T"](r_interior)
    else:
        raise ValueError(f"Unknown initial_profile: {initial_profile}")
    
    P_init = rho_init * BOLTZMANN * T_init / PROTON_MASS
    
    # Set initial conditions for INTERIOR CELLS ONLY (using proper slicing)
    dens[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] = rho_init
    xmom[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] = rho_init * v_init
    ymom[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] = 0.0
    
    e_int = P_init / (rho_init * (gamma - 1.0))
    e_kin = 0.5 * v_init**2
    ener[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] = rho_init * (e_int + e_kin)
    
    # Create custom BC wrapper
    wrapper = PyroCustomBCWrapper(pyro, gamma=gamma)
    
    # Define BC function using the provided functions
    def boundary_condition(r_array, t):
        """
        Apply time-dependent boundary conditions at the inner boundary.
        
        Parameters
        ----------
        r_array : array
            Radial positions at the boundary (should be r_inner or r_outer)
        t : float
            Current simulation time (s)
        
        Returns
        -------
        rho : array
            Density at the boundary (g/cm³)
        v : array
            Radial velocity at the boundary (cm/s)
        T : array
            Temperature at the boundary (K)
        """
        # Get time-dependent BC values
        rho_bc = rho_bc_func(t)
        v_bc = v_bc_func(t)
        T_bc = T_bc_func(t)
        
        # Apply to all boundary points (typically just one value)
        rho = rho_bc * np.ones_like(r_array)
        v = v_bc * np.ones_like(r_array)
        T = T_bc * np.ones_like(r_array)
        
        return rho, v, T
    
    wrapper.set_custom_bc(boundary_condition)
    
    # Apply custom boundary conditions immediately to initialize ghost cells properly
    # This ensures the BCs match the target values, not the Parker solution
    pyro.sim.cc_data.fill_BC_all()
    wrapper.apply_custom_bc()
    
    if verbose:
        print("Running simulation...")
        print(f"Target time: {t_max/86400.0:.2f} days ({t_max:.2e} s)")
        print()
    
    # Prepare to collect snapshots
    t_grid_idx = 0
    next_snapshot_time = t_grid[0]
    
    # Storage for results
    rho_snapshots = []
    v_snapshots = []
    T_snapshots = []
    P_snapshots = []
    t_snapshots = []
    
    # Initialize test particles if requested
    # Uses custom midpoint method (RK2) for advection, same as pyro's built-in particles
    # Handle both int and dict specifications
    particles_enabled = False
    particle_groups = {}
    total_particles = 0
    
    if isinstance(num_particles, dict):
        # Dictionary mode: multiple particle groups
        particles_enabled = True
        
        # particle_injection_rate must also be a dict with matching keys
        if not isinstance(particle_injection_rate, dict):
            raise ValueError("If num_particles is a dict, particle_injection_rate must also be a dict with matching keys")
        
        if set(num_particles.keys()) != set(particle_injection_rate.keys()):
            raise ValueError("Keys in num_particles and particle_injection_rate must match")
        
        for group_name in num_particles.keys():
            n_particles = num_particles[group_name]
            injection_spec = particle_injection_rate[group_name]
            
            # Convert injection spec to array of times
            if isinstance(injection_spec, (list, np.ndarray)):
                injection_times = np.asarray(injection_spec)
                if len(injection_times) != n_particles:
                    raise ValueError(f"Group '{group_name}': injection times array length ({len(injection_times)}) "
                                   f"must match num_particles ({n_particles})")
            else:
                raise ValueError(f"Group '{group_name}': particle_injection_rate must be an array of times")
            
            particle_groups[group_name] = {
                'n_particles': n_particles,
                'injection_times': injection_times,
                'particle_r': [],  # Store full trajectory for output
                'particle_v': [],
                'particle_t': [],
                'particle_t_inject': [],
                'particle_active': [],
                'particles_injected': 0,
            }
            total_particles += n_particles
        
        if verbose:
            print(f"Test particles enabled: {total_particles} particles in {len(particle_groups)} groups")
            for group_name, group in particle_groups.items():
                times = group['injection_times']
                print(f"  Group '{group_name}': {group['n_particles']} particles, "
                      f"t = [{times.min()/86400:.4f}, {times.max()/86400:.4f}] days")
            print()
            
    elif isinstance(num_particles, int) and num_particles > 0:
        # Single group mode (original behavior)
        particles_enabled = True
        group_name = 'default'
        
        # Determine injection schedule
        if particle_injection_rate is None:
            # Inject all particles at t=0
            injection_times = np.zeros(num_particles)
        elif isinstance(particle_injection_rate, (list, np.ndarray)):
            # Explicit injection times provided
            injection_times = np.asarray(particle_injection_rate)
            if len(injection_times) != num_particles:
                raise ValueError(f"Injection times array length ({len(injection_times)}) "
                               f"must match num_particles ({num_particles})")
        elif callable(particle_injection_rate):
            # Will inject dynamically based on rate function
            injection_times = None
        else:
            # Constant injection rate - distribute uniformly
            if num_particles > 1:
                injection_times = np.linspace(0, t_max, num_particles)
            else:
                injection_times = np.array([0.0])
        
        particle_groups[group_name] = {
            'n_particles': num_particles,
            'injection_times': injection_times,
            'particle_r': [],
            'particle_v': [],
            'particle_t': [],
            'particle_t_inject': [],
            'particle_active': [],
            'particles_injected': 0,
            'next_particle_time': 0.0,  # For dynamic injection
            'rate_func': particle_injection_rate if callable(particle_injection_rate) else None,
        }
        total_particles = num_particles
        
        if verbose:
            print(f"Test particles enabled: {num_particles} particles")
            if injection_times is not None:
                print(f"  Injection schedule: t = [{injection_times[0]/86400:.4f}, {injection_times[-1]/86400:.4f}] days")
            elif callable(particle_injection_rate):
                print(f"  Dynamic injection using rate function")
            print()
    
    step_count = 0
    solve_start_time = time_module.time()
    
    # Main time-stepping loop
    try:
        while not wrapper.finished() and t_grid_idx < len(t_grid):
            # Get current state
            state = wrapper.get_state()
            current_t = state['t']
            r_state = state['r']
            v_state = state['v']
            
            # Inject new particles if needed
            if particles_enabled:
                for group_name, group in particle_groups.items():
                    injection_times = group['injection_times']
                    
                    if injection_times is not None:
                        # Pre-scheduled injection from array
                        while (group['particles_injected'] < group['n_particles'] and 
                               injection_times[group['particles_injected']] <= current_t):
                            # Inject particle at inner boundary
                            particle_idx = group['particles_injected']
                            t_inj = injection_times[particle_idx]
                            
                            # Position at inner boundary
                            r_init = r_state[0]
                            v_init = v_state[0]
                            
                            # Store for trajectory
                            group['particle_r'].append([r_init])
                            group['particle_v'].append([v_init])
                            group['particle_t'].append([current_t])
                            group['particle_t_inject'].append(t_inj)
                            group['particle_active'].append(True)
                            group['particles_injected'] += 1
                            
                            if verbose and group['particles_injected'] % max(1, group['n_particles'] // 10) == 0:
                                print(f"  Group '{group_name}': injected particle {group['particles_injected']}/{group['n_particles']} "
                                      f"at t={current_t/86400:.4f} days")
                    else:
                        # Dynamic injection based on rate function (only for single group mode)
                        rate_func = group.get('rate_func')
                        if rate_func is not None:
                            rate = rate_func(current_t)
                            # Inject if enough time has passed
                            if (group['particles_injected'] < group['n_particles'] and 
                                current_t >= group['next_particle_time']):
                                
                                # Position at inner boundary
                                r_init = r_state[0]
                                v_init = v_state[0]
                                
                                group['particle_r'].append([r_init])
                                group['particle_v'].append([v_init])
                                group['particle_t'].append([current_t])
                                group['particle_t_inject'].append(current_t)
                                group['particle_active'].append(True)
                                group['particles_injected'] += 1
                                
                                # Schedule next particle
                                if rate > 0:
                                    group['next_particle_time'] = current_t + 1.0 / rate
                                else:
                                    group['next_particle_time'] = t_max + 1  # No more injections
                                
                                if verbose and group['particles_injected'] % max(1, group['n_particles'] // 10) == 0:
                                    print(f"  Injected particle {group['particles_injected']}/{group['n_particles']} "
                                          f"at t={current_t/86400:.4f} days")
            
            # Check if we need to save this snapshot
            if current_t >= next_snapshot_time:
                # Save snapshot
                t_snapshots.append(current_t)
                rho_snapshots.append(state['rho'].copy())
                v_snapshots.append(state['v'].copy())
                T_snapshots.append(state['T'].copy())
                P_snapshots.append(state['P'].copy())
                
                if verbose:
                    print(f"  Snapshot {t_grid_idx + 1}/{len(t_grid)} at t={current_t/86400.0:.4f} days (step {step_count})")
                
                # Move to next snapshot time
                t_grid_idx += 1
                if t_grid_idx < len(t_grid):
                    next_snapshot_time = t_grid[t_grid_idx]
            
            # Take one time step
            dt = wrapper.single_step_with_custom_bc()
            step_count += 1
            
            # Advect particles using midpoint method (RK2) - same method as pyro's particles
            if particles_enabled:
                for group_name, group in particle_groups.items():
                    particle_r = group['particle_r']
                    particle_v = group['particle_v']
                    particle_t = group['particle_t']
                    particle_active = group['particle_active']
                    
                    for i in range(len(particle_active)):
                        if particle_active[i]:
                            # Get current particle position
                            r_p = particle_r[i][-1]
                            
                            # Check if still in domain
                            if r_p < r_state[0] or r_p > r_state[-1]:
                                particle_active[i] = False
                            else:
                                # Midpoint method (RK2) for advection
                                # Step 1: Interpolate velocity at current position
                                v_p = np.interp(r_p, r_state, v_state)
                                
                                # Step 2: Predict position at midpoint
                                r_mid = r_p + 0.5 * v_p * dt
                                
                                # Step 3: Interpolate velocity at midpoint
                                if r_mid >= r_state[0] and r_mid <= r_state[-1]:
                                    v_mid = np.interp(r_mid, r_state, v_state)
                                else:
                                    # If midpoint is outside, use endpoint velocity
                                    v_mid = v_p
                                
                                # Step 4: Update position using midpoint velocity
                                r_p_new = r_p + v_mid * dt
                                
                                # Store new position
                                particle_r[i].append(r_p_new)
                                particle_v[i].append(v_mid)
                                particle_t[i].append(current_t + dt)
            
            # Periodic diagnostics
            if verbose and (step_count == 1 or step_count % 500 == 0):
                msg = f"  Step {step_count}, t={current_t/86400.0:.4f} days, dt={dt:.2e} s"
                if particles_enabled:
                    total_injected = sum(g['particles_injected'] for g in particle_groups.values())
                    total_active = sum(sum(g['particle_active'][:g['particles_injected']]) 
                                      for g in particle_groups.values())
                    msg += f", particles: {total_injected}/{total_particles} injected, {total_active} active"
                print(msg)
    
    except Exception as e:
        if verbose:
            print(f"\nSimulation stopped at t={current_t:.2e} s: {e}")
        # Continue to return whatever data we have
    
    solve_time = time_module.time() - solve_start_time
    
    # Get final state if we haven't reached all snapshots
    if t_grid_idx < len(t_grid):
        state = wrapper.get_state()
        if verbose:
            print(f"\nWarning: Simulation ended at t={state['t']/86400.0:.4f} days")
            print(f"         before reaching final requested time t={t_max/86400.0:.4f} days")
    
    if verbose:
        print(f"\n{'='*70}")
        print("SIMULATION COMPLETE")
        print(f"{'='*70}")
        print(f"Final time: {state['t']/86400.0:.4f} days ({state['t']:.2e} s)")
        print(f"Total steps: {step_count}")
        print(f"Solve time: {solve_time:.3f} s")
        print(f"Average time per step: {solve_time/step_count*1000:.2f} ms")
        print(f"Collected {len(t_snapshots)} snapshots")
        print()
    
    # Convert to arrays
    t_out = np.array(t_snapshots)
    rho_out = np.array(rho_snapshots)
    v_out = np.array(v_snapshots)
    T_out = np.array(T_snapshots)
    P_out = np.array(P_snapshots)
    
    # Get radial grid (use from last state)
    r_out = state['r']
    
    # Compute number density
    n_out = rho_out / PROTON_MASS
    
    # Package results
    results = {
        "r": r_out,
        "t": t_out,
        "rho": rho_out,
        "v": v_out,
        "T": T_out,
        "P": P_out,
        "n": n_out,
        "step_count": step_count,
        "solve_time": solve_time,
    }
    
    # Add particle trajectories if enabled
    if particles_enabled:
        if len(particle_groups) == 1 and 'default' in particle_groups:
            # Single group mode - return flat structure for backward compatibility
            group = particle_groups['default']
            particle_r = group['particle_r']
            particle_v = group['particle_v']
            particle_t = group['particle_t']
            particle_t_inject = group['particle_t_inject']
            particle_active = group['particle_active']
            n_particles = group['n_particles']
            
            # Find max length for ragged arrays
            max_len = max(len(particle_r[i]) for i in range(len(particle_r))) if len(particle_r) > 0 else 0
            
            # Create arrays filled with NaN for inactive/uninjected particles
            particle_r_array = np.full((n_particles, max_len), np.nan)
            particle_v_array = np.full((n_particles, max_len), np.nan)
            particle_t_array = np.full((n_particles, max_len), np.nan)
            
            for i in range(len(particle_r)):
                n_pts = len(particle_r[i])
                particle_r_array[i, :n_pts] = particle_r[i]
                particle_v_array[i, :n_pts] = particle_v[i]
                particle_t_array[i, :n_pts] = particle_t[i]
            
            # Pad arrays for uninjected particles
            particle_active_array = np.array(particle_active + [False] * (n_particles - len(particle_active)))
            particle_t_inject_array = np.array(particle_t_inject + [np.nan] * (n_particles - len(particle_t_inject)))
            
            results["particles"] = {
                "r": particle_r_array,
                "v": particle_v_array,
                "t": particle_t_array,
                "t_inject": particle_t_inject_array,
                "active": particle_active_array,
            }
            
            if verbose:
                print(f"Particle tracking summary:")
                print(f"  Total particles: {n_particles}")
                print(f"  Injected: {len(particle_r)}")
                print(f"  Still active: {sum(particle_active)}")
                print(f"  Exited domain: {sum(not a for a in particle_active)}")
        else:
            # Multi-group mode - return grouped structure
            particle_data = {"groups": {}}
            
            # Process each group
            all_r = []
            all_v = []
            all_t = []
            all_t_inject = []
            all_active = []
            
            for group_name, group in particle_groups.items():
                particle_r = group['particle_r']
                particle_v = group['particle_v']
                particle_t = group['particle_t']
                particle_t_inject = group['particle_t_inject']
                particle_active = group['particle_active']
                n_particles = group['n_particles']
                
                # Find max length for this group
                max_len = max(len(particle_r[i]) for i in range(len(particle_r))) if len(particle_r) > 0 else 0
                
                # Create arrays for this group
                particle_r_array = np.full((n_particles, max_len), np.nan)
                particle_v_array = np.full((n_particles, max_len), np.nan)
                particle_t_array = np.full((n_particles, max_len), np.nan)
                
                for i in range(len(particle_r)):
                    n_pts = len(particle_r[i])
                    particle_r_array[i, :n_pts] = particle_r[i]
                    particle_v_array[i, :n_pts] = particle_v[i]
                    particle_t_array[i, :n_pts] = particle_t[i]
                
                particle_active_array = np.array(particle_active + [False] * (n_particles - len(particle_active)))
                particle_t_inject_array = np.array(particle_t_inject + [np.nan] * (n_particles - len(particle_t_inject)))
                
                particle_data["groups"][group_name] = {
                    "r": particle_r_array,
                    "v": particle_v_array,
                    "t": particle_t_array,
                    "t_inject": particle_t_inject_array,
                    "active": particle_active_array,
                    "n_particles": n_particles,
                }
                
                # Accumulate for combined arrays
                all_r.append(particle_r_array)
                all_v.append(particle_v_array)
                all_t.append(particle_t_array)
                all_t_inject.append(particle_t_inject_array)
                all_active.append(particle_active_array)
            
            # Create combined arrays (different groups may have different trajectory lengths)
            max_len_all = max(arr.shape[1] for arr in all_r) if all_r else 0
            total_n = sum(g['n_particles'] for g in particle_groups.values())
            
            combined_r = np.full((total_n, max_len_all), np.nan)
            combined_v = np.full((total_n, max_len_all), np.nan)
            combined_t = np.full((total_n, max_len_all), np.nan)
            
            idx = 0
            for arr_r, arr_v, arr_t in zip(all_r, all_v, all_t):
                n = arr_r.shape[0]
                len_traj = arr_r.shape[1]
                combined_r[idx:idx+n, :len_traj] = arr_r
                combined_v[idx:idx+n, :len_traj] = arr_v
                combined_t[idx:idx+n, :len_traj] = arr_t
                idx += n
            
            combined_t_inject = np.concatenate(all_t_inject)
            combined_active = np.concatenate(all_active)
            
            particle_data["all_r"] = combined_r
            particle_data["all_v"] = combined_v
            particle_data["all_t"] = combined_t
            particle_data["all_t_inject"] = combined_t_inject
            particle_data["all_active"] = combined_active
            
            results["particles"] = particle_data
            
            if verbose:
                print(f"Particle tracking summary:")
                print(f"  Total particles: {total_n} in {len(particle_groups)} groups")
                for group_name, group_data in particle_data["groups"].items():
                    n_inj = np.sum(~np.isnan(group_data["t_inject"]))
                    n_act = np.sum(group_data["active"])
                    print(f"    Group '{group_name}': {n_inj} injected, {n_act} active")
    
    return results
