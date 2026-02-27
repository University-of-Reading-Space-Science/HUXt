"""
Compressible hydrodynamics solver for HUXt.

This module provides numerical methods for solving the 1D spherical 
compressible Euler equations using a finite-volume formulation with
area-weighted fluxes and geometric source terms.

Spherical geometry is handled through:
1. Interface areas: A = 4πr² 
2. Cell volumes: V = 4/3 π(r³_out - r³_in)
3. Geometric source term: S = [0, 2p/r, 0] for momentum equation

The combination of area-weighted fluxes + the 2p/r source term ensures:
- Mass, momentum, and energy are conserved correctly
- Parker nozzle acceleration emerges naturally
- Shocks and stream interactions are handled correctly by the Riemann solver

Riemann Solver:
- HLLC (Harten-Lax-van Leer-Contact) - fast, robust, captures contact waves

Reconstruction Methods:
- 'plm': Piecewise Linear Method (2nd order) with MC limiter [default]
- 'pcm': Piecewise Constant Method (1st order, Godunov)

Time Integration:
- 'euler': Forward Euler (1st order) [default]
- 'rk2': Runge-Kutta 2nd order (Heun's method)

Example Usage:
    from huxt.huxt_solvers import create_solver
    
    # Create solver with PLM reconstruction (default)
    solver = create_solver(r_grid, gamma=1.5, method='hllc-plm')
    
    # Or specify all components
    solver = create_solver(r_grid, gamma=1.5, method='hllc-plm-rk2')
    
    # Run simulation
    result = solver.solve(t_grid, v_bc_func, rho_bc_func, T_bc_func)
"""

import numpy as np
from numba import njit

__all__ = [
    'CompressibleSolver',
    'create_solver',
    'benchmark_solvers',
    'list_available_methods',
    'K_B_SI', 'M_P_SI',
]

# Physical constants (SI units)
K_B_SI = 1.380649e-23       # J/K (Boltzmann constant)
M_P_SI = 1.67262192e-27     # kg (proton mass)

# Numerical floor values
SMALL_RHO = 1e-30
SMALL_P = 1e-30


# =============================================================================
# Particle Advection
# =============================================================================

@njit
def _advect_particle_rk2(r_p, v_grid, r_grid, dt, behavior):
    """
    Advect a single particle using RK2 integration.
    
    Parameters
    ----------
    r_p : float
        Current particle radius (m)
    v_grid : ndarray
        Velocity on radial grid (m/s)
    r_grid : ndarray
        Radial grid points (m)
    dt : float
        Time step (s)
    behavior : int
        0: delete particle if out of bounds
        1: clamp to outer boundary
        
    Returns
    -------
    r_new : float
        New particle radius (m)
    v_new : float
        Velocity at new position (m/s)
    is_active : bool
        Whether particle is still active
    """
    # Check current bounds
    if r_p < r_grid[0]: 
        r_new = r_grid[0]
    elif r_p > r_grid[-1]:
        if behavior == 0:
            return r_p, 0.0, False
        else:
            r_new = r_grid[-1]
    else:
        r_new = r_p
        
    r_p = r_new

    # RK2 integration
    v_p = np.interp(r_p, r_grid, v_grid)
    r_mid = r_p + 0.5 * v_p * dt
    
    # Clamp intermediate position
    if r_mid < r_grid[0]: r_mid = r_grid[0]
    if r_mid > r_grid[-1]: r_mid = r_grid[-1]
    
    v_mid = np.interp(r_mid, r_grid, v_grid)
    r_final = r_p + v_mid * dt
    
    # Boundary handling
    is_active = True
    if r_final < r_grid[0]:
        r_final = r_grid[0]
    elif r_final > r_grid[-1]:
        if behavior == 0:
            is_active = False
        else:
            r_final = r_grid[-1]
            
    return r_final, v_mid, is_active


# =============================================================================
# Equation of State
# =============================================================================

@njit(cache=True)
def _get_rhoe(p, gamma):
    """Internal energy density from pressure: rho*e = p / (gamma - 1)"""
    return p / (gamma - 1.0)


@njit(cache=True)
def _prim_to_cons(q, gamma):
    """
    Convert primitive [rho, v, p] to conserved [rho, rho*v, E] variables.
    """
    rho, v, p = q[0], q[1], q[2]
    mom = rho * v
    rhoe = _get_rhoe(p, gamma)
    ener = rhoe + 0.5 * rho * v**2
    return np.array([rho, mom, ener])


# =============================================================================
# HLLC Riemann Solver
# =============================================================================

@njit(cache=True)
def _riemann_hllc(U_l, U_r, gamma):
    """
    HLLC Riemann solver for the Euler equations.
    
    Computes the interface flux given left and right conserved states.
    Resolves contact discontinuities accurately.
    
    Parameters
    ----------
    U_l, U_r : ndarray
        Left and right conserved states [rho, rho*v, E]
    gamma : float
        Adiabatic index
        
    Returns
    -------
    F : ndarray
        Interface flux [mass flux, momentum flux, energy flux]
    """
    # Left state primitives
    rho_l = max(U_l[0], SMALL_RHO)
    v_l = U_l[1] / rho_l
    E_l = U_l[2]
    p_l = max((E_l - 0.5*rho_l*v_l**2) * (gamma - 1.0), SMALL_P)
    c_l = np.sqrt(gamma * p_l / rho_l)
    
    # Right state primitives
    rho_r = max(U_r[0], SMALL_RHO)
    v_r = U_r[1] / rho_r
    E_r = U_r[2]
    p_r = max((E_r - 0.5*rho_r*v_r**2) * (gamma - 1.0), SMALL_P)
    c_r = np.sqrt(gamma * p_r / rho_r)
    
    # Wave speed estimates
    S_l = min(v_l - c_l, v_r - c_r)
    S_r = max(v_l + c_l, v_r + c_r)
    
    F = np.zeros(3)
    
    if S_l >= 0:
        # Supersonic flow to the right
        F[0] = rho_l * v_l
        F[1] = rho_l * v_l * v_l + p_l
        F[2] = (E_l + p_l) * v_l
        return F
    elif S_r <= 0:
        # Supersonic flow to the left
        F[0] = rho_r * v_r
        F[1] = rho_r * v_r * v_r + p_r
        F[2] = (E_r + p_r) * v_r
        return F
    
    # Contact wave speed
    denom = rho_l * (S_l - v_l) - rho_r * (S_r - v_r)
    if abs(denom) < 1e-14:
        S_c = 0.5 * (v_l + v_r)
    else:
        S_c = (p_r - p_l + rho_l * v_l * (S_l - v_l) - rho_r * v_r * (S_r - v_r)) / denom
    
    if S_c >= 0:
        # Left star state
        factor = rho_l * (S_l - v_l) / (S_l - S_c)
        U_star_0 = factor
        U_star_1 = factor * S_c
        U_star_2 = factor * (E_l/rho_l + (S_c - v_l) * (S_c + p_l/(rho_l*(S_l - v_l))))
        
        F_l_0 = rho_l * v_l
        F_l_1 = rho_l * v_l * v_l + p_l
        F_l_2 = (E_l + p_l) * v_l
        
        F[0] = F_l_0 + S_l * (U_star_0 - U_l[0])
        F[1] = F_l_1 + S_l * (U_star_1 - U_l[1])
        F[2] = F_l_2 + S_l * (U_star_2 - U_l[2])
    else:
        # Right star state
        factor = rho_r * (S_r - v_r) / (S_r - S_c)
        U_star_0 = factor
        U_star_1 = factor * S_c
        U_star_2 = factor * (E_r/rho_r + (S_c - v_r) * (S_c + p_r/(rho_r*(S_r - v_r))))
        
        F_r_0 = rho_r * v_r
        F_r_1 = rho_r * v_r * v_r + p_r
        F_r_2 = (E_r + p_r) * v_r
        
        F[0] = F_r_0 + S_r * (U_star_0 - U_r[0])
        F[1] = F_r_1 + S_r * (U_star_1 - U_r[1])
        F[2] = F_r_2 + S_r * (U_star_2 - U_r[2])
    
    return F


# =============================================================================
# Slope Limiter
# =============================================================================

@njit(cache=True)
def _mc_limiter(a, b):
    """MC (Monotonized Central) slope limiter."""
    c = 0.5 * (a + b)
    if a * b <= 0:
        return 0.0
    else:
        return np.sign(c) * min(abs(c), 2*abs(a), 2*abs(b))


# =============================================================================
# Flux Computation (JIT-compiled)
# =============================================================================

@njit(cache=True)
def _compute_fluxes_pcm(U, nr, gamma):
    """
    Compute interface fluxes using PCM (1st order, Godunov).
    """
    fluxes = np.zeros((nr + 1, 3))
    
    for i in range(nr + 1):
        if i == 0:
            U_l = U[0]
            U_r = U[0]
        elif i == nr:
            U_l = U[nr-1]
            U_r = U[nr-1]
        else:
            U_l = U[i-1]
            U_r = U[i]
        
        fluxes[i] = _riemann_hllc(U_l, U_r, gamma)
    
    return fluxes


@njit(cache=True)
def _compute_fluxes_plm(U, nr, gamma):
    """
    Compute interface fluxes using PLM (2nd order) reconstruction.
    
    Batch-converts all cells to primitives once, then applies MC limiter
    for slope reconstruction at each interface.
    """
    fluxes = np.zeros((nr + 1, 3))
    
    # Convert all cells to primitives [rho, v, p]
    q_all = np.zeros((nr, 3))
    q_valid = np.ones(nr, dtype=np.int32)
    for j in range(nr):
        rho = max(U[j, 0], SMALL_RHO)
        v = U[j, 1] / rho
        eint = (U[j, 2] - 0.5 * rho * v * v) / rho
        p = max(rho * eint * (gamma - 1.0), SMALL_P)
        q_all[j, 0] = rho
        q_all[j, 1] = v
        q_all[j, 2] = p
        if rho <= 0 or p <= 0:
            q_valid[j] = 0
    
    for i in range(nr + 1):
        if i == 0:
            U_l = U[0]
            U_r = U[0]
        elif i == nr:
            # Outer boundary - 2nd order extrapolation
            if nr >= 2:
                q_last = q_all[nr-1]
                q_prev = q_all[nr-2]
                q_face = np.zeros(3)
                q_face[0] = max(q_last[0] + 0.5 * (q_last[0] - q_prev[0]), SMALL_RHO)
                q_face[1] = q_last[1] + 0.5 * (q_last[1] - q_prev[1])
                q_face[2] = max(q_last[2] + 0.5 * (q_last[2] - q_prev[2]), SMALL_P)
                U_face = _prim_to_cons(q_face, gamma)
                U_l = U_face
                U_r = U_face
            else:
                U_l = U[nr-1]
                U_r = U[nr-1]
        else:
            idx = i - 1  # left cell of interface
            
            if idx >= nr - 1:
                U_l = U[idx].copy()
                U_r = U[idx].copy()
            elif idx == 0:
                # Inner boundary special case
                if nr < 3 or q_valid[0] == 0 or q_valid[1] == 0 or q_valid[2] == 0:
                    U_l = U[0].copy()
                    U_r = U[1].copy()
                else:
                    q_L = np.zeros(3)
                    q_R = np.zeros(3)
                    for n in range(3):
                        slope_0 = q_all[1, n] - q_all[0, n]
                        slope_L = q_all[1, n] - q_all[0, n]
                        slope_R = q_all[2, n] - q_all[1, n]
                        dq_1 = _mc_limiter(slope_L, slope_R)
                        q_L[n] = q_all[0, n] + 0.5 * slope_0
                        q_R[n] = q_all[1, n] - 0.5 * dq_1
                    q_L[0] = max(q_L[0], SMALL_RHO); q_L[2] = max(q_L[2], SMALL_P)
                    q_R[0] = max(q_R[0], SMALL_RHO); q_R[2] = max(q_R[2], SMALL_P)
                    U_l = _prim_to_cons(q_L, gamma)
                    U_r = _prim_to_cons(q_R, gamma)
            else:
                # Standard internal interface
                if q_valid[idx-1] == 0 or q_valid[idx] == 0 or q_valid[idx+1] == 0:
                    U_l = U[idx].copy()
                    U_r = U[idx+1].copy()
                else:
                    dq_l = np.zeros(3)
                    dq_r = np.zeros(3)
                    for n in range(3):
                        slope_l = q_all[idx, n] - q_all[idx-1, n]
                        slope_r = q_all[idx+1, n] - q_all[idx, n]
                        dq_l[n] = _mc_limiter(slope_l, slope_r)
                        
                        if idx < nr - 2:
                            slope_l2 = q_all[idx+1, n] - q_all[idx, n]
                            slope_r2 = q_all[idx+2, n] - q_all[idx+1, n]
                            dq_r[n] = _mc_limiter(slope_l2, slope_r2)
                    
                    q_L = q_all[idx] + 0.5 * dq_l
                    q_R = q_all[idx+1] - 0.5 * dq_r
                    q_L[0] = max(q_L[0], SMALL_RHO); q_L[2] = max(q_L[2], SMALL_P)
                    q_R[0] = max(q_R[0], SMALL_RHO); q_R[2] = max(q_R[2], SMALL_P)
                    U_l = _prim_to_cons(q_L, gamma)
                    U_r = _prim_to_cons(q_R, gamma)
        
        fluxes[i] = _riemann_hllc(U_l, U_r, gamma)
    
    return fluxes


# =============================================================================
# Time Stepping (JIT-compiled)
# =============================================================================

@njit(cache=True)
def _get_dt(U, nr, min_dx, gamma, cfl):
    """Compute CFL-limited timestep."""
    max_speed = 0.0
    for i in range(nr):
        rho = max(U[i, 0], SMALL_RHO)
        v = U[i, 1] / rho
        eint = (U[i, 2] - 0.5 * rho * v * v) / rho
        p = max(rho * eint * (gamma - 1.0), SMALL_P)
        c = np.sqrt(gamma * p / rho)
        speed = abs(v) + c
        if speed > max_speed:
            max_speed = speed
    
    if max_speed == 0:
        return 1.0
    
    return cfl * min_dx / max_speed


@njit(cache=True)
def _extract_snapshot(U, nr, gamma, M_P_val, K_B_val):
    """
    Extract primitive variables (v, rho, T) from conserved state.
    """
    v_out = np.zeros(nr)
    rho_out = np.zeros(nr)
    T_out = np.zeros(nr)
    for i in range(nr):
        rho = max(U[i, 0], SMALL_RHO)
        v = U[i, 1] / rho
        eint = (U[i, 2] - 0.5 * rho * v * v) / rho
        p = max(rho * eint * (gamma - 1.0), SMALL_P)
        v_out[i] = v
        rho_out[i] = rho
        T_out[i] = p * M_P_val / (rho * K_B_val)
    return v_out, rho_out, T_out


@njit(cache=True)
def _step_euler(U, U_bc, nr, A, V, dt, gamma, use_plm):
    """
    Forward Euler time step with area-weighted fluxes and geometric source.
    """
    if use_plm:
        fluxes = _compute_fluxes_plm(U, nr, gamma)
    else:
        fluxes = _compute_fluxes_pcm(U, nr, gamma)
    
    U_new = np.zeros_like(U)
    
    for i in range(nr):
        if V[i] <= 0:
            U_new[i] = U[i]
            continue
        
        # Geometric source term (momentum equation only)
        rho = max(U[i, 0], SMALL_RHO)
        v = U[i, 1] / rho
        eint = (U[i, 2] - 0.5 * rho * v * v) / rho
        p = max(rho * eint * (gamma - 1.0), SMALL_P)
        S_mom = p * (A[i+1] - A[i]) / V[i]
        
        inv_V = 1.0 / V[i]
        for j in range(3):
            dUdt = -(A[i+1] * fluxes[i+1, j] - A[i] * fluxes[i, j]) * inv_V
            if j == 1:
                dUdt += S_mom
            U_new[i, j] = U[i, j] + dt * dUdt
        
        # Reset if unphysical
        if U_new[i, 0] < SMALL_RHO or U_new[i, 2] < SMALL_P:
            U_new[i] = U[i]
    
    U_new[0] = U_bc
    return U_new


@njit(cache=True)
def _step_rk2(U, U_bc, nr, A, V, dt, gamma, use_plm):
    """
    RK2 (Heun's method) time step.
    """
    U1 = _step_euler(U, U_bc, nr, A, V, dt, gamma, use_plm)
    U2 = _step_euler(U1, U_bc, nr, A, V, dt, gamma, use_plm)
    
    U_new = np.zeros_like(U)
    for i in range(nr):
        for j in range(3):
            U_new[i, j] = 0.5 * (U[i, j] + U2[i, j])
    
    U_new[0] = U_bc
    return U_new


# =============================================================================
# Main Solver Class
# =============================================================================

class CompressibleSolver:
    """
    1D spherical compressible Euler solver using HLLC Riemann solver.
    
    Solves the compressible Euler equations in spherical geometry with
    area-weighted fluxes and geometric source terms.
    
    Parameters
    ----------
    r_grid : ndarray
        Radial grid cell centers in m (SI units)
    gamma : float, optional
        Adiabatic index (default 5/3)
    cfl : float, optional
        CFL number (default 0.4 for PLM, 0.8 for PCM)
    reconstruction : str, optional
        'pcm' (1st order) or 'plm' (2nd order, default)
    time_integration : str, optional
        'euler' (1st order, default) or 'rk2' (2nd order)
    verbose : bool, optional
        Print progress information
        
    Example
    -------
    >>> solver = CompressibleSolver(r_grid, gamma=1.5, reconstruction='plm')
    >>> results = solver.solve(t_grid, v_bc_func, rho_bc_func, T_bc_func)
    """
    
    def __init__(self, r_grid, gamma=5.0/3.0, cfl=None, 
                 riemann='hllc', reconstruction='plm', time_integration='euler',
                 verbose=False):
        self.r = r_grid.copy()
        self.nr = len(r_grid)
        self.gamma = gamma
        self.verbose = verbose
        
        self.reconstruction = reconstruction.lower()
        self.time_integration = time_integration.lower()
        
        # Set CFL based on reconstruction order
        if cfl is None:
            self.cfl = 0.4 if self.reconstruction == 'plm' else 0.8
        else:
            self.cfl = cfl
        
        # Grid geometry
        self.dr = np.diff(r_grid)
        self.r_int = np.zeros(self.nr + 1)
        self.r_int[1:-1] = 0.5 * (self.r[1:] + self.r[:-1])
        self.r_int[0] = self.r[0] - 0.5 * self.dr[0]
        self.r_int[-1] = self.r[-1] + 0.5 * self.dr[-1]
        self.dx = np.diff(self.r_int)
        
        # Interface areas and cell volumes (spherical geometry)
        self.A = 4.0 * np.pi * self.r_int**2
        self.V = 4.0/3.0 * np.pi * (self.r_int[1:]**3 - self.r_int[:-1]**3)
        
        # State array [rho, rho*v, E]
        self.U = np.zeros((self.nr, 3))
        self.time = 0.0
        
        # Precompute min_dx for CFL
        self.min_dx = np.min(self.dx)
        
    def _set_initial_conditions(self, rho, v, T):
        """Set initial conditions from primitive variables (SI units)."""
        for i in range(self.nr):
            p = rho[i] * K_B_SI * T[i] / M_P_SI
            p = max(p, SMALL_P)
            self.U[i] = _prim_to_cons(np.array([rho[i], v[i], p]), self.gamma)
    
    def _get_dt(self):
        """Compute CFL-limited timestep."""
        return _get_dt(self.U, self.nr, self.min_dx, self.gamma, self.cfl)
    
    def _step(self, dt, bc_func):
        """Advance solution by dt."""
        rho_bc, v_bc, T_bc = bc_func(self.time + dt)
        p_bc = rho_bc * K_B_SI * T_bc / M_P_SI
        U_bc = _prim_to_cons(np.array([rho_bc, v_bc, p_bc]), self.gamma)
        
        use_plm = (self.reconstruction == 'plm')
        
        if self.time_integration == 'rk2':
            self.U = _step_rk2(self.U, U_bc, self.nr, self.A, self.V,
                               dt, self.gamma, use_plm)
        else:
            self.U = _step_euler(self.U, U_bc, self.nr, self.A, self.V,
                                 dt, self.gamma, use_plm)
        
        self.time += dt
        return self.U
    
    def solve(self, t_grid, v_bc_func, rho_bc_func, T_bc_func, 
              num_particles=0, particle_injection_rate=None, particle_release_rate=None):
        """
        Run simulation over time grid.
        
        Parameters
        ----------
        t_grid : ndarray
            Output times (seconds)
        v_bc_func : callable
            Velocity boundary condition v(t) in m/s
        rho_bc_func : callable
            Density boundary condition rho(t) in kg/m³
        T_bc_func : callable
            Temperature boundary condition T(t) in K
        num_particles : int or dict, optional
            Number of tracer particles to track
        particle_injection_rate : array or dict, optional
            Particle injection times (seconds)
        particle_release_rate : array or dict, optional
            Particle release times (seconds)
            
        Returns
        -------
        dict
            Results with keys 't', 'r', 'v', 'rho', 'T', 'solver_info',
            and optionally 'particles'
        """
        import time as time_module
        
        if self.verbose:
            print(f"CompressibleSolver starting:")
            print(f"  Reconstruction: {self.reconstruction}")
            print(f"  Time integration: {self.time_integration}")
        
        # Initialize with power-law scaling
        v0 = v_bc_func(t_grid[0])
        rho0 = rho_bc_func(t_grid[0])
        T0 = T_bc_func(t_grid[0])
        
        r0 = self.r[0]
        v_init = np.ones_like(self.r) * v0
        rho_init = np.zeros_like(self.r)
        T_init = np.zeros_like(self.r)
        alpha = 2 * (self.gamma - 1)
        
        for i, r in enumerate(self.r):
            r_ratio = r0 / r
            rho_init[i] = rho0 * r_ratio**2
            T_init[i] = T0 * r_ratio**alpha
        
        self._set_initial_conditions(rho_init, v_init, T_init)
        self.time = t_grid[0]
        
        # Output arrays
        nt = len(t_grid)
        v_out = np.zeros((nt, self.nr))
        rho_out = np.zeros((nt, self.nr))
        T_out = np.zeros((nt, self.nr))
        
        # Particle tracking setup
        particles_enabled = False
        particle_groups = {}
        
        if isinstance(num_particles, dict):
            particles_enabled = True
            for group_name, n_p in num_particles.items():
                inj_times = particle_injection_rate[group_name]
                rel_times = particle_release_rate[group_name] if particle_release_rate and group_name in particle_release_rate else inj_times
                behavior = 1 if 'cme' in group_name.lower() else 0
                
                particle_groups[group_name] = {
                    'n_particles': n_p,
                    'injection_times': inj_times,
                    'release_times': rel_times,
                    'r': [], 'v': [], 't': [], 't_inject': [], 'active': [],
                    'particles_injected': 0,
                    'behavior': behavior
                }
        elif isinstance(num_particles, int) and num_particles > 0:
            particles_enabled = True
            inj_times = particle_injection_rate if particle_injection_rate is not None else np.zeros(num_particles)
            rel_times = particle_release_rate if particle_release_rate is not None else inj_times
            particle_groups['default'] = {
                'n_particles': num_particles,
                'injection_times': inj_times,
                'release_times': rel_times,
                'r': [], 'v': [], 't': [], 't_inject': [], 'active': [],
                'particles_injected': 0,
                'behavior': 0
            }
        
        # Time loop
        t_idx = 0
        start_time = time_module.time()
        
        # Save first snapshot
        if abs(t_grid[0] - self.time) < 1e-5:
            v_snap, rho_snap, T_snap = _extract_snapshot(
                self.U, self.nr, self.gamma, M_P_SI, K_B_SI)
            v_out[0, :] = v_snap
            rho_out[0, :] = rho_snap
            T_out[0, :] = T_snap
            t_idx = 1
        
        while t_idx < nt:
            target_time = t_grid[t_idx]
            
            while self.time < target_time:
                dt = self._get_dt()
                if self.time + dt > target_time:
                    dt = target_time - self.time
                
                self._step(dt, lambda t: (rho_bc_func(t), v_bc_func(t), T_bc_func(t)))
                
                # Particle advection
                if particles_enabled:
                    v_curr = self.U[:, 1] / self.U[:, 0]
                    
                    for group in particle_groups.values():
                        # Inject new particles
                        while (group['particles_injected'] < group['n_particles'] and 
                               group['injection_times'][group['particles_injected']] <= self.time):
                            group['r'].append([self.r[0]])
                            group['v'].append([v_curr[0]])
                            group['t'].append([self.time])
                            group['t_inject'].append(group['injection_times'][group['particles_injected']])
                            group['active'].append(True)
                            group['particles_injected'] += 1
                        
                        # Advect particles
                        behavior = group.get('behavior', 0)
                        for i in range(len(group['active'])):
                            if group['active'][i]:
                                if self.time < group['release_times'][i]:
                                    group['r'][i].append(self.r[0])
                                    group['v'][i].append(v_curr[0])
                                    group['t'][i].append(self.time)
                                    continue
                                
                                r_p = group['r'][i][-1]
                                r_new, v_new, is_active = _advect_particle_rk2(
                                    r_p, v_curr, self.r, dt, behavior
                                )
                                
                                group['active'][i] = is_active
                                if is_active:
                                    group['r'][i].append(r_new)
                                    group['v'][i].append(v_new)
                                    group['t'][i].append(self.time)
            
            # Save snapshot
            v_snap, rho_snap, T_snap = _extract_snapshot(
                self.U, self.nr, self.gamma, M_P_SI, K_B_SI)
            v_out[t_idx, :] = v_snap
            rho_out[t_idx, :] = rho_snap
            T_out[t_idx, :] = T_snap
            t_idx += 1
        
        if self.verbose:
            elapsed = time_module.time() - start_time
            print(f"  Completed in {elapsed:.2f} seconds")
        
        results = {
            't': t_grid,
            'r': self.r,
            'v': v_out,
            'rho': rho_out,
            'T': T_out,
            'solver_info': {
                'reconstruction': self.reconstruction,
                'time_integration': self.time_integration,
            }
        }
        
        # Format particle data
        if particles_enabled:
            for name, g in particle_groups.items():
                if len(g['r']) > 0:
                    max_len = max(len(traj) for traj in g['r'])
                    n_p = len(g['r'])
                    
                    r_arr = np.full((n_p, max_len), np.nan)
                    v_arr = np.full((n_p, max_len), np.nan)
                    t_arr = np.full((n_p, max_len), np.nan)
                    
                    for i in range(n_p):
                        traj_len = len(g['r'][i])
                        r_arr[i, :traj_len] = g['r'][i]
                        v_arr[i, :traj_len] = g['v'][i]
                        t_arr[i, :traj_len] = g['t'][i]
                    
                    g['r'] = r_arr
                    g['v'] = v_arr
                    g['t'] = t_arr
                    g['t_inject'] = np.array(g['t_inject'])
                    g['active'] = np.array(g['active'])
            
            if len(particle_groups) == 1 and 'default' in particle_groups:
                g = particle_groups['default']
                results['particles'] = {
                    'r': g['r'], 'v': g['v'], 't': g['t'],
                    't_inject': g['t_inject'], 'active': g['active']
                }
            else:
                results['particles'] = {'groups': particle_groups}
        
        return results


# =============================================================================
# Factory Function
# =============================================================================

def create_solver(r_grid, gamma=5.0/3.0, method='hllc-plm', cfl=None, verbose=False):
    """
    Create a compressible solver with specified method.
    
    Parameters
    ----------
    r_grid : ndarray
        Radial grid in m (SI units)
    gamma : float, optional
        Adiabatic index (default 5/3)
    method : str, optional
        Solver configuration: 'hllc-{pcm|plm}[-{euler|rk2}]'
        Examples: 'hllc-plm', 'hllc-pcm', 'hllc-plm-rk2'
    cfl : float, optional
        CFL number (default depends on reconstruction)
    verbose : bool, optional
        Print progress
        
    Returns
    -------
    CompressibleSolver
        Configured solver instance
    """
    parts = method.lower().split('-')
    
    reconstruction = parts[1] if len(parts) > 1 else 'plm'
    time_int = parts[2] if len(parts) > 2 else 'euler'
    
    return CompressibleSolver(
        r_grid, gamma=gamma, cfl=cfl,
        reconstruction=reconstruction,
        time_integration=time_int, verbose=verbose
    )


# =============================================================================
# Benchmark and Utility Functions
# =============================================================================

def benchmark_solvers(r_grid, t_end, v_bc, rho_bc, T_bc, gamma=5.0/3.0,
                      methods=None, reference='hllc-plm-rk2'):
    """
    Compare different solver configurations on the same problem.
    
    Parameters
    ----------
    r_grid : ndarray
        Radial grid in m (SI units)
    t_end : float
        End time in seconds
    v_bc, rho_bc, T_bc : float
        Boundary condition values (constant)
    gamma : float, optional
        Adiabatic index
    methods : list, optional
        List of method strings to compare
    reference : str, optional
        Reference solution method for error calculation
        
    Returns
    -------
    dict
        Results for each method including timing and errors
    """
    import time as time_module
    
    if methods is None:
        methods = [
            'hllc-pcm',
            'hllc-plm',
            'hllc-plm-rk2',
        ]
    
    dt_out = t_end / 100
    t_grid = np.arange(0, t_end + dt_out, dt_out)
    
    v_bc_func = lambda t: v_bc
    rho_bc_func = lambda t: rho_bc
    T_bc_func = lambda t: T_bc
    
    results = {}
    
    print(f"Running reference: {reference}")
    solver = create_solver(r_grid, gamma=gamma, method=reference, verbose=False)
    t0 = time_module.time()
    ref_result = solver.solve(t_grid, v_bc_func, rho_bc_func, T_bc_func)
    ref_time = time_module.time() - t0
    results[reference] = {
        'result': ref_result,
        'time': ref_time,
        'error_v': 0.0,
        'error_rho': 0.0,
    }
    
    for method in methods:
        if method == reference:
            continue
            
        print(f"Running: {method}")
        solver = create_solver(r_grid, gamma=gamma, method=method, verbose=False)
        t0 = time_module.time()
        result = solver.solve(t_grid, v_bc_func, rho_bc_func, T_bc_func)
        elapsed = time_module.time() - t0
        
        error_v = np.sqrt(np.mean((result['v'][-1] - ref_result['v'][-1])**2))
        error_rho = np.sqrt(np.mean((result['rho'][-1] - ref_result['rho'][-1])**2))
        
        results[method] = {
            'result': result,
            'time': elapsed,
            'error_v': error_v,
            'error_rho': error_rho,
        }
    
    print("\n" + "="*70)
    print(f"{'Method':<20} {'Time (s)':<12} {'Error v':<15} {'Error rho':<15}")
    print("="*70)
    for method, data in results.items():
        print(f"{method:<20} {data['time']:<12.3f} {data['error_v']:<15.2e} {data['error_rho']:<15.2e}")
    
    return results


def list_available_methods():
    """
    Print available solver methods and configurations.
    
    Returns
    -------
    dict
        Dictionary of available options
    """
    print("="*70)
    print("Available Compressible Solver Methods")
    print("="*70)
    
    print("\nReconstruction Methods:")
    print("-" * 50)
    print("  'pcm': Piecewise Constant Method (1st order, Godunov)")
    print("  'plm': Piecewise Linear Method (2nd order) with MC limiter [DEFAULT]")
    
    print("\nTime Integration:")
    print("-" * 50)
    print("  'euler': Forward Euler (1st order) [DEFAULT]")
    print("  'rk2': Runge-Kutta 2nd order (Heun method)")
    
    print("\nMethod String Format:")
    print("-" * 50)
    print("  'hllc[-{reconstruction}[-{time_integrator}]]'")
    print("\nExamples:")
    print("  'hllc'        -> HLLC + PLM + Euler (default)")
    print("  'hllc-plm'    -> HLLC + PLM + Euler")
    print("  'hllc-pcm'    -> HLLC + PCM + Euler (1st order)")
    print("  'hllc-plm-rk2'-> HLLC + PLM + RK2 (full 2nd order)")
    print()
    
    return {
        'reconstruction': ['pcm', 'plm'],
        'time_integrator': ['euler', 'rk2'],
    }
