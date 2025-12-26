"""
Compressible hydrodynamics solvers for HUXt.

This module provides multiple numerical methods for solving the 1D spherical 
compressible Euler equations, allowing easy comparison of accuracy and performance.

Available Riemann Solvers:
- 'hllc': HLLC (Harten-Lax-van Leer-Contact) - fast, robust
- 'hll': HLL (Harten-Lax-van Leer) - faster, more diffusive  
- 'roe': Roe's approximate solver - accurate for smooth flows
- 'rusanov': Rusanov/Local Lax-Friedrichs - most robust, most diffusive

Available Reconstruction Methods:
- 'plm': Piecewise Linear Method (2nd order) with MC limiter
- 'pcm': Piecewise Constant Method (1st order, Godunov)

Available Time Integration:
- 'euler': Forward Euler (1st order)
- 'rk2': Runge-Kutta 2nd order (Heun's method)

Example Usage:
    from huxt.compressible_solvers import create_solver
    
    # Create solver with specific methods
    solver = create_solver(r_grid, gamma=1.5, method='hllc-plm')
    
    # Or specify each component
    solver = create_solver(r_grid, gamma=1.5, 
                          riemann='hllc', reconstruction='plm', time_integrator='euler')
    
    # Run simulation
    result = solver.solve(t_grid, v_bc_func, rho_bc_func, T_bc_func)

Author: HUXt Development Team
"""

import numpy as np
from numba import njit

__all__ = [
    'CompressibleSolver',
    'create_solver',
    'benchmark_solvers',
    'list_available_methods',
    # Constants
    'K_B_CGS', 'M_P_CGS', 'KM_TO_CM', 'KGM3_TO_GCM3',
]

# Constants
SMALL_RHO = 1e-30
SMALL_P = 1e-30

# Physical constants (CGS units)
K_B_CGS = 1.380649e-16      # erg/K
M_P_CGS = 1.67262192e-24    # g
K_B_SI = 1.38064852e-23     # J/K  
M_P_SI = 1.67262192e-27     # kg
KM_TO_CM = 1e5              # cm/km
KGM3_TO_GCM3 = 1e-3         # g/cm³ per kg/m³


# =============================================================================
# Equation of State
# =============================================================================

@njit(cache=True)
def get_pressure(rho, eint, gamma):
    """Ideal gas equation of state: p = rho * e * (gamma - 1)"""
    return rho * eint * (gamma - 1.0)

@njit(cache=True)
def get_rhoe(p, gamma):
    """Internal energy density from pressure: rho * e = p / (gamma - 1)"""
    return p / (gamma - 1.0)

@njit(cache=True)
def get_sound_speed(rho, p, gamma):
    """Sound speed: c = sqrt(gamma * p / rho)"""
    return np.sqrt(gamma * max(p, SMALL_P) / max(rho, SMALL_RHO))

@njit(cache=True)
def cons_to_prim(U, gamma):
    """
    Convert conserved variables to primitive variables.
    U: [rho, mom, ener]
    Returns: [rho, v, p]
    """
    rho = max(U[0], SMALL_RHO)
    v = U[1] / rho
    eint = (U[2] - 0.5 * rho * v**2) / rho
    p = max(get_pressure(rho, eint, gamma), SMALL_P)
    return np.array([rho, v, p])

@njit(cache=True)
def prim_to_cons(q, gamma):
    """
    Convert primitive variables to conserved variables.
    q: [rho, v, p]
    Returns: [rho, mom, ener]
    """
    rho, v, p = q[0], q[1], q[2]
    mom = rho * v
    rhoe = get_rhoe(p, gamma)
    ener = rhoe + 0.5 * rho * v**2
    return np.array([rho, mom, ener])

@njit(cache=True)
def get_flux(U, gamma):
    """
    Compute flux vector from conserved state.
    F = [rho*v, rho*v^2 + p, (E+p)*v]
    """
    rho = max(U[0], SMALL_RHO)
    v = U[1] / rho
    E = U[2]
    eint = (E - 0.5 * rho * v**2) / rho
    p = max(get_pressure(rho, eint, gamma), SMALL_P)
    
    F = np.zeros(3)
    F[0] = rho * v
    F[1] = rho * v**2 + p
    F[2] = (E + p) * v
    return F


# =============================================================================
# Riemann Solvers
# =============================================================================

@njit(cache=True)
def riemann_rusanov(U_l, U_r, gamma):
    """
    Rusanov (Local Lax-Friedrichs) Riemann solver.
    Most robust but most diffusive. Good for strong shocks.
    
    F = 0.5*(F_l + F_r) - 0.5*S_max*(U_r - U_l)
    """
    # Left state
    rho_l = max(U_l[0], SMALL_RHO)
    v_l = U_l[1] / rho_l
    p_l = max(get_pressure(rho_l, (U_l[2] - 0.5*rho_l*v_l**2)/rho_l, gamma), SMALL_P)
    c_l = get_sound_speed(rho_l, p_l, gamma)
    
    # Right state
    rho_r = max(U_r[0], SMALL_RHO)
    v_r = U_r[1] / rho_r
    p_r = max(get_pressure(rho_r, (U_r[2] - 0.5*rho_r*v_r**2)/rho_r, gamma), SMALL_P)
    c_r = get_sound_speed(rho_r, p_r, gamma)
    
    # Maximum wave speed
    S_max = max(abs(v_l) + c_l, abs(v_r) + c_r)
    
    # Fluxes
    F_l = get_flux(U_l, gamma)
    F_r = get_flux(U_r, gamma)
    
    # Rusanov flux
    F = np.zeros(3)
    for i in range(3):
        F[i] = 0.5 * (F_l[i] + F_r[i]) - 0.5 * S_max * (U_r[i] - U_l[i])
    
    return F


@njit(cache=True)
def riemann_hll(U_l, U_r, gamma):
    """
    HLL (Harten-Lax-van Leer) Riemann solver.
    Fast and robust. More diffusive than HLLC but simpler.
    Does not resolve contact discontinuities.
    """
    # Left state
    rho_l = max(U_l[0], SMALL_RHO)
    v_l = U_l[1] / rho_l
    p_l = max(get_pressure(rho_l, (U_l[2] - 0.5*rho_l*v_l**2)/rho_l, gamma), SMALL_P)
    c_l = get_sound_speed(rho_l, p_l, gamma)
    
    # Right state
    rho_r = max(U_r[0], SMALL_RHO)
    v_r = U_r[1] / rho_r
    p_r = max(get_pressure(rho_r, (U_r[2] - 0.5*rho_r*v_r**2)/rho_r, gamma), SMALL_P)
    c_r = get_sound_speed(rho_r, p_r, gamma)
    
    # Wave speed estimates (Davis)
    S_l = min(v_l - c_l, v_r - c_r)
    S_r = max(v_l + c_l, v_r + c_r)
    
    if S_l >= 0:
        return get_flux(U_l, gamma)
    elif S_r <= 0:
        return get_flux(U_r, gamma)
    else:
        # HLL average state
        F_l = get_flux(U_l, gamma)
        F_r = get_flux(U_r, gamma)
        
        F = np.zeros(3)
        for i in range(3):
            F[i] = (S_r * F_l[i] - S_l * F_r[i] + S_l * S_r * (U_r[i] - U_l[i])) / (S_r - S_l)
        return F


@njit(cache=True)
def riemann_hllc(U_l, U_r, gamma):
    """
    HLLC (Harten-Lax-van Leer-Contact) Riemann solver.
    Good balance of accuracy and speed. Resolves contact discontinuities.
    
    This is currently the default "CGF" solver despite the naming.
    """
    # Left state
    rho_l = max(U_l[0], SMALL_RHO)
    v_l = U_l[1] / rho_l
    E_l = U_l[2]
    p_l = max(get_pressure(rho_l, (E_l - 0.5*rho_l*v_l**2)/rho_l, gamma), SMALL_P)
    c_l = get_sound_speed(rho_l, p_l, gamma)
    
    # Right state
    rho_r = max(U_r[0], SMALL_RHO)
    v_r = U_r[1] / rho_r
    E_r = U_r[2]
    p_r = max(get_pressure(rho_r, (E_r - 0.5*rho_r*v_r**2)/rho_r, gamma), SMALL_P)
    c_r = get_sound_speed(rho_r, p_r, gamma)
    
    # Wave speed estimates
    S_l = min(v_l - c_l, v_r - c_r)
    S_r = max(v_l + c_l, v_r + c_r)
    
    if S_l >= 0:
        return get_flux(U_l, gamma)
    elif S_r <= 0:
        return get_flux(U_r, gamma)
    
    # Contact wave speed
    denom = rho_l * (S_l - v_l) - rho_r * (S_r - v_r)
    if abs(denom) < 1e-14:
        S_c = 0.5 * (v_l + v_r)
    else:
        S_c = (p_r - p_l + rho_l * v_l * (S_l - v_l) - rho_r * v_r * (S_r - v_r)) / denom
    
    if S_c >= 0:
        # Left star state
        factor = rho_l * (S_l - v_l) / (S_l - S_c)
        U_star = np.zeros(3)
        U_star[0] = factor
        U_star[1] = factor * S_c
        U_star[2] = factor * (E_l/rho_l + (S_c - v_l) * (S_c + p_l/(rho_l*(S_l - v_l))))
        
        F_l = get_flux(U_l, gamma)
        F = np.zeros(3)
        for i in range(3):
            F[i] = F_l[i] + S_l * (U_star[i] - U_l[i])
        return F
    else:
        # Right star state
        factor = rho_r * (S_r - v_r) / (S_r - S_c)
        U_star = np.zeros(3)
        U_star[0] = factor
        U_star[1] = factor * S_c
        U_star[2] = factor * (E_r/rho_r + (S_c - v_r) * (S_c + p_r/(rho_r*(S_r - v_r))))
        
        F_r = get_flux(U_r, gamma)
        F = np.zeros(3)
        for i in range(3):
            F[i] = F_r[i] + S_r * (U_star[i] - U_r[i])
        return F


@njit(cache=True)
def riemann_roe(U_l, U_r, gamma):
    """
    Roe's approximate Riemann solver with entropy fix.
    More accurate for smooth flows but can fail at strong shocks
    without entropy fix.
    """
    # Left state
    rho_l = max(U_l[0], SMALL_RHO)
    v_l = U_l[1] / rho_l
    E_l = U_l[2]
    p_l = max(get_pressure(rho_l, (E_l - 0.5*rho_l*v_l**2)/rho_l, gamma), SMALL_P)
    H_l = (E_l + p_l) / rho_l  # Total enthalpy
    c_l = get_sound_speed(rho_l, p_l, gamma)
    
    # Right state
    rho_r = max(U_r[0], SMALL_RHO)
    v_r = U_r[1] / rho_r
    E_r = U_r[2]
    p_r = max(get_pressure(rho_r, (E_r - 0.5*rho_r*v_r**2)/rho_r, gamma), SMALL_P)
    H_r = (E_r + p_r) / rho_r
    c_r = get_sound_speed(rho_r, p_r, gamma)
    
    # Roe averages
    sqrt_rho_l = np.sqrt(rho_l)
    sqrt_rho_r = np.sqrt(rho_r)
    denom = sqrt_rho_l + sqrt_rho_r
    
    v_roe = (sqrt_rho_l * v_l + sqrt_rho_r * v_r) / denom
    H_roe = (sqrt_rho_l * H_l + sqrt_rho_r * H_r) / denom
    c_roe = np.sqrt((gamma - 1) * (H_roe - 0.5 * v_roe**2))
    
    # Eigenvalues
    lambda1 = v_roe - c_roe
    lambda2 = v_roe
    lambda3 = v_roe + c_roe
    
    # Entropy fix (Harten-Hyman)
    eps = 0.1 * c_roe
    if abs(lambda1) < eps:
        lambda1 = 0.5 * (lambda1**2 / eps + eps) * np.sign(lambda1)
    if abs(lambda3) < eps:
        lambda3 = 0.5 * (lambda3**2 / eps + eps) * np.sign(lambda3)
    
    # Wave strengths
    drho = rho_r - rho_l
    dv = v_r - v_l
    dp = p_r - p_l
    
    alpha1 = (dp - rho_l * c_roe * dv) / (2 * c_roe**2)
    alpha2 = drho - dp / c_roe**2
    alpha3 = (dp + rho_l * c_roe * dv) / (2 * c_roe**2)
    
    # Eigenvectors (right)
    r1 = np.array([1.0, v_roe - c_roe, H_roe - v_roe * c_roe])
    r2 = np.array([1.0, v_roe, 0.5 * v_roe**2])
    r3 = np.array([1.0, v_roe + c_roe, H_roe + v_roe * c_roe])
    
    # Flux
    F_l = get_flux(U_l, gamma)
    F_r = get_flux(U_r, gamma)
    
    F = np.zeros(3)
    for i in range(3):
        F[i] = 0.5 * (F_l[i] + F_r[i]) - 0.5 * (
            abs(lambda1) * alpha1 * r1[i] +
            abs(lambda2) * alpha2 * r2[i] +
            abs(lambda3) * alpha3 * r3[i]
        )
    
    return F


# =============================================================================
# Reconstruction Methods (Slope Limiters)
# =============================================================================

@njit(cache=True)
def minmod(a, b):
    """Minmod limiter - most diffusive TVD limiter"""
    if a * b <= 0:
        return 0.0
    elif abs(a) < abs(b):
        return a
    else:
        return b

@njit(cache=True)
def mc_limiter(a, b):
    """MC (Monotonized Central) limiter - good balance"""
    c = 0.5 * (a + b)
    if a * b <= 0:
        return 0.0
    else:
        return np.sign(c) * min(abs(c), 2*abs(a), 2*abs(b))

@njit(cache=True)
def reconstruct_pcm(U, i):
    """
    Piecewise Constant Method (1st order, Godunov).
    No reconstruction - just use cell averages.
    """
    return U[i].copy(), U[i].copy()

@njit(cache=True)
def reconstruct_plm(U, i, nr, gamma):
    """
    Piecewise Linear Method (2nd order) with MC limiter.
    Returns left and right states at cell i's right interface.
    """
    if i == 0 or i >= nr - 1:
        return U[i].copy(), U[i+1].copy() if i < nr-1 else U[i].copy()
    
    # Check for valid states first
    for j in [i-1, i, i+1]:
        if U[j, 0] <= SMALL_RHO or U[j, 2] <= 0:
            # Fall back to PCM if state is invalid
            return U[i].copy(), U[i+1].copy()
    
    # Convert to primitives for limiting
    q_im1 = cons_to_prim(U[i-1], gamma)
    q_i = cons_to_prim(U[i], gamma)
    q_ip1 = cons_to_prim(U[i+1], gamma)
    
    # Check for valid primitives
    if q_im1[0] <= 0 or q_im1[2] <= 0 or q_i[0] <= 0 or q_i[2] <= 0 or q_ip1[0] <= 0 or q_ip1[2] <= 0:
        return U[i].copy(), U[i+1].copy()
    
    # Slopes with MC limiter
    dq_l = np.zeros(3)
    dq_r = np.zeros(3)
    
    for n in range(3):
        slope_l = q_i[n] - q_im1[n]
        slope_r = q_ip1[n] - q_i[n]
        dq_l[n] = mc_limiter(slope_l, slope_r)
        
        if i < nr - 2:
            q_ip2 = cons_to_prim(U[i+2], gamma)
            slope_l2 = q_ip1[n] - q_i[n]
            slope_r2 = q_ip2[n] - q_ip1[n]
            dq_r[n] = mc_limiter(slope_l2, slope_r2)
        else:
            dq_r[n] = 0.0
    
    # Reconstruct at interface
    q_L = q_i + 0.5 * dq_l
    q_R = q_ip1 - 0.5 * dq_r
    
    # Ensure positivity
    q_L[0] = max(q_L[0], SMALL_RHO)
    q_L[2] = max(q_L[2], SMALL_P)
    q_R[0] = max(q_R[0], SMALL_RHO)
    q_R[2] = max(q_R[2], SMALL_P)
    
    return prim_to_cons(q_L, gamma), prim_to_cons(q_R, gamma)


# =============================================================================
# JIT-Compiled Full Step Functions (for performance)
# =============================================================================

@njit(cache=True)
def _compute_fluxes_pcm(U, nr, gamma, riemann_type):
    """
    Compute fluxes at all interfaces using PCM reconstruction.
    riemann_type: 0=rusanov, 1=hll, 2=hllc, 3=roe
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
        
        if riemann_type == 0:
            fluxes[i] = riemann_rusanov(U_l, U_r, gamma)
        elif riemann_type == 1:
            fluxes[i] = riemann_hll(U_l, U_r, gamma)
        elif riemann_type == 2:
            fluxes[i] = riemann_hllc(U_l, U_r, gamma)
        else:
            fluxes[i] = riemann_roe(U_l, U_r, gamma)
    
    return fluxes


@njit(cache=True)
def _compute_fluxes_plm(U, nr, gamma, riemann_type):
    """
    Compute fluxes at all interfaces using PLM reconstruction.
    riemann_type: 0=rusanov, 1=hll, 2=hllc, 3=roe
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
            # PLM reconstruction
            U_l, U_r = reconstruct_plm(U, i-1, nr, gamma)
        
        if riemann_type == 0:
            fluxes[i] = riemann_rusanov(U_l, U_r, gamma)
        elif riemann_type == 1:
            fluxes[i] = riemann_hll(U_l, U_r, gamma)
        elif riemann_type == 2:
            fluxes[i] = riemann_hllc(U_l, U_r, gamma)
        else:
            fluxes[i] = riemann_roe(U_l, U_r, gamma)
    
    return fluxes


@njit(cache=True)
def _compute_source(U, r, nr, gamma):
    """Compute geometric source terms for spherical coordinates."""
    S = np.zeros((nr, 3))
    for i in range(nr):
        q = cons_to_prim(U[i], gamma)
        # Geometric pressure term: 2p/r
        S[i, 1] = 2.0 * q[2] / r[i]
    return S


@njit(cache=True)
def _get_dt(U, nr, dx, gamma, cfl):
    """Compute CFL-limited timestep."""
    max_speed = 0.0
    for i in range(nr):
        q = cons_to_prim(U[i], gamma)
        c = get_sound_speed(q[0], q[2], gamma)
        speed = abs(q[1]) + c
        if speed > max_speed:
            max_speed = speed
    
    if max_speed == 0:
        return 1.0
    
    min_dx = dx[0]
    for i in range(1, len(dx)):
        if dx[i] < min_dx:
            min_dx = dx[i]
    
    return cfl * min_dx / max_speed


@njit(cache=True)
def _step_euler_jit(U, U_bc, nr, r, A, V, dt, gamma, riemann_type, use_plm):
    """
    JIT-compiled Euler step.
    riemann_type: 0=rusanov, 1=hll, 2=hllc, 3=roe
    use_plm: True for PLM, False for PCM
    """
    # Compute fluxes
    if use_plm:
        fluxes = _compute_fluxes_plm(U, nr, gamma, riemann_type)
    else:
        fluxes = _compute_fluxes_pcm(U, nr, gamma, riemann_type)
    
    # Compute source
    S = _compute_source(U, r, nr, gamma)
    
    # Update
    U_new = np.zeros_like(U)
    for i in range(nr):
        # Check for valid volume
        if V[i] <= 0:
            U_new[i] = U[i]
            continue
            
        dUdt = -(A[i+1] * fluxes[i+1] - A[i] * fluxes[i]) / V[i] + S[i]
        U_new[i] = U[i] + dt * dUdt
        
        # Check for negative density or energy and reset if needed
        if U_new[i, 0] < SMALL_RHO or U_new[i, 2] < SMALL_P:
            U_new[i] = U[i]  # Keep old state
    
    # Apply inner BC
    U_new[0] = U_bc
    
    return U_new


@njit(cache=True)
def _step_rk2_jit(U, U_bc, nr, r, A, V, dt, gamma, riemann_type, use_plm):
    """
    JIT-compiled RK2 (Heun) step.
    """
    # Stage 1: Euler predictor
    U1 = _step_euler_jit(U, U_bc, nr, r, A, V, dt, gamma, riemann_type, use_plm)
    
    # Stage 2: Corrector
    U2 = _step_euler_jit(U1, U_bc, nr, r, A, V, dt, gamma, riemann_type, use_plm)
    
    # Average
    U_new = np.zeros_like(U)
    for i in range(nr):
        for j in range(3):
            U_new[i, j] = 0.5 * (U[i, j] + U2[i, j])
    
    # Apply BC
    U_new[0] = U_bc
    
    return U_new


# =============================================================================
# Main Solver Class
# =============================================================================

class CompressibleSolver:
    """
    Modular 1D spherical compressible Euler solver.
    
    Allows selection of different numerical methods:
    - Riemann solvers: 'rusanov', 'hll', 'hllc', 'roe'
    - Reconstruction: 'pcm' (1st order), 'plm' (2nd order)
    - Time integration: 'euler', 'rk2'
    
    Example:
        solver = CompressibleSolver(r_grid, gamma=5/3, 
                                    riemann='hllc', 
                                    reconstruction='plm',
                                    time_integration='rk2')
        results = solver.solve(t_grid, v_bc_func, rho_bc_func, T_bc_func)
    """
    
    RIEMANN_SOLVERS = {
        'rusanov': riemann_rusanov,
        'hll': riemann_hll,
        'hllc': riemann_hllc,
        'roe': riemann_roe,
    }
    
    def __init__(self, r_grid, gamma=5.0/3.0, cfl=None, 
                 riemann='hllc', reconstruction='plm', time_integration='euler',
                 verbose=False):
        """
        Initialize the solver.
        
        Args:
            r_grid: Radial grid (cell centers) in cm
            gamma: Adiabatic index (default 5/3 for monoatomic gas)
            cfl: CFL number for timestep control (default: 0.8 for PCM, 0.4 for PLM)
            riemann: Riemann solver choice ('rusanov', 'hll', 'hllc', 'roe')
            reconstruction: Reconstruction method ('pcm', 'plm')
            time_integration: Time integration ('euler', 'rk2')
            verbose: Print progress information
        """
        self.r = r_grid.copy()
        self.nr = len(r_grid)
        self.gamma = gamma
        self.verbose = verbose
        
        # Method selection
        self.riemann_name = riemann.lower()
        self.reconstruction = reconstruction.lower()
        self.time_integration = time_integration.lower()
        
        # Set CFL based on reconstruction order if not specified
        if cfl is None:
            if self.reconstruction == 'plm':
                self.cfl = 0.4  # Lower CFL for 2nd order
            else:
                self.cfl = 0.8  # Higher CFL for 1st order
        else:
            self.cfl = cfl
        
        if self.riemann_name not in self.RIEMANN_SOLVERS:
            raise ValueError(f"Unknown Riemann solver: {riemann}. "
                           f"Available: {list(self.RIEMANN_SOLVERS.keys())}")
        
        self.riemann_solver = self.RIEMANN_SOLVERS[self.riemann_name]
        
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
        
        # State
        self.U = np.zeros((self.nr, 3))
        self.time = 0.0
        
    def set_initial_conditions(self, rho, v, T):
        """Set initial conditions from primitive variables (CGS units)."""
        for i in range(self.nr):
            p = rho[i] * K_B_CGS * T[i] / M_P_CGS
            p = max(p, SMALL_P)
            self.U[i] = prim_to_cons(np.array([rho[i], v[i], p]), self.gamma)
            
    def get_dt(self):
        """Compute CFL-limited timestep."""
        max_speed = 0.0
        for i in range(self.nr):
            q = cons_to_prim(self.U[i], self.gamma)
            c = get_sound_speed(q[0], q[2], self.gamma)
            max_speed = max(max_speed, abs(q[1]) + c)
        
        if max_speed == 0:
            return 1.0
        return self.cfl * np.min(self.dx) / max_speed
    
    def compute_fluxes(self, U):
        """Compute fluxes at all interfaces using selected methods."""
        fluxes = np.zeros((self.nr + 1, 3))
        
        for i in range(self.nr + 1):
            if i == 0:
                # Inner boundary - use BC state
                U_l = U[0]
                U_r = U[0]
            elif i == self.nr:
                # Outer boundary - zero gradient
                U_l = U[-1]
                U_r = U[-1]
            else:
                # Interior interfaces
                if self.reconstruction == 'pcm':
                    U_l = U[i-1]
                    U_r = U[i]
                else:  # plm
                    U_l, U_r = reconstruct_plm(U, i-1, self.nr, self.gamma)
            
            fluxes[i] = self.riemann_solver(U_l, U_r, self.gamma)
        
        return fluxes
    
    def compute_source(self, U):
        """Compute geometric source terms for spherical coordinates."""
        S = np.zeros((self.nr, 3))
        for i in range(self.nr):
            q = cons_to_prim(U[i], self.gamma)
            # Geometric pressure term: 2p/r
            S[i, 1] = 2.0 * q[2] / self.r[i]
        return S
    
    def step_euler(self, dt, U_bc):
        """Forward Euler time step."""
        # Set BC
        U = self.U.copy()
        
        # Fluxes
        fluxes = self.compute_fluxes(U)
        
        # Update
        S = self.compute_source(U)
        
        for i in range(self.nr):
            dUdt = -(self.A[i+1] * fluxes[i+1] - self.A[i] * fluxes[i]) / self.V[i] + S[i]
            U[i] = U[i] + dt * dUdt
            
            # Apply inner BC
            if i == 0:
                U[i] = U_bc
                
        return U
    
    def step_rk2(self, dt, U_bc):
        """RK2 (Heun) time step - 2nd order in time."""
        # Stage 1: Euler predictor
        U1 = self.step_euler(dt, U_bc)
        
        # Stage 2: Corrector
        U_save = self.U.copy()
        self.U = U1
        U2 = self.step_euler(dt, U_bc)
        self.U = U_save
        
        # Average
        U_new = 0.5 * (self.U + U2)
        U_new[0] = U_bc  # Apply BC
        
        return U_new
    
    def step(self, dt, bc_func):
        """Advance solution by dt using selected time integration."""
        rho_bc, v_bc, T_bc = bc_func(self.time + dt)
        p_bc = rho_bc * K_B_CGS * T_bc / M_P_CGS
        U_bc = prim_to_cons(np.array([rho_bc, v_bc, p_bc]), self.gamma)
        
        # Use JIT-compiled step functions for speed
        riemann_type = {'rusanov': 0, 'hll': 1, 'hllc': 2, 'roe': 3}[self.riemann_name]
        use_plm = (self.reconstruction == 'plm')
        
        if self.time_integration == 'rk2':
            self.U = _step_rk2_jit(self.U, U_bc, self.nr, self.r, self.A, self.V,
                                    dt, self.gamma, riemann_type, use_plm)
        else:
            self.U = _step_euler_jit(self.U, U_bc, self.nr, self.r, self.A, self.V,
                                      dt, self.gamma, riemann_type, use_plm)
            
        self.time += dt
        return self.U
    
    def get_dt_jit(self):
        """Compute CFL-limited timestep using JIT function."""
        return _get_dt(self.U, self.nr, self.dx, self.gamma, self.cfl)
    
    def solve(self, t_grid, v_bc_func, rho_bc_func, T_bc_func, 
              num_particles=0, particle_injection_rate=None):
        """
        Run simulation over time grid.
        
        Args:
            t_grid: Output times (seconds)
            v_bc_func: Velocity boundary condition function v(t) in cm/s
            rho_bc_func: Density boundary condition function rho(t) in g/cm³
            T_bc_func: Temperature boundary condition function T(t) in K
            num_particles: Number of tracer particles or dict of particle groups
            particle_injection_rate: Particle injection times
            
        Returns:
            dict: Results with 't', 'r', 'v', 'rho', 'T', and optionally 'particles'
        """
        import time as time_module
        
        if self.verbose:
            print(f"CompressibleSolver starting:")
            print(f"  Riemann solver: {self.riemann_name}")
            print(f"  Reconstruction: {self.reconstruction}")
            print(f"  Time integration: {self.time_integration}")
        
        # Initialize
        v0 = v_bc_func(t_grid[0])
        rho0 = rho_bc_func(t_grid[0])
        T0 = T_bc_func(t_grid[0])
        
        # Initialize using Parker solution scaling
        from huxt.huxt_insitu import map_density_parker, map_temperature_parker
        import astropy.units as u
        
        r0 = self.r[0]
        v_init = np.ones_like(self.r) * v0
        
        # Use Parker solution to scale density and temperature with radius
        rho_init = np.zeros_like(self.r)
        T_init = np.zeros_like(self.r)
        for i, r in enumerate(self.r):
            rho_init[i] = map_density_parker(rho0, r0*u.solRad, r*u.solRad)
            T_init[i] = map_temperature_parker(T0 * u.K, r0*u.solRad, r*u.solRad, gamma=self.gamma).value
        
        self.set_initial_conditions(rho_init, v_init, T_init)
        self.time = t_grid[0]
        
        # Output arrays
        nt = len(t_grid)
        v_out = np.zeros((nt, self.nr))
        rho_out = np.zeros((nt, self.nr))
        T_out = np.zeros((nt, self.nr))
        
        # Particle tracking (same as CGFSolver)
        particles_enabled = False
        particle_groups = {}
        
        if isinstance(num_particles, dict):
            particles_enabled = True
            for group_name, n_p in num_particles.items():
                inj_times = particle_injection_rate[group_name]
                particle_groups[group_name] = {
                    'n_particles': n_p,
                    'injection_times': inj_times,
                    'r': [], 'v': [], 't': [], 't_inject': [], 'active': [],
                    'particles_injected': 0
                }
        elif isinstance(num_particles, int) and num_particles > 0:
            particles_enabled = True
            inj_times = particle_injection_rate if particle_injection_rate is not None else np.zeros(num_particles)
            particle_groups['default'] = {
                'n_particles': num_particles,
                'injection_times': inj_times,
                'r': [], 'v': [], 't': [], 't_inject': [], 'active': [],
                'particles_injected': 0
            }
        
        # Time loop
        t_idx = 0
        start_time = time_module.time()
        
        # Save first snapshot
        if abs(t_grid[0] - self.time) < 1e-5:
            for i in range(self.nr):
                q = cons_to_prim(self.U[i], self.gamma)
                v_out[0, i] = q[1]
                rho_out[0, i] = q[0]
                T_out[0, i] = q[2] * M_P_CGS / (q[0] * K_B_CGS)
            t_idx = 1
        
        while t_idx < nt:
            target_time = t_grid[t_idx]
            
            while self.time < target_time:
                dt = self.get_dt_jit()  # Use JIT version for speed
                if self.time + dt > target_time:
                    dt = target_time - self.time
                
                self.step(dt, lambda t: (rho_bc_func(t), v_bc_func(t), T_bc_func(t)))
                
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
                        
                        # Advect particles (RK2)
                        for i in range(len(group['active'])):
                            if group['active'][i]:
                                r_p = group['r'][i][-1]
                                if r_p < self.r[0] or r_p > self.r[-1]:
                                    group['active'][i] = False
                                    continue
                                
                                v_p = np.interp(r_p, self.r, v_curr)
                                r_mid = r_p + 0.5 * v_p * dt
                                
                                if r_mid < self.r[0] or r_mid > self.r[-1]:
                                    v_mid = v_p
                                else:
                                    v_mid = np.interp(r_mid, self.r, v_curr)
                                
                                r_new = r_p + v_mid * dt
                                group['r'][i].append(r_new)
                                group['v'][i].append(v_mid)
                                group['t'][i].append(self.time)
            
            # Save snapshot
            for i in range(self.nr):
                q = cons_to_prim(self.U[i], self.gamma)
                v_out[t_idx, i] = q[1]
                rho_out[t_idx, i] = q[0]
                T_out[t_idx, i] = q[2] * M_P_CGS / (q[0] * K_B_CGS)
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
                'riemann': self.riemann_name,
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
# Convenience Factory Function
# =============================================================================

def create_solver(r_grid, gamma=5.0/3.0, method='hllc-plm', cfl=0.8, verbose=False):
    """
    Create a compressible solver with specified method.
    
    Args:
        r_grid: Radial grid in cm
        gamma: Adiabatic index
        method: Solver configuration string, e.g.:
                'hllc-plm' (default) - HLLC Riemann solver with PLM reconstruction
                'rusanov-pcm' - Rusanov with 1st order (most robust)
                'roe-plm-rk2' - Roe solver with PLM and RK2 time integration
        cfl: CFL number
        verbose: Print progress
        
    Returns:
        CompressibleSolver instance
    """
    parts = method.lower().split('-')
    
    riemann = parts[0] if len(parts) > 0 else 'hllc'
    reconstruction = parts[1] if len(parts) > 1 else 'plm'
    time_int = parts[2] if len(parts) > 2 else 'euler'
    
    return CompressibleSolver(
        r_grid, gamma=gamma, cfl=cfl,
        riemann=riemann, reconstruction=reconstruction,
        time_integration=time_int, verbose=verbose
    )


# =============================================================================
# Benchmark Function
# =============================================================================

def benchmark_solvers(r_grid, t_end, v_bc, rho_bc, T_bc, gamma=5.0/3.0,
                      methods=None, reference='hllc-plm-rk2'):
    """
    Compare different solver methods on the same problem.
    
    Args:
        r_grid: Radial grid in cm
        t_end: End time in seconds
        v_bc, rho_bc, T_bc: Boundary condition values (constant)
        gamma: Adiabatic index
        methods: List of method strings to compare (default: all combinations)
        reference: Reference solution method for error calculation
        
    Returns:
        dict: Results for each method including timing and errors
    """
    import time as time_module
    
    if methods is None:
        methods = [
            'rusanov-pcm',
            'hll-pcm',
            'hllc-pcm',
            'rusanov-plm',
            'hll-plm',
            'hllc-plm',
            'roe-plm',
            'hllc-plm-rk2',
        ]
    
    # Time grid
    dt_out = t_end / 100
    t_grid = np.arange(0, t_end + dt_out, dt_out)
    
    # BC functions (constant)
    v_bc_func = lambda t: v_bc
    rho_bc_func = lambda t: rho_bc
    T_bc_func = lambda t: T_bc
    
    results = {}
    
    # Run reference first
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
    
    # Run other methods
    for method in methods:
        if method == reference:
            continue
            
        print(f"Running: {method}")
        solver = create_solver(r_grid, gamma=gamma, method=method, verbose=False)
        t0 = time_module.time()
        result = solver.solve(t_grid, v_bc_func, rho_bc_func, T_bc_func)
        elapsed = time_module.time() - t0
        
        # Compute errors relative to reference
        error_v = np.sqrt(np.mean((result['v'][-1] - ref_result['v'][-1])**2))
        error_rho = np.sqrt(np.mean((result['rho'][-1] - ref_result['rho'][-1])**2))
        
        results[method] = {
            'result': result,
            'time': elapsed,
            'error_v': error_v,
            'error_rho': error_rho,
        }
    
    # Print summary
    print("\n" + "="*70)
    print(f"{'Method':<20} {'Time (s)':<12} {'Error v':<15} {'Error rho':<15}")
    print("="*70)
    for method, data in results.items():
        print(f"{method:<20} {data['time']:<12.3f} {data['error_v']:<15.2e} {data['error_rho']:<15.2e}")
    
    return results


def list_available_methods():
    """
    Print available solver methods and their descriptions.
    
    This function lists all available combinations of Riemann solvers,
    reconstruction methods, and time integrators that can be used
    with the CompressibleSolver class.
    """
    print("="*70)
    print("Available Compressible Solver Methods")
    print("="*70)
    
    print("\nRiemann Solvers (accuracy roughly increases down the list):")
    print("-" * 50)
    riemann_info = {
        'rusanov': 'Rusanov/Local Lax-Friedrichs - most robust, most diffusive',
        'hll': 'HLL (Harten-Lax-van Leer) - fast, diffusive',
        'hllc': 'HLLC (HLL-Contact) - fast, captures contact waves [DEFAULT]',
        'roe': "Roe's solver - accurate for smooth flows",
    }
    for name, desc in riemann_info.items():
        print(f"  '{name}': {desc}")
    
    print("\nReconstruction Methods:")
    print("-" * 50)
    recon_info = {
        'pcm': 'Piecewise Constant Method (1st order, Godunov)',
        'plm': 'Piecewise Linear Method (2nd order) with MC limiter [DEFAULT]',
    }
    for name, desc in recon_info.items():
        print(f"  '{name}': {desc}")
    
    print("\nTime Integration:")
    print("-" * 50)
    time_info = {
        'euler': 'Forward Euler (1st order) [DEFAULT]',
        'rk2': 'Runge-Kutta 2nd order (Heun method)',
    }
    for name, desc in time_info.items():
        print(f"  '{name}': {desc}")
    
    print("\nMethod String Format:")
    print("-" * 50)
    print("  '{riemann}[-{reconstruction}[-{time_integrator}]]'")
    print("\nExamples:")
    print("  'hllc'        -> HLLC + PLM + Euler (default)")
    print("  'hllc-plm'    -> HLLC + PLM + Euler")
    print("  'rusanov-pcm' -> Rusanov + PCM + Euler (1st order)")
    print("  'hllc-plm-rk2'-> HLLC + PLM + RK2 (full 2nd order)")
    print("  'roe-plm'     -> Roe + PLM + Euler")
    print()
    
    return {
        'riemann': list(riemann_info.keys()),
        'reconstruction': list(recon_info.keys()),
        'time_integrator': list(time_info.keys()),
    }
