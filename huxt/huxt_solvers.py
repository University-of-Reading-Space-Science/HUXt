"""
Compressible hydrodynamics solvers for HUXt.

This module provides numerical methods for solving the 1D spherical 
compressible Euler equations using a finite-volume formulation with
area-weighted fluxes AND geometric source terms.

Spherical geometry is handled through:
1. Interface areas: A = 4πr² 
2. Cell volumes: V = 4/3 π(r³_out - r³_in)
3. Geometric source term: S = [0, 2p/r, 0] for momentum equation

The combination of area-weighted fluxes + the 2p/r source term ensures:
- Mass, momentum, and energy are conserved correctly
- Parker nozzle acceleration emerges naturally
- Shocks and stream interactions are handled correctly by the Riemann solver

Riemann Solver:
- 'hllc': HLLC (Harten-Lax-van Leer-Contact) - fast, robust, captures contact waves

Available Reconstruction Methods:
- 'plm': Piecewise Linear Method (2nd order) with MC limiter
- 'pcm': Piecewise Constant Method (1st order, Godunov)

Available Time Integration:
- 'euler': Forward Euler (1st order)
- 'rk2': Runge-Kutta 2nd order (Heun's method)

Example Usage:
    from huxt.huxt_solvers import create_solver
    
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
import astropy.units as u
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

@njit
def _advect_particle_rk2(r_p, v_grid, r_grid, dt, behavior):
    """
    Advect a single particle using RK2 integration.
    
    Parameters
    ----------
    r_p : float
        Current particle radius
    v_grid : array_like
        Velocity on radial grid
    r_grid : array_like
        Radial grid points
    dt : float
        Time step
    behavior : int
        0: 'delete' (set active=False if out of bounds)
        1: 'clamp' (clamp to outer boundary)
        
    Returns
    -------
    r_new : float
        New particle radius
    v_new : float
        Velocity of particle for this step
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
            # Since we are clamping, if we are at the boundary, we stay there if flow is outward
            # But we still check velocity to see if it brings us back? 
            # Usually solar wind is outward.
    else:
        r_new = r_p
        
    r_p = r_new

    # RK2 Integration
    # 1. Evaluate v at current position
    v_p = np.interp(r_p, r_grid, v_grid)
    
    # 2. Half step
    r_mid = r_p + 0.5 * v_p * dt
    
    # Clamp intermediate position to grid for stability
    if r_mid < r_grid[0]: r_mid = r_grid[0]
    if r_mid > r_grid[-1]: r_mid = r_grid[-1]
    
    # 3. Evaluate v at mid point
    v_mid = np.interp(r_mid, r_grid, v_grid)
    
    # 4. Full step
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
def riemann_hllc(U_l, U_r, gamma):
    """
    HLLC (Harten-Lax-van Leer-Contact) Riemann solver.
    Good balance of accuracy and speed. Resolves contact discontinuities.
    
    Optimized: computes fluxes inline from already-known primitives
    to avoid redundant cons_to_prim conversions in get_flux.
    """
    # Left state
    rho_l = max(U_l[0], SMALL_RHO)
    v_l = U_l[1] / rho_l
    E_l = U_l[2]
    p_l = max((E_l - 0.5*rho_l*v_l**2) * (gamma - 1.0), SMALL_P)
    c_l = np.sqrt(gamma * p_l / rho_l)
    
    # Right state
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
        # All waves move right; use left flux (inline, no get_flux call)
        F[0] = rho_l * v_l
        F[1] = rho_l * v_l * v_l + p_l
        F[2] = (E_l + p_l) * v_l
        return F
    elif S_r <= 0:
        # All waves move left; use right flux (inline)
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
        
        # Inline left flux + HLL correction
        F_l_0 = rho_l * v_l
        F_l_1 = rho_l * v_l * v_l + p_l
        F_l_2 = (E_l + p_l) * v_l
        
        F[0] = F_l_0 + S_l * (U_star_0 - U_l[0])
        F[1] = F_l_1 + S_l * (U_star_1 - U_l[1])
        F[2] = F_l_2 + S_l * (U_star_2 - U_l[2])
        return F
    else:
        # Right star state
        factor = rho_r * (S_r - v_r) / (S_r - S_c)
        U_star_0 = factor
        U_star_1 = factor * S_c
        U_star_2 = factor * (E_r/rho_r + (S_c - v_r) * (S_c + p_r/(rho_r*(S_r - v_r))))
        
        # Inline right flux + HLL correction
        F_r_0 = rho_r * v_r
        F_r_1 = rho_r * v_r * v_r + p_r
        F_r_2 = (E_r + p_r) * v_r
        
        F[0] = F_r_0 + S_r * (U_star_0 - U_r[0])
        F[1] = F_r_1 + S_r * (U_star_1 - U_r[1])
        F[2] = F_r_2 + S_r * (U_star_2 - U_r[2])
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
    if i >= nr - 1:
        return U[i].copy(), U[i].copy()
    
    # Special handling for inner boundary (i=0) to improve conservation
    if i == 0:
        if nr < 3: # Fallback for tiny grids
            return U[0].copy(), U[1].copy()
            
        q_0 = cons_to_prim(U[0], gamma)
        q_1 = cons_to_prim(U[1], gamma)
        q_2 = cons_to_prim(U[2], gamma)
        
        # Check validity
        if q_0[0] <= 0 or q_0[2] <= 0 or q_1[0] <= 0 or q_1[2] <= 0 or q_2[0] <= 0 or q_2[2] <= 0:
             return U[0].copy(), U[1].copy()

        q_L = np.zeros(3)
        q_R = np.zeros(3)
        
        for n in range(3):
            # For cell 0 (Left state): Use forward difference slope as approximation
            # This is better than 0 slope (PCM) for smooth flows
            slope_0 = q_1[n] - q_0[n]
            # No limiter possible for cell 0 without ghost cells, but for smooth 
            # Parker wind this is safe. For robustness, we could limit to 0, 
            # but that causes the error you saw.
            
            # For cell 1 (Right state): We have full stencil 0,1,2
            slope_L = q_1[n] - q_0[n]
            slope_R = q_2[n] - q_1[n]
            dq_1 = mc_limiter(slope_L, slope_R)
            
            # Reconstruct
            # Interface is between 0 and 1
            # q_L comes from 0 projected to right
            # q_R comes from 1 projected to left
            q_L[n] = q_0[n] + 0.5 * slope_0
            q_R[n] = q_1[n] - 0.5 * dq_1
            
        # Ensure positivity
        q_L[0] = max(q_L[0], SMALL_RHO); q_L[2] = max(q_L[2], SMALL_P)
        q_R[0] = max(q_R[0], SMALL_RHO); q_R[2] = max(q_R[2], SMALL_P)
        
        return prim_to_cons(q_L, gamma), prim_to_cons(q_R, gamma)

    # Standard internal interfaces
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
    
    # Pre-compute q_ip2 once outside the component loop (was called 3x before)
    if i < nr - 2:
        q_ip2 = cons_to_prim(U[i+2], gamma)
    
    for n in range(3):
        slope_l = q_i[n] - q_im1[n]
        slope_r = q_ip1[n] - q_i[n]
        dq_l[n] = mc_limiter(slope_l, slope_r)
        
        if i < nr - 2:
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
    Compute fluxes at all interfaces using PCM (1st order).
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
        
        # Use HLLC Riemann solver
        fluxes[i] = riemann_hllc(U_l, U_r, gamma)
    
    return fluxes


@njit(cache=True)
def _compute_fluxes_plm(U, nr, gamma, riemann_type):
    """
    Compute fluxes at all interfaces using PLM reconstruction.
    
    Optimized: batch-converts ALL cells to primitives once upfront,
    then reuses across interfaces. Previously each cell was converted
    up to 4 times (as q_im1, q_i, q_ip1, q_ip2).
    """
    fluxes = np.zeros((nr + 1, 3))
    
    # Batch convert all cells to primitives ONCE
    q_all = np.zeros((nr, 3))  # [rho, v, p] for each cell
    q_valid = np.ones(nr, dtype=np.int32)  # Track validity
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
                U_face = prim_to_cons(q_face, gamma)
                U_l = U_face
                U_r = U_face
            else:
                U_l = U[nr-1]
                U_r = U[nr-1]
        else:
            # PLM reconstruction using pre-computed primitives
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
                        dq_1 = mc_limiter(slope_L, slope_R)
                        q_L[n] = q_all[0, n] + 0.5 * slope_0
                        q_R[n] = q_all[1, n] - 0.5 * dq_1
                    q_L[0] = max(q_L[0], SMALL_RHO); q_L[2] = max(q_L[2], SMALL_P)
                    q_R[0] = max(q_R[0], SMALL_RHO); q_R[2] = max(q_R[2], SMALL_P)
                    U_l = prim_to_cons(q_L, gamma)
                    U_r = prim_to_cons(q_R, gamma)
            else:
                # Standard internal - check validity
                if q_valid[idx-1] == 0 or q_valid[idx] == 0 or q_valid[idx+1] == 0:
                    U_l = U[idx].copy()
                    U_r = U[idx+1].copy()
                else:
                    # Left state slopes
                    dq_l = np.zeros(3)
                    dq_r = np.zeros(3)
                    for n in range(3):
                        slope_l = q_all[idx, n] - q_all[idx-1, n]
                        slope_r = q_all[idx+1, n] - q_all[idx, n]
                        dq_l[n] = mc_limiter(slope_l, slope_r)
                        
                        if idx < nr - 2:
                            slope_l2 = q_all[idx+1, n] - q_all[idx, n]
                            slope_r2 = q_all[idx+2, n] - q_all[idx+1, n]
                            dq_r[n] = mc_limiter(slope_l2, slope_r2)
                    
                    q_L = q_all[idx] + 0.5 * dq_l
                    q_R = q_all[idx+1] - 0.5 * dq_r
                    q_L[0] = max(q_L[0], SMALL_RHO); q_L[2] = max(q_L[2], SMALL_P)
                    q_R[0] = max(q_R[0], SMALL_RHO); q_R[2] = max(q_R[2], SMALL_P)
                    U_l = prim_to_cons(q_L, gamma)
                    U_r = prim_to_cons(q_R, gamma)
        
        fluxes[i] = riemann_hllc(U_l, U_r, gamma)
    
    return fluxes


@njit(cache=True)
def _compute_source(U, A, V, nr, gamma):
    """
    Compute geometric source terms for spherical coordinates.
    Optimized: inline pressure computation to avoid cons_to_prim overhead.
    """
    S = np.zeros((nr, 3))
    for i in range(nr):
        rho = max(U[i, 0], SMALL_RHO)
        v = U[i, 1] / rho
        eint = (U[i, 2] - 0.5 * rho * v * v) / rho
        p = max(rho * eint * (gamma - 1.0), SMALL_P)
        S[i, 1] = p * (A[i+1] - A[i]) / V[i]
    return S


@njit(cache=True)
def _get_dt(U, nr, dx, gamma, cfl):
    """Compute CFL-limited timestep. Uses precomputed min_dx when available."""
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
    
    min_dx = dx[0]
    for i in range(1, len(dx)):
        if dx[i] < min_dx:
            min_dx = dx[i]
    
    return cfl * min_dx / max_speed


@njit(cache=True)
def _get_dt_fast(U, nr, min_dx, gamma, cfl):
    """Compute CFL-limited timestep with precomputed min_dx (avoids recomputing each step)."""
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
def _extract_snapshot(U, nr, gamma, M_P_CGS_val, K_B_CGS_val):
    """
    JIT-compiled snapshot extraction: convert conserved to (v, rho, T) arrays.
    Avoids per-cell Python overhead in the main solve loop.
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
        T_out[i] = p * M_P_CGS_val / (rho * K_B_CGS_val)
    return v_out, rho_out, T_out


@njit(cache=True)
def _step_euler_jit(U, U_bc, nr, r, A, V, dt, gamma, riemann_type, use_plm):
    """
    JIT-compiled Euler step using HLLC Riemann solver.
    Optimized: computes source terms inline during update loop
    to avoid a separate cons_to_prim pass via _compute_source.
    """
    # Compute fluxes
    if use_plm:
        fluxes = _compute_fluxes_plm(U, nr, gamma, riemann_type)
    else:
        fluxes = _compute_fluxes_pcm(U, nr, gamma, riemann_type)
    
    # Update with area-weighted flux divergence PLUS inline geometric source term
    U_new = np.zeros_like(U)
    
    for i in range(nr):
        # Check for valid volume
        if V[i] <= 0:
            U_new[i] = U[i]
            continue
        
        # Inline source term: only momentum equation needs p * dA/V
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
    1D spherical compressible Euler solver using HLLC Riemann solver.
    
    Allows selection of different numerical methods:
    - Riemann solver: 'hllc' (HLLC - Harten-Lax-van Leer-Contact)
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
        'hllc': riemann_hllc,
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
            riemann: Riemann solver choice ('hllc' - only option)
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
        
        # Precompute min_dx for faster CFL computation (grid never changes)
        self.min_dx = np.min(self.dx)
        
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
        """
        Compute geometric source term for spherical coordinates.
        
        Only momentum equation needs source term: S = [0, p*dA/V, 0]
        
        The area-weighted flux formulation already handles mass and energy
        conservation correctly. The source term represents pressure work on
        the expanding spherical cross-section (lateral walls).
        """
        S = np.zeros((self.nr, 3))
        for i in range(self.nr):
            q = cons_to_prim(U[i], self.gamma)
            p = q[2]
            # Geometric pressure term: p * dA/V (ONLY for momentum equation)
            S[i, 1] = p * (self.A[i+1] - self.A[i]) / self.V[i]
        return S
    
    def step_euler(self, dt, U_bc):
        """Forward Euler time step."""
        # Set BC
        U = self.U.copy()
        
        # Fluxes
        fluxes = self.compute_fluxes(U)
        
        # Update with area-weighted flux divergence PLUS geometric source term
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
        riemann_type = 0  # Only HLLC is available
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
        """Compute CFL-limited timestep using JIT function with precomputed min_dx."""
        return _get_dt_fast(self.U, self.nr, self.min_dx, self.gamma, self.cfl)
    
    def solve(self, t_grid, v_bc_func, rho_bc_func, T_bc_func, 
              num_particles=0, particle_injection_rate=None, particle_release_rate=None):
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
        
        # Initialize using simple power law scaling 
        r0 = self.r[0]
        v_init = np.ones_like(self.r) * v0
        
        # Simple power law scaling: density ~ 1/r^2, temperature ~ r^(-2(γ-1))
        rho_init = np.zeros_like(self.r)
        T_init = np.zeros_like(self.r)
        alpha = 2 * (self.gamma - 1)  # Adiabatic temperature scaling exponent
        for i, r in enumerate(self.r):
            r_ratio = r0 / r
            rho_init[i] = rho0 * r_ratio**2
            T_init[i] = T0 * r_ratio**alpha
        
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
                rel_times = particle_release_rate[group_name] if particle_release_rate and group_name in particle_release_rate else inj_times
                
                # Determine boundary behavior
                # Default is delete (0), CMEs are clamped (1)
                behavior = 0
                if 'cme' in group_name.lower():
                    behavior = 1
                
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
                self.U, self.nr, self.gamma, M_P_CGS, K_B_CGS)
            v_out[0, :] = v_snap
            rho_out[0, :] = rho_snap
            T_out[0, :] = T_snap
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
                        behavior = group.get('behavior', 0)
                        for i in range(len(group['active'])):
                            if group['active'][i]:
                                # Check if particle is pinned to boundary
                                if self.time < group['release_times'][i]:
                                    r_new = self.r[0]
                                    v_new = v_curr[0]
                                    
                                    group['r'][i].append(r_new)
                                    group['v'][i].append(v_new)
                                    group['t'][i].append(self.time)
                                    continue
                                
                                r_p = group['r'][i][-1]
                                
                                # Use compiled function
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
                self.U, self.nr, self.gamma, M_P_CGS, K_B_CGS)
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
    
    print("\nRiemann Solver:")
    print("-" * 50)
    riemann_info = {
        'hllc': 'HLLC (Harten-Lax-van Leer-Contact) - fast, robust, captures contact waves',
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
    print("  'hllc-pcm'    -> HLLC + PCM + Euler (1st order)")
    print("  'hllc-plm-rk2'-> HLLC + PLM + RK2 (full 2nd order)")
    print()
    
    return {
        'riemann': list(riemann_info.keys()),
        'reconstruction': list(recon_info.keys()),
        'time_integrator': list(time_info.keys()),
    }
