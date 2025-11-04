"""
Compare HUXt compressible solvers against analytical Parker nozzle solution.

The Parker nozzle solution treats spherical expansion as flow through an expanding
nozzle where A(r) = r². This gives an analytical solution for the supersonic solar wind
assuming:
- Fixed gamma (adiabatic index)
- Steady-state isentropic flow
- No heat addition or viscosity
- Perfect gas equation of state

This is the gold standard test for compressible solar wind codes.
"""

import numpy as np
from astropy.constants import k_B, m_p
import astropy.units as u
from scipy.optimize import bisect
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import huxt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import huxt as H

# ==============================================================================
# Analytical Parker Nozzle Solution
# ==============================================================================

def compute_parker_nozzle_solution(r_grid, U0, n_rho0, T0, gamma=1.5):
    """
    Compute the analytical Parker nozzle solution for solar wind expansion.
    
    Args:
        r_grid: Radial grid in solar radii
        U0: Initial velocity (km/s)
        n_rho0: Initial number density (cm^-3)
        T0: Initial temperature (K)
        gamma: Adiabatic index
    
    Returns:
        U, n_rho, T, rho: Velocity, number density, temperature, mass density
    """
    # Convert to proper units
    U0 = U0 * (u.km / u.s)
    n_rho0 = n_rho0 / (u.cm ** 3)
    m_rho0 = (m_p * n_rho0).to(u.kg / u.m**3)
    T0 = T0 * u.K
    R_gas = k_B / m_p  # Specific gas constant
    
    # Compute Mach number at inner boundary
    M0 = (U0 / np.sqrt((gamma * R_gas * T0))).to(u.dimensionless_unscaled)
    
    # Compute stagnation (total) conditions
    T_t = T0 * (1 + ((gamma - 1)/2)*(M0*M0))
    p_t = (m_rho0 * R_gas * T0) * ((1 + ((gamma - 1)/2)*(M0*M0)) ** (gamma/(gamma - 1)))
    rho_t = m_rho0 * ((1 + ((gamma - 1)/2)*(M0*M0)) ** (1/(gamma - 1)))
    
    # Compute reference area at sonic point (A*)
    A0 = r_grid[0]**2  # Area at inner boundary
    
    def A_norm_calc(M, gamma):
        """Normalized area-Mach relation for quasi-1D nozzle flow"""
        a = 2 / (gamma + 1)
        b = (gamma - 1) / 2
        c = (gamma + 1) / (2*(gamma - 1))
        A_norm = (1/M) * (a*(1 + b*M*M))**c
        return A_norm
    
    A0_norm = A_norm_calc(M0.value, gamma)
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
    T = T_t / (1 + ((gamma - 1)/2)*(M*M))
    p = p_t * (T/T_t) ** (gamma/(gamma - 1))
    rho = rho_t * (T/T_t) ** (1/(gamma - 1))
    
    # Convert to output units
    rho = rho.to(u.kg / u.m**3)
    n_rho = (rho / m_p).to(1/u.cm**3)
    
    # Compute velocity from Mach number
    c = np.sqrt(gamma * R_gas * T)
    U = (M * c).to(u.km / u.s)
    
    return U, n_rho, T, rho


# ==============================================================================
# HUXt Solver Comparison
# ==============================================================================

def run_huxt_solver(solver_name, v0, rho0, T0, gamma, r_min=30, r_max=220, 
                    nr=200, simtime=5, dtscale=0.5):
    """
    Run a HUXt compressible solver with given initial conditions.
    
    Args:
        solver_name: 'upwind', 'hll', 'hll_fv', 'mol', or 'cgf'
        v0: Initial velocity (km/s)
        rho0: Initial density (kg/m^3)
        T0: Initial temperature (K)
        gamma: Adiabatic index
        r_min: Inner boundary (solar radii)
        r_max: Outer boundary (solar radii)
        nr: Number of radial grid points
        simtime: Simulation time (days)
        dtscale: Timestep safety factor
    
    Returns:
        model: HUXt model object with solution
    """
    # Create HUXt model
    model = H.HUXt(
        v_boundary=np.ones(128) * v0 * (u.km/u.s),
        r_min=r_min * u.solRad,
        r_max=r_max * u.solRad,
        simtime=simtime * u.day,
        dt_scale=dtscale,
        cr_num=np.nan,
        cr_lon_init=0 * u.deg,
        latitude=0 * u.deg,
        lon_out=0 * u.deg,
        frame='sidereal',
        rho_boundary=np.ones(128) * rho0 * (u.kg/u.m**3),
        temp_boundary=np.ones(128) * T0 * u.K,
        compressible=True,
        solver=solver_name
    )
    
    # Set gamma and alpha directly after initialization
    model.gamma = gamma
    model.alpha = 0.0  # No empirical acceleration
    model.model_params[10] = gamma  # Update model_params array
    
    # Run the model
    model.solve([])
    
    return model


# ==============================================================================
# Main Comparison
# ==============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("Comparing HUXt Solvers vs Analytical Parker Nozzle Solution")
    print("="*70)
    
    # Initial conditions matching the notebook
    U0 = 400  # km/s
    n_rho0 = 20  # cm^-3
    T0 = 1e6  # K
    gamma = 1.5
    
    # Convert number density to mass density
    rho0 = (n_rho0 / u.cm**3 * m_p).to(u.kg / u.m**3).value
    
    print(f"\nInitial conditions:")
    print(f"  Velocity: {U0} km/s")
    print(f"  Number density: {n_rho0} cm^-3")
    print(f"  Mass density: {rho0:.3e} kg/m^3")
    print(f"  Temperature: {T0:.2e} K")
    print(f"  Gamma: {gamma}")
    
    # Radial grid
    r_min = 30
    r_max = 220
    nr = 200
    r_grid = np.linspace(r_min, r_max, nr)
    
    # Compute analytical solution
    print("\nComputing analytical Parker nozzle solution...")
    U_ana, n_ana, T_ana, rho_ana = compute_parker_nozzle_solution(
        r_grid, U0, n_rho0, T0, gamma
    )
    
    # Run HUXt solvers
    solvers = ['upwind', 'hll', 'hll_fv']
    models = {}
    
    simtime = 5  # days - long enough to reach steady state
    
    for solver in solvers:
        print(f"\nRunning HUXt with '{solver}' solver...")
        models[solver] = run_huxt_solver(
            solver, U0, rho0, T0, gamma,
            r_min=r_min, r_max=r_max, nr=nr,
            simtime=simtime, dtscale=0.5
        )
    
    # ==============================================================================
    # Extract HUXt results at final time and central longitude
    # ==============================================================================
    
    results = {}
    for solver in solvers:
        model = models[solver]
        # Get radial profile at final time, central longitude (index 0 since lon_out=0)
        r_huxt = model.r.to(u.solRad).value
        v_huxt = model.v_grid[-1, :, 0].to(u.km/u.s).value
        rho_huxt = model.rho_grid[-1, :, 0].to(u.kg/u.m**3).value
        T_huxt = model.temp_grid[-1, :, 0].to(u.K).value
        n_huxt = (rho_huxt * u.kg/u.m**3 / m_p).to(1/u.cm**3).value
        
        results[solver] = {
            'r': r_huxt,
            'v': v_huxt,
            'rho': rho_huxt,
            'n': n_huxt,
            'T': T_huxt
        }
    
    # ==============================================================================
    # Compute errors relative to analytical solution
    # ==============================================================================
    
    print("\n" + "="*70)
    print("Errors vs Analytical Solution (RMS percentage error)")
    print("="*70)
    
    # Interpolate analytical solution onto HUXt grid for fair comparison
    r_huxt = results['upwind']['r']
    U_ana_interp = np.interp(r_huxt, r_grid, U_ana.value)
    n_ana_interp = np.interp(r_huxt, r_grid, n_ana.value)
    T_ana_interp = np.interp(r_huxt, r_grid, T_ana.value)
    
    for solver in solvers:
        v_err = np.sqrt(np.mean(((results[solver]['v'] - U_ana_interp) / U_ana_interp * 100)**2))
        n_err = np.sqrt(np.mean(((results[solver]['n'] - n_ana_interp) / n_ana_interp * 100)**2))
        T_err = np.sqrt(np.mean(((results[solver]['T'] - T_ana_interp) / T_ana_interp * 100)**2))
        
        print(f"\n{solver:15s}:")
        print(f"  Velocity:    {v_err:6.2f}%")
        print(f"  Density:     {n_err:6.2f}%")
        print(f"  Temperature: {T_err:6.2f}%")
    
    # ==============================================================================
    # Create comparison plots
    # ==============================================================================
    
    print("\n" + "="*70)
    print("Creating comparison plots...")
    print("="*70)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    colors = {
        'upwind': 'C0',
        'hll': 'C2',
        'hll_fv': 'C3'
    }
    
    # Left column: Absolute values
    ax = axes[0, 0]
    ax.plot(r_grid, U_ana.value, 'k-', linewidth=2, label='Analytical')
    for solver in solvers:
        ax.plot(results[solver]['r'], results[solver]['v'], 
                '--', color=colors[solver], label=solver, alpha=0.7)
    ax.set_ylabel('Velocity (km/s)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title('Solar Wind Profiles')
    
    ax = axes[1, 0]
    ax.plot(r_grid, n_ana.value, 'k-', linewidth=2, label='Analytical')
    for solver in solvers:
        ax.plot(results[solver]['r'], results[solver]['n'],
                '--', color=colors[solver], label=solver, alpha=0.7)
    ax.set_ylabel('Number Density (cm$^{-3}$)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 0]
    ax.plot(r_grid, T_ana.value/1e6, 'k-', linewidth=2, label='Analytical')
    for solver in solvers:
        ax.plot(results[solver]['r'], results[solver]['T']/1e6,
                '--', color=colors[solver], label=solver, alpha=0.7)
    ax.set_ylabel('Temperature (MK)')
    ax.set_xlabel('Radius (R$_\\odot$)')
    ax.grid(True, alpha=0.3)
    
    # Right column: Percentage errors
    ax = axes[0, 1]
    for solver in solvers:
        v_err = (results[solver]['v'] - U_ana_interp) / U_ana_interp * 100
        ax.plot(results[solver]['r'], v_err, color=colors[solver], label=solver)
    ax.axhline(0, color='k', linestyle='-', linewidth=1)
    ax.set_ylabel('Velocity Error (%)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title('Percentage Errors vs Analytical')
    
    ax = axes[1, 1]
    for solver in solvers:
        n_err = (results[solver]['n'] - n_ana_interp) / n_ana_interp * 100
        ax.plot(results[solver]['r'], n_err, color=colors[solver], label=solver)
    ax.axhline(0, color='k', linestyle='-', linewidth=1)
    ax.set_ylabel('Density Error (%)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    for solver in solvers:
        T_err = (results[solver]['T'] - T_ana_interp) / T_ana_interp * 100
        ax.plot(results[solver]['r'], T_err, color=colors[solver], label=solver)
    ax.axhline(0, color='k', linestyle='-', linewidth=1)
    ax.set_ylabel('Temperature Error (%)')
    ax.set_xlabel('Radius (R$_\\odot$)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    outfile = 'parker_nozzle_comparison.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {outfile}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("Comparison complete!")
    print("="*70)
