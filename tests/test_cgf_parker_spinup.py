"""
Test script to verify CGF solver spin-up and Parker solution approximation.

1. Verifies that the CGF solver correctly handles the spin-up period (output starts at t=0, not t=-spinup).
2. Checks if the solution approximates a Parker solar wind (constant v, rho ~ 1/r^2) for constant boundary conditions.
3. Verifies mass flux conservation through the domain.
"""

import numpy as np
import astropy.units as u
from astropy.constants import k_B, m_p
from scipy.optimize import bisect
import matplotlib.pyplot as plt
import huxt.huxt as H

# ==============================================================================
# Analytical Parker Nozzle Solution
# ==============================================================================

def compute_parker_nozzle_solution(r_grid, U0, n_rho0, T0, gamma=1.5):
    """
    Compute the analytical Parker nozzle solution for solar wind expansion.
    
    Args:
        r_grid: Radial grid in solar radii (or consistent units if careful)
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
        try:
            M[i] = bisect(invert_A_for_M, m_min, m_max, args=(gamma, a_n))
        except ValueError:
            # Fallback if bisect fails (e.g. very close to sonic point or numerical issues)
            # For solar wind, we are usually well supersonic.
            M[i] = np.nan

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
    
    return U.value, n_rho.value, T.value, rho.value

def test_cgf_parker_spinup():
    # Setup parameters
    # 1.5 day spinup is default in HUXt if not specified, but let's be explicit if possible.
    # HUXt calculates spinup internally based on min velocity and domain size.
    
    simtime = 5 * u.day
    dt_scale = 4
    cr = 2210
    
    # Constant boundary conditions
    v_boundary = np.ones(128) * 400 * (u.km/u.s)
    
    print("Initializing HUXt with CGF solver...")
    # lon_out=0.0*u.rad makes it effectively 1D for the output
    model = H.HUXt(v_boundary=v_boundary, simtime=simtime, dt_scale=dt_scale, 
                   solver='cgf', frame='sidereal', lon_out=0.0*u.rad)
    
    print(f"Model time range: {model.time_out.min()} to {model.time_out.max()}")
    print(f"Spin-up time included in initialization: {model.time_init}")
    
    # Solve with no CMEs
    print("Solving...")
    model.solve([])
    
    # 1. Verify Spin-up
    # The output time_out should start at roughly 0 (or whatever simtime start is)
    # and NOT include the negative spin-up time.
    print("\n--- Spin-up Verification ---")
    print(f"First output time: {model.time_out[0].to(u.day)}")
    if model.time_out[0] >= 0 * u.s:
        print("PASS: Output starts at or after t=0 (spin-up excluded from output).")
    else:
        print(f"FAIL: Output starts at {model.time_out[0]} (spin-up included?).")

    # 2. Parker Solution Approximation
    print("\n--- Parker Solution Check ---")
    
    # Get final state from model
    r_km = model.r.to(u.km).value
    v_model = model.v_grid[-1, :, 0].value # km/s
    rho_model = model.rho_grid[-1, :, 0].value # kg/m^3
    T_model = model.temp_grid[-1, :, 0].value # K
    
    # Initial conditions at inner boundary
    v0 = v_model[0]
    rho0 = rho_model[0]
    T0 = T_model[0]
    
    # Convert rho0 to n_rho0 (cm^-3) for the analytical function
    m_p_kg = 1.6726219e-27
    n_rho0 = (rho0 / m_p_kg) / 1e6 # m^-3 -> cm^-3
    
    # Compute analytical Parker solution
    # Note: r_grid needs to be in consistent units. The function uses r_grid[0] to define A0.
    # It treats r as just a coordinate, so units cancel in A/A* ratio as long as consistent.
    # But we pass r_km.
    
    # HUXt uses gamma = 1.5 by default for polytropic solar wind?
    # Let's check what gamma HUXt uses.
    # HUXt defaults to gamma=1.5 in huxt.py constants or init?
    # Actually, HUXt usually sets polytropic index alpha.
    # For CGF solver, we pass gamma. Let's assume gamma=1.5 (standard for solar wind models in HUXt context often).
    # Wait, CGF solver default gamma is 5/3 (1.667) in cgf_solver.py __init__.
    # But HUXt might override it.
    # Let's check model.gamma
    gamma = model.gamma
    print(f"Using gamma = {gamma}")

    v_parker, n_rho_parker, T_parker, rho_parker = compute_parker_nozzle_solution(
        r_km, v0, n_rho0, T0, gamma=gamma
    )
    
    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    
    # Velocity
    ax[0].plot(r_km/1.5e8, v_model, 'b-', label='CGF Model')
    ax[0].plot(r_km/1.5e8, v_parker, 'r--', label='Analytical Parker')
    ax[0].set_ylabel('Velocity (km/s)')
    ax[0].set_title('Radial Velocity Profile')
    ax[0].legend()
    ax[0].grid(True)
    
    # Density
    ax[1].plot(r_km/1.5e8, rho_model, 'b-', label='CGF Model')
    ax[1].plot(r_km/1.5e8, rho_parker, 'r--', label='Analytical Parker')
    ax[1].set_ylabel('Density (kg/m^3)')
    ax[1].set_xlabel('Radius (AU)')
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_title('Density Profile')
    
    # Temperature
    ax[2].plot(r_km/1.5e8, T_model, 'b-', label='CGF Model')
    ax[2].plot(r_km/1.5e8, T_parker, 'r--', label='Analytical Parker')
    ax[2].set_ylabel('Temperature (K)')
    ax[2].set_xlabel('Radius (AU)')
    ax[2].legend()
    ax[2].grid(True)
    ax[2].set_title('Temperature Profile')

    # Calculate error
    # Ignore NaNs in Parker solution (if any)
    valid = ~np.isnan(v_parker)
    v_err = np.mean(np.abs(v_model[valid] - v_parker[valid]) / v_parker[valid]) * 100
    rho_err = np.mean(np.abs(rho_model[valid] - rho_parker[valid]) / rho_parker[valid]) * 100
    
    print(f"Mean Velocity Error: {v_err:.2f}%")
    print(f"Mean Density Error: {rho_err:.2f}%")

    # 3. Mass Flux Conservation
    print("\n--- Mass Flux Conservation ---")
    # Flux = 4 * pi * r^2 * rho * v
    # We'll compute this at the last timestep
    
    # Convert to consistent units
    # r in m
    r_m = r_km * 1000.0
    # v in m/s
    v_ms = v_model * 1000.0
    # rho in kg/m^3
    
    flux = 4 * np.pi * r_m**2 * rho_model * v_ms
    
    mean_flux = np.mean(flux)
    std_flux = np.std(flux)
    max_dev = np.max(np.abs(flux - mean_flux)) / mean_flux * 100
    
    print(f"Mean Mass Flux: {mean_flux:.4e} kg/s")
    print(f"Std Dev: {std_flux:.4e} kg/s")
    print(f"Max Deviation from Mean: {max_dev:.4f}%")
    
    if max_dev < 3.0: # Allow some deviation due to discretization/settling
        print("PASS: Mass flux conserved to within 3%.")
    else:
        print("WARNING: Mass flux deviation > 3%.")

    # Plot Mass Flux
    ax[2].plot(r_km/1.5e8, flux)
    ax[2].axhline(mean_flux, color='r', linestyle='--', label='Mean')
    ax[2].set_ylabel('Mass Flux (kg/s)')
    ax[2].set_xlabel('Radius (AU)')
    ax[2].set_title(f'Mass Flux vs Radius (Max Dev: {max_dev:.2f}%)')
    ax[2].legend()
    ax[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('parker_check.png')
    print("Saved parker_check.png")

if __name__ == "__main__":
    test_cgf_parker_spinup()
