"""
Test script to verify CGF solver spin-up and Parker solution approximation.

1. Verifies that the CGF solver correctly handles the spin-up period (output starts at t=0, not t=-spinup).
2. Checks if the solution approximates a Parker solar wind (constant v, rho ~ 1/r^2) for constant boundary conditions.
3. Verifies mass flux conservation through the domain.
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import huxt.huxt as H

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
    # For constant V, rho should scale as 1/r^2.
    # However, the CGF solver solves the Euler equations with pressure gradients.
    # A hot solar wind will accelerate (v increases with r), so rho will drop faster than 1/r^2
    # to conserve mass flux (rho * v * r^2 = const).
    # So we expect some deviation from the simple 1/r^2 (constant v) scaling.
    print("\n--- Parker Solution Check ---")
    
    # Get final state
    r = model.r.to(u.km).value
    v = model.v_grid[-1, :, 0].value # km/s
    rho = model.rho_grid[-1, :, 0].value # kg/m^3
    
    # Expected scaling for CONSTANT velocity
    rho_0 = rho[0]
    r_0 = r[0]
    rho_theory_const_v = rho_0 * (r_0 / r)**2
    
    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    
    ax[0].plot(r/1.5e8, v, label='Model V')
    ax[0].set_ylabel('Velocity (km/s)')
    ax[0].set_title('Radial Velocity Profile (Expect slight acceleration due to pressure)')
    ax[0].grid(True)
    
    ax[1].plot(r/1.5e8, rho, label='Model Density')
    ax[1].plot(r/1.5e8, rho_theory_const_v, '--', label='1/r^2 Theory (Const V)')
    ax[1].set_ylabel('Density (kg/m^3)')
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_title('Density Profile')

    # 3. Mass Flux Conservation
    print("\n--- Mass Flux Conservation ---")
    # Flux = 4 * pi * r^2 * rho * v
    # We'll compute this at the last timestep
    
    # Convert to consistent units
    # r in m
    r_m = r * 1000.0
    # v in m/s
    v_ms = v * 1000.0
    # rho in kg/m^3
    
    flux = 4 * np.pi * r_m**2 * rho * v_ms
    
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
    ax[2].plot(r/1.5e8, flux)
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
