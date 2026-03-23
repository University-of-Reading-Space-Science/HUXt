"""
Compare HUXt numerical solutions against analytical Parker nozzle solution.

This script runs HUXt with constant boundary conditions to steady state
and compares the results against the Parker nozzle analytical solution.

Boundary conditions:
- v = 400 km/s at inner boundary (0.1 AU)
- T = 1 MK at inner boundary  
- n = 600 cm^-3 at inner boundary
- gamma = 1.5 (polytropic index)

Run with: python test_huxt_vs_parker.py
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import time
import os
import huxt.huxt as H

# Physical constants
Rs_cm = 6.957e10  # Solar radius in cm
M_P_CGS = 1.67262192e-24  # Proton mass in g (correct CGS value)
K_B_CGS = 1.380649e-16  # Boltzmann constant in erg/K


def compute_parker_solution(r_cm, v0_cms, n0_cm3, T0_K, gamma=1.5):
    """
    Compute analytical Parker nozzle solution for supersonic solar wind.
    
    Args:
        r_cm: Radial grid in cm
        v0_cms: Velocity at inner boundary (cm/s)
        n0_cm3: Number density at inner boundary (cm^-3)
        T0_K: Temperature at inner boundary (K)
        gamma: Adiabatic index
        
    Returns:
        v, n, T: Arrays of velocity (cm/s), density (cm^-3), temperature (K)
    """
    # Initial conditions
    rho0 = n0_cm3 * M_P_CGS  # g/cm^3
    p0 = n0_cm3 * K_B_CGS * T0_K  # dyne/cm^2
    c0 = np.sqrt(gamma * p0 / rho0)  # Sound speed
    M0 = v0_cms / c0  # Mach number
    
    # Stagnation (total) conditions
    T_t = T0_K * (1 + (gamma - 1)/2 * M0**2)
    p_t = p0 * (1 + (gamma - 1)/2 * M0**2)**(gamma/(gamma-1))
    rho_t = rho0 * (1 + (gamma - 1)/2 * M0**2)**(1/(gamma-1))
    
    # Area-Mach relation for quasi-1D nozzle flow
    def area_mach(M):
        """Normalized area A/A* as function of Mach number"""
        a = 2 / (gamma + 1)
        b = (gamma - 1) / 2
        c = (gamma + 1) / (2 * (gamma - 1))
        return (1/M) * (a * (1 + b * M**2))**c
    
    # Reference area
    r0 = r_cm[0]
    A0 = r0**2
    A0_norm = area_mach(M0)
    A_star = A0 / A0_norm
    
    # Solve for Mach number at each radius (simple iterative method)
    M = np.zeros(len(r_cm))
    for i in range(len(r_cm)):
        A = r_cm[i]**2
        A_norm = A / A_star
        
        # Newton iteration for supersonic branch
        M_guess = M0 * (r_cm[i] / r0)**0.5  # Initial guess
        for _ in range(50):
            f = area_mach(M_guess) - A_norm
            # Numerical derivative
            dM = 1e-6
            df = (area_mach(M_guess + dM) - area_mach(M_guess - dM)) / (2*dM)
            M_new = M_guess - f / df
            if abs(M_new - M_guess) < 1e-8:
                break
            M_guess = M_new
        M[i] = M_guess
    
    # Isentropic relations
    T = T_t / (1 + (gamma - 1)/2 * M**2)
    p = p_t * (T / T_t)**(gamma / (gamma - 1))
    rho = rho_t * (T / T_t)**(1 / (gamma - 1))
    
    # Velocity and density
    c = np.sqrt(gamma * K_B_CGS * T / M_P_CGS)
    v = M * c
    n = rho / M_P_CGS
    
    return v, n, T


def main():
    print("="*70)
    print("HUXt vs Parker Nozzle Analytical Solution")
    print("="*70)
    
    # Boundary conditions
    v_bc_kms = 400  # km/s
    T_bc = 1e6  # K
    n_bc = 600  # cm^-3
    gamma = 1.5
    simtime = 10 * u.day  # Run to steady state
    
    print(f"\nBoundary conditions:")
    print(f"  v = {v_bc_kms} km/s at 0.1 AU")
    print(f"  T = {T_bc/1e6:.1f} MK")
    print(f"  n = {n_bc} cm^-3")
    print(f"  gamma = {gamma}")
    print(f"  Simulation time: {simtime}")
    
    # Set up HUXt boundary conditions
    mp = 1.6726219e-27 * u.kg  # Proton mass
    vr_in = v_bc_kms * np.ones(128) * u.km/u.s
    rho_in = n_bc * np.ones(128)  * u.cm**-3
    rho_in = rho_in.to(u.m**-3) * mp * 1000  # Convert to kg/m^3
    T_in = T_bc * np.ones(128) * u.K
    
    # Run models with different solvers
    methods = [
        ('hllc-pcm', 'SURF-hydro (PCM)'),
        ('hllc-plm-rk2', 'SURF-hydro (PLM)'),
    ]
    
    results = {}
    for method_key, method_name in methods:
        print(f"\nRunning {method_name}...", end=" ", flush=True)
        t0 = time.time()
        
        model = H.HUXt(
            v_boundary=vr_in, 
            rho_boundary=rho_in, 
            temp_boundary=T_in,  
            simtime=simtime, 
            dt_scale=4, 
            r_max=215*u.solRad, 
            r_min=21.5*u.solRad,
            solver=method_key, 
            lon_out=0.0*u.rad
        )
        model.solve([])
        
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        
        # Extract final steady-state solution
        v_final = model.v_grid[-1, :, 0]  # Final time, all radii, first longitude
        rho_final = model.rho_grid[-1, :, 0]
        temp_final = model.temp_grid[-1, :, 0]
        r_grid = model.r
        
        results[method_key] = {
            'name': method_name,
            'v': v_final,
            'rho': rho_final,
            'T': temp_final,
            'r': r_grid,
            'time': elapsed
        }
    
    # Compute Parker analytical solution
    print("\nComputing Parker nozzle solution...", end=" ", flush=True)
    r_au = results['hllc-pcm']['r'].to(u.AU).value
    r_cm = results['hllc-pcm']['r'].to(u.cm).value
    r_Rs = r_cm / Rs_cm
    
    v_bc_cms = v_bc_kms * 1e5  # km/s to cm/s
    v_parker, n_parker, T_parker = compute_parker_solution(r_cm, v_bc_cms, n_bc, T_bc, gamma)
    print("done")
    
    # Print comparison at 1 AU
    idx_1AU = np.argmin(np.abs(r_au - 1.0))
    print(f"\nComparison at 1 AU:")
    print(f"{'Method':<20} {'v (km/s)':<12} {'n (cm^-3)':<12} {'T (kK)':<12}")
    print("-"*60)
    print(f"{'Parker nozzle':<20} {v_parker[idx_1AU]/1e5:<12.1f} {n_parker[idx_1AU]:<12.1f} {T_parker[idx_1AU]/1e3:<12.1f}")
    
    for method_key, method_name in methods:
        res = results[method_key]
        v_1au = res['v'][idx_1AU].to(u.km/u.s).value
        n_1au = (res['rho'][idx_1AU] / (mp * 1000)).to(u.cm**-3).value
        T_1au = res['T'][idx_1AU].value
        print(f"{method_name:<20} {v_1au:<12.1f} {n_1au:<12.1f} {T_1au/1e3:<12.1f}")
    
    # Compute RMS errors
    print(f"\nRMS errors vs Parker nozzle:")
    print(f"{'Method':<20} {'v (%)':<12} {'n (%)':<12} {'T (%)':<12}")
    print("-"*60)
    
    for method_key, method_name in methods:
        res = results[method_key]
        v_num = res['v'].to(u.cm/u.s).value
        n_num = (res['rho'] / (mp * 1000)).to(u.cm**-3).value
        T_num = res['T'].value
        
        err_v = np.sqrt(np.mean(((v_num - v_parker) / v_parker)**2)) * 100
        err_n = np.sqrt(np.mean(((n_num - n_parker) / n_parker)**2)) * 100
        err_T = np.sqrt(np.mean(((T_num - T_parker) / T_parker)**2)) * 100
        
        print(f"{method_name:<20} {err_v:<12.2f} {err_n:<12.2f} {err_T:<12.2f}")
    
    # Plotting
    plt.rcParams['font.size'] = 16
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ["#0396ff", "#fd0707"]
    
    # Velocity
    ax = axes[0, 0]
    ax.plot(r_Rs/215, v_parker/1e5, 'k-', linewidth=2.5, label='Parker nozzle')
    for i, (method_key, method_name) in enumerate(methods):
        res = results[method_key]
        v = res['v'].to(u.km/u.s).value
        ax.plot(r_Rs/215, v, '--', color=colors[i], alpha=0.8, linewidth=3, label=method_name)
    ax.set_ylabel(r'Radial velocity, $V$ [km s$^{-1}$]', fontsize=18)
    ax.tick_params(labelbottom=False)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=18, fontweight='bold',
            verticalalignment='top')
    
    # Density
    ax = axes[0, 1]
    ax.semilogy(r_Rs/215, n_parker, 'k-', linewidth=2.5, label='Parker nozzle')
    for i, (method_key, method_name) in enumerate(methods):
        res = results[method_key]
        n = (res['rho'] / (mp * 1000)).to(u.cm**-3).value
        ax.semilogy(r_Rs/215, n, '--', color=colors[i], alpha=0.8, linewidth=3, label=method_name)
    ax.set_ylabel(r'Number density, $n$ [cm$^{-3}$]', fontsize=18)
    ax.set_ylim(n_parker.min() * 0.8, n_parker.max() * 1.8)
    ax.tick_params(labelbottom=False)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=18, fontweight='bold',
            verticalalignment='top')
    
    # Temperature
    ax = axes[1, 0]
    ax.semilogy(r_Rs/215, T_parker, 'k-', linewidth=2.5, label='Parker nozzle')
    for i, (method_key, method_name) in enumerate(methods):
        res = results[method_key]
        T = res['T'].value
        ax.semilogy(r_Rs/215, T, '--', color=colors[i], alpha=0.8, linewidth=3, label=method_name)
    ax.set_xlabel('Distance [AU]', fontsize=18)
    ax.set_ylabel(r'Temperature, $T$ [K]', fontsize=18)
    ax.set_ylim(T_parker.min() * 0.8, T_parker.max() * 1.8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16)
    ax.text(0.02, 0.98, '(c)', transform=ax.transAxes, fontsize=18, fontweight='bold',
            verticalalignment='top')
    
    # Mass flux conservation
    ax = axes[1, 1]
    for i, (method_key, method_name) in enumerate(methods):
        res = results[method_key]
        rho_cgs = res['rho'].to(u.g/u.cm**3).value
        v_cgs = res['v'].to(u.cm/u.s).value
        r_cgs = res['r'].to(u.cm).value
        mass_flux = rho_cgs * v_cgs * r_cgs**2
        mass_flux_norm = mass_flux / mass_flux[0]
        ax.plot(r_Rs/215, mass_flux_norm, '-', color=colors[i], alpha=0.8, linewidth=3, label=method_name)
    ax.axhline(1.0, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Perfect conservation')
    ax.set_xlabel('Distance [AU]', fontsize=18)
    ax.set_ylabel('Normalized mass flux', fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.text(0.02, 0.98, '(d)', transform=ax.transAxes, fontsize=18, fontweight='bold',
            verticalalignment='top')
    
    plt.tight_layout()
    
    # Save figure
    dbox = os.getenv('DBOX')
    if dbox:
        save_dir = os.path.join(dbox, 'Apps', 'Overleaf', 'SHUXt')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'huxt_vs_parker.pdf')
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    print("\nDone!")


if __name__ == '__main__':
    main()
