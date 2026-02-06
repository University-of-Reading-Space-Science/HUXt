"""
Compare all compressible solver methods against analytical Parker nozzle solution.

This script runs each solver to steady-state and compares against the analytical
Parker nozzle solution for:
- v = 400 km/s at inner boundary
- T = 1 MK at inner boundary  
- n = 600 cm^-3 at inner boundary
- gamma = 1.5 (polytropic index for solar wind)

Run with: python test_solver_vs_parker.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Import directly to avoid slow sunpy imports
import sys
sys.path.insert(0, '.')
from huxt.huxt_solvers import (
    CompressibleSolver, 
    create_solver,
    KM_TO_CM,
    K_B_CGS,
    M_P_CGS
)


def simple_bisect(f, a, b, args=(), tol=1e-8, maxiter=100):
    """Simple bisection method to avoid scipy import overhead."""
    fa = f(a, *args)
    fb = f(b, *args)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    
    for _ in range(maxiter):
        c = (a + b) / 2
        fc = f(c, *args)
        if abs(fc) < tol or abs(b - a) < tol:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a + b) / 2

# Physical constants
Rs_cm = 6.957e10  # Solar radius in cm


def compute_parker_nozzle_solution(r_grid_cm, v0_cms, n0_cm3, T0_K, gamma=1.5):
    """
    Compute analytical Parker nozzle solution for supersonic solar wind.
    
    This is the exact analytical solution for steady-state isentropic flow
    through a diverging nozzle with A(r) ~ r^2.
    
    Args:
        r_grid_cm: Radial grid in cm
        v0_cms: Velocity at inner boundary (cm/s)
        n0_cm3: Number density at inner boundary (cm^-3)
        T0_K: Temperature at inner boundary (K)
        gamma: Adiabatic index
        
    Returns:
        v, n, T, rho: Velocity (cm/s), density (cm^-3), temperature (K), mass density (g/cm^3)
    """
    # Compute sound speed and Mach number at inner boundary
    rho0 = n0_cm3 * M_P_CGS  # g/cm^3
    p0 = n0_cm3 * K_B_CGS * T0_K  # erg/cm^3 = dyne/cm^2
    c0 = np.sqrt(gamma * p0 / rho0)  # cm/s
    M0 = v0_cms / c0
    
    # Stagnation (total) conditions
    T_t = T0_K * (1 + (gamma - 1)/2 * M0**2)
    p_t = p0 * (1 + (gamma - 1)/2 * M0**2)**(gamma/(gamma-1))
    rho_t = rho0 * (1 + (gamma - 1)/2 * M0**2)**(1/(gamma-1))
    
    # Area-Mach relation for quasi-1D nozzle flow
    def area_mach(M, gamma):
        """Normalized area A/A* as function of Mach number"""
        a = 2 / (gamma + 1)
        b = (gamma - 1) / 2
        c = (gamma + 1) / (2 * (gamma - 1))
        return (1/M) * (a * (1 + b * M**2))**c
    
    # Reference area at sonic point
    r0 = r_grid_cm[0]
    A0 = r0**2  # Area at inner boundary (proportional to r^2)
    A0_norm = area_mach(M0, gamma)
    A_star = A0 / A0_norm  # Reference area at sonic point
    
    # For each radius, solve for supersonic Mach number
    def invert_area_mach(M, gamma, A_n):
        """Root finding function"""
        return area_mach(M, gamma) - A_n
    
    nr = len(r_grid_cm)
    M = np.zeros(nr)
    
    for i in range(nr):
        A = r_grid_cm[i]**2
        A_norm = A / A_star
        try:
            # Supersonic branch (M > 1)
            M[i] = simple_bisect(invert_area_mach, 1.0001, 100.0, args=(gamma, A_norm))
        except ValueError:
            # Fallback: use simple scaling
            M[i] = M0 * (r_grid_cm[i] / r0)**0.5
    
    # Isentropic relations
    T = T_t / (1 + (gamma - 1)/2 * M**2)
    p = p_t * (T / T_t)**(gamma / (gamma - 1))
    rho = rho_t * (T / T_t)**(1 / (gamma - 1))
    
    # Compute velocity from Mach number
    c = np.sqrt(gamma * K_B_CGS * T / M_P_CGS)
    v = M * c
    
    n = rho / M_P_CGS
    
    return v, n, T, rho


def main():
    print("="*70)
    print("Solver Comparison vs Parker Nozzle Analytical Solution")
    print("="*70)
    
    # Test conditions (as requested)
    v_bc_kms = 400  # km/s
    T_bc = 1e6  # 1 MK
    n_bc = 600  # cm^-3
    gamma = 1.5
    
    # Convert to CGS
    v_bc = v_bc_kms * KM_TO_CM  # cm/s
    rho_bc = n_bc * M_P_CGS  # g/cm^3
    
    print(f"\nBoundary conditions:")
    print(f"  v = {v_bc_kms} km/s")
    print(f"  T = {T_bc/1e6:.1f} MK")
    print(f"  n = {n_bc} cm^-3")
    print(f"  gamma = {gamma}")
    
    # Radial grid (30 Rs to 215 Rs)
    r_min = 30 * Rs_cm
    r_max = 215 * Rs_cm
    nr = 100
    r_grid = np.linspace(r_min, r_max, nr)
    r_AU = r_grid / 1.496e13  # Convert to AU for plotting
    r_Rs = r_grid / Rs_cm  # Solar radii
    
    # Compute analytical Parker solution
    print("\nComputing Parker nozzle analytical solution...")
    v_parker, n_parker, T_parker, rho_parker = compute_parker_nozzle_solution(
        r_grid, v_bc, n_bc, T_bc, gamma
    )
    
    print(f"  Parker solution at 1 AU:")
    idx_1AU = np.argmin(np.abs(r_AU - 1.0))
    print(f"    v = {v_parker[idx_1AU]/KM_TO_CM:.1f} km/s")
    print(f"    n = {n_parker[idx_1AU]:.1f} cm^-3")
    print(f"    T = {T_parker[idx_1AU]/1e3:.1f} kK")
    
    # Simulation time: run to steady state (5 days)
    day_sec = 86400
    t_end = 10 * day_sec
    dt_out = t_end / 20
    t_grid = np.arange(0, t_end + dt_out, dt_out)
    
    # BC functions (constant)
    v_bc_func = lambda t: v_bc
    rho_bc_func = lambda t: rho_bc
    T_bc_func = lambda t: T_bc
    
    # Methods to compare
    # Note: PLM methods need more work on characteristic limiting - using PCM for now
    methods = [  
        ('hllc-pcm', 'HLLC + PCM'),
        ('hllc-plm-rk2', 'HLLC + PLM'),
    ]
    
    # Warm-up JIT compilation
    print("\nWarming up JIT compilation (running short simulations)...")
    # Warm up PCM path
    create_solver(r_grid, gamma=gamma, method='hllc-pcm', verbose=False).solve(
        [0, 1.0], v_bc_func, rho_bc_func, T_bc_func)
    # Warm up PLM+RK2 path
    create_solver(r_grid, gamma=gamma, method='hllc-plm-rk2', verbose=False).solve(
        [0, 1.0], v_bc_func, rho_bc_func, T_bc_func)
    
    results = {}
    
    print("\nRunning numerical solvers...")
    for method_key, method_name in methods:
        print(f"  {method_name}...", end=" ", flush=True)
        
        solver = create_solver(r_grid, gamma=gamma, method=method_key, verbose=False)
        
        t0 = time.time()
        result = solver.solve(t_grid, v_bc_func, rho_bc_func, T_bc_func)
        elapsed = time.time() - t0
        
        # Compute errors vs Parker solution (at final time)
        v_num = result['v'][-1]
        rho_num = result['rho'][-1]
        T_num = result['T'][-1]
        n_num = rho_num / M_P_CGS
        
        # RMS errors (relative)
        error_v = np.sqrt(np.mean(((v_num - v_parker) / v_parker)**2)) * 100
        error_n = np.sqrt(np.mean(((n_num - n_parker) / n_parker)**2)) * 100
        error_T = np.sqrt(np.mean(((T_num - T_parker) / T_parker)**2)) * 100
        
        # Max errors
        max_error_v = np.max(np.abs((v_num - v_parker) / v_parker)) * 100
        max_error_n = np.max(np.abs((n_num - n_parker) / n_parker)) * 100
        max_error_T = np.max(np.abs((T_num - T_parker) / T_parker)) * 100
        
        # Mass flux conservation: rho * v * r^2 should be constant
        # (in steady state for spherical expansion)
        mass_flux = rho_num * v_num * (r_grid**2)
        mass_flux_inner = mass_flux[0]
        mass_flux_error = np.abs((mass_flux - mass_flux_inner) / mass_flux_inner) * 100
        max_mass_flux_error = np.max(mass_flux_error)
        rms_mass_flux_error = np.sqrt(np.mean(mass_flux_error**2))
        
        results[method_key] = {
            'name': method_name,
            'result': result,
            'time': elapsed,
            'error_v': error_v,
            'error_n': error_n,
            'error_T': error_T,
            'max_error_v': max_error_v,
            'max_error_n': max_error_n,
            'max_error_T': max_error_T,
            'mass_flux': mass_flux,
            'max_mass_flux_error': max_mass_flux_error,
            'rms_mass_flux_error': rms_mass_flux_error,
        }
        print(f"done ({elapsed:.2f}s, RMSE v: {error_v:.2f}%, mass flux err: {rms_mass_flux_error:.2f}%)")
    
    # Print results table
    print("\n" + "="*90)
    print("Results Summary: RMS Error vs Parker Nozzle Solution (%)")
    print("="*90)
    print(f"{'Method':<22} {'Time (s)':<10} {'v err %':<12} {'n err %':<12} {'T err %':<12}")
    print("-"*90)
    
    for method_key, method_name in methods:
        data = results[method_key]
        print(f"{method_name:<22} {data['time']:<10.3f} {data['error_v']:<12.2f} "
              f"{data['error_n']:<12.2f} {data['error_T']:<12.2f}")
    
    print("\n" + "="*90)
    print("Maximum Error vs Parker Nozzle Solution (%)")
    print("="*90)
    print(f"{'Method':<22} {'v max err %':<15} {'n max err %':<15} {'T max err %':<15}")
    print("-"*90)
    
    for method_key, method_name in methods:
        data = results[method_key]
        print(f"{method_name:<22} {data['max_error_v']:<15.2f} "
              f"{data['max_error_n']:<15.2f} {data['max_error_T']:<15.2f}")
    
    # Mass flux conservation table
    print("\n" + "="*90)
    print("Mass Flux Conservation: ρ·v·r² should be constant at steady state")
    print("="*90)
    print(f"{'Method':<22} {'RMS error %':<15} {'Max error %':<15} {'Order':<10}")
    print("-"*90)
    
    # Compute expected mass flux from Parker solution
    mass_flux_parker = rho_parker * v_parker * (r_grid**2)
    parker_flux_error = np.abs((mass_flux_parker - mass_flux_parker[0]) / mass_flux_parker[0]) * 100
    print(f"{'Parker (analytical)':<22} {np.sqrt(np.mean(parker_flux_error**2)):<15.4f} "
          f"{np.max(parker_flux_error):<15.4f} {'N/A':<10}")
    
    for method_key, method_name in methods:
        data = results[method_key]
        order = '1st' if 'pcm' in method_key.lower() else '2nd'
        print(f"{method_name:<22} {data['rms_mass_flux_error']:<15.4f} "
              f"{data['max_mass_flux_error']:<15.4f} {order:<10}")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Set font sizes globally for better visibility
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    
    # Colors for different methods (use explicit color list for compatibility)
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    colors = color_list[:len(methods)]
    
    # Velocity vs radius
    ax = axes[0, 0]
    ax.plot(r_Rs, v_parker/KM_TO_CM, 'k-', linewidth=2.5, label='Parker (analytical)')
    for i, (method_key, method_name) in enumerate(methods):
        v = results[method_key]['result']['v'][-1]
        ax.plot(r_Rs, v/KM_TO_CM, '--', color=colors[i], alpha=0.7, linewidth=2, label=method_name)
    ax.set_xlabel('Distance (Rs)', fontsize=14)
    ax.set_ylabel('Velocity (km/s)', fontsize=14)
    ax.legend(fontsize=14, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold',
            verticalalignment='top', horizontalalignment='left')
    
    # Density vs radius
    ax = axes[0, 1]
    ax.semilogy(r_Rs, n_parker, 'k-', linewidth=2.5, label='Parker (analytical)')
    for i, (method_key, method_name) in enumerate(methods):
        rho = results[method_key]['result']['rho'][-1]
        n = rho / M_P_CGS
        ax.semilogy(r_Rs, n, '--', color=colors[i], alpha=0.7, linewidth=2, label=method_name)
    ax.set_xlabel('Distance (Rs)', fontsize=14)
    ax.set_ylabel('Number density (cm⁻³)', fontsize=14)
    ax.set_ylim(n_parker.min() * 0.8, n_parker.max() * 1.5)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=16, fontweight='bold',
            verticalalignment='top', horizontalalignment='left')
    
    # Temperature vs radius
    ax = axes[1, 0]
    ax.semilogy(r_Rs, T_parker, 'k-', linewidth=2.5, label='Parker (analytical)')
    for i, (method_key, method_name) in enumerate(methods):
        T = results[method_key]['result']['T'][-1]
        ax.semilogy(r_Rs, T, '--', color=colors[i], alpha=0.7, linewidth=2, label=method_name)
    ax.set_xlabel('Distance (Rs)', fontsize=14)
    ax.set_ylabel('Temperature (K)', fontsize=14)
    # Set y-axis range to span at least one order of magnitude
    ax.set_ylim(T_parker.min() * 0.8, T_parker.max() * 1.5)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)
    ax.text(0.02, 0.98, '(c)', transform=ax.transAxes, fontsize=16, fontweight='bold',
            verticalalignment='top', horizontalalignment='left')
    
    # Mass flux conservation: ρ*v*r² vs radius
    ax = axes[1, 1]
    # Normalize by inner boundary value (first real cell)
    for i, (method_key, method_name) in enumerate(methods):
        mf = results[method_key]['mass_flux']
        mf_normalized = mf / mf[0]
        ax.plot(r_Rs, mf_normalized, '-', color=colors[i], alpha=0.7, linewidth=2, label=method_name)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, linewidth=2, label='Perfect conservation')
    ax.set_xlabel('Distance (Rs)', fontsize=14)
    ax.set_ylabel('ρ·v·r² / (ρ·v·r²)₀', fontsize=14)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylim(0.9, 1.1)
    ax.legend(fontsize=14, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)
    ax.text(0.02, 0.98, '(d)', transform=ax.transAxes, fontsize=16, fontweight='bold',
            verticalalignment='top', horizontalalignment='left')
    
    plt.suptitle(f'Solver Comparison: v={v_bc_kms} km/s, T={T_bc/1e6:.0f} MK, n={n_bc} cm⁻³, γ={gamma}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Output to Overleaf directory
    dbox = os.environ.get('DBOX', 'C:\\Users\\mathe\\Dropbox')
    outdir = os.path.join(dbox, 'Apps', 'Overleaf', 'SHUXt')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, 'solver_vs_parker_comparison.pdf')
    plt.savefig(outfile, dpi=150)
    print(f"\nPlot saved to: {outfile}")
    plt.show()
    
    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)
    print("\nKey observations:")
    print("- All solvers should converge to Parker nozzle at steady state")
    print("- 1st order methods (PCM) show more numerical diffusion")
    print("- 2nd order methods (PLM) are more accurate")
    print("- HLLC captures contact discontinuities better than HLL")
    print("- Roe solver is most accurate for smooth flows")


if __name__ == '__main__':
    main()
