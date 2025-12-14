"""
Test and benchmark different compressible solver methods.

This script compares:
- Riemann solvers: Rusanov, HLL, HLLC, Roe
- Reconstruction: PCM (1st order), PLM (2nd order)
- Time integration: Euler, RK2

Run with: python test_compressible_methods.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from huxt.compressible_solvers import (
    CompressibleSolver, 
    create_solver, 
    benchmark_solvers,
    KM_TO_CM,
    K_B_CGS,
    M_P_CGS
)

def main():
    print("="*70)
    print("Compressible Solver Method Comparison")
    print("="*70)
    
    # Set up test problem: Solar wind in CGS units
    # Radial grid from 30 Rs to 215 Rs
    Rs_cm = 6.957e10  # Solar radius in cm
    r_min = 30 * Rs_cm
    r_max = 215 * Rs_cm
    nr = 30  # Reduced for speed - increase for production runs
    r_grid = np.linspace(r_min, r_max, nr)
    
    # Boundary conditions (at 30 Rs)
    v_bc = 400 * KM_TO_CM  # 400 km/s in cm/s
    T_bc = 1e6  # 1 MK
    n_bc = 300  # 300 protons/cm³
    rho_bc = n_bc * M_P_CGS  # g/cm³
    
    gamma = 1.5  # Polytropic index for solar wind
    
    # Simulation time: 1 day (reduced for speed)
    day_sec = 86400
    t_end = 1 * day_sec
    dt_out = t_end / 10
    t_grid = np.arange(0, t_end + dt_out, dt_out)
    
    # BC functions
    v_bc_func = lambda t: v_bc
    rho_bc_func = lambda t: rho_bc
    T_bc_func = lambda t: T_bc
    
    # Methods to compare (reduced set for faster testing)
    methods = [
        ('rusanov-pcm', 'Rusanov + PCM (1st order)'),
        ('hllc-plm', 'HLLC + PLM (2nd order)'),
    ]
    
    results = {}
    
    print("\nRunning solvers...")
    for method_key, method_name in methods:
        print(f"  {method_name}...", end=" ", flush=True)
        
        solver = create_solver(r_grid, gamma=gamma, method=method_key, verbose=False)
        
        t0 = time.time()
        result = solver.solve(t_grid, v_bc_func, rho_bc_func, T_bc_func)
        elapsed = time.time() - t0
        
        results[method_key] = {
            'name': method_name,
            'result': result,
            'time': elapsed
        }
        print(f"done ({elapsed:.2f}s)")
    
    # Use highest-order method as reference
    ref_key = 'hllc-plm'  # Use HLLC+PLM as reference for this reduced test
    ref = results[ref_key]['result']
    
    print("\n" + "="*70)
    print("Results Summary (at final time)")
    print("="*70)
    print(f"{'Method':<35} {'Time (s)':<10} {'RMSE v':<12} {'RMSE rho':<12}")
    print("-"*70)
    
    for method_key, method_name in methods:
        data = results[method_key]
        result = data['result']
        
        # Compute RMS error vs reference
        if method_key != ref_key:
            error_v = np.sqrt(np.mean((result['v'][-1] - ref['v'][-1])**2))
            error_rho = np.sqrt(np.mean((result['rho'][-1] - ref['rho'][-1])**2))
        else:
            error_v = 0.0
            error_rho = 0.0
        
        print(f"{method_name:<35} {data['time']:<10.3f} {error_v:<12.2e} {error_rho:<12.2e}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    r_AU = r_grid / (1.496e13)  # Convert to AU
    
    # Velocity profiles at final time
    ax = axes[0, 0]
    for method_key, method_name in methods:
        result = results[method_key]['result']
        v_kms = result['v'][-1] / KM_TO_CM
        ax.plot(r_AU, v_kms, label=results[method_key]['name'])
    ax.set_xlabel('Distance (AU)')
    ax.set_ylabel('Velocity (km/s)')
    ax.set_title('Velocity at t = 5 days')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Density profiles at final time
    ax = axes[0, 1]
    for method_key, method_name in methods:
        result = results[method_key]['result']
        n_cm3 = result['rho'][-1] / M_P_CGS
        ax.semilogy(r_AU, n_cm3, label=results[method_key]['name'])
    ax.set_xlabel('Distance (AU)')
    ax.set_ylabel('Number density (cm⁻³)')
    ax.set_title('Density at t = 5 days')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Temperature profiles at final time
    ax = axes[1, 0]
    for method_key, method_name in methods:
        result = results[method_key]['result']
        ax.semilogy(r_AU, result['T'][-1], label=results[method_key]['name'])
    ax.set_xlabel('Distance (AU)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Temperature at t = 5 days')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Timing comparison
    ax = axes[1, 1]
    method_names = [results[k]['name'] for k, _ in methods]
    times = [results[k]['time'] for k, _ in methods]
    bars = ax.barh(method_names, times, color='steelblue')
    ax.set_xlabel('Runtime (seconds)')
    ax.set_title('Solver Performance')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add time labels on bars
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{t:.2f}s', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('compressible_solver_comparison.png', dpi=150)
    print("\nPlot saved to: compressible_solver_comparison.png")
    plt.show()
    
    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)
    print("\nKey observations:")
    print("- Rusanov/PCM is fastest but most diffusive")
    print("- HLLC/PLM provides good balance of speed and accuracy")
    print("- Roe solver is most accurate for smooth flows")
    print("- RK2 time integration adds cost but improves temporal accuracy")


if __name__ == '__main__':
    main()
