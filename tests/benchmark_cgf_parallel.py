"""
Benchmark script to compare serial vs parallel performance of CGF solver.

This script runs the CGF solver with different longitude resolutions
and compares serial vs parallel execution times.
"""
import numpy as np
import astropy.units as u
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from huxt import HUXt

def benchmark_solver(parallel=False, simtime=1.0):
    """
    Run HUXt with CGF solver and time execution.
    
    Args:
        nlon: Number of longitude points (1, 64, 128, etc.)
        parallel: Whether to use parallel execution
        simtime: Simulation time in days
    
    Returns:
        Total execution time in seconds
    """
    print(f"\n{'='*70}")
    print(f"Benchmark: parallel={parallel}, simtime={simtime} days")
    print(f"{'='*70}")
    
    # Create model with uniform 400 km/s boundary
    if nlon == 1:
        # Single longitude (1D run)
        model = HUXt(
            v_boundary=400.0 * u.km / u.s * np.ones(128),
            simtime=simtime * u.day,
            lon_out=0.0 * u.deg,
            compressible=True,
            solver='cgf',
            parallel=parallel
        )
    else:
        # Multi-longitude run
        model = HUXt(
            v_boundary=400.0 * u.km / u.s * np.ones(128),
            simtime=simtime * u.day,
            compressible=True,
            solver='cgf',
            parallel=parallel
        )
    
    # Run solver and time it
    start = time.time()
    model.solve([])
    end = time.time()
    
    elapsed = end - start
    
    print(f"\nTotal time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    if nlon > 1:
        print(f"Time per longitude: {elapsed/nlon:.2f} seconds")
    
    return elapsed


def main():
    """Run benchmarks comparing serial vs parallel execution."""
    
    print("\n" + "="*70)
    print("CGF SOLVER PARALLELIZATION BENCHMARK")
    print("="*70)
    
    # Test configurations
    test_cases = [
        {'nlon': 1, 'simtime': 1.0, 'label': '1D run (1 longitude)'},
        {'nlon': 8, 'simtime': 1.0, 'label': '8 longitudes'},
        {'nlon': 16, 'simtime': 1.0, 'label': '16 longitudes'},
        {'nlon': 32, 'simtime': 1.0, 'label': '32 longitudes'},
    ]
    
    results = []
    
    for case in test_cases:
        nlon = case['nlon']
        simtime = case['simtime']
        label = case['label']
        
        print(f"\n\n{'#'*70}")
        print(f"# TEST: {label}")
        print(f"{'#'*70}")
        
        # Run serial
        print(f"\n--- Serial execution ---")
        time_serial = benchmark_solver(nlon, parallel=False, simtime=simtime)
        
        if nlon > 1:
            # Run parallel (only makes sense for multi-longitude)
            print(f"\n--- Parallel execution ---")
            time_parallel = benchmark_solver(nlon, parallel=True, simtime=simtime)
            
            speedup = time_serial / time_parallel
            efficiency = speedup / nlon * 100
            
            results.append({
                'nlon': nlon,
                'serial': time_serial,
                'parallel': time_parallel,
                'speedup': speedup,
                'efficiency': efficiency,
                'label': label
            })
        else:
            results.append({
                'nlon': nlon,
                'serial': time_serial,
                'parallel': None,
                'speedup': None,
                'efficiency': None,
                'label': label
            })
    
    # Print summary
    print("\n\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Test':<25} {'Serial (s)':<12} {'Parallel (s)':<14} {'Speedup':<10} {'Efficiency'}")
    print("-"*70)
    
    for r in results:
        if r['parallel'] is not None:
            print(f"{r['label']:<25} {r['serial']:>10.2f}   {r['parallel']:>12.2f}   "
                  f"{r['speedup']:>8.2f}x   {r['efficiency']:>6.1f}%")
        else:
            print(f"{r['label']:<25} {r['serial']:>10.2f}   {'N/A':<14} {'N/A':<10} {'N/A'}")
    
    print("="*70)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if len(results) > 1 and results[1]['speedup'] is not None:
        if results[1]['speedup'] > 1.5:
            print("✓ Parallel execution provides significant speedup")
            print(f"  - Best speedup: {max([r['speedup'] for r in results if r['speedup'] is not None]):.2f}x")
        elif results[1]['speedup'] > 1.1:
            print("⚠ Parallel execution provides modest speedup")
            print("  - May be worth using for larger runs")
        else:
            print("✗ Parallel execution shows minimal benefit")
            print("  - Serial execution may be better for small problems")
    
    print("\n")


if __name__ == '__main__':
    main()
