"""
Minimal test to check if imports work.
"""
import sys
sys.stdout = open('test_minimal_output.txt', 'w')
sys.stderr = sys.stdout

try:
    print("Starting imports...")
    import numpy as np
    print("Numpy imported")
    import astropy.units as u
    print("Astropy imported")
    from huxt import HUXt
    print("HUXt imported")
    
    print("\nCreating model...")
    model = HUXt(
        simtime=1 * u.day,
        dt_scale=8,
        v_boundary=np.ones(128) * 400.0 * (u.km / u.s),
        solver='cgf',
        parallel=False
    )
    print(f"Model created: compressible={model.compressible}")
    
    print("\nRunning solve...")
    model.solve([])
    print("SUCCESS!")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    sys.stdout.close()
