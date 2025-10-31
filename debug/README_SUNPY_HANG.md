# Sunpy Import Hang Issue

## Problem Summary

**HUXt cannot be imported on Python 3.13.9** due to an incompatibility between Python 3.13 and sunpy 7.0.3.

## Root Cause

- Python 3.13 was released in October 2024 (very recent)
- Sunpy 7.0.3 (latest stable release) does not fully support Python 3.13
- The import hangs indefinitely when attempting to `import sunpy`
- This blocks all HUXt functionality since `huxt.py` imports `from sunpy.coordinates import sun`

## Evidence

### Diagnostic Tests Performed

1. **Individual import testing** (`test_individual_imports.py`):
   - ✓ All standard library imports work
   - ✓ NumPy, Astropy, H5py, Numba, Joblib all work
   - ✗ Sunpy hangs on import

2. **Alternate import testing** (`test_sunpy_alternate.py`):
   - ✗ `import sunpy` hangs
   - ✗ `import sunpy.coordinates` hangs
   - ✗ `from sunpy.coordinates import sun` hangs

3. **HUXt import testing** (`test_import_stepwise.py`):
   - ✗ `import huxt` hangs (because huxt.py imports sunpy)
   - ✗ `from huxt import huxt` hangs

### Environment Details

```
Python: 3.13.9 (conda-forge, Oct 22 2025)
Sunpy: 7.0.3 (latest stable)
Platform: Windows, PowerShell
```

## Solution

### Recommended: Downgrade to Python 3.12

Python 3.12 is the officially supported version for sunpy 7.0.3.

```powershell
# Create new conda environment with Python 3.12
mamba create -n huxt312 python=3.12 -c conda-forge

# Activate it
mamba activate huxt312

# Install HUXt dependencies
mamba install numpy scipy astropy h5py numba joblib sunpy -c conda-forge

# Or use existing environment.yml if available
mamba env create -f environment.yml
```

### Alternative Solutions

1. **Wait for sunpy update**: Monitor sunpy releases for Python 3.13 support
2. **Use sunpy development version**: Check if bleeding-edge sunpy has 3.13 fixes
3. **Use Python 3.11**: Also officially supported by sunpy

## Demonstration Script

Run `demonstrate_sunpy_hang.py` to reproduce the issue:

```powershell
python debug/demonstrate_sunpy_hang.py
```

This will:
1. Show your Python version
2. Test all dependencies sequentially
3. Demonstrate the hang on sunpy import
4. Require manual termination (Ctrl+C)

## Impact on Optimization Work

This issue **blocks all HUXt development and testing** because:
- Cannot import HUXt module
- Cannot run tests
- Cannot implement or verify optimizations
- Cannot use the model at all

**Resolution is required before continuing optimization work** (Numba parallel mode, vectorization, etc.)

## Timeline

- **2024-10-27**: Python 3.13 incompatibility identified
- **Status**: Blocking issue
- **Priority**: Critical - must resolve before any other work

## Next Steps

1. ✅ Document the issue (this file)
2. ✅ Create demonstration script
3. ⏳ User to decide: downgrade Python or wait for sunpy update
4. ⏳ Once resolved, continue with optimization #2 (Numba parallel mode)
