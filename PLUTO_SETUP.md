# PLUTO Setup Guide for HUXt

This guide explains how to set up PLUTO (a multidimensional hydrodynamics code) for use with the HUXt solar wind model on the `pluto` branch.

## Overview

PLUTO is an external, standalone hydrodynamics solver that HUXt can integrate with to provide compressible solar wind simulations. Unlike standard Python packages, PLUTO must be downloaded, compiled from source, and configured on your system.

## System Requirements

Before installing PLUTO, ensure you have:
- **Fortran compiler**: `gfortran`, `ifort`, or equivalent
- **C compiler**: `gcc`, `clang`, or equivalent
- **GNU Make** or compatible build tools
- On Windows: Consider using Windows Subsystem for Linux (WSL) or MSYS2 for easier compilation

## Step 1: Download PLUTO

1. Visit the official PLUTO website: http://plutocode.ph.unito.it/

2. Download the latest stable release (typically a `.tar.gz` archive)

3. Extract the archive:
   ```bash
   tar -xzf Pluto_vX.X.tar.gz
   cd Pluto_vX.X
   ```

## Step 2: Build PLUTO

1. **Configure the build** (interactive setup):
   ```bash
   cd Source
   ./Makefile.setup
   ```
   This will prompt you to select:
   - Physics module (e.g., `HD` for hydrodynamics, `MHD` for magnetohydrodynamics)
   - Geometry (e.g., `CARTESIAN`, `CYLINDRICAL`, `SPHERICAL`)
   - Boundary conditions
   - Output format

2. **Compile**:
   ```bash
   make clean
   make
   ```
   This will create PLUTO executables in the `Bin/` directory.

3. **Verify installation**:
   ```bash
   ./Bin/pluto --version
   ```

## Step 3: Configure Environment Variables

Set up your system to find PLUTO:

### On Linux/macOS:
Add these lines to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):
```bash
export PLUTO_DIR="/path/to/Pluto_vX.X"
export PATH="$PLUTO_DIR/Bin:$PATH"
```

Reload your shell:
```bash
source ~/.bashrc
```

### On Windows (using WSL or MSYS2):
Set environment variables in your shell initialization file or use the Windows System Properties dialog:
```cmd
set PLUTO_DIR=C:\path\to\Pluto_vX.X
set PATH=%PLUTO_DIR%\Bin;%PATH%
```

### In Conda/Python Environment:
Alternatively, set environment variables in your conda environment:
```bash
conda activate huxt
conda env config vars set PLUTO_DIR="/path/to/Pluto_vX.X"
conda activate huxt  # Re-activate to apply
```

## Step 4: Verify HUXt Can Access PLUTO

Test the integration in Python:
```python
from huxt import HUXt
import astropy.units as u
import numpy as np

# Create a simple model with compressible solver
v_boundary = 400 * np.ones(128) * (u.km / u.s)
model = HUXt(
    v_boundary=v_boundary,
    cr_num=2050,
    compressible=True,  # Enable compressible solver
    solver='pluto',     # Use PLUTO solver
)
model.solve()
```

If this runs without errors, PLUTO is properly configured!

## Troubleshooting

### Error: PLUTO executable not found
- Verify `PLUTO_DIR` is set correctly: `echo $PLUTO_DIR`
- Verify PLUTO binaries exist: `ls $PLUTO_DIR/Bin/`
- Check `$PATH` includes PLUTO: `echo $PATH | grep pluto`

### Error: Fortran compiler not found
- **Linux**: `sudo apt-get install gfortran` (Ubuntu/Debian) or `sudo yum install gcc-gfortran` (RedHat/CentOS)
- **macOS**: `brew install gcc` or use Xcode Command Line Tools
- **Windows**: Install MinGW or use Windows Subsystem for Linux

### Error: PLUTO build fails with "No rule to make target"
- Ensure you ran `./Makefile.setup` first
- Check that the `Makefile` was created in `Source/`
- Try `make clean` before rebuilding

### Segmentation fault or runtime errors
- Ensure PLUTO was compiled with matching compiler flags for your system
- Check HUXt debug output for more details
- Review PLUTO output logs in the working directory

## Using Compressible Solvers in HUXt

Once PLUTO is installed and configured, you can use compressible solvers:

```python
import numpy as np
import astropy.units as u
from huxt import HUXt

# Create boundary conditions
v_boundary = 400 * np.ones(128) * (u.km / u.s)
rho_boundary = 5.0 * np.ones(128)  # Number density, cm^-3
T_boundary = 1e6 * np.ones(128)     # Temperature, K

# Create model with PLUTO solver
model = HUXt(
    v_boundary=v_boundary,
    rho_boundary=rho_boundary,
    T_boundary=T_boundary,
    cr_num=2050,
    compressible=True,
    solver='pluto',  # or 'cgf' for alternative solver
)

# Solve
model.solve(cme_list=[])
```

## PLUTO Documentation

For detailed PLUTO documentation, configuration options, and advanced usage:
- Official Manual: http://plutocode.ph.unito.it/userguide.pdf
- Online Documentation: http://plutocode.ph.unito.it/

## Related HUXt Branches

- **`master`**: Standard incompressible solver with parallelization
- **`dev/parallelization`**: Master branch with recent parallelization improvements
- **`pluto`**: Compressible solvers (CGF and PLUTO) integrated with parallelization

## Notes

- PLUTO must be installed **before** using compressible solver features
- If PLUTO is not available, HUXt will gracefully fall back to standard solvers
- The `parallel=True` option works with all solvers including PLUTO-based ones
- PLUTO builds can be slow (5-15 minutes depending on system)
- Different PLUTO configurations can produce different outputs; document your setup for reproducibility
