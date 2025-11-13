import sys
print("Python version:", sys.version)
print("Testing individual imports...")



print("1. copy...", end=" ")
import copy
print("OK")

print("2. errno...", end=" ")
import errno
print("OK")

print("3. os...", end=" ")
import os
print("OK")

print("4. appdirs...", end=" ")
from appdirs import user_data_dir
print("OK")

print("5. astropy.units...", end=" ")
import astropy.units as u
print("OK")

print("6. astropy.time...", end=" ")
from astropy.time import Time, TimeDelta
print("OK")

print("7. h5py...", end=" ")
import h5py
print("OK")

print("8. joblib...", end=" ")
from joblib import Parallel, delayed
print("OK")

print("9. numpy...", end=" ")
import numpy as np
print("OK")

print("10. numba...", end=" ")
from numba import jit
print("OK")

print("11. pathlib...", end=" ")
from pathlib import Path
print("OK")

print("12. sunpy.coordinates...", end=" ")
from sunpy.coordinates import sun
print("OK")


print("\nAll imports successful!")
