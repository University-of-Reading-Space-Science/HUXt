# HUXt - a lightweight solar wind model


## Introduction

This repository provides an implementation of the HUXt model (Heliospheric Upwind Extrapolation with time dependence) in Python, as described by [Owens et al. (2020)](https://doi.org/10.1007/s11207-020-01605-3). This is a simple 1D incompressible hydrodynamic model, which essentially solves Burgers equation using the upwind numerical scheme. For more details on the models background, refer to [Owens et al. (2020)](https://doi.org/10.1007/s11207-020-01605-3).

## Installation
 ``HUXt`` is written in Python 3.7.3 and requires ``numpy``, ``scipy``, ``scikit-image``, ``matplotlib``, ``astropy``, ``sunpy``, ``h5py``, and ``moviepy v1.0.1``. Currently ``moviepy v1.0.1`` is not available on ``conda``, but can be downloaded from ``pip``. Additionally, to make animations, ``moviepy`` requires ``ffmpeg`` to be installed. Specific dependencies can be found in the ``requirements.txt`` and ``environment.yml`` files.

After cloning or downloading ``HUXt``, users should update [``code/config.dat``](code/config.dat) so that ``root`` points to the local directory where HUXt is installed.

The simplest way to work with ``HUXt`` in ``conda`` is to create its own environment. With the anaconda prompt, in the root directory of ``HUXt``, this can be done as:
```
>>conda env create -f environment.yml
>>conda activate huxt
``` 
Then the examples can be accessed through 
```
>>jupyter lab code/HUXt_example.ipynb
```

## Usage
Some examples of how to use HUXt can be found in [``HUXt_example.ipynb``](code/HUXt_example.ipynb).

``HUXt`` requires an inner boundary condition for longitudinal solar wind speed profile. This can either be prescribed by the user or derived from other sources. For convenience, a [folder of boundary conditions](data/boundary_conditions) is provided containing the equatorial solar wind speed profiles derived from [HelioMAS](https://doi.org/10.1029/2000JA000121) for Carrington rotations 1625 - 2210.

## Contact
Please contact either [Mathew Owens](https://github.com/mathewjowens) or [Luke Barnard](https://github.com/lukebarnard). 

## Citation
Please cite this software as Owens et al. (2020),  *A Computationally Efficient, Time-Dependent Model of the Solar Wind for Use as a Surrogate to Three-Dimensional Numerical Magnetohydrodynamic Simulations*,  Sol Phys, DOI: [10.1007/s11207-020-01605-3](https://doi.org/10.1007/s11207-020-01605-3)

