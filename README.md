# HUXt - a lightweight solar wind model


## Introduction

This repository provides an implementation of the HUXt model (Heliospheric Upwind Extrapolation with time dependence) in Python, as described by Owens et al. (2020). This is a simple 1D incompressible hydrodynamic model, which essentially solves Burgers equation using the upwind numerical scheme. For more details on the models background, refer to Owens et al. (2020).

## Installation
 ``HUXt`` requires ``numpy`` ``matplotlib`` ``astropy`` ``h5py`` and ``moviepy v1.0.1``. Currently ``moviepy v1.0.1`` is not available on ``conda``, but can be downloaded from ``pip``. Additionally, to make animations, ``moviepy`` requires ``ffmpeg`` to be installed. 

After cloning or downloading ``HUXt``, users should update [``code/config.dat``](code/config.dat) so that ``root`` points to the local directory where HUXt is installed.

## Usage
Some examples of how to use HUXt can be found in [``HUXt_example.ipynb``](code/HUXt_example.ipynb).

``HUXt`` requires an inner boundary condition for longitudinal solar wind speed profile. This can either be prescribed by the user or derived from other sources. For convenience, a [folder of boundary conditions](data/boundary_conditions) is provided containing the equatorial solar wind speed profiles derived from [HelioMAS](https://doi.org/10.1029/2000JA000121) for Carrington rotations 1625 - 2210.

## Contact
Please contact either [Mathew Owens](https://github.com/mathewjowens) or [Luke Barnard](https://github.com/lukebarnard). 

## Citation
Please cite this software with the following doi

## Notes 
