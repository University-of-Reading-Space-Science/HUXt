# HUXt - a lightweight solar wind model


## Introduction

This repository provides an implementation of the HUXt model (Heliospheric Upwind Extrapolation with time dependence) in Python, as described by [Owens et al. (2020)](https://doi.org/10.1007/s11207-020-01605-3). This is a simple 1D incompressible hydrodynamic model, which essentially solves Burgers equation using the upwind numerical scheme. For more details on the models background, refer to [Owens et al. (2020)](https://doi.org/10.1007/s11207-020-01605-3).

## Installation
 ``HUXt`` is written in Python 3.7.3 and has a range of dependencies, which are listed in the ``requirements.txt`` and ``environment.yml`` files. Because of these dependencies, the simplest way to work with ``HUXt`` in ``conda`` is to create its own environment. With the anaconda prompt, in the root directory of ``HUXt``, this can be done as:
```
>>conda env create -f environment.yml
>>conda activate huxt
``` 
Then the examples can be accessed through 
```
>>jupyter lab code/HUXt_example.ipynb
```
Please note that after cloning or downloading ``HUXt``, users should update [``code/config.dat``](code/config.dat) so that ``root`` points to the local directory where HUXt is installed. Otherwise HUXt won't know where to save and load things from.

## Usage
Some examples of how to use HUXt can be found in [``HUXt_example.ipynb``](code/HUXt_example.ipynb).

``HUXt`` requires an inner boundary condition for longitudinal solar wind speed profile. This can either be prescribed by the user or derived from other sources. For convenience,  [``HUXt_inputs.py``](code/HUXt_inputs.py) provides some functions for downloading solar wind speed profiles from the [HelioMAS](https://doi.org/10.1029/2000JA000121) solutions (an example is provided in the examples workbook).

## Contact
Please contact either [Mathew Owens](https://github.com/mathewjowens) or [Luke Barnard](https://github.com/lukebarnard). 

## Citation
Please cite this software as Owens et al. (2020),  *A Computationally Efficient, Time-Dependent Model of the Solar Wind for Use as a Surrogate to Three-Dimensional Numerical Magnetohydrodynamic Simulations*,  Sol Phys, DOI: [10.1007/s11207-020-01605-3](https://doi.org/10.1007/s11207-020-01605-3)
