[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4889326.svg)](https://doi.org/10.5281/zenodo.4889326)
# HUXt - a lightweight solar wind model


## Introduction

This repository provides an implementation of the HUXt model (Heliospheric Upwind Extrapolation with time dependence) in Python, as described by [Owens et al. (2020)](https://doi.org/10.1007/s11207-020-01605-3). This is a simple 1D incompressible hydrodynamic model, which essentially solves Burgers equation using the upwind numerical scheme. For more details on the models background, refer to [Owens et al. (2020)](https://doi.org/10.1007/s11207-020-01605-3).

## Installation
 `HUXt` is written in Python 3.9.13 and has a range of dependencies, which are listed in the `requirements.txt` and `environment.yml` files. Because of these dependencies, the simplest way to work with `HUXt` is to use `conda` to create a virtual environment for `HUXt`. We recommend using and up-to-date version of [miniconda](https://docs.anaconda.com/free/miniconda/index.html). With the anaconda prompt, in the root directory of `HUXt`, this can be done as:
```
>>conda env create -f environment.yml
>>conda activate huxt
``` 
Then the examples can be accessed through 
```
>>jupyter lab code/HUXt_examples.ipynb
```

## Testing
A small test suite is included in ['test_huxt.py'](code/test_huxt.py), which compares tests a HUXt installation against a simple analytical solution and some reference simulation data included in this repository. You should run the test suite after installation as a (limited) check that everything is working as intended. The test suite uses `pytest`, which is included in the `huxt` environment. Using the anaconda prompt from the root directory of `HUXt`, these tests can be performed by running:
```
pytest code/test_huxt.py
```
The four tests should take around 30s to complete on a modest laptop. These tests are not an exhaustive test of all of the features in `huxt`, but they do cover all of the core functionality and expected common use cases.

## Usage
Some examples of how to use HUXt can be found in [`HUXt_examples.ipynb`](code/HUXt_examples.ipynb).

`HUXt` requires an inner boundary condition for longitudinal solar wind speed profile. This can either be prescribed by the user or derived from other sources. For convenience,  [`huxt_inputs.py`](code/huxt_inputs.py) provides some functions for downloading and generating longitudinal solar wind speed profiles from the [HelioMAS](https://doi.org/10.1029/2000JA000121), Wang-Sheeley-Arge (WSA), and Potential Field Source Surface (PFSS) models, as well as from Coronal Tomography (CorTom). Examples of each is provided in the examples workbook. Routines for plotting and animating HUXt solutions can be found in  [`huxt_analysis.py`](code/huxt_analysis.py). Again, examples are provided in the workbook.

## Contact
Please contact either [Mathew Owens](https://github.com/mathewjowens) or [Luke Barnard](https://github.com/lukebarnard). 

## Citations

If you use HUXt in a publication or presentation, please cite the software using the Zenodo reference with DOI:[10.5281/zenodo.4889326](https://doi.org/10.5281/zenodo.4889326) 

To cite this project, including the scientific basis and functionality of HUXt, please use: 

Barnard and Owens. (2022), *HUXt - An open source, computationally efficient reduced-physics solar wind model, written in Python*, Frontiers in Physics [10.3389/fphy.2022.1005621](https://doi.org/10.3389/fphy.2022.1005621)

Owens et al. (2020),  *A Computationally Efficient, Time-Dependent Model of the Solar Wind for Use as a Surrogate to Three-Dimensional Numerical Magnetohydrodynamic Simulations*,  Sol Phys, DOI:[10.1007/s11207-020-01605-3.svg](https://doi.org/10.1007/s11207-020-01605-3)
