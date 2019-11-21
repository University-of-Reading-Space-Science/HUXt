# HUXt - a lightweight solar wind model
---

This repository provides an implementation of the HUXt model (Heliospheric Upwind Extrapolation with time dependence) in Python, as described by Owens et al. (2019). It's dependencies are ``numpy`` ``matplotlib`` ``astropy`` ``h5py`` and ``moviepy``. Additionally, to make animations, ``moviepy`` requires ``ffmpeg`` to be installed. 

Users should update ``root`` in ``code/config.dat`` to point to the local directory where HUXt is installed.

Some examples of how to use HUXt can be found in ``code/HUXt_example.ipynb``.

``data/boundary_conditions`` contains the equatorial solar wind speed profiles used as the inner boundary conditions to HUXt. Need some more details here...