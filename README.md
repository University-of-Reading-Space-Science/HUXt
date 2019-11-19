# HUXt - a lightweight solar wind model
---

This repository provides an implementation of the HUXt model (Heliospheric Upwind Extrapolation with time dependence) in Python, as described by Owens et al. (2019). It's dependencies are ``numpy`` ``matplotlib`` ``astropy`` ``h5py`` and ``moviepy``. Additionally, to make animations, ``moviepy`` requires ``ffmpeg`` to be installed. 

An example of how to use HUXt can be found in ``code\HUXt_example.ipynb``.

Users should update ``code\config.dat`` to point to local directories where data, figures and animations will be saved.