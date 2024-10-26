# HUXt Changelog

# V4.2.3
## Bug fixes
- HUXt was intermittently erroring due to an assumption that `coneCME.parameter_array()` would an array of length 10. This assumption was broken with the addition of the `cme_expansion` and `cme_fixed_duration` options. This has now been fixed.
- There was an error in load_HUXt_run, which was incorrectly loading ConeCME objects, as it wasn't loading the initial_height parameter, and was instead defaulting to 30 Rs. 

# V4.2.2
## Additions
- Added a `__version__` attribute to `huxt.ConeCME` and `huxt.HUXt` classes
- Added functionality to `huxt.ConeCME` to give the CME a fixed injection duration. This alleviates a feature/bug of the cone CME geometry where slower CMEs result in longer duration perturbations to the boundary conditions. This is activated by setting `cme_fixed_duration` to `True` and providing the CME duration in hours to `fixed_duration`.
- Added functionality to `huxt.ConeCME` to give the CME an expanding profile, with the flag `cme_expansion`. If this is `True`, then the CME velocity profile on the inner boundary decreases with time, corresponding to an expanding CME. As per Owens et al. 2005.

# V4.2.1

## Additions
- Add functions to `huxt_inputs` for downloading CME paramters and producing ConeCME objects from the DONKI catalogue

# V4.2.0
## Bug fixes
- Fixed plotting error for streaklines which assumed streaklines should be plotted at all longitudes.
- Some bug fixes in `huxt_analysis.plot()`
- The fields `streak_particles_r` and `hcs_particles_r` were not being saved by `huxt.save()`, which is now fixed. 

## Additions
- Updated environment to solve dependency security issues
- A Dynamic-Time-Warping method has been included for generating HUXt boundary conditions from OMNI data, `huxt_inputs.generate_vCarr_from_OMNI_DTW()`.
- Increased support for running and plotting HUXt solutions to the outer planets.
- Added a script to generate the reference simulations for the test suite to `/data/test_data`. This should be left alone - they only need to be run for certain breaking changes.

## Breaking changes
- `v_max` set to 3000km/s, up from 2000km/s. This increases simulation times, but enables simulation of faster CMEs.

# V4.1.1
## Bug fixes
- In `huxt_inputs.set_time_dependent_boundary()` added a check to ensure input vgrid_Carr matches the longitudes of HUXt.
- In `huxt_inputs.generate_vCarr_from_OMNI()` added checks and warnings to highlight if generated longitude series does not match default HUXt longitudes.

## Additions
- Dependence on `moviepy` removed from `huxt_analysis.animate()`. This now uses only matplotlib. Syntax remains unchanged so is backwards compatible.

# V4.1.0

## Bug fixes
- In `huxt.solve()` tracing field lines was failing in the updated environment due to the default argument of an empty list. This was fixed by being replaced with an empty quantity
- In the updated environment, `huxt_inputs.generate_vCarr_from_OMNI()` was failing due to an unexplained issue with a pandas dataframe containing an astropy quantity, and that pandas could not copy this data frame. Units have been removed from the dataframe, which solves the issue. 

## Additions
- Updated environment file to solve dependency sercurity issues
- Added `ConeCME.compute_arrival_at_location()`, to compute CME arrival at a specified radius and longitude.
- Added CME expansion to CME boundary paramterisation
- Updated CorTom loader to work with the IDL save files in the Aberystwyth repository

# V4.0.1

## Bug Fixes
- Fixed bug `HUXt.solve()` that incorrectly adjusted CME longitudes when running HUXt in the sidereal frame. In this bug, the error in the adjusted CME longitude grew linearly with the CME launch time relative to model initialisation time.

# V4.0.0

## Bug fixes
- Fixed bug that assumed ConeCMEs were always initialised at 30 Rs. This caused errors in calculations of the Cone CME radius when HUXt was configured to use an inner boundary radius other than 30 Rs.  
- Fixed bug in `ConeCME.compute_arrival_at_body()`, that was incorrectly centering the model longitudes on the CME nose. This lead to computing the arrival time for CME longitudes offset from the expected longitude unless the CME source longitude was 0 (fully Earth directed).
- `ConeCME._track_` had a bug due to incorrectly wrapping longitudes of the CME particles. This meant that the plotting routines drew the incorrect CME boundary for CMEs with noses at longitudes just less than 360 deg, with flanks that cross the 360-0 wrapover.  
- `huxt/load_HUXt_run()` now raises an FileNotFound exception if given an invalid filepath. Before it returned empty lists in place of the HUXt and ConeCME classes.
- `HUXt.buffertime`, which controls the model spin-up time, was set to a fixed value which was too low. For some edge cases that included very low solar wind speeds, this didn't give the model enough time to spin-up properly and so the model solution contained some artefacts of the spin-up for some time steps in the outer limits of the radial grid. `HUXt.buffertime` is now a function of the minimum inner boundary speed and works as intended for uniform inner boundary speeds between 200km/s and 1000 km/s. 
- `radial_grid()` returned incorrect relative radial grid values (`rrel`) for HUXt instances with inner boundaries other than  30Rs. This unfortunately resuleted in increasingly large errors as lower inner boundaries were used. For example, at 10Rs, the model solutions were, on average, wrong by 8%. 
- Updated dependencies in `environment.yml` and `requirements.txt` to mititgate security issues raised in the Dependabot alerts.
- Removed Gaussian smoothing of the CME front in `ConeCME._track_`, as this made CME tracking inconsistent between HUXt runs that spanned different regions of the CME. Now CME kinematics calculations match along a specific longitude irrespective of whether the HUXt domain partially or wholly covers the CME width. 
- Fixed `ConeCME.compute_arrival_at_body()` so that it also works with HUXt solutions for only one longitude coordinate. Now HUXt simulations along one, many, or all longitudes should return consistent CME arrival times. This requires a different approach to the method for multiple longitudes, and can result in small differences in the CME arrival time, of the order of the model timestep. However, this is much smaller than all other sources of uncertainty in this simulation and calculation.

## Additions
- Ephemeris data updated to include positions of Jupiter and Saturn, for plotting with outer heliospheric simulations.
- Added `generate_vCarr_from_OMNI` function that uses Sunpy.Fido to download OMNI solar wind speed data from the VSO, and processes these data to generate a time-dependent input time series for driving HUXt with.
- `test_huxt.py` provides a small test suite that compares huxt solutions against a simple anlytical solution for a uniform and constant inner boundary condition, as well as comparing a time dependent solution with a Cone CME against a reference scenario. This can be used to test that a version of HUXt is performing in a manner consistent with some of our expectations. This test suite will be expanded in future versions.
- `huxt_analysis` includes function `huxt_streakline` for computing streaklines in the HUXt solutions. These can be used to trace the location of the heliospheric current sheet with `trace_HCS` and map the polarity of the heliospheric magnetic field throughout the model domain with `add_bgrid`. 
- Now `ConeCME.compute_arrival_at_body()` returns the arrival statistics in a dictionary rather than returning a tuple of parameters.
- Updated HUXt and ConeCME classes so that the flow speed at cme tracer particle coordinates is also stored, and the arrival speed is returned by `ConeCME.compute_arrival_at_body()`.
- A small test suite is included in "test_huxt.py", which runs under pytest. Currently includes tests against a simple analytic solution for a uniform boundary and against a time-dependent solution. In the time-dependent solution test, the solar wind speed solution and tracked ConeCME are checked for consistency with a reference simulation.
- Checks on ConeCME inputs to exclude ConeCMEs with no overlap with a HUXt domain.
- New `huxt_inputs/import_cone2bc_parameters` and `huxt_inputs/ConeFile_to_ConeCME_list()`functions to create ConeCME instances from the Cone files used to input Cone CMEs into the Enlil solar wind model.
- New `huxt_inputs/get_CorTom_vr_map` and `huxt_inputs/get_CorTom_long_profile` functions to read and process Coronal Tomography output for use with HUXt
- New `huxt_inputs/get_PFSS_maps` and `huxt_inputs/get_PFSS_long_profile` functions to read and process PFSS output for use with HUXt
- New `huxt_inputs/get_WSA_maps` and `huxt_inputs/get_WSA_long_profile` functions to read and process standard Met Office WSA output for use with HUXt
- New `huxt_inputs/datetime2huxtinputs` function to convert a datetime into the cr_num and cr_lon_init parameters required to initialise HUXt
- Changed the `huxt.solve()` method logic to first compute complete boundary conditions at each longitude, then solve the Burgers' equation. This allows fully time-dependent boundary conditions to specified.

## Breaking changes
- Renamed `huxt_inputs/get_MAS_vrmap` and `huxt_inputs/get_MAS_brmap` to `huxt_inputs/get_MAS_vr_map` and `huxt_inputs/get_MAS_br_map`. HUXt3D code changed to be consistent with correctly transposed Vr and Br inner boundary conditions



