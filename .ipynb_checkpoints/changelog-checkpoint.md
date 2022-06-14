# HUXt Changelog

# V4.0.0

## Bug fixes
- Fixed bug in `ConeCME.compute_arrival_at_body()`, that was incorrectly centering the model longitudes on the CME nose. This lead to computing the arrival time for CME longitudes offset from the expected longitude unless the CME source longitude was 0 (fully Earth directed).
- `ConeCME._track_` had a bug due to incorrectly wrapping longitudes of the CME particles. This meant that the plotting routines drew the incorrect CME boundary for CMEs with noses at longitudes just less than 360 deg, with flanks that cross the 360-0 wrapover.  
- `huxt/load_HUXt_run()` now raises an FileNotFound exception if given an invalid filepath. Before it returned empty lists in place of the HUXt and ConeCME classes.
- `HUXt.buffertime`, which controls the model spin-up time, was set to a fixed value which was too low. For some edge cases that included very low solar wind speeds, this didn't give the model enough time to spin-up properly and so the model solution contained some artefacts of the spin-up for some time steps in the outer limits of the radial grid. `HUXt.buffertime` is now a function of the minimum inner boundary speed and works as intended for uniform inner boundary speeds between 200km/s and 1000 km/s. 
- `radial_grid()` returned incorrect relative radial grid values (`rrel`) for HUXt instances with inner boundaries other than  30Rs. This unfortunately resuleted in increasingly large errors as lower inner boundaries were used. For example, at 10Rs, the model solutions were, on average, wrong by 8%. 
- Updated dependencies in `environment.yml` and `requirements.txt` to mititgate security issues raised in the Dependabot alerts.
- Removed Gaussian smoothing of the CME front in `ConeCME._track_`, as this made CME tracking inconsistent between HUXt runs that spanned different regions of the CME. Now CME kinematics calculations match along a specific longitude irrespective of whether the HUXt domain partially or wholly covers the CME width. 
- Fixed `ConeCME.compute_arrival_at_body()` so that it also works with HUXt solutions for only one longitude coordinate. Now HUXt simulations along one, many, or all longitudes should return consistent CME arrival times. This requires a different approach to the method for multiple longitudes, and can result in small differences in the CME arrival time, of the order of the model timestep. However, this is much smaller than all other sources of uncertainty in this simulation and calculation.

## Additions
- Now `ConeCME.compute_arrival_at_body()` returns the arrival statistics in a dictionary rather than returning a tuple of parameters.
- Updated HUXt and ConeCME classes so that the flow speed at cme tracer particle coordinates is also stored, and the arrival speed is returned by `ConeCME.compute_arrival_at_body()`.
- A small test suite is included in "test_huxt.py", which runs under pytest. Currently includes tests against a simple analytic solution for a uniform boundary and against a time-dependent solution. In the time-dependent solution test, the solar wind speed solution and tracked ConeCME are checked for consistency with a reference simulation.
- Checks on ConeCME inputs to exclude ConeCMEs with no overlap with a HUXt domain.
- New `huxt_inputs/import_cone2bc_parameters` and `huxt_inputs/ConeFile_to_ConeCME_list()`functions to create ConeCME instances from the Cone files used to input Cone CMEs into the Enlil solar wind model.
- New `huxt_inputs/get_WSA_maps` function to read and process standard Met Office WSA output for use with HUXt
- New `huxt_inputs/datetime2huxtinputs` function to convert a datetime into the cr_num and cr_lon_init parameters required to initialise HUXt
- Changed the `huxt.solve()` method logic to first compute complete boundary conditions at each longitude, then solve the Burgers' equation. This allows fully time-dependent boundary conditions to specified.

## Breaking changes
- Renamed `huxt_inputs/get_MAS_vrmap` and `huxt_inputs/get_MAS_brmap` to `huxt_inputs/get_MAS_vr_map` and `huxt_inputs/get_MAS_br_map`. HUXt3D code changed to be consistent with correctly transposed Vr and Br inner boundary conditions



