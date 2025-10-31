"""
Diagnostic script: run a short CGF 2D (multiple longitudes) and print v_grid occupancy and stats per longitude.
"""
import numpy as np
import astropy.units as u
from huxt import HUXt

simtime = 1 * u.day
v_boundary = np.ones(128) * 400.0 * (u.km / u.s)

model = HUXt(simtime=simtime, dt_scale=4, v_boundary=v_boundary,
             r_min=21.5*u.solRad, r_max=240*u.solRad, lon_start=330*u.deg,
             lon_stop=30*u.deg, frame='sidereal', solver='cgf', parallel=False)

print('Model created: compressible=', model.compressible, ' nlon=', model.lon.size)

# Run minimal solve (no CMEs)
model.solve([])

# After solve, inspect v_grid
v = model.v_grid  # Quantity
rho = model.rho_grid if hasattr(model, 'rho_grid') else None

print('\nGrid shapes: v_grid', v.shape, ' rho_grid', rho.shape if rho is not None else None)

# For each longitude compute NaN fraction and min/max at inner radial index
for i in range(model.lon.size):
    col = v[:, 0, i].value  # first radial cell
    nans = np.isnan(col).sum()
    tot = col.size
    print(f'Lon {i+1}/{model.lon.size}: NaNs {nans}/{tot} ({100.0*nans/tot:.1f}%), min {np.nanmin(col):.1f}, max {np.nanmax(col):.1f}')

# Also global stats
allvals = v[:, 0, :].value.flatten()
print('\nOverall inner-radial stats: total points', allvals.size, ' NaNs', np.isnan(allvals).sum())
print('Min', np.nanmin(allvals), 'Max', np.nanmax(allvals))
