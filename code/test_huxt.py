import numpy as np
import astropy.units as u
from astropy.time import Time
import huxt as H


def test_analytic_solution():
    
    
    #Form longitudinal boundary conditions - background wind of 350 km/s.
    v_boundary = np.ones(128) * 350 * (u.km/u.s)

    # Setup HUXt to do a 1 day simulation, with model output every 8 timesteps (roughly an hour time step), looking at 0 longitude
    model = H.HUXt(v_boundary=v_boundary, lon_out=0.0*u.deg, simtime=5*u.day, dt_scale=8)

    # Solve these conditions, with no ConeCMEs added.
    cme_list = []
    model.solve(cme_list)
    
    # Compute analytical solution from equation 5 in Owens et al 2020
    const = H.huxt_constants()
    alpha  = const['alpha']
    rh = const['r_accel']

    v0 = model.v_boundary[0]
    r = model.r
    r0 = r[0]
    v = v0 * (1 + alpha*(1 - np.exp((r0 - r)/rh)))
    
    # Compute fractional differnce of model solution with analytical 
    dv = np.abs((model.v_grid.squeeze().value - v.value) / v.value)
    
    # These differences should be less than 1e-3 in an absolute sense.
    assert np.allclose(dv, np.zeros(dv.shape), atol=1e-3)
    
    return
    
    

    
    assert x == y
# Analytic test of uniform wind

# Time dependent test with saved file

#    - ambient wind no CME
    
#    - ambient with cme
    
    #    - compare flow fields
        
    #    - compare CME particles
        
    #    - compare CME arrival time