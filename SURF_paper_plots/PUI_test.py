import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SURF_DIR = os.path.join(PROJECT_ROOT, 'surf')
if SURF_DIR not in sys.path:
    sys.path.insert(0, SURF_DIR)

import datetime
import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import surf as surf
import surf_analysis as surfA



v_boundary = np.ones(128) * 400 * (u.km/u.s)
model_huxt = surf.SURF(v_boundary=v_boundary, lon_out=0.0*u.deg, simtime=0.5*u.day, dt_scale=4, 
                  r_min=21.5*u.solRad, r_max=6000*u.solRad, solver = 'huxt')

model_huxt.solve([])

model_pui = surf.SURF(v_boundary=v_boundary, lon_out=0.0*u.deg, simtime=0.5*u.day, dt_scale=4, 
                  r_min=21.5*u.solRad, r_max=6000*u.solRad, solver = 'huxt-pui')
model_pui.solve([])

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(model_huxt.r/215, model_huxt.v_grid[0, :, 0], label='HUXt', color='r')
ax.plot(model_pui.r/215, model_pui.v_grid[0, :, 0], label='HUXt-PUI', color='b', ls='--')
ax.set_xlabel('r (AU)')
ax.set_ylabel('v (km/s)')   
ax.legend()

plt.show()