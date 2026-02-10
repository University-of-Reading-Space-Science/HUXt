import numpy as np
import astropy.units as u
import matplotlib
from sympy import solve
matplotlib.use('TkAgg')  # Set backend explicitly for Windows
import matplotlib.pyplot as plt
import datetime
import os
import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin
import huxt.huxt_insitu as Hinsitu



simtime = 5*u.day

plt.ion()  # Turn on interactive mode

# start with a solar wind speed at 1 au of 400 km/s
v_1au = 400*u.km/u.s

#backmap this to 0.1 AU
v_0p1au, n_0p1au, T_0p1au, lon_0p1au = Hin.map_v_inwards_parker(v_1au, 215*u.solRad, 0.0*u.rad, 21.5*u.solRad, gamma=1.5)

vr_in = np.ones(128) * v_0p1au 

#now run SURF-hydro out to 1 AU
model_plm = H.HUXt(v_boundary=vr_in, 
                        simtime=simtime, dt_scale=4,  r_min=21.5*u.solRad, r_max=220*u.solRad,
                    solver ='hllc-plm-rk2', lon_out=0.0*u.rad) 
model_plm.solve([])


HA.plot_earth_timeseries(model_plm, plot_omni=False)

plt.show(block=True)
print("\nPlots displayed. Press Enter to close all plots and exit...")
input()