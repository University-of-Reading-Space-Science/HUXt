import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import datetime


import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin
import huxt.huxt_insitu as Hinsitu

stime = datetime.datetime(2020,1,1)
ftime = datetime.datetime(2020,6,1)

cr, cr_lon_init = Hin.datetime2huxtinputs(ftime)
model = Hinsitu.omniHUXt_reconstruction(stime, ftime , 
                        rmin=21.5*u.solRad, rmax=250*u.solRad, 
                        dt_scale=4, dt=1*u.day,
                        run_2d=True)
model.solve([])

# Plot Earth timeseries
HA.plot_earth_timeseries(model)
plt.show()