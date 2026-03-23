import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import datetime

import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin
import huxt.huxt_insitu as Hinsitu

stime = datetime.datetime(2020,1,1)
ftime = datetime.datetime(2020,3,1)
include_cone_cmes = False
r_min = 21.5*u.solRad
r_max = 450*u.solRad
solver = 'upwind' # use 'hllc' if you need density and temperature too

#set up model using OMNI observations backmapped to 0.1 AU as the inner boundary condition
print('Setting up HUXt model with OMNI backmapped to 0.1 AU as inner boundary condition...')
model = Hinsitu.omniHUXt_reconstruction(stime, ftime, 
                        rmin=r_min, rmax=r_max, 
                        dt_scale=20, dt=1*u.day,
                        run_2d=True, solver = solver)

cme_list = []
#maybe include CMEs from the DONKI catalogue?
if include_cone_cmes:
    print('Getting CMEs from DONKI catalogue...')
    cme_list = Hin.get_DONKI_cme_list(model, stime, ftime)

#solve the model
print('Solving HUXt between {} and {}...'.format(r_min, r_max))
print('Run duration: ',stime, ' to ', ftime )
model.solve(cme_list)

# Plot Earth timeseries
HA.plot_earth_timeseries(model)

#plot Mars timeseries
ts_mars = HA.get_observer_timeseries(model, observer='Mars', suppress_warning=False)
fig, ax = plt.subplots()
ax.plot(ts_mars['time'], ts_mars['vsw'], label='Mars')

plt.show()