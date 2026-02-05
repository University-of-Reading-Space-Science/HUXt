# import huxt.huxt as H
# import astropy.units as u
# import datetime


# n = H.map_properties_parker(400, 215*u.solRad, 21.5*u.solRad, 5, 50000, gamma=1.5)
# t0 = datetime.datetime.now()
# n = H.map_properties_parker(400, 215*u.solRad, 21.5*u.solRad, 5, 50000, gamma=1.5)
# dt_pcm = (datetime.datetime.now() - t0).total_seconds()
# print(f"Solve complete in {dt_pcm:.2f}s!")

# print(n)



import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use('TkAgg')  # Set backend explicitly for Windows
import matplotlib.pyplot as plt
import datetime
import os
import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin
import huxt.huxt_insitu as Hinsitu


simtime = 5*u.day

print(f"Matplotlib backend: {matplotlib.get_backend()}")
plt.ion()  # Turn on interactive mode




# Set up HUXt over a limited longitude range.
dirs = H._setup_dirs_()
data_path=dirs['example_inputs']
print(data_path)
filepath_pkl = os.path.join(data_path, 'tomo_8.0_20250110.pkl')

vr_map, vr_longs, vr_lats = Hin.get_CorTom_vr_map(filepath_pkl)
print(vr_map.shape)


plt.figure()
im1 = plt.pcolormesh(vr_longs.value, vr_lats.value, vr_map.value, shading='auto')
plt.colorbar(im1, label='Vr (km/s)')
plt.xlabel('Carr lon')
plt.ylabel('Lat')

vr_in = Hin.get_CorTom_long_profile(filepath_pkl)
print(len(vr_in))
plt.figure()
plt.plot(vr_in)


filepath_dat = os.path.join(data_path, 'tomo_sta_cor2_20250101131424_8-0.dat')

vr_map, vr_longs, vr_lats = Hin.get_CorTom_vr_map(filepath_dat)
print(vr_map.shape)


plt.figure()
plt.title(filepath_dat)
im2 = plt.pcolormesh(vr_longs.value, vr_lats.value, vr_map.value, shading='auto')
plt.colorbar(im2, label='Vr (km/s)')
plt.xlabel('Carr lon')
plt.ylabel('Lat')

vr_in = Hin.get_CorTom_long_profile(filepath_dat)
print(len(vr_in))

plt.figure()
plt.plot(vr_in)
plt.xlabel('Carr lon')
plt.ylabel('Vr (km/s)') 

plt.show(block=True)
print("\nPlots displayed. Press Enter to close all plots and exit...")
input()