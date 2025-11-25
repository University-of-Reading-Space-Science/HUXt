"""Quick test for CGF solver with streakline particles"""
import numpy as np
import astropy.units as u
import huxt.huxt as H

print("Testing CGF solver with streakline particles...")

# Simple test case
lon_grid = np.array([0, np.pi]) * u.rad

model = H.HUXt(simtime=0.5*u.day, dt_scale=4, solver='cgf')
model.solve([], streak_carr=lon_grid)

print("SUCCESS: CGF solver with streakline particles completed without errors!")
