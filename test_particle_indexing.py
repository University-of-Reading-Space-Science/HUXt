"""
Minimal test for CGF solver particle tracking with streaklines
"""
import astropy.units as u
import huxt as H

# Create minimal model
model = H.HUXt(simtime=2*u.day, dt_scale=4, solver='cgf')

# Set up streakline longitudes
lon_grid = [0, 90, 180, 270] * u.deg

# Run with streaklines (no CMEs to keep it simple)
print("Testing CGF solver with streakline particles...")
try:
    model.solve([], streak_carr=lon_grid)
    print("SUCCESS: Model solved without errors")
    print(f"Streakline particles shape: {model.streak_particles_r.shape}")
    print(f"Number of output times: {len(model.time_out)}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
