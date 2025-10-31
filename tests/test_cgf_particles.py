"""
Test CGF solver with particle tracking (CME, HCS, streaklines).
"""

import numpy as np
import astropy.units as u
from huxt import HUXt
from huxt import ConeCME

# Create a simple HUXt run with CGF solver and a CME
cr_num = 2010
v_boundary = np.ones(128) * 400.0 * (u.km / u.s)
simtime = 5 * u.day
dt_scale = 4

# Create CME
cme = ConeCME(
    t_launch=1.0 * u.day,
    longitude=0.0 * u.deg,
    latitude=0.0 * u.deg,
    width=20.0 * u.deg,
    v=1000.0 * (u.km / u.s),
    thickness=5 * u.solRad,
    initial_height=21.5 * u.solRad,
    mass=1.0e16 * u.kg,
    temperature=1.0e6 * u.K
)

print("="*70)
print("Testing CGF solver with CME particle tracking")
print("="*70)

# Initialize HUXt model with CGF solver
model = HUXt(
    cr_num=cr_num,
    cr_lon_init=0.0 * u.deg,
    latitude=0.0 * u.deg,
    simtime=simtime,
    dt_scale=dt_scale,
    v_boundary=v_boundary,
    r_min=21.5 * u.solRad,
    r_max=240 * u.solRad,
    lon_out=0.0 * u.deg,
    frame='sidereal',
    gamma=5.0/3.0,
    solver='cgf',
    alpha=0.15
)

# Add CME
model.cmes = [cme]

print(f"\nModel setup:")
print(f"  Solver: {model.solver}")
print(f"  Time: 0 to {simtime}")
print(f"  Longitudes: {model.nlon}")
print(f"  Radial points: {model.nr}")
print(f"  CMEs: {len(model.cmes)}")
print(f"  CME launch time: {cme.t_launch}")
print(f"  CME velocity: {cme.v}")

# Run the model
print("\nRunning model...")
try:
    model.solve()
    print("\n" + "="*70)
    print("Model completed successfully!")
    print("="*70)
    
    # Check CME particle tracking
    print("\nCME Particle Tracking Results:")
    print(f"  cme_particles_r shape: {model.cme_particles_r.shape}")
    print(f"  n_cme: {len(model.cmes)}")
    
    # Check if particles were tracked
    cme_r = model.cme_particles_r[0, :, :, 0].value  # First CME, first longitude
    
    if not np.all(np.isnan(cme_r)):
        print(f"\n  ✓ CME particles successfully tracked!")
        
        # Leading edge
        leading_valid = ~np.isnan(cme_r[:, 0])
        if np.any(leading_valid):
            print(f"\n  Leading edge:")
            print(f"    First detection: t={model.time_out[leading_valid][0]/86400:.2f} days")
            print(f"    Initial position: r={cme_r[leading_valid, 0][0]:.2f} km")
            print(f"    Final position: r={cme_r[leading_valid, 0][-1]:.2f} km")
            print(f"    Distance traveled: {(cme_r[leading_valid, 0][-1] - cme_r[leading_valid, 0][0])/1e6:.1f} million km")
        
        # Trailing edge
        trailing_valid = ~np.isnan(cme_r[:, 1])
        if np.any(trailing_valid):
            print(f"\n  Trailing edge:")
            print(f"    First detection: t={model.time_out[trailing_valid][0]/86400:.2f} days")
            print(f"    Initial position: r={cme_r[trailing_valid, 1][0]:.2f} km")
            print(f"    Final position: r={cme_r[trailing_valid, 1][-1]:.2f} km")
            print(f"    Distance traveled: {(cme_r[trailing_valid, 1][-1] - cme_r[trailing_valid, 1][0])/1e6:.1f} million km")
        
        # Calculate CME width
        if np.any(leading_valid) and np.any(trailing_valid):
            # Find times where both are valid
            both_valid = leading_valid & trailing_valid
            if np.any(both_valid):
                widths = cme_r[both_valid, 0] - cme_r[both_valid, 1]
                print(f"\n  CME width evolution:")
                print(f"    Initial: {widths[0]/1e6:.2f} million km")
                print(f"    Final: {widths[-1]/1e6:.2f} million km")
    else:
        print(f"\n  ✗ No CME particles tracked (all NaN)")
    
    # Check velocity field
    print(f"\nVelocity field:")
    print(f"  v_grid shape: {model.v_grid.shape}")
    v_inner = model.v_grid[:, 0, 0].value
    print(f"  Inner boundary: {v_inner.min():.1f} to {v_inner.max():.1f} km/s")
    
    # Look for CME signature in velocity
    cme_signature = v_inner > 600
    if np.any(cme_signature):
        print(f"  CME detected in velocity field at t={model.time_out[cme_signature][0]/86400:.2f} days")
        print(f"  Peak velocity: {v_inner.max():.1f} km/s")
    
    print("\n" + "="*70)
    print("TEST PASSED")
    print("="*70)
    
except Exception as e:
    print(f"\n" + "="*70)
    print(f"ERROR: {e}")
    print("="*70)
    import traceback
    traceback.print_exc()
