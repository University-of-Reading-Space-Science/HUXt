"""
test_huxt_fluxrope.py

Quick test and visualisation script for the huxt_fluxrope module.
Exercises:
  1. Axis geometry - shape for different n (front-flattening) values
  2. Cross-section radius profile along the axis
  3. Lundquist magnetic field radial profile
  4. 3D axis and field-line visualisation
  5. Simple Lundquist cylinder time series
  6. FRiED synthetic in-situ at 1 AU (static CME)
  7. FRiED synthetic in-situ with propagation + expansion
  8. Effect of twist on in-situ signature
  9. Effect of impact parameter on in-situ signature
 10. Effect of tilt angle on in-situ HEEQ components
"""

import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from huxt.huxt_fluxrope import (
    FRiEDCME,
    fri3d_axis,
    fri3d_axis_3d,
    cross_section_radius,
    lundquist_field,
    B0_from_flux,
    make_lundquist_timeseries,
    plot_fluxrope_insitu,
)

plt.ion()

# =============================================================================
# 1. Axis shape for different front-flattening coefficients n
# =============================================================================
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))
fig1.suptitle('FRiED Axis Geometry (Eq. 14, Isavnin 2016)', fontsize=13)

hw_deg = 60.0
hw_rad = np.deg2rad(hw_deg)
phi_arr = np.linspace(0, 2 * hw_rad, 400)

ax = axes1[0]
for n_val in [0.5, 0.75, 1.0, 1.5, 2.0]:
    r_norm = fri3d_axis(phi_arr, hw_rad, n=n_val)
    ax.plot(np.rad2deg(phi_arr), r_norm, label=f'n = {n_val}')
ax.set_xlabel('phi (deg)')
ax.set_ylabel('Normalised radial distance')
ax.set_title('Axis shape r(phi) for various n')
ax.legend(fontsize=9)
ax.axvline(hw_deg, color='k', linestyle='--', linewidth=0.8, label='apex')

# Polar plot of axis
ax2 = fig1.add_subplot(122, projection='polar')
for n_val in [0.5, 1.0, 2.0]:
    r_norm = fri3d_axis(phi_arr, hw_rad, n=n_val)
    ax2.plot(phi_arr - hw_rad, r_norm, label=f'n = {n_val}')
ax2.set_title('Polar axis shape', pad=15)
ax2.legend(fontsize=9, loc='lower right')
fig1.tight_layout()

# =============================================================================
# 2. Cross-section radius along the axis
# =============================================================================
fig2, ax = plt.subplots(figsize=(7, 4))
ax.set_title('Cross-section radius along the CME axis', fontsize=12)

Rp_vals = [10, 15, 20]
for Rp_val in Rp_vals:
    r_cs = cross_section_radius(phi_arr, hw_rad, Rp=Rp_val, n=1.0)
    ax.plot(np.rad2deg(phi_arr), r_cs, label=f'Rp = {Rp_val} Rs')
ax.axvline(hw_deg, color='k', linestyle='--', linewidth=0.8)
ax.set_xlabel('phi along axis (deg)')
ax.set_ylabel('Cross-section radius (solar radii)')
ax.legend()
fig2.tight_layout()

# =============================================================================
# 3. Lundquist field radial profile
# =============================================================================
fig3, ax = plt.subplots(figsize=(7, 4))
ax.set_title('Lundquist field profile (Eq. 16, Isavnin 2016)', fontsize=12)

rho_norm = np.linspace(0, 1, 200)
B_ax, B_az = lundquist_field(rho_norm, B0=20.0)
B_total = np.sqrt(B_ax**2 + B_az**2)
ax.plot(rho_norm, B_ax, label='Axial (J0)', color='C0')
ax.plot(rho_norm, B_az, label='Azimuthal (J1)', color='C1')
ax.plot(rho_norm, B_total, 'k--', label='|B| total', linewidth=1.5)
ax.axhline(0, color='grey', linewidth=0.5)
ax.set_xlabel('Normalised poloidal distance ρ/R')
ax.set_ylabel('B (nT)')
ax.legend()
fig3.tight_layout()

# =============================================================================
# 4. Magnetic flux conservation: B0 vs cross-section radius
# =============================================================================
fig4, ax = plt.subplots(figsize=(7, 4))
ax.set_title('Flux conservation: B0 varies along axis (Eq. 18)', fontsize=12)

Phi = 1e13  # Wb
R_apex_m = 15 * u.solRad.to(u.m)
B0_apex = B0_from_flux(Phi, R_apex_m)

r_cs_vals = cross_section_radius(phi_arr, hw_rad, Rp=15.0, n=1.0)
R_m = r_cs_vals * u.solRad.to(u.m)
B0_along = np.where(R_m > 0, B0_from_flux(Phi, np.maximum(R_m, 1e-10)), 0.0)

# Convert to nT
ax.plot(np.rad2deg(phi_arr), B0_along * 1e9, color='C3')
ax.set_xlabel('phi along axis (deg)')
ax.set_ylabel('Core field B0 (nT)')
ax.set_title('Flux conservation: B0 along axis (Phi = 10¹³ Wb)', fontsize=12)
fig4.tight_layout()

# =============================================================================
# 5. Simple Lundquist cylinder time series - various impact parameters
# =============================================================================
fig5, axes5 = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
fig5.suptitle('Lundquist cylinder: effect of impact parameter', fontsize=12)

labels = ['|B| total', 'Bx (radial)', 'By (tangential)', 'Bz (normal)']
for d0 in [0.0, 0.3, 0.6, 0.85]:
    t, Bx, By, Bz, Bt = make_lundquist_timeseries(
        B0=20.0, R=0.1, v_transit=400.0, impact_param=d0,
        chirality=1, polarity=1, tilt=0.0, dt=60.0)
    t_hr = t / 3600.0
    for i, comp in enumerate([Bt, Bx, By, Bz]):
        axes5[i].plot(t_hr, comp, label=f'd={d0}')

for i, lab in enumerate(labels):
    axes5[i].set_ylabel(lab + ' (nT)')
    axes5[i].axhline(0, color='grey', linewidth=0.5, linestyle='--')
axes5[0].legend(fontsize=9, ncol=4)
axes5[-1].set_xlabel('Time (hours)')
fig5.tight_layout()

# =============================================================================
# 6. Lundquist cylinder: effect of chirality and polarity
# =============================================================================
fig6, axes6 = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
fig6.suptitle('Lundquist cylinder: chirality & polarity', fontsize=12)

combos = [(1, 1, 'chi=+1, pol=+1'), (-1, 1, 'chi=-1, pol=+1'),
          (1, -1, 'chi=+1, pol=-1'), (-1, -1, 'chi=-1, pol=-1')]
for chi, pol, lbl in combos:
    t, Bx, By, Bz, Bt = make_lundquist_timeseries(
        B0=20.0, R=0.1, v_transit=400.0, impact_param=0.0,
        chirality=chi, polarity=pol, tilt=0.0, dt=60.0)
    t_hr = t / 3600.0
    for i, comp in enumerate([Bt, Bx, By, Bz]):
        axes6[i].plot(t_hr, comp, label=lbl)

for i, lab in enumerate(labels):
    axes6[i].set_ylabel(lab + ' (nT)')
    axes6[i].axhline(0, color='grey', linewidth=0.5, linestyle='--')
axes6[0].legend(fontsize=8, ncol=2)
axes6[-1].set_xlabel('Time (hours)')
fig6.tight_layout()

# =============================================================================
# 7. FRiED synthetic in-situ: static snapshot (case study params from paper)
# =============================================================================
print("Running FRiED static snapshot...")

cme_static = FRiEDCME(
    Rt=215.0 * u.solRad,
    Rp=29.3 * u.solRad,
    half_width=66.8 * u.deg,
    n=0.2,
    tilt=0.0 * u.deg,
    theta=0.0 * u.deg,
    phi_dir=0.0 * u.deg,
    delta=16.0 * u.deg,
    sigma=0.0 * u.deg,
    twist=2.0,
    Phi=1e13 * u.Wb,
    polarity=1,
    chirality=1,
    v_propagation=0.0 * u.km / u.s,
    v_expansion=0.0 * u.km / u.s,
)

# Static: spacecraft moves at CME speed through the frozen CME
v_sc = 545.0  # km/s (CME speed)
solrad_km = u.solRad.to(u.km)
Rt_km = cme_static.Rt.to(u.km).value
Rp_km = cme_static.Rp.to(u.km).value

# Crossing time estimate
t_cross = 2.0 * Rp_km / v_sc  # seconds
t_arr = np.arange(-t_cross * 1.2, t_cross * 1.2, 120.0)  # 2-minute steps

def sc_pos_static(t):
    # Spacecraft at fixed x=215 Rs, drifts through CME
    x = 215.0 - v_sc * t / solrad_km
    return (x, 0.0, 0.0)

times_s, B_heeq_s, B_mag_s = cme_static.synthetic_insitu(
    sc_pos_static,
    t_start=(t_arr[0] * u.s),
    t_end=(t_arr[-1] * u.s),
    dt=120.0 * u.s,
    evolving=False,
)

fig7, axes7 = plot_fluxrope_insitu(times_s, B_heeq_s * 1e9, B_mag=B_mag_s * 1e9,
                                    units='nT', figsize=(10, 8))
fig7.suptitle('FRiED static snapshot (no evolution)', fontsize=12)

# =============================================================================
# 8. FRiED synthetic in-situ: with toroidal propagation + poloidal expansion
# =============================================================================
print("Running FRiED evolving CME...")

v_prop = 545.0  # km/s
v_exp = 20.0    # km/s poloidal expansion

cme_evolving = FRiEDCME(
    Rt=100.0 * u.solRad,   # starts closer
    Rp=15.0 * u.solRad,
    half_width=66.8 * u.deg,
    n=0.2,
    tilt=0.0 * u.deg,
    theta=0.0 * u.deg,
    phi_dir=0.0 * u.deg,
    delta=0.0 * u.deg,  # no pancaking initially
    sigma=0.0 * u.deg,
    twist=2.0,
    Phi=1e13 * u.Wb,
    polarity=1,
    chirality=1,
    v_propagation=v_prop * u.km / u.s,
    v_expansion=v_exp * u.km / u.s,
)

# Transit time to 1 AU
t_transit_s = (215.0 * solrad_km) / v_prop
# Rp at 1 AU
Rp_at_arrival = 15.0 + v_exp * t_transit_s / solrad_km
t_cross_ev = 2.0 * (Rp_at_arrival * solrad_km) / v_prop

print(f"  Transit time: {t_transit_s/86400:.2f} days")
print(f"  Rp at 1 AU: {Rp_at_arrival:.1f} Rs")

def sc_pos_fixed(t):
    return (215.0, 0.0, 0.0)

times_e, B_heeq_e, B_mag_e = cme_evolving.synthetic_insitu(
    sc_pos_fixed,
    t_start=(t_transit_s - t_cross_ev * 1.2) * u.s,
    t_end=(t_transit_s + t_cross_ev * 1.2) * u.s,
    dt=120.0 * u.s,
    evolving=True,
)

# Shift to CME-relative time
times_e_rel = times_e - t_transit_s

fig8, axes8 = plot_fluxrope_insitu(times_e_rel, B_heeq_e * 1e9, B_mag=B_mag_e * 1e9,
                                    units='nT', figsize=(10, 8))
fig8.suptitle('FRiED evolving CME (propagation + expansion)', fontsize=12)

# =============================================================================
# 9. Effect of twist on in-situ signature
# =============================================================================
print("Running FRiED twist sweep...")

fig9, axes9 = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
fig9.suptitle('FRiED: effect of twist', fontsize=12)

for tau in [0.5, 1.0, 2.0, 4.0]:
    cme_tw = FRiEDCME(
        Rt=215.0 * u.solRad, Rp=29.3 * u.solRad,
        half_width=66.8 * u.deg, n=0.2, twist=tau,
        Phi=1e13 * u.Wb, polarity=1, chirality=1,
    )
    cme_tw.v_propagation = v_prop * u.km / u.s

    def sc_pos_1au(t):
        return (215.0 - v_prop * (t - t_transit_s) / solrad_km, 0.0, 0.0)

    times_tw, B_tw, Bmag_tw = cme_tw.synthetic_insitu(
        sc_pos_fixed,
        t_start=(t_transit_s - t_cross * 1.2) * u.s,
        t_end=(t_transit_s + t_cross * 1.2) * u.s,
        dt=120.0 * u.s,
        evolving=False,
    )
    t_rel = (times_tw - t_transit_s) / 3600.0
    for i, comp in enumerate([Bmag_tw, B_tw[:, 0], B_tw[:, 1], B_tw[:, 2]]):
        axes9[i].plot(t_rel, comp * 1e9, label=f'τ={tau}')

for i, lab in enumerate(['|B| (nT)', 'Bx (nT)', 'By (nT)', 'Bz (nT)']):
    axes9[i].set_ylabel(lab)
    if i > 0:
        axes9[i].axhline(0, color='grey', linewidth=0.5, linestyle='--')
axes9[0].legend(ncol=4, fontsize=9)
axes9[-1].set_xlabel('Time relative to CME centre (hours)')
fig9.tight_layout()

# =============================================================================
# 10. Effect of tilt angle
# =============================================================================
print("Running FRiED tilt sweep...")

fig10, axes10 = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
fig10.suptitle('FRiED: effect of flux-rope tilt angle', fontsize=12)

for tilt_deg in [0, 30, 60, 90]:
    cme_tilt = FRiEDCME(
        Rt=215.0 * u.solRad, Rp=29.3 * u.solRad,
        half_width=66.8 * u.deg, n=0.2, twist=2.0,
        tilt=tilt_deg * u.deg,
        Phi=1e13 * u.Wb, polarity=1, chirality=1,
    )
    times_tl, B_tl, Bmag_tl = cme_tilt.synthetic_insitu(
        sc_pos_fixed,
        t_start=(t_transit_s - t_cross * 1.2) * u.s,
        t_end=(t_transit_s + t_cross * 1.2) * u.s,
        dt=120.0 * u.s, evolving=False,
    )
    t_rel = (times_tl - t_transit_s) / 3600.0
    for i, comp in enumerate([Bmag_tl, B_tl[:, 0], B_tl[:, 1], B_tl[:, 2]]):
        axes10[i].plot(t_rel, comp * 1e9, label=f'γ={tilt_deg}°')

for i, lab in enumerate(['|B| (nT)', 'Bx (nT)', 'By (nT)', 'Bz (nT)']):
    axes10[i].set_ylabel(lab)
    if i > 0:
        axes10[i].axhline(0, color='grey', linewidth=0.5, linestyle='--')
axes10[0].legend(ncol=4, fontsize=9)
axes10[-1].set_xlabel('Time relative to CME centre (hours)')
fig10.tight_layout()

# =============================================================================
# 11. 3D structure plot
# =============================================================================
print("Plotting 3D structure...")
from huxt.huxt_fluxrope import plot_fluxrope_3d

cme_3d = FRiEDCME(
    Rt=100.0 * u.solRad, Rp=14.0 * u.solRad,
    half_width=60.0 * u.deg, n=0.5, tilt=20.0 * u.deg,
    delta=15.0 * u.deg, sigma=5.0 * u.deg,
    twist=2.0, Phi=1e13 * u.Wb, polarity=1, chirality=1,
)
fig11, ax11 = plot_fluxrope_3d(cme_3d, n_field_lines=15)
ax11.set_title('FRiED 3D CME (Rt=100 Rs, tilt=20°, delta=15°)', fontsize=11)

print("\nAll plots complete.")
plt.ioff()
plt.show()
