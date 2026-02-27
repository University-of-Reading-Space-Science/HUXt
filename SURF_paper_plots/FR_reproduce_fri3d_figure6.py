"""
reproduce_fri3d_figure6.py

Reproduces Figure 6 of Isavnin (2016, ApJ, 833, 267) — FRiED model.

Three panels of synthetic in-situ HEEQ magnetic field at L1 (215 Rs):
  Top:    Static (non-evolving) CME — spacecraft traverses frozen CME at v_CME.
          This is a spatial snapshot converted to time: v_spacecraft -> infinity.
  Middle: Propagating CME — Rt grows at v_prop, Rp held fixed.
          Toroidal flux conservation weakens B_azimuthal as the axis stretches,
          producing the front-loaded asymmetry ("magnetic expansion").
  Bottom: Propagating + expanding CME — Rt grows at v_prop, Rp grows at v_exp.
          Field decreases faster because BOTH axis stretching AND cross-section
          growth dilute the flux.

Figure 6 model parameters (from text of Isavnin 2016):
    lambda   = 40 deg      (angular half-width, phi_hw)
    Rt       = 215 Rs      (toroidal height at epoch / observation time)
    Rp       = 0.15 AU     (poloidal height at epoch)
    theta    = 0 deg       (HEEQ latitude)
    phi      = 0 deg       (HEEQ longitude, Sun-Earth line)
    gamma    = 0 deg       (tilt)
    n        = 0.5         (front-flattening)
    delta    = 30 deg      (pancaking, theta_p)
    sigma    = 0 deg       (skewing, phi_S)
    tau      = 3 turns     (twist)
    Phi      = 5e14 Wb     (magnetic flux)
    polarity = +1          (east-west)
    chirality= +1          (right-handed)
    v_prop   = 545 km/s
    v_exp    = 10 km/s     (for bottom panel only)
"""

import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from huxt.huxt_fluxrope import FRiEDCME, B0_from_flux

# =============================================================================
# Shared CME parameters (Isavnin 2016, Figure 6)
# =============================================================================
PARAMS = dict(
    Rt=215.0 * u.solRad,
    Rp=0.15 * u.AU,
    half_width=40.0 * u.deg,
    n=0.5,
    tilt=5.0 * u.deg,
    theta=5.0 * u.deg,
    phi_dir=5.0 * u.deg,
    delta=30.0 * u.deg,
    sigma=5.0 * u.deg,
    twist=3.0,
    Phi=5e14 * u.Wb,
    polarity=1,
    chirality=1,
)

V_PROP = 545.0  # km/s
V_EXP  = 10.0   # km/s (poloidal expansion for bottom panel)

SOLRAD_KM = u.solRad.to(u.km)  # km per solar radius
R_L1 = 215.0                    # solar radii

DT = 120.0  # time step in seconds

# =============================================================================
# Time window helper
# =============================================================================
def crossing_window(Rp_km, v_km_s, pad_frac=0.5):
    """Return (t_start, t_end) seconds, centred on apex crossing at t=0."""
    t_half = Rp_km / v_km_s
    pad = pad_frac * t_half
    return -(t_half + pad), (t_half + pad)


Rp_km = PARAMS['Rp'].to(u.km).value
t0, t1 = crossing_window(Rp_km, V_PROP)

# =============================================================================
# Panel 1: Static (non-evolving) snapshot
#
# The CME is frozen at Rt = 215 Rs. The spacecraft moves at v_CME in the -x
# direction through the CME cross-section near the apex (equivalent to the
# CME sweeping over a fixed spacecraft at infinite speed — a spatial snapshot).
# At t = 0: spacecraft coincides with the CME apex (x = 215 Rs).
# No Rt_ref: field is the same everywhere because nothing evolves.
# =============================================================================
print("Panel 1: Static snapshot...")

cme_static = FRiEDCME(**PARAMS, v_propagation=0.0 * u.km / u.s,
                       v_expansion=0.0 * u.km / u.s)

t_arr_static = np.arange(t0, t1 + DT, DT)

n_static = len(t_arr_static)
B_static = np.zeros((n_static, 3))
for i, t in enumerate(t_arr_static):
    # Spacecraft moves at v_prop in -x direction through frozen CME
    x = R_L1 - V_PROP * t / SOLRAD_KM
    B_static[i] = cme_static.magnetic_field_at_point(x, 0.0, 0.0)

B_static_nT = B_static * 1e9
Bmag_static = np.sqrt(np.sum(B_static_nT**2, axis=1))

# =============================================================================
# Panel 2: Propagating CME, Rp fixed (dynamic pancaking / magnetic expansion)
#
# Spacecraft fixed at (215 Rs, 0, 0).
# Rt(t) = 215 + v_prop * t   =>  t = 0: apex at 1 AU.
# Rp and delta are constant.
#
# Key physics: as Rt grows, the axis length L grows proportionally. Toroidal
# flux conservation requires B_azimuthal ~ 1/L ~ 1/Rt.  This weakens the
# poloidal field as the CME propagates outward, so the FRONT of the
# crossing (when Rt was smaller) is stronger than the BACK (Rt larger).
# =============================================================================
print("Panel 2: Propagating CME (fixed Rp)...")

cme_prop = FRiEDCME(**PARAMS, v_propagation=V_PROP * u.km / u.s,
                     v_expansion=0.0 * u.km / u.s)

t_arr_prop = np.arange(t0, t1 + DT, DT)

n_prop = len(t_arr_prop)
B_prop = np.zeros((n_prop, 3))
Rt_ref = PARAMS['Rt'].value  # reference Rt for magnetic expansion

for i, t in enumerate(t_arr_prop):
    Rt_t, Rp_t = cme_prop._evolve_params(t)
    B_prop[i] = cme_prop.magnetic_field_at_point(R_L1, 0.0, 0.0,
                                                  Rt=Rt_t, Rp=Rp_t,
                                                  Rt_ref=Rt_ref)
B_prop_nT = B_prop * 1e9
Bmag_prop = np.sqrt(np.sum(B_prop_nT**2, axis=1))

# =============================================================================
# Panel 3: Propagating + expanding CME
#
# Rt(t) = 215 + v_prop * t
# Rp(t) = 14  + v_exp  * t
#
# Two effects combine:
#   1. Toroidal flux conservation (B_az ~ 1/Rt) — same as Panel 2.
#   2. Cross-section growth: B0 ~ 1/Rp^2 from poloidal flux conservation,
#      so as Rp grows the ENTIRE field weakens more rapidly.
# Both shift the field maximum toward the start of measurement.
# =============================================================================
print("Panel 3: Propagating + expanding CME...")

cme_exp = FRiEDCME(**PARAMS, v_propagation=V_PROP * u.km / u.s,
                    v_expansion=V_EXP * u.km / u.s)

# Rp is larger at arrival => extend time window
Rp_at_arrival = Rp_km + V_EXP * (-t0)
t0_exp, t1_exp = crossing_window(Rp_at_arrival, V_PROP, pad_frac=0.5)

t_arr_exp = np.arange(t0_exp, t1_exp + DT, DT)

n_exp = len(t_arr_exp)
B_exp = np.zeros((n_exp, 3))

for i, t in enumerate(t_arr_exp):
    Rt_t, Rp_t = cme_exp._evolve_params(t)
    B_exp[i] = cme_exp.magnetic_field_at_point(R_L1, 0.0, 0.0,
                                                Rt=Rt_t, Rp=Rp_t,
                                                Rt_ref=Rt_ref)
B_exp_nT = B_exp * 1e9
Bmag_exp = np.sqrt(np.sum(B_exp_nT**2, axis=1))

# Print diagnostic information
R_apex_m = PARAMS['Rp'].to(u.m).value
B0_apex = B0_from_flux(PARAMS['Phi'].value, R_apex_m)
Rt_lead = R_L1 + V_PROP * t0 / SOLRAD_KM
Rt_trail = R_L1 + V_PROP * t1 / SOLRAD_KM
print(f"\nCore field B0 at apex (reference): {B0_apex*1e9:.1f} nT")
print(f"Static crossing time: {(t1-t0)/3600:.1f} h")
print(f"Panel 2 Rt range: {Rt_lead:.1f} - {Rt_trail:.1f} Rs")
print(f"  Self-similar B scaling at leading edge: (215/{Rt_lead:.1f})^2 = "
      f"{(215/Rt_lead)**2:.3f}")
print(f"  Self-similar B scaling at trailing edge: (215/{Rt_trail:.1f})^2 = "
      f"{(215/Rt_trail)**2:.3f}")
print(f"  Front/back field ratio: {(Rt_trail/Rt_lead)**2:.2f}")
print(f"Panel 3 Rp at leading edge: {Rp_at_arrival/SOLRAD_KM:.1f} Rs")

# Verify asymmetry: print max |B| position
idx_peak_static = np.argmax(Bmag_static)
idx_peak_prop = np.argmax(Bmag_prop)
idx_peak_exp = np.argmax(Bmag_exp)
print(f"\nPeak |B| positions (hours from apex):")
print(f"  Panel 1 (static): t = {t_arr_static[idx_peak_static]/3600:.2f} h, "
      f"|B| = {Bmag_static[idx_peak_static]:.1f} nT")
print(f"  Panel 2 (prop):   t = {t_arr_prop[idx_peak_prop]/3600:.2f} h, "
      f"|B| = {Bmag_prop[idx_peak_prop]:.1f} nT")
print(f"  Panel 3 (exp):    t = {t_arr_exp[idx_peak_exp]/3600:.2f} h, "
      f"|B| = {Bmag_exp[idx_peak_exp]:.1f} nT")

# =============================================================================
# Plot — reproduce Figure 6 layout
# =============================================================================
fig = plt.figure(figsize=(8, 12))
gs = gridspec.GridSpec(3, 1, hspace=0.35)

panel_data = [
    (t_arr_static / 3600, B_static_nT, Bmag_static,
     'Static CME snapshot (no evolution)'),
    (t_arr_prop / 3600, B_prop_nT, Bmag_prop,
     f'Propagating CME ($v_{{prop}}$={V_PROP:.0f} km/s, $R_p$ fixed)'),
    (t_arr_exp / 3600, B_exp_nT, Bmag_exp,
     f'Propagating + expanding ($v_{{exp}}$={V_EXP:.0f} km/s)'),
]

COLORS = {'|B|': 'k', 'Bx': '#1f77b4', 'By': '#ff7f0e', 'Bz': '#2ca02c'}

for row, (t_hr, B_nT, Bmag, title) in enumerate(panel_data):
    ax = fig.add_subplot(gs[row])
    ax.plot(t_hr, Bmag,       color=COLORS['|B|'], linewidth=1.8,
            label='|B|', zorder=4)
    ax.plot(t_hr, B_nT[:, 0], color=COLORS['Bx'],  linewidth=1.2, label='Bx')
    ax.plot(t_hr, B_nT[:, 1], color=COLORS['By'],  linewidth=1.2, label='By')
    ax.plot(t_hr, B_nT[:, 2], color=COLORS['Bz'],  linewidth=1.2, label='Bz')
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--', zorder=0)

    ax.set_ylabel('B (nT)', fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.legend(loc='upper right', fontsize=9, ncol=4)
    ax.set_xlim(t_hr[0], t_hr[-1])

    # Shade the CME body
    inside = Bmag > 0
    if inside.any():
        t_enter = t_hr[np.argmax(inside)]
        t_exit  = t_hr[len(inside) - 1 - np.argmax(inside[::-1])]
        ax.axvspan(t_enter, t_exit, alpha=0.07, color='steelblue', zorder=0)

fig.text(0.5, 0.02, 'Time (hours) relative to apex at L1', ha='center',
         fontsize=11)
fig.text(0.01, 0.5, 'Magnetic field (nT) — HEEQ', va='center',
         rotation='vertical', fontsize=11)
fig.suptitle(
    'FRiED synthetic in-situ magnetic field\n'
    '(Reproducing Figure 6 of Isavnin 2016, ApJ 833 267)',
    fontsize=12, y=0.99,
)

plt.savefig('fri3d_figure6_reproduction.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: fri3d_figure6_reproduction.png")
plt.show()
