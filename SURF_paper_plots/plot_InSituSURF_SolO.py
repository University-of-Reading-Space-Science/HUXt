import matplotlib
matplotlib.use('TkAgg')  # Set backend explicitly for Windows
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import os
import numpy as np
import pandas as pd
import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin
import huxt.huxt_insitu as Hinsitu
import astropy.units as u
from astropy.time import Time
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.coordinates import get_horizons_coord
import sunpy.timeseries
import sunpy.coordinates
import sunpy_soar # this is required


# ============================================================
# Set up the reconstruction interval
# ============================================================
run_start = datetime.datetime(2025, 1, 15)
run_end = run_start + datetime.timedelta(days=30)

is_model = Hinsitu.omniHUXt_reconstruction(run_start, run_end,
                            rmin=21.5*u.solRad, rmax=230*u.solRad,
                            dt_scale=4, dt=1*u.day,
                            omni_input=None,
                            run_2d=True,
                            solver='hllc')

is_model.solve([])

# ============================================================
# Get Solar Orbiter coordinates
# ============================================================
body = 'Solar Orbiter'
coord = get_horizons_coord(body, {'start': Time(run_start), 'stop': Time(run_end), 'step': '1H'})
smjd = Time(run_start).mjd
fmjd = Time(run_end).mjd

coords = pd.DataFrame()
coords['r_AU'] = coord.radius.value
coords['lon_heeq'] = H.zerototwopi(coord.lon.value * np.pi / 180)
coords['lat_heeq'] = coord.lat.value * np.pi / 180
coords['mjd'] = np.linspace(smjd, fmjd, len(coords))
coords['datetime'] = Time(coords['mjd'] + 2400000.5, format='jd').to_datetime()

# ============================================================
# Download Solar Orbiter SWA-PAS-MOM data
# ============================================================
time_range = a.Time(run_start, run_end)
res = Fido.search(time_range, a.soar.Product("swa-pas-mom"))
v_files = Fido.fetch(res)
ts = sunpy.timeseries.TimeSeries(v_files, concatenate=True)
v_df = ts.to_dataframe()

# Average to 1 hour
v_df_hourly = v_df.resample('1h', label='right', closed='right').mean()
v_df_hourly.index = v_df_hourly.index - pd.Timedelta(minutes=30)

# ============================================================
# Compute mean observer latitudes and SolO distance for plot info
# ============================================================
earth_obs = is_model.get_observer('Earth')
E_lat_deg = np.nanmean(earth_obs.lat_c.to(u.deg).value)
solo_lat_deg = np.degrees(np.nanmean(coords['lat_heeq']))
solo_r_au = np.nanmean(coords['r_AU'])

# ============================================================
# Extract model solution at Earth and Solar Orbiter
# ============================================================
ts_earth = HA.get_observer_timeseries(is_model, observer='Earth')
ts_solo = HA.get_HUXt_at_position_HEEQ(is_model, coords['mjd'].values,
                                         coords['r_AU'].values * 215,
                                         coords['lon_heeq'].values)

# ============================================================
# Download OMNI data for Earth comparison
# ============================================================
omni = Hinsitu.get_omni(run_start, run_end)

# ============================================================
# Plot: Earth and Solar Orbiter comparison
# ============================================================
is_compressible = hasattr(is_model, 'compressible') and is_model.compressible

if is_compressible:
    n_panels = 6  # V, n, T at Earth + V, n, T at SolO
else:
    n_panels = 2  # V at Earth + V at SolO

fig, axs = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels), sharex=True)

xx = (run_start, run_end)
panel = 0

# --- Earth: Speed ---
ax = axs[panel]
ax.plot(omni['datetime'], omni['V'], 'k', label='OMNI', alpha=0.7)
ax.plot(ts_earth['time'], ts_earth['vsw'], 'b', label='OMNI-HUXt (Earth)')
ax.set_ylabel(r'$V_{SW}$ [km/s]')
ax.set_xlim(xx)
ax.set_ylim((250, 800))
ax.legend(loc='upper right', ncol=2, fontsize=9)
ax.set_title(f'Earth  (lat = {E_lat_deg:.1f}°)', fontsize=14)
ax.text(0.01, 0.95, f"({chr(97 + panel)})", transform=ax.transAxes, ha='left', va='top',
        fontsize=14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
panel += 1

if is_compressible:
    # --- Earth: Density ---
    ax = axs[panel]
    ax.plot(omni['datetime'], omni['N'], 'k', label='OMNI', alpha=0.7)
    ax.plot(ts_earth['time'], ts_earth['n'], 'b', label='OMNI-HUXt (Earth)')
    ax.set_ylabel(r'$n$ [cm$^{-3}$]')
    ax.set_xlim(xx)
    ax.set_yscale('log')
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    ax.text(0.01, 0.95, f"({chr(97 + panel)})", transform=ax.transAxes, ha='left', va='top',
            fontsize=14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    panel += 1

    # --- Earth: Temperature ---
    ax = axs[panel]
    ax.plot(omni['datetime'], omni['T'], 'k', label='OMNI', alpha=0.7)
    ax.plot(ts_earth['time'], ts_earth['T'], 'b', label='OMNI-HUXt (Earth)')
    ax.set_ylabel(r'$T$ [K]')
    ax.set_xlim(xx)
    ax.set_yscale('log')
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    ax.text(0.01, 0.95, f"({chr(97 + panel)})", transform=ax.transAxes, ha='left', va='top',
            fontsize=14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    panel += 1

# --- SolO: Speed ---
ax = axs[panel]
ax.plot(v_df_hourly.index, v_df_hourly['velocity_0'], 'k', label='Solar Orbiter SWA', alpha=0.7)
ax.plot(ts_solo['time'], ts_solo['vsw'], 'r', label='OMNI-SURF-hydro')
ax.set_ylabel(r'$V_{SW}$ [km/s]')
ax.set_xlim(xx)
ax.set_ylim((250, 800))
ax.legend(loc='upper right', ncol=2, fontsize=9)
ax.set_title(f'Solar Orbiter  (r = {solo_r_au:.2f} AU,  lat = {solo_lat_deg:.1f}°)', fontsize=14)
ax.text(0.01, 0.95, f"({chr(97 + panel)})", transform=ax.transAxes, ha='left', va='top',
        fontsize=14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
panel += 1

if is_compressible:
    # --- SolO: Density ---
    ax = axs[panel]
    ax.plot(v_df_hourly.index, v_df_hourly['density'], 'k', label='Solar Orbiter SWA', alpha=0.7)
    ax.plot(ts_solo['time'], ts_solo['n'], 'r', label='OMNI-SURF-hydro')
    ax.set_ylabel(r'$n$ [cm$^{-3}$]')
    ax.set_xlim(xx)
    ax.set_yscale('log')
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    ax.text(0.01, 0.95, f"({chr(97 + panel)})", transform=ax.transAxes, ha='left', va='top',
            fontsize=14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    panel += 1

    # --- SolO: Temperature ---
    # SWA temperature is in eV, convert to K (1 eV = 11604.5 K)
    ax = axs[panel]
    ax.plot(v_df_hourly.index, v_df_hourly['temperature'] * 11604.5, 'k', label='Solar Orbiter SWA', alpha=0.7)
    ax.plot(ts_solo['time'], ts_solo['T'], 'r', label='OMNI-SURF-hydro')
    ax.set_ylabel(r'$T$ [K]')
    ax.set_xlim(xx)
    ax.set_yscale('log')
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    ax.text(0.01, 0.95, f"({chr(97 + panel)})", transform=ax.transAxes, ha='left', va='top',
            fontsize=14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    panel += 1

# Format x-axis
axs[-1].xaxis.set_major_locator(mdates.DayLocator(interval=5))
axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
axs[-1].set_xlabel('Date')
fig.autofmt_xdate(rotation=30, ha='right')
fig.tight_layout()

# ============================================================
# New figure: SolO V time series (left) + dV vs observed V (right)
# ============================================================

# Interpolate model V onto the SWA hourly time axis
solo_mjd_hourly = (v_df_hourly.index - pd.Timestamp("1858-11-17")) / pd.Timedelta(days=1)
model_vsw_interp = np.interp(solo_mjd_hourly.values,
                             ts_solo['mjd'].values,
                             ts_solo['vsw'].values,
                             left=np.nan, right=np.nan)

v_obs = v_df_hourly['velocity_0'].values
v_mod = model_vsw_interp
dV = v_mod - v_obs

# Only use points where both obs and model are valid
mask = np.isfinite(v_obs) & np.isfinite(dV)
v_obs_clean = v_obs[mask]
dV_clean = dV[mask]

fig2, (ax_ts, ax_dv) = plt.subplots(1, 2, figsize=(12, 4),
                                     gridspec_kw={'width_ratios': [2, 1]})

# --- Left: V time series ---
ax_ts.plot(v_df_hourly.index, v_df_hourly['velocity_0'], 'k', alpha=0.7,
           label='Solar Orbiter SWA')
ax_ts.plot(ts_solo['time'], ts_solo['vsw'], 'r', label='OMNI-SURF-hydro')
ax_ts.set_ylabel(r'$V_{SW}$ [km/s]')
ax_ts.set_xlim(xx)
ax_ts.set_ylim((250, 800))
ax_ts.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=0, ha='center')
ax_ts.set_xlabel(f'Date ({run_start.year})')
ax_ts.legend(loc='upper right', fontsize=14)
ax_ts.set_title(f'Solar Orbiter  (r = {solo_r_au:.2f} AU,  lat = {solo_lat_deg:.1f}°)',
                fontsize=12)
ax_ts.text(0.01, 0.97, '(a)', transform=ax_ts.transAxes, ha='left', va='top',
           fontsize=13, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

# --- Right: dV vs observed V ---
ax_dv.scatter(v_obs_clean, dV_clean, s=4, alpha=0.4, color='k')
ax_dv.axhline(0, color='r', lw=1.5, ls='--')

# Bin mean and std
bin_edges = np.arange(250, 850, 50)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_mean = [np.nanmean(dV_clean[(v_obs_clean >= e0) & (v_obs_clean < e1)])
            for e0, e1 in zip(bin_edges[:-1], bin_edges[1:])]
bin_std  = [np.nanstd( dV_clean[(v_obs_clean >= e0) & (v_obs_clean < e1)])
            for e0, e1 in zip(bin_edges[:-1], bin_edges[1:])]
ax_dv.errorbar(bin_centres, bin_mean, yerr=bin_std, fmt='r-o', ms=5,
               capsize=3, label='Bin mean ± std')

ax_dv.set_xlabel(r'$V_{obs}$ [km/s]')
ax_dv.set_ylabel(r'$\Delta V$ [km/s]')
ax_dv.set_xlim((250, 750))
ax_dv.xaxis.set_major_locator(plt.MultipleLocator(100))
plt.setp(ax_dv.xaxis.get_majorticklabels(), rotation=0, ha='center')
#ax_dv.legend(fontsize=14)
ax_dv.text(0.01, 0.97, '(b)', transform=ax_dv.transAxes, ha='left', va='top',
           fontsize=13, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

fig2.tight_layout()

plt.show()