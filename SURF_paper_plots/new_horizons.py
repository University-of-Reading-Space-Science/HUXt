# New Horizons solar wind study - 2013
# Runs omniSURF_reconstruction with three solvers (huxt, huxt-pui, hydro) and
# plots the modelled solar wind speed at New Horizons on a single figure.

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SURF_DIR = os.path.join(PROJECT_ROOT, 'surf')
if SURF_DIR not in sys.path:
    sys.path.insert(0, SURF_DIR)

import datetime
import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import surf
import surf_analysis as surfA
import surf_insitu as surfIS

# ── New Horizons NAIF code ────────────────────────────────────────────────────
NH_NAIF_CODE = -98
NH_NAME = 'New Horizons'

# ── Run period ────────────────────────────────────────────────────────────────
run_start = datetime.datetime(2012, 1, 1)
run_end   = datetime.datetime(2020, 1, 1)

# ── Solvers to compare ────────────────────────────────────────────────────────
SOLVERS = [
    ('huxt',     'HUXt',         'red',   '-'),
    ('huxt-pui', 'HUXt + PUI',   'blue',  '-'),
    ('hydro',    'Hydro (HLLC)', 'black', '-'),
]

# ── Step 1: New Horizons ephemeris ────────────────────────────────────────────
t_start_astro = Time(run_start)
t_stop_astro  = Time(run_end)

print("Fetching New Horizons ephemeris from JPL Horizons ...")
nh_pos = surfA.get_horizons_body_for_SURF(t_start_astro, t_stop_astro,
                                          step='1d',
                                          naif_code=NH_NAIF_CODE,
                                          body_name=NH_NAME)

r_max_rs = nh_pos['r_rs'].max()
print(f"New Horizons max distance: {r_max_rs:.0f} Rs "
      f"({r_max_rs * 695700 / 1.496e8:.2f} AU)")

rmax_model = np.ceil(r_max_rs / 50) * 50 * u.solRad
print(f"Model rmax = {rmax_model}\n")

# ── Step 2: run each solver and extract the time series ──────────────────────
results = {}   # solver_name -> DataFrame

for solver, label, colour, ls in SOLVERS:
    print(f"[{label}] Building model ...")
    model = surfIS.omniSURF_reconstruction(
        run_start, run_end,
        rmin=21.5 * u.solRad,
        rmax=rmax_model,
        dt_scale=50,
        dt=1 * u.day,
        run_2d=True,
        solver=solver,
        rho_source='speed',
        temp_source='speed',
    )
    print(f"[{label}] Solving (chunked) ...")
    surf.solve_chunked(model, [], chunk_simtime=100*u.day)

    print(f"[{label}] Extracting time series at New Horizons ...")
    ts = surfA.get_SURF_at_position_HEEQ(model,
                                          nh_pos['mjd'],
                                          nh_pos['r_rs'],
                                          nh_pos['lon_rad'])
    results[solver] = ts
    print(f"[{label}] Done.\n")

# ── Step 3: plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

for solver, label, colour, ls in SOLVERS:
    ts = results[solver]
    ax.plot(ts['time'], ts['vsw'], color=colour, linestyle=ls,
            linewidth=1.5, label=label)

# Grey out first 6 months as spin-up
spinup_end = run_start + datetime.timedelta(days=182)
ax.axvspan(run_start, spinup_end, color='grey', alpha=0.2, zorder=0)
ax.axvline(spinup_end, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)

ax.set_ylabel('V$_{sw}$ (km s$^{-1}$)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylim(250, 500)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
fig.autofmt_xdate(rotation=0, ha='center')

# Annotate with NH distance
r_au = nh_pos['r_rs'] * 695700 / 1.496e8
ax2 = ax.twinx()
ax2.plot(results[list(results.keys())[0]]['time'], r_au[:len(results[list(results.keys())[0]])],
         'grey', linewidth=0.8, linestyle='-', alpha=0.5)
ax2.set_ylabel('New Horizons distance (AU)', fontsize=10, color='grey')
ax2.tick_params(axis='y', labelcolor='grey')

fig.suptitle('SURF solar wind speed at New Horizons – 2013\n(solver comparison)',
             fontsize=13)
plt.tight_layout()
fig_path = os.path.join(SCRIPT_DIR, 'new_horizons_2013_solver_comparison.png')
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"Figure saved to {fig_path}")

