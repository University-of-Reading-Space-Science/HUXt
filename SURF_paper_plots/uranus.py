import os
import sys
import datetime

import astropy.units as u
from astropy.time import Time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SURF_DIR = os.path.join(PROJECT_ROOT, 'surf')
if SURF_DIR not in sys.path:
    sys.path.insert(0, SURF_DIR)

import surf
import surf_analysis as surfA
import surf_insitu as surfIS


URANUS_NAIF_CODE = 799
URANUS_NAME = 'Uranus'

run_start = datetime.datetime(2021, 1, 1)
run_end = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

SOLVERS = [
    ('huxt', 'HUXt', 'red', '-'),
    ('huxt-pui', 'HUXt + PUI', 'blue', '-'),
    ('hydro', 'Hydro (HLLC)', 'black', '-'),
]

CHUNK_SIMTIME = 120 * u.day
EPHEMERIS_STEP = '1d'


def build_output_table(ephem, results):
    hydro_ts = results['hydro']
    output = pd.DataFrame({
        'time_iso': pd.to_datetime(hydro_ts['time']).dt.strftime('%Y-%m-%dT%H:%M:%S'),
        'time_mjd': hydro_ts['mjd'].astype(float),
        'uranus_r_rs': np.asarray(ephem['r_rs'], dtype=float),
        'uranus_r_au': np.asarray(ephem['r_rs'], dtype=float) * 695700.0 / 1.496e8,
        'uranus_lon_rad': np.asarray(ephem['lon_rad'], dtype=float),
        'huxt_vsw_kms': results['huxt']['vsw'].astype(float),
        'huxt_pui_vsw_kms': results['huxt-pui']['vsw'].astype(float),
        'hydro_vsw_kms': results['hydro']['vsw'].astype(float),
        'hydro_n_cm3': results['hydro']['n'].astype(float),
        'hydro_T_K': results['hydro']['T'].astype(float),
    })
    return output


def main():
    print(f'Running Uranus study from {run_start:%Y-%m-%d} to {run_end:%Y-%m-%d}')

    t_start_astro = Time(run_start)
    t_stop_astro = Time(run_end)

    print('Fetching Uranus ephemeris from JPL Horizons ...')
    uranus_pos = surfA.get_horizons_body_for_SURF(
        t_start_astro,
        t_stop_astro,
        step=EPHEMERIS_STEP,
        naif_code=URANUS_NAIF_CODE,
        body_name=URANUS_NAME,
    )

    r_max_rs = np.nanmax(uranus_pos['r_rs'])
    print(f'Uranus max distance: {r_max_rs:.0f} Rs ({r_max_rs * 695700.0 / 1.496e8:.2f} AU)')

    rmax_model = np.ceil(r_max_rs / 50.0) * 50.0 * u.solRad
    print(f'Model rmax = {rmax_model}\n')

    results = {}

    for solver, label, colour, ls in SOLVERS:
        print(f'[{label}] Building model ...')
        if solver in ('hydro', 'hydro-pcm'):
            model = surfIS.omniSURF_1au_out(
                run_start,
                run_end,
                rmax=rmax_model,
                dt_scale=50,
                dt=1 * u.day,
                run_2d=True,
                solver=solver,
            )
        else:
            model = surfIS.omniSURF_reconstruction(
                run_start,
                run_end,
                rmin=21.5 * u.solRad,
                rmax=rmax_model,
                dt_scale=50,
                dt=1 * u.day,
                run_2d=True,
                solver=solver,
            )

        print(f'[{label}] Solving (chunked, {CHUNK_SIMTIME.to(u.day).value:.0f}-day chunks) ...')
        surf.solve_chunked(model, [], chunk_simtime=CHUNK_SIMTIME)

        print(f'[{label}] Extracting time series at Uranus ...')
        ts = surfA.get_SURF_at_position_HEEQ(
            model,
            uranus_pos['mjd'],
            uranus_pos['r_rs'],
            uranus_pos['lon_rad'],
        )
        results[solver] = ts
        print(f'[{label}] Done.\n')

    output = build_output_table(uranus_pos, results)
    csv_path = os.path.join(SCRIPT_DIR, 'uranus_2021_now_timeseries.csv')
    output.to_csv(csv_path, index=False)
    print(f'Wrote CSV to {csv_path}')

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    speed_ax = axes[0]
    for solver, label, colour, ls in SOLVERS:
        ts = results[solver]
        speed_ax.plot(ts['time'], ts['vsw'], color=colour, linestyle=ls,
                      linewidth=1.5, label=label)
    speed_ax.set_ylabel('V$_{sw}$ (km s$^{-1}$)')
    speed_ax.set_title(f'SURF at Uranus ({run_start:%Y-%m-%d} to {run_end:%Y-%m-%d})')
    speed_ax.grid(True, alpha=0.3)
    speed_ax.legend(fontsize=10)

    density_ax = axes[1]
    density_ax.plot(results['hydro']['time'], results['hydro']['n'], color='black', linewidth=1.5)
    density_ax.set_ylabel('n (cm$^{-3}$)')
    density_ax.set_yscale('log')
    density_ax.grid(True, alpha=0.3)
    density_ax.set_title('Hydro density')

    temp_ax = axes[2]
    temp_ax.plot(results['hydro']['time'], results['hydro']['T'], color='black', linewidth=1.5)
    temp_ax.set_ylabel('T (K)')
    temp_ax.set_xlabel('Date')
    temp_ax.set_yscale('log')
    temp_ax.grid(True, alpha=0.3)
    temp_ax.set_title('Hydro temperature')

    temp_ax.xaxis.set_major_locator(mdates.YearLocator())
    temp_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    temp_ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.autofmt_xdate(rotation=0, ha='center')

    fig.tight_layout()
    fig_path = os.path.join(SCRIPT_DIR, 'uranus_2021_now_timeseries.png')
    fig.savefig(fig_path, dpi=150)
    print(f'Figure saved to {fig_path}')
    plt.show()


if __name__ == '__main__':
    main()