from astropy.time import Time
import astropy.units as u
import astropy.coordinates as acoords
import datetime
import h5py
import numpy as np
import requests
import sunpy.coordinates as coords

from huxt import huxt as h


def get_naif_body_codes_dict():
    """ Return a dictionary with the names and naif codes of bodies that can be looked up in Horizons for use in
    generating an offline ephemeris for HUXt."""

    bodies = {'PSP': -96,
              'SOLO': -144,
              'STA': -234,
              'STB': -235,
              'ACE': -92,
              'MERCURY': 199,
              'VENUS': 299,
              'EARTH': 399,
              'MARS': 499,
              'JUPITER': 599,
              'SATURN': 699}
    return bodies


def zerototwopi(angles):
    """
    Function to constrain angles to the 0 - 2pi domain.
    Args:
        angles: a numpy array of angles
    Returns:
        angles_out: a numpy array of angles in the 0 - 2pi domain.
    """
    twopi = 2.0 * np.pi
    angles_out = angles
    a = -np.floor_divide(angles_out, twopi)
    angles_out = angles_out + (a * twopi)
    return angles_out


def main():
    """
    This function uses SunPy's JPL Horizons integration to generate an ephemeris file for use with HUXt. This enables
    using HUXt offline. The ephemeris includes the planets out to Saturn, and has a 6-hour cadence. It also includes
    the ephemeris for ACE, STEREO-A, STEREO-B, Parker Solar Probe, and Solar Orbiter, at 3-hour cadence. JPL Horizons
    only provides ephemeris data for ACE and STEREO-A for a short window ahead (roughly 70 days and 100 days,
    respectively). So the ephemeris file will need periodically updating if HUXt is going to be used for case studies
    of new events or for forecasting.
    Returns:

    """

    dirs = h._setup_dirs_()
    ephemeris_path = dirs['ephemeris']
    print(f"Updating ephemeris file: {ephemeris_path}")
    ephem = h5py.File(ephemeris_path, 'w')

    bodies_dict = get_naif_body_codes_dict()
    for body, naif_code in bodies_dict.items():

        body_group = ephem.create_group(body)

        # Get start and stop times for the ephemeris, based on the body
        t_start = Time('1963-01-01T00:00:00').to_datetime()
        t_stop = Time('2029-01-01T00:00:00').to_datetime()
        if body == 'STA':
            t_start = Time('2007-01-01T00:00:00')
            t_stop = get_STA_latest_horizons_date()
        elif body == 'STB':
            t_start = Time('2007-01-01T00:00:00')
            t_stop = Time('2024-10-25T00:00:00').to_datetime()
        elif body == 'PSP':
            t_start = Time('2018-08-13T00:00:00')
        elif body == 'SOLO':
            t_start = Time('2020-02-11T00:00:00')
        elif body == 'ACE':
            t_start = Time('2006-07-03T00:00:00')
            t_stop = get_ACE_latest_horizons_date()

        if body in ['STA', 'STB', 'PSP', 'SOLO', 'ACE']:
            time_lookup = {'start': t_start, 'stop': t_stop, 'step': '3H'}
        else:
            time_lookup = {'start': t_start, 'stop': t_stop, 'step': '12H'}

        body_coords = coords.get_horizons_coord(naif_code, time_lookup)

        for coord_sys in ['CARR', 'HEEQ', 'HAE']:
            coord_group = body_group.create_group(coord_sys)
            coord_group.create_dataset('time', data=body_coords.obstime.jd)

            if coord_sys == 'CARR':
                this_coord = body_coords.transform_to(coords.HeliographicCarrington(observer="self"))
            elif coord_sys == 'HEEQ':
                this_coord = body_coords.transform_to(coords.HeliographicStonyhurst())
            elif coord_sys == 'HAE':
                this_coord = body_coords.transform_to(acoords.HeliocentricMeanEcliptic())

            if coord_sys == 'HAE':
                rad = coord_group.create_dataset('radius', data=this_coord.distance.to(u.km).value)
            else:
                rad = coord_group.create_dataset('radius', data=this_coord.radius.to(u.km).value)

            rad.attrs['unit'] = u.km.to_string()
            lon = coord_group.create_dataset('longitude', data=np.rad2deg(zerototwopi(this_coord.lon.radian)))
            lon.attrs['unit'] = u.deg.to_string()
            lat = coord_group.create_dataset('latitude', data=np.rad2deg(this_coord.lat.radian))
            lat.attrs['unit'] = u.deg.to_string()

            ephem.flush()

    ephem.close()
    return


def get_STA_latest_horizons_date():
    """
    This function queries JPL Horizons to find the current latest available date of ephemeris data for STEREO-A.
    Typically, JPL Horizons only has 3-4 months of planned predicted ephemeris data for STEREO-A
    """
    # In the query, -234 is the NAIF code of STEREO-A. This query URL was given in an email exchange with JPL Horizons
    # by Jon Giorgini.
    query = "https://ssd.jpl.nasa.gov/api/horizons.api?format=text&OBJ_DATA='YES'&MAKE_EPHEM='NO'&COMMAND='-234'"
    response = requests.get(query)
    # This response is some paragraphs and a table. We want the date in the bottom right corner of the table.
    lines = response.text.split("\n")
    final_line = lines[-3].split()
    final_date = final_line[-1]
    print(f"STEREO-A ephemeris data available out to {final_date}")
    ephemeris_date_limit = datetime.datetime.strptime(final_date, "%Y-%b-%d")
    # Take a day off so that we don't bump into the limit
    ephemeris_date_limit = ephemeris_date_limit - datetime.timedelta(days=1)
    return ephemeris_date_limit


def get_ACE_latest_horizons_date():
    """
    This function queries JPL Horizons to find the current latest available date of ephemeris data for ACE.
    Typically, JPL Horizons only has 2-3 months of planned predicted ephemeris data for ACE
    """
    # In the query, -92 is the NAIF code of ACE. This query URL was given in an email exchange with JPL Horizons
    # by Jon Giorgini.
    query = "https://ssd.jpl.nasa.gov/api/horizons.api?format=text&OBJ_DATA='YES'&MAKE_EPHEM='NO'&COMMAND='-92'"
    response = requests.get(query)
    # This response is some paragraphs and a table. We want the date in the bottom right corner of the table.
    lines = response.text.split("\n")
    final_line = lines[-3].split()
    final_date = final_line[-1]
    print(f"ACE ephemeris data available out to {final_date}")
    ephemeris_date_limit = datetime.datetime.strptime(final_date, "%Y-%b-%d")
    # Take a day off so that we don't bump into the limit
    ephemeris_date_limit = ephemeris_date_limit - datetime.timedelta(days=1)
    return ephemeris_date_limit


if __name__ == "__main__":

    main()
