"""
Light source configurations and their conditioning.

The standard result is that the photometric stereo system is well conditioned
as long as the three or more light vectors are linearly independent, that is,
not along a common azimuth. The rigs below let that be tested rather than
assumed.
"""
import numpy as np


def cone_rig(m=3, slant_deg=35.0, azimuth_offset_deg=0.0):
    """
    Standard rig. m lights evenly spaced in azimuth on a cone of half angle
    slant_deg about the viewing axis z.
    """
    slant = np.deg2rad(slant_deg)
    az = np.deg2rad(azimuth_offset_deg) + np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    return np.stack([
        np.sin(slant) * np.cos(az),
        np.sin(slant) * np.sin(az),
        np.full(m, np.cos(slant)),
    ], axis=1)


def near_coplanar_rig(m=3, slant_deg=35.0, azimuth_spread_deg=180.0):
    """
    Rig for the conditioning ablation. Azimuths are squeezed into a wedge of
    azimuth_spread_deg. As the spread shrinks the lights approach a common
    azimuth and the system becomes ill conditioned.

    The wedge endpoints are open at 360 degrees, since a closed interval would
    place the first and last light at the same azimuth and quietly give one
    fewer distinct direction than requested.
    """
    slant = np.deg2rad(slant_deg)
    spread = np.deg2rad(azimuth_spread_deg)
    closed = azimuth_spread_deg < 360.0
    az = np.linspace(-spread / 2.0, spread / 2.0, m, endpoint=closed)
    return np.stack([
        np.sin(slant) * np.cos(az),
        np.sin(slant) * np.sin(az),
        np.full(m, np.cos(slant)),
    ], axis=1)


def condition_number(lights):
    """Ratio of largest to smallest singular value of the m by 3 light matrix."""
    return float(np.linalg.cond(lights))


def view_direction():
    """Orthographic camera looks down the z axis, so the view vector is +z."""
    return np.array([0.0, 0.0, 1.0])
