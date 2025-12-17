import time
import warnings
from logging import getLogger
from typing import Callable

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.time import Time

from astroplan import Observer

from aas2rto import utils
from aas2rto.observatory.ephem_info import EphemInfo
from aas2rto.exc import InvalidEphemWarning
from aas2rto.target import Target

logger = getLogger("default_obs_scoring")


def compute_visibility_integral(
    t_grid: np.ndarray,
    alt_grid: np.ndarray,
    min_alt: float = 30.0,
    ref_alt: float = 90.0,
):
    # Make sure it's always at least zero...
    alt_above_minimum = np.maximum(alt_grid, min_alt) - min_alt

    integral = np.trapezoid(alt_above_minimum, x=t_grid)
    norm = (t_grid[-1] - t_grid[0]) * (ref_alt - min_alt)
    return integral / norm  # invert later!


def calc_visibility_factor(
    ephem_info: EphemInfo, t_ref: Time, min_alt: float = 30.0, ref_alt: float = 90.0
):
    t_ref = t_ref or Time.now()

    if t_ref > ephem_info.t_grid[-1]:
        t_ref_str = t_ref.strftime("%y-%m-%d %H:%M")
        last_ephem_str = ephem_info.t_grid[-1].strftime("%y-%m-%d %H:%M")
        msg = f"{t_ref_str} (t_ref) is AFTER latest ephem_info {last_ephem_str}"
        warnings.warn(InvalidEphemWarning(msg))
        return -1.0, [msg]

    # If it's actually already past sunset, pick NOW.
    t_start = max(t_ref, ephem_info.sunset)
    t_grid = ephem_info.t_grid
    t_stop = ephem_info.sunrise

    night_mask = (t_start < t_grid) & (t_grid < t_stop)

    relevant_alt = ephem_info.target_altaz.alt.deg[night_mask]  # alt during NIGHT.
    relevant_mjd = t_grid.mjd[night_mask]

    comments = []
    if all(relevant_alt <= min_alt):
        comments.append(f"always < min_alt ({min_alt:.1f}deg)")
        comments.append(f"set vis_factor=-1.0")
        vis_factor = -1.0
    else:
        norm_integral = compute_visibility_integral(
            relevant_mjd, relevant_alt, min_alt=min_alt, ref_alt=ref_alt
        )

        vis_factor = 1.0 / norm_integral
        comments.append(f"vis_factor={vis_factor:.3f}")
        if not np.isfinite(vis_factor):
            comments.append(f"set non-finite vis_factor=-1.0")
            vis_factor = -1.0

    return vis_factor, comments


def compute_inv_airmass(alt: float):
    """
    This is just sin(alt), with alt in deg.
    motivated by X~sec(90-alt), where X is airmass.
    promote high alt, so low airmass --> 1/X ~ sin(alt)
    """
    return np.sin(alt * np.pi / 180.0)


def calc_altitude_factor(ephem_info: EphemInfo, t_ref: Time, min_alt: float = 30.0):
    # If it's actually already past sunset, pick NOW.
    t_start = max(t_ref, ephem_info.sunset)

    t_remain_mask = ephem_info.t_grid > t_start
    altaz_grid = ephem_info.target_altaz[t_remain_mask]
    if len(altaz_grid) == 0:
        t_ref_str = t_ref.strftime("%y-%m-%d %H:%M")
        last_ephem_str = ephem_info.t_grid[-1].strftime("%y-%m-%d %H:%M")
        msg = f"{t_ref_str} (t_ref) is AFTER latest ephem_info  {last_ephem_str}"
        warnings.warn(InvalidEphemWarning(msg))
        return -1.0, [msg]

    curr_alt = altaz_grid.alt.deg[0]  # the first alt!

    comments = []
    if curr_alt > min_alt:
        factor = compute_inv_airmass(curr_alt)
        alt_comm = f"alt_factor={factor:.3f} alt={curr_alt:.1f}"
        comments.append(alt_comm)
    else:
        factor = -1.0
        comm = f"alt_factor=-1.0: curr_alt {curr_alt:.1f} < {min_alt:1f} (min)"
        comments.append(comm)
    return factor, comments


class DefaultObservatoryScoring:
    """
    Doc string here

    """

    __name__ = "default_observatory_scoring"

    def __init__(
        self,
        min_altitude: float = 30.0,
        ref_altitude: float = 90.0,
        moon_sep: str = 30.0,
        method: str = "visibility",
    ):
        self.min_altitude = min_altitude
        self.ref_altitude = ref_altitude
        self.moon_sep = moon_sep
        self.method = method

        exp_methods = ["visibility", "altitude"]
        if self.method not in exp_methods:
            msg = f"'method' should be one of {exp_methods},\n    not {method}"
            raise ValueError(msg)

    def __call__(self, target: Target, observatory: Observer, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if observatory is None:
            raise TypeError(
                "observatory must be of type `astroplan.Observer`, not `None`"
            )
        obs_name = utils.get_observatory_name(observatory)

        target_id = target.target_id

        scoring_comments = []
        factors = {}
        # reject = False  # Observatory factors should NOT reject targets.
        exclude = False

        # Get the ephem_info
        ephem_info: EphemInfo = target.ephem_info.get(obs_name, None)
        if ephem_info is None:
            msg = f"no ephem_info for {target_id} at observatory '{obs_name}'"
            logger.warning(msg)
            return -1.0, [msg]

        if ephem_info.sunset is None or ephem_info.sunrise is None:
            scoring_comments.append(f"sun never sets at this observatory")

        if self.method == "altitude":
            scoring_comments.append("use method='altitude'")
            alt_factor, comms = calc_altitude_factor(
                ephem_info, t_ref, min_alt=self.min_altitude
            )
            if alt_factor < 0.0:
                exclude = True
            else:
                factors["alt_factor"] = alt_factor
            scoring_comments.extend(comms)
        elif self.method == "visibility":
            scoring_comments.append("use method='visibility'")
            vis_factor, comms = calc_visibility_factor(
                ephem_info, t_ref, min_alt=self.min_altitude, ref_alt=self.ref_altitude
            )
            if vis_factor < 0.0:
                exclude = True
            else:
                factors["vis_factor"] = vis_factor
            scoring_comments.extend(comms)
        else:
            comm = f"unexpected method {self.method}: vis_factor = 1.0"
            scoring_comments.append(comm)
            logger.warning(comm)
            factors["bad_method"] = 1.0

        scoring_factors = np.array(list(factors.values()))
        if not all(scoring_factors > 0):
            neg_factors = "\n".join(
                f"    {k}={v}" for k, v in factors.items() if not v > 0
            )
            scoring_comments.append(neg_factors)
            logger.warning(f"{target_id} exclude: negative factors:\n{neg_factors}")
            exclude = True
        if not all(np.isfinite(scoring_factors)):
            inf_factors = "\n".join(
                f"    {k}={v}" for k, v in factors.items() if not np.isfinite(v)
            )
            scoring_comments.append(inf_factors)
            logger.warning(f"{target_id} exclude inf factors:\n{inf_factors}")
            exclude = True

        score = np.prod(scoring_factors)

        if exclude:
            score = -1.0
        return score, scoring_comments
