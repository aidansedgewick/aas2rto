import time
from logging import getLogger
from typing import Callable

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.time import Time

from astroplan import Observer

from aas2rto.target import Target

logger = getLogger("default_obs_scoring")


def calc_visibility_factor(alt_grid, min_alt: float = 30.0, ref_alt: float = 45.0):
    alt_above_minimum = np.maximum(alt_grid, min_alt) - min_alt
    integral = np.trapz(alt_above_minimum)

    if min_alt > ref_alt:
        raise ValueError(f"norm_alt > min_altitude ({ref_alt} > {min_alt})")
    norm = len(alt_grid) * (ref_alt - min_alt)

    return 1.0 / (integral / norm)


def calc_altitude_factor(alt: float):
    """
    This is just sin(alt), with alt in deg.
    motivated by X~sec(90-alt), where X is airmass.
    promote high alt, so low airmass --> 1/X ~ sin(alt)
    """
    return np.sin(alt * np.pi / 180.0)


class DefaultObservatoryScoring:
    """
    Doc string here

    """

    __name__ = "default_observatory_scoring"

    def __init__(
        self,
        min_altitude: float = 30.0,
        ref_altitude: float = 90.0,
        exclude_daytime: bool = False,
        visibility_method: str = "visibility",
    ):
        self.min_altitude = min_altitude
        self.ref_altitude = ref_altitude
        self.exclude_daytime = exclude_daytime
        self.visibility_method = visibility_method

        expected_vis_methods = ["visibility", "altitude"]
        if self.visibility_method not in expected_vis_methods:
            msg = (
                f"visibility_factor should be one of {expected_vis_methods},\n    "
                f"not {visibility_method}"
            )
            raise ValueError(msg)

    def __call__(self, target: Target, observatory: Observer, t_ref: Time = None):

        if observatory is None:
            raise ValueError(
                "observatory must be of type `astroplan.Observer`, not `None`"
            )

        min_alt = self.min_altitude

        # to keep track
        scoring_comments = []
        reject_comments = []
        factors = {}
        # reject = False  # Observatory factors should NOT reject targets.
        exclude = False  # If true, don't reject, but not interesting right now.

        objectId = target.objectId

        timing_lookup = {}  # Internal use, tracking code performance.

        obs_name = getattr(observatory, "name", None)
        if obs_name is None:
            raise ValueError(f"observatory {observatory} has no name!!")

        t_str = t_ref.iso

        # Get the observatory info
        obs_info = target.observatory_info.get(obs_name, None)
        if obs_info is None:
            logger.warning(f"obs_info=None for {obs_name} {objectId}")
            target_altaz = None
            logger.warning("calc curr_sun_alt. this is slow!")
            scoring_comments.append("calculating alt is slow!")
            curr_sun_alt = observatory.sun_altaz(t_ref).alt.deg
        else:
            sunset = obs_info.sunset
            sunrise = obs_info.sunrise
            target_altaz = obs_info.target_altaz
            curr_sun_alt = obs_info.sun_altaz.alt.deg[0]

            if obs_info.sunset is None or obs_info.sunrise is None:
                scoring_comments.append(f"sun never sets at this observatory")
                exclude = True

        if self.exclude_daytime and curr_sun_alt > -18.0:
            sun_alt_comm = f"exclude as sun up ({curr_sun_alt:.2f}) deg at {t_str}"
            scoring_comments.append(sun_alt_comm)
            exclude = True

        if target_altaz is not None:
            curr_alt = target_altaz.alt.deg[0]
        else:
            curr_alt = observatory.altaz(t_ref, target.coord).alt.deg
        if curr_alt < min_alt:
            alt_comm = f"exclude as {curr_alt:.2f}deg < {min_alt:.2f}deg at {t_str}"
            scoring_comments.append(alt_comm)
            exclude = True

        vis_method = self.visibility_method  # If we need to default to 'altitude'
        if target_altaz is None:
            vis_method = "altitude"

        if exclude:
            vis_factor = 1.0
        else:
            if vis_method == "altitude":
                vis_factor = calc_altitude_factor(curr_alt)
            elif vis_method == "visibility":
                t_grid = obs_info.t_grid
                night_mask = (sunset.mjd < t_grid.mjd) & (t_grid.mjd < sunrise.mjd)
                target_night_alt = target_altaz.alt.deg[night_mask]
                if all(target_night_alt < min_alt):
                    always_set_comm = f"target always below min_alt={min_alt:.2f}"
                    scoring_comments.append(always_set_comm)
                    exclude = True
                    vis_factor = 1.0
                else:
                    vis_factor = calc_visibility_factor(
                        target_night_alt, min_alt=min_alt, ref_alt=self.ref_altitude
                    )
            else:
                unexp_comm = f"unexpected method {vis_method}: vis_factor = 1.0"
                scoring_comments.append(unexp_comm)
                logger.warning(unexp_comm)
                vis_factor = 1.0
            vis_comm = f"vis_factor={vis_factor:.2f} from alt={curr_alt:.2f}deg at {t_str} (method='{vis_method}')"
            scoring_comments.append(vis_comm)

        factors["vis_factor"] = vis_factor

        scoring_factors = np.array(list(factors.values()))
        if not all(scoring_factors > 0):
            neg_factors = "\n".join(
                f"    {k}={v}" for k, v in factors.items() if not v > 0
            )
            scoring_comments.append(neg_factors)
            logger.warning(f"{objectId} has negative factors:\n{neg_factors}")
            exclude = True
        if not all(np.isfinite(scoring_factors)):
            inf_factors = "\n".join(
                f"    {k}={v}" for k, v in factors.items() if not np.isfinite(v)
            )
            scoring_comments.append(inf_factors)
            logger.warning(f"{objectId} has inf factors:\n{inf_factors}")

        score = np.prod(scoring_factors)

        if exclude:
            score = -1.0
        return score, scoring_comments, reject_comments
