import copy
import sys
import traceback
import warnings
from logging import getLogger

import numpy as np

from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning

from astroplan import Observer

logger = getLogger(__name__.split(".")[-1])


def observatory_tonight(
    observatory: Observer, t_ref: Time = None, horizon=-18.0 * u.deg
):
    """
    This is almost exactly like `astroplan.Observer.tonight()`, but it catches
    some exceptions:
        - If it's permanent night: sunset=t_ref, sunrise=None
        - Don't crash if there's an exception (ie, permanent day):
            sunrise=None, sunset=None
    """
    t_ref = t_ref or Time.now()
    try:
        sunset, sunrise = observatory.tonight(t_ref, horizon=horizon)
    except TypeError as e:
        sunset, sunrise = None, None  # Permanent day causes TypeError??
    if sunrise is not None and not isinstance(sunrise.jd, float):
        sunrise = None  # During permanent night.
    return sunset, sunrise


def get_next_valid_sunset_sunrise(
    observatory: Observer,
    t_ref: Time = None,
    horizon=-18.0 * u.deg,
    delta_t=30 * u.min,
):
    """
    If the next sunrise is closer than delta_t, compute the one after.

    """
    t_ref = t_ref or Time.now()

    sunset, sunrise = observatory_tonight(observatory, horizon=horizon, t_ref=t_ref)
    if sunset is None or sunrise is None:
        return sunset, sunrise

    time_to_sunrise = sunrise - t_ref
    if time_to_sunrise < delta_t:
        invalid_sunset = sunset
        invalid_sunrise = sunrise
        t_shift = sunrise + 2.0 * delta_t
        sunset, sunrise = observatory_tonight(
            observatory, horizon=horizon, t_ref=t_shift
        )
        t_ref_str = t_ref.strftime("%y-%m-%d %H:%M")
        t_shift_str = t_shift.strftime("%y-%m-%d %H:%M")
        dt_sr_min = time_to_sunrise.to(u.min)
        inv_sr_str = invalid_sunrise.strftime("%y-%m-%d %H:%M")
        sr_str = sunrise.strftime("%y-%m-%d %H:%M")

        msg_lines = [
            f"\033[33mBad sunrise for {observatory.name}\033[0m\n"
            f"    Sunrise (end of night) is too near (<{delta_t.to(u.min):.1f}):\n"
            f"    At t_ref={t_ref_str} sunrise is {dt_sr_min:.1f} away (sr={inv_sr_str}).\n"
            f"    Compute next sunset/rise from {t_shift_str} (new sr={sr_str}).\n"
        ]
        msg = "".join(msg_lines)
        logger.warning(msg)
    return sunset, sunrise


class EphemInfo:
    """
    For computing and storing ephemeris information:
        sun_altaz, moon_altaz, target_altaz, sunrise, sunset, horizon
        and t_grid they were computed on

    Will not upgrade to @dataclass - too much additional logic.
    """

    default_horizon = -18.0 * u.deg
    default_dt = 0.25 * u.hour
    default_forecast = 1.0 * u.day
    default_backcast = 0.0 * u.day

    def __init__(
        self,
        observatory: Observer,
        t_ref: Time = None,
        t_grid: Time = None,
        horizon: u.Quantity = None,
        forecast: u.Quantity = None,
        backcast: u.Quantity = None,
        dt: u.Quantity = None,
    ):
        """
        Parameters
        ----------
        observatory : astroplan.Observer
        horizon : astropy.quantity, default=-18.0 * u.deg
        t_ref : astropy.Time, default=now
            if t_grid is passed instead, t_ref is set to t_grid[0]
        t_grid : astropy.Time, default = None
            computed as t_ref +
        """
        self.observatory = observatory
        self.t_ref = t_ref
        self.t_grid = t_grid

        # Need ugly if statements here, because u.Quantity has no 'truthiness'
        # eg. if dt = 1 * u.hr, then `self.dt = dt or default_dt` raises ValueError.
        if horizon is None:
            horizon = self.default_horizon
        self.horizon = horizon
        if forecast is None:
            forecast = self.default_forecast
        self.forecast = forecast
        if backcast is None:
            backcast = self.default_backcast
        self.backcast = backcast
        if dt is None:
            dt = self.default_dt
        self.dt = dt

        self.sun_altaz = None
        self.moon_altaz = None
        self.target_altaz = None

        self.recompute_info(t_ref=self.t_ref, t_grid=self.t_grid)

    def recompute_info(self, t_ref: Time = None, t_grid: Time = None):
        if self.t_ref and self.t_grid:
            msg = (
                "pass only t_grid or t_ref, not both! "
                "provided t_ref will be overwritten with t_grid[0]"
            )
            logger.warning(msg)
            warnings.warn(SyntaxWarning(msg))

        if t_grid is None:
            t_ref = t_ref or Time.now()
            diff = self.backcast + self.forecast
            N = int((diff) / self.dt) + 1
            t_grid = t_ref - self.backcast + np.linspace(0.0, 1.0, N) * diff
        else:
            if not isinstance(t_grid, Time):
                raise TypeError(f"t_grid should have type Time, not {type(t_grid)}")
            t_ref = t_grid[0]  # Overwrite t_ref, as t_ref and grid[0] should match!

        delta_t = t_grid[1] - t_grid[0]

        sun_altaz = self.observatory.sun_altaz(t_grid)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=AstropyDeprecationWarning)
            moon_altaz = self.observatory.moon_altaz(t_grid)

        sunset, sunrise = get_next_valid_sunset_sunrise(
            self.observatory, horizon=self.horizon, t_ref=t_ref, delta_t=delta_t
        )

        self.t_grid = t_grid
        self.sun_altaz = sun_altaz
        self.moon_altaz = moon_altaz
        self.target_altaz = None
        self.sunrise = sunrise
        self.sunset = sunset
        self.t_ref = t_ref

    def set_target_altaz(self, coord: SkyCoord):
        if not isinstance(coord, SkyCoord):
            raise TypeError(f"'coord' should be type 'SkyCoord' not {type(coord)}")
        self.target_altaz = self.observatory.altaz(self.t_grid, coord)
