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


default_dt = 0.5 / 24.0
default_forecast = 1.0
default_backcast = 0.0


def observatory_tonight(
    observatory: Observer, horizon=-18.0 * u.deg, t_ref: Time = None
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
        sunset, sunrise = None, None
    if sunrise is not None and not isinstance(sunrise.jd, float):
        sunrise = None  # During permanent night.
    return sunset, sunrise


def get_next_valid_sunset_sunrise(
    observatory: Observer, horizon=-18.0 * u.deg, t_ref: Time = None, delta_t=30 * u.min
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


class ObservatoryInfo:
    """
    For storing precomputed information:
        sun_altaz, moon_altaz, target_altaz, sunrise, sunset
        and t_grid they were computed on

    TODO: update to DataClass?
    """

    def __init__(
        self,
        t_grid: Time,
        sun_altaz: AltAz,
        moon_altaz: AltAz,
        target_altaz: AltAz = None,
        sunrise: Time = None,
        sunset: Time = None,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()

        if t_grid is not None:
            if not isinstance(t_grid, Time):
                raise TypeError(f"t_grid should be Time, not {type(t_grid)}")
            t_ref = t_grid[0]  # Overwrite t_ref, as t_ref and grid[0] should match!

        self.t_grid = t_grid
        self.sun_altaz = sun_altaz
        self.moon_altaz = moon_altaz
        self.target_altaz = target_altaz
        self.sunrise = sunrise
        self.sunset = sunset
        self.t_ref = t_ref

    @classmethod
    def for_observatory(
        cls,
        observatory: Observer,
        horizon=-18.0 * u.deg,
        t_ref: Time = None,
        t_grid: Time = None,
        forecast: float = None,
        backcast: float = None,
        dt: float = None,
    ):
        """
        Parameters
        ----------
        observatory : astroplan.Observer
        horizon : astropy.quantity, default=-18.0 * u.deg
        t_ref: astropy.Time, default=now


        """

        if t_ref is None:
            logger.warning("t_ref is None! Defaulting to Time.now()")
        t_ref = t_ref or Time.now()

        if t_grid is None:
            if dt is None:
                logger.warning(f"should not pass dt as None! set as {default_dt}")
            dt = dt or default_dt  # sometimes dt is explitly passed as None ?!
            N_samples = int(round((1 / dt))) + 1  # for endpoint
            forecast = forecast or default_forecast
            backcast = backcast or default_backcast
            t_grid = t_ref + np.linspace(-backcast, forecast, N_samples) * u.day
        else:
            if not isinstance(t_grid, Time):
                raise TypeError(f"t_grid should be Time, not {type(t_grid)}")
            t_ref = t_grid[0]  # Overwrite t_ref, as t_ref and grid[0] should match!

        delta_t = t_grid[1] - t_grid[0]

        try:
            sun_altaz = observatory.sun_altaz(t_grid)
        except Exception as e:
            if observatory is None:
                msg = (
                    "\n    You're trying to compute obs_info for \033[31;1mobservatory=None\033[0m."
                    f"\n    This is probably why there's an exception, type={type(e).__name__}\n"
                )
                logger.error(msg)
            raise  # Bare raise re-raises the exception we just caught.

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=AstropyDeprecationWarning)
            moon_altaz = observatory.moon_altaz(t_grid)

        sunset, sunrise = get_next_valid_sunset_sunrise(
            observatory, horizon=horizon, t_ref=t_ref, delta_t=delta_t
        )

        return cls(
            t_grid=t_grid,
            moon_altaz=moon_altaz,
            sun_altaz=sun_altaz,
            sunset=sunset,
            sunrise=sunrise,
            t_ref=t_ref,
        )

    def set_target_altaz(self, coord: SkyCoord, observatory: Observer):
        if self.t_grid is None:
            raise ValueError("Can't compute altaz, t_grid is None")
        self.target_altaz = observatory.altaz(self.t_grid, coord)

    def __copy__(self):
        return ObservatoryInfo(
            t_grid=self.t_grid,  # .copy(),
            sun_altaz=self.sun_altaz,  # .copy(),
            moon_altaz=self.moon_altaz,  # .copy(),
            target_altaz=copy.deepcopy(self.target_altaz),
            sunrise=self.sunrise,  # copy.deepcopy(self.sunrise),
            sunset=self.sunset,  # copy.deepcopy(self.sunset),
            t_ref=self.t_ref,
        )

    copy = __copy__

    # __deepcopy__ = deepcopy
