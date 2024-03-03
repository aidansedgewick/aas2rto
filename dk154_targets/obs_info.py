import copy
import warnings
from logging import getLogger

import numpy as np

from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning

from astroplan import Observer

logger = getLogger(__name__.split(".")[-1])


class ObservatoryInfo:
    """For storing precomputed"""

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

    @staticmethod
    def observatory_tonight(
        observatory: Observer, horizon=-18.0 * u.deg, t_ref: Time = None
    ):
        try:
            sunset, sunrise = observatory.tonight(t_ref, horizon=horizon)
        except TypeError as e:
            sunset, sunrise = None, None
        if sunrise is not None and not isinstance(sunrise.jd, float):
            sunrise = None  # During permanent night.
        return sunset, sunrise

    @classmethod
    def for_observatory(
        cls,
        observatory: Observer,
        horizon=-18.0 * u.deg,
        t_ref: Time = None,
        t_grid: Time = None,
        forecast=1.0,
        dt=None,
    ):
        t_ref = t_ref or Time.now()

        if t_grid is None:
            dt = dt or 1 / (24.0 * 4.0)
            N_samples = int((1 / dt)) + 1  # for endpoint
            t_grid = t_ref + np.linspace(0, forecast, N_samples) * u.day
        else:
            if not isinstance(t_grid, Time):
                raise TypeError(f"t_grid should be Time, not {type(t_grid)}")
            t_ref = t_grid[0]  # Overwrite t_ref, as t_ref and grid[0] should match!

        delta_t = t_grid[1] - t_grid[0]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=AstropyDeprecationWarning)
            moon_altaz = observatory.moon_altaz(t_grid)
        sun_altaz = observatory.sun_altaz(t_grid)

        sunset, sunrise = cls.observatory_tonight(
            observatory, horizon=horizon, t_ref=t_ref
        )
        if sunset is not None and sunrise is not None:
            time_to_sunrise = sunrise - sunset
            if time_to_sunrise < 2.0 * delta_t:
                old_sunset = sunset
                old_sunrise = sunrise
                t_ref_shift = sunrise + delta_t
                sunset, sunrise = cls.observatory_tonight(
                    observatory, horizon=horizon, t_ref=t_ref_shift
                )
                msg = (
                    f"    At t_ref={t_ref.strftime('%y-%m-%d %H:%M')}, "
                    f"sunrise {time_to_sunrise.to(u.min):.2f} away "
                    f"(sunrise={old_sunrise.strftime('%y-%m-%d %H:%M')}).\n"
                    f"    Compute next sunset/rise from "
                    f"{t_ref_shift.strftime('%y-%m-%d %H:%M')}."
                )
                logger.warning(
                    f"\033[33mBad sunset for {observatory.name} (too near)\033[0m\n{msg}"
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
