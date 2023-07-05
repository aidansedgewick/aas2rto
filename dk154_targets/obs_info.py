import copy

from logging import getLogger

import numpy as np

from astropy import units as u
from astropy.coordinates import AltAz
from astropy.time import Time

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

        self.t_grid = t_grid
        self.sun_altaz = sun_altaz
        self.moon_altaz = moon_altaz
        self.target_altaz = target_altaz
        self.sunrise = sunrise
        self.sunset = sunset
        self.t_ref = t_ref

    @classmethod
    def from_observatory(
        cls, observatory: Observer, horizon=-18 * u.deg, t_ref: Time = None
    ):
        t_grid = t_ref + np.linspace(0, 24.0, 24 * 4) * u.hour
        moon_altaz = observatory.moon_altaz(t_grid)
        sun_altaz = observatory.sun_altaz(t_grid)
        try:
            sunset, sunrise = observatory.tonight(t_ref, horizon=horizon)
        except TypeError as e:
            sunset, sunrise = None, None
        if sunrise is not None:
            if not isinstance(sunrise.jd, float):
                # During permanent night.
                sunrise = None

        return cls(
            t_grid=t_grid,
            moon_altaz=moon_altaz,
            sun_altaz=sun_altaz,
            sunset=sunset,
            sunrise=sunrise,
            t_ref=t_ref,
        )

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
