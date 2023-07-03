from logging import getLogger

from astropy.coordinates import AltAz
from astropy.time import Time

logger = getLogger(__name__.split(".")[-1])


class ObservatoryInfo:
    """For storing precomputed"""

    def __init__(
        self,
        t_grid: Time,
        sun_altaz: AltAz,
        moon_altaz: AltAz,
        target_altaz: AltAz,
        time_computed: Time,
        sunrise: Time = None,
        sunset: Time = None,
    ):
        self.t_grid = t_grid
        self.sun_altaz = sun_altaz
        self.moon_altaz = moon_altaz
        self.target_altaz = target_altaz
        self.time_computed = time_computed
        self.sunrise = sunrise
        self.sunset = sunset
