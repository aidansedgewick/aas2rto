import pytest

import numpy as np

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from astroplan import Observer, TargetAlwaysUpWarning, TargetNeverUpWarning

from dk154_targets.target import Target
from dk154_targets.obs_info import (
    ObservatoryInfo,
    observatory_tonight,
    get_next_valid_sunset_sunrise,
)


@pytest.fixture
def ucph():
    el = EarthLocation(lat=55.6761, lon=12.5683)
    return Observer(el, name="ucph")


@pytest.fixture
def spt():
    el = EarthLocation(lat=-89.5, lon=0.0)
    return Observer(el, name="spt")


@pytest.fixture
def longyearbyen():
    el = EarthLocation(lat=78.2232, lon=15.6267)
    return Observer(el, name="northpole")


class Test__ObservatoryTonight:

    def test__observatory_tonight_during_day(self, ucph):
        t_ref = Time("2023-03-21T12:00:00", format="isot")
        # https://www.timeanddate.com/sun/denmark/copenhagen?month=3&year=2023
        exp_sunset = Time("2023-03-21T19:32:00", format="isot")  # in UT !!!
        exp_sunrise = Time("2023-03-22T03:00:00", format="isot")  # in UT !!!

        sunset, sunrise = observatory_tonight(ucph, t_ref=t_ref)

        assert isinstance(sunset, Time)
        assert isinstance(sunrise, Time)

        # atol 5 min
        assert np.isclose(sunset.mjd, exp_sunset.mjd, atol=0.0035)
        assert np.isclose(sunrise.mjd, exp_sunrise.mjd, atol=0.0035)
        assert np.isclose(sunset.mjd % 1, exp_sunset.mjd % 1, atol=0.0035)
        assert np.isclose(sunrise.mjd % 1, exp_sunrise.mjd % 1, atol=0.0035)

        # mjd mod 1: 60014.5674 % 1 == 0.5674 -- want to check the end bit!

    def test__observatory_tonight_during_night(self, ucph):
        t_ref = Time("2023-03-22T00:05:00", format="isot")
        exp_sunrise = Time("2023-03-22T03:00:00", format="isot")  # in UT !!!

        sunset, sunrise = observatory_tonight(ucph, t_ref=t_ref)

        assert isinstance(sunset, Time)
        assert isinstance(sunrise, Time)

        # atol 5 min
        assert np.isclose(sunset.mjd, t_ref.mjd, atol=0.0035)  # Same as t_ref!
        assert np.isclose(sunrise.mjd, exp_sunrise.mjd, atol=0.0035)
        assert np.isclose(sunset.mjd % 1, t_ref.mjd % 1, atol=0.0035)
        assert np.isclose(sunrise.mjd % 1, exp_sunrise.mjd % 1, atol=0.0035)

    def test__observatory_permanent_night(self, longyearbyen):
        t_ref = Time("2023-12-21T12:00:00", format="isot")
        exp_sunset = Time("2023-12-21T15:13:00")  # in UT
        exp_sunrise = Time("2023-12-22T06:37:00")  # in UT

        # Test astronomical night still has sunset/rise
        sunset, sunrise = observatory_tonight(longyearbyen, t_ref=t_ref)

        # atol 5 min
        assert np.isclose(sunset.mjd, exp_sunset.mjd, atol=0.0035)
        assert np.isclose(sunrise.mjd, exp_sunrise.mjd, atol=0.0035)
        assert np.isclose(sunset.mjd % 1, exp_sunset.mjd % 1, atol=0.0035)
        assert np.isclose(sunrise.mjd % 1, exp_sunrise.mjd % 1, atol=0.0035)

        # Test civil is permanent:
        with pytest.warns(TargetNeverUpWarning):
            sunset, sunrise = observatory_tonight(
                longyearbyen, horizon=-6 * u.deg, t_ref=t_ref
            )
        assert np.isclose(sunset.mjd, t_ref.mjd, atol=0.0035)
        assert np.isclose(sunset.mjd % 1, t_ref.mjd % 1, atol=0.0035)

        assert sunrise is None

    def test__permanent_day(self, longyearbyen):
        t_ref = Time("2023-06-21T12:00:00", format="isot")

        with pytest.warns(TargetAlwaysUpWarning):
            with pytest.raises(TypeError):
                longyearbyen.tonight(t_ref)

        with pytest.warns(TargetAlwaysUpWarning):
            sunset, sunrise = observatory_tonight(longyearbyen, t_ref=t_ref)

        assert sunset is None
        assert sunrise is None


class Test__GetValidSunsetSunrise:

    def test__normal(self, ucph):
        t_ref = Time("2023-03-21T12:00:00", format="isot")
        # https://www.timeanddate.com/sun/denmark/copenhagen?month=3&year=2023
        exp_sunset = Time("2023-03-21T19:32:00", format="isot")  # in UT !!!
        exp_sunrise = Time("2023-03-22T03:00:00", format="isot")  # in UT !!!

        sunset, sunrise = get_next_valid_sunset_sunrise(ucph, t_ref=t_ref)

        assert isinstance(sunset, Time)
        assert isinstance(sunrise, Time)

        # atol 5 min
        assert np.isclose(sunset.mjd, exp_sunset.mjd, atol=0.0035)
        assert np.isclose(sunrise.mjd, exp_sunrise.mjd, atol=0.0035)
        assert np.isclose(sunset.mjd % 1, exp_sunset.mjd % 1, atol=0.0035)
        assert np.isclose(sunrise.mjd % 1, exp_sunrise.mjd % 1, atol=0.0035)

        # mjd mod 1: 60014.5674 % 1 == 0.5674 -- want to check the end bit!

    def test__close_to_sunset(self, ucph):
        t_ref = Time("2023-03-22T02:40:00", format="isot")
        # This is very close to the sunrise in 20 minutes. Get the one after!

        # expect the next sunset and sunrise!
        exp_sunset = Time("2023-03-22T19:34:00", format="isot")  # in UT !!!
        exp_sunrise = Time("2023-03-23T02:57:00", format="isot")  # in UT !!!

        sunset, sunrise = get_next_valid_sunset_sunrise(ucph, t_ref=t_ref)

        assert isinstance(sunset, Time)
        assert isinstance(sunrise, Time)

        # atol 5 min
        assert np.isclose(sunset.mjd, exp_sunset.mjd, atol=0.0035)
        assert np.isclose(sunrise.mjd, exp_sunrise.mjd, atol=0.0035)
        assert np.isclose(sunset.mjd % 1, exp_sunset.mjd % 1, atol=0.0035)
        assert np.isclose(sunrise.mjd % 1, exp_sunrise.mjd % 1, atol=0.0035)


class ObsInfoInit:

    def test__init(self):
        obs_info = ObservatoryInfo()

        assert isinstance(obs_info.t_ref, Time)

    def test__for_obs(self, ucph):

        obs_info = ObservatoryInfo.for_observatory(ucph)

        assert isinstance()
