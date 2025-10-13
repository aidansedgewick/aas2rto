import copy
import pytest
from pathlib import Path

import numpy as np

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time, TimeDelta

from astroplan import Observer, TargetAlwaysUpWarning, TargetNeverUpWarning

from aas2rto.ephem_info import (
    EphemInfo,
    observatory_tonight,
    get_next_valid_sunset_sunrise,
)
from aas2rto.target import Target


@pytest.fixture(scope="module")
def greenwich():
    # use greenwich as local time lines up with UTC...
    return Observer.at_site("greenwich")


@pytest.fixture(scope="module")
def north_pole():
    return Observer(location=EarthLocation(lat=89.99, lon=0.0), name="northpole")


@pytest.fixture(scope="module")
def mar_equinox():
    # Travelling from la silla to la serena to santiago...
    return Time("2025-03-20T12:00:00", format="isot")


@pytest.fixture(scope="module")
def jun_solstice():
    return Time("2025-06-21T12:00:00", format="isot")


@pytest.fixture(scope="module")
def dec_solstice():
    return Time("2025-12-21T12:00:00", format="isot")


@pytest.fixture
def basic_ephem_info(greenwich: Observer, mar_equinox: Time):
    return EphemInfo(greenwich, t_ref=mar_equinox)


class Test__ObsTonightHelper:
    def test__equinox(self, greenwich: Observer, mar_equinox: Time):

        # Act
        sunset, sunrise = observatory_tonight(greenwich, t_ref=mar_equinox)

        # Assert
        assert isinstance(sunset, Time)
        assert isinstance(sunrise, Time)

    def test__perm_night_no_fail(self, north_pole: Observer, dec_solstice: Time):
        # Act
        with pytest.warns(TargetNeverUpWarning):
            sunset, sunrise = observatory_tonight(north_pole, t_ref=dec_solstice)

        # Assert
        assert np.isclose(
            sunset.mjd, dec_solstice.mjd, rtol=1e-8
        )  # sunset == dark == now!
        assert sunrise is None

    def test__perm_day_no_fail(self, north_pole: Observer, jun_solstice):
        # Act
        with pytest.warns(TargetAlwaysUpWarning):
            sunset, sunrise = observatory_tonight(north_pole, t_ref=jun_solstice)

        # Assert
        assert sunset is None
        assert sunrise is None


class Test__ValidSunriseHelper:
    def test__not_close_to_sunset(self, greenwich: Observer, mar_equinox: Time):
        # Arrange
        sunset, sunrise = greenwich.tonight(mar_equinox, horizon=-18.0 * u.deg)

        # Act
        valid_sunset, valid_sunrise = get_next_valid_sunset_sunrise(
            greenwich, t_ref=mar_equinox
        )

        # Assert
        assert np.isclose(valid_sunset.mjd, sunset.mjd, rtol=1e-8)
        assert np.isclose(valid_sunrise.mjd, sunrise.mjd, rtol=1e-8)

    def test__close_to_sunrise_picks_next(self, greenwich: Observer, mar_equinox: Time):
        # Arrange
        invalid_sunset, invalid_sunrise = greenwich.tonight(
            mar_equinox, horizon=-18.0 * u.deg
        )
        almost_sunrise = invalid_sunrise - TimeDelta(15 * u.min)
        assert invalid_sunset < almost_sunrise
        assert almost_sunrise < invalid_sunrise
        tomorrow = mar_equinox + TimeDelta(24 * u.hour)
        tomorrow_sunset, tomorrow_sunrise = greenwich.tonight(
            tomorrow, horizon=-18.0 * u.deg
        )

        # Act
        valid_sunset, valid_sunrise = get_next_valid_sunset_sunrise(
            greenwich, t_ref=almost_sunrise
        )

        # Assert
        assert np.isclose(valid_sunset.mjd, tomorrow_sunset.mjd, rtol=1e-8)
        assert np.isclose(valid_sunrise.mjd, tomorrow_sunrise.mjd, rtol=1e-8)

    def test__perm_night_no_fail(self, north_pole: Observer, dec_solstice: Time):
        # Act
        with pytest.warns(TargetNeverUpWarning):
            sunset, sunrise = get_next_valid_sunset_sunrise(
                north_pole, t_ref=dec_solstice
            )

        # Assert
        assert np.isclose(sunset.mjd, dec_solstice.mjd)
        assert sunrise is None

    def test__perm_day_no_fail(self, north_pole: Observer, jun_solstice: Time):
        # Act
        with pytest.warns(TargetAlwaysUpWarning):
            sunset, sunrise = get_next_valid_sunset_sunrise(
                north_pole, t_ref=jun_solstice
            )

        # Assert
        assert sunset is None
        assert sunrise is None


class Test__InitAndUpdate:
    def test__with_t_ref(self, greenwich: Observer, mar_equinox: Time):
        # Act
        ephem_info = EphemInfo(greenwich, t_ref=mar_equinox)

        # Assert
        assert np.isclose(ephem_info.t_ref.mjd, mar_equinox.mjd)
        assert isinstance(ephem_info.sun_altaz, SkyCoord)  # why not AltAz?
        assert isinstance(ephem_info.moon_altaz, SkyCoord)

        assert isinstance(ephem_info.t_ref, Time)

        assert isinstance(ephem_info.t_grid, Time)

    def test__create_t_grid(self, greenwich: Observer, mar_equinox: Time):
        # Act
        ephem_info = EphemInfo(
            greenwich,
            t_ref=mar_equinox,
            backcast=0.0 * u.day,
            forecast=1.0 * u.day,
            dt=1.0 * u.hour,
        )

        # Assert
        assert len(ephem_info.t_grid) == (24 * 1) + 1  # 1hr + endpoint
        tgrid_range = ephem_info.t_grid[-1] - ephem_info.t_grid[0]
        assert np.isclose(tgrid_range.to(u.hour).value, 24.0)

    def test__t_ref_and_t_grid_warns(self, greenwich: Observer, mar_equinox: Time):
        # Arrange
        t_grid = mar_equinox + 30 * u.min + np.linspace(0, 1, 25) * u.day
        assert not np.isclose(t_grid[0].mjd, mar_equinox.mjd, rtol=1e-8)

        # Act
        with pytest.warns(SyntaxWarning):
            ephem_info = EphemInfo(greenwich, t_ref=mar_equinox, t_grid=t_grid)

        # Assert
        assert np.isclose(ephem_info.t_ref.mjd, t_grid[0].mjd, rtol=1e-8)
        assert len(ephem_info.t_grid) == 25


class Test__AddAltAz:
    def test__valid_coord(self, basic_ephem_info: EphemInfo, basic_target: Target):
        assert basic_ephem_info.target_altaz is None

        # Act
        basic_ephem_info.set_target_altaz(basic_target.coord)

        # Assert
        assert isinstance(basic_ephem_info.target_altaz, SkyCoord)

    def test__invalid_coord_raises(
        self, basic_ephem_info: EphemInfo, basic_target: Target
    ):
        # Act
        with pytest.raises(TypeError):
            basic_ephem_info.set_target_altaz(None)


class Test__CopyMethods:
    def test__stdlib(self, basic_ephem_info):
        # Act
        copy_ephem_info = copy.copy(basic_ephem_info)

        # Assert
        assert isinstance(copy_ephem_info, EphemInfo)
        assert id(copy_ephem_info) != id(basic_ephem_info)

        assert id(copy_ephem_info.observatory) == id(basic_ephem_info.observatory)
        assert id(copy_ephem_info.t_grid) == id(basic_ephem_info.t_grid)
        assert id(copy_ephem_info.sun_altaz) == id(basic_ephem_info.sun_altaz)
        assert id(copy_ephem_info.moon_altaz) == id(basic_ephem_info.moon_altaz)

    def test__new_t_altaz_works(self, basic_ephem_info: EphemInfo):
        # Arrange
        ephem_1 = copy.copy(basic_ephem_info)
        ephem_1.set_target_altaz(SkyCoord(180.0, 0.0, unit="deg"))

        # Act
        ephem_2 = copy.copy(basic_ephem_info)

        # Assert
        assert isinstance(ephem_1.target_altaz, SkyCoord)
        assert ephem_2.target_altaz is None
