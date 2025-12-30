import pytest

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from astroplan import Observer

from aas2rto.exc import InvalidEphemWarning
from aas2rto.observatory.ephem_info import EphemInfo
from aas2rto.scoring.default_obs_scoring import (
    DefaultObservatoryScoring,
    compute_visibility_integral,
    calc_visibility_factor,
    compute_inv_airmass,
    calc_altitude_factor,
)
from aas2rto.target import Target


@pytest.fixture
def midnight(lasilla: Observer, t_fixed: Time):
    return lasilla.midnight(t_fixed, which="next", n_grid_points=40)


@pytest.fixture
def t_early(midnight: Time):
    return midnight - 3.0 * u.hour


@pytest.fixture
def t_late(midnight: Time):
    return midnight + 3.0 * u.hour


@pytest.fixture
def sunset(lasilla: Observer, midnight: Time):
    return lasilla.sun_set_time(midnight, which="previous")


@pytest.fixture
def ephem_early(lasilla: Observer, t_fixed: Time, t_early: Time):
    lst = lasilla.local_sidereal_time(t_early)
    o_lat = lasilla.location.lat  # dec = lat +/- ZD, ZD = (90-transit_alt)
    coord = SkyCoord(ra=lst, dec=o_lat + (90.0 - 60.0) * u.deg)
    print(coord)
    return EphemInfo(lasilla, t_ref=t_fixed, target_coord=coord)


@pytest.fixture
def ephem_late(lasilla: Observer, t_fixed: Time, t_late: Time):
    lst = lasilla.local_sidereal_time(t_late)
    o_lat = lasilla.location.lat  # dec = lat +/- ZD, ZD = (90-transit_alt)
    coord = SkyCoord(ra=lst, dec=o_lat + (90.0 - 60.0) * u.deg)
    print(coord)
    return EphemInfo(lasilla, t_ref=t_fixed, target_coord=coord)


@pytest.fixture
def target_late(ephem_late: EphemInfo):
    target = Target("T_late", ephem_late.target_coord)
    target.ephem_info["lasilla"] = ephem_late
    return target


@pytest.fixture
def ephem_low(lasilla: Observer, t_fixed: Time, t_late: Time):
    lst = lasilla.local_sidereal_time(t_late)
    o_lat = lasilla.location.lat  # dec = lat +/- ZD, ZD = (90-transit_alt)
    coord = SkyCoord(ra=lst, dec=o_lat + (90.0 - 45.0) * u.deg)
    print(coord)
    return EphemInfo(lasilla, t_ref=t_fixed, target_coord=coord)


class Test__VisIntegral:
    def test__all_equal_ref(self):
        # Arrange
        t_grid = np.linspace(60000.0, 60000.5, 100)
        alt_grid = np.full(len(t_grid), 45.0)

        # Act
        integral = compute_visibility_integral(
            t_grid=t_grid, alt_grid=alt_grid, min_alt=30.0, ref_alt=45.0
        )

        # Assert
        assert np.isclose(integral, 1.0)

    def test__all_below_min_is_zero(self):
        # Arrange
        t_grid = np.linspace(60000.0, 60000.5, 100)
        alt_grid = np.full(len(t_grid), 20.0)

        # Act
        integral = compute_visibility_integral(
            t_grid=t_grid, alt_grid=alt_grid, min_alt=30.0, ref_alt=45.0
        )

        # Assert
        assert np.isclose(integral, 0.0)

    def test__sloping(self):
        # Arrange
        t_grid = np.linspace(60000.0, 60000.5, 100)
        alt_grid = -30.0 * (t_grid - 60000.0) + 45
        assert np.isclose(alt_grid[0], 45.0)  # can I do y=mx+c properly?
        assert np.isclose(alt_grid[-1], 30.0)

        # Act
        integral = compute_visibility_integral(
            t_grid=t_grid, alt_grid=alt_grid, min_alt=30.0, ref_alt=45.0
        )

        # Assert
        assert np.isclose(integral, 0.5)


class Test__CalcVisibilityFactor:
    def test__calc_vis_factor(self, t_early: Time, ephem_early: EphemInfo):
        # Act
        vis_factor, comms = calc_visibility_factor(ephem_early, t_ref=t_early)

        # Arrange
        assert vis_factor > 0.0
        assert isinstance(comms, list)

        comm_str = " ".join(comms)
        assert "vis_factor=" in comm_str

    def test__not_risen_positive(
        self, lasilla: Observer, t_early: Time, ephem_late: EphemInfo
    ):
        # Arrange
        curr_alt = lasilla.altaz(t_early, ephem_late.target_coord).alt.deg
        assert curr_alt < 30.0
        assert ephem_late.target_transit > t_early

        # Act
        vis_factor, _ = calc_visibility_factor(ephem_late, t_ref=t_early)

        # Assert
        assert vis_factor > 0.0

    def test__target_set_excluded(
        self, lasilla: Observer, t_late: Time, ephem_early: EphemInfo
    ):
        # Arrange
        curr_alt = lasilla.altaz(t_late, ephem_early.target_coord).alt.deg
        assert curr_alt < 30.0
        assert ephem_early.target_transit < t_late

        # Act
        vis_factor, comms = calc_visibility_factor(ephem_early, t_ref=t_late)

        # Assert
        assert np.isclose(vis_factor, -1.0)
        assert isinstance(comms, list)

        comm_str = " ".join(comms)
        assert "always < min_alt" in comm_str
        assert "set vis_factor=-1.0" in comm_str

    def test__early_target_promoted(
        self, t_early: Time, ephem_early: Target, ephem_late: EphemInfo
    ):
        # Act
        vis_early, _ = calc_visibility_factor(ephem_early, t_ref=t_early)
        vis_late, _ = calc_visibility_factor(ephem_late, t_ref=t_early)

        # Assert
        assert vis_early > vis_late  # As T_early sets sooner...!

    def test__low_target_promoted(
        self, midnight: Time, ephem_late: EphemInfo, ephem_low: EphemInfo
    ):
        # Act
        vis_high_alt, _ = calc_visibility_factor(ephem_late, t_ref=midnight)
        vis_low_alt, _ = calc_visibility_factor(ephem_low, t_ref=midnight)

        # Assert
        assert vis_low_alt > 0.0
        assert vis_low_alt > vis_high_alt

    def test__v_low_target_exluded(self, t_early: Time, ephem_low: EphemInfo):
        # Act
        f_normal, c1 = calc_visibility_factor(ephem_low, t_ref=t_early)
        f_strict, c2 = calc_visibility_factor(ephem_low, t_ref=t_early, min_alt=60.0)

        # Assert
        assert f_normal > 0.0
        assert np.isclose(f_strict, -1.0)

        comm_str = " ".join(c2)
        assert "always < min_alt" in comm_str
        assert "set vis_factor=-1.0" in comm_str

    def test__old_ephem_no_fail(self, ephem_late: EphemInfo):
        # Arrange
        t_much_later = Time(60010.0, format="mjd")

        # Act
        with pytest.warns(InvalidEphemWarning):
            vis_factor, comms = calc_visibility_factor(ephem_late, t_ref=t_much_later)

        # Assert
        assert np.isclose(vis_factor, -1.0)

        comm_str = " ".join(comms)
        assert "t_ref is AFTER latest ephem_info"


class Test__ComputeInvAirmass:
    def test__inv_airmass(self):
        # Act
        inv_airmass = compute_inv_airmass(30.0)

        # Assert
        assert np.isclose(inv_airmass, 0.5)


class Test__CalcAltFactor:
    def test__alt_factor(self, ephem_early: EphemInfo, t_early: Time):
        # Act
        alt_factor, comms = calc_altitude_factor(ephem_early, t_early)

        # Assert
        assert np.isclose(alt_factor, 0.86603, rtol=0.1)  # sin(60deg)
        assert isinstance(comms, list)

        comm_str = " ".join(comms)

    def test__exclude_low_target(self, ephem_early: EphemInfo, t_early: Time):
        # Act
        alt_factor, comms = calc_altitude_factor(ephem_early, t_early, min_alt=70.0)

        # Assert
        assert np.isclose(alt_factor, -1.0)  # sin(60deg)
        assert isinstance(comms, list)

        comm_str = " ".join(comms)

    def test__old_ephem_no_fail(self, ephem_late: EphemInfo):
        # Arrange
        t_much_later = Time(60010.0, format="mjd")

        # Act
        with pytest.warns(InvalidEphemWarning):
            alt_factor, comms = calc_altitude_factor(ephem_late, t_ref=t_much_later)

        # Assert
        assert np.isclose(alt_factor, -1.0)

        comm_str = " ".join(comms)
        assert "t_ref is AFTER latest ephem_info"


class Test__ObsScoringInit:
    def test__obs_scoring_init(self):
        # Act
        scoring_func = DefaultObservatoryScoring()

        # Assert
        assert scoring_func.method in ["visibility", "altitude"]  # don't care which
        assert isinstance(scoring_func.min_altitude, float)
        assert isinstance(scoring_func.ref_altitude, float)

    def test__bad_method_raises(self):
        # Act
        with pytest.raises(ValueError):
            scoring_func = DefaultObservatoryScoring(method="bad_method")


class Test__ObsVisMethod:
    def test__obs_vis_method(
        self, lasilla: Observer, target_late: Target, t_late: Time
    ):
        # Arrange
        scoring_func = DefaultObservatoryScoring(method="visibility")

        # Act
        vis_factor, comms = scoring_func(target_late, lasilla, t_late)

        # Assert
        assert vis_factor > 0.0
        assert isinstance(comms, list)

        comm_str = " ".join(comms)
        assert "use method='visibility'" in comm_str
        assert "vis_factor=" in comm_str

    def test__obs_vis_excludes(
        self, lasilla: Observer, target_late: Target, t_late: Time
    ):
        # Arrange
        scoring_func = DefaultObservatoryScoring(method="visibility", min_altitude=70.0)

        # Act
        vis_factor, comms = scoring_func(target_late, lasilla, t_late)

        # Assert
        assert vis_factor < 0.0
        assert isinstance(comms, list)

        comm_str = " ".join(comms)
        assert "use method='visibility'" in comm_str
        assert "vis_factor=" in comm_str

    def test__obs_alt_method(
        self, lasilla: Observer, target_late: Target, t_late: Time
    ):
        # Arrange
        scoring_func = DefaultObservatoryScoring(method="altitude")

        # Act
        alt_factor, comms = scoring_func(target_late, lasilla, t_late)

        # Assert
        assert alt_factor > 0.0
        assert isinstance(comms, list)

        comm_str = " ".join(comms)
        assert "use method='altitude'" in comm_str
        assert "alt_factor=" in comm_str

    def test__obs_alt_exclude(
        self, lasilla: Observer, target_late: Target, t_late: Time
    ):
        # Arrange
        scoring_func = DefaultObservatoryScoring(method="altitude", min_altitude=70.0)

        # Act
        alt_factor, comms = scoring_func(target_late, lasilla, t_late)

        # Assert
        assert np.isclose(alt_factor, -1.0)
        assert isinstance(comms, list)

        comm_str = " ".join(comms)
        assert "use method='altitude'" in comm_str
        assert "alt_factor=-1.0: curr_alt" in comm_str

    def test__no_ephem_no_fail(
        self, lasilla: Observer, target_late: Target, t_late: Time
    ):
        # Arrange
        target_late.ephem_info = {}
        scoring_func = DefaultObservatoryScoring()

        # Act
        obs_factor, comms = scoring_func(target_late, lasilla, t_late)

        # Assert
        assert np.isclose(obs_factor, -1.0)

        comm_str = " ".join(comms)
        assert "no ephem_info for T_late at observatory 'lasilla'" in comm_str

    def test__obs_none_raises(self, target_late: Target, t_late: Time):
        # Arrange
        scoring_func = DefaultObservatoryScoring()

        # Act
        with pytest.raises(TypeError):
            scoring_func(target_late, None, t_late)
