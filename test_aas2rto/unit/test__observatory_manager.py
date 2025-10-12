import pytest
from pathlib import Path

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

from astroplan import Observer

from aas2rto.exc import UnexpectedKeysWarning
from aas2rto.ephem_info import EphemInfo
from aas2rto.observatory_manager import ObservatoryManager
from aas2rto.path_manager import PathManager
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup


@pytest.fixture
def empty_config():
    return {}


@pytest.fixture
def basic_config():
    return {
        "sites": {
            "lasilla": "lasilla",
            "astrolab": {"lat": 54.767, "lon": -1.5742, "height": 10},
        }
    }


@pytest.fixture
def greenwich():
    return Observer.at_site("greenwich")


@pytest.fixture(scope="module")
def mar_equinox():
    # Travelling from la silla to la serena to santiago...
    return Time("2025-03-20T12:00:00", format="isot")


class Test__ObsMgrInit:
    def test__no_sites(
        self, empty_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Act
        obs_manager = ObservatoryManager(empty_config, tlookup, path_mgr)

        # Assert
        assert set(obs_manager.sites.keys()) == set(["no_observatory"])

    def test__with_sites(
        self, basic_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Act
        obs_manager = ObservatoryManager(basic_config, tlookup, path_mgr)

        # Assert
        assert set(obs_manager.sites) == set(["no_observatory", "lasilla", "astrolab"])

        expected_config_keys = ["dt", "dt_unit", "horizon", "sites"]
        assert set(obs_manager.config.keys()) == set(expected_config_keys)

        assert obs_manager.sites["no_observatory"] is None
        assert isinstance(obs_manager.sites["lasilla"], Observer)
        assert isinstance(obs_manager.sites["astrolab"], Observer)

        assert obs_manager.sites["lasilla"].name == "lasilla"
        assert obs_manager.sites["astrolab"].name == "astrolab"

        assert u.isclose(obs_manager.sites["astrolab"].elevation, 10.0 * u.m)

    def test__unexpected_key_warns(
        self, basic_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Arrange
        basic_config["bad_kwarg"] = 10.0

        # Act
        with pytest.warns(UnexpectedKeysWarning):
            obs_manager = ObservatoryManager(basic_config, tlookup, path_mgr)

        # Assert
        assert isinstance(obs_manager, ObservatoryManager)  # still init...

    def test__units_applied(
        self, basic_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Arrange
        basic_config["dt"] = 0.5
        basic_config["dt_unit"] = "day"

        # Act
        obs_manager = ObservatoryManager(basic_config, tlookup, path_mgr)

        # Assert
        assert isinstance(obs_manager.config["horizon"], u.Quantity)
        assert obs_manager.config["horizon"].unit.is_equivalent(u.deg)
        assert isinstance(obs_manager.config["dt"], u.Quantity)
        assert obs_manager.config["dt"].unit.is_equivalent(u.s)
        dt_hour = obs_manager.config["dt"].to(u.hour).value
        assert np.isclose(dt_hour, 12.0)

    def test__bad_dt_units_raises(
        self, basic_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Arrange
        basic_config["dt"] = 0.5
        basic_config["dt_unit"] = "meter"

        # Act
        with pytest.raises(u.UnitTypeError):
            obs_mananger = ObservatoryManager(basic_config, tlookup, path_mgr)


class Test__UpdateEphemInfo:
    def test__ephem_info(self, obs_mgr: ObservatoryManager, mar_equinox: Time):
        # Act
        obs_mgr.update_ephem_info(obs_mgr.sites["astrolab"], mar_equinox)

        # Assert
        set(obs_mgr.current_ephem_info.keys()) == set(["astrolab"])
        assert isinstance(obs_mgr.current_ephem_info["astrolab"], EphemInfo)

        assert obs_mgr.ephem_updated["astrolab"]

        ephem_t_ref = obs_mgr.current_ephem_info["astrolab"].t_ref
        assert np.isclose(ephem_t_ref.mjd, mar_equinox.mjd, rtol=1e-8)


class Test__CheckAndUpdate:

    def test__missing_ephem_added(self, obs_mgr: ObservatoryManager, mar_equinox: Time):
        # Act
        obs_mgr.check_and_update_ephem_info(t_ref=mar_equinox)

    def test__no_unnecessary_updates(
        self, obs_mgr: ObservatoryManager, mar_equinox: Time
    ):
        # Arrange
        t_earlier = mar_equinox - 1 * u.hour
        obs_mgr.check_and_update_ephem_info(t_ref=mar_equinox)
        obs_mgr.ephem_updated["lasilla"] = False
        obs_mgr.ephem_updated["astrolab"] = False

        # Act

        # if ephem is from the later than t_ref, shouldn't update
        obs_mgr.check_and_update_ephem_info(t_ref=t_earlier)

        # Assert
        assert obs_mgr.ephem_updated["lasilla"] is False
        assert obs_mgr.ephem_updated["astrolab"] is False


class Test__ApplyEphem:
    def test__apply(self, obs_mgr: ObservatoryManager, mar_equinox: Time):
        # Act
        obs_mgr.apply_ephem_info(t_ref=mar_equinox)

        # Assert
        tl = obs_mgr.target_lookup
        assert isinstance(tl["T00"].ephem_info["astrolab"], EphemInfo)
        assert isinstance(tl["T00"].ephem_info["astrolab"].target_altaz, SkyCoord)
        assert isinstance(tl["T00"].ephem_info["lasilla"], EphemInfo)
        assert isinstance(tl["T00"].ephem_info["lasilla"].target_altaz, SkyCoord)

    def test__no_unnecessary_apply(
        self, obs_mgr: ObservatoryManager, mar_equinox: Time
    ):
        # Arrange
        obs_mgr.apply_ephem_info(t_ref=mar_equinox)
        tl = obs_mgr.target_lookup
        tl["T00"].ephem_info["lasilla"].mark = True
        tl["T01"].ephem_info["astrolab"].mark = True
        tl["T00"].ephem_info.pop("astrolab")  # should now be updated...
        tl["T01"].ephem_info.pop("lasilla")  # should now be updated...

        # Act
        obs_mgr.apply_ephem_info(t_ref=mar_equinox)

        # Assert
        tl = obs_mgr.target_lookup
        assert hasattr(tl["T00"].ephem_info["lasilla"], "mark")
        assert hasattr(tl["T01"].ephem_info["astrolab"], "mark")

        assert not hasattr(tl["T00"].ephem_info["astrolab"], "mark")
        assert isinstance(tl["T00"].ephem_info["astrolab"], EphemInfo)
        assert not hasattr(tl["T01"].ephem_info["lasilla"], "mark")
        assert isinstance(tl["T01"].ephem_info["lasilla"], EphemInfo)
