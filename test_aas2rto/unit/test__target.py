import pytest
from pathlib import Path

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from astroplan import Observer

from aas2rto.exc import UnknownObservatoryWarning
from aas2rto.target import Target
from aas2rto.target_data import TargetData


@pytest.fixture
def t_ref():
    return Time(60000.0, format="mjd")


@pytest.fixture
def mock_coord():
    return SkyCoord(ra=180.0, dec=0.0, unit=u.deg)


@pytest.fixture
def obs_lapalma():
    return Observer.at_site("lapalma")


@pytest.fixture
def mock_target(mock_coord, t_ref):
    return Target("T001", mock_coord, t_ref=t_ref)


@pytest.fixture
def mock_target_data():
    return TargetData(parameters={"redshift": 1.0, "salt_x1": 0.0, "salt_c": 0.1})


@pytest.fixture
def mock_target_with_data(mock_target: Target, mock_target_data: TargetData):
    mock_target.target_data["src01"] = mock_target_data
    return mock_target


@pytest.fixture
def mock_target_with_history(
    mock_target: Target, lasilla: Observer, obs_lapalma: Observer, t_ref: Time
):
    t_later = Time(60001.0, format="mjd")

    # Add data for
    mock_target.update_science_score_history(10.0, t_ref=t_ref)
    mock_target.update_science_score_history(8.0, t_ref=t_later)

    mock_target.update_science_rank_history(1, t_ref=t_ref)
    mock_target.update_science_rank_history(2, t_ref=t_later)

    mock_target.update_obs_score_history(5.0, observatory=lasilla, t_ref=t_ref)
    mock_target.update_obs_score_history(4.0, observatory=lasilla, t_ref=t_later)

    mock_target.update_obs_rank_history(11, observatory=lasilla, t_ref=t_ref)
    mock_target.update_obs_rank_history(12, observatory=lasilla, t_ref=t_later)

    mock_target.update_obs_score_history(2.5, observatory=obs_lapalma, t_ref=t_ref)
    mock_target.update_obs_score_history(2.0, observatory=obs_lapalma, t_ref=t_later)

    mock_target.update_obs_rank_history(21, observatory=obs_lapalma, t_ref=t_ref)
    mock_target.update_obs_rank_history(22, observatory=obs_lapalma, t_ref=t_later)
    return mock_target


class Test__TargetInit:

    def test__basic_init(self, mock_coord, t_ref):
        # Act
        target = Target("mock_target", mock_coord, t_ref=t_ref)

        # Assert
        assert isinstance(target, Target)  # sort of important...
        assert np.isclose(target.coord.ra.deg, 180.0)
        assert np.isclose(target.coord.dec.deg, 0.0)

        assert isinstance(target.coord, SkyCoord)

        assert isinstance(target.ephem_info, dict)
        assert set(target.ephem_info.keys()) == set()

        assert isinstance(target.models, dict)
        assert len(target.models) == 0
        assert isinstance(target.models_t_ref, dict)
        assert len(target.models_t_ref) == 0

        assert isinstance(target.target_data, dict)
        assert len(target.target_data) == 0

        assert isinstance(target.alt_ids, dict)
        assert set(target.alt_ids.keys()) == set(["<unknown>"])
        assert target.alt_ids["<unknown>"] == "mock_target"

        assert np.isclose(target.creation_time.mjd, 60000.0)

        assert target.target_of_opportunity is False
        assert target.updated is False
        assert target.to_reject is False

        assert isinstance(target.update_messages, list)
        assert isinstance(target.sudo_messages, list)

    def test__init_with_source(self):
        # Arrange
        alt_ids = {"src02": "target_A"}

        # Act
        target = Target("T001", 180.0, 0.0, source="src01", alt_ids=alt_ids)

        # Assert
        assert set(target.alt_ids.keys()) == set(["src01", "src02"])
        assert target.alt_ids["src01"] == "T001"
        assert target.alt_ids["src02"] == "target_A"

    def test__init_no_source(self):
        # Arrange
        alt_ids = {"src01": "T001", "src02": "target_A"}

        # Act
        target = Target("T001", 180.0, 0.0, alt_ids=alt_ids)

        # Assert
        assert set(target.alt_ids.keys()) == set(["src01", "src02"])
        assert target.alt_ids["src01"] == "T001"
        assert target.alt_ids["src02"] == "target_A"


# class Test__CoordinatesMethods:
#    def test__update_coordinates(self, mock_target):
#        pass


class Test__TargetDataMethods:
    def test__get_missing_data(self, mock_target: Target):
        # Arrange
        assert "src01" not in mock_target.target_data

        # Act
        target_data = mock_target.get_target_data("src01")

        # Assert
        assert isinstance(target_data, TargetData)
        assert "src01" in mock_target.target_data

    def test__get_existing_data(self, mock_target_with_data: Target):
        # Act
        t_data = mock_target_with_data.get_target_data("src01")

        # Assert
        assert t_data.lightcurve is None
        assert isinstance(t_data.parameters, dict)
        assert set(t_data.parameters.keys()) == set(["redshift", "salt_x1", "salt_c"])

    def test__updating_target_data(self, mock_target: Target):
        # Act
        t_data = mock_target.get_target_data("src02")
        t_data.parameters["m0"] = -19.6

        # Assert
        assert np.isclose(mock_target.target_data["src02"].parameters["m0"], -19.6)


class Test__UpdateScoreHistMethods:
    def test__update_sci_score_hist(self, mock_target: Target, t_ref: Time):
        # Arrange
        assert len(mock_target.science_score_history) == 0

        # Act
        mock_target.update_science_score_history(1.0, t_ref=t_ref)

        # Assert
        assert len(mock_target.science_score_history) == 1
        score_hist0 = mock_target.science_score_history[0]

        assert isinstance(score_hist0, tuple)
        assert len(score_hist0) == 2
        assert np.isclose(score_hist0[0], 1.0)
        assert np.isclose(score_hist0[1], 60000.0)

    def test__update_obs_score_hist_observer(
        self, mock_target: Target, lasilla: Observer, t_ref: Time
    ):
        # Arrange
        assert set(mock_target.obs_score_history.keys()) == set()

        # Act
        mock_target.update_obs_score_history(1.0, lasilla, t_ref=t_ref)

        # Assert
        assert set(mock_target.obs_score_history.keys()) == set(["lasilla"])

        assert len(mock_target.obs_score_history["lasilla"]) == 1
        score_hist0 = mock_target.obs_score_history["lasilla"][0]

        assert isinstance(score_hist0, tuple)
        assert len(score_hist0) == 2
        assert np.isclose(score_hist0[0], 1.0)
        assert np.isclose(score_hist0[1], 60000.0)

    def test__update_obs_score_history_str(
        self, mock_target: Target, lasilla: Observer, t_ref: Time
    ):
        # Arrange
        assert set(mock_target.obs_score_history.keys()) == set()

        # Act
        mock_target.update_obs_score_history(1.0, "lasilla", t_ref=t_ref)

        # Assert
        assert set(mock_target.obs_score_history.keys()) == set(["lasilla"])

        assert len(mock_target.obs_score_history["lasilla"]) == 1
        score_hist0 = mock_target.obs_score_history["lasilla"][0]

        assert isinstance(score_hist0, tuple)
        assert len(score_hist0) == 2
        assert np.isclose(score_hist0[0], 1.0)
        assert np.isclose(score_hist0[1], 60000.0)


class Test__GetScoreHistMethods:

    def test__get_sci_score_hist(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history

        # Act
        score_hist = target.get_science_score_history()

        # Assert
        assert isinstance(score_hist, pd.DataFrame)
        assert len(score_hist) == 2
        assert np.allclose(score_hist["score"].values, np.array([10.0, 8.0]))

    def test__get_obs_score_hist_observer(
        self, mock_target_with_history: Target, lasilla: Observer
    ):
        # Arrange
        target = mock_target_with_history

        # Act
        score_hist = target.get_obs_score_history(lasilla)

        # Assert
        assert isinstance(score_hist, pd.DataFrame)
        assert len(score_hist) == 2
        assert np.allclose(score_hist["score"].values, np.array([5.0, 4.0]))

    def test__get_obs_score_hist_str(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history

        # Act
        score_hist = target.get_obs_score_history("lasilla")

        # Assert
        assert isinstance(score_hist, pd.DataFrame)
        assert len(score_hist) == 2
        assert np.allclose(score_hist["score"].values, np.array([5.0, 4.0]))

    def test__get_sci_score_hist_empty_no_fail(self, mock_target: Target):
        # Act
        score_hist = mock_target.get_science_score_history()

        # Assert
        assert isinstance(score_hist, pd.DataFrame)
        assert score_hist.empty
        assert set(score_hist.columns) == set(["score", "mjd"])

    def test__get_sci_hist_missing_obs_no_fail(self, mock_target: Target):
        # Act
        with pytest.warns(UnknownObservatoryWarning):
            score_hist = mock_target.get_obs_score_history("palomar")

        # Assert
        assert isinstance(score_hist, pd.DataFrame)
        assert score_hist.empty
        assert set(score_hist.columns) == set(["score", "mjd"])

    def test__get_sci_score_history_limit_t_ref(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history
        t_mid = Time(60000.5, format="mjd")

        # Act
        score_hist = target.get_science_score_history(t_ref=t_mid)

        # Assert
        assert len(score_hist) == 1

        # only keep the ones before mjd=60000.5
        assert np.isclose(score_hist["mjd"].iloc[0] - 60000.0, 0.0)

    def test__get_last_science_score(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        no_obs_score = target.get_latest_science_score()

        # Assert
        assert np.isclose(no_obs_score, 8.0)

    def test__get_last_obs_score_str(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        lasilla_score = target.get_latest_obs_score("lasilla")

        # Assert
        assert np.isclose(lasilla_score, 4.0)

    def test__get_last_obs_score_observer(
        self, mock_target_with_history: Target, lasilla: Observer
    ):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        lasilla_score = target.get_latest_obs_score(lasilla)

        # Assert
        assert np.isclose(lasilla_score, 4.0)

    def test__get_last_score_returns_time(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        last_score, mjd = target.get_latest_science_score(return_time=True)

        # Assert
        assert np.isclose(last_score, 8.0)
        assert np.isclose(mjd - 60001.0, 0.0)

    def test__get_last_score_missing_obs_no_fail(
        self, mock_target_with_history: Target
    ):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        # with pytest.warns(UnknownObservatoryWarning):
        palomar_score = target.get_latest_obs_score("palomar")

        assert palomar_score is None


class Test__UpdateRankHistMethods:
    def test__update_sci_rank_hist(self, mock_target: Target, t_ref: Time):
        # Act
        mock_target.update_science_rank_history(1, t_ref=t_ref)

        # Assert
        assert len(mock_target.science_rank_history) == 1
        rank_hist0 = mock_target.science_rank_history[0]

        assert isinstance(rank_hist0, tuple)
        assert len(rank_hist0) == 2
        assert np.isclose(rank_hist0[0], 1)
        assert np.isclose(rank_hist0[1], 60000.0)

    def test__update_obs_rank_hist_str(self, mock_target: Target, t_ref: Time):
        # Arrange
        assert set(mock_target.obs_rank_history.keys()) == set()

        # Act
        mock_target.update_obs_rank_history(99, "lasilla", t_ref=t_ref)

        # Assert
        assert set(mock_target.obs_rank_history.keys()) == set(["lasilla"])

        assert len(mock_target.obs_rank_history["lasilla"]) == 1
        rank_hist0 = mock_target.obs_rank_history["lasilla"][0]

        assert isinstance(rank_hist0, tuple)
        assert len(rank_hist0) == 2
        assert np.isclose(rank_hist0[0], 99)
        assert np.isclose(rank_hist0[1], 60000.0)

    def test__update_obs_rank_history_observer(
        self, mock_target: Target, lasilla: Observer, t_ref: Time
    ):
        # Act
        mock_target.update_obs_rank_history(1.0, lasilla, t_ref=t_ref)

        # Assert
        assert set(mock_target.obs_rank_history.keys()) == set(["lasilla"])

        assert len(mock_target.obs_rank_history["lasilla"]) == 1
        rank_hist0 = mock_target.obs_rank_history["lasilla"][0]

        assert isinstance(rank_hist0, tuple)
        assert len(rank_hist0) == 2
        assert np.isclose(rank_hist0[0], 1)
        assert np.isclose(rank_hist0[1], 60000.0)


class Test__GetRankHistMethods:
    def test__get_sci_rank_history(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        rank_hist = target.get_science_rank_history()

        # Assert
        assert isinstance(rank_hist, pd.DataFrame)
        assert len(rank_hist) == 2
        assert set(rank_hist.columns) == set(["ranking", "mjd"])

    def test__get_sci_rank_hist_empty_no_fail(self, mock_target: Target):
        # Act
        rank_hist = mock_target.get_science_rank_history()

        # Assert
        assert isinstance(rank_hist, pd.DataFrame)
        assert rank_hist.empty
        assert set(rank_hist.columns) == set(["ranking", "mjd"])

    def test__get_obs_rank_hist_str(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history

        # Act
        rank_hist = target.get_obs_rank_history(observatory="lasilla")

        # Assert
        assert isinstance(rank_hist, pd.DataFrame)
        assert len(rank_hist) == 2
        assert np.allclose(rank_hist["ranking"].values, np.array([11, 12]))

    def test__get_obs_rank_hist_observer(
        self, mock_target_with_history: Target, lasilla: Observer
    ):
        # Arrange
        target = mock_target_with_history

        # Act
        rank_hist = target.get_obs_rank_history(lasilla)

        # Assert
        assert isinstance(rank_hist, pd.DataFrame)
        assert len(rank_hist) == 2
        assert np.allclose(rank_hist["ranking"].values, np.array([11, 12]))

    def test__get_rank_hist_missing_obs_no_fail(self, mock_target: Target):
        # Act
        with pytest.warns(UnknownObservatoryWarning):
            rank_hist = mock_target.get_obs_rank_history(observatory="palomar")

        # Assert
        assert isinstance(rank_hist, pd.DataFrame)
        assert rank_hist.empty
        assert set(rank_hist.columns) == set(["ranking", "mjd"])

    def test__get_rank_hist_limit_t_ref(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history
        t_mid = Time(60000.5, format="mjd")

        # Act
        rank_hist = target.get_science_rank_history(t_ref=t_mid)

        # Assert
        assert len(rank_hist) == 1

        # only keep the ones before mjd=60000.5
        assert np.allclose(rank_hist["mjd"] - 60000.0, 0.0)

    def test__get_last_sci_rank(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        no_obs_rank = target.get_latest_science_rank()

        # Assert
        assert no_obs_rank == 2

    def test__get_last_obs_rank_str(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        lasilla_rank = target.get_latest_obs_rank("lasilla")

        # Assert
        assert lasilla_rank == 12

    def test__get_last_obs_rank_observer(
        self, mock_target_with_history: Target, lasilla: Observer
    ):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        lasilla_rank = target.get_latest_obs_rank(lasilla)

        # Assert
        assert lasilla_rank == 12

    def test__get_last_rank_returns_time(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        last_rank, mjd = target.get_latest_science_rank(return_time=True)

        # Assert
        assert last_rank == 2
        assert np.isclose(mjd, 60001.0)

    def test__get_last_rank_missing_obs_no_fail(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        #  with pytest.warns(UnknownObservatoryWarning):
        palomar_rank = target.get_latest_obs_rank(observatory="palomar")

        assert palomar_rank is None


class Test__InfoLines:
    # These tests are pretty silly - just check no crash.
    def test__target_id_lines(self, mock_target: Target):
        # Arrange
        mock_target.alt_ids["ztf"] = "ZTF25abc"
        mock_target.alt_ids["tns"] = "2025xyz"
        mock_target.alt_ids["yse"] = "fYSE25_001"

        # Act
        id_lines = mock_target.get_target_id_info_lines()
        info_str = " ".join(id_lines)

        # Assert
        assert "Aliases and brokers" in info_str
        assert "FINK: fink-portal.org/ZTF25abc" in info_str
        assert "Lasair: lasair-ztf.lsst.ac.uk/objects/ZTF25abc" in info_str
        assert "ALeRCE: alerce.online/object/ZTF25abc" in info_str
        assert "TNS: wis-tns.org/object/2025xyz" in info_str
        assert "YSE: ziggy.ucolick.org/yse/transient_detail/fYSE25_001" in info_str

        assert "alt names" in info_str

    def test__coordinate_info_lines(self, mock_target: Target):
        # Act
        info_lines = mock_target.get_coordinate_info_lines()
        info_str = " ".join(info_lines)

        # Assert
        assert "equatorial (ra, dec)" in info_str
        assert "galactic (l, b)" in info_str


class Test__WriteComments:
    def test__no_available_comments(
        self, mock_target: Target, tmp_path: Path, t_ref: Time
    ):
        # Arrange
        expected_comms_path = tmp_path / "T001_comments.txt"
        assert not expected_comms_path.exists()

        # Act
        mock_target.write_comments(tmp_path, t_ref=t_ref)

        # Assert
        assert expected_comms_path.exists()
        with open(expected_comms_path) as f:
            comms = f.readlines()
        comm_str = " ".join(comms)

        assert "Target T001 at " in comm_str
        assert "Aliases and brokers" in comm_str
        assert "Coordinates" in comm_str
        assert "Photometry" in comm_str

        assert "no score_comments provided" in comm_str

    def test__more_comments_available(
        self, mock_target: Target, tmp_path: Path, t_fixed: Time
    ):
        # Arrange
        expected_comms_path = tmp_path / "T001_comments.txt"
        mock_target.update_science_score_history(1.0)
        mock_target.science_comments = ["the score is 1.0"]
        mock_target.update_obs_score_history(0.5, "lasilla")
        mock_target.obs_comments["lasilla"] = ["lasilla score is 0.5"]

        # Act
        mock_target.write_comments(tmp_path, t_ref=t_fixed)

        # Assert
        assert expected_comms_path.exists()
        with open(expected_comms_path) as f:
            comms = f.readlines()
        comm_str = " ".join(comms)

        assert "Target T001 at " in comm_str
        assert "Aliases and brokers" in comm_str
        assert "Coordinates" in comm_str
        assert "Photometry" in comm_str
