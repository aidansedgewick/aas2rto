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
    mock_target.update_score_history(10.0, t_ref=t_ref)
    mock_target.update_score_history(8.0, t_ref=t_later)

    mock_target.update_rank_history(1, t_ref=t_ref)
    mock_target.update_rank_history(2, t_ref=t_later)

    mock_target.update_score_history(5.0, observatory=lasilla, t_ref=t_ref)
    mock_target.update_score_history(4.0, observatory=lasilla, t_ref=t_later)

    mock_target.update_rank_history(11, observatory=lasilla, t_ref=t_ref)
    mock_target.update_rank_history(12, observatory=lasilla, t_ref=t_later)

    mock_target.update_score_history(2.5, observatory=obs_lapalma, t_ref=t_ref)
    mock_target.update_score_history(2.0, observatory=obs_lapalma, t_ref=t_later)

    mock_target.update_rank_history(21, observatory=obs_lapalma, t_ref=t_ref)
    mock_target.update_rank_history(22, observatory=obs_lapalma, t_ref=t_later)
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


class Test__ScoreHistoryMethods:
    def test__update_score_history_no_observatory(
        self, mock_target: Target, t_ref: Time
    ):
        # Arrange
        assert len(mock_target.score_history["no_observatory"]) == 0

        # Act
        mock_target.update_score_history(1.0, t_ref=t_ref)

        # Assert
        assert len(mock_target.score_history["no_observatory"]) == 1
        score_hist0 = mock_target.score_history["no_observatory"][0]

        assert isinstance(score_hist0, tuple)
        assert len(score_hist0) == 2
        assert np.isclose(score_hist0[0], 1.0)
        assert np.isclose(score_hist0[1], 60000.0)

    def test__update_score_history_observer_class(
        self, mock_target: Target, lasilla: Observer, t_ref: Time
    ):
        # Arrange
        assert set(mock_target.score_history.keys()) == set(["no_observatory"])

        # Act
        mock_target.update_score_history(1.0, observatory=lasilla, t_ref=t_ref)

        # Assert
        expected_keys = ["no_observatory", "lasilla"]
        assert set(mock_target.score_history.keys()) == set(expected_keys)

        assert len(mock_target.score_history["no_observatory"]) == 0
        assert len(mock_target.score_history["lasilla"]) == 1
        score_hist0 = mock_target.score_history["lasilla"][0]

        assert isinstance(score_hist0, tuple)
        assert len(score_hist0) == 2
        assert np.isclose(score_hist0[0], 1.0)
        assert np.isclose(score_hist0[1], 60000.0)

    def test__update_score_history_str_observer(
        self, mock_target: Target, lasilla: Observer, t_ref: Time
    ):
        # Arrange
        assert set(mock_target.score_history.keys()) == set(["no_observatory"])

        # Act
        mock_target.update_score_history(1.0, observatory=lasilla, t_ref=t_ref)

        # Assert
        expected_keys = ["no_observatory", "lasilla"]
        assert set(mock_target.score_history.keys()) == set(expected_keys)

        assert len(mock_target.score_history["no_observatory"]) == 0
        assert len(mock_target.score_history["lasilla"]) == 1
        score_hist0 = mock_target.score_history["lasilla"][0]

        assert isinstance(score_hist0, tuple)
        assert len(score_hist0) == 2
        assert np.isclose(score_hist0[0], 1.0)
        assert np.isclose(score_hist0[1], 60000.0)

    def test__get_score_history_empty_no_fail(self, mock_target: Target):
        # Act
        score_hist = mock_target.get_score_history()

        # Assert
        assert isinstance(score_hist, pd.DataFrame)
        assert score_hist.empty
        assert set(score_hist.columns) == set(["score", "mjd", "observatory"])

    def test__get_rank_history_missing_obs_no_fail(self, mock_target: Target):
        # Act
        with pytest.warns(UnknownObservatoryWarning):
            score_hist = mock_target.get_score_history(observatory="palomar")

        # Assert
        assert isinstance(score_hist, pd.DataFrame)
        assert score_hist.empty
        assert set(score_hist.columns) == set(["score", "mjd", "observatory"])

    def test__get_score_history_no_obs(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        score_hist = target.get_score_history()

        # Assert
        assert isinstance(score_hist, pd.DataFrame)
        assert len(score_hist) == 6
        assert set(score_hist.columns) == set(["score", "mjd", "observatory"])

        expected_obs_names = ["no_observatory", "lasilla", "lapalma"]
        assert set(score_hist["observatory"].values) == set(expected_obs_names)

        # result should be sorted in (obs, mjd) - lasilla first alphabetically
        assert np.isclose(score_hist["score"].iloc[0], 2.5)
        assert np.isclose(score_hist["mjd"].iloc[0], 60000.0)
        assert score_hist["observatory"].iloc[0] == "lapalma"

        assert np.isclose(score_hist["score"].iloc[1], 2.0)
        assert np.isclose(score_hist["mjd"].iloc[1], 60001.0)
        assert score_hist["observatory"].iloc[1] == "lapalma"

        assert np.isclose(score_hist["score"].iloc[3], 4.0)
        assert np.isclose(score_hist["mjd"].iloc[3], 60001.0)
        assert score_hist["observatory"].iloc[3] == "lasilla"

        assert np.isclose(score_hist["score"].iloc[4], 10.0)
        assert np.isclose(score_hist["mjd"].iloc[4], 60000.0)
        assert score_hist["observatory"].iloc[4] == "no_observatory"

    def test__get_score_history_obs_name(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history

        # Act
        score_hist = target.get_score_history(observatory="lasilla")

        # Assert
        assert len(score_hist) == 2
        assert set(score_hist["observatory"]) == set(["lasilla"])

    def test__get_score_history_obs_none(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history

        # Act
        score_hist = target.get_score_history(observatory=None)

        # Assert
        assert len(score_hist) == 2
        assert set(score_hist["observatory"]) == set(["no_observatory"])
        assert np.allclose(score_hist["score"].values, np.array([10.0, 8.0]))

    def test__get_score_hist_observer_object(
        self, mock_target_with_history: Target, lasilla: Observer
    ):
        # Arrange
        target = mock_target_with_history

        # Act
        score_hist = target.get_score_history(observatory=lasilla)

        # Assert
        assert len(score_hist) == 2
        assert set(score_hist["observatory"]) == set(["lasilla"])
        assert np.allclose(score_hist["score"].values, np.array([5.0, 4.0]))

    def test__get_score_history_limit_t_ref(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history
        t_mid = Time(60000.5, format="mjd")

        # Act
        score_hist = target.get_score_history(t_ref=t_mid)

        # Assert
        assert len(score_hist) == 3

        # only keep the ones before mjd=60000.5
        assert np.allclose(score_hist["mjd"], 60000.0)

    def test__get_last_score_no_obs(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        no_obs_score = target.get_last_score()

        # Assert
        assert np.isclose(no_obs_score, 8.0)

    def test__get_last_score_str_observer(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        lasilla_score = target.get_last_score(observatory="lasilla")

        # Assert
        assert np.isclose(lasilla_score, 4.0)

    def test__get_last_score_observer_obj(
        self, mock_target_with_history: Target, lasilla: Observer
    ):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        lasilla_score = target.get_last_score(observatory=lasilla)

        # Assert
        assert np.isclose(lasilla_score, 4.0)

    def test__get_last_score_returns_time(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        last_score, mjd = target.get_last_score(return_time=True)

        # Assert
        assert np.isclose(last_score, 8.0)
        assert np.isclose(mjd - 60001.0, 0.0)

    def test__get_last_score_missing_obs_no_fail(
        self, mock_target_with_history: Target
    ):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        with pytest.warns(UnknownObservatoryWarning):
            palomar_score = target.get_last_score(observatory="palomar")

        assert palomar_score is None


class Test__RankHistoryMethods:
    def test__update_rank_history_no_observatory(
        self, mock_target: Target, t_ref: Time
    ):
        # Arrange
        assert len(mock_target.rank_history["no_observatory"]) == 0

        # Act
        mock_target.update_rank_history(1, t_ref=t_ref)

        # Assert
        assert len(mock_target.rank_history["no_observatory"]) == 1
        rank_hist0 = mock_target.rank_history["no_observatory"][0]

        assert isinstance(rank_hist0, tuple)
        assert len(rank_hist0) == 2
        assert np.isclose(rank_hist0[0], 1)
        assert np.isclose(rank_hist0[1], 60000.0)

    def test__update_rank_history_observer_class(
        self, mock_target: Target, lasilla: Observer, t_ref: Time
    ):
        # Arrange
        assert set(mock_target.rank_history.keys()) == set(["no_observatory"])

        # Act
        mock_target.update_rank_history(1.0, observatory=lasilla, t_ref=t_ref)

        # Assert
        expected_keys = ["no_observatory", "lasilla"]
        assert set(mock_target.rank_history.keys()) == set(expected_keys)

        assert len(mock_target.rank_history["no_observatory"]) == 0
        assert len(mock_target.rank_history["lasilla"]) == 1
        rank_hist0 = mock_target.rank_history["lasilla"][0]

        assert isinstance(rank_hist0, tuple)
        assert len(rank_hist0) == 2
        assert np.isclose(rank_hist0[0], 1)
        assert np.isclose(rank_hist0[1], 60000.0)

    def test__update_rank_history_str_observer(self, mock_target: Target, t_ref: Time):
        # Arrange
        assert set(mock_target.rank_history.keys()) == set(["no_observatory"])

        # Act
        mock_target.update_rank_history(99, observatory="lasilla", t_ref=t_ref)

        # Assert
        expected_keys = ["no_observatory", "lasilla"]
        assert set(mock_target.rank_history.keys()) == set(expected_keys)

        assert len(mock_target.rank_history["no_observatory"]) == 0
        assert len(mock_target.rank_history["lasilla"]) == 1
        rank_hist0 = mock_target.rank_history["lasilla"][0]

        assert isinstance(rank_hist0, tuple)
        assert len(rank_hist0) == 2
        assert np.isclose(rank_hist0[0], 99)
        assert np.isclose(rank_hist0[1], 60000.0)

    def test__get_rank_history_empty_no_fail(self, mock_target: Target):
        # Act
        rank_hist = mock_target.get_rank_history()

        # Assert
        assert isinstance(rank_hist, pd.DataFrame)
        assert rank_hist.empty
        assert set(rank_hist.columns) == set(["ranking", "mjd", "observatory"])

    def test__get_rank_history_missing_obs_no_fail(self, mock_target: Target):
        # Act
        with pytest.warns(UnknownObservatoryWarning):
            rank_hist = mock_target.get_rank_history(observatory="palomar")

        # Assert
        assert isinstance(rank_hist, pd.DataFrame)
        assert rank_hist.empty
        assert set(rank_hist.columns) == set(["ranking", "mjd", "observatory"])

    def test__get_rank_history_no_obs(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        rank_hist = target.get_rank_history()

        # Assert
        assert isinstance(rank_hist, pd.DataFrame)
        assert len(rank_hist) == 6
        assert set(rank_hist.columns) == set(["ranking", "mjd", "observatory"])

        expected_obs_names = ["no_observatory", "lasilla", "lapalma"]
        assert set(rank_hist["observatory"].values) == set(expected_obs_names)

        # result should be sorted in (obs, mjd) - lasilla first alphabetically
        assert np.isclose(rank_hist["ranking"].iloc[0], 21)
        assert np.isclose(rank_hist["mjd"].iloc[0], 60000.0)
        assert rank_hist["observatory"].iloc[0] == "lapalma"

        assert np.isclose(rank_hist["ranking"].iloc[1], 22)
        assert np.isclose(rank_hist["mjd"].iloc[1], 60001.0)
        assert rank_hist["observatory"].iloc[1] == "lapalma"

        assert np.isclose(rank_hist["ranking"].iloc[3], 12)
        assert np.isclose(rank_hist["mjd"].iloc[3], 60001.0)
        assert rank_hist["observatory"].iloc[3] == "lasilla"

        assert np.isclose(rank_hist["ranking"].iloc[4], 1)
        assert np.isclose(rank_hist["mjd"].iloc[4], 60000.0)
        assert rank_hist["observatory"].iloc[4] == "no_observatory"

    def test__get_rank_history_obs_name(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history

        # Act
        rank_hist = target.get_rank_history(observatory="lasilla")

        # Assert
        assert len(rank_hist) == 2
        assert set(rank_hist["observatory"]) == set(["lasilla"])

    def test__get_rank_history_obs_none(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history

        # Act
        rank_hist = target.get_rank_history(observatory=None)

        # Assert
        assert len(rank_hist) == 2
        assert set(rank_hist["observatory"]) == set(["no_observatory"])
        assert np.allclose(rank_hist["ranking"].values, np.array([1, 2]))

    def test__get_rank_hist_observer_object(
        self, mock_target_with_history: Target, lasilla: Observer
    ):
        # Arrange
        target = mock_target_with_history

        # Act
        rank_hist = target.get_rank_history(observatory=lasilla)

        # Assert
        assert len(rank_hist) == 2
        assert set(rank_hist["observatory"]) == set(["lasilla"])
        assert np.allclose(rank_hist["ranking"].values, np.array([11, 12]))

    def test__get_rank_history_limit_t_ref(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history
        t_mid = Time(60000.5, format="mjd")

        # Act
        rank_hist = target.get_rank_history(t_ref=t_mid)

        # Assert
        assert len(rank_hist) == 3

        # only keep the ones before mjd=60000.5
        assert np.allclose(rank_hist["mjd"], 60000.0)

    def test__get_last_rank_no_obs(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        no_obs_rank = target.get_last_rank()

        # Assert
        assert no_obs_rank == 2

    def test__get_last_rank_str_observer(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        lasilla_rank = target.get_last_rank(observatory="lasilla")

        # Assert
        assert lasilla_rank == 12

    def test__get_last_rank_observer_obj(
        self, mock_target_with_history: Target, lasilla: Observer
    ):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        lasilla_rank = target.get_last_rank(observatory=lasilla)

        # Assert
        assert lasilla_rank == 12

    def test__get_last_rank_returns_time(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        last_rank, mjd = target.get_last_rank(return_time=True)

        # Assert
        assert last_rank == 2
        assert np.isclose(mjd, 60001.0)

    def test__get_last_rank_missing_obs_no_fail(self, mock_target_with_history: Target):
        # Arrange
        target = mock_target_with_history  # shorter name is nicer...

        # Act
        with pytest.warns(UnknownObservatoryWarning):
            palomar_rank = target.get_last_rank(observatory="palomar")

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
        self, mock_target: Target, tmp_path: Path, t_ref: Time
    ):
        # Arrange
        expected_comms_path = tmp_path / "T001_comments.txt"
        mock_target.update_score_history(1.0)
        mock_target.score_comments["no_observatory"] = ["the score is 1.0"]
        mock_target.update_score_history(0.5, observatory="lasilla")
        mock_target.score_comments["lasilla"] = ["lasilla score is 0.5"]

        # Act
        mock_target.write_comments(tmp_path)

        # Assert
        assert expected_comms_path.exists()
        with open(expected_comms_path) as f:
            comms = f.readlines()
        comm_str = " ".join(comms)

        assert "Target T001 at " in comm_str
        assert "Aliases and brokers" in comm_str
        assert "Coordinates" in comm_str
        assert "Photometry" in comm_str
