import pytest

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time

from astroplan import FixedTarget, Observer

from dk154_targets.exc import MissingDateError, UnknownObservatoryWarning
from dk154_targets.obs_info import ObservatoryInfo
from dk154_targets.target import Target, TargetData, SettingLightcurveDirectlyWarning


@pytest.fixture
def mock_lc_rows():
    return [
        (60000.0, 1, 20.5, 0.1, "upperlim"),
        (60001.0, 2, 20.4, 0.1, "upperlim"),
        (60002.0, 3, 20.3, 0.1, "nondet"),
        (60003.0, 4, 20.2, 0.1, "nondet"),
        (60004.0, 5, 20.1, 0.2, "badquality"),
        (60005.0, 6, 20.1, 0.2, "badqual"),
        (60006.0, 7, 20.1, 0.2, "dubious"),
        (60007.0, 8, 20.0, 0.1, "valid"),
        (60008.0, 9, 20.0, 0.1, "valid"),
        (60009.0, 10, 20.0, 0.1, "valid"),
    ]


@pytest.fixture
def mock_lc(mock_lc_rows):
    return pd.DataFrame(mock_lc_rows, columns="mjd obsId mag magerr tag".split())


@pytest.fixture
def mock_lc_astropy(mock_lc_rows):
    return Table(rows=mock_lc_rows, names="mjd obsId mag magerr tag".split())


@pytest.fixture
def mock_lc_updates_rows():
    return [
        (60010.0, 11, 19.9, 0.05, "valid"),
        (60011.0, 12, 19.8, 0.05, "valid"),
        (60012.0, 13, 19.7, 0.2, "badqual"),
    ]


@pytest.fixture
def mock_lc_updates(mock_lc_updates_rows):
    return pd.DataFrame(
        mock_lc_updates_rows, columns="mjd obsId mag magerr tag".split()
    )


@pytest.fixture
def mock_lc_updates_astropy(mock_lc_updates_rows):
    return Table(rows=mock_lc_updates_rows, names="mjd obsId mag magerr tag".split())


@pytest.fixture
def mock_target():
    return Target("T101", ra=45.0, dec=60.0)


@pytest.fixture
def test_observer():
    location = EarthLocation(lat=55.6802, lon=12.5724, height=0.0)
    return Observer(location, name="ucph")


def basic_lc_compiler(target: Target, t_ref: Time):
    lc = target.target_data["ztf"].detections.copy()
    lc.loc[:, "flux"] = 3631.0 * 10 ** (-0.4 * lc["mag"])
    lc.loc[:, "fluxerr"] = lc["flux"] * lc["magerr"]
    lc.loc[:, "band"] = "ztf-w"
    return lc


class Test__TargetData:
    def test__target_data_init(self):
        td = TargetData()

        assert isinstance(td.meta, dict)
        assert len(td.meta) == 0
        assert td.lightcurve is None
        assert td.detections is None
        assert td.badqual is None
        assert td.non_detections is None

        assert isinstance(td.probabilities, dict)
        assert len(td.probabilities) == 0
        assert isinstance(td.parameters, dict)
        assert len(td.parameters) == 0
        assert isinstance(td.cutouts, dict)
        assert len(td.cutouts) == 0

        assert set(td.valid_tags) == set(["valid", "detection", "det"])
        assert set(td.badqual_tags) == set(["badquality", "badqual", "dubious"])
        assert set(td.nondet_tags) == set(["upperlim", "nondet"])

    def test__change_tags(self):
        td = TargetData(valid_tags=(["valid"]))
        assert set(td.valid_tags) == set(["valid"])

    def test__init_with_lc_no_tag(self, mock_lc):
        mock_lc.drop("tag", axis=1, inplace=True)
        assert not "tag" in mock_lc.columns

        td = TargetData(lightcurve=mock_lc)
        assert len(td.lightcurve) == 10
        assert len(td.detections) == 10
        assert td.badqual is None
        assert td.non_detections is None

    def test__init_with_tag(self, mock_lc):
        td = TargetData(lightcurve=mock_lc, include_badqual=False)  # default False

        assert len(td.lightcurve) == 10
        assert len(td.detections) == 3
        assert len(td.badqual) == 3
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([8, 9, 10])
        assert set(td.badqual["obsId"]) == set([5, 6, 7])
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__init_badqual_true(self, mock_lc):
        td = TargetData(lightcurve=mock_lc, include_badqual=True)

        assert len(td.lightcurve) == 10
        assert len(td.detections) == 6
        assert len(td.badqual) == 0
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([5, 6, 7, 8, 9, 10])
        assert set(td.badqual["obsId"]) == set()  # Test the column name is there.
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__no_date_column_raises_error(self, mock_lc):
        mock_lc.drop("mjd", axis=1, inplace=True)

        td = TargetData()
        with pytest.raises(MissingDateError):
            td.add_lightcurve(mock_lc)

    def test__set_lc_directly_raises_warning(self, mock_lc):
        td = TargetData()
        with pytest.warns(SettingLightcurveDirectlyWarning):
            td.lightcurve = mock_lc

        assert isinstance(td.lightcurve, pd.DataFrame)
        assert len(td.lightcurve) == 10

    def test__empty_cutouts(self):
        td = TargetData()
        result = td.empty_cutouts()
        assert isinstance(result, dict)
        assert len(result) == 0


class Test__IntegratingUpdates:
    def test__integrate_equality_no_lightcurve(self, mock_lc_updates):
        td = TargetData()
        assert td.lightcurve is None

        result = td.integrate_lightcurve_updates_equality(
            mock_lc_updates, column="obsId"
        )
        assert len(result) == 3
        assert len(td.lightcurve) == 3
        assert len(td.detections) == 2
        assert len(td.badqual) == 1

    def test__integrate_equality(self, mock_lc, mock_lc_updates):
        td = TargetData(lightcurve=mock_lc)

        result = td.integrate_lightcurve_updates_equality(
            mock_lc_updates, column="obsId"
        )
        assert len(result) == 13
        assert set(result["obsId"]) == set(range(1, 14))

    def test__integrate_equality_keep_updated(self, mock_lc, mock_lc_updates):
        updates = pd.concat([mock_lc[7:], mock_lc_updates], ignore_index=True)
        assert len(updates) == 6
        updates["source"] = "updates"
        mock_lc["source"] = "original"

        td = TargetData(lightcurve=mock_lc)
        result = td.integrate_lightcurve_updates_equality(updates, column="obsId")
        assert len(result) == 13
        assert all(result.iloc[:7].source == "original")
        assert all(result.iloc[7:].source == "updates")
        assert len(result[result["source"] == "original"]) == 7
        assert len(result[result["source"] == "updates"]) == 6

    def test__integrate_equality_keep_original(self, mock_lc, mock_lc_updates):
        updates = pd.concat([mock_lc[7:], mock_lc_updates], ignore_index=True)
        assert len(updates) == 6
        updates["source"] = "updates"
        mock_lc["source"] = "original"

        td = TargetData(lightcurve=mock_lc)
        result = td.integrate_lightcurve_updates_equality(
            updates, column="obsId", keep_updates=False
        )
        assert len(result) == 13
        assert all(result.iloc[:10].source == "original")
        assert all(result.iloc[10:].source == "updates")
        assert len(result[result["source"] == "original"]) == 10
        assert len(result[result["source"] == "updates"]) == 3

    def test__integrate_updates(self, mock_lc, mock_lc_updates):
        td = TargetData(lightcurve=mock_lc)
        td.integrate_lightcurve_updates_equality(mock_lc_updates, column="obsId")

        assert len(td.lightcurve) == 13
        assert len(td.detections) == 5
        assert len(td.badqual) == 4
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([8, 9, 10, 11, 12])
        assert set(td.badqual["obsId"]) == set([5, 6, 7, 13])
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__integrate_updates_badqual_true(self, mock_lc, mock_lc_updates):
        td = TargetData(lightcurve=mock_lc)
        td.integrate_lightcurve_updates_equality(
            mock_lc_updates, column="obsId", include_badqual=True
        )

        assert len(td.lightcurve) == 13
        assert len(td.detections) == 9
        assert len(td.badqual) == 0
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([5, 6, 7, 8, 9, 10, 11, 12, 13])
        assert set(td.badqual["obsId"]) == set()  # Test the column name is there.
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__integrate_updates_drop_repeated_rows(self, mock_lc, mock_lc_updates):
        mock_lc.loc[mock_lc["tag"] == "upperlim", "obsId"] = -1
        mock_lc.loc[mock_lc["tag"] == "nondet", "obsId"] = -1

        assert all(mock_lc["obsId"].iloc[:3].values == -1)

        td = TargetData(lightcurve=mock_lc)
        assert len(td.non_detections) == 4
        assert len(td.badqual) == 3
        assert len(td.detections) == 3

        td.integrate_lightcurve_updates_equality(mock_lc_updates, column="obsId")

        assert len(td.lightcurve) == 10  # 4 rows with obsId==-1 -- drop all but last!
        assert len(td.non_detections) == 1
        assert np.isclose(td.non_detections["mjd"].iloc[0], 60003.0)
        assert len(td.badqual) == 4
        assert len(td.detections) == 5

    def test__integrate_updates_keep_repeated_rows(self, mock_lc, mock_lc_updates):
        mock_lc.loc[mock_lc["tag"] == "upperlim", "obsId"] = -1
        mock_lc.loc[mock_lc["tag"] == "nondet", "obsId"] = -1

        assert all(mock_lc["obsId"].iloc[:3].values == -1)

        td = TargetData(lightcurve=mock_lc)
        assert len(td.non_detections) == 4
        assert len(td.badqual) == 3
        assert len(td.detections) == 3

        td.integrate_lightcurve_updates_equality(
            mock_lc_updates, column="obsId", ignore_values=[-1]
        )

        assert len(td.lightcurve) == 13
        assert len(td.non_detections) == 4
        assert np.isclose(td.non_detections["mjd"].iloc[0], 60000.0)
        assert np.isclose(td.non_detections["mjd"].iloc[3], 60003.0)
        assert len(td.badqual) == 4
        assert len(td.detections) == 5


class Test__TargetDataAstropyTableCompatibility:
    def test__init_with_table_no_tag(self, mock_lc_astropy):
        mock_lc_astropy.remove_column("tag")
        assert "tag" not in mock_lc_astropy.columns

        td = TargetData(lightcurve=mock_lc_astropy)

        assert len(td.lightcurve) == 10
        # assert td.detections is None
        assert len(td.detections) == 10
        assert td.badqual is None
        assert td.non_detections is None

    def test__init_with_table_with_tag(self, mock_lc_astropy):
        td = TargetData(mock_lc_astropy, include_badqual=False)

        assert len(td.lightcurve) == 10
        assert len(td.detections) == 3
        assert len(td.badqual) == 3
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([8, 9, 10])
        assert set(td.badqual["obsId"]) == set([5, 6, 7])
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__init_with_table_badqual_true(self, mock_lc_astropy):
        td = TargetData(lightcurve=mock_lc_astropy, include_badqual=True)

        assert len(td.lightcurve) == 10
        assert len(td.detections) == 6
        assert len(td.badqual) == 0
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([5, 6, 7, 8, 9, 10])
        assert set(td.badqual["obsId"]) == set()
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__equality_no_lc_astropy(self, mock_lc_updates):
        td = TargetData()
        result = td.integrate_lightcurve_updates_equality(
            mock_lc_updates, column="obsId"
        )
        assert len(result) == 3

    def test__integrate_equality(self, mock_lc, mock_lc_updates):
        td = TargetData(lightcurve=mock_lc)

        result = td.integrate_lightcurve_updates_equality(
            mock_lc_updates, column="obsId"
        )
        assert len(result) == 13
        assert set(result["obsId"]) == set(range(1, 14))

    def test__equality_keep_updated_astropy(
        self, mock_lc_astropy, mock_lc_updates_astropy
    ):
        updates = vstack([mock_lc_astropy[7:], mock_lc_updates_astropy])
        assert len(updates) == 6
        updates["source"] = "updates"
        mock_lc_astropy["source"] = "original"

        td = TargetData(lightcurve=mock_lc_astropy)
        result = td.integrate_lightcurve_updates_equality(updates, column="obsId")
        assert len(result) == 13
        assert all(result[:7]["source"] == "original")
        assert all(result[7:]["source"] == "updates")
        assert len(result[result["source"] == "original"]) == 7
        assert len(result[result["source"] == "updates"]) == 6

    def test__integrate_equality_keep_original(
        self, mock_lc_astropy, mock_lc_updates_astropy
    ):
        updates = vstack([mock_lc_astropy[7:], mock_lc_updates_astropy])
        assert len(updates) == 6
        updates["source"] = "updates"
        mock_lc_astropy["source"] = "original"

        td = TargetData(lightcurve=mock_lc_astropy)
        result = td.integrate_lightcurve_updates_equality(
            updates, column="obsId", keep_updates=False
        )
        assert len(result) == 13
        assert all(result[:10]["source"] == "original")
        assert all(result[10:]["source"] == "updates")
        assert len(result[result["source"] == "original"]) == 10
        assert len(result[result["source"] == "updates"]) == 3


class Test__TargetInit:
    def test__target_init(self):
        t_create = Time(60000.0, format="mjd")
        t = Target("T101", ra=45.0, dec=60.0, t_ref=t_create)

        assert t.objectId == "T101"
        assert np.isclose(t.ra, 45.0)
        assert np.isclose(t.dec, 60.0)
        assert isinstance(t.coord, SkyCoord)
        assert isinstance(t.astroplan_target, FixedTarget)
        assert np.isclose(t.base_score, 1.0)
        assert t.compiled_lightcurve is None

        assert isinstance(t.target_data, dict)
        assert len(t.target_data) == 0
        assert isinstance(t.observatory_info, dict)
        assert set(t.observatory_info.keys()) == set(["no_observatory"])

        assert isinstance(t.models, dict)
        assert len(t.models) == 0
        assert isinstance(t.models_t_ref, dict)
        assert len(t.models) == 0

        assert isinstance(t.score_history, dict)
        assert set(t.score_history.keys()) == set(["no_observatory"])
        assert isinstance(t.score_history["no_observatory"], list)
        assert len(t.score_history["no_observatory"]) == 0
        assert isinstance(t.score_comments, dict)
        assert set(t.score_comments.keys()) == set(["no_observatory"])
        assert isinstance(t.reject_comments, dict)
        assert set(t.score_comments["no_observatory"]) == set()
        assert set(t.reject_comments.keys()) == set(["no_observatory"])
        assert set(t.reject_comments["no_observatory"]) == set()
        assert isinstance(t.rank_history, dict)
        assert set(t.rank_history.keys()) == set(["no_observatory"])
        assert isinstance(t.rank_history["no_observatory"], list)
        assert len(t.rank_history["no_observatory"]) == 0

        assert isinstance(t.creation_time, Time)
        assert np.isclose(t.creation_time.mjd, 60000.0)
        assert t.target_of_opportunity is False
        assert t.updated is False
        assert t.to_reject is False
        assert t.send_updates is False
        assert isinstance(t.update_messages, list)
        assert len(t.update_messages) == 0
        assert isinstance(t.sudo_messages, list)
        assert len(t.sudo_messages) == 0

    def test__init_with_target_data(self, mock_lc):
        td = TargetData(lightcurve=mock_lc)
        t = Target("T101", ra=45.0, dec=60.0, target_data={"data_source": td})

        assert set(t.target_data.keys()) == set(["data_source"])
        assert isinstance(t.target_data["data_source"], TargetData)
        assert len(t.target_data["data_source"].lightcurve) == 10

    def test__modify_base_score(self):
        t = Target("T101", ra=45.0, dec=60.0, base_score=1000.0)

        assert np.isclose(t.base_score, 1000.0)

    def test__set_target_of_opportunity(self):
        t = Target("T101", ra=45.0, dec=60.0, target_of_opportunity=True)

        assert t.target_of_opportunity is True


class Test__TargetConvenienceMethods:
    def test__update_coordinate(self, mock_target):
        mock_target.update_coordinates(60.0, 70.0)

        assert np.isclose(mock_target.ra, 60.0)
        assert np.isclose(mock_target.dec, 70.0)
        assert isinstance(mock_target.coord, SkyCoord)
        assert np.isclose(mock_target.coord.ra.deg, 60.0)
        assert np.isclose(mock_target.coord.dec.deg, 70.0)
        assert isinstance(mock_target.astroplan_target, FixedTarget)

    def test__get_target_data_create_missing(self, mock_target):
        assert isinstance(mock_target.target_data, dict)
        assert "my_source" not in mock_target.target_data
        result = mock_target.get_target_data("my_source")
        assert isinstance(result, TargetData)
        assert result.lightcurve is None

    def test__get_target_data_existing(self, mock_target):
        td = TargetData(parameters=dict(a=1, b=10, c=100))
        mock_target.target_data["test_data"] = td

        result = mock_target.get_target_data("test_data")
        assert result.parameters["b"] == 10

    def test__update_score_history(self, mock_target):
        t_ref = Time(60000.0, format="mjd")
        mock_target.update_score_history(1000.0, None, t_ref=t_ref)

        hist = mock_target.score_history
        assert len(hist["no_observatory"]) == 1
        assert isinstance(hist["no_observatory"][0], tuple)
        assert np.isclose(hist["no_observatory"][0][0], 1000.0)
        assert isinstance(hist["no_observatory"][0][1], Time)
        assert np.isclose(hist["no_observatory"][0][1].mjd, 60000.0)  # Correctly set.

    def test__update_score_history_new_observatory(self, mock_target, test_observer):
        t_ref = Time(60000.0, format="mjd")
        assert set(mock_target.score_history.keys()) == set(["no_observatory"])

        mock_target.update_score_history(1000.0, test_observer, t_ref=t_ref)

        assert set(mock_target.score_history.keys()) == set(["no_observatory", "ucph"])
        assert len(mock_target.score_history["ucph"]) == 1
        assert isinstance(mock_target.score_history["ucph"][0], tuple)
        assert np.isclose(mock_target.score_history["ucph"][0][0], 1000.0)
        assert isinstance(mock_target.score_history["ucph"][0][1], Time)
        assert np.isclose(
            mock_target.score_history["ucph"][0][1].mjd, 60000.0
        )  # Correctly set.

    def test__get_last_score_no_obs(self, mock_target, test_observer):
        t_ref_01 = Time(60000.0, format="mjd")
        mock_target.update_score_history(50.0, None, t_ref=t_ref_01)
        mock_target.update_score_history(60.0, test_observer, t_ref=t_ref_01)
        t_ref_02 = Time(60010.0, format="mjd")
        mock_target.update_score_history(25.0, None, t_ref=t_ref_02)
        mock_target.update_score_history(35.0, test_observer, t_ref=t_ref_02)

        result = mock_target.get_last_score()  # No args means obs=None
        assert np.isclose(result, 25.0)

        assert np.isclose(mock_target.get_last_score(None), 25.0)
        assert np.isclose(mock_target.get_last_score("no_observatory"), 25.0)

        result_with_time = mock_target.get_last_score(return_time=True)
        assert isinstance(result_with_time, tuple)
        assert np.isclose(result_with_time[0], 25.0)
        assert isinstance(result_with_time[1], Time)
        assert np.isclose(result_with_time[1].mjd, 60010.0)

    def test__get_last_score_observatory(self, mock_target, test_observer):
        t_ref_01 = Time(60000.0, format="mjd")
        mock_target.update_score_history(50.0, None, t_ref=t_ref_01)
        mock_target.update_score_history(60.0, test_observer, t_ref=t_ref_01)
        t_ref_02 = Time(60010.0, format="mjd")
        mock_target.update_score_history(25.0, None, t_ref=t_ref_02)
        mock_target.update_score_history(35.0, test_observer, t_ref=t_ref_02)

        result = mock_target.get_last_score(test_observer)
        assert np.isclose(result, 35.0)
        assert np.isclose(mock_target.get_last_score("ucph"), 35.0)

        result_with_time = mock_target.get_last_score(test_observer, return_time=True)
        assert isinstance(result_with_time, tuple)
        assert np.isclose(result_with_time[0], 35.0)
        assert isinstance(result_with_time[1], Time)
        assert np.isclose(result_with_time[1].mjd, 60010.0)

    def test__no_exception_on_missing_score(self, mock_target):
        assert len(mock_target.score_history["no_observatory"]) == 0
        result = mock_target.get_last_score()
        assert result is None

        result_with_time = mock_target.get_last_score(return_time=True)
        assert result_with_time[0] is None
        assert result_with_time[1] is None

    def test__get_last_score_warn_on_missing(self, mock_target, test_observer):
        assert "ucph" not in mock_target.score_history.keys()

        with pytest.warns(UnknownObservatoryWarning):
            result = mock_target.get_last_score(test_observer)
        assert result is None

        with pytest.warns(UnknownObservatoryWarning):
            result = mock_target.get_last_score("ucph")
        assert result is None


class Test__TargetStringMethods:
    def test__dunder_str(self, mock_target):
        s = str(mock_target)
        assert isinstance(s, str)

    def test__get_info_string(self, mock_target):
        t_ref = Time(60000.0, format="mjd")

        info_str = mock_target.get_info_string(t_ref=t_ref)

        assert "2023-02-25" in info_str
        assert "FINK" in info_str
        assert "Lasair" in info_str
        assert "coordinates" in info_str
        assert "equatorial" in info_str
        assert "galactic" in info_str

    def test__get_info_str_with_detections(self, mock_target, mock_lc):
        t_ref = Time(60000.0, format="mjd")
        td = TargetData(lightcurve=mock_lc)
        mock_target.target_data["ztf"] = td
        mock_target.compiled_lightcurve = basic_lc_compiler(mock_target, t_ref)
        print(mock_target.compiled_lightcurve)

        info_str = mock_target.get_info_string()
        assert "detections" in info_str
        assert "3 ztf-w" in info_str


class Test__WriteComments:
    def test__write_comments_no_comms(self, mock_target, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        exp_comments_path = tmp_path / "T101_comments.txt"

        mock_target.write_comments(tmp_path, t_ref=t_ref)

        assert exp_comments_path.exists()
        with open(exp_comments_path, "r") as f:
            lines = f.readlines()
        comments = " ".join(lines)

        assert "no_observatory" in comments
        assert "no score_comments provided" in comments

    def test__write_comments(self, mock_target, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        exp_comments_path = tmp_path / "T101_comments.txt"
        mock_target.score_comments["no_observatory"] = ["some comment"]
        mock_target.write_comments(tmp_path)

        assert exp_comments_path.exists()
        with open(exp_comments_path, "r") as f:
            lines = f.readlines()
        comments = " ".join(lines)

        assert "some comment" in comments


# class Test__TargetPlottingMethods:
#     def test__plot_lightcurve_no_data(self, mock_target, tmp_path):
#         t_ref = Time(60000.0, format="mjd")
#         fig_path = tmp_path / "target_func_no_data.pdf"
#         assert not fig_path.exists()

#         mock_target.plot_lightcurve(t_ref=t_ref, fig_path=fig_path)

#         assert fig_path.exists()
#         assert mock_target.lamock_lc_fig_path == fig_path

#     def test__plot_lightcurve_with_data(
#         self, mock_target, ext_mock_lc, mock_cutouts, tmp_path
#     ):
#         t_ref = Time(60000.0, format="mjd")
#         fig_path = tmp_path / "target_func_with_data.pdf"

#         mock_target.target_data["fink"] = TargetData(lightcurve=ext_mock_lc)
#         mock_target.plot_lightcurve(t_ref=t_ref, fig_path=fig_path)

#         assert fig_path.exists()
#         assert mock_target.lamock_lc_fig_path == fig_path

#     def test__plot_observing_chart(self, mock_target, test_observer, tmp_path):
#         t_ref = Time(60000.0, format="mjd")
#         fig_path = tmp_path / "target_func_obs_chart.pdf"
#         assert not fig_path.exists()

#         mock_target.plot_observing_chart(test_observer, fig_path=fig_path)

#         assert mock_target.latest_oc_fig_paths["ucph"] == fig_path
#         assert fig_path.exists()
