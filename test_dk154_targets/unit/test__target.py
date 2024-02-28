import pytest

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time

from astroplan import FixedTarget, Observer

from dk154_targets import Target, TargetData
from dk154_targets.exc import MissingDateError, UnknownObservatoryWarning
from dk154_targets.obs_info import ObservatoryInfo
from dk154_targets.target import (
    plot_default_lightcurve,
    DefaultLightcurvePlotter,
    plot_observing_chart,
    ObservingChartPlotter,
)
from dk154_targets.target import SettingLightcurveDirectlyWarning


@pytest.fixture
def test_lc_rows():
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
def test_lc(test_lc_rows):
    return pd.DataFrame(test_lc_rows, columns="mjd obsId mag magerr tag".split())


@pytest.fixture
def test_lc_astropy(test_lc_rows):
    return Table(rows=test_lc_rows, names="mjd obsId mag magerr tag".split())


@pytest.fixture
def test_lc_updates_rows():
    return [
        (60010.0, 11, 19.9, 0.05, "valid"),
        (60011.0, 12, 19.8, 0.05, "valid"),
        (60012.0, 13, 19.7, 0.2, "badqual"),
    ]


@pytest.fixture
def test_lc_updates(test_lc_updates_rows):
    return pd.DataFrame(
        test_lc_updates_rows, columns="mjd obsId mag magerr tag".split()
    )


@pytest.fixture
def test_lc_updates_astropy(test_lc_updates_rows):
    return Table(rows=test_lc_updates_rows, names="mjd obsId mag magerr tag".split())


@pytest.fixture
def test_target():
    return Target("T101", ra=45.0, dec=60.0)


@pytest.fixture
def test_observer():
    location = EarthLocation(lat=55.6802, lon=12.5724, height=0.0)
    return Observer(location, name="ucph")


# @pytest.fixture(scope="module")
# def plotting_path(tmp_path_factory)
#    pl_path = tmp_path_factory.mktemp("default_plotters")
#    return pl_path


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

        assert td.valid_tags == (
            "valid",
            "detection",
        )
        assert td.badqual_tags == (
            "badquality",
            "badqual",
            "dubious",
        )
        assert td.nondet_tags == ("upperlim", "nondet")

    def test__change_tags(self):
        td = TargetData(valid_tags=("valid",))
        assert td.valid_tags == ("valid",)

    def test__init_with_lc_no_tag(self, test_lc):
        test_lc.drop("tag", axis=1, inplace=True)
        assert not "tag" in test_lc.columns

        td = TargetData(lightcurve=test_lc)
        assert len(td.lightcurve) == 10
        assert len(td.detections) == 10
        assert td.badqual is None
        assert td.non_detections is None

    def test__init_with_lc_with_tag(self, test_lc):
        td = TargetData(lightcurve=test_lc)

        assert len(td.lightcurve) == 10
        assert len(td.detections) == 6
        assert len(td.badqual) == 0
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([5, 6, 7, 8, 9, 10])
        assert set(td.badqual["obsId"]) == set()  # Test the column name is there.
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__init_badqual_false(self, test_lc):
        td = TargetData(lightcurve=test_lc, include_badqual=False)

        assert len(td.lightcurve) == 10
        assert len(td.detections) == 3
        assert len(td.badqual) == 3
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([8, 9, 10])
        assert set(td.badqual["obsId"]) == set([5, 6, 7])
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__no_date_column_raises_error(self, test_lc):
        test_lc.drop("mjd", axis=1, inplace=True)

        td = TargetData()
        with pytest.raises(MissingDateError):
            td.add_lightcurve(test_lc)

    def test__set_lc_directly_raises_warning(self, test_lc):
        td = TargetData()
        with pytest.warns(SettingLightcurveDirectlyWarning):
            td.lightcurve = test_lc

        assert isinstance(td.lightcurve, pd.DataFrame)
        assert len(td.lightcurve) == 10

    def test__empty_cutouts(self):
        td = TargetData()
        result = td.empty_cutouts()
        assert isinstance(result, dict)
        assert len(result) == 0

    def test__integrate_equality_no_lightcurve(self, test_lc_updates):
        td = TargetData()
        result = td.integrate_equality(test_lc_updates, column="obsId")
        assert len(result) == 3

    def test__integrate_equality(self, test_lc, test_lc_updates):
        td = TargetData(lightcurve=test_lc)

        result = td.integrate_equality(test_lc_updates, column="obsId")
        assert len(result) == 13
        assert set(result["obsId"]) == set(range(1, 14))

    def test__integrate_equality_keep_updated(self, test_lc, test_lc_updates):
        updates = pd.concat([test_lc[7:], test_lc_updates], ignore_index=True)
        assert len(updates) == 6
        updates["source"] = "updates"
        test_lc["source"] = "original"

        td = TargetData(lightcurve=test_lc)
        result = td.integrate_equality(updates, column="obsId")
        assert len(result) == 13
        assert all(result.iloc[:7].source == "original")
        assert all(result.iloc[7:].source == "updates")
        assert len(result[result["source"] == "original"]) == 7
        assert len(result[result["source"] == "updates"]) == 6

    def test__integrate_equality_keep_original(self, test_lc, test_lc_updates):
        updates = pd.concat([test_lc[7:], test_lc_updates], ignore_index=True)
        assert len(updates) == 6
        updates["source"] = "updates"
        test_lc["source"] = "original"

        td = TargetData(lightcurve=test_lc)
        result = td.integrate_equality(updates, column="obsId", keep_updates=False)
        assert len(result) == 13
        assert all(result.iloc[:10].source == "original")
        assert all(result.iloc[10:].source == "updates")
        assert len(result[result["source"] == "original"]) == 10
        assert len(result[result["source"] == "updates"]) == 3

    def test__integrate_updates(self, test_lc, test_lc_updates):
        td = TargetData(lightcurve=test_lc)
        td.integrate_lightcurve_updates(test_lc_updates, column="obsId")

        assert len(td.lightcurve) == 13
        assert len(td.detections) == 9
        assert len(td.badqual) == 0
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([5, 6, 7, 8, 9, 10, 11, 12, 13])
        assert set(td.badqual["obsId"]) == set()  # Test the column name is there.
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__integrate_updates_badqual_false(self, test_lc, test_lc_updates):
        td = TargetData(lightcurve=test_lc)
        td.integrate_lightcurve_updates(
            test_lc_updates, column="obsId", include_badqual=False
        )

        assert len(td.lightcurve) == 13
        assert len(td.detections) == 5
        assert len(td.badqual) == 4
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([8, 9, 10, 11, 12])
        assert set(td.badqual["obsId"]) == set([5, 6, 7, 13])
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])


class Test__TargetDataAstropyTableCompatibility:
    def test__init_with_table_no_tag(self, test_lc_astropy):
        test_lc_astropy.remove_column("tag")
        assert "tag" not in test_lc_astropy.columns

        td = TargetData(lightcurve=test_lc_astropy)

        assert len(td.lightcurve) == 10
        # assert td.detections is None
        assert len(td.detections) == 10
        assert td.non_detections is None

    def test__init_with_table_with_tag(self, test_lc_astropy):
        td = TargetData(lightcurve=test_lc_astropy)

        assert len(td.lightcurve) == 10
        assert len(td.detections) == 6
        assert len(td.badqual) == 0
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([5, 6, 7, 8, 9, 10])
        assert set(td.badqual["obsId"]) == set()
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__init_with_table_badqual_false(self, test_lc_astropy):
        td = TargetData(test_lc_astropy, include_badqual=False)

        assert len(td.lightcurve) == 10
        assert len(td.detections) == 3
        assert len(td.badqual) == 3
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([8, 9, 10])
        assert set(td.badqual["obsId"]) == set([5, 6, 7])
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__equality_no_lc_astropy(self, test_lc_updates):
        td = TargetData()
        result = td.integrate_equality(test_lc_updates, column="obsId")
        assert len(result) == 3

    def test__integrate_equality(self, test_lc, test_lc_updates):
        td = TargetData(lightcurve=test_lc)

        result = td.integrate_equality(test_lc_updates, column="obsId")
        assert len(result) == 13
        assert set(result["obsId"]) == set(range(1, 14))

    def test__equality_keep_updated_astropy(
        self, test_lc_astropy, test_lc_updates_astropy
    ):
        updates = vstack([test_lc_astropy[7:], test_lc_updates_astropy])
        assert len(updates) == 6
        updates["source"] = "updates"
        test_lc_astropy["source"] = "original"

        td = TargetData(lightcurve=test_lc_astropy)
        result = td.integrate_equality(updates, column="obsId")
        assert len(result) == 13
        assert all(result[:7]["source"] == "original")
        assert all(result[7:]["source"] == "updates")
        assert len(result[result["source"] == "original"]) == 7
        assert len(result[result["source"] == "updates"]) == 6

    def test__integrate_equality_keep_original(
        self, test_lc_astropy, test_lc_updates_astropy
    ):
        updates = vstack([test_lc_astropy[7:], test_lc_updates_astropy])
        assert len(updates) == 6
        updates["source"] = "updates"
        test_lc_astropy["source"] = "original"

        td = TargetData(lightcurve=test_lc_astropy)
        result = td.integrate_equality(updates, column="obsId", keep_updates=False)
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
        assert np.isclose(t.base_score, 100.0)
        assert t.compiled_lightcurve is None

        assert isinstance(t.target_data, dict)
        assert len(t.target_data) == 0
        assert isinstance(t.observatory_info, dict)
        assert set(t.observatory_info.keys()) == set(["no_observatory"])

        assert isinstance(t.models, dict)
        assert len(t.models) == 0
        assert isinstance(t.models_tref, dict)
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

        assert t.latest_lc_fig is None
        assert t.latest_lc_fig_path is None
        assert isinstance(t.latest_oc_figs, dict)
        assert len(t.latest_oc_figs) == 0
        assert isinstance(t.latest_oc_fig_paths, dict)
        assert len(t.latest_oc_fig_paths) == 0

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

    def test__init_with_target_data(self, test_lc):
        td = TargetData(lightcurve=test_lc)
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
    def test__update_coordinate(self, test_target):
        test_target.update_coordinates(60.0, 70.0)

        assert np.isclose(test_target.ra, 60.0)
        assert np.isclose(test_target.dec, 70.0)
        assert isinstance(test_target.coord, SkyCoord)
        assert np.isclose(test_target.coord.ra.deg, 60.0)
        assert np.isclose(test_target.coord.dec.deg, 70.0)
        assert isinstance(test_target.astroplan_target, FixedTarget)

    def test__get_target_data_create_missing(self, test_target):
        assert isinstance(test_target.target_data, dict)
        assert "my_source" not in test_target.target_data
        result = test_target.get_target_data("my_source")
        assert isinstance(result, TargetData)
        assert result.lightcurve is None

    def test__get_target_data_existing(self, test_target):
        td = TargetData(parameters=dict(a=1, b=10, c=100))
        test_target.target_data["test_data"] = td

        result = test_target.get_target_data("test_data")
        assert result.parameters["b"] == 10

    def test__update_score_history(self, test_target):
        t_ref = Time(60000.0, format="mjd")
        test_target.update_score_history(1000.0, None, t_ref=t_ref)

        hist = test_target.score_history
        assert len(hist["no_observatory"]) == 1
        assert isinstance(hist["no_observatory"][0], tuple)
        assert np.isclose(hist["no_observatory"][0][0], 1000.0)
        assert isinstance(hist["no_observatory"][0][1], Time)
        assert np.isclose(hist["no_observatory"][0][1].mjd, 60000.0)  # Correctly set.

    def test__update_score_history_new_observatory(self, test_target, test_observer):
        t_ref = Time(60000.0, format="mjd")
        assert set(test_target.score_history.keys()) == set(["no_observatory"])

        test_target.update_score_history(1000.0, test_observer, t_ref=t_ref)

        assert set(test_target.score_history.keys()) == set(["no_observatory", "ucph"])
        assert len(test_target.score_history["ucph"]) == 1
        assert isinstance(test_target.score_history["ucph"][0], tuple)
        assert np.isclose(test_target.score_history["ucph"][0][0], 1000.0)
        assert isinstance(test_target.score_history["ucph"][0][1], Time)
        assert np.isclose(
            test_target.score_history["ucph"][0][1].mjd, 60000.0
        )  # Correctly set.

    def test__get_last_score_no_obs(self, test_target, test_observer):
        t_ref_01 = Time(60000.0, format="mjd")
        test_target.update_score_history(50.0, None, t_ref=t_ref_01)
        test_target.update_score_history(60.0, test_observer, t_ref=t_ref_01)
        t_ref_02 = Time(60010.0, format="mjd")
        test_target.update_score_history(25.0, None, t_ref=t_ref_02)
        test_target.update_score_history(35.0, test_observer, t_ref=t_ref_02)

        result = test_target.get_last_score()  # No args means obs=None
        assert np.isclose(result, 25.0)

        assert np.isclose(test_target.get_last_score(None), 25.0)
        assert np.isclose(test_target.get_last_score("no_observatory"), 25.0)

        result_with_time = test_target.get_last_score(return_time=True)
        assert isinstance(result_with_time, tuple)
        assert np.isclose(result_with_time[0], 25.0)
        assert isinstance(result_with_time[1], Time)
        assert np.isclose(result_with_time[1].mjd, 60010.0)

    def test__get_last_score_observatory(self, test_target, test_observer):
        t_ref_01 = Time(60000.0, format="mjd")
        test_target.update_score_history(50.0, None, t_ref=t_ref_01)
        test_target.update_score_history(60.0, test_observer, t_ref=t_ref_01)
        t_ref_02 = Time(60010.0, format="mjd")
        test_target.update_score_history(25.0, None, t_ref=t_ref_02)
        test_target.update_score_history(35.0, test_observer, t_ref=t_ref_02)

        result = test_target.get_last_score(test_observer)
        assert np.isclose(result, 35.0)
        assert np.isclose(test_target.get_last_score("ucph"), 35.0)

        result_with_time = test_target.get_last_score(test_observer, return_time=True)
        assert isinstance(result_with_time, tuple)
        assert np.isclose(result_with_time[0], 35.0)
        assert isinstance(result_with_time[1], Time)
        assert np.isclose(result_with_time[1].mjd, 60010.0)

    def test__no_exception_on_missing_score(self, test_target):
        assert len(test_target.score_history["no_observatory"]) == 0
        result = test_target.get_last_score()
        assert result is None

        result_with_time = test_target.get_last_score(return_time=True)
        assert result_with_time[0] is None
        assert result_with_time[1] is None

    def test__get_last_score_warn_on_missing(self, test_target, test_observer):
        assert "ucph" not in test_target.score_history.keys()

        with pytest.warns(UnknownObservatoryWarning):
            result = test_target.get_last_score(test_observer)
        assert result is None

        with pytest.warns(UnknownObservatoryWarning):
            result = test_target.get_last_score("ucph")
        assert result is None


def basic_scoring_function(target, obs, t_ref):
    if obs is None:
        return target.base_score
    return 3 * target.base_score


def basic_scoring_function_with_comments(target, obs, t_ref):
    score_comments = ["some comment"]
    reject_comments = []
    if obs is None:
        return target.base_score, score_comments, reject_comments
    return 3 * target.base_score, score_comments, reject_comments


class ScoringClass:
    __name__ = "score_class"

    def __init__(self, default_score=75.0):
        self.default_score = default_score

    def __call__(self, target, obs, t_ref):
        if obs is None:
            return self.default_score
        return 3 * self.default_score


class ScoringClassWithComments:
    __name__ = "score_class_with_comments"

    def __init__(self, default_score=75.0):
        self.default_score = default_score

    def __call__(self, target, obs, t_ref):
        score_comments = ["this is from a class"]
        reject_comments = []
        if obs is None:
            return self.default_score, score_comments, reject_comments
        return 3 * self.default_score, score_comments, reject_comments


def scoring_function_bad_return(target, obs, t_ref):
    return 50.0, ["a single list"]


def scoring_function_bad_signature(target, obs):
    return 50.0  # No t_ref in signature.


class Test__EvaluateTarget:
    def test__evaluate_target_basic(self, test_target):
        t_ref = Time(60000.0, format="mjd")

        result = test_target.evaluate_target(basic_scoring_function, None, t_ref=t_ref)

        assert np.isclose(result, 100.0)
        assert len(test_target.score_history["no_observatory"]) == 1
        assert np.isclose(test_target.score_history["no_observatory"][0][0], 100.0)
        assert test_target.get_last_score() == 100.0
        assert set(test_target.score_comments["no_observatory"]) == set()
        assert set(test_target.reject_comments["no_observatory"]) == set()

    def test__evaluate_target_with_comments(self, test_target):
        t_ref = Time(60000.0, format="jd")

        result = test_target.evaluate_target(
            basic_scoring_function_with_comments, None, t_ref
        )

        assert np.isclose(result, 100.0)
        assert len(test_target.score_history["no_observatory"]) == 1
        assert np.isclose(test_target.score_history["no_observatory"][0][0], 100.0)
        assert test_target.get_last_score() == 100.0
        assert isinstance(test_target.score_comments["no_observatory"], list)
        assert set(test_target.score_comments["no_observatory"]) == set(
            ["some comment"]
        )
        assert isinstance(test_target.reject_comments["no_observatory"], list)
        assert set(test_target.reject_comments["no_observatory"]) == set()

    def test__evaluate_target_with_obs(self, test_target, test_observer):
        t_ref = Time(60000.0, format="mjd")

        result = test_target.evaluate_target(
            basic_scoring_function, test_observer, t_ref=t_ref
        )

        assert np.isclose(result, 300.0)
        assert set(test_target.score_history.keys()) == set(["no_observatory", "ucph"])
        assert len(test_target.score_history["no_observatory"]) == 0
        assert len(test_target.score_history["ucph"]) == 1
        assert np.isclose(test_target.score_history["ucph"][0][0], 300.0)
        assert np.isclose(test_target.score_history["ucph"][0][1].mjd, 60000.0)

        assert set(test_target.score_comments["ucph"]) == set()
        assert set(test_target.reject_comments["ucph"]) == set()
        assert set(test_target.score_comments["no_observatory"]) == set()
        assert set(test_target.reject_comments["no_observatory"]) == set()

    def test__evaluate_target_with_obs_comments(self, test_target, test_observer):
        t_ref = Time(60000.0, format="mjd")

        result = test_target.evaluate_target(
            basic_scoring_function_with_comments, test_observer, t_ref=t_ref
        )

        assert np.isclose(result, 300.0)
        assert set(test_target.score_history.keys()) == set(["no_observatory", "ucph"])
        assert len(test_target.score_history["ucph"]) == 1
        assert test_target.get_last_score(test_observer) == 300.0
        assert np.isclose(test_target.score_history["ucph"][0][0], 300.0)
        assert np.isclose(test_target.score_history["ucph"][0][1].mjd, 60000.0)
        assert set(test_target.score_comments["ucph"]) == set(["some comment"])
        assert set(test_target.reject_comments["ucph"]) == set()

        assert len(test_target.score_history["no_observatory"]) == 0
        assert set(test_target.score_comments["no_observatory"]) == set()
        assert set(test_target.reject_comments["no_observatory"]) == set()

    def test__evaluate_target_class(self, test_target):
        t_ref = Time(60000.0, format="mjd")
        scoring_func = ScoringClass()

        result = test_target.evaluate_target(scoring_func, None, t_ref=t_ref)

        assert np.isclose(result, 75.0)
        assert set(test_target.score_history.keys()) == set(["no_observatory"])
        assert len(test_target.score_history["no_observatory"]) == 1
        assert np.isclose(test_target.get_last_score(), 75.0)
        assert set(test_target.score_comments["no_observatory"]) == set()
        assert set(test_target.reject_comments["no_observatory"]) == set()

    def test__evaluate_target_class_comments(self, test_target):
        t_ref = Time(60000.0, format="mjd")
        scoring_func = ScoringClassWithComments()

        result = test_target.evaluate_target(scoring_func, None, t_ref=t_ref)

        assert np.isclose(result, 75.0)
        assert len(test_target.score_history["no_observatory"]) == 1
        assert np.isclose(test_target.score_history["no_observatory"][0][0], 75.0)
        assert test_target.get_last_score() == 75.0
        assert set(test_target.score_comments["no_observatory"]) == set(
            ["this is from a class"]
        )
        assert set(test_target.reject_comments["no_observatory"]) == set()

    def test__evaluate_target_class_obs_comments(self, test_target, test_observer):
        t_ref = Time(60000.0, format="mjd")
        scoring_func = ScoringClassWithComments()

        result = test_target.evaluate_target(scoring_func, test_observer, t_ref=t_ref)

        assert np.isclose(result, 225.0)
        assert len(test_target.score_history["ucph"]) == 1
        assert np.isclose(test_target.score_history["ucph"][0][0], 225.0)
        assert test_target.get_last_score(test_observer) == 225.0
        assert test_target.get_last_score("ucph") == 225.0
        assert set(test_target.score_comments["ucph"]) == set(["this is from a class"])
        assert set(test_target.reject_comments["ucph"]) == set()

        assert len(test_target.score_history["no_observatory"]) == 0
        assert set(test_target.score_comments["no_observatory"]) == set()
        assert set(test_target.reject_comments["no_observatory"]) == set()

    def test__exception_with_bad_return(self, test_target):
        t_ref = Time(60000.0, format="mjd")

        with pytest.raises(ValueError):
            result = test_target.evaluate_target(
                scoring_function_bad_return, None, t_ref=t_ref
            )

    def test__warning_with_class_not_instance(self, test_target):
        t_ref = Time(60000.0, format="mjd")

        scoring_func = ScoringClass  # Oops, we for got to make this an instance.
        with pytest.warns(UserWarning):
            result = test_target.evaluate_target(scoring_func, None, t_ref=t_ref)


def basic_lc_compiler(target):
    detections = target.target_data["test_data"].detections.copy()
    badqual = target.target_data["test_data"].badqual.copy()
    limits = target.target_data["test_data"].non_detections.copy()

    detections.loc[:, "flux"] = 3631.0 * 10 ** (-detections["mag"] * 0.4)
    detections.loc[:, "tag"] = "valid"
    badqual.loc[:, "tag"] = "is_badqual"  # Can definitely check these have changed.
    limits.loc[:, "tag"] = "a_limit"
    lc = pd.concat([limits, badqual, detections], ignore_index=True)
    lc.loc[:, "band"] = "ztf-w"
    lc.loc[:, "jd"] = Time(lc["mjd"], format="mjd").jd
    return lc


class LCCompilerClass:
    def __init__(self):
        pass

    def __call__(self, target):
        lc = basic_lc_compiler(target)
        lc.loc[:, "band"] = "ztf-q"
        return lc


class Test__CompileLightcurve:
    def test__basic_compile(self, test_target, test_lc):
        td = TargetData(lightcurve=test_lc, include_badqual=False)
        test_target.target_data["test_data"] = td

        result = test_target.build_compiled_lightcurve(basic_lc_compiler)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert len(result[result["tag"] == "a_limit"]) == 4
        assert len(result[result["tag"] == "is_badqual"]) == 3
        assert len(result[result["tag"] == "valid"]) == 3

        assert "band" in result.columns
        assert "flux" in result.columns

        assert isinstance(test_target.compiled_lightcurve, pd.DataFrame)
        assert len(test_target.compiled_lightcurve) == 10
        assert all(test_target.compiled_lightcurve == "ztf-w")

    def test__compile_with_class(self, test_target, test_lc):
        td = TargetData(lightcurve=test_lc, include_badqual=False)
        test_target.target_data["test_data"] = td
        compile_func = LCCompilerClass()

        result = test_target.build_compiled_lightcurve(compile_func)
        assert all(result["band"] == "ztf-q")

    def test__lazy_compile_lightcurve(self, test_target, test_lc):
        td = TargetData(lightcurve=test_lc, include_badqual=False)
        test_target.target_data["test_data"] = td

        # compiled_lightcurve should be update when it is None, even if lazy
        assert test_target.updated is False
        assert test_target.compiled_lightcurve is None
        test_target.build_compiled_lightcurve(basic_lc_compiler, lazy=True)
        assert isinstance(test_target.compiled_lightcurve, pd.DataFrame)  # Is updated

        # ... but once it already exists, skip if not updated.
        test_target.compiled_lightcurve.drop("band", axis=1, inplace=True)
        assert test_target.updated is False
        test_target.build_compiled_lightcurve(basic_lc_compiler, lazy=True)
        assert "band" not in test_target.compiled_lightcurve.columns

        # ...and updates if lazy but target is updated.
        test_target.updated = True
        test_target.build_compiled_lightcurve(basic_lc_compiler, lazy=True)
        assert "band" in test_target.compiled_lightcurve.columns


class SimpleModel:
    def __init__(self, target, chi2=1.0):
        self.flux = 1e-6
        self.chi2 = chi2


def simple_model(target):
    return SimpleModel(target)


class BuildSimpleModel:
    __name__ = "simple_model_class"

    def __init__(self, chi2=1.0):
        self.chi2 = chi2

    def __call__(self, target):
        return SimpleModel(target, chi2=self.chi2)


class Test__BuildModel:
    def test__build_simple_model(self, test_target):
        t_ref = Time(60000.0, format="mjd")

        result = test_target.build_model(simple_model, t_ref=t_ref)
        assert isinstance(result, SimpleModel)

        assert set(test_target.models.keys()) == set(["simple_model"])
        assert isinstance(test_target.models["simple_model"], SimpleModel)
        assert set(test_target.models_tref.keys()) == set(["simple_model"])
        assert isinstance(test_target.models_tref["simple_model"], Time)
        assert np.isclose(test_target.models_tref["simple_model"].mjd, 60000.0)

    def test__tref_updated(self, test_target):
        t_ref = Time(60000.0, format="mjd")
        t_future = Time(60050.0, format="mjd")

        result = test_target.build_model(simple_model, t_ref=t_ref)
        assert np.isclose(test_target.models_tref["simple_model"].mjd, 60000.0)

        result2 = test_target.build_model(simple_model, t_ref=t_future)
        assert np.isclose(test_target.models_tref["simple_model"].mjd, 60050.0)


class Test__TargetStringMethods:
    def test__dunder_str(self, test_target):
        s = str(test_target)
        assert isinstance(s, str)

    def test__get_info_string(self, test_target):
        t_ref = Time(60000.0, format="mjd")

        info_str = test_target.get_info_string(t_ref=t_ref)

        assert "2023-02-25" in info_str
        assert "FINK" in info_str
        assert "Lasair" in info_str
        assert "coordinates" in info_str
        assert "equatorial" in info_str
        assert "galactic" in info_str

    def test__get_info_str(self, test_target, test_lc):
        t_ref = Time(60000.0, format="mjd")
        td = TargetData(lightcurve=test_lc)
        test_target.target_data["test_data"] = td
        test_target.build_compiled_lightcurve(basic_lc_compiler)

        info_str = test_target.get_info_string()
        assert "detections" in info_str
        assert "6 ztf-w" in info_str


class Test__WriteComments:
    def test__write_comments_no_comms(self, test_target, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        exp_comments_path = tmp_path / "T101_comments.txt"

        test_target.write_comments(tmp_path, t_ref=t_ref)

        assert exp_comments_path.exists()
        with open(exp_comments_path, "r") as f:
            lines = f.readlines()
        comments = " ".join(lines)

        assert "no_observatory" in comments
        assert "no score_comments provided" in comments

    def test__write_comments(self, test_target, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        exp_comments_path = tmp_path / "T101_comments.txt"
        test_target.evaluate_target(basic_scoring_function_with_comments, None)

        test_target.write_comments(tmp_path)

        assert exp_comments_path.exists()
        with open(exp_comments_path, "r") as f:
            lines = f.readlines()
        comments = " ".join(lines)

        assert "some comment" in comments


@pytest.fixture
def test_lc_g_rows():
    return [
        (60000.0, 1, 20.5, 0.1, 20.2, "upperlim"),
        (60001.0, 2, 20.4, 0.1, 20.3, "upperlim"),
        (60002.0, 3, 20.3, 0.1, 20.2, "upperlim"),
        (60003.0, 4, 20.2, 0.1, 20.1, "upperlim"),
        (60004.0, 5, 20.1, 0.2, 20.0, "badquality"),
        (60005.0, 6, 20.1, 0.2, 20.0, "badquality"),
        (60006.0, 7, 20.1, 0.2, 20.0, "badquality"),
        (60007.0, 8, 20.0, 0.1, 20.0, "valid"),
        (60008.0, 9, 19.7, 0.1, 20.0, "valid"),
        (60009.0, 10, 19.5, 0.1, 20.0, "valid"),
    ]


@pytest.fixture
def test_lc_r_rows():
    return [
        (60000.5, 1, 19.5, 0.1, 19.2, "upperlim"),
        (60001.5, 2, 19.4, 0.1, 19.3, "upperlim"),
        (60002.5, 3, 19.3, 0.1, 19.2, "upperlim"),
        (60003.5, 4, 19.2, 0.1, 19.1, "upperlim"),
        (60004.5, 5, 19.1, 0.2, 19.0, "badquality"),
        (60005.5, 6, 19.1, 0.2, 19.0, "badquality"),
        (60006.5, 7, 19.1, 0.2, 19.0, "badquality"),
        (60007.5, 8, 19.0, 0.1, 19.0, "valid"),
        (60008.5, 9, 18.7, 0.1, 19.0, "valid"),
        (60009.5, 10, 19.4, 0.1, 19.0, "valid"),
    ]


@pytest.fixture
def ext_test_lc(test_lc_g_rows, test_lc_r_rows):
    columns = "mjd obsId mag magerr diffmaglim tag".split()
    g_df = pd.DataFrame(test_lc_g_rows, columns=columns)
    g_df.loc[:, "band"] = "ztfg"
    r_df = pd.DataFrame(test_lc_r_rows, columns=columns)
    r_df.loc[:, "band"] = "ztfr"

    lc = pd.concat([g_df, r_df], ignore_index=True)
    lc.loc[:, "obsId"] = np.arange(1, len(lc) + 1)
    lc.sort_values("mjd", inplace=True)
    return lc


@pytest.fixture
def mock_cutouts():
    # These are more elaborate than necessary...
    grid = np.linspace(-4, 4, 100)
    xx, yy = np.meshgrid(grid, grid)
    template = xx**2 + yy**2
    source = np.maximum(20 - (5 * xx**2 + 5 * yy**2), 0.0)
    return {
        "science": template + source,
        "difference": source,
        "template": template,
    }


def basic_ztf_lc_compiler(target):
    lc = target.target_data["fink"].lightcurve.copy()
    lc.loc[:, "jd"] = Time(lc["mjd"], format="mjd").jd
    return lc


class Test__TargetPlottingMethods:
    def test__plot_lightcurve_no_data(self, test_target, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        fig_path = tmp_path / "target_func_no_data.pdf"
        assert not fig_path.exists()

        test_target.plot_lightcurve(t_ref=t_ref, fig_path=fig_path)

        assert fig_path.exists()
        assert test_target.latest_lc_fig_path == fig_path

    def test__plot_lightcurve_with_data(
        self, test_target, ext_test_lc, mock_cutouts, tmp_path
    ):
        t_ref = Time(60000.0, format="mjd")
        fig_path = tmp_path / "target_func_with_data.pdf"

        test_target.target_data["fink"] = TargetData(lightcurve=ext_test_lc)
        test_target.plot_lightcurve(t_ref=t_ref, fig_path=fig_path)

        assert fig_path.exists()
        assert test_target.latest_lc_fig_path == fig_path

    def test__plot_observing_chart(self, test_target, test_observer, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        fig_path = tmp_path / "target_func_obs_chart.pdf"
        assert not fig_path.exists()

        test_target.plot_observing_chart(test_observer, fig_path=fig_path)

        assert test_target.latest_oc_fig_paths["ucph"] == fig_path
        assert fig_path.exists()


class Test__DefaultLightcurvePlotter:
    def test__init(self):
        plotter = DefaultLightcurvePlotter()

        assert hasattr(plotter, "lc_gs")
        assert hasattr(plotter, "zscaler")

        assert hasattr(plotter, "figsize")

        assert hasattr(plotter, "ztf_colors")
        assert hasattr(plotter, "atlas_colors")
        assert hasattr(plotter, "det_kwargs")
        assert hasattr(plotter, "ulim_kwargs")
        assert hasattr(plotter, "badqual_kwargs")

        assert hasattr(plotter, "tag_col")
        assert hasattr(plotter, "valid_tag")
        assert hasattr(plotter, "badqual_tag")

        assert hasattr(plotter, "band_col")

        assert plotter.photometry_plotted is False
        assert plotter.cutouts_added is False
        assert plotter.axes_formatted is False
        assert plotter.comments_added is False

    def test__fig_init(self):
        plotter = DefaultLightcurvePlotter()

        plotter.init_fig()
        assert isinstance(plotter.fig, plt.Figure)
        assert isinstance(plotter.ax, plt.Axes)

    def tset__no_exception_plot_photometry_blank(self, test_target, tmp_path):
        t_ref = Time(60012.0, format="mjd")
        fig_path = tmp_path / "lc_no_photom.pdf"
        assert not fig_path.exists()

        plotter = DefaultLightcurvePlotter()
        plotter.init_fig()

        plotter.plot_photometry(test_target, t_ref=t_ref)

        plotter.fig.save_fig(fig_path)
        assert fig_path.exists()

    def test__plot_photometry(self, test_target, ext_test_lc, tmp_path):
        t_ref = Time(60015.0, format="mjd")
        test_target.target_data["fink"] = TargetData(lightcurve=ext_test_lc)
        test_target.build_compiled_lightcurve(basic_ztf_lc_compiler)
        fig_path = tmp_path / "lc_with_photom.pdf"
        assert not fig_path.exists()

        plotter = DefaultLightcurvePlotter()
        plotter.init_fig()
        plotter.plot_photometry(test_target, t_ref=t_ref)

        assert plotter.photometry_plotted is True
        assert plotter.cutouts_added is False
        assert plotter.axes_formatted is False
        assert plotter.comments_added is False

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()

    def test__add_postage_stamps(self, test_target, mock_cutouts, tmp_path):
        test_target.target_data["fink"] = TargetData(cutouts=mock_cutouts)
        fig_path = tmp_path / "lc_add_stamps.pdf"
        assert not fig_path.exists()

        plotter = DefaultLightcurvePlotter()
        plotter.init_fig()
        plotter.add_cutouts(test_target)

        assert plotter.photometry_plotted is False
        assert plotter.cutouts_added is True
        assert plotter.axes_formatted is False
        assert plotter.comments_added is False

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()

    def test__plot_class_method(
        self, test_target, ext_test_lc, mock_cutouts, test_observer, tmp_path
    ):
        t_ref = Time(60015.0, format="mjd")
        td = TargetData(lightcurve=ext_test_lc, cutouts=mock_cutouts)
        test_target.target_data["fink"] = td
        test_target.target_data["tns"] = TargetData(parameters={"Redshift": 0.1})
        test_target.build_compiled_lightcurve(basic_ztf_lc_compiler)
        test_target.score_comments["no_observatory"] = [
            "comm a",
            "comm b",
            "comm c",
            "comm d",
            "comm e",
            "comm f",
        ]
        plotter = DefaultLightcurvePlotter.plot(test_target, t_ref=t_ref)
        fig_path = tmp_path / "lc_plotter_class_method.pdf"
        assert not fig_path.exists()

        assert plotter.photometry_plotted
        assert plotter.cutouts_added
        assert plotter.axes_formatted
        assert plotter.comments_added

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()


class Test__ObservingChartPlotter:
    def test__init(self, test_observer):
        t_ref = Time(60000.0, format="mjd")
        plotter = ObservingChartPlotter(test_observer, t_ref=t_ref)

        assert np.isclose(plotter.t_grid[0].mjd, 60000.0)
        assert np.isclose(plotter.t_grid[-1].mjd, 60001.0)

        print(type(plotter.obs_info.moon_altaz))
        assert isinstance(plotter.obs_info, ObservatoryInfo)
        assert isinstance(plotter.obs_info.moon_altaz, SkyCoord)  # apparenlty not AltAz
        assert isinstance(plotter.obs_info.sun_altaz, SkyCoord)
        assert hasattr(plotter.obs_info.moon_altaz, "alt")
        assert hasattr(plotter.obs_info.sun_altaz, "alt")
        assert plotter.obs_info.target_altaz is None

        assert plotter.axes_initialized is False
        assert plotter.altitude_plotted is False
        assert plotter.sky_plotted is False
        assert plotter.moon_plotted is False
        assert plotter.sun_plotted is False
        assert plotter.axes_formatted is False

    def test__plot_method(self, test_target, test_observer, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        fig_path = tmp_path / "oc_plot_class_method.pdf"
        assert not fig_path.exists()

        plotter = ObservingChartPlotter.plot(test_observer, test_target, t_ref=t_ref)

        assert np.allclose(plotter.fig.get_size_inches(), [6.0, 8.0])
        assert isinstance(plotter.alt_ax, plt.Axes)
        assert isinstance(plotter.sky_ax, plt.Axes)

        assert plotter.axes_initialized
        assert plotter.altitude_plotted
        assert plotter.sky_plotted
        assert plotter.moon_plotted
        assert plotter.sun_plotted
        assert plotter.axes_formatted

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()

    def test__plot_method_no_alt(self, test_observer, test_target, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        fig_path = tmp_path / "oc_plot_no_alt.pdf"
        assert not fig_path.exists()

        plotter = ObservingChartPlotter.plot(
            test_observer, test_target, t_ref=t_ref, alt_ax=False
        )

        assert np.allclose(plotter.fig.get_size_inches(), [6.0, 4.0])
        assert plotter.alt_ax is None
        assert isinstance(plotter.sky_ax, plt.Axes)

        assert plotter.axes_initialized
        assert plotter.altitude_plotted is False
        assert plotter.sky_plotted
        assert plotter.moon_plotted
        assert plotter.sun_plotted
        assert plotter.axes_formatted

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()

    def test__plot_method_no_sky(self, test_observer, test_target, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        fig_path = tmp_path / "oc_plot_no_sky.pdf"
        assert not fig_path.exists()

        plotter = ObservingChartPlotter.plot(
            test_observer, test_target, t_ref=t_ref, sky_ax=False
        )

        assert np.allclose(plotter.fig.get_size_inches(), [6.0, 4.0])
        assert isinstance(plotter.alt_ax, plt.Axes)
        assert plotter.sky_ax is None

        assert plotter.axes_initialized
        assert plotter.altitude_plotted
        assert plotter.sky_plotted is False
        assert plotter.moon_plotted
        assert plotter.sun_plotted
        assert plotter.axes_formatted

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()

    def test__plot_method_no_sun(self, test_observer, test_target, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        fig_path = tmp_path / "oc_plot_no_sun.pdf"
        assert not fig_path.exists()

        plotter = ObservingChartPlotter.plot(
            test_observer, test_target, t_ref=t_ref, sun=False
        )

        assert np.allclose(plotter.fig.get_size_inches(), [6.0, 8.0])
        assert isinstance(plotter.alt_ax, plt.Axes)
        assert isinstance(plotter.sky_ax, plt.Axes)

        assert plotter.axes_initialized
        assert plotter.altitude_plotted
        assert plotter.sky_plotted
        assert plotter.moon_plotted
        assert plotter.sun_plotted is False
        assert plotter.axes_formatted

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()

    def test__plot_method_no_moon(self, test_observer, test_target, tmp_path):
        t_ref = Time(60000.0, format="mjd")
        fig_path = tmp_path / "oc_plot_no_moon.pdf"
        assert not fig_path.exists()

        plotter = ObservingChartPlotter.plot(
            test_observer, test_target, t_ref=t_ref, moon=False
        )

        assert np.allclose(plotter.fig.get_size_inches(), [6.0, 8.0])
        assert isinstance(plotter.alt_ax, plt.Axes)
        assert isinstance(plotter.sky_ax, plt.Axes)

        assert plotter.axes_initialized
        assert plotter.altitude_plotted
        assert plotter.sky_plotted
        assert plotter.moon_plotted is False
        assert plotter.sun_plotted
        assert plotter.axes_formatted

        plotter.fig.savefig(fig_path)
        assert fig_path.exists()
