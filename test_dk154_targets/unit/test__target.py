import pytest
import yaml

import numpy as np

import pandas as pd

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from astroplan import Observer

from dk154_targets import Target, TargetData
from dk154_targets import target


class BasicTestModel:
    def __init__(self, *args):
        pass

    def get_band_flux(self, band):
        return 100.0


class AnotherBasicModel:
    def __init__(self, *args):
        pass

    def get_band_flux(self, band):
        return 20.0


class TestTargetData:
    def test__target_data_init(self):
        td = TargetData()
        assert td.lightcurve is None
        assert td.detections is None
        assert td.non_detections is None
        assert td.probabilities is None
        assert isinstance(td.parameters, dict)
        assert len(td.parameters) == 0
        assert isinstance(td.cutouts, dict)
        assert len(td.cutouts) == 0
        assert isinstance(td.meta, dict)
        assert len(td.meta) == 0

    def test__target_data_add_lightcurve(self):
        td = TargetData()
        rows = [(60000.1, 20001, 18.0, 0.1), (60000.0, 20000, 17.9, 0.1)]
        lc = pd.DataFrame(rows, columns="mjd candid mag magerr".split())
        assert "jd" not in lc.columns

        td.add_lightcurve(lc)
        assert td.lightcurve is not None
        assert td.detections is not None
        assert td.non_detections is None
        assert len(td.lightcurve) == 2
        assert len(td.detections) == 2
        assert "jd" in td.lightcurve.columns  # added
        assert td.lightcurve.iloc[0].candid == 20000  # correctly sorted
        assert td.lightcurve.iloc[1].candid == 20001  # correctly sorted
        assert np.isclose(td.lightcurve.iloc[0].jd - 2460000.0, 0.5)  # jd calc is good
        assert np.isclose(td.lightcurve.iloc[1].jd - 2460000.0, 0.6)


class TestTarget:
    def test__target_init(self):
        t1 = Target("test1", ra=45.0, dec=45.0)

        # Boring tests, will break if __init__ changes.
        assert isinstance(t1.coord, SkyCoord)
        assert isinstance(t1.alerce_data, TargetData)
        assert isinstance(t1.atlas_data, TargetData)
        assert isinstance(t1.fink_data, TargetData)
        assert isinstance(t1.tns_data, TargetData)

        assert set(t1.observatory_night.keys()) == set(["no_observatory"])

        assert len(t1.models) == 0
        assert set(t1.score_history.keys()) == set(["no_observatory"])
        assert set(t1.rank_history.keys()) == set(["no_observatory"])

        assert not t1.target_of_opportunity
        assert not t1.updated

    def test__basic_evaluate_target(self):
        def basic_score(target, observer):
            return 50.0

        t1 = Target("test1", ra=45.0, dec=45.0)

        t_ref = Time("2023-02-25T00:00:00", format="isot")  # mjd=60000.
        test_obs = Observer(location=EarthLocation.of_site("palomar"), name="test_obs")

        # Evaluate
        t1.evaluate_target(basic_score, test_obs, t_ref=t_ref)
        assert set(t1.score_history.keys()) == set(["no_observatory", "test_obs"])
        assert len(t1.score_history["no_observatory"]) == 0
        assert len(t1.score_history["test_obs"]) == 1
        assert isinstance(t1.score_history["test_obs"][-1], tuple)
        assert isinstance(t1.score_history["test_obs"][-1][0], float)
        assert isinstance(t1.score_history["test_obs"][-1][1], Time)

        assert t1.score_history["test_obs"][-1][0] == 50.0
        assert t1.score_history["test_obs"][-1][1].mjd == 60000.0

        assert t1.score_comments["no_observatory"] is None
        assert t1.reject_comments["no_observatory"] is None
        assert t1.score_comments["test_obs"] is None
        assert t1.reject_comments["test_obs"] is None

    def test__evaluate_target_with_comments(self):
        def score_with_comments(target, observer):
            return 50.0, ["this is a comment"], None

        t1 = Target("test1", ra=45.0, dec=45.0)

        t_ref = Time("2023-02-25T00:00:00", format="isot")  # mjd=60000.
        test_obs = Observer(location=EarthLocation.of_site("palomar"), name="test_obs")

        # Evaluate
        t1.evaluate_target(score_with_comments, test_obs, t_ref=t_ref)
        assert set(t1.score_history.keys()) == set(["no_observatory", "test_obs"])
        assert len(t1.score_history["no_observatory"]) == 0
        assert len(t1.score_history["test_obs"]) == 1
        assert isinstance(t1.score_history["test_obs"][-1], tuple)
        assert isinstance(t1.score_history["test_obs"][-1][0], float)
        assert isinstance(t1.score_history["test_obs"][-1][1], Time)

        assert t1.score_history["test_obs"][-1][1].mjd == 60000.0
        assert t1.score_history["test_obs"][-1][0] == 50.0

        assert t1.score_comments["no_observatory"] is None
        assert t1.reject_comments["no_observatory"] is None
        assert len(t1.score_comments["test_obs"]) == 1
        assert t1.score_comments["test_obs"][0] == "this is a comment"
        assert t1.reject_comments["test_obs"] is None

    def test__evaluate_target_bad_functions(self):
        def bad_return_function(target, observer):
            return 1.5, ["blah blah"]

        no_obs = None

        t1 = Target("t1", ra=45.0, dec=0.0)
        with pytest.raises(ValueError):
            t1.evaluate_target(bad_return_function, no_obs)

        def other_bad_return_function(target, obs):
            return 10.0, [], [], []

        with pytest.raises(ValueError):
            t1.evaluate_target(other_bad_return_function, no_obs)

    def test__get_last_score(self):
        def increasing_score(target, obs):
            if obs is None:
                obs_name = "no_observatory"
            else:
                obs_name = obs.name
            history = target.score_history.get(obs_name, [])
            return len(history) + 1.0

        palomar = Observer(location=EarthLocation.of_site("palomar"), name="palomar")
        t1 = Target("t1", ra=30.0, dec=45.0)
        assert set(t1.score_history.keys()) == set(["no_observatory"])

        # Check get_last_score() returns None if there is no score.
        last_score = t1.get_last_score()
        assert last_score is None
        last_score_tuple = t1.get_last_score(return_time=True)
        assert len(last_score_tuple) == 2
        assert not any(last_score_tuple)  # any(lst) is True if any are True-like.

        # Check the same behaviour for a named observatory.
        last_palomar_score = t1.get_last_score("palomar")
        assert last_palomar_score is None
        last_palomar_score_tuple = t1.get_last_score("palomar", return_time=True)
        assert len(last_palomar_score_tuple) == 2
        assert not any(last_palomar_score_tuple)

        # Evaluate the target 5 times for `no_observatory`.
        t_ref = Time("2023-02-25T00:00:00", format="isot")
        for ii in range(5):
            t_eval = t_ref + ii * u.day
            t1.evaluate_target(increasing_score, None, t_ref=t_eval)

        # Check that the last score is 5.
        last_score = t1.get_last_score()
        assert last_score == 5.0
        last_score_tuple = t1.get_last_score(return_time=True)
        assert last_score_tuple[0] == 5.0
        assert last_score_tuple[1].mjd == 60004.0
        no_obs_last_score = t1.get_last_score("no_observatory")
        assert no_obs_last_score == 5.0

        t1.evaluate_target(increasing_score, palomar, t_ref=t_ref)
        assert t1.get_last_score("palomar") == 1.0
        palomar_t1_last_score_tuple = t1.get_last_score("palomar", return_time=True)
        assert palomar_t1_last_score_tuple[0] == 1.0
        assert palomar_t1_last_score_tuple[1].mjd == 60000.0

        for ii in range(1, 3):
            t_eval = t_ref + ii * u.day
            t1.evaluate_target(increasing_score, palomar, t_ref=t_eval)
        assert t1.get_last_score("palomar") == 3.0

    def test__custom_compile_lightcurve(self):
        def basic_test_function(target):
            dat = target.alerce_data.lightcurve
            dat["jd"] = dat["jd"] + 1.0
            return dat

        lc = pd.DataFrame(
            dict(
                jd=np.arange(60000.0, 60005.0, 1.0),
                mag=np.array([20.1, 20.2, 20.3, 20.4, 20.5]),
                magerr=np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
            )
        )

        alerce_data = TargetData(lightcurve=lc)
        t1 = Target("t1", 45.0, 60.0, alerce_data=alerce_data)
        assert t1.compiled_lightcurve is None

        t_ref = Time(60006.0, format="jd")
        t1.build_compiled_lightcurve(compile_function=basic_test_function, t_ref=t_ref)
        assert len(t1.compiled_lightcurve) == 5
        assert np.allclose(
            t1.compiled_lightcurve["jd"].values,
            [60001.0, 60002.0, 60003.0, 60004.0, 60005.0],
        )

    def test__build_model(self):
        def build_basic_model(target):
            return BasicTestModel()

        def build_other_basic_model(target):
            return AnotherBasicModel()

        tt = Target("tt", 0.0, 0.0)
        m1 = build_basic_model(tt)
        m2 = build_other_basic_model(tt)
        assert isinstance(m1, BasicTestModel)
        assert isinstance(m2, AnotherBasicModel)

        t1 = Target("t1", ra=45.0, dec=60.0)
        assert isinstance(t1.models, dict)
        assert len(t1.models) == 0

        t_ref1 = Time("2023-02-26T00:00:00", format="isot")  # mjd=60001.
        t1.build_models(build_basic_model, lazy=False, t_ref=t_ref1)
        assert set(t1.models.keys()) == set(["BasicTestModel"])
        assert isinstance(t1.models["BasicTestModel"], BasicTestModel)
        assert np.isclose(t1.models_tref["BasicTestModel"].mjd - 60000.0, 1.0)

        t_ref2 = Time("2023-02-27T00:00:00", format="isot")  # mjd=60002.
        t1.build_models(
            [build_basic_model, build_other_basic_model], lazy=False, t_ref=t_ref2
        )
        assert set(t1.models.keys()) == set(["BasicTestModel", "AnotherBasicModel"])
        assert isinstance(t1.models["BasicTestModel"], BasicTestModel)
        assert isinstance(t1.models["AnotherBasicModel"], AnotherBasicModel)
        assert np.isclose(t1.models_tref["BasicTestModel"].mjd - 60000.0, 2.0)
        assert np.isclose(t1.models_tref["AnotherBasicModel"].mjd - 60000.0, 2.0)


#     def test__check_broker_priority():
#         test_broker_priority = ("lasair", "fink", "alerce")
#         t1 = Target("t1", ra=45.0, dec=80.0, broker_priority=test_broker_priority)
#         target._check_broker_priority(t1.broker_priority)

#         priority_will_fail = ("fink", "alerce", "non_existant")
#         with pytest.raises(ValueError):
#             t2 = Target("t2", ra=30.0, dec=23.0, broker_priority=priority_will_fail)

#         t1.broker_priority = priority_will_fail
#         with pytest.raises(ValueError):
#             target._check_broker_priority(t1.broker_priority)
# '
