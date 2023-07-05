import os
import pytest
import yaml

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from astroplan import Observer

from dk154_targets import Target, TargetData
from dk154_targets import target
from dk154_targets.target import default_plot_lightcurve
from dk154_targets.lightcurve_compilers import default_compile_lightcurve

from dk154_targets import paths


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


@pytest.fixture
def example_target():
    return Target("test1", ra=45.0, dec=30)


@pytest.fixture
def fink_rows():
    return [
        (0, 2460001.5, np.nan, np.nan, 17.0, np.nan, np.nan, 1, "upperlim"),
        (0, 2460002.5, np.nan, np.nan, 17.0, np.nan, np.nan, 2, "upperlim"),
        (0, 2460003.5, np.nan, np.nan, 17.0, np.nan, np.nan, 2, "upperlim"),
        (0, 2460004.5, 18.5, 0.5, 19.0, np.nan, np.nan, 1, "badquality"),
        (23000_10000_20000_5005, 2460005.5, 18.2, 0.1, 19.0, 30.0, 45.0, 2, "valid"),
        (23000_10000_20000_5006, 2460006.5, 18.3, 0.1, 19.5, 30.0, 45.0, 1, "valid"),
        (23000_10000_20000_5007, 2460007.5, 18.0, 0.1, 19.5, 30.0, 45.0, 1, "valid"),
    ]


@pytest.fixture
def fink_lc(fink_rows):
    return pd.DataFrame(
        fink_rows, columns="candid jd magpsf sigmapsf diffmaglim ra dec fid tag".split()
    )


@pytest.fixture
def atlas_lc():
    mjd_dat = np.arange(60000.1, 60005.1, 0.5)
    rows = [
        (20.0, 0.1, 19.0, "c"),  # below 5 sig
        (19.9, 0.1, 21.0, "o"),
        (-19.8, 0.1, 21.0, "c"),  # negative mag
        (-19.7, 0.1, 21.0, "o"),  # negative mag
        (19.6, 0.1, 21.0, "o"),
        (19.5, 0.1, 21.0, "c"),
        (19.4, 0.1, 21.0, "c"),
        (19.3, 0.1, 19.0, "o"),  # below 5 sig
        (19.2, 0.1, 21.0, "o"),
        (19.1, 0.1, 21.0, "o"),  #
    ]

    fake_lc = pd.DataFrame(rows, columns="m dm mag5sig F".split())
    fake_lc.insert(0, "mjd", pd.Series(mjd_dat, name="mjd"))
    return fake_lc


@pytest.fixture
def fake_cutouts():
    cutouts = {}
    cutouts["science"] = np.random.random((60, 60))
    cutouts["template"] = np.random.random((60, 60))
    cutouts["difference"] = np.random.random((60, 60))

    return cutouts


@pytest.fixture
def target_with_data(example_target, fink_lc, atlas_lc, fake_cutouts):
    example_target.fink_data.add_lightcurve(fink_lc)
    example_target.atlas_data.add_lightcurve(atlas_lc)
    example_target.cutouts = fake_cutouts
    return example_target


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

        assert set(t1.observatory_info.keys()) == set(["no_observatory"])

        assert len(t1.models) == 0
        assert set(t1.score_history.keys()) == set(["no_observatory"])
        assert set(t1.rank_history.keys()) == set(["no_observatory"])

        assert not t1.target_of_opportunity
        assert not t1.updated

    def test__basic_evaluate_target(self, example_target: Target):
        def basic_score(target, observer):
            return 50.0

        t1 = Target("test1", ra=45.0, dec=45.0)

        t_ref = Time("2023-02-25T00:00:00", format="isot")  # mjd=60000.
        test_obs = Observer(location=EarthLocation.of_site("palomar"), name="test_obs")

        # Evaluate
        example_target.evaluate_target(basic_score, test_obs, t_ref=t_ref)
        assert set(example_target.score_history.keys()) == set(
            ["no_observatory", "test_obs"]
        )
        assert len(example_target.score_history["no_observatory"]) == 0
        assert len(example_target.score_history["test_obs"]) == 1
        assert isinstance(example_target.score_history["test_obs"][-1], tuple)
        assert isinstance(example_target.score_history["test_obs"][-1][0], float)
        assert isinstance(example_target.score_history["test_obs"][-1][1], Time)

        assert example_target.score_history["test_obs"][-1][0] == 50.0
        assert example_target.score_history["test_obs"][-1][1].mjd == 60000.0

        assert example_target.score_comments["no_observatory"] is None
        assert example_target.reject_comments["no_observatory"] is None
        assert example_target.score_comments["test_obs"] is None
        assert example_target.reject_comments["test_obs"] is None

    def test__evaluate_target_with_comments(self):
        def score_with_comments(target, observer):
            return 50.0, ["this is a comment"], None

        example_target = Target("tesexample_target", ra=45.0, dec=45.0)

        t_ref = Time("2023-02-25T00:00:00", format="isot")  # mjd=60000.
        test_obs = Observer(location=EarthLocation.of_site("palomar"), name="test_obs")

        # Evaluate
        example_target.evaluate_target(score_with_comments, test_obs, t_ref=t_ref)
        assert set(example_target.score_history.keys()) == set(
            ["no_observatory", "test_obs"]
        )
        assert len(example_target.score_history["no_observatory"]) == 0
        assert len(example_target.score_history["test_obs"]) == 1
        assert isinstance(example_target.score_history["test_obs"][-1], tuple)
        assert isinstance(example_target.score_history["test_obs"][-1][0], float)
        assert isinstance(example_target.score_history["test_obs"][-1][1], Time)

        assert example_target.score_history["test_obs"][-1][1].mjd == 60000.0
        assert example_target.score_history["test_obs"][-1][0] == 50.0

        assert example_target.score_comments["no_observatory"] is None
        assert example_target.reject_comments["no_observatory"] is None
        assert len(example_target.score_comments["test_obs"]) == 1
        assert example_target.score_comments["test_obs"][0] == "this is a comment"
        assert example_target.reject_comments["test_obs"] is None

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

    def test__custom_compile_lightcurve(self, example_target, fink_lc):
        def basic_test_function(target):
            dat = target.fink_data.lightcurve
            dat["jd"] = dat["jd"] + 1.0
            return dat

        example_target.fink_data.add_lightcurve(fink_lc)
        assert example_target.compiled_lightcurve is None

        t_ref = Time(60006.0, format="mjd")
        example_target.build_compiled_lightcurve(
            compile_function=basic_test_function, t_ref=t_ref
        )
        assert len(example_target.compiled_lightcurve) == 4
        assert np.allclose(
            example_target.compiled_lightcurve["jd"].values,
            np.array([4.0, 5.0, 6.0, 7.0]) + 2_460_000.5,
        )

    def test__build_model(self):
        def basic_model(target):
            return BasicTestModel()

        def other_basic_model(target):
            return AnotherBasicModel()

        tt = Target("tt", 0.0, 0.0)
        m1 = basic_model(tt)
        m2 = other_basic_model(tt)
        assert isinstance(m1, BasicTestModel)
        assert isinstance(m2, AnotherBasicModel)

        t1 = Target("t1", ra=45.0, dec=60.0)
        assert isinstance(t1.models, dict)
        assert len(t1.models) == 0

        t_ref1 = Time("2023-02-26T00:00:00", format="isot")  # mjd=60001.
        t1.build_model(basic_model, lazy=False, t_ref=t_ref1)
        assert set(t1.models.keys()) == set(["basic_model"])
        assert isinstance(t1.models["basic_model"], BasicTestModel)
        assert np.isclose(t1.models_tref["basic_model"].mjd - 60000.0, 1.0)

        t_ref2 = Time("2023-02-27T00:00:00", format="isot")  # mjd=60002.
        t1.build_model(basic_model, lazy=False, t_ref=t_ref2)
        t1.build_model(other_basic_model, lazy=False, t_ref=t_ref2)

        assert set(t1.models.keys()) == set(["basic_model", "other_basic_model"])
        assert isinstance(t1.models["basic_model"], BasicTestModel)
        assert isinstance(t1.models["other_basic_model"], AnotherBasicModel)
        assert np.isclose(t1.models_tref["basic_model"].mjd - 60000.0, 2.0)
        assert np.isclose(t1.models_tref["other_basic_model"].mjd - 60000.0, 2.0)

    def test__lc_plotting(self, target_with_data: Target):
        def return_blank_fig(target, t_ref=None, fig=None):
            return plt.figure()

        fig_path = paths.test_data_path / "test_lc_fig.png"
        if fig_path.exists():
            os.remove(fig_path)
        fig = target_with_data.plot_lightcurve(
            lc_plotting_function=return_blank_fig, figpath=fig_path
        )
        assert isinstance(fig, plt.Figure)

        plt.close(fig=fig)
        assert fig_path.exists()  # Correctly saved!
        os.remove(fig_path)

        ## works with no function (ie, correctly uses default_lc_pl)
        fig_path = paths.test_data_path / "test_lc_fig.png"
        fig = target_with_data.plot_lightcurve(figpath=fig_path)
        assert isinstance(fig, plt.Figure)
        assert fig_path.exists()
        os.remove(fig_path)

    def test__oc_plotting(self, example_target: Target):
        observatory = Observer(location=EarthLocation.of_site("greenwich"))
        t_ref = Time(60000.0, format="mjd")
        oc_fig = example_target.plot_observing_chart(observatory, t_ref=t_ref)
        plt.close(fig=oc_fig)


def test__default_lightcurve(target_with_data: Target):
    ### basically testing to see that it doesn't crash when not wrapped in try: except:

    t_ref = Time(60010, format="mjd")

    target_with_data.build_compiled_lightcurve(default_compile_lightcurve)

    lc_fig1 = default_plot_lightcurve(target_with_data, t_ref=t_ref)
    assert isinstance(lc_fig1, plt.Figure)
    assert len(lc_fig1.axes) == 4
    plt.close(fig=lc_fig1)

    target_with_data.score_comments["no_observatory"] = "comm1 comm2 comm3".split()
    target_with_data.fink_data.cutouts = {}

    lc_fig2 = default_plot_lightcurve(target_with_data, t_ref=t_ref)
    assert len(lc_fig2.axes) == 4  # even if no cutouts

    plt.close(lc_fig2)


def test__plot_observing_chart(example_target: Target):
    observatory = Observer(location=EarthLocation.of_site("greenwich"))

    t_ref = Time(60010, format="mjd")
    # Check no errors raised
    oc_fig = target.plot_observing_chart(
        observatory, example_target, t_ref=t_ref, warn=False
    )
    plt.close(fig=oc_fig)
