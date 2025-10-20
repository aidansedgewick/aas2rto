import pytest
from typing import Callable

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import sncosmo

from aas2rto.scoring.supernova_peak import (
    SupernovaPeakScore,
    logistic,
    gauss,
    calc_mag_factor,
    calc_timespan_factor,
    calc_rising_fraction,
    calc_chisq_factor,
    calc_cv_prob_penalty,
    calc_color_factor,
)
from aas2rto.target import Target
from aas2rto.target_data import TargetData


@pytest.fixture
def t_score():
    return Time(60010.0, format="mjd")


@pytest.fixture
def lc_ztf(lc_pandas: pd.DataFrame):
    col_mapping = {
        "obsid": "candid",
        "mag": "magpsf",
        "magerr": "sigmapsf",
        "band": "fid",
    }
    lc_pandas.rename(col_mapping, inplace=True, axis=1)
    lc_pandas.loc[:, "blah"] = 100.0
    print(lc_pandas)
    return lc_pandas


@pytest.fixture
def ztf_td(lc_ztf: pd.DataFrame):
    return TargetData(lightcurve=lc_ztf)


@pytest.fixture
def salt_model():
    model = sncosmo.Model(source="salt3")
    model.update(dict(z=0.05, x0=1e-2, x1=0.0, c=0.0, t0=60010.0))
    result = dict(chisq=5.0, ndof=5, vparam_names=["z", "x0", "x1", "c", "t0"])
    model.result = result
    return model


@pytest.fixture
def salt_model_with_samples(salt_model):
    mu_vals = np.array([0.05, 1e-2, 0.0, 0.0, 60010.0])  # z, x0, x1, c, t0
    sig_vals = np.array([0.01, 1e-4, 0.05, 0.01, 0.3])

    samples = np.column_stack(
        [np.random.normal(mu, sig, 500) for mu, sig in zip(mu_vals, sig_vals)]
    )

    salt_model.result["samples"] = samples
    return salt_model


@pytest.fixture
def target_to_score(
    basic_target: Target, ztf_td: TargetData, lc_compiler: Callable, t_score: Time
):
    basic_target.target_data["ztf"] = ztf_td
    basic_target.compiled_lightcurve = lc_compiler(basic_target, t_ref=t_score)
    return basic_target


@pytest.fixture
def scoring_func():
    return SupernovaPeakScore(faint_limit=20.0)  # so we definitely get the target.


class Test__HelperFunctions:
    def test__logistic(self):
        # Act
        y1 = logistic(-10.0, 1.0, -1.0, 0.0)
        y2 = logistic(0.0, 1.0, -1.0, 0.0)
        y3 = logistic(10.0, 1.0, -1.0, 0.0)

        # Assert
        assert y1 > y2  # Check k definition
        assert y2 > y3

    def test__gauss(self):
        # Act
        gauss(0.0, 1.0, 0.0, 1.0)  # No crash...


class Test__MagFactor:
    def test__mag_factor(self):
        # Act
        mf1 = calc_mag_factor(20.0)
        mf2 = calc_mag_factor(18.0)
        mf3 = calc_mag_factor(16.0)

        # Assert
        assert mf1 < mf2
        assert mf2 < mf3


class Test__TimespanFactor:
    def test__timespan(self):
        # Assert
        assert np.isclose(calc_timespan_factor(50.0), 1e-3)


class Test__RiseFactor:
    def test__rising_lc(self):
        # Arrange
        df = pd.DataFrame({"mag": [20.0, 19.0, 18.0, 17.0, 18.0]})

        # Assert
        assert np.isclose(calc_rising_fraction(df), 0.75)

    def test__falling_lc(self):
        # Arrange
        df = pd.DataFrame({"mag": [16.0, 17.0, 18.0, 19.0]})

        # Assert
        assert np.isclose(calc_rising_fraction(df), 0.05)  # minimum

    def test__single_point(self):
        # Arrange
        df = pd.DataFrame({"mag": [16.0]})

        # Act
        rf = calc_rising_fraction(df, single_point_value=0.3)

        # Assert
        assert np.isclose(rf, 0.3)


class Test__ChiSqFactor:
    def test__mid_val(self):
        # Assert
        assert np.isclose(calc_chisq_factor(1.0), 1.0)

    def test__mid_high_val(self):
        # Assert
        assert np.isclose(calc_chisq_factor(7.0), 0.5)

    def test__mid_low_val(self):
        # Assert
        assert np.isclose(calc_chisq_factor(0.15), 0.5)

    def test__v_high(self):
        # Assert
        assert np.isclose(calc_chisq_factor(20.0), 0.1)

    def test__v_low(self):
        # Assert
        assert np.isclose(calc_chisq_factor(0.05), 0.1)


class Test__ColorFactor:
    def test__red_no_penalty(self):
        # Assert
        assert calc_color_factor(20.0, 15.0) > 0.99  # no penalty for red sources

    def test__promotes_blue(self):
        # Assert
        assert calc_color_factor(18.0, 20.0) > 2.0  # no penalty for red sources


class Test__CVPenalty:
    def test__no_penalty(
        self,
        target_to_score: Target,
        salt_model: sncosmo.Model,
    ):
        # Arrange
        lc = target_to_score.compiled_lightcurve
        det = lc[lc["tag"] == "valid"]
        ulim = lc[lc["tag"] == "upperlim"]

        # Act
        cv_factor, comms = calc_cv_prob_penalty(det, ulim, salt_model)

        # Assert
        assert np.isclose(cv_factor, 1.0)
        assert "no CV penalty" in comms[0]

    def test__gives_penalty(
        self,
        target_to_score: Target,
        salt_model: sncosmo.Model,
    ):
        # Arrange
        salt_model.update(dict(t0=60001.5))  # so there are no detections pre-t0
        lc = target_to_score.compiled_lightcurve
        det = lc[lc["tag"] == "valid"]
        ulim = lc[lc["tag"] == "upperlim"]

        # Act
        cv_factor, comms = calc_cv_prob_penalty(det, ulim, salt_model)
        assert cv_factor < 1.0
        comm_str = " ".join(comms)
        assert "CV penalty:" in comm_str


class Test__ScoringClassInit:
    def test__init_class(self):
        # Act
        scoring_func = SupernovaPeakScore(faint_limit=22.0)

        # Assert
        assert hasattr(scoring_func, "__name__")
        assert np.isclose(scoring_func.faint_limit, 22.0)


class Test__ScoreTargetNoModel:
    def test__no_lc_no_fail(
        self, basic_target: Target, scoring_func: Callable, t_score: Time
    ):
        # Act
        score, comms = scoring_func(basic_target, t_score)

        # Assert
        assert np.isclose(score, -1.0)
        assert len(comms) == 1
        assert comms[0] == "no compiled lightcurve"

    def test__basic_call(
        self, target_to_score: Target, scoring_func: Callable, t_score: Time
    ):
        # Act
        score, comms = scoring_func(target_to_score, t_ref=t_score)

        # Assert
        assert score > 0.0
        assert isinstance(comms, list)

        comm_str = " ".join(comms)
        assert "mag_factor=" in comm_str
        assert "timespan f=" in comm_str and "from timespan=" in comm_str
        assert "f_rise:" in comm_str
        assert "no 'sncosmo_salt' model to score" in comm_str

        assert "REJECT: MW centre" not in comm_str
        assert "REJECT: disc" not in comm_str
        assert "REJECT: target age is" not in comm_str

    def test__exclude_no_detections(
        self, target_to_score: Target, scoring_func: Callable, t_score: Time
    ):
        # Arrange
        lc = target_to_score.compiled_lightcurve
        target_to_score.compiled_lightcurve = lc[lc["tag"] == "upperlim"]

        # Act
        score, comms = scoring_func(target_to_score, t_score)

        # Assert
        assert np.isclose(score, -1.0)

        comm_str = " ".join(comms)
        assert "exclude: no detections" in comm_str

    def test__exclude_insuff_detections(self, target_to_score: Target, t_score: Time):
        # Arrange
        scoring_func = SupernovaPeakScore(min_detections=20)

        # Act
        score, comms = scoring_func(target_to_score, t_score)

        # Assert
        assert np.isclose(score, -1.0)

        comm_str = " ".join(comms)
        assert "exclude: N_det=" in comm_str

    def test__exclude_faint(self, target_to_score: Target, t_score: Time):
        # Arrange
        scoring_func = SupernovaPeakScore(faint_limit=18.0)

        # Act
        score, comms = scoring_func(target_to_score, t_ref=t_score)

        # Assert
        assert np.isclose(score, -1.0)
        assert isinstance(comms, list)

        comm_str = " ".join(comms)
        assert "exclude: faint mag"

    def test__reject_old_target(
        self,
        target_to_score: Target,
        scoring_func: Callable,
    ):
        # Arrange
        t_later = Time(60050.0, format="mjd")

        # Act
        score, comms = scoring_func(target_to_score, t_ref=t_later)

        # Assert
        assert not np.isfinite(score)
        assert isinstance(comms, list)

        comm_str = " ".join(comms)
        assert "REJECT: MW centre" not in comm_str
        assert "REJECT: disc" not in comm_str
        assert "REJECT: target age is" in comm_str

    def test__reject_mw_centre_target(
        self, target_to_score: Target, scoring_func: Callable, t_score: Time
    ):
        # Arrange
        hmsdms = (u.hourangle, u.deg)
        target_to_score.coord = SkyCoord(ra="17:45:40", dec="-29:00:28", unit=hmsdms)

        # Act
        score, comms = scoring_func(target_to_score, t_ref=t_score)

        # Assert
        assert not np.isfinite(score)
        assert isinstance(comms, list)

        comm_str = " ".join(comms)
        assert "REJECT: MW centre" in comm_str
        assert "REJECT: disc" in comm_str
        assert "REJECT: target age is" not in comm_str

    def test__reject_mw_disc_target(
        self, target_to_score: Target, scoring_func: Callable, t_score: Time
    ):
        # Arrange
        hmsdms = (u.hourangle, u.deg)
        target_to_score.coord = SkyCoord(ra="05:45:40", dec="+28:00:28", unit=hmsdms)

        # Act
        score, comms = scoring_func(target_to_score, t_ref=t_score)

        # Assert
        assert not np.isfinite(score)
        assert isinstance(comms, list)

        comm_str = " ".join(comms)
        assert "REJECT: MW centre" not in comm_str
        assert "REJECT: disc" in comm_str
        assert "REJECT: target age is" not in comm_str


class Test__ScoreTargetWithModel:
    def test__call_with_model(
        self,
        target_to_score: Target,
        salt_model: sncosmo.Model,
        scoring_func: Callable,
        t_score: Time,
    ):
        # Arrange
        target_to_score.models["sncosmo_salt"] = salt_model

        # Act
        score, comms = scoring_func(target_to_score, t_score)

        # Assert
        assert score > 0.0
        assert isinstance(comms, list)

    def test__reject_past_peak(
        self, target_to_score: Target, salt_model: sncosmo.Model, scoring_func: Callable
    ):
        # Arrange
        target_to_score.models["sncosmo_salt"] = salt_model
        t_late = Time(60026.0, format="mjd")  # lc-age rejected at 60034.

        # Act
        score, comms = scoring_func(target_to_score, t_late)

        # Assert
        assert not np.isfinite(score)
        assert isinstance(comms, list)

        comm_str = " ".join(comms)
        assert "REJECT: target age is" not in comm_str
        assert "REJECT: too far past peak" in comm_str

    def test__model_bad_chisq(
        self,
        target_to_score: Target,
        salt_model: sncosmo.Model,
        scoring_func: Callable,
        t_score: Time,
    ):
        # Arrange
        salt_model.result["chisq"] = np.nan
        target_to_score.models["sncosmo_salt"] = salt_model

        # Act
        score, comms = scoring_func(target_to_score, t_score)

        # Assert
        assert score > 0
        assert isinstance(comms, list)
        comm_str = " ".join(comms)
        print(comm_str)
        assert "no f_chisq: non-float chisq_nu=" in comm_str
        assert "ignore f_interest" in comm_str and "non-float chisq_nu" in comm_str

    def test__model_bad_ndof(
        self,
        target_to_score: Target,
        salt_model: sncosmo.Model,
        scoring_func: Callable,
        t_score: Time,
    ):
        # Arrange
        salt_model.result["ndof"] = 0
        target_to_score.models["sncosmo_salt"] = salt_model

        # Act
        score, comms = scoring_func(target_to_score, t_score)

        # Assert
        assert score > 0
        assert isinstance(comms, list)
        comm_str = " ".join(comms)
        assert "model has bad ndof" in comm_str
        assert "ignore f_interest" in comm_str and "non-float chisq_nu" in comm_str

    def test__high_chisq_ignore_interest(
        self,
        target_to_score: Target,
        salt_model: sncosmo.Model,
        scoring_func: Callable,
        t_score: Time,
    ):
        # Arrange
        salt_model.result["chisq"] = 10000.0
        target_to_score.models["sncosmo_salt"] = salt_model

        # Act
        score, comms = scoring_func(target_to_score, t_score)

        # Assert
        assert score > 0
        assert isinstance(comms, list)
        comm_str = " ".join(comms)
        assert "ignore f_interest" in comm_str and "bad value chisq_nu=" in comm_str

    def test__model_bad_t0(
        self,
        target_to_score: Target,
        salt_model: sncosmo.Model,
        scoring_func: Callable,
        t_score: Time,
    ):
        # Arrange
        salt_model.update(dict(t0=np.nan))
        target_to_score.models["sncosmo_salt"] = salt_model

        # Act
        score, comms = scoring_func(target_to_score, t_score)

        # Assert
        assert score > 0
        assert isinstance(comms, list)
        comm_str = " ".join(comms)
        assert "interest_factor not finite" in comm_str

    def test__model_with_samples(
        self,
        target_to_score: Target,
        salt_model_with_samples: sncosmo.Model,
        scoring_func: Callable,
        t_score: Time,
    ):
        # Arrange
        target_to_score.models["sncosmo_salt"] = salt_model_with_samples

        # Act
        score, comms = scoring_func(target_to_score, t_score)

        comm_str = " ".join(comms)

        # Assert
        assert score > 0
        assert isinstance(comms, list)
        comm_str = " ".join(comms)
