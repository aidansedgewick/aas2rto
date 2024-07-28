import pytest

import numpy as np

import pandas as pd

from astropy.time import Time

import sncosmo

from dk154_targets.target import Target, TargetData
from dk154_targets.scoring.supernova_peak import (
    calc_mag_factor,
    calc_timespan_factor,
    calc_rising_fraction,
    calc_color_factor,
    peak_only_interest_function,
    SupernovaPeakScore,
)


@pytest.fixture
def rising_lc_rows():
    return [
        (60000.0, 18.6, 1),
        (60000.1, 18.5, 2),
        (60001.0, 17.6, 1),
        (60001.1, 17.5, 2),
        (60002.0, 16.6, 1),
        (60002.1, 16.5, 2),
    ]


@pytest.fixture
def rising_lc(rising_lc_rows):
    return pd.DataFrame(rising_lc_rows, columns="mjd magpsf fid".split())


@pytest.fixture
def mock_salt_model():
    m = sncosmo.Model(source="salt2")
    m.set(t0=60005.0)
    return m


@pytest.fixture
def mock_target(rising_lc):
    data = TargetData(lightcurve=rising_lc)
    target = Target("ZTF00abc", ra=30.0, dec=60.0, target_data={"fink": data})
    return target


@pytest.fixture
def mock_target_with_model(mock_target, mock_salt_model):
    mock_target.models["sncosmo_salt"] = mock_salt_model
    return mock_target


class Test__CalcMagFactor:
    def test__values(self):
        x1 = calc_mag_factor(18.5)
        assert np.isclose(x1, 1.0)

        x2 = calc_mag_factor(16.5)
        assert np.isclose(x2, 10.0)

        x3 = calc_mag_factor(14.5)
        assert np.isclose(x3, 100.0)


class Test__ScoringClassInit:
    def test__no_kwargs(self):

        func = SupernovaPeakScore()
        assert func.__name__ == "supernova_peak_score"
        assert isinstance(func.faint_limit, float)
        assert isinstance(func.min_timespan, float)
        assert isinstance(func.max_timespan, float)
        assert isinstance(func.min_rising_fraction, float)
        assert isinstance(func.min_ztf_detections, int)
        assert isinstance(func.default_color_factor, float)
        assert isinstance(func.broker_priority, tuple)

    def test__change_kwargs(self):
        func = SupernovaPeakScore(faint_limit=20.0, min_timespan=3.0)

        assert np.isclose(func.faint_limit, 20.0)
        assert np.isclose(func.min_timespan, 3.0)


class Test__Lightcurve:
    def test__mag_factor(self, rising_lc: pd.DataFrame):
        mag = rising_lc["magpsf"].iloc[-1]
        mag_factor = calc_mag_factor(mag)
        assert np.isclose(mag_factor, 10.0)

    def test__rising_fractions(self, rising_lc: pd.DataFrame):
        for fid, fid_lc in rising_lc.groupby("fid"):
            fid_fact = calc_rising_fraction(fid_lc)
            assert np.isclose(fid_fact, 1.0)


class Test__CallScoringClass:

    def test__target_no_model(self, mock_target: Target):
        func = SupernovaPeakScore()
        t_ref = Time(60005.0, format="mjd")

        score, comms, reject_comms = func(mock_target, t_ref)

        assert np.isclose(score, 10.0)

    def test__target_with_model(self, mock_target_with_model: Target):
        func = SupernovaPeakScore()
        t_ref = Time(60005.0, format="mjd")

        score, comms, reject_comms = func(mock_target_with_model, t_ref)

        assert np.isclose(score, 100.0)

    def test__low_score_old(self, mock_target: Target):
        func = SupernovaPeakScore()
        t_ref = Time(60025.0, format="mjd")

        score, comms, reject_comms = func(mock_target, t_ref)

        assert np.isclose(score, 5.0)

    def test__target_reject_old(self, mock_target: Target):
        func = SupernovaPeakScore()
        t_ref = Time(60031.0, format="mjd")

        score, comms, reject_comms = func(mock_target, t_ref)

        assert not np.isfinite(score)
