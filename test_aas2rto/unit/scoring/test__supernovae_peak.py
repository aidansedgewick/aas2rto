import pytest

import numpy as np

import pandas as pd

from astropy.time import Time

import sncosmo

from aas2rto.target import Target, TargetData
from aas2rto.scoring.supernova_peak import (
    calc_mag_factor,
    calc_timespan_factor,
    calc_rising_fraction,
    calc_color_factor,
    peak_only_interest_function,
    SupernovaPeakScore,
)

from aas2rto.lightcurve_compilers import DefaultLightcurveCompiler

lc_compiler = DefaultLightcurveCompiler()


@pytest.fixture
def rising_lc_rows():
    return [
        (60000.0, 18.6, "ztfg", "valid", "ztf"),
        (60000.1, 18.5, "ztfr", "valid", "ztf"),
        (60001.0, 17.6, "ztfg", "valid", "ztf"),
        (60001.1, 17.5, "ztfr", "valid", "ztf"),
        (60002.0, 16.6, "ztfg", "valid", "ztf"),
        (60002.1, 16.5, "ztfr", "valid", "ztf"),
    ]


@pytest.fixture
def rising_lc(rising_lc_rows):
    return pd.DataFrame(rising_lc_rows, columns="mjd mag band tag source".split())


@pytest.fixture
def mock_salt_model():
    m = sncosmo.Model(source="salt2")
    m.set(t0=60005.0)
    return m


@pytest.fixture
def target_t_ref():
    return Time(60010.0, format="mjd")


@pytest.fixture
def mock_target(rising_lc, target_t_ref):
    target = Target("ZTF00abc", ra=30.0, dec=-60.0)  # not in the disc!
    target.compiled_lightcurve = rising_lc
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
        assert isinstance(func.min_detections, int)
        assert isinstance(func.default_color_factor, float)
        assert isinstance(func.broker_priority, tuple)

    def test__change_kwargs(self):
        func = SupernovaPeakScore(faint_limit=20.0, min_timespan=3.0)

        assert np.isclose(func.faint_limit, 20.0)
        assert np.isclose(func.min_timespan, 3.0)


class Test__Lightcurve:
    def test__mag_factor(self, rising_lc: pd.DataFrame):
        mag = rising_lc["mag"].iloc[-1]
        mag_factor = calc_mag_factor(mag)
        assert np.isclose(mag_factor, 10.0)

    def test__rising_fractions(self, rising_lc: pd.DataFrame):
        for fid, fid_lc in rising_lc.groupby("band"):
            fid_fact = calc_rising_fraction(fid_lc)
            assert np.isclose(fid_fact, 1.0)


class Test__CallScoringClass:

    def test__target_no_model(self, mock_target: Target):
        func = SupernovaPeakScore()
        t_ref = Time(60005.0, format="mjd")

        score, comms = func(mock_target, t_ref)

        print(comms)

        assert np.isclose(score, 10.0)

    def test__target_with_model(self, mock_target_with_model: Target):
        func = SupernovaPeakScore()
        t_ref = Time(60005.0, format="mjd")

        score, comms = func(mock_target_with_model, t_ref)

        # x_mag = 10, x_peak = 30.
        assert np.isclose(score, 300.0)

    def test__low_score_old(self, mock_target: Target):
        func = SupernovaPeakScore()
        t_ref = Time(60025.0, format="mjd")

        score, comms = func(mock_target, t_ref)

        assert np.isclose(score, 5.0)

    def test__target_reject_old(self, mock_target: Target):
        func = SupernovaPeakScore()
        t_ref = Time(60031.0, format="mjd")

        score, comms = func(mock_target, t_ref)

        assert not np.isfinite(score)

    def test__no_crash_with_no_data(self, mock_target: Target):
        t_ref = Time(60005.0, format="mjd")

        mock_target.compiled_lightcurve = None

        func = SupernovaPeakScore()

        score, comms = func(mock_target, t_ref)

        assert np.isclose(score, -1.0)
        assert comms[-1].startswith("no compiled lightcurve")

    def test__exclude_target_too_faint(self, mock_target: Target):
        t_ref = Time(60005.0, format="mjd")

        faint_limit = 15.0
        func = SupernovaPeakScore(faint_limit=faint_limit)

        mag_vals = mock_target.compiled_lightcurve["mag"].values
        assert all(mag_vals > faint_limit)

        score, comms = func(mock_target, t_ref)

        assert np.isclose(score, -1.0)

        comment_found = False
        for comm in comms:
            if "exclude:" in comm and "(faint lim)" in comm:
                comment_found = True
                break
        assert comment_found

    def test__exclude_target_too_young(self, mock_target: Target):
        t_ref = Time(60005.0, format="mjd")

        min_timespan = 6.0
        func = SupernovaPeakScore(min_timespan=min_timespan)

        mjd_vals = mock_target.compiled_lightcurve["mjd"].values

        assert t_ref.mjd - mjd_vals.min() < min_timespan

        score, comms = func(mock_target, t_ref)

        assert np.isclose(score, -1.0)

        comment_found = False
        for comm in comms:
            if "exclude: timespan" in comm and "(min)" in comm:
                comment_found = True
                break
        assert comment_found

    def test__exclude_target_too_few_obs(self, mock_target: Target):
        t_ref = Time(60005.0, format="mjd")

        min_obs = 8
        func = SupernovaPeakScore(min_detections=min_obs)

        assert len(mock_target.compiled_lightcurve) < min_obs

        score, comms = func(mock_target, t_ref)

        assert np.isclose(score, -1.0)

        comment_found = False
        for comm in comms:
            if "exclude" in comm and "detections insufficient" in comm:
                comment_found = True
                break
        assert comment_found

    def test__no_fail_on_inf_model_t0(self, mock_target: Target, mock_salt_model):
        t_ref = Time(60005.0, format="mjd")

        mock_salt_model["t0"] = np.inf

        mock_target.models["sncosmo_salt"] = mock_salt_model

        func = SupernovaPeakScore()

        score, comms = func(mock_target, t_ref)

        assert np.isclose(score, 10.0)

        comment_found = False
        for comm in comms:
            if f"ZTF00abc interest_factor not finite" in comm:
                comment_found = True
                break
        assert comment_found

    def test__no_fail_on_nan_model_t0(self, mock_target: Target, mock_salt_model):
        t_ref = Time(60005.0, format="mjd")

        mock_salt_model["t0"] = np.nan

        mock_target.models["sncosmo_salt"] = mock_salt_model

        func = SupernovaPeakScore()

        score, comms = func(mock_target, t_ref)

        print(comms)

        assert np.isclose(score, 10.0)

        comment_found = False
        for comm in comms:
            if f"ZTF00abc interest_factor not finite" in comm:
                comment_found = True
                break
        assert comment_found
