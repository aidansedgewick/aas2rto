import pytest

import numpy as np

import pandas as pd

from astropy.time import Time

from dk154_targets.lightcurve_compilers import (
    default_compile_lightcurve,
    prepare_atlas_data,
)
from dk154_targets.target import Target, TargetData


@pytest.fixture
def atlas_lc():
    mjd_dat = np.arange(60000.0, 60005.0, 0.5)
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
def short_atlas_lc():
    atlas_dict = dict(
        mjd=np.arange(60000.5, 60005.5, 1.0),  # last is 60004.5
        m=np.array([19.5, 19.6, 19.7, 19.8, 19.9]),
        dm=[0.2] * 5,
        mag5sig=[20.0] * 5,
        F=list("oocco"),
    )
    return pd.DataFrame(atlas_dict)


@pytest.fixture
def short_alerce_lc():
    alerce_dict = dict(
        mjd=np.arange(60000.0, 60005.0, 1.0),  # last is 60004.
        magpsf=np.array([20.1, 20.2, 20.3, 20.4, 20.5]),
        sigmapsf=np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        fid=[1, 1, 2, 1, 2],
        # band=np.array("ztfg ztfg ztfr ztfg ztfr".split()),
    )
    return pd.DataFrame(alerce_dict)


def test__prepare_atlas_data(atlas_lc):
    formatted_df = prepare_atlas_data(atlas_lc)

    expected_columns = "mag magerr diffmaglim band jd mjd tag".split()
    assert set(formatted_df.columns) == set(expected_columns)
    assert len(formatted_df) == 8

    expected_mag_values = np.array([20.0, 19.9, 19.6, 19.5, 19.4, 19.3, 19.2, 19.1])
    assert np.allclose(formatted_df["mag"].values, expected_mag_values)

    assert np.allclose(formatted_df["magerr"].values, 0.1)

    expected_bands = tuple(["atlas" + b for b in "cooccooo"])
    assert tuple(formatted_df["band"].values) == expected_bands


def test__default_compile_lightcurve(short_alerce_lc, short_atlas_lc):
    alerce_data = TargetData(lightcurve=short_alerce_lc)

    atlas_data = TargetData(lightcurve=short_atlas_lc)
    t1 = Target("t1", ra=45.0, dec=60.0, alerce_data=alerce_data, atlas_data=atlas_data)
    t_ref = Time(60003.75, format="mjd")  # cut off the last from ztf and atlas.

    assert t1.compiled_lightcurve is None
    t1.build_compiled_lightcurve(default_compile_lightcurve, t_ref=t_ref)
    assert len(t1.compiled_lightcurve) == 8

    expected_columns = ["jd", "mag", "magerr", "band"]
    assert all(col in t1.compiled_lightcurve.columns for col in expected_columns)
    missing_cols = [
        col for col in expected_columns if col not in t1.compiled_lightcurve.columns
    ]
    assert len(missing_cols) == 0

    expected_bands = "ztfg atlaso ztfg atlaso ztfr atlasc ztfg atlasc".split()
    assert list(t1.compiled_lightcurve["band"].values) == expected_bands
