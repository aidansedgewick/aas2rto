import pytest

import numpy as np

import pandas as pd

from astropy.time import Time

from aas2rto.lightcurve_compilers import (
    prepare_ztf_data,
    prepare_atlas_data,
    prepare_yse_data,
    DefaultLightcurveCompiler,
)
from aas2rto.target import Target, TargetData


@pytest.fixture
def ztf_df_rows():
    """
    candid, jd, objectId, magpsf, sigmapsf, diffmaglim, ra, dec, fid, tag
    """
    return [
        (-1, 2460001.5, "T101", np.nan, np.nan, 17.0, np.nan, np.nan, 1, "upperlim"),
        (-1, 2460002.5, "T101", np.nan, np.nan, 17.0, np.nan, np.nan, 2, "upperlim"),
        (-1, 2460003.5, "T101", np.nan, np.nan, 17.0, np.nan, np.nan, 2, "nondet"),
        (-1, 2460004.5, "T101", 18.5, 0.5, 19.0, np.nan, np.nan, 2, "badquality"),
        (-1, 2460004.5, "T101", 18.4, 0.5, 19.0, np.nan, np.nan, 1, "dubious"),
        (5006, 2460006.5, "T101", 18.2, 0.1, 19.0, 30.0, 45.0, 1, "valid"),
        (5007, 2460007.5, "T101", 18.3, 0.1, 19.5, 30.0, 45.0, 1, "detection"),
        (5008, 2460008.5, "T101", 18.0, 0.1, 19.5, 30.0, 45.0, 2, "valid"),
    ]


@pytest.fixture
def ztf_df(ztf_df_rows):
    """
    A fake lightcurve. Only use relevant columns...
    """

    return pd.DataFrame(
        ztf_df_rows,
        columns="candid jd objectId magpsf sigmapsf diffmaglim ra dec fid tag".split(),
    )


@pytest.fixture
def ztf_target_data(ztf_df):
    return TargetData(lightcurve=ztf_df)


@pytest.fixture
def mock_target():
    return Target("T101", ra=45.0, dec=30.0)


class Test__PrepareZtfData:

    def test__mock_target_data(self, ztf_target_data):
        # Shouldn't strictly be testing TargetData here, but as a reminder...

        assert hasattr(ztf_target_data, "lightcurve")
        assert len(ztf_target_data.lightcurve) == 8

        assert hasattr(ztf_target_data, "detections")
        assert hasattr(ztf_target_data, "badqual")
        assert hasattr(ztf_target_data, "non_detections")
        assert len(ztf_target_data.detections) == 3
        assert len(ztf_target_data.badqual) == 2
        assert len(ztf_target_data.non_detections) == 3

    def test__normal_behaviour(self, ztf_target_data):

        ztf_lc = prepare_ztf_data(ztf_target_data)

        assert "mjd" in ztf_lc.columns
        assert "band" in ztf_lc.columns
        assert "mag" in ztf_lc.columns
        assert "magerr" in ztf_lc.columns
        assert len(ztf_lc) == 8

        # all the names have been properly fixed.
        assert sum(ztf_lc["tag"] == "valid") == 3
        assert sum(ztf_lc["tag"] == "badqual") == 2
        assert sum(ztf_lc["tag"] == "upperlim") == 3

    def test__change_valid_tag(self, ztf_target_data):
        ztf_lc = prepare_ztf_data(ztf_target_data, valid_tag="good_det")

        assert sum(ztf_lc["tag"] == "valid") == 0
        assert sum(ztf_lc["tag"] == "good_det") == 3
        assert sum(ztf_lc["tag"] == "badqual") == 2
        assert sum(ztf_lc["tag"] == "upperlim") == 3

    def test__no_detections_attr(self, ztf_target_data):
        ztf_target_data.detections = None
        ztf_target_data.badqual = None
        ztf_target_data.non_detections = None

        ztf_lc = prepare_ztf_data(ztf_target_data)
        assert len(ztf_lc) == 8
        assert ztf_lc.iloc[2]["tag"] == "nondet"  # tags not properly fixed.


@pytest.fixture
def atlas_df_rows():
    return [
        (60000.0, 20.0, 0.1, 19.0, "c", "01"),  # below 5 sig
        (60000.5, 19.9, 0.1, 21.0, "o", "02"),
        (60001.0, -19.8, 0.1, 21.0, "c", "03"),  # negative mag
        (60001.5, -19.7, 0.1, 21.0, "o", "04"),  # negative mag
        (60002.0, 19.6, 0.1, 21.0, "o", "05"),
        (60002.49, 19.7, 0.1, 21.0, "c", "06"),  # \
        (60002.50, 19.5, 0.1, 21.0, "c", "07"),  # | These four are all
        (60002.51, 19.4, 0.1, 21.0, "c", "08"),  # | with mean m = 19.55
        (60002.52, 19.6, 0.1, 21.0, "c", "09"),  # /
        (60002.75, 19.5, 0.1, 21.0, "o", "10"),
        (60003.00, 19.4, 0.1, 21.0, "c", "11"),  # \ These two are different
        (60003.01, 19.4, 0.1, 21.0, "o", "12"),  # / filters.
        (60004.00, 19.2, 0.1, 20.0, "c", "13"),
        (60004.50, 19.3, 0.1, 19.0, "o", "14"),  # \ ulim
        (60004.51, 19.3, 0.1, 19.5, "o", "15"),  # | ave to ulim, but #15 valid.
        (60004.52, 19.3, 0.1, 19.0, "o", "16"),  # / ulim
        (60005.5, 19.1, 0.1, 21.0, "o", "17"),
    ]


@pytest.fixture
def atlas_df(atlas_df_rows):
    return pd.DataFrame(atlas_df_rows, columns="mjd m dm mag5sig F Obs".split())


@pytest.fixture
def atlas_target_data(atlas_df):
    return TargetData(lightcurve=atlas_df)


class Test__PrepareAtlasData:
    def test__atlas_target_data(self, atlas_target_data):
        assert len(atlas_target_data.lightcurve) == 17

        assert len(atlas_target_data.detections) == 17
        assert atlas_target_data.badqual is None
        assert atlas_target_data.non_detections is None

    def test__no_average(self, atlas_target_data):

        atlas_lc = prepare_atlas_data(atlas_target_data, average_epochs=False)
        assert len(atlas_lc) == 17

        assert sum(atlas_lc["tag"] == "valid") == 12
        assert sum(atlas_lc["tag"] == "upperlim") == 5

    def test__average_detections(self, atlas_target_data):

        atlas_lc = prepare_atlas_data(atlas_target_data)

        assert len(atlas_lc) == 12
        assert sum(atlas_lc["tag"] == "valid") == 8
        assert sum(atlas_lc["tag"] == "upperlim") == 4

        assert atlas_lc["N_exp"].iloc[5] == 4
        assert np.isclose(atlas_lc["mjd"].iloc[5], 60002.5005)
        assert np.isclose(atlas_lc["mag"].iloc[5], 19.567, atol=0.01)

        assert atlas_lc["N_exp"].iloc[10] == 3
        assert np.isclose(atlas_lc["mjd"].iloc[10], 60004.51)


class Test__PrepareYseData:
    pass


class Test__DefaultLightcurveCompiler:
    def test__init(self):
        compiler = DefaultLightcurveCompiler()

        assert compiler.valid_tag == "valid"
        assert compiler.badqual_tag == "badqual"
        assert compiler.ulimit_tag == "upperlim"

        assert isinstance(compiler.ztf_broker_priority, tuple)
        assert compiler.ztf_broker_priority == ("fink", "alerce", "lasair", "antares")

    def test__target_with_no_data(self, mock_target):
        t_ref = Time(60020.0, format="mjd")

        assert len(mock_target.target_data) == 0

        compiler = DefaultLightcurveCompiler()
        compiled_lc = compiler(mock_target, t_ref)

        assert compiled_lc is None

    def test__normal_ztf_behaviour(self, mock_target, ztf_df):
        t_ref = Time(60020.0, format="mjd")

        fink_data = TargetData(lightcurve=ztf_df)
        mock_target.target_data["fink"] = fink_data

        compiler = DefaultLightcurveCompiler()
        compiled_lc = compiler(mock_target, t_ref)

        assert isinstance(compiled_lc, pd.DataFrame)
        assert len(compiled_lc) == 8

    def test__two_ztf_target_data(self, mock_target, ztf_df):
        t_ref = Time(60020.0, format="mjd")

        fink_data = TargetData(lightcurve=ztf_df.iloc[:-1])
        alerce_data = TargetData(lightcurve=ztf_df)
        mock_target.target_data["fink"] = fink_data
        mock_target.target_data["alerce"] = alerce_data

        compiler = DefaultLightcurveCompiler()
        compiled_lc = compiler(mock_target, t_ref)

        assert len(compiled_lc) == 7  # Choose FINK because ztf_priority.

    def test__normal_ztf_atlas_behaviour(self, mock_target, ztf_df, atlas_df):
        t_ref = Time(60020.0, format="mjd")

        fink_data = TargetData(lightcurve=ztf_df)
        atlas_data = TargetData(lightcurve=atlas_df)
        mock_target.target_data["fink"] = fink_data
        mock_target.target_data["atlas"] = atlas_data

        compiler = DefaultLightcurveCompiler()
        compiled_lc = compiler(mock_target, t_ref)

        assert len(compiled_lc) == 20  # 8 + 12
