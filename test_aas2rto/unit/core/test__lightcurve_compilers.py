import pytest
from typing import List

import numpy as np

import pandas as pd

from astropy.time import Time

from aas2rto.lightcurve_compilers import (
    DefaultLightcurveCompiler,
    prepare_atlas_data,
    prepare_lsst_data,
    prepare_yse_data,
    prepare_ztf_data,
)
from aas2rto.target import Target
from aas2rto.target_data import TargetData


class Test__PrepZTF:
    def test__prep_ztf(self, ztf_td: TargetData):
        # Act
        processed_ztf = prepare_ztf_data(ztf_td)

        # Assert
        assert isinstance(processed_ztf, pd.DataFrame)
        exp_columns = "mjd mag magerr diffmaglim band tag alert_id source".split()
        assert set(processed_ztf.columns) == set(exp_columns)


class Test__PrepLSST:
    def test__prep_lsst(self, lsst_td: TargetData):
        # Act
        processed_lsst = prepare_lsst_data(lsst_td)

        # Assert
        assert isinstance(processed_lsst, pd.DataFrame)
        exp_columns = "mjd mag magerr diffmaglim band tag alert_id source".split()
        assert set(processed_lsst.columns) == set(exp_columns)

    def test__prep_lsst_tag(self, lsst_lc: TargetData):
        # Arrange
        tag_data = ["badqual", "badqual", "badqual", "valid", "valid", "valid"]
        lsst_lc.loc[:, "tag"] = tag_data
        td = TargetData(lightcurve=lsst_lc)

        # Act
        processed_lsst = prepare_lsst_data(td)

        # Assert
        print(processed_lsst[["tag", "alert_id", "mjd"]])
        assert processed_lsst["tag"].iloc[2] == "badqual"
        assert processed_lsst["tag"].iloc[3] == "valid"


class Test__PrepAtlas:
    def test__prep_atlas(self, atlas_td: TargetData):
        # Act
        result = prepare_atlas_data(atlas_td)

        # Assert
        assert isinstance(result, pd.DataFrame)
        exp_columns = "mjd mag magerr diffmaglim tag band N_exp source".split()
        set(result.columns) == set(exp_columns)
        assert len(result) == 5

        assert np.isclose(result["mjd"].iloc[0] - 60000.0, 0.005)
        assert result["N_exp"].iloc[0] == 2
        assert result["tag"].iloc[0] == "upperlim"
        assert result["band"].iloc[0] == "atlaso"

        assert np.isclose(result["mjd"].iloc[1] - 60000.0, 1.00)
        assert result["N_exp"].iloc[1] == 1
        assert result["tag"].iloc[1] == "valid"
        assert result["band"].iloc[1] == "atlasc"

        assert np.isclose(result["mjd"].iloc[2] - 60000.0, 1.005)
        assert result["N_exp"].iloc[2] == 4
        assert result["tag"].iloc[2] == "valid"
        assert result["band"].iloc[2] == "atlaso"

        assert np.isclose(result["mjd"].iloc[3] - 60000.0, 2.00)
        assert result["N_exp"].iloc[3] == 2
        assert result["tag"].iloc[3] == "valid"
        assert result["band"].iloc[3] == "atlasc"

        assert np.isclose(result["mjd"].iloc[4] - 60000.0, 3.00)
        assert result["N_exp"].iloc[4] == 3
        assert result["tag"].iloc[4] == "valid"
        assert result["band"].iloc[4] == "atlasc"
        assert np.isclose(result["mag"].iloc[4], 18.0, atol=0.1)  # weights ok.

    def test__no_average(self, atlas_td: TargetData):
        # Act
        result = prepare_atlas_data(atlas_td, average_epochs=False)

        # Assert
        assert len(result) == 12


class Test__PrepYSE:
    def test__prep_yse(self, yse_td: TargetData):
        # Act
        result = prepare_yse_data(yse_td, use_all_sources=True)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        exp_columns = "mjd mag magerr tag band source".split()
        assert set(result.columns) == set(exp_columns)

        assert result["band"].iloc[0] == "ps1::g"  # column correctly mapped...
        assert result["band"].iloc[1] == "ps1::r"
        assert result["band"].iloc[2] == "uvot::uvw1"  #  3rd row dq 'BAD' is ignored.

    def test__not_all_sources(self, yse_td: TargetData):
        # Act
        result = prepare_yse_data(yse_td, use_all_sources=False)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

        assert result["band"].iloc[0] == "ps1::g"
        assert result["band"].iloc[1] == "ps1::r"
        assert result["band"].iloc[2] == "ps1::g"  #  3rd row dq 'BAD' has been ignored.

    def test__additional_sources(self, yse_td: TargetData):
        # Act
        result = prepare_yse_data(
            yse_td, use_all_sources=False, additional_sources="swift"
        )

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

        assert result["band"].iloc[0] == "ps1::g"
        assert result["band"].iloc[1] == "ps1::r"
        assert result["band"].iloc[2] == "uvot::uvw1"  #  3rd row dq 'BAD' is ignored.


class Test__DefaultLCCompiler:

    def test__class_init(self):
        # Act
        lc_compiler = DefaultLightcurveCompiler()

        # Assert
        assert hasattr(lc_compiler, "__name__")

    def test__only_ztf(
        self,
        lc_compiler: DefaultLightcurveCompiler,
        basic_target: Target,
        ztf_td: TargetData,
        t_fixed: Time,
    ):
        # Arrange
        basic_target.target_data["ztf"] = ztf_td

        # Act
        result = lc_compiler(basic_target, t_ref=t_fixed)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 14
        assert set(result["source"].values) == set(["ztf"])

    def test__ztf_priority(
        self,
        lc_compiler: DefaultLightcurveCompiler,
        basic_target: Target,
        ztf_td: TargetData,
        t_fixed: Time,
    ):
        # Arrange
        basic_target.target_data["alerce_ztf"] = ztf_td

        # Act
        result = lc_compiler(basic_target, t_ref=t_fixed)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 14
        assert set(result["source"].values) == set(["ztf"])

    def test__only_atlas(
        self,
        lc_compiler: DefaultLightcurveCompiler,
        basic_target: Target,
        atlas_td: TargetData,
        t_fixed: Time,
    ):
        # Arrange
        basic_target.target_data["atlas"] = atlas_td

        # Act
        result = lc_compiler(basic_target, t_ref=t_fixed)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert set(result["source"].values) == set(["atlas"])

    def test__atlas_no_average(
        self,
        basic_target: Target,
        atlas_td: TargetData,
        t_fixed: Time,
    ):
        # Arrange
        lcc = DefaultLightcurveCompiler(atlas_average_epochs=False)
        basic_target.target_data["atlas"] = atlas_td

        # Act
        result = lcc(basic_target, t_ref=t_fixed)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 12
        assert set(result["source"].values) == set(["atlas"])

    def test__only_yse(
        self,
        lc_compiler: DefaultLightcurveCompiler,
        basic_target: Target,
        yse_td: TargetData,
        t_fixed: Time,
    ):
        # Arrange
        basic_target.target_data["yse"] = yse_td

        # Act
        result = lc_compiler(basic_target, t_ref=t_fixed)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # dq BAD in 3rd row ignored
        assert set(result["source"].values) == set(["yse", "swift", "unknown"])

    def test__all_data(
        self,
        lc_compiler: DefaultLightcurveCompiler,
        basic_target: Target,
        ztf_td: TargetData,
        atlas_td: TargetData,
        yse_td: TargetData,
        t_fixed: Time,
    ):
        # Arrange
        basic_target.target_data["ztf"] = ztf_td
        basic_target.target_data["atlas"] = atlas_td
        basic_target.target_data["yse"] = yse_td

        # Act
        result = lc_compiler(basic_target, t_ref=t_fixed)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24  # 14 ztf + 5 atlas ave + 5 yse (exl. bad)
        assert set(result["source"].values) == set(
            ["yse", "swift", "atlas", "ztf", "unknown"]
        )
