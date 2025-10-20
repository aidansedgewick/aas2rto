import pytest
from pathlib import Path

import numpy as np

import pandas as pd

from astropy.table import Table, vstack

from aas2rto.exc import SettingLightcurveDirectlyWarning, UnknownPhotometryTagWarning
from aas2rto.target import Target
from aas2rto.target_data import TargetData


@pytest.fixture
def tdata():
    return TargetData()


@pytest.fixture
def updates_rows(det_rows, extra_det_rows):
    rows = det_rows[2:4] + extra_det_rows
    for row in rows:
        row[-2] = row[-2] + 2  #
    return rows


@pytest.fixture
def updates_pandas(updates_rows, lc_col_names):
    df = pd.DataFrame(updates_rows, columns=lc_col_names)
    assert len(df) == 6
    return df


@pytest.fixture
def updates_astropy(updates_rows, lc_col_names):
    tab = Table(rows=updates_rows, names=lc_col_names)
    assert len(tab) == 6
    return tab


class Test__TargetDataInit:
    def test__empty_init(self):
        # Act
        td = TargetData()

        # Assert
        assert set(td.valid_tags) == set(["valid", "detection", "det"])
        assert set(td.badqual_tags) == set(["badqual", "badquality", "dubious"])
        assert set(td.nondet_tags) == set(["nondet", "upperlim", "ulimit", "ulim"])

        assert td.lightcurve is None
        assert td.detections is None
        assert td.badqual is None
        assert td.non_detections is None

        assert isinstance(td.meta, dict)
        assert len(td.meta) == 0

        assert isinstance(td.probabilities, dict)
        assert len(td.probabilities) == 0

        assert isinstance(td.parameters, dict)
        assert len(td.parameters) == 0

        assert isinstance(td.cutouts, dict)
        assert len(td.parameters) == 0


class Test__AddLightcurvePandas:
    def test__default_behaviour(self, tdata: TargetData, lc_pandas: pd.DataFrame):
        # Act
        tdata.add_lightcurve(lc_pandas)

        # Assert
        assert isinstance(tdata.lightcurve, pd.DataFrame)
        assert len(tdata.lightcurve) == 14

        assert isinstance(tdata.detections, pd.DataFrame)
        assert len(tdata.detections) == 6
        assert np.isclose(tdata.detections["mjd"].iloc[0], 60004.0)

        assert isinstance(tdata.badqual, pd.DataFrame)
        assert len(tdata.badqual) == 4
        assert np.isclose(tdata.badqual["mjd"].iloc[0], 60002.0)

        assert isinstance(tdata.non_detections, pd.DataFrame)
        assert len(tdata.non_detections) == 4
        assert np.isclose(tdata.non_detections["mjd"].iloc[0], 60000.0)

    def test__det_with_badqual(self, tdata: TargetData, lc_pandas: pd.DataFrame):
        # Act
        tdata.add_lightcurve(lc_pandas, include_badqual=True)

        # Assert
        assert isinstance(tdata.lightcurve, pd.DataFrame)
        assert len(tdata.lightcurve) == 14

        assert isinstance(tdata.detections, pd.DataFrame)
        assert len(tdata.detections) == 10
        assert np.isclose(tdata.detections["mjd"].iloc[0], 60002.0)

        assert isinstance(tdata.badqual, pd.DataFrame)
        assert len(tdata.badqual) == 0

        assert isinstance(tdata.non_detections, pd.DataFrame)
        assert len(tdata.non_detections) == 4
        assert np.isclose(tdata.non_detections["mjd"].iloc[0], 60000.0)

    def test__new_tags(self, lc_pandas: pd.DataFrame):
        # Arrange
        lc = lc_pandas
        lc.loc[lc["tag"] == "valid", "tag"] = "ok"
        lc.loc[lc["tag"] == "badqual", "tag"] = "bq"
        lc.loc[lc["tag"] == "ulim", "tag"] = "ndet"
        td = TargetData(valid_tags="ok", badqual_tags="bq", nondet_tags="ndet")

        # Act
        td.add_lightcurve(lc)

        # Assert
        assert isinstance(td.lightcurve, pd.DataFrame)
        assert len(td.lightcurve) == 14

        assert isinstance(td.detections, pd.DataFrame)
        assert len(td.detections) == 6
        assert np.isclose(td.detections["mjd"].iloc[0], 60004.0)

        assert isinstance(td.badqual, pd.DataFrame)
        assert len(td.badqual) == 4
        assert np.isclose(td.badqual["mjd"].iloc[0], 60002.0)

        assert isinstance(td.non_detections, pd.DataFrame)
        assert len(td.non_detections) == 4
        assert np.isclose(td.non_detections["mjd"].iloc[0], 60000.0)

    def test__unknown_tags_warns(self, tdata: TargetData, lc_pandas: pd.DataFrame):
        # Arrange
        lc_pandas.loc[-2:, "tag"] = "ok"

        # Act
        with pytest.warns(UnknownPhotometryTagWarning):
            tdata.add_lightcurve(lc_pandas)

    def test__no_tag_no_fail(self, tdata: TargetData, lc_pandas: pd.DataFrame):
        # Arrange
        lc_pandas.drop("tag", axis=1, inplace=True)

        # Act
        tdata.add_lightcurve(lc_pandas)

        # Assert
        assert len(tdata.lightcurve) == 14
        assert tdata.detections is None
        assert tdata.badqual is None
        assert tdata.non_detections is None

    def test__setting_directly_warns(self, tdata: TargetData, lc_pandas: pd.DataFrame):
        # Act
        with pytest.warns(SettingLightcurveDirectlyWarning):
            tdata.lightcurve = lc_pandas

        # Assert
        assert len(tdata.lightcurve) == 14
        assert tdata.detections is None
        assert tdata.badqual is None
        assert tdata.non_detections is None

    def test__remove_method_works(self, tdata_lc_pandas: TargetData):
        # Act
        tdata_lc_pandas.remove_lightcurve()

        # Assert
        assert tdata_lc_pandas.lightcurve is None
        assert tdata_lc_pandas.detections is None
        assert tdata_lc_pandas.badqual is None
        assert tdata_lc_pandas.non_detections is None


class Test__AddLightcurveAstropy:
    def test__default_behaviour(self, tdata: TargetData, lc_astropy: Table):
        # Act
        tdata.add_lightcurve(lc_astropy)

        # Assert
        assert isinstance(tdata.lightcurve, Table)
        assert len(tdata.lightcurve) == 14

        assert isinstance(tdata.detections, Table)
        assert len(tdata.detections) == 6
        assert np.isclose(tdata.detections["mjd"][0], 60004.0)

        assert isinstance(tdata.badqual, Table)
        assert len(tdata.badqual) == 4
        assert np.isclose(tdata.badqual["mjd"][0], 60002.0)

        assert isinstance(tdata.non_detections, Table)
        assert len(tdata.non_detections) == 4
        assert np.isclose(tdata.non_detections["mjd"][0], 60000.0)

    def test__new_tags(self, lc_astropy: Table):
        # Arrange
        lc = lc_astropy
        lc["tag"][lc["tag"] == "valid"] = "ok"
        lc["tag"][lc["tag"] == "badqual"] = "bq"
        lc["tag"][lc["tag"] == "ulim"] = "ndet"
        td = TargetData(valid_tags="ok", badqual_tags="bq", nondet_tags="ndet")

        # Act
        td.add_lightcurve(lc)

        # Assert
        assert isinstance(td.lightcurve, Table)
        assert len(td.lightcurve) == 14

        assert isinstance(td.detections, Table)
        assert len(td.detections) == 6
        assert np.isclose(td.detections["mjd"][0], 60004.0)

        assert isinstance(td.badqual, Table)
        assert len(td.badqual) == 4
        assert np.isclose(td.badqual["mjd"][0], 60002.0)

        assert isinstance(td.non_detections, Table)
        assert len(td.non_detections) == 4
        assert np.isclose(td.non_detections["mjd"][0], 60000.0)

    def test__setting_directly_warns(self, tdata: TargetData, lc_astropy: Table):
        # Act
        with pytest.warns(SettingLightcurveDirectlyWarning):
            tdata.lightcurve = lc_astropy

        # Assert
        assert len(tdata.lightcurve) == 14
        assert tdata.detections is None
        assert tdata.badqual is None
        assert tdata.non_detections is None

    def test__unknown_tags_warns(self, tdata: TargetData, lc_astropy: Table):
        # Arrange
        lc_astropy["tag"][-2:] = "ok"
        # Act
        with pytest.warns(UnknownPhotometryTagWarning):
            tdata.add_lightcurve(lc_astropy)

    def test__no_tag_no_fail(self, tdata: TargetData, lc_astropy: Table):
        # Arrange
        lc_astropy.remove_column("tag")

        # Act
        tdata.add_lightcurve(lc_astropy)

        # Assert
        assert len(tdata.lightcurve) == 14
        assert tdata.detections is None
        assert tdata.badqual is None
        assert tdata.non_detections is None

    def test__remove_method_works(self, tdata_lc_astropy: TargetData):
        # Act
        tdata_lc_astropy.remove_lightcurve()

        # Assert
        assert tdata_lc_astropy.lightcurve is None
        assert tdata_lc_astropy.detections is None
        assert tdata_lc_astropy.badqual is None
        assert tdata_lc_astropy.non_detections is None


# class Test__UpdatesLCExactPandas:
#     def test__no_duplicate_values(
#         self, tdata: TargetData, lc_pandas: pd.DataFrame, extra_det_pandas: pd.DataFrame
#     ):
#         # Arrange
#         tdata.add_lightcurve(lc_pandas)

#         # Act
#         tdata.update_lc_exact_match(extra_det_pandas, column="obsid")

#         # Assert
#         assert len(tdata.lightcurve) == 18  # 14 + 4 new ones

#         # check all NaNs are preserved
#         assert len(tdata.non_detections) == 4
#         expected_ndet_mjds = 60000.0 + np.array([0.0, 0.1, 1.0, 1.1])
#         assert np.allclose(tdata.non_detections["mjd"], expected_ndet_mjds)

#         assert len(tdata.badqual) == 4
#         assert len(tdata.detections) == 10

#     def test__dup_values_keep_updates(
#         self,
#         tdata: TargetData,
#         lc_pandas: pd.DataFrame,
#         det_rows: list,
#         extra_det_rows: list,
#     ):
#         # Arrange
#         tdata.add_lightcurve(lc_pandas)
#         updates_rows = det_rows[2:4] + extra_det_rows
#         updates = pd.DataFrame(updates_rows, columns=lc_pandas.columns)
#         updates.loc[:, "band"] = updates.loc[:, "band"] + 2  # check mod values kept.
#         assert len(updates) == 6

#         # Act
#         tdata.update_lc_exact_match(
#             updates,
#             column="obsid",
#             keep_updates=True,  # Make sure we keep 'updates' rows where clash.
#         )

#         # Assert
#         assert len(tdata.lightcurve) == 18  # Duplicates removed

#         assert len(tdata.detections) == 10  # same as before, ok...

#         #                              o n o o n n u u u u
#         expected_band_vals = np.array([1, 2, 3, 4, 1, 2, 3, 4, 3, 4])
#         assert np.allclose(tdata.detections["band"], expected_band_vals)

#         assert len(tdata.non_detections) == 4
#         assert len(tdata.badqual) == 4

#     def test__dup_values_keep_original(
#         self,
#         tdata: TargetData,
#         lc_pandas: pd.DataFrame,
#         det_rows: list,
#         extra_det_rows: list,
#     ):
#         # Arrange
#         tdata.add_lightcurve(lc_pandas)
#         updates_rows = det_rows[2:4] + extra_det_rows
#         updates = pd.DataFrame(updates_rows, columns=lc_pandas.columns)
#         updates.loc[:, "band"] = updates.loc[:, "band"] + 2  # check mod values kept.
#         assert len(updates) == 6

#         # Act
#         tdata.update_lc_exact_match(
#             updates,
#             column="obsid",
#             keep_updates=False,  # Keep existing lc where rows clash.
#         )

#         # Assert
#         assert len(tdata.lightcurve) == 18  # Duplicates removed

#         assert len(tdata.detections) == 10  # same as before, ok...

#         #                              o  o  o  o  o  o  u  u  u  u
#         expected_band_vals = np.array([1, 2, 1, 2, 1, 2, 3, 4, 3, 4])
#         assert np.allclose(tdata.detections["band"], expected_band_vals)

#         assert len(tdata.non_detections) == 4
#         assert len(tdata.badqual) == 4

#     def test__no_existing_lightcurve(
#         self, tdata: TargetData, extra_det_pandas: pd.DataFrame
#     ):
#         # Act
#         tdata.update_lc_exact_match(extra_det_pandas, column="obsid")

#         # Assert
#         assert len(tdata.lightcurve) == 4
#         assert len(tdata.detections) == 4
#         assert tdata.badqual.empty
#         assert tdata.non_detections.empty

#     def test__type_mismatch_fails(
#         self,
#         tdata: TargetData,
#         lc_pandas: pd.DataFrame,
#         extra_det_astropy: Table,  # Not a typo!!!
#     ):
#         # Arrange
#         tdata.add_lightcurve(lc_pandas)

#         # Act
#         with pytest.raises(TypeError):
#             tdata.update_lc_exact_match(extra_det_astropy, column="obsid")


# class Test__UpdateLCExactAstropy:
#     def test__no_duplicate_values(
#         self, tdata: TargetData, lc_astropy: Table, extra_det_astropy: Table
#     ):
#         # Arrange
#         tdata.add_lightcurve(lc_astropy)

#         # Act
#         tdata.update_lc_exact_match(
#             extra_det_astropy, column="obsid", sort_column="mjd"
#         )

#         # Assert
#         assert len(tdata.lightcurve) == 18  # 14 + 4 new ones

#         # check all NaNs are preserved
#         assert len(tdata.non_detections) == 4
#         expected_ndet_mjds = 60000.0 + np.array([0.0, 0.1, 1.0, 1.1])
#         assert np.allclose(tdata.non_detections["mjd"], expected_ndet_mjds)

#         assert len(tdata.badqual) == 4
#         assert len(tdata.detections) == 10

#     def test__dup_values_keep_updates(
#         self, tdata: TargetData, lc_astropy: Table, updates_astropy: Table
#     ):
#         # Arrange
#         tdata.add_lightcurve(lc_astropy)

#         # Act
#         tdata.update_lc_exact_match(
#             updates_astropy,
#             column="obsid",
#             sort_column="mjd",
#             keep_updates=True,  # Make sure we keep 'updates' rows where clash.
#         )

#         # Assert
#         assert len(tdata.lightcurve) == 18  # Duplicates removed

#         assert len(tdata.detections) == 10  # same as before, ok...

#         #                              o  o  n  n  o  o  u  u  u  u
#         expected_band_vals = np.array([1, 2, 3, 4, 1, 2, 3, 4, 3, 4])
#         assert np.allclose(tdata.detections["band"], expected_band_vals)

#         assert len(tdata.non_detections) == 4
#         assert len(tdata.badqual) == 4

#     def test__keep_updates_with_ignore(
#         self, tdata: TargetData, lc_astropy: Table, updates_astropy: Table
#     ):
#         # Arrange
#         nan_mask = np.isnan(lc_astropy["obsid"])
#         lc_astropy["obsid"][nan_mask] = -1
#         tdata.add_lightcurve(lc_astropy)

#         # Act
#         tdata.update_lc_exact_match(
#             updates_astropy,
#             column="obsid",
#             sort_column="mjd",
#             keep_updates=True,  # Make sure we keep 'updates' rows where clash.
#             ignore_values=[-1],
#         )

#         # Assert
#         assert len(tdata.lightcurve) == 18  # Duplicates removed

#         assert len(tdata.detections) == 10  # same as before, ok...

#         #                              o  o  n  n  o  o  u  u  u  u
#         expected_band_vals = np.array([1, 2, 3, 4, 1, 2, 3, 4, 3, 4])
#         assert np.allclose(tdata.detections["band"], expected_band_vals)

#         assert len(tdata.non_detections) == 4
#         assert len(tdata.badqual) == 4

#     def test__dup_values_keep_original(
#         self, tdata: TargetData, lc_astropy: Table, updates_astropy: Table
#     ):
#         # Arrange
#         tdata.add_lightcurve(lc_astropy)

#         # Act
#         tdata.update_lc_exact_match(
#             updates_astropy,
#             column="obsid",
#             sort_column="mjd",
#             keep_updates=False,  # Keep existing lc where rows clash.
#         )

#         # Assert
#         assert len(tdata.lightcurve) == 18  # Duplicates removed

#         assert len(tdata.detections) == 10  # same as before, ok...

#         #                              o  o  o  o  o  o  u  u  u  u
#         expected_band_vals = np.array([1, 2, 1, 2, 1, 2, 3, 4, 3, 4])
#         assert np.allclose(tdata.detections["band"], expected_band_vals)

#         assert len(tdata.non_detections) == 4
#         assert len(tdata.badqual) == 4

#     def test__keep_original_with_ignore(
#         self, tdata: TargetData, lc_astropy: Table, updates_astropy: Table
#     ):
#         # Arrange
#         nan_mask = np.isnan(lc_astropy["obsid"])
#         lc_astropy["obsid"][nan_mask] = -1
#         tdata.add_lightcurve(lc_astropy)

#         # Act
#         tdata.update_lc_exact_match(
#             updates_astropy,
#             column="obsid",
#             sort_column="mjd",
#             keep_updates=False,  # Keep existing lc where rows clash.
#         )

#         # Assert
#         assert len(tdata.lightcurve) == 18  # Duplicates removed

#         assert len(tdata.detections) == 10  # same as before, ok...

#         #                              o  o  o  o  o  o  u  u  u  u
#         expected_band_vals = np.array([1, 2, 1, 2, 1, 2, 3, 4, 3, 4])
#         assert np.allclose(tdata.detections["band"], expected_band_vals)

#         assert len(tdata.non_detections) == 4
#         assert len(tdata.badqual) == 4

#     def test__no_existing_lightcurve(self, tdata: TargetData, extra_det_astropy: Table):
#         # Act
#         tdata.update_lc_exact_match(extra_det_astropy, column="obsid")

#         # Assert
#         assert len(tdata.lightcurve) == 4
#         assert len(tdata.detections) == 4
#         assert len(tdata.badqual) == 0
#         assert len(tdata.non_detections) == 0

#     def test__type_mismatch_fails(
#         self,
#         tdata: TargetData,
#         lc_astropy: Table,
#         extra_det_pandas: pd.DataFrame,  # Not a typo!!!
#     ):
#         # Arrange
#         tdata.add_lightcurve(lc_astropy)

#         # Act
#         with pytest.raises(TypeError):
#             tdata.update_lc_exact_match(extra_det_pandas, column="obsid")


# class Test__UpdateLCContinuousPandas:
#     def test__no_conflits(
#         self, tdata: TargetData, lc_pandas: pd.DataFrame, extra_det_pandas: pd.DataFrame
#     ):
#         # Arrange
#         tdata.add_lightcurve(lc_pandas)

#         # Act
#         tdata.update_lc_continuous(extra_det_pandas)

#         # Assert
#         assert len(tdata.lightcurve) == 18  # 14 + 4 new ones

#         # check all NaNs are preserved
#         assert len(tdata.non_detections) == 4
#         expected_ndet_mjds = 60000.0 + np.array([0.0, 0.1, 1.0, 1.1])
#         assert np.allclose(tdata.non_detections["mjd"], expected_ndet_mjds)

#         assert len(tdata.badqual) == 4
#         assert len(tdata.detections) == 10

#     def test__conflicting_rows_resolved(
#         self, tdata: TargetData, lc_pandas: pd.DataFrame, updates_pandas: pd.DataFrame
#     ):
#         # Arrange
#         tdata.add_lightcurve(lc_pandas)

#         # Act
#         tdata.update_lc_continuous(updates_pandas)

#         # Assert
#         # Removed first updates        o  o  o  o  o  o  u  u  u  u
#         expected_band_vals = np.array([1, 2, 1, 2, 1, 2, 3, 4, 3, 4])
#         assert np.allclose(tdata.detections["band"], expected_band_vals)

#         assert len(tdata.non_detections) == 4
#         assert len(tdata.badqual) == 4

#     def test__no_existing_lightcurve(
#         self, tdata: TargetData, extra_det_pandas: pd.DataFrame
#     ):
#         # Act
#         tdata.update_lc_continuous(extra_det_pandas)

#         # Assert
#         assert len(tdata.lightcurve) == 4
#         assert len(tdata.detections) == 4
#         assert tdata.badqual.empty
#         assert tdata.non_detections.empty

#     def test__type_mismatch_fails(
#         self,
#         tdata: TargetData,
#         lc_pandas: pd.DataFrame,
#         extra_det_astropy: Table,  # Not a typo!!!
#     ):
#         # Arrange
#         tdata.add_lightcurve(lc_pandas)

#         # Act
#         with pytest.raises(TypeError):
#             tdata.update_lc_continuous(extra_det_astropy)


# class Test__UpdateLCContinuousAstropy:
#     def test__no_conflicts(
#         self, tdata: TargetData, lc_astropy: Table, extra_det_astropy: Table
#     ):
#         # Arrange
#         tdata.add_lightcurve(lc_astropy)

#         # Act
#         tdata.update_lc_continuous(extra_det_astropy)

#         # Assert
#         assert len(tdata.lightcurve) == 18  # 14 + 4 new ones

#         # check all NaNs are preserved
#         assert len(tdata.non_detections) == 4
#         expected_ndet_mjds = 60000.0 + np.array([0.0, 0.1, 1.0, 1.1])
#         assert np.allclose(tdata.non_detections["mjd"], expected_ndet_mjds)

#         assert len(tdata.badqual) == 4
#         assert len(tdata.detections) == 10

#     def test__conflicting_rows_resolved(
#         self, tdata: TargetData, lc_astropy: Table, updates_astropy: Table
#     ):
#         # Arrange
#         tdata.add_lightcurve(lc_astropy)

#         # Act
#         tdata.update_lc_continuous(updates_astropy)

#         # Assert
#         # Removed first updates        o  o  o  o  o  o  u  u  u  u
#         expected_band_vals = np.array([1, 2, 1, 2, 1, 2, 3, 4, 3, 4])
#         assert np.allclose(tdata.detections["band"], expected_band_vals)

#         assert len(tdata.non_detections) == 4
#         assert len(tdata.badqual) == 4

#     def test__no_existing_lightcurve(self, tdata: TargetData, extra_det_astropy: Table):
#         # Act
#         tdata.update_lc_continuous(extra_det_astropy)

#         # Assert
#         assert len(tdata.lightcurve) == 4
#         assert len(tdata.detections) == 4
#         assert len(tdata.badqual) == 0
#         assert len(tdata.non_detections) == 0

#     def test__type_mismatch_fails(
#         self,
#         tdata: TargetData,
#         lc_astropy: Table,
#         extra_det_pandas: pd.DataFrame,  # Not a typo!!!
#     ):
#         # Arrange
#         tdata.add_lightcurve(lc_astropy)

#         # Act
#         with pytest.raises(TypeError):
#             tdata.update_lc_continuous(extra_det_pandas)
