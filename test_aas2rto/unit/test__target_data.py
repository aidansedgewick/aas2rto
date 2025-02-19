import pytest

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time

from astroplan import FixedTarget, Observer

from aas2rto.exc import (
    MissingDateError,
    UnknownObservatoryWarning,
    SettingLightcurveDirectlyWarning,
)
from aas2rto.obs_info import ObservatoryInfo
from aas2rto.target import Target
from aas2rto.target_data import TargetData


@pytest.fixture
def mock_lc_rows():
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
def mock_lc(mock_lc_rows):
    return pd.DataFrame(mock_lc_rows, columns="mjd obsId mag magerr tag".split())


@pytest.fixture
def mock_lc_astropy(mock_lc_rows):
    return Table(rows=mock_lc_rows, names="mjd obsId mag magerr tag".split())


@pytest.fixture
def mock_lc_updates_rows():
    return [
        (60010.0, 11, 19.9, 0.05, "valid"),
        (60011.0, 12, 19.8, 0.05, "valid"),
        (60012.0, 13, 19.7, 0.2, "badqual"),
    ]


@pytest.fixture
def mock_lc_updates(mock_lc_updates_rows):
    return pd.DataFrame(
        mock_lc_updates_rows, columns="mjd obsId mag magerr tag".split()
    )


@pytest.fixture
def mock_lc_updates_astropy(mock_lc_updates_rows):
    return Table(rows=mock_lc_updates_rows, names="mjd obsId mag magerr tag".split())


@pytest.fixture
def mock_target():
    return Target("T101", ra=45.0, dec=60.0)


@pytest.fixture
def test_observer():
    location = EarthLocation(lat=55.6802, lon=12.5724, height=0.0)
    return Observer(location, name="ucph")


def basic_lc_compiler(target: Target, t_ref: Time):
    lc = target.target_data["ztf"].detections.copy()
    lc.loc[:, "flux"] = 3631.0 * 10 ** (-0.4 * lc["mag"])
    lc.loc[:, "fluxerr"] = lc["flux"] * lc["magerr"]
    lc.loc[:, "band"] = "ztf-w"
    return lc


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

        assert set(td.valid_tags) == set(["valid", "detection", "det"])
        assert set(td.badqual_tags) == set(["badquality", "badqual", "dubious"])
        assert set(td.nondet_tags) == set(["upperlim", "nondet"])

    def test__change_tags(self):
        td = TargetData(valid_tags=(["valid"]))
        assert set(td.valid_tags) == set(["valid"])

    def test__init_with_lc_no_tag(self, mock_lc):
        mock_lc.drop("tag", axis=1, inplace=True)
        assert not "tag" in mock_lc.columns

        td = TargetData(lightcurve=mock_lc)
        assert len(td.lightcurve) == 10
        assert len(td.detections) == 10
        assert td.badqual is None
        assert td.non_detections is None

    def test__init_with_tag(self, mock_lc):
        td = TargetData(lightcurve=mock_lc, include_badqual=False)  # default False

        assert len(td.lightcurve) == 10
        assert len(td.detections) == 3
        assert len(td.badqual) == 3
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([8, 9, 10])
        assert set(td.badqual["obsId"]) == set([5, 6, 7])
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__init_badqual_true(self, mock_lc):
        td = TargetData(lightcurve=mock_lc, include_badqual=True)

        assert len(td.lightcurve) == 10
        assert len(td.detections) == 6
        assert len(td.badqual) == 0
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([5, 6, 7, 8, 9, 10])
        assert set(td.badqual["obsId"]) == set()  # Test the column name is there.
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__no_date_column_raises_error(self, mock_lc):
        mock_lc.drop("mjd", axis=1, inplace=True)

        td = TargetData()
        with pytest.raises(MissingDateError):
            td.add_lightcurve(mock_lc)

    def test__set_lc_directly_raises_warning(self, mock_lc):
        td = TargetData()
        with pytest.warns(SettingLightcurveDirectlyWarning):
            td.lightcurve = mock_lc

        assert isinstance(td.lightcurve, pd.DataFrame)
        assert len(td.lightcurve) == 10

    def test__empty_cutouts(self):
        td = TargetData()
        result = td.empty_cutouts()
        assert isinstance(result, dict)
        assert len(result) == 0


class Test__IntegratingUpdates:
    def test__integrate_equality_no_lightcurve(self, mock_lc_updates):
        td = TargetData()
        assert td.lightcurve is None

        result = td.integrate_lightcurve_updates_equality(
            mock_lc_updates, column="obsId"
        )
        assert len(result) == 3
        assert len(td.lightcurve) == 3
        assert len(td.detections) == 2
        assert len(td.badqual) == 1

    def test__integrate_equality(self, mock_lc, mock_lc_updates):
        td = TargetData(lightcurve=mock_lc)

        result = td.integrate_lightcurve_updates_equality(
            mock_lc_updates, column="obsId"
        )
        assert len(result) == 13
        assert set(result["obsId"]) == set(range(1, 14))

    def test__integrate_equality_keep_updated(self, mock_lc, mock_lc_updates):
        updates = pd.concat([mock_lc[7:], mock_lc_updates], ignore_index=True)
        assert len(updates) == 6
        updates["source"] = "updates"
        mock_lc["source"] = "original"

        td = TargetData(lightcurve=mock_lc)
        result = td.integrate_lightcurve_updates_equality(updates, column="obsId")
        assert len(result) == 13
        assert all(result.iloc[:7].source == "original")
        assert all(result.iloc[7:].source == "updates")
        assert len(result[result["source"] == "original"]) == 7
        assert len(result[result["source"] == "updates"]) == 6

    def test__integrate_equality_keep_original(self, mock_lc, mock_lc_updates):
        updates = pd.concat([mock_lc[7:], mock_lc_updates], ignore_index=True)
        assert len(updates) == 6
        updates["source"] = "updates"
        mock_lc["source"] = "original"

        td = TargetData(lightcurve=mock_lc)
        result = td.integrate_lightcurve_updates_equality(
            updates, column="obsId", keep_updates=False
        )
        assert len(result) == 13
        assert all(result.iloc[:10].source == "original")
        assert all(result.iloc[10:].source == "updates")
        assert len(result[result["source"] == "original"]) == 10
        assert len(result[result["source"] == "updates"]) == 3

    def test__integrate_updates(self, mock_lc, mock_lc_updates):
        td = TargetData(lightcurve=mock_lc)
        td.integrate_lightcurve_updates_equality(mock_lc_updates, column="obsId")

        assert len(td.lightcurve) == 13
        assert len(td.detections) == 5
        assert len(td.badqual) == 4
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([8, 9, 10, 11, 12])
        assert set(td.badqual["obsId"]) == set([5, 6, 7, 13])
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__integrate_updates_badqual_true(self, mock_lc, mock_lc_updates):
        td = TargetData(lightcurve=mock_lc)
        td.integrate_lightcurve_updates_equality(
            mock_lc_updates, column="obsId", include_badqual=True
        )

        assert len(td.lightcurve) == 13
        assert len(td.detections) == 9
        assert len(td.badqual) == 0
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([5, 6, 7, 8, 9, 10, 11, 12, 13])
        assert set(td.badqual["obsId"]) == set()  # Test the column name is there.
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__integrate_updates_drop_repeated_rows(self, mock_lc, mock_lc_updates):
        mock_lc.loc[mock_lc["tag"] == "upperlim", "obsId"] = -1
        mock_lc.loc[mock_lc["tag"] == "nondet", "obsId"] = -1

        assert all(mock_lc["obsId"].iloc[:3].values == -1)

        td = TargetData(lightcurve=mock_lc)
        assert len(td.non_detections) == 4
        assert len(td.badqual) == 3
        assert len(td.detections) == 3

        td.integrate_lightcurve_updates_equality(mock_lc_updates, column="obsId")

        assert len(td.lightcurve) == 10  # 4 rows with obsId==-1 -- drop all but last!
        assert len(td.non_detections) == 1
        assert np.isclose(td.non_detections["mjd"].iloc[0], 60003.0)
        assert len(td.badqual) == 4
        assert len(td.detections) == 5

    def test__integrate_updates_keep_repeated_rows(self, mock_lc, mock_lc_updates):
        mock_lc.loc[mock_lc["tag"] == "upperlim", "obsId"] = -1
        mock_lc.loc[mock_lc["tag"] == "nondet", "obsId"] = -1

        assert all(mock_lc["obsId"].iloc[:3].values == -1)

        td = TargetData(lightcurve=mock_lc)
        assert len(td.non_detections) == 4
        assert len(td.badqual) == 3
        assert len(td.detections) == 3

        td.integrate_lightcurve_updates_equality(
            mock_lc_updates, column="obsId", ignore_values=[-1]
        )

        assert len(td.lightcurve) == 13
        assert len(td.non_detections) == 4
        assert np.isclose(td.non_detections["mjd"].iloc[0], 60000.0)
        assert np.isclose(td.non_detections["mjd"].iloc[3], 60003.0)
        assert len(td.badqual) == 4
        assert len(td.detections) == 5


class Test__TargetDataAstropyTableCompatibility:
    def test__init_with_table_no_tag(self, mock_lc_astropy):
        mock_lc_astropy.remove_column("tag")
        assert "tag" not in mock_lc_astropy.columns

        td = TargetData(lightcurve=mock_lc_astropy)

        assert len(td.lightcurve) == 10
        # assert td.detections is None
        assert len(td.detections) == 10
        assert td.badqual is None
        assert td.non_detections is None

    def test__init_with_table_with_tag(self, mock_lc_astropy):
        td = TargetData(mock_lc_astropy, include_badqual=False)

        assert len(td.lightcurve) == 10
        assert len(td.detections) == 3
        assert len(td.badqual) == 3
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([8, 9, 10])
        assert set(td.badqual["obsId"]) == set([5, 6, 7])
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__init_with_table_badqual_true(self, mock_lc_astropy):
        td = TargetData(lightcurve=mock_lc_astropy, include_badqual=True)

        assert len(td.lightcurve) == 10
        assert len(td.detections) == 6
        assert len(td.badqual) == 0
        assert len(td.non_detections) == 4

        assert set(td.detections["obsId"]) == set([5, 6, 7, 8, 9, 10])
        assert set(td.badqual["obsId"]) == set()
        assert set(td.non_detections["obsId"]) == set([1, 2, 3, 4])

    def test__equality_no_lc_astropy(self, mock_lc_updates):
        td = TargetData()
        result = td.integrate_lightcurve_updates_equality(
            mock_lc_updates, column="obsId"
        )
        assert len(result) == 3

    def test__integrate_equality(self, mock_lc, mock_lc_updates):
        td = TargetData(lightcurve=mock_lc)

        result = td.integrate_lightcurve_updates_equality(
            mock_lc_updates, column="obsId"
        )
        assert len(result) == 13
        assert set(result["obsId"]) == set(range(1, 14))

    def test__equality_keep_updated_astropy(
        self, mock_lc_astropy, mock_lc_updates_astropy
    ):
        updates = vstack([mock_lc_astropy[7:], mock_lc_updates_astropy])
        assert len(updates) == 6
        updates["source"] = "updates"
        mock_lc_astropy["source"] = "original"

        td = TargetData(lightcurve=mock_lc_astropy)
        result = td.integrate_lightcurve_updates_equality(updates, column="obsId")
        assert len(result) == 13
        assert all(result[:7]["source"] == "original")
        assert all(result[7:]["source"] == "updates")
        assert len(result[result["source"] == "original"]) == 7
        assert len(result[result["source"] == "updates"]) == 6

    def test__integrate_equality_keep_original(
        self, mock_lc_astropy, mock_lc_updates_astropy
    ):
        updates = vstack([mock_lc_astropy[7:], mock_lc_updates_astropy])
        assert len(updates) == 6
        updates["source"] = "updates"
        mock_lc_astropy["source"] = "original"

        td = TargetData(lightcurve=mock_lc_astropy)
        result = td.integrate_lightcurve_updates_equality(
            updates, column="obsId", keep_updates=False
        )
        assert len(result) == 13
        assert all(result[:10]["source"] == "original")
        assert all(result[10:]["source"] == "updates")
        assert len(result[result["source"] == "original"]) == 10
        assert len(result[result["source"] == "updates"]) == 3
