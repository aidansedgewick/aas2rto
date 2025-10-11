import pytest
from pathlib import Path

import numpy as np

import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from aas2rto.target import Target
from aas2rto.target_data import TargetData
from aas2rto.target_lookup import TargetLookup

from aas2rto.path_manager import PathManager


@pytest.fixture
def t_fixed():
    return Time(60000.0, format="mjd")


@pytest.fixture
def basic_target(t_fixed: Time):
    alt_ids = {"src02": "target_A"}
    coord = SkyCoord(ra=180.0, dec=0.0, unit="deg")
    return Target("T00", coord=coord, source="src01", alt_ids=alt_ids, t_ref=t_fixed)


@pytest.fixture
def other_target(t_fixed: Time):
    alt_ids = {"src02": "target_B"}
    coord = SkyCoord(ra=90.0, dec=30.0, unit="deg")
    return Target("T01", coord=coord, source="src01", alt_ids=alt_ids, t_ref=t_fixed)


@pytest.fixture
def target_config_example():
    return {
        "target_id": "T99",
        "ra": 90.0,
        "dec": -30.0,
        "base_score": 100.0,
        "alt_ids": {"src02": "target_Z"},
        "target_of_opportunity": True,
    }


@pytest.fixture
def mjd0():
    return 60000.0


@pytest.fixture
def id0():
    return 1000_2000_3000_4000


@pytest.fixture
def ulim_rows(mjd0):
    # obsid mjd mag magerr maglim filt tag
    return [
        [np.nan, mjd0 + 0.0, np.nan, np.nan, 22.0, 1, "ulim"],
        [np.nan, mjd0 + 0.1, np.nan, np.nan, 21.8, 2, "ulim"],
        [np.nan, mjd0 + 1.0, np.nan, np.nan, 22.0, 1, "ulim"],
        [np.nan, mjd0 + 1.1, np.nan, np.nan, 21.8, 2, "ulim"],
    ]


@pytest.fixture
def badqual_rows(mjd0):
    return [
        [np.nan, mjd0 + 2.0, 21.0, 0.5, 22.0, 1, "badqual"],
        [np.nan, mjd0 + 2.1, 21.0, 0.5, 21.8, 2, "badqual"],
        [np.nan, mjd0 + 3.0, 20.0, 0.3, 22.0, 1, "badqual"],
        [np.nan, mjd0 + 3.1, 20.0, 0.3, 21.8, 2, "badqual"],
    ]


@pytest.fixture
def det_rows(mjd0, id0):
    return [
        [id0 + 1, mjd0 + 4.1, 21.0, 0.05, 21.8, 2, "valid"],
        [id0 + 0, mjd0 + 4.0, 21.0, 0.08, 22.0, 1, "valid"],
        [id0 + 2, mjd0 + 5.0, 20.0, 0.08, 22.0, 1, "valid"],
        [id0 + 3, mjd0 + 5.1, 20.0, 0.05, 21.8, 2, "valid"],
        [id0 + 4, mjd0 + 6.0, 19.0, 0.08, 22.0, 1, "valid"],
        [id0 + 5, mjd0 + 6.1, 19.0, 0.05, 21.8, 2, "valid"],
    ]


@pytest.fixture
def lc_rows(ulim_rows, badqual_rows, det_rows):
    return ulim_rows + badqual_rows + det_rows


@pytest.fixture
def lc_col_names():
    return "obsid mjd mag magerr maglim band tag".split()


@pytest.fixture
def lc_pandas(lc_rows, lc_col_names):
    return pd.DataFrame(lc_rows, columns=lc_col_names)


@pytest.fixture
def tdata_lc_pandas(lc_pandas):
    return TargetData(lightcurve=lc_pandas)


@pytest.fixture
def lc_astropy(lc_rows, lc_col_names):
    return Table(rows=lc_rows, names=lc_col_names)


@pytest.fixture
def tdata_lc_astropy(lc_astropy):
    return TargetData(lightcurve=lc_astropy)


@pytest.fixture
def extra_det_rows(mjd0, id0):
    return [
        [id0 + 6, mjd0 + 7.0, 20.0, 0.08, 22.0, 1, "valid"],
        [id0 + 7, mjd0 + 7.1, 20.0, 0.05, 21.8, 2, "valid"],
        [id0 + 8, mjd0 + 8.0, 19.0, 0.08, 22.0, 1, "valid"],
        [id0 + 9, mjd0 + 8.1, 19.0, 0.05, 21.8, 2, "valid"],
    ]


@pytest.fixture
def extra_det_pandas(extra_det_rows, lc_col_names):
    return pd.DataFrame(extra_det_rows, columns=lc_col_names)


@pytest.fixture
def extra_det_astropy(extra_det_rows, lc_col_names):
    return Table(rows=extra_det_rows, names=lc_col_names)


@pytest.fixture
def path_mgr_config(tmp_path: Path):
    return {"base_path": tmp_path, "project_name": "test"}


@pytest.fixture
def path_mgr(path_mgr_config: dict):
    return PathManager(path_mgr_config)


@pytest.fixture
def tlookup(
    basic_target: Target,
    other_target: Target,
    lc_pandas: pd.DataFrame,
    extra_det_pandas: pd.DataFrame,
):
    tl = TargetLookup()
    basic_target.target_data["src01"] = TargetData(lightcurve=lc_pandas)
    other_target.target_data["src01"] = TargetData(lightcurve=extra_det_pandas)
    tl.add_target(basic_target)
    tl.add_target(other_target)
