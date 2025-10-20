import pytest
from pathlib import Path
from typing import List

import numpy as np

import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from astroplan import Observer

from aas2rto.lightcurve_compilers import DefaultLightcurveCompiler
from aas2rto.modeling.modeling_manager import ModelingManager
from aas2rto.observatory_manager import ObservatoryManager
from aas2rto.path_manager import PathManager
from aas2rto.plotting.plotting_manager import PlottingManager
from aas2rto.scoring.scoring_manager import ScoringManager

from aas2rto.target import Target
from aas2rto.target_data import TargetData
from aas2rto.target_lookup import TargetLookup


@pytest.fixture(scope="session")
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
def southern_target(t_fixed: Time):
    alt_ids = {"src02": "target_C"}
    coord = SkyCoord(ra=270.0, dec=-30.0, unit="deg")
    return Target("T02", coord=coord, source="src01", alt_ids=alt_ids, t_ref=t_fixed)


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
def lasilla():
    return Observer.at_site("lasilla")


@pytest.fixture
def id0():
    return 1000_2000_3000_4000


@pytest.fixture
def missing_id():
    return -1


@pytest.fixture
def ulim_rows(missing_id: int, t_fixed: Time):
    # obsid mjd mag magerr maglim filt tag
    return [
        [missing_id, t_fixed.mjd + 0.0, np.nan, np.nan, 22.0, 1, "ulim"],
        [missing_id, t_fixed.mjd + 0.1, np.nan, np.nan, 21.8, 2, "ulim"],
        [missing_id, t_fixed.mjd + 1.0, np.nan, np.nan, 22.0, 1, "ulim"],
        [missing_id, t_fixed.mjd + 1.1, np.nan, np.nan, 21.8, 2, "ulim"],
    ]


@pytest.fixture
def badqual_rows(missing_id: int, t_fixed: Time):
    return [
        [missing_id, t_fixed.mjd + 2.0, 21.0, 0.5, 22.0, 1, "badqual"],
        [missing_id, t_fixed.mjd + 2.1, 21.0, 0.5, 21.8, 2, "badqual"],
        [missing_id, t_fixed.mjd + 3.0, 20.0, 0.3, 22.0, 1, "badqual"],
        [missing_id, t_fixed.mjd + 3.1, 20.0, 0.3, 21.8, 2, "badqual"],
    ]


@pytest.fixture
def det_rows(t_fixed: Time, id0: int):
    return [
        [id0 + 1, t_fixed.mjd + 4.1, 21.0, 0.05, 21.8, 2, "valid"],
        [id0 + 0, t_fixed.mjd + 4.0, 21.0, 0.08, 22.0, 1, "valid"],
        [id0 + 2, t_fixed.mjd + 5.0, 20.0, 0.08, 22.0, 1, "valid"],
        [id0 + 3, t_fixed.mjd + 5.1, 20.0, 0.05, 21.8, 2, "valid"],
        [id0 + 4, t_fixed.mjd + 6.0, 19.0, 0.08, 22.0, 1, "valid"],
        [id0 + 5, t_fixed.mjd + 6.1, 19.0, 0.05, 21.8, 2, "valid"],
    ]


@pytest.fixture
def lc_rows(ulim_rows: list, badqual_rows: list, det_rows: list):
    return ulim_rows + badqual_rows + det_rows


@pytest.fixture
def lc_col_names():
    return "obsid mjd mag magerr diffmaglim band tag".split()


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
def extra_det_rows(t_fixed: Time, id0: int):
    return [
        [id0 + 6, t_fixed.mjd + 7.0, 20.0, 0.08, 22.0, 1, "valid"],
        [id0 + 7, t_fixed.mjd + 7.1, 20.0, 0.05, 21.8, 2, "valid"],
        [id0 + 8, t_fixed.mjd + 8.0, 19.0, 0.08, 22.0, 1, "valid"],
        [id0 + 9, t_fixed.mjd + 8.1, 19.0, 0.05, 21.8, 2, "valid"],
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
def path_mgr_no_paths(path_mgr_config: dict):
    return PathManager(path_mgr_config, create_paths=False)


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
    return tl


@pytest.fixture
def obs_mgr_config():
    return {
        "dt": 60.0,
        "sites": {
            "lasilla": "lasilla",
            "astrolab": {"lat": 54.767, "lon": -1.5742, "height": 10},
        },
    }


@pytest.fixture
def obs_mgr(obs_mgr_config: dict, tlookup: TargetLookup, path_mgr: PathManager):
    return ObservatoryManager(obs_mgr_config, tlookup, path_mgr)


@pytest.fixture
def modeling_mgr_config():
    return {}


@pytest.fixture
def modeling_mgr(
    modeling_mgr_config: dict, tlookup: TargetLookup, path_mgr: PathManager
):
    return ModelingManager(modeling_mgr_config, tlookup, path_mgr)


@pytest.fixture
def scoring_mgr(
    tlookup: TargetLookup, path_mgr: PathManager, obs_mgr: ObservatoryManager
):
    return ScoringManager({}, tlookup, path_mgr, obs_mgr)


@pytest.fixture
def plotting_mgr(
    tlookup: Target, path_mgr: PathManager, obs_mgr: ObservatoryManager, t_fixed: Time
):
    mgr = PlottingManager({}, tlookup, path_mgr, obs_mgr)
    mgr.observatory_manager.apply_ephem_info(t_ref=t_fixed)
    return mgr


@pytest.fixture()
def lc_compiler():
    return DefaultLightcurveCompiler()


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
def compiled_ztf_lc(
    ztf_td: TargetData, lc_compiler: DefaultLightcurveCompiler, t_fixed
):
    return lc_compiler(ztf_td, t_ref=t_fixed)  # t_ref does nothing.


@pytest.fixture
def atlas_rows(t_fixed: Time):
    return [
        [t_fixed.mjd + 0.00, 21.0, 0.1, 20.0, "o", "obs_o_a1"],  # grp0 - ulim
        [t_fixed.mjd + 0.01, 21.0, 0.1, 20.0, "o", "obs_o_a1"],
        [t_fixed.mjd + 0.99, 20.1, 0.1, 20.0, "o", "obs_o_b1"],  # grp1
        [t_fixed.mjd + 1.00, 20.0, 0.1, 20.0, "o", "obs_o_b2"],
        [t_fixed.mjd + 1.00, 19.9, 0.1, 20.0, "c", "obs_c_b1"],  # grp2
        [t_fixed.mjd + 1.01, 19.9, 0.1, 20.0, "o", "obs_o_b3"],  # grp1 again
        [t_fixed.mjd + 1.02, 19.8, 0.1, 20.0, "o", "obs_o_b4"],
        [t_fixed.mjd + 1.99, 19.1, 0.1, 20.0, "c", "obs_c_c1"],  # grp3
        [t_fixed.mjd + 2.01, 18.9, 0.1, 20.0, "c", "obs_c_c2"],
        [t_fixed.mjd + 3.00, 18.5, 3.0, 20.0, "c", "obs_c_d1"],  # grp 4
        [t_fixed.mjd + 3.00, 18.5, 3.0, 20.0, "c", "obs_c_d2"],
        [t_fixed.mjd + 3.00, 18.0, 0.1, 20.0, "c", "obs_c_d3"],
    ]


@pytest.fixture
def lc_atlas(atlas_rows: List[List]):
    colnames = "mjd m dm mag5sig F Obs".split()
    return pd.DataFrame(atlas_rows, columns=colnames)


@pytest.fixture
def atlas_td(lc_atlas: pd.DataFrame):
    return TargetData(lightcurve=lc_atlas)


@pytest.fixture
def yse_rows(t_fixed):
    return [
        [t_fixed.mjd + 0.0, 21.0, 0.1, 22.0, "g", "Pan-STARRS1", 100.0],
        [t_fixed.mjd + 0.1, 21.0, 0.1, 22.0, "r", "Pan-STARRS2", 100.0],
        [t_fixed.mjd + 0.2, 21.0, 0.1, 22.0, "i", "Pan-STARRS1", 100.0],
        [t_fixed.mjd + 1.0, 20.0, 0.1, 21.0, "UVW1", "Swift", 100.0],
        [t_fixed.mjd + 2.0, 19.0, 0.1, 22.0, "g", "Pan-STARRS1", 100.0],
    ]


@pytest.fixture
def lc_yse(yse_rows: List[List], t_fixed: Time):
    colnames = "mjd mag magerr diffmaglim flt instrument blah".split()
    return pd.DataFrame(yse_rows, columns=colnames)


@pytest.fixture
def yse_td(lc_yse):
    return TargetData(lightcurve=lc_yse)
