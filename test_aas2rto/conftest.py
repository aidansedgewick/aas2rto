import copy
import pytest
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from astroplan import Observer

import matplotlib.pyplot as plt

from aas2rto.lightcurve_compilers import DefaultLightcurveCompiler, prepare_ztf_data
from aas2rto.messaging.messaging_manager import MessagingManager
from aas2rto.messaging.slack_messenger import SlackMessenger, SlackApiError
from aas2rto.messaging.telegram_messenger import TelegramMessenger
from aas2rto.modeling.modeling_manager import ModelingManager
from aas2rto.observatory.observatory_manager import ObservatoryManager
from aas2rto.outputs.outputs_manager import OutputsManager
from aas2rto.path_manager import PathManager
from aas2rto.plotting.plotting_manager import PlottingManager
from aas2rto.scoring.scoring_manager import ScoringManager

from aas2rto.target import Target
from aas2rto.target_data import TargetData
from aas2rto.target_lookup import TargetLookup


def remove_empty_filetree(dirpath: Path, depth=0, max_depth=6):
    if depth > max_depth:
        return
    for filepath in dirpath.iterdir():
        if filepath.is_dir():
            remove_empty_filetree(filepath, depth=depth + 1, max_depth=max_depth)
    dirpath_is_empty = not list(dirpath.iterdir())
    if dirpath_is_empty:
        dirpath.rmdir()


@pytest.fixture()  # do NOT auto use here. Often want to check empty dirs 'by hand'...
def remove_tmp_dirs(tmp_path: Path):
    # Arrange
    pass  # Code BEFORE yield in fixture is setup. No setup here...

    # Act
    yield  # Test is run here.

    # Cleanup
    remove_empty_filetree(tmp_path)  # Code AFTER yield in fixture is cleanup/teardown


@pytest.fixture(autouse=True)
def no_subprocess(monkeypatch: pytest.MonkeyPatch):
    # define HERE (main conftest.py), so that no subprocess commands ever run.
    def dummy_check_output(*args, **kwargs):
        return "".encode()

    monkeypatch.setattr(subprocess, "check_output", dummy_check_output)


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
    # obs_id mjd mag magerr maglim filt tag
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
        [id0 + 1, t_fixed.mjd + 4.1, 21.0, 0.05, 21.8, 2, "valid"],  # note wrong order!
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
    return "obs_id mjd mag magerr diffmaglim band tag".split()


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
def ztf_lc(lc_pandas: pd.DataFrame):
    col_mapping = {
        "obs_id": "candid",
        "mag": "magpsf",
        "magerr": "sigmapsf",
        "band": "fid",
    }
    lc_pandas.rename(col_mapping, inplace=True, axis=1)
    lc_pandas.loc[:, "blah"] = 100.0
    return lc_pandas


@pytest.fixture
def ztf_td(ztf_lc: pd.DataFrame):
    return TargetData(lightcurve=ztf_lc)


@pytest.fixture
def compiled_ztf_lc(ztf_td: TargetData) -> pd.DataFrame:
    return prepare_ztf_data(ztf_td)


@pytest.fixture
def ztf_cutouts():
    return {
        "science": np.random.normal(0, 1.0, (20, 20)),
        "template": np.random.normal(0, 1.0, (20, 20)),
        "difference": np.random.normal(0, 1.0, (20, 20)),
    }


@pytest.fixture
def lsst_id0():
    return 1234_5678_9000


@pytest.fixture
def lsst_rows(lsst_id0: int, t_fixed: Time):
    return [
        [lsst_id0 + 0, t_fixed.mjd + 0.0, 1000.0, 100.0, "g"],  # ~23.9+-0.1
        [lsst_id0 + 1, t_fixed.mjd + 1.0, 2000.0, 200.0, "r"],  # ~23.1+-0.1
        [lsst_id0 + 2, t_fixed.mjd + 2.0, 5000.0, 500.0, "i"],  # ~22.2+-0.1
        [lsst_id0 + 3, t_fixed.mjd + 3.0, 10000.0, 100.0, "g"],  # ~21.4+-0.01
        [lsst_id0 + 4, t_fixed.mjd + 4.0, 20000.0, 200.0, "r"],  # ~20.6+-0.01
        [lsst_id0 + 5, t_fixed.mjd + 5.0, 50000.0, 500.0, "i"],  # ~19.7+-0.01
    ]


@pytest.fixture
def lsst_lc(lsst_rows: list[list]):
    col_names = "diaSourceId midpointMjdTai psfFlux psfFluxErr band".split()
    return pd.DataFrame(lsst_rows, columns=col_names)


@pytest.fixture
def lsst_td(lsst_lc: pd.DataFrame):
    return TargetData(lightcurve=lsst_lc)


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
def atlas_lc(atlas_rows: List[List]):
    col_names = "mjd m dm mag5sig F Obs".split()
    return pd.DataFrame(atlas_rows, columns=col_names)


@pytest.fixture
def atlas_td(atlas_lc: pd.DataFrame):
    return TargetData(lightcurve=atlas_lc)


@pytest.fixture
def yse_rows(t_fixed):
    return [
        [t_fixed.mjd + 0.0, 21.0, 0.1, 22.0, "g", "Pan-STARRS1", "None"],
        [t_fixed.mjd + 0.1, 21.0, 0.1, 22.0, "r", "Pan-STARRS2", "None"],
        [t_fixed.mjd + 0.2, 21.0, 0.1, 22.0, "i", "Pan-STARRS1", "BAD"],
        [t_fixed.mjd + 1.0, 20.0, 0.1, 21.0, "UVW1", "Swift", "None"],
        [t_fixed.mjd + 1.1, 20.0, 0.1, 21.0, "Unkown", "Unknown", "None"],
        [t_fixed.mjd + 2.0, 19.0, 0.1, 22.0, "g", "Pan-STARRS1", "None"],
    ]


@pytest.fixture
def yse_columns():
    return "mjd mag magerr diffmaglim flt instrument dq".split()


@pytest.fixture
def yse_lc(yse_columns: list[str], yse_rows: list[list], t_fixed: Time):
    return pd.DataFrame(yse_rows, columns=yse_columns)


@pytest.fixture
def yse_td(yse_lc: pd.DataFrame):
    return TargetData(lightcurve=yse_lc)


@pytest.fixture
def yse_explorer_columns():
    return "name classification ra dec n_det".split()


@pytest.fixture
def yse_explorer_rows():
    return [
        ["2023J", "SN", 180.0, 0.0, 5],
        ["2023K", "SNIa", 90.0, 30.0, 5],
        ["2023M", "", 355.0, -80.0, 5],
    ]


@pytest.fixture
def tns_td():
    return TargetData(parameters={"redshift": 0.02})


@pytest.fixture
def t_plot():
    return Time(60010.0, format="mjd")


@pytest.fixture
def mod_tl(tlookup: TargetLookup, lasilla: Observer, t_fixed: Time):
    tlookup["T00"].update_science_score_history(10.0, t_ref=t_fixed)
    tlookup["T00"].update_obs_score_history(3.0, lasilla, t_ref=t_fixed)
    tlookup["T00"].update_obs_score_history(0.5, "astrolab", t_ref=t_fixed)
    tlookup["T00"].science_comments = ["T00 comment"]
    tlookup["T00"].obs_comments["lasilla"] = ["T00 lasilla comment"]
    tlookup["T00"].obs_comments["astrolab"] = ["T00 astrolab comment"]

    tlookup["T01"].update_science_score_history(5.0, t_ref=t_fixed)
    tlookup["T01"].update_obs_score_history(1.0, lasilla, t_ref=t_fixed)
    tlookup["T01"].update_obs_score_history(2.0, "astrolab", t_ref=t_fixed)
    tlookup["T01"].science_comments = ["T01 comment"]
    tlookup["T01"].obs_comments["lasilla"] = ["T01 lasilla comment"]
    tlookup["T01"].obs_comments["astrolab"] = ["T01 astrolab comment"]
    return tlookup


@pytest.fixture
def outputs_mgr(
    mod_tl: TargetLookup, path_mgr: PathManager, obs_mgr: ObservatoryManager
):
    return OutputsManager({}, mod_tl, path_mgr, obs_mgr)


def blank_plot_helper(filepath, title=None):
    fig, ax = plt.subplots(figsize=(1, 1))
    if title is not None:
        fig.suptitle(title)
    fig.savefig(filepath)
    plt.close(fig)
    return


@pytest.fixture
def outputs_mgr_with_plots(
    mod_tl: TargetLookup, path_mgr: PathManager, obs_mgr: ObservatoryManager
):
    omgr = OutputsManager({}, mod_tl, path_mgr, obs_mgr)
    for target_id, target in mod_tl.items():
        lc_path = path_mgr.get_lightcurve_plot_path(target_id)
        blank_plot_helper(lc_path)
        target.lc_fig_path = lc_path
        for obs_name in obs_mgr.sites.keys():
            vis_path = path_mgr.get_visibility_plot_path(target_id, obs_name)
            blank_plot_helper(vis_path)
            target.vis_fig_paths[obs_name] = vis_path
    return omgr


@pytest.fixture
def tl_vis_targets(lasilla: Observer, t_fixed: Time):
    # Reverse engineer some targets which will be visible and not visible at t_fixed
    midnight = lasilla.midnight(t_fixed, which="next", n_grid_points=40)
    t_early = midnight - 3.0 * u.hour  # earlier
    t_later = midnight + 3.0 * u.hour  # later

    lst_e = lasilla.local_sidereal_time(t_early)  # has units deg
    lst_m = lasilla.local_sidereal_time(midnight)
    lst_l = lasilla.local_sidereal_time(t_later)

    o_lat = lasilla.location.lat  # dec = lat +/- ZD, ZD = (90-transit_alt)
    c00 = SkyCoord(ra=lst_m, dec=o_lat - (90.0 - 80.0) * u.deg)  # 00:00 local
    c01 = SkyCoord(ra=lst_l, dec=o_lat - (90.0 - 80.0) * u.deg)  # 03:00 local
    c02 = SkyCoord(ra=lst_m, dec=o_lat + (90.0 - 20.0) * u.deg)  # too low (North)
    c03 = SkyCoord(ra=lst_e, dec=o_lat - (90.0 - 80.0) * u.deg)  # 21:00 local
    c04 = SkyCoord(ra=lst_m, dec=o_lat - (90.0 - 80.0) * u.deg)  # bad score
    c05 = SkyCoord(ra=lst_m, dec=o_lat - (90.0 - 80.0) * u.deg)  # no score

    target_list = [
        Target("Tv00", c00.transform_to("icrs")),
        Target("Tv01", c01.transform_to("icrs")),
        Target("Tv02", c02.transform_to("icrs")),
        Target("Tv03", c03.transform_to("icrs")),
        Target("Tv04", c04.transform_to("icrs")),
        Target("Tv05", c05.transform_to("icrs")),
    ]

    tl = TargetLookup()
    tl.add_target_list(target_list)
    tl["Tv00"].update_science_score_history(1.0, midnight)
    tl["Tv01"].update_science_score_history(1.0, midnight)
    tl["Tv02"].update_science_score_history(1.0, midnight)
    tl["Tv03"].update_science_score_history(1.0, midnight)
    tl["Tv04"].update_science_score_history(-1.0, midnight)
    # Tv05 has no score!
    return tl


@pytest.fixture  #
def om_vis_targets(
    tl_vis_targets: TargetLookup,
    path_mgr: PathManager,
    obs_mgr_config: dict,
    t_fixed: Time,
):
    obs_mgr = ObservatoryManager(obs_mgr_config, tl_vis_targets, path_mgr)
    om = OutputsManager({}, tl_vis_targets, path_mgr, obs_mgr)
    om.observatory_manager.apply_ephem_info(t_ref=t_fixed)
    return om


@pytest.fixture
def target_to_plot(
    basic_target: Target,
    ztf_td: TargetData,
    atlas_td: TargetData,
    tns_td: TargetData,
    ztf_cutouts: Dict[str, np.ndarray],
    lc_compiler: DefaultLightcurveCompiler,
    t_plot: Time,
):
    ztf_td.cutouts = ztf_cutouts
    basic_target.target_data["ztf"] = ztf_td
    basic_target.target_data["atlas"] = atlas_td
    basic_target.target_data["tns"] = tns_td
    basic_target.compiled_lightcurve = lc_compiler(basic_target, t_ref=t_plot)
    basic_target.science_comments = ["a comment here", "another comment"]
    return basic_target


@pytest.fixture
def slack_config():
    return {"token": "123456", "channel_id": "channel:abcdef"}


def mock_chat_postMessage(channel=None, text=None):
    if channel is None:
        raise ValueError("Should not pass channel=None to chat_postMessage()!")
    if text is None:
        raise ValueError("Should not pass text=None to chat_postMessage()")
    return


def mock_files_upload_v2(channel=None, initial_comment=None, file=None):
    if channel is None:
        raise ValueError("Should not pass channel=None to files_upload_v2()!")
    if file is None:
        raise ValueError("Should not pass file=None to files_upload_v2()")
    if isinstance(file, str) or isinstance(file, Path):
        raise TypeError("file should be FILE DATA, not Path/str!")
    return


def mock_chat_postMessage_raises(**kwargs):
    raise SlackApiError("raised in mock function", 404)


def mock_upload_files_v2_raises(**kwargs):
    raise SlackApiError("raised in mock function", 404)


@pytest.fixture
def slack_msgr(slack_config: dict, monkeypatch: pytest.MonkeyPatch):
    msgr = SlackMessenger(slack_config)
    monkeypatch.setattr(msgr.client, "chat_postMessage", mock_chat_postMessage)
    monkeypatch.setattr(msgr.client, "files_upload_v2", mock_files_upload_v2)
    return msgr


@pytest.fixture
def raising_slack_msgr(slack_config: dict, monkeypatch: pytest.MonkeyPatch):
    msgr = SlackMessenger(slack_config)
    monkeypatch.setattr(msgr.client, "chat_postMessage", mock_chat_postMessage_raises)
    monkeypatch.setattr(msgr.client, "files_upload_v2", mock_upload_files_v2_raises)
    return msgr


@pytest.fixture
def telegram_config():
    return {
        "token": "abcdef",
        "users": {101: "user1", 102: "user2"},
        "sudoers": {901: "sudoer1"},
    }


@pytest.fixture
def telegram_msgr(telegram_config: dict):
    return TelegramMessenger(telegram_config)


class MockBot:
    def __init__(self):
        self.token = "mock_token"

    async def send_message(
        self,
        chat_id: int = None,
        text: str = None,
        disable_web_page_preview: bool = None,
    ):
        if not isinstance(text, str):
            raise TypeError(f"text should be str, not {type(text)}")

        return dict(user=chat_id, msg_type="text", length=len(text))

    async def send_media_group(self, chat_id: int = None, media: list = None):
        if not isinstance(media, list):
            raise TypeError(f"media_group should be list, not type {type(media)}!")
        return dict(user=chat_id, msg_type="media_group", length=len(media))

    async def send_photo(
        self, chat_id: int = None, photo: bytes = None, caption: str = None
    ):
        return dict(user=chat_id, msg_type="single_photo", length="n/a")


@pytest.fixture
def telegram_msgr(telegram_config: dict, monkeypatch: pytest.MonkeyPatch):
    def get_mock_bot():
        return MockBot()

    msgr = TelegramMessenger(telegram_config)
    monkeypatch.setattr(msgr, "get_bot", get_mock_bot)
    return msgr


@pytest.fixture
def msg_mgr_config(slack_config: dict, telegram_config: dict):
    return {
        "slack": {"use": True, **slack_config},
        "telegram": {"use": True, **telegram_config},
    }


@pytest.fixture
def msg_mgr(
    slack_msgr: SlackMessenger,
    telegram_msgr: TelegramMessenger,
    tlookup: TargetLookup,
    path_mgr: PathManager,
):
    mgr = MessagingManager({}, tlookup, path_mgr)
    mgr.slack_messenger = slack_msgr
    mgr.telegram_messenger = telegram_msgr
    return mgr


@pytest.fixture
def fink_kafka_config():
    return {
        "username": "user",
        "group.id": 1234,
        "bootstrap.servers": "http://fink.blah.org",
        "topics": ["cool_sne"],
        "survey": "cool_survey",
    }


@pytest.fixture
def fink_config(fink_kafka_config: dict):
    return {"kafka": fink_kafka_config, "fink_classes": "cool_sne"}


@pytest.fixture
def lasair_ztf_kafka_config():
    topic_keys = {"lasair_id": "objectId", "ra": "ramean", "dec": "decmean"}
    return {
        "kafka_server": "lasair-blah.org",
        "group_id": "test_group",
        "topics": {
            "cool_sne": topic_keys,
        },
        "survey": "ztf",
    }


@pytest.fixture
def lasair_ztf_config(lasair_ztf_kafka_config: dict):
    return {"kafka": lasair_ztf_kafka_config, "token": "example_token"}


@pytest.fixture
def lasair_lsst_kafka_config():
    topic_keys = {"lasair_id": "diaObjectId", "ra": "ramean", "dec": "decmean"}
    return {
        "kafka_server": "lasair-blah.org",
        "group_id": "test_group",
        "topics": {
            "cool_sne": topic_keys,
        },
        "survey": "lsst",
    }


@pytest.fixture
def lasair_lsst_config(lasair_lsst_kafka_config: dict):
    return {"kafka": lasair_lsst_kafka_config, "token": "example_token"}


@pytest.fixture
def atlas_credentials():
    return {"token": 1234}


@pytest.fixture
def atlas_config(atlas_credentials: dict):
    return {"credentials": atlas_credentials, "project_identifier": "test"}


@pytest.fixture
def tns_credentials():
    return {"user": "test_user", "uid": 1234}


@pytest.fixture
def tns_config(tns_credentials: dict):
    return {"credentials": tns_credentials}


@pytest.fixture
def yse_credentials():
    return {"username": "user", "password": "password"}


@pytest.fixture
def yse_query_name():
    return "yse_test_query"


@pytest.fixture
def yse_explorer_queries(yse_query_name: str):
    return {
        yse_query_name: {
            "query_id": 101,
            "target_id_col": "name",
            "coordinate_cols": ("ra", "dec"),
            "comparison_col": "n_det",
        }
    }


@pytest.fixture
def yse_config(yse_credentials: dict, yse_explorer_queries: dict):
    return {"credentials": yse_credentials, "explorer_queries": yse_explorer_queries}


@pytest.fixture
def global_qm_config(
    fink_config: dict,
    lasair_ztf_config: dict,
    lasair_lsst_config: dict,
    atlas_config: dict,
    yse_config: dict,
    tns_config: dict,
):
    # Deepcopy, else sub-dicts not modified correctly for tests of config-checkers.
    return {
        "fink_lsst": copy.deepcopy(fink_config),
        "fink_ztf": copy.deepcopy(fink_config),
        # "lasair_ztf": copy.deepcopy(lasair_ztf_config),
        # "lasair_lsst": copy.deepcopy(lasair_lsst_config),
        "atlas": copy.deepcopy(atlas_config),
        "yse": copy.deepcopy(yse_config),
        "tns": copy.deepcopy(tns_config),
    }
