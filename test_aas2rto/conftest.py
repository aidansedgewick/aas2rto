import copy
import pytest
from pathlib import Path
from typing import Dict, List

import numpy as np

import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from astroplan import Observer

from aas2rto.lightcurve_compilers import DefaultLightcurveCompiler
from aas2rto.messaging.messaging_manager import MessagingManager
from aas2rto.messaging.slack_messenger import SlackMessenger, SlackApiError
from aas2rto.messaging.telegram_messenger import TelegramMessenger
from aas2rto.modeling.modeling_manager import ModelingManager
from aas2rto.observatory_manager import ObservatoryManager
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


@pytest.fixture()  # do NOT auto use. Sometimes we want to check the empty dirs exist.
def remove_tmp_dirs(tmp_path: Path):
    # Arrange
    pass  # Code BEFORE yield in fixture is setup. No setup here...

    # Act
    yield  # Test is run here.

    # Cleanup
    remove_empty_filetree(tmp_path)  # Code AFTER yield in fixture is cleanup/teardown


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
def lc_ztf(lc_pandas: pd.DataFrame):
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
def ztf_td(lc_ztf: pd.DataFrame):
    return TargetData(lightcurve=lc_ztf)


@pytest.fixture
def compiled_ztf_lc(
    ztf_td: TargetData, lc_compiler: DefaultLightcurveCompiler, t_fixed
):
    return lc_compiler(ztf_td, t_ref=t_fixed)  # t_ref does nothing.


@pytest.fixture
def ztf_cutouts():
    return {
        "science": np.random.normal(0, 1.0, (20, 20)),
        "template": np.random.normal(0, 1.0, (20, 20)),
        "difference": np.random.normal(0, 1.0, (20, 20)),
    }


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


@pytest.fixture
def tns_td():
    return TargetData(parameters={"redshift": 0.02})


@pytest.fixture
def t_plot():
    return Time(60010.0, format="mjd")


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
    }


@pytest.fixture
def fink_config(fink_kafka_config: dict):
    return {"kafka": fink_kafka_config, "fink_classes": "cool_sne"}


@pytest.fixture
def atlas_credentials():
    return {"token": 1234}


@pytest.fixture
def atlas_config(atlas_credentials: dict):
    return {"credentials": atlas_credentials, "project_identifier": "test"}


@pytest.fixture
def global_qm_config(fink_config: dict, atlas_config: dict):
    return {
        "fink_lsst": copy.deepcopy(fink_config),
        "fink_ztf": copy.deepcopy(fink_config),
        "atlas": copy.deepcopy(atlas_config),
    }
