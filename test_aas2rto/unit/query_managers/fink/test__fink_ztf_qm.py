import copy
import pickle
import pytest
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.time import Time

from aas2rto.query_managers.fink.fink_base import FinkAlert, BaseFinkQueryManager
from aas2rto.query_managers.fink.fink_ztf import (
    FinkZTFQueryManager,
    process_ztf_alert,
    process_ztf_lightcurve,
    apply_ztf_updates_to_target,
    target_from_ztf_alert,
)
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup


@pytest.fixture
def ztf_alert_extras():
    return {
        "cdsxmatch": 1.0,
        "rf_snia_vs_nonia": 1.0,
        "snn_snia_vs_nonia": 1.0,
        "snn_sn_vs_all": 1.0,
        "mulens": 0.1,
        "roid": 0.1,
        "nalerthist": 4,
        "rf_kn_vs_nonkn": 0.3,
    }


@pytest.fixture
def ztf_alert_base(ztf_alert_extras):
    return {
        "objectId": "ZTF00abc",
        "candidate": {"ra": 180.0, "dec": -30.0},
        **ztf_alert_extras,
    }


@pytest.fixture
def mock_cutouts():
    return dict(
        cutoutScience=dict(stampData=np.random.normal(0.0, 0.1, (10, 10))),
        cutoutDifference=dict(stampData=np.random.normal(0.0, 0.1, (10, 10))),
        cutoutTemplate=dict(stampData=np.random.normal(0.0, 0.1, (10, 10))),
    )


@pytest.fixture
def fink_ztf_alert_list(
    ztf_alert_base: dict, mock_cutouts: Dict[str, np.array], t_fixed: Time
):
    alert_data_list = []
    for ii in range(5):
        # prepare "main" data
        alert = copy.deepcopy(ztf_alert_base)  # so nested dict is not just VIEW
        alert["candidate"]["jd"] = (t_fixed + ii * u.day).jd
        alert["candidate"]["magpsf"] = 20.0 - ii * 0.5
        alert["candidate"]["sigpsf"] = 0.1
        alert["candidate"]["fid"] = (ii % 2) + 1  # 1 or 2
        alert["candid"] = 1000_2000_3000_4000 + ii  # some long integer
        # now add cutouts
        alert_cutouts = copy.deepcopy(mock_cutouts)
        science_data = alert_cutouts["cutoutScience"]["stampData"]
        alert_cutouts["cutoutScience"]["stampData"] = science_data + ii
        alert.update(alert_cutouts)  # cutouts need to be in "top level"

        alert_data = ("cool_sne", alert, "some_string")
        alert_data_list.append(alert_data)
    return alert_data_list


@pytest.fixture
def patched_ztf_qm(
    fink_config: dict,
    tlookup: TargetLookup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):

    def mock_readstamp(data, return_type="array"):
        return data

    qm = FinkZTFQueryManager(fink_config, tlookup, parent_path=tmp_path)
    monkeypatch.setattr(
        "aas2rto.query_managers.fink.fink_ztf.readstamp", mock_readstamp
    )
    return qm


@pytest.fixture
def processed_ztf_alert_list(
    patched_ztf_qm: FinkZTFQueryManager,
    fink_ztf_alert_list: list[FinkAlert],
    tmp_path: Path,
) -> list[dict]:
    processed_alerts = []
    alert_dir = tmp_path / "fink_ztf/alerts/ZTF00abc"
    cutouts_dir = tmp_path / "fink_ztf/cutouts/ZTF00abc"
    for alert_data in fink_ztf_alert_list:
        processed_alert = patched_ztf_qm.process_single_alert(alert_data)
        processed_alerts.append(processed_alert)

        # Ensure alerts are dumped!
        candid = processed_alert["candid"]
        alert_path = alert_dir / f"{candid}.json"
        assert alert_path.exists()  #
        cutouts_path = cutouts_dir / f"{candid}.pkl"
        assert cutouts_path.exists()
    return processed_alerts


@pytest.fixture
def unprocessed_lc_ztf(lc_ztf: pd.DataFrame):
    lc = lc_ztf.copy()
    lc.loc[:, "objectId"] = "ZTF00xyz"
    lc["jd"] = Time(lc["mjd"].values, format="mjd").jd
    lc.drop("mjd", axis=1, inplace=True)
    return lc


##===== Tests start here! =====##


class Test__ProcessZTFAlert:
    def test__process_ztf_alert(self):
        pass


class Test__ProcessAlerts:
    def test__alert_processed(
        self,
        patched_ztf_qm: FinkZTFQueryManager,
        fink_ztf_alert_list: List[Tuple],
        tmp_path: Path,
    ):
        # Arrange
        alert_data = fink_ztf_alert_list[0]
        exp_alerts_dir = tmp_path / "fink_ztf/alerts/ZTF00abc"
        assert not exp_alerts_dir.exists()
        exp_cutouts_dir = tmp_path / "fink_ztf/cutouts/ZTF00abc"
        assert not exp_cutouts_dir.exists()

        # Act
        processed_alert = patched_ztf_qm.process_single_alert(alert_data)

        # Assert
        assert "objectId" in processed_alert
        assert "topic" in processed_alert
        assert "tag" in processed_alert
        assert "candid" in processed_alert
        assert "mjd" in processed_alert

        # check data are dumped to the correct path
        assert exp_alerts_dir.exists()
        alert_filepath = exp_alerts_dir / "1000200030004000.json"
        assert alert_filepath.exists()
        cutout_filepath = exp_cutouts_dir / "1000200030004000.pkl"
        assert cutout_filepath.exists()


class Test__TargetFromAlert:
    def test__target_from_alert(
        self,
        patched_ztf_qm: FinkZTFQueryManager,
        processed_ztf_alert_list: list[FinkAlert],
    ):
        # Arrange
        alert = processed_ztf_alert_list[0]

        # Act
        target = target_from_ztf_alert(alert)

        # Assert
        assert isinstance(target, Target)
        assert target.target_id == "ZTF00abc"

    def test__target_is_added(
        self, patched_ztf_qm: FinkZTFQueryManager, processed_ztf_alert_list: list[dict]
    ):
        # Arrange
        alert = processed_ztf_alert_list[0]

        # Act
        target = patched_ztf_qm.add_target_from_alert(alert)

        # Assert
        assert isinstance(target, Target)
        assert target.target_id == "ZTF00abc"

        # now check it's in the target_lookup
        assert "ZTF00abc" in patched_ztf_qm.target_lookup


class Test__ApplyUpdatesFromTarget:
    def test__apply_ztf_updates_to_target(
        self, basic_target: Target, processed_ztf_alert_list: list[dict], t_fixed: Time
    ):
        # Arrange
        alert = processed_ztf_alert_list[0]

        # Act
        apply_ztf_updates_to_target(basic_target, alert, t_ref=t_fixed)

        # Assert
        assert len(basic_target.info_messages) == 1
        assert basic_target.updated

    def test__apply_update_from_alert(
        self,
        patched_ztf_qm: FinkZTFQueryManager,
        processed_ztf_alert_list: list[dict],
        t_fixed: Time,
    ):
        # Arrange
        alert = processed_ztf_alert_list[0]
        patched_ztf_qm.add_target_from_alert(alert)

        # Act
        patched_ztf_qm.apply_updates_from_alert(alert)

        # Assert
        ZTF00abc = patched_ztf_qm.target_lookup["ZTF00abc"]
        assert len(ZTF00abc.info_messages) == 1

    def test__skip_missing_target(
        self,
        patched_ztf_qm: FinkZTFQueryManager,
        processed_ztf_alert_list: list[dict],
        t_fixed: Time,
    ):
        # Arrange
        alert = processed_ztf_alert_list[0]
        # DON'T add target this time.

        # Act
        patched_ztf_qm.apply_updates_from_alert(alert)  # Basically test no fail...


class Test__LoadMissingAlerts:
    def test__no_existing_lc(
        self,
        patched_ztf_qm: FinkZTFQueryManager,
        processed_ztf_alert_list: list[dict],
        t_fixed: Time,
        tmp_path: Path,
    ):
        # Arrange
        alert_dir = tmp_path / "fink_ztf/alerts/ZTF00abc"
        assert alert_dir.exists()
        # alerts are dumped in creation of processed_ztf_alert_list
        alert = processed_ztf_alert_list[0]
        ZTF00abc = patched_ztf_qm.add_target_from_alert(alert, t_ref=t_fixed)
        assert set(ZTF00abc.target_data.keys()) == set()

        # Act
        alerts_loaded = patched_ztf_qm.load_missing_alerts_for_target("ZTF00abc")

        # Assert
        assert alerts_loaded
        assert set(ZTF00abc.target_data.keys()) == set(["fink_ztf"])
        ztf_data = ZTF00abc.target_data["fink_ztf"]
        assert len(ztf_data.lightcurve) == 5

    def test__no_dumped_alerts(
        self,
        patched_ztf_qm: FinkZTFQueryManager,
        fink_ztf_alert_list: list[FinkAlert],
        t_fixed: Time,
        tmp_path: Path,
    ):
        # Arrange
        alert = process_ztf_alert(fink_ztf_alert_list[0])
        alert_dir = tmp_path / "fink_ztf/alerts/ZTF00abc"
        assert not alert_dir.exists()
        ZTF00abc = patched_ztf_qm.add_target_from_alert(alert, t_ref=t_fixed)

        # Act
        alerts_loaded = patched_ztf_qm.load_missing_alerts_for_target("ZTF00abc")

        # Assert
        assert not alerts_loaded
        assert not alert_dir.exists()  # dir not created while checking for alerts

    def test__few_missing(
        self,
        patched_ztf_qm: FinkZTFQueryManager,
        processed_ztf_alert_list: list[dict],
        t_fixed: Time,
        tmp_path: Path,
    ):
        # Arrange
        alert = processed_ztf_alert_list[0]
        ZTF00abc = patched_ztf_qm.add_target_from_alert(alert, t_ref=t_fixed)
        lc = pd.DataFrame(processed_ztf_alert_list[:3])
        lc.loc[:, "tag"] = "badqual"  # to check we don't re-load unnecessarily
        ztf_data = ZTF00abc.get_target_data("fink_ztf")
        ztf_data.add_lightcurve(lc)

        # Act
        alerts_loaded = patched_ztf_qm.load_missing_alerts_for_target("ZTF00abc")

        # Assert
        assert alerts_loaded
        assert len(ztf_data.lightcurve) == 5
        assert len(ztf_data.detections) == 2  # the others all kept "badqual"

    def test__all_already_loaded(
        self,
        patched_ztf_qm: FinkZTFQueryManager,
        processed_ztf_alert_list: list[dict],
        t_fixed: Time,
    ):
        # Arrange
        alert = processed_ztf_alert_list[0]
        ZTF00abc = patched_ztf_qm.add_target_from_alert(alert, t_ref=t_fixed)
        lc = pd.DataFrame(processed_ztf_alert_list)
        ztf_data = ZTF00abc.get_target_data("fink_ztf")
        ztf_data.add_lightcurve(lc)

        # Act
        alerts_loaded = patched_ztf_qm.load_missing_alerts_for_target("ZTF00abc")

        # Assert
        assert not alerts_loaded


class Test__ProcessZTFLC:
    def test__process_ztf_lc(self, unprocessed_lc_ztf: pd.DataFrame):
        # Assert
        assert "mjd" not in unprocessed_lc_ztf.columns
        assert "candid" in unprocessed_lc_ztf.columns

        # Act
        lc = process_ztf_lightcurve(unprocessed_lc_ztf)

        # Assert
        assert "mjd" in lc.columns

    def test__candid_inserted(self, unprocessed_lc_ztf: pd.DataFrame):
        # Arrange
        unprocessed_lc_ztf.drop("candid", axis=1, inplace=True)

        # Act
        lc = process_ztf_lightcurve(unprocessed_lc_ztf)

        # Assert
        assert "candid" in lc.columns


class Test__FinkZTFQMInit:
    def test__fink_ztf_init(self, tlookup: TargetLookup, tmp_path: Path):
        # Act
        qm = FinkZTFQueryManager({}, tlookup, parent_path=tmp_path)

        # Assert
        assert isinstance(qm, BaseFinkQueryManager)
        assert hasattr(qm, "config")  # OK this is now just testing subclassing...
