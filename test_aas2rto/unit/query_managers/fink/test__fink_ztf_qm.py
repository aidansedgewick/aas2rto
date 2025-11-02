import copy
import pytest
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from astropy import units as u
from astropy.time import Time

from aas2rto.query_managers.fink.fink_ztf import (
    FinkZTFQueryManager,
    process_ztf_alert,
    apply_ztf_updates_to_target,
    target_from_ztf_alert,
)
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
        cutoutScience=dict(stampData=None),
        cutoutDifference=dict(stampData=None),
        cutoutTemplate=dict(stampData=None),
    )


@pytest.fixture
def fink_ztf_alert_stream(
    ztf_alert_base: dict, mock_cutouts: Dict[str, np.array], t_fixed: Time
):
    alert_data_list = []
    for ii in range(5):
        alert = copy.deepcopy(ztf_alert_base)
        alert["candidate"]["jd"] = (t_fixed + ii * u.day).mjd
        alert["candidate"]["mag"] = 20.0 - ii * 0.5
        alert["candidate"]["magerr"] = 0.1
        alert["candidate"]["fid"] = (ii % 2) + 1  # 1 or 2
        alert["candid"] = 1000_2000_3000_4000 + ii  # some long integer.

        alert_data = ("cool_sne", alert, "some_string")
        alert_data_list.append(alert_data)
    return alert_data_list


@pytest.fixture
def patched_ztf_qm(fink_config: dict, tlookup: TargetLookup, tmp_path: Path):
    qm = FinkZTFQueryManager(fink_config, tlookup, parent_path=tmp_path)
    return qm


class Test__ProcessZTFAlert:
    def test__process_ztf_alert(
        self,
    ):
        pass


class Test__ProcessAlerts:
    def test__alert_processed(
        self, patched_ztf_qm: FinkZTFQueryManager, fink_ztf_alert_stream: List[Tuple]
    ):
        # Arrange
        alert_data = fink_ztf_alert_stream[0]

        # Act
        processed_alert = patched_ztf_qm.process_single_alert(alert_data)

        # Assert"
        assert "objectId" in processed_alert
        assert "topic" in processed_alert
        assert "tag" in processed_alert
        assert "candid" in processed_alert
        assert "mjd" in processed_alert


class Test__ListenForAlerts:
    def test__listen_for_alerts(self):
        pass


class Test__FinkZTFQMInit:
    def test__fink_ztf_init(self):
        pass
