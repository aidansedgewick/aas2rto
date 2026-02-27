import copy
import pytest
from pathlib import Path
from typing import NoReturn

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.time import Time

from aas2rto.query_managers.fink.fink_base import FinkAlert
from aas2rto.query_managers.fink.fink_lsst import (
    FinkLSSTQueryManager,
    process_fink_lsst_alert,
    target_from_fink_lsst_alert,
    apply_fink_lsst_updates_to_target,
    process_fink_lsst_lightcurve,
    EXTRA_FINK_LSST_ALERT_KEYS,
)
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup


@pytest.fixture
def mock_cutouts():
    return dict(
        cutoutScience=dict(stampData=np.random.normal(0.0, 0.1, (10, 10))),
        cutoutDifference=dict(stampData=np.random.normal(0.0, 0.1, (10, 10))),
        cutoutTemplate=dict(stampData=np.random.normal(0.0, 0.1, (10, 10))),
    )


@pytest.fixture
def fink_lsst_alert_list(
    lsst_alert_base: dict, t_fixed: Time, fink_alert_extras: dict
) -> list[FinkAlert]:

    lsst_filters = "g r i".split()

    alert_data_list = []
    for ii in range(5):
        alert = copy.deepcopy(lsst_alert_base)

        alert["diaSource"]["midpointMjdTai"] = t_fixed.mjd + ii
        alert["diaSource"]["psfFlux"] = 10 ** (-0.4 * ((22.0 - ii * 0.5) - 31.4))
        alert["diaSource"]["psfFluxErr"] = alert["diaSource"]["psfFlux"] / 10.0
        alert["diaSource"]["band"] = lsst_filters[ii % 3]

        alert.update(copy.deepcopy(fink_alert_extras))

        alert_data = ("lsst_cool_sne", alert, "schema_string")
        alert_data_list.append(alert_data)
    return alert_data_list


@pytest.fixture
def patch_readstamp(monkeypatch: pytest.MonkeyPatch) -> NoReturn:

    def mock_readstamp(data, return_array="array"):
        return data

    monkeypatch.setattr(
        "aas2rto.query_managers.fink.fink_lsst.readstamp", mock_readstamp
    )


@pytest.fixture
def patched_lsst_qm(
    fink_config: dict, tlookup: TargetLookup, tmp_path: Path, patch_readstamp: NoReturn
):
    qm = FinkLSSTQueryManager(fink_config, tlookup, parent_path=tmp_path)
    return qm


@pytest.fixture
def processed_lsst_alert_list(
    patched_lsst_qm: FinkLSSTQueryManager,
    fink_lsst_alert_list: list[FinkAlert],
    tmp_path: Path,
) -> list[dict]:
    processed_alerts = []
    alert_dir = tmp_path / "fink_lsst/alerts/1000"
    cutouts_dir = tmp_path / "fink_lsst/cutouts/1000"
    for alert_data in fink_lsst_alert_list:
        processed_alert = patched_lsst_qm.process_single_alert(alert_data)
        processed_alerts.append(processed_alert)

        # Ensure alerts are dumped...
        alert_id = processed_alert["diaSourceId"]
        alert_path = alert_dir / f"{alert_id}.json"
        assert alert_path.exists()
        cutouts_path = cutouts_dir / f"{alert_id}.pkl"
        assert cutouts_path.exists()
    return processed_alerts


@pytest.fixture
def unprocessed_lsst_lc(lsst_lc: pd.DataFrame):
    lc = lsst_lc.copy()
    return lc


##===== Tests start here! =====##


class Test__ProcessLSSTAlert:

    @pytest.fixture
    def lsst_alert_data(
        self, lsst_alert_base: dict, fink_alert_extras: dict
    ) -> FinkAlert:
        # Arrange
        alert = copy.deepcopy(lsst_alert_base)
        alert.update(fink_alert_extras)
        return ("some_topic", alert, "some_schema")

    def test__process_lsst_alert(
        self, lsst_alert_data: FinkAlert, patch_readstamp: NoReturn
    ):

        # Act
        proc_alert = process_fink_lsst_alert(lsst_alert_data)

        # Assert
        candidate_keys = (
            "diaObjectId ra dec diaSourceId midpointMjdTai psfFlux psfFluxErr".split()
        )
        cutout_keys = [f"cutout{x}" for x in "Science Difference Template".split()]

        assert all([k in proc_alert.keys() for k in candidate_keys])
        assert all([k in proc_alert.keys() for k in EXTRA_FINK_LSST_ALERT_KEYS])  # OK!
        # cutout removed before JSON!
        assert all([k not in proc_alert.keys() for k in cutout_keys])

        # Cutouts are removed from alert in order to json!
        assert "topic" in proc_alert
        assert "diaSource" not in proc_alert.keys()  # flattened.

    def test__write_alert_data(
        self, lsst_alert_data: FinkAlert, patch_readstamp: NoReturn, tmp_path: Path
    ):
        # Arrange
        alert_path = tmp_path / "alert.json"
        cutouts_path = tmp_path / "cutouts.pkl"

        # Act
        proc_alert = process_fink_lsst_alert(
            lsst_alert_data, alert_filepath=alert_path, cutouts_filepath=cutouts_path
        )

        # Assert
        assert alert_path.exists()
        assert cutouts_path.exists()


class Test__TargetFromAlert:
    def test__target_from_alert(self, processed_lsst_alert_list: list[dict]):
        # Arrange
        processed_alert = processed_lsst_alert_list[0]

        # Act
        target = target_from_fink_lsst_alert(processed_alert)

        # Assert
        assert isinstance(target, Target)
        assert isinstance(target.target_id, str)  # int diaObjectId converted!
        assert target.target_id == "1000"

        assert set(target.alt_ids.keys()) == set(["lsst", "fink_lsst"])

    def test__target_is_added(
        self,
        processed_lsst_alert_list: list[dict],
        patched_lsst_qm: FinkLSSTQueryManager,
    ):
        # Arrange
        processed_alert = processed_lsst_alert_list[0]

        # Act
        targets_added = patched_lsst_qm.add_targets_from_alerts([processed_alert])

        # Assert
        assert set(targets_added) == set(["1000"])

        assert "1000" in patched_lsst_qm.target_lookup

    def test__target_(self):
        pass


class Test__ApplyUpdatesFromAlert:
    def test__apply_updates_to_target(
        self,
        processed_lsst_alert_list: list[dict],
    ):
        # Arrange
        processed_alert = processed_lsst_alert_list[0]
        t_1000 = target_from_fink_lsst_alert(processed_alert)

        # Act
        apply_fink_lsst_updates_to_target(t_1000, processed_alert)

        # Assert
        assert t_1000.updated

        assert len(t_1000.info_messages) == 1
        assert t_1000.info_messages[0].startswith("New FINK-LSST")

    def test__use_qm_function(
        self, processed_lsst_alert_list: list, patched_lsst_qm: FinkLSSTQueryManager
    ):
        # Arrange
        processed_alert = processed_lsst_alert_list[0]
        patched_lsst_qm.add_targets_from_alerts([processed_alert])

        # Act
        patched_lsst_qm.update_info_messages([processed_alert])


# class Test__ProcessLSSTLightcurve:
#     def test__process_lightcurve(
#         self,
#     ):
#         pass
