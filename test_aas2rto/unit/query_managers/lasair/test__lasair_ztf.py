import copy
import json
import pytest
from pathlib import Path

import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.time import Time

from aas2rto.query_managers.lasair.lasair_ztf import (
    ZTF_ALERT_ID_KEY,
    ZTF_TARGET_ID_KEY,
    LasairZTFQueryManager,
    process_lasair_ztf_object_data,
)
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup


@pytest.fixture
def processed_lasair_alert(t_fixed: Time):
    return {
        "objectId": "ZTF00abc",
        "candid": 1001_2001_3001_4001_5001,
        "ramean": 180.0,
        "decmean": -60.0,
    }


@pytest.fixture
def lasair_ztf_object_data(ztf_lc: pd.DataFrame, t_fixed: Time):
    lc = ztf_lc.to_dict("records")

    candidates = []
    for rec in lc:
        candid = copy.deepcopy(rec)
        alert_id = rec[ZTF_ALERT_ID_KEY]
        imtypes = ["Difference", "Template", "Science"]
        image_url_rec = {imtype: f"{imtype}_{alert_id}" for imtype in imtypes}
        image_url_rec[ZTF_ALERT_ID_KEY] = alert_id
        candid["image_urls"] = image_url_rec
        candidates.append(candid)

    forced_phot = [
        {"jd": t_fixed.mjd, "diffmag": 22.0},
    ]

    return {
        "diaObjectId": "T00",
        "objectData": {
            "ramean": "60.0",
            "decmean": "-45.0",
            "other_key": 100.0,
        },
        "candidates": candidates,
        "forcedphot": forced_phot,
        "sherlock": {
            "classified_classifier": "SN",  # Who knew!?
        },
    }


@pytest.fixture
def lasair_ztf_qm(lasair_ztf_config: dict, tlookup: TargetLookup, tmp_path: Path):
    return LasairZTFQueryManager(lasair_ztf_config, tlookup, tmp_path)


class Test__ProcessDataFunction:
    def test__process_data(self, lasair_ztf_object_data: dict, tmp_path: Path):
        # Arrange
        obj_data_filepath = tmp_path / "object_data.json"
        lc_filepath = tmp_path / "lightcurve.csv"
        fp_filepath = tmp_path / "forced_photom.csv"
        image_urls_filepath = tmp_path / "lighcurves.csv"
        lasair_context_filepath = tmp_path / "context.json"

        # Act
        result = process_lasair_ztf_object_data(
            lasair_ztf_object_data,
            object_data_filepath=obj_data_filepath,
            lightcurve_filepath=lc_filepath,
            forced_photom_filepath=fp_filepath,
            image_urls_filepath=image_urls_filepath,
            lasair_context_filepath=lasair_context_filepath,
        )

        # Assert
        assert obj_data_filepath.exists()
        assert lc_filepath.exists()
        assert fp_filepath.exists()
        assert image_urls_filepath.exists()
        assert lasair_context_filepath.exists()

        with open(obj_data_filepath, "r") as f:
            data = json.load(f)
        assert set(data.keys()) == set(["ramean", "decmean", "other_key"])

        lc = pd.read_csv(lc_filepath)
        assert len(lc) == 14


class Test__ApplyUpdatesMethod:
    def test__apply_updates_method(
        self,
        lasair_ztf_qm: LasairZTFQueryManager,
        processed_lasair_alert: dict,
    ):
        # Arrange
        lasair_ztf_qm.add_targets_from_alerts([processed_lasair_alert])
        new_target = lasair_ztf_qm.target_lookup["ZTF00abc"]

        # Act
        lasair_ztf_qm.apply_updates_from_alert(processed_lasair_alert)

        # Assert
        assert len(new_target.info_messages) == 1

        assert new_target.updated
        assert new_target.info_messages[0].startswith("New Lasair-ZTF alert")
