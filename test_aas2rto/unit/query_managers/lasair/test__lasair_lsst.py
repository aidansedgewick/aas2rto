import json
import pytest
from pathlib import Path

import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.time import Time

from aas2rto.query_managers.lasair.lasair_lsst import (
    LSST_ALERT_ID_KEY,
    LSST_TARGET_ID_KEY,
    LasairLSSTQueryManager,
    process_lasair_lsst_object_data,
)
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup


@pytest.fixture
def lasair_lsst_object_data(lsst_lc: pd.DataFrame, t_fixed: Time):
    lc = lsst_lc.to_dict("records")

    imageUrls = []
    for rec in lc:
        alert_id = rec[LSST_ALERT_ID_KEY]
        imtypes = ["Difference", "Template", "Science"]
        image_url_rec = {imtype: f"{imtype}_{alert_id}" for imtype in imtypes}
        image_url_rec[LSST_ALERT_ID_KEY] = alert_id
        imageUrls.append(image_url_rec)

    forced_phot = [
        {"midpointMjdTai": t_fixed.mjd, "psfFlux": 10000.0},
    ]

    return {
        "diaObjectId": "T00",
        "diaObject": {
            "ra": "60.0",
            "decl": "-45.0",
            "other_key": 100.0,
        },
        "diaSourcesList": lc,
        "diaForcedSourcesList": forced_phot,
        "lasairData": {
            "imageUrls": imageUrls,
            "sherlock_classifier": "SN",  # Who knew!?
        },
    }


@pytest.fixture
def processed_lasair_alert(t_fixed: Time):
    return {
        "diaObjectId": 1000_2000_3000_4000_5000,
        "midpointMjdTai": 1001_2001_3001_4001_5001,
        "ra": 180.0,
        "decl": -60.0,
    }


@pytest.fixture
def lasair_lsst_qm(lasair_lsst_config: dict, tlookup: TargetLookup, tmp_path: Path):
    return LasairLSSTQueryManager(lasair_lsst_config, tlookup, tmp_path)


##===== Tests start here!! =====##


class Test__Constants:
    # quite silly tests - but helpful for catching typos...

    def test__target_id_key(self):
        # Assert
        assert LSST_TARGET_ID_KEY == "diaObjectId"

    def test__alert_id_key(self):
        # Assert
        assert LSST_ALERT_ID_KEY == "diaSourceId"


class Test__ProcessDataFunction:
    def test__process_data(self, lasair_lsst_object_data: dict, tmp_path: Path):
        # Arrange
        obj_data_filepath = tmp_path / "object_data.json"
        lc_filepath = tmp_path / "lightcurve.csv"
        fp_filepath = tmp_path / "forced_photom.csv"
        image_urls_filepath = tmp_path / "lighcurves.csv"
        lasair_context_filepath = tmp_path / "context.json"

        # Act
        result = process_lasair_lsst_object_data(
            lasair_lsst_object_data,
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
        assert set(data.keys()) == set(["ra", "decl", "other_key"])

        lc = pd.read_csv(lc_filepath)
        assert len(lc) == 6


class Test__QueryManagerInit:
    def test__lasair_lsst_qm_init(
        self, lasair_lsst_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Act
        qm = LasairLSSTQueryManager(lasair_lsst_config, tlookup, tmp_path)

        # Assert
        assert isinstance(qm.default_topic_keys, dict)
        assert isinstance(qm.default_topic_keys["id_key"], str)
        assert qm.default_topic_keys["id_key"] == "diaObjectId"  # @property def works!


class Test__NewTargetFromAlert:
    def test__new_target(
        self, processed_lasair_alert: dict, lasair_lsst_qm: LasairLSSTQueryManager
    ):
        # Act
        target = lasair_lsst_qm.new_target_from_alert(processed_lasair_alert)

        # Assert
        assert target.target_id == str(1000_2000_3000_4000_5000)
        assert isinstance(target, Target)
        assert isinstance(target.coord, SkyCoord)

    def test__new_target_no_coords(
        self, processed_lasair_alert: dict, lasair_lsst_qm: LasairLSSTQueryManager
    ):
        # Arrange
        processed_lasair_alert.pop("ra")
        processed_lasair_alert.pop("decl")

        # Act
        target = lasair_lsst_qm.new_target_from_alert(processed_lasair_alert)

        # Assert
        assert isinstance(target, Target)
        assert target.coord is None


class Test__ApplyUpdatesMethod:
    def test__apply_updates_method(
        self,
        processed_lasair_alert: dict,
        lasair_lsst_qm: LasairLSSTQueryManager,
        t_fixed: Time,
    ):
        # Arrange
        lasair_lsst_qm.add_targets_from_alerts([processed_lasair_alert])
        lsst_id = str(1000_2000_3000_4000_5000)
        new_target = lasair_lsst_qm.target_lookup[lsst_id]

        # Act
        target = lasair_lsst_qm.apply_updates_from_alert(
            processed_lasair_alert, t_ref=t_fixed
        )

        # Assert
        assert len(new_target.info_messages) == 1

        assert new_target.updated
        assert new_target.info_messages[0].startswith("New Lasair-LSST alert")
