import pytest

from aas2rto.path_manager import PathManager

from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.query_managers.primary import (
    EXPECTED_QUERY_MANAGERS,
    PrimaryQueryManager,
)

from aas2rto.target_lookup import TargetLookup


class Test__InitPrimaryQM:
    def test__no_qms(self, tlookup: TargetLookup, path_mgr: PathManager):
        # Arrange
        empty_config = {}

        # Act
        pqm = PrimaryQueryManager(empty_config, tlookup, path_mgr)

        # Assert
        assert isinstance(pqm.query_managers, dict)
        assert set(pqm.query_managers.keys()) == set()

    def test__all_qms_init(
        self, global_qm_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Act
        pqm = PrimaryQueryManager(global_qm_config, tlookup, path_mgr)

        # Assert
        exp_qms = ["atlas", "fink_ztf", "fink_lsst", "tns"]
        assert set(pqm.query_managers.keys()) == set(exp_qms)

        untested_qms = set(EXPECTED_QUERY_MANAGERS.keys()) - set(
            pqm.query_managers.keys()
        )
        if untested_qms:
            msg = (
                f"you should include example configs for the QMs {untested_qms} "
                "in the `global_qm_config` fixturein test_aas2rto/unit/conftest.py "
                "so that they are properly tested"
            )
            raise ValueError(msg)

    def test__respect_use_flag(
        self, atlas_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Arrange
        config = {"atlas": atlas_config}
        config["atlas"]["use"] = False

        # Act
        pqm = PrimaryQueryManager(config, tlookup, path_mgr)
        # Assert
        assert set(pqm.query_managers.keys()) == set()

    def test__unknown_qm_raises(self, tlookup: TargetLookup, path_mgr: PathManager):
        # Arrange
        config = {"unknown_qm": {}}

        # Act
        with pytest.raises(ValueError):
            pqm = PrimaryQueryManager(config, tlookup, path_mgr)
