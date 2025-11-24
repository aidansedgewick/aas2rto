import pytest

from aas2rto.path_manager import PathManager

from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.query_managers.global_query_manager import (
    EXPECTED_QUERY_MANAGERS,
    GlobalQueryManager,
)

from aas2rto.target_lookup import TargetLookup


class Test__InitGlobalQM:
    def test__no_qms(self, tlookup: TargetLookup, path_mgr: PathManager):
        # Arrange
        empty_config = {}

        # Act
        gqm = GlobalQueryManager(empty_config, tlookup, path_mgr)

        # Assert
        assert isinstance(gqm.query_managers, dict)
        assert set(gqm.query_managers.keys()) == set()

    def test__all_qms_init(
        self, global_qm_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Act
        gqm = GlobalQueryManager(global_qm_config, tlookup, path_mgr)

        # Assert
        exp_qms = ["fink_ztf", "fink_lsst", "atlas"]
        assert set(gqm.query_managers.keys()) == set(exp_qms)

        untested_qms = set(EXPECTED_QUERY_MANAGERS.keys()) - set(
            gqm.query_managers.keys()
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
        gqm = GlobalQueryManager(config, tlookup, path_mgr)
        # Assert
        assert set(gqm.query_managers.keys()) == set()

    def test__unknown_qm_raises(self, tlookup: TargetLookup, path_mgr: PathManager):
        # Arrange
        config = {"unknown_qm": {}}

        # Act
        with pytest.raises(ValueError):
            gqm = GlobalQueryManager(config, tlookup, path_mgr)
