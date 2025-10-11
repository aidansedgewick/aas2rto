import json
import pytest
from pathlib import Path

from astropy.time import Time

from aas2rto.exc import UnexpectedKeysWarning
from aas2rto.path_manager import PathManager
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup
from aas2rto.recovery_manager import RecoveryManager


@pytest.fixture
def basic_config():
    return {}


@pytest.fixture
def rec_mgr(basic_config: dict, tlookup: TargetLookup, path_mgr: PathManager):
    return RecoveryManager(basic_config, tlookup, path_mgr)


class Test__RecMgrInit:
    def test__normal_behaviour(
        self, basic_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Act
        rec_mgr = RecoveryManager(basic_config, tlookup, path_mgr)

        # Assert
        assert set(rec_mgr.config.keys()) == set(
            ["retained_recovery_files", "load_rank_history"]
        )

    def test__unexp_key_warns(
        self, basic_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Arrange
        basic_config["bad_kwarg"] = "some_str"

        # Act
        with pytest.warns(UnexpectedKeysWarning):
            rec_mgr = RecoveryManager(basic_config, tlookup, path_mgr)


class Test__WriteRecFiles:
    def test__write_basic(
        self, rec_mgr: RecoveryManager, t_fixed: Time, tmp_path: Path
    ):
        # Act
        rec_mgr.write_recovery_file(t_fixed)

        # Assert
        rec_path = rec_mgr.path_manager.lookup["recovery"]
        exp_path = rec_path / "recover_230205_000000.json"
        assert exp_path.exists()

        with open(exp_path, "r") as f:
            data = json.load(f)

        assert set(data.keys()) == set(["T00", "T01"])
