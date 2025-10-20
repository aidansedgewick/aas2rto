import json
import pytest
from pathlib import Path

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.time import Time

from aas2rto.exc import MissingFileWarning, UnexpectedKeysWarning
from aas2rto.path_manager import PathManager
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup
from aas2rto.recovery_manager import RecoveryManager


@pytest.fixture
def basic_config():
    return {}


@pytest.fixture
def rec_mgr(
    basic_config: dict, tlookup: TargetLookup, path_mgr: PathManager, tmp_path: Path
):
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
    def test__write_basic(self, rec_mgr: RecoveryManager, t_fixed: Time):
        # Arrange
        rec_mgr.target_lookup["T00"].base_score = 100.0
        rec_mgr.target_lookup["T01"].alt_ids = {}  # Remove alt_ids

        # Act
        recovery_file = rec_mgr.write_recovery_file(t_fixed)

        # Assert
        exp_path = rec_mgr.path_manager.get_current_recovery_file(t_ref=t_fixed)
        assert exp_path.exists()

        with open(exp_path, "r") as f:
            data = json.load(f)

        assert set(data.keys()) == set(["T00", "T01"])

        expected_keys = ["target_id", "ra", "dec", "base_score", "alt_ids"]
        assert set(data["T00"].keys()) == set(expected_keys)
        assert np.isclose(data["T00"]["ra"], 180.0)
        assert np.isclose(data["T00"]["dec"], 0.0)
        assert np.isclose(data["T00"]["base_score"], 100.0)
        assert set(data["T00"]["alt_ids"].keys()) == set(["src01", "src02"])

        assert set(data["T01"].keys()) == set(expected_keys)
        assert np.isclose(data["T01"]["ra"], 90.0)
        assert np.isclose(data["T01"]["dec"], 30.0)
        assert np.isclose(data["T01"]["base_score"], 1.0)
        assert set(data["T01"]["alt_ids"].keys()) == set()

    def test__old_files_removed(self, rec_mgr: RecoveryManager, t_fixed: Time):
        # Arrange
        rec_path = rec_mgr.path_manager.lookup["recovery"]
        rec_mgr.config["retained_recovery_files"] = 2
        rec_filepath1 = rec_path / "recover_230223_000000.json"
        with open(rec_filepath1, "w+") as f:
            f.write("0")
        rec_filepath2 = rec_path / "recover_230224_000000.json"
        with open(rec_filepath2, "w+") as f:
            f.write("0")
        assert rec_filepath1.exists()
        assert rec_filepath2.exists()

        # Act
        rec_mgr.write_recovery_file(t_fixed)

        # Assert
        exp_path = rec_mgr.path_manager.get_current_recovery_file(t_ref=t_fixed)
        assert exp_path.exists()

        assert not rec_filepath1.exists()  # correctly removed!
        assert rec_filepath2.exists()  # kept!

    def test__clear_old_files(self, rec_mgr: RecoveryManager, t_fixed: Time):
        # Arrange
        rec_mgr.target_lookup.pop("T00")
        rec_mgr.target_lookup.pop("T01")
        assert len(rec_mgr.target_lookup) == 0

        # Act
        recovery_file = rec_mgr.write_recovery_file(t_ref=t_fixed)

        # Assert
        assert recovery_file is None

        exp_path = rec_mgr.path_manager.get_current_recovery_file(t_ref=t_fixed)
        assert not exp_path.exists()


class Test__WriteRankHistFiles:
    def test__write_basic(self, rec_mgr: RecoveryManager, t_fixed: Time):
        # Arrange
        t1 = Time(59998.0, format="mjd")
        t2 = Time(59999.0, format="mjd")
        T00 = rec_mgr.target_lookup["T00"]
        T01 = rec_mgr.target_lookup["T01"]
        T00.update_science_rank_history(1, t_ref=t1)  # T00 len >1
        T00.update_science_rank_history(2, t_ref=t2)
        T01.update_science_rank_history(3, t_ref=t2)  # T01 len == 1

        # Act
        rec_mgr.write_rank_histories(t_ref=t_fixed)

        # Assert
        exp_path = rec_mgr.path_manager.get_current_rank_history_file(t_ref=t_fixed)
        assert exp_path.exists()
        with open(exp_path, "r") as f:
            data = json.load(f)

        assert set(data.keys()) == set(["T00", "T01"])

        assert set(data["T00"].keys()) == set(["science"])
        assert len(data["T00"]["science"]) == 2
        t00_no_obs_data = data["T00"]["science"]
        assert t00_no_obs_data[0][0] == 1
        assert t00_no_obs_data[1][0] == 2
        assert np.isclose(t00_no_obs_data[0][1], 59998.0)
        assert np.isclose(t00_no_obs_data[1][1], 59999.0)

        assert set(data["T01"].keys()) == set(["science"])
        assert len(data["T01"]["science"]) == 1
        t01_no_obs_data = data["T01"]["science"]
        assert t01_no_obs_data[0][0] == 3
        assert np.isclose(t01_no_obs_data[0][1], 59999.0)

    def test__write_additional_obs(self, rec_mgr: RecoveryManager, t_fixed: Time):
        # Arrange
        t1 = Time(59998.0, format="mjd")
        t2 = Time(59999.0, format="mjd")
        T00 = rec_mgr.target_lookup["T00"]
        T01 = rec_mgr.target_lookup["T01"]
        T00.update_science_rank_history(1, t_ref=t1)
        T00.update_science_rank_history(2, t_ref=t2)
        T00.update_obs_rank_history(1, observatory="lasilla", t_ref=t2)

        T01.update_science_rank_history(3, t_ref=t2)

        # Act
        rec_mgr.write_rank_histories(observatories=True, t_ref=t_fixed)

        # Assert
        exp_path = rec_mgr.path_manager.get_current_rank_history_file(t_ref=t_fixed)
        assert exp_path.exists()
        with open(exp_path, "r") as f:
            data = json.load(f)

        assert set(data.keys()) == set(["T00", "T01"])

        assert set(data["T00"].keys()) == set(["science", "observatory"])
        assert len(data["T00"]["science"]) == 2
        assert set(data["T00"]["observatory"].keys()) == set(["lasilla"])
        assert len(data["T00"]["observatory"]["lasilla"]) == 1

        assert set(data["T01"].keys()) == set(["science", "observatory"])
        assert len(data["T01"]["science"]) == 1
        assert set(data["T01"]["observatory"].keys()) == set()

    def test__no_hist_no_fail(self, rec_mgr: RecoveryManager, t_fixed: Time):
        # Arrange
        T00 = rec_mgr.target_lookup["T00"]
        T00.update_science_rank_history(1, t_ref=t_fixed)

        # Act
        rec_mgr.write_rank_histories(observatories=True, t_ref=t_fixed)

        # Assert
        exp_path = rec_mgr.path_manager.get_current_rank_history_file(t_ref=t_fixed)
        assert exp_path.exists()
        with open(exp_path, "r") as f:
            data = json.load(f)

        assert set(data.keys()) == set(["T00"])

    def test__clear_old_files(self, rec_mgr: RecoveryManager, t_fixed: Time):
        # Arrange
        rec_path = rec_mgr.path_manager.lookup["recovery"]
        rec_mgr.config["retained_recovery_files"] = 2
        rec_filepath1 = rec_path / "rank_history_230223_000000.json"
        with open(rec_filepath1, "w+") as f:
            f.write("0")
        rec_filepath2 = rec_path / "rank_history_230224_000000.json"
        with open(rec_filepath2, "w+") as f:
            f.write("0")
        assert rec_filepath1.exists()
        assert rec_filepath2.exists()

        rec_mgr.target_lookup["T00"].update_science_rank_history(1, t_ref=t_fixed)

        # Act
        rec_mgr.write_rank_histories(t_ref=t_fixed)

        exp_path = rec_mgr.path_manager.get_current_rank_history_file(t_ref=t_fixed)
        assert exp_path.exists()
        assert not rec_filepath1.exists()
        assert rec_filepath2.exists()


class Test__RecoverFromFile:
    def test__recover_with_rank_hist(
        self,
        rec_mgr: RecoveryManager,
        basic_config: dict,
        path_mgr: PathManager,
        t_fixed: Time,
    ):
        # Arrange
        rec_mgr.target_lookup["T00"].update_science_rank_history(1, t_ref=t_fixed)
        rec_mgr.target_lookup["T00"].base_score = 100.0
        rec_mgr.target_lookup["T01"].alt_ids = {}
        rec_mgr.write_recovery_file(t_ref=t_fixed)
        rec_mgr.write_rank_histories(t_ref=t_fixed)
        # Need new rec_mgr to test tlookup is correctly populated...
        new_rec_mgr = RecoveryManager(basic_config, TargetLookup(), path_mgr)

        # Act
        recovered_targets = new_rec_mgr.recover_targets_from_file()

        # Assert
        assert len(recovered_targets) == 2
        assert set(new_rec_mgr.target_lookup.keys()) == set(["T00", "T01"])

        T00 = new_rec_mgr.target_lookup["T00"]
        assert isinstance(T00.coord, SkyCoord)
        assert np.isclose(T00.coord.ra.deg, 180.0)
        assert np.isclose(T00.coord.dec.deg, 0.0)
        assert np.isclose(T00.base_score, 100.0)
        assert set(T00.alt_ids.keys()) == set(["src01", "src02"])
        assert T00.alt_ids["src01"] == "T00"
        assert T00.alt_ids["src02"] == "target_A"

        assert len(T00.science_rank_history) == 1

        T01 = new_rec_mgr.target_lookup["T01"]
        assert np.isclose(T01.coord.ra.deg, 90.0)
        assert np.isclose(T01.coord.dec.deg, 30.0)
        assert np.isclose(T01.base_score, 1.0)

    def test__recover_no_rank_hist(
        self,
        rec_mgr: RecoveryManager,
        basic_config: dict,
        path_mgr: PathManager,
        t_fixed: Time,
    ):
        # Arrange
        rec_mgr.target_lookup["T00"].update_science_rank_history(1, t_ref=t_fixed)
        rec_mgr.target_lookup["T00"].base_score = 100.0
        rec_mgr.target_lookup["T01"].alt_ids = {}
        rec_mgr.write_recovery_file(t_ref=t_fixed)  # Don't write rank hist now...
        # Need new rec_mgr to test tlookup is correctly populated...
        new_rec_mgr = RecoveryManager(basic_config, TargetLookup(), path_mgr)

        # Act
        recovered_targets = new_rec_mgr.recover_targets_from_file()

        # Assert
        assert len(recovered_targets) == 2
        assert set(new_rec_mgr.target_lookup.keys()) == set(["T00", "T01"])

    def test__recover_named_file(
        self,
        rec_mgr: RecoveryManager,
        basic_config: dict,
        path_mgr: PathManager,
        t_fixed: Time,
    ):
        # Arrange
        t1 = Time(59990.0, format="mjd")
        rec_mgr.write_recovery_file(t_ref=t1)
        early_file = rec_mgr.path_manager.get_current_recovery_file(t_ref=t1)
        rec_mgr.target_lookup.pop("T01")
        rec_mgr.write_recovery_file(t_ref=t_fixed)
        later_file = rec_mgr.path_manager.get_current_recovery_file(t_ref=t_fixed)
        # Need new rec_mgr to test tlookup is correctly populated...
        rec_mgr_1 = RecoveryManager(basic_config, TargetLookup(), path_mgr)
        rec_mgr_2 = RecoveryManager(basic_config, TargetLookup(), path_mgr)

        # Act
        rec_targets1 = rec_mgr_1.recover_targets_from_file(recovery_file=early_file)
        rec_targets2 = rec_mgr_2.recover_targets_from_file()

        # Assert
        assert set(rec_mgr_1.target_lookup.keys()) == set(["T00", "T01"])
        assert set(rec_mgr_2.target_lookup.keys()) == set(["T00"])

    def test__no_files_warns(self, basic_config: dict, path_mgr: PathManager):
        # Arrange
        new_rec_mgr = RecoveryManager(basic_config, TargetLookup(), path_mgr)

        # Act
        with pytest.warns(MissingFileWarning):
            recovered_targets = new_rec_mgr.recover_targets_from_file()

        # Assert
        recovered_targets is None

    def test__malformed_file_warns(
        self, basic_config: dict, path_mgr: PathManager, t_fixed: Time
    ):
        # Arrange
        new_rec_mgr = RecoveryManager(basic_config, TargetLookup(), path_mgr)
        rec_filepath = new_rec_mgr.path_manager.get_current_recovery_file(t_ref=t_fixed)
        rec_filepath.touch()

        # Act
        with pytest.warns(MissingFileWarning):
            recovered_targets = new_rec_mgr.recover_targets_from_file()

        # Assert
        recovered_targets is None

    def test__missing_named_file_raises(
        self, basic_config: dict, path_mgr: PathManager, t_fixed: Time
    ):
        # Arrange
        new_rec_mgr = RecoveryManager(basic_config, TargetLookup(), path_mgr)
        rec_filepath = new_rec_mgr.path_manager.get_current_recovery_file(t_ref=t_fixed)
        assert not rec_filepath.exists()

        # Act
        with pytest.raises(FileNotFoundError):
            new_rec_mgr.recover_targets_from_file(recovery_file=rec_filepath)
