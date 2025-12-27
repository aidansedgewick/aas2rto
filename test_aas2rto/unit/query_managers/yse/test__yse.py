import os
import pytest
from pathlib import Path
from typing import NoReturn

import pandas as pd

from astropy import units as u
from astropy.time import Time

from aas2rto.exc import MissingKeysError, UnexpectedKeysError
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.query_managers.yse.yse import YSEQueryManager, get_yse_id_from_target
from aas2rto.query_managers.yse.yse_client import YSEClient, YSEClientError
from aas2rto.target import Target
from aas2rto.target_data import TargetData
from aas2rto.target_lookup import TargetLookup


@pytest.fixture
def yse_qm(yse_config: dict, tlookup: TargetLookup, tmp_path: Path):
    tlookup["T00"].alt_ids["tns"] = "2023J"
    tlookup.update_target_id_mappings()
    return YSEQueryManager(yse_config, tlookup, tmp_path)


@pytest.fixture
def existing_query_results_filepath(yse_query_name: str, tmp_path: Path):
    results_dir = tmp_path / f"yse/query_results"
    results_dir.mkdir(exist_ok=True, parents=True)
    return results_dir / f"{yse_query_name}.csv"


@pytest.fixture
def write_existing_explorer_results(
    existing_query_results_filepath: Path, yse_explorer_columns: list[str]
):

    rows = [
        ["2023J", "SN", 180.0, 0.0, 5],  # ndet=5: same as what query
        ["2023K", "", 90.0, 30.0, 4],  # query will return ndet=5: updated!
        ["2023L", "", 355.0, -80.0, 5],  # NOT included in new query results
        # query will also return an extra row 2023M
    ]
    results = pd.DataFrame(rows, columns=yse_explorer_columns)
    results.to_csv(existing_query_results_filepath, index=False)


@pytest.fixture
def existing_lightcurve_filepath(tmp_path: Path):
    lc_dir = tmp_path / f"yse/lightcurves"
    lc_dir.mkdir(exist_ok=True, parents=True)
    return lc_dir / "2023J.csv"


@pytest.fixture
def write_existing_lightcurve(existing_lightcurve_filepath: Path, yse_lc: pd.DataFrame):
    yse_lc.to_csv(existing_lightcurve_filepath, index=False)
    return None


class Test__GetYSEId:
    def test__yse_only(self, basic_target: Target):
        # Arrange
        assert set(basic_target.alt_ids.keys()) == set(["src01", "src02"])  # a reminder
        basic_target.alt_ids["yse"] = "YSE_01"

        # Act
        yse_id = get_yse_id_from_target(basic_target)

        # Assert
        assert yse_id == "YSE_01"

    def test__prefer_tns(self, basic_target: Target):
        # Arrange
        assert set(basic_target.alt_ids.keys()) == set(["src01", "src02"])  # a reminder
        basic_target.alt_ids["yse"] = "YSE_01"
        basic_target.alt_ids["tns"] = "2023J"

        # Act
        yse_id = get_yse_id_from_target(basic_target)

        # Assert
        assert yse_id == "2023J"

    def test__neither(self, basic_target: Target):
        # Arrange
        assert set(basic_target.alt_ids.keys()) == set(["src01", "src02"])  # a reminder

        # Act
        yse_id = get_yse_id_from_target(basic_target)

        # Assert
        assert yse_id is None


class Test__YSEQMInit:
    def test__init_valid_config(
        self, yse_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Act
        qm = YSEQueryManager(yse_config, tlookup, tmp_path)

        # Assert
        assert isinstance(qm, BaseQueryManager)
        assert isinstance(qm.yse_client, YSEClient)

    def test__bad_keyword_raises(
        self,
        yse_config: dict,
        tlookup: TargetLookup,
        tmp_path: Path,
        remove_tmp_dirs: NoReturn,
    ):
        # Arrange
        yse_config["bad_key"] = 100.0

        # Act
        with pytest.raises(UnexpectedKeysError):
            qm = YSEQueryManager(yse_config, tlookup, tmp_path)

    def test__missing_credential_raises(
        self,
        yse_config: dict,
        tlookup: TargetLookup,
        tmp_path: Path,
        remove_tmp_dirs: NoReturn,
    ):
        # Arrange
        yse_config.pop("credentials")

        # Act
        with pytest.raises(YSEClientError):
            qm = YSEQueryManager(yse_config, tlookup, tmp_path)

    def test__credentials_missing_key_raises(
        self,
        yse_config: dict,
        tlookup: TargetLookup,
        tmp_path: Path,
        remove_tmp_dirs: NoReturn,
    ):
        # Arrange
        yse_config["credentials"].pop("username")

        # Act
        with pytest.raises(MissingKeysError):
            qm = YSEQueryManager(yse_config, tlookup, tmp_path)

    def test__malformed_explorer_query_raises(
        self,
        yse_config: dict,
        tlookup: TargetLookup,
        tmp_path: Path,
        remove_tmp_dirs: NoReturn,
    ):
        # Arrange
        yse_config["explorer_queries"]["bad_query"] = {"blah": 100.0}

        # Act
        with pytest.raises(MissingKeysError):
            qm = YSEQueryManager(yse_config, tlookup, tmp_path)

    def test__bad_type_query_raises(
        self,
        yse_config: dict,
        tlookup: TargetLookup,
        tmp_path: Path,
        remove_tmp_dirs: NoReturn,
    ):
        # Arrange
        yse_config["explorer_queries"]["bad_query"] = 102  # Should be a dict!

        # Act
        with pytest.raises(TypeError):
            qm = YSEQueryManager(yse_config, tlookup, tmp_path)


class Test__GetPathMethods:
    def test__query_results(
        self, yse_qm: YSEQueryManager, tmp_path: Path, remove_tmp_dirs: NoReturn
    ):
        # Act
        filepath = yse_qm.get_query_results_filepath("test_query")

        # Assert
        exp_path = tmp_path / "yse/query_results/test_query.csv"
        assert filepath == exp_path

    def test__yse_lightcurve_filepath(
        self, yse_qm: YSEQueryManager, tmp_path: Path, remove_tmp_dirs: NoReturn
    ):
        # Act
        filepath = yse_qm.get_lightcurve_filepath("2025xyz")

        # Assert
        exp_path = tmp_path / "yse/lightcurves/2025xyz.csv"
        assert filepath == exp_path


class Test__QueryExplorers:

    def test__no_exising_results(
        self,
        yse_qm: YSEQueryManager,
        existing_query_results_filepath: Path,
    ):
        # Act
        records = yse_qm.explorer_queries()  # t_now

        # Assert
        assert len(records) == 3
        returned_names = set([r["yse_id"] for r in records])
        qnames = set(r["query_name"] for r in records)
        assert len(records) == 3
        assert set(returned_names) == set(["2023J", "2023K", "2023M"])

        # now check results were written
        assert existing_query_results_filepath.exists()
        results = pd.read_csv(existing_query_results_filepath)
        assert len(results) == 3

    def test__compare_with_existing_results(
        self,
        yse_qm: YSEQueryManager,
        write_existing_explorer_results: NoReturn,
        existing_query_results_filepath: Path,
        tmp_path: Path,
    ):
        # Arrange
        t_future = Time.now() + 2.0 * u.day

        # Act
        records = yse_qm.explorer_queries(t_ref=t_future)
        # use t_future so query is not skipped bc. "newly" written explorer_results

        # Assert
        # check the correct diff was returned
        returned_names = set([r["yse_id"] for r in records])
        qnames = set(r["query_name"] for r in records)
        assert len(records) == 2
        assert set(returned_names) == set(["2023K", "2023M"])
        assert set(qnames) == set(["yse_test_query"])
        # see test_aas2rto/conftest: 'yse_query_name'

        # now check that the combined reuslts were written correctly
        assert existing_query_results_filepath.exists()
        results = pd.read_csv(existing_query_results_filepath)
        assert len(results) == 4
        print(results)
        assert set(results["name"]) == set(["2023J", "2023K", "2023L", "2023M"])

    def test__no_requery_recent_results(
        self,
        yse_qm: YSEQueryManager,
        write_existing_explorer_results: NoReturn,
    ):
        # Act
        records = yse_qm.explorer_queries()  # t_now

        # Assert
        assert len(records) == 0


class Test__NewTargetsFromRecord:
    def test__add_new_target(self, yse_qm: YSEQueryManager, t_fixed: Time):
        # Arrange
        record = {"yse_id": "YSE_01", "ra": 225.0, "dec": -45.0}

        # Act
        added, skipped = yse_qm.new_targets_from_query_records([record], t_ref=t_fixed)

        # Assert
        assert set(added) == set(["YSE_01"])
        assert set(skipped) == set()
        assert set(yse_qm.target_lookup.keys()) == set(["T00", "T01", "YSE_01"])

    def test__skip_existing_target(self, yse_qm: YSEQueryManager, t_fixed):
        # Arrange
        record = {"yse_id": "T00", "ra": 315.0, "dec": -45.0}

        # Act
        added, skipped = yse_qm.new_targets_from_query_records([record], t_ref=t_fixed)

        # Assert
        assert set(added) == set()
        assert set(skipped) == set(["T00"])
        assert set(yse_qm.target_lookup.keys()) == set(["T00", "T01"])

    def test__repeated_record(self, yse_qm: YSEQueryManager, t_fixed: Time):
        # Arrange
        records = [
            {"yse_id": "YSE_01", "ra": 225.0, "dec": -45.0},
            {"yse_id": "YSE_01", "ra": 225.0, "dec": -45.0},
        ]

        # Act
        added, skipped = yse_qm.new_targets_from_query_records(records, t_ref=t_fixed)

        # Assert
        assert set(added) == set(["YSE_01"])
        assert set(skipped) == set()  # "YSE_01" not in added and skipped
        assert set(yse_qm.target_lookup.keys()) == set(["T00", "T01", "YSE_01"])

    def test__add_info_to_records(self, yse_qm: YSEQueryManager, t_fixed: Time):
        # Arrange
        records = [
            {"yse_id": "T00", "query_name": "query_A"},
            {"yse_id": "T00", "query_name": "query_B"},
        ]

        # Act
        yse_qm.apply_update_messages(records, t_ref=t_fixed)

        # Assert
        T00 = yse_qm.target_lookup["T00"]
        assert len(T00.info_messages) == 2
        assert "query_A" in T00.info_messages[0]
        assert "query_B" in T00.info_messages[1]


class Test__GetLightcurvesToQuery:
    def test__get_lcs_to_query_no_lc(self, yse_qm: YSEQueryManager, t_fixed: Time):
        # Arrange
        t_future = t_fixed + 2.0 * u.day

        # Act
        to_query = yse_qm.get_lightcurves_to_query(t_ref=t_future)

        # Assert
        assert set(to_query) == set(["2023J"])  # NOT T01, as it has no tns/yse id

    def test__get_lcs_to_query_old_lc(
        self,
        yse_qm: YSEQueryManager,
        write_existing_lightcurve: NoReturn,
        existing_lightcurve_filepath: Path,
        t_fixed: Time,
    ):
        # Arrange
        t_future = t_fixed + 2.0 * u.day
        yse_qm.config["lightcurve_update_interval"] = 1.0
        os.utime(existing_lightcurve_filepath, (t_fixed.unix, t_fixed.unix))

        # Act
        to_query = yse_qm.get_lightcurves_to_query(t_ref=t_future)

        # Assert
        assert set(to_query) == set(["2023J"])  # NOT T01, as it has no tns/yse_id

    def test__skip_recently_updated(
        self,
        yse_qm: YSEQueryManager,
        write_existing_lightcurve: NoReturn,
        existing_lightcurve_filepath: Path,
        t_fixed: Time,
    ):
        # Arrange
        t_future = t_fixed + 0.5 * u.day
        yse_qm.config["lightcurve_update_interval"] = 1.0
        os.utime(existing_lightcurve_filepath, (t_fixed.unix, t_fixed.unix))

        # Act
        to_query = yse_qm.get_lightcurves_to_query(t_ref=t_future)

        # Assert
        assert set(to_query) == set()  # NOT T01, as it has no tns/yse_id


class Test__QueryLightcurves:
    def test__no_existing(self, yse_qm: YSEQueryManager, t_fixed: Time):
        # Arrange
        t_future = t_fixed + 2.0 * u.day

        # Act
        success, failed = yse_qm.query_lightcurves(t_ref=t_future)

        # Assert
        assert set(success) == set(["2023J"])

    def test__requery_old_lcs(
        self,
        yse_qm: YSEQueryManager,
        write_existing_lightcurve: NoReturn,
        existing_lightcurve_filepath: Path,
        t_fixed: Time,
    ):
        # Arrange
        t_future = t_fixed + 2.0 * u.day
        yse_qm.config["lightcurve_update_interval"] = 1.0
        os.utime(existing_lightcurve_filepath, (t_fixed.unix, t_fixed.unix))

        # Act
        success, failed = yse_qm.query_lightcurves(t_ref=t_future)

        # Assert
        assert set(success) == set(["2023J"])  # NOT T01, as it has no tns/yse_id

    def test__skip_recent_lcs(
        self,
        yse_qm: YSEQueryManager,
        write_existing_lightcurve: NoReturn,
        existing_lightcurve_filepath: Path,
        t_fixed: Time,
    ):
        # Arrange
        t_future = t_fixed + 0.5 * u.day
        yse_qm.config["lightcurve_update_interval"] = 1.0
        os.utime(existing_lightcurve_filepath, (t_fixed.unix, t_fixed.unix))

        # Act
        success, failed = yse_qm.query_lightcurves(t_ref=t_future)

        # Assert
        assert set(success) == set(["2023J"])  # NOT T01, as it has no tns/yse_id

    def test__query_named_target_anyway(
        self,
        yse_qm: YSEQueryManager,
        write_existing_lightcurve: NoReturn,
        existing_lightcurve_filepath: Path,
        t_fixed: Time,
    ):
        # Arrange
        t_future = t_fixed + 2.0 * u.day
        yse_qm.config["lightcurve_update_interval"] = 1.0
        os.utime(existing_lightcurve_filepath, (t_fixed.unix, t_fixed.unix))

        # Act
        success, failed = yse_qm.query_lightcurves(id_list=["2023J"], t_ref=t_future)

        # Assert
        assert set(success) == set(["2023J"])

    def test__quit_after_max_failures(self, yse_qm: YSEQueryManager, t_fixed: Time):
        # Arrange
        t_future = t_fixed + 2.0 * u.day
        yse_qm.config["max_failed_queries"] = 3
        to_query = ["fail_01", "fail_02", "fail_03", "fail_04", "T00"]

        # Act
        success, failed = yse_qm.query_lightcurves(id_list=to_query, t_ref=t_future)

        # Assert
        assert len(failed) == 3
        assert set(failed) == set(["fail_01", "fail_02", "fail_03"])
        assert set(success) == set()

    def test__quit_after_max_time(self, yse_qm: YSEQueryManager, t_fixed: Time):
        # Arrange
        t_future = t_fixed + 2.0 * u.day
        yse_qm.config["max_query_time"] = 0.3
        to_query = ["sleep_01", "sleep_02", "sleep_03"]

        # Act
        success, failed = yse_qm.query_lightcurves(id_list=to_query, t_ref=t_future)

        # Assert
        assert set(success) == set(["sleep_01", "sleep_02"])


class Test__LoadSingleLightcurve:
    def test__load_existing(
        self,
        yse_qm: YSEQueryManager,
        write_existing_lightcurve: NoReturn,
    ):
        # Act
        lc = yse_qm.load_single_lightcurve("T00")  # T00 corresponds to 2023J

        # Assert
        assert isinstance(lc, pd.DataFrame)
        assert len(lc) == 6

    def test__load_missing(
        self,
        yse_qm: YSEQueryManager,
    ):
        # Act
        lc = yse_qm.load_single_lightcurve("T01")

        # Assert
        assert lc is None


class Test__LoadAllLightcurves:
    def test__load_all(
        self,
        yse_qm: YSEQueryManager,
        write_existing_lightcurve: NoReturn,
    ):
        # Act
        loaded, skipped, missing = yse_qm.load_target_lightcurves()

        # Assert
        T00 = yse_qm.target_lookup["T00"]
        assert set(loaded) == set(["T00"])  # corresponds to 2023J
        assert set(missing) == set(["T01"])  # T01 has no TNS/YSE id

        assert "yse" in T00.target_data
        td = T00.target_data["yse"]
        assert isinstance(td, TargetData)

        assert isinstance(td.lightcurve, pd.DataFrame)
        assert len(td.lightcurve) == 6


class Test__PerformAllTasks:
    def test__iteration0(
        self,
        yse_qm: YSEQueryManager,
        write_existing_lightcurve: NoReturn,
        t_fixed: Time,
    ):
        # Act
        yse_qm.perform_all_tasks(iteration=0, t_ref=t_fixed)

        # Assert
        T00 = yse_qm.target_lookup["T00"]

        assert "yse" in T00.target_data
        td = T00.target_data["yse"]
        assert isinstance(td, TargetData)

        assert isinstance(td.lightcurve, pd.DataFrame)
        assert len(td.lightcurve) == 6

    def test__non_zero_iteration(self, yse_qm: YSEQueryManager, t_fixed: Time):
        # Arrange
        print(yse_qm.target_lookup.id_mapping)
        assert yse_qm.target_lookup["2023J"].target_id == "T00"

        # Act
        yse_qm.perform_all_tasks(iteration=1, t_ref=t_fixed)

        # Assert
        exp_lc_path = yse_qm.parent_path / "yse/lightcurves/2023J.csv"
        assert exp_lc_path.exists()

        exp_qresults_path = yse_qm.parent_path / "yse/query_results/yse_test_query.csv"
        assert exp_qresults_path.exists()

        exp_target_ids = "T00 T01 2023K 2023M".split()  # T00 already has alt_id 2023J
        assert set(yse_qm.target_lookup.keys()) == set(exp_target_ids)
