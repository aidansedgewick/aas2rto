import pytest
from pathlib import Path

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.time import Time

from aas2rto.target_lookup import TargetLookup

from aas2rto.query_managers.tns.tns import TNSQueryManager, TNSCredentialError
from aas2rto.query_managers.tns.tns_client import TNSClient, TNSClientError


@pytest.fixture
def tns_qm(tns_config: dict, tlookup: TargetLookup, tmp_path: Path):
    tns_config["query_sleep_time"] = 0.0
    return TNSQueryManager(tns_config, tlookup, parent_path=tmp_path)


@pytest.fixture
def write_mock_daily_delta(
    daily_delta_rows: list[list], tns_delta_columns: list[str], tns_qm: TNSQueryManager
):
    df = pd.DataFrame(daily_delta_rows, columns=tns_delta_columns)

    t0 = Time("2023-02-24 23:59:00")
    fpath = tns_qm.get_daily_delta_filepath(t0)

    df.to_csv(fpath, index=False)


@pytest.fixture
def write_mock_hourly_delta(
    hourly_delta_rows: list[list], tns_delta_columns: list[str], tns_qm: TNSQueryManager
):
    df = pd.DataFrame(hourly_delta_rows, columns=tns_delta_columns)

    t0 = Time("2023-02-25 00:59:00")
    fpath = tns_qm.get_hourly_delta_filepath(t0)

    df.to_csv(fpath, index=False)


class Test__InitQM:
    def test__normal(self, tns_config: dict, tlookup: TargetLookup, tmp_path: Path):
        # Act
        qm = TNSQueryManager(tns_config, tlookup, parent_path=tmp_path)

        # Assert
        assert set(qm.paths_lookup) == set(["query_results"])
        exp_path = tmp_path / "tns/query_results"
        assert qm.paths_lookup["query_results"] == exp_path
        assert exp_path.exists()

        # See that
        assert isinstance(qm.config["sep_limit"], u.Quantity)
        qm.config["sep_limit"].to(u.arcsec)  # Will fail if unconvertible

    def test__missing_credentials(
        self, tns_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Arrange
        tns_config.pop("credentials")

        # Act
        with pytest.raises(TNSCredentialError):
            qm = TNSQueryManager(tns_config, tlookup, tmp_path)

    def test__missing_credential_key(
        self, tns_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Arrange
        tns_config["credentials"].pop("user")

        # Act
        with pytest.raises(TNSCredentialError):
            qm = TNSQueryManager(tns_config, tlookup, tmp_path)


class Test__FilePathMethods:

    @pytest.fixture(autouse=True)
    def _remove_empty_tmp_dirs(self, remove_tmp_dirs: None) -> None:
        pass  # remove_tmp_dirs defined in unit/contfest.py, executed with autouse=True

    def test__daily_delta_filepath(self, tns_qm: TNSQueryManager):
        # Arrange
        t0 = Time("2025-11-01 12:34:56")

        # Act
        fpath = tns_qm.get_daily_delta_filepath(t0)

        # Assert
        assert fpath.name == "tns_delta_2025-11-01.csv"
        assert not fpath.exists()

    def test__hourly_delta_filepath(self, tns_qm: TNSQueryManager):
        # Arrange
        t0 = Time("2025-11-01 12:34:56")

        # Act
        fpath = tns_qm.get_hourly_delta_filepath(t0)

        # Assert
        assert fpath.stem == "tns_delta_2025-11-01_12h"
        assert not fpath.exists()


class Test__LoadExistingResults:

    def test__no_results(self, tns_qm: TNSQueryManager):
        # Act
        tns_qm.load_existing_tns_results()

        # Assert
        assert isinstance(tns_qm.tns_results, pd.DataFrame)
        assert tns_qm.tns_results.empty

    def test__existing_daily(
        self, write_mock_daily_delta: None, tns_qm: TNSQueryManager
    ):
        # Act
        tns_qm.load_existing_tns_results()

        # Assert
        assert len(tns_qm.tns_results) == 3
        assert tns_qm.tns_results.iloc[1]["name"] == "2023K"
        assert tns_qm.tns_results.iloc[1]["lastmodified"] == "2023-02-24 13:30:00"

    def test__existing_daily_and_hourly(
        self,
        write_mock_daily_delta: None,
        write_mock_hourly_delta: None,
        tns_qm: TNSQueryManager,
    ):
        # Act
        tns_qm.load_existing_tns_results()

        # Assert
        assert len(tns_qm.tns_results) == 4
        assert tns_qm.tns_results.iloc[1]["name"] == "2023K"
        assert tns_qm.tns_results.iloc[1]["lastmodified"] == "2023-02-25 00:20:00"


class Test__PerformAllTasks:
    def test__iter0_no_existing(self, tns_qm: TNSQueryManager):
        # Act
        tns_qm.perform_all_tasks(iteration=0)

        # Assert
        T00 = tns_qm.target_lookup["T00"]
        assert "tns" not in T00.target_data.keys()

    def test__iter0_with_daily_existing(
        self, write_mock_daily_delta: None, tns_qm: TNSQueryManager
    ):
        # Act
        tns_qm.perform_all_tasks(iteration=0)

        # Assert
        T00 = tns_qm.target_lookup["T00"]
        T01 = tns_qm.target_lookup["T01"]
        assert "tns" in T00.target_data.keys()
        T00_tns_data = T00.target_data["tns"]
        assert T00.alt_ids["tns"] == "2023J"
        assert T00.alt_ids["ztf"] == "ZTF00A"

        assert T01.alt_ids["tns"] == "2023K"
        assert T01.alt_ids["ztf"] == "ZTF00B"

    def test__non_iter0(self, tns_qm: TNSQueryManager):
        # Arrange
        tns_qm.config["delta_lookback"] = 2  # No point in writing 14 empty files
        t0 = Time("2023-02-25 01:59:59")  # Get "full" hourly delta and one empty.

        # Act
        tns_qm.perform_all_tasks(iteration=1, t_ref=t0)

        # Assert
        # Did we collect all the missing deltas?
        exp_tstamps = ["2023-02-24", "2023-02-23", "2023-02-25_00h", "2023-02-25_01h"]
        exp_filestems = [f"tns_delta_{x}" for x in exp_tstamps]
        found_filestems = [
            f.stem for f in tns_qm.paths_lookup["query_results"].glob("tns_delta*")
        ]
        assert set(found_filestems) == set(exp_filestems)

        # Did we do the matching correctly?
        T00 = tns_qm.target_lookup["T00"]
        T01 = tns_qm.target_lookup["T01"]
        assert "tns" in T00.target_data.keys()
        T00_tns_data = T00.target_data["tns"]
        assert T00.alt_ids["tns"] == "2023J"
        assert T00.alt_ids["ztf"] == "ZTF00A"

        assert T01.alt_ids["tns"] == "2023K"
        assert T01.alt_ids["ztf"] == "ZTF00B"
