import pytest

import numpy as np

import pandas as pd

from astropy.table import Table
from astropy.time import Time

from aas2rto.query_managers.tns.tns_client import (
    TNSClient,
    TNSClientError,
    TNSClientWarning,
)


@pytest.fixture
def tclient():
    return TNSClient("test_user", 1234, query_sleep_time=0.0)


class Test__StaticProcessRespone:
    # This also tests that requests.post and zipfile.ZipFile are correctly patched

    def test__process_to_pandas(self, mock_daily_delta: str):
        # Arrange
        bytes_data = mock_daily_delta.encode("utf-8")

        # Act
        df = TNSClient.process_zip_content(bytes_data, return_type="pandas")

        # Assert
        assert isinstance(df, pd.DataFrame)

        exp_columns = (
            "objid name_prefix name type ra declination redshift "
            "internal_names reporters discoverydate lastmodified"
        ).split()
        assert set(df.columns) == set(exp_columns)

        assert len(df) == 3
        assert df.iloc[0]["name"] == "2023J"
        assert isinstance(df.iloc[0]["redshift"], float)  # Not broken by missing vals
        assert np.isclose(df.iloc[0]["redshift"], 0.01)  # in other rows

        assert np.isnan(df.iloc[1]["redshift"])  # Empty str is converted ok

        assert df.iloc[2]["reporters"] == "A. Aa, B. Bb"  # str w/ commas parsed ok

    def test__process_to_empty_pandas(self):
        # Arrange
        bytes_data = "2023-02-23 00:00 - 23:59".encode()  # header line ONLY

        # Act
        df = TNSClient.process_zip_content(bytes_data, return_type="pandas")

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(["name", "lastmodified"])
        assert df.empty

    def test__process_to_astropy(self, mock_daily_delta: str):
        # Arrange
        bytes_data = mock_daily_delta.encode("utf-8")

        # Act
        tab = TNSClient.process_zip_content(bytes_data, return_type="astropy")

        # Assert
        assert isinstance(tab, Table)

        exp_columns = (
            "objid name_prefix name type ra declination redshift "
            "internal_names reporters discoverydate lastmodified"
        ).split()
        assert set(tab.columns) == set(exp_columns)

        assert len(tab) == 3
        assert tab[0]["name"] == "2023J"
        assert isinstance(tab[0]["redshift"], float)  # Not broken by missing vals
        assert np.isclose(tab[0]["redshift"], 0.01)  # in other rows

        print(tab[1]["redshift"])
        assert np.ma.is_masked(tab[1]["redshift"])  # Empty str is converted ok

        assert tab[2]["reporters"] == "A. Aa, B. Bb"  # str w/ commas parsed ok

    def test__process_to_empty_astropy(self):
        # Arrange
        bytes_data = "2023-02-23 00:00 - 23:59".encode()  # header line ONLY

        # Act
        tab = TNSClient.process_zip_content(bytes_data, return_type="astropy")

        # Assert
        assert isinstance(tab, Table)
        assert set(tab.columns) == set(["name", "lastmodified"])
        assert len(tab) == 0


class Test__StaticCheckReturnType:
    def test__bad_type(self):
        # Act
        with pytest.raises(TNSClientError):
            TNSClient.check_return_type("bad_type")


class Test__SubmitPublicObjectRequest:
    def test__bad_request(self, tclient: TNSClient):
        # Arrange
        url = f"{TNSClient.tns_public_objects_url}/bad_request.csv.zip"

        # Act
        with pytest.warns(TNSClientWarning):
            resp = tclient.request_delta(url, process=False)


class Test__GetEmptyDeltaResults:
    def test__empty_pandas(self):
        # Act
        results = TNSClient.get_empty_delta_results(return_type="pandas")

        # Assert
        assert isinstance(results, pd.DataFrame)
        assert results.empty

    def test__empty_astropy(self):
        # Act
        results = TNSClient.get_empty_delta_results(return_type="astropy")

        # Assert
        assert isinstance(results, Table)
        assert len(results) == 0

    def test__empty_records(self):
        # Act
        results = TNSClient.get_empty_delta_results(return_type="records")

        # Assert
        assert isinstance(results, list)
        assert len(results) == 0


class Test__GetDailyDelta:
    def test__return_pandas(self, tclient: TNSClient):
        # Arrange
        t_ref = Time("2023-02-24 00:00:00")  # The day before t_fixed

        # Act
        df = tclient.get_tns_daily_delta(t_ref, return_type="pandas")

        # Assert
        assert isinstance(df, pd.DataFrame)

        exp_columns = (
            "objid name_prefix name type ra declination redshift "
            "internal_names reporters discoverydate lastmodified"
        ).split()
        assert set(df.columns) == set(exp_columns)

    def test__return_empty_pandas(self, tclient: TNSClient):
        # Arrange
        t_ref = Time("2023-02-23 00:00:00")  # TWO days before t_fixed

        # Act
        df = tclient.get_tns_daily_delta(t_ref, return_type="pandas")

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(["name", "lastmodified"])
        assert df.empty

    def test__return_astropy(self, tclient: TNSClient):
        # Arrange
        t_ref = Time("2023-02-24 00:00:00")  # The day before t_fixed

        # Act
        tab = tclient.get_tns_daily_delta(t_ref, return_type="astropy")

        # Assert
        assert isinstance(tab, Table)

        exp_columns = (
            "objid name_prefix name type ra declination redshift "
            "internal_names reporters discoverydate lastmodified"
        ).split()
        assert set(tab.columns) == set(exp_columns)

    def test__return_empty_astropy(self, tclient: TNSClient):
        # Arrange
        t_ref = Time("2023-02-23 00:00:00")  # TWO days before t_fixed

        # Act
        tab = tclient.get_tns_daily_delta(t_ref, return_type="astropy")

        # Assert
        assert isinstance(tab, Table)
        assert set(tab.columns) == set(["name", "lastmodified"])
        assert len(tab) == 0

    def test__return_records(self, tclient: TNSClient):
        # Arrange
        t_ref = Time("2023-02-24 00:00:00")  # The day before t_fixed

        # Act
        records = tclient.get_tns_daily_delta(t_ref, return_type="records")

        # Assert
        assert isinstance(records, list)
        assert isinstance(records[0], dict)

        exp_columns = (
            "objid name_prefix name type ra declination redshift "
            "internal_names reporters discoverydate lastmodified"
        ).split()
        assert set(records[0].keys()) == set(exp_columns)

    def test__return_unprocessed(self, tclient: TNSClient):
        # Arrange
        t_ref = Time("2023-02-24 00:00:00")  # The day before t_fixed

        # Act
        resp = tclient.get_tns_daily_delta(t_ref, process=False)

        # Assert
        assert hasattr(resp, "content")


class Test__GetHourlyDelta:

    # TNSClient does NOT delete old hourly deltas. That happens in TNS QM

    def test__return_pandas(self, tclient: TNSClient):
        # Act
        df = tclient.get_tns_hourly_delta(0, return_type="pandas")

        # Assert
        assert isinstance(df, pd.DataFrame)
        exp_columns = (
            "objid name_prefix name type ra declination redshift "
            "internal_names reporters discoverydate lastmodified"
        ).split()
        assert set(df.columns) == set(exp_columns)

    def test__return_empty_pandas(self, tclient: TNSClient):
        # Act
        df = tclient.get_tns_hourly_delta(1, return_type="pandas")

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(["name", "lastmodified"])
        assert df.empty

    def test__return_astropy(self, tclient: TNSClient):
        # Act
        tab = tclient.get_tns_hourly_delta(0, return_type="astropy")

        # Assert
        assert isinstance(tab, Table)
        exp_columns = (
            "objid name_prefix name type ra declination redshift "
            "internal_names reporters discoverydate lastmodified"
        ).split()
        assert set(tab.columns) == set(exp_columns)

    def test__return_empty_astropy(self, tclient: TNSClient):
        # Act
        tab = tclient.get_tns_hourly_delta(1, return_type="astropy")

        # Assert
        assert isinstance(tab, Table)
        assert set(tab.columns) == set(["name", "lastmodified"])
        assert len(tab) == 0

    def test__return_records(self, tclient: TNSClient):
        # Act
        records = tclient.get_tns_hourly_delta(0, return_type="records")

        # Assert
        assert isinstance(records, list)
        assert isinstance(records[0], dict)

    def test__return_unprocessed(self, tclient: TNSClient):
        # Act
        resp = tclient.get_tns_hourly_delta(0, process=False)

        # Assert
        assert hasattr(resp, "content")
