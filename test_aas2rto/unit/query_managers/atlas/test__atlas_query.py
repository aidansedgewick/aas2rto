import pytest
import requests

import pandas as pd

from astropy.table import Table

from aas2rto.query_managers.atlas.atlas_query import AtlasQuery, AtlasQueryWarning


# Monkeypatch of AtlasQuery is in test_aa2rto/unit/query_managers/atlas/conftest.py


@pytest.fixture
def mock_headers():
    return AtlasQuery.build_headers(token="1234")


##===== Testing starts here =====##


class Test__InitClass:
    def test__atlas_query_init(self):
        # Act
        q = AtlasQuery()


class Test__AtlasQueryStatics:

    def test__build_headers(self):
        # Act
        headers = AtlasQuery.build_headers("my_token")

        # Assert
        assert isinstance(headers, dict)
        assert set(headers.keys()) == set(["Authorization", "Accept"])

    def test__raise_return_type(self):
        # Act
        with pytest.raises(ValueError):
            AtlasQuery.raise_return_type("bad_type")  # basically


class Test__GetToken:
    def test__get_token(self):
        # Act
        token = AtlasQuery.get_atlas_token(username="user", password="shh_its_a_secret")

        # Assert
        assert token == 1234  # Basically testing mock and no typos.


class Test__EmptyLC:
    def test__empty_lc_pandas(self):
        # Act
        lc = AtlasQuery.get_empty_lightcurve(return_type="pandas")

        # Assert
        assert isinstance(lc, pd.DataFrame)
        assert lc.empty

        cols_str = (
            "MJD m dm uJy duJy F err chi/N RA Dec x y maj min phi apfit mag5sig Sky Obs"
        )
        assert set(lc.columns) == set(cols_str.split())

    def test__empty_lc_astropy(self):
        # Act
        lc = AtlasQuery.get_empty_lightcurve(return_type="astropy")

        # Assert
        assert isinstance(lc, Table)
        assert len(lc) == 0

        cols_str = (
            "MJD m dm uJy duJy F err chi/N RA Dec x y maj min phi apfit mag5sig Sky Obs"
        )
        assert set(lc.columns) == set(cols_str.split())

    def test__empty_lc_records(self):
        # Act
        lc = AtlasQuery.get_empty_lightcurve(return_type="records")

        # Assert
        assert isinstance(lc, list)
        assert len(lc) == 0


class Test__SubmitQuery:
    def test__submit_query(self, mock_headers: dict):
        # Arrange
        data = dict(ra=90.0, dec=180.0, comment="target:test")

        # Act
        resp = AtlasQuery.submit_forced_photom_query(data, headers=mock_headers)

        # Assert
        assert resp.__class__.__name__ == "MockPostResponse"
        # Can't import from conftest
        assert resp.status_code == 201

    def test__bad_data_warns(self, mock_headers: dict):
        # Arrange
        data = dict(weird_kwarg=100.0)

        # Act
        with pytest.warns(AtlasQueryWarning):
            resp = AtlasQuery.submit_forced_photom_query(data, headers=mock_headers)

    def test__throttle(self):
        # Arrange
        data = dict(comment="T_throttle:test")

        # Act
        resp = AtlasQuery.submit_forced_photom_query(data, headers=mock_headers)

        # Assert
        assert resp.status_code == 429


class Test__GetExisting:
    def test__get_existing_no_url(self, mock_headers: dict):
        # Act
        existing = AtlasQuery.get_existing_queries(headers=mock_headers)

        # Assert
        assert isinstance(existing, dict)

        assert existing["next"].endswith("/?cursor_ABCDE")
        assert isinstance(existing["results"], list)
        assert len(existing["results"]) == 5
        assert isinstance(existing["results"][0], dict)
        assert existing["results"][0]["url"] == "http://atlas.mock/T00"

    def test__get_existing_url(self, mock_headers: dict):
        # Act
        existing = AtlasQuery.get_existing_queries(
            headers=mock_headers, url="https://atlas.mock/queue/?cursor_otherpage"
        )

        # Assert
        assert isinstance(existing, dict)

        assert existing["next"] is None  # We got the second page
        assert isinstance(existing["results"], list)
        assert len(existing["results"]) == 2
        assert isinstance(existing["results"][0], dict)
        assert existing["results"][0]["url"] == "http://atlas.mock/T01"


class Test__IterateExisting:
    def test__iterate_existing(self, mock_headers: dict):
        # Arrange
        result_list = []

        # Act
        for result in AtlasQuery.iterate_existing_queries(headers=mock_headers):
            result_list.append(result)

        # Assert
        assert len(result_list) == 7

        assert result_list[-1]["url"] == "http://atlas.mock/T01"
        assert result_list[0]["url"] == "http://atlas.mock/T_bad_query00"
        # remember first page returned in rev.

    def test__quit_after_timelimit(self, mock_headers: dict):

        # Arrange
        result_list = []

        # Act
        for result in AtlasQuery.iterate_existing_queries(
            headers=mock_headers, max_query_time=0.1
        ):
            result_list.append(result)

        # Assert
        assert len(result_list) == 5
        # Same as before, but we quit because of the 0.2 sec sleep in mocked AtlasQuery


class Test__RecoverTaskData:
    def test__success_pandas(self, mock_headers: dict):
        # Act
        status, lc = AtlasQuery.recover_task_data(
            "T_ready_00", mock_headers, return_type="pandas"
        )

        # Assert
        assert status == 200
        assert isinstance(lc, pd.DataFrame)
        assert not lc.empty

    def test__success_astropy(self, mock_headers: dict):
        # Act
        status, lc = AtlasQuery.recover_task_data(
            "T_ready_00", mock_headers, return_type="astropy"
        )

        # Assert
        assert status == 200
        assert isinstance(lc, Table)
        assert len(lc) > 0

    def test__still_waiting(self, mock_headers: dict):
        # Act
        status, lc = AtlasQuery.recover_task_data(
            "T00", mock_headers, return_type="pandas"
        )

        # Assert
        assert status == 201
        assert lc is None

    def test__no_data_pandas(self, mock_headers: dict):
        # Act
        status, lc = AtlasQuery.recover_task_data(
            "T_no_data", mock_headers, return_type="pandas"
        )

        # Assert
        assert status == 200
        assert isinstance(lc, pd.DataFrame)
        assert lc.empty

    def test__no_data_astropy(self, mock_headers: dict):
        # Act
        status, lc = AtlasQuery.recover_task_data(
            "T_no_data", mock_headers, return_type="astropy"
        )

        # Asset
        assert status == 200
        assert isinstance(lc, Table)
        assert len(lc) == 0

    def test__bad_return_type_raises(self, mock_headers: dict):
        # Act
        with pytest.raises(ValueError):
            status, lc = AtlasQuery.recover_task_data(
                "T_ready_00", mock_headers, return_type="blah"
            )


class Test__ProcessTaskData:
    def test__return_pandas(self, task_response_data: str):
        # Act
        lc = AtlasQuery.process_task_data(task_response_data, return_type="pandas")

        # Assert
        assert isinstance(lc, pd.DataFrame)

    def test__return_astropy(self, task_response_data: str):
        # Act
        lc = AtlasQuery.process_task_data(task_response_data, return_type="astropy")

        # Assert
        assert isinstance(lc, Table)

    def test__return_records(self, task_response_data: str):
        # Act
        lc = AtlasQuery.process_task_data(task_response_data, return_type="records")

        # Assert
        assert isinstance(lc, list)

    def test__bad_return_type_raises(self, task_response_data: str):
        # Act
        with pytest.raises(ValueError):
            lc = AtlasQuery.process_task_data(task_response_data, return_type="blah")
