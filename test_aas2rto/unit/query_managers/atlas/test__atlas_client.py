import pytest
import requests

import pandas as pd

from astropy.table import Table

from aas2rto.query_managers.atlas.atlas_client import AtlasClient, AtlasClientWarning


# Monkeypatch of AtlasClient is in test_aa2rto/unit/query_managers/atlas/conftest.py


@pytest.fixture
def atlas_client(atlas_credentials: dict):
    return AtlasClient(atlas_credentials["token"])


##===== Testing starts here =====##


class Test__InitClass:
    def test__atlas_client_init(self):
        # Act
        aq = AtlasClient("1234")


class Test__AtlasClientStatics:

    def test__build_headers(self):
        # Act
        headers = AtlasClient.build_atlas_headers("my_token")

        # Assert
        assert isinstance(headers, dict)
        assert set(headers.keys()) == set(["Authorization", "Accept"])

    def test__raise_return_type(self):
        # Act
        with pytest.raises(ValueError):
            AtlasClient.check_return_type("bad_type")  # basically


class Test__GetToken:
    def test__get_token(self):
        # Act
        token = AtlasClient.get_atlas_token(
            username="user", password="shh_its_a_secret"
        )

        # Assert
        assert token == 1234  # Basically testing mock and no typos.


class Test__EmptyLC:
    def test__empty_lc_pandas(self):
        # Act
        lc = AtlasClient.get_empty_lightcurve(return_type="pandas")

        # Assert
        assert isinstance(lc, pd.DataFrame)
        assert lc.empty

        cols_str = (
            "MJD m dm uJy duJy F err chi/N RA Dec x y maj min phi apfit mag5sig Sky Obs"
        )
        assert set(lc.columns) == set(cols_str.split())

    def test__empty_lc_astropy(self):
        # Act
        lc = AtlasClient.get_empty_lightcurve(return_type="astropy")

        # Assert
        assert isinstance(lc, Table)
        assert len(lc) == 0

        cols_str = (
            "MJD m dm uJy duJy F err chi/N RA Dec x y maj min phi apfit mag5sig Sky Obs"
        )
        assert set(lc.columns) == set(cols_str.split())

    def test__empty_lc_records(self):
        # Act
        lc = AtlasClient.get_empty_lightcurve(return_type="records")

        # Assert
        assert isinstance(lc, list)
        assert len(lc) == 0


class Test__ProcessTaskData:
    def test__return_pandas(self, task_response_data: str):
        # Act
        lc = AtlasClient.process_task_data(task_response_data, return_type="pandas")

        # Assert
        assert isinstance(lc, pd.DataFrame)

    def test__return_astropy(self, task_response_data: str):
        # Act
        lc = AtlasClient.process_task_data(task_response_data, return_type="astropy")

        # Assert
        assert isinstance(lc, Table)

    def test__return_records(self, task_response_data: str):
        # Act
        lc = AtlasClient.process_task_data(task_response_data, return_type="records")

        # Assert
        assert isinstance(lc, list)

    def test__bad_return_type_raises(self, task_response_data: str):
        # Act
        with pytest.raises(ValueError):
            lc = AtlasClient.process_task_data(task_response_data, return_type="blah")


class Test__SubmitQuery:
    def test__submit_query(self, atlas_client: AtlasClient):
        # Arrange
        data = dict(ra=90.0, dec=180.0, comment="target:test")

        # Act
        resp = atlas_client.submit_forced_photom_query(data)

        # Assert
        assert resp.__class__.__name__ == "MockPostResponse"
        # Can't import from conftest
        assert resp.status_code == 201

    def test__bad_data_warns(self, atlas_client: AtlasClient):
        # Arrange
        data = dict(weird_kwarg=100.0)

        # Act
        with pytest.warns(AtlasClientWarning):
            resp = atlas_client.submit_forced_photom_query(data)

    def test__throttle(self, atlas_client: AtlasClient):
        # Arrange
        data = dict(comment="T_throttle:test")

        # Act
        resp = atlas_client.submit_forced_photom_query(data)

        # Assert
        assert resp.status_code == 429


class Test__GetExisting:
    def test__get_existing_no_url(self, atlas_client: AtlasClient):
        # Act
        existing = atlas_client.get_existing_queries()

        # Assert
        assert isinstance(existing, dict)

        assert existing["next"].endswith("/?cursor_ABCDE")
        assert isinstance(existing["results"], list)
        assert len(existing["results"]) == 5
        assert isinstance(existing["results"][0], dict)
        assert existing["results"][0]["url"] == "http://atlas.mock/T00"

    def test__get_existing_with_url(self, atlas_client: AtlasClient):
        # Act
        existing = atlas_client.get_existing_queries(
            url="https://atlas.mock/queue/?cursor_otherpage"
        )

        # Assert
        assert isinstance(existing, dict)

        assert existing["next"] is None  # We got the second page
        assert isinstance(existing["results"], list)
        assert len(existing["results"]) == 2
        assert isinstance(existing["results"][0], dict)
        assert existing["results"][0]["url"] == "http://atlas.mock/T01"


class Test__IterateExisting:
    def test__iterate_existing(self, atlas_client: AtlasClient):
        # Arrange
        result_list = []

        # Act
        for result in atlas_client.iterate_existing_queries():
            result_list.append(result)

        # Assert
        assert len(result_list) == 7

        assert result_list[-1]["url"] == "http://atlas.mock/T01"
        assert result_list[0]["url"] == "http://atlas.mock/T_bad_query00"
        # remember first page returned in rev.

    def test__quit_after_timelimit(self, atlas_client: AtlasClient):

        # Arrange
        result_list = []

        # Act
        for result in atlas_client.iterate_existing_queries(max_query_time=0.1):
            result_list.append(result)

        # Assert
        assert len(result_list) == 5
        # Same as before, but we quit because of the 0.2 sec sleep in mocked AtlasClient


class Test__RecoverTaskData:
    def test__success_pandas(self, atlas_client: AtlasClient):
        # Act
        status, lc = atlas_client.recover_task_data("T_ready_00", return_type="pandas")

        # Assert
        assert status == 200
        assert isinstance(lc, pd.DataFrame)
        assert not lc.empty

    def test__success_astropy(self, atlas_client: AtlasClient):
        # Act
        status, lc = atlas_client.recover_task_data("T_ready_00", return_type="astropy")

        # Assert
        assert status == 200
        assert isinstance(lc, Table)
        assert len(lc) > 0

    def test__still_waiting(self, atlas_client: AtlasClient):
        # Act
        status, lc = atlas_client.recover_task_data("T00", return_type="pandas")

        # Assert
        assert status == 201
        assert lc is None

    def test__no_data_pandas(self, atlas_client: AtlasClient):
        # Act
        status, lc = atlas_client.recover_task_data("T_no_data", return_type="pandas")

        # Assert
        assert status == 200
        assert isinstance(lc, pd.DataFrame)
        assert lc.empty

    def test__no_data_astropy(self, atlas_client: AtlasClient):
        # Act
        status, lc = atlas_client.recover_task_data("T_no_data", return_type="astropy")

        # Asset
        assert status == 200
        assert isinstance(lc, Table)
        assert len(lc) == 0

    def test__bad_return_type_raises(self, atlas_client: AtlasClient):
        # Act
        with pytest.raises(ValueError):
            status, lc = atlas_client.recover_task_data(
                "T_ready_00", return_type="blah"
            )
