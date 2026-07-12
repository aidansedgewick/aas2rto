import gzip
import json as jsonlib  # parameter of client is 'json'
import pytest
from io import BytesIO

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from aas2rto.query_managers.fink.fink_portal_client import (
    FinkBasePortalClient,
    FinkZTFPortalClient,
    FinkLSSTPortalClient,
    FinkPortalClientError,
    fix_dict_keys_inplace,
    readstamp,
)

##===== Some helper functions here =====##


class FinkCoolPortalClient(FinkBasePortalClient):
    # DON'T use api.cool.fink-portal.org, so no chance of traffic to fink-portal.org
    api_url = "https://fink_cool.org/api/v1"
    imtypes = ("Difference", "Template")
    target_id_key = "target_id"
    alert_id_key = "alert_id"

    def query_lightcurve(self, *args, **payload):
        pass  # Doesn't matter, we don't call it for testing generic methods

    def query_classifiers(self, *args, **payload):
        pass


# class FinkBadQuery(FinkPortalClientError):
#     pass


class MockElapsed:
    def __init__(self, s):
        self.s = s

    def total_seconds(self):
        return self.s


class MockPostResponse:
    def __init__(self, *args, json=None):
        self.args = args

        # json = json or {}
        self.status_code = json.pop("status_code", 200)
        self.reason = json.pop("reason", "OK")
        self._data = json.pop("content", {})
        self.content = jsonlib.dumps(self._data)  # use renamed 'json' module...
        self.elapsed = MockElapsed(json.pop("elapsed", 1.0))
        self.payload = json  # whatever remains...

    def json(self):
        return self._data


class MockGetResponse:
    def __init__(self, *args, status_code: int = 200, params=None):
        self.args = args
        # json = json or {}
        self.status_code = params.pop("status_code", 200)
        self.reason = params.pop("reason", "OK")
        self._data = params.pop("content", {})
        self.content = jsonlib.dumps(self._data)  # renamed 'json' module...
        self.elapsed = MockElapsed(params.pop("elapsed", 1.0))
        self.payload = params  # whatever remains...

    def json(self):
        return self._data


###===== Pytest fixtures here =====###


@pytest.fixture
def fink_stamp_data():
    return np.arange(100).reshape(10, 10)


@pytest.fixture
def fink_cutout_array_data():
    return {
        "Difference": np.arange(0, 100).reshape(10, 10).tolist(),
        "Template": np.arange(0, 100).reshape(10, 10).tolist(),
        "Science": np.arange(0, 100).reshape(10, 10).tolist(),
    }


@pytest.fixture
def fink_stamp_bytes(fink_stamp_data: np.ndarray) -> bytes:

    buffer = BytesIO()
    hdul = fits.PrimaryHDU(data=fink_stamp_data)
    hdul.writeto(buffer)
    return buffer.getvalue()


@pytest.fixture
def fink_stamp_gzip_bytes(fink_stamp_data: np.ndarray) -> bytes:
    buffer = BytesIO()
    hdul = fits.PrimaryHDU(data=fink_stamp_data)
    with gzip.GzipFile(fileobj=buffer, mode="wb") as gzip_file:
        hdul.writeto(gzip_file)
    return buffer.getvalue()


@pytest.fixture
def mock_table_data():
    return [
        {"a:col1": 1, "a:col2": 10, "a:col3": 100},
        {"a:col1": 2, "a:col2": 20, "a:col3": 200},
    ]


@pytest.fixture(autouse=True)
def patch_post_request(
    monkeypatch: pytest.MonkeyPatch,
    fink_stamp_data: np.ndarray,
    mock_table_data: list[dict],
):

    TABLE_LIKE_ENPOINTS = ["latests", "sources", "objects", "tags"]

    def mock_post(*args, json: dict = None):
        json = json or {}

        new_content = None

        endpoint = args[0].split("/")[-1]
        if endpoint == "cutouts":
            output_format = json.get("output-format")
            kinds = [json.get("kind")]
            if kinds[0] == "All":
                kinds = ["Difference", "Template", "Science"]

            if output_format == "array":
                new_content = {f"cutout{k}": fink_stamp_data.tolist() for k in kinds}
            elif output_format == "FITS":
                raise NotImplementedError("Write this test...")
        elif endpoint in TABLE_LIKE_ENPOINTS:
            print("we are about to return table-like")
            new_content = mock_table_data
        else:
            print(f"we are not replacing content with endpoint {endpoint}")

        provided_content = json.get("content")
        if provided_content and new_content:
            raise ValueError("Can't overwrite provided content and new content!")
        json["content"] = new_content

        return MockPostResponse(*args, json=json)

    monkeypatch.setattr("requests.post", mock_post)


@pytest.fixture(autouse=True)
def patch_get_request(monkeypatch: pytest.MonkeyPatch):

    def mock_get(*args, **kwargs):
        return MockGetResponse(*args, **kwargs)

    monkeypatch.setattr("requests.get", mock_get)


###===== Test helper functions here =====###


class Test__Readstamp:
    def test__readstamp(self, fink_stamp_gzip_bytes: bytes):
        # Act
        recovered = readstamp(fink_stamp_gzip_bytes, gzipped=True)
        # gzipped is True BY DEFAULT - but explict is helpful reminder here...

        # Assert
        assert isinstance(recovered, np.ndarray)

        assert recovered.shape == (10, 10)

    def test__readstamp_gzipped(self, fink_stamp_bytes: bytes):
        # Act
        recovered = readstamp(fink_stamp_bytes, gzipped=False)

        # Assert
        assert isinstance(recovered, np.ndarray)
        assert recovered.shape == (10, 10)


class Test__ExamplePortalClient:

    def test__post_is_patched(self):
        # Act
        res = FinkCoolPortalClient().do_request(
            "some_service", method="post", process=False
        )

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0] == "https://fink_cool.org/api/v1/some_service"
        assert res.status_code == 200

    def test__post_bad_status_raises(self):
        # Act
        with pytest.raises(FinkPortalClientError):
            res = FinkCoolPortalClient().do_request(
                "service", method="post", status_code=404, content="some_msg"
            )

    def test__get_is_patched(self):
        # Act
        res = FinkCoolPortalClient().do_request(
            "some_service", method="get", process=False
        )

        # Assert
        assert isinstance(res, MockGetResponse)
        assert res.args[0] == "https://fink_cool.org/api/v1/some_service"
        assert res.status_code == 200

    def test__get_bad_status_code_raises(self):
        # Act
        with pytest.raises(FinkPortalClientError):
            res = FinkCoolPortalClient().do_request(
                "service", method="get", status_code=404, content="some_msg"
            )


##====== Actual tests start here! =====###


class Test__PortalClientsInit:
    def test__ztf_query_init(self):
        # Act
        cli = FinkZTFPortalClient()

    def test__lsst_query_init(self):
        # Act
        cli = FinkLSSTPortalClient()


class Test__HelperFunctions:
    def test__fix_keys(self):
        # Arrange
        data = {"p:key01": 1.0, "key02": 1.0, "p:q:key03": 1.0}

        # Act
        fix_dict_keys_inplace(data)

        # Assert
        assert set(data.keys()) == set("key01 key02 key03".split())

    def test__fix_kwargs(self):
        # Arrange
        kwargs = {"key01": 1.0, "key02_": 1.0, "key03__": 1.0}

        # Act
        fixed = FinkCoolPortalClient.process_kwargs(**kwargs)

        # Assert
        assert set(fixed.keys()) == set("key01 key02 key03".split())

    def test__process_table_data(self, mock_table_data: list[dict]):
        # Act
        processed_data = FinkCoolPortalClient.process_table_data(
            mock_table_data, return_type="records"
        )

        # Assert
        assert isinstance(processed_data, list)
        assert len(processed_data) == 2
        assert isinstance(processed_data[0], dict)
        assert set(processed_data[0].keys()) == set("col1 col2 col3".split())  # fixed

    def test__process_table_data_pandas(self, mock_table_data: list[dict]):

        # Act
        df = FinkCoolPortalClient.process_table_data(
            mock_table_data, return_type="pandas"
        )

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test__process_table_data_astropy(self, mock_table_data: list[dict]):
        # Act
        table = FinkCoolPortalClient.process_table_data(
            mock_table_data, return_type="astropy"
        )

        # Assert
        assert isinstance(table, Table)
        assert len(table) == 2

    def test__process_table_bad_return_type(self, mock_table_data: list[dict]):
        # Act
        with pytest.raises(ValueError):
            table = FinkCoolPortalClient.process_table_data(
                mock_table_data, return_type="bad_type"
            )

    # def test__process_response(self):
    #     # Arrange
    #     data = [{"i:key01": 1.0, "i:key02": 10.0}, {"i:key01": 2.0, "i:key02": 20.0}]
    #     res = MockPostResponse(json=dict(content=data))  # json.dumps INSIDE __init__

    #     # Act
    #     processed = FinkCoolPortalClient().process_response(res, return_type="records")

    #     # Assert
    #     assert isinstance(processed, list)
    #     assert isinstance(processed[0], dict)
    #     assert set(processed[0].keys()) == set("key01 key02".split())

    # def test__process_bad_return_fals(self):
    #     # Arrange
    #     data = [{"i:key01": 1.0, "i:key02": 10.0}, {"i:key01": 2.0, "i:key02": 20.0}]
    #     res = MockPostResponse(json=dict(content=data))  # json.dumps INSIDE __init__

    #     # Act
    #     with pytest.raises(ValueError):
    #         processed = FinkCoolPortalClient().process_response(res, return_type="blah")


class Test__CutoutsEndpoint:

    def test__cutouts(self):
        # Act
        res = FinkCoolPortalClient().cutouts(kind="Difference", process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "cutouts"

    def test__raise_with_all_not_array(self):
        # Act
        with pytest.raises(ValueError):
            res = FinkCoolPortalClient().cutouts(
                {"kind": "All", "output-format": "other"}
            )

    def test__process_cutout_data_array(self, fink_cutout_array_data: dict):
        # Act
        res = FinkCoolPortalClient.process_cutout_data(fink_cutout_array_data, "array")

        # Assert
        assert isinstance(res, dict)
        assert set(res.keys()) == set(["Template", "Science", "Difference"])
        assert res["Difference"].shape == (10, 10)
        assert res["Science"].shape == (10, 10)
        assert res["Template"].shape == (10, 10)


class Test__CommonEndpoints:

    def test__cutouts_bad_kind(self):
        # Act
        with pytest.raises(ValueError):
            res = FinkCoolPortalClient().cutouts(kind="Unknown", process=False)

    # def test__cutouts_missing_kind(self):
    #     # Act
    #     with pytest.raises(ValueError):
    #         res = FinkCoolPortalClient().cutouts(process=False)

    def test_conesearch(self):
        # Act
        res = FinkCoolPortalClient().conesearch(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "conesearch"

    def test_sso(self):
        # Act
        res = FinkCoolPortalClient().sso(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "sso"

    def test__resolver(self):
        # Act
        res = FinkCoolPortalClient().resolver(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "resolver"

    def test__schema(self):
        # Act
        res = FinkCoolPortalClient().schema(process=False)

        # Assert
        assert isinstance(res, MockGetResponse)
        assert res.args[0].split("/")[-1] == "schema"

    def test__skymap(self):
        # Act
        res = FinkCoolPortalClient().skymap(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "skymap"

    def test__statistics(self):
        # Act
        res = FinkCoolPortalClient().statistics(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "statistics"


class Test__ZTFEndpoints:

    @pytest.fixture(autouse=True)
    def _patch_api_url(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(FinkZTFPortalClient, "api_url", "fink-ztf-fake.org")

    def test__anomaly(self):
        # Act
        res = FinkZTFPortalClient().anomaly(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "anomaly"

    def test__latests(self):
        # Act
        res = FinkZTFPortalClient().latests(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "latests"

    def test__objects(self):
        # Act
        res = FinkZTFPortalClient().objects(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "objects"

    # explorer is gone now???
    # def test_explorer(self):
    #     # Act
    #     res = FinkZTFPortalClient().explorer(process=False)

    #     # Assert
    #     assert isinstance(res, MockPostResponse)
    #     assert res.args[0].split("/")[-1] == "explorer"

    def test__ssobulk(self):
        # Act
        res = FinkZTFPortalClient().ssobulk(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "ssobulk"

    def test__ssocand(self):
        # Act
        res = FinkZTFPortalClient().ssocand(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "ssocand"

    def test__ssoft(self):
        # Act
        res = FinkZTFPortalClient().ssoft(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "ssoft"

    def test__tracklet(self):
        # Act
        res = FinkZTFPortalClient().tracklet(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "tracklet"

    def test__query_lightcurve(self):
        # Act
        res = FinkZTFPortalClient().query_lightcurve(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "objects"  # OBJECTS enpoint for ZTF


class Test__LSSTEndpoints:

    @pytest.fixture(autouse=True)
    def _patch_api_url(self, monkeypatch: pytest.MonkeyPatch):
        # No chance of pinging servers...
        monkeypatch.setattr(FinkLSSTPortalClient, "api_url", "fink-lsst-fake.org")

    def test__blocks(self):
        # Act
        res = FinkLSSTPortalClient().blocks(process=False)

        # Assert
        assert isinstance(res, MockGetResponse)
        assert res.args[0].split("/")[-1] == "blocks"

    def test__fp(self):
        # Act
        res = FinkLSSTPortalClient().fp(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "fp"

    def test__objects(self):
        # Act
        res = FinkLSSTPortalClient().objects(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "objects"

    def test__sources(self):
        # Act
        res = FinkLSSTPortalClient().sources(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "sources"

    def test__tags(self):
        # Act
        res = FinkLSSTPortalClient().tags(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "tags"

    def test__query_lightcurve(self):
        # Act
        res = FinkLSSTPortalClient().query_lightcurve(process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "sources"  # SOURCES enpoint for lsst


class Test__QueryClassifiers:
    def test__query_and_collate_pandas(self, t_fixed: Time):
        # Arrange
        t_later = t_fixed + 1.0 * u.day

        # Act
        FinkZTFPortalClient().query_classifiers(
            t_start=t_fixed, t_stop=t_later, n=10, return_type="pandas"
        )

    def test__query_and_collate_astropy(self, t_fixed: Time):
        # Arrange
        t_later = t_fixed + 1.0 * u.day

        # Act
        FinkZTFPortalClient().query_classifiers(
            t_start=t_fixed, t_stop=t_later, n=10, return_type="astropy"
        )

    def test__query_and_collate_records(self, t_fixed: Time):
        # Arrange
        t_later = t_fixed + 1.0 * u.day

        # Act
        FinkZTFPortalClient().query_classifiers(
            t_start=t_fixed, t_stop=t_later, n=10, return_type="records"
        )
