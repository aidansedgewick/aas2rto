import json as json_tools  # parameter of query is 'json'
import pytest

import pandas as pd

from astropy import units as u
from astropy.table import Table
from astropy.time import Time

from aas2rto.query_managers.fink.fink_portal_client import (
    FinkBasePortalClient,
    FinkZTFPortalClient,
    FinkLSSTPortalClient,
    FinkPortalClientError,
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
        content = json.pop("content", {})
        self.content = json_tools.dumps(content)  # use renamed 'json' module...
        self.elapsed = MockElapsed(json.pop("elapsed", 1.0))
        self.payload = json  # whatever remains...


class MockGetResponse:
    def __init__(self, *args, status_code: int = 200, params=None):
        self.args = args
        # json = json or {}
        self.status_code = params.pop("status_code", 200)
        self.reason = params.pop("reason", "OK")
        content = params.pop("content", {})
        self.content = json_tools.dumps(content)  # use renamed 'json' module...
        self.elapsed = MockElapsed(params.pop("elapsed", 1.0))
        self.payload = params  # whatever remains...


###===== Pytest fixtures here =====###


@pytest.fixture(autouse=True)
def patch_requests(monkeypatch: pytest.MonkeyPatch):
    def mock_post(*args, **kwargs):
        return MockPostResponse(*args, **kwargs)

    def mock_get(*args, **kwargs):
        return MockGetResponse(*args, **kwargs)

    monkeypatch.setattr("requests.post", mock_post)
    monkeypatch.setattr("requests.get", mock_get)


###===== Test helper functions here =====###


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
        q = FinkZTFPortalClient()

    def test__lsst_query_init(self):
        # Act
        q = FinkLSSTPortalClient()


class Test__HelperFunctions:
    def test__fix_keys(self):
        # Arrange
        data = {"p:key01": 1.0, "key02": 1.0, "p:q:key03": 1.0}

        # Act
        FinkCoolPortalClient().fix_dict_keys_inplace(data)

        # Assert
        assert set(data.keys()) == set("key01 key02 key03".split())

    def test__fix_kwargs(self):
        # Arrange
        kwargs = {"key01": 1.0, "key02_": 1.0, "key03__": 1.0}

        # Act
        fixed = FinkCoolPortalClient().process_kwargs(**kwargs)

        # Assert
        assert set(fixed.keys()) == set("key01 key02 key03".split())

    def test__process_data(self):
        # Arrage
        data = [{"i:key01": 1.0, "i:key02": 10.0}, {"i:key01": 2.0, "i:key02": 20.0}]

        # Act
        processed_data = FinkCoolPortalClient().process_data(
            data, return_type="records"
        )

        # Assert
        assert isinstance(processed_data, list)
        assert len(processed_data) == 2
        assert isinstance(processed_data[0], dict)
        assert set(processed_data[0].keys()) == set("key01 key02".split())

    def test__process_data_pandas(self):
        # Arrange
        data = [{"i:key01": 1.0, "i:key02": 10.0}, {"i:key01": 2.0, "i:key02": 20.0}]

        # Act
        df = FinkCoolPortalClient().process_data(data, return_type="pandas")

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test__process_data_astropy(self):
        # Arrange
        data = [{"i:key01": 1.0, "i:key02": 10.0}, {"i:key01": 2.0, "i:key02": 20.0}]

        # Act
        table = FinkCoolPortalClient().process_data(data, return_type="astropy")

        # Assert
        assert isinstance(table, Table)
        assert len(table) == 2

    def test__process_response(self):
        # Arrange
        data = [{"i:key01": 1.0, "i:key02": 10.0}, {"i:key01": 2.0, "i:key02": 20.0}]
        res = MockPostResponse(json=dict(content=data))  # json.dumps INSIDE __init__

        # Act
        processed = FinkCoolPortalClient().process_response(res, return_type="records")

        # Assert
        assert isinstance(processed, list)
        assert isinstance(processed[0], dict)
        assert set(processed[0].keys()) == set("key01 key02".split())

    def test__process_bad_return_fals(self):
        # Arrange
        data = [{"i:key01": 1.0, "i:key02": 10.0}, {"i:key01": 2.0, "i:key02": 20.0}]
        res = MockPostResponse(json=dict(content=data))  # json.dumps INSIDE __init__

        # Act
        with pytest.raises(ValueError):
            processed = FinkCoolPortalClient().process_response(res, return_type="blah")


class Test__CommonEndpoints:

    def test__cutouts(self):
        # Act
        res = FinkCoolPortalClient().cutouts(kind="Difference", process=False)

        # Assert
        assert isinstance(res, MockPostResponse)
        assert res.args[0].split("/")[-1] == "cutouts"

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
            t_fixed, t_stop=t_later, n=10, return_type="pandas"
        )

    def test__query_and_collate_astropy(self, t_fixed: Time):
        # Arrange
        t_later = t_fixed + 1.0 * u.day

        # Act
        FinkZTFPortalClient().query_classifiers(
            t_fixed, t_stop=t_later, n=10, return_type="astropy"
        )

    def test__query_and_collate_records(self, t_fixed: Time):
        # Arrange
        t_later = t_fixed + 1.0 * u.day

        # Act
        FinkZTFPortalClient().query_classifiers(
            t_fixed, t_stop=t_later, n=10, return_type="records"
        )
