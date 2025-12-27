import pytest
import requests
from typing import NoReturn

import pandas as pd

from astropy.table import Table

from aas2rto.query_managers.yse.yse_client import (
    YSEClient,
    YSEClientError,
    YSEClientBadEndpointError,
)


@pytest.fixture
def yse_client() -> YSEClient:
    return YSEClient("user", "password")


@pytest.fixture
def bad_client() -> YSEClient:
    return YSEClient("bad_user", "bad_pass")


class Test__InitQueryClass:
    def test__init(self):
        # Act
        q = YSEClient("user", "password")

        # Assert
        assert isinstance(q.yse_auth, requests.auth.HTTPBasicAuth)


class Test__ProcessExplorerResult:
    def test__return_type_pandas(self, yse_explorer_data: str):
        # Act
        result = YSEClient.process_explorer_result(yse_explorer_data)

        # Assert
        assert isinstance(result, pd.DataFrame)
        exp_columns = ["name", "classification", "ra", "dec", "n_det"]
        assert set(result.columns) == set(exp_columns)
        assert len(result) == 3

    def test__return_type_astropy(self, yse_explorer_data: str):
        # Act
        result = YSEClient.process_explorer_result(
            yse_explorer_data, return_type="astropy"
        )

        # Assert
        assert isinstance(result, Table)
        exp_columns = ["name", "classification", "ra", "dec", "n_det"]
        assert set(result.columns) == set(exp_columns)

    def test__return_type_records(self, yse_explorer_data: str):
        # Act
        result = YSEClient.process_explorer_result(
            yse_explorer_data, return_type="records"
        )

        # Assert
        assert isinstance(result, list)
        assert len(result) == 3
        assert isinstance(result[0], dict)
        exp_columns = ["name", "classification", "ra", "dec", "n_det"]
        assert set(result[0].keys()) == set(exp_columns)

    def test__bad_return_type_raises(self, yse_explorer_data: str):
        # Act
        with pytest.raises(ValueError):
            YSEClient.process_explorer_result(yse_explorer_data, return_type="aaagh")


class Test__EmptyQueryExplorerResults:
    def test__empty_pandas(self):
        # Act
        result = YSEClient.get_empty_query_results(return_type="pandas")

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert set(result.columns) == set(["name"])

    def test__empty_astropy(self):
        # Act
        result = YSEClient.get_empty_query_results(return_type="astropy")

        # Assert
        assert isinstance(result, Table)
        assert len(result) == 0
        assert set(result.columns) == set(["name"])

    def test__empty_records(self):
        # Act
        result = YSEClient.get_empty_query_results(return_type="records")

        # Assert
        assert isinstance(result, list)
        assert len(result) == 0
        # no columns to check in empty list...

    def test__bad_type_rasies(self):
        # Act
        with pytest.raises(ValueError):
            result = YSEClient.get_empty_query_results(return_type="bad_type")


class Test__QueryExplorer:
    def test__bad_auth(self, bad_client: YSEClient):
        # Act
        result = bad_client.query_explorer(101)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test__good_query_pandas(self, yse_client: YSEClient):
        # Act
        result = yse_client.query_explorer(101, return_type="pandas")

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        exp_columns = ["name", "classification", "ra", "dec", "n_det"]
        assert set(result.columns) == set(exp_columns)

    def test__good_query_pandas(self, yse_client: YSEClient):
        # Act
        result = yse_client.query_explorer(101, return_type="astropy")

        # Assert
        assert isinstance(result, Table)
        assert len(result) == 3
        exp_columns = ["name", "classification", "ra", "dec", "n_det"]
        assert set(result.columns) == set(exp_columns)

    def test__missing_query_id(self, yse_client: YSEClient):
        # Act
        result = yse_client.query_explorer(999)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test__no_process(self, yse_client: YSEClient):
        # Act
        result = yse_client.query_explorer(101, process=False)

        # Assert
        assert type(result).__name__ == "MockResponse"
        # Can't import actual class to do isinstance() test


class Test__EmptyLC:
    def test__empty_pandas(self):
        # Act
        lc = YSEClient.get_empty_lightcurve(return_type="pandas")

        # Assert
        assert isinstance(lc, pd.DataFrame)
        exp_cols = "mjd flt fluxcal fluxcalerr mag magerr magsys telescope instrument dq".split()
        assert set(lc.columns) == set(exp_cols)

    def test__empty_astropy(self):
        # Act
        lc = YSEClient.get_empty_lightcurve(return_type="astropy")

        # Assert
        assert isinstance(lc, Table)
        exp_cols = "mjd flt fluxcal fluxcalerr mag magerr magsys telescope instrument dq".split()
        assert set(lc.columns) == set(exp_cols)
        assert len(lc) == 0

    def test__empty_records(self):
        # Act
        lc = YSEClient.get_empty_lightcurve(return_type="records")

        # Assert
        assert isinstance(lc, list)
        assert len(lc) == 0
        # no columns in an empty list...

    def test__bad_type_raises(self):
        # Act
        with pytest.raises(ValueError):
            YSEClient.get_empty_lightcurve(return_type="bad_type")


class Test__QueryLC:
    def test__bad_auth(self, bad_client: YSEClient):
        # Act
        lc = bad_client.query_lightcurve("2023J")

        # Assert
        assert isinstance(lc, pd.DataFrame)
        assert lc.empty

    def test__known_target_pandas(self, yse_client: YSEClient):
        # Act
        lc = yse_client.query_lightcurve("2023J", return_type="pandas")

        # Assert
        assert isinstance(lc, pd.DataFrame)
        exp_cols = "mjd mag magerr diffmaglim flt instrument dq".split()
        assert set(lc.columns) == set(exp_cols)
        assert len(lc) == 6

    def test__unknown_target_pandas(self, yse_client: YSEClient):
        # Act
        lc = yse_client.query_lightcurve("2023K", return_type="pandas")

        # Assert
        assert isinstance(lc, pd.DataFrame)
        assert lc.empty

    def test__known_target_astropy(self, yse_client: YSEClient):
        # Act
        lc = yse_client.query_lightcurve("2023J", return_type="astropy")

        # Assert
        assert isinstance(lc, Table)
        assert len(lc) == 6

    def test__unknown_target_astropy(self, yse_client: YSEClient):
        # Act
        lc = yse_client.query_lightcurve("2023K", return_type="astropy")

        # Assert
        assert isinstance(lc, Table)
        assert len(lc) == 0

    def test__known_target_records(self, yse_client: YSEClient):
        # Act
        lc = yse_client.query_lightcurve("2023J", return_type="records")

        # Assert
        assert isinstance(lc, list)
        assert len(lc) == 6

    def test__unknown_target_records(self, yse_client: YSEClient):
        # Act
        lc = yse_client.query_lightcurve("2023K", return_type="records")

        # Assert
        assert isinstance(lc, list)
        assert len(lc) == 0


class Test__PrepareQueryURL:
    def test__bad_endpoint_raises(self):
        # Act
        with pytest.raises(YSEClientBadEndpointError):
            url = YSEClient.prepare_query_url("bad_endpoint", {"blah": 100.0})

    def test__survey_fields_url(self):
        # Act
        url = YSEClient.prepare_query_url("surveyfields", {"field_id": 1000})

        # Assert
        assert url.endswith("api/surveyfields/?field_id=1000")

    def test__survey_fields_bad_param_raises(self):
        # Act
        with pytest.raises(YSEClientError):
            url = YSEClient.prepare_query_url("surveyfields", {"blah": 100.0})

    def test__survey_field_msbs_url(self):
        # Act
        url = YSEClient.prepare_query_url("surveyfieldmsbs", {"name": "field_A"})

        # Assert
        assert url.endswith("api/surveyfieldmsbs/?name=field_A")

    def test__survey_field_msbs_bad_param_raises(self):
        # Act
        with pytest.raises(YSEClientError):
            url = YSEClient.prepare_query_url("surveyfieldmsbs", {"blah": 100.0})

    def test__survey_observations_url(self):
        # Act
        url = YSEClient.prepare_query_url(
            "surveyobservations", {"obs_mjd_gte": 60000.0}
        )

        # Assert
        assert url.endswith("api/surveyobservations/?obs_mjd_gte=60000.0")

    def test__survey_obesrvations_param_raises(self):
        # Act
        with pytest.raises(YSEClientError):
            url = YSEClient.prepare_query_url("surveyfieldmsbs", {"blah": 100.0})

    def test__transients_url(self):
        # Act
        url = YSEClient.prepare_query_url(
            "transients", {"dec_gte": 0.0, "dec_lte": 1.0}
        )

        # Assert
        assert url.endswith("api/transients/?dec_gte=0.0&dec_lte=1.0")

    def test__transients_bad_param_raises(self):
        # Act
        with pytest.raises(YSEClientError):
            url = YSEClient.prepare_query_url("transients", {"blah": 100.0})


class Test__QueryEndpoints:
    def test__query_survey_fields_pandas(self, yse_client: YSEClient):
        # Act
        result = yse_client.query_survey_fields({}, return_type="pandas")

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert set(result.columns) == set(["field_id"])

    def test__query_survey_fields_astropy(self, yse_client: YSEClient):
        # Act
        result = yse_client.query_survey_fields({}, return_type="astropy")

        # Assert
        assert isinstance(result, Table)
        assert len(result) == 2
        assert set(result.columns) == set(["field_id"])

    def test__query_survey_field_msbs_pandas(self, yse_client: YSEClient):
        # Act
        result = yse_client.query_survey_field_msbs({}, return_type="pandas")

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert set(result.columns) == set(["name", "active"])

    def test__query_survey_field_msbs_astropy(self, yse_client: YSEClient):
        # Act
        result = yse_client.query_survey_field_msbs({}, return_type="astropy")

        # Assert
        assert isinstance(result, Table)
        assert len(result) == 2
        assert set(result.columns) == set(["name", "active"])

    def test__query_survey_observations_pandas(self, yse_client: YSEClient):
        # Act
        result = yse_client.query_survey_observations({}, return_type="pandas")

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert set(result.columns) == set(["ra", "dec"])

    def test__query_survey_observations_astropy(self, yse_client: YSEClient):
        # Act
        result = yse_client.query_survey_observations({}, return_type="astropy")

        # Assert
        assert isinstance(result, Table)
        assert len(result) == 2
        assert set(result.columns) == set(["ra", "dec"])

    def test__query_transients_pandas(self, yse_client: YSEClient):
        # Act
        result = yse_client.query_transients({}, return_type="pandas")

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert set(result.columns) == set(["name", "classification"])

    def test__query_survey_observations_astropy(self, yse_client: YSEClient):
        # Act
        result = yse_client.query_transients({}, return_type="astropy")

        # Assert
        assert isinstance(result, Table)
        assert len(result) == 2
        assert set(result.columns) == set(["name", "classification"])


class Test__QueryNamedTransient:
    def test__query_good_transient(self, yse_client: YSEClient):
        # Act
        result = yse_client.query_named_transient("2023J")

        # Assert
        assert isinstance(result, dict)
        assert set(result.keys()) == set(["param1", "param2"])

    def test__query_unknown_transient(self, yse_client: YSEClient):
        # Act
        result = yse_client.query_named_transient("9999xyz")

        # Assert
        assert isinstance(result, dict)
        assert set(result.keys()) == set()
