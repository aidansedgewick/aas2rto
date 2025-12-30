import pytest
import requests
import time


class MockResponse:
    def __init__(self, data, status_code: int = 200, reason: str = "OK"):
        self.data = data
        self.status_code = status_code
        self.text = data
        self.reason = reason

    def json(self):
        return {"results": self.data}


@pytest.fixture
def yse_lightcurve_data(yse_rows: list[list], yse_columns: list[str]):
    str_rows = [" ".join(str(x) for x in row) for row in yse_rows]
    return (
        "# COMM1: some_data\n"
        + "# COMM2: other_data\n"
        + "\n\n"
        + " ".join(yse_columns)
        + "\n"
        + "\n".join(str_rows)
        + "\n"
    )


@pytest.fixture
def yse_lightcurve_response(yse_lightcurve_data):
    return MockResponse(yse_lightcurve_data)


@pytest.fixture
def yse_transient_parameter_data():
    return [{"param1": 1.0, "param2": 10.0}]


@pytest.fixture
def yse_transient_parameter_response(yse_transient_parameter_data):
    return MockResponse(yse_transient_parameter_data)


@pytest.fixture
def yse_explorer_data(yse_explorer_columns: list[str], yse_explorer_rows: list[list]):
    str_rows = [",".join(str(x) for x in row) for row in yse_explorer_rows]
    return ",".join(yse_explorer_columns) + "\n" + "\n".join(str_rows)


@pytest.fixture
def yse_explorer_response(yse_explorer_data):
    return MockResponse(yse_explorer_data)


@pytest.fixture
def yse_transients_query_data():
    return [
        {"name": "2023J", "classification": "SN"},
        {"name": "2023K", "classification": "SN Ia"},
    ]


@pytest.fixture
def yse_transients_query_response(yse_transients_query_data):
    return MockResponse(yse_transients_query_data)


@pytest.fixture
def yse_survey_fields_query_data():
    return [
        {"field_id": 1000},
        {"field_id": 1001},
    ]


@pytest.fixture
def yse_survey_fields_query_response(yse_survey_fields_query_data):
    return MockResponse(yse_survey_fields_query_data)


@pytest.fixture
def yse_survey_observations_query_data():
    return [
        {"ra": 180.0, "dec": 0.0},
        {"ra": 270.0, "dec": 30.0},
    ]


@pytest.fixture
def yse_survey_observations_query_response(yse_survey_observations_query_data):
    return MockResponse(yse_survey_observations_query_data)


@pytest.fixture
def yse_survey_fields_msb_query_data():
    return [
        {"name": "field_A", "active": 1},
        {"name": "field_B", "active": 0},
    ]


@pytest.fixture
def yse_survey_fields_msb_query_response(yse_survey_fields_msb_query_data):
    return MockResponse(yse_survey_fields_msb_query_data)


@pytest.fixture(autouse=True)
def monkeypatch_yse_query(
    yse_lightcurve_response: MockResponse,
    yse_transient_parameter_response: MockResponse,
    yse_explorer_response: MockResponse,
    yse_survey_fields_query_response: MockResponse,
    yse_survey_fields_msb_query_response: MockResponse,
    yse_survey_observations_query_response: MockResponse,
    yse_transients_query_response: MockResponse,
    monkeypatch: pytest.MonkeyPatch,
):

    def mock_get(url: str, auth: requests.auth.HTTPBasicAuth = None):
        if not ((auth.username == "user") & (auth.password == "password")):
            return MockResponse(None, status_code=401, reason="Unauthorized")

        if "explorer" in url:
            if "101" in url:
                return yse_explorer_response
            else:
                return MockResponse(None, status_code=404, reason="Not Found")

        elif "download_photometry" in url:
            if "2023J" in url:
                return yse_lightcurve_response
            if "fail" in url:
                raise Exception("url asked to raise exc.")
            if "sleep" in url:
                time.sleep(0.2)
                return MockResponse(None, status_code=500, reason="Server Error")
            else:
                return MockResponse(None, status_code=500, reason="Server Error")

        elif "surveyfields/?" in url:
            return yse_survey_fields_query_response

        elif "surveyfieldmsbs/?" in url:
            return yse_survey_fields_msb_query_response

        elif "surveyobservations/?" in url:
            return yse_survey_observations_query_response

        elif "transients/?" in url:
            if "name=" in url:
                if "name=2023J" in url:
                    return yse_transient_parameter_response
                else:
                    return MockResponse([{}])
            else:
                return yse_transients_query_response

        else:
            raise ValueError(f"'mock_get()' not prepared for '{url}'")

    def mock_post(*args, **kwargs):
        raise NotImplementedError(f"requests.post mock not prepared yet")

    monkeypatch.setattr("requests.get", mock_get)
    monkeypatch.setattr("requests.post", mock_post)
