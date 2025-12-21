import pytest
import time

from aas2rto.query_managers.atlas.atlas import AtlasQueryManager
from aas2rto.query_managers.atlas.atlas_client import AtlasClient


class MockGetResponse:

    reason_lookup = {200: "ok", 201: "submitted", 429: "bad_request"}

    def __init__(
        self,
        url: str,
        headers: dict[str, str] = None,
        data: dict[str, str] = None,
        timeout: float = None,
        json_data: dict = None,
        status_code: int = 200,
    ):
        self.url = url
        self.headers = headers
        self.text = None
        self.json_data = json_data
        self.status_code = status_code
        self.reason = self.reason_lookup.get(
            status_code, f"uh-oh, don't know {status_code}"
        )

    def json(self):
        if self.json_data is None:
            raise ValueError("You must set json_data in __init__ or mock function...")
        return self.json_data


class MockPostResponse:

    reason_lookup = {200: "ok", 201: "submitted", 429: "server throttled!"}

    def __init__(
        self,
        url: str,
        headers: dict[str, str] = None,
        data: dict[str, str] = None,
        timeout: float = None,
        json_data: dict = None,
        status_code: int = 201,
    ):
        self.url = url
        self.headers = headers
        self.data = data
        self.timeout = timeout

        self.json_data = json_data
        self.status_code = status_code
        self.reason = self.reason_lookup.get(
            status_code, f"uh-oh, don't know {status_code}"
        )

    def json(self) -> dict:
        if self.json_data is None:
            raise ValueError("You must set json_data in __init__ or mock function...")
        return self.json_data


def mock_post_wrapper(
    url: str,
    headers: dict[str, str] = None,
    data: dict[str, str] = None,
    timeout: float = None,
):
    service = url.strip("/").split("/")[-1]  # remove trailing slash first
    if service == "api-token-auth":
        return mock_token_response(url, headers=headers, data=data, timeout=timeout)

    # If it's not asking for the token, it must be a forced photom request.

    target_id = data.get("comment", "QUERY_PAYLOAD_MISSING_COMMENT").split(":")[0]
    json_data = {"url": f"http://atlas.mock/{target_id}"}
    if headers is None:
        raise ValueError("you didn't provide headers!")
    response = MockPostResponse(
        url, headers=headers, data=data, timeout=timeout, json_data=json_data
    )
    comment = data.get("comment", "")
    if "T_sleep" in comment:
        time.sleep(0.2)
    if "T_error" in comment:
        response.status_code = AtlasClient.QUERY_BAD_REQUEST
    if "T_throttle" in comment:
        response.status_code = AtlasClient.QUERY_THROTTLED
    return response


def mock_token_response(
    url: str,
    headers: dict[str, str] = None,
    data: dict[str, str] = None,
    timeout: float = None,
):
    if "username" not in data.keys() or "password" not in data.keys():
        raise KeyError("Must provide 'username' and 'password'")
    # No headers check for token request.
    token_data = {"token": 1234}
    return MockPostResponse(
        url,
        headers=headers,
        data=data,
        timeout=timeout,
        status_code=200,
        json_data=token_data,
    )


def mock_task_page_response(url: str, headers: dict[str, str] = None):
    last_word = url.rstrip("/").split("/")[-1]
    if last_word == "queue":  # the default, first page
        next_url = AtlasClient.atlas_default_queue_url + "/?cursor_ABCDE"
        results = [
            {"url": "http://atlas.mock/T00", "comment": "T00:test"},
            {"url": "http://atlas.mock/T901", "comment": "T901:other_project"},
            {"url": "http://atlas.mock/T_ready00", "comment": "T_ready00:test"},
            {"url": "http://atlas.mock/T_no_data00", "comment": "T_no_data00:test"},
            {"url": "http://atlas.mock/T_bad_query00", "comment": "T_bad_query00:test"},
        ]
        page_data = {"next": next_url, "results": results}
        time.sleep(0.2)

    else:
        page_data = {
            "next": None,
            "results": [
                {"url": "http://atlas.mock/T01", "comment": "T01:test"},
                {"url": "http://atlas.mock/web_task00"},  # No comment - skip web tasks
            ],
        }
    return MockGetResponse(url, headers=headers, json_data=page_data)


def mock_task_response(url: str, headers: dict):

    target_id = url.rstrip("/").split("/")[-1]
    if "T_ready" in url:
        data = {
            "result_url": f"{url}.txt",
            "finishtimestamp": 60000.0,
            "comment": f"{target_id}:test",
        }
        status_code = AtlasClient.QUERY_EXISTS
    elif "T_no_data" in url:
        data = {
            "error_msg": "No data returned",
            "finishtimestamp": 60000.0,
            "comment": f"{target_id}:test",
        }
        status_code = AtlasClient.QUERY_EXISTS
    elif "T_bad_query" in url:
        data = {
            "error_msg": "some other error",
            "finishtimestamp": 60000.0,
            "comment": "T_bad_query:test",
        }
        status_code = AtlasClient.QUERY_BAD_REQUEST
    else:
        data = {"finishtimestamp": None, "comment": f"{target_id}:test"}
        status_code = AtlasClient.QUERY_SUBMITTED
    return MockGetResponse(
        url, headers=headers, json_data=data, status_code=status_code
    )


@pytest.fixture
def task_response_data(atlas_rows: list[list]):
    data = "###MJD m dm mag5sig F Obs\n"
    for row in atlas_rows:
        row_str = " ".join(str(v) for v in row)
        data = data + row_str + "\n"
    return data


@pytest.fixture(autouse=True)
def monkeypatch_atlas_query(task_response_data: str, monkeypatch: pytest.MonkeyPatch):

    def mock_post(*args, **kwargs):
        return mock_post_wrapper(*args, **kwargs)

    def mock_session_post(self, *args, **kwargs):
        return mock_post(*args, **kwargs)

    def mock_get(url: str = None, headers: dict[str, str] = None, timeout=None):
        if headers is None:
            raise ValueError("you didn't provide headers!")

        if url.endswith(".txt"):
            # We're asking for the results
            response = MockGetResponse(url, headers=headers, json_data={})
            response.text = task_response_data
            return response

        last_word = url.rstrip("/").split("/")[-1]
        if last_word == "queue" or last_word.startswith("?cursor"):
            return mock_task_page_response(url, headers)
        else:
            return mock_task_response(url, headers)

    def mock_session_get(self, *args, **kwargs):
        return mock_get(*args, **kwargs)

    def mock_delete(*args, **kwargs):
        pass

    def mock_session_delete(self, *args, **kwargs):
        return mock_delete(*args, **kwargs)

    monkeypatch.setattr("requests.get", mock_get)
    monkeypatch.setattr("requests.Session.get", mock_session_get)
    monkeypatch.setattr("requests.post", mock_post)
    monkeypatch.setattr("requests.Session.post", mock_post)
    monkeypatch.setattr("requests.delete", mock_delete)
    monkeypatch.setattr("requests.Session.delete", mock_session_delete)
    return
