import io
import pytest
import re
from typing import Any


class MockPostResponse:
    def __init__(
        self,
        url: str,
        req_headers: dict = None,
        content: Any = None,
        status_code: int = 200,
        resp_headers: dict[str, int] = None,
        reason: str = "",
    ):

        if req_headers is None:
            raise ValueError("You need to send headers!")

        if content is None:
            raise ValueError("you should set content!")

        self.url = url
        self.status_code = status_code
        self.reason = reason
        self.content = content
        self.headers = resp_headers or {
            "x-rate-limit-limit": 20,
            "x-rate-limit-remaining": 19,
            "x-rate-limit-reset": 58.0,
        }


def mock_post_wrapper(url: str, headers: dict, content=None):
    return MockPostResponse(url, req_headers=headers, content=content)


class MockZipFile:
    def __init__(self, data_list: list):
        if not isinstance(data_list, list):
            data_list = [data_list]
        self.data_list = data_list

    def namelist(self):
        return self.data_list

    def open(self, data: io.BytesIO):
        return data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def tns_delta_columns() -> list[str]:
    return (
        "objid name_prefix name type ra declination redshift "
        "internal_names reporters discoverydate lastmodified"
    ).split()


@pytest.fixture
def tns_delta_colstr(tns_delta_columns) -> str:
    formatted_string = ",".join(f'"{col}"' for col in tns_delta_columns)
    return formatted_string


@pytest.fixture
def daily_delta_rows():
    # 2023J matches T00, 2023K matches T01, 2023L matches nothing
    d0 = "2023-02-20 00:00:00"
    d1 = "2023-02-24 13:30:00"
    return [
        # id prfx  name    type  ra     dec  z     int.name  <-reporters->   dd  lm
        [1, "SN", "2023J", "SN", 180.0, 0.0, 0.01, "ZTF00A", "A. Aa, B. Bb", d0, d1],
        [2, "AT", "2023K", "", 90.0, 30.0, "", "ZTF00B", "A. Aa, B. Bb", d0, d1],
        [3, "SN", "2023L", "SN", 345.0, -80.0, 0.01, "", "A. Aa, B. Bb", d0, d1],
    ]


@pytest.fixture
def daily_delta_data(daily_delta_rows: list[list]):
    data_str = ""
    for row in daily_delta_rows:
        row_str = ",".join(f'"{str(x)}"' for x in row)
        data_str = data_str + row_str + "\n"
    return data_str


@pytest.fixture
def mock_daily_delta(tns_delta_colstr: str, daily_delta_data: str):
    return "20230224 00:00:00 - 23:59:59\n" + f"{tns_delta_colstr}\n" + daily_delta_data


@pytest.fixture
def hourly_delta_rows() -> list[list]:
    # 2023K matches with T01, 2023M matches nothing
    d0 = "2023-02-20 00:00:00"
    d1 = "2023-02-25 00:20:00"
    return [
        [4, "SN", "2023K", "SNIa", 90.0, 30.0, 0.02, "ZTF00B", "A. Aa, B. Bb", d0, d1],
        [5, "AT", "2023M", "", 355.0, -80.0, "", "", "A. Aa, B. Bb", d0, d1],
    ]


@pytest.fixture
def hourly_delta_data(hourly_delta_rows: list[list]) -> str:
    data_str = ""
    for row in hourly_delta_rows:
        row_str = ",".join(f'"{str(x)}"' for x in row)
        data_str = data_str + row_str + "\n"
    return data_str


@pytest.fixture
def mock_hourly_delta(tns_delta_colstr: str, hourly_delta_data: str):
    return (
        "20230225 00:00:00 - 00:59:59\n" + f"{tns_delta_colstr}\n" + hourly_delta_data
    )


@pytest.fixture(autouse=True)
def monkeypatched_tns_query(
    mock_daily_delta: str, mock_hourly_delta: str, monkeypatch: pytest.MonkeyPatch
):

    def mock_post(url: str, headers: dict = None):

        if "bad_request" in url:
            reas = "Bad request"
            return MockPostResponse(
                url, req_headers=headers, status_code=400, content=reas, reason=reas
            )
        elif "last_request" in url:
            reas = "OK"
            resp_headers = {
                "x-rate-limit-limit": 20,
                "x-rate-limit-remaining": 1,
                "x-rate-limit-reset": 0.1,
            }
            return MockPostResponse(
                url,
                req_headers=headers,
                content=reas,
                reason=reas,
                resp_headers=resp_headers,
            )

        elif "repeat_request" in url:
            reas = "Request repeat for this test"
            return MockPostResponse(
                url, req_headers=headers, status_code=429, content=reas, reason=reas
            )

        else:
            filename = url.split("/")[-1]
            filestem = filename.split(".")[0]
            if filestem == "tns_public_objects":
                content = mock_daily_delta.encode("utf-8")
            else:
                tstamp = filestem.split("tns_public_objects_", 1)[1]
                if len(tstamp) == 8:
                    # Asking for daily_delta:
                    if tstamp == "20230224":  # The last full day before t_ref
                        content = mock_daily_delta.encode("utf-8")
                    else:
                        content = f"{tstamp} 00:00:00 23:59:59".encode()
                elif len(tstamp) == 2:
                    # Asking for hourly_delta:
                    if tstamp == "00":
                        content = mock_hourly_delta.encode()
                    else:
                        content = f"230225 {tstamp}:00:00 - {tstamp}:59:59".encode()
                else:
                    msg = f"unknown request {filename} (extracted tstamp {tstamp})"
                    raise ValueError(msg)
            return MockPostResponse(url, headers, content=content)

    monkeypatch.setattr("requests.post", mock_post)
    monkeypatch.setattr("zipfile.ZipFile", MockZipFile)
    return
