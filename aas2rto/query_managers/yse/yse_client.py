import io
import requests
from logging import getLogger

import pandas as pd

from astropy.table import Table


logger = getLogger(__name__.split(".")[-1])


class YSEClientError(Exception):
    pass


class YSEClientBadEndpointError(Exception):
    pass


class YSEClient:
    """See also https://github.com/berres2002/prep"""

    base_url = "https://ziggy.ucolick.org/yse"

    known_return_types = ("pandas", "astropy", "records")

    known_endpoints = {
        "surveyfields": ("field_id", "obs_group"),
        "surveyfieldmsbs": ("name", "active"),
        "surveyobservations": (
            "status_in",
            "obs_mjd_gte",
            "obs_mjd_lte",
            "mjd_requested_gte",
            "mjd_requested_lte",
            "survey_field",
            "obs_group",
            "ra_gt",
            "ra_lt",
            "dec_gt",
            "dec_lt",
        ),
        "transients": (
            "created_date_gte",
            "modified_date_gte",
            "status_in",
            "ra_gte",
            "ra_lte",
            "dec_gte",
            "dec_lte",
            "tag_in",
            "name",
        ),
        "classicalresources": ("instrument_name"),
    }

    def __init__(self, yse_username: str, yse_password: str):
        self.yse_auth = requests.auth.HTTPBasicAuth(yse_username, yse_password)

    @staticmethod
    def check_return_type(return_type: str):
        known_types_str = ", ".join(f"'{x}'" for x in YSEClient.known_return_types)
        return_type_err_msg = (
            f"Unknown return_type='\033[33;1m{return_type}\033[0m'. Choose from:\n    "
            f"{known_types_str}"
        )
        if return_type not in YSEClient.known_return_types:
            raise ValueError(return_type_err_msg)

    def query_explorer(
        self, query_id: int, process: bool = True, return_type: str = "pandas"
    ):
        url = f"{self.base_url}/explorer/{query_id}/download"
        response = requests.get(url, auth=self.yse_auth)
        if not process:
            return response

        if response.status_code != 200:
            return self.get_empty_query_results(return_type=return_type)

        # Remove some junk characters at the beginning...
        result_text = response.text.strip("ï»¿")
        return self.process_explorer_result(result_text, return_type=return_type)

    @staticmethod
    def process_explorer_result(text, return_type="pandas"):
        YSEClient.check_return_type(return_type)
        if return_type == "records":
            rows = text.split("\n")
            cols = rows[0].split(",")
            records = [
                {k: v for k, v in zip(cols, line.split(","))} for line in rows[1:]
            ]
            return records
        elif return_type == "pandas":
            return pd.read_csv(io.StringIO(text))
        elif return_type == "astropy":
            return Table.read(
                io.BytesIO(text.strip().encode()),
                data_start=1,
                format="csv",
                delimiter=",",
            )

    @staticmethod
    def get_empty_query_results(return_type="pandas", columns=("name",)):
        YSEClient.check_return_type(return_type)

        if return_type == "pandas":
            return pd.DataFrame(columns=columns)
        elif return_type == "astropy":
            return Table(names=columns)
        elif return_type == "records":
            return []
        else:
            raise Exception  # How did we get here?!

    def query_named_transient(self, name: str, process: bool = True):
        response: requests.Response = self.query_transients(
            {"name": name}, process=False
        )
        if response.status_code == 200:
            return response.json()["results"][0]
        return {}

    @staticmethod
    def prepare_query_url(endpoint: str, params: dict):
        expected_keys = YSEClient.known_endpoints.get(endpoint, None)
        known_endpoints_str = ",".join(
            f"'{x}" for x in YSEClient.known_endpoints.keys()
        )
        if expected_keys is None:
            msg = f"unknown endpoint '{endpoint}', choose from:\n    {known_endpoints_str}"
            raise YSEClientBadEndpointError(msg)
        unexpected_keys = set(params.keys()) - set(expected_keys)
        if len(unexpected_keys) > 0:
            exp_keys_str = ",".join(f"'{x}'" for x in expected_keys)
            unexp_keys_str = ",".join(f"'{x}" for x in unexpected_keys)
            raise YSEClientError(
                f"Unexpected keys {unexp_keys_str} for endpoint '{endpoint}, "
                f"choose from:\n   {exp_keys_str}"
            )
        query_str = "&".join(f"{key}={str(val)}" for key, val in params.items())
        return f"{YSEClient.base_url}/api/{endpoint}/?{query_str}"

    def query_survey_fields(
        self, params: dict, process: bool = True, return_type="pandas"
    ):
        url = self.prepare_query_url(endpoint="surveyfields", params=params)
        return self.query(url, process=process, return_type=return_type)

    def query_survey_field_msbs(
        self, params: dict, process: bool = True, return_type="pandas"
    ):
        url = self.prepare_query_url(endpoint="surveyfieldmsbs", params=params)
        return self.query(url, process=process, return_type=return_type)

    def query_survey_observations(
        self, params: dict, process: bool = True, return_type="pandas"
    ):
        url = self.prepare_query_url(endpoint="surveyobservations", params=params)
        return self.query(url, process=process, return_type=return_type)

    def query_transients(
        self, params: dict, process: bool = True, return_type="pandas"
    ):
        url = self.prepare_query_url(endpoint="transients", params=params)
        return self.query(url, process=process, return_type=return_type)

    def query(self, url: str, process: bool = True, return_type="pandas"):
        response = requests.get(url, auth=self.yse_auth)
        if not process:
            return response
        return self.process_query_response(response, return_type=return_type)

    @staticmethod
    def process_query_response(response: requests.Response, return_type="pandas"):
        YSEClient.check_return_type(return_type)

        result = response.json()["results"]
        if return_type == "records":
            return result
        elif return_type == "pandas":
            return pd.DataFrame.from_records(result)
        elif return_type == "astropy":
            return Table(rows=result)

    def query_lightcurve(
        self,
        name: str,
        process: bool = True,
        return_type="pandas",
        return_header: bool = False,
    ):
        url = f"{self.base_url}/download_photometry/{name}/"
        response = requests.get(url, auth=self.yse_auth)

        msg = f"Request for '{name}': status {response.status_code} {response.reason}"
        logger.debug(msg)
        if not process:
            return response
        if response.status_code != 200:
            lc = self.get_empty_lightcurve(return_type=return_type)
            if return_header:
                return lc, None
            return lc
        return self.process_lightcurve(
            response.text, return_type=return_type, return_header=return_header
        )

    @staticmethod
    def process_lightcurve(
        text: str, return_type: str = "pandas", return_header: bool = False
    ):
        all_rows = text.split("\n")
        header = {}
        columns = None
        data_rows = []
        for row in all_rows:
            if row.startswith("#"):
                # It must be a comment
                key, val = row[2:].split(":", 1)
                header[key.lower().strip()] = val.strip() or None
                continue
            if len(row.strip()) == 0:
                continue  # It must be a blank row
            if columns is None:
                # It must be the first non-comment row - the col names!
                columns = row.lower().split()
                continue
            data_rows.append(row.split())

        if len(data_rows) == 0:
            lc = YSEClient.get_empty_lightcurve(return_type=return_type)
        else:
            if return_type == "pandas":
                lc = pd.DataFrame(data=data_rows, columns=columns)
            elif return_type == "astropy":
                print(data_rows)
                print(columns)
                lc = Table(rows=data_rows, names=columns)
            elif return_type == "records":
                lc = [{k: v for k, v in zip(columns, row)} for row in data_rows]
            else:
                raise Exception  # How did we get here?
        if return_header:
            return lc, header
        return lc

    @staticmethod
    def get_empty_lightcurve(return_type="pandas"):
        YSEClient.check_return_type(return_type)

        columns = "mjd flt fluxcal fluxcalerr mag magerr magsys telescope instrument dq".split()
        if return_type == "pandas":
            return pd.DataFrame(columns=columns)
        elif return_type == "astropy":
            return Table(names=columns)
        elif return_type == "records":
            return []
