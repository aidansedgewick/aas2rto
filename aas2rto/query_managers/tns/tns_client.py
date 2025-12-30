import io
import json
import requests
import time
import warnings
import zipfile
from logging import getLogger
from itertools import chain

logger = getLogger(__name__.split(".")[-1])

import pandas as pd

from astropy import units as u
from astropy.io.ascii.core import InconsistentTableError
from astropy.table import Table, vstack
from astropy.time import Time, TimeDelta


class TNSClientWarning(UserWarning):
    pass


class TNSClientError(Exception):
    pass


class TNSClient:
    tns_base_url = f"https://www.wis-tns.org"
    tns_search_url = f"{tns_base_url}/search"  # NOTE: no trailing slash
    tns_public_objects_url = f"{tns_base_url}/system/files/tns_public_objects"
    max_num_page = 50

    known_return_types = ("records", "pandas", "astropy")

    @staticmethod
    def build_tns_headers(tns_user: str, tns_uid: str):
        """Returns a dict with the correctly formatted string."""

        marker_dict = dict(tns_id=str(tns_uid), type="user", name=tns_user)
        marker_str = json.dumps(marker_dict)
        # marker_str is a literal string of a dict:
        #       '{"tns_id": "1234", "type": "user", "name": "your_name"}'

        tns_marker = f"tns_marker{marker_str}"
        return {"User-Agent": tns_marker}

    @staticmethod
    def check_search_parameters(parameters: dict):
        raise NotImplementedError()
        unknown_parameters = []
        for k, v in parameters.items():
            if k not in ALLOWED_TNS_PARAMETERS:
                unknown_parameters.append(k)

        if unknown_parameters:
            unk_param_str = "\n    ".join(unknown_parameters)
            raise TNSClientError(f"unknown_parameters\n    {unk_param_str}")

    @staticmethod
    def check_return_type(return_type: str):
        known_types_str = ", ".join(f"'{x}'" for x in TNSClient.known_return_types)
        return_type_err_msg = (
            f"Unknown return_type='\033[33;1m{return_type}\033[0m'. Choose from:\n    "
            f"{known_types_str}"
        )
        if return_type not in TNSClient.known_return_types:
            raise TNSClientError(return_type_err_msg)

    @staticmethod
    def get_empty_delta_results(return_type="pandas"):
        TNSClient.check_return_type(return_type)
        columns = "name ra declination lastmodified".split()
        if return_type == "records":
            return []
        elif return_type == "pandas":
            return pd.DataFrame(columns=columns)
        elif return_type == "astropy":
            return Table(names=columns)

    @staticmethod
    def build_search_parameter_string(search_params: dict):
        raise NotImplementedError()
        url_components = []
        for k, v in search_params.items():
            if isinstance(v, bool):
                v = int(v)
            url_components.append(f"&{k}={str(v)}")
        return "".join(url_components)

    @staticmethod
    def process_zip_content(content: bytes, return_type="pandas"):
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            with z.open(z.namelist()[0]) as f:
                if return_type == "records":
                    lines = f.read().decode().rstrip("\n").split("\n")
                    if len(lines) == 1:
                        return []
                    # line 0 is just the date eg. literal "25-11-24 00:00 - 23:59"
                    columns = lines[1].split('"')[1::2]
                    # splitting logic: literal str '"one", "two", "three,four"'
                    # would become: '' 'one' ',' 'two' ',' 'three,four' ''
                    # Note the first and last are empty, every 2nd is a literal comma.
                    records = []
                    for line in lines[2:]:
                        data = line.split('"')[1::2]
                        rec = {col: val for col, val in zip(columns, data)}
                        records.append(rec)
                    return records

                elif return_type == "pandas":
                    try:
                        df = pd.read_csv(f, skiprows=1)
                        # Need to skip the first row, which is just the date
                    except pd.errors.EmptyDataError:
                        df = pd.DataFrame(columns=["name", "lastmodified"])
                    return df
                elif return_type == "astropy":
                    try:
                        tab = Table.read(
                            f, header_start=1, data_start=2, format="ascii"
                        )
                    except InconsistentTableError:
                        tab = Table(names=["name", "lastmodified"])
                    return tab

    def __init__(self, tns_user: str, tns_uid: int, query_sleep_time: float = 3.0):
        self.tns_user = tns_user
        self.tns_uid = str(tns_uid)
        self.query_sleep_time = query_sleep_time

        self.tns_headers = self.build_tns_headers(self.tns_user, self.tns_uid)

    def do_post(self, url) -> requests.Response:
        return requests.post(url, headers=self.tns_headers)

    def wait_after_request(self, response: requests.Response):
        req_limit = response.headers.get("x-rate-limit-limit")
        req_remain = response.headers.get("x-rate-limit-remaining")
        req_reset = response.headers.get("x-rate-limit-reset")  # time until reset.

        logger.info(f"{req_remain}/{req_limit} requests left (resets in {req_reset}s)")

        try:
            if int(req_remain) < 2:
                logger.info(f"waiting {req_reset}s for reset...")
                time.sleep(int(req_reset) + 1.0)
            else:
                time.sleep(self.query_sleep_time)  # Should wait a little bit anyway...
        except TypeError as e:
            logger.warning(f"error: {type(e).__name__}: {e}")
            time.sleep(2.0)

    def request_delta(
        self,
        url: str,
        return_type: str = "pandas",
        process: bool = True,
        retries: int = 0,
        max_retries: int = 3,
    ):
        response = self.do_post(url)

        self.check_return_type(return_type)

        filename = url.split("/")[-1]

        self.wait_after_request(response)
        if response.status_code == 200:
            if process:
                return self.process_zip_content(
                    response.content, return_type=return_type
                )
            return response

        if response.status_code in [429] and retries < max_retries:
            # Repeat 429[Too many requests] - called wait_after() so should be ok now...
            msg = (
                f"status {response.status_code}: resubmit this request!"
                f" try {retries}/{max_retries} {filename}"
            )
            logger.error(msg)
            return self.request_delta(
                url,
                return_type=return_type,
                process=process,
                retries=retries + 1,
                max_retries=max_retries,
            )  # Recursive call to this function

        # We must have a bad response, and re-trying is not appropriate.
        msg = (
            f"{url}:\n    "
            f"response {response.status_code}, failed after {retries + 1} tries:\n"
            f"    Reason '{response.reason}')"
        )
        logger.error(msg)
        warnings.warn(TNSClientWarning(msg))
        if not process:
            return response
        return self.get_empty_delta_results(return_type=return_type)

    def get_tns_daily_delta(
        self, t_ref: Time = None, return_type="pandas", process=True
    ):
        if t_ref is None:
            t_ref = Time.now() - TimeDelta(24 * u.hour)
        timestamp = t_ref.strftime("%Y%m%d")
        filename = f"tns_public_objects_{timestamp}.csv.zip"
        url = f"{self.tns_public_objects_url}/{filename}"
        return self.request_delta(url, return_type=return_type, process=process)

    def get_tns_hourly_delta(
        self, hour: int = None, return_type="pandas", process=True
    ):
        if hour is None:
            t_ref = Time.now() - TimeDelta(1 * u.hour)
            hour = t_ref.datetime.hour

        filename = f"tns_public_objects_{hour:02d}.csv.zip"
        url = f"{self.tns_public_objects_url}/{filename}"
        return self.request_delta(url, return_type=return_type, process=process)

    def get_full_tns_archive(self, return_type="pandas", process=True):
        filename = f"tns_public_objects.csv.zip"
        url = f"{self.tns_public_objects_url}/{filename}"
        return self.request_delta(url, return_type=return_type, process=process)

    def process_query_response(self, response: requests.Response):
        raise NotImplementedError()

    def query(self, search_params: dict, return_type="pandas"):
        raise NotImplementedError()

        num_page = search_params.get("num_page", 50)
        if num_page > self.max_num_page:
            logger.warning(f"limit num_page={self.max_num_page} (from {num_page})")
            num_page = self.max_num_page
        search_params["num_page"] = num_page

        if search_params.get("format", "csv") != "csv":
            logger.warning(f"overwrite format='{format_param}' to 'csv'")
        search_params["format"] = "csv"

        self.check_search_parameters(search_params)
        param_str = self.build_search_parameter_string(search_params)
        page = 0
        results_list = []
        while True:
            url = f"{self.tns_base_url}/search?{param_str}&page={page}"
            response = self.do_post(url)
            self.wait_after_request(response)
            results = self.process_query_response(response, return_type=return_type)
            results_list.append(results)
            if len(results) < num_page:
                break
            page = page + 1

        if len(results_list) > 0:
            if return_type == "records":
                return chain.from_iterable(results_list)
            elif return_type == "pandas":
                return pd.concat(results_list, ignore_index=True)
            elif return_type == "astropy":
                return vstack(results_list)
        return


ALLOWED_TNS_PARAMETERS = (
    "discovered_period_value",
    "discovered_period_units",
    "unclassified_at",
    "classified_sne",
    "include_frb",
    "name",
    "name_like",
    "isTNS_AT",
    "public",
    "ra",
    "decl",
    "radius",
    "coords_unit",
    "reporting_groupid[]",
    "groupid[]",
    "classifier_groupid[]",
    "objtype[]",
    "at_type[]",
    "date_start[date]",
    "date_end[date]",
    "discovery_mag_min",
    "discovery_mag_max",
    "internal_name",
    "discoverer",
    "classifier",
    "spectra_count",
    "redshift_min",
    "redshift_max",
    "hostname",
    "ext_catid",
    "ra_range_min",
    "ra_range_max",
    "decl_range_min",
    "decl_range_max",
    "discovery_instrument[]",
    "classification_instrument[]",
    "associated_groups[]",
    "official_discovery",
    "official_classification",
    "at_rep_remarks",
    "class_rep_remarks",
    "frb_repeat",
    "frb_repeater_of_objid",
    "frb_measured_redshift",
    "frb_dm_range_min",
    "frb_dm_range_max",
    "frb_rm_range_min",
    "frb_rm_range_max",
    "frb_snr_range_min",
    "frb_snr_range_max",
    "frb_flux_range_min",
    "frb_flux_range_max",
    "format",
    "num_page",
)
