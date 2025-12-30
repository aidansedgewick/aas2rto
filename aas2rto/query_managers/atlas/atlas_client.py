import getpass
import io
import requests
import time
import warnings
from collections.abc import Iterator
from logging import getLogger
from typing import Any

import pandas as pd

from astropy.table import Table


logger = getLogger(__name__.split(".")[-1])


class AtlasClientWarning(UserWarning):
    pass


class AtlasClient:
    atlas_base_url = "https://fallingstar-data.com/forcedphot"
    atlas_default_queue_url = f"{atlas_base_url}/queue/"
    atlas_default_timeout = 90.0
    known_return_types = ("pandas", "astropy", "records")

    expected_forcedphot_kwargs = (
        "radeclist",
        "ra",
        "dec",
        "mjd_min",
        "mjd_max",
        "comment",
        "enable_propermotion",
        "radec_epoch_year",
        "propermotion_ra",
        "propermotion_dec",
        "send_email",
    )

    # these are the normal http response codes...
    QUERY_EXISTS = 200
    QUERY_SUBMITTED = 201
    QUERY_BAD_REQUEST = 400
    QUERY_THROTTLED = 429

    @staticmethod
    def build_atlas_headers(token: str):
        return dict(Authorization=f"Token {token}", Accept="application/json")

    @staticmethod
    def get_atlas_token(username=None, password=None, display=True):
        logger.info("requesting ATLAS token from servers...")
        username = username or input("ATLAS username: ")
        password = password or getpass.getpass("ATLAS password [input hidden]: ")
        url = f"{AtlasClient.atlas_base_url}/api-token-auth/"  # NOTE TRAILING SLASH!
        response = requests.post(url, data=dict(username=username, password=password))
        try:
            token = response.json()["token"]
            if display:
                print(f"Your ATLAS token (copy this into your config)\n:    {token}")
            return token
        except KeyError:
            print(response.json())
            raise

    @staticmethod
    def check_return_type(return_type: str):
        known_types_str = ", ".join(f"'{x}'" for x in AtlasClient.known_return_types)
        return_type_err_msg = (
            f"Unknown return_type='\033[33;1m{return_type}\033[0m'. Choose from:\n    "
            f"{known_types_str}"
        )
        if return_type not in AtlasClient.known_return_types:
            raise ValueError(return_type_err_msg)

    @staticmethod
    def get_empty_lightcurve(return_type: str = "pandas"):
        AtlasClient.check_return_type(return_type)

        columns = "MJD m dm uJy duJy F err chi/N RA Dec x y maj min phi apfit mag5sig Sky Obs".split()
        if return_type == "records":
            return []
        elif return_type == "pandas":
            return pd.DataFrame(columns=columns)
        elif return_type == "astropy":
            return Table(names=columns)
        else:
            raise Exception("How did we get here?")

    @staticmethod
    def check_forcedphot_keys(data: dict):
        # Try to avoid using aas2rto.utils here...
        unexected_kwargs = set(data.keys()) - set(
            AtlasClient.expected_forcedphot_kwargs
        )
        if unexected_kwargs:
            unexp_str = " ".join(f"'\033[31;1m{x}\033[0m'" for x in unexected_kwargs)
            exp_str = " ".join("'{x}'" for x in AtlasClient.expected_forcedphot_kwargs)
            msg = (
                f"unexpected keys when submitting forcedphot query: \n    {unexp_str}\n"
                f"known kwargs:\n    {exp_str}"
            )
            warnings.warn(AtlasClientWarning(msg))

    @staticmethod
    def process_task_data(photom_data: str, return_type: str = "pandas"):
        AtlasClient.check_return_type(return_type)

        if isinstance(photom_data, requests.Response):
            photom_data = photom_data.text

        photom_data = photom_data.replace("###", "").rstrip()  # remove trailing "\n"
        rows = photom_data.split("\n")

        if return_type == "records":
            cols = rows[0].split()
            records = [{k: v for k, v in zip(cols, line.split())} for line in rows[1:]]
            return records
        elif return_type == "pandas":
            input_bytes = io.StringIO(photom_data)
            return pd.read_csv(input_bytes, sep=r"\s+")
        elif return_type == "astropy":
            return Table.read(rows, data_start=1, format="ascii", delimiter=r"\s")
        else:
            raise Exception("How did we get here?")  # Exc raised in check_return_type()

    def __init__(self, token: str):
        self.token = token
        self.headers = self.build_atlas_headers(self.token)

    def submit_forced_photom_query(
        self, data: dict, timeout: float = atlas_default_timeout
    ) -> requests.Response:
        self.check_forcedphot_keys(data)
        response = requests.post(
            url=self.atlas_default_queue_url,
            headers=self.headers,
            data=data,
            timeout=timeout,
        )
        return response

    def get_existing_queries(self, url=None, timeout=atlas_default_timeout) -> dict:
        if url is None:
            url = self.atlas_default_queue_url
        res = requests.get(url=url, headers=self.headers, timeout=timeout)
        return res.json()

    def iterate_existing_queries(self, max_query_time: float = 600.0) -> Iterator[dict]:
        next_url = self.atlas_default_queue_url
        start_time = time.perf_counter()
        while next_url is not None:
            query_time = time.perf_counter() - start_time
            if query_time > max_query_time:
                msg = (
                    f"Don't try next page url. "
                    f"Break as query time {query_time:.1f}s > max {max_query_time:.1f}s"
                )
                logger.warning(msg)
                break
            try:
                query_data = self.get_existing_queries(url=next_url)
            except Exception as e:
                msg = f"from url {next_url}, got exception {type(e)}: {e}"
                logger.warning(msg)
                warnings.warn(AtlasClientWarning(msg))
                break

            next_url = query_data["next"]  # prepare for next query...
            query_results = query_data["results"]
            for query_result in query_results[::-1]:
                # Loop backwards to not destroy list order if queries are deleted (??)
                yield query_result

    def recover_task_data(
        self,
        task_url: str,
        return_type: str = None,
        timeout: float = None,
        delete_finished=False,
    ) -> tuple[int, Any]:
        """
        Returns
        -------
        status_code : int
        lightcurve : Any (depends on input return_type)
        """
        self.check_return_type(return_type)

        with requests.Session() as s:
            query_response: requests.Response = s.get(
                task_url, headers=self.headers, timeout=timeout
            )
            query_data: dict = query_response.json()

            # Is the query finished?
            finishtimestamp = query_data.get("finishtimestamp", None)
            if finishtimestamp is None:
                return query_response.status_code, None

            result_url = query_data.get("result_url", None)
            if result_url is None:
                # If there is finish_tstamp but no result_url, there must be an error.
                lightcurve = self.get_empty_lightcurve(return_type=return_type)
                error_msg = query_data.get("error_msg", None)
                result_str = task_url.split("/")[-1]
                comment = query_data.get("comment", "")
                status_code = query_response.status_code
                msg = f"For {result_str} ('{comment}'):\n    {status_code}: {error_msg}"
                if error_msg == "No data returned":
                    logger.debug(msg)
                    status = self.QUERY_EXISTS
                    # Overrule resp.status_code - we don't count 'empty' as an error!
                else:
                    logger.warning(msg)
                    status = query_response.status_code
            else:
                # There's a result_url - download and return it!
                lightcurve_response = s.get(result_url, headers=self.headers)
                lightcurve = self.process_task_data(
                    lightcurve_response.text, return_type=return_type
                )
                status = query_response.status_code
        if delete_finished:
            self.delete_tasks(task_url)
        return status, lightcurve

    def delete_tasks(self, task_url_list: str):
        if isinstance(task_url_list, str):
            task_url_list = list(task_url_list)

        logger.info(f"deleting {len(task_url_list)} tasks")
        with requests.Session() as s:
            for task_url in task_url_list:
                s.delete(task_url, headers=self.headers)
