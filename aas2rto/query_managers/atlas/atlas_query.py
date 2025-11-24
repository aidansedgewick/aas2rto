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


class AtlasQueryWarning(UserWarning):
    pass


class AtlasQuery:
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

    def __init__(self):
        pass

    @staticmethod
    def build_headers(token: str):
        return dict(Authorization=f"Token {token}", Accept="application/json")

    @classmethod
    def get_atlas_token(cls, username=None, password=None, display=True):
        logger.info("requesting ATLAS token from servers...")
        username = username or input("ATLAS username: ")
        password = password or getpass.getpass("ATLAS password [input hidden]: ")
        url = f"{AtlasQuery.atlas_base_url}/api-token-auth/"  # NOTE TRAILING SLASH!
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
    def check_return_type(return_type):
        known_types_str = ", ".join(f"'{x}'" for x in AtlasQuery.known_return_types)
        return_type_err_msg = (
            f"Unknown return_type='\033[33;1m{return_type}\033[0m'. Choose from:\n    "
            f"{known_types_str}"
        )
        if return_type not in AtlasQuery.known_return_types:
            raise ValueError(return_type_err_msg)

    @classmethod
    def get_empty_lightcurve(cls, return_type: str):
        cls.check_return_type(return_type)

        columns = "MJD m dm uJy duJy F err chi/N RA Dec x y maj min phi apfit mag5sig Sky Obs".split()
        if return_type == "records":
            return []
        elif return_type == "pandas":
            return pd.DataFrame(columns=columns)
        elif return_type == "astropy":
            return Table(names=columns)
        else:
            cls.check_return_type(return_type)  # Should never reach this...

    @classmethod
    def check_forcedphot_keys(cls, data: dict):
        # Try to avoid using aas2rto.utils here...
        unexected_kwargs = set(data.keys()) - set(cls.expected_forcedphot_kwargs)
        if unexected_kwargs:
            unexp_str = " ".join(f"'\033[31;1m{x}\033[0m'" for x in unexected_kwargs)
            exp_str = " ".join("'{x}'" for x in cls.expected_forcedphot_kwargs)
            msg = (
                f"unexpected keys when submitting forcedphot query: \n    {unexp_str}\n"
                f"known kwargs:\n    {exp_str}"
            )
            warnings.warn(AtlasQueryWarning(msg))

    @classmethod
    def submit_forced_photom_query(
        cls, data: dict, headers: dict, timeout: float = atlas_default_timeout
    ) -> requests.Response:
        cls.check_forcedphot_keys(data)
        res = requests.post(
            url=cls.atlas_default_queue_url, headers=headers, data=data, timeout=timeout
        )
        return res

    @classmethod
    def get_existing_queries(
        cls, headers, url=None, timeout=atlas_default_timeout
    ) -> dict:
        if url is None:
            url = cls.atlas_default_queue_url
        res = requests.get(url=url, headers=headers, timeout=timeout)
        return res.json()

    @classmethod
    def iterate_existing_queries(
        cls, headers: dict, max_query_time: float = 600.0
    ) -> Iterator[dict]:
        next_url = cls.atlas_default_queue_url
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
                query_data = cls.get_existing_queries(url=next_url, headers=headers)
            except Exception as e:
                msg = f"from url {next_url}, got exception {type(e)}: {e}"
                logger.warning(msg)
                warnings.warn(AtlasQueryWarning(msg))
                break

            next_url = query_data["next"]  # prepare for next query...
            query_results = query_data["results"]
            for query_result in query_results[::-1]:
                # Loop backwards to not destroy list order if queries are deleted (??)
                yield query_result

    @classmethod
    def recover_task_data(
        cls,
        task_url: str,
        headers: dict,
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
        cls.check_return_type(return_type)

        with requests.Session() as s:
            query_response: requests.Response = s.get(
                task_url, headers=headers, timeout=timeout
            )
            query_data: dict = query_response.json()

            # Is the query finished?
            finishtimestamp = query_data.get("finishtimestamp", None)
            if finishtimestamp is None:
                return query_response.status_code, None

            result_url = query_data.get("result_url", None)
            if result_url is None:
                # If there is finish_tstamp but no result_url, there must be an error.
                lightcurve = cls.get_empty_lightcurve(return_type=return_type)
                error_msg = query_data.get("error_msg", None)
                result_str = task_url.split("/")[-1]
                comment = query_data.get("comment", "")
                status_code = query_response.status_code
                msg = f"For {result_str} ('{comment}'):\n    {status_code}: {error_msg}"
                if error_msg == "No data returned":
                    logger.debug(msg)
                    status = cls.QUERY_EXISTS
                    # Overrule resp.status_code - we don't count 'empty' as an error!
                else:
                    logger.warning(msg)
                    status = query_response.status_code
            else:
                # There's a result_url - download and return it!
                lightcurve_response = s.get(result_url, headers=headers)
                lightcurve = cls.process_task_data(
                    lightcurve_response.text, return_type=return_type
                )
                status = query_response.status_code
        if delete_finished:
            cls.delete_tasks(task_url, headers=headers)
        return status, lightcurve

    @classmethod
    def process_task_data(cls, photom_data: str, return_type=None):
        if return_type not in cls.known_return_types:
            cls.raise_return_type(return_type)

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
            cls.raise_return_type(return_type)

    @classmethod
    def delete_tasks(cls, task_url_list: str, headers):
        if isinstance(task_url_list, str):
            task_url_list = list(task_url_list)

        logger.info(f"deleting {len(task_url_list)} tasks")
        with requests.Session() as s:
            for task_url in task_url_list:
                s.delete(task_url, headers=task_url)
