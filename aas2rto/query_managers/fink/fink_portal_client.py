from __future__ import annotations

import abc
import copy
import gzip
import itertools
import json
import requests
import time
from io import BytesIO
from logging import getLogger
from typing import Callable

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.time import Time

# No aas2rto imports: try to keep this standalone

logger = getLogger(__name__.split(".")[-1])


class FinkPortalClientError(Exception):
    pass


class FinkBadEndpointError(Exception):
    pass


class FinkDisallowedMethodError(Exception):
    pass


class FinkMissingRequiredParametersError(Exception):
    pass


def readstamp(
    stamp: str, return_type: str = "array", gzipped: bool = True, hdu: int = 0
) -> np.array:
    """Read the stamp data inside an alert.
    Modified from Fink's utils:
    https://github.com/astrolabsoftware/fink-science-portal/blob/master/apps/utils.py#L216 ...

    Parameters
    ----------
    stamp: str
        String containing binary data for the stamp
    return_type: str
        Data block of HDU (`array`) or original FITS uncompressed (`FITS`) as file-object.
        Default is `array`.
    hdu: int
        If return_type is not None:


    Returns
    -------
    data: np.array
        2D array containing image data (`array`) or FITS file uncompressed as file-object (`FITS`)
    """

    def extract_stamp(fitsdata):
        with fits.open(fitsdata, ignore_missing_simple=True) as hdul:
            if return_type == "array":
                data = hdul[hdu].data
            elif return_type == "FITS":
                data = BytesIO()
                hdul.writeto(data)
                data.seek(0)
        return data

    if not isinstance(stamp, BytesIO):
        stamp = BytesIO(stamp)

    if gzipped:
        try:
            with gzip.open(stamp, "rb") as f:
                return extract_stamp(BytesIO(f.read()))
        except gzip.BadGzipFile as e:
            logger.error(e)
            return extract_stamp(stamp)
    else:
        return extract_stamp(stamp)


TABLE_RESPONSE_PROCESSORS = {
    "pandas": pd.DataFrame,
    "astropy": lambda data: Table(rows=data),
    "records": lambda data: data,
}
CUTOUT_RESPONSE_PROCESSORS = {"array": lambda data: np.array(data), "fits": readstamp}
DEFAULT_TABLE_RETURN_TYPE = "records"


def fix_dict_keys_inplace(data: dict):
    key_lookup = {k: k.split(":")[-1] for k in data.keys()}
    for old_key, new_key in key_lookup.items():
        # Not dangerous, as not iterating over the dict we're updating.
        data[new_key] = data.pop(old_key)
    # NO RETURN - dictionary is modified "in place"...


class FinkBasePortalClient(abc.ABC):

    TIMEOUT_LIKELY = 58.0

    @property
    @abc.abstractmethod
    def api_url(self) -> str:
        """The url for the fink endpoint eg. api.{survey}.fink-portal.org"""

    @property
    @abc.abstractmethod
    def imtypes(self) -> tuple[str]:
        """"""

    @property
    @abc.abstractmethod
    def target_id_key(self) -> str:
        """"""

    @property
    @abc.abstractmethod
    def alert_id_key(self) -> str:
        """"""

    @abc.abstractmethod
    def query_lightcurve(self, *args, **payload) -> pd.DataFrame | Table | list[dict]:
        """Define which endpoint should be used for querying a lightcurve.
        eg. they are different for LSST and ZTF!"""

    @abc.abstractmethod
    def query_classifiers(self, *args, **payload):
        """Define which endpoint should be used for querying classifiers.
        eg. they are different for LSST and ZTF"""

    def __init__(self):
        pass  # In case of credentials required, implement here

    @staticmethod
    def process_kwargs(**kwargs):
        """Remove trailing underscores from eg. 'class_'
        (python keywords which are) used as keywords
        use dict unpacking (ie. **kwargs), instead of single arg as sometimes there
        may be NO kwargs to process.
        """
        return {k.rstrip("_"): v for k, v in kwargs.items()}

    @staticmethod
    def disallow_http_method(method, disallowed, endpoint: str = "<unknown>"):
        if method == disallowed:
            msg = f"method '{method}' disallowed for endpoint '{endpoint}'"
            raise FinkDisallowedMethodError(msg)
        return

    @staticmethod
    def raise_client_error(response: requests.Response, endpoint="<unknown>"):
        status = response.status_code
        reason = response.reason
        msg = f"query for '{endpoint}' returned status {status} ({reason})"
        logger.error(msg)
        if isinstance(response.content, bytes):
            raise FinkPortalClientError(response.content.decode())
        else:
            raise FinkPortalClientError(response.content)

    def do_request(
        self,
        endpoint: str,
        method: str = "post",
        **payload,
    ):
        payload = self.process_kwargs(**payload)
        if method == "post":
            response: requests.Response = requests.post(
                f"{self.api_url}/{endpoint}", json=payload
            )
        elif method == "get":
            response: requests.Response = requests.get(
                f"{self.api_url}/{endpoint}", params=payload
            )
        else:
            msg = f"Unknown method '{method}'. Choose 'post' or 'get'"
            raise ValueError(msg)
        if response.status_code != 200:
            self.raise_client_error(response, endpoint=endpoint)
        return response

    @staticmethod
    def process_table_data(data, fix_keys: str = True, return_type: str = None):
        return_type = return_type or DEFAULT_TABLE_RETURN_TYPE
        if fix_keys:
            for row in data:
                fix_dict_keys_inplace(row)
        processor = TABLE_RESPONSE_PROCESSORS.get(return_type, None)
        if processor is None:
            processors_str = ", ".join(
                f"'{x}'" for x in TABLE_RESPONSE_PROCESSORS.keys()
            )
            return_type_err_msg = (
                f"Unknown return_type='\033[33;1m{return_type}\033[0m'. "
                f"Choose from:\n    {processors_str}"
            )
            raise ValueError(return_type_err_msg)
        return processor(data)

    @staticmethod
    def process_cutout_data(data: dict, output_format: str, fix_keys: bool = True):

        if fix_keys:
            fix_dict_keys_inplace(data)
        processor = CUTOUT_RESPONSE_PROCESSORS[output_format]

        cutout_data = {}
        for key, data in data.items():
            cutout_data[key] = processor(data)
        return cutout_data

    ##===== Methods which look common among surveys =====##
    # ...and do the same thing in each survey.

    def cutouts(
        self,
        method: str = "post",
        process: bool = True,
        fix_keys: bool = True,
        **payload,
    ):
        cutouts_kind = payload.get("kind", None)
        output_format = payload.get("output-format", None)

        # Do some checks on parameters.
        if cutouts_kind is not None:
            if cutouts_kind != "All":
                if cutouts_kind not in self.imtypes:
                    raise ValueError(f"cutouts kind must be 'All' '{self.imtypes}'")
            if cutouts_kind == "All":
                if output_format is None:
                    output_format = "array"
                    payload["output-format"] = output_format
                else:
                    if output_format != "array":
                        msg = f"for kind 'All', output-format must be 'array'"
                        raise ValueError(msg)
        else:
            cutouts_kind = "All"
            output_format = "array"
            payload["kind"] = cutouts_kind
            payload["output-format"] = output_format

        response: requests.Response = self.do_request(
            "cutouts", method=method, **payload
        )
        if process:
            return self.process_cutout_data(
                response, output_format=output_format, fix_keys=fix_keys
            )
        return response

    def conesearch(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        response: requests.Response = self.do_request(
            "conesearch", method=method, **payload
        )
        if process:
            return self.process_table_data(fix_keys=fix_keys, return_type=return_type)
        return response

    def sso(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        response: requests.Response = self.do_request("sso", method=method, **payload)
        if process:
            return self.process_table_data(
                response.json(), fix_keys=fix_keys, return_type=return_type
            )
        return response

    def schema(self, method: str = "get", process: bool = True, **payload):
        self.disallow_http_method(method, disallowed="post", endpoint="schema")
        response: requests.Response = self.do_request(
            "schema", method=method, **payload
        )
        if process:
            return response.json()
        return response

    def skymap(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        response: requests.Response = self.do_request(
            "skymap", method=method, **payload
        )
        if process:
            return self.process_table_data(
                response.json(), fix_keys=fix_keys, return_type=return_type
            )
        return response

    def statistics(
        self,
        process: bool = True,
        fix_keys: bool = True,
        return_type: str = None,
        **payload,
    ):
        response: requests.Response = self.do_request("statistics", **payload)
        if process:
            return self.process_table_data(
                response.json(), fix_keys=fix_keys, return_type=return_type
            )
        return response

    def resolver(self, method="post", process=True, fix_keys=True, **payload):
        response: requests.Response = self.do_request(
            "resolver", method=method, **payload
        )
        if process:
            return self.process_table_data(
                response.json(), fix_keys=fix_keys, return_type="records"
            )
        return response

    def query_and_consolidate(
        self,
        endpoint_method: Callable,
        t_start: Time = None,
        t_stop: Time = None,
        step: u.Quantity = 24 * u.h,
        return_type: str = "pandas",
        fix_keys: bool = True,
        n: int = 1000,
        depth: int = 0,
        max_depth: int = 6,
        **payload,
    ):
        """
        For an endpoint with `startdate` and `stopdate`, call query on a time grid,
        and consolidate the results.

        If the number of rows returned equals the maximum allowed, call recursively and
        try a smaller interval

        Parameters
        ----------
        endpoint_method : Callable
            eg. ztf_client.latests
        t_start : astropy.time.Time, optional
            Defaults to Time.now() - 1 day
        t_stop : astropy.time.Time, optional
            Defaults to Time.now()
        step : astropy.units.Quantity
            Default is 3.0 * u.h
        return_type : str
            Choose 'records', 'astropy', 'pandas'
        fix_keys : bool, default=True
            Remove FINK-added column prefixes
        n : int, default = 1000
            Maximum rows per query
        **payload
            All other parameters passed to the endpoint you are querying.

        """

        if depth > max_depth:
            raise RecursionError(f"Depth {depth} > {max_depth}")

        ##===== Prepare a grid of dates.
        t_stop = t_stop or Time.now()
        t_start = t_start or t_stop - 1.0 * u.day
        step_day = step.to(u.day).value
        t_grid = Time(
            np.arange(t_start.mjd, (t_stop + step).mjd, step_day), format="mjd"
        )

        query_results = []
        payload["n"] = n

        ##===== Loop through a grid of dates.
        for t1, t2 in zip(t_grid[:-1], t_grid[1:]):
            payload = copy.deepcopy(payload)
            t1_str = t1.strftime("%Y-%m-%d %H:%M")
            t2_str = t2.strftime("%Y-%m-%d %H:%M")
            payload["startdate"] = t1.iso
            payload["stopdate"] = t2.iso
            logger.info(f"start query {t1_str} - {t2_str}")
            result = endpoint_method(
                fix_keys=fix_keys, return_type=return_type, **payload
            )
            logger.info("  " * depth + f"returned {len(result)}")

            if len(result) == n:
                msg = f"returned {n} rows: recurse depth={depth+1}"
                logger.warning(msg)
                try:
                    sub_result = self.query_and_consolidate(
                        endpoint_method,
                        **dict(t_start=t1, t_stop=t2, step=step / 6.0),  # grid params
                        **dict(return_type=return_type, fix_keys=fix_keys),  # client
                        **dict(depth=depth + 1, max_depth=max_depth),  # recursion
                        **payload,
                    )
                    query_results.append(sub_result)
                except RecursionError:
                    logger.error("Cannot recurse deeper")
                    query_results.append(result)  #
            else:
                if len(result) > 0:
                    logger.info("  " * depth + f"returned {len(result)}")
                    query_results.append(result)

        ##===== Now combine all the results.
        if return_type == "records":
            return itertools.chain.from_iterable(result)
        elif return_type == "pandas":
            if len(query_results) > 0:
                return pd.concat(query_results)
            return pd.DataFrame(columns=[self.target_id_key])
        elif return_type == "astropy":
            if len(query_results) > 0:
                return vstack(query_results)
            return Table(names=[self.target_id_key])
        else:
            msg = f"Unknown return_type='{return_type}': Use None, 'pandas', 'astropy'"
            raise ValueError(msg)


class FinkZTFPortalClient(FinkBasePortalClient):
    api_url = "https://api.ztf.fink-portal.org/api/v1"
    target_id_key = "objectId"
    alert_id_key = "candid"
    imtypes = ("Difference", "Science", "Template")

    def anomaly(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        response: requests.Response = self.do_request(
            "anomaly", method=method, **payload
        )
        if process:
            return self.process_table_data(
                response.json(), fix_keys=fix_keys, return_type=return_type
            )
        return response

    def classes(
        self, method="get", process=True, fix_keys=True, return_type=None, **payload
    ):
        self.disallow_http_method(method, disallowed="post", endpoint="classes")
        response: requests.Response = self.do_request(
            "classes", method="get", **payload
        )
        if process:
            return response.json()
        return response

    def latests(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        response: requests.Response = self.do_request(
            "latests", method=method, **payload
        )
        if process:
            return self.process_table_data(
                response.json(), fix_keys=fix_keys, return_type=return_type
            )
        return response

    def objects(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        # NOT a base class method, as it has different behaviour in each survey.
        response: requests.Response = self.do_request(
            "objects", method=method, **payload
        )
        if process:
            return self.process_table_data(
                response.json(), fix_keys=fix_keys, return_type=return_type
            )
        return response

    def schema(self, method: str = "post", **payload):
        """'get' method disallowed for ZTF schema endpoint"""
        self.disallow_http_method(method, disallowed="post", endpoint="schema")
        return super().schema(method=method, **payload)

    def ssobulk(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        response: requests.Response = self.do_request(
            "ssobulk", method=method, **payload
        )
        if process:
            return self.process_table_data(
                response.json(), fix_keys=fix_keys, return_type=return_type
            )
        return response

    def ssocand(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        response: requests.Response = self.do_request(
            "ssocand", method=method, **payload
        )
        if process:
            return self.process_table_data(
                response.json(), fix_keys=fix_keys, return_type=return_type
            )
        return response

    def ssoft(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        response: requests.Response = self.do_request("ssoft", method=method, **payload)
        if process:
            return self.process_table_data(
                response.json(), fix_keys=fix_keys, return_type=return_type
            )
        return response

    def tracklet(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        response: requests.Response = self.do_request(
            "tracklet", method=method, **payload
        )
        if process:
            return self.process_table_data(
                response.json(), fix_keys=fix_keys, return_type=return_type
            )
        return response

    ##===== Extra methods =====##

    def consolidate_latests(self, **kwargs):
        """
        All kwargs from query_and_consolidate
        """
        return self.query_and_consolidate(self.latests, **kwargs)

    # def consolidate_anomaly(self, **kwargs):
    #    TODO: Need some way to swap the 'startdate' for 'start_date'
    #    return self.query_and_consolidate(self.anomaly, **kwargs)

    def query_lightcurve(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        """ZTF lightcurves are queried with the `objects` endpoint"""
        return self.objects(
            method=method,
            process=process,
            fix_keys=fix_keys,
            return_type=return_type,
            **payload,
        )

    def query_classifiers(self, **payload):
        """ZTF classifiers are queried with the `latests` endpoint

        All kwargs are the same as from query_and_consolidate.
        """
        return self.consolidate_latests(**payload)


class FinkLSSTPortalClient(FinkBasePortalClient):
    api_url = "https://api.lsst.fink-portal.org/api/v1"
    target_id_key = "diaObjectId"
    alert_id_key = "diaSourceId"
    imtypes = ("Difference", "Science", "Template")

    def blocks(
        self, method="get", process: bool = True, fix_keys: bool = True, **payload
    ):
        """No method keyword here - only http 'get' method is allowed"""
        self.disallow_http_method(method, disallowed="post", endpoint="blocks")
        response: requests.Response = self.do_request("blocks", method="get", **payload)
        if process:
            return response.json()
        return response

    def fp(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):

        response: requests.Response = self.do_request("fp", method=method, **payload)
        if process:
            return self.process_table_data(
                response.json(), fix_keys=True, return_type=return_type
            )
        return response

    def objects(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        response: requests.Response = self.do_request(
            "objects", method=method, **payload
        )
        if process:
            return self.process_table_data(
                response.json(), fix_keys=True, return_type=return_type
            )
        return response

    def sources(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        response: requests.Response = self.do_request(
            "sources", method=method, **payload
        )
        if process:
            return self.process_table_data(
                response.json(), fix_keys=True, return_type=return_type
            )
        return response

    def tags(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        response: requests.Response = self.do_request("tags", method=method, **payload)
        if process:
            return self.process_table_data(
                response.json(), fix_keys=True, return_type=return_type
            )
        return response

    ##===== Extra methods =====##

    def consolidate_tags(self, **kwargs):
        return self.query_and_consolidate(self.tags, **kwargs)

    def query_lightcurve(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        """LSST lightcurves are queried with the `sources` endpoint"""
        return self.sources(
            method=method,
            process=process,
            fix_keys=fix_keys,
            return_type=return_type,
            **payload,
        )

    def query_classifiers(self, **kwargs):
        """
        LSST classifiers are queried with the `tags` endpoint.
        """

        return self.consolidate_tags(**kwargs)
