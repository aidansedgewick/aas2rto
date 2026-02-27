from __future__ import annotations

import abc
import copy
import itertools
import json
import requests
import time
from logging import getLogger
from typing import Callable

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.table import Table, vstack
from astropy.time import Time

logger = getLogger(__name__.split(".")[-1])


class FinkPortalClientError(Exception):  # Try to keep this standalone...
    pass


class FinkBadEndpointError(Exception):
    pass


class FinkDisallowedMethodError(Exception):
    pass


class FinkMissingRequiredParametersError(Exception):
    pass


RESPONSE_PROCESSORS = {
    "pandas": pd.DataFrame,
    "astropy": lambda data: Table(rows=data),
    "records": lambda data: data,
}


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
        """"""

    def __init__(self):
        pass  # In case of credentials required, implement here

    @staticmethod
    def fix_dict_keys_inplace(data: dict):
        key_lookup = {k: k.split(":")[-1] for k in data.keys()}
        for old_key, new_key in key_lookup.items():
            # Not dangerous, as not iterating over the dict we're updating.
            data[new_key] = data.pop(old_key)
        # NO RETURN - dictionary is modified "in place"...

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
        process=True,
        fix_keys=True,
        return_type=None,
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

        if process:
            return self.process_response(
                response, fix_keys=fix_keys, return_type=return_type
            )
        return response

    def process_response(
        self,
        response: requests.Response,
        fix_keys: bool = True,
        return_type: bool = None,
    ):
        data = json.loads(response.content)
        return self.process_data(data, fix_keys=fix_keys, return_type=return_type)

    def process_data(self, data, fix_keys=True, return_type="records", df_kwargs=None):
        if fix_keys:
            for row in data:
                self.fix_dict_keys_inplace(row)
        processor = RESPONSE_PROCESSORS.get(return_type)
        if processor is None:
            processors_str = ", ".join(f"'{x}'" for x in RESPONSE_PROCESSORS.keys())
            return_type_err_msg = (
                f"Unknown return_type='\033[33;1m{return_type}\033[0m'. Choose from:\n    "
                f"{processors_str}"
            )
            raise ValueError(return_type_err_msg)
        return processor(data)

    ##===== Methods which look common among surveys =====##
    # ...and do the same thing in each survey.

    def cutouts(self, method: str = "post", **payload):
        cutouts_kind = payload.get("kind", None)
        if (cutouts_kind is not None) and (cutouts_kind not in self.imtypes):
            raise ValueError(f"cutouts kind must be in 'imtypes'")

        client_kwargs = dict(method=method)
        return self.do_request("cutouts", **client_kwargs, **payload)

    def conesearch(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("conesearch", **client_kwargs, **payload)

    def sso(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("sso", **client_kwargs, **payload)

    def schema(self, method="get", **payload):
        client_kwargs = dict(method=method, fix_keys=False, return_type="records")
        return self.do_request("schema", **client_kwargs, **payload)

    def skymap(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("skymap", **client_kwargs, **payload)

    def statistics(self, fix_keys=True, return_type=None, **payload):
        return self.do_request(
            "statistics", fix_keys=fix_keys, return_type=return_type, **payload
        )

    def resolver(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("resolver", **client_kwargs, **payload)


class FinkZTFPortalClient(FinkBasePortalClient):
    api_url = "https://api.ztf.fink-portal.org/api/v1"
    target_id_key = "objectId"
    alert_id_key = "candid"
    imtypes = ("Difference", "Science", "Template")

    def anomaly(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("anomaly", **client_kwargs, **payload)

    def classes(
        self, method="get", process=True, fix_keys=True, return_type=None, **payload
    ):
        self.disallow_http_method(method, disallowed="post", endpoint="classes")
        client_kwargs = dict(
            method="get", process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("classes", **client_kwargs, **payload)

    def latests(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("latests", **client_kwargs, **payload)

    def objects(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        # NOT a base class method, as it has different behaviour in each survey.
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("objects", **client_kwargs, **payload)

    def schema(self, method="get", **payload):
        """get method disallowed for ZTF schema endpoint"""
        self.disallow_http_method(method, disallowed="post", endpoint="schema")
        return super().schema(method=method, **payload)

    def ssobulk(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("ssobulk", **client_kwargs, **payload)

    def ssocand(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("ssocand", **client_kwargs, **payload)

    def ssoft(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("ssoft", **client_kwargs, **payload)

    def tracklet(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("tracklet", **client_kwargs, **payload)

    ##===== Extra methods =====##

    def query_classifiers(
        self,
        t_start: Time,
        t_stop: Time = None,
        step=3 * u.h,
        return_type="pandas",
        fix_keys=True,
        n=1000,
        **payload,
    ):
        t_stop = t_stop or Time.now()
        step_day = step.to(u.day).value
        t_grid = Time(
            np.arange(t_start.mjd, (t_stop + step).mjd, step_day), format="mjd"
        )

        results = []
        payload["n"] = n
        for t1, t2 in zip(t_grid[:-1], t_grid[1:]):
            payload = copy.deepcopy(payload)
            t1_str = t1.strftime("%Y-%m-%d %H:%M")
            t2_str = t2.strftime("%Y-%m-%d %H:%M")
            payload["startdate"] = t1.iso
            payload["stopdate"] = t2.iso
            logger.info(f"start query {t1_str} - {t2_str}")
            try:
                result = self.latests(
                    fix_keys=fix_keys, return_type=return_type, **payload
                )
            except FinkBadEndpointError as e:
                break

            if len(result) == n:
                msg = f"{t1_str} - {t2_str} returned {n} rows. choose shorter timestep!"
                logger.warning(msg)
            if len(result) > 0:
                results.append(result)
        if return_type == "records":
            return itertools.chain.from_iterable(result)
        elif return_type == "pandas":
            if len(results) > 0:
                return pd.concat(results)
            return pd.DataFrame(columns=[self.target_id_key])
        elif return_type == "astropy":
            if len(results) > 0:
                return vstack(results)
            return Table(names=[self.target_id_key])
        else:
            msg = f"Unknown return_type='{return_type}': Use None, 'pandas', 'astropy'"
            raise ValueError(msg)

    def query_lightcurve(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        """ZTF lightcurves are queried with the OBJECTS endpoint"""
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.objects(**client_kwargs, **payload)


class FinkLSSTPortalClient(FinkBasePortalClient):
    api_url = "https://api.lsst.fink-portal.org/api/v1"
    target_id_key = "diaObjectId"
    alert_id_key = "diaSourceId"
    imtypes = ("Difference", "Science", "Template")

    def blocks(self, process=True, fix_keys=True, return_type=None, **payload):
        """No method keyword here - only http 'get' method is allowed"""
        client_kwargs = dict(
            method="get", process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("blocks", **client_kwargs, **payload)

    def fp(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("fp", **client_kwargs, **payload)

    def objects(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("objects", **client_kwargs, **payload)

    def sources(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("sources", **client_kwargs, **payload)

    def tags(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.do_request("tags", **client_kwargs, **payload)

    ##===== Extra methods =====##

    def query_lightcurve(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        """LSST lightcurves are queried with the 'SOURCES' endpoint"""
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        return self.sources(**client_kwargs, **payload)

    def query_classifiers(
        self, method="post", process=True, fix_keys=True, return_type=None, **payload
    ):
        client_kwargs = dict(
            method=method, process=process, fix_keys=fix_keys, return_type=return_type
        )
        raise NotImplementedError("No query classifiers and collate for LSST")


### Fancy decorator solution does not work for disallowing http methods because
### decorator does not know about default arguments. Would need to use inspect,
### which seems sketchy...
#
# def disallow_http_method(disallowed_method: str):
#     def disallow_decorator(func):
#         def check_method(endpoint, *args, method: str = None, **payload):
#             if method == disallowed_method:
#                 msg = f"method '{method}' disallowed for endpoint '{endpoint}'"
#                 raise FinkDisallowedMethodError(msg)
#             return func(endpoint, *args, method=method, **payload)

#         return check_method

#     return disallow_decorator
