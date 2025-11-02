import abc
import copy
import json
import requests
import time
from logging import getLogger

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.time import Time

logger = getLogger(__name__.split(".")[-1])


class FinkQueryError(Exception):
    pass


class FinkBaseQuery(abc.ABC):

    @property
    @abc.abstractmethod
    def api_url(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def imtypes(self) -> tuple[str]:
        pass

    @property
    @abc.abstractmethod
    def id_key(self) -> str:
        pass

    def __init__(self):
        pass

    @staticmethod
    def fix_dict_keys_inplace(data: dict):
        key_lookup = {k: k.split(":")[-1] for k in data.keys()}
        for old_key, new_key in key_lookup.items():
            # Not dangerous, as not iterating over the dict we're updating.
            data[new_key] = data.pop(old_key)
        # don't return dictionary, dict is modified inplace...

    @classmethod
    def process_data(cls, data, fix_keys=True, return_df=True, df_kwargs=None):
        if fix_keys:
            for row in data:
                cls.fix_dict_keys_inplace(row)
        if not return_df:
            return data
        return pd.DataFrame(data)

    @classmethod
    def process_response(cls, res: requests.Response, fix_keys=True, return_df=True):
        if res.status_code in [404, 500, 504]:
            logger.error("\033[31;1FinkQuery: error rasied\033[0m")
            if res.elapsed.total_seconds() > 58.0:
                logger.error("likely a timeout error")
            raise FinkQueryError(res.content.decode())
        data = json.loads(res.content)
        return cls.process_data(data, fix_keys=fix_keys, return_df=return_df)

    @classmethod
    def do_post(
        cls, service: str, process=True, fix_keys=True, return_df=True, **kwargs
    ):
        kwargs = cls.process_kwargs(**kwargs)
        response = requests.post(f"{cls.api_url}/{service}", json=kwargs)
        if response.status_code != 200:
            time.sleep(0.1)
            msg = f"query for '{service}' returned status {response.status_code}"
            logger.warning(msg)
        if process:
            return cls.process_response(
                response, fix_keys=fix_keys, return_df=return_df
            )
        return response

    @classmethod
    def do_get(cls, service: str, **kwargs):
        response = requests.get(f"{cls.api_url}/{service}", json=kwargs)
        if response.status_code != 200:
            raise FinkQueryError(response.content.decode())
        try:
            return json.loads(response.content)
        except Exception as e:
            return response.content.decode()

    @staticmethod
    def process_kwargs(**kwargs):
        """Remove trailing underscores from eg. 'class_'
        (python keywords which are) used as keywords"""
        return {k.rstrip("_"): v for k, v in kwargs.items()}

    @classmethod
    def objects(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post("objects", fix_keys=fix_keys, return_df=return_df, **kwargs)

    @classmethod
    def cutouts(cls, **kwargs):
        if kwargs.get("kind", None) not in cls.imtypes:
            raise ValueError(f"provide `kind` as one of {cls.imtypes}")
        return cls.do_post("cutouts", fix_keys=False, return_df=False, **kwargs)

    @classmethod
    def latests(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post("latests", fix_keys=fix_keys, return_df=return_df, **kwargs)

    @classmethod
    def latests_query_and_collate(
        cls,
        t_start: Time,
        t_stop: Time = None,
        step=3 * u.h,
        return_df=True,
        fix_keys=True,
        n=1000,
        **kwargs,
    ):
        t_stop = t_stop or Time.now()
        step_day = step.to(u.day).value
        t_grid = Time(
            np.arange(t_start.mjd, (t_stop + step).mjd, step_day), format="mjd"
        )

        results = []
        kwargs["n"] = n
        for t1, t2 in zip(t_grid[:-1], t_grid[1:]):
            payload = copy.deepcopy(kwargs)
            t1_str = t1.strftime("%Y-%m-%d %H:%M")
            t2_str = t2.strftime("%Y-%m-%d %H:%M")
            payload["startdate"] = t1.iso
            payload["stopdate"] = t2.iso
            logger.info(f"start query {t1_str} - {t2_str}")
            result = cls.latests(fix_keys=fix_keys, return_df=return_df, **payload)
            if len(result) == n:
                msg = f"{t1_str} - {t2_str} returned {n} rows. choose shorter timestep!"
                logger.warning(msg)
            if len(result) > 0:
                results.append(result)
        if return_df:
            if len(results) > 0:
                return pd.concat(results)
            return pd.DataFrame(columns=[cls.id_key])
        return results

    @classmethod
    def classes(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_get("classes", **kwargs)

    @classmethod
    def explorer(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post("explorer", fix_keys=fix_keys, return_df=return_df, **kwargs)

    @classmethod
    def conesearch(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post(
            "conesearch", fix_keys=fix_keys, return_df=return_df, **kwargs
        )

    @classmethod
    def sso(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post("sso", fix_keys=fix_keys, return_df=return_df, **kwargs)

    @classmethod
    def ssocand(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post("ssocand", fix_keys=fix_keys, return_df=return_df, **kwargs)

    @classmethod
    def resolver(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post("resolver", fix_keys=fix_keys, return_df=return_df, **kwargs)

    @classmethod
    def tracklet(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post("tracklet", fix_keys=fix_keys, return_df=return_df, **kwargs)

    @classmethod
    def schema(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_get("schema", fix_keys=fix_keys, return_df=return_df, **kwargs)

    @classmethod
    def skymap(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post("skymap", fix_keys=fix_keys, return_df=return_df, **kwargs)

    @classmethod
    def statistics(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post(
            "statistics", fix_keys=fix_keys, return_df=return_df, **kwargs
        )

    @classmethod
    def anomaly(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post("anomaly", fix_keys=fix_keys, return_df=return_df, **kwargs)

    @classmethod
    def ssoft(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post("ssoft", fix_keys=fix_keys, return_df=return_df, **kwargs)


class FinkZTFQuery(FinkBaseQuery):
    api_url = "https://api.ztf.fink-portal.org/api/v1"
    id_key = "objectId"
    imtypes = ("Difference", "Template", "Science")


class FinkLSSTQuery(FinkBaseQuery):
    api_url = "https://api.lsst.fink-portal.org/api/v1"
    id_key = "diaObjectId"
    imtypes = ("Difference", "Template")
