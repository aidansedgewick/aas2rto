import copy
import getpass
import io
import requests
import time
import yaml
from logging import getLogger
from typing import Dict, List

import numpy as np

import pandas as pd

from astropy.time import Time

from dk154_targets import Target

from dk154_targets.query_managers.base import BaseQueryManager
from dk154_targets.query_managers.exc import BadKafkaConfigError, MissingObjectIdError
from dk154_targets.utils import calc_file_age

logger = getLogger(__name__.split(".")[-1])


def process_atlas_lightcurve(detections: pd.DataFrame):
    detections.rename({"MJD": "mjd"}, inplace=True, axis=1)

    jd_dat = Time(detections["mjd"], format="mjd").jd
    detections.insert(1, "jd", jd_dat)
    return detections


def target_from_atlas_lightcurve(lightcurve: pd.DataFrame):
    pass


def get_empty_atlas_lightcurve():
    cols = "mjd,jd,m,dm,uJy,duJy,F,err,chi/N,RA,Dec,x,y,maj,min,phi,apfit,mag5sig,Sky,Obs".split(
        ","
    )
    return pd.DataFrame([], columns=cols)


class AtlasQueryManager(BaseQueryManager):
    name = "atlas"

    # these are the normal http response codes...
    QUERY_EXISTS = 200
    QUERY_SUBMITTED = 201
    QUERY_BAD_REQUEST = 400
    QUERY_THROTTLED = 429

    default_query_parameters = {
        "lookback_time": 30.0,
        "interval": 2.0,
        "max_submitted": 10,
        "requests_timeout": 20.0,
    }

    def __init__(
        self,
        atlas_config,
        target_lookup: Dict[str, Target],
        data_path=None,
        create_paths=True,
    ):
        self.atlas_config = atlas_config
        self.target_lookup = target_lookup

        token = self.atlas_config.get("token", None)
        if token is None:
            raise ValueError("no token provided in `query_parameters`: `atlas`")

        self.atlas_headers = dict(
            Authorization=f"Token {token}", Accept="application/json"
        )

        self.query_parameters = self.default_query_parameters.copy()
        query_params = self.atlas_config.get("query_parameters", {})
        self.query_parameters.update(query_params)

        self.submitted_queries = {}
        self.throttled_queries = []

        self.process_paths(data_path=data_path, create_paths=create_paths)

    def recover_finished_queries(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        finished_queries = []
        ongoing_queries = []

        finished_task_results = {}

        next_url = AtlasQuery.atlas_default_queue_url

        while next_url is not None:
            task_response = AtlasQuery.get_existing_queries(
                headers=self.atlas_headers, url=next_url
            )
            task_results = task_response["results"]
            for task_result in task_results[::-1]:
                objectId = task_result.get("comment", None)
                if objectId is None:
                    logger.warning("existing query has no objectId")
                    continue
                task_url = task_result.get("url", None)
                status = self.recover_query_data(objectId, task_url)
                if status == self.QUERY_SUBMITTED:
                    self.submitted_queries[objectId] = task_url
                    ongoing_queries.append(objectId)
                elif status == self.QUERY_EXISTS:
                    self.submitted_queries.pop(
                        objectId, None
                    )  # remove from submitted queries.
                    finished_task_results[objectId] = task_url
                    finished_queries.append(objectId)
            next_url = task_response["next"]

        for objectId, finished_task_url in finished_task_results.items():
            with requests.Session() as s:
                s.delete(finished_task_url, headers=self.atlas_headers)

        logger.info(f"{len(finished_queries)} finished, {len(ongoing_queries)} ongoing")
        return finished_queries, ongoing_queries

    def recover_query_data(self, objectId, task_url, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        with requests.Session() as s:
            query_response = s.get(task_url, headers=self.atlas_headers)
            query_data = query_response.json()

            # Is the query finished
            finishtimestamp = query_data.get("finishtimestamp", None)
            if finishtimestamp is None:
                logger.debug(f"{objectId} query not finished")
                return self.QUERY_SUBMITTED

            result_url = query_data.get("result_url", None)
            if result_url is None:
                error_msg = query_data.get("error_msg", None)
                if error_msg is not None:
                    if error_msg == "No data returned":
                        lightcurve = get_empty_atlas_lightcurve()
                    else:
                        logger.warning(f"{objectId} unexpected error {error_msg}")
            else:
                lightcurve_data = s.get(result_url, headers=self.atlas_headers)
                raw_lightcurve = AtlasQuery.process_response(lightcurve_data)
                lightcurve = process_atlas_lightcurve(raw_lightcurve)

            lightcurve_file = self.get_lightcurve_file(objectId)
            if len(lightcurve) == 0:
                if lightcurve_file.exists():
                    # If there is existing data, might as well keep it, and update
                    # the file timestamp, so we don't just try again needlessly.
                    lightcurve = pd.read_csv()
            # else:
            #    if lightcurve_file.exists():
            lightcurve.to_csv(lightcurve_file, index=False)

            return self.QUERY_EXISTS

    def retry_throttled_queries(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        logger.info(f"retry {len(self.throttled_queries)} throttled queries")

        old_throttled_list = self.throttled_queries
        self.throttled_queries = []
        submitted_queries, throttled_queries = self.submit_new_queries(
            self.throttled_queries, t_ref=t_ref
        )
        if len(self.throttled_queries) != len(throttled_queries):
            raise ValueError("throttled queries not correctly set")

    def submit_new_queries(self, objectId_list: List[str], t_ref: Time = None):
        t_ref = t_ref or Time.now()

        submitted = []
        throttled = []
        self_throttled = False
        atlas_throttled = False

        for objectId in objectId_list:
            if objectId in self.submitted_queries:
                continue  # Already submitted
            if objectId in self.throttled_queries:
                continue  # Already waiting

            if atlas_throttled:
                throttled.append(objectId)
                continue

            target = self.target_lookup.get(objectId, None)
            if target is None:
                logger.warning(f"{objectId} target does not exist!")
                continue

            if len(self.submitted_queries) >= self.query_parameters["max_submitted"]:
                self_throttled = True
                throttled.append(objectId)
                continue

            query_status = self.submit_query(target, t_ref=t_ref)
            if query_status == self.QUERY_SUBMITTED:
                submitted.append(objectId)
            elif query_status == self.QUERY_THROTTLED:
                throttled.append(objectId)
                atlas_throttled = True
                msg = "\033[33;1mATLAS THROTTLED\033[0m: no more queries for now..."
                logger.warning(msg)

        self.throttled_queries.extend(throttled)
        logger.info(f"{len(submitted)} new submitted, {len(throttled)} throttled")
        return submitted, throttled

    def submit_query(self, target: Target, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        query_data = self.prepare_query_data(target, t_ref=t_ref)
        if query_data.get("ra", None) is None or query_data.get("dec", None) is None:
            logger.warning(
                f"\033[33m{target.objectId} ra/dec is None!\033[0m skip submit."
            )
            return self.QUERY_BAD_REQUEST

        res = AtlasQuery.atlas_query(query_data, headers=self.atlas_headers)
        if res.status_code == self.QUERY_SUBMITTED:
            task_url = res.json()["url"]
            self.submitted_queries[target.objectId] = task_url
            return res.status_code
        elif res.status_code == self.QUERY_THROTTLED:
            self.throttled_queries.append(target.objectId)
            return res.status_code
        else:
            msg = f"{target.objectId} query status \033[33;1m{res.status_code}\033[0m: {res.reason}"
            logger.error(msg)
            return res.status_code

    def prepare_query_data(self, target: Target, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if target.atlas_data.lightcurve is None:
            mjd_min = t_ref.mjd - self.query_parameters["lookback_time"]
        else:
            mjd_min = target.atlas_data.lightcurve["MJD"].min() - 1e-3

        return dict(
            ra=target.ra,
            dec=target.dec,
            mjd_min=mjd_min,
            mjd_max=t_ref.mjd - 1e-3,
            send_email=False,
            comment=target.objectId,
        )

    def select_query_candidates(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        score_lookup = {}
        for objectId, target in self.target_lookup.items():
            last_score = target.get_last_score(obs_name="no_observatory")
            if last_score is None:
                continue
            lightcurve_file = self.get_lightcurve_file(objectId)
            lightcurve_file_age = calc_file_age(lightcurve_file, t_ref)
            if lightcurve_file_age < self.query_parameters["interval"]:
                continue
            score_lookup[objectId] = last_score
        object_series = pd.Series(score_lookup)
        object_series.sort_values(inplace=True, ascending=False)
        return object_series.index

    def load_target_lightcurves(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        loaded = []
        t_start = time.perf_counter()
        for objectId, target in self.target_lookup.items():
            lightcurve_file = self.get_lightcurve_file(objectId)
            if not lightcurve_file.exists():
                continue
            lightcurve = pd.read_csv(lightcurve_file)
            if lightcurve.empty:
                continue
            existing_lightcurve = target.atlas_data.lightcurve
            if existing_lightcurve is None:
                target.atlas_data.add_lightcurve(lightcurve)
            else:
                if len(lightcurve) > len(existing_lightcurve):
                    target.atlas_data.add_lightcurve(lightcurve)
                else:
                    continue
            loaded.append(objectId)
            target.updated = True
            target.update_messages.append("New atlas data!")
        t_end = time.perf_counter()
        if len(loaded) > 0:
            logger.info(f"{len(loaded)} lightcurves loaded in {(t_end-t_start):.1f}s")

    def perform_all_tasks(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        self.recover_finished_queries(t_ref=t_ref)  # also populates "submitted queries"
        self.retry_throttled_queries(t_ref=t_ref)
        query_candidates = self.select_query_candidates(t_ref=t_ref)
        self.submit_new_queries(query_candidates, t_ref=t_ref)
        # self.recover_finished_queries(t_ref=t_ref)
        self.load_target_lightcurves(t_ref=t_ref)


class AtlasQuery:
    atlas_base_url = "https://fallingstar-data.com/forcedphot"
    atlas_default_queue_url = f"{atlas_base_url}/queue/"

    def __init__(self):
        pass

    @staticmethod
    def get_atlas_token():
        username = input("Username: ")
        password = getpass.getpass("Password [input hidden]: ")
        url = f"{AtlasQuery.atlas_base_url}/api-token-auth/"  # NOTE TRAILING SLASH!
        response = requests.post(url, data=dict(username=username, password=password))
        try:
            token = response.json()["token"]
            print(f"Your ATLAS token is: {token}")
        except KeyError:
            print(response.json())

    @classmethod
    def atlas_query(cls, data, headers):
        res = requests.post(url=cls.atlas_default_queue_url, headers=headers, data=data)
        return res

    @classmethod
    def get_existing_queries(cls, headers, url=None):
        if url is None:
            url = cls.atlas_default_queue_url
        res = requests.get(url=url, headers=headers)
        return res.json()

    @staticmethod
    def process_response(photom_data, text_processed=False):
        if not text_processed:
            textdata = photom_data.text
        else:
            textdata = photom_data
        lightcurve = pd.read_csv(
            io.StringIO(textdata.replace("###", "")), delim_whitespace=True
        )
        return lightcurve
