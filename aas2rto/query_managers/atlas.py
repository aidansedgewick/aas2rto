import copy
import getpass
import io
import requests
import shutil
import time
import traceback
import yaml
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import numpy as np

import pandas as pd

from astropy.time import Time

from aas2rto import Target, TargetData
from aas2rto import utils
from aas2rto.exc import BadKafkaConfigError, MissingTargetIdError
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.utils import calc_file_age

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

    comment_delim = ":"

    default_query_parameters = {
        "lightcurve_query_lookback": 30.0,
        "lightcurve_update_interval": 2.0,
        "max_submitted": 20,
        "max_query_time": 180.0,
        "timeout": 20.0,
        "submit_faint_limit": 20.0,
        "submit_stale_time": 7.0,
        "alt_id_priority": ("tns", "lsst", "ztf", "yse", "ls4"),
    }
    expected_config_params = ("query_parameters", "token", "project_identifier")

    def __init__(
        self,
        atlas_config,
        target_lookup: Dict[str, Target],
        parent_path=None,
        create_paths=True,
    ):
        self.atlas_config = atlas_config
        utils.check_unexpected_config_keys(
            self.atlas_config, self.expected_config_params, name="atlas_config"
        )
        self.target_lookup = target_lookup

        self.project_identifier = self.atlas_config.get("project_identifier", None)
        if self.project_identifier is None:
            self.project_identifier = Path(parent_path).parent.stem
            msg = (
                f"\n    \033[33mATLAS config missing project_identifier set to {self.project_identifier}\033[0m"
                f"\n    atlas_config should contain a unique project_identifier"
                f"\n    so that atlas queries aren't deleted on the server by another project."
            )
            logger.warning(msg)

        token = self.atlas_config.get("token", None)
        if token is None:
            raise ValueError("no token provided in `query_parameters`: `atlas`")

        self.atlas_headers = dict(
            Authorization=f"Token {token}", Accept="application/json"
        )

        self.query_parameters = self.default_query_parameters.copy()
        query_params = self.atlas_config.get("query_parameters", {})
        self.query_parameters.update(query_params)
        utils.check_unexpected_config_keys(
            self.query_parameters,
            self.default_query_parameters,
            name="atlas.query_parameters",
        )

        self.submitted_queries = {}
        self.throttled_queries = []

        # Keep track of who stopped new queries...
        self.local_throttled = False  # ...us?
        self.server_throttled = False  # ...or the server?

        self.process_paths(parent_path=parent_path, create_paths=create_paths)

    def recover_finished_queries(
        self, t_ref: Time = None, delete_finished_queries=True
    ):
        t_ref = t_ref or Time.now()

        timeout = None  # self.query_parameters["timeout"]
        max_query_time = self.query_parameters["max_query_time"]

        finished_queries = []
        ongoing_queries = []

        finished_task_results = {}

        next_url = AtlasQuery.atlas_default_queue_url

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

            ## Loop through "pages" of results, with N results per page.
            try:
                task_response = AtlasQuery.get_existing_queries(
                    headers=self.atlas_headers, url=next_url, timeout=timeout
                )
            except Exception as e:
                logger.warning("\033[33mget_existing_queries failed.\033[0m")
                tr = traceback.format_exc()
                print(tr)
                break
            task_results = task_response["results"]
            for task_result in task_results[::-1]:
                query_time = time.perf_counter() - start_time
                if query_time > max_query_time:
                    msg = (
                        f"Stop requesting finished queries for now. "
                        f"Break as query time {query_time:.1f}s > max {max_query_time:.1f}s"
                    )
                    logger.warning(msg)
                    break

                submit_comment = task_result.get("comment", None)
                if submit_comment is None:
                    logger.warning("existing query has no comment")
                    continue
                kv_split = submit_comment.split(self.comment_delim, 1)
                if len(kv_split) != 2:
                    continue
                target_id, project_str = kv_split
                if project_str != self.project_identifier:
                    # In this case, the identifier means the query is for another project.
                    # Don't retrieve it! When the other project looks for them, it'll crash!
                    continue

                task_url = task_result.get("url", None)
                try:
                    status = self.recover_query_data(target_id, task_url)
                except requests.exceptions.ReadTimeout as e:
                    logger.error("Timeout: break")
                    break

                if status == self.QUERY_SUBMITTED:
                    self.submitted_queries[target_id] = task_url
                    ongoing_queries.append(target_id)
                elif status == self.QUERY_EXISTS:
                    self.submitted_queries.pop(
                        target_id, None
                    )  # remove from submitted queries.
                    finished_task_results[target_id] = task_url
                    finished_queries.append(target_id)
            next_url = task_response["next"]

        if delete_finished_queries:
            for target_id, finished_task_url in finished_task_results.items():
                with requests.Session() as s:
                    s.delete(finished_task_url, headers=self.atlas_headers)

        logger.info(f"{len(finished_queries)} finished, {len(ongoing_queries)} ongoing")
        return finished_queries, ongoing_queries

    def recover_query_data(self, target_id, task_url, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        timeout = self.query_parameters["timeout"]
        max_query_time = self.query_parameters["max_query_time"]

        start_time = time.perf_counter()
        with requests.Session() as s:
            query_response = s.get(
                task_url, headers=self.atlas_headers, timeout=timeout
            )
            query_data = query_response.json()

            # Is the query finished
            finishtimestamp = query_data.get("finishtimestamp", None)
            if finishtimestamp is None:
                logger.debug(f"{target_id} query not finished")
                return self.QUERY_SUBMITTED

            result_url = query_data.get("result_url", None)
            if result_url is None:
                error_msg = query_data.get("error_msg", None)
                if error_msg is not None:
                    if error_msg == "No data returned":
                        lightcurve = get_empty_atlas_lightcurve()
                    else:
                        logger.warning(f"{target_id} unexpected error {error_msg}")
                        return self.QUERY_BAD_REQUEST
            else:
                lightcurve_data = s.get(result_url, headers=self.atlas_headers)
                raw_lightcurve = AtlasQuery.process_response(lightcurve_data)
                lightcurve = process_atlas_lightcurve(raw_lightcurve)

            lightcurve_filepath = self.get_lightcurve_file(target_id)
            if len(lightcurve) == 0:
                if lightcurve_filepath.exists():
                    # If there is existing data, might as well keep it, and update
                    # the file timestamp, so we don't submit again needlessly next loop.
                    lightcurve = pd.read_csv(lightcurve_filepath)
                    if len(lightcurve) > 0:
                        logger.warning(
                            f"\033[33matlas returned no data\033[0m\n"
                            f"existing {target_id} lightcurve has"
                            f"len {len(lightcurve)}, but new query returned zero!"
                        )
            lightcurve.to_csv(lightcurve_filepath, index=False)

            return self.QUERY_EXISTS

    def retry_throttled_queries(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        logger.info(f"retry {len(self.throttled_queries)} throttled queries")

        old_throttled_queries = self.throttled_queries
        self.throttled_queries = []
        submitted_queries, throttled_queries = self.submit_new_queries(
            old_throttled_queries, t_ref=t_ref
        )
        if len(self.throttled_queries) != len(throttled_queries):
            raise ValueError("throttled queries not correctly set")

    def submit_new_queries(self, target_id_list: List[str], t_ref: Time = None):
        t_ref = t_ref or Time.now()

        submitted = []
        throttled = []
        self.local_throttled = False
        self.server_throttled = False

        max_query_time = self.query_parameters["max_query_time"]

        start_time = time.perf_counter()
        for target_id in target_id_list:
            query_time = time.perf_counter() - start_time
            if query_time > max_query_time:
                msg = f"Querying time {query_time:.1f}s > max allowed {max_query_time:.1f}s. Break for now."
                logger.warning(msg)
                break

            if target_id in self.submitted_queries:
                continue  # Already submitted
            if target_id in self.throttled_queries:
                continue  # Already waiting

            if self.server_throttled:
                throttled.append(target_id)
                continue

            target = self.target_lookup.get(target_id, None)
            if target is None:
                logger.warning(f"{target_id} target does not exist!")
                continue

            if len(self.submitted_queries) >= self.query_parameters["max_submitted"]:
                self.local_throttled = True
                throttled.append(target_id)
                continue

            try:
                query_status = self.submit_query(target, t_ref=t_ref)
            except requests.exceptions.ReadTimeout as e:
                logger.info(f"break after {target.target_id} query submit ReadTimeout")
                break
            if query_status == self.QUERY_SUBMITTED:
                submitted.append(target_id)
            elif query_status == self.QUERY_THROTTLED:
                throttled.append(target_id)
                self.server_throttled = True
                msg = "\033[33;1mATLAS THROTTLED\033[0m: no more queries for now..."
                logger.warning(msg)

        self.throttled_queries.extend(throttled)
        logger.info(f"{len(submitted)} new submitted, {len(throttled)} throttled")
        return submitted, throttled

    def submit_query(self, target: Target, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        timeout = None  # self.query_parameters["timeout"]

        query_data = self.prepare_query_data(target, t_ref=t_ref)
        if query_data.get("ra", None) is None or query_data.get("dec", None) is None:
            logger.warning(
                f"\033[33m{target.target_id} ra/dec is None!\033[0m skip submit."
            )
            return self.QUERY_BAD_REQUEST

        res = AtlasQuery.atlas_query(
            query_data, headers=self.atlas_headers, timeout=timeout
        )
        if res.status_code == self.QUERY_SUBMITTED:
            task_url = res.json()["url"]
            self.submitted_queries[target.target_id] = task_url
            return res.status_code
        elif res.status_code == self.QUERY_THROTTLED:
            self.throttled_queries.append(target.target_id)
            return res.status_code
        else:
            msg = f"{target.target_id} query status \033[33;1m{res.status_code}\033[0m: {res.reason}"
            logger.error(msg)
            return res.status_code

    def get_atlas_query_comment(self, target_id):
        return f"{target_id}{self.comment_delim}{self.project_identifier}"

    def prepare_query_data(self, target: Target, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        atlas_data = target.get_target_data("atlas")
        if atlas_data.lightcurve is not None:
            mjd_min = atlas_data.lightcurve["mjd"].min() - 1e-3
        else:
            mjd_min = t_ref.mjd - self.query_parameters["lightcurve_query_lookback"]

        comment = self.get_atlas_query_comment(target.target_id)

        return dict(
            ra=target.ra,
            dec=target.dec,
            mjd_min=mjd_min,
            mjd_max=t_ref.mjd - 1e-3,
            send_email=False,
            comment=comment,
        )

    def select_query_candidates(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        too_faint = 0
        too_stale = 0
        no_score = 0
        recent_updates = 0

        faint_limit = self.query_parameters["submit_faint_limit"]
        stale_time = self.query_parameters["submit_stale_time"]
        update_interval = self.query_parameters["lightcurve_update_interval"]

        score_lookup = {}
        for target_id, target in self.target_lookup.items():
            last_score = (
                target.get_last_score()
            )  # no observatory means None -> "no_observatory"
            if last_score is None:
                no_score = no_score + 1
                continue
            lightcurve_filepath = self.get_lightcurve_file(target_id)
            lightcurve_file_age = calc_file_age(lightcurve_filepath, t_ref)
            if lightcurve_file_age < update_interval:
                recent_updates = recent_updates + 1
                continue

            lc = target.compiled_lightcurve
            if lc is None:
                continue
            valid_lc = lc[lc["tag"] == "valid"]
            if valid_lc["mag"].min() > faint_limit:
                too_faint = too_faint + 1
                continue
            delta_t = t_ref.mjd - valid_lc.iloc[-1]["mjd"]
            if delta_t > stale_time:
                too_stale = too_stale + 1
                continue

            score_lookup[target_id] = last_score

        msg = (
            f"Targets not considered:\n"
            f"    {no_score} with no valid score\n"
            f"    {recent_updates} updated <{update_interval:.1f}d ago\n"
            f"    {too_faint} with no detections <{faint_limit:.1f}mag\n"
            f"    {too_stale} with no detections <{stale_time:.1f}d ago"
        )
        logger.info(msg)

        object_series = pd.Series(score_lookup)
        object_series.sort_values(inplace=True, ascending=False)
        logger.info(f"{len(object_series)} suitable targets")
        return object_series.index

    def load_single_lightcurve(self, target_id: str, t_ref=None):
        # logger.debug(f"loading {target_id}")

        target = self.target_lookup[target_id]

        lightcurve_filepath = None
        for alt_key in self.query_parameters["alt_id_priority"]:
            alt_id = target.alt_ids.get(alt_key, None)
            if alt_id is not None:
                test_filepath = self.get_lightcurve_file(alt_id)
                if test_filepath.exists():
                    lightcurve_filepath = test_filepath
                    break

        if lightcurve_filepath is None:
            logger.debug(f"{target_id} is missing lightcurve ")
            return None
        lightcurve = pd.read_csv(lightcurve_filepath)
        if lightcurve.empty:
            logger.warning(f"{target_id} empty!")
            return None
        return lightcurve

    def update_lightcurve_filenames(self):
        alt_id_priority = self.default_query_parameters["alt_id_priority"]

        t_start = time.perf_counter()
        renamed = 0
        skipped = 0
        missing = 0
        for target_id, target in self.target_lookup.items():
            best_id = None
            for alt_key in alt_id_priority:
                alt_id = target.alt_ids.get(alt_key, None)
                if alt_id is None:
                    continue
                if best_id is None:
                    best_id = alt_id
                lc_filepath = self.get_lightcurve_file(alt_id)
                if lc_filepath.exists():
                    best_filepath = self.get_lightcurve_file(best_id)
                    if alt_id == best_id:
                        skipped = skipped + 1
                    else:
                        shutil.move(lc_filepath, best_filepath)
                        renamed = renamed + 1
                    break
        t_end = time.perf_counter()

        logger.info(f"rename {renamed} LCs in {t_end-t_start:.1f}s (skipped {skipped})")

    def perform_all_tasks(self, startup=False, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        self.update_lightcurve_filenames()

        if startup:
            logger.info("perform no ATLAS queries on iter 0: only load existing")
            self.load_target_lightcurves(t_ref=t_ref, flag_only_existing=False)
            return

        try:
            self.recover_finished_queries(t_ref=t_ref)  # and popul. submitted_queries
        except requests.exceptions.JSONDecodeError as e:
            return e
        except requests.exceptions.Timeout as e:
            return e

        try:
            self.retry_throttled_queries(t_ref=t_ref)
        except ConnectionError as e:
            return e
        except requests.exceptions.Timeout as e:
            return e

        query_candidates = self.select_query_candidates(t_ref=t_ref)
        try:
            self.submit_new_queries(query_candidates, t_ref=t_ref)
        except requests.exceptions.Timeout as e:
            return e
        # self.recover_finished_queries(t_ref=t_ref)

        self.load_target_lightcurves(t_ref=t_ref, flag_only_existing=False)
        # flag_only_exisitng=False means new LCs make the target updated.


class AtlasQuery:
    atlas_base_url = "https://fallingstar-data.com/forcedphot"
    atlas_default_queue_url = f"{atlas_base_url}/queue/"
    atlas_default_timeout = 90.0

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
    def atlas_query(cls, data, headers, timeout=atlas_default_timeout):
        res = requests.post(
            url=cls.atlas_default_queue_url, headers=headers, data=data, timeout=timeout
        )
        return res

    @classmethod
    def get_existing_queries(cls, headers, url=None, timeout=atlas_default_timeout):
        if url is None:
            url = cls.atlas_default_queue_url
        res = requests.get(url=url, headers=headers, timeout=timeout)
        return res.json()

    @staticmethod
    def process_response(photom_data, text_processed=False):
        if not text_processed:
            textdata = photom_data.text
        else:
            textdata = photom_data
        lightcurve = pd.read_csv(io.StringIO(textdata.replace("###", "")), sep="\s+")
        return lightcurve
