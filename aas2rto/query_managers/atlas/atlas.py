import requests
import shutil
import time
import warnings
from logging import getLogger
from pathlib import Path

import pandas as pd

from astropy.time import Time

from aas2rto import utils
from aas2rto.exc import UnknownTargetWarning
from aas2rto.query_managers.atlas.atlas_query import AtlasQuery
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup

logger = getLogger(__name__.split(".")[-1])


def process_atlas_lightcurve(lc: pd.DataFrame):
    processed_lc = lc.copy()  # Avoid SettingWithCopyWarning?

    processed_lc.rename({"MJD": "mjd"}, inplace=True, axis=1)
    if processed_lc.empty:
        jd_dat = []
    else:
        jd_dat = Time(processed_lc["mjd"], format="mjd").jd
    processed_lc.insert(1, "jd", jd_dat)
    return processed_lc


class AtlasQueryManager(BaseQueryManager):
    name = "atlas"

    comment_delim = ":"
    DEFAULT_ID_PRIORITY = ("tns", "atlas", "yse", "ztf", "ls4", "lsst")
    default_config = {
        "credentials": None,
        "alt_id_priority": DEFAULT_ID_PRIORITY,
        "project_identifier": None,
        "lightcurve_query_lookback": 30.0,
        "max_submitted": 20,
        "max_query_time": 180.0,
        "timeout": 20.0,
        "query_faint_limit": 22.0,
        "query_stale_limit": 7.0,
        "query_lightcurve_age": 2.0,
        "query_minimum_score": 0.0,
    }
    config_comments = {
        "credentials": "**REQUIRED**: {'token': <>} OR {'username': <>, 'password': <>}",
        "alt_id_priority": f"Naming preference for LCs, default={DEFAULT_ID_PRIORITY}",
        "project_identifier": f"A unique string for this project eg. 'bright_sne_ia'",
        "lightcurve_query_lookback": "How long [DAYS] should LCs query in the past?",
        "max_submitted": "How many queries (per project!!) in queue at once? MAX~25",
        "max_query_time": "Quit after queries that take longer than x [SEC]",
        "timeout": "<unused> - for now",
        "query_faint_limit": "If last data fainter than x [MAG], don't submit query",
        "query_stale_limit": "If no data more recent than x [DAYS], don't submit query",
        "query_lightcurve_age": "If last query less x [DAYS] ago, don't requery",
        "query_minimum_score": "Skip targets if last science_score less than x",
    }
    required_directories = ("lightcurves",)

    def __init__(self, config: dict, target_lookup: TargetLookup, parent_path: Path):

        self.config = self.default_config.copy()
        self.config.update(config)

        self.process_credentials()  # BEFORE cfg_chk: credentials error raised first.
        utils.check_unexpected_config_keys(
            self.config, self.default_config, name="atlas", raise_exc=True
        )

        self.target_lookup = target_lookup

        self.project_identifier = self.config.get("project_identifier", None)
        if self.project_identifier is None:
            self.project_identifier = Path(parent_path).parent.stem
            msg = (
                f"\n    \033[33mATLAS config missing project_identifier set to {self.project_identifier}\033[0m"
                f"\n    atlas_config should contain a unique project_identifier"
                f"\n    so that atlas queries aren't deleted on the server by another project."
            )
            logger.warning(msg)

        self.submitted_queries = {}
        self.throttled_queries = []

        # Keep track of who stopped new queries...
        self.local_throttled = False  # ...us?
        self.server_throttled = False  # ...or the server?

        self.process_paths(
            parent_path=parent_path, directories=self.required_directories
        )

    def get_lightcurve_filepath(self, target_id: str, fmt="csv"):
        return self.paths_lookup["lightcurves"] / f"{target_id}.{fmt}"

    def process_credentials(self):
        credentials: dict = self.config.get("credentials", None)
        bad_credential_msg = (
            "ATLAS config must contain dict 'credentials' with "
            "key 'token' OR 'username' and 'password'. eg.\n    "
            "credentials: {token: 292834932}\n        ----OR----\n    "  # NO f-str
            "credentials: {username: myuser, password: mypass}"  # NO f-str
        )
        if credentials is None:
            raise ValueError(bad_credential_msg)

        self.token = credentials.get("token", None)  # attr mainly for testing.
        username = credentials.get("username", None)  # Not an attr
        password = credentials.get("password", None)  # DEFINIETLY not an attr.

        if self.token is not None:
            if username is not None or password is not None:
                msg = f"IGNORING usr '{username}' pwd '***': using provided token"
                logger.warning(msg)
            self.atlas_headers = AtlasQuery.build_headers(self.token)
            return

        # Have to try to retrieve using the usr/pwd combo provided.
        if username is None or password is None:
            raise ValueError(bad_credential_msg)
        self.token = AtlasQuery.get_atlas_token(
            username=username, password=password, display=False
        )
        self.atlas_headers = AtlasQuery.build_headers(self.token)

    def recover_query_data(
        self, target_id: str, task_url: str, timeout: float = None, t_ref: Time = None
    ):
        """
        Does NOT add lightcurves to target.
        """
        t_ref = t_ref or Time.now()

        status, unprocessed_lightcurve = AtlasQuery.recover_task_data(
            task_url,
            headers=self.atlas_headers,
            return_type="pandas",
            timeout=timeout,
            delete_finished=False,  # NO - delete them all in one go.
        )
        if unprocessed_lightcurve is None:
            return status, unprocessed_lightcurve

        lightcurve = process_atlas_lightcurve(unprocessed_lightcurve)

        lightcurve_filepath = self.get_lightcurve_filepath(target_id)
        if len(lightcurve) == 0:
            if lightcurve_filepath.exists():
                lightcurve = pd.read_csv(lightcurve_filepath)
                # We may as well just re-write the existing lightcurve.
        lightcurve.to_csv(lightcurve_filepath, index=False)
        return status, lightcurve

    def recover_existing_queries(
        self, t_ref: Time = None, delete_finished_queries: bool = True
    ):
        t_ref = t_ref or Time.now()

        timeout = None  # self.query_parameters["timeout"]
        max_query_time = self.config["max_query_time"]

        finished_queries = []
        ongoing_queries = []
        error_queries = []

        finished_task_urls = []

        start_time = time.perf_counter()
        for task_result in AtlasQuery.iterate_existing_queries(
            headers=self.atlas_headers, max_query_time=max_query_time
        ):
            # Is this task relevant to us?
            query_comment: str = task_result.get("comment", None)
            if query_comment is None:
                # This must be a query put by hand in the webform... Don't delete it!
                continue

            target_id, task_project = query_comment.split(self.comment_delim, 1)
            if task_project != self.project_identifier:
                # This task doesn't belong to our project. Don't retrieve it!
                # Otherwise, if the other project looks for this task, it might crash!
                continue

            # Is the query in progress, or finished?
            task_url = task_result.get("url", None)
            if task_url is not None:
                try:
                    status, lc = self.recover_query_data(target_id, task_url)
                except requests.exceptions.ReadTimeout as e:
                    logger.error("Timeout: break")
                    break
            else:
                logger.info(f"no task_url (key 'url') for {target_id}")

            if status == AtlasQuery.QUERY_SUBMITTED:
                self.submitted_queries[target_id] = task_url
                ongoing_queries.append(target_id)
            elif status == AtlasQuery.QUERY_EXISTS:
                self.submitted_queries.pop(target_id, None)
                finished_task_urls.append(task_url)
                finished_queries.append(target_id)
            elif status == AtlasQuery.QUERY_BAD_REQUEST:
                self.submitted_queries.pop(target_id, None)
                error_queries.append(target_id)

        if delete_finished_queries:
            AtlasQuery.delete_tasks(finished_task_urls, headers=self.atlas_headers)

        logger.info(f"{len(finished_queries)} finished, {len(ongoing_queries)} ongoing")
        return finished_queries, ongoing_queries, error_queries

    def select_query_candidates(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        too_faint = 0
        too_stale = 0
        no_score = 0
        low_score = 0
        recently_updated = 0

        faint_limit = self.config["query_faint_limit"]
        stale_time = self.config["query_stale_limit"]
        update_interval = self.config["query_lightcurve_age"]
        minimum_score = self.config["query_minimum_score"]

        score_lookup = {}
        for target_id, target in self.target_lookup.items():
            last_score = target.get_latest_science_score()
            if last_score is None:
                no_score = no_score + 1
                continue
            if last_score < minimum_score:
                low_score = low_score + 1
                continue
            lightcurve_filepath = self.get_lightcurve_filepath(target_id)
            lightcurve_file_age = utils.calc_file_age(lightcurve_filepath, t_ref)
            if lightcurve_file_age < update_interval:
                recently_updated = recently_updated + 1
                continue

            lc = target.compiled_lightcurve
            if lc is None:
                continue
            valid_lc = lc[lc["tag"] == "valid"]
            if len(valid_lc) == 0:
                continue

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
            f"    {recently_updated} updated <{update_interval:.1f}d ago\n"
            f"    {too_faint} with no detections <{faint_limit:.1f}mag\n"
            f"    {too_stale} with no detections <{stale_time:.1f}d ago"
        )
        logger.info(msg)

        object_series = pd.Series(score_lookup)
        object_series.sort_values(inplace=True, ascending=False)
        logger.info(f"{len(object_series)} suitable targets")
        return object_series.index

    def submit_query(self, target: Target, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        timeout = None  # self.query_parameters["timeout"]

        query_data = self.prepare_query_payload(target, t_ref=t_ref)
        if query_data.get("ra", None) is None or query_data.get("dec", None) is None:
            logger.warning(
                f"\033[33m{target.target_id} ra/dec is None!\033[0m skip submit."
            )
            return None

        res = AtlasQuery.submit_forced_photom_query(
            query_data, headers=self.atlas_headers, timeout=timeout
        )
        if res.status_code == AtlasQuery.QUERY_SUBMITTED:
            self.submitted_queries[target.target_id] = res.json()["url"]
        elif res.status_code == AtlasQuery.QUERY_THROTTLED:
            self.throttled_queries.append(target.target_id)
        else:
            msg = f"{target.target_id} query status \033[33;1m{res.status_code}\033[0m: {res.reason}"
            logger.error(msg)
        return res

    def get_atlas_query_comment(self, target_id: str) -> str:
        return f"{target_id}{self.comment_delim}{self.project_identifier}"

    def prepare_query_payload(self, target: Target, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        atlas_data = target.get_target_data("atlas")
        if atlas_data.lightcurve is not None:
            mjd_min = atlas_data.lightcurve["mjd"].min() - 1e-3
        else:
            mjd_min = t_ref.mjd - self.config["lightcurve_query_lookback"]
        return dict(
            ra=target.coord.ra.deg,
            dec=target.coord.dec.deg,
            mjd_min=mjd_min,
            mjd_max=t_ref.mjd - 1e-3,
            send_email=False,
            comment=self.get_atlas_query_comment(target.target_id),
        )

    def submit_new_queries(self, target_id_list: list[str], t_ref: Time = None):
        t_ref = t_ref or Time.now()

        submitted = []
        throttled = []
        self.local_throttled = False
        self.server_throttled = False

        max_query_time = self.config["max_query_time"]

        start_time = time.perf_counter()
        for target_id in target_id_list:
            query_time = time.perf_counter() - start_time
            if query_time > max_query_time:
                msg = (
                    f"Query time {query_time:.1f}s > max {max_query_time:.1f}s. "
                    "Break for now."
                )
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
                msg = f"{target_id} target does not exist!"
                logger.warning(msg)
                warnings.warn(UnknownTargetWarning(msg))
                continue

            if len(self.submitted_queries) >= self.config["max_submitted"]:
                self.local_throttled = True
                throttled.append(target_id)
                continue

            try:
                query_response = self.submit_query(target, t_ref=t_ref)
                if query_response is None:
                    continue
                query_status = query_response.status_code
            except requests.exceptions.ReadTimeout as e:
                logger.info(f"break after {target.target_id} query submit ReadTimeout")
                break
            if query_status == AtlasQuery.QUERY_SUBMITTED:
                submitted.append(target_id)
            elif query_status == AtlasQuery.QUERY_THROTTLED:
                throttled.append(target_id)
                self.server_throttled = True
                msg = "\033[33;1mATLAS THROTTLED\033[0m: no more queries for now..."
                logger.warning(msg)

        self.throttled_queries.extend(throttled)
        logger.info(f"{len(submitted)} new submitted, {len(throttled)} throttled")
        return submitted, throttled

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
        return submitted_queries, throttled_queries

    def load_single_lightcurve(self, target_id: str, t_ref=None):
        # logger.debug(f"loading {target_id}")

        target = self.target_lookup[target_id]

        lightcurve_filepath = None

        for alt_key in self.config["alt_id_priority"]:
            alt_id = target.alt_ids.get(alt_key, None)
            if alt_id is not None:
                candidate_filepath = self.get_lightcurve_filepath(alt_id)
                if candidate_filepath.exists():
                    lightcurve_filepath = candidate_filepath
                    break

        if lightcurve_filepath is None:
            logger.debug(f"{target_id} is missing lightcurve")
            return None
        lightcurve = pd.read_csv(lightcurve_filepath)
        if lightcurve.empty:
            return None
        return lightcurve

    def update_lightcurve_filenames(self):
        alt_id_priority = self.config["alt_id_priority"]

        t_start = time.perf_counter()
        renamed = []
        skipped = []
        for target_id, target in self.target_lookup.items():
            best_id = None
            for alt_key in alt_id_priority:
                alt_id = target.alt_ids.get(alt_key, None)
                if alt_id is None:
                    continue
                if best_id is None:
                    best_id = alt_id
                lc_filepath = self.get_lightcurve_filepath(alt_id)
                if lc_filepath.exists():
                    best_filepath = self.get_lightcurve_filepath(best_id)
                    if alt_id == best_id:
                        skipped.append(target_id)
                    else:
                        shutil.move(lc_filepath, best_filepath)
                        renamed.append(target_id)
                    break
        t_end = time.perf_counter()
        if len(renamed) > 0:
            dt = t_end - t_start
            msg = f"rename {len(renamed)} LCs in {dt:.1f}s (skipped {len(skipped)})"
            logger.info(msg)
        return renamed, skipped

    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        # Are we in startup?
        if iteration == 0:
            logger.info("Skip atlas queries on iter0: only load existing")
            self.load_target_lightcurves(t_ref=t_ref, only_flag_updated=False)
            return

        # Check queries that already exist on server
        self.recover_existing_queries()

        self.retry_throttled_queries()

        query_candidates = self.select_query_candidates(t_ref=t_ref)
        self.submit_new_queries(query_candidates)

        self.load_target_lightcurves(t_ref=t_ref, only_flag_updated=False)
        # flag_only_exisitng=False means new LCs make the target updated.

        self.clear_stale_files()
