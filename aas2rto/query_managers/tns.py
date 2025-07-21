import io
import json
import requests
import time
import traceback as tr
import warnings
import zipfile
from logging import getLogger
from pathlib import Path
from typing import Dict

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.time import Time, TimeDelta

from aas2rto import utils
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.target import Target
from aas2rto.target_data import TargetData
from aas2rto.utils import calc_file_age

logger = getLogger(__name__.split(".")[-1])


def process_tns_query_results():
    pass


def keep_latest_rows(df: pd.DataFrame):
    df = df.copy()
    df.sort_values(["name", "mjdmod"], inplace=True)
    df.drop_duplicates("name", keep="last", inplace=True)
    return df


class TnsQueryManager(BaseQueryManager):
    name = "tns"

    expected_tns_keys = ("user", "uid", "query_parameters", "tns_parameters")
    default_tns_search_parameters = {
        "num_page": "50",  # 50 is max allowed?
        "format": "csv",  # probably don't change this one...
    }
    default_query_parameters = {
        "query_interval": 0.125,  # days
        "lookback_time": 60.0,  # days
        "query_sleep_time": 3.0,
    }

    def __init__(
        self,
        tns_config: dict,
        target_lookup: Dict[str, Target],
        parent_path=None,
        create_paths=True,
    ):
        self.tns_config = tns_config
        utils.check_unexpected_config_keys(
            self.tns_config, self.expected_tns_keys, name="tns_config"
        )

        self.target_lookup = target_lookup

        # self.recent_coordinate_searches = set()
        # self.query_results = None

        self.tns_results = None

        tns_user = self.tns_config.get("user")
        tns_uid = self.tns_config.get("uid")
        if (tns_user is None) or (tns_uid is None):
            msg = f"user ({tns_user}) or uid ({tns_uid}) is None"
            raise ValueError(msg)

        # self.tns_headers = build_tns_headers(tns_user, tns_uid)
        self.tns_query = TnsQuery(tns_user, tns_uid)

        self.query_parameters = self.default_query_parameters.copy()
        query_parameters = self.tns_config.get("query_parameters", {})
        self.query_parameters.update(query_parameters)
        utils.check_unexpected_config_keys(
            self.query_parameters, self.default_query_parameters
        )

        self.tns_search_parameters = self.default_tns_search_parameters.copy()
        tns_search_parameters = self.tns_config.get("tns_search_parameters", {})
        self.tns_search_parameters.update(tns_search_parameters)
        utils.check_unexpected_config_keys(
            self.tns_search_parameters,
            ALLOWED_TNS_PARAMETERS,
            name="tns.tns_parameters",
        )

        self.process_paths(
            parent_path=parent_path,
            create_paths=create_paths,
            directories=["query_results"],
        )

    def load_existing_tns_results(self):

        existing_results = sorted(list(self.query_results_path.glob("tns_delta*")))

        df_list = []
        for filepath in existing_results:
            try:
                df = pd.read_csv(filepath)
                logger.info(f"load {filepath.stem}")
            except pd.errors.EmptyDataError:
                logger.info(f"cannot read {filepath.stem}")
                continue
            df_list.append(df)

        if len(df_list) > 0:
            tns_results = pd.concat(df_list)
            if not "mjdmod" in tns_results.columns:
                tns_results["mjdmod"] = Time(tns_results["lastmodified"]).mjd
            self.tns_results = tns_results
        else:
            logger.info("no existing TNS information!")

    def collect_missing_daily_deltas(self, t_ref=None):
        t_ref = t_ref or Time.now()

        new_deltas = []
        for ii in range(14):
            t_ref = t_ref - TimeDelta(24 * u.hour)
            datestr = t_ref.strftime("%Y%m%d")
            filepath = self.get_query_results_file(f"tns_delta_{datestr}")
            if filepath.exists():
                continue
            df = self.tns_query.get_tns_daily_delta(t_ref=t_ref)
            if df is not None:
                if df.empty:
                    logger.warning(f"{filepath.name} is empty")
                df.to_csv(filepath, index=False)
                new_deltas.append(df)

        if len(new_deltas) > 0:
            results = pd.concat(new_deltas)
            if "mjdmod" not in results.columns:
                results["mjdmod"] = Time(results["lastmodified"], format="iso").mjd
            return results
        else:
            return pd.DataFrame(columns=["name", "lastmodified", "mjdmod"])

    def collect_missing_hourly_deltas(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        curr_hour = t_ref.ymdhms.hour
        curr_datestr = t_ref.strftime("%Y%m%d")

        yday = t_ref - TimeDelta(24 * u.hour)
        yday_datestr = yday.strftime("%Y%m%d")
        yday_glob_pattern = f"tns_delta_{yday_datestr}_h*"
        yday_hourly = sorted(self.query_results_path.glob(yday_glob_pattern))
        if len(yday_hourly) > 0:
            logger.info("delete yesterday's hourly delta files")
        for filepath in yday_hourly:
            filepath.unlink()  # delete!

        new_deltas = []
        for hour in range(curr_hour):
            fstem = f"tns_delta_{curr_datestr}_h{hour:02d}"
            filepath = self.get_query_results_file(fstem)
            if filepath.exists():
                continue
            df = self.tns_query.get_tns_hourly_delta(hour)
            if df is not None:
                if df.empty:
                    logger.warning(f"{filepath.stem} is empty")
                df.to_csv(filepath, index=False)
                new_deltas.append(df)

        if len(new_deltas) > 0:
            results = pd.concat(new_deltas)
            if "mjdmod" not in results.columns:
                results["mjdmod"] = Time(results["lastmodified"], format="iso").mjd
            return results
        else:
            return pd.DataFrame(columns=["name", "lastmodified", "mjdmod"])

    def match_tns_results_by_coordinates(
        self,
        results=None,
        seplimit=5 * u.arcsec,
        skip_matched=False,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()
        logger.info(f"match TNS data on coords <{seplimit:.1f}")

        if results is None:
            logger.info("re-match ALL tns results")
            results = self.tns_results
            if self.tns_results is None:
                logger.info("no existing TNS results to match!")
                return

        tns_candidate_coords = SkyCoord(
            results["ra"], results["declination"], unit=(u.deg, u.deg)
        )

        target_candidate_coords = []
        target_candidate_target_ids = []
        for target_id, target in self.target_lookup.items():
            tns_data = target.get_target_data("tns")
            if tns_data.parameters is None and skip_matched:
                continue
            target_candidate_coords.append(target.coord)
            target_candidate_target_ids.append(target_id)

        if len(target_candidate_coords) == 0:
            logger.info("no targets left to TNS match")
            return

        target_candidate_coords = SkyCoord(target_candidate_coords)

        target_match_idx, tns_match_idx, skysep, _ = search_around_sky(
            target_candidate_coords, tns_candidate_coords, seplimit
        )
        logger.info(f"coordinate match for {len(target_match_idx)} TNS objects")

        # Who refers to targets by TNS name only?
        tns_alt_keys = ["tns", "yse"]

        # Who has their own internal names for targets?
        alt_id_prefixes = {
            "ZTF": "ztf",
            "ATLAS": "atlas",
            "PS": "panstarrs",
            "LSST": "lsst",
        }

        for ii, (idx1, idx2, skysep) in enumerate(
            zip(target_match_idx, tns_match_idx, skysep)
        ):
            target_id = target_candidate_target_ids[idx1]
            target = self.target_lookup[target_id]
            tns_data = target.get_target_data("tns")
            curr_tns_name = target.alt_ids.get("tns", None)

            tns_row = self.tns_results.iloc[idx2]

            tns_name = tns_row["name"]
            if curr_tns_name is not None and tns_name != curr_tns_name:
                msg = f"    \n{target_id} new tns match {tns_name} does not match old tns_match {curr_tns_name}"
                logger.warning(msg)

            for alt_key in tns_alt_keys:
                target.alt_ids[alt_key] = tns_name
            internal_names = str(tns_row["internal_names"]).replace(" ", "").split(",")
            for name in internal_names:
                for prefix, alt_key in alt_id_prefixes.items():
                    if name.startswith(prefix):
                        curr_alt_id = target.alt_ids.get(alt_key)
                        if curr_alt_id is not None and curr_alt_id != name:
                            msg = f"{target_id}/{tns_name}: new {alt_key}_id {name} does not match existing {curr_alt_id}"
                            logger.warning(msg)
                        # else:
                        target.alt_ids[alt_key] = name

            existing_parameters = tns_data.parameters or {}
            params = "objid redshift type discoverydate lastmodified mjdmod".split()
            updated_parameters = tns_row[params].to_dict()
            tns_data.parameters = updated_parameters

            if updated_parameters["mjdmod"] > existing_parameters.get("mjdmod", 0.0):
                target.updated = True

        if not len(np.unique(target_match_idx)) == len(target_match_idx):
            logger.warning("target matches multiple TNS rows")

    def perform_all_tasks(self, startup=False, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if startup:
            self.load_existing_tns_results()
        else:
            try:
                new_daily_deltas = self.collect_missing_daily_deltas(t_ref=t_ref)
                new_hourly_deltas = self.collect_missing_hourly_deltas(t_ref=t_ref)
            except Exception as e:
                logger.error("exception in tns tasks")
                return e

            if not new_daily_deltas.empty or not new_hourly_deltas.empty:

                self.tns_results = pd.concat(
                    [self.tns_results, new_daily_deltas, new_hourly_deltas]
                )

                self.tns_results.sort_values(["name", "mjdmod"], inplace=True)
                self.tns_results.drop_duplicates(inplace=True, keep="last")

        self.match_tns_results_by_coordinates()


def build_tns_headers(tns_user, tns_uid):
    """Returns a dict with the correctly formatted string."""

    if tns_user is None or tns_uid is None:
        wrn = utils.MissingKeysWarning(f"user={tns_user} or uid={tns_uid} is None!")
        warnings.warn(wrn)
        return None

    marker_dict = dict(tns_id=str(tns_uid), type="user", name=tns_user)
    marker_str = json.dumps(marker_dict)
    # marker_str is a literal string of a dict:
    #       '{"tns_id": "1234", "type": "user", "name": "your_name"}'

    tns_marker = f"tns_marker{marker_str}"
    return {"User-Agent": tns_marker}


class TnsQuery:

    tns_base_url = "https://www.wis-tns.org"
    tns_search_url = f"{tns_base_url}/search"
    tns_public_objects_url = f"{tns_base_url}/system/files/tns_public_objects"

    def __init__(self, tns_user: str, tns_uid: str, query_sleep_time=3.0):
        self.tns_user = tns_user
        self.tns_uid = str(tns_uid)
        self.query_sleep_time = query_sleep_time

        self.tns_headers = build_tns_headers(self.tns_user, self.tns_uid)

    def do_post(self, url):
        return requests.post(url, headers=self.tns_headers)

    def tns_search(self, parameters):
        self.check_search_parameters(parameters)

    @staticmethod
    def check_search_parameters(parameters: dict):
        unknown_parameters = []
        for k, v in parameters.items():
            if k not in ALLOWED_TNS_PARAMETERS:
                unknown_parameters.append(k)

        if unknown_parameters:
            raise ValueError("unknown_parameters")

    def submit_public_object_request(self, url, retries=0, max_retries=3):
        response = self.do_post(url)

        self.wait_after_request(response)
        if response.status_code == 200:
            return self.read_csv_zip_response(response)
        if response.status_code in [404, 429] and retries < max_retries:

            msg = (
                f"status {response.status_code}: resubmit this request!"
                f" try {retries}/{max_retries} {url.split('/')[-1]}"
            )
            logger.error(msg)
            return self.submit_public_object_request(url, retries=retries + 1)
        else:
            logger.error(f"{url} returned response {response.status_code}, ")
            return pd.DataFrame(columns=["name", "lastmodified", "mjdmod"])

    def wait_after_request(self, response: requests.Response):
        req_limit = response.headers.get("x-rate-limit-limit")
        req_remain = response.headers.get("x-rate-limit-remaining")
        req_reset = response.headers.get("x-rate-limit-reset")  # time

        logger.info(f"{req_remain}/{req_limit} requests left (reset {req_reset}s)")

        if int(req_limit) < 2:
            logger.info(f"waiting {req_reset}s for reset...")
            time.sleep(int(req_reset) + 1.0)
        else:
            time.sleep(self.query_sleep_time)

    def get_full_tns_history(self):
        raise NotImplementedError()

    def get_tns_daily_delta(self, t_ref=None, response_only=False):
        if t_ref is None:
            t_ref = Time.now() - TimeDelta(24 * u.hour)

        timestamp = t_ref.strftime("%Y%m%d")
        filename = f"tns_public_objects_{timestamp}.csv.zip"
        url = f"{self.tns_public_objects_url}/{filename}"
        return self.submit_public_object_request(url)

    def get_tns_hourly_delta(self, hour, response_only=False):
        filename = f"tns_public_objects_{hour:02d}.csv.zip"
        url = f"{self.tns_public_objects_url}/{filename}"
        return self.submit_public_object_request(url)

    def read_csv_zip_response(self, response: requests.Response):
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open(z.namelist()[0]) as f:
                try:
                    df = pd.read_csv(f, skiprows=1)  # if actual data, skip header row
                except pd.errors.EmptyDataError as e:
                    df = pd.DataFrame(columns=["name", "lastmodified", "mjdmod"])
                    return df

        lastmod = df["lastmodified"].values.astype(str)
        df["mjdmod"] = Time(lastmod, format="iso").mjd
        print(df)
        return df


ALLOWED_TNS_PARAMETERS = [
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
]
