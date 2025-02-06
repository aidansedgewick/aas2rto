import copy
import json
import re
import requests
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.time import Time, TimeDelta

from aas2rto import Target, TargetData
from aas2rto import utils
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.utils import calc_file_age

logger = getLogger(__name__.split(".")[-1])

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


def build_tns_headers(tns_user: str, tns_uid: int):
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


def process_tns_query_results(raw_query_results: pd.DataFrame):
    query_results = raw_query_results.copy()
    query_results.drop_duplicates("Name", keep="last", inplace=True)
    return query_results


def empty_data_frame():
    return pd.DataFrame(columns=["Name"])


class TnsQueryManager(BaseQueryManager):
    name = "tns"

    expected_tns_keys = ("user", "uid", "query_parameters", "tns_parameters")
    default_tns_parameters = {
        "num_page": "50",  # 50 is max allowed?
        "format": "csv",  # probably don't change this one...
    }
    default_query_parameters = {
        "query_interval": 0.125,  # days
        "lookback_time": 60.0,  # days
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

        self.recent_coordinate_searches = set()
        self.query_results = None

        tns_user = self.tns_config.get("user")
        tns_uid = self.tns_config.get("uid")
        if (tns_user is None) or (tns_uid is None):
            msg = f"user ({tns_user}) or uid ({tns_uid}) is None"
            raise ValueError(msg)

        self.tns_headers = build_tns_headers(tns_user, tns_uid)

        self.query_parameters = self.default_query_parameters.copy()
        query_parameters = self.tns_config.get("query_parameters", {})
        self.query_parameters.update(query_parameters)
        utils.check_unexpected_config_keys(
            self.query_parameters, self.default_query_parameters
        )

        self.tns_parameters = self.default_tns_parameters.copy()
        tns_parameters = self.tns_config.get("tns_parameters", {})
        self.tns_parameters.update(tns_parameters)
        utils.check_unexpected_config_keys(
            self.tns_parameters, ALLOWED_TNS_PARAMETERS, name="tns.tns_parameters"
        )

        self.process_paths(parent_path=parent_path, create_paths=create_paths)

    def query_for_updates(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        query_results_file = self.get_query_results_file("tns_results")
        query_results_file_age = calc_file_age(query_results_file, t_ref)

        if self.query_results is None and query_results_file.exists():
            query_results = pd.read_csv(query_results_file)
            query_results = process_tns_query_results(query_results)
            self.query_results = query_results
        if query_results_file_age < self.query_parameters["query_interval"]:
            return None

        query_parameters = self.get_query_data()
        query_updates = TnsQuery.query(query_parameters, self.tns_headers)

        existing_results = self.query_results
        if existing_results is None:
            existing_results = empty_data_frame()
        query_results = pd.concat([existing_results, query_updates], ignore_index=True)
        query_results.to_csv(query_results_file, index=False)
        self.query_results = query_results

        # Make sure that we crossmatch everything again.
        self.recent_coordinate_searches = set()

    def get_query_data(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        query_parameters = {**self.tns_parameters}

        lookback_time = self.query_parameters.get("lookback_time", 60.0)

        t_start = t_ref - TimeDelta(lookback_time, format="jd")
        query_parameters["date_start[date]"] = t_start.iso.split()[0]

        return query_parameters

    # def update_matched_tns_names(self, t_ref: Time = None):
    #     t_ref = t_ref or Time.now()

    #     matched_tns_names = []
    #     for target_id, target in self.target_lookup.items():
    #         tns_name = target.tns_data.parameters.get("Name", None)
    #         if tns_name is not None:
    #             matched_tns_names[tns_name] = target_id

    def match_tns_on_names(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        logger.info("match TNS data by target_id")

        unmatched_rows = []
        matched = 0
        for idx, tns_row in self.query_results.iterrows():
            disc_name = tns_row["Disc. Internal Name"]
            if disc_name in self.target_lookup:
                target = self.target_lookup[disc_name]
                tns_data = target.get_target_data("tns")
                tns_data.parameters = tns_row.to_dict()
                target.alt_ids["tns"] = tns_data.parameters["Name"]
                matched = matched + 1
            else:
                unmatched_rows.append(tns_row)

        logger.info(f"matched {matched} TNS objects by name")
        self.tns_results = pd.DataFrame(unmatched_rows)

    def match_tns_on_coordinates(self, seplimit=5 * u.arcsec, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        logger.info(f"match TNS data on coords <{seplimit:.1f}")

        tns_candidate_coords = SkyCoord(
            self.tns_results["RA"], self.tns_results["DEC"], unit=(u.hourangle, u.deg)
        )

        target_candidate_coords = []
        target_candidate_target_ids = []
        for target_id, target in self.target_lookup.items():
            if target_id in self.recent_coordinate_searches:
                continue
            tns_data = target.target_data.get("tns", None)
            if tns_data:
                if tns_data.parameters:
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

        if not len(np.unique(target_match_idx)) == len(target_match_idx):
            logger.warning("target matches multiple TNS rows")
        if not len(np.unique(tns_match_idx)) == len(tns_match_idx):
            logger.warning("TNS row matches multiple targets")

        logger.info(f"coordinate match for {len(target_match_idx)} TNS objects")
        for ii, (idx1, idx2, skysep) in enumerate(
            zip(target_match_idx, tns_match_idx, skysep)
        ):
            target_id = target_candidate_target_ids[idx1]
            target = self.target_lookup[target_id]

            tns_row = self.tns_results.iloc[idx2]
            try:
                tns_parameters = tns_row.to_dict()
            except Exception as e:
                logger.warning(f"tns_row.to_dict() failed for {target_id}")
                print(e)
                raise ValueError(e)
            tns_data = target.get_target_data("tns")
            tns_data.parameters = tns_parameters
            target.alt_ids["tns"] = tns_parameters["Name"]

        self.recent_coordinate_searches.update(target_candidate_target_ids)
        # This set is emptied every time TNS-results is read in.

    def perform_all_tasks(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        self.query_for_updates(t_ref=t_ref)

        self.match_tns_on_names(t_ref=t_ref)  # If the "Disc. Name" directly matches...
        self.match_tns_on_coordinates(t_ref=t_ref)


class TnsQuery:
    # allowed_parameters = allowed_tns_parameters
    tns_base_url = "https://www.wis-tns.org/search"
    sleep_time = 3.0

    def __init__(self):
        pass

    @staticmethod
    def check_parameters(
        search_params: dict, allowed_parameters=ALLOWED_TNS_PARAMETERS
    ):
        for k, v in search_params.items():
            if not k in allowed_parameters:
                logger.warning(f"unexpected TNS keyword {k}")

    @staticmethod
    def build_param_url(search_params: dict):
        url_components = []
        for k, v in search_params.items():
            if isinstance(v, bool):
                v = int(v)
            url_components.append(f"&{k}={str(v)}")
        return "".join(url_components)

    @classmethod
    def do_post(cls, search_params: dict, tns_headers: dict, page: int):
        param_url = cls.build_param_url(search_params)
        url = f"{cls.tns_base_url}?{param_url}&page={page}"
        return requests.post(url, headers=tns_headers)

    @classmethod
    def process_response(cls, response):
        response_data = response.text.splitlines()

        columns = response_data[0].replace('"', "").split(",")
        # Find everything in quotes, eg. "SN 2022sai", "ASAS-SN, ALeRCE"
        # Can't use split(",") here, as some entries eg. "ASAS-SN, ALeRCE" would split.
        regex_quoted = re.compile('"[^"]*"')
        df_data = [
            [dat.replace('"', "") or np.nan for dat in regex_quoted.findall(row)]
            for row in response_data[1:]
        ]
        df = pd.DataFrame(df_data, columns=columns)
        return df

    @classmethod
    def query(cls, search_params: dict, tns_headers: dict):
        utils.check_unexpected_config_keys(
            search_params, ALLOWED_TNS_PARAMETERS, name="tns_query.search_params"
        )

        num_page = int(search_params.get("num_page", 50))
        if num_page > 50:
            logger.warning(f"num_page(={num_page}) > max allowed(=50). fixed.")
            num_page = 50
        search_params["num_page"] = num_page

        page = 0
        df_list = []
        while True:
            t1 = time.perf_counter()
            response = cls.do_post(search_params, tns_headers, page)
            df = cls.process_response(response)
            N_results = len(df)

            status = response.status_code
            req_limit = response.headers.get("x-rate-limit-limit")
            req_remaining = response.headers.get("x-rate-limit-remaining")
            req_reset = response.headers.get("x-rate-limit-reset")

            logger.info(
                f"{req_remaining}/{req_limit} requests left (reset {req_reset}s)"
            )
            if (response.status_code != 200) or N_results == 0:
                logger.info(f"break: page {page} status={status}, len={N_results}")
                logger.info(
                    f"{req_remaining}/{req_limit} requests left (reset {req_reset}s)"
                )
                break

            df_list.append(df)

            if N_results < int(num_page):
                logger.info(f"break after {N_results} results < {num_page}")
                break

            t2 = time.perf_counter()
            query_time = t2 - t1

            if cls.sleep_time is not None:
                if cls.sleep_time - query_time > 0:
                    time.sleep(cls.sleep_time - query_time)

            logger.info(f"{N_results} results from page {page} in {query_time:.2f}")
            if int(req_remaining) < 2:
                logger.info(f"waiting {req_reset}s for reset...")
                time.sleep(int(req_reset) + 1.0)
            page = page + 1

        if not df_list:
            return pd.DataFrame(columns="Name")  # empty dataframe.

        results_df = pd.concat(df_list, ignore_index=True)
        return results_df
