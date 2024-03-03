import copy
import json
import re
import requests
import time
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.time import Time, TimeDelta

from dk154_targets import Target, TargetData
from dk154_targets.query_managers.base import BaseQueryManager
from dk154_targets.utils import calc_file_age

logger = getLogger(__name__.split(".")[-1])

allowed_tns_parameters = [
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

    marker_dict = dict(tns_id=str(tns_uid), type="user", name=tns_user)
    marker_str = json.dumps(marker_dict)
    # marker_str is a literal string:
    #       '{"tns_id": "1234", "type": "user", "name": "your_name"}'

    tns_marker = f"tns_marker{marker_str}"
    return {"User-Agent": tns_marker}


class TnsQueryManager(BaseQueryManager):
    name = "tns"

    default_tns_parameters = {
        "num_page": "50",  # 50 is max allowed?
        "format": "csv",  # probably don't change this one...
    }

    default_query_parameters = {
        "query_interval": 0.25,
        "lookback_time": 60.0,
    }

    def __init__(
        self,
        tns_config: dict,
        target_lookup: Dict[str, Target],
        data_path=None,
        create_paths=True,
    ):
        self.tns_config = tns_config
        self.target_lookup = target_lookup

        self.recent_coordinate_searches = set()
        self.tns_results = None

        tns_user = self.tns_config.get("user")
        tns_uid = self.tns_config.get("uid")

        self.tns_headers = build_tns_headers(tns_user, tns_uid)

        self.query_parameters = self.default_query_parameters.copy()
        query_parameters = self.tns_config.get("query_parameters", {})
        self.query_parameters.update(query_parameters)

        self.tns_parameters = self.default_tns_parameters.copy()
        tns_parameters = self.tns_config.get("tns_parameters", {})
        self.tns_parameters.update(tns_parameters)

        self.process_paths(data_path=data_path, create_paths=create_paths)

    def perform_query(self, tns_parameters: dict = None, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        tns_parameters = tns_parameters or {}

        search_params = copy.deepcopy(self.tns_parameters)
        search_params.update(tns_parameters)

        if search_params.get("date_start[date]", None) is None:
            date_start = t_ref - TimeDelta(60.0, format="jd")
            search_params["date_start[date]"] = date_start.iso.split()[0]

        return TnsQuery.query(search_params, self.tns_headers)

    def concatenate_results(
        self, new_results: pd.DataFrame, existing_results_path: Path, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        if not existing_results_path.exists():
            return new_results
        existing_results = pd.read_csv(existing_results_path)
        logger.info(f"merge with {len(existing_results)} existing results")

        results = pd.concat([existing_results, new_results])
        results.drop_duplicates(subset="Name", keep="last", inplace=True)
        return results

    def update_matched_tns_names(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        for objectId, target in self.target_lookup.items():
            tns_name = target.tns_data.parameters.get("Name", None)
            if tns_name is not None:
                self.matched_tns_names[tns_name] = objectId

    def match_tns_on_names(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        logger.info("match TNS data by objectId")

        unmatched_rows = []
        matched = 0
        for idx, row in self.tns_results.iterrows():
            disc_name = row["Disc. Internal Name"]
            if disc_name in self.target_lookup:
                target = self.target_lookup[disc_name]
                target.tns_data.parameters = row.to_dict()
                matched = matched + 1
            else:
                unmatched_rows.append(row)

        logger.info(f"matched {matched} TNS objects by name")
        self.tns_results = pd.DataFrame(unmatched_rows)

    def match_tns_on_coordinates(self, seplimit=5 * u.arcsec, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        logger.info(f"match TNS data on coords <{seplimit:.1f}")

        tns_candidate_coords = SkyCoord(
            self.tns_results["RA"], self.tns_results["DEC"], unit=(u.hourangle, u.deg)
        )

        target_candidate_coords = []
        target_candidate_objectIds = []
        for objectId, target in self.target_lookup.items():
            if target.tns_data.parameters:
                continue
            if objectId in self.recent_coordinate_searches:
                continue
            target_candidate_coords.append(target.coord)
            target_candidate_objectIds.append(objectId)

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
            objectId = target_candidate_objectIds[idx1]
            target = self.target_lookup[objectId]

            tns_data = self.tns_results.iloc[idx2]
            try:
                target.tns_data.parameters = tns_data.to_dict()
            except Exception as e:
                print(e)
                print(objectId)
                print(tns_data)
                raise ValueError(e)

        self.recent_coordinate_searches.update(target_candidate_objectIds)
        # This set is emptied every time TNS-results is read in.

    def perform_all_tasks(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        existing_results_path = self.query_results_path / "tns_results.csv"

        results_age = calc_file_age(existing_results_path, t_ref=t_ref)
        if results_age > self.query_parameters["query_interval"]:
            logger.info("re-query for TNS data")
            new_results = self.perform_query()
            logger.info(f"{len(new_results)} TNS entries")
            results = self.concatenate_results(
                new_results, existing_results_path, t_ref=t_ref
            )
            results.to_csv(existing_results_path, index=False)
            self.tns_results = None  # guarantee re-read in the next step.

        if self.tns_results is None:
            logger.info("read TNS results from file")
            results = pd.read_csv(existing_results_path)
            self.tns_results = results
            self.recent_coordinate_searches = set()

        self.match_tns_on_names(t_ref=t_ref)  # If the "Disc. Name" directly matches...
        self.match_tns_on_coordinates(t_ref=t_ref)


class TnsQuery:
    allowed_parameters = allowed_tns_parameters
    tns_base_url = "https://www.wis-tns.org/search"
    sleep_time = 3.0

    def __init__(self):
        pass

    @classmethod
    def check_parameters(cls, search_params: dict):
        for k, v in search_params.items():
            if not k in cls.allowed_parameters:
                logger.warning(f"unexpected TNS keyword {k}")

    @classmethod
    def build_param_url(cls, search_params: dict):
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
        cls.check_parameters(search_params)

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
            requests_limit = response.headers.get("x-rate-limit-limit")
            requests_remaining = response.headers.get("x-rate-limit-remaining")
            requests_reset = response.headers.get("x-rate-limit-reset")

            logger.info(
                f"{requests_remaining}/{requests_limit} requests left (reset {requests_reset}s)"
            )
            if (response.status_code != 200) or N_results == 0:
                logger.info(f"break: page {page} status={status}, len={N_results}")
                logger.info(
                    f"{requests_remaining}/{requests_limit} requests left (reset {requests_reset}s)"
                )
                break

            df_list.append(df)

            t2 = time.perf_counter()

            if N_results < int(num_page):
                logger.info(f"break after {N_results} results < {num_page}")
                break

            logger.info(f"{N_results} results from page {page}")
            if int(requests_remaining) < 2:
                logger.info(f"waiting {requests_reset}s for reset...")
                time.sleep(int(requests_reset) + 1.0)

            page = page + 1

            if cls.sleep_time is not None:
                query_time = t2 - t1
                if cls.sleep_time - query_time > 0:
                    time.sleep(cls.sleep_time - query_time)

        if not df_list:
            return pd.DataFrame()  # empty dataframe.

        results_df = pd.concat(df_list, ignore_index=True)
        return results_df


if __name__ == "__main__":
    tns_config = {
        "user": "aidan",
        "uid": 1234,
        # password: dk154_targets  # not important?
        "tns_parameters": {
            "classified_sne": "1",
            "date_start[date]": "2023-07-22",
            "num_page": "50",  # 50 is max allowed?
            "format": "csv",  # probably don't change this one...
        },
        "query_parameters": {"query_interval": 1.0},
    }

    qm = TnsQueryManager(tns_config, {})

    print(qm.tns_marker)

    qm.perform_all_tasks()
