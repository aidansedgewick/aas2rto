import io
import json
import os
import requests
import time
import warnings
from logging import getLogger
from requests.auth import HTTPBasicAuth
from typing import Dict, List, Tuple

import numpy as np

import pandas as pd

from astropy.time import Time

from aas2rto import utils
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup

logger = getLogger(__name__.split(".")[-1])


def target_from_yse():
    pass


def target_from_yse_query_row(
    target_id, row: pd.Series, coordinate_columns: Tuple[str, str] = None
):
    if not isinstance(row, pd.Series):
        row = pd.Series(row)

    if coordinate_columns is not None and len(coordinate_columns) == 2:
        col_pairs = [coordinate_columns] + [
            pair for pair in YseQueryManager.default_coordinate_columns
        ]
    else:
        col_pairs = YseQueryManager.default_coordinate_columns
    ra, dec = None, None
    for ra_col, dec_col in col_pairs:
        if ra_col in row.index and dec_col in row.index:
            ra = row[ra_col]
            dec = row[dec_col]
            break
    if ra is None or dec is None:
        logger.debug(f"can't guess coordinate columns for {target_id}")

    target = Target(target_id, ra=ra, dec=dec, source="yse")
    return target


def process_yse_query_results(
    input_query_updates: pd.DataFrame, sorting_columns="name"
):
    query_updates = input_query_updates.copy()
    query_updates.sort_values(sorting_columns, inplace=True)
    query_updates.drop_duplicates(subset="name", keep="last", inplace=True)
    query_updates.set_index("name", verify_integrity=True, inplace=True)
    return query_updates


def process_yse_lightcurve(
    input_lightcurve: pd.DataFrame, additional_sources=None, use_all_sources=False
):
    lightcurve = input_lightcurve.copy()
    lightcurve.drop(["varlist:"], inplace=True, axis=1)
    lightcurve.sort_values("mjd", inplace=True)
    jd_dat = Time(lightcurve["mjd"].values, format="mjd").jd
    lightcurve.insert(1, "jd", jd_dat)
    # if only_yse_data:
    if not use_all_sources:
        mask = lightcurve["telescope"].str.startswith("Pan-STARRS")
        if additional_sources:
            if isinstance(additional_sources, str):
                additional_sources = [additional_sources]
            for source_key in additional_sources:
                # make sure eg. "ZTF" is translated to "P48" YSE-convention...
                source = ADDITIONAL_SOURCES_LOOKUP[source_key]

                source_mask = lightcurve["telescope"].str.lower() == source.lower()
                mask = mask | source_mask

        lightcurve = lightcurve[mask]

    ### FOR NOW, JUST IGNORE EVERYTHING WITH DQ 'BAD'
    bad_dq_mask = lightcurve["dq"].astype("str").str.lower() == "bad"
    lightcurve = lightcurve[~bad_dq_mask]

    lightcurve = lightcurve[(0 < lightcurve["magerr"]) & (lightcurve["magerr"] < 1.0)]
    lightcurve = lightcurve[lightcurve["flt"].str.lower() != "unknown"]

    lightcurve.reset_index(drop=True, inplace=True)

    if len(lightcurve) == 0:
        return lightcurve
    lightcurve.loc[:, "tag"] = "valid"
    low_snr_mask = 1.0 / lightcurve["magerr"] < 3

    bad_dq_mask = lightcurve["dq"].astype("str").str.lower() == "bad"

    badquality_mask = low_snr_mask | bad_dq_mask
    lightcurve.loc[badquality_mask, "tag"] = "badqual"
    return lightcurve


def get_object_updates(
    existing_results: pd.DataFrame,
    updated_results: pd.DataFrame,
    comparison_col="ndetections",
    name_col="name",
):
    if existing_results is None:
        return updated_results

    if updated_results is None:
        return pd.DataFrame(columns=[name_col, comparison_col])

    existing_results = existing_results.copy()
    existing_results.set_index(name_col, inplace=True, drop=False)

    updated_results = updated_results.copy()
    updated_results.set_index(name_col, inplace=True, drop=False)

    # get INNER JOIN

    updates = []
    for name, updated_row in updated_results.iterrows():
        if name in existing_results.index:
            existing_row = existing_results.loc[name]
            if updated_row[comparison_col] > existing_row[comparison_col]:
                logger.info(
                    f"updated!: {updated_row[comparison_col]} vs {existing_row[comparison_col]}"
                )
                updates.append(updated_row)
            else:
                logger.info(
                    f"skip: {updated_row[comparison_col]} vs {existing_row[comparison_col]}"
                )
    if len(updates) > 0:
        updates_df = pd.DataFrame(updates)
        updates_df.sort_values("name")
        return updates_df
    else:
        return None


def yse_id_from_target(target, resolving_order=("yse", "tns")):

    for source_name in resolving_order:
        yse_id = target.alt_ids.get(source_name, None)
        if yse_id is not None:
            return yse_id
    return None


ADDITIONAL_SOURCES_LOOKUP = {
    "ATLAS": "ATLAS",
    "P48": "P48",
    "ZTF": "P48",
    "Swift": "Swift",
    "UVOT": "Swift",
}


class YseQueryManager(BaseQueryManager):
    name = "yse"
    default_query_parameters = {
        "n_objects": 25,
        "object_query_interval": 1.0,  # how often to query for new objects
        "lightcurve_update_interval": 1.0,  # how often to update each target
        "max_latest_lookback": 30.0,  # bad if latest data is older than 30 days
        "max_earliest_lookback": 70.0,  # bad if the youngest data is older than 70 days (AGN!)
        "max_failed_queries": 10,  # after X failed queries, stop trying
        "max_total_query_time": 600,  # total time to spend in each stage seconds
        "use_all_sources": True,
        "additional_sources": None,
    }
    default_coordinate_columns = (("transient_RA", "transient_Dec"),)
    default_update_indicator_columns = ("number_of_detection", "latest_detection")

    expected_object_query_keys = (
        "query_id",
        "name_col",
        "coordinate_cols",
        "comparison_col",
    )
    required_object_query_keys = ["query_id"]

    expected_additional_sources = ADDITIONAL_SOURCES_LOOKUP

    def __init__(
        self,
        yse_config: Dict,
        target_lookup: TargetLookup,
        parent_path=None,
        create_paths=True,
    ):
        self.yse_config = yse_config
        self.target_lookup = target_lookup

        self.credential_config = self.yse_config.get("credential_config", {})
        self.auth = YseQuery.prepare_auth(self.credential_config)
        self.check_query_parameters()

        object_queries = self.yse_config.get("object_queries", [])
        self.object_queries = object_queries
        self.check_yse_queries()

        self.query_results = {}
        self.query_updates = {}

        self.process_paths(
            parent_path=parent_path,
            create_paths=create_paths,
            directories=["lightcurves", "parameters", "query_results"],
        )

    def check_query_parameters(self, t_ref: Time = None):
        self.query_parameters = self.default_query_parameters.copy()
        query_params = self.yse_config.get("query_parameters", {})
        unknown_kwargs = [
            key for key in query_params if key not in self.query_parameters
        ]
        if len(unknown_kwargs) > 0:
            msg = f"\033[33munexpected query_parameters:\033[0m\n    {unknown_kwargs}"
            logger.warning(msg)
        self.query_parameters.update(query_params)

        additional_sources = query_params.get("additional_sources", None)

        if additional_sources is not None:
            unknown_sources = utils.check_unexpected_config_keys(
                additional_sources,
                self.expected_additional_sources.keys(),
                name="yse.additional_sources",
            )
            if unknown_sources:

                rev_sources = {
                    s: [] for s in set(self.expected_additional_sources.values())
                }
                for alias, source in self.expected_additional_sources.items():
                    rev_sources[source].append(alias)

                raise ValueError(
                    f"unknown 'additional sources' {unknown_sources}.\nChoose from (sources and allowed aliases):\n"
                    f"{rev_sources}\n"
                )

    def check_yse_queries(self):

        for query_name, query_pattern in self.object_queries.items():
            unexpected_keys = utils.check_unexpected_config_keys(
                query_pattern.keys(),
                self.expected_object_query_keys,
                name=f"yse.object_queries.{query_name}",
            )
            if unexpected_keys:
                logger.warning(f"{unexpected_keys}")

            missing_keys = utils.check_missing_config_keys(
                query_pattern.keys(),
                self.required_object_query_keys,
                name=f"yse.object_queries.{query_name}",
            )
            if len(missing_keys) > 0:
                msg = f"yse.object_query.{query_name} missing required keys {missing_keys}"
                raise ValueError(msg)

    def load_existing_query_results(self):
        for query_name, query_parameters in self.yse_queries.items():
            query_results_filepath = self.get_query_results_file(query_name)
            logger.info(f"load {query_name} results")
            try:
                df = pd.read_csv(query_results_filepath)
                self.query_results[query_name] = df
            except pd.errors.EmptyDataError as e:
                logger.error(f"empty data for query {query_name}")

    def query_for_updates(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        for query_name, query_pattern in self.object_queries.items():
            query_id = query_pattern["query_id"]

            query_results_file = self.get_query_results_file(query_name)
            results_file_age = utils.calc_file_age(query_results_file, t_ref=t_ref)

            if results_file_age < self.query_parameters["object_query_interval"]:
                continue  # don't need to make this query

            else:
                logger.info(f"perform query: {query_name} (id={query_id})")
                existing_results = self.query_results.get(query_name)

                new_results = YseQuery.query_explorer(query_id, self.auth)
                new_results.to_csv(query_results_file, index=False)

                if new_results.empty:
                    continue

                comparison_col = query_pattern.get("comparison_col")
                updates = get_object_updates(
                    existing_results, new_results, comparison_col=comparison_col
                )

                self.query_updates[query_name] = updates

    def get_updated_targets(self):

        updated_targets = []
        new_targets = []
        for query_name, updates in self.query_updates.items():

            if updates is None:
                logger.info(f"no updates for {query_name}")
                continue

            if not isinstance(updates, pd.DataFrame):
                msg = f"updates from {query_name} is type {type(updates)}, not pd.DF!"
                warnings.warn(Warning(msg))
                continue

            logger.info("{len(updates)} for {query_nae}")

            query_pattern = self.object_queries[query_name]
            coordinate_cols = query_pattern.get("coordinate_cols")
            name_col = query_pattern.get("name_col")

            if name_col not in updates:
                msg = f"{name_col} not in updates columns:\n    {updates.columns}"
                warnings.warn(Warning(msg))
                logger.warning(msg)

            for ii, row in updates.iterrows():
                target_id = row[name_col]
                if target_id in self.target_lookup:
                    target = self.target_lookup.get(target_id)
                    yse_id = yse_id_from_target(target)
                    updated_targets.append(yse_id)
                    target.updated = True

                    target.update_messages.append("YSE data updated")
                else:
                    target = target_from_yse_query_row(target_id, row, coordinate_cols)
                    target.updated = True
                    target.update_messages.append("New YSE target")
                    self.target_lookup.add_target(target)
                    updated_targets.append(target_id)
                    new_targets.append(target_id)

        new_targets = list(set(new_targets))
        updated_targets = list(set(new_targets))
        logger.info(
            f"updates for {len(updated_targets)} targets ({len(new_targets)} are new)"
        )
        return updated_targets

    def reset_query_updates(self):

        for query_name in self.query_updates.keys():
            self.query_updates[query_name] = None

    def get_transient_parameters_to_query(
        self, yse_id_list: List[str] = None, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        if yse_id_list is None:
            yse_id_list = []
            for target_id, target in self.target_lookup.items():
                yse_id = yse_id_from_target(target)
                if yse_id is None:
                    continue
                yse_id_list.append(yse_id)

        to_query = []
        for yse_id in yse_id_list:
            parameters_file = self.get_parameters_file(yse_id, fmt="json")
            parameters_file_age = utils.calc_file_age(
                parameters_file, t_ref, allow_missing=True
            )
            if (
                parameters_file_age
                > self.query_parameters["lightcurve_update_interval"]
            ):
                to_query.append(yse_id)
        return to_query

    def perform_transient_parameters_queries(self, target_id_list, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        success = []
        failed = []

        logger.info(f"attempt {len(target_id_list)} transient parameter queries")
        t_start = time.perf_counter()
        for target_id in target_id_list:
            if len(failed) >= self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed transient parameters queries ({len(failed)}). Stop for now."
                logger.info(msg)
                break
            logger.debug(f"transient query for {target_id}")
            try:
                transient_parameters = YseQuery.query_transient_parameters(
                    target_id, auth=self.auth
                )
            except Exception as e:
                failed.append(target_id)
                logger.error(f"{target_id} params query failed: {e}")
                continue
            if transient_parameters is None:
                continue
            if len(transient_parameters) == 0:
                continue
            parameters_file = self.get_parameters_file(target_id, fmt="json")
            with open(parameters_file, "w+") as f:
                json.dump(transient_parameters, f)
            success.append(target_id)

        N_success = len(success)
        N_failed = len(failed)
        t_end = time.perf_counter()
        if N_success > 0 or N_failed > 0:
            logger.info(f"parameters queries in {t_end-t_start:.1f}s")
            logger.info(f"{N_success} successful, {N_failed} failed queries")
        return success, failed

    def load_transient_parameters(
        self, target_id_list: List[str] = None, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        if target_id_list is None:
            target_id_list = list(self.target_lookup.keys())

        loaded = []
        missing = []
        coords_updated = []
        t_start = time.perf_counter()
        for target_id in target_id_list:
            target = self.target_lookup.get(target_id, None)
            if target is None:
                logger.warning(f"\033[33mmissing target {target_id}\033[0m")
                continue
            parameters_file = self.get_parameters_file(target_id, fmt="json")
            if not parameters_file.exists():
                missing.append(target_id)
                continue
            with open(parameters_file, "r") as f:
                parameters = json.load(f)

            yse_data = target.get_target_data("yse")
            yse_data.parameters = parameters
            if target.ra is None or target.dec is None:
                ra = parameters.get("ra", None)
                dec = parameters.get("dec", None)
                if (ra is not None) and (dec is not None):
                    target.update_coordinates(ra, dec)
                    coords_updated.append(target_id)

        t_end = time.perf_counter()
        N_loaded = len(loaded)
        N_missing = len(missing)
        if N_loaded > 0:
            logger.info(f"loaded {N_loaded} parameter files in {t_end-t_start:.1f}s")
        if N_missing > 0:
            logger.info(f"{N_missing} parameter files missing!")
        if len(coords_updated) > 0:
            logger.info(f"{len(coords_updated)} coordinates updated")
        return loaded, missing

    def check_coordinates(self):
        missing_coordinates = []
        for target_id, target in self.target_lookup.items():
            if target.ra is None or target.dec is None:
                missing_coordinates.append(target_id)
        if len(missing_coordinates) > 0:
            logger.warning(f"\033[33m{len(missing_coordinates)} missing ra/dec!\033[0m")

    def update_yse_ids(self):

        missing = []
        changed = []
        for target_id, target in self.target_lookup.items():
            tns_name = target.alt_ids.get("tns", None)
            if tns_name is not None:
                yse_id = target.alt_ids.get("yse", None)
                if yse_id is None:
                    missing.append(target_id)
                if yse_id != tns_name:
                    changed.append(target_id)
                target.alt_ids["yse"] = tns_name
        if len(missing) > 0:
            logger.info(f"{len(missing)} targets added yse_id from tns")
        if len(changed) > 0:
            logger.info(f"{len(changed)} targets have yse_id CHANGED!")

    def get_lightcurves_to_query(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        to_query = []
        for target_id, target in self.target_lookup.items():
            yse_id = yse_id_from_target(target)
            if yse_id is None:
                continue
            lightcurve_file = self.get_lightcurve_file(yse_id)
            lightcurve_file_age = utils.calc_file_age(
                lightcurve_file, t_ref, allow_missing=True
            )

            interval = self.query_parameters["lightcurve_update_interval"]
            if lightcurve_file_age > interval:
                to_query.append(yse_id)
        return to_query

    def perform_lightcurve_queries(self, yse_id_list=None, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        success = []
        failed = []

        logger.info(f"attempt {len(yse_id_list)} lightcurve queries")
        t_start = time.perf_counter()
        for ii, yse_id in enumerate(yse_id_list):
            if ii % 25 == 0 & ii > 0:
                logger.info(f"{ii} queries...")

            lightcurve_file = self.get_lightcurve_file(yse_id)
            if len(failed) >= self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed queries ({len(failed)}), stop for now"
                logger.info(msg)
                break
            t_now = time.perf_counter()
            if t_now - t_start > self.query_parameters["max_total_query_time"]:
                logger.info(f"querying time ({t_now - t_start:.1f}s) exceeded max")
                break

            try:
                logger.debug(f"query for {yse_id} lc")
                lightcurve = YseQuery.query_lightcurve(yse_id, auth=self.auth)
            except Exception as e:
                print(e)
                logger.warning(f"{yse_id} lightcurve query failed")
                failed.append(yse_id)
                continue
            if lightcurve.empty:
                lightcurve.to_csv(lightcurve_file, index=True)

            lightcurve.to_csv(lightcurve_file, index=False)
            success.append(yse_id)

        N_success = len(success)
        N_failed = len(failed)
        if N_success > 0 or N_failed > 0:
            t_end = time.perf_counter()
            logger.info(f"lightcurve queries in {t_end - t_start:.1f}s")
            logger.info(f"{N_success} successful, {N_failed} failed lc queries")
        return success, failed

    def load_single_lightcurve(self, yse_id, t_ref=None):
        lightcurve_filepath = self.get_lightcurve_file(yse_id)
        if not lightcurve_filepath.exists():
            logger.warning(f"{yse_id} is missing lightcurve")
            return None
        try:
            lightcurve = pd.read_csv(lightcurve_filepath)
            additional_sources = self.query_parameters["additional_sources"]
            use_all_sources = self.query_parameters["use_all_sources"]
            try:
                lightcurve = process_yse_lightcurve(
                    lightcurve,
                    additional_sources=additional_sources,
                    use_all_sources=use_all_sources,
                )
                return lightcurve
            except KeyError as e:
                logger.error(f"{yse_id}")
                raise ValueError

        except pd.errors.EmptyDataError as e:
            print(e)
            logger.warning(f"bad (empty) lightcurve file for {yse_id}")
            return None

    def perform_all_tasks(self, startup=False, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if startup:
            logger.info("don't query for new updates at startup")
        else:
            try:
                self.query_for_updates(t_ref=t_ref)
            except Exception as e:
                return e

            try:
                updated_targets = self.get_updated_targets()
            except Exception as e:
                return e
            self.reset_query_updates()

            if updated_targets:
                try:
                    self.perform_lightcurve_queries(updated_targets)
                except Exception as e:
                    return e

            to_query = self.get_lightcurves_to_query(t_ref=t_ref)
            success, failed = self.perform_lightcurve_queries(to_query, t_ref=t_ref)

            # to_query = self.get_transient_parameters_to_query(t_ref=t_ref)
            # success, failed = self.perform_transient_parameters_queries(
            #    to_query, t_ref=t_ref
            # )

        self.update_yse_ids()
        loaded, missing = self.load_transient_parameters(t_ref=t_ref)
        loaded, missing = self.load_target_lightcurves(
            t_ref=t_ref, id_from_target_function=yse_id_from_target
        )

        self.check_coordinates()


class YseQuery:
    """See also https://github.com/berres2002/prep"""

    base_url = "https://ziggy.ucolick.org/yse/"

    required_credential_parameters = ("username", "password")

    @classmethod
    def prepare_auth(cls, credential_config) -> HTTPBasicAuth:
        required = cls.required_credential_parameters
        missing_keys = utils.check_missing_config_keys(
            credential_config, required, name="yse.auth_config"
        )
        if len(missing_keys) > 0:
            errormsg = (
                f"yse: credential_config: provide {required} (missing {missing_keys})"
            )
            raise ValueError(errormsg)

        return requests.auth.HTTPBasicAuth(
            credential_config.get("username"),
            credential_config.get("password"),
        )

    @classmethod
    def query_explorer(cls, query_id: int, auth: HTTPBasicAuth, process_result=True):
        url = f"{cls.base_url}/explorer/{query_id}/download"
        response = requests.get(url, auth=auth)
        if not process_result:
            return response
        # Remove some junk characters at the beginning...
        result_text = response.text.strip("ï»¿")
        return pd.read_csv(io.StringIO(result_text))

    @classmethod
    def query_transient_parameters(
        cls, name: str, auth: HTTPBasicAuth, process_result=True
    ):
        url = f"{cls.base_url}/api/transients/?name={name}"
        response = requests.get(url, auth=auth)
        if not process_result:
            return response
        return response.json()["results"][0]

    @classmethod
    def query_lightcurve(cls, name: str, auth: HTTPBasicAuth, process_result=True):
        url = f"{cls.base_url}/download_photometry/{name}/"
        response = requests.get(url, auth=auth)
        if not process_result:
            return response
        return cls.process_lightcurve(response.text)

    @classmethod
    def process_lightcurve(cls, lc_text: str, return_header=False):
        header = {}
        columns = None
        data = []
        for line in lc_text.split("\n"):
            if len(line.strip()) == 0:
                continue
            if line.startswith("#"):
                key, val = line.split("#")[1].strip().split(":", 1)
                # print(key_val)
                header[key.lower().strip()] = val.strip() or None
                continue
            # The first line that doesn't start with a # is the column names
            if columns is None:
                columns = line.lower()
                data.append(columns)
                continue
            data.append(line)

        if len(data) == 1:
            lightcurve = pd.DataFrame([], columns=columns)
        else:
            lightcurve = pd.read_csv(io.StringIO("\n".join(data)), sep="\s+")
        if return_header:
            return lightcurve, header
        return lightcurve
