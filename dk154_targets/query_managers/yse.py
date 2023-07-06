import io
import json
import os
import requests
import time
from logging import getLogger
from requests.auth import HTTPBasicAuth
from typing import Dict, List, Tuple

import numpy as np

import pandas as pd

from astropy.time import Time

from dk154_targets import Target
from dk154_targets.query_managers.base import BaseQueryManager
from dk154_targets.utils import calc_file_age

logger = getLogger(__name__.split(".")[-1])


def target_from_yse():
    pass


def target_from_yse_query_row(
    objectId, row: pd.Series, coordinate_columns: Tuple[str, str] = None
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
        logger.debug(f"can't guess coordinate columns for {objectId}")

    target = Target(objectId, ra=ra, dec=dec)
    return target


def process_yse_query_results(
    input_query_updates: pd.DataFrame, sorting_columns="name"
):
    query_updates = input_query_updates.copy()
    query_updates.sort_values(sorting_columns, inplace=True)
    query_updates.drop_duplicates(subset="name", keep="last", inplace=True)
    query_updates.set_index("name", verify_integrity=True, inplace=True)
    return query_updates


def process_yse_lightcurve(input_lightcurve: pd.DataFrame, only_yse_data=True):
    lightcurve = input_lightcurve.copy()
    lightcurve.drop(["varlist:"], inplace=True, axis=1)
    lightcurve.sort_values("mjd")
    jd_dat = Time(lightcurve["mjd"].values, format="mjd").jd
    lightcurve.insert(1, "jd", jd_dat)
    if only_yse_data:
        lightcurve.query("instrument=='GPC1'", inplace=True)
    lightcurve.query("(0<magerr) & (magerr < 1)", inplace=True)
    lightcurve.reset_index(drop=True)

    lightcurve["tag"] = "valid"
    bad_qual_mask = 1.0 / lightcurve["magerr"] < 3
    lightcurve.loc[bad_qual_mask, "tag"] = "badquality"
    return lightcurve


def _get_indicator_guesses(col):
    if col is None:
        return YseQueryManager.default_update_indicator_columns
    return [col] + list(YseQueryManager.default_update_indicator_columns)


def _get_indicator_column(query_config, query_updates: pd.DataFrame):
    if not isinstance(query_updates, pd.DataFrame):
        return None
    indicator_col = query_config.get("updated_indicator", None)
    indicator_guesses = _get_indicator_guesses(indicator_col)
    update_indicator = None
    for col in indicator_guesses:
        if col in query_updates.columns:
            update_indicator = col
            break
    return update_indicator


class YseQueryManager(BaseQueryManager):
    name = "yse"
    default_query_parameters = {
        "n_objects": 25,
        "interval": 1.0,  # how often to query for new objects
        "update": 2.0,  # how often to update each target
        "max_latest_lookback": 30.0,  # bad if latest data is older than 30 days
        "max_earliest_lookback": 70.0,  # bad if the younest data is older than 70 days (AGN!)
        "max_failed_queries": 10,  # after X failed queries, stop trying
        "max_total_query_time": 600,  # total time to spend in each stage seconds
        "use_only_yse": True,
    }
    default_coordinate_columns = (("transient_RA", "transient_Dec"),)
    default_update_indicator_columns = ("number_of_detection", "latest_detction")
    default_yse_only = True

    def __init__(
        self,
        yse_config: Dict,
        target_lookup: Dict[str, Target],
        data_path=None,
        create_paths=True,
    ):
        self.yse_config = yse_config
        self.target_lookup = target_lookup

        self.credential_config = self.yse_config.get("credential_config", {})
        self.auth = YseQuery.prepare_auth(self.credential_config)

        yse_queries = self.yse_config.get("yse_queries", [])
        self.yse_queries = yse_queries

        self.process_query_parameters()

        self.query_results = {}
        self.query_updates = {}
        self.query_updates_available = False

        self.process_paths(data_path=data_path, create_paths=create_paths)

    def process_query_parameters(self, t_ref: Time = None):
        self.query_parameters = self.default_query_parameters.copy()
        query_params = self.yse_config.get("query_parameters", {})
        unknown_kwargs = [
            key for key in query_params if key not in self.query_parameters
        ]
        if len(unknown_kwargs) > 0:
            msg = f"\033[33munexpected query_parameters:\033[0m\n    {unknown_kwargs}"
            logger.warning(msg)
        self.query_parameters.update(query_params)

    def query_for_updates(self, t_ref: Time = None) -> pd.DataFrame:
        t_ref = t_ref or Time.now()

        for query_id, column_data in self.yse_queries.items():
            query_name = f"yse_{query_id}"
            query_results_file = self.get_query_results_file(query_name)
            results_file_age = calc_file_age(
                query_results_file, t_ref, allow_missing=True
            )
            if results_file_age < self.query_parameters["interval"]:
                if query_results_file.stat().st_size > 1:
                    logger.debug("read")
                    logger.info(f"read existing yse id {query_id} results")
                    raw_query_updates = pd.read_csv(query_results_file)

                    query_updates = process_yse_query_results(raw_query_updates)
                    if query_updates.empty:
                        continue
                    self.query_updates[query_id] = query_updates
                    # It's fine that we don't make the query now (ie, don't go to "else:")
                    # we just assume that there were no results if the file is tiny.

            else:
                # if we need to make the query...
                logger.info(f"query for {query_name}")
                t_start = time.perf_counter()
                raw_query_updates = YseQuery.query_explorer(query_id, self.auth)
                if raw_query_updates.empty:
                    raw_query_updates.to_csv(query_results_file, index=False)
                raw_query_updates["query_id"] = query_id
                raw_query_updates.to_csv(query_results_file, index=False)
                query_updates = process_yse_query_results(raw_query_updates)
                self.query_updates[query_id] = query_updates
                t_end = time.perf_counter()
                logger.info(
                    f"{query_name} returned {len(query_updates)} in {t_end-t_start:.1f}s"
                )
                self.query_updates[query_id] = query_updates
                self.query_updates_available = True

    def new_targets_from_updates(self, updates: pd.DataFrame, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        new_targets = []
        if updates is None:
            return new_targets

        logger.info(f"{len(updates)} updates to process")
        for objectId, row in updates.iterrows():
            target = self.target_lookup.get(objectId, None)
            if target is not None:
                continue
            if "query_id" in row.index:
                query_id = row.query_id

                coordinate_columns = self.yse_queries[query_id].get("coordinates", None)
                if coordinate_columns is not None:
                    if len(coordinate_columns) != 2:
                        coordinate_columns = None
                        logger.info(
                            "provide pair of coordinate columns, eg. "
                            "['transient_RA', 'transient_Dec']"
                        )
            else:
                coordinate_columns = None
            target = target_from_yse_query_row(
                objectId, row, coordinate_columns=coordinate_columns
            )
            self.target_lookup[objectId] = target
            new_targets.append(objectId)
        return new_targets

    def target_updates_from_query_results(self, t_ref: Time = None):
        update_dfs = []
        for query_id, updated_results in self.query_updates.items():
            # Decide which column to use to get updated targets:
            update_indicator = _get_indicator_column(
                self.yse_queries[query_id], updated_results
            )

            updated_targets = []
            existing_results = self.query_results.get(query_id, None)

            if existing_results is None:
                self.query_results[query_id] = updated_results
                logger.info(
                    f"no existing id={query_id} results, use updates {len(updated_results)}"
                )
                continue
            # existing_results.sort_values(["oid", "lastmjd"], inplace=True)
            for objectId, updated_row in updated_results.iterrows():
                if objectId in existing_results.index:
                    existing_row = existing_results.loc[objectId]
                    if update_indicator is None:
                        updated_targets.append(objectId)
                        continue
                    if updated_row[update_indicator] > existing_row[update_indicator]:
                        updated_targets.append(objectId)
                else:
                    updated_targets.append(objectId)
            self.query_results[query_id] = updated_results
            updated = updated_results.loc[updated_targets]
            update_dfs.append(updated)

        if len(update_dfs) > 0:
            # yse_updates = updated_results.loc[updated_targets]
            yse_updates = pd.concat(update_dfs)
            logger.info(f"{len(yse_updates)} yse target updates")
        else:
            yse_updates = None
        return yse_updates

    def get_transient_parameters_to_query(
        self, objectId_list: List[str] = None, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        if objectId_list is None:
            objectId_list = list(self.target_lookup.keys())

        to_query = []
        for objectId in objectId_list:
            parameters_file = self.get_parameters_file(objectId, fmt="json")
            parameters_file_age = calc_file_age(
                parameters_file, t_ref, allow_missing=True
            )
            if parameters_file_age > self.query_parameters["interval"]:
                to_query.append(objectId)
        return to_query

    def perform_transient_parameters_queries(self, objectId_list, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        success = []
        failed = []

        logger.info(f"attempt {len(objectId_list)} transient parameter queries")
        t_start = time.perf_counter()
        for objectId in objectId_list:
            if len(failed) >= self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed transient parameters queries ({len(failed)}). Stop for now."
                logger.info(msg)
                break
            logger.debug(f"transient query for {objectId}")
            try:
                transient_parameters = YseQuery.query_transient_parameters(
                    objectId, auth=self.auth
                )
            except Exception as e:
                failed.append(objectId)
                print(f"failed", e)
                continue
            if transient_parameters is None:
                continue
            if len(transient_parameters) == 0:
                continue
            parameters_file = self.get_parameters_file(objectId, fmt="json")
            with open(parameters_file, "w+") as f:
                json.dump(transient_parameters, f)
            success.append(objectId)

        N_success = len(success)
        N_failed = len(failed)
        t_end = time.perf_counter()
        if N_success > 0 or N_failed > 0:
            logger.info(f"parameters queries in {t_end-t_start:.1f}s")
            logger.info(f"{N_success} successful, {N_failed} failed queries")
        return success, failed

    def load_transient_parameters(
        self, objectId_list: List[str] = None, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        if objectId_list is None:
            objectId_list = list(self.target_lookup.keys())

        loaded = []
        missing = []
        coords_updated = []
        t_start = time.perf_counter()
        for objectId in objectId_list:
            target = self.target_lookup.get(objectId, None)
            if target is None:
                logger.warning(f"\033[33mmissing target {objectId}\033[0m")
                continue
            parameters_file = self.get_parameters_file(objectId, fmt="json")
            if not parameters_file.exists():
                missing.append(objectId)
            with open(parameters_file, "r") as f:
                parameters = json.load(f)
            target.yse_data.parameters = parameters
            if target.ra is None or target.dec is None:
                ra = parameters.get("ra", None)
                dec = parameters.get("dec", None)
                if (ra is not None) and (dec is not None):
                    target.update_coordinates(ra, dec)
                    coords_updated.append(objectId)

        t_end = time.perf_counter()
        N_loaded = len(loaded)
        N_missing = len(missing)
        if N_loaded > 0:
            logger.info(f"loaded {N_loaded} parameter files in {t_end-t_start:.1f}s")
        if N_missing > 0:
            logger.info(f"{N_loaded} parameter files missing!")
        if len(coords_updated) > 0:
            logger.info(f"{len(coords_updated)} coordinates updated")
        return loaded, missing

    def check_coordinates(self):
        missing_coordinates = []
        for objectId, target in self.target_lookup.items():
            if target.ra is None or target.dec is None:
                missing_coordinates.append(objectId)
        if len(missing_coordinates) > 0:
            logger.warning(f"\033[33m{len(missing_coordinates)} missing ra/dec!\033[0m")

    def get_lightcurves_to_query(
        self, objectId_list: List[str] = None, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        if objectId_list is None:
            objectId_list = list(self.target_lookup.keys())
        to_query = []
        for objectId in objectId_list:
            lightcurve_file = self.get_lightcurve_file(objectId)
            lightcurve_file_age = calc_file_age(
                lightcurve_file, t_ref, allow_missing=True
            )
            if lightcurve_file_age > self.query_parameters["update"]:
                to_query.append(objectId)
        return to_query

    def perform_lightcurve_queries(self, objectId_list, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        success = []
        failed = []
        logger.info(f"attempt {len(objectId_list)} lightcurve queries")
        t_start = time.perf_counter()
        for objectId in objectId_list:
            lightcurve_file = self.get_lightcurve_file(objectId)
            if len(failed) >= self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed queries ({len(failed)}), stop for now"
                logger.info(msg)
                break
            t_now = time.perf_counter()
            if t_now - t_start > self.query_parameters["max_total_query_time"]:
                logger.info(f"querying time ({t_now - t_start:.1f}s) exceeded max")
                break
            try:
                logger.debug(f"query for {objectId} lc")
                lightcurve = YseQuery.query_lightcurve(objectId, auth=self.auth)
            except Exception as e:
                print(e)
                logger.warning(f"{objectId} lightcurve query failed")
                failed.append(objectId)
                continue
            if lightcurve.empty:
                lightcurve.to_csv(lightcurve_file, index=True)

            try:
                lightcurve = process_yse_lightcurve(lightcurve)
            except KeyError as e:
                logger.error(f"{objectId}")
                raise ValueError
            lightcurve.to_csv(lightcurve_file, index=False)
            success.append(objectId)

        N_success = len(success)
        N_failed = len(failed)
        if N_success > 0 or N_failed > 0:
            t_end = time.perf_counter()
            logger.info(f"lightcurve queries in {t_end - t_start:.1f}s")
            logger.info(f"{N_success} successful, {N_failed} failed lc queries")

        return success, failed

    def load_lightcurves(self, objectId_list=None, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if objectId_list is None:
            objectId_list = list(self.target_lookup.keys())

        loaded = []
        missing = []
        t_start = time.perf_counter()
        for objectId in objectId_list:
            lightcurve_file = self.get_lightcurve_file(objectId)
            if not lightcurve_file.exists():
                missing.append(objectId)
                continue
            target = self.target_lookup.get(objectId, None)
            lightcurve = pd.read_csv(lightcurve_file)
            # TODO: not optimum to read every time... but not a bottleneck for now.
            if target is None:
                logger.warning(f"target {objectId} missing")
            else:
                existing_lightcurve = target.yse_data.lightcurve
                if existing_lightcurve is not None:
                    if len(lightcurve) == len(existing_lightcurve):
                        continue
            # lightcurve.sort_values("jd", inplace=True)
            loaded.append(objectId)
            target.yse_data.add_lightcurve(lightcurve)
            target.updated = True
            # if target.yse_data.lightcurve.iloc[-1, "candid"]
        t_end = time.perf_counter()

        N_loaded = len(loaded)
        N_missing = len(missing)
        if N_loaded > 0:
            logger.info(f"loaded {N_loaded} lightcurves in {t_end-t_start:.1f}s")
        if N_missing > 0:
            logger.warning(f"{N_missing} lightcurves missing...")
        return loaded, missing

    def perform_all_tasks(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        self.query_for_updates(t_ref=t_ref)
        if len(self.query_results) == 0:
            # This will only happen on the initial pass.
            for query_name, query_updates in self.query_updates.items():
                new_targets = self.new_targets_from_updates(query_updates, t_ref=t_ref)
                msg = f"{len(new_targets)} new targets from yse query id={query_name}"
                logger.info(msg)
                self.query_results[query_name] = query_updates
        if self.query_updates_available:
            logger.info("query updates are available!")
            yse_updates = self.target_updates_from_query_results(t_ref=t_ref)
            new_targets = self.new_targets_from_updates(yse_updates)
            msg = f"{len(new_targets)} new targets from yse query id={query_name}"
            logger.info(msg)

        to_query = self.get_transient_parameters_to_query(t_ref=t_ref)
        success, failed = self.perform_transient_parameters_queries(
            to_query, t_ref=t_ref
        )
        loaded, missing = self.load_transient_parameters(t_ref=t_ref)

        to_query = self.get_lightcurves_to_query(t_ref=t_ref)
        success, failed = self.perform_lightcurve_queries(to_query, t_ref=t_ref)
        loaded, missing = self.load_lightcurves(t_ref=t_ref)

        self.check_coordinates()
        self.query_updates_available = False

    def apply_messenger_updates(self, alerts):
        for alert in alerts:
            objectId = alert["objectId"]
            target = self.target_lookup.get(objectId, None)
            if target is None:
                continue
            target.send_updates = True
            topic_str = alert["topic"]
            alert_text = f"ipdates
            target.update_messages.append(alert_text)


class YseQuery:
    base_url = "https://ziggy.ucolick.org/yse/"

    required_credential_parameters = ("username", "password")

    @classmethod
    def prepare_auth(cls, credential_config) -> HTTPBasicAuth:
        required = cls.required_credential_parameters
        missing = [kw for kw in required if kw not in credential_config]
        if len(missing) > 0:
            errormsg = (
                f"`yse`: `credential_config` must contain "
                + " ".join(required)
                + ":\n    \033[31;1mmissing "
                + " ".join(missing)
                + "\033[0m"
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
        print(response.text)
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
            lightcurve = pd.read_csv(
                io.StringIO("\n".join(data)), delim_whitespace=True
            )
        if return_header:
            return lightcurve, header
        return lightcurve
