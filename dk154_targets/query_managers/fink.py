import gzip
import io
import json
import os
import pickle
import requests
import time
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.io import fits
from astropy.time import Time


try:
    import fink_client
except ModuleNotFoundError as e:
    fink_client = None

if fink_client is not None:
    from fink_client.avroUtils import write_alert, _get_alert_schema, AlertReader
    from fink_client.consumer import AlertConsumer

from dk154_targets import Target, TargetData
from dk154_targets.exc import (
    BadKafkaConfigError,
    MissingObjectIdError,
    MissingCoordinatesError,
)
from dk154_targets.query_managers.base import BaseQueryManager
from dk154_targets import utils
from dk154_targets.utils import calc_file_age

from dk154_targets import paths

logger = getLogger(__name__.split(".")[-1])


def combine_fink_detections_non_detections(
    detections: pd.DataFrame, non_detections: pd.DataFrame
):
    """
    Combine fink "detections" and "non-detections", correctly preserving `candid`.

    Parameters
    ----------
    detections
        the output of `FinkQuery.query_objects(objectId="ZTF23blahbla")`
    non_detections
        the output of `FinkQuery.query_objects(objectId="ZTF23blahbla", withupperlim=True)`

    Querying objects using `withupperlim=True` will return valid, badqual, and upperlim
    BUT values `candid` and other long ints from `valid` entries
    will have been irreversibly converted to floats (removing the last few digits).
    So we remove them from non-detections, add 0 to non_detections `candid` so that the
    int -> float conversion does not break these values, and re-concatenate detections.
    """

    # Check to see if we're working with sensible data...
    try:
        objectId = detections["objectId"].iloc[0]
    except:
        objectId = "bad_objectId"

    if "tag" in non_detections:
        if "valid" in np.unique(non_detections["tag"]):
            ldet = len(detections)
            lvalid = len(non_detections.query("tag=='valid'"))
            if not ldet == lvalid:
                msg = f"{objectId}: mismatch : detections ({ldet}) != 'valid' non_detections ({lvalid})"
                raise ValueError(msg)
    else:
        if (not non_detections.empty) and (not detections.empty):
            msg = (
                f"{objectId}: non_detections has no 'tag' (no detections), "
                "but detections is not empty"
            )
            raise ValueError(msg)

    # fix detections
    detections["tag"] = "valid"
    if not detections.empty:
        detections["candid_str"] = detections["candid"].astype(str)
    if "candid" not in detections.columns:
        detections["candid"] = 0

    # fix non-detections
    if not non_detections.empty:
        non_detections.query("tag!='valid'", inplace=True)
        if "candid" in non_detections.columns:
            if not all(pd.isnull(non_detections["candid"])):
                raise ValueError(
                    f"{objectId}: not all non-detections have Null `candid`"
                )

    non_detections["candid"] = -1
    lightcurve = pd.concat([detections, non_detections])
    return lightcurve


def process_fink_lightcurve(raw_lightcurve: pd.DataFrame):
    lightcurve = raw_lightcurve.copy(deep=True)
    if not lightcurve.empty:
        lightcurve.sort_values("jd", inplace=True)
        mjd_col = Time(lightcurve["jd"], format="jd").mjd
        lightcurve.insert(1, "mjd", mjd_col)
    else:
        lightcurve["jd"] = 0  # Fails later without a date column...
        lightcurve["mjd"] = 0
    if "candid" not in lightcurve.columns:
        # ie, LC of only non-detections/badqual - add, else fails to integrate alerts.
        lightcurve["candid"] = -1
    return lightcurve


def target_from_fink_alert(alert: dict, t_ref: Time = None) -> Target:
    t_ref = t_ref or Time.now()

    objectId = alert.get("objectId", None)
    if objectId is None:
        raise MissingObjectIdError(f"alert has no objectId: {alert.keys()}")
    ra = alert.get("ra", None)
    dec = alert.get("dec", None)
    if (ra is None) or (dec is None):
        raise MissingCoordinatesError(f"one of (ra, dec)=({ra}, {dec}) is None")
    return Target(objectId, ra=ra, dec=dec, t_ref=t_ref)


def target_from_fink_lightcurve(lightcurve: pd.DataFrame, t_ref: Time = None) -> Target:
    t_ref = t_ref or Time.now()

    if "objectId" not in lightcurve.columns:
        raise MissingObjectIdError("lightcurve has no 'objectId' column")
    objectId_options = lightcurve["objectId"].dropna().unique()
    if len(objectId_options) > 1:
        logger.warning(f"several objectId options:\n {objectId_options}")
    objectId = objectId_options[0]

    ra = None
    dec = None
    if "ra" in lightcurve.columns:
        ra_vals = lightcurve["ra"].dropna()
        ra = np.average(ra_vals)
    if "dec" in lightcurve.columns:
        dec_vals = lightcurve["dec"].dropna()
        dec = np.average(dec_vals)

    if (ra is None) or (dec is None):
        raise MissingCoordinatesError(f"missing ra/dec from {lightcurve.columns}")

    fink_data = TargetData(lightcurve=lightcurve)
    target = Target(objectId, ra=ra, dec=dec)
    target.target_data["fink"] = fink_data
    target.updated = True
    return target


def process_fink_query_results(raw_query_results: pd.DataFrame, comparison="ndethist"):
    query_results = raw_query_results.copy()
    query_results.sort_values(["objectId", comparison], inplace=True)
    query_results.drop_duplicates("objectId", keep="last", inplace=True)
    # query_results.set_index("objectId", verify_integrity=True, drop=False, inplace=True)
    return query_results


def empty_query_results(comparison="ndethist"):
    return pd.DataFrame(columns=["objectId", comparison])


def get_updates_from_query_results(
    existing_results: pd.DataFrame, updated_results: pd.DataFrame, comparison="ndethist"
):
    """
    Compare two dataframes of query_results. Get the rows which have been updated, or are new.
    """

    if updated_results is None or updated_results.empty:
        # Even if it's None, that's fine.
        return empty_query_results(comparison=comparison)

    updated_results = process_fink_query_results(updated_results, comparison=comparison)

    if existing_results is None or existing_results.empty:
        return updated_results

    existing_results = process_fink_query_results(
        existing_results, comparison=comparison
    )  # Makes a COPY.
    existing_results.set_index("objectId", verify_integrity=True)

    updates = []
    for objectId, updated_row in updated_results.iterrows():
        if objectId not in existing_results.index:
            updates.append(updated_row)
            continue
        existing_row = existing_results.loc[objectId]
        if updated_row[comparison] > updated_row[comparison]:
            updates.append(updated_row)

    if len(updates) == 0:
        return empty_query_results(comparison=comparison)
    updates_df = pd.DataFrame(updates)
    updates_df.sort_values("objectId", inplace=True)
    # updates_df.set_index("objectId", verify_integrity=True, inplace=True, drop=False)
    return updates_df


def target_from_fink_query_row(data: pd.Series, t_ref: Time = None):
    t_ref = t_ref or Time.now()

    if isinstance(data, dict):
        data = pd.Series(data)

    objectId = data.get("objectId", None)
    if objectId is None:
        raise MissingObjectIdError(f"query_row has no objectId in:\n    {data.keys()}")
    ra = data.get("ra", None)
    dec = data.get("dec", None)
    if (ra is None) or (dec is None):
        raise MissingCoordinatesError(f"one of (ra, dec)=({ra}, {dec}) is None")
    return Target(objectId, ra=ra, dec=dec, t_ref=t_ref)


class FinkQueryManager(BaseQueryManager):
    name = "fink"
    expected_fink_parameters = ("object_queries", "query_parameters", "kafka_config")
    default_query_parameters = {
        "object_query_interval": 1.0,  # How often to query for new data [day]
        "object_query_lookback": 20.0,  # How far back in time to check for alerts [day]
        "object_query_timestep": 0.1,  # Queries in short steps, not one big chunk. [day]
        "lightcurve_update_interval": 2.0,  # How often to update existing LCs.
        "max_failed_queries": 10,
        "max_total_query_time": 300,  # Max time allowed in each query stage. [sec]
    }
    default_kafka_parameters = {"n_alerts": 10, "timeout": 10.0}
    required_kafka_parameters = ("username", "group_id", "bootstrap.servers", "topics")
    alert_extra_keys = (
        # "timestamp", # doesn't exist anymore?!
        "cdsxmatch",
        "rf_snia_vs_nonia",
        "snn_snia_vs_nonia",
        "snn_sn_vs_all",
        "mulens",
        "roid",
        "nalerthist",
        "rf_kn_vs_nonkn",
    )

    def __init__(
        self,
        fink_config: dict,
        target_lookup: Dict[str, Target],
        parent_path=None,
        create_paths=True,
    ):
        self.fink_config = fink_config
        self.target_lookup = target_lookup

        utils.check_unexpected_config_keys(
            self.fink_config,
            self.expected_fink_parameters,
            name="fink_config",
        )

        self.kafka_config = self.get_kafka_parameters()
        if (self.kafka_config) and (fink_client is None):
            raise ValueError(
                "fink_client module not imported correctly!\n"
                "Either install with `\033[32;1mpython3 -m pip install fink_client\033[0m` "
                "(you may also need to install `\033[32;1mfastavro\033[0m`), "
                "or switch `use: False` in fink config, in config file."
            )

        object_queries = self.fink_config.get("object_queries", [])
        if isinstance(object_queries, str):
            object_queries = [object_queries]
        self.object_queries = object_queries

        self.query_parameters = self.default_query_parameters.copy()
        query_params = self.fink_config.get("query_parameters", {})
        self.query_parameters.update(query_params)
        utils.check_config_keys(
            self.query_parameters,
            self.default_query_parameters,
            name="fink_config.query_parameters",
        )

        self.query_results = {}

        self.process_paths(parent_path=parent_path, create_paths=create_paths)

    def get_kafka_parameters(self):
        kafka_parameters = self.default_kafka_parameters.copy()

        kafka_config = self.fink_config.get("kafka_config", None)
        if kafka_config is None:
            logger.warning(
                "no kafka_config: \033[33;1mwill not listen for alerts.\033[0m"
            )
            return None
        kafka_parameters.update(kafka_config)
        expected_keys = (
            tuple(self.default_kafka_parameters.keys()) + self.required_kafka_parameters
        )

        missing_kafka_keys = utils.check_missing_config_keys(
            kafka_parameters, self.required_kafka_parameters, name="fink.kafka_config"
        )
        if len(missing_kafka_keys) > 0:
            err_msg = f"kafka_config: provide {self.required_kafka_parameters}"
            raise BadKafkaConfigError(err_msg)

        topics = kafka_parameters.get("topics", None)
        if isinstance(topics, str):
            # kafka_parameters["topics"] should be a list!
            topics = [topics]
            kafka_parameters["topics"] = topics

        return kafka_parameters

    def listen_for_alerts(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if fink_client is None:
            logger.warning("no fink_client module: can't listen for alerts!")
            return

        if self.kafka_config is None:
            logger.info("can't listen for alerts: no kafka config!")
            return None

        new_alerts = []
        for topic in self.kafka_config["topics"]:
            with AlertConsumer([topic], self.kafka_config) as consumer:
                # TODO: change from `poll` to `consume`?
                for ii in range(self.kafka_config["n_alerts"]):
                    try:
                        alert_data = consumer.poll(timeout=self.kafka_config["timeout"])
                    except Exception as e:
                        print(e)
                        alert_data = (None, None, None)  # topic, alert, key
                    if any([x is None for x in alert_data]):
                        logger.info(f"break after {len(new_alerts)}")
                        break
                    new_alerts.append(alert_data)
                logger.info(f"received {len(new_alerts)} {topic} alerts")
        return new_alerts

    def process_alerts(
        self,
        alerts: List[Tuple[str, Dict, str]],
        save_alerts=True,
        save_cutouts=True,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()

        processed_alerts = []
        for topic, alert, key in alerts:
            candidate = alert["candidate"]
            objectId = alert["objectId"]

            candidate["topic"] = topic
            candidate["tag"] = "valid"
            candidate["mjd"] = Time(candidate["jd"], format="jd").mjd
            candidate["objectId"] = objectId
            candidate["candid"] = alert["candid"]

            valid_keys = [k for k in self.alert_extra_keys if k in alert]
            missing_keys = set(self.alert_extra_keys) - set(valid_keys)
            if len(missing_keys) > 0:
                logger.info(f"{objectId} missing keys: {missing_keys}")

            extra_data = {k: alert[k] for k in valid_keys}
            candidate.update(extra_data)
            processed_alerts.append(candidate)

            if save_alerts:
                alert_filepath = self.get_alert_file(objectId, candidate["candid"])
                with open(alert_filepath, "w") as f:
                    json.dump(candidate, f, indent=2)
            if save_cutouts:
                cutouts = {}
                for imtype in FinkQuery.imtypes:
                    data = alert.get("cutout" + imtype, {}).get("stampData", None)
                    if data is None:
                        continue
                    cutout = readstamp(data, return_type="array")
                    cutouts[imtype.lower()] = cutout
                if len(cutouts) > 0:
                    cutouts_filepath = self.get_cutouts_file(
                        objectId, candidate["candid"]
                    )
                    with open(cutouts_filepath, "wb+") as f:
                        pickle.dump(cutouts, f)
        return processed_alerts

    def new_targets_from_alerts(self, alerts: List[Dict], t_ref: None = Time):
        t_ref = t_ref or Time.now()

        added = []
        existing = []
        for alert in alerts:
            objectId = alert.get("objectId", None)
            if objectId is None:
                raise MissingObjectIdError("objectId is `None` in fink_alert")
            target = self.target_lookup.get(objectId, None)
            if target is not None:
                existing.append(objectId)  # Don't need to make a target.
                continue
            target = target_from_fink_alert(alert, t_ref=t_ref)
            self.target_lookup[objectId] = target
            added.append(objectId)
        if len(added) > 0 or len(existing) > 0:
            logger.info(f"{len(added)} targets init, {len(existing)} existing skipped")
        return added, existing

    def query_for_object_updates(self, comparison="ndethist", t_ref: Time = None):
        """
        Query for updates for each fink_class in fink_config["object_queries"]
        Compare with what we currently know about - if any are new, or updated,
        return these rows.

        Parameters
        ----------
        comparison [str]:
            column_name to compare to existing results to determine updates.
            default="ndethist"
        """
        t_ref = t_ref or Time.now()

        update_dfs = []

        step = self.query_parameters["object_query_timestep"]
        lookback = self.query_parameters["object_query_lookback"]

        for fink_class in self.object_queries:
            query_results_filepath = self.get_query_results_file(fink_class)
            query_results_file_age = calc_file_age(query_results_filepath, t_ref)
            # Age is np.inf if missing!

            if self.query_results.get(fink_class) is None:
                if query_results_filepath.exists():
                    query_results = pd.read_csv(query_results_filepath)
                    query_results = process_fink_query_results(query_results)
                    self.query_results[fink_class] = query_results
            if query_results_file_age < self.query_parameters["object_query_interval"]:
                continue  # Don't requery if the file is "recent"

            query_updates = FinkQuery.query_and_collate_latests(
                fink_class, step=step, lookback=lookback, t_ref=t_ref
            )
            if query_updates is None:
                updated_results = empty_query_results(comparison=comparison)
            else:
                updated_results = process_fink_query_results(
                    query_updates, comparison=comparison
                )
            existing_results = self.query_results.get(fink_class, None)
            if existing_results is None:
                existing_results = empty_query_results(comparison=comparison)
            updates = get_updates_from_query_results(existing_results, updated_results)

            if not updates.empty:
                update_dfs.append(updates)

            query_results = pd.concat(
                [existing_results, updated_results], ignore_index=True
            )
            query_results = process_fink_query_results(
                query_results, comparison=comparison
            )
            # Save even if empty, so don't re-query unnecessarily.
            query_results.to_csv(query_results_filepath, index=False)
            self.query_results[fink_class] = query_results

        if len(update_dfs) == 0:
            return empty_query_results(comparison=comparison)
        updated_objects = pd.concat(update_dfs, ignore_index=True)
        updated_objects = process_fink_query_results(updated_objects)
        return updated_objects

    def new_targets_from_object_updates(
        self, updates: pd.DataFrame, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        new_targets = []
        if updates is None or updates.empty:
            return new_targets

        for idx, row in updates.iterrows():
            objectId = row["objectId"]
            target = self.target_lookup.get(objectId, None)
            if target is not None:
                continue
            target = target_from_fink_query_row(row, t_ref=t_ref)
            if target is None:
                continue
            self.target_lookup[objectId] = target
            new_targets.append(objectId)
        logger.info(f"{len(new_targets)} targets added from updated queries")
        return new_targets

    def get_lightcurves_to_query(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        to_update = []
        for objectId, target in self.target_lookup.items():
            lightcurve_filepath = self.get_lightcurve_file(objectId)
            lightcurve_file_age = calc_file_age(
                lightcurve_filepath, t_ref, allow_missing=True
            )
            interval = self.query_parameters["lightcurve_update_interval"]
            if lightcurve_file_age > interval:
                to_update.append(objectId)
        return to_update

    def perform_lightcurve_queries(
        self,
        objectId_list: List[str],
        t_ref: Time = None,
        chunk_size=10,
        bulk_filepath=None,
    ):
        t_ref = t_ref or Time.now()

        if len(objectId_list) > 0:
            logger.info(f"attempt {len(objectId_list)} lightcurve queries")

        success = []
        failed = []
        failed_queries = 0

        bulk_results = []
        t_start = time.perf_counter()
        for objectId_chunk in utils.chunk_list(objectId_list, chunk_size=chunk_size):
            if failed_queries > self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed LC queries ({failed_queries}), stop for now"
                logger.info(msg)
                break
            t_now = time.perf_counter()
            if t_now - t_start > self.query_parameters["max_total_query_time"]:
                logger.info(f"querying time ({t_now - t_start:.1f}s) exceeded max")
                break

            objectId_str = ",".join(objectId for objectId in objectId_chunk)
            try:
                logger.info(f"query {len(objectId_chunk)} LCs...")
                chunk_result = FinkQuery.query_objects(
                    objectId=objectId_str, withupperlim=True, return_df=True
                )
                if not chunk_result.empty:
                    bulk_results.append(chunk_result)
            except:
                logger.info(f"fink query")
                failed.extend(objectId_chunk)
                failed_queries = failed_queries + 1
                continue

            if "objectId" not in chunk_result:
                # Still need this column to be able to split...
                chunk_result["objectId"] = "0"

            for objectId in objectId_chunk:
                raw_lightcurve = chunk_result[chunk_result["objectId"] == objectId]
                lightcurve_filepath = self.get_lightcurve_file(objectId)
                # raw_lightcurve = non_detections
                # raw_lightcurve = combine_fink_detections_non_detections(detections, non_detections)
                lightcurve = process_fink_lightcurve(raw_lightcurve)
                if lightcurve.empty:
                    logger.warning(f"\033[33m{objectId} lightcurve empty!\033[0m")
                lightcurve.to_csv(lightcurve_filepath, index=False)
                success.append(objectId)

        if len(bulk_results) > 0 and bulk_filepath is not None:
            bulk_lc = pd.concat(bulk_results, ignore_index=True)
            bulk_lc.to_csv(bulk_filepath, index=False)

        if len(success) > 0 or len(failed) > 0:
            t_end = time.perf_counter()
            logger.info(f"lightcurve queries in {t_end - t_start:.1f}s")
            logger.info(f"{len(success)} successful, {len(failed)} failed lc queries")
        return success, failed

    def load_target_lightcurves(
        self, objectId_list: List[str] = None, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        loaded = []
        missing = []
        skipped = []
        t_start = time.perf_counter()

        if objectId_list is None:
            objectId_list = list(self.target_lookup.keys())
            logger.info(f"try loading all {len(objectId_list)} lcs in target_lookup")
        else:
            logger.info(f"try loading {len(objectId_list)} lcs")

        for objectId in objectId_list:
            # TODO: not optimum to read every time... but not a bottleneck for now.
            lightcurve = self.load_single_lightcurve(objectId, t_ref=t_ref)
            if lightcurve is None:
                missing.append(objectId)
                logger.info(f"{objectId} lightcurve is bad")
                continue

            target = self.target_lookup.get(objectId, None)
            if target is None:
                logger.warning(f"load_lightcurve: {objectId} not in target_lookup!")
                missing.append(objectId)
                continue

            fink_data = target.get_target_data("fink")
            existing_lightcurve = fink_data.lightcurve
            if existing_lightcurve is not None:
                if len(lightcurve) <= len(existing_lightcurve):
                    skipped.append(objectId)
                    continue

            fink_data.add_lightcurve(lightcurve)
            loaded.append(objectId)
            target.updated = True
        t_end = time.perf_counter()

        N_loaded = len(loaded)
        N_missing = len(missing)
        N_skipped = len(skipped)
        t_load = t_end - t_start
        msg = f"loaded {N_loaded}, missing {N_missing} lightcurves in {t_load:.1f}s"
        logger.info(msg)
        return loaded, missing

    def load_single_lightcurve(self, objectId: str, t_ref=None):
        lightcurve_filepath = self.get_lightcurve_file(objectId)
        if not lightcurve_filepath.exists():
            logger.warning(f"{objectId} is missing lightcurve")
            return None

        try:
            lightcurve = pd.read_csv(lightcurve_filepath, dtype={"candid": "Int64"})
        except pd.errors.EmptyDataError as e:
            logger.warning(f"bad lightcurve file for {objectId}")
            return None
        if "mjd" not in lightcurve.columns:
            mjd_dat = Time(lightcurve["jd"], format="jd").mjd
            lightcurve.insert(1, "mjd", mjd_dat)

        # print(f"{objectId} {len(lightcurve)} rows", time.perf_counter() - t1)
        return lightcurve

    def load_missing_alerts(self, objectId: str):
        target = self.target_lookup.get(objectId, None)
        if target is None:
            raise ValueError(f"{objectId} has no entry in target_lookup")
        alert_dir = self.get_alert_dir(objectId)
        if not alert_dir.exists():
            return []
        alert_list = alert_dir.glob("*.json")

        fink_data = target.get_target_data("fink")
        if fink_data.lightcurve is None:
            existing_candids = []
        else:
            existing_candids = fink_data.lightcurve["candid"].values

        loaded_alerts = []
        for alert_filepath in alert_list:
            if int(alert_filepath.stem) in existing_candids:
                continue
            with open(alert_filepath, "r") as f:
                alert = json.load(f)
                if "mjd" not in alert:
                    alert["mjd"] = Time(alert["jd"], format="jd").mjd
                loaded_alerts.append(alert)
        return loaded_alerts

    def integrate_alerts(self, save_lightcurve=True):
        integrated_alerts = []
        for objectId, target in self.target_lookup.items():
            # Loop through all targets - not a bottleneck for now.
            loaded_alerts = self.load_missing_alerts(objectId)
            if len(loaded_alerts) == 0:
                continue

            alert_df = pd.DataFrame(loaded_alerts)
            if "mjd" not in alert_df.columns:
                alert_df["mjd"] = Time(alert_df["jd"], format="jd").mjd
            fink_data = target.get_target_data("fink")
            fink_data.integrate_lightcurve_updates(alert_df, column="candid")

            if save_lightcurve:
                lightcurve_filepath = self.get_lightcurve_file(objectId)
                lightcurve = fink_data.lightcurve
                lightcurve.to_csv(lightcurve_filepath, index=False)
            integrated_alerts.append(objectId)

            target.updated = True
        if len(integrated_alerts):
            logger.info(
                f"integrate alerts into LC for {len(integrated_alerts)} targets"
            )
        return integrated_alerts

    def load_cutouts(self):
        loaded_cutouts = []
        for objectId, target in self.target_lookup.items():
            fink_data = target.get_target_data("fink")
            cutouts_are_None = [im is None for k, im in fink_data.cutouts.items()]
            for candid in fink_data.detections["candid"][::-1]:
                cutouts_filepath = self.get_cutouts_file(objectId, candid)
                if cutouts_filepath.exists():
                    cutouts_candid = fink_data.meta.get("cutouts_candid", None)
                    if cutouts_candid == candid:
                        # If the existing cutouts are from this candid,
                        # they must already be the latest (as we're searching in rev.)
                        break
                    with open(cutouts_filepath, "rb") as f:
                        try:
                            cutouts = pickle.load(f)
                        except EOFError as e:
                            msg = f"{objectId}: {candid} bad cutouts - deleting file {cutouts_filepath}"
                            logger.error(msg)
                            cutouts_filepath.unlink()
                            continue
                    fink_data.cutouts = cutouts
                    fink_data.meta["cutouts_candid"] = candid
                    loaded_cutouts.append(objectId)
                    break  # Leave candid loop, back to outer target loop
        logger.info(f"{len(loaded_cutouts)} cutouts loaded")

    def apply_messenger_updates(self, alerts):
        for alert in alerts:
            objectId = alert["objectId"]
            target = self.target_lookup.get(objectId, None)
            if target is None:
                continue
            target.send_updates = True
            topic_str = alert["topic"]
            alert_jd = alert.get("jd")
            alert_time = Time(alert_jd, format="jd")
            timestamp = alert_time.strftime("%Y-%m-%d %H:%M:%S")
            alert_text = (
                f"FINK alert from {topic_str}\n"
                f"     broadcast at mjd={alert_time.mjd:.5f}={timestamp}\n"
            )
            target.update_messages.append(alert_text)

    def perform_all_tasks(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        alerts = self.listen_for_alerts()
        processed_alerts = self.process_alerts(alerts, t_ref=t_ref)
        self.new_targets_from_alerts(processed_alerts, t_ref=t_ref)

        new_targets = list(set([alert["objectId"] for alert in processed_alerts]))
        success, failed = self.perform_lightcurve_queries(new_targets)

        # updated_objects = self.query_for_object_updates(t_ref=t_ref)
        # success, failed = self.perform_lightcurve_queries(updated_objects["objectId"].values)

        lcs_to_query = self.get_lightcurves_to_query(t_ref=t_ref)
        logger.info(f"update {len(lcs_to_query)} old lightcurves")
        success, failed = self.perform_lightcurve_queries(lcs_to_query)

        loaded_lcs, missing_lcs = self.load_target_lightcurves(t_ref=t_ref)

        self.integrate_alerts()

        self.load_cutouts()

        self.apply_messenger_updates(processed_alerts)


class FinkQueryError(Exception):
    pass


class FinkQuery:
    """
    See https://fink-portal.org/api
    Really, using a class as a namespace is not pythonic...

    can do either:
    >>> fq = FinkQuery()
    >>> lightcurve = fq.query_objects("ZTF23example")

    or directly:
    >>> lc = FinkQuery.query_objects("ZTF23example")
    """

    api_url = "https://fink-portal.org/api/v1"
    imtypes = ("Science", "Template", "Difference")
    longint_cols = ("candid",)

    def __init__(self):
        pass

    @classmethod
    def fix_dict_keys(cls, data: Dict):
        key_lookup = {k: k.split(":")[-1] for k in data.keys()}
        for old_key, new_key in key_lookup.items():
            # Not dangerous, as not iterating over the dict we're updating.
            data[new_key] = data.pop(old_key)

    @classmethod
    def process_data(cls, data, fix_keys=True, return_df=True, df_kwargs=None):
        if fix_keys:
            for row in data:
                cls.fix_dict_keys(row)
        if not return_df:
            return data
        return pd.DataFrame(data)

    @classmethod
    def process_response(cls, res, fix_keys=True, return_df=True):
        if res.status_code in [404, 500, 504]:
            logger.error("\033[31;1FinkQuery: error rasied\033[0m")
            if res.elapsed.total_seconds() > 58.0:
                logger.error("likely a timeout error")
            raise FinkQueryError(res.content.decode())
        data = json.loads(res.content)
        return cls.process_data(data, fix_keys=fix_keys, return_df=return_df)

    @classmethod
    def do_post(cls, service, process=True, fix_keys=True, return_df=True, **kwargs):
        kwargs = cls.process_kwargs(**kwargs)
        response = requests.post(f"{cls.api_url}/{service}", json=kwargs)
        if response.status_code != 200:
            msg = f"query for {service} returned status {response.status_code}"
            logger.warning(msg)
        if process:
            return cls.process_response(
                response, fix_keys=fix_keys, return_df=return_df
            )
        return response

    @classmethod
    def process_kwargs(cls, **kwargs):
        return {k.rstrip("_"): v for k, v in kwargs.items()}

    @classmethod
    def query_objects(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post("objects", fix_keys=fix_keys, return_df=return_df, **kwargs)

    @classmethod
    def query_explorer(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post("explorer", fix_keys=fix_keys, return_df=return_df, **kwargs)

    @classmethod
    def query_latests(cls, fix_keys=True, return_df=True, **kwargs):
        return cls.do_post("latests", fix_keys=fix_keys, return_df=return_df, **kwargs)

    @classmethod
    def query_cutouts(cls, **kwargs):
        if kwargs.get("kind", None) not in cls.imtypes:
            raise ValueError(f"provide `kind` as one of {cls.imtypes}")
        return cls.do_post("cutouts", fix_keys=False, return_df=False, **kwargs)

    @classmethod
    def query_and_collate_latests(
        cls, fink_class, n=1000, lookback=1.0, query_timestep=0.1, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        step = query_timestep
        lookback_grid = t_ref + np.arange(-lookback, step, step) * u.day

        update_dfs = []
        for start_jd, stop_jd in zip(lookback_grid[:-1], lookback_grid[1:]):
            query_data = dict(
                class_=fink_class, n=n, startdate=start_jd.iso, stopdate=stop_jd.iso
            )
            try:
                updates = FinkQuery.query_latests(**query_data)
            except Exception as e:
                print(e)
                logger.warning(f"{fink_class} {start_jd:.1f}<jd<{stop_jd:.1f} failed")
                updates = None

            if updates is None:
                return None

            n = query_data["n"]
            if len(updates) == n:
                msg = f"`{fink_class}` query returned max updates={n}.\nChoose a shorter `query_timestep`!"
                logger.warning(msg)
            if len(updates) > 0:
                update_dfs.append(updates)

        if len(update_dfs) > 0:
            return pd.concat(update_dfs)
        return pd.DataFrame([], columns=["objectId"])


def readstamp(stamp: str, return_type="array") -> np.array:
    """
    copied and pasted directly from
    https://github.com/astrolabsoftware/fink-science-portal/blob/master/apps/utils.py#L216 ...
    Read the stamp data inside an alert.

    Parameters
    ----------
    alert: dictionary
        dictionary containing alert data
    field: string
        Name of the stamps: cutoutScience, cutoutTemplate, cutoutDifference
    return_type: str
        Data block of HDU 0 (`array`) or original FITS uncompressed (`FITS`) as file-object.
        Default is `array`.

    Returns
    ----------
    data: np.array
        2D array containing image data (`array`) or FITS file uncompressed as file-object (`FITS`)
    """
    with gzip.open(io.BytesIO(stamp), "rb") as f:
        with fits.open(io.BytesIO(f.read())) as hdul:
            if return_type == "array":
                data = hdul[0].data
            elif return_type == "FITS":
                data = io.BytesIO()
                hdul.writeto(data)
                data.seek(0)
    return data
