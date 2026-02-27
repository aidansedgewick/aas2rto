from __future__ import annotations

import abc
import gzip
import io
import json
import pickle
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import NewType

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

from aas2rto.exc import (
    BadKafkaConfigError,
    MissingCoordinatesError,
    MissingTargetIdError,
)
from aas2rto.query_managers.base import LightcurveQueryManager, KafkaQueryManager
from aas2rto.query_managers.fink.fink_portal_client import FinkBasePortalClient
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup
from aas2rto.utils import (
    calc_file_age,
    check_safe_to_query,
    check_missing_config_keys,
    check_unexpected_config_keys,
    chunk_list,
)

try:
    import fink_client
except ModuleNotFoundError as e:
    fink_client = None

if fink_client is not None:
    from fink_client.avro_utils import write_alert, _get_alert_schema, AlertReader
    from fink_client.consumer import AlertConsumer

logger = getLogger(__name__.split(".")[-1])


FinkAlert = NewType("FinkAlert", tuple[str, dict, str])


def updates_from_classifier_queries(
    existing_results: pd.DataFrame,
    new_results: pd.DataFrame,
    id_key: str = None,
    date_key: str = "lastdate",
) -> pd.DataFrame:
    if id_key is None:
        raise ValueError("must provide 'id_key=<key>'")

    existing_results = existing_results.copy()
    existing_results.sort_values([id_key, date_key], inplace=True, ignore_index=True)
    existing_results.drop_duplicates(id_key, keep="last", inplace=True)
    existing_results.set_index(id_key, inplace=True, drop=False, verify_integrity=True)

    new_results = new_results.copy()
    new_results.sort_values([id_key, date_key], inplace=True, ignore_index=True)
    new_results.drop_duplicates(id_key, keep="last", inplace=True)
    new_results.set_index(id_key, inplace=True, drop=False, verify_integrity=True)

    updated_rows = []
    for target_id, new_row in new_results.iterrows():
        if target_id not in existing_results.index:
            updated_rows.append(new_row)
        else:
            existing_row = existing_results.loc[target_id]
            if new_row[date_key] > existing_row[date_key]:
                updated_rows.append(new_row)

    if len(updated_rows) == 0:
        return pd.DataFrame(columns=new_results.columns)
    return pd.DataFrame(updated_rows)


# re-inherit from abc.ABC to indicate that FinkBaseQM should be subclassed...
class FinkBaseQueryManager(LightcurveQueryManager, KafkaQueryManager, abc.ABC):

    @property
    @abc.abstractmethod
    def id_resolving_order(self) -> tuple[str]:
        """The order that the QM should prefer IDs for queries to the FINK portal.

        eg. ("lsst", "fink_lsst", "tns") would first check in a target's alt_ids
        for a 'lsst' ID, then a 'fink_lsst' ID, and finally a 'tns' ID."""

    @property
    @abc.abstractmethod
    def target_id_key(self) -> str:
        """The key used as target_id from FINK alerts/lightcurves.

        eg. 'objectId' for ZTF, 'diaObjectId' for LSST"""

    @property
    @abc.abstractmethod
    def alert_id_key(self) -> str:
        """The key used as alert_id from FINK alerts/lightcurves

        eg. 'candid' for ZTF, 'diaSourceId' for LSST"""

    @property
    @abc.abstractmethod
    def portal_client_class(self) -> type[FinkBasePortalClient]:
        """The class -- NOT instance -- of the PortalClient which should be used

        See aas2rto.query_managers.fink.fink_portal_client
        """

    ##===== Methods for subclasses to define here =====##

    @abc.abstractmethod
    def process_single_alert(self, alert_data: FinkAlert) -> dict:
        """process_single_alert(self, alert_data: tuple[str, dict, str]):

        Returns a dict (the processed alert), or None (if ignoring the alert).
        The argument alert_data [tuple] is: topic [str], alert[dict], key [str].
        You should also save any alert and cutouts in this function."""

    @abc.abstractmethod
    def apply_updates_from_alert(self, processed_alert: dict, t_ref: Time = None):
        """apply_updates_from_alert(self, processed_alert: dict, t_ref: Time):

        Accept a processed alert (from your process_single_alert() method),
        which you can use to  mark any targets as updated, and add messages
        that will be sent out by eg. slack or telegram."""

    @abc.abstractmethod
    def new_target_from_alert(
        self, processed_alert: dict, t_ref: Time = None
    ) -> Target:
        """new_target_from_alert(self, alert: dict, t_ref: Time)

        Create a single `Target` from a processed alert (dict)"""

    @abc.abstractmethod
    def new_target_from_record(self, query_record: dict) -> Target:
        """new_target_from_record(self, alert: dict, t_ref: Time):\n"

        Create a single `Target` from dict - the result of a fink 'latests' query,
        with whatever attributes this query result may have."""

    @abc.abstractmethod
    def process_fink_lightcurve(self, unprocessed_lc: pd.DataFrame) -> pd.DataFrame:
        """process_fink_lc(self, unprocessed_lc: pd.DataFrame):"""

    @abc.abstractmethod
    def load_missing_alerts_for_target(self, fink_id: str) -> bool:
        """load_missing_alerts_for_target(self, target_id: str):"""

    @abc.abstractmethod
    def load_cutouts_for_alert(self, fink_id: str, alert_id: int) -> dict | None:
        """load_cutouts(self, fink_id: int, alert_id: int):

        Your function should accept fink_id [str] and alert_id [int].
        If the cutouts for this id exist, return a dict.
        If they don't exist, return None."""

    ##===== Generic methods start here =====##

    DEFAULT_TASKS = ("alerts", "queries", "lightcurves", "cutouts", "stale_files")
    REQUIRED_KAFKA_PARAMS = (
        "username",
        "group.id",
        "bootstrap.servers",
        "topics",
        "survey",
    )
    REQUIRED_DIRECTORIES = ("alerts", "cutouts", "lightcurves", "query_results")

    default_config = {
        "max_failed_queries": 10,
        "max_query_time": 300.0,
        "lightcurve_update_interval": 2.0,
        "lightcurve_chunk_size": 25,
        "lightcurve_columns": None,
        "kafka": None,
        # "portal_credentials": None, # Placeholder - currently no creds for portal...
        "n_alerts": 10,
        "alert_timeout": 10.0,
        "fink_classes": None,
        "query_interval": 1.0,
        "query_lookback": 20.0,
        "stale_file_age": 60.0,
        "tasks": DEFAULT_TASKS,
    }
    config_comments = {
        "max_failed_queries": "Quit after how many failed queries?",
        "max_query_time": "Quit after how long spent making queries?",
        "lightcurve_update_interval": "How often [DAYS] to check for updated LCs?",
        "lightcurve_chunk_size": "How many LCs to query at once? [MAX=25]",
        "kafka": f"dict: must provide {REQUIRED_KAFKA_PARAMS}",
        "n_alerts": "How many alerts to listen for [PER KAFKA TOPIC] per loop?",
        "alert_timeout": "How long to listen for new alerts?",
        "fink_classes": "To query with 'latests' - try eg. FinkZTFQuery.classes()",
        "query_interval": "How long [DAYS] to wait until new 'latests' query?",
        "query_lookback": "How far back [DAYS] to query for",
        "stale_file_age": "How old [DAYS] should files be deleted?",
        "tasks": f"which subset to perform, from: {DEFAULT_TASKS}",
    }

    ##===== initialisation here =====##

    def __init__(self, config: dict, target_lookup: TargetLookup, parent_path: Path):
        # Config steps
        self.config = self.default_config.copy()
        self.config.update(config)

        check_unexpected_config_keys(
            self.config, self.default_config, name=f"{self.name}_config", raise_exc=True
        )
        self.process_kafka_config()

        # Set the TargetLookup
        self.target_lookup = target_lookup

        # Initialize the portal client - maybe credentials needed in future...
        self.portal_client: FinkBasePortalClient = self.portal_client_class()

        # Process the paths
        self.process_paths(
            parent_path=parent_path, directories=self.REQUIRED_DIRECTORIES
        )

        # Are there any ZTF tasks we want to skip?
        self.tasks = self.config.get("tasks")
        check_unexpected_config_keys(
            self.tasks,
            self.DEFAULT_TASKS,
            name=f"{self.name}.config.tasks",
            raise_exc=True,
        )

    def process_kafka_config(self):
        self.kafka_config = self.config.get("kafka", None)
        if self.kafka_config is None:
            logger.info(f"No 'kafka' in {self.name} config: will not listen for alerts")
            return
        else:
            missing_keys = check_missing_config_keys(
                self.kafka_config,
                self.REQUIRED_KAFKA_PARAMS,
                name=f"{self.name}.kafka",
                raise_exc=True,
                exc_class=BadKafkaConfigError,
            )

        topics = self.kafka_config["topics"]
        if isinstance(topics, str):
            self.kafka_config["topics"] = [topics]
            logger.info("converted kafka topics 'str' to 'list'")

        if fink_client is None:
            msg = (
                f"Error importing 'fink_client'.\n"
                f"    Fix broken module with \033[32 python3 -m pip install fink_client\033[0m\n"
                f"    or remove `kafka` from {self.name} config"
            )
            raise ModuleNotFoundError(msg)

    ##===== Path methods here =====##

    def get_alert_directory(self, fink_id: str, mkdir: bool = True):
        alert_dir = self.paths_lookup["alerts"] / str(fink_id)
        if mkdir:
            alert_dir.mkdir(exist_ok=True, parents=True)
        return alert_dir

    def get_alert_filepath(
        self, fink_id: str, alert_id: int, mkdir: bool = True, fmt="json"
    ):
        alert_dir = self.get_alert_directory(fink_id, mkdir=mkdir)
        return alert_dir / f"{alert_id}.{fmt}"

    def get_cutouts_directory(self, fink_id: str, mkdir: bool = True):
        cutouts_dir = self.paths_lookup["cutouts"] / str(fink_id)
        if mkdir:
            cutouts_dir.mkdir(exist_ok=True, parents=True)
        return cutouts_dir

    def get_cutouts_filepath(
        self, fink_id: str, alert_id: int, mkdir: bool = True, fmt="pkl"
    ):
        cutouts_dir = self.get_cutouts_directory(fink_id, mkdir=mkdir)
        return cutouts_dir / f"{alert_id}.{fmt}"

    def get_lightcurve_filepath(self, fink_id: str, fmt="csv"):
        return self.paths_lookup["lightcurves"] / f"{fink_id}.{fmt}"

    def get_query_results_filepath(self, fink_class: str, fmt="csv"):
        file_stem = fink_class.replace(" ", "_")
        return self.paths_lookup["query_results"] / f"{file_stem}.{fmt}"

    ##===== Generic methods for FINK here =====##

    def listen_for_alerts(self, t_ref: Time = None) -> list[dict]:
        if self.kafka_config is None:
            logger.debug("no kafka config - skip alerts")
            return []

        n_alerts = self.config["n_alerts"]
        timeout = self.config["alert_timeout"]
        survey = self.kafka_config["survey"]

        alerts = []
        for topic in self.kafka_config["topics"]:
            topic_alerts = []
            with AlertConsumer([topic], self.kafka_config, survey) as consumer:
                logger.info(f"listen for up to {n_alerts} from {topic}")
                for ii in range(n_alerts):
                    try:
                        alert_data: FinkAlert = consumer.poll(timeout=timeout)
                    except Exception as e:
                        logger.warning(f"In poll:\n    {type(e)}: {e}")
                        alert_data = (None, None, None)  # match return from no-alert
                    if alert_data[0] is None:
                        logger.info(f"break after {len(topic_alerts)}")
                        break
                    processed_alert = self.process_single_alert(alert_data)
                    if processed_alert is not None:
                        topic_alerts.append(processed_alert)
                logger.info(f"{len(topic_alerts)} alerts from {topic}")
            alerts.extend(topic_alerts)
        logger.info(f"{len(alerts)} alerts from all topics")
        return alerts

    def update_info_messages(self, processed_alerts: list[dict], t_ref: Time = None):
        t_ref = t_ref or Time.now()

        for alert in processed_alerts:
            self.apply_updates_from_alert(alert, t_ref=t_ref)

    def load_single_lightcurve(self, target_id: str, t_ref: Time = None):
        target = self.target_lookup.get(target_id, None)
        if target is None:
            return None

        fink_id = self.get_relevant_id_from_target(target)

        lightcurve_filepath = self.get_lightcurve_filepath(fink_id)
        if lightcurve_filepath.exists():
            try:
                lightcurve = pd.read_csv(lightcurve_filepath)
            except Exception as e:
                msg = f"During read lc for {fink_id}:\n    {type(e).__name__}: {e}"
                logger.warning(msg)
                lightcurve = None
        else:
            lightcurve = None
        return lightcurve

    def fink_classifier_queries(self, t_ref: Time = None) -> list[dict]:
        t_ref = t_ref or Time.now()

        fink_classes = self.config["fink_classes"]
        if fink_classes is None:
            return []

        if isinstance(fink_classes, str):
            fink_classes = [fink_classes]

        query_interval = self.config["query_interval"]

        results_list = []
        for fink_class in fink_classes:
            ##== Decide if we need to make the query
            query_results_filepath = self.get_query_results_filepath(fink_class)
            file_age = calc_file_age(query_results_filepath, t_ref)
            if file_age < query_interval:
                logger.info(f"{fink_class} has results {file_age:.1f}d old - skip")
                continue
            logger.info(f"query 'latests' for class '{fink_class}'")

            ##== Are there any existing results? Decides how far back we need to query
            if query_results_filepath.exists():
                existing_results = pd.read_csv(query_results_filepath)
                lookback = file_age
            else:
                empty_df_keys = [self.target_id_key, "lastdate"]
                existing_results = pd.DataFrame(columns=empty_df_keys)
                lookback = self.config["query_lookback"]

            ##== Perform the query
            t_start = t_ref - lookback * u.day
            new_results = self.portal_client.query_classifiers(
                t_start=t_start, class_=fink_class, return_type="pandas"
            )  # trailing "_"  from 'class_' is stripped...
            if len(new_results) == 0:
                existing_results.to_csv(query_results_filepath, index=False)
                continue  # write empty file anyway so we don't re-query in 5 minutes!

            ##== Combine with any existing results
            if existing_results.empty:
                combined_results = new_results
            else:
                combined_results = pd.concat([existing_results, new_results])

            ##== Only keep the latest row for each target in the combined df.
            key_cols = [self.target_id_key, "lastdate"]
            combined_results.sort_values(key_cols, inplace=True, ignore_index=True)
            combined_results.drop_duplicates(
                self.target_id_key, keep="last", inplace=True
            )
            combined_results.to_csv(query_results_filepath, index=False)

            ##== Check what's different from the exisi
            classifier_updates = updates_from_classifier_queries(
                existing_results, new_results, id_key=self.target_id_key
            )
            classifier_updates.loc[:, "fink_class"] = fink_class
            if len(classifier_updates) > 0:
                results_list.append(classifier_updates)
                logger.info(f"{len(classifier_updates)} updates for '{fink_class}'")

        if len(results_list) > 0:
            results = pd.concat(results_list)
            results.drop_duplicates(self.target_id_key, keep="first", inplace=True)
            logger.info(f"{len(results)} updates in from all fink_classes")
            return results.to_dict("records")
        return []

    def add_targets_from_query_records(self, query_records: list[dict]):
        targets_added = []
        existing_skipped = []
        for record in query_records:
            target_id = record[self.target_id_key]

            ##== Do we already know about it?
            existing_target = self.target_lookup.get(target_id, None)
            if existing_target is not None:
                existing_skipped.append(target_id)
                continue

            target = self.new_target_from_record(record)
            self.target_lookup.add_target(target)
            targets_added.append(target.target_id)

        N_added = len(targets_added)
        N_skipped = len(existing_skipped)
        logger.info(f"added {N_added} new targets (skipped {N_skipped} existing)")
        return targets_added

    def query_lightcurves(
        self,
        fink_id_list: list[str] = None,
        t_ref: Time = None,
        bulk_filepath: Path = None,
    ):
        """
        Returns
        -------
        success, missing, failed: int
        """

        t_ref = t_ref or Time.now()

        if fink_id_list is None:
            # Can't just query all in target_lookup, some may not have valid fink_id...
            fink_id_list = self.select_lightcurves_to_query(
                t_ref=t_ref
            )  # super() method
        logger.info(f"attempt to query {len(fink_id_list)} LCs")

        chunk_size = self.config["lightcurve_chunk_size"]

        success = []
        missing = []
        failed = []
        chunk_results = []
        failed_queries = 0

        t_start = time.perf_counter()
        for ii, fink_id_chunk in enumerate(
            chunk_list(fink_id_list, chunk_size=chunk_size)
        ):
            if not check_safe_to_query(
                t_start=t_start,
                failed_queries=failed_queries,
                max_query_time=self.config["max_query_time"],
                max_failed_queries=self.config["max_failed_queries"],
            ):
                logger.warning("Stop LC queries for now")
                break

            chunk_str = ",".join([str(fink_id) for fink_id in fink_id_chunk])
            logger.info(f"start LC query chunk {ii+1} ({len(fink_id_chunk)} LCs)")
            payload = {
                self.target_id_key: chunk_str,
                "withupperlim": True,
            }
            lightcurve_columns = self.config["lightcurve_columns"]
            if lightcurve_columns is not None:
                payload["columns"] = lightcurve_columns

            t1 = time.perf_counter()
            try:
                result: pd.DataFrame = self.portal_client.query_lightcurve(
                    return_type="pandas", **payload
                )
                chunk_results.append(result)
            except Exception as e:
                msg = f"LC query chunk {ii+1} failed:\n    {type(e).__name__}: {e}"
                logger.error(msg)
                failed_queries = failed_queries + 1
                failed.extend(fink_id_chunk)
                continue
            t2 = time.perf_counter()  # continue doesn't matter - t1 re-set every loop
            logger.info(f"LC query chunk {ii+1} in {t2-t1:.1f}s")

            if self.target_id_key not in result:
                msg = (
                    f"\033[33;1mkey '{self.target_id_key}' not in returned columns\033[0m:"
                    f"\n {result.columns.values}"
                )
                logger.warning(msg)
                result[self.target_id_key] = "0"  # Still need to to split on target_id

            result[self.target_id_key] = result[self.target_id_key].astype(str)

            for fink_id in fink_id_chunk:

                id_mask = result[self.target_id_key] == fink_id
                unprocessed_lc = result[id_mask]
                processed_lc = self.process_fink_lightcurve(unprocessed_lc)

                if processed_lc.empty:
                    missing.append(fink_id)
                else:
                    success.append(fink_id)

                lightcurve_filepath = self.get_lightcurve_filepath(fink_id)
                processed_lc.to_csv(lightcurve_filepath, index=False)

        if bulk_filepath is not None:
            if len(chunk_results) > 0:
                bulk_lc = pd.concat(chunk_results)
            else:
                cols = [self.target_id_key, self.alert_id_key, "mjd"]
                bulk_lc = pd.DataFrame(columns=cols)
            bulk_lc.to_csv(bulk_filepath, index=False)
            logger.info(f"written bulk results to {bulk_filepath}")

        logger.info(f"{len(success)} LCs queried ok, {len(missing)} returned no LC")
        if len(failed) > 0:
            logger.warning(f"{len(failed)} LCs were in {failed_queries} failed queries")
        return success, missing, failed

    def integrate_alerts(self):
        targets_modified = []
        for target_id, target in self.target_lookup.items():
            fink_id = self.get_relevant_id_from_target(target)
            if fink_id is None:
                continue
            alerts_loaded = self.load_missing_alerts_for_target(fink_id)
            # this load() must to be defined per-survey, as alert schema is different
            if alerts_loaded:
                targets_modified.append(target_id)
        logger.info(f"loaded missing alerts for {len(targets_modified)} targets")
        return targets_modified

    def query_latest_cutouts(self):

        success = []
        missing = []
        failed = []
        failed_queries = 0

        t_start = time.perf_counter()
        for target_id, target in self.target_lookup.items():

            if not check_safe_to_query(
                t_start=t_start,
                failed_queries=failed_queries,
                max_query_time=self.config["max_query_time"],
                max_failed_queries=self.config["max_failed_queries"],
            ):
                logger.warning("Stop cutouts queries for now")
                break

            fink_id = self.get_relevant_id_from_target(target)
            if fink_id is None:
                continue  # There is no relevant ID to ask FINK for cutouts.
            fink_data = target.target_data.get(self.name, None)
            if fink_data is None:
                continue  # The target has no fink data.
            if fink_data.detections is None:
                continue  # The target has no detections
            latest_alert_id = fink_data.detections[self.alert_id_key].iloc[-1]
            latest_cutout_filepath = self.get_cutouts_filepath(fink_id, latest_alert_id)

            if not latest_cutout_filepath.exists():
                payload = {
                    self.target_id_key: fink_id,
                    self.alert_id_key: latest_alert_id,
                }
                try:
                    cutouts = self.portal_client.cutouts(**payload)
                except Exception as e:
                    failed_queries = failed_queries + 1
                    continue

                with open(latest_cutout_filepath, "rb") as f:
                    pickle.dump(cutouts, f)

    def load_cutouts(self):
        loaded_cutouts = []
        missing_cutouts = []
        skipped_reload = []
        for target_id, target in self.target_lookup.items():
            fink_id = self.get_relevant_id_from_target(target)
            if fink_id is None:
                continue
            fink_data = target.target_data.get(self.name, None)
            if fink_data is None:
                continue
            if fink_data.detections is None:
                continue
            if self.alert_id_key not in fink_data.detections:
                highlight = f"'{self.alert_id_key}' not in {self.name}_data.lightcurve"
                msg = f"{target_id}: for {self.name} \033[33;1m{highlight}\033[0m"
                logger.warning(msg)
                continue

            current_cutouts_id = fink_data.meta.get("cutouts_alert_id", -1)
            for alert_id in fink_data.lightcurve[self.alert_id_key].values[::-1]:
                # Step through alert_id from most recent to oldest.
                if alert_id < 0:
                    continue
                if alert_id == current_cutouts_id:
                    skipped_reload.append(target_id)
                    break  # We've already loaded this one.
                cutouts = self.load_cutouts_for_alert(fink_id, alert_id)
                if cutouts is not None:
                    fink_data.cutouts = cutouts
                    fink_data.meta["cutouts_alert_id"] = alert_id
                    loaded_cutouts.append(target_id)
                    break
            else:
                missing_cutouts.append(target_id)  # executed if no break statement met!

        N_loaded = len(loaded_cutouts)
        N_skipped = len(skipped_reload)
        logger.info(f"loaded cutouts for {N_loaded} targets (skipped {N_skipped})")
        if len(missing_cutouts) > 0:
            logger.info(f"missing cutouts for {len(missing_cutouts)}!!")
        return loaded_cutouts, missing_cutouts, skipped_reload

    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        # Are we in startup?
        if iteration == 0:
            # If yes, we don't want to do anything other than load existing data.
            self.load_target_lightcurves()  # Use per-QM load_single_lightcurve()
            self.integrate_alerts()
            self.load_cutouts()
            return

        # Are there any new alerts?
        if "alerts" in self.tasks:
            processed_alerts = self.listen_for_alerts()
            self.add_targets_from_alerts(processed_alerts, t_ref=t_ref)
            self.update_info_messages(processed_alerts, t_ref=t_ref)

        # Any targets flagged as interesting *not* from alerts?
        if "queries" in self.tasks:
            updated_query_records = self.fink_classifier_queries()
            self.add_targets_from_query_records(updated_query_records)

        # Who needs a lightcurve updating?
        if "lightcurves" in self.tasks:
            success, missing, failed = self.query_lightcurves()
            self.load_target_lightcurves()  # This is defined in BaseQueryManager!

        # Are there any fresh alerts not included in the lightcurves?
        if "alerts" in self.tasks:
            self.integrate_alerts()

        # Load any cutouts
        if "cutouts" in self.tasks:
            self.query_latest_cutouts()
            self.load_cutouts()

        # Clear stale files
        self.clear_stale_files(self.config["stale_file_age"])
        return


def readstamp(stamp: str, return_type="array", gzipped=True) -> np.array:
    """Read the stamp data inside an alert.
    Copied directly from Fink's utils:
    https://github.com/astrolabsoftware/fink-science-portal/blob/master/apps/utils.py#L216 ...

    Parameters
    ----------
    stamp: str
        String containing binary data for the stamp
    return_type: str
        Data block of HDU 0 (`array`) or original FITS uncompressed (`FITS`) as file-object.
        Default is `array`.

    Returns
    -------
    data: np.array
        2D array containing image data (`array`) or FITS file uncompressed as file-object (`FITS`)
    """

    def extract_stamp(fitsdata):
        with fits.open(fitsdata, ignore_missing_simple=True) as hdul:
            if return_type == "array":
                data = hdul[0].data
            elif return_type == "FITS":
                data = io.BytesIO()
                hdul.writeto(data)
                data.seek(0)
        return data

    if not isinstance(stamp, io.BytesIO):
        stamp = io.BytesIO(stamp)

    if gzipped:
        with gzip.open(stamp, "rb") as f:
            return extract_stamp(io.BytesIO(f.read()))
    else:
        return extract_stamp(stamp)
