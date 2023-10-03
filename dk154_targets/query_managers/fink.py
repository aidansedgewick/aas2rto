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
from dk154_targets.utils import calc_file_age

from dk154_targets.query_managers.base import BaseQueryManager
from dk154_targets.query_managers.exc import (
    BadKafkaConfigError,
    MissingObjectIdError,
    MissingCoordinateColumnsError,
)

from dk154_targets import paths

logger = getLogger(__name__.split(".")[-1])


def process_fink_lightcurve(detections: pd.DataFrame, non_detections: pd.DataFrame):
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
    if "tag" in non_detections:
        if "valid" in np.unique(non_detections["tag"]):
            ldet = len(detections)
            lvalid = len(non_detections.query("tag=='valid'"))
            if not ldet == lvalid:
                msg = f"mismatch : detections ({ldet}) != 'valid' non_detections ({lvalid})"
                raise ValueError(msg)
    else:
        if (not non_detections.empty) and (not detections.empty):
            raise ValueError(
                "non_detections has no 'tag' (no detections), but detections is not empty"
            )

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
                print(non_detections["candid"])
                raise ValueError("not all non-detections have Null `candid`")

    non_detections["candid"] = 0

    lightcurve = pd.concat([detections, non_detections])
    if not lightcurve.empty:
        lightcurve.sort_values("jd", inplace=True)
    else:
        lightcurve["jd"] = 0  # Fails later without a date column...
        lightcurve["mjd"] = 0
    return lightcurve


def target_from_fink_alert(alert: dict, t_ref: Time = None) -> Target:
    t_ref = t_ref or Time.now()

    objectId = alert.get("objectId", None)
    if objectId is None:
        raise MissingObjectIdError(f"alert has no objectId: {alert.keys()}")
    ra = alert.get("ra", None)
    dec = alert.get("dec", None)
    if (ra is None) or (dec is None):
        raise ValueError(f"one of (ra, dec)=({ra}, {dec}) is None")
    return Target(objectId, ra=ra, dec=dec, t_ref=t_ref)


def target_from_fink_lightcurve(
    lightcurve: pd.DataFrame, objectId: str, t_ref: Time = None
) -> Target:
    t_ref = t_ref or Time.now()

    ra = None
    dec = None
    if "ra" in lightcurve.columns:
        ra = lightcurve["ra"].iloc[-1]
    if "dec" in lightcurve.columns:
        dec = lightcurve["dec"].iloc[-1]

    if (ra is None) or (dec is None):
        raise MissingCoordinateColumnsError(f"missing ra/dec from {lightcurve.columns}")

    fink_data = TargetData(lightcurve=lightcurve)
    target = Target(objectId, ra=ra, dec=dec, fink_data=fink_data)
    target.updated = True
    return target


def process_fink_query_results(raw_query_results: pd.DataFrame):
    query_results = raw_query_results.copy()
    query_results.sort_values(["objectId", "ndethist"], inplace=True)
    query_results.drop_duplicates("objectId", keep="last", inplace=True)
    query_results.set_index("objectId", verify_integrity=True, inplace=True)
    return query_results


def target_from_fink_query_row(objectId: str, data: pd.Series):
    if isinstance(data, dict):
        data = pd.Series(data)
    if "ra" not in data or "dec" not in data:
        msg = (
            f"\033[33m{objectId} target_from_alerce_query_row\033[0m"
            f"\n     missing 'meanra'/'meandec' from row {data.index}"
        )
        logger.warning(msg)
        return None
    if isinstance(data, pd.DataFrame):
        if isinstance(data["ra"], pd.Series):
            if len(data["ra"]) > 1:
                logger.error(
                    f"\033[31mtarget_from_alerce_query_row\033[0m has data\n{data}"
                )
                raise ValueError("data passed has length greater than>1")

    return Target(objectId, ra=data["ra"], dec=data["dec"])


class FinkQueryManager(BaseQueryManager):
    name = "fink"
    default_query_parameters = {
        "update": 1.0,
        "interval": 2.0,
        "query_timespan": 0.1,
        "lookback_time": 20.0,
        "max_failed_queries": 10,
        "max_total_query_time": 300,  # total time to spend in each stage seconds
    }
    default_kafka_parameters = {"n_alerts": 10, "timeout": 10.0}
    required_kafka_parameters = ("username", "group_id", "bootstrap.servers", "topics")
    alert_extra_keys = (
        "objectId",
        "candid",
        "timestamp",
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
        data_path=None,
        create_paths=True,
    ):
        self.fink_config = fink_config
        self.target_lookup = target_lookup
        if fink_client is None:
            raise ValueError(
                "fink_client module not imported correctly!\n"
                "either install with `\033[32;1mpython3 -m pip install fink_client\033[0m` "
                "(you may also need to install `\033[32;1mfastavro\033[0m`), " 
                "or switch `use: False` in config."
            )

        self.kafka_config = self.get_kafka_parameters()

        object_queries = self.fink_config.get("object_queries", [])
        if isinstance(object_queries, str):
            object_queries = [object_queries]
        self.object_queries = object_queries

        self.query_parameters = self.default_query_parameters.copy()
        query_params = self.fink_config.get("query_parameters", {})
        self.query_parameters.update(query_params)

        self.query_results = {}
        self.query_updates = {}

        self.process_paths(data_path=data_path, create_paths=create_paths)

    def get_kafka_parameters(self):
        kafka_parameters = self.default_kafka_parameters.copy()

        kafka_config = self.fink_config.get("kafka_config", None)
        if kafka_config is None:
            logger.warning(
                "no kafka_config: \033[31;1mwill not listen for alerts.\033[0m"
            )
            return None
        kafka_parameters.update(kafka_config)

        if any([kw not in kafka_parameters for kw in self.required_kafka_parameters]):
            err_msg = f"kafka_config: provide {self.required_kafka_parameters}"
            raise BadKafkaConfigError(err_msg)

        topics = kafka_parameters.get("topics", None)
        if isinstance(topics, str):
            topics = [topics]
            kafka_parameters["topics"] = topics

        return kafka_parameters

    def listen_for_alerts(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if fink_client is None:
            logger.warning("no fink_client module: can't listen for alerts!")

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

    def read_simulated_alerts(self, alert_dir: Path):
        alert_dir = Path(alert_dir)

        alerts = []
        for alert_path in alert_dir.glob("*.json"):
            with open(alert_path, "r") as f:
                alert = json.load(f)
                alerts.append(alert)

        logger.info(f"read {len(alerts)} simulated alerts")
        return alerts

    def process_alerts(
        self,
        alerts: List[Tuple[str, Dict, str]],
        save_alerts=True,
        save_cutouts=True,
        simulated_alerts=False,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()

        if simulated_alerts:
            return alerts

        processed_alerts = []
        for topic, alert, key in alerts:
            candidate = alert["candidate"]
            objectId = alert["objectId"]

            candidate["topic"] = topic
            candidate["tag"] = "valid"
            extra_data = {k: alert[k] for k in self.alert_extra_keys}
            candidate.update(extra_data)
            processed_alerts.append(candidate)

            if save_alerts:
                alert_file = self.get_alert_file(objectId, candidate["candid"])
                with open(alert_file, "w") as f:
                    json.dump(candidate, f, indent=2)
            if not simulated_alerts and save_cutouts:
                cutouts = {}
                for imtype in FinkQuery.imtypes:
                    data = alert.get("cutout" + imtype, {}).get("stampData", None)
                    if data is None:
                        continue
                    cutout = readstamp(data, return_type="array")
                    cutouts[imtype.lower()] = cutout
                if len(cutouts) > 0:
                    cutout_file = self.get_cutouts_file(objectId, candidate["candid"])
                    with open(cutout_file, "wb+") as f:
                        pickle.dump(cutouts, f)
        return processed_alerts

    def new_targets_from_alerts(self, alerts: List[Dict], t_ref: None = Time):
        t_ref = t_ref or Time.now()

        added = []
        existing = []
        for alert in alerts:
            objectId = alert.get("objectId", None)
            if objectId is None:
                raise MissingObjectIdError("objectId is None in fink_alert")
            target = self.target_lookup.get(objectId, None)
            if target is not None:
                # Don't need to make a target.
                existing.append(objectId)
                continue
            target = target_from_fink_alert(alert, t_ref=t_ref)
            self.target_lookup[objectId] = target
            added.append(objectId)
        if len(added) > 0 or len(existing) > 0:
            logger.info(f"{len(added)} targets init, {len(existing)} existing skipped")
        return added, existing

    def query_for_updates(self, results_stem=None, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        for fink_class in self.object_queries:
            results_stem = results_stem or fink_class
            query_results_file = self.get_query_results_file(results_stem)
            query_results_file_age = calc_file_age(query_results_file, t_ref)
            if query_results_file_age < self.query_parameters["update"]:
                raw_query_updates = pd.read_csv(query_results_file)
                if raw_query_updates.empty:
                    query_updates = raw_query_updates
                else:
                    query_updates = process_fink_query_results(raw_query_updates)
                self.query_updates[fink_class] = query_updates
            else:
                logger.info(f"query for {fink_class}")
                t_start = time.perf_counter()
                raw_query_updates = self.query_and_collate_pages(
                    fink_class, t_ref=t_ref
                )
                if raw_query_updates is None:
                    continue
                raw_query_updates.to_csv(query_results_file, index=False)
                if raw_query_updates.empty:
                    query_updates = raw_query_updates
                else:
                    query_updates = process_fink_query_results(raw_query_updates)
                self.query_updates[fink_class] = query_updates
                t_end = time.perf_counter()
                logger.info(
                    f"{fink_class} returned {len(query_updates)} in {t_end-t_start:.1f}s"
                )
                self.query_results_updated = True

    def query_and_collate_pages(
        self, fink_class, lookback=None, step=None, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        step = step or self.query_parameters["query_timespan"]
        lookback = lookback or self.query_parameters["lookback_time"]
        lookback_grid = t_ref + np.arange(-lookback, step, step) * u.day

        update_dfs = []

        for start_jd, stop_jd in zip(lookback_grid[:-1], lookback_grid[1:]):
            query_data = self.get_query_data(fink_class, start_jd, stop_jd)

            try:
                updates = FinkQuery.query_latests(**query_data)
            except Exception as e:
                logger.warning(f"{fink_class} {start_jd:.1f}<jd<{stop_jd:.1f} failed")
                updates = None
            if updates is None:
                return None

            n = query_data["n"]
            if len(updates) == n:
                msg = f"`{fink_class}` query returned max updates={n}. Choose a shorter `query_timespan`!"
                logger.warning(msg)
            if len(updates) > 0:
                update_dfs.append(updates)

        if len(update_dfs) > 0:
            return pd.concat(update_dfs)
        return pd.DataFrame([], columns=["objectId"])

    def get_query_data(self, fink_class, start_jd: Time, stop_jd: Time):
        return dict(
            class_=fink_class, n=1000, startdate=start_jd.iso, stopdate=stop_jd.iso
        )

    def target_updates_from_query_results(self, t_ref: Time = None) -> pd.DataFrame:
        t_ref = t_ref or Time.now()

        update_dfs = []
        for query_name, updated_results in self.query_updates.items():
            # updated_results.sort_values(["oid", "lastmjd"], inplace=True)
            updated_targets = []
            existing_results = self.query_results.get(query_name, None)
            if existing_results is None:
                self.query_results[query_name] = updated_results
                logger.info(
                    f"no existing {query_name} results, use updates {len(updated_results)}"
                )
                continue
            # existing_results.sort_values(["oid", "lastmjd"], inplace=True)
            for objectId, updated_row in updated_results.iterrows():
                if objectId in existing_results.index:
                    existing_row = existing_results.loc[objectId]
                    if updated_row["ndethist"] > existing_row["ndethist"]:
                        updated_targets.append(objectId)
                else:
                    updated_targets.append(objectId)
            self.query_results[query_name] = updated_results
            updated = updated_results.loc[updated_targets]
            update_dfs.append(updated)

        if len(update_dfs) > 0:
            # alerce_updates = updated_results.loc[updated_targets]
            fink_updates = pd.concat(update_dfs)
            logger.info(f"{len(fink_updates)} alerce targets updates")
        else:
            fink_updates = None
        return fink_updates

    def new_targets_from_updates(self, updates: pd.DataFrame, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        new_targets = []
        if updates is None:
            return new_targets

        for objectId, row in updates.iterrows():
            target = self.target_lookup.get(objectId, None)
            if target is not None:
                continue
            target = target_from_fink_query_row(objectId, row)
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
            lightcurve_file = self.get_lightcurve_file(objectId)
            lightcurve_file_age = calc_file_age(
                lightcurve_file, t_ref, allow_missing=True
            )
            if lightcurve_file_age > self.query_parameters["interval"]:
                to_update.append(objectId)
        return to_update

    def perform_lightcurve_queries(self, objectId_list: List[str], t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if len(objectId_list) > 0:
            logger.info(f"attempt {len(objectId_list)} lightcurve queries")

        success = []
        failed = []
        t_start = time.perf_counter()
        for objectId in objectId_list:
            lightcurve_file = self.get_lightcurve_file(objectId)
            if len(failed) > self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed queries ({len(failed)}), stop for now"
                logger.info(msg)
                break
            t_now = time.perf_counter()
            if t_now - t_start > self.query_parameters["max_total_query_time"]:
                logger.info(f"querying time ({t_now - t_start:.1f}s) exceeded max")
                break
            try:
                detections = FinkQuery.query_objects(
                    objectId=objectId,
                    withupperlim=False,
                    return_df=True,
                    fix_keys=True,
                )
                non_detections = FinkQuery.query_objects(
                    objectId=objectId,
                    withupperlim=True,
                    return_df=True,
                    fix_keys=True,
                )
            except Exception as e:
                print(e)
                logger.warning(f"{objectId} lightcurve query failed")
                failed.append(objectId)
                continue
            lightcurve = process_fink_lightcurve(detections, non_detections)
            if lightcurve.empty:
                logger.warning(f"\033[33m{objectId} lightcurve empty!\033[0m")
            lightcurve.to_csv(lightcurve_file, index=False)
            success.append(objectId)

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
        t_start = time.perf_counter()

        if objectId_list is None:
            objectId_list = list(self.target_lookup.keys())

        for objectId in objectId_list:
            lightcurve_file = self.get_lightcurve_file(objectId)
            if not lightcurve_file.exists():
                missing.append(objectId)
                continue
            target = self.target_lookup.get(objectId, None)
            lightcurve = pd.read_csv(lightcurve_file, dtype={"candid": "Int64"})
            # TODO: not optimum to read every time... but not a bottleneck for now.
            if target is None:
                try:
                    target = target_from_fink_lightcurve(lightcurve, objectId)
                except MissingCoordinateColumnsError as e:
                    logger.warning(f"{objectId}: {e}")
                    continue
                self.target_lookup[objectId] = target
            else:
                existing_lightcurve = target.fink_data.lightcurve
                if existing_lightcurve is not None:
                    if len(lightcurve) <= len(existing_lightcurve):
                        continue
            loaded.append(objectId)
            lightcurve.query("jd<@t_ref.jd", inplace=True)
            target.fink_data.add_lightcurve(lightcurve)
            target.updated = True
        t_end = time.perf_counter()

        N_loaded = len(loaded)
        N_missing = len(missing)
        # if N_loaded > 0:
        logger.info(
            f"loaded {N_loaded}, missing {N_missing} lightcurves in {t_end-t_start:.1f}s"
        )
        return loaded, missing

    def load_missing_alerts(self, objectId: str):
        target = self.target_lookup.get(objectId, None)
        if target is None:
            raise ValueError(f"{objectId} has no entry in target_lookup")
        alert_dir = self.get_alert_dir(objectId)
        if not alert_dir.exists():
            return []
        alert_list = alert_dir.glob("*.json")

        if target.fink_data.lightcurve is None:
            existing_candids = []
        else:
            existing_candids = target.fink_data.lightcurve["candid"].values

        loaded_alerts = []
        for alert_file in alert_list:
            if int(alert_file.stem) in existing_candids:
                continue
            with open(alert_file, "r") as f:
                alert = json.load(f)
                loaded_alerts.append(alert)
        return loaded_alerts

    def integrate_alerts(self, save_lightcurve=True):
        integrated_alerts = []
        for objectId, target in self.target_lookup.items():
            loaded_alerts = self.load_missing_alerts(objectId)
            if len(loaded_alerts) == 0:
                continue

            alert_df = pd.DataFrame(loaded_alerts)
            target.fink_data.integrate_lightcurve_updates(alert_df, column="candid")

            if save_lightcurve:
                lightcurve_file = self.get_lightcurve_file(objectId)
                target.fink_data.lightcurve.to_csv(lightcurve_file, index=False)
            integrated_alerts.append(objectId)
        if len(integrated_alerts):
            logger.info(
                f"integrate alerts into LC for {len(integrated_alerts)} targets"
            )

    def load_cutouts(self):
        loaded_cutouts = []
        for objectId, target in self.target_lookup.items():
            cutouts_are_None = [
                im is None for k, im in target.fink_data.cutouts.items()
            ]
            for candid in target.fink_data.detections["candid"][::-1]:
                cutouts_file = self.get_cutouts_file(objectId, candid)
                if cutouts_file.exists():
                    cutouts_candid = target.fink_data.meta.get("cutouts_candid", None)
                    if cutouts_candid == candid:
                        # If the existing cutouts are from this candid,
                        # they must already be the latest (as we're searching in rev.)
                        break
                    with open(cutouts_file, "rb") as f:
                        cutouts = pickle.load(f)
                    target.fink_data.cutouts = cutouts
                    target.fink_data.meta["cutouts_candid"] = candid
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
            timestamp = alert.get("timestamp")
            alert_jd = alert.get("jd")
            alert_text = (
                f"FINK alert from {topic_str}\n"
                f"     broadcast at jd={alert_jd:.5f}={timestamp}\n"
            )
            target.update_messages.append(alert_text)

    def perform_all_tasks(
        self, simulated_alerts=False, alert_dir=None, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        # get alerts
        if simulated_alerts:
            if alert_dir is None:
                raise ValueError(
                    "must also provide alert_path, where to read simulated alerts"
                )
            alerts = self.read_simulated_alerts(alert_dir)
        else:
            alerts = self.listen_for_alerts()
        processed_alerts = self.process_alerts(
            alerts, simulated_alerts=simulated_alerts, t_ref=t_ref
        )
        self.new_targets_from_alerts(processed_alerts, t_ref=t_ref)

        if not simulated_alerts:
            new_alerts = set([alert["objectId"] for alert in processed_alerts])
            success, failed = self.perform_lightcurve_queries(new_alerts)

            lcs_to_query = self.get_lightcurves_to_query(t_ref=t_ref)
            success, failed = self.perform_lightcurve_queries(lcs_to_query)

        loaded_lcs, missing_lcs = self.load_target_lightcurves(t_ref=t_ref)

        if not simulated_alerts:
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
            logger.error("\033[31;1merror rasied\033[0m")
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
