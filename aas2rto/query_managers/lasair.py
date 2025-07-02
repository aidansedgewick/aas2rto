import json
import time
from logging import getLogger
from pathlib import Path
from typing import Dict, Set, List

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.time import Time

try:
    import lasair
    from lasair import lasair_client, lasair_consumer
except ModuleNotFoundError as e:
    lasair = None

from aas2rto import utils
from aas2rto.target import Target, TargetData
from aas2rto.exc import (
    BadKafkaConfigError,
    MissingTargetIdError,
    MissingCoordinatesError,
)
from aas2rto.query_managers.base import BaseQueryManager

from aas2rto import paths

logger = getLogger(__name__.split(".")[-1])


def process_lasair_lightcurve(lc: Dict, t_ref: Time = None) -> pd.DataFrame:
    t_ref = t_ref or Time.now()

    target_id = lc["objectId"]
    det = [row for row in lc["candidates"] if row.get("candid")]
    non_det = [r for r in lc["candidates"] if not r.get("candid")]
    assert len(det) + len(non_det) == len(lc["candidates"])

    for r in det:
        r["candid_str"] = str(r["candid"])
        r["tag"] = "valid"

    for r in non_det:
        assert not r.get("candid")
        r["candid"] = -1
        r["candid_str"] = "-1"
        r["tag"] = "upperlim"

    detections = pd.DataFrame(det)
    non_detections = pd.DataFrame(non_det)

    lightcurve = pd.concat([detections, non_detections])
    lightcurve["objectId"] = target_id
    if lightcurve.empty:
        lightcurve["jd"] = 0  # fails later without date column.
        lightcurve["mjd"] = 0
    else:
        lightcurve["mjd"] = Time(lightcurve["jd"].values, format="jd").mjd

    lightcurve.sort_values("jd", inplace=True)
    return lightcurve


def target_from_lasair_lightcurve(
    lightcurve: pd.DataFrame, t_ref: Time = None
) -> Target:
    t_ref = t_ref or Time.now()

    if not isinstance(lightcurve, pd.DataFrame):
        lightcurve = process_lasair_lightcurve(lightcurve)

    if "objectId" not in lightcurve.columns:
        raise MissingTargetIdError("lightcurve has no 'objectId' column")
    target_id_options = lightcurve["objectId"].dropna().unique()
    if len(target_id_options) > 1:
        logger.warning(f"several target_id options:\n {target_id_options}")
    target_id = target_id_options[0]

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

    lasair_data = TargetData(lightcurve=lightcurve)
    alt_ids = {"ztf": target_id}
    target = Target(
        target_id, ra=ra, dec=dec, source="lasair", alt_ids=alt_ids, t_ref=t_ref
    )
    target.target_data["lasair"] = lasair_data
    return target


def target_from_lasair_alert(alert: Dict, t_ref: Time = None):
    t_ref = t_ref or Time.now()

    target_id = alert.get("objectId", None)
    if target_id is None:
        raise MissingTargetIdError(f"alert has no 'objectId': {alert.keys()}")
    for ra_col, dec_col in LasairQueryManager.default_coordinate_guesses:
        ra = alert.get(ra_col, None)
        dec = alert.get(dec_col, None)
        if all([ra is not None, dec is not None]):
            break
    topic = alert.get("topic", "<Unknown topic>")
    if (ra is None) or (dec is None):
        logger.warning(
            f"\033[033m{target_id} missing coords\033[0m!\n"
            f"    topic {topic} can't guess coords\n"
            f"    alert has keys {alert.keys()}.\n"
            f"    Can you updated the LASAIR filter?"
        )
    alt_ids = {"ztf": target_id}
    return Target(
        target_id, ra=ra, dec=dec, source="lasair", alt_ids=alt_ids, t_ref=t_ref
    )


def _get_alert_timestamp(alert: dict, t_ref: Time = None):
    t_ref = t_ref or Time.now()

    target_id = alert["objectId"]

    try:
        if "timestamp" in alert:
            timestamp = Time(alert["timestamp"])
        elif "UTC" in alert:
            timestamp = Time(alert["UTC"])
        elif "mjdmax" in alert:
            timestamp = Time(alert["mjdmax"], format="mjd")
        elif "jdmax" in alert:
            timestamp = Time(alert["jdmax"], format="jd")
        else:
            logger.warning(f"{target_id} has no timestamp candidates...")
            timestamp = t_ref
    except Exception as e:
        print(e)
        timestamp = t_ref

    return timestamp.isot


def lasair_id_from_target(target: Target, resolving_order=("lasair", "ztf", "lsst")):
    for source_name in resolving_order:
        # eg. loop through ["lasair", "ztf", "lsst"] until we get a hit.
        lasair_id = target.alt_ids.get(source_name, None)
        if lasair_id is not None:
            return lasair_id
    # last resort...
    target_id = target.target_id
    if target_id.lower().startswith("ztf") or target_id.lower().startswith("lsst"):
        return target_id  # must be the lasair id.
    return None


class LasairQueryManager(BaseQueryManager):
    name = "lasair"
    expected_lasair_parameters = (
        "object_queries",
        "client_token",
        "query_parameters",
        "kafka_config",
    )
    default_query_parameters = {
        "object_query_interval": 1.0,
        "lightcurve_update_interval": 2.0,
        "max_failed_queries": 10,
    }
    required_client_parameters = "token"
    default_kafka_parameters = {"n_alerts": 10, "timeout": 20.0}
    required_kafka_parameters = ("host", "group_id", "topics")

    default_coordinate_guesses = (("ramean", "decmean"), ("ra", "dec"), ("RA", "Dec"))

    def __init__(
        self,
        lasair_config: dict,
        target_lookup: Dict[str, Target],
        parent_path=None,
        create_paths=True,
    ):
        self.lasair_config = lasair_config
        self.target_lookup = target_lookup
        utils.check_unexpected_config_keys(
            self.lasair_config, self.expected_lasair_parameters, name="lasair_config"
        )

        if lasair is None:
            raise ValueError(
                "lasair module not imported correctly! "
                "either install with \033[31;1m`python3 -m pip install lasair`\033[0m, "
                "or switch `use: False` in config."
            )
        self.client_token = self.lasair_config.get("client_token", None)
        if self.client_token is None:
            logger.warning("\033[33;1mno client_token in lasair config!\033[0m")

        self.kafka_config = self.get_kafka_parameters()
        if (self.kafka_config) and (lasair is None):
            raise ValueError(
                "lasair module not imported correctly!\n"
                "Either install with `\033[32;1mpython3 -m pip install lasair\033[0m` "
                "(you may also need to install `\033[32;1mfastavro\033[0m`), "
                "or switch `use: False` in lasair config, in config file."
            )

        self.query_parameters = self.default_query_parameters.copy()
        query_params = self.lasair_config.get("query_parameters", {})
        self.query_parameters.update(query_params)
        utils.check_config_keys(
            self.query_parameters,
            self.default_query_parameters,
            name="lasair_config.query_parameters",
        )

        self.process_paths(parent_path=parent_path, create_paths=create_paths)

    def get_kafka_parameters(self):
        kafka_parameters = self.default_kafka_parameters.copy()

        kafka_config = self.lasair_config.get("kafka_config", None)
        if kafka_config is None:
            logger.warning(
                "no kafka_config: \033[31;1mwill not listen for alerts.\033[0m"
            )
            return None
        kafka_parameters.update(kafka_config)

        missing_kafka_keys = utils.check_missing_config_keys(
            kafka_parameters,
            self.required_kafka_parameters,
            name="lasair_config.kafka_config",
        )
        if len(missing_kafka_keys) > 0:
            err_msg = f"kafka_config: provide {self.required_kafka_parameters}"
            raise BadKafkaConfigError(err_msg)

        topics = kafka_parameters.get("topics", None)
        if isinstance(topics, str):
            topics = [topics]
            kafka_parameters["topics"] = topics

        return kafka_parameters

    def listen_for_alerts(self, t_ref: Time = None) -> List[Dict]:
        t_ref = t_ref or Time.now()

        if lasair is None:
            logger.info("no lasair module: can't listen for alerts!")
            return []
        if not self.kafka_config:
            logger.info("can't listen for alerts: no kafka config!")
            return []

        new_alerts = []
        for topic in self.kafka_config["topics"]:
            consumer = lasair_consumer(
                self.kafka_config["host"], self.kafka_config["group_id"], topic
            )
            for ii in range(self.kafka_config["n_alerts"]):
                msg = consumer.poll(timeout=self.kafka_config["timeout"])
                if msg is None:
                    break
                if msg.error():
                    logger.warning(str(msg.error()))
                    continue
                data = json.loads(msg.value())
                data["topic"] = topic
                new_alerts.append(data)
            logger.info(f"received {len(new_alerts)} {topic} alerts")
        return new_alerts

    def process_alerts(self, alerts: List[Dict], save_alerts=True, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        for alert in alerts:
            target_id = alert.get("objectId")

            timestamp = _get_alert_timestamp(alert)
            alert["alert_timestamp"] = timestamp

            if save_alerts:
                alert_filepath = self.get_alert_file(target_id, alert["candid"])
                with open(alert_filepath, "w") as f:
                    json.dump(alert, f)
        return alerts

    def new_targets_from_alerts(self, alerts: List[Dict], t_ref: Time = None):
        t_ref = t_ref or Time.now()

        added = []
        existing = []
        for alert in alerts:
            target_id = alert.get("objectId", None)
            if target_id is None:
                logger.warning(
                    f"\033[33m'objectId' is None in alert\033[0m:\n    {alert}"
                )
            target = self.target_lookup.get(target_id)
            if target is None:
                target = target_from_lasair_alert(alert, t_ref=t_ref)
                self.add_target(target)
                added.append(target_id)
            else:
                existing.append(target_id)
        if len(added) > 0 or len(existing) > 0:
            logger.info(f"{len(added)} targets init, {len(existing)} existing skipped")
        return added, existing

    def update_target_alt_ids(self):
        for target_id, target in self.target_lookup.items():
            lasair_id = lasair_id_from_target(target)
            if lasair_id is None:
                continue
            if "lasair" not in target.alt_ids:
                target.alt_ids["lasair"] = lasair_id

    def get_lightcurves_to_query(self, t_ref: Time = None) -> List[str]:
        t_ref = t_ref or Time.now()

        to_query = []
        for target_id, target in self.target_lookup.items():
            lasair_id = lasair_id_from_target(target)
            if lasair_id is None:
                continue

            lightcurve_file = self.get_lightcurve_file(lasair_id)
            lightcurve_file_age = utils.calc_file_age(
                lightcurve_file, t_ref, allow_missing=True
            )
            interval = self.query_parameters["lightcurve_update_interval"]
            if lightcurve_file_age > interval:
                to_query.append(lasair_id)
        return to_query

    def perform_lightcurve_queries(
        self, lasair_id_list: List, chunk_size=25, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        if self.client_token is None:
            err_msg = f"Must provied `client_token` for lightcurves"
            raise ValueError(err_msg)

        if len(lasair_id_list) > 0:
            logger.info(f"attempt {len(lasair_id_list)} lightcurve queries")

        success = []
        failed = []
        t_start = time.perf_counter()
        for ii, lasair_id_chunk in enumerate(
            utils.chunk_list(lasair_id_list, chunk_size=chunk_size)
        ):
            if len(failed) > self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed LC queries ({len(failed)}). Stop for now."
                logger.warning(msg)
                break
            client = lasair_client(self.client_token)
            try:
                lc_data_chunk = client.lightcurves(lasair_id_chunk)  # a LIST.
            except Exception as e:
                logger.warning(e)
                failed.extend(lasair_id_chunk)
                continue
            if not len(lc_data_chunk) == len(lasair_id_chunk):
                msg = f"lasair_client returned {len(lc_data_chunk)} - request {len(lasair_id_chunk)}"
                logger.warning(msg)
            for lc_data in lc_data_chunk:
                target_id = lc_data["objectId"]
                lightcurve = process_lasair_lightcurve(lc_data)
                lightcurve_filepath = self.get_lightcurve_file(target_id)
                lightcurve.to_csv(lightcurve_filepath, index=False)
                success.append(target_id)

        if len(success) > 0 or len(failed) > 0:
            t_end = time.perf_counter()
            logger.info(f"lightcurve queries in {t_end - t_start:.1f}s")
            logger.info(f"{len(success)} successful, {len(failed)} failed lc queries")
        return success, failed

    # def load_target_lightcurves(
    #     self, lasair_id_list: List[str] = None, t_ref: Time = None
    # ):
    #     t_ref = t_ref or Time.now()

    #     loaded = []
    #     skipped = []
    #     missing = []
    #     t_start = time.perf_counter()

    #     if lasair_id_list is None:
    #         lasair_id_list = []
    #         for target_id, target in self.target_lookup.items():
    #             lasair_id = lasair_id_from_target(target)
    #             if lasair_id is not None:
    #                 lasair_id_list.append(lasair_id)
    #         logger.info(f"try loading all {len(lasair_id_list)} lcs in target_lookup")
    #     else:
    #         logger.info(f"try loading {len(lasair_id_list)} lcs")

    #     for lasair_id in lasair_id_list:
    #         lightcurve = self.load_single_lightcurve(target_id)
    #         if lightcurve is None:
    #             missing.append(target_id)
    #             continue

    #         target = self.target_lookup.get(lasair_id, None)
    #         if target is None:
    #             logger.warning(f"load_lightcurve: {target_id} not in target_lookup!")
    #             missing.append(target_id)
    #             continue

    #         lasair_data = target.get_target_data("lasair")
    #         existing_lightcurve = lasair_data.lightcurve
    #         if existing_lightcurve is not None:
    #             if len(lightcurve) <= len(existing_lightcurve):
    #                 skipped.append(target_id)
    #                 continue
    #             # Not updated if there was no data to begin with...
    #             target.updated = True

    #         if target.coord is None:
    #             ra_vals = lightcurve["ra"].dropna()
    #             dec_vals = lightcurve["dec"].dropna()
    #             try:
    #                 target.update_coordinates(np.average(ra_vals), np.average(dec_vals))
    #             except Exception as e:
    #                 logger.error(f"failed to add missing coords to {target_id}")
    #         lasair_data.add_lightcurve(lightcurve)
    #         loaded.append(target_id)
    #     t_end = time.perf_counter()

    #     N_loaded = len(loaded)
    #     N_missing = len(missing)
    #     t_load = t_end - t_start
    #     msg = f"loaded {N_loaded}, missing {N_missing} lightcurves in {t_load:.1f}s"
    #     logger.info(msg)
    #     return loaded, missing

    def load_single_lightcurve(self, target_id: str):
        lightcurve_filepath = self.get_lightcurve_file(target_id)
        if not lightcurve_filepath.exists():
            return None

        try:
            lightcurve = pd.read_csv(lightcurve_filepath, dtype={"candid": "Int64"})
        except pd.errors.EmptyDataError as e:
            logger.warning(f"bad lightcurve file for {target_id}")
            return None

        return lightcurve

    def apply_messenger_updates(self, alerts: List[Dict]):
        for alert in alerts:
            target_id = alert["objectId"]
            target = self.target_lookup.get(target_id, None)
            if target is None:
                msg = f"No target {target_id} in target lookup after alert..."
                logger.warning(msg)
                continue
            target.updated = True
            target.send_updates = True
            topic_str = alert["topic"]
            timestamp = alert["alert_timestamp"]
            alert_text = (
                f"LASAIR alert from {topic_str}\n" f"     broadcast at jd={timestamp}\n"
            )
            target.update_messages.append(alert_text)

    def perform_all_tasks(self, startup=False, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if startup:
            logger.info("skip listening for alerts on startup step (iter 0)")
        else:
            alerts = self.listen_for_alerts(t_ref=t_ref)
            processed_alerts = self.process_alerts(alerts)
            self.new_targets_from_alerts(processed_alerts, t_ref=t_ref)

        self.update_target_alt_ids()

        # Always query for alert LCs - alert means that the LC should be updated!
        alert_target_ids = [alert["objectId"] for alert in alerts]
        success, failed = self.perform_lightcurve_queries(alert_target_ids, t_ref=t_ref)

        if startup:
            logger.info("skip query lightcurves on startup")
        else:
            to_query = self.get_lightcurves_to_query(t_ref=t_ref)
            logger.info(f"lightcurves for {len(to_query)}")
            success, failed = self.perform_lightcurve_queries(to_query, t_ref=t_ref)
        loaded, missing = self.load_target_lightcurves(t_ref=t_ref)

        self.apply_messenger_updates(processed_alerts)
