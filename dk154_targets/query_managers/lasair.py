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

from dk154_targets import Target, TargetData
from dk154_targets.utils import calc_file_age

from dk154_targets.query_managers.base import BaseQueryManager
from dk154_targets.query_managers.exc import BadKafkaConfigError, MissingObjectIdError

from dk154_targets import paths

logger = getLogger(__name__.split(".")[-1])


def process_lasair_lightcurve(lc: Dict, t_ref: Time = None) -> pd.DataFrame:
    t_ref = t_ref or Time.now()
    det = [r for r in lc["candidates"] if r.get("candid")]
    non_det = [r for r in lc["candidates"] if not r.get("candid")]
    assert len(det) + len(non_det) == len(lc["candidates"])

    for r in det:
        r["candid_int"] = r["candid"]
        r["candid"] = str(r["candid"])
        r["tag"] = "valid"

    for r in non_det:
        assert not r.get("candid")
        r["candid_int"] = 0
        r["candid"] = ""
        r["tag"] = "upperlim"

    detections = pd.DataFrame(det)
    non_detections = pd.DataFrame(non_det)
    lightcurve = pd.concat([detections, non_detections])
    lightcurve.sort_values("jd", inplace=True)
    return lightcurve


def target_from_lasair_lightcurve(
    objectId: str, lightcurve: pd.DataFrame, t_ref: Time = None
) -> Target:
    t_ref = t_ref or Time.now()
    ra = lightcurve["ra"].iloc[-1]
    dec = lightcurve["dec"].iloc[-1]
    lasair_data = TargetData(lightcurve=lightcurve)

    target = Target(objectId, ra=ra, dec=dec, lasair_data=lasair_data)
    target.updated = True

    return target


def target_from_lasair_alert(alert: Dict, t_ref: Time = None):
    t_ref = t_ref or Time.now()

    objectId = alert.get("objectId", None)
    if objectId is None:
        raise MissingObjectIdError(f"alert has no objectId: {alert.keys()}")
    for racol, deccol in LasairQueryManager.default_coordinate_guesses:
        ra = alert.get(racol, None)
        dec = alert.get(deccol, None)
        if all([ra is not None, dec is not None]):
            break
    topic = alert.get("topic", "<Unknown topic>")
    if (ra is None) or (dec is None):
        logger.warning(
            f"\033[033m{objectId} missing coords\033[0m\n"
            f"    topic {topic} can't guess coords\n"
            f"    alert has keys {alert.keys()}"
        )
    return Target(objectId, ra=ra, dec=dec, t_ref=t_ref)


def _get_alert_timestamp(alert: dict, t_ref: Time = None):
    t_ref = t_ref or Time.now()

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
            logger.warning(f"{objectId} has no timestamp candidates...")
            timestamp = t_ref
    except Exception as e:
        print(e)
        timestamp = t_ref

    return timestamp.isot


class LasairQueryManager(BaseQueryManager):
    name = "lasair"
    default_query_parameters = {"interval": 2.0, "max_failed_queries": 10}
    default_kafka_parameters = {"n_alerts": 10, "timeout": 10.0}
    required_kafka_parameters = ("host", "group_id", "topics")

    default_coordinate_guesses = (
        ("ramean", "decmean"),
        ("ra", "dec"),
    )

    def __init__(
        self,
        lasair_config: dict,
        target_lookup: Dict[str, Target],
        data_path=None,
        create_paths=True,
    ):
        self.lasair_config = lasair_config
        self.target_lookup = target_lookup
        if lasair is None:
            raise ValueError(
                "lasair module not imported correctly! "
                "either install with \033[31;1m`python3 -m pip install lasair`\033[0m, "
                "or switch `use: False` in config."
            )

        self.client_config = self.lasair_config.get("client_config", {})

        self.kafka_config = self.get_kafka_parameters()

        self.query_parameters = self.default_query_parameters.copy()
        query_params = self.lasair_config.get("query_parameters", {})
        self.query_parameters.update(query_params)

        self.process_paths(data_path=data_path, create_paths=create_paths)

    def get_kafka_parameters(self):
        kafka_parameters = self.default_kafka_parameters.copy()

        kafka_config = self.lasair_config.get("kafka_config", None)
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

    def listen_for_alerts(self, dump_alerts=True, t_ref: Time = None) -> List[Dict]:
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
            objectId = alert.get("objectId")

            timestamp = _get_alert_timestamp(alert)
            alert["alert_timestamp"] = timestamp

            if save_alerts:
                alert_file = self.get_alert_file(objectId, timestamp)
                with open(alert_file, "w") as f:
                    json.dump(alert, f)
        return alerts

    def read_simulated_alerts(self, t_ref: Time = None) -> List[Dict]:
        t_ref = t_ref or Time.now()
        raise NotImplementedError()
        return alerts

    def new_targets_from_alerts(self, alerts: List[Dict], t_ref: Time = None):
        t_ref = t_ref or Time.now()

        for alert in alerts:
            objectId = alert.get("objectId", None)
            if objectId is None:
                logger.warning(
                    f"\033[33mobjectId is None in alert\033[0m:\n    {alert}"
                )
            target = self.target_lookup.get(objectId)
            if target is None:
                target = target_from_lasair_alert(alert, t_ref=t_ref)
                self.add_target(target)

    def get_lightcurves_to_query(self, t_ref: Time = None) -> List[str]:
        t_ref = t_ref or Time.now()

        to_query = []
        for objectId, target in self.target_lookup.items():
            lightcurve_file = self.get_lightcurve_file(objectId)
            lightcurve_file_age = calc_file_age(
                lightcurve_file, t_ref, allow_missing=True
            )
            if lightcurve_file_age > self.query_parameters["update"]:
                to_query.append(objectId)
        return to_query

    def perform_lightcurve_queries(self, objectId_list: List, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        token = self.client_config.get("token", None)
        if token is None:
            err_msg = f"Must provied `client_config`: `token` for lightcurves"
            raise ValueError(err_msg)

        if len(objectId_list) > 0:
            logger.info(f"attempt {len(objectId_list)} lightcurve queries")

        success = []
        failed = []
        t_start = time.perf_counter()
        for objectId in objectId_list:
            if len(failed) > self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed lightcurve queries ({N_failed}). Stop for now."
                logger.info(msg)
                break
            client = lasair_client(token)
            try:
                lc = client.lightcurves([objectId])
            except Exception as e:
                logger.warning(e)
                failed.append(objectId)
                continue
            assert len(lc) == 1
            lightcurve = process_lasair_lightcurve(lc[0])
            lightcurve_file = self.get_lightcurve_file(objectId)
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

            lightcurve = pd.read_csv(lightcurve_file)
            lightcurve.query(f"jd < @t_ref.jd", inplace=True)
            lightcurve.sort_values("jd", inplace=True)
            target = self.target_lookup.get(objectId, None)
            if target is None:
                target = target_from_lasair_lightcurve(
                    objectId, lightcurve=lightcurve, t_ref=t_ref
                )
                self.target_lookup[objectId] = target
            else:
                existing_lightcurve = target.lasair_data.lightcurve
                if existing_lightcurve is not None:
                    if len(lightcurve) <= len(existing_lightcurve):
                        continue
            target.lasair_data.add_lightcurve(lightcurve)
            loaded.append(objectId)
            target.updated = True
        t_end = time.perf_counter()

        N_loaded = len(loaded)
        N_missing = len(missing)
        if N_loaded > 0:
            logger.info(
                f"loaded {N_loaded}, missing {N_missing} lightcurves in {t_end-t_start:.1f}s"
            )
        return loaded, missing

    def apply_messenger_updates(self, alerts: List[Dict]):
        for alert in alerts:
            objectId = alert["objectId"]
            target = self.target_lookup.get(objectId, None)
            if target is None:
                continue
            target.send_updates = True
            topic_str = alert["topic"]
            timestamp = alert["alert_timestamp"]
            alert_text = (
                f"LASAIR alert from {topic_str}\n"
                f"     broadcast at jd={timestamp}\n"
                f"     see lasair-ztf.lsst.ac.uk/objects/{objectId}/"
            )
            target.update_messages.append(alert_text)

    def perform_all_tasks(self, t_ref: Time = None, simulated_alerts=None):
        t_ref = t_ref or Time.now()
        if simulated_alerts:
            alerts = self.read_simulated_alerts(t_ref=t_ref)
        else:
            alerts = self.listen_for_alerts(t_ref=t_ref)

        processed_alerts = self.process_alerts(alerts)

        self.new_targets_from_alerts(processed_alerts, t_ref=t_ref)

        # Always query for alert LCs - alert means that the LC should be updated!
        alert_objectIds = [alert["objectId"] for alert in alerts]
        success, failed = self.perform_lightcurve_queries(alert_objectIds, t_ref=t_ref)

        to_query = self.get_lightcurves_to_query(t_ref=t_ref)
        logger.info(f"lightcurves for {len(to_query)}")
        success, failed = self.perform_lightcurve_queries(to_query, t_ref=t_ref)
        loaded, missing = self.load_target_lightcurves(t_ref=t_ref)

        self.apply_messenger_updates(processed_alerts)
