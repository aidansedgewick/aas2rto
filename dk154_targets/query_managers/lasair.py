import json
from logging import getLogger
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

    for r in non_det:
        assert not r.get("candid")
        r["candid"] = ""
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


class LasairQueryManager(BaseQueryManager):
    name = "lasair"
    default_query_parameters = {"interval": 2.0, "max_failed_queries": 10}
    default_kafka_parameters = {"n_alerts": 10, "timeout": 10.0}
    required_kafka_parameters = ("host", "group_id", "topics")

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

    def listen_for_alerts(self, t_ref: Time = None) -> List[Dict]:
        t_ref = t_ref or Time.now()

        if lasair is None:
            logger.info("no lasair module: can't listen for alerts!")
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
                data = json.loads(msg.value())
                data["topic"] = topic
                new_alerts.append(data)
            logger.info(f"received {len(new_alerts)} {topic} alerts")
        return new_alerts

    def read_simulated_alerts(self, t_ref: Time = None) -> List[Dict]:
        t_ref = t_ref or Time.now()
        raise NotImplementedError
        return alerts

    def find_old_lightcurves(self, t_ref: Time = None) -> List[str]:
        t_ref = t_ref or Time.now()

        objectIds_to_update = []
        for objectId, target in self.target_lookup.items():
            lightcurve_file = self.get_lightcurve_file(objectId)
            lightcurve_file_age = calc_file_age(
                lightcurve_file, t_ref, allow_missing=True
            )
            if lightcurve_file_age > self.query_parameters["update"]:
                objectIds_to_update.append(objectId)
        return objectIds_to_update

    def perform_lightcurve_queries(
        self, objectId_list: List, t_ref: Time = None, discard_existing=True
    ):
        t_ref = t_ref or Time.now()
        token = self.client_config.get("token", None)
        if token is None:
            err_msg = f"Must provied `client_config`: `token` for lightcurves"
            raise ValueError(err_msg)
        N_success = 0
        N_failed = 0
        lightcurves_queried = []
        for objectId in objectId_list:
            if N_failed > self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed lightcurve queries ({N_failed}). Stop for now."
                logger.info(msg)
                break

            client = lasair_client(token)
            try:
                lc = client.lightcurves([objectId])
                N_success = N_success + 1
            except Exception as e:
                logger.warning(e)
                N_failed = N_failed + 1
                continue
            assert len(lc) == 1
            lightcurve = process_lasair_lightcurve(lc[0])
            lightcurve_file = self.get_lightcurve_file(objectId)

            lightcurves_queried.append(objectId)
            if lightcurve_file.exists() and discard_existing:
                existing_lightcurve = pd.read_csv(lightcurve_file)
                if existing_lightcurve["jd"].max() >= lightcurve["jd"].max():
                    continue
            lightcurve.to_csv(lightcurve_file, index=False)

        return lightcurves_queried

    def load_target_lightcurves(self, objectId_list: List[str], t_ref: Time = None):
        t_ref = t_ref or Time.now()
        for objectId in objectId_list:
            lightcurve_file = self.get_lightcurve_file(objectId)
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
                target.lasair_data.lightcurve = lightcurve
                target.updated = True

    def perform_all_tasks(self, t_ref: Time = None, simulated_alerts=None):
        t_ref = t_ref or Time.now()
        if simulated_alerts:
            alerts = self.read_simulated_alerts(t_ref=t_ref)
        else:
            alerts = self.listen_for_alerts(t_ref=t_ref)
        # if simulated_alerts:
        #    pass
        # else:
        alert_objectIds = [alert["objectId"] for alert in alerts]
        logger.info(f"lightcurves for {len(alert_objectIds)} alerts")
        alert_lightcurves = self.perform_lightcurve_queries(
            alert_objectIds, t_ref=t_ref
        )
        old_lightcurve_objectIds = self.find_old_lightcurves(t_ref=t_ref)
        logger.info(f"lightcurves for {len(old_lightcurve_objectIds)}")
        updated_lightcurves = self.perform_lightcurve_queries(
            old_lightcurve_objectIds, t_ref=t_ref
        )
        objectId_list = list(set(alert_lightcurves + updated_lightcurves))
        self.load_target_lightcurves(objectId_list)

        for alert in alerts:
            objectId = alert["objectId"]
            topic = alert.get("topic", None)
            target = self.target_lookup[objectId]
            target.send_updates = True
            if topic:
                target.update_messages.append(f"lasair topic {topic}")
