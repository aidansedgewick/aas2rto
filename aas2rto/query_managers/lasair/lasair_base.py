from __future__ import annotations


import abc
import json
from logging import getLogger
from pathlib import Path

from confluent_kafka.cimpl import Message

from astropy.time import Time

from aas2rto.exc import BadKafkaConfigError, MissingKeysError
from aas2rto.query_managers.base import LightcurveQueryManager, KafkaQueryManager
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
    import lasair
except ModuleNotFoundError as e:
    lasair = None


logger = getLogger(__name__.split("/")[-1])


# Do NOT register LasairBaseQueryManager
# Re-inheirt from abc.ABC to indicate LasairBaseQM should be subclasses...
class LasairBaseQueryManager(LightcurveQueryManager, KafkaQueryManager, abc.ABC):

    @property
    @abc.abstractmethod
    def id_resolving_order(self) -> tuple[str]:
        """The order that the QM should prefer IDs for queries to the Lasair client.

        eg. ("lasair_lsst", "lsst", "tns") would first check in a target's alt_ids
        for a 'lasair_lsst' ID, then a 'lsst' ID, and finally a 'tns' ID."""

    @property
    @abc.abstractmethod
    def target_id_key(self) -> str:
        """The key used as target_id from Lasair alerts/lightcurves.

        eg. 'objectId' for ZTF, 'diaObjectId' for LSST"""

    @property
    @abc.abstractmethod
    def alert_id_key(self) -> str:
        """The key used as alert_id from Lasair alerts/lightcurves

        eg. 'candid' for ZTF, 'diaSourceId' for LSST"""

    ##===== Methods for subclass to define here =====##

    @abc.abstractmethod
    def process_single_alert(alert: dict, topic_keys: dict):
        pass

    DEFAULT_TASKS = ("alerts", "queries", "lightcurves", "cutouts", "stale_files")
    DEFAULT_TOPIC_KEYS: dict = {
        "target_id": target_id_key,  # Use the abstract property
        "ra": "ra",
        "decl": "decl",
        "date": "mjd",
        # "alert_id" # diaSourceId/ObjectId is not available in Lasair alerts.
    }
    REQUIRED_DIRECTORIES = ("alerts", "cutouts", "lightcurves", "sherlock")
    REQUIRED_KAFKA_PARAMS = ("group_id", "server", "topics")
    default_config = {
        "client_token": None,
        "kafka": None,
    }

    ##===== Initialisation here =====##

    def __init__(
        self, config: dict, target_lookup: TargetLookup, parent_path: Path = None
    ):
        # Config steps
        self.config = self.default_config.copy()
        self.config.update(config)

        check_unexpected_config_keys(
            self.config, self.default_config, name="{self.name}_config", raise_exc=True
        )
        self.process_kafka_config()

        # Set the target_lookup
        self.target_lookup = target_lookup

        # Check 'lasair' tools imported correctly.
        if lasair is None:
            msg = (
                "Error importing 'lasair'.\n"
                f"    Fix broken module with \033[32 python3 -m pip install lasair\033[0m\n"
                f"    or remove {self.name} from config"
            )
            raise ModuleNotFoundError(msg)

        # Set the lasair client correctly
        client_token = self.config["client_token"]
        if client_token is None:
            msg = f"must provide 'client_token' in {self.name} config"
            raise MissingKeysError(msg)

        self.lasair_client = lasair.lasair_client(client_token)

        # Process paths
        self.process_paths(
            parent_path=parent_path, directories=self.REQUIRED_DIRECTORIES
        )

    def process_kafka_config(self) -> None:
        self.kafka_config = self.config.get("kafka", None)
        if self.kafka_config is None:
            logger.info(f"No 'kafka' in {self.name} config: will not listen for alerts")
            return

        missing_keys = check_missing_config_keys(
            self.kafka_config,
            self.REQUIRED_KAFKA_PARAMS,
            raise_exc=True,
            exc_class=BadKafkaConfigError,
        )

        # Make sure that 'topics' is a dict.
        topics = self.kafka_config["topics"]
        if isinstance(topics, str):
            logger.info(f"converted kafka topics 'str' to 'list'")
            self.kafka_config["topics"] = [topics]

        topics = self.kafka_config["topics"]
        if isinstance(topics, list):
            updated_topics = {}
            for topic_name in topics:
                updated_topics[topic_name] = self.DEFAULT_TOPIC_KEYS.copy()
            self.kafka_config["topics"] = updated_topics

        # If any of the topics are str, convert them to dict with DEFAULT_KEYS.
        updated_topics = {}  # Add to new dict, can't update item iter'ing.
        errors = {}
        for topic_name, topic_keys in self.kafka_config["topics"].items():
            bad_keys = check_unexpected_config_keys(
                topic_keys,
                self.DEFAULT_TOPIC_KEYS,
                name=f"{self.name}.{topic_name}.keys",
                warn=False,  # Don't need to warn - it's raised in a minute anyway.
            )
            if bad_keys:
                errors[topic_name] = bad_keys

            # Assume defaults, update what we've got.
            fixed_topic_keys = self.DEFAULT_TOPIC_KEYS.copy()
            fixed_topic_keys.update(topic_keys)  # Make sure that any left out are incl.
            updated_topics[topic_name] = fixed_topic_keys
        self.kafka_config["topics"] = updated_topics

        if errors:
            msg = f"unexpected keys in {self.name}.kafka.topics:\n"
            for t, keys in errors.items():
                msg = msg + f"    - {t}: " + ", ".join(f"'{k}'" for k in keys) + "\n"
            msg = msg + "expect " + ", ".join(f"'{k}'" for k in self.DEFAULT_TOPIC_KEYS)

            raise BadKafkaConfigError(msg)

    ##===== Path methods here =====##

    def get_alert_directory(self, lasair_id: str, mkdir: bool = True):
        alert_dir = self.paths_lookup["alerts"] / str(lasair_id)
        if mkdir:
            alert_dir.mkdir(exist_ok=True, parents=True)
        return alert_dir

    def get_alert_filepath(
        self, lasair_id: str, alert_id: int, mkdir: bool = True, fmt="json"
    ):
        alert_dir = self.get_alert_directory(lasair_id, mkdir=mkdir)
        return alert_dir / f"{alert_id}.{fmt}"

    def get_cutouts_directory(self, lasair_id: str, mkdir: bool = True):
        cutouts_dir = self.paths_lookup["cutouts"] / str(lasair_id)
        if mkdir:
            cutouts_dir.mkdir(exist_ok=True, parents=True)
        return cutouts_dir

    def get_cutouts_filepath(
        self, lasair_id: str, alert_id: int, mkdir: bool = True, fmt="pkl"
    ):
        cutouts_dir = self.get_cutouts_directory(lasair_id, mkdir=mkdir)
        return cutouts_dir / f"{alert_id}.{fmt}"

    def get_lightcurve_filepath(self, lasair_id: str, fmt="csv"):
        return self.paths_lookup["lightcurves"] / f"{lasair_id}.{fmt}"

    def get_sherlock_filepath(self, lasair_id: str, fmt="json"):
        return self.paths_lookup["sherlock"] / f"{lasair_id}.{fmt}"

    ##===== Generic methods for Lasair here =====##

    def listen_for_alerts(self, t_ref: Time = None) -> list[dict]:
        t_ref = t_ref or Time.now()
        if self.kafka_config is None:
            logger.info("No kafka config - skip alerts")
            return []

        n_alerts = self.config["n_alerts"]
        timeout = self.config["timeout"]

        server = self.kafka_config["server"]
        group_id = self.kafka_config["group_id"]

        alerts = []
        for topic, topic_keys in self.kafka_config["topics"]:
            topic_alerts = []
            consumer = lasair.lasair_consumer(server, group_id, topic)
            logger.info(f"listen to up to {n_alerts} from {topic}")
            for ii in range(n_alerts):
                msg: Message = consumer.poll(timeout=timeout)
                if msg is None:
                    logger.info(f"Break after {len(topic_alert)}")
                    break
                if msg.error():
                    logger.error(f"Kafka error! \n{msg.error()}")
                    logger.info(f"break after {len(topic_alerts)}")

                processed_alert = self.process_single_alert(msg, topic_keys)
                if processed_alert is not None:
                    topic_alerts.append(processed_alert)

            logger.info(f"{len(topic_alerts)} from {topic}")
        alerts.extend(topic_alerts)
        logger.info(f"{len(alerts)} from all topics")
        return alerts
