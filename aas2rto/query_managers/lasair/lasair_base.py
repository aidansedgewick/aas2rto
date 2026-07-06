from __future__ import annotations

import abc
import json
import pickle
import requests
from io import BytesIO
from logging import getLogger
from pathlib import Path

from confluent_kafka import Consumer
from confluent_kafka.cimpl import Message

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

from aas2rto.exc import BadKafkaConfigError, MissingKeysError
from aas2rto.query_managers.base import (
    BaseQueryManager,
    LightcurveQueryManager,
    KafkaQueryManager,
)
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup
from aas2rto.utils import (
    QueryTracker,
    calc_file_age,
    # check_safe_to_query,
    check_missing_config_keys,
    check_unexpected_config_keys,
    chunk_list,
)

try:
    import lasair
except ModuleNotFoundError as e:
    lasair = None


logger = getLogger(__name__.split(".")[-1])


# Do NOT register LasairBaseQueryManager with @qm_registry.register
# Re-inheirt from abc.ABC to indicate LasairBaseQM should be subclasses...
class LasairBaseQueryManager(
    KafkaQueryManager, LightcurveQueryManager, BaseQueryManager, abc.ABC
):

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

    @property
    @abc.abstractmethod
    def ra_key(self) -> str:
        """The key for 'R.A.' for different surveys

        eg. 'ramean' for ZTF, 'ra' for LSST"""

    @property
    @abc.abstractmethod
    def dec_key(self) -> str:
        """The key for 'R.A.' for different surveys

        eg. 'decmean' for ZTF, 'decl' for LSST"""

    @property
    @abc.abstractmethod
    def lightcurve_key(self) -> str:
        """The key used to pick the 'lightcurve' data from Lasair 'object' request

        eg. 'candidates' for ZTF, 'diaSourcesList' for LSST"""

    # @property
    # @abc.abstractmethod
    # def object_data_key(self) -> str:
    #     """The key used to pick 'object' data from Lasair 'object' request

    #     eg. 'objectData' for ZTF, 'diaObject' for LSST"""

    # @property
    # @abc.abstractmethod
    # def image_urls_key(self) -> str:
    #     """The key used to pick 'image URLs' from Lasair 'object' request

    #     eg. 'image_urls' from ZTF, 'imageUrls' from LSST"""

    @property
    @abc.abstractmethod
    def lasair_client_endpoint(self) -> str:
        """The default endpoint used for the client in different surveys.

        Can additionally be read from user-provided config as 'client_endpoint'."""

    ##===== Methods for subclass to define here =====##
    # ...in addition to those required by BaseQM,

    @abc.abstractmethod
    def apply_updates_from_alert(
        self, processed_alert: dict, t_ref: Time = None
    ) -> None:
        """apply_updates_from_alert(self, processed_alert: dict, t_ref: Time):

        Accept a processed alert (from the process_single_alert() method),
        which you can use to mark any targets as updated, and add messages
        that will be sent out by eg. slack or telegram."""

    @abc.abstractmethod
    def process_lasair_object_data(self, target_data: dict, t_ref: Time = None) -> None:
        """process_lasair_object_data(self, target_data: dict, t_ref: Time=None)

        Process with the result of `lasair_client.object` request.
        The layout of the result is sufficiently different for ZTF, LSST that it needs
        to be implemented differently for each subclass.

        You should save lightcurve, object_data, image_urls, etc."""

    ##===== Some default config stuff here =====##

    default_tasks = ("alerts", "object_queries", "data_queries", "cutouts")

    @property  # Use property here, to lookup class attrs at run time.
    def default_topic_keys(self) -> dict:
        return {
            "id_key": self.target_id_key,  # Use abstract property (defined in subclasses)
            "ra_key": self.ra_key,
            "dec_key": self.dec_key,
        }

    additional_topic_keys: dict = None

    required_directories = (
        "alerts",
        "cutouts",
        "lightcurves",
        "forced_photom",
        "object_data",
        "lasair_context",
    )
    required_kafka_params = ("group.id", "bootstrap.servers", "topics")
    additional_kafka_params = {
        "default.topic.config": {"auto.offset.reset": "smallest"}
    }
    default_config = {
        "client_token": None,
        "client_endpoint": None,
        "client_lite": True,
        "kafka": None,
        "max_query_time": 180.0,
        "max_failed_queries": 10,
        "n_alerts": 25,
        "n_cutouts": 1,
        "cutouts_timeout": 5.0,
        "timeout": 20.0,
        "stale_file_age": 60.0,
        "tasks": default_tasks,
    }

    ##===== Initialisation here =====##

    def __init__(
        self, config: dict, target_lookup: TargetLookup, parent_path: Path = None
    ):
        # Combine default_config and config, set target_lookup, populate paths_lookup
        super().__init__(config, target_lookup, parent_path)

        # Get kafta cofg
        self.process_kafka_config()

        # Check 'lasair' tools imported correctly.
        if lasair is None:
            msg = (
                "Error importing 'lasair'.\n"
                f"    Fix broken module with \033[32 python3 -m pip install lasair\033[0m\n"
                f"    or remove {self.name} from config"
            )
            raise ModuleNotFoundError(msg)

        self.initialize_lasair_client()

        self.tasks = self.config.get("tasks")
        check_unexpected_config_keys(
            self.tasks,
            self.default_tasks,
            name=f"{self.name}.tasks",
            raise_exc=True,
        )

    def process_kafka_config(self) -> None:
        self.kafka_config = self.config.get("kafka", None)
        if self.kafka_config is None:
            msg = f"No 'kafka' in {self.name} config: will not listen for alerts"
            self.logger.info(msg)
            return

        missing_keys = check_missing_config_keys(
            self.kafka_config,
            self.required_kafka_params,
            raise_exc=True,
            exc_class=BadKafkaConfigError,
        )

        # Make sure that 'topics' is a dict.
        topics = self.kafka_config["topics"]
        if isinstance(topics, str):
            self.logger.info(f"converted kafka topics 'str' to 'list'")
            self.kafka_config["topics"] = [topics]

        topics = self.kafka_config["topics"]
        if isinstance(topics, list):
            updated_topics = {}
            for topic_name in topics:
                updated_topics[topic_name] = self.default_topic_keys.copy()
            self.kafka_config["topics"] = updated_topics

        # If any of the topics are str, convert them to dict with DEFAULT_KEYS.
        updated_topics = {}  # Add to new dict, can't update item iter'ing.
        errors = {}

        for topic_name, topic_keys in self.kafka_config["topics"].items():
            bad_keys = check_unexpected_config_keys(
                topic_keys,
                self.default_topic_keys,
                name=f"{self.name}.{topic_name}.keys",
                warn=False,  # Don't need to warn - it's raised in a minute anyway.
            )
            if bad_keys:
                errors[topic_name] = bad_keys

            # Assume defaults, update what we've got.
            fixed_topic_keys = self.default_topic_keys.copy()
            fixed_topic_keys.update(topic_keys)  # Make sure that any left out are incl.
            updated_topics[topic_name] = fixed_topic_keys
        self.kafka_config["topics"] = updated_topics

        if errors:
            msg = f"unexpected keys in {self.name}.kafka.topics:\n"
            for t, keys in errors.items():
                msg = msg + f"    - {t}: " + ", ".join(f"'{k}'" for k in keys) + "\n"
            msg = msg + "expect " + ", ".join(f"'{k}'" for k in self.default_topic_keys)

            raise BadKafkaConfigError(msg)

    def initialize_lasair_client(self):
        client_token = self.config["client_token"]
        if client_token is None:
            msg = f"must provide 'client_token' in {self.name} config"
            raise MissingKeysError(msg)

        endpoint = self.config["client_endpoint"]
        if endpoint is None:
            endpoint = self.lasair_client_endpoint
            msg = f"Use default lasair_client endpoint:\n    {endpoint}"
            self.logger.info(msg)
        else:
            msg = f"Use user-provided lasair_client endpoint:\n    {endpoint}"
            self.logger.info(msg)

        self.lasair_client = lasair.lasair_client(client_token, endpoint=endpoint)

    ##===== Path methods here =====##

    def get_alert_directory(self, lasair_id: str, topic: str, mkdir: bool = True):
        alert_dir = self.paths_lookup["alerts"] / topic / str(lasair_id)
        if mkdir:
            alert_dir.mkdir(exist_ok=True, parents=True)
        return alert_dir

    def get_alert_filepath(
        self, lasair_id: str, topic: str, t_str: str, mkdir: bool = True, fmt="json"
    ):
        """
        Use t_str - alert_id (ie. 'candid' or 'diaSourceId') is not provided with lasair
        alerts.
        """
        alert_dir = self.get_alert_directory(lasair_id, topic, mkdir=mkdir)
        return alert_dir / f"{t_str}.{fmt}"

    def get_object_data_filepath(self, lasair_id: str, fmt: str = "json"):
        return self.paths_lookup["object_data"] / f"{lasair_id}.{fmt}"

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

    def get_cutout_urls_filepath(self, lasair_id: str, mkdir: bool = True, fmt="csv"):
        cutout_urls_dir = self.paths_lookup["cutouts"] / "cutout_urls"
        if mkdir:
            cutout_urls_dir.mkdir(exist_ok=True, parents=True)
        return cutout_urls_dir / f"{lasair_id}_cutout_urls.csv"

    def get_lightcurve_filepath(self, lasair_id: str, fmt="csv"):
        return self.paths_lookup["lightcurves"] / f"{lasair_id}.{fmt}"

    def get_forced_photom_filepath(self, lasair_id: str, fmt="csv"):
        return self.paths_lookup["forced_photom"] / f"{lasair_id}.{fmt}"

    def get_lasair_context_filepath(self, lasair_id: str, fmt="json"):
        return self.paths_lookup["lasair_context"] / f"{lasair_id}.{fmt}"

    ##===== Generic methods for Lasair here =====##

    def listen_for_alerts(self, t_ref: Time = None) -> list[dict]:
        t_ref = t_ref or Time.now()
        if self.kafka_config is None:
            self.logger.info("No kafka config - skip alerts")
            return []

        n_alerts = self.config["n_alerts"]
        timeout = self.config["timeout"]

        consumer_config = {
            "bootstrap.servers": self.kafka_config["bootstrap.servers"],
            "group.id": self.kafka_config["group.id"],
            **self.additional_kafka_params,
        }

        alerts = []
        for topic, topic_keys in self.kafka_config["topics"].items():
            topic_alerts = []
            self.logger.info(f"listen to up to {n_alerts} alerts from '{topic}'")
            # currently prefer direct confluent_kafka.Consumer() with context mgr
            # over lasair.lasair_consumer()
            with Consumer(consumer_config) as consumer:
                consumer.subscribe([topic])
                for ii in range(n_alerts):
                    msg: Message = consumer.poll(timeout=timeout)
                    if msg is None:
                        self.logger.info(f"Break after {len(topic_alerts)}")
                        break
                    if msg.error():
                        self.logger.error(f"Kafka error! \n{msg.error()}")
                        break

                    processed_alert = self.process_single_alert(
                        msg, topic_keys, t_ref=t_ref
                    )
                    if processed_alert is not None:
                        topic_alerts.append(processed_alert)

            self.logger.info(f"{len(topic_alerts)} from {topic}")
            alerts.extend(topic_alerts)
        self.logger.info(f"{len(alerts)} from all topics")
        return alerts

    def process_single_alert(
        self, message: Message, topic_keys: dict, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        ##== Decode the message, add basic information
        alert: dict = json.loads(message.value())

        alert_topic = message.topic()
        alert["topic"] = alert_topic

        lasair_id_key = topic_keys["id_key"]
        lasair_id = str(alert[lasair_id_key])
        if lasair_id_key != self.target_id_key:
            alert[self.target_id_key] = lasair_id

        ##== Check to see if we have co-ordinate information.
        topic_ra_key = topic_keys["ra_key"]
        topic_dec_key = topic_keys["dec_key"]

        if topic_ra_key not in alert or topic_dec_key not in alert:
            coordinates_missing = True
        else:
            coordinates_missing = False
            if topic_ra_key != self.ra_key or topic_dec_key != topic_dec_key:
                # Standardize the coordinate keys for later...
                alert[self.ra_key] = alert[topic_ra_key]
                alert[self.dec_key] = alert[topic_dec_key]

        ##== Get the alert timestamp (or t_ref, if it fails.)
        ts_type, timestamp_ms = message.timestamp()  # Time in ms since epoch 1970
        if timestamp_ms is not None:
            timestamp = Time(timestamp_ms / 1000, format="unix")
        else:
            timestamp = t_ref
        alert_timestamp = t_ref.strftime("%Y%m%d_%H%M%S")
        alert["alert_mjd"] = timestamp.mjd

        ##== Remove (large) lightcurve data from the alert before dumping.
        lightcurve_records: list[dict] = alert.pop(self.lightcurve_key, None)
        if lightcurve_records is not None:
            lightcurve = pd.DataFrame(lightcurve_records)
            lightcurve_filepath = self.get_lightcurve_filepath(lasair_id)
            lightcurve.to_csv(lightcurve_filepath, index=False)

            if coordinates_missing:
                ra = lightcurve_records[-1].get(self.ra_key)
                dec = lightcurve_records[-1].get(self.dec_key)
                alert[self.ra_key] = ra
                alert[self.dec_key] = dec

        ##== Finally dump the alert
        alert_filepath = self.get_alert_filepath(
            lasair_id, alert_topic, alert_timestamp
        )
        with open(alert_filepath, "w+") as f:
            json.dump(alert, f)
        return alert

    def query_target_data(self, lasair_id_list: list[str] = None, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        if lasair_id_list is None:
            lasair_id_list = self.select_lightcurves_to_query()
        self.logger.info(f"attempt to query {len(lasair_id_list)} objects")

        qtracker = QueryTracker.start(
            self.config["max_query_time"], self.config["max_failed_queries"]
        )
        for lasair_id in lasair_id_list:
            if not qtracker.safe_to_query():
                self.logger.info("Stop query lightcurves for now.")
                break
            try:
                target_data = self.lasair_client.object(
                    lasair_id, lite=self.config["client_lite"]
                )
            except lasair.lasair.LasairError as e:
                logger.error(e)
                qtracker.track_failed(lasair_id)
                continue

            self.process_lasair_object_data(target_data, t_ref=t_ref)
            qtracker.track_success(lasair_id)

        qtracker.log_summary(name=f"{self.name} lightcurves")
        return qtracker.success, qtracker.missing, qtracker.failed

    def load_missing_coord_information(self):
        coords_fixed = []
        for target_id, target in self.target_lookup.items():
            lasair_id = self.get_relevant_id_from_target(target)
            if lasair_id is None:
                continue

            if target.coord is None:
                self.update_target_coord(lasair_id)
                coords_fixed.append(target_id)

        self.logger.info(f"Update coord info for {len(coords_fixed)} targets")
        return coords_fixed

    def update_target_coord(self, lasair_id: str):
        target = self.target_lookup[lasair_id]
        object_data_filepath = self.get_object_data_filepath(lasair_id)
        with open(object_data_filepath) as f:
            data = json.load(f)
        coord = SkyCoord(ra=data[self.ra_key], dec=data[self.dec_key], unit=u.deg)
        target.update_coordinates(coord)

    def load_single_lightcurve(self, target_id: int, t_ref: Time = None):

        target = self.target_lookup.get(target_id, None)
        if target is None:
            return None

        lasair_id = self.get_relevant_id_from_target(target)

        lightcurve_filepath = self.get_lightcurve_filepath(lasair_id)
        if not lightcurve_filepath.exists():
            return None
        return pd.read_csv(lightcurve_filepath)

    def query_latest_cutouts(self):
        # TODO: success/fail tracking better here...

        qtracker = QueryTracker.start(max_failed_queries=10, max_query_time=180.0)

        self.logger.info("query latest cutouts")
        for target_id, target in self.target_lookup.items():
            lasair_id = self.get_relevant_id_from_target(target)
            if lasair_id is None:
                continue

            if not qtracker.safe_to_query():
                self.logger.info("stop cutout queries for now")
                break
            try:
                success, missing, failed = self.query_latest_target_cutouts(lasair_id)
            except requests.exceptions.ConnectionError as e:
                qtracker.track_failed(lasair_id)
                continue
            except requests.exceptions.Timeout:
                qtracker.track_failed(lasair_id)
                continue
            if missing:
                qtracker.track_missing(missing)
            if success:
                qtracker.track_success(success)
            if failed:
                qtracker.track_failed(failed)

        qtracker.log_summary(name=f"{self.name} cutouts")
        return qtracker.success, qtracker.missing, qtracker.failed

    def query_latest_target_cutouts(self, lasair_id: str):

        cutout_urls_filepath = self.get_cutout_urls_filepath(lasair_id, mkdir=False)

        qtracker = QueryTracker.start(
            self.config["max_query_time"], self.config["max_failed_queries"]
        )
        if not cutout_urls_filepath.exists():
            qtracker.track_missing(lasair_id)
            return qtracker.success, qtracker.missing, qtracker.failed

        cutout_data = pd.read_csv(cutout_urls_filepath)
        cutout_data.sort_values("mjd", inplace=True, ascending=False)  # NEWEST first!

        n_cutouts = min(self.config["n_cutouts"], len(cutout_data))
        for ii in range(n_cutouts):
            row = cutout_data.iloc[ii]  # Take the top N
            alert_id = row[self.alert_id_key]

            cutout_id = f"{lasair_id}-{alert_id}"

            cutouts_filepath = self.get_cutouts_filepath(lasair_id, alert_id)
            if cutouts_filepath.exists():
                self.logger.debug(f"{cutout_id} cutouts exist")
                continue

            logger.info(f"try {lasair_id} cutouts")

            cutouts = {}
            for imtype in ["Science", "Template", "Difference"]:
                url = row[imtype]
                try:
                    im_cutout = request_cutout(url)
                except requests.exceptions.HTTPError as e:
                    break
                cutouts[imtype.lower()] = im_cutout
                if im_cutout is None:
                    break

            if len(cutouts) != 3:
                qtracker.track_failed(alert_id)
                continue

            cutouts["meta"] = {"mjd": row["mjd"], "band": row["band"]}

            with open(cutouts_filepath, "wb+") as f:
                pickle.dump(cutouts, f)
            qtracker.track_success(cutout_id)
            self.logger.debug(f"{lasair_id} cutouts success")
        return qtracker.success, qtracker.missing, qtracker.failed

    def load_cutouts(self, t_ref: Time = None):
        loaded_cutouts = []
        missing_cutouts = []
        skipped_reload = []
        for target_id, target in self.target_lookup.items():
            lasair_id = self.get_relevant_id_from_target(target)
            if lasair_id is None:
                continue
            lasair_data = target.target_data.get(self.name, None)
            if lasair_data is None:
                continue
            if self.alert_id_key not in lasair_data.lightcurve:
                highlight = f"'{self.alert_id_key}' not in {self.name}_data.lightcurve"
                msg = f"{target_id}: for {self.name} \033[33;1m{highlight}\033[0m"
                self.logger.warning(msg)
                continue

            current_cutouts_id = lasair_data.meta.get("cutouts_alert_id", -1)
            for alert_id in lasair_data.lightcurve[self.alert_id_key].values[::-1]:
                # Step through alert_id from most recent to oldest.
                if alert_id < 0:
                    continue
                if alert_id == current_cutouts_id:
                    skipped_reload.append(target_id)
                    break  # We've already loaded this one.
                cutouts = self.load_cutouts_for_alert(lasair_id, alert_id)
                if cutouts is not None:
                    lasair_data.cutouts = cutouts
                    lasair_data.meta["cutouts_alert_id"] = alert_id
                    loaded_cutouts.append(target_id)
                    target.update = True  # re-plot with new cutouts! Don't need to msg.
                    break
            else:
                missing_cutouts.append(target_id)  # Executed if no break statement met!

        N_loaded = len(loaded_cutouts)
        N_skipped = len(skipped_reload)
        self.logger.info(f"loaded cutouts for {N_loaded} targets (skipped {N_skipped})")
        if len(missing_cutouts) > 0:
            self.logger.info(f"missing cutouts for {len(missing_cutouts)}!!")
        return loaded_cutouts, missing_cutouts, skipped_reload

    def load_cutouts_for_alert(self, lasair_id: int, alert_id: int):
        cutouts_filepath = self.get_cutouts_filepath(lasair_id, alert_id, mkdir=False)
        if not cutouts_filepath.exists():
            return None

        with open(cutouts_filepath, "rb") as f:
            cutouts = pickle.load(f)
        return cutouts

    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        if iteration == 0:
            self.logger.info("Skip many tasks on first iteration")
            self.load_target_lightcurves()
            self.load_cutouts()
            return

        if "alerts" in self.tasks:
            processed_alerts = self.listen_for_alerts()
            self.add_targets_from_alerts(processed_alerts, t_ref=t_ref)
            self.add_messages_from_alerts(processed_alerts, t_ref=t_ref)

        if "object_queries" in self.tasks:
            pass  # Currently not implemented...

        if "data_queries" in self.tasks:
            success, missing, failed = self.query_target_data()
            self.load_target_lightcurves()
            self.load_missing_coord_information()

        if "cutouts" in self.tasks:
            self.query_latest_cutouts()
            self.load_cutouts()

        self.clear_stale_files(self.config["stale_file_age"])
        return


def request_cutout(cutout_url: str):

    sci_resp = requests.get(cutout_url)

    if sci_resp.status_code != 200:
        logger.debug(f"cutout status {sci_resp.status_code}: {cutout_url}")
        if sci_resp.status_code in [429]:
            raise requests.exceptions.HTTPError("Cutouts request throttled!")
        return None

    bytestream = BytesIO(sci_resp.content)
    return fits.open(bytestream)[0].data
