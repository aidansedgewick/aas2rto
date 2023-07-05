import gzip
import io
import json
import os
import pickle
import requests
import time
from logging import getLogger
from typing import Dict, List, Set, Tuple

import numpy as np

import pandas as pd

from astropy import utils as u
from astropy.io import fits
from astropy.time import Time


try:
    import fink_client
    from fink_client.avroUtils import write_alert, _get_alert_schema, AlertReader
    from fink_client.consumer import AlertConsumer
except ModuleNotFoundError as e:
    fink_client = None

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
    if "valid" in np.unique(non_detections["tag"]):
        ldet = len(detections)
        lvalid = len(non_detections.query("tag=='valid'"))
        if not ldet == lvalid:
            msg = f"mismatch : detections ({ldet}) != 'valid' non_detections ({lvalid})"
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
                print(non_detections["candid"])
                raise ValueError("not all non-detections have Null `candid`")
    non_detections["candid"] = 0

    lightcurve = pd.concat([detections, non_detections])
    lightcurve.sort_values("jd", inplace=True)
    return lightcurve


def target_from_fink_alert(alert: dict, objectId: str, t_ref: Time = None) -> Target:
    ra = alert.get("ra", None)
    dec = alert.get("dec", None)
    if (ra is None) or (dec is None):
        raise
    target = Target(objectId, ra=ra, dec=dec)
    return target


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


class FinkQueryManager(BaseQueryManager):
    name = "fink"
    default_query_parameters = {
        "interval": 2.0,
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
                "fink_client module not imported correctly! "
                "either install with `python3 -m pip install fink_client`, "
                "or switch `use: False` in config."
            )

        self.kafka_config = self.get_kafka_parameters()

        self.query_parameters = self.default_query_parameters.copy()
        query_params = self.fink_config.get("query_parameters", {})
        self.query_parameters.update(query_params)

        self.lc_queries_to_retry = []
        self.alerts_integrations_to_retry = []

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

        successful_queries = []
        N_failed = 0
        if len(objectId_list) > 0:
            logger.info(f"attempt {len(objectId_list)} lightcurve queries")
        t_start = time.perf_counter()
        for objectId in objectId_list:
            lightcurve_file = self.get_lightcurve_file(objectId)
            if N_failed > self.query_parameters["max_failed_queries"]:
                msg = f"Too many failed queries ({N_failed}), stop for now"
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
                N_failed = N_failed + 1
                continue
            lightcurve = process_fink_lightcurve(detections, non_detections)
            lightcurve.to_csv(lightcurve_file, index=False)
            successful_queries.append(objectId)
        N_success = len(successful_queries)
        if N_success > 0 or N_failed > 0:
            t_end = time.perf_counter()
            logger.info(f"lightcurve queries in {t_end - t_start:.1f}s")
            logger.info(f"{N_success} successful, {N_failed} failed lc queries")
        return successful_queries

    def new_targets_from_alerts(self, alerts: List[Dict]):
        for alert in alerts:
            objectId = alert.get("objectId", None)
            if objectId is None:
                raise MissingObjectIdError("objectId is None in fink_alert")
            target = self.target_lookup.get(objectId, None)
            if target is not None:
                # Don't need to make a target.
                continue
            target = Target(objectId=objectId, ra=alert["ra"], dec=alert["dec"])
            self.target_lookup[objectId] = target

    def load_target_lightcurves(
        self, objectId_list: List[str] = None, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        loaded_lightcurves = []
        missing_lightcurves = []
        t_start = time.perf_counter()

        if objectId_list is None:
            objectId_list = list(self.target_lookup.keys())

        # for objectId, target in self.target_lookup.items():
        for objectId in objectId_list:
            lightcurve_file = self.get_lightcurve_file(objectId)
            if not lightcurve_file.exists():
                missing_lightcurves.append(objectId)
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
                    if len(lightcurve) == len(existing_lightcurve):
                        continue
            lightcurve.sort_values("jd", inplace=True)
            loaded_lightcurves.append(objectId)
            target.fink_data.add_lightcurve(lightcurve)
            target.updated = True
        t_end = time.perf_counter()

        N_loaded = len(loaded_lightcurves)
        if N_loaded > 0:
            logger.info(f"loaded {N_loaded} lightcurves in {t_end-t_start:.1f}s")
        return loaded_lightcurves, missing_lightcurves

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
                        continue
                    with open(cutouts_file, "rb") as f:
                        cutouts = pickle.load(f)
                    target.fink_data.cutouts = cutouts
                    target.fink_data.meta["cutouts_candid"] = candid
                    loaded_cutouts.append(objectId)
        logger.info(f"{len(loaded_cutouts)} cutouts loaded")

    def apply_messenger_updates(self, alerts):
        for alert in alerts:
            objectId = alert["objectId"]
            target = self.target_lookup.get(objectId, None)
            if target is None:
                continue
            target.send_updates = True
            topic_str = alert["topic"]
            alert_text = f"FINK alert {topic_str}\nsee fink-portal.org/{objectId}"
            target.update_messages.append(alert_text)

    def perform_all_tasks(self, simulated_alerts=False, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        # get alerts
        if simulated_alerts:
            pass
        else:
            alerts = self.listen_for_alerts()
        processed_alerts = self.process_alerts(alerts, t_ref=t_ref)
        self.new_targets_from_alerts(processed_alerts)

        lcs_to_query = self.get_lightcurves_to_query(t_ref=t_ref)
        successful_lc_queries = self.perform_lightcurve_queries(lcs_to_query)

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

    # @classmethod
    # def fix_keys(cls, df):
    #     column_lookup = {
    #         col: col.split(":")[1] if ":" in col else col for col in df.columns
    #     }
    #     return df.rename(column_lookup, axis=1)

    @classmethod
    def fix_dict_keys(cls, data: List[Dict]):
        for row_dict in data:
            key_lookup = {k: k.split(":")[-1] for k in row_dict.keys()}
            for old_key, new_key in key_lookup.items():
                # Not dangerous, as not iterating over the dict we're updating.
                row_dict[new_key] = row_dict.pop(old_key)

    @classmethod
    def process_data(cls, data, fix_keys=True, return_df=True, df_kwargs=None):
        if fix_keys:
            cls.fix_dict_keys(data)
        if not return_df:
            return data
        return pd.DataFrame(data)

    @classmethod
    def process_response(
        cls,
        res,
        fix_keys=True,
        return_df=True,
    ):
        if res.status_code in [404, 500, 504]:
            logger.error("\033[31;1merror rasied\033[0m")
            if res.elapsed.total_seconds() > 58.0:
                logger.error("likely a timeout error")
            raise FinkQueryError(res.content.decode())
        data = json.loads(res.content)
        return cls.process_data(data, fix_keys=fix_keys, return_df=return_df)

    @classmethod
    def do_post(cls, service, process=True, fix_keys=True, return_df=True, **kwargs):
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
