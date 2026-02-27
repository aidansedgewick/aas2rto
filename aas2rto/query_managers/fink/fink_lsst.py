import pickle
import json
from logging import getLogger
from typing import NoReturn

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from aas2rto import utils
from aas2rto.exc import MissingCoordinatesError
from aas2rto.query_managers.fink.fink_base import (
    FinkBaseQueryManager,
    FinkAlert,
    readstamp,
)
from aas2rto.query_managers.fink.fink_portal_client import FinkLSSTPortalClient
from aas2rto.target import Target


logger = getLogger(__name__.split(".")[-1])

LSST_TARGET_ID_KEY = "diaObjectId"
LSST_ALERT_ID_KEY = "diaSourceId"

EXTRA_FINK_LSST_ALERT_KEYS = ()


class FinkLSSTQueryManager(FinkBaseQueryManager):
    name = "fink_lsst"
    id_resolving_order = ("lsst", "fink_lsst", "tns")
    target_id_key = LSST_TARGET_ID_KEY
    alert_id_key = LSST_ALERT_ID_KEY
    portal_client_class = FinkLSSTPortalClient
    # DO NOT init portal_client here: happens in QM init, in case credentials needed...

    config_extras = {"reliability_cutoff": 0.5}

    def process_single_alert(self, alert_data: FinkAlert, t_ref: Time = None):
        topic, data, key = alert_data
        fink_id = data["diaObject"][self.target_id_key]
        alert_id = data["diaSource"][self.alert_id_key]
        return process_fink_lsst_alert(
            alert_data,
            alert_filepath=self.get_alert_filepath(fink_id, alert_id),
            cutouts_filepath=self.get_cutouts_filepath(fink_id, alert_id),
            t_ref=t_ref,
        )

    def new_target_from_alert(self, processed_alert: dict, t_ref: Time = None):
        return target_from_fink_lsst_alert(processed_alert, t_ref=t_ref)

    def new_target_from_record(self, query_record: dict, t_ref: Time = None):
        # Wait to read FINK LSST Docs?
        return target_from_fink_lsst_alert(query_record, t_ref=t_ref)

    def apply_updates_from_alert(
        self, processed_alert: dict, t_ref: Time = None
    ) -> NoReturn:
        fink_id = processed_alert[self.target_id_key]

        target = self.target_lookup.get(fink_id, None)
        if target is None:
            logger.warning(f"No target '{fink_id}' - can't apply updates!")
            return
        apply_fink_lsst_updates_to_target(target, processed_alert, t_ref)

    def process_fink_lightcurve(self, unprocessed_lc: pd.DataFrame):
        return process_fink_lsst_lightcurve(unprocessed_lc)

    def load_missing_alerts_for_target(self, fink_id: str):
        pass
        # There should be no missing alerts in the LC - fink ingests LSST immediately.

    def load_cutouts_for_alert(self, fink_id: str, alert_id: int):
        pass


def process_fink_lsst_alert(
    alert_data: FinkAlert,
    alert_filepath=None,
    cutouts_filepath=None,
    t_ref: Time = None,
):
    topic, data, key = alert_data

    object_data: dict = data["diaObject"]
    alert: dict = data["diaSource"]

    fink_id: int = str(object_data[LSST_TARGET_ID_KEY])
    alert_id: int = alert[LSST_ALERT_ID_KEY]

    # Now modify the alert dict - diaSourceId (alert_id) is already included
    alert[LSST_TARGET_ID_KEY] = fink_id
    alert["topic"] = topic
    alert["mjd"] = alert["midpointMjdTai"]
    alert["tag"] = "valid"  # if it's arrived as an alert, it must be valid...
    alert["nDiaSources"] = object_data["nDiaSources"]

    extra_data = {k: data[k] for k in EXTRA_FINK_LSST_ALERT_KEYS if k in data}
    utils.check_missing_config_keys(
        data, EXTRA_FINK_LSST_ALERT_KEYS, name=f"fink_lsst.{topic}.{fink_id}.{alert_id}"
    )
    alert.update(extra_data)

    # POP any cutouts first, so we don't try to serialise them into JSON - fails!
    cutouts = {}
    for imtype in FinkLSSTPortalClient.imtypes:
        cutout_key = f"cutout{imtype}"
        alert_data = alert.pop(cutout_key, None)
        if alert_data is not None:
            cutout = readstamp(alert_data)
            cutouts[imtype.lower()] = cutout

    if alert_filepath is not None:
        with open(alert_filepath, "w+") as f:
            json.dump(alert, f)

    if cutouts_filepath is not None:
        if len(cutouts) > 0:
            with open(cutouts_filepath, "wb+") as f:
                pickle.dump(cutouts, f)

    return alert


def target_from_fink_lsst_alert(processed_alert: dict, t_ref: Time = None):
    t_ref = t_ref or Time.now()

    target_id = str(processed_alert[LSST_TARGET_ID_KEY])
    ra = processed_alert.get("ra", None)
    dec = processed_alert.get("dec", None)
    if (ra is None) or (dec is None):
        raise MissingCoordinatesError(f"ra: {ra} or dec: {dec} is None!")
    coord = SkyCoord(ra=ra, dec=dec, unit=u.deg)
    alt_ids = {"lsst": target_id, "fink_lsst": target_id}
    return Target(target_id, coord, source="fink_lsst", alt_ids=alt_ids, t_ref=t_ref)


def apply_fink_lsst_updates_to_target(
    target: Target, processed_alert: dict, t_ref: Time = None
) -> NoReturn:
    t_ref = t_ref or Time.now()

    fink_id = str(processed_alert[LSST_TARGET_ID_KEY])
    topic = processed_alert["topic"]
    alert_id = processed_alert[LSST_ALERT_ID_KEY]
    flux = processed_alert["psfFlux"]  # flux is in nJy
    mag = -2.5 * np.log10(flux) + 31.4
    band = processed_alert["band"]
    mjd = processed_alert["mjd"]
    t_str = Time(mjd, format="mjd").strftime("%y-%m-%d %H:%M")

    msg = (
        f"New FINK-LSST alert for {target.target_id} ({fink_id}) from {topic}!"
        f"\n    mag={mag:.1f} in band '{band}' at {t_str} (alert_id={alert_id})"
    )
    target.updated = True
    target.info_messages.append(msg)
    return


def process_fink_lsst_lightcurve(unprocessed_lc: pd.DataFrame):
    lightcurve = unprocessed_lc.copy(deep=True)

    if lightcurve.empty:
        req_columns = "diaObjectId mjd diaSourceId tag".split()
        return pd.DataFrame(columns=req_columns)

    if LSST_ALERT_ID_KEY not in lightcurve.columns:
        lightcurve.loc[:, LSST_ALERT_ID_KEY] = -1

    return lightcurve
