from __future__ import annotations

import gzip
import io
import json
import pickle
from logging import getLogger
from pathlib import Path
from typing import NoReturn

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

from aas2rto.exc import MissingCoordinatesError
from aas2rto.query_managers.fink.fink_base import BaseFinkQueryManager, FinkAlert
from aas2rto.query_managers.fink.fink_portal_client import FinkZTFPortalClient
from aas2rto.target import Target
from aas2rto.utils import check_missing_config_keys

logger = getLogger(__name__.split(".")[-1])


class FinkZTFQueryManager(BaseFinkQueryManager):
    name = "fink_ztf"
    id_resolving_order = ("ztf", "ztf_fink", "tns")
    target_id_key = "objectId"
    alert_id_key = "candid"
    portal_client_class = FinkZTFPortalClient
    # DO NOT init portal_client here: happens in QM init, in case credentials needed...

    def process_single_alert(self, alert_data: FinkAlert, t_ref: Time = None) -> dict:
        topic, data, key = alert_data
        fink_id = data["objectId"]
        candid = data["candid"]
        return process_ztf_alert(
            alert_data,
            alert_filepath=self.get_alert_filepath(fink_id, candid),
            cutouts_filepath=self.get_cutouts_filepath(fink_id, candid),
            t_ref=t_ref,
        )

    def add_target_from_alert(self, processed_alert: dict, t_ref=None) -> Target:
        t_ref = t_ref or Time.now()

        target = target_from_ztf_alert(processed_alert, t_ref=t_ref)
        self.target_lookup.add_target(target)
        return target

    def add_target_from_record(self, query_record: dict, t_ref: Time = None) -> Target:
        # Actually, the result of a ZTF classifier query
        # looks just like a processed alert...
        # so we can just re-use that function!
        return self.add_target_from_alert(processed_alert=query_record, t_ref=t_ref)

    def apply_updates_from_alert(self, processed_alert: dict, t_ref=None) -> NoReturn:
        t_ref = t_ref or Time.now()

        fink_id = processed_alert["objectId"]

        target = self.target_lookup.get(fink_id, None)
        if target is None:
            logger.warning(f"No target {fink_id} - can't apply updates!")
            return
        apply_ztf_updates_to_target(target, processed_alert, t_ref=t_ref)

    def process_fink_lightcurve(self, unprocessed_lc: pd.DataFrame) -> pd.DataFrame:
        return process_ztf_lightcurve(unprocessed_lc)

    def load_missing_alerts_for_target(self, fink_id: str) -> Target | None:
        target = self.target_lookup.get(fink_id, None)
        if target is None:
            return False

        fink_data = target.get_target_data(self.name)
        alert_directory = self.get_alert_directory(fink_id, mkdir=False)
        if not alert_directory.exists():
            return False  # There obviously aren't any alerts here.

        available_alert_files = alert_directory.glob("*.json")
        available_alert_ids = [int(a.stem) for a in available_alert_files]

        if fink_data.lightcurve is None:
            loaded_alert_ids = []
        else:
            loaded_alert_ids = fink_data.lightcurve[self.alert_id_key].values
        unloaded_alert_ids = set(available_alert_ids) - set(loaded_alert_ids)
        if not unloaded_alert_ids:
            return False

        new_alerts = []
        for alert_id in unloaded_alert_ids:
            alert_filepath = self.get_alert_filepath(fink_id, alert_id, mkdir=False)
            with open(alert_filepath, "r") as f:
                alert = json.load(f)
            new_alerts.append(alert)

        new_alerts_df = pd.DataFrame(new_alerts)
        if fink_data.lightcurve is None:
            updated_lc = new_alerts_df
        else:
            updated_lc = pd.concat([fink_data.lightcurve, new_alerts_df])
        fink_data.add_lightcurve(updated_lc)
        return True

    def load_cutouts_for_alert(self, fink_id: int, alert_id: int):
        cutouts_filepath = self.get_cutouts_filepath(fink_id, alert_id, mkdir=False)
        if not cutouts_filepath.exists():
            return None

        with open(cutouts_filepath, "rb") as f:
            cutouts = pickle.load(f)
        return cutouts


def target_from_ztf_alert(processed_alert: dict, t_ref: Time = None):

    target_id = processed_alert["objectId"]
    ra = processed_alert.get("ra", None)
    dec = processed_alert.get("dec", None)
    if (ra is None) or (dec is None):
        raise MissingCoordinatesError(f"ra: {ra} or dec: {dec} is None!")

    alt_ids = {"ztf": target_id, "fink_ztf": target_id}
    coord = SkyCoord(ra=ra, dec=dec, unit=u.deg)
    return Target(target_id, coord, source="fink_ztf", alt_ids=alt_ids, t_ref=t_ref)


def process_ztf_lightcurve(unprocessed_lc: pd.DataFrame) -> pd.DataFrame:
    lightcurve = unprocessed_lc.copy(deep=True)

    if lightcurve.empty:
        return pd.DataFrame(columns="objectId mjd jd candid tag".split())

    lightcurve.sort_values("jd", inplace=True, ignore_index=True)
    mjd = Time(lightcurve["jd"], format="jd").mjd
    lightcurve.insert(1, "mjd", mjd)
    if "candid" not in lightcurve.columns:
        lightcurve.loc[:, "candid"] = -1  # INTEGER!
    return lightcurve


def apply_ztf_updates_to_target(
    target: Target, processed_alert: dict, t_ref: Time = None
):
    t_ref = t_ref or Time.now()

    fink_id = processed_alert["objectId"]
    topic = processed_alert["topic"]
    candid = processed_alert["candid"]
    mag = processed_alert["magpsf"]
    band = processed_alert["fid"]
    mjd = processed_alert["mjd"]
    t_str = Time(mjd, format="mjd").strftime("%y-%m-%d %H:%M")

    msg = (
        f"New FINK-ZTF alert for {target.target_id} ({fink_id}) from {topic}!"
        f"\n    mag={mag:.1f} in band '{band}' at {t_str} (alert_id={candid})"
    )
    target.updated = True
    target.info_messages.append(msg)
    return


extra_ztf_alert_keys = (
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


def process_ztf_alert(
    alert_data: tuple[str, dict, str],
    alert_filepath: Path = None,
    cutouts_filepath: Path = None,
    t_ref: Time = None,
):
    topic, data, key = alert_data  # Unpack tuple...

    fink_id: str = data["objectId"]
    alert: dict = data["candidate"]
    candid: int = data["candid"]

    alert["objectId"] = fink_id
    alert["topic"] = topic
    alert["tag"] = "valid"  # must be valid if it's arrived as an alert
    if "mjd" not in alert:
        alert["mjd"] = Time(alert["jd"], format="jd").mjd
    alert["candid"] = candid  # the long ~20 digit ID number.

    extra_data = {k: data[k] for k in extra_ztf_alert_keys if k in data}
    check_missing_config_keys(
        data, extra_ztf_alert_keys, name=f"fink.{topic}.{fink_id}.{candid}"
    )
    alert.update(extra_data)

    if alert_filepath is not None:
        with open(alert_filepath, "w+") as f:
            json.dump(alert, f)

    if cutouts_filepath is not None:
        cutouts = {}
        for imtype in FinkZTFPortalClient.imtypes:
            cutout_data = data.get("cutout" + imtype, {}).get("stampData", None)
            if cutout_data is None:
                continue
            cutout = readstamp(cutout_data, return_type="array")
            cutouts[imtype.lower()] = cutout
        if len(cutouts) > 0:
            with open(cutouts_filepath, "wb+") as f:
                pickle.dump(cutouts, f)
    return alert


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
