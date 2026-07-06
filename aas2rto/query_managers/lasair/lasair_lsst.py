from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from confluent_kafka.cimpl import Message  # Only for type hints

from aas2rto.target import Target
from aas2rto.query_managers.registry import qm_registry
from aas2rto.query_managers.lasair.lasair_base import LasairBaseQueryManager

LSST_TARGET_ID_KEY = "diaObjectId"
LSST_ALERT_ID_KEY = "diaSourceId"


@qm_registry.register()
class LasairLSSTQueryManager(LasairBaseQueryManager):
    name = "lasair_lsst"
    target_id_key = LSST_TARGET_ID_KEY
    alert_id_key = LSST_ALERT_ID_KEY
    ra_key = "ra"
    dec_key = "decl"
    id_resolving_order = ("lasair_lsst", "broker_lsst")
    lightcurve_key = "diaSourceList"
    lasair_client_endpoint = "https://api.lasair.lsst.ac.uk/api"

    def new_target_from_alert(self, processed_alert: dict, t_ref: Time = None):
        """Called by in LasairBaseQM.new_targets_from_alerts"""

        lsst_id = str(processed_alert[self.target_id_key])
        if lsst_id in self.target_lookup:
            return

        ra = processed_alert.get(self.ra_key, None)
        dec = processed_alert.get(self.dec_key, None)

        if ra is not None and dec is not None:
            coord = SkyCoord(ra, dec, unit=u.deg)
        else:
            coord = None

        alt_ids = {"lasair_lsst": lsst_id, "broker_lsst": lsst_id}
        return Target(lsst_id, coord, alt_ids=alt_ids)

    def apply_updates_from_alert(self, processed_alert: dict, t_ref=None):

        lsst_id = str(processed_alert[self.target_id_key])

        target = self.target_lookup.get(lsst_id)
        if target is None:
            msg = f"Lasair alert with {self.target_id_key}={lsst_id}: target not found!"
            self.logger.warning(msg)
            return

        topic = processed_alert.get("topic", "<unknown-topic>")

        msg = f"New Lasair-LSST alert for {target.target_id} ({lsst_id}) from {topic}!"
        target.updated = True
        target.info_messages.append(msg)
        return

    def process_lasair_object_data(self, lasair_object_data: dict, t_ref: Time = None):
        t_ref = Time.now()

        lasair_id: str = str(lasair_object_data[self.target_id_key])

        process_lasair_lsst_object_data(
            lasair_object_data,
            object_data_filepath=self.get_object_data_filepath(lasair_id),
            lightcurve_filepath=self.get_lightcurve_filepath(lasair_id),
            forced_photom_filepath=self.get_forced_photom_filepath(lasair_id),
            image_urls_filepath=self.get_cutout_urls_filepath(lasair_id),
            lasair_context_filepath=self.get_lasair_context_filepath(lasair_id),
        )


def apply_lasair_lsst_updates_to_target(
    target: Target, processed_alert: dict, t_ref: Time = None
):
    t_ref = t_ref or Time.now()

    lasair_id = str(processed_alert[LSST_TARGET_ID_KEY])


def process_lasair_lsst_object_data(
    target_data,
    object_data_filepath: Path = None,
    lightcurve_filepath: Path = None,
    forced_photom_filepath: Path = None,
    image_urls_filepath: Path = None,
    lasair_context_filepath: Path = None,
):

    # Object data
    if object_data_filepath:
        object_data: dict = target_data["diaObject"]
        with open(object_data_filepath, "w+") as f:
            json.dump(object_data, f)

    lightcurve_records: dict = target_data.get("diaSourcesList", None)
    # Also keep track of the alert_ids to dump some info later.
    alert_id_lookup = {}
    if lightcurve_records:
        alert_id_lookup = {rec[LSST_ALERT_ID_KEY]: rec for rec in lightcurve_records}

    # Keep the lightcurves
    if lightcurve_filepath:
        lightcurve = pd.DataFrame(lightcurve_records)
        lightcurve.to_csv(lightcurve_filepath, index=False)

    # Keep forced photom. filepath
    if forced_photom_filepath:
        forced_photom_records = target_data["diaForcedSourcesList"]
        forced_photom = pd.DataFrame(forced_photom_records)
        forced_photom.to_csv(forced_photom_filepath, index=False)

    # Keep the image URLs.
    if image_urls_filepath:
        image_urls_records: dict = target_data["lasairData"].pop("imageUrls")
        if image_urls_records and len(alert_id_lookup) > 0:
            for rec in image_urls_records:
                alert_id = rec[LSST_ALERT_ID_KEY]
                rec["band"] = alert_id_lookup[alert_id]["band"]
                rec["mjd"] = alert_id_lookup[alert_id]["midpointMjdTai"]

        image_urls = pd.DataFrame(image_urls_records)
        image_urls.to_csv(image_urls_filepath, index=False)

    # Keep the lasair data
    if lasair_context_filepath:
        lasair_context = target_data["lasairData"]
        with open(lasair_context_filepath, "w+") as f:
            json.dump(lasair_context, f, indent=2)
