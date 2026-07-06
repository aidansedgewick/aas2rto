import json
from pathlib import Path

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from aas2rto.query_managers.lasair.lasair_base import LasairBaseQueryManager
from aas2rto.query_managers.registry import qm_registry
from aas2rto.target import Target

ZTF_TARGET_ID_KEY = "objectId"
ZTF_ALERT_ID_KEY = "candid"

ZTF_BAND_LOOKUP = {1: "ZTF-g", 2: "ZTF-r", 3: "ZTF-i"}


@qm_registry.register()
class LasairZTFQueryManager(LasairBaseQueryManager):
    name = "lasair_ztf"
    target_id_key = ZTF_TARGET_ID_KEY
    alert_id_key = ZTF_ALERT_ID_KEY
    ra_key = "meanra"
    dec_key = "meandec"
    id_resolving_order = ("lasair_ztf", "ztf")
    lightcurve_key = "candidates"
    lasair_client_endpoint = "https://lasair-ztf.lsst.ac.uk/api"

    def new_target_from_alert(self, processed_alert: dict, t_ref=None):
        ztf_id = str(processed_alert[self.target_id_key])

        target = self.target_lookup.get(ztf_id)

        ra = processed_alert.get(self.ra_key, None)
        dec = processed_alert.get(self.dec_key, None)

        if ra is not None and dec is not None:
            coord = SkyCoord(ra, dec, unit=u.deg)
        else:
            coord = None

        alt_ids = {"lasair_ztf": ztf_id, "ztf": ztf_id}
        return Target(ztf_id, coord, alt_ids=alt_ids)

    def apply_updates_from_alert(self, processed_alert: dict, t_ref: Time = None):

        ztf_id = str(processed_alert[self.target_id_key])

        target = self.target_lookup.get(ztf_id)

        if target is None:
            msg = f"Lasair alert with {self.target_id_key}={ztf_id}: target not found!"
            self.logger.warning(msg)
            return

        topic = processed_alert.get("topic", "<unknown-topic>")

        msg = f"New Lasair-ZTF alert for {target.target_id} ({ztf_id}) from {topic}!"
        target.updated = True
        target.info_messages.append(msg)
        return

    def process_lasair_object_data(self, lasair_object_data: dict, t_ref: Time = None):
        t_ref = Time.now()

        lasair_id: str = str(lasair_object_data[self.target_id_key])

        process_lasair_ztf_object_data(
            lasair_object_data,
            object_data_filepath=self.get_object_data_filepath(lasair_id),
            lightcurve_filepath=self.get_lightcurve_filepath(lasair_id),
            forced_photom_filepath=self.get_forced_photom_filepath(lasair_id),
            image_urls_filepath=self.get_cutout_urls_filepath(lasair_id),
            lasair_context_filepath=self.get_lasair_context_filepath(lasair_id),
        )


def process_lasair_ztf_object_data(
    target_data,
    object_data_filepath: Path = None,
    lightcurve_filepath: Path = None,
    forced_photom_filepath: Path = None,
    image_urls_filepath: Path = None,
    lasair_context_filepath: Path = None,
):
    # Object data
    if object_data_filepath:
        object_data: dict = target_data["objectData"]
        with open(object_data_filepath, "w+") as f:
            json.dump(object_data, f)

    # Separate the lightcurve and image urls:
    candidate_records = target_data.get("candidates", None)
    lightcurve_records = []
    image_urls_records = []
    for rec in candidate_records:
        urls_rec = rec.get("image_urls")
        fid = rec["fid"]
        urls_rec["band"] = ZTF_BAND_LOOKUP.get(fid, f"band={fid}")
        urls_rec["mjd"] = Time(rec["jd"], format="jd").mjd
        lightcurve_rec = {k: v for k, v in rec.items() if k != "image_urls"}

        lightcurve_records.append(lightcurve_rec)
        image_urls_records.append(urls_rec)

    if lightcurve_filepath:
        lightcurve = pd.DataFrame(lightcurve_records)
        lightcurve.to_csv(lightcurve_filepath, index=False)

    if image_urls_filepath:
        image_urls = pd.DataFrame(image_urls_records)
        image_urls.to_csv(image_urls_filepath, index=False)

    if forced_photom_filepath:
        forced_phot_records = target_data["forcedphot"]
        forced_phot = pd.DataFrame(forced_phot_records)
        forced_phot.to_csv(forced_photom_filepath, index=False)

    lasair_context = target_data.get("sherlock", None)
    if lasair_context_filepath and lasair_context:
        with open(lasair_context_filepath, "w+") as f:
            json.dump(lasair_context, f)
