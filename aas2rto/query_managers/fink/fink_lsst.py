import json
from logging import getLogger
from typing import Dict, List, Tuple

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from aas2rto.exc import MissingCoordinatesError
from aas2rto.query_managers.fink.fink_base import FinkBaseQueryManager
from aas2rto.query_managers.fink.fink_query import FinkLSSTQuery
from aas2rto.target import Target


logger = getLogger(__name__.split(".")[-1])


class FinkLSSTQueryManager(FinkBaseQueryManager):
    name = "fink_lsst"
    id_resolving_order = ("lsst", "fink_lsst", "tns")
    fink_query = FinkLSSTQuery
    target_id_key = "diaObjectId"
    alert_id_key = "diaSourceId"

    def process_single_alert(
        self, alert_data: Tuple[str, Dict, str], t_ref: Time = None
    ):
        topic, data, key = alert_data
        fink_id = data["diaObject"]["diaObjectId"]
        alert_id = data["diaSource"]["diaSourceId"]
        return process_lsst_alert(
            alert_data,
            alert_filepath=self.get_alert_filepath(fink_id, alert_id),
            cutout_filepath=self.get_cutouts_filepath(fink_id, alert_id),
            t_ref=t_ref,
        )

    def add_target_from_alert(self, alert: Dict, t_ref: Time = None):
        return target_from_lsst_alert(alert, t_ref=t_ref)

    def add_target_from_record(self, query_record: dict):
        # Wait to read FINK LSST Docs?
        pass


def target_from_lsst_alert(alert: dict, t_ref: Time = None):
    t_ref = t_ref or Time.now()

    target_id = alert["diaObjectId"]
    ra = alert.get("ra", None)
    dec = alert.get("dec", None)
    if (ra is None) or (dec is None):
        raise MissingCoordinatesError(f"ra: {ra} or dec: {dec} is None!")
    coord = SkyCoord(ra=ra, dec=dec, unit=u.deg)
    alt_ids = {"lsst": target_id, "fink_lsst": target_id}
    return Target(target_id, coord, source="fink_lsst", alt_ids=alt_ids, t_ref=t_ref)


def process_lsst_alert(
    alert_data, alert_filepath=None, cutout_filepath=None, t_ref: Time = None
):
    topic, data, key = alert_data
    alert = data["diaSource"]
    alert["topic"] = topic
    alert["mjd"] = alert["midpointMjdTai"]

    diaObject = data["diaObject"]
    alert["diaObjectId"] = diaObject["diaObjectId"]
    alert["nDiaSources"] = diaObject["nDiaSources"]
    alert["tag"] = "valid"

    if alert_filepath is not None:
        with open(alert_filepath, "w+") as f:
            json.dump(alert, f)

    if cutout_filepath is not None:
        logger.warning("don't know how LSST cutouts are handled!")
    return alert
