from __future__ import annotations

import json

from astropy.time import Time

from confluent_kafka.cimpl import Message  # Only for type hints

from aas2rto.query_managers.registry import qm_registry
from aas2rto.query_managers.lasair.lasair_base import LasairBaseQueryManager


@qm_registry.register()
class LasairLSSTQUeryManager(LasairBaseQueryManager):
    name = "lasair_lsst"
    target_id_key = "diaObjectId"
    alert_id_key = "diaSourceId"
    id_resolving_order = ("lasair_lsst", "lsst")

    def process_single_alert(
        self, message: Message, topic_keys: dict, t_ref: Time = None
    ):
        alert = json.loads(message.value())

        lasair_id_key = topic_keys.get("target_id", None)
        lasair_id = alert.get[lasair_id_key]
