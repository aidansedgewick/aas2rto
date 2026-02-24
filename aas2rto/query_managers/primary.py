import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict

from astropy.time import Time

from aas2rto import utils
from aas2rto.path_manager import PathManager
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.target_lookup import TargetLookup

# from aas2rto.query_managers.alerce import AlerceQueryManager
from aas2rto.query_managers.atlas import AtlasQueryManager
from aas2rto.query_managers.fink import FinkLSSTQueryManager, FinkZTFQueryManager
from aas2rto.query_managers.lasair import LasairLSSTQueryManager, LasairZTFQueryManager
from aas2rto.query_managers.tns import TNSQueryManager
from aas2rto.query_managers.yse import YSEQueryManager

# from aas2rto.query_managers.sdss import SdssQueryManager


logger = getLogger(__name__.split(".")[-1])

EXPECTED_QUERY_MANAGERS = {
    # "alerce": AlerceQueryManager,
    "atlas": AtlasQueryManager,
    "fink_lsst": FinkLSSTQueryManager,
    "fink_ztf": FinkZTFQueryManager,
    "lasair_lsst": LasairLSSTQueryManager,
    "lasair_ztf": LasairZTFQueryManager,
    "tns": TNSQueryManager,
    "yse": YSEQueryManager,
}


class PrimaryQueryManager:

    def __init__(
        self,
        config: Dict,
        target_lookup: TargetLookup,
        path_manager: PathManager,
    ):

        self.config = config
        unknown_qm_configs = utils.check_unexpected_config_keys(
            self.config.keys(),
            EXPECTED_QUERY_MANAGERS.keys(),
            warn=False,
            name="query_managers",
        )
        if len(unknown_qm_configs):
            unk_qms_str = ", ".join(f"'{x}'" for x in unknown_qm_configs)
            exp_qms_str = ", ".join(f"'{x}'" for x in EXPECTED_QUERY_MANAGERS.keys())
            msg = (
                f"\033[31;1mUnknown QueryManager config(s)\033[0m:\n    {unk_qms_str}\n"
                f"Currently known:\n    {exp_qms_str}"
            )
            logger.error(msg)
            raise ValueError(msg)

        self.target_lookup = target_lookup
        self.path_manager = path_manager

        self.query_managers = self._initialise_query_manager_lookup()
        self.initialize_query_managers()

    def _initialise_query_manager_lookup(self) -> Dict[str, BaseQueryManager]:
        """Only for type hinting..."""
        return {}

    def initialize_query_managers(self):
        self.query_managers = self._initialise_query_manager_lookup()
        self.qm_order = []
        for qm_name, qm_config in self.config.items():

            if qm_config is None:
                continue

            use_qm = qm_config.pop("use", True)
            if not use_qm:
                logger.info(f"Skip {qm_name} init (parameter 'use'={use_qm})")
                continue
            self.qm_order.append(qm_name)  # In case the config order is very important.

            QMClass = EXPECTED_QUERY_MANAGERS[qm_name]
            qm = QMClass(
                qm_config, self.target_lookup, parent_path=self.path_manager.data_path
            )
            self.query_managers[qm_name] = qm

        if len(self.query_managers) == 0:
            logger.warning("no query managers initialised!")

    def perform_all_query_manager_tasks(self, iteration: int, t_ref: Time = None):
        """
        for each of the query managers `qm` in target_selector.query_managers,
        call the method `qm.perform_all_tasks(iteration=iteration, t_ref=t_ref)`

        Parameters
        ----------
        t_ref : astropy.time.Time, default=Time.now()
            used in the perform_all_tasks() call.
        """

        t_ref = t_ref or Time.now()

        logger.info("begin query manager tasks")
        for qm_name, qm in self.query_managers.items():
            t_start = time.perf_counter()
            logger.info(f"begin {qm_name} tasks")
            query_result = qm.perform_all_tasks(iteration=iteration, t_ref=t_ref)
            if isinstance(query_result, Exception):
                msg = f"EXCEPTION IN {qm_name} [caught, so no crash]\n{query_result}"
                warnings.warn(UserWarning(msg))
                # self.send_crash_reports(text=text)
            t_end = time.perf_counter()
            logger.info(f"{qm_name} tasks in {t_end-t_start:.1f} sec")
