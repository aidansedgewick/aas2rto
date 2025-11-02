import time
from logging import getLogger
from pathlib import Path
from typing import Dict

from astropy.time import Time

from aas2rto import utils
from aas2rto.path_manager import PathManager
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.target_lookup import TargetLookup

# from aas2rto.query_managers.alerce import AlerceQueryManager
# from aas2rto.query_managers.atlas import AtlasQueryManager
# from aas2rto.query_managers.fink import FinkQueryManager

# from aas2rto.query_managers.lasair import LasairQueryManager
# from aas2rto.query_managers.tns import TnsQueryManager
# from aas2rto.query_managers.yse import YseQueryManager

# from aas2rto.query_managers.sdss import SdssQueryManager


logger = getLogger(__name__.split(".")[-1])

EXPECTED_QUERY_MANAGERS = {
    # "alerce": AlerceQueryManager,
    # "atlas": AtlasQueryManager,
    # "fink_ztf": FinkZTFQueryManager,
    # "lasair": LasairQueryManager,
    # "tns": TnsQueryManager,
    # "yse": YseQueryManager,
}


class GlobalQueryManager:

    def __init__(
        self,
        config: Dict,
        target_lookup: TargetLookup,
        path_manager: PathManager,
        create_paths=True,
    ):

        self.config = config
        unknown_qm_configs = utils.check_unexpected_config_keys(
            self.config.keys(), EXPECTED_QUERY_MANAGERS.keys(), name="QM_configs"
        )
        if len(unknown_qm_configs):
            msg = (
                f"\033[31;1mUnknown QueryManager config(s)\033[0m:\n    {unknown_qm_configs}"
                f"Currently known:\n    {EXPECTED_QUERY_MANAGERS.keys()}"
            )
            logger.error(msg)
            raise ValueError(msg)

        self.target_lookup = target_lookup
        self.path_manager = path_manager

        self.query_managers = self._initialise_query_manager_lookup()
        self.initialize_query_managers(create_paths=create_paths)

    def _initialise_query_manager_lookup(self) -> Dict[str, BaseQueryManager]:
        """Only for type hinting..."""
        return {}

    def initialize_query_managers(self, create_paths=True):
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
                qm_config,
                self.target_lookup,
                parent_path=self.path_manager.data_path,
                create_paths=create_paths,
            )
            self.query_managers[qm_name] = qm

        if len(self.query_managers) == 0:
            logger.warning("no query managers initialised!")

    def add_query_manager(
        self, qm_config: Dict, QueryManagerClass, create_paths=True, **kwargs
    ):
        qm_name = getattr(QueryManagerClass, "name", None)
        if qm_name is None:
            qm_name = QueryManagerClass.__name__
            QueryManagerClass.name = qm_name  # Monkey patch!
            msg = f"new query manager {qm_name} should have class attribute 'name'"
            logger.warning(msg)

        has_tasks_method = hasattr(QueryManagerClass, "perform_all_tasks")
        if not has_tasks_method:
            msg = f"query manager {qm_name} must have method 'perform_all_tasks()'"
            raise AttributeError(msg)

        qm_config.update(kwargs)
        qm = QueryManagerClass(
            qm_config,
            self.target_lookup,
            parent_path=self.path_manager.data_path,
            create_paths=create_paths,
        )
        self.query_managers[qm_name] = qm

    def perform_all_query_manager_tasks(self, iteration, t_ref: Time = None):
        """
        for each of the query managers `qm` in target_selector.query_managers,
        call the method `qm.perform_all_tasks(t_ref=t_ref)`

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
                text = [f"EXCEPTION IN {qm_name} [no crash]"]
                # self.send_crash_reports(text=text)
            t_end = time.perf_counter()
            logger.info(f"{qm_name} tasks in {t_end-t_start:.1f} sec")
