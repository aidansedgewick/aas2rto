import time
import warnings
from logging import getLogger
from pathlib import Path

from astropy.time import Time

from aas2rto import utils
from aas2rto.exc import UnknownQueryManagerError
from aas2rto.path_manager import PathManager
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.target_lookup import TargetLookup

from aas2rto.query_managers.registry import qm_registry  # qm_reg. is a SINGLETON

logger = getLogger(__name__.split(".")[-1])


class PrimaryQueryManager:

    def __init__(
        self,
        config: dict,
        target_lookup: TargetLookup,
        path_manager: PathManager,
    ):

        self.config = config
        unknown_qm_configs = utils.check_unexpected_config_keys(
            self.config.keys(),
            qm_registry.all(),
            warn=False,
            name="query_managers",
        )
        if len(unknown_qm_configs):
            unk_qms_str = ", ".join(f"'{x}'" for x in unknown_qm_configs)
            exp_qms_str = ", ".join(f"'{x}'" for x in qm_registry.all()) or "--empty--"
            msg = (
                f"\033[31;1mUnknown QueryManager config(s)\033[0m:\n    {unk_qms_str}\n"
                f"Currently in registry:\n    {exp_qms_str}\n"
                f"Did you remember to register your QM?"
            )
            logger.error(msg)
            raise UnknownQueryManagerError(msg)

        self.target_lookup = target_lookup
        self.path_manager = path_manager

        self.query_managers: dict[str, BaseQueryManager] = {}
        self.initialize_query_managers()

    def initialize_query_managers(self):
        self.qm_order: list = []
        for qm_name, qm_config in self.config.items():

            if qm_config is None:
                continue

            use_qm = qm_config.pop("use", True)
            if not use_qm:
                logger.info(f"Skip {qm_name} init (parameter 'use'={use_qm})")
                continue
            self.qm_order.append(qm_name)  # In case the config order is very important.

            QMClass = qm_registry.get(qm_name)
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
