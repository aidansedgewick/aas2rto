import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import getLogger
from pathlib import Path

from astropy.time import Time

from aas2rto.exc import UnknownQueryManagerError
from aas2rto.path_manager import PathManager
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.target_lookup import TargetLookup
from aas2rto.query_managers.registry import qm_registry  # qm_reg. is a SINGLETON
from aas2rto.query_managers.result import QueryManagerResult
from aas2rto.utils.config import check_unexpected_config_keys

logger = getLogger(__name__.split(".")[-1])


class PrimaryQueryManager:

    default_primary_config = {
        "concurrent": False,
        "max_workers": 8,
    }

    def __init__(
        self,
        config: dict,
        target_lookup: TargetLookup,
        path_manager: PathManager,
    ):

        # First pop the dict for configuring the PRIMARY (ie. this class)
        primary_config = config.pop("primary", {})
        self.primary_config = self.default_primary_config.copy()
        self.primary_config.update(primary_config)
        check_unexpected_config_keys(
            self.primary_config,
            self.default_primary_config,
            name="primary_qm",
            raise_exc=True,
        )

        # Now set check that all the QMs we have are correct
        self.qm_configs = config  # Everything else is for QMs to be initialised.
        unknown_qm_configs = check_unexpected_config_keys(
            self.qm_configs.keys(),
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
        for qm_name, qm_config in self.qm_configs.items():

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

        if self.primary_config["concurrent"]:
            return self.perform_qm_tasks_concurrently(iteration, t_ref=t_ref)
        else:
            return self.perform_qm_tasks_sequentially(iteration, t_ref=t_ref)

    def perform_qm_tasks_concurrently(self, iteration: int, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        concurrent_qms: list[BaseQueryManager] = []
        sequential_qms: list[BaseQueryManager] = []

        for qm_name, qm in self.query_managers.items():
            if qm.can_run_concurrently:
                concurrent_qms.append(qm)
            else:
                sequential_qms.append(qm)

        results = []
        if concurrent_qms:
            max_workers = min(self.primary_config["max_workers"], len(concurrent_qms))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                logger.info(f"perform {len(concurrent_qms)} QM tasks concurrently")
                futures = [
                    executor.submit(_qm_task_wrapper, query_manager, iteration, t_ref)
                    for query_manager in concurrent_qms
                ]

                for future in as_completed(futures):
                    qm_result = future.result()
                    qm_result.ran_concurrently = True  # Only for
                    results.append(qm_result)

        if sequential_qms:
            logger.info(f"perform {len(sequential_qms)} QM tasks sequentially")

            for qm in sequential_qms:
                qm_result = _qm_task_wrapper(qm, iteration, t_ref)
                results.append(qm_result)
        else:
            logger.info(f"no sequential QM tasks to perform")

        return qm_result

    def perform_qm_tasks_sequentially(self, iteration: int, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        logger.info("perform QM tasks sequentially")
        results = []
        for qm_name, qm in self.query_managers.items():
            qm_result = _qm_task_wrapper(qm, iteration, t_ref)
            results.append(qm_result)
        return results


def _qm_task_wrapper(
    query_manager: BaseQueryManager, iteration: int, t_ref: Time  # t_ref ARG
) -> QueryManagerResult:

    t_start = time.perf_counter()
    logger.info(f"begin {query_manager.name} tasks")
    try:
        qm_result = query_manager.perform_all_tasks(iteration=iteration, t_ref=t_ref)
    except Exception as e:
        logger.exception(f"Error in {query_manager.name}")
        return QueryManagerResult(name=qm_result.name, success=False, error_msg=e)

    t_stop = time.perf_counter()

    if isinstance(qm_result, QueryManagerResult):
        if qm_result.error_message:
            err_msg = qm_result.error_message
            msg = f"EXCEPTION IN {qm_result.name} [caught, so no crash]\n{err_msg}"
            warnings.warn(UserWarning(msg))
    elif qm_result is None:
        pass
    else:
        msg = (
            f"{query_manager.name} should return type 'QueryManagerResult', "
            f"not {type(qm_result)}"
        )
        logger.warning(msg)
        warnings.warn(UserWarning(msg))

    logger.info(f"{query_manager.name} tasks in {t_stop-t_start:.1f} sec")
    return qm_result
