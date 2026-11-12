from __future__ import annotations

import time
from logging import getLogger
from typing import Any

import numpy as np

logger = getLogger("transport_utils")


class RequestTracker:
    """
    Convenience class to track how query successes and failures, and to check/quit
    if there have been too many failed queries, or querying is taking too long.

    """

    @classmethod
    def start(cls, max_query_time: float = np.inf, max_failed_queries: int = np.inf):
        return cls(time.perf_counter(), max_query_time, max_failed_queries)

    def __init__(self, t_start: float, max_query_time: float, max_failed_queries: int):
        self.t_start = t_start
        self.max_query_time = max_query_time
        self.max_failed_queries = max_failed_queries

        self.n_failed_queries = 0
        self.n_success_queries = 0
        self.failed = []
        self.missing = []
        self.success = []

    def track_failed(self, failed: Any):
        if isinstance(failed, list) or isinstance(failed, tuple):
            self.failed.extend(failed)
        else:
            self.failed.append(failed)
        self.n_failed_queries = self.n_failed_queries + 1

    def track_missing(self, missing: Any):
        if isinstance(missing, list) or isinstance(missing, tuple):
            self.missing.extend(missing)
        else:
            self.missing.append(missing)

    def track_success(self, success: Any):
        if isinstance(success, list) or isinstance(success, tuple):
            self.success.extend(success)
        else:
            self.success.append(success)
        self.n_success_queries = self.n_success_queries + 1

    @property
    def n_failed(self) -> int:
        return len(self.failed)

    @property
    def n_missing(self) -> int:
        return len(self.missing)

    @property
    def n_success(self) -> int:
        return len(self.success)

    def safe_to_query(self):
        # Is it sensible to conitnue with queries, or is everything failing?
        if self.n_failed_queries >= self.max_failed_queries:
            logger.warning(f"Too many failures ({self.n_failed})")
            return False

        t_elapsed = time.perf_counter() - self.t_start
        if t_elapsed > self.max_query_time:
            msg = (
                f"Tracker timeout: "
                f"({t_elapsed:.1f}s > max {self.max_query_time:.1f}s)"
            )
            logger.warning(msg)
            return False
        return True

    def log_summary(
        self,
        name: str = "",
        tasks_str: str = "queries",
        data_str: str = "objects",
        **kwargs,
    ):
        """
        All kwargs here are just for extra description in log messages.
        Unknown kwargs are ignorred.
        """
        msg = (
            f"{name} summary:\n"
            f"    {self.n_success_queries} succesful {tasks_str} ({self.n_success} {data_str})"
        )
        if self.n_missing > 0:
            msg = msg + f"\n    {self.n_missing} {data_str} missing/returned no data"

        if self.n_failed_queries > 0:
            fail_msg = (
                f"\n    {self.n_failed_queries} failed {tasks_str} "
                f"({self.n_failed} {data_str})"
            )
            msg = msg + fail_msg
        logger.info(msg)
