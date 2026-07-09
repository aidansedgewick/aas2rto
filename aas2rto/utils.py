from __future__ import annotations

import shutil
import time
import warnings
from logging import getLogger
from typing import Any

from pathlib import Path

import numpy as np

from astropy.coordinates import EarthLocation
from astropy.time import Time

from astroplan import Observer

from aas2rto.exc import (
    MissingKeysError,
    MissingKeysWarning,
    UnexpectedKeysError,
    UnexpectedKeysWarning,
)

logger = getLogger("aas2rto_utils")


def chunk_list(iterable, chunk_size=100):
    for ii in range(0, len(iterable), chunk_size):
        yield iterable[ii : ii + chunk_size]


def calc_file_age(filepath, t_ref, allow_missing=True):
    """
    return the age of a filepath in days.
    """

    filepath = Path(filepath)
    if not filepath.exists():
        if allow_missing:
            return np.inf
        else:
            msg = "File does not exist: `calc_file_age` with `allow_missing=False`"
            raise IOError(msg)
    file_mod_time = Time(filepath.stat().st_mtime, format="unix")
    dt = t_ref - file_mod_time
    return dt.jd


def clear_stale_files(dir: Path, t_ref=None, stale_age=60.0, depth=0, max_depth=3):
    if depth > max_depth:
        return

    t_ref = t_ref or Time.now()

    N_dirs = 0
    N_files = 0

    for filepath in dir.glob("*"):
        if filepath.is_dir():
            N_subdirs, N_subfiles = clear_stale_files(
                filepath,
                t_ref,
                stale_age=stale_age,
                depth=depth + 1,
                max_depth=max_depth,
            )
            N_dirs = N_dirs + N_subdirs
            N_files = N_files + N_subfiles
            subdir_is_empty = not list(filepath.iterdir())
            if subdir_is_empty:
                logger.debug(f"removing empty subdir {filepath.name}")
                filepath.rmdir()
                N_dirs = N_dirs + 1
        else:
            file_age = calc_file_age(filepath, t_ref=t_ref)
            if file_age > stale_age:
                logger.debug(f"removing stale {filepath.name}")
                filepath.unlink()
                N_files = N_files + 1
    return N_dirs, N_files


def print_header(s: str) -> None:
    """
    Print a nicely formatted header line.

    Parameters
    ----------
    s [str]

    eg. if your string s="Hello!", print:\n
    `### ============ Hello! ============ ###`
    """
    try:
        tsize = shutil.get_terminal_size()
        cols = tsize.columns
    except Exception:
        cols = 60
    pad = (cols - (len(s) + 2) - 6) // 2

    pad = max(pad, 1)
    fmt_s = "\n###" + "=" * pad + f" {s} " + "=" * pad + "###"
    print(fmt_s)  # Actually write to the screen


class QueryTracker:
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
            logger.warning(f"Too many failed queries ({self.n_failed})")
            return False

        t_elapsed = time.perf_counter() - self.t_start
        if t_elapsed > self.max_query_time:
            msg = (
                f"Queries taking too long "
                f"({t_elapsed:.1f}s > max {self.max_query_time:.1f}s)"
            )
            logger.warning(msg)
            return False
        return True

    def log_summary(self, name: str = ""):
        msg = (
            f"{name} query summary:\n"
            f"    {self.n_success_queries} succesful queries ({self.n_success} objects)"
        )
        if self.n_missing > 0:
            msg = msg + f"\n    {self.n_missing} objects missing/returned no data"

        if self.n_failed_queries > 0:
            fail_msg = (
                f"\n    {self.n_failed_queries} failed queries "
                f"({self.n_failed} objects)"
            )
            msg = msg + fail_msg
        logger.info(msg)


def check_config_keys(
    provided, expected, name: str = None, warn=True
) -> tuple[list, list]:
    if isinstance(provided, dict):
        provided = provided.keys()
    if isinstance(expected, dict):
        expected = expected.keys()
    unexpected_keys = check_unexpected_config_keys(
        provided, expected, name=name, warn=warn
    )
    missing_keys = check_missing_config_keys(provided, expected, name=name, warn=warn)
    return unexpected_keys, missing_keys


def check_unexpected_config_keys(
    provided,
    expected,
    name: str = None,
    warn=True,
    raise_exc=False,
    exc_class=UnexpectedKeysError,
) -> list:
    if isinstance(provided, dict):
        provided = provided.keys()
    if isinstance(expected, dict):
        expected = expected.keys()

    unexpected_keys = set(provided) - set(expected)
    if len(unexpected_keys) > 0:
        msg = "\033[33;1munexpected keys\033[0m"
        if name is not None:
            msg = msg + f" in {name}"

        keys_str = ", ".join(f"\033[33;1m'{k}'\033[0m" for k in unexpected_keys)
        msg = msg + ":\n    " + keys_str
        if raise_exc:
            logger.error(msg)
            raise exc_class(msg)
        if warn:
            logger.warning(msg)
            warnings.warn(UnexpectedKeysWarning(msg))
    return list(unexpected_keys)


def check_missing_config_keys(
    provided,
    expected,
    name: str = None,
    warn=True,
    raise_exc=False,
    exc_class=MissingKeysError,
) -> list:

    if isinstance(provided, dict):
        provided = provided.keys()
    if isinstance(expected, dict):
        expected = expected.keys()

    missing_keys = set(expected) - set(provided)
    if missing_keys:
        msg = "\033[33;1mmissing keys\033[0m"
        if name is not None:
            msg = msg + f" in {name}"
        keys_str = ", ".join(f"\033[33;1m'{k}'\033[0m" for k in missing_keys)
        msg = msg + ":\n    " + keys_str
        if raise_exc:
            logger.error(msg)
            raise exc_class(msg)
        if warn:
            logger.warning(msg)
            warnings.warn(MissingKeysWarning(msg))
    return list(missing_keys)


def get_observatory_name(observatory: Observer | str | None):
    if isinstance(observatory, str):
        return observatory
    return observatory.name


# def retreive_nested_data(
#     data: dict, keys_str: str, default_value: Any = None, raise_exc: bool = False
# ):

#     keys = keys_str.split(".")

#     result = data
#     for key in keys:
#         result = result.get(key, {})
#         if result is isinstance(dict) and not result and raise_exc:
#             msg = f"retrive_nested exited early ('{key}' not found)"
#             raise MissingKeysError(msg)
#     return result or default_value


def format_link_as_markdown(link: str, text: str = None):
    if text is not None:
        return f"<{link}|{str(text)}>"
    return f"<{link}>"


def format_link_as_html(link: str, text: str = None, prefix: str = "//"):
    text = text or ""
    return f'<a href="{prefix}{link}">{text}</a>'
