import shutil
import warnings
from logging import getLogger
from typing import List, Tuple, Union

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


def chunk_list(l, chunk_size=100):
    for ii in range(0, len(l), chunk_size):
        yield l[ii : ii + chunk_size]


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
    print(fmt_s)


def check_config_keys(
    provided, expected, name: str = None, warn=True
) -> Tuple[List, List]:
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


def get_observatory_name(observatory: Union[Observer, None, str]):
    if isinstance(observatory, str):
        return observatory
    return observatory.name
