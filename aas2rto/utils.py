import shutil
import warnings
from logging import getLogger
from typing import List, Tuple, Union

from pathlib import Path

import numpy as np

from astropy.coordinates import EarthLocation
from astropy.time import Time

from astroplan import Observer

from aas2rto.exc import UnexpectedKeysWarning, MissingKeysWarning

logger = getLogger("aas2rto_utils")


def chunk_list(l, chunk_size=100):
    for ii in range(0, len(l), chunk_size):
        yield l[ii : ii + chunk_size]


def calc_file_age(filepath, t_ref, allow_missing=True):
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


def print_header(s: str) -> None:
    """
    Print a nicely formatted header line.

    Parameters
    ----------
    s [str]

    eg. if your string s="Hello!", print:\n
    \#\#\# ============ Hello! ============ \#\#\#
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


def check_config_keys(provided, expected, name: str = None) -> Tuple[List, List]:
    if isinstance(provided, dict):
        provided = provided.keys()
    if isinstance(expected, dict):
        expected = expected.keys()
    unexpected_keys = check_unexpected_config_keys(provided, expected, name=name)
    missing_keys = check_missing_config_keys(provided, expected, name=name)
    return unexpected_keys, missing_keys


def check_unexpected_config_keys(provided, expected, name: str = None) -> list:
    if isinstance(provided, dict):
        provided = provided.keys()
    if isinstance(expected, dict):
        expected = expected.keys()

    unexpected_keys = set(provided) - set(expected)
    if len(unexpected_keys) > 0:
        msg = "\033[33;1munexpected keys\033[0m"
        if name is not None:
            msg = msg + f" in {name}"
        msg = msg + ":\n    " + " ".join(unexpected_keys)
        warnings.warn(UnexpectedKeysWarning(msg))
    return list(unexpected_keys)


def check_missing_config_keys(provided, expected, name: str = None) -> list:
    if isinstance(provided, dict):
        provided = provided.keys()
    if isinstance(expected, dict):
        expected = expected.keys()

    missing_keys = set(expected) - set(provided)
    if missing_keys:
        msg = "\033[33;1mmissing keys\033[0m"
        if name is not None:
            msg = msg + f" in {name}"
        msg = msg + ":\n    " + " ".join(missing_keys)
        warnings.warn(MissingKeysWarning(msg))
    return list(missing_keys)


def haversine(loc1: EarthLocation, loc2: EarthLocation, r=6371):
    dlon = loc2.lon.rad - loc1.lon.rad
    dlat = loc2.lat.rad - loc1.lat.rad
    sin_hdlat = np.sin(dlat / 2.0)
    sin_hdlon = np.sin(dlon / 2.0)
    print(sin_hdlat, sin_hdlon)

    a = sin_hdlat**2 + np.cos(loc1.lat.rad) * np.cos(loc2.lat.rad) * sin_hdlon**2

    c = 2.0 * np.arcsin(np.sqrt(a))
    return r * c


def get_observatory_name(observatory: Union[Observer, None, str]):
    if isinstance(observatory, str):
        return observatory
    if observatory is None:
        return "no_observatory"
    return observatory.name


def init_sfd_dustmaps():
    try:
        import dustmaps
        from dustmaps import sfd
    except ModuleNotFoundError as e:
        msg = (
            "`dusmaps` not imported properly. try:"
            "\n    \033[33;1mpython3 -m pip install dustmaps\033[0m"
        )
        raise ModuleNotFoundError(msg)

    logger.info("calling dustmaps.sfd.fetch()")
    sfd.fetch()
