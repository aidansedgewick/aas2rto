import shutil
from logging import getLogger
from typing import Union

from pathlib import Path

import numpy as np

from astropy.time import Time

from astroplan import Observer

logger = getLogger("utils")


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


def print_header(s):
    try:
        tsize = shutil.get_terminal_size()
        cols = tsize.columns
    except Exception as e:
        cols = 60
    pad = (cols - (len(s) + 2) - 6) // 2

    pad = max(pad, 1)
    fmt_s = "\n###" + "=" * pad + f" {s} " + "=" * pad + "###"
    print(fmt_s)


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
