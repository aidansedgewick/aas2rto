import shutil
from logging import getLogger

from pathlib import Path

import numpy as np

from astropy.time import Time

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
        tsize = shutil.get((80, 24))
        cols = tsize.columns
    except:
        cols = 60
    pad = cols - (len(s) + 2) - 6

    pad = max(pad, 1)
    fmt_s = "\n###" + "=" * pad + f" {s} " + "=" * pad + "###"
    print(fmt_s)


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
