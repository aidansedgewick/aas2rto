from logging import getLogger
from pathlib import Path

import numpy as np

from astropy.time import Time

logger = getLogger("file_utils")


def calc_file_age(filepath: Path, t_ref: Time, allow_missing=True):
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
