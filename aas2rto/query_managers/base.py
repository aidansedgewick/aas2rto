import abc
import time
import warnings
from logging import getLogger
from typing import Callable, Dict, List

from pathlib import Path

import pandas as pd

from astropy.table import Table
from astropy.time import Time

from aas2rto.exc import UnknownTargetWarning
from aas2rto.target import Target
from aas2rto.target_data import TargetData
from aas2rto.target_lookup import TargetLookup
from aas2rto import paths

from aas2rto import utils

logger = getLogger("base_qm")


DEFAULT_DIRECTORIES = [
    "lightcurves",
    "alerts",
    "probabilities",
    "parameters",
    "magstats",
    "query_results",
    "cutouts",
    "photometry",
]


class BaseQueryManager(abc.ABC):

    target_lookup: TargetLookup = None

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def perform_all_tasks(self):
        msg = f"query_manager {self.name}: implement a function which accepts kwargs:\n    startup:bool, t_ref=:astropy.time.Time"
        raise NotImplementedError(msg)

    def process_paths(
        self,
        data_path: Path = None,
        parent_path: Path = None,
        create_paths: bool = True,
        directories: list = None,
        extra_directories: list = None,
    ):
        """

        Parameters
        ----------
        data_path : Path, default=None
            where should data for this query manager be stored? often it's more
            convenient to use parent_path (see below)
        parent_path : Path, default=None
            the parent directory of where QM data should be stored - the data_path
            is then set as data_path = parent_path / cls.name (ie. using the name param
            that you provided when defining the class!)
        create_paths : bool, default=True
            actually create the directories
        directories : list, default=`DEFAULT_DIRECTORIES`
        """
        if data_path is None:
            if parent_path is None:
                raise ValueError("Must provide 'data_path' or 'parent_path'")
            parent_path = Path(parent_path)
            data_path = parent_path / self.name
        else:
            data_path = Path(data_path)
            if parent_path is not None:
                msg = f"Ignoring parent_path={parent_path}: set to {data_path.parent}"
                logger.warning(msg)
            parent_path = data_path.parent

        self.parent_path = parent_path.absolute()
        self.data_path = data_path.absolute()

        directories = directories or DEFAULT_DIRECTORIES
        if isinstance(directories, str):
            directories = [directories]

        extra_directories = extra_directories or []
        if isinstance(extra_directories, str):
            extra_directories = [extra_directories]

        self.paths_lookup: Dict[str, Path] = {}
        for dir_name in list(directories) + list(extra_directories):
            path = self.data_path / dir_name
            self.paths_lookup[dir_name] = path
            setattr(self, f"{dir_name}_path", path)

        if create_paths:
            self._create_paths()

    def _create_paths(self):

        self.data_path.mkdir(exist_ok=True, parents=True)
        for dir_name, path in self.paths_lookup.items():
            path.mkdir(exist_ok=True)

    def load_single_lightcurve(self, target_id: str, t_ref: Time = None):
        logger.error(
            f"{self.name}: to call load_target_lightcurves, must implement load_single_lightcurve"
        )
        msg = f"QueryManager for {self.name}: load_single_lightcurve() not implemented"
        raise NotImplementedError(msg)

    def load_target_lightcurves(
        self,
        id_list: List[str] = None,
        only_flag_updated=True,
        t_ref: Time = None,
    ):
        """
        Parameters
        ----------
        id_list : list, default = None
            list of (eg.) fink_ids or target_ids to load.
            if None, try to load for all targets
        flag_only_existing : bool, default True
            if True, only targets which already have an existing (eg.) fink lightcurve
            will have `target.updated` flag set as True, if the latest lightcurve has
            more data than the existing.
            if False, targets with prev. missing (eg.) Fink lightcurves are flagged as
            updated if there is now a fink lightcurve available.
        """

        t_ref = t_ref or Time.now()

        loaded = []
        missing = []
        skipped = []

        t_start = time.perf_counter()

        ## ===== Decide which targets we want to load.
        if id_list is None:
            id_list = list(self.target_lookup.keys())

        ## ===== Actually try to load them.
        for target_id in id_list:
            target = self.target_lookup.get(target_id, None)
            if target is None:
                msg = f"load_lightcurve: {target_id} not in target_lookup"
                logger.warning(msg)
                warnings.warn(UnknownTargetWarning(msg))
                missing.append(target_id)
                continue

            lightcurve: pd.DataFrame | Table = self.load_single_lightcurve(
                target_id, t_ref=t_ref
            )
            if lightcurve is None:
                # logger.warning(f"{self.name}: loaded {id_} LC is bad")
                missing.append(target_id)
                continue

            try:
                if len(lightcurve) == 0:
                    missing.append(target_id)
                    continue
            except Exception as e:
                pass

            qm_data = target.get_target_data(self.name)
            # get eg. target.target_data["fink_ztf"], target.target_data["yse"], ...

            existing_lightcurve = qm_data.lightcurve
            if existing_lightcurve is not None:
                if len(lightcurve) <= len(existing_lightcurve):
                    skipped.append(target_id)
                    continue
                target.updated = True

            # If we've reached this point, the new LC must be longer than existing one
            # (or the prev. one was None),
            if not only_flag_updated:
                logger.debug(f"set {target_id} updated")
                target.updated = True
            qm_data.add_lightcurve(lightcurve)
            loaded.append(target_id)
        t_end = time.perf_counter()

        ## ===== Some summary stats.
        N_loaded = len(loaded)
        N_missing = len(missing)
        N_skipped = len(skipped)
        t_load = t_end - t_start
        msg = f"{self.name}: load {N_loaded}, missing {N_missing} LCs in {t_load:.1f}s"
        logger.info(msg)
        return loaded, skipped, missing

    def clear_stale_files(
        self, stale_age=60.0, max_depth=3, dry_run=False, t_ref: Time = None
    ):
        """
        Clear files/directories older than eg. 60 days.

        stale_time
            how many days to check

        """

        t_ref = t_ref or Time.now()

        for dir_name, top_level_dir in self.paths_lookup.items():
            N_dirs, N_files = utils.clear_stale_files(
                top_level_dir, t_ref=t_ref, stale_age=stale_age, max_depth=max_depth
            )
            if N_dirs > 0 or N_files > 0:
                logger.info(
                    f"Removing stale files in {dir_name}:\n    "
                    f"del {N_files} files >{stale_age:.1f}d old, {N_dirs} empty subdirs"
                )
