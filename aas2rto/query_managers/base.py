from __future__ import annotations  # Must come first

import abc
import time
import warnings
from logging import getLogger

from pathlib import Path

import pandas as pd

from astropy.table import Table
from astropy.time import Time

from aas2rto import paths
from aas2rto.exc import UnknownTargetWarning
from aas2rto.target import Target
from aas2rto.target_data import TargetData
from aas2rto.target_lookup import TargetLookup
from aas2rto.utils import calc_file_age, clear_stale_files

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
    """
    The base class for all query managers.
    Only implements abstract methods and some path-processing methods.
    """

    config: dict = None
    target_lookup: TargetLookup = None

    @property
    @abc.abstractmethod
    def name(self):
        """Class attribute - the NAME of the query manager eg. 'atlas', 'fink_ztf', ..."""

    @abc.abstractmethod
    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        """perform_all_tasks(self, iteration: int, t_ref: Time=None)

        The method that the TargetSelector will use to operate your QueryManager
        Any methods the QM needs (outside of __init__) should be called here.

        Parameters
        ----------
        iteration: int
            `iteration` counter can optionally be used to modify behaviour of QM
            eg. on iteration=0 (startup), FinkQMs do not make new queries, only load
            existing data
        """

    def __init__(self, config: dict, target_lookup: TargetLookup, parent_path: Path):
        self.config = config
        self.target_lookup = target_lookup
        self.parent_path = parent_path

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
        parent_path : Path, default=None
            the parent directory of where QM data should be stored - the data_path
            is then set as `data_path = parent_path / cls.name`
            (ie. using the name param that you provided when defining the class!)
        data_path : Path, default=None
            where should data for this query manager be stored? often it's more
            convenient to use parent_path (above)
        create_paths : bool, default=True
            actually create the directories
        directories : list
            sub directories to create
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

        self.paths_lookup: dict[str, Path] = {}
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
            N_dirs, N_files = clear_stale_files(
                top_level_dir, t_ref=t_ref, stale_age=stale_age, max_depth=max_depth
            )
            if N_dirs > 0 or N_files > 0:
                logger.info(
                    f"Removing stale files in {dir_name}:\n    "
                    f"del {N_files} files >{stale_age:.1f}d old, {N_dirs} empty subdirs"
                )


class LightcurveQueryManager(BaseQueryManager, abc.ABC):
    """
    A base class for query managers that are primarily used for querying lightcurves

    """

    DEFAULT_LIGHTCURVE_UPDATE_INTERVAL = 2.0

    @property
    @abc.abstractmethod
    def id_resolving_order(self) -> tuple[str]:
        """tuple of strings.
        what order should we look for the relevant id in target.alt_ids?

        eg1. for FinkZTDQueryManager, id_resolving_order = ("fink_ztf", "ztf")

        If there is a specific name that FINK uses for this ZTF target by (unlikely...),
        take that first. Then, if there's a 'generic' ZTF name, take that. Otherwise,
        it's not a target known in the ZTF survey.

        eg2. for YSEQueryManager, id_resolving_order = ("tns", "yse")

        Even if the object has a YSE name, the YSE servers still prefer the TNS name."""

    @abc.abstractmethod
    def load_single_lightcurve(
        self, target_id: str, t_ref: Time = None
    ) -> pd.DataFrame | Table:
        """load_single_lightcurve(self, target_id: str, t_ref: Time=None)

        return pd.DataFrame OR astropy.table.Table
        """

    @abc.abstractmethod
    def get_lightcurve_filepath(self, id_: str):
        """get_lightcurve_filepath(self, idd: str)

        return the path of a lightcurve which should be under name 'id_'
        You should use a more sensible variable name for id_ in subclasses
        (eg. fink_id, atlas_id...)
        """

    def get_relevant_id_from_target(self, target: Target):
        """Loop over self.id_resolving_order, return the first value from key
        found in target.alt_ids

        For example: if FinkQM.id_resolving_order = ("fink", "fink_ztf", "tns")
        and target.alt_ids = {"fink_ztf": "ZTFabc", "other_srv": "T001"}
        would return "ZTFabc"
        """

        for alt_key in self.id_resolving_order:
            relevant_id = target.alt_ids.get(alt_key, None)
            if relevant_id is not None:
                return relevant_id
        return None

    def select_lightcurves_to_query(self, t_ref: Time = None):
        """
        Which lightcurves should be re-queried?

        Check the age of each file.
        """

        t_ref = t_ref or Time.now()

        to_query = []
        no_relevant_id = []
        max_age = self.config.get(
            "lightcurve_update_interval", self.DEFAULT_LIGHTCURVE_UPDATE_INTERVAL
        )
        for target_id, target in self.target_lookup.items():
            relevant_id = self.get_relevant_id_from_target(target)
            if relevant_id is None:
                no_relevant_id.append(target_id)
                continue  # There is no relevant ID to ask FINK for lightcurve
            lightcurve_filepath = self.get_lightcurve_filepath(relevant_id)
            lc_age = calc_file_age(lightcurve_filepath, t_ref=t_ref)
            if lc_age > max_age:
                to_query.append(relevant_id)
        msg = f"LCs for {len(to_query)} targets need updating (age > {max_age:.1f}d or missing)"
        logger.info(msg)
        if len(no_relevant_id) > 0:
            logger.info(
                f"({len(no_relevant_id)} have no relevant id for '{self.name}')"
            )
        return to_query

    def load_target_lightcurves(
        self,
        id_list: list[str] = None,
        only_flag_updated: bool = True,
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

            qm_data = target.get_target_data(self.name)  # returns empty TD() if missing
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


class KafkaQueryManager(BaseQueryManager, abc.ABC):
    """Used for adding targets that receive data from Kafka streams"""

    @property
    @abc.abstractmethod
    def target_id_key() -> str:
        """how do the alerts refer to targets?

        eg. `FinkZTFQueryManager` has `target_id_key='objectId', ``"""

    @abc.abstractmethod
    def listen_for_alerts(self) -> list[dict]:
        """listen_for_alerts(self)

        return a list of processed alerts. Each alert should contain at LEAST the value
        defined by `target_id_key`.

        eg. `FinkZTFQueryManager` has `target_id_key='objectId'`,
        so all processed_alerts must contain `objectId`"""

    @abc.abstractmethod
    def new_target_from_alert(
        self, processed_alert: dict, t_ref: Time = None
    ) -> Target:
        """new_target_from_alert(self, processed_alert: dict, t_ref: Time=None)

        return a target, based"""

    def add_targets_from_alerts(
        self, processed_alerts: list[dict], t_ref: Time = None
    ) -> list[str]:
        t_ref = t_ref or Time.now()

        targets_added = []
        existing_skipped = []
        for alert in processed_alerts:
            target_id = alert[self.target_id_key]

            ##== Do we already know about it?
            existing_target = self.target_lookup.get(target_id, None)
            if existing_target is not None:
                existing_skipped.append(target_id)
                continue

            ##== Otherwise add it...
            target = self.new_target_from_alert(alert, t_ref=t_ref)
            if not isinstance(target, Target):
                msg = (
                    f"implementation of {self.__class__}.new_target_from_alert"
                    " should return Target"
                )
                raise TypeError()
            self.target_lookup.add_target(target)
            targets_added.append(target.target_id)

        N_added = len(targets_added)
        N_skipped = len(existing_skipped)
        logger.info(f"added {N_added} new targets (skipped {N_skipped} existing)")
        return targets_added
