import abc
import time
from logging import getLogger
from typing import Callable, List

from pathlib import Path

from astropy.time import Time

from aas2rto.target import TargetData
from aas2rto.target_lookup import TargetLookup
from aas2rto import paths

from aas2rto import utils

logger = getLogger("base_qm")


EXPECTED_DIRECTORIES = [
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
        raise NotImplementedError

    def add_target(self, target):
        if target.target_id in self.target_lookup:
            raise ValueError(
                f"{self.name}: obj {target.target_id} already in target_lookup"
            )
        self.target_lookup[target.target_id] = target

    def process_paths(
        self,
        data_path: Path = None,
        parent_path: Path = None,
        create_paths: bool = True,
        # paths: list = EXPECTED_DIRECTORIES,
    ):
        """
        If data path is None
        """
        if data_path is None:
            if parent_path is None:
                parent_path = paths.base_path / paths.default_data_dir
            parent_path = Path(parent_path)
            data_path = parent_path / self.name
        else:
            data_path = Path(data_path)
            parent_path = data_path.parent

        self.parent_path = parent_path
        self.data_path = data_path

        # utils.check_unexpected_config_keys(
        #    dirs, EXPECTED_DIRECTORIES, name=f"{self.name} qm __init__(paths)"
        # )

        self.lightcurves_path = self.data_path / "lightcurves"
        self.alerts_path = self.data_path / "alerts"
        self.probabilities_path = self.data_path / "probabilities"
        self.parameters_path = self.data_path / "parameters"
        self.magstats_path = self.data_path / "magstats"
        self.query_results_path = self.data_path / "query_results"
        self.cutouts_path = self.data_path / "cutouts"
        if create_paths:
            self.create_paths()

    def create_paths(self):
        self.data_path.mkdir(exist_ok=True, parents=True)
        self.lightcurves_path.mkdir(exist_ok=True, parents=True)
        self.alerts_path.mkdir(exist_ok=True, parents=True)
        self.probabilities_path.mkdir(exist_ok=True, parents=True)
        self.parameters_path.mkdir(exist_ok=True, parents=True)
        self.magstats_path.mkdir(exist_ok=True, parents=True)
        self.query_results_path.mkdir(exist_ok=True, parents=True)
        self.cutouts_path.mkdir(exist_ok=True, parents=True)

    def init_missing_target_data(self):
        for target_id, target in self.target_lookup.items():
            qm_data = target.target_data.get(self.name, None)
            if qm_data is None:
                assert self.name not in target.target_data
                target.target_data[self.name] = TargetData()

    def get_query_results_file(self, query_name, fmt="csv") -> Path:
        return self.query_results_path / f"{query_name}.{fmt}"

    def get_alert_dir(self, target_id) -> Path:
        return self.alerts_path / f"{target_id}"

    def get_alert_file(self, target_id, candid, fmt="json", mkdir=True) -> Path:
        alert_dir = self.get_alert_dir(target_id)
        if mkdir:
            alert_dir.mkdir(exist_ok=True, parents=True)
        return alert_dir / f"{candid}.{fmt}"

    def get_magstats_file(self, target_id) -> Path:
        return self.magstats_path / f"{target_id}.csv"

    def get_lightcurve_file(self, target_id, fmt="csv") -> Path:
        return self.lightcurves_path / f"{target_id}.{fmt}"

    def get_probabilities_file(self, target_id, fmt="csv") -> Path:
        return self.probabilities_path / f"{target_id}.{fmt}"

    def get_cutouts_dir(self, target_id) -> Path:
        return self.cutouts_path / f"{target_id}"

    def get_cutouts_file(self, target_id, candid, fmt="pkl", mkdir=True) -> Path:
        cutouts_dir = self.get_cutouts_dir(target_id)
        if mkdir:
            cutouts_dir.mkdir(exist_ok=True, parents=True)
        return cutouts_dir / f"{candid}.{fmt}"

    def get_parameters_file(self, target_id, fmt="pkl") -> Path:
        return self.parameters_path / f"{target_id}.{fmt}"

    def load_single_lightcurve(self, id_, t_ref=None):
        logger.error(
            f"{self.name}: to call load_target_lightcurves, must implement load_single_lightcurve"
        )
        msg = f"QueryManager for {self.name}: load_single_lightcurve() not implemented"
        raise NotImplementedError(msg)

    def load_target_lightcurves(
        self,
        id_list: List[str] = None,
        id_from_target_function: Callable = None,
        flag_only_existing=True,
        t_ref: Time = None,
    ):
        """
        Parameters
        ----------
        id_list : list, default = None
            list of (eg.) fink_ids or target_ids to load.
            if None, try to load for all targets
        func_id_from_target : Callable, default = None
            if provided, use this function on each target to see if a suitable
            id can be found for searching for lightcurve.
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

        ## ===== Decide which targets we should load lightcurves of.
        if id_list is None:
            if id_from_target_function is None:
                id_list = list(self.target_lookup.keys())
                msg = f"{self.name}: try loading lcs for all {len(id_list)} targets"
            else:
                id_list = []
                for target_id, target in self.target_lookup.items():
                    specific_id = id_from_target_function(target)
                    if specific_id is not None:
                        id_list.append(specific_id)
                msg = f"{self.name}: try loading lcs for {len(id_list)} targets with {self.name}_id"
            logger.info(msg)

        ## ===== Actually try to load them.
        for id_ in id_list:
            target = self.target_lookup.get(id_, None)
            if target is None:
                logger.warning(f"load_lightcurve: {id_} not in target_lookup")
                missing.append(id_)
                continue

            lightcurve = self.load_single_lightcurve(id_, t_ref=t_ref)
            if lightcurve is None:
                logger.warning(f"{self.name}: loaded {id_} LC is bad")
                missing.append(id_)
                continue

            try:
                if lightcurve.empty:
                    skipped.append(id_)
                    continue
            except Exception as e:
                pass

            # get eg. target.target_data["fink"], target.target_data["yse"], ...
            qm_data = target.get_target_data(self.name)
            existing_lightcurve = qm_data.lightcurve
            if existing_lightcurve is not None:
                if len(lightcurve) <= len(existing_lightcurve):
                    skipped.append(id_)
                    continue
                target.updated = True
            if not flag_only_existing:
                # lightcurve here cannot be None, and must be longer than existing.
                logger.debug(f"set {id_} updated")
                target.updated = True
            qm_data.add_lightcurve(lightcurve)
            loaded.append(id_)
        t_end = time.perf_counter()

        ## ===== Some summary stats.
        N_loaded = len(loaded)
        N_missing = len(missing)
        N_skipped = len(skipped)
        t_load = t_end - t_start
        msg = f"{self.name}: load {N_loaded}, missing {N_missing} LCs in {t_load:.1f}s"
        logger.info(msg)
        return loaded, missing

    def clear_stale_files(self, stale_time=60.0, dry_run=False, t_ref: Time = None):
        """
        Clear files/directories older than eg. 60 days.

        stale_time
            how many days to check

        """

        t_ref = t_ref or Time.now()

        for dir in [
            self.lightcurves_path,
            self.alerts_path,
            self.probabilities_path,
            self.parameters_path,
            self.magstats_path,
            self.query_results_path,
            self.cutouts_path,
        ]:
            for filepath in dir.glob("*"):
                filepath = Path(filepath)
                file_age = utils.calc_file_age(filepath, t_ref)

                if file_age < stale_time:
                    continue

                if filepath.is_dir():
                    for filepath_ii in filepath.iterdir():
                        if filepath_ii.is_file():
                            if not dry_run:
                                filepath_ii.unlink()
                            print(f"del {filepath_ii}")
                    print(f"del dir {filepath}")
                    try:
                        if not dry_run:
                            filepath.rmdir()
                    except Exception as e:
                        pass
                else:
                    if not dry_run:
                        filepath.unlink()
                    print(f"del {filepath}")
