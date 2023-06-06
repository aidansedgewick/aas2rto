import logging
import time
import warnings
import yaml
from pathlib import Path
from typing import Callable

import numpy as np

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from astroplan import Observer

from dk154_targets import query_managers
from dk154_targets import Target

from dk154_targets import paths

logger = logging.getLogger(__name__)


class TargetSelector:
    """
    The main api.

    Parameters
    ----------
    selector_config
        A dictionary, with keys {expected_config_keys}

    """

    expected_config_keys = (
        "query_managers",
        "observatories",
        "modelling",
        "messengers",
    )

    def __init__(self, selector_config: dict):
        # Unpack configs.
        self.selector_config = selector_config
        self.query_manager_config = self.selector_config.get("query_managers", {})
        self.observatory_config = self.selector_config.get("observatories", {})
        self.messenger_config = self.selector_config.get("messengers", {})
        self.modelling_config = self.selector_config.get("modelling", {})
        self.paths_config = self.selector_config.get("paths", {})

        # to keep the targets. Do this here as initQM needs to know about it.
        self.target_lookup = {}

        # Prepare paths
        self.process_paths()
        self.make_directories()

        # Initialise some things.
        self.initialize_query_managers()
        self.initialize_observatories()
        self.initialize_messengers()

    @classmethod
    def from_config(cls, config_path):
        config_path = Path(config_path)

        with open(config_path, "r") as f:
            selector_config = yaml.load(f, Loader=yaml.FullLoader)
            print(selector_config)
            selector = cls(selector_config)
            return selector

    def process_paths(
        self,
    ):
        self.base_path = paths.base_path
        project_path = self.paths_config.pop("project_path", "default")
        if project_path == "default":
            project_path = self.base_path
        self.project_path = project_path
        self.paths = {"base_path": self.base_path, "project_path": self.project_path}
        for location, path in self.paths_config.items():
            formatted_parts = [
                self.instance_paths[p[1:]] if p.startswith("$") else p
                for p in path.split("/")
            ]
            self.instance_paths[location] = Path(formatted_parts)
        if "data_path" not in self.paths:
            self.paths["data_path"] = paths.default_data_path
        self.data_path = self.paths["data_path"]
        if "ouputs_path" not in self.paths:
            self.paths["outputs_path"] = paths.default_outputs_path
        self.outputs_path = self.paths["outputs_path"]

    def make_directories(
        self,
    ):
        for location, path in self.paths.items():
            path.mkdir(exist_ok=True, parents=True)

    def initialize_query_managers(
        self,
    ):
        self.query_managers = {}
        self.qm_order = []
        for qm_name, qm_config in self.query_manager_config.items():
            if not qm_config.get("use", True):
                logger.info(f"Skip {qm_name} init")
                continue
            self.qm_order.append(qm_name)  # In case the config order is very important.

            if qm_name == "alerce":
                raise NotImplementedError(f"{qm_name.capitalize()}QueryManager")
            if qm_name == "atlas":
                raise NotImplementedError(f"{qm_name.capitalize()}QueryManager")
            if qm_name == "fink":
                raise NotImplementedError(f"{qm_name.capitalize()}QueryManager")
            if qm_name == "lasair":
                raise NotImplementedError(f"{qm_name.capitalize()}QueryManager")
            if qm_name == "tns":
                raise NotImplementedError(f"{qm_name.capitalize()}QueryManager")
            else:
                # Mainly for testing.
                warnings.warn(
                    f"no known query manager for {qm_name}",
                    query_managers.UsingGenericWarning,
                )
                qm = query_managers.GenericQueryManager(
                    qm_config, self.target_lookup, data_path=self.data_path
                )
            self.query_managers[qm_name] = qm

        if len(self.query_managers) == 0:
            logger.warning("no query managers initialised!")

    def initialize_observatories(
        self,
    ):
        self.observatories = {"no_observatory": None}
        for obs_name, location_identifier in self.observatory_config.items():
            if isinstance(location_identifier, str):
                earth_loc = EarthLocation.of_site(location_identifier)
            else:
                earth_loc = EarthLocation(**location_identifier)
            observatory = Observer(location=earth_loc, name=obs_name)
            self.observatories[obs_name] = observatory
        logger.info("init {len(self.observatories)} observatories")
        logger.info("     including `no_observatory`")

    def initialize_messengers(
        self,
    ):
        # TODO: add telegram, slack, etc.
        pass

    def add_target(self, target: Target):
        if target.objectId in self.target_lookup:
            raise ValueError(f"obj {target.objectId} already in target_lookup")
        self.target_lookup[target.objectId] = target

    def compute_observatory_nights(
        self, t_ref: Time = None, horizon: u.Quantity = None
    ):
        """
        save the result of astrolan.Observer().tonight() in each target
        as it's not a cheap function if computing many times.

        access in a target with eg. `my_target.observatory_nights["lasilla"]`
        """
        t_ref = t_ref or Time.now()
        horizon = horizon or -18.0 * u.deg
        for obs_name, observatory in self.observatories.items():
            if observatory is None:
                continue
            tonight = observatory.tonight(t_ref, horizon=horizon)
            for objectId, target in self.target_lookup.items():
                target.observatory_night[obs_name] = tonight

    def perform_query_manager_tasks(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        logger.info("begin query manager tasks")
        for qm_name, qm in self.query_managers.items():
            t_start = time.perf_counter()
            qm.perform_all_tasks(t_ref=t_ref)
            t_end = time.perf_counter()
            logger.info(f"{qm_name} tasks in {t_end-t_start:.1f} sec")

    def evaluate_all_targets(self, scoring_function: Callable, t_ref: Time = None):
        """
        Evaluate all targets according to a provided `scoring_function`, for
        each observatory and `no_observatory` (`None`).

        Parameters
        ----------
        scoring_function
            your scoring function, with signature
            (target: Target, observatory: astroplan.Observer)
        t_ref
            astropy.time.Time, optional - if not provided, defaults to Time.now()

        """

        t_ref = t_ref or Time.now()
        for obs_name, observatory in self.observatories.items():
            for objectId, target in self.target_lookup.items():
                target.evaluate_target(scoring_function, observatory, t_ref=t_ref)

    def new_target_initial_check(self, scoring_function: Callable, t_ref: Time = None):
        t_ref = t_ref or Time.now()
        new_targets = []
        new_scores = []
        for objectId, target in self.target_lookup.items():
            last_score = target.get_last_score("no_observatory")
            if last_score is not None:
                continue
            score = target.evaluate_target(scoring_function, None, t_ref=t_ref)
            new_scores.append(score)
            new_targets.append(objectId)
        return new_targets

    def reject_bad_targets(self):
        to_remove = []
        for objectId, target in self.target_lookup.items():
            raw_score = target.get_last_score("no_observatory")
            if target.target_of_opportunity:
                continue
            if not np.isfinite(raw_score):
                to_remove.append(objectId)
        removed_targets = []
        for objectId in to_remove:
            target = self.target_lookup.pop(objectId)
            removed_targets.append(target)
            assert objectId not in self.target_lookup
        return removed_targets
