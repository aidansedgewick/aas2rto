import logging
import os
import shutil
import sys
import time
import traceback
import warnings
import yaml
from pathlib import Path
from typing import Callable, Dict, List, Set

import numpy as np

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from astroplan import Observer
from astroplan.plots import plot_altitude

from dk154_targets import messengers
from dk154_targets import query_managers
from dk154_targets import utils
from dk154_targets.lightcurve_compilers import DefaultLightcurveCompiler
from dk154_targets.obs_info import ObservatoryInfo
from dk154_targets.plotters import plot_default_lightcurve, plot_visibility
from dk154_targets.scoring.default_obs_scoring import DefaultObservatoryScoring
from dk154_targets.target import Target

from dk154_targets import paths

logger = logging.getLogger(__name__.split(".")[-1])

matplotlib.use("Agg")


VALID_SKIP_TASKS = (
    "qm_tasks",
    "obs_info",
    "pre_check",
    "modeling",
    "evaluate",
    "ranking",
    "reject",
    "plotting",
    "write_targets",
    "messaging",
)


class TargetSelector:
    """
    The main api.

    Parameters
    ----------
    selector_config [dict]
        A dictionary, with keys {expected_config_keys}
    create_paths

    """

    expected_config_keys = (
        "selector_parameters",
        "query_managers",
        "observatories",
        "messengers",
        "paths",
    )
    default_selector_parameters = {
        "project_name": None,
        "sleep_time": 600.0,
        "unranked_value": 9999,
        "minimum_score": 0.0,
        "retained_recovery_files": 5,
        "skip_tasks": [],
        "obs_info_dt": 0.5 / 24.0,
        "lazy_modeling": True,
        "lazy_compile": False,
        "lazy_plotting": True,
        "plotting_interval": 0.25,
        "write_comments": True,
    }
    default_base_path = paths.wkdir
    expected_messenger_keys = ("telegram", "slack")

    def __init__(self, selector_config: dict, create_paths=True):
        # Unpack configs.
        self.selector_config = selector_config
        utils.check_unexpected_config_keys(
            selector_config, self.expected_config_keys, name="selector_config"
        )

        self.selector_parameters = self.default_selector_parameters.copy()
        selector_parameters = self.selector_config.get("selector_parameters", {})
        self.selector_parameters.update(selector_parameters)
        utils.check_unexpected_config_keys(
            selector_parameters,
            self.default_selector_parameters,
            name="selector_parameters",
        )

        self.query_managers_config = self.selector_config.get("query_managers", {})
        self.observatories_config = self.selector_config.get("observatories", {})
        self.messengers_config = self.selector_config.get("messengers", {})
        self.paths_config = self.selector_config.get("paths", {})

        # to keep the targets. Do this here as initQM needs to know about it.
        self.target_lookup = self._create_empty_target_lookup()

        self.ranked_lists = {}

        # Prepare paths
        self.process_paths(create_paths=create_paths)

        # Initialise some things.
        self.initialize_query_managers(create_paths=create_paths)
        self.initialize_observatories()
        self.initialize_messengers()

    def __setitem__(self, objectId, target):
        if not isinstance(target, Target):
            class_name = target.__class__.__name__
            msg = f"Cannot add {objectId} (type={class_name}) to target list."
            raise ValueError(msg)
        self.target_lookup[objectId] = target

    def __getitem__(self, objectId):
        return self.target_lookup[objectId]

    def __iter__(self):
        for target, objectId in self.target_lookup.items():
            yield target, objectId

    def __contains__(self, member):
        return member in self.target_lookup

    @classmethod
    def from_config(cls, config_path, create_paths=True):
        config_path = Path(config_path)

        with open(config_path, "r") as f:
            selector_config = yaml.load(f, Loader=yaml.FullLoader)
        selector = cls(selector_config, create_paths=create_paths)
        return selector

    def _create_empty_target_lookup(self) -> Dict[str, Target]:
        """Returns an empty dictionary. Only for type hinting."""
        return dict()

    def process_paths(self, create_paths=True):
        base_path = self.paths_config.pop("base_path", "default")
        if base_path == "default":
            base_path = self.default_base_path
        self.base_path = Path(base_path)

        project_path = self.paths_config.pop("project_path", None)
        if project_path is None or project_path == "default":
            project_name = self.selector_parameters.get("project_name", None)
            if project_name is None:
                msg = (
                    "You can set the name of the project_path by providing "
                    "'project_name' in 'selector_parameters'. set to 'default'"
                )
                logger.info(msg)
                project_name = "default"
            projects_base = self.base_path / "projects"
            projects_base.mkdir(exist_ok=True, parents=True)
            project_path = projects_base / project_name
        self.project_path = Path(project_path)

        msg = (
            f"set project path at:\n    \033[36;1m{self.project_path.absolute()}\033[0m"
        )
        logger.info(msg)

        self.paths = {"base_path": self.base_path, "project_path": self.project_path}

        for location, raw_path in self.paths_config.items():
            if Path(raw_path).is_absolute():
                formatted_path = Path(raw_path)
            else:
                parts = str(raw_path).split("/")
                if parts[0].startswith("$"):
                    # eg. replace `$my_cool_dir/blah/blah` with `paths["my_cool_dir"]`
                    parent_name = parts[0][1:]
                    parent = self.paths[parent_name]
                    formatted_parts = [Path(p) for p in parts[1:]]
                else:
                    parent = self.project_path
                    formatted_parts = [Path(p) for p in parts]
                formatted_path = parent.joinpath(*formatted_parts)
            self.paths[location] = formatted_path

        if "data_path" not in self.paths:
            self.paths["data_path"] = self.project_path / paths.default_data_dir
        self.data_path = self.paths["data_path"]

        if "outputs_path" not in self.paths:
            self.paths["outputs_path"] = self.project_path / paths.default_outputs_dir
        self.outputs_path = self.paths["outputs_path"]

        if "opp_targets_path" not in self.paths:
            self.paths["opp_targets_path"] = (
                self.project_path / paths.default_opp_targets_dir
            )
        self.opp_targets_path = self.paths["opp_targets_path"]

        if "scratch_path" not in self.paths:
            self.paths["scratch_path"] = self.project_path / paths.default_scratch_dir
        self.scratch_path = self.paths["scratch_path"]
        self.lc_scratch_path = self.scratch_path / "lc"
        self.paths["lc_scratch_path"] = self.lc_scratch_path
        self.vis_scratch_path = self.scratch_path / "vis"
        self.paths["vis_scratch_path"] = self.vis_scratch_path

        if "comments_path" not in self.paths:
            self.paths["comments_path"] = self.project_path / "comments"
        self.comments_path = self.paths["comments_path"]

        if "rejected_targets_path" not in self.paths:
            self.paths["rejected_targets_path"] = self.project_path / "rejected_targets"
        self.rejected_targets_path = self.paths["rejected_targets_path"]

        if "existing_targets_path" not in self.paths:
            self.paths["existing_targets_path"] = self.project_path / "existing_targets"
        self.existing_targets_path = self.paths["existing_targets_path"]

        if create_paths:
            for path_name, path_val in self.paths.items():
                path_val.mkdir(exist_ok=True, parents=True)

    def _initialise_query_manager_lookup(
        self,
    ) -> Dict[str, query_managers.BaseQueryManager]:
        """Only for type hinting..."""
        return {}

    def initialize_query_managers(self, create_paths=True):
        self.query_managers = self._initialise_query_manager_lookup()
        self.qm_order = []
        for qm_name, qm_config in self.query_managers_config.items():
            use_qm = qm_config.pop("use", True)
            if not use_qm:
                logger.info(f"Skip {qm_name} init")
                continue
            self.qm_order.append(qm_name)  # In case the config order is very important.

            qm_args = (qm_config, self.target_lookup)
            qm_kwargs = dict(parent_path=self.data_path, create_paths=create_paths)
            if qm_name == "alerce":
                qm = query_managers.AlerceQueryManager(*qm_args, **qm_kwargs)
                self.alerce_query_manager = qm
            elif qm_name == "atlas":
                qm = query_managers.AtlasQueryManager(*qm_args, **qm_kwargs)
                self.atlas_query_manager = qm
            elif qm_name == "fink":
                qm = query_managers.FinkQueryManager(*qm_args, **qm_kwargs)
                self.fink_query_manager = qm
            elif qm_name == "lasair":
                qm = query_managers.LasairQueryManager(*qm_args, **qm_kwargs)
                self.lasair_query_manager = qm
            elif qm_name == "sdss":
                raise NotImplementedError("sdss qm not implemented.")
                qm = query_managers.SdssQueryManager(*qm_args, **qm_kwargs)
            elif qm_name == "tns":
                qm = query_managers.TnsQueryManager(*qm_args, **qm_kwargs)
                self.tns_query_manager = qm
            elif qm_name == "yse":
                raise NotImplementedError("yse not yet implemented")
                qm = query_managers.YseQueryManager(*qm_args, **qm_kwargs)
                self.yse_query_manager = qm
            else:
                logger.warning(f"no known query manager for {qm_name}")
                logger.info(f"unused config for {qm_name}")
                continue
            self.query_managers[qm_name] = qm

        if len(self.query_managers) == 0:
            logger.warning("no query managers initialised!")

    def add_query_manager(
        self, qm_config, QueryManagerClass, create_paths=True, **kwargs
    ):
        qm_name = getattr(QueryManagerClass, "name", None)
        if qm_name is None:
            qm_name = QueryManagerClass.__name__
            QueryManagerClass.name = qm_name  # Monkey patch!
            msg = f"new query manager {qm_name} should name class attribute 'name'"
            logger.warning(msg)

        has_tasks_method = hasattr(QueryManagerClass, "perform_all_tasks")
        if not has_tasks_method:
            msg = f"query manager {qm_name} must have method 'perform_all_tasks()'"
            raise AttributeError(msg)

        qm_config.update(kwargs)
        qm = QueryManagerClass(
            qm_config,
            self.target_lookup,
            parent_path=self.data_path,
            create_paths=create_paths,
        )
        self.query_managers[qm_name] = qm

    def _get_empty_observatory_lookup(self) -> Dict[str, Observer]:
        return {}

    def initialize_observatories(self):
        logger.info(f"init observatories")
        self.observatories = (
            self._get_empty_observatory_lookup()
        )  # {"no_observatory": None}
        self.observatories["no_observatory"] = None
        for obs_name, location_val in self.observatories_config.items():
            if isinstance(location_val, str):
                earth_loc = EarthLocation.of_site(location_val)
            else:
                if "lat" not in location_val or "lon" not in location_val:
                    msg = f"{obs_name} " + "should be a dict with {lat=, lon=}"
                    logger.warning(f"Likey an exception:\n     {msg}")
                earth_loc = EarthLocation.from_geodetic(**location_val)
            observatory = Observer(location=earth_loc, name=obs_name)
            self.observatories[obs_name] = observatory
        logger.info(f"init {len(self.observatories)} including `no_observatory`")

    def initialize_messengers(self):
        self.messengers = {}
        self.telegram_messenger = None
        self.slack_messenger = None
        for msgr_name, msgr_config in self.messengers_config.items():
            use_msgr = msgr_config.pop("use", True)
            if not use_msgr:
                logger.info(f"Skip messenger {msgr_name} init")
                continue
            if msgr_name == "telegram":
                msgr = messengers.TelegramMessenger(msgr_config)
                self.telegram_messenger = msgr
            elif msgr_name == "slack":
                msgr = messengers.SlackMessenger(msgr_config)
                self.slack_messenger = msgr
            else:
                raise NotImplementedError(f"No messenger {msgr_name}")
            self.messengers[msgr_name] = msgr

    def add_target(self, target: Target):
        if target.objectId in self.target_lookup:
            raise ValueError(f"obj {target.objectId} already in target_lookup")
        self.target_lookup[target.objectId] = target

    def compute_observatory_info(
        self, t_ref: Time = None, horizon: u.Quantity = -18 * u.deg, dt=0.5 / 24.0
    ):
        """
        Precompute some expensive information about the altitude for each observatory.

        access in a target with eg. `my_target.observatory_info["lasilla"]`
        """
        t_ref = t_ref or Time.now()

        for obs_name, observatory in self.observatories.items():
            if observatory is None:
                continue

            obs_info = ObservatoryInfo.for_observatory(
                observatory, t_ref=t_ref, horizon=horizon, dt=dt
            )

            for objectId, target in self.target_lookup.items():
                target_obs_info = obs_info.copy()
                assert target_obs_info.target_altaz is None
                if target.coord is not None:
                    target_obs_info.set_target_altaz(target.coord, observatory)
                target.observatory_info[obs_name] = target_obs_info
                if target.observatory_info[obs_name].target_altaz is None:
                    msg = f"\033[33m{objectId} {obs_name} altaz missing\033[0m"
                    logger.warning(msg)

    def check_for_targets_of_opportunity(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        logger.info("check for targets of opportunity")
        opp_target_file_list = self.opp_targets_path.glob("*.yaml")
        failed_targets = []
        existing_targets = []
        successful_targets = []
        for opp_target_file in opp_target_file_list:
            with open(opp_target_file, "r") as f:
                target_config = yaml.load(f, Loader=yaml.FullLoader)
                objectId = target_config.get("objectId", None)
                ra = target_config.get("ra", None)
                dec = target_config.get("dec", None)
                msg = (
                    f"Your config has {target_config.keys()}. "
                    "You should provide minimum of: "
                    "\033[31;1mobjectId, ra, dec\033[0m"  # codes colour red.
                )
                if any([x is None for x in [objectId, ra, dec]]):
                    logger.warning(msg)
                    failed_targets.append(opp_target_file)
                    continue
                base_score = target_config.get("base_score", None)
            if objectId in self.target_lookup:
                logger.info(f"{objectId} already in target list!")
                existing_target = self.target_lookup[objectId]
                if base_score is not None:
                    existing_target.base_score = base_score
                existing_target.target_of_opportunity = True
                existing_targets.append(opp_target_file)
            else:
                opp_target = Target(
                    objectId, ra=ra, dec=dec, base_score=base_score, t_ref=t_ref
                )  # Target()
                opp_target.target_of_opportunity = True
                self.add_target(opp_target)
                successful_targets.append(opp_target_file)
        if len(failed_targets) > 0:
            failed_list = [f.name for f in failed_targets]
            logger.info(f"{len(failed_targets)} failed targets:\n    {failed_list}")
        msg = f"{len(successful_targets)} new opp targets"
        if len(existing_targets) > 0:
            msg = msg + f" ({len(existing_targets)} already exist)"
        for target_file in successful_targets + existing_targets:
            os.remove(target_file)
            assert not target_file.exists()

        return successful_targets, existing_targets, failed_targets

    def compile_target_lightcurves(
        self, lightcurve_compiler: Callable = None, lazy=False, t_ref=None
    ):
        """
        Compile all the data from the target_data into a convenient location,
        This could be useful for eg. scoring, modeling, plotting.

        Parameters
        ----------
        lightcurve_compiler [Callable]
            your function that builds a convenient single lightcurve.
            see dk154_targets.lightcurve_compilers.DefaultLigthcurveCompiler for example.
        t_ref [`astropy.time.Time`]
        """
        t_ref = t_ref or Time.now()

        if lightcurve_compiler is None:
            lightcurve_compiler = DefaultLightcurveCompiler()

        try:
            lc_compiler_name = lightcurve_compiler.__name__
        except AttributeError as e:
            lc_compiler_name = type(lightcurve_compiler).__name__
        logger.info("compile photometric data:")
        logger.info(f"use {lc_compiler_name}")

        compiled = []
        skipped = []
        failed = []
        for objectId, target in self.target_lookup.items():
            compiled_exists = target.compiled_lightcurve is not None
            if compiled_exists and (not target.updated) and lazy:
                skipped.append(objectId)
                continue
            compiled_lc = lightcurve_compiler(target, t_ref=t_ref)
            target.compiled_lightcurve = compiled_lc
            if compiled_lc is None:
                failed.append(objectId)
                continue
            compiled.append(objectId)

        logger.info(f"compiled:{len(compiled)}, skipped (lazy):{len(skipped)}")
        if len(failed) > 0:
            logger.warning(f"failed:{len(failed)} (no compiled lc!)")
        return compiled, failed

    def perform_query_manager_tasks(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        logger.info("begin query manager tasks")
        for qm_name, qm in self.query_managers.items():
            t_start = time.perf_counter()
            logger.info(f"begin {qm_name} tasks")
            query_result = qm.perform_all_tasks(t_ref=t_ref)
            if isinstance(query_result, Exception):
                text = [f"EXCEPTION IN {qm_name}"]
                self.send_crash_reports(text=text)
            t_end = time.perf_counter()
            logger.info(f"{qm_name} tasks in {t_end-t_start:.1f} sec")

    def evaluate_targets(
        self,
        science_scoring_function: Callable,
        observatory_scoring_function: Callable = None,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()
        logger.info("evalutate all targets")

        if observatory_scoring_function is None:
            observatory_scoring_function = DefaultObservatoryScoring()

        for objectId, target in self.target_lookup.items():
            self.evaluate_single_target(
                target,
                science_scoring_function,
                observatory_scoring_function=observatory_scoring_function,
                t_ref=t_ref,
            )

    def evaluate_single_target(
        self,
        target: Target,
        science_scoring_function: Callable,
        observatory_scoring_function: Observer = None,
        t_ref: Time = None,
    ):
        """
        Evaluate a single target with the provided `science_scoring_function`,
        to give a `science_score`.
        If `observatory_scoring_function` is provided, modify the `science_score`
        by this result.
        This function is used mainly by `evaluate_all_targets`.

        Parameters
        ----------
        target : Target
        science_scoring_function : Callable
            Your scoring function, with signature:
            (target: `Target`, t_ref: `Time`)
        observatory_scoring_function : Callable, default: None
            Your observatory_scoring_function, with score
            if observatory_scoring_function is not None, multiply the result
            of science_scoring_function by the result of this function.
            It should have signature:
            (target: `Target`, observatory: `astroplan.Observer`, t_ref: `Time`)
        t_ref : astropy.time.Time, optional
            if not provided, defaults to Time.now()
        """
        t_ref = t_ref or Time.now()
        minimum_score = self.selector_parameters["minimum_score"]

        # ===== Compute score according to
        science_score, score_comments, reject_comments = (
            self._evaluate_target_science_score(
                science_scoring_function, target, t_ref=t_ref
            )
        )

        obs_name = "no_observatory"
        target.update_score_history(science_score, obs_name, t_ref=t_ref)
        target.score_comments[obs_name] = score_comments
        target.reject_comments[obs_name] = reject_comments

        if observatory_scoring_function is None:
            return

        for obs_name, observatory in self.observatories.items():
            if observatory is None:
                if not obs_name == "no_observatory":
                    raise ValueError(f"observatory {obs_name} is None!")
                continue

            if science_score > minimum_score and np.isfinite(science_score):
                # ===== Modify score for each observatory
                obs_factors, score_comments, reject_comments = (
                    self._evaluate_target_observatory_score(
                        observatory_scoring_function, target, observatory, t_ref
                    )
                )
                observatory_score = science_score * obs_factors
            else:
                obs_factors = 1.0
                observatory_score = science_score
                score_comments = ["excluded by no_observatory score"]
                reject_comments = []

            target.update_score_history(observatory_score, obs_name, t_ref=t_ref)
            target.score_comments[obs_name] = score_comments
            target.reject_comments[obs_name] = reject_comments
        return

    def _evaluate_target_science_score(
        self, science_scoring_function: Callable, target: Target, t_ref: Time = None
    ):
        """
        Wrapper function around user-provided science_scoring_function, to catch
        exceptions.

        """

        t_ref = t_ref or Time.now()

        obs_name = "no_observatory"
        try:
            scoring_res = science_scoring_function(target, t_ref)
        except Exception as e:
            details = (
                f"    For target {target.objectId} at obs {obs_name} at {t_ref.isot},\n"
                f"    scoring with {science_scoring_function.__name__} failed.\n"
                f"    Set score to -1.0, to exclude.\n"
            )
            scoring_res = (-1.0, details, [])
            self.send_crash_reports(text=details)

        if isinstance(scoring_res, tuple):
            if len(scoring_res) == 3:
                science_score, score_comments, reject_comments = scoring_res
            else:
                msg = f"your scoring_function {science_scoring_function.__name__} should return\n    "
                f"score [float], score_comments [List[str]], reject_comments [List[str]]"
                raise ValueError(msg)
        elif isinstance(scoring_res, float) or isinstance(scoring_res, int):
            science_score = scoring_res
            score_comments = ["no score_comments provided"]
            reject_comments = ["no reject_comments provided"]
        else:
            raise ValueError

        return science_score, score_comments, reject_comments

    def _evaluate_target_observatory_score(
        self,
        observatory_scoring_function: Callable,
        target: Target,
        observatory: Observer,
        t_ref: Time,
    ):

        obs_name = observatory.name
        try:
            scoring_res = observatory_scoring_function(target, observatory, t_ref=t_ref)
        except Exception as e:
            details = (
                f"    For target {target.objectId} at obs {obs_name} at {t_ref.isot},\n"
                f"    scoring with {observatory_scoring_function.__name__} failed.\n"
                f"    Set score to -1.0, to exclude.\n"
            )
            scoring_res = (-1.0, details, [])
            self.send_crash_reports(text=details)

        if isinstance(scoring_res, tuple):
            if len(scoring_res) == 3:
                obs_factors, score_comments, reject_comments = scoring_res
            else:
                raise ValueError(
                    "your scoring_function should return float, or tuple len 3:\n"
                    "score [float], or (score [float], score_comms [list], reject_comms [list])"
                )
        elif isinstance(scoring_res, float) or isinstance(scoring_res, int):
            obs_factors = scoring_res
            score_comments = ["no score_comments provided"]
            reject_comments = ["no reject_comments provided"]
        else:
            raise ValueError
        if not np.isfinite(obs_factors):
            msg = (
                f"observatory_score not finite ={obs_factors} "
                f"for {target.objectId} at {obs_name}."
            )
            logger.warning(msg)

        return obs_factors, score_comments, reject_comments

    def new_target_initial_check(
        self, scoring_function: Callable, t_ref: Time = None
    ) -> List[str]:
        """
        Evaluate the score for targets which have not been scored before (ie, new targets)
        with fixed observatory = `None`.

        This is useful for removing targets which are obviously rubbish
        before any expensive modellng.

        It is unlikely that a user will need to call this function directly.

        Parameters
        ----------
        scoring_function: `Callable`
            Your scoring function, with signature:
            (target: `Target`, observatory: `astroplan.Observer`, t_ref: `Time`)
        t_ref: `astropy.time.Time`, optional
            if not provided, defaults to Time.now()
        """

        t_ref = t_ref or Time.now()
        new_targets = []
        for objectId, target in self.target_lookup.items():
            last_score = target.get_last_score("no_observatory")
            if last_score is not None:
                continue
            self.evaluate_single_target(
                target,
                scoring_function,
                observatory_scoring_function=None,
                t_ref=t_ref,
            )
            new_targets.append(objectId)
        return new_targets

    def remove_rejected_targets(
        self, objectId_list=None, t_ref=None, write_comments=True
    ) -> List[Target]:
        t_ref = t_ref or Time.now()

        if objectId_list is None:
            objectId_list = list(self.target_lookup.keys())

        removed_targets = []
        for objectId in objectId_list:
            target = self.target_lookup.get(objectId, None)
            if target is None:
                logger.warning(f"can't remove non-existent target {objectId}")
                continue
            last_score = target.get_last_score()  # at no_observatory.
            if last_score is None:
                logger.warning(f"in reject: {objectId} has no score")
            if np.isfinite(last_score):
                continue  # it can stay...
            target = self.target_lookup.pop(objectId)
            if write_comments:
                target.write_comments(self.rejected_targets_path, t_ref=t_ref)
                target.write_comments(self.comments_path, t_ref=t_ref)
            removed_targets.append(target)
            assert objectId not in self.target_lookup
        return removed_targets

    def build_target_models(
        self, modeling_functions: Callable, t_ref: Time = None, lazy=False
    ):
        """
        Parameters
        ----------
        modeling_function
            your function, which should accept one argument, `Target` and return
            a model (which you can access later)
        t_ref
            an `astropy.time.Time`

        """
        t_ref = t_ref or Time.now()

        if not isinstance(modeling_functions, list):
            modeling_functions = [modeling_functions]
        for func in modeling_functions:
            try:
                model_key = func.__name__
            except AttributeError as e:
                model_key = type(func).__name__
                msg = f"\n    Your modeling_function {model_key} should have attribute __name__."
                logger.warning(msg)

            logger.info(f"build {model_key} models")
            built = []
            failed = []
            skipped = []
            for objectId, target in self.target_lookup.items():
                model_exists = model_key in target.models
                latest_model = target.models.get(model_key, None)
                if (not target.updated) and lazy and (model_exists):
                    skipped.append(objectId)
                    continue
                try:
                    model = func(target)
                    built.append(objectId)
                except Exception as e:
                    print(traceback.format_exc())
                    model = None
                    failed.append(objectId)
                    logger.warning(f"{model_key} failed for {objectId}")
                target.models[model_key] = model
                target.models_t_ref[model_key] = t_ref

            logger.info(f"built:{len(built)} skipped (lazy):{len(skipped)}")
            if len(failed) > 0:
                logger.warning(f"failed:{len(failed)}")

    def write_target_comments(
        self, target_list: List[Target] = None, outdir: Path = None, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        logger.info("writing target comments")

        outdir = outdir or self.comments_path
        outdir = Path(outdir)

        if target_list is None:
            target_list = [t for o, t in self.target_lookup.items()]

        for objectId, target in self.target_lookup.items():
            target.write_comments(outdir, t_ref=t_ref)

    def plot_target_lightcurves(
        self,
        plotting_function: Callable = None,
        lazy=False,
        interval=0.25,
        t_ref: Time = None,
    ):
        plotted = []
        skipped = []
        logger.info(f"plot lightcurves")

        if plotting_function is None:
            plotting_function = plot_default_lightcurve

        for objectId, target in self.target_lookup.items():
            fig_path = self.get_lightcurve_plot_path(objectId)
            fig_age = utils.calc_file_age(fig_path, t_ref, allow_missing=True)
            if lazy and (not target.updated) and (fig_age < interval):
                skipped.append(objectId)
                msg = f"skip {objectId} lc: age {fig_age:.2f} < {interval:.2f}"
                logger.debug(msg)
                continue
            fig = plotting_function(target, t_ref=t_ref)
            fig.savefig(fig_path)
            plt.close(fig=fig)
            plotted.append(objectId)
        if len(plotted) > 0 or len(skipped) > 0:
            msg = f"plotted {len(plotted)}, re-use {len(skipped)}"
            logger.info(msg)
        return plotted, skipped

    def plot_target_visibilities(self, lazy=False, interval=0.25, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        for obs_name, observatory in self.observatories.items():
            if observatory is None:
                continue
            logger.info(f"visibility plots for {obs_name}")
            skipped = []
            plotted = []
            for objectId, target in self.target_lookup.items():
                fig_path = self.get_visibility_plot_path(objectId, obs_name)
                fig_age = utils.calc_file_age(fig_path, t_ref, allow_missing=True)
                if lazy and (not target.updated) and (fig_age < interval):
                    msg = f"skip {objectId} {obs_name} vis: age {fig_age:.2f} < {interval:.2f}"
                    logger.debug(msg)
                    skipped.append(objectId)
                    continue
                obs_info = target.observatory_info.get(obs_name, None)
                fig = plot_visibility(
                    observatory, target, t_ref=t_ref, obs_info=obs_info
                )
                fig.savefig(fig_path)
                plt.close(fig=fig)
                plotted.append(objectId)
            if len(plotted) > 0 or len(skipped) > 0:
                msg = f"plotted {len(plotted)}, reused {len(skipped)}"
                logger.info(msg)

    def get_lightcurve_plot_path(self, objectId):
        return self.lc_scratch_path / f"{objectId}_lc.png"

    def get_visibility_plot_path(self, objectId, obs_name):
        return self.vis_scratch_path / f"{objectId}_{obs_name}_vis.png"

    def build_ranked_target_lists(
        self, plots=True, write_list=True, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        logger.info("build ranked target lists:")
        for obs_name, observatory in self.observatories.items():
            self.build_ranked_target_list_at_observatory(
                observatory, plots=plots, write_list=write_list, t_ref=t_ref
            )

    def build_ranked_target_list_at_observatory(
        self, observatory: Observer, plots=True, write_list=True, t_ref: Time = None
    ):
        """
        Rank all of the targets in TargetLookup for a particular observatory.

        Parameters
        ----------
        observatory: `astroplan.Observer` or `None`
            The observatory to rank the targets for.
        plots: default=True
            whether or not to produce plots for each target.
            lightcurve plots are copied from scratch_path, altitude/obs charts
            are produced from scratch.
        write_list: default=True
            Whether or not to write the ranked list to paths
        t_ref: default = Time.now()
            the score history will be saved for each target, along with t_ref

        Returns ranked_df
        """
        t_ref = t_ref or Time.now()

        obs_name = getattr(observatory, "name", "no_observatory")
        if obs_name == "no_observatory":
            assert observatory is None
        logger.info(f"ranked lists for {obs_name}")

        data_list = []
        for objectId, target in self.target_lookup.items():
            last_score = target.get_last_score(obs_name)
            if last_score is None:
                continue
            data_list.append(
                dict(objectId=objectId, score=last_score, ra=target.ra, dec=target.dec)
            )

        if len(data_list) == 0:
            logger.warning("no targets in target lookup!")
            return

        score_df = pd.DataFrame(data_list)
        score_df.sort_values("score", inplace=True, ascending=False, ignore_index=True)
        score_df["ranking"] = np.arange(1, len(score_df) + 1)
        # should call is "ranking" not "rank", as rank is a df/series method
        # so score_df.rank / row.rank fails!!

        minimum_score = self.selector_parameters.get("minimum_score")
        unranked_value = self.selector_parameters.get("unranked_value")
        negative_score = score_df["score"] < minimum_score
        score_df.loc[negative_score, "ranking"] = unranked_value

        for ii, row in score_df.iterrows():
            objectId = row.objectId
            target = self.target_lookup[objectId]
            target.update_rank_history(row["ranking"], obs_name, t_ref=t_ref)

        score_df.query(f"score>{minimum_score}", inplace=True)
        if write_list:
            ranked_list_path = self.get_ranked_list_path(obs_name)
            score_df.to_csv(ranked_list_path, index=False)

        self.ranked_lists[obs_name] = score_df

        if plots:
            for ii, row in score_df.iterrows():
                objectId = row.objectId
                target = self.target_lookup[objectId]
                self.collect_plots(target, obs_name, row["ranking"])
        logger.info(f"{sum(negative_score)} targets excluded, {len(score_df)} ranked")
        return score_df

    def get_ranked_list_path(self, obs_name, mkdir=True) -> Path:
        ranked_lists_path = self.outputs_path / "ranked_lists"
        if mkdir:
            ranked_lists_path.mkdir(exist_ok=True, parents=True)
        return ranked_lists_path / f"{obs_name}.csv"

    def get_output_plots_path(self, obs_name, mkdir=True) -> Path:
        plots_path = self.outputs_path / f"plots/{obs_name}"
        if mkdir:
            plots_path.mkdir(exist_ok=True, parents=True)
        return plots_path

    def clear_output_plots(self, fig_fmt="png"):
        for obs_name, observatory in self.observatories.items():
            plot_dir = self.get_output_plots_path(obs_name, mkdir=False)
            if plot_dir.exists():
                for plot in plot_dir.glob(f"*.{fig_fmt}"):
                    os.remove(plot)

    def collect_plots(self, target: Target, obs_name: str, ranking: int, fmt="png"):
        """
        It's much cheaper to create all the plots once,
        and then just copy them to a new directory.
        """

        plots_path = self.get_output_plots_path(obs_name)
        objectId = target.objectId

        lc_fig_file = self.get_lightcurve_plot_path(objectId)
        if lc_fig_file is not None:
            new_lc_fig_stem = f"{int(ranking):03d}_{objectId}_lc"
            new_lc_fig_file = plots_path / f"{new_lc_fig_stem}.{fmt}"
            if lc_fig_file.exists():
                try:
                    shutil.copy2(lc_fig_file, new_lc_fig_file)
                except FileNotFoundError as e:
                    msg = (
                        f"\033[33mlc_fig {lc_fig_file} missing!\033[0m"
                        + "\n    the cause is likely that you have two projects "
                        + "with the same project_path, and one has cleared plots"
                    )
                    logger.error(msg)

        vis_fig_file = self.get_visibility_plot_path(objectId, obs_name)
        # Don't need obs_name in new stem - separate dir for each!
        new_vis_fig_stem = f"{int(ranking):03d}_{objectId}_vis"
        new_vis_fig_file = plots_path / f"{new_vis_fig_stem}.{fmt}"
        if vis_fig_file.exists():
            try:
                shutil.copy2(vis_fig_file, new_vis_fig_file)
            except FileNotFoundError as e:
                msg = (
                    f"\033[33mvisibility fig {vis_fig_file} missing!\033[0m"
                    + f"\n    the likely cause is you have two projects with the"
                    + f"same project_path, and one has cleared plots"
                )
                logger.error(msg)

    def reset_target_figures(self):
        for objectId, target in self.target_lookup.items():
            target.reset_figures()

    def reset_updated_targets(self, t_ref: Time = None):
        for objectId, target in self.target_lookup.items():
            target.updated = False
            target.send_updates = False
            target.update_messages = []

    def write_existing_target_list(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        rows = []
        if len(self.target_lookup) == 0:
            logger.info("no existing targets to write...")
            return
        for objectId, target in self.target_lookup.items():
            data = dict(
                objectId=objectId,
                ra=target.ra,
                dec=target.dec,
                base_score=target.base_score,
            )
            rows.append(data)
        existing_targets_df = pd.DataFrame(rows)

        timestamp = t_ref.strftime("%y%m%d_%H%M%S")
        filename = f"recover_{timestamp}"
        existing_targets_file = self.existing_targets_path / f"{filename}.csv"
        existing_targets_df.to_csv(existing_targets_file, index=False)
        try:
            print_path = existing_targets_file.relative_to(self.project_path.parent)
        except Exception as e:
            print_path = existing_targets_file
        logger.info(f"write existing targets to:\n    {print_path}")

        targets_files = sorted(self.existing_targets_path.glob("*.csv"))
        N = self.selector_parameters.get("retained_recovery_files", 5)
        targets_files_to_remove = targets_files[:-N]
        for filepath in targets_files_to_remove:
            os.remove(filepath)

    def write_score_histories(self, t_ref: Time = None):
        for objectId, target in self.target_lookup.items():
            score_history_df = target.get_score_history()
            if score_history_df is None:
                continue
            score_history_file = self.get_score_history_file(objectId)
            score_history_df.to_csv(score_history_file, index=False)

    def get_score_history_file(self, objectId: str):
        score_history_path = self.paths.get("score_history_path", None)
        if score_history_path is None:
            score_history_path = self.existing_targets_path / "score_history"
        score_history_path.mkdir(exist_ok=True, parents=True)
        return score_history_path / f"{objectId}.csv"

    def write_rank_histories(self, t_ref: Time = None):
        for objectId, target in self.target_lookup.items():
            rank_history_df = target.get_rank_history()
            if rank_history_df is None:
                continue
            rank_history_file = self.get_rank_history_file(objectId)
            rank_history_df.to_csv(rank_history_file, index=False)

    def get_rank_history_file(self, objectId: str):
        rank_history_path = self.paths.get("rank_history_path", None)
        if rank_history_path is None:
            rank_history_path = self.existing_targets_path / "rank_history"
        rank_history_path.mkdir(exist_ok=True, parents=True)
        return rank_history_path / f"{objectId}.csv"

    def recover_existing_targets(self, existing_targets_file="last"):
        """
        During each iteration of the selector, the names of all of the existing targets
        are dumped into a file.

        These can be reloaded in the event of the program stopping.
        """

        if existing_targets_file == "last":
            targets_files = sorted(self.existing_targets_path.glob("*.csv"))
            if len(targets_files) == 0:
                logger.info(f"No existing targets file to recover")
                return
            existing_targets_file = targets_files[-1]
        existing_targets_file = Path(existing_targets_file)

        if not existing_targets_file.exists():
            logger.info(f"{existing_targets_file.file} missing")
            return
        if existing_targets_file.stat().st_size < 2:
            logger.info("file too small - don't attempt read...")
            return

        logger.info(f"recover targets from\n    {existing_targets_file}")
        existing_targets_df = pd.read_csv(existing_targets_file)
        logger.info(f"attempt {len(existing_targets_df)} targets")
        recovered_targets = []
        for ii, row in existing_targets_df.iterrows():
            objectId = row.objectId
            if objectId in self.target_lookup:
                logger.warning(f"skip load existing {objectId}")
                continue
            target = Target(objectId, ra=row.ra, dec=row.dec, base_score=row.base_score)
            # self.recover_score_history(target)
            # self.recover_rank_history(target)
            self.add_target(target)
            recovered_targets.append(objectId)
        logger.info(f"recovered {len(recovered_targets)} existing targets")
        return recovered_targets

    def recover_score_history(self, target: Target):
        score_history_file = self.get_score_history_file(target.objectId)
        if not score_history_file.exists():
            return
        score_history = pd.read_csv(score_history_file)
        for obs_name, history in score_history.groupby("observatory"):
            for ii, row in history.iterrows():
                score_t_ref = Time(row.mjd, format="mjd")
                target.update_score_history(row.score, obs_name, t_ref=score_t_ref)

    def recover_rank_history(self, target: Target):
        rank_history_file = self.get_rank_history_file(target.objectId)
        if not rank_history_file.exists():
            return
        rank_history = pd.read_csv(rank_history_file)
        for obs_name, history in rank_history.groupby("observatory"):
            for ii, row in history.iterrows():
                rank_t_ref = Time(row.mjd, format="mjd")
                target.update_rank_history(row["ranking"], obs_name, t_ref=rank_t_ref)

    def perform_messaging_tasks(self, t_ref: Time = None):
        """
        Loop through each target in TargetLookup.
        If there are any messages attached to a target, send them via the available messengers.
        Messages are (mostly) attached to targets in the TargetLookup.

        Parameters
        ----------
        t_ref
        """

        t_ref = t_ref or Time.now()

        skipped = []
        no_updates = []
        sent = []

        minimum_score = self.selector_parameters.get("minimum_score")
        logger.info("perform messaging tasks")
        for objectId, target in self.target_lookup.items():
            logger.debug(f"messaging for {objectId}")
            if len(target.update_messages) == 0:
                logger.debug(f"no messages")
                no_updates.append(objectId)
                continue
            if not target.updated:
                logger.debug(f"not updated; skip")
                skipped.append(objectId)
                continue
            last_score = target.get_last_score()
            if last_score is None:
                skipped.append(objectId)
                logger.debug(f"last score is None; skip")
                continue

            if last_score < minimum_score:
                skipped.append(objectId)
                logger.debug(f"last score: {last_score} < {minimum_score}; skip")
                continue

            intro = f"Updates for {objectId}"
            messages = [target.get_info_string()] + target.update_messages
            message_text = "\n".join(msg for msg in messages)

            lc_fig_path = self.get_lightcurve_plot_path(objectId)
            vis_fig_paths = []
            for obs_name in self.observatories.keys():
                if obs_name == "no_observatory":
                    continue
                vis_fig_paths.append(self.get_visibility_plot_path(objectId, obs_name))

            if self.telegram_messenger is not None:
                self.telegram_messenger.message_users(texts=message_text)
                self.telegram_messenger.message_users(img_paths=lc_fig_path)
                self.telegram_messenger.message_users(
                    texts="visibilities", img_paths=vis_fig_paths
                )
            if self.slack_messenger is not None:
                self.slack_messenger.send_messages(
                    texts=message_text, img_paths=lc_fig_path
                )
            sent.append(objectId)
            time.sleep(2.0)

        logger.info(f"no updates to send for {len(no_updates)} targets")
        logger.info(f"skipped messages for {len(skipped)} targets")
        logger.info(f"sent messages for {len(sent)} targets")
        return sent, skipped, no_updates

    def send_crash_reports(self, text: str = None):
        """
        Convenience method for sending most recent traceback via telegram to sudoers.

        Parameters
        ----------
        text : str or list of str, default=None
            str or list of str of text to accompany the traceback
        """
        if text is None:
            text = []
        if isinstance(text, str):
            text = [text]
        tr = text + [traceback.format_exc()]
        logger.error("\n" + "\n".join(tr))
        if self.telegram_messenger is not None:
            self.telegram_messenger.message_users(users="sudoers", texts=tr)

    def perform_iteration(
        self,
        scoring_function: Callable,
        observatory_scoring_function: Callable = None,
        modeling_function: Callable = None,
        lightcurve_compiler: Callable = None,
        lc_plotting_function: Callable = None,
        skip_tasks: list = None,
        t_ref: Time = None,
    ):
        """
        The actual prioritisation loop.

        Most of the parameters are callable functions, which you can provide.
        Only one is mandatory: scoring_function

        All but one of the user provided functions should have signature:
            func(target, t_ref) :
                target : dk154_targets.target.Target
                t_ref : astropy.time.Time
        The only exception is observatory_scoring_function, which should have signature
            func(target, obs: astroplan.Observer, t_ref)

        Parameters
        ----------
        scoring_function : Callable
            The (science) scoring function to prioritise targets.
            It should have the 'standard' signature defined above.
            It should return:
                score : float
                comments : List of str, optional
        observatory_scoring_function : Callable, optional
            The scoring function to evaluate at observatories.
                If not provided, defaults to dk154_targets.scoring.DefaultObsScore
                It should have signature func(target, obs, t_ref), where arguments:
                    target : dk154_targets.target.Target
                    obs : astroplan.Observer
                    t_ref : astropy.time.Time
            It should return:
                obs_factors : float
                    The output of this will be multiplied by 'score'
                    from scoring_function, to get the score for this observatory.
                comms : list of str, optional
        modeling_function : Callable or list of Callable
            Functions which produce models based on target.
            Exceptions in modeling_function are caught, and the model will be set to None.
            It should have the 'standard' signature defined above.
            It should return:
                model : Any
                    the model which describes your source. You can access it in
                    (eg.) scoring functions with target.models[<your_func_name>]
        lightcurve_compiler : Callable
            Function which produces a convenient single lightcurve including all data
            sources. Helpful in scoring, plotting.
            It should have the 'standard' signature defined above.
            It should return:
                compiled_lc : pd.DataFrame or astropy.table.Table
        lc_plotting_function : Callable
            Generate a lightcurve figure for each target.
            It should have the 'standard' signature defined above.
            it should return:
                figure : matplotlib.pyplot.Figure
        skip_tasks : list of str, optional
            Task(s) which should be skipped.
            Must be one of
                "qm_tasks", "obs_info", "pre_check", "modeling", "evaluate",
                "ranking", "reject", "plotting", "write_targets", "messaging",
        """

        t_ref = t_ref or Time.now()

        t_str = t_ref.strftime("%Y-%m-%d %H:%M:%S")
        utils.print_header(f"iteration at {t_str}")
        perf_times = {}

        # ================= Get some parameters from config ================= #
        write_comments = self.selector_parameters.get("write_comments", True)
        obs_info_dt = self.selector_parameters.get("obs_info_dt", 0.5 / 24.0)
        lazy_modeling = self.selector_parameters.get("lazy_modeling", True)
        lazy_compile = self.selector_parameters.get("lazy_compile", False)
        lazy_plotting = self.selector_parameters.get("lazy_plotting", True)
        # self.clear_scratch_plots() # NO - lazy plotting re-uses existing plots.

        # =============== Are there any tasks we should skip? =============== #
        config_skip_tasks = self.selector_parameters.get("skip_tasks", [])
        skip_tasks = skip_tasks or []
        skip_tasks = skip_tasks + config_skip_tasks
        invalid_skip_tasks = utils.check_unexpected_config_keys(
            skip_tasks, VALID_SKIP_TASKS, name="perform_iteration.skip_tasks"
        )
        if len(invalid_skip_tasks) > 0:
            errmsg = (
                f"invalid tasks in 'skip_tasks': {invalid_skip_tasks}\n"
                f"    choose from {VALID_SKIP_TASKS}"
            )
            raise ValueError(errmsg)

        # =========================== Get new data =========================== #
        self.check_for_targets_of_opportunity()
        if not "qm_tasks" in skip_tasks:
            self.perform_query_manager_tasks(t_ref=t_ref)
        else:
            logger.info("skip query manager tasks")

        # ================= Prep before modeling and scoring ================= #
        t1 = time.perf_counter()
        if not "obs_info" in skip_tasks:
            self.compute_observatory_info(t_ref=t_ref, dt=obs_info_dt)
        else:
            logger.info("skip compute new obs_info for each target")
        perf_times["obs_info"] = time.perf_counter() - t1

        t1 = time.perf_counter()
        self.compile_target_lightcurves(
            lightcurve_compiler=lightcurve_compiler, lazy=lazy_compile, t_ref=t_ref
        )
        perf_times["compile"] = time.perf_counter() - t1

        # =========== Remove any targets that aren't interesting. ============ #
        t1 = time.perf_counter()
        if not "pre_check" in skip_tasks:
            logger.info(f"{len(self.target_lookup)} targets before check")
            new_targets = self.new_target_initial_check(scoring_function, t_ref=t_ref)
            removed_before_modeling = self.remove_rejected_targets(
                objectId_list=new_targets, write_comments=write_comments
            )
            logger.info(
                f"reject {len(removed_before_modeling)} targets before modeling"
            )
        perf_times["precheck"] = time.perf_counter() - t1

        # =========================== Build models =========================== #
        t1 = time.perf_counter()
        if not "modeling" in skip_tasks:
            self.build_target_models(modeling_function, t_ref=t_ref, lazy=lazy_modeling)
        perf_times["modeling"] = time.perf_counter() - t1

        # ========================= Do the scoring, ========================== #
        t1 = time.perf_counter()
        if "evaluate" not in skip_tasks:
            self.evaluate_targets(
                scoring_function,
                observatory_scoring_function=observatory_scoring_function,
                t_ref=t_ref,
            )
            logger.info(f"{len(self.target_lookup)} targets after evaluation")
        perf_times["evaluate"] = time.perf_counter() - t1

        # ===================== Remove rejected targets ====================== #
        t1 = time.perf_counter()
        if "reject" not in skip_tasks:
            removed_targets = self.remove_rejected_targets(
                write_comments=write_comments, t_ref=t_ref
            )
            logger.info(f"rejected {len(removed_targets)} targets")
            logger.info(f"{len(self.target_lookup)} remain")
        perf_times["reject"] = time.perf_counter() - t1

        # ========================= Write comments ========================== #
        t1 = time.perf_counter()
        if write_comments:
            self.write_target_comments(t_ref=t_ref)
        else:
            logger.info("skip writing comments for targets")
        perf_times["write"] = time.perf_counter() - t1

        # ============================ Plotting ============================ #
        t1 = time.perf_counter()
        if not "plotting" in skip_tasks:
            plotting_interval = self.selector_parameters.get("plotting_interval", 0.25)
            if lazy_plotting:
                logger.info(f"re-use vis/lc plots <{plotting_interval*24:.1f}hr old")

            self.plot_target_lightcurves(
                plotting_function=lc_plotting_function,
                lazy=lazy_plotting,
                interval=plotting_interval,
                t_ref=t_ref,
            )
            self.plot_target_visibilities(
                lazy=lazy_plotting, interval=plotting_interval, t_ref=t_ref
            )
        else:
            logger.info("skip plotting")
        perf_times["plotting"] = time.perf_counter() - t1

        # ============================= Ranking ============================= #
        t1 = time.perf_counter()
        if "ranking" not in skip_tasks:
            self.clear_output_plots()  # In prep for the new outputs
            self.build_ranked_target_lists(t_ref=t_ref, plots=True, write_list=True)
        perf_times["ranking"] = time.perf_counter() - t1

        # =============== Checkpoint the current target list ================ #
        t1 = time.perf_counter()
        if "write_targets" not in skip_tasks:
            self.write_existing_target_list(t_ref=t_ref)
            # self.write_score_histories(t_ref=t_ref)
            # self.write_rank_histories(t_ref=t_ref)
        perf_times["checkpoint"] = time.perf_counter() - t1

        # ====================== Broadcast any messages ====================== #
        t1 = time.perf_counter()
        if "messaging" not in skip_tasks:
            self.perform_messaging_tasks()
        else:
            logger.info("skip messaging tasks")

        # ======== Reset all targets to un-updated for the next loop ========= #
        self.reset_updated_targets()

        perf_times["messaging"] = time.perf_counter() - t1

        print(
            f"time summary:\n    "
            + f"\n    ".join(f"{k}={v:.5e}" for k, v in perf_times.items())
        )

        return None

    def start(
        self,
        scoring_function: Callable = None,
        observatory_scoring_function: Callable = None,
        modeling_function: List[Callable] = None,
        lightcurve_compiler: Callable = None,
        lc_plotting_function: Callable = None,
        existing_targets_file=False,
        skip_tasks=None,
        iterations=None,
    ):
        """
        A convenience function to perform iterations

        Parameters
        ----------
        scoring_function: Callable
            the user-built scoring function
        modeling_function: Callable or List[Callable]
            function(s) to build models for targets
        lightcurve_compiler: Callable [optional]

        lc_plotting_function: Callable

        existing_targets_file: optional, default=False
            path to an existing_targets_file, or "last"


        Examples
        --------
        >>> from dk154_targets import paths
        >>> from dk154_targets import TargetSelector
        >>> from dk154_targets.scoring.example_functions import latest_flux

        >>> config_path = paths.config_path / "examples/fink_supernovae.yaml"
        >>> selector = TargetSelector.from_config(config_path)
        >>> selector.start(scoring_function=latest_flux)

        """
        t_ref = Time.now()

        # ===================== Get and set some parameters ================== #
        N_iterations = 0
        sleep_time = self.selector_parameters.get("sleep_time")
        if lightcurve_compiler is None:
            lightcurve_compiler = DefaultLightcurveCompiler()

        if existing_targets_file:
            self.recover_existing_targets(existing_targets_file=existing_targets_file)

        if self.telegram_messenger is not None:
            # Send some messages on start-up.
            try:
                nodename = os.uname().nodename
            except Exception as e:
                nodename = "<unknown node>"
            msg = (
                f"starting at {t_ref.isot} on {nodename} with:\n"
                f"observatories:\n    {', '.join(k for k in self.observatories)}\n"
                f"query_managers:\n    {', '.join(k for k in self.query_managers)}\n"
                f"modeling_function: {modeling_function.__name__}\n"
                f"scoring_function: {scoring_function.__name__}"
            )
            self.telegram_messenger.message_users(texts=msg, users="sudoers")

        # ========================= loop indefinitely ======================== #
        while True:
            t_ref = Time.now()

            loop_skip_tasks = skip_tasks or []
            if N_iterations == 0:
                loop_skip_tasks.append("messaging")

            try:
                self.perform_iteration(
                    scoring_function=scoring_function,
                    observatory_scoring_function=observatory_scoring_function,
                    modeling_function=modeling_function,
                    lightcurve_compiler=lightcurve_compiler,
                    lc_plotting_function=lc_plotting_function,
                    skip_tasks=loop_skip_tasks,
                    t_ref=t_ref,
                )
            except Exception as e:
                t_str = t_ref.strftime("%Y-%m-%d %H:%M:%S")
                crash_text = [f"CRASH at UT {t_str}"]
                self.send_crash_reports(text=crash_text)
                sys.exit()

            # Some post-loop tasks
            N_iterations = N_iterations + 1
            if iterations is not None:
                if N_iterations >= iterations:
                    break

            logger.info(f"sleep for {sleep_time} sec")
            time.sleep(sleep_time)
