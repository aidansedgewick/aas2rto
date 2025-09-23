import json
import logging
import os
import shutil
import sys
import time
import traceback
import warnings
import yaml
from functools import partial
from multiprocessing import Pool
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

from aas2rto import query_managers
from aas2rto import utils
from aas2rto.lightcurve_compilers import DefaultLightcurveCompiler
from aas2rto.messaging.messaging_manager import MessagingManager
from aas2rto.modeling.utils import modeling_wrapper, pool_modeling_wrapper
from aas2rto.obs_info import ObservatoryInfo
from aas2rto.outputs_manager import OutputsManager
from aas2rto.path_manager import PathManager
from aas2rto.plotting import PlottingManager, plot_default_lightcurve, plot_visibility
from aas2rto.recovery_manager import RecoveryManager
from aas2rto.scoring.default_obs_scoring import DefaultObservatoryScoring
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup

from aas2rto import paths

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
    selector_config : dict
        A dictionary, with keys {expected_config_keys}
    create_paths : bool
    """

    expected_config_keys = (
        "selector_parameters",
        "query_managers",
        "observatories",
        "messaging",
        "paths",
        "recovery",
        "plotting",
        "outputs",
    )
    default_selector_parameters = {
        "project_name": None,
        "sleep_time": 600.0,
        "ncpu": None,
        "skip_tasks": [],
        "obs_info_dt": 0.5 / 24.0,
        "obs_info_update": 2.0 / 24.0,
        "minimum_score": 0.0,
        "lazy_modeling": True,
        "lazy_compile": True,
        "consolidate_seplim": 5 * u.arcsec,
        "write_comments": True,
    }
    expected_messenger_keys = ("telegram", "slack", "git_web")

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

        self.paths_config = self.selector_config.get("paths", {})
        self.query_managers_config = self.selector_config.get("query_managers", {})
        self.observatories_config = self.selector_config.get("observatories", {})

        self.plotting_config = self.selector_config.get("plotting", {})
        self.outputs_config = self.selector_config.get("outputs", {})
        self.recovery_config = self.selector_config.get("recovery", {})
        self.messaging_config = self.selector_config.get("messaging", {})

        # to keep the targets. Do this here as initQM needs to know about it.
        self.target_lookup = TargetLookup()  # self._create_empty_target_lookup()
        self.ranked_lists = {}

        # Initialise PathManager first - others depend on it.
        self.path_manager = PathManager(self.paths_config)

        # Initialise some other managers.
        self.plotting_manager = PlottingManager(
            self.plotting_config, self.target_lookup, self.path_manager
        )
        self.outputs_manager = OutputsManager(
            self.outputs_config, self.target_lookup, self.path_manager
        )
        self.recovery_manager = RecoveryManager(
            self.recovery_config, self.target_lookup, self.path_manager
        )
        self.messaging_manager = MessagingManager(
            self.messaging_config, self.target_lookup, self.path_manager
        )
        self.initialize_query_managers(create_paths=create_paths)
        self.initialize_observatories()

    def __setitem__(self, target_id, target):
        if not isinstance(target, Target):
            class_name = target.__class__.__name__
            msg = f"Cannot add {target_id} (type={class_name}) to target list."
            raise ValueError(msg)
        self.target_lookup[target_id] = target

    def __getitem__(self, target_id):
        return self.target_lookup[target_id]

    def __iter__(self):
        for target, target_id in self.target_lookup.items():
            yield target, target_id

    def __contains__(self, member):
        return member in self.target_lookup

    @classmethod
    def from_config(cls, config_path, create_paths=True):
        config_path = Path(config_path)

        with open(config_path, "r") as f:
            selector_config = yaml.load(f, Loader=yaml.FullLoader)
        selector = cls(selector_config, create_paths=create_paths)
        return selector

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
            qm_kwargs = dict(
                parent_path=self.path_manager.data_path, create_paths=create_paths
            )
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
            msg = f"new query manager {qm_name} should have class attribute 'name'"
            logger.warning(msg)

        has_tasks_method = hasattr(QueryManagerClass, "perform_all_tasks")
        if not has_tasks_method:
            msg = f"query manager {qm_name} must have method 'perform_all_tasks()'"
            raise AttributeError(msg)

        qm_config.update(kwargs)
        qm = QueryManagerClass(
            qm_config,
            self.target_lookup,
            parent_path=self.path_manager.data_path,
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

    def add_target(self, target: Target):
        if target.target_id in self.target_lookup:
            raise ValueError(f"obj {target.target_id} already in target_lookup")
        self.target_lookup[target.target_id] = target

    def compute_observatory_info(
        self, t_ref: Time = None, horizon: u.Quantity = -18 * u.deg, dt=0.5 / 24.0
    ):
        """
        Precompute some expensive information about the altitude for each observatory.

        access in a target with eg. `my_target.observatory_info["lasilla"]`
        """
        t_ref = t_ref or Time.now()
        obs_info_update = self.selector_parameters["obs_info_update"]

        counter = {}
        for obs_name, observatory in self.observatories.items():
            if observatory is None:
                continue
            msg = f"{obs_name}: obs_info missing/older than {obs_info_update*24:.1f}hr"
            logger.info(msg)

            counter = 0
            obs_info = ObservatoryInfo.for_observatory(
                observatory, t_ref=t_ref, horizon=horizon, dt=dt
            )
            for target_id, target in self.target_lookup.items():
                existing_info = target.observatory_info.get(obs_name)
                if existing_info is not None:
                    if t_ref.mjd - existing_info.t_ref.mjd < obs_info_update:
                        continue
                target_obs_info = obs_info.copy()
                assert target_obs_info.target_altaz is None
                if target.coord is not None:
                    target_obs_info.set_target_altaz(target.coord, observatory)
                target.observatory_info[obs_name] = target_obs_info
                if target.observatory_info[obs_name].target_altaz is None:
                    msg = f"\033[33m{target_id} {obs_name} altaz missing\033[0m"
                    logger.warning(msg)
                counter = counter + 1
            logger.info(f"updated obs_info for {counter} targets")

    def check_for_targets_of_opportunity(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        logger.info("check for targets of opportunity")
        opp_target_file_list = self.path_manager.opp_targets_path.glob("*.yaml")
        failed_targets = []
        existing_targets = []
        successful_targets = []
        for opp_target_file in opp_target_file_list:
            with open(opp_target_file, "r") as f:
                target_config = yaml.load(f, Loader=yaml.FullLoader)
                target_id = target_config.get("target_id", None)
                ra = target_config.get("ra", None)
                dec = target_config.get("dec", None)
                msg = (
                    f"Your config has {target_config.keys()}. "
                    "You should provide minimum of: "
                    "\033[31;1mtarget_id, ra, dec\033[0m"  # codes colour red.
                )
                if any([x is None for x in [target_id, ra, dec]]):
                    logger.warning(msg)
                    failed_targets.append(opp_target_file)
                    continue
                base_score = target_config.get("base_score", None)
            if target_id in self.target_lookup:
                logger.info(f"{target_id} already in target list!")
                existing_target = self.target_lookup[target_id]
                if base_score is not None:
                    existing_target.base_score = base_score
                existing_target.target_of_opportunity = True
                existing_targets.append(opp_target_file)
            else:
                opp_target = Target(
                    target_id, ra=ra, dec=dec, base_score=base_score, t_ref=t_ref
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
        lightcurve_compiler : Callable, optional
            your function that builds a convenient single lightcurve.
            see `aas2rto.lightcurve_compilers.DefaultLigthcurveCompiler` for example.
        t_ref : astropy.time.Time, optional
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
        for target_id, target in self.target_lookup.items():
            compiled_exists = target.compiled_lightcurve is not None
            if compiled_exists and (not target.updated) and lazy:
                skipped.append(target_id)
                continue
            compiled_lc = lightcurve_compiler(target, t_ref=t_ref)
            target.compiled_lightcurve = compiled_lc
            if compiled_lc is None:
                failed.append(target_id)
                continue
            compiled.append(target_id)

        logger.info(f"compiled:{len(compiled)}, skipped (lazy):{len(skipped)}")
        if len(failed) > 0:
            logger.warning(f"failed:{len(failed)} (no compiled lc!)")
        return compiled, failed

    def perform_query_manager_tasks(self, startup=False, t_ref: Time = None):
        """
        for each of the query managers `qm` in target_selector.query_managers,
        call the method `qm.perform_all_tasks(t_ref=t_ref)`

        Parameters
        ----------
        t_ref : astropy.time.Time, default=Time.now()
            used in the perform_all_tasks() call.
        """

        t_ref = t_ref or Time.now()

        logger.info("begin query manager tasks")
        for qm_name, qm in self.query_managers.items():
            t_start = time.perf_counter()
            logger.info(f"begin {qm_name} tasks")
            query_result = qm.perform_all_tasks(startup=startup, t_ref=t_ref)
            if isinstance(query_result, Exception):
                text = [f"EXCEPTION IN {qm_name} [no crash]"]
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

        for target_id, target in self.target_lookup.items():
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

        # ===== Compute score according to science scoring ===== #
        science_score, score_comments = self._evaluate_target_science_score(
            science_scoring_function, target, t_ref=t_ref
        )

        obs_name = "no_observatory"
        target.update_score_history(science_score, obs_name, t_ref=t_ref)
        target.score_comments[obs_name] = score_comments

        if observatory_scoring_function is None:
            return

        for obs_name, observatory in self.observatories.items():
            if observatory is None:
                if not obs_name == "no_observatory":
                    raise ValueError(f"observatory {obs_name} is None!")
                continue

            if science_score > minimum_score and np.isfinite(science_score):
                # ===== Modify score for each observatory
                obs_factors, score_comments = self._evaluate_target_observatory_score(
                    observatory_scoring_function, target, observatory, t_ref
                )
                observatory_score = science_score * obs_factors
            else:
                obs_factors = 1.0
                observatory_score = science_score
                score_comments = ["excluded by no_observatory (science) score"]

            target.update_score_history(observatory_score, obs_name, t_ref=t_ref)
            target.score_comments[obs_name] = score_comments
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
                f"    For target {target.target_id} at obs {obs_name} at {t_ref.isot},\n"
                f"    scoring with {science_scoring_function.__name__} failed.\n"
                f"    Set score to -1.0, to exclude.\n"
            )
            scoring_res = (-1.0, [details])
            self.send_crash_reports(text=details)

        func_name = science_scoring_function.__name__
        science_score, score_comments = self._parse_scoring_result(
            scoring_res, func_name=func_name
        )
        return science_score, score_comments

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
                f"    For target {target.target_id} at obs {obs_name} at {t_ref.isot},\n"
                f"    scoring with {observatory_scoring_function.__name__} failed.\n"
                f"    Set score to -1.0, to exclude.\n"
            )
            scoring_res = (-1.0, [details])
            self.send_crash_reports(text=details)

        func_name = observatory_scoring_function.__name__
        obs_factors, obs_comments = self._parse_scoring_result(scoring_res)
        if not np.isfinite(obs_factors):
            msg = (
                f"observatory_score not finite (={obs_factors}) "
                f"for {target.target_id} at {obs_name}."
            )
            logger.warning(msg)

        return obs_factors, obs_comments

    def _parse_scoring_result(self, scoring_res, func_name=""):
        err_msg = (
            f"scoring function {func_name} should return\n"
            "score [float] or tuple of (score [float], comments [list[str]])"
        )

        if isinstance(scoring_res, tuple):
            if len(scoring_res) == 2:
                score, score_comments = scoring_res
            else:
                raise ValueError(err_msg)
        elif isinstance(scoring_res, float) or isinstance(scoring_res, int):
            score = scoring_res
            score_comments = [f"function {func_name}: no score_comments provided"]
        else:
            raise ValueError(err_msg)

        return score, score_comments

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
        for target_id, target in self.target_lookup.items():
            last_score = target.get_last_score("no_observatory")
            if last_score is not None:
                continue
            self.evaluate_single_target(
                target,
                scoring_function,
                observatory_scoring_function=None,
                t_ref=t_ref,
            )
            new_targets.append(target_id)
        return new_targets

    def remove_rejected_targets(
        self, target_id_list=None, t_ref=None, write_comments=True
    ) -> List[Target]:
        t_ref = t_ref or Time.now()

        if target_id_list is None:
            target_id_list = list(self.target_lookup.keys())

        removed_targets = []
        for target_id in target_id_list:
            target = self.target_lookup.get(target_id, None)
            if target is None:
                logger.warning(f"can't remove non-existent target {target_id}")
                continue
            last_score = target.get_last_score()  # at no_observatory.
            if last_score is None:
                logger.warning(f"in reject: {target_id} has no score")
            if np.isfinite(last_score):
                continue  # it can stay...
            target = self.target_lookup.pop(target_id)
            if write_comments:
                target.write_comments(
                    self.path_manager.rejected_targets_path, t_ref=t_ref
                )
                target.write_comments(self.path_manager.comments_path, t_ref=t_ref)
            removed_targets.append(target)
            assert target_id not in self.target_lookup
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
        for modeling_func in modeling_functions:

            try:
                model_key = modeling_func.__name__
            except AttributeError as e:
                model_key = type(modeling_func).__name__
                msg = f"\n    Your modeling_function {model_key} should have attribute __name__."
                logger.warning(msg)

            skipped = []
            targets_to_model = []
            for target_id, target in self.target_lookup.items():
                model_exists = model_key in target.models
                if (not target.updated) and lazy and model_exists:
                    skipped.append(target_id)
                    continue
                targets_to_model.append(target)

            if lazy and len(skipped) > 0:
                logger.info(
                    f"skip {len(skipped)} non-updated targets (lazy_modeling=True)"
                )

            if len(targets_to_model) == 0:
                logger.info("no targets to model - continue!")
                continue  # NOT return - we are still in a loop...

            logger.info(f"build {model_key} models for {len(targets_to_model)}")
            ncpu = self.selector_parameters.get("ncpu", None)
            if ncpu is not None:
                logger.info("cannot currently multiprocess models")
            serial = True
            if serial:
                logger.info("build in serial")
                result_list = []
                for target in targets_to_model:
                    result = modeling_wrapper(modeling_func, target, t_ref=t_ref)
                    result_list.append(result)
            else:
                if not isinstance(ncpu, int):
                    raise ValueError(f"'ncpu' must be integer, not {type(ncpu)}")
                logger.info(f"build with {ncpu} workers")
                with Pool(ncpu) as p:
                    args_kwargs = [
                        ((modeling_func, target), dict(t_ref=t_ref))
                        for target in targets_to_model
                    ]
                    result_list = p.map(pool_modeling_wrapper, args_kwargs)

            built = []
            failed = []
            no_model = []
            fail_str = ""
            for result in result_list:
                target_id = result.target_id
                target = self.target_lookup.get(target_id)
                target.models[model_key] = result.model
                if result.success:
                    if result.model is None:
                        no_model.append(target_id)
                    else:
                        built.append(target_id)
                else:
                    failed.append(target_id)
                    fail_str = fail_str + f"{target_id}: {result.reason}\n"

            logger.info(f"{model_key} built:{len(built)}, {len(no_model)} 'None'")
            if len(failed) > 0:
                logger.warning(f"failed: {len(failed)}, reasons:\n{fail_str}")

    def build_ranked_target_lists(
        self, plots=True, write_list=True, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        logger.info("build ranked target lists")
        for obs_name, observatory in self.observatories.items():
            ranked_list = self.outputs_manager.build_ranked_target_list_at_observatory(
                observatory, plots=plots, write_list=write_list, t_ref=t_ref
            )
            self.ranked_lists[obs_name] = ranked_list

    def recover_targets_from_file(self, recovery_file: str):

        recovered_targets = self.recovery_manager.recover_targets_from_file(
            recovery_file=recovery_file
        )
        for target in recovered_targets:
            self.add_target(target)
        self.target_lookup.update_target_id_mappings()

    def clear_output_plots(self, fig_fmt="png"):
        for obs_name, observatory in self.observatories.items():
            plot_dir = self.path_manager.get_output_plots_path(obs_name, mkdir=False)
            if plot_dir.exists():
                for plot in plot_dir.glob(f"*.{fig_fmt}"):
                    os.remove(plot)

    def reset_target_figures(self):
        for target_id, target in self.target_lookup.items():
            target.reset_figures()

    def reset_updated_targets(self, t_ref: Time = None):
        for target_id, target in self.target_lookup.items():
            target.updated = False
            target.send_updates = False
            target.update_messages = []

    def perform_web_tasks(self, t_ref: Time = None):
        # TODO: somehow move to messaging_manager
        t_ref = t_ref or Time.now()
        if self.messaging_manager.git_webpage_manager is not None:
            self.messaging_manager.git_webpage_manager.update_webpages(
                self.target_lookup, self.ranked_lists, t_ref=t_ref
            )

    def send_crash_reports(self, text: str = None):
        self.messaging_manager.send_crash_reports(text=text)

    def perform_iteration(
        self,
        scoring_function: Callable,
        observatory_scoring_function: Callable = None,
        modeling_function: Callable = None,
        lightcurve_compiler: Callable = None,
        lc_plotting_function: Callable = None,
        extra_plotting_functions: List[Callable] = None,
        skip_tasks: list = None,
        startup: bool = False,
        t_ref: Time = None,
    ):
        """
        Perform a single iteration.

        Parameters
        ----------
        scoring_function : Callable
        observatory_scoring_function : Callable, optional
        modeling_function : Callable, optional
        lightcurve_compiler : Callable, optional
        lc_plotting_function : Callable, optional
        skip_tasks : list, optional
        startup : bool, default=False
            passed to the "perform_all_tasks()" function for query_managers.
            useful for skipping certain steps on first iteration.
        t_ref : `astropy.time.Time`, default=Time.now()
            the reference time at the start of the iteration.
            useful for simulations.

        See 'TargetSelector.start() documentation for details of user-provided
        Callable functions.

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
        seplimit = self.selector_parameters.get("consolidate_seplimit", 5 * u.arcsec)
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
        t1 = time.perf_counter()
        self.check_for_targets_of_opportunity()
        if not "qm_tasks" in skip_tasks:
            self.perform_query_manager_tasks(startup=startup, t_ref=t_ref)
        else:
            logger.info("skip query manager tasks")
        perf_times["qm_tasks"] = time.perf_counter() - t1

        # ================== Merge + sort duplicated targets ================= #
        if isinstance(self.target_lookup, TargetLookup):
            self.target_lookup.consolidate_targets(seplimit=seplimit)
            self.target_lookup.update_target_id_mappings()
            self.target_lookup.update_to_preferred_target_id()

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

        # ============ Remove any targets that aren't interesting ============ #
        t1 = time.perf_counter()
        if not "pre_check" in skip_tasks:
            logger.info(f"{len(self.target_lookup)} targets before check")
            new_targets = self.new_target_initial_check(scoring_function, t_ref=t_ref)
            removed_before_modeling = self.remove_rejected_targets(
                target_id_list=new_targets, write_comments=write_comments
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

        # ========================== Do the scoring ========================== #
        t1 = time.perf_counter()
        if "evaluate" not in skip_tasks:
            if scoring_function is None:
                msg = "You must provide a scoring function."
                raise ValueError(msg)
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
            self.outputs_manager.write_target_comments(t_ref=t_ref)
        else:
            logger.info("skip writing comments for targets")
        perf_times["write"] = time.perf_counter() - t1

        # ============================ Plotting ============================ #
        t1 = time.perf_counter()
        if not "plotting" in skip_tasks:
            plotting_interval = self.selector_parameters.get("plotting_interval", 0.25)

            self.plotting_manager.plot_all_target_lightcurves(
                plotting_function=lc_plotting_function, t_ref=t_ref
            )
            for obs_name, observatory in self.observatories.items():
                self.plotting_manager.plot_all_target_visibilities(
                    observatory, t_ref=t_ref
                )

            # self.plotting_manager.plot_rank_histories(t_ref=t_ref)
            if extra_plotting_functions is not None:
                for plotting_func in extra_plotting_functions:
                    self.plotting_manager.plot_additional_figures(
                        plotting_func, t_ref=t_ref
                    )

        else:
            logger.info("skip plotting")
        perf_times["plotting"] = time.perf_counter() - t1

        # ======================== Ranking and lists ======================== #
        t1 = time.perf_counter()
        if "ranking" not in skip_tasks:
            self.clear_output_plots()  # In prep for the new outputs
            self.build_ranked_target_lists(t_ref=t_ref, plots=True, write_list=True)
            # self.build_visible_target_lists(t_ref=t_ref, plots=True, write_list=True)
        perf_times["ranking"] = time.perf_counter() - t1

        # =============== Checkpoint the current target list ================ #
        t1 = time.perf_counter()
        if "write_targets" not in skip_tasks:
            self.recovery_manager.write_recovery_file(t_ref=t_ref)
            # self.write_score_histories(t_ref=t_ref)
            self.recovery_manager.write_rank_histories(t_ref=t_ref)
        perf_times["recovery"] = time.perf_counter() - t1

        # ====================== Broadcast any messages ====================== #
        t1 = time.perf_counter()
        if "messaging" not in skip_tasks:
            self.messaging_manager.perform_messaging_tasks()
        else:
            logger.info("skip messaging tasks")
        perf_times["messaging"] = time.perf_counter() - t1

        # ========================== Perform web tasks ======================= #
        t1 = time.perf_counter()
        if "web_tasks" not in skip_tasks:
            self.perform_web_tasks(t_ref=t_ref)
        else:
            logger.info("skip web tasks")
        perf_times["web_tasks"] = time.perf_counter() - t1

        # ======== Reset all targets to un-updated for the next loop ========= #
        self.reset_updated_targets()

        print(
            f"time summary:\n    "
            + f"\n    ".join(f"{k:10} = {v: 6.2f}s" for k, v in perf_times.items())
        )

        return None

    def start(
        self,
        scoring_function: Callable = None,
        observatory_scoring_function: Callable = None,
        modeling_function: List[Callable] = None,
        lightcurve_compiler: Callable = None,
        lc_plotting_function: Callable = None,
        extra_plotting_functions: List[Callable] = None,
        recovery_file=False,
        skip_tasks=None,
        iterations=None,
    ):
        """
        The actual prioritisation loop.

        Most of the parameters are callable functions, which you can provide.
        Only one is required: scoring_function

        All but one of the Callable types (functions) should have a 'standard' signature:
            `func(target, t_ref)`

        The only exception is observatory_scoring_function, which should have signature
            `func(target, obs: astroplan.Observer, t_ref)`

        Parameters
        ----------
        scoring_function : Callable
            The (science) scoring function to prioritise targets.
            It should have the 'standard' signature defined above.
            It should return:\n
                `score : float`
                `comments : List of str, optional`
        observatory_scoring_function : Callable, optional
            The scoring function to evaluate at observatories.
            If not provided, defaults to `aas2rto.scoring.DefaultObsScore`.
            It should have signature func(target, obs, t_ref), where arguments:
                `target` : `aas2rto.target.Target`
                `obs` : `astroplan.Observer`
                `t_ref` : `astropy.time.Time`

            The output of this will be multiplied by 'score'
            It should return:
                obs_factors : float
                    from scoring_function, to get the score for this observatory.
                comms : list of str, optional
        modeling_function : Callable or list of Callable
            Functions which produce models based on target.
            Exceptions in modeling_function are caught, and the model will be set to None.
            It should have the 'standard' signature defined above.
            It should return:\n
                `model` : `Any`\n
                    the model which describes your source. You can access it in
                    (eg.) scoring functions with target.models[<your_func_name>]
        lightcurve_compiler : Callable
            Function which produces a convenient single lightcurve including all data
            sources. Helpful in scoring, plotting.
            It should have the 'standard' signature defined above.
            It should return:
                `compiled_lc` : `pd.DataFrame` or `astropy.table.Table`
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
        iterations : int, optional
            How many iterations to perform before exiting
            default infinite loop.

        Examples
        --------
        >>> from aas2rto import paths
        >>> from aas2rto import TargetSelector
        >>> from aas2rto.scoring.example_functions import latest_flux

        >>> config_path = paths.config_path / "examples/fink_supernovae.yaml"
        >>> selector = TargetSelector.from_config(config_path)
        >>> selector.start(scoring_function=latest_flux)

        """
        t_ref = Time.now()

        # ===================== Get and set some parameters ================== #
        N_iterations = 0
        sleep_time = self.selector_parameters.get("sleep_time")

        if scoring_function is None:
            msg = "You must provide scoring_function=<some-callable>."
            raise ValueError(msg)

        if observatory_scoring_function is None:
            observatory_scoring_function = DefaultObservatoryScoring()
        if lightcurve_compiler is None:
            lightcurve_compiler = DefaultLightcurveCompiler()

        if recovery_file:
            self.recover_targets_from_file(recovery_file=recovery_file)

        # Send some messages on start-up.
        try:
            nodename = os.uname().nodename
        except Exception as e:
            nodename = "<unknown node>"
        msg = (
            f"starting at {t_ref.isot} on {nodename} with:\n"
            f"observatories:\n    {', '.join(k for k in self.observatories)}\n"
            f"query_managers:\n    {', '.join(k for k in self.query_managers)}\n"
            f"modeling_function:\n    {modeling_function.__name__}\n"
            f"scoring_function:\n    {scoring_function.__name__}\n"
            f"observatory_scoring_function:\n    {observatory_scoring_function.__name__}\n"
        )
        self.messaging_manager.send_sudo_messages(texts=msg)

        # ========================= loop indefinitely ======================== #
        while True:
            t_ref = Time.now()

            t_start = time.perf_counter()
            loop_skip_tasks = skip_tasks or []
            if N_iterations == 0:
                # Don't send messages on the first loop.
                # If many targets are recovered, 100s of messages could be sent...
                loop_skip_tasks.append("messaging")
                loop_skip_tasks.append("write_targets")
                # loop_skip_tasks.append("plotting")
                startup = True
            else:
                startup = False  # skips some query_manager tasks on startup

            try:
                self.perform_iteration(
                    scoring_function=scoring_function,
                    observatory_scoring_function=observatory_scoring_function,
                    modeling_function=modeling_function,
                    lightcurve_compiler=lightcurve_compiler,
                    lc_plotting_function=lc_plotting_function,
                    extra_plotting_functions=extra_plotting_functions,
                    skip_tasks=loop_skip_tasks,
                    startup=startup,
                    t_ref=t_ref,
                )
            except Exception as e:
                t_str = t_ref.strftime("%Y-%m-%d %H:%M:%S")
                crash_text = [f"CRASH at UT {t_str}"]
                self.messaging_manager.send_crash_reports(text=crash_text)
                sys.exit()

            # Some post-loop tasks
            N_iterations = N_iterations + 1
            if iterations is not None:
                if N_iterations >= iterations:
                    break

            if startup:
                logger.info("no sleep after startup")
            else:
                exc_time = time.perf_counter() - t_start
                sleep_time_actual = max(sleep_time - exc_time, 60.0)

                logger.info(f"loop execution time = {exc_time:.1f}sec")
                logger.info(f"...so sleep for {sleep_time_actual:.1f}sec")
                time.sleep(sleep_time_actual)
