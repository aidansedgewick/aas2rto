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

from aas2rto import utils
from aas2rto.lightcurve_compilers import DefaultLightcurveCompiler
from aas2rto.messaging.messaging_manager import MessagingManager
from aas2rto.modeling.modeling_manager import ModelingManager
from aas2rto.scoring.scoring_manager import ScoringManager
from aas2rto.observatory.ephem_info import EphemInfo
from aas2rto.observatory.observatory_manager import ObservatoryManager
from aas2rto.outputs.outputs_manager import OutputsManager
from aas2rto.path_manager import PathManager
from aas2rto.query_managers.primary import PrimaryQueryManager
from aas2rto.plotting import PlottingManager, plot_default_lightcurve, plot_visibility
from aas2rto.recovery.recovery_manager import RecoveryManager
from aas2rto.scoring.default_obs_scoring import DefaultObservatoryScoring
from aas2rto.web.web_manager import WebManager
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
    "sleep",
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
        "selector",
        "query_managers",
        "observatories",
        "messaging",
        "modeling",
        "scoring",
        "paths",
        "recovery",
        "plotting",
        "outputs",
        "web",
    )
    default_selector_config = {
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
        "delete_opp_target_configs": True,
    }
    # expected_messenger_keys = ("telegram", "slack", "web")

    def __init__(self, aas2rto_config: dict, create_paths: bool = True):
        # Check config sections
        utils.check_unexpected_config_keys(
            aas2rto_config,
            self.expected_config_keys,
            name="selector_config",
            raise_exc=True,
        )
        self.aas2rto_config = aas2rto_config

        # Top level section
        self.selector_config = self.default_selector_config.copy()
        selector_config = aas2rto_config.get("selector", {})
        self.selector_config.update(selector_config)
        utils.check_unexpected_config_keys(
            self.selector_config,
            self.default_selector_config,
            name="selector",
            raise_exc=True,
        )

        # Unpack configs
        self.paths_config = aas2rto_config.get("paths", {})
        self.observatories_config = aas2rto_config.get("observatories", {})

        self.query_managers_config = aas2rto_config.get("query_managers", {})
        self.scoring_config = aas2rto_config.get("scoring", {})
        self.modeling_config = aas2rto_config.get("modeling", {})
        self.plotting_config = aas2rto_config.get("plotting", {})
        self.outputs_config = aas2rto_config.get("outputs", {})
        self.recovery_config = aas2rto_config.get("recovery", {})
        self.messaging_config = aas2rto_config.get("messaging", {})
        self.web_config = aas2rto_config.get("web", {})

        # Init TargetLookup first as everything else depends on it.
        self.target_lookup = TargetLookup()

        # Initialise PathManager next - others Managers depend on it.
        self.path_manager = PathManager(self.paths_config, create_paths=create_paths)

        # Initialise ObservatoryManager next - others depend on obs_manager and path_manager
        self.observatory_manager = ObservatoryManager(
            self.observatories_config, self.target_lookup, self.path_manager
        )

        # Initialise some which need target_lookup + path_manager
        self.recovery_manager = RecoveryManager(
            self.recovery_config, self.target_lookup, self.path_manager
        )
        self.primary_query_manager = PrimaryQueryManager(
            self.query_managers_config,
            self.target_lookup,
            self.path_manager,
        )
        self.modeling_manager = ModelingManager(
            self.modeling_config, self.target_lookup, self.path_manager
        )
        self.messaging_manager = MessagingManager(
            self.messaging_config, self.target_lookup, self.path_manager
        )

        # Now some which need t_lookup + path_manager AND observatory_manager.
        self.scoring_manager = ScoringManager(
            self.scoring_config,
            self.target_lookup,
            self.path_manager,
            self.observatory_manager,
        )
        self.plotting_manager = PlottingManager(
            self.plotting_config,
            self.target_lookup,
            self.path_manager,
            self.observatory_manager,
        )
        self.outputs_manager = OutputsManager(
            self.outputs_config,
            self.target_lookup,
            self.path_manager,
            self.observatory_manager,
        )

        # Finally - web manager is a little peculiar in that it depends
        # on OUTPUTS_MANAGER, not target_lookup
        self.web_manager = WebManager(
            self.web_config, self.outputs_manager, self.path_manager
        )

    @classmethod
    def from_config(cls, config_path: Path, create_paths: bool = True):
        config_path = Path(config_path)

        with open(config_path, "r") as f:
            selector_config = yaml.load(f, Loader=yaml.FullLoader)
        selector = cls(selector_config, create_paths=create_paths)
        return selector

    def load_targets_of_opportunity(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        logger.info("load targets of opportunity")
        opp_target_file_list = self.path_manager.opp_targets_path.glob("*.yaml")
        failed_targets = []
        successful_targets = []
        for opp_target_filepath in opp_target_file_list:
            try:
                self.target_lookup.add_target_from_file(
                    opp_target_filepath, t_ref=t_ref
                )
                successful_targets.append(opp_target_filepath)
            except Exception as e:
                logger.error(f"target in file '{opp_target_filepath.name}' failed")
                logger.error(f"{type(e)}: {e}")
                failed_targets.append(opp_target_filepath)

        if len(failed_targets) > 0:
            failed_list = "\n".join(f"    {f.name}" for f in failed_targets)
            msg = f"failed to add {len(failed_targets)} targets:\n{failed_targets}"
            logger.warning(msg)
        if len(successful_targets) > 0:
            logger.info(f"{len(successful_targets)} new opp targets")

        if self.selector_config.get("delete_opp_target_configs", True):
            for target_file in successful_targets:
                os.remove(target_file)
                assert not target_file.exists()

        return successful_targets, failed_targets

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

    def clear_output_plots(self, fig_fmt="png"):
        for obs_name, observatory in self.observatory_manager.sites.items():
            plot_dir = self.path_manager.get_output_plots_path(obs_name, mkdir=False)
            if plot_dir.exists():
                for plot in plot_dir.glob(f"*.{fig_fmt}"):
                    os.remove(plot)

    def perform_web_tasks(self, t_ref: Time = None):
        pass

    #     # TODO: somehow move to messaging_manager
    #     t_ref = t_ref or Time.now()
    #     if self.messaging_manager.html_webpage_manager is not None:
    #         self.messaging_manager.html_webpage_manager.update_webpages(
    #             self.target_lookup, self.outputs_manager.ranked_lists, t_ref=t_ref
    #         )

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
        iteration: int = -1,
        t_ref: Time = None,
    ):
        """
        Perform a single iteration.

        See 'TargetSelector.start() documentation for details of user-provided
        Callable functions.


        Parameters
        ----------
        scoring_function : Callable
        observatory_scoring_function : Callable, optional
        modeling_function : Callable, optional
        lightcurve_compiler : Callable, optional
        lc_plotting_function : Callable, optional
        skip_tasks : list, optional
        iteration : int, default=-1
            passed to the "perform_all_tasks()" function for query_managers.
            useful for skipping certain steps on iteration=0.
        t_ref : `astropy.time.Time`, default=Time.now()
            the reference time at the start of the iteration.
            useful for simulations.


        """

        t_ref = t_ref or Time.now()

        t_str = t_ref.strftime("%Y-%m-%d %H:%M:%S")
        utils.print_header(f"iteration at {t_str}")
        perf_times = {}

        # ================= Get some parameters from config ================= #
        write_comments = self.selector_config.get("write_comments", True)
        obs_info_dt = self.selector_config.get("obs_info_dt", 0.5 / 24.0)
        # lazy_modeling = self.selector_parameters.get("lazy_modeling", True)
        lazy_compile = self.selector_config.get("lazy_compile", False)
        # lazy_plotting = self.selector_parameters.get("lazy_plotting", True)
        seplimit = self.selector_config.get("consolidate_seplimit", 5.0) * u.arcsec
        # self.clear_scratch_plots() # NO - lazy plotting re-uses existing plots.

        # =============== Are there any tasks we should skip? =============== #
        config_skip_tasks = self.selector_config.get("skip_tasks", [])
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
        self.load_targets_of_opportunity(t_ref=t_ref)
        if not "qm_tasks" in skip_tasks:
            self.primary_query_manager.perform_all_query_manager_tasks(
                iteration=iteration, t_ref=t_ref
            )
        else:
            logger.info("skip query manager tasks")
        perf_times["qm_tasks"] = time.perf_counter() - t1

        # ================== Merge + sort duplicated targets ================= #
        self.target_lookup.consolidate_targets(seplimit=seplimit)
        self.target_lookup.update_target_id_mappings()
        self.target_lookup.update_to_preferred_target_id()

        # ================= Prep before modeling and scoring ================= #
        t1 = time.perf_counter()
        if not "ephem_info" in skip_tasks:
            self.observatory_manager.apply_ephem_info(t_ref=t_ref)
        else:
            logger.info("skip compute new obs_info for each target")
        perf_times["ephem_info"] = time.perf_counter() - t1

        t1 = time.perf_counter()
        self.compile_target_lightcurves(
            lightcurve_compiler=lightcurve_compiler, lazy=lazy_compile, t_ref=t_ref
        )
        perf_times["compile"] = time.perf_counter() - t1

        # ============ Remove any targets that aren't interesting ============ #
        t1 = time.perf_counter()
        if not "pre_check" in skip_tasks:
            logger.info(f"{len(self.target_lookup)} targets before check")
            new_targets = self.scoring_manager.new_target_initial_check(
                scoring_function, t_ref=t_ref
            )
            pre_removed_targets = self.target_lookup.remove_rejected_targets(
                target_id_list=new_targets
            )
            if write_comments:
                self.outputs_manager.write_target_comments(
                    target_list=pre_removed_targets, t_ref=t_ref
                )
                self.outputs_manager.write_target_comments(
                    target_list=pre_removed_targets,
                    outdir=self.path_manager.rejected_targets_path,
                    t_ref=t_ref,
                )
            logger.info(f"reject {len(pre_removed_targets)} targets before modeling")
        perf_times["precheck"] = time.perf_counter() - t1

        # =========================== Build models =========================== #
        t1 = time.perf_counter()
        if not "modeling" in skip_tasks:
            self.modeling_manager.build_target_models(modeling_function, t_ref=t_ref)
        perf_times["modeling"] = time.perf_counter() - t1

        # ========================== Do the scoring ========================== #
        t1 = time.perf_counter()
        if "evaluate" not in skip_tasks:
            if scoring_function is None:
                msg = "You must provide a scoring function."
                raise ValueError(msg)
            self.scoring_manager.evaluate_targets(
                scoring_function,
                observatory_scoring_function=observatory_scoring_function,
                t_ref=t_ref,
            )
            logger.info(f"{len(self.target_lookup)} targets after evaluation")
        perf_times["evaluate"] = time.perf_counter() - t1

        # ===================== Remove rejected targets ====================== #
        t1 = time.perf_counter()
        if "reject" not in skip_tasks:
            removed_targets = self.target_lookup.remove_rejected_targets(t_ref=t_ref)
            if write_comments:
                self.outputs_manager.write_target_comments(
                    target_list=removed_targets, t_ref=t_ref
                )
                self.outputs_manager.write_target_comments(
                    target_list=removed_targets,
                    outdir=self.path_manager.rejected_targets_path,
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
            self.plotting_manager.plot_all_target_lightcurves(
                plotting_function=lc_plotting_function, t_ref=t_ref
            )
            self.plotting_manager.plot_all_target_visibilities(t_ref=t_ref)

            # self.plotting_manager.plot_rank_histories(t_ref=t_ref)
            if extra_plotting_functions is not None:
                for plotting_func in extra_plotting_functions:
                    self.plotting_manager.plot_additional_target_figures(
                        plotting_func, t_ref=t_ref
                    )

        else:
            logger.info("skip plotting")
        perf_times["plotting"] = time.perf_counter() - t1

        # ======================== Ranking and lists ======================== #
        t1 = time.perf_counter()
        if "ranking" not in skip_tasks:
            self.clear_output_plots()  # In prep for the new outputs
            self.outputs_manager.build_ranked_target_lists(
                t_ref=t_ref, plots=True, write_list=True
            )
            self.outputs_manager.create_visible_target_lists(
                t_ref=t_ref, plots=True, write_list=True
            )
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
        self.target_lookup.reset_updated_targets()

        logger.info(
            f"time summary:\n    "
            + f"\n    ".join(f"{k:10} = {v: 6.2f}s" for k, v in perf_times.items())
        )  # space in f string formatter is for float padding!

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
                    (eg.) scoring functions with `target.models[<your_func_name>]`

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
            default None (infinite loop).

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
        sleep_time = self.selector_config["sleep_time"]

        if scoring_function is None:
            msg = "You must provide scoring_function=<some-callable>."
            raise ValueError(msg)

        if observatory_scoring_function is None:
            observatory_scoring_function = DefaultObservatoryScoring()
        if lightcurve_compiler is None:
            lightcurve_compiler = DefaultLightcurveCompiler()

        if recovery_file:
            recovered_targets = self.recovery_manager.recover_targets_from_file(
                recovery_file=recovery_file
            )

        # Send some messages on start-up.
        try:
            nodename = os.uname().nodename
        except Exception as e:
            nodename = "<unknown node>"
        t_start_str = t_ref.strftime("%Y-%m-%d %H:%M:%S UTC")
        obs_list = list(self.observatory_manager.sites.keys())
        qm_list = list(self.primary_query_manager.query_managers.keys())
        if not isinstance(modeling_function, list):
            modeling_function = [modeling_function]
        model_func_list = [f.__name__ for f in modeling_function]
        msg = (
            f"starting at {t_start_str} on {nodename} with:\n"
            f"observatories:\n    {', '.join(k for k in obs_list)}\n"
            f"query_managers:\n    {', '.join(qm_list)}\n"
            f"modeling_function:\n    {', '.join(model_func_list)}\n"
            f"scoring_function:\n    {scoring_function.__name__}\n"
            f"observatory_scoring_function:\n    {observatory_scoring_function.__name__}\n"
        )
        logger.info("sending startup message")
        self.messaging_manager.send_sudo_messages(texts=msg)

        # ========================= loop indefinitely ======================== #
        iteration_idx = 0
        while True:
            t_ref = Time.now()

            t_start = time.perf_counter()
            loop_skip_tasks = skip_tasks or []
            if iteration_idx == 0:
                # Don't send messages on the first loop.
                # If many targets are recovered, 100s of messages could be sent...
                loop_skip_tasks.append("messaging")
                loop_skip_tasks.append("write_targets")
                # loop_skip_tasks.append("plotting")

            try:
                self.perform_iteration(
                    scoring_function=scoring_function,
                    observatory_scoring_function=observatory_scoring_function,
                    modeling_function=modeling_function,
                    lightcurve_compiler=lightcurve_compiler,
                    lc_plotting_function=lc_plotting_function,
                    extra_plotting_functions=extra_plotting_functions,
                    skip_tasks=loop_skip_tasks,
                    iteration=iteration_idx,
                    t_ref=t_ref,
                )
            except Exception as e:
                t_str = t_ref.strftime("%Y-%m-%d %H:%M:%S")
                crash_text = [f"CRASH at UT {t_str}"]
                self.messaging_manager.send_crash_reports(text=crash_text)
                sys.exit()

            # Some post-loop tasks
            if iterations is not None:
                if iteration_idx >= iterations:
                    break

            if iteration_idx == 0:
                logger.info("no sleep after startup")
            else:
                exc_time = time.perf_counter() - t_start
                sleep_time_actual = max(sleep_time - exc_time, 60.0)

                logger.info(f"loop execution time = {exc_time:.1f}sec")
                logger.info(f"...so sleep for {sleep_time_actual:.1f}sec")
                time.sleep(sleep_time_actual)

            # ...don't forget to increment the iterations counter!
            iteration_idx = iteration_idx + 1
