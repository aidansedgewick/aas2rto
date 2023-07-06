import logging
import os
import shutil
import sys
import time
import warnings
import yaml
from pathlib import Path
from typing import Callable, Dict, List, Set

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from astroplan import Observer
from astroplan.plots import plot_altitude

from dk154_targets import Target
from dk154_targets import query_managers
from dk154_targets import messengers
from dk154_targets.lightcurve_compilers import default_compile_lightcurve
from dk154_targets.obs_info import ObservatoryInfo
from dk154_targets.utils import print_header

from dk154_targets import paths

logger = logging.getLogger(__name__.split(".")[-1])


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

    default_sleep_time = 600.0
    default_unranked_value = 9999
    minimum_score = 0.0

    def __init__(self, selector_config: dict):
        # Unpack configs.
        self.selector_config = selector_config
        self.selector_parameters = self.selector_config.get("selector_parameters", {})
        self.query_manager_config = self.selector_config.get("query_managers", {})
        self.observatory_config = self.selector_config.get("observatories", {})
        self.messenger_config = self.selector_config.get("messengers", {})
        self.modelling_config = self.selector_config.get("modelling", {})
        self.paths_config = self.selector_config.get("paths", {})

        # to keep the targets. Do this here as initQM needs to know about it.
        self.target_lookup = self._create_empty_target_lookup()

        # Prepare paths
        self.process_paths()
        self.make_directories()

        # Initialise some things.
        self.initialize_query_managers()
        self.initialize_observatories()
        self.initialize_messengers()

    def __setitem__(self, objectId, target):
        self.target_lookup[objectId] = target

    def __getitem__(self, objectId):
        return self.target_lookup[objectId]

    def __iter__(self):
        for target, objectId in self.target_lookup.items():
            yield target, objectId

    def __contains__(self, member):
        return member in self.target_lookup

    @classmethod
    def from_config(cls, config_path):
        config_path = Path(config_path)

        with open(config_path, "r") as f:
            selector_config = yaml.load(f, Loader=yaml.FullLoader)
            selector = cls(selector_config)
            return selector

    def _create_empty_target_lookup(self) -> Dict[str, Target]:
        """Returns an empty dictionary. Only for type hinting."""
        return dict()

    def _create_empty_target_set(self) -> Set[str]:
        """Returns an empty set. Only for type hinting."""
        return set()

    def process_paths(self):
        self.base_path = paths.wkdir
        project_path = self.paths_config.pop("project_path", "default")
        if project_path == "default":
            project_path = self.base_path / "projects/default"
        self.project_path = Path(project_path)
        self.paths = {"base_path": self.base_path, "project_path": self.project_path}
        for location, path in self.paths_config.items():
            parts = path.split("/")
            if parts[0].startswith("$"):
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
        self.lc_scratch_path.mkdir(exist_ok=True, parents=True)
        self.oc_scratch_path = self.scratch_path / "oc"
        self.oc_scratch_path.mkdir(exist_ok=True, parents=True)
        if "existing_targets_file" not in self.paths:
            self.paths["existing_targets_file"] = (
                self.data_path / "existing_targets.csv"
            )
        self.existing_targets_file = self.paths["existing_targets_file"]

    def make_directories(self):
        for location, path in self.paths.items():
            if location.endswith("file"):
                continue
            path.mkdir(exist_ok=True, parents=True)

    def initialize_query_managers(self):
        self.query_managers = {}
        self.qm_order = []
        for qm_name, qm_config in self.query_manager_config.items():
            if not qm_config.get("use", True):
                logger.info(f"Skip {qm_name} init")
                continue
            self.qm_order.append(qm_name)  # In case the config order is very important.

            qm_args = (qm_config, self.target_lookup)
            qm_kwargs = dict(data_path=self.data_path)
            if qm_name == "alerce":
                qm = query_managers.AlerceQueryManager(*qm_args, **qm_kwargs)
            elif qm_name == "atlas":
                qm = query_managers.AtlasQueryManager(*qm_args, **qm_kwargs)
            elif qm_name == "fink":
                qm = query_managers.FinkQueryManager(*qm_args, **qm_kwargs)
            elif qm_name == "lasair":
                qm = query_managers.LasairQueryManager(*qm_args, **qm_kwargs)
            elif qm_name == "tns":
                raise NotImplementedError(f"{qm_name.capitalize()}QueryManager")
            elif qm_name == "yse":
                qm = query_managers.YseQueryManager(*qm_args, **qm_kwargs)
            else:
                # Mainly for testing.
                logger.warning(f"no known query manager for {qm_name}")
                # query_managers.UsingGenericWarning,
                qm = query_managers.GenericQueryManager(*qm_args, **qm_kwargs)
            self.query_managers[qm_name] = qm

        if len(self.query_managers) == 0:
            logger.warning("no query managers initialised!")

    def initialize_observatories(self) -> Dict[str, Observer]:
        logger.info(f"init observatories")
        self.observatories = {"no_observatory": None}
        for obs_name, location_identifier in self.observatory_config.items():
            if isinstance(location_identifier, str):
                earth_loc = EarthLocation.of_site(location_identifier)
            else:
                earth_loc = EarthLocation(**location_identifier)
            observatory = Observer(location=earth_loc, name=obs_name)
            self.observatories[obs_name] = observatory
        logger.info(f"init {len(self.observatories)} including `no_observatory`")

    def initialize_messengers(self):
        # TODO: add telegram, slack, etc.
        self.messengers = {}
        self.telegram_messenger = None
        self.slack_messenger = None
        for m_name, msgr_config in self.messenger_config.items():
            if not msgr_config.get("use", True):
                logger.info(f"Skip messenger {m_name} init")
                continue
            if m_name == "telegram":
                msgr = messengers.TelegramMessenger(msgr_config)
                self.telegram_messenger = msgr
            elif m_name == "slack":
                msgr = messengers.SlackMessenger(msgr_config)
                self.slack_messenger = msgr
            else:
                raise NotImplementedError(f"No messenger {m_name}")
            self.messengers[m_name] = msgr
        telegram_config = self.messenger_config.get("telegram", None)

    def add_target(self, target: Target):
        if target.objectId in self.target_lookup:
            raise ValueError(f"obj {target.objectId} already in target_lookup")
        self.target_lookup[target.objectId] = target

    def compute_observatory_info(
        self, t_ref: Time = None, horizon: u.Quantity = -18 * u.deg
    ):
        """
        save the result of astrolan.Observer() functions.

        access in a target with eg. `my_target.observatory_nights["lasilla"]`
        """
        t_ref = t_ref or Time.now()

        for obs_name, observatory in self.observatories.items():
            if observatory is None:
                continue

            obs_info = ObservatoryInfo.from_observatory(
                observatory, t_ref=t_ref, horizon=horizon
            )

            for objectId, target in self.target_lookup.items():
                target_obs_info = obs_info.copy()
                assert target_obs_info.target_altaz is None
                if target.coord is not None:
                    target_altaz = observatory.altaz(obs_info.t_grid, target.coord)
                    target_obs_info.target_altaz = target_altaz
                target.observatory_info[obs_name] = target_obs_info
                if target.observatory_info[obs_name].target_altaz is None:
                    msg = f"\033[33m{objectId} {obs_name} altaz missing\033[0m"
                    logger.warning(msg)

    def check_for_targets_of_opportunity(self):
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
                existing_target.base_score = base_score
                existing_target.target_of_opportunity = True
                existing_targets.append(opp_target_file)
            else:
                opp_target = Target(objectId, ra=ra, dec=dec, base_score=base_score)
                opp_target.target_of_opportunity = True
                self.add_target(opp_target)
            successful_targets.append(opp_target_file)
        if len(failed_targets) > 0:
            failed_list = [f.name for f in failed_targets]
            logger.info(f"{len(failed_targets)} failed targets:\n    {failed_list}")
        msg = f"{len(successful_targets)} new opp targets"
        if len(existing_targets) > 0:
            msg = msg + f" ({len(existing_targets)} already exist)"
        for target_file in successful_targets:
            os.remove(target_file)
            assert not target_file.exists()

        return successful_targets, existing_targets, failed_targets

    def compile_target_lightcurves(self, t_ref=None, compile_function: Callable = None):
        t_ref = t_ref or Time.now()

        if compile_function is None:
            compile_function = default_compile_lightcurve
        logger.info("compile photometric data")

        not_compiled = []
        for objectId, target in self.target_lookup.items():
            target.build_compiled_lightcurve(
                compile_function=compile_function, t_ref=t_ref
            )
            if target.compiled_lightcurve is None:
                not_compiled.append(objectId)
        if len(not_compiled) > 0:
            logger.info(f"{len(not_compiled)} have no compiled lightcurve")

    def perform_query_manager_tasks(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        logger.info("begin query manager tasks")
        for qm_name, qm in self.query_managers.items():
            t_start = time.perf_counter()
            logger.info(f"begin {qm_name} tasks")
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
        logger.info(f"evaluate {len(self.target_lookup)} targets")
        logger.info(f"evaluate with {scoring_function.__name__}")
        for obs_name, observatory in self.observatories.items():
            logger.info(f"evaluate for {obs_name}")
            self.evaluate_all_targets_at_observatory(
                scoring_function, observatory, t_ref=t_ref
            )
            # for objectId, target in self.target_lookup.items():
            #    target.evaluate_target(scoring_function, observatory, t_ref=t_ref)

    def evaluate_all_targets_at_observatory(
        self, scoring_function: Callable, observatory: Observer, t_ref: Time = None
    ):
        t_ref = t_ref or Time()
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

    def remove_bad_targets(self, t_ref=None):
        t_ref = t_ref or Time.now()

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
            # self.rejected_targets.add(objectId)
            removed_targets.append(target)
            assert objectId not in self.target_lookup
        return removed_targets

    def reset_updated_target_flags(self):
        for objectId, target in self.target_lookup.items():
            target.updated = False

    def build_target_models(
        self, modeling_functions: Callable, t_ref: Time = None, lazy=True
    ):
        t_ref = t_ref or Time.now()

        logger.info(f"build models for {len(self.target_lookup)} targets")

        if not isinstance(modeling_functions, list):
            modeling_functions = [modeling_functions]
        failed_models = {func.__name__: 0 for func in modeling_functions}
        for objectId, target in self.target_lookup.items():
            for func in modeling_functions:
                target.build_model(func, t_ref=t_ref, lazy=lazy)
                if target.models.get(func.__name__, None) is None:
                    failed_models[func.__name__] = failed_models[func.__name__] + 1
        for func_name, N_failed in failed_models.items():
            if N_failed > 0:
                logger.warning(f"{func_name} failed for {N_failed} targets")

    def build_lightcurve_plots(
        self, lc_plotting_function: Callable = None, t_ref: Time = None
    ):
        logger.info(f"build {len(self.target_lookup)} lightcurve plots")
        for objectId, target in self.target_lookup.items():
            figpath = self.lc_scratch_path / f"{objectId}_lc.png"
            fig = target.plot_lightcurve(
                lc_plotting_function=lc_plotting_function, t_ref=t_ref, figpath=figpath
            )
            plt.close(fig=fig)

    def build_observing_charts(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        for obs_name, observatory in self.observatories.items():
            if observatory is None:
                continue
            logger.info(f"build observing charts for {obs_name}")
            for objectId, target in self.target_lookup.items():
                figpath = self.oc_scratch_path / f"{objectId}_{obs_name}_oc.png"
                fig = target.plot_observing_chart(
                    observatory, t_ref=t_ref, figpath=figpath
                )
                plt.close(fig=fig)

    def get_output_plots_dir(self, obs_name, mkdir=True) -> Path:
        plots_dir = self.outputs_path / f"plots/{obs_name}"
        if mkdir:
            plots_dir.mkdir(exist_ok=True, parents=True)
        return plots_dir

    def get_output_lists_dir(self, mkdir=True) -> Path:
        ranked_list_dir = self.outputs_path / "ranked_lists"
        if mkdir:
            ranked_list_dir.mkdir(exist_ok=True, parents=True)
        return ranked_list_dir

    def clear_scratch_plots(self):
        for plot in self.lc_scratch_path.glob("*.png"):
            os.remove(plot)
        for plot in self.lc_scratch_path.glob("*.png"):
            os.remove(plot)

    def clear_output_directories(self):
        ranked_list_dir = self.get_output_lists_dir(mkdir=True)
        if ranked_list_dir.exists():
            for listfile in ranked_list_dir.glob("*.png"):
                os.remove(listfile)
        for obs_name, observatory in self.observatories.items():
            plot_dir = self.get_output_plots_dir(obs_name, mkdir=False)
            if plot_dir.exists():
                for plot in plot_dir.glob("*.png"):
                    os.remove(plot)

    def build_all_ranked_target_lists(
        self, plots=True, write_list=True, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()
        for obs_name, observatory in self.observatories.items():
            self.build_ranked_target_list(
                observatory, plots=plots, write_list=write_list, t_ref=t_ref
            )

    def build_ranked_target_list(
        self, observatory: Observer, plots=True, write_list=True, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        obs_name = getattr(observatory, "name", "no_observatory")
        if obs_name == "no_observatory":
            assert observatory is None

        logger.info(f"ranked lists for {obs_name}")

        data_list = []
        for objectId, target in self.target_lookup.items():
            last_score = target.get_last_score(obs_name)
            data_list.append(
                dict(objectId=objectId, score=last_score, ra=target.ra, dec=target.dec)
            )

        if len(data_list) == 0:
            logger.warning("no targets in target lookup!")
            return
        score_df = pd.DataFrame(data_list)
        score_df.sort_values("score", inplace=True, ascending=False)
        score_df.set_index("objectId", inplace=True)
        score_df["ranking"] = np.arange(1, len(score_df) + 1)
        negative_score = score_df["score"] < self.minimum_score
        score_df.loc[negative_score, "ranking"] = self.default_unranked_value
        for objectId, row in score_df.iterrows():
            target = self.target_lookup[objectId]
            if obs_name not in target.rank_history:
                target.rank_history[obs_name] = []
            target.rank_history[obs_name].append((row.ranking, t_ref))

        score_df.query(f"score>{self.minimum_score}", inplace=True)
        if write_list:
            ranked_list_dir = self.get_output_lists_dir()
            ranked_list_file = ranked_list_dir / f"{obs_name}.csv"
            score_df.to_csv(ranked_list_file, index=True)

        if plots:
            for objectId, row in score_df.iterrows():
                target = self.target_lookup[objectId]
                self.collect_plots(target, obs_name, row.ranking)

    def collect_plots(self, target: Target, obs_name: str, rank: int):
        plots_dir = self.get_output_plots_dir(obs_name)

        lcfig_file = target.latest_lc_fig_path
        if lcfig_file is not None:
            new_lcfig_filename = f"{int(rank):03d}_{target.objectId}_lc.png"
            new_lcfig_file = plots_dir / new_lcfig_filename
            if lcfig_file.exists():
                shutil.copy2(lcfig_file, new_lcfig_file)
        ocfig_file = target.latest_oc_fig_paths.get(obs_name, None)
        if ocfig_file is not None:
            new_ocfig_filename = f"{int(rank):03d}_{target.objectId}_oc.png"
            new_ocfig_file = plots_dir / new_ocfig_filename
            if ocfig_file.exists():
                shutil.copy2(ocfig_file, new_ocfig_file)

    def perform_messaging_tasks(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        for objectId, target in self.target_lookup.items():
            if len(target.update_messages) == 0:
                continue
            intro = f"Updates for {objectId} at {t_ref.isot}"
            messages = [intro] + target.update_messages
            img_paths = [target.latest_lc_fig_path] + list(
                target.latest_oc_fig_paths.values()
            )
            if self.telegram_messenger is not None:
                self.telegram_messenger.message_users(texts=messages)
                self.telegram_messenger.message_users(
                    img_paths=target.latest_lc_fig_path, caption="lightcurve"
                )
                self.telegram_messenger.message_users(
                    img_paths=target.latest_oc_fig_paths.values(),
                    caption="observing charts",
                )
            if self.slack_messenger is not None:
                self.slack_messenger.send_messages(
                    texts=messages, img_paths=target.latest_lc_fig_path
                )

    def reset_target_figures(self):
        for objectId, target in self.target_lookup.items():
            target.reset_figures()

    def reset_updated_targets(self):
        for objectId, target in self.target_lookup.items():
            target.updated = False
            target.send_updates = False
            target.update_messages = []

    def write_existing_target_list(self):
        rows = []
        if len(self.target_lookup) == 0:
            logger.info("no existing targets to write...")
        for objectId, target in self.target_lookup.items():
            data = dict(
                objectId=objectId,
                ra=target.ra,
                dec=target.dec,
                base_score=target.base_score,
            )
            rows.append(data)
        existing_targets_df = pd.DataFrame(rows)
        existing_targets_df.to_csv(self.existing_targets_file, index=False)

    def recover_existing_targets(self):
        if not self.existing_targets_file.exists():
            logger.info("no existing targets to recover...")
            return
        if self.existing_targets_file.stat().st_size < 2:
            logger.info("file too small - don't attempt read...")
            return
        existing_targets_df = pd.read_csv(self.existing_targets_file)
        recovered_targets = 0
        for ii, row in existing_targets_df.iterrows():
            objectId = row.objectId
            if objectId in self.target_lookup:
                logger.warning(f"skip load existing {objectId}")
                continue
            target = Target(objectId, ra=row.ra, dec=row.dec, base_score=row.base_score)
            self.add_target(target)
            recovered_targets = recovered_targets + 1
        logger.info(f"recovered {recovered_targets} existing targets")
        return

    def perform_iteration(
        self,
        scoring_function: Callable = None,
        modeling_function: Callable = None,
        lightcurve_compile_function: Callable = None,
        lc_plotting_function: Callable = None,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()

        t_str = t_ref.strftime("%Y-%m-%d %H:%M:%S")
        print_header(f"iteration at {t_str}")

        self.clear_scratch_plots()

        # Get new data
        self.check_for_targets_of_opportunity()
        self.perform_query_manager_tasks(t_ref=t_ref)

        # Set some things before modelling and scoring.
        self.compute_observatory_info(t_ref=t_ref)
        self.compile_target_lightcurves(
            t_ref=t_ref, compile_function=lightcurve_compile_function
        )

        # Remove any targets that aren't interesting.
        logger.info(f"{len(self.target_lookup)} targets before check")
        self.new_target_initial_check(scoring_function, t_ref=t_ref)
        removed_before_modeling = self.remove_bad_targets()
        logger.info(f"removed {len(removed_before_modeling)} before modelling")

        # Build models
        lazy_modeling = self.selector_parameters.get("lazy_modeling", True)
        self.build_target_models(modeling_function, t_ref=t_ref, lazy=lazy_modeling)

        # Do the scoring, remove the bad targets
        self.evaluate_all_targets(scoring_function)
        logger.info(f"{len(self.target_lookup)} targets before removing bad")
        removed_targets = self.remove_bad_targets()
        logger.info(
            f"removed {len(removed_targets)} targets, {len(self.target_lookup)} remain"
        )

        # Plotting
        self.build_lightcurve_plots(
            lc_plotting_function=lc_plotting_function, t_ref=t_ref
        )
        self.build_observing_charts(t_ref=t_ref)

        # Ranking
        self.clear_output_directories()  # In prep for the new outputs
        self.build_all_ranked_target_lists(t_ref=t_ref, plots=True, write_list=True)

    def start(
        self,
        scoring_function: Callable = None,
        modeling_function: List[Callable] = None,
        lightcurve_compile_function: Callable = None,
        lc_plotting_function: Callable = None,
        existing_targets=True,
        iterations=None,
    ):
        N_iterations = 0
        sleep_time = self.selector_parameters.get("sleep_time", self.default_sleep_time)

        if existing_targets is True or existing_targets == "read":
            self.recover_existing_targets()
        if existing_targets == "clear":
            logger.warning("clear existing targets...")
            if self.existing_targets_file.exists():
                os.remove(self.existing_targets_file)

        while True:
            t_ref = Time.now()
            # try:
            self.perform_iteration(
                scoring_function=scoring_function,
                modeling_function=modeling_function,
                lightcurve_compile_function=lightcurve_compile_function,
                lc_plotting_function=lc_plotting_function,
                t_ref=t_ref,
            )
            # except Exception as e:
            #     tr = traceback.format_exc()
            #     print(tr)
            #     if self.telegram_messenger is not None:
            #         self.telegram_messenger.send_crash_reports(tr)
            #     sys.exit()

            # if iterations > 1:
            self.perform_messaging_tasks()

            self.reset_target_figures()
            self.reset_updated_targets()

            if existing_targets:
                self.write_existing_target_list()

            N_iterations = N_iterations + 1
            if iterations is not None:
                if N_iterations >= iterations:
                    break

            logger.info(f"sleep for {sleep_time} sec")
            time.sleep(sleep_time)
