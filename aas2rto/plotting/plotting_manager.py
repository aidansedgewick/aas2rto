from logging import getLogger
from typing import Callable

import matplotlib.pyplot as plt
from astropy.time import Time

from aas2rto.path_manager import PathManager
from aas2rto.plotting.default_plotter import plot_default_lightcurve
from aas2rto.plotting.visibility_plotter import plot_visibility
from aas2rto.target_lookup import TargetLookup
from aas2rto import utils


logger = getLogger(__name__.split(".")[-1])


class PlottingManager:

    default_config = {
        "plotting_interval": 0.25,
        "lazy_plotting": True,
        "minimum_rank": 20,
        "rank_lookback": 7.0,
    }

    def __init__(
        self,
        plotting_config: dict,
        target_lookup: TargetLookup,
        path_manager: PathManager,
    ):
        self.config = self.default_config.copy()
        self.config.update(plotting_config)
        utils.check_unexpected_config_keys(
            self.config,
            expected=self.default_config.keys(),
            name="plotting_manager",
        )

        self.target_lookup = target_lookup
        self.path_manager = path_manager

    def plot_all_target_lightcurves(
        self,
        plotting_function: Callable = None,
        lazy_plotting=False,
        plotting_interval=None,
        t_ref: Time = None,
    ):
        plotted = []
        skipped = []
        logger.info(f"plot lightcurves")

        if plotting_function is None:
            plotting_function = plot_default_lightcurve

        interval = plotting_interval or self.config["plotting_interval"]

        for target_id, target in self.target_lookup.items():
            fig_path = self.path_manager.get_lightcurve_plot_path(target_id)
            target.lc_fig_path = fig_path  # useful to attach fig path to target
            fig_age = utils.calc_file_age(fig_path, t_ref, allow_missing=True)
            if lazy_plotting and (not target.updated) and (fig_age < interval):
                skipped.append(target_id)
                msg = f"skip {target_id} lc: age {fig_age:.2f} < {interval:.2f}"
                logger.debug(msg)
                continue
            fig = plotting_function(target, t_ref=t_ref)
            fig.savefig(fig_path)
            plt.close(fig=fig)
            plotted.append(target_id)
        if len(plotted) > 0 or len(skipped) > 0:
            msg = f"plotted {len(plotted)}, re-use {len(skipped)}"
            logger.info(msg)
        return plotted, skipped

    def plot_all_target_visibilities(
        self,
        observatory,
        lazy_plotting=False,
        plotting_interval=0.25,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()

        obs_name = utils.get_observatory_name(observatory)
        if observatory is None:
            return
        logger.info(f"visibility plots for {obs_name}")

        interval = plotting_interval or self.config["plotting_interval"]
        skipped = []
        plotted = []
        for target_id, target in self.target_lookup.items():
            fig_path = self.path_manager.get_visibility_plot_path(target_id, obs_name)
            target.vis_fig_paths[obs_name] = fig_path
            fig_age = utils.calc_file_age(fig_path, t_ref, allow_missing=True)
            if lazy_plotting and (not target.updated) and (fig_age < interval):
                msg = f"skip {target_id} {obs_name} vis: age {fig_age:.2f} < {interval:.2f}"
                logger.debug(msg)
                skipped.append(target_id)
                continue
            obs_info = target.observatory_info.get(obs_name, None)
            fig = plot_visibility(observatory, target, t_ref=t_ref, obs_info=obs_info)
            fig.savefig(fig_path)
            plt.close(fig=fig)
            plotted.append(target_id)
        if len(plotted) > 0 or len(skipped) > 0:
            msg = f"plotted {len(plotted)}, reused {len(skipped)}"
            logger.info(msg)

    def plot_rank_histories(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        fig, ax = plt.subplots()

        minimum_rank = self.config["minimum_rank"]
        rank_lookback = self.config["rank_lookback"]

        for target_id, target in self.target_lookup.items():
            rank_history = target.get_rank_history(obs_name="no_observatory")
            if all(rank_history > minimum_rank):
                continue

            recent_mask = t_ref.mjd - rank_history["mjd"] < rank_lookback
            recent_history = rank_history[recent_mask]

            ax.plot(recent_history["mjd"], recent_history["rank"], label=target_id)

        ax.set_ylim(minimum_rank + 0.5, 0.5)
        ax.set_yaxis("Rank")
        ax.set_xlabel("")
