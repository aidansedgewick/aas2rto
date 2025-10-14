import warnings
from logging import getLogger
from typing import Callable

import matplotlib.pyplot as plt

from astropy.time import Time

from astroplan import Observer

from aas2rto.observatory_manager import ObservatoryManager
from aas2rto.path_manager import PathManager
from aas2rto.plotting.default_plotter import plot_default_lightcurve
from aas2rto.plotting.rank_hist_plotter import plot_rank_histories
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
        observatory_manager: ObservatoryManager,
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
        self.observatory_manager = observatory_manager

    def plot_all_target_lightcurves(
        self,
        plotting_function: Callable = None,
        lazy_plotting: bool = None,
        plotting_interval: float = None,
        t_ref: Time = None,
    ):
        logger.info(f"plot lightcurves")

        t_ref = t_ref or Time.now()

        if plotting_function is None:
            plotting_function = plot_default_lightcurve

        interval = plotting_interval or self.config["plotting_interval"]
        lazy_plotting = lazy_plotting or self.config["lazy_plotting"]

        if lazy_plotting:
            logger.info(f"reuse lc plots <{interval*24:.1f}hr")

        plotted = []
        skipped = []
        for target_id, target in self.target_lookup.items():
            fig_path = self.path_manager.get_lightcurve_plot_path(target_id)
            target.lc_fig_path = fig_path  # useful to attach fig path to target
            fig_age = utils.calc_file_age(fig_path, t_ref, allow_missing=True)
            if lazy_plotting and (fig_age < interval) and not target.updated:
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
        lazy_plotting: bool = None,
        plotting_interval: float = None,
        t_ref: Time = None,
    ):
        for obs_name, observatory in self.observatory_manager.sites.items():
            self.plot_all_target_visibilities_for_observatory(
                observatory,
                lazy_plotting=lazy_plotting,
                plotting_interval=plotting_interval,
                t_ref=t_ref,
            )

    def plot_all_target_visibilities_for_observatory(
        self,
        observatory: Observer,
        lazy_plotting=None,
        plotting_interval=None,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()

        obs_name = utils.get_observatory_name(observatory)
        if observatory is None:
            return
        logger.info(f"visibility plots for {obs_name}")

        interval = plotting_interval or self.config["plotting_interval"]
        lazy_plotting = lazy_plotting or self.config["lazy_plotting"]

        if lazy_plotting:
            logger.info(f"reuse {obs_name} vis plots <{interval*24:.1f}hr")

        interval = plotting_interval or self.config["plotting_interval"]
        skipped = []
        plotted = []
        for target_id, target in self.target_lookup.items():
            fig_path = self.path_manager.get_visibility_plot_path(target_id, obs_name)
            target.vis_fig_paths[obs_name] = fig_path
            fig_age = utils.calc_file_age(fig_path, t_ref, allow_missing=True)
            if lazy_plotting and (fig_age < interval):
                msg = f"skip {target_id} {obs_name} vis: age {fig_age:.2f} < {interval:.2f}"
                logger.debug(msg)
                skipped.append(target_id)
                continue

            ephem_info = target.ephem_info.get(obs_name, None)
            fig = plot_visibility(
                observatory, target, t_ref=t_ref, ephem_info=ephem_info
            )
            fig.savefig(fig_path)
            plt.close(fig=fig)
            plotted.append(target_id)
        if len(plotted) > 0 or len(skipped) > 0:
            msg = f"plotted {len(plotted)}, reused {len(skipped)}"
            logger.info(msg)
        return plotted, skipped

    def plot_rank_histories(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        fig, ax = plt.subplots()

        plotter = plot_rank_histories(
            self.target_lookup, "no_observatory", t_ref=t_ref, return_plotter=True
        )

        fig = plotter.fig

        fig_path = self.path_manager.scratch_path / "rank_histories.png"
        fig.savefig(fig_path)
        plt.close(fig)

        return plotter.targets_plotted, plotter.targets_skipped

    def plot_additional_target_figures(self, func: Callable, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        try:
            plot_name = func.__name__
        except AttributeError as e:
            plot_name = type(func).__name__

        lazy_plotting = getattr(func, "lazy_plotting", None)
        if lazy_plotting is None:
            lazy_plotting = self.config["lazy_plotting"]
            # TODO or not: warning here?

        interval = getattr(func, "plotting_interval", None)
        if interval is None:
            interval = self.config["plotting_interval"]
            # TODO or not: raise warining here?

        output_dir = self.path_manager.scratch_path / "extra_plots" / plot_name
        output_dir.mkdir(exist_ok=True, parents=True)

        plotted = []
        skipped = []
        returned_none = []
        failed = []
        for target_id, target in self.target_lookup.items():
            fig_path = output_dir / f"{target_id}.png"
            fig_age = utils.calc_file_age(fig_path, t_ref, allow_missing=True)
            if lazy_plotting and (fig_age < interval) and not target.updated:
                skipped.append(target_id)
                msg = (
                    f"skip {target_id} {plot_name}: age {fig_age:.2f} < {interval:.2f}"
                )
                logger.debug(msg)
                continue

            try:
                fig = func(target, t_ref=t_ref)
            except Exception as e:
                if len(failed) < 3:
                    msg = f"{target_id}: {type(e)}, {e}"
                    logger.error(f"{target_id}: {type(e)}, {e}")
                    warnings.warn(UserWarning(msg))
                failed.append(target_id)
                continue
            if fig is None:
                returned_none.append(target_id)
                continue

            plotted.append(target_id)
            target.additional_fig_paths[plot_name] = fig_path

            fig.savefig(fig_path)
            plt.close(fig)

        if len(failed) > 0:
            logger.error(f"{len(failed)} failed to plot!")

        logger.info(
            f"{len(plotted)} plotted, {len(skipped)} skipped, {len(returned_none)} returned 'None'"
        )
        return plotted, skipped
