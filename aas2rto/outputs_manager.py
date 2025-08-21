import shutil
from logging import getLogger
from pathlib import Path
from typing import List

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.time import Time

from astroplan import Observer

from aas2rto.path_manager import PathManager
from aas2rto.plotting.visibility_plotter import plot_visibility
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup
from aas2rto import utils
import matplotlib.pyplot as plt


logger = getLogger(__name__.split(".")[-1])


class OutputsManager:

    default_config = {
        "unranked_value": 9999,
        "minimum_score": 0.0,
        "minimum_altitude": 40.0,
        "rank_history_lookback": 7.0,
    }

    def __init__(
        self,
        outputs_config,
        target_lookup: TargetLookup,
        path_manager: PathManager,
    ):
        self.config = self.default_config.copy()
        self.config.update(outputs_config)
        utils.check_unexpected_config_keys(
            self.config.keys(), self.default_config, name="outputs"
        )

        self.target_lookup = target_lookup
        self.path_manager = path_manager

    def write_target_comments(
        self, target_list: List[Target] = None, outdir: Path = None, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        logger.info("writing target comments")

        outdir = outdir or self.path_manager.comments_path
        outdir = Path(outdir)

        if target_list is None:
            target_list = [t for o, t in self.target_lookup.items()]

        for target_id, target in self.target_lookup.items():
            target.write_comments(outdir, t_ref=t_ref)

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
        for target_id, target in self.target_lookup.items():
            data = target.get_target_summary_data(obs_name)
            data_list.append(data)

        if len(data_list) == 0:
            logger.warning("no targets in target lookup!")
            return

        score_df = pd.DataFrame(data_list)
        score_df.sort_values("score", inplace=True, ascending=False, ignore_index=True)
        score_df["ranking"] = np.arange(1, len(score_df) + 1)
        # use column name "ranking", not "rank", as rank is a df/series method.
        # column/row lookup with dot syntax `score_df.rank` / `row.rank`` fails!

        minimum_score = self.config["minimum_score"]
        unranked_value = self.config["unranked_value"]
        negative_score = score_df["score"] < minimum_score  # bool mask.
        score_df.loc[negative_score, "ranking"] = unranked_value

        for ii, row in score_df.iterrows():
            target_id = row.target_id
            target = self.target_lookup[target_id]
            last_rank = target.get_last_rank(obs_name)
            new_rank = row["ranking"]
            if (last_rank is None) or (last_rank != new_rank):
                target.update_rank_history(row["ranking"], obs_name, t_ref=t_ref)

        score_df.query(f"score>{minimum_score}", inplace=True)
        if write_list:
            ranked_list_path = self.path_manager.get_ranked_list_path(obs_name)
            score_df.to_csv(ranked_list_path, index=False)

        if plots:
            for ii, row in score_df.iterrows():
                target_id = row.target_id
                target = self.target_lookup[target_id]
                self.collect_plots(target, obs_name, row["ranking"])
        logger.info(f"{sum(negative_score)} targets excluded, {len(score_df)} ranked")
        return score_df

    def build_visible_target_lists_at_observatory(
        self, observatory: Observer, plots=False, write_list=True, t_ref: Time = None
    ):

        minimum_alt = self.config["minimum_altitude"]
        minimum_score = self.config["minimum_score"]

        obs_name = observatory.name
        sunset = observatory.sun_set_time(t_ref, which="next", horizon=-18.0 * u.deg)
        sunrise = observatory.sun_rise_time(
            sunset, which="nearest", horizon=-18.0 * u.deg
        )

        visible_targets = []
        data_list = []

        for target_id, target in self.target_lookup.items():
            obs_info = target.observatory_info[obs_name]
            data = target.get_target_summary_data(obs_name)
            if data["score"] is None or data["score"] < minimum_score:
                continue

            alt_grid = observatory.al

        if len(data_list) == 0:
            return

        vis_targets_list = pd.DataFrame(data_list)
        vis_targets_list.sort_values("score", inplace=True)
        if write_list:
            vis_targets_list_path = self.path_manager.get_visible_targets_list_path(
                obs_name
            )
            vis_targets_list.to_csv(vis_targets_list_path, index=False)

        if plots:
            fig = plot_visibility(observatory, *visible_targets, sky_ax=False)
            fig_path = self.path_manager.outputs_path / f"{obs_name}_visible.png"
            fig.savefig(fig_path)
            plt.close(fig=fig)

    # def plot_rank_history(self):

    #    fig, ax = plt.su

    #    for target_id, target in self.target_lookup.items():

    def collect_plots(self, target: Target, obs_name: str, ranking: int, fmt="png"):
        """
        It's much cheaper to create all the plots once,
        and then just copy them to a new directory.
        """

        plots_path = self.path_manager.get_output_plots_path(obs_name)
        target_id = target.target_id

        lc_fig_file = target.lc_fig_path
        if lc_fig_file is not None:
            new_lc_fig_stem = f"{int(ranking):03d}_{target_id}_lc"
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

        vis_fig_file = target.vis_fig_paths.get(obs_name, None)
        # Don't need obs_name in new stem - separate dir for each!
        new_vis_fig_stem = f"{int(ranking):03d}_{target_id}_vis"
        new_vis_fig_file = plots_path / f"{new_vis_fig_stem}.{fmt}"
        if vis_fig_file is not None and vis_fig_file.exists():
            try:
                shutil.copy2(vis_fig_file, new_vis_fig_file)
            except FileNotFoundError as e:
                msg = (
                    f"\033[33mvisibility fig {vis_fig_file} missing!\033[0m"
                    + f"\n    the likely cause is you have two projects with the"
                    + f"same project_path, and one has cleared plots"
                )
                logger.error(msg)
