import shutil
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.time import Time

from astroplan import Observer

from aas2rto import utils
from aas2rto.exc import MissingEphemInfoWarning, UnknownTargetWarning
from aas2rto.observatory_manager import ObservatoryManager
from aas2rto.path_manager import PathManager
from aas2rto.plotting.visibility_plotter import plot_visibility
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup

import matplotlib.pyplot as plt


logger = getLogger(__name__.split(".")[-1])


def get_target_summary_data(target):
    ra = target.coord.ra
    dec = target.coord.dec
    data = dict(
        target_id=target.target_id,
        ra=ra.to_string(unit=u.hourangle, sep=":", precision=2, pad=True),
        dec=dec.to_string(unit=u.deg, sep=":", precision=2, alwayssign=True, pad=True),
        ra_deg=f"{target.coord.ra.deg:010.6f}",
        dec_deg=f"{target.coord.dec.deg:+09.5f}",
    )
    return data


class OutputsManager:

    default_config = {
        "unranked_value": 9999,
        "minimum_score": 0.0,
        "minimum_altitude": 40.0,
        "horizon": -18.0,
    }

    def __init__(
        self,
        outputs_config,
        target_lookup: TargetLookup,
        path_manager: PathManager,
        observatory_manager: ObservatoryManager,
    ):
        self.config = self.default_config.copy()
        self.config.update(outputs_config)
        utils.check_unexpected_config_keys(
            self.config.keys(), self.default_config, name="outputs"
        )
        self._apply_units_to_config()

        self.target_lookup = target_lookup
        self.path_manager = path_manager
        self.observatory_manager = observatory_manager

        self.science_ranked_list = None
        self.obs_ranked_lists = {}
        self.obs_visible_lists = {}

    def _apply_units_to_config(self):
        self.config["minimum_altitude"] = self.config["minimum_altitude"] * u.deg
        self.config["horizon"] = self.config["horizon"] * u.deg

    def write_target_comments(
        self,
        target_list: List[Target] = None,
        outdir: Path = None,
        t_ref: Time = None,
    ):
        t_ref = t_ref or Time.now()

        logger.info("writing target comments")

        outdir = outdir or self.path_manager.comments_path
        outdir = Path(outdir)

        if target_list is None:
            target_list = list(self.target_lookup.values())

        for target in target_list:
            if target is None:
                msg = f"cannot write comments for non-existant target {target_id}"
                logger.warning(msg)
                warnings.warn(UnknownTargetWarning(msg))
                continue
            target.write_comments(outdir, t_ref=t_ref)

    def build_ranked_target_lists(
        self, plots=True, write_list=True, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        logger.info("build ranked target lists")
        ranked_list = self.rank_targets_on_science_score(
            plots=plots, write_list=write_list, t_ref=t_ref
        )
        self.science_ranked_list = ranked_list

        if ranked_list.empty:
            logger.info("Skip attempt obs ranked lists...")
            return

        for obs_name, observatory in self.observatory_manager.sites.items():
            ranked_list = self.rank_targets_on_obs_score(
                observatory, plots=plots, write_list=write_list, t_ref=t_ref
            )
            self.obs_ranked_lists[obs_name] = ranked_list

    def rank_targets_on_science_score(
        self, plots=False, write_list=True, t_ref: Time = None
    ):
        t_ref = t_ref or Time.now()

        plots_path = self.path_manager.get_output_plots_path("science_score")

        minimum_score = self.config["minimum_score"]
        unranked_value = self.config["unranked_value"]

        data_list = []
        for target_id, target in self.target_lookup.items():
            data = get_target_summary_data(target)
            data["score"] = target.get_latest_science_score()
            data_list.append(data)

        if len(data_list) == 0:
            score_df = pd.DataFrame(columns=["score"])
            logger.warning("no targets in initial score df!")
            return score_df

        score_df = pd.DataFrame(data_list)

        score_df.sort_values("score", ascending=False, inplace=True)
        score_df["ranking"] = np.arange(1, len(score_df) + 1)
        # use col name "RANKING" to avoid clash with pd.DF method "df.rank()"

        bad_score = score_df["score"] < minimum_score
        score_df.loc[bad_score, "ranking"] = unranked_value
        for ii, row in score_df.iterrows():
            target_id = row.target_id
            target = self.target_lookup[target_id]
            prev_rank = target.get_latest_science_rank()
            new_rank = row.ranking
            if prev_rank is None or prev_rank != new_rank:
                target.update_science_rank_history(new_rank, t_ref)

            if plots and row.score > minimum_score:
                self.collect_plots(target, plots_path, new_rank)  # no obs_name here

        score_df = score_df[score_df["score"] > minimum_score]
        if write_list:
            ranked_list_path = self.path_manager.get_ranked_list_path("science_score")
            score_df.to_csv(ranked_list_path, index=False)
        return score_df

    def rank_targets_on_obs_score(
        self, observatory: Observer, plots=True, write_list=True, t_ref: Time = None
    ):

        t_ref = t_ref or Time.now()

        obs_name = utils.get_observatory_name(observatory)
        list_name = f"obs_{obs_name}"
        plots_path = self.path_manager.get_output_plots_path(f"obs_{obs_name}")

        minimum_score = self.config["minimum_score"]
        unranked_value = self.config["unranked_value"]

        data_list = []
        for target_id, target in self.target_lookup.items():
            data = get_target_summary_data(target)
            data["score"] = target.get_latest_obs_score(obs_name)
            data_list.append(data)

        if len(data_list) == 0:
            return pd.DataFrame(columns=["target_id", "score", "ranking"])

        score_df = pd.DataFrame(data_list)

        score_df.sort_values("score", ascending=False, inplace=True)
        score_df["ranking"] = np.arange(1, len(score_df) + 1)
        # use col name "RANKING" to avoid clash with pd.DF method "df.rank()"

        bad_score = score_df["score"] < minimum_score
        score_df.loc[bad_score, "ranking"] = unranked_value

        for ii, row in score_df.iterrows():
            target_id = row.target_id
            target = self.target_lookup[target_id]
            prev_rank = target.get_latest_obs_rank(obs_name)
            new_rank = row.ranking
            if prev_rank is None or prev_rank != new_rank:
                target.update_obs_rank_history(new_rank, obs_name, t_ref)

            if plots and row.score > minimum_score:
                self.collect_plots(target, plots_path, new_rank, obs_name=obs_name)

        score_df = score_df[score_df["score"] > minimum_score]
        if write_list:
            ranked_list_path = self.path_manager.get_ranked_list_path(list_name)
            score_df.to_csv(ranked_list_path, index=False)
        return score_df

    def create_visible_target_lists(
        self, plots=True, write_list=True, t_ref: Time = None
    ):
        for obs_name, observatory in self.observatory_manager.sites.items():
            vis_df = self.create_visible_target_list_for_obs(
                observatory, write_list=write_list, plots=plots, t_ref=t_ref
            )
            self.obs_visible_lists[obs_name] = vis_df

    def create_visible_target_list_for_obs(
        self, observatory: Observer, plots=False, write_list=True, t_ref: Time = None
    ):

        t_ref = t_ref or Time.now()
        obs_name = utils.get_observatory_name(observatory)
        logger.info(f"write visible targets list at {obs_name}")

        minimum_score = self.config["minimum_score"]
        minimum_alt = self.config["minimum_altitude"]
        horizon = self.config["horizon"]

        data_list = []
        N_warn = 0
        missing_ephem = 0
        for target_id, target in self.target_lookup.items():
            ephem_info = target.ephem_info.get(obs_name, None)
            if ephem_info is None:
                missing_ephem = missing_ephem + 1
                if N_warn < 3:
                    msg = f"no ephem_info for {target_id} at {obs_name}"
                    logger.warning(msg)
                    warnings.warn(MissingEphemInfoWarning(msg))
                    N_warn = N_warn + 1
                continue

            last_score = target.get_latest_science_score()
            if last_score is None:
                continue
            if last_score < minimum_score:
                continue

            night_mask = ephem_info.sun_altaz.alt < horizon
            target_up_mask = ephem_info.target_altaz.alt > minimum_alt
            visible_mask = night_mask & target_up_mask
            if sum(visible_mask) == 0:
                continue

            data = get_target_summary_data(target)
            transit_time = ephem_info.target_transit
            data["score"] = last_score
            data["transit_mjd"] = transit_time.mjd
            data["transit_time"] = transit_time.strftime("%H:%M")
            data_list.append(data)

        if missing_ephem > 0:
            msg = f"{missing_ephem} targets missing ephem_info at {obs_name}"
            logger.warning(msg)
            warnings.warn(MissingEphemInfoWarning(msg))

        if len(data_list) == 0:
            columns = ["target_id", "score", "transit_mjd", "transit_time"]
            return pd.DataFrame(columns=columns)

        visible_df = pd.DataFrame(data_list)
        visible_df.sort_values("transit_mjd", inplace=True, ignore_index=True)
        if write_list:
            list_path = self.path_manager.get_visible_targets_list_path(obs_name)
            visible_df.to_csv(list_path, index=False)
        return visible_df

    def collect_plots(
        self,
        target: Target,
        output_path: Path,
        ranking: int,
        lc_fig: bool = True,
        obs_name: str = None,
    ):
        target_id = target.target_id

        lc_fig_file = target.lc_fig_path
        if lc_fig_file is not None and lc_fig:
            new_lc_fig_file = output_path / f"{ranking:03d}_{lc_fig_file.name}"

            if not lc_fig_file.exists():
                logger.warning(f"{target_id}: lc figure missing!")
            else:
                try:
                    shutil.copy(lc_fig_file, new_lc_fig_file)
                except Exception:
                    logger.error(f"failed in copy {lc_fig_file.name}")

        if obs_name is not None:
            vis_fig_file = target.vis_fig_paths.get(obs_name, None)

            if vis_fig_file is not None:
                new_vis_fig_file = output_path / f"{ranking:03d}_{vis_fig_file.name}"

                if not vis_fig_file.exists():
                    logger.warning(f"{vis_fig_file.name} missing!")
                else:
                    try:
                        shutil.copy(vis_fig_file, new_vis_fig_file)
                    except Exception:
                        logger.error(f"failed in copy {vis_fig_file.name}")
