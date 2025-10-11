import json
import os
import time
from logging import getLogger
from pathlib import Path

import numpy as np

import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.time import Time

from aas2rto import utils
from aas2rto.path_manager import PathManager
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup

logger = getLogger(__name__.split(".")[-1])


class RecoveryManager:

    default_config = {
        "retained_recovery_files": 5,
        "load_rank_history": True,
    }

    def __init__(
        self,
        recovery_config: dict,
        target_lookup: TargetLookup,
        path_manager: PathManager,
    ):
        self.config = self.default_config.copy()
        self.config.update(recovery_config)
        utils.check_unexpected_config_keys(
            self.config, self.default_config, name="recovery"
        )

        self.target_lookup = target_lookup
        self.path_manager = path_manager

    def write_recovery_file(self, t_ref: Time = None, fmt="json"):
        t_ref = t_ref or Time.now()

        if len(self.target_lookup) == 0:
            logger.info("no existing targets to write...")
            return

        data = {}
        for target_id, target in self.target_lookup.items():
            target_info = dict(
                target_id=target_id,
                ra=target.coord.ra.deg,
                dec=target.coord.dec.deg,
                base_score=target.base_score,
                alt_ids=target.alt_ids,
            )
            data[target_id] = target_info

        recovery_file = self.path_manager.get_current_recovery_file(t_ref=t_ref)
        with open(recovery_file, "w+") as f:
            json.dump(data, f, indent=2)
        try:
            print_path = recovery_file.relative_to(
                self.path_manager.project_path.parent
            )
        except Exception as e:
            print_path = recovery_file
        logger.info(f"write existing targets to:\n    {print_path}")

        existing_recovery_files = self.path_manager.get_existing_recovery_files(fmt=fmt)
        N = self.config["retained_recovery_files"]
        targets_files_to_remove = existing_recovery_files[:-N]
        for filepath in targets_files_to_remove:
            os.remove(filepath)

    def write_rank_histories(self, obs_name="no_observatory", t_ref: Time = None):
        t_ref = t_ref or Time.now()

        data = {}
        for target_id, target in self.target_lookup.items():
            data[target_id] = target.rank_history[obs_name]

        rank_history_file = self.path_manager.get_current_rank_history_file(t_ref=t_ref)
        with open(rank_history_file, "w+") as f:
            json.dump(data, f)

        existing_rank_history_files = (
            self.path_manager.get_existing_rank_history_files()
        )

        N = self.config["retained_recovery_files"]
        files_to_remove = existing_rank_history_files[:-N]
        for filepath in files_to_remove:
            os.remove(filepath)

    def recover_targets_from_file(self, recovery_file="last", fmt="json"):
        """
        During each iteration of the selector, the names of all of the existing targets
        are dumped into a file.

        These can be reloaded in the event of the program stopping.
        """

        if recovery_file == "last":
            existing_recovery_files = self.path_manager.get_existing_recovery_files(
                fmt=fmt
            )
            if len(existing_recovery_files) == 0:
                return
            recovery_file = existing_recovery_files[-1]
        recovery_file = Path(recovery_file)

        if not recovery_file.exists():
            logger.warning(f"{recovery_file.file} missing")
            return
        if recovery_file.stat().st_size < 2:
            logger.info("file too small - don't attempt read...")
            return

        logger.info(f"recover targets from\n    {recovery_file}")
        t_start = time.perf_counter()
        with open(recovery_file) as f:
            known_targets = json.load(f)
        t_end = time.perf_counter()
        logger.info(f"...load data from file in {t_end-t_start:.2f}sec")

        load_rank_history = self.config["load_rank_history"]
        recovered_rank_history = None
        if load_rank_history:
            recovered_rank_history_files = (
                self.path_manager.get_existing_rank_history_files()
            )
            if len(recovered_rank_history_files) == 0:
                logger.info("no files to recover rank history")
            else:
                logger.info("also load rank history")
                rank_history_file = recovered_rank_history_files[-1]
                t_start = time.perf_counter()
                with open(rank_history_file) as f:
                    recovered_rank_history = json.load(f)
                t_end = time.perf_counter()
                logger.info(f"...load data from file in {t_end-t_start:.2f}sec")

        recovered_targets = []
        missing_rank_history = []
        for target_id, target_info in known_targets.items():
            ra = target_info.pop("ra")
            dec = target_info.pop("dec")
            target = Target(
                target_id,
                coord=coord,
                base_score=target_info["base_score"],
                alt_ids=target_info.get("alt_ids", {}),
            )

            if recovered_rank_history is not None:
                target_rank_history = recovered_rank_history.get(target_id, [])
                if len(target_rank_history) == 0:
                    missing_rank_history.append(target_id)
                target.rank_history["no_observatory"] = target_rank_history

            recovered_targets.append(target)

        logger.info(f"recovered {len(recovered_targets)} targets from file")
        if recovered_rank_history is not None:
            logger.info(f"{len(missing_rank_history)} are missing rank history")
        return recovered_targets

    # def recover_score_history(self, target: Target):
    #     score_history_file = self.path_manager.get_score_history_file(target.target_id)
    #     if not score_history_file.exists():
    #         return
    #     score_history = pd.read_csv(score_history_file)
    #     for obs_name, history in score_history.groupby("observatory"):
    #         for ii, row in history.iterrows():
    #             score_t_ref = Time(row.mjd, format="mjd")
    #             target.update_score_history(row.score, obs_name, t_ref=score_t_ref)

    # def recover_rank_history(self, target: Target):
    #     rank_history_file = self.path_manager.get_rank_history_file(target.target_id)
    #     if not rank_history_file.exists():
    #         return
    #     rank_history = pd.read_csv(rank_history_file)
    #     for obs_name, history in rank_history.groupby("observatory"):
    #         for ii, row in history.iterrows():
    #             rank_t_ref = Time(row.mjd, format="mjd")
    #             target.update_rank_history(row["ranking"], obs_name, t_ref=rank_t_ref)
