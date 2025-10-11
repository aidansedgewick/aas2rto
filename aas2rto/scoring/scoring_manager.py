from logging import getLogger
from pathlib import Path
from typing import Callable, List

import numpy as np

from astropy.time import Time

from astroplan import Observer

from aas2rto import utils
from aas2rto.observatory_manager import ObservatoryManager
from aas2rto.path_manager import PathManager
from aas2rto.scoring.default_obs_scoring import DefaultObservatoryScoring
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup


logger = getLogger(__name__.split(".")[-1])


def parse_scoring_result(scoring_res, func_name=""):
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


class ScoringManager:

    default_scoring_config = {
        "minimum_score": 0.0,
    }

    def __init__(
        self,
        config: dict,
        target_lookup: TargetLookup,
        path_manager: PathManager,
        observatory_manager: ObservatoryManager,
    ):
        self.config = self.default_scoring_config.copy()
        self.config.update(config)
        utils.check_unexpected_config_keys(
            self.config, self.default_scoring_config, name="scoring_manager"
        )

        self.target_lookup = target_lookup
        self.path_manager = path_manager
        self.observatory_manager = observatory_manager

        return

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
        minimum_score = self.config["minimum_score"]

        # ===== Compute score according to science scoring ===== #
        science_score, score_comments = self._evaluate_target_science_score(
            science_scoring_function, target, t_ref=t_ref
        )

        obs_name = "no_observatory"
        target.update_score_history(science_score, obs_name, t_ref=t_ref)
        target.score_comments[obs_name] = score_comments

        if observatory_scoring_function is None:
            return

        for obs_name, observatory in self.observatory_manager.sites.items():
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

        func_name = science_scoring_function.__name__
        science_score, score_comments = parse_scoring_result(
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
        obs_factors, obs_comments = parse_scoring_result(scoring_res)
        if not np.isfinite(obs_factors):
            msg = (
                f"observatory_score not finite (={obs_factors}) "
                f"for {target.target_id} at {obs_name}."
            )
            logger.warning(msg)

        return obs_factors, obs_comments

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
