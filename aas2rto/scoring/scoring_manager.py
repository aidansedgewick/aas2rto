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


def parse_scoring_result(result, func_name=""):
    err_msg = (
        f"scoring function {func_name} should return\n"
        f"    score : float, or\n"
        f"    (score, comment_list) : Tuple[float, List[str]]"
    )
    missing_comments = [f"function {func_name}: no score_comments provided"]

    if isinstance(result, tuple):
        if len(result) == 2:
            score, comments = result
        elif len(result) == 1:
            score, comments = result[0], missing_comments
        else:
            raise ValueError(err_msg + f"\n your Tuple has len {len(result)}")
    elif isinstance(result, float) or isinstance(result, int):
        score, comments = result, missing_comments
    else:
        raise ValueError(err_msg + f"\nyour function returned type {type(result)}")

    if not isinstance(score, float) or isinstance(result, int):
        raise ValueError(err_msg + f"\nyour 'score' is type {type(score)}")

    if comments is None:
        comments = missing_comments
    if not isinstance(comments, list):
        raise ValueError(err_msg + f"\nyour 'comment_list' is type {type(comments)}")
    return score, comments


def science_score_wrapper(
    science_scoring_function: Callable,
    target: Target,
    t_ref: Time = None,
):
    """
    Wrapper function around user-provided science_scoring_function, to catch
    exceptions.

    """

    t_ref = t_ref or Time.now()

    try:
        func_name = science_scoring_function.__name__
    except AttributeError as e:
        func_name = type(science_scoring_function).__name__
    obs_name = "no_observatory"
    try:
        scoring_res = science_scoring_function(target, t_ref)
    except Exception as e:
        t_str = t_ref.strftime("%y-%m-%d %H:%M:%S")
        details = (
            f"For target {target.target_id} at {t_str},\n"
            f"    science score with '{func_name}' failed.\n"
            f"    Set science score to -1.0 to exclude. Details:\n    {type(e)}: {e}"
        )
        scoring_res = (-1.0, [details])

    science_score, score_comments = parse_scoring_result(
        scoring_res, func_name=func_name
    )
    return science_score, score_comments


def observatory_score_wrapper(
    observatory_scoring_function: Callable,
    target: Target,
    observatory: Observer,
    t_ref: Time,
):

    obs_name = observatory.name

    try:
        func_name = observatory_scoring_function.__name__
    except AttributeError as e:
        func_name = type(observatory_scoring_function).__name__

    t_str = t_ref.strftime("%y-%m-%d %H:%M:%S")

    try:
        scoring_res = observatory_scoring_function(target, observatory, t_ref=t_ref)
    except Exception as e:
        details = (
            f"For target {target.target_id} at obs {obs_name} at {t_str},\n"
            f"    observatory score with '{func_name}' failed.\n"
            f"    Set {obs_name} score to -1.0 to exclude. Details:\n    {type(e)}: {e}"
        )
        scoring_res = (-1.0, [details])

    obs_factors, obs_comments = parse_scoring_result(scoring_res, func_name=func_name)
    if not np.isfinite(obs_factors):
        msg = (
            f"For target {target.target_id} at obs {obs_name} at {t_str},\n"
            f"   obs score with '{func_name}' is non-finite (={obs_factors}). \n"
            f"    Set to -1.0"
        )
        obs_factors = -1.0
        obs_comments.append(msg)
        logger.warning(msg)

    return obs_factors, obs_comments


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
        observatory_scoring_function: Callable = None,
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
        science_score, score_comments = science_score_wrapper(
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
                obs_factors, score_comments = observatory_score_wrapper(
                    observatory_scoring_function, target, observatory, t_ref
                )
                observatory_score = science_score * obs_factors
            else:
                obs_factors = 1.0
                observatory_score = science_score
                score_comments = ["excluded by science score"]

            target.update_score_history(observatory_score, obs_name, t_ref=t_ref)
            target.score_comments[obs_name] = score_comments
        return

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
