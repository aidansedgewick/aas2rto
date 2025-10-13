import pytest

import numpy as np

from astropy import units as u
from astropy.time import Time

from astroplan import Observer

from aas2rto.exc import UnexpectedKeysWarning
from aas2rto.observatory_manager import ObservatoryManager
from aas2rto.path_manager import PathManager
from aas2rto.scoring.scoring_manager import (
    ScoringManager,
    science_score_wrapper,
    observatory_score_wrapper,
    parse_scoring_result,
)
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup


@pytest.fixture
def lasilla():
    return Observer.at_site("lasilla")


# ===== define some mock functions here ===== #


def scoring_func(target: Target, t_ref: Time):
    if target.coord.dec.deg > 15.0:
        return -10.0, ["target is excluded"]
    return 10.0, ["comment"]


def scoring_func_no_comms(target: Target, t_ref: Time):
    return 5.0


class ScoringFunc:
    def __init__(self, m=1.0):
        self.m = m

    def __call__(self, target: Target, t_ref: Time):
        return self.m, ["comment from ScoringFunc inst"]


def scoring_function_reject(target: Target, t_ref: Time):
    return -np.inf, ["reject"]


def bad_scoring_func(target: Target, t_ref: Time):
    raise Exception


def obs_scoring_func(target: Target, observatory: Observer, t_ref: Time):
    if observatory.location.lat.deg > 0.0:
        return 0.2, ["obs in north"]
    return 0.5, ["obs in south"]


def bad_obs_scoring_func(target: Target, observatory: Observer, t_ref: Time):
    raise Exception


def obs_scoring_not_finite(target: Target, observatory: Observer, t_ref: Time):
    return -np.inf, ["non-finite obs score"]


# ===== actual tests start here ===== #


class Test__ParseResult:

    def test__score_and_comms(self):
        # Arrange
        result = (1.0, ["comment1", "comment2"])

        # Act
        score, comments = parse_scoring_result(result)

        # Assert
        assert np.isclose(score, 1.0)
        assert isinstance(comments, list)

    def test__score_only(self):
        # Arrange
        result = 1.0

        # Act
        score, comments = parse_scoring_result(result)

        # Assert
        assert len(comments) == 1
        assert "no score_comments provided" in comments[0]

    def test__one_tuple_ok(self):
        # Arrange
        result = (1.0,)

        # Act
        score, comments = parse_scoring_result(result)

        # Assert
        assert np.isclose(score, 1.0)
        assert len(comments) == 1
        assert "no score_comments provided" in comments[0]

    def test__comments_none_no_raise(self):
        # Arrange
        result = 1.0, None

        # Act
        score, comments = parse_scoring_result(result)

        # Assert
        assert np.isclose(score, 1.0)
        assert len(comments) == 1
        assert "no score_comments provided" in comments[0]

    def test__bad_tuple_raises(self):
        # Arrange
        result = (1.0, 2.0, 3.0)

        # Act
        with pytest.raises(ValueError):
            parse_scoring_result(result)

    def test__bad_type_raises(self):
        # Arrange
        result = None

        # Act
        with pytest.raises(ValueError):
            parse_scoring_result(result)

    def test__bad_score_type_raises(self):
        # Arrange
        result = (None, None)

        # Act
        with pytest.raises(ValueError):
            parse_scoring_result(result)

    def test__bad_comments_type_raises(self):
        # Arrange
        result = (1.0, "comments")

        # Act
        with pytest.raises(ValueError):
            parse_scoring_result(result)


class Test__EvalSciScoreWrapper:
    def test__scoring_f(self, basic_target: Target, t_fixed: Time):
        # Act
        result = science_score_wrapper(scoring_func, basic_target, t_fixed)

        # Assert
        assert isinstance(result, tuple)
        assert isinstance(result[0], float)
        assert np.isclose(result[0], 10.0)

        assert isinstance(result[1], list)
        assert result[1][0] == "comment"

    def test__scoring_f_no_comms(self, basic_target: Target, t_fixed: Time):
        # Act
        result = science_score_wrapper(scoring_func_no_comms, basic_target, t_fixed)

        # Assert
        assert isinstance(result, tuple)
        assert isinstance(result[0], float)
        assert np.isclose(result[0], 5.0)

        assert isinstance(result[1], list)
        assert "no score_comments provided" in result[1][0]

    def test__bad_func_no_raise(self, basic_target: Target, t_fixed: Time):
        # Act
        result = science_score_wrapper(bad_scoring_func, basic_target, t_fixed)

        # Assert
        assert isinstance(result, tuple)
        assert isinstance(result[0], float)
        assert np.isclose(result[0], -1.0)

        assert isinstance(result[1], list)
        assert "Set science score to -1.0 to exclude" in result[1][0]

    def test__class_inst_scoring(self, basic_target: Target, t_fixed: Time):
        # Arrange
        inst_func = ScoringFunc(m=3.0)

        # Act
        result = science_score_wrapper(inst_func, basic_target, t_fixed)

        # Assert
        assert isinstance(result, tuple)
        assert isinstance(result[0], float)
        assert np.isclose(result[0], 3.0)

        assert isinstance(result[1], list)
        assert result[1][0] == "comment from ScoringFunc inst"


class Test__ObsScoreWrapper:
    def test__obs_scoring_f(
        self, basic_target: Target, lasilla: Observer, t_fixed: Time
    ):
        # Act
        result = observatory_score_wrapper(
            obs_scoring_func, basic_target, lasilla, t_fixed
        )

        # Assert
        assert isinstance(result, tuple)
        assert isinstance(result[0], float)
        assert np.isclose(result[0], 0.5)  #

        assert isinstance(result[1], list)
        assert result[1][0] == "obs in south"

    def test__bad_func_no_raise(
        self, basic_target: Target, lasilla: Observer, t_fixed: Time
    ):
        # Act
        result = observatory_score_wrapper(
            bad_obs_scoring_func, basic_target, lasilla, t_fixed
        )

        # Assert
        assert isinstance(result, tuple)
        assert isinstance(result[0], float)
        assert np.isclose(result[0], -1.0)

        assert isinstance(result[1], list)
        assert "Set lasilla score to -1.0 to exclude" in result[1][0]

    def test__obs_score_inf_caught(
        self, basic_target: Target, lasilla: Observer, t_fixed: Time
    ):
        # Act
        obs_factors, obs_comm = observatory_score_wrapper(
            obs_scoring_not_finite, basic_target, lasilla, t_fixed
        )

        # Assert
        assert np.isclose(obs_factors, -1.0)
        assert len(obs_comm) == 2
        assert "non-finite obs score" in obs_comm[0]
        assert "is non-finite" in obs_comm[1]


class Test__ScoreMgrInit:
    def test__score_mgr_init(
        self, tlookup: TargetLookup, path_mgr: PathManager, obs_mgr: ObservatoryManager
    ):
        # Act
        score_mgr = ScoringManager({}, tlookup, path_mgr, obs_mgr)

    def test__bad_config_warns(
        self, tlookup: TargetLookup, path_mgr: PathManager, obs_mgr: ObservatoryManager
    ):
        # Arrange
        config = {"bad_kwarg": True}

        # Act
        with pytest.warns(UnexpectedKeysWarning):
            score_mgr = ScoringManager(config, tlookup, path_mgr, obs_mgr)


class Test__EvalSingleTargetMethod:
    def test__score_target(self, scoring_mgr: ScoringManager, t_fixed: Time):
        # Arrange
        T00 = scoring_mgr.target_lookup["T00"]

        # Act
        scoring_mgr.evaluate_single_target(
            T00,
            scoring_func,
            observatory_scoring_function=obs_scoring_func,
            t_ref=t_fixed,
        )

        # Assert
        expected_keys = ["no_observatory", "lasilla", "astrolab"]
        assert set(T00.score_history.keys()) == set(expected_keys)
        assert np.isclose(T00.get_last_score(), 10.0)
        # check science_score and obs_score are COMBINED.
        assert np.isclose(T00.get_last_score("lasilla"), 5.0)  # 10. * 0.5
        assert np.isclose(T00.get_last_score("astrolab"), 2.0)  # 10. * 0.2

        assert set(T00.score_comments.keys()) == set(expected_keys)
        assert T00.score_comments["no_observatory"][0] == "comment"
        assert T00.score_comments["lasilla"][0] == "obs in south"
        assert T00.score_comments["astrolab"][0] == "obs in north"

    def test__score_target_bad_func(self, scoring_mgr: ScoringManager, t_fixed: Time):
        # Arrange
        T00 = scoring_mgr.target_lookup["T00"]

        # Act
        scoring_mgr.evaluate_single_target(
            T00,
            bad_scoring_func,
            observatory_scoring_function=obs_scoring_func,
            t_ref=t_fixed,
        )

        # Assert
        expected_keys = ["no_observatory", "lasilla", "astrolab"]
        assert set(T00.score_history.keys()) == set(expected_keys)
        assert np.isclose(T00.get_last_score(), -1.0)
        assert np.isclose(T00.get_last_score("lasilla"), -1.0)
        assert np.isclose(T00.get_last_score("astrolab"), -1.0)

        assert set(T00.score_comments.keys()) == set(expected_keys)
        exp_comm_str = "Set science score to -1.0 to exclude"
        assert exp_comm_str in T00.score_comments["no_observatory"][0]
        assert T00.score_comments["lasilla"][0] == "excluded by science score"
        assert T00.score_comments["astrolab"][0] == "excluded by science score"

    def test__obs_scoring_none(self, scoring_mgr: ScoringManager, t_fixed: Time):
        # Arrange
        T00 = scoring_mgr.target_lookup["T00"]

        # Act
        scoring_mgr.evaluate_single_target(
            T00,
            scoring_func,
            observatory_scoring_function=None,
            t_ref=t_fixed,
        )

        # Assert
        assert np.isclose(T00.get_last_score(), 10.0)

        assert set(T00.score_comments.keys()) == set(["no_observatory"])
        assert T00.score_comments["no_observatory"][0] == "comment"


class Test__EvaluateTargets:
    def test__all_evaluated(self, scoring_mgr: ScoringManager, t_fixed: Time):
        # Act
        scoring_mgr.evaluate_targets(
            scoring_func, observatory_scoring_function=obs_scoring_func, t_ref=t_fixed
        )

        # Assert
        T00 = scoring_mgr.target_lookup["T00"]
        T01 = scoring_mgr.target_lookup["T01"]
        expected_keys = ["no_observatory", "lasilla", "astrolab"]

        # Check all of T00 first...
        assert set(T00.score_history.keys()) == set(expected_keys)
        assert np.isclose(T00.get_last_score(), 10.0)
        assert np.isclose(T00.get_last_score("lasilla"), 5.0)
        assert np.isclose(T00.get_last_score("astrolab"), 2.0)

        assert set(T00.score_comments.keys()) == set(expected_keys)
        assert T00.score_comments["no_observatory"][0] == "comment"
        assert T00.score_comments["lasilla"][0] == "obs in south"
        assert T00.score_comments["astrolab"][0] == "obs in north"

        # Now check T01
        assert set(T01.score_history.keys()) == set(expected_keys)
        assert np.isclose(T01.get_last_score(), -10.0)
        assert np.isclose(T01.get_last_score("lasilla"), -10.0)  # exactly no_obs score
        assert np.isclose(T01.get_last_score("astrolab"), -10.0)

        assert set(T01.score_comments.keys()) == set(expected_keys)
        assert T01.score_comments["no_observatory"][0] == "target is excluded"
        assert T01.score_comments["lasilla"][0] == "excluded by science score"
        assert T01.score_comments["astrolab"][0] == "excluded by science score"


class Test__NewTargetInitCheck:
    def test__initial_check_score_ok(
        self, scoring_mgr: ScoringManager, southern_target: Target, t_fixed: Time
    ):
        # Arrange
        scoring_mgr.evaluate_targets(
            scoring_func, observatory_scoring_function=obs_scoring_func, t_ref=t_fixed
        )
        scoring_mgr.target_lookup.add_target(southern_target)
        t_later = t_fixed + 1.0 * u.day

        # Act
        scored_targets = scoring_mgr.new_target_initial_check(
            scoring_func, t_ref=t_later
        )

        # Assert
        assert set(scored_targets) == set(["T02"])

        # Check indiv targets.
        T00 = scoring_mgr.target_lookup["T00"]
        T00_exp_keys = ["no_observatory", "lasilla", "astrolab"]
        assert set(T00.score_history.keys()) == set(T00_exp_keys)
        assert len(T00.score_history["no_observatory"]) == 1
        sc_T00, t_T00 = T00.get_last_score(return_time=True)
        assert np.isclose(t_T00 - 60000.0, 0.0)  # ie. not re-scored at t_later...

        T02 = scoring_mgr.target_lookup["T02"]
        assert set(T02.score_history.keys()) == set(["no_observatory"])
        assert len(T02.score_history["no_observatory"]) == 1
        sc_T02, t_T02 = T02.get_last_score(return_time=True)
        assert np.isclose(sc_T02, 10.0)
        assert np.isclose(t_T02 - 60001.0, 0.0)

    def test__init_score_no_new_targets(self):
        pass
