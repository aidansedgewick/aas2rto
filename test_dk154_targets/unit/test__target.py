import yaml

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from astroplan import Observer

from dk154_targets import Target, TargetData


def test__target_init():
    t1 = Target("test1", ra=45.0, dec=45.0)

    # Boring tests, will break if __init__ changes.
    assert isinstance(t1.coord, SkyCoord)
    assert isinstance(t1.alerce_data, TargetData)
    assert isinstance(t1.atlas_data, TargetData)
    assert isinstance(t1.fink_data, TargetData)
    assert isinstance(t1.tns_data, TargetData)

    assert set(t1.observatory_night.keys()) == set(["no_observatory"])

    assert len(t1.models) == 0
    assert set(t1.score_history.keys()) == set(["no_observatory"])
    assert set(t1.rank_history.keys()) == set(["no_observatory"])

    assert not t1.target_of_opportunity
    assert not t1.updated


def test__basic_evaluate_target():
    def basic_score(target, observer):
        return 50.0

    t1 = Target("test1", ra=45.0, dec=45.0)

    t_ref = Time("2023-02-25T00:00:00", format="isot")  # mjd=60000.
    test_obs = Observer(location=EarthLocation.of_site("palomar"), name="test_obs")

    # Evaluate
    t1.evaluate_target(basic_score, test_obs, t_ref=t_ref)
    assert set(t1.score_history.keys()) == set(["no_observatory", "test_obs"])
    assert len(t1.score_history["no_observatory"]) == 0
    assert len(t1.score_history["test_obs"]) == 1
    assert isinstance(t1.score_history["test_obs"][-1], tuple)
    assert isinstance(t1.score_history["test_obs"][-1][0], float)
    assert isinstance(t1.score_history["test_obs"][-1][1], Time)

    assert t1.score_history["test_obs"][-1][0] == 50.0
    assert t1.score_history["test_obs"][-1][1].mjd == 60000.0

    assert t1.score_comments["no_observatory"] is None
    assert t1.reject_comments["no_observatory"] is None
    assert t1.score_comments["test_obs"] is None
    assert t1.reject_comments["test_obs"] is None


def test__evaluate_target_with_comments():
    def score_with_comments(target, observer):
        return 50.0, ["this is a comment"], None

    t1 = Target("test1", ra=45.0, dec=45.0)

    t_ref = Time("2023-02-25T00:00:00", format="isot")  # mjd=60000.
    test_obs = Observer(location=EarthLocation.of_site("palomar"), name="test_obs")

    # Evaluate
    t1.evaluate_target(score_with_comments, test_obs, t_ref=t_ref)
    assert set(t1.score_history.keys()) == set(["no_observatory", "test_obs"])
    assert len(t1.score_history["no_observatory"]) == 0
    assert len(t1.score_history["test_obs"]) == 1
    assert isinstance(t1.score_history["test_obs"][-1], tuple)
    assert isinstance(t1.score_history["test_obs"][-1][0], float)
    assert isinstance(t1.score_history["test_obs"][-1][1], Time)

    assert t1.score_history["test_obs"][-1][1].mjd == 60000.0
    assert t1.score_history["test_obs"][-1][0] == 50.0

    assert t1.score_comments["no_observatory"] is None
    assert t1.reject_comments["no_observatory"] is None
    assert len(t1.score_comments["test_obs"]) == 1
    assert t1.score_comments["test_obs"][0] == "this is a comment"
    assert t1.reject_comments["test_obs"] is None


def test__get_last_score():
    def increasing_score(target, obs):
        if obs is None:
            obs_name = "no_observatory"
        else:
            obs_name = obs.name
        history = target.score_history.get(obs_name, [])
        return len(history) + 1.0

    palomar = Observer(location=EarthLocation.of_site("palomar"), name="palomar")
    t1 = Target("t1", ra=30.0, dec=45.0)
    assert set(t1.score_history.keys()) == set(["no_observatory"])

    # Check get_last_score() returns None if there is no score.
    last_score = t1.get_last_score()
    assert last_score is None
    last_score_tuple = t1.get_last_score(return_time=True)
    assert len(last_score_tuple) == 2
    assert not any(last_score_tuple)  # any(lst) is True if any are True-like.

    # Check the same behaviour for a named observatory.
    last_palomar_score = t1.get_last_score("palomar")
    assert last_palomar_score is None
    last_palomar_score_tuple = t1.get_last_score("palomar", return_time=True)
    assert len(last_palomar_score_tuple) == 2
    assert not any(last_palomar_score_tuple)

    # Evaluate the target 5 times for `no_observatory`.
    t_ref = Time("2023-02-25T00:00:00", format="isot")
    for ii in range(5):
        t_eval = t_ref + ii * u.day
        t1.evaluate_target(increasing_score, None, t_ref=t_eval)

    # Check that the last score is 5.
    last_score = t1.get_last_score()
    assert last_score == 5.0
    last_score_tuple = t1.get_last_score(return_time=True)
    assert last_score_tuple[0] == 5.0
    assert last_score_tuple[1].mjd == 60004.0
    no_obs_last_score = t1.get_last_score("no_observatory")
    assert no_obs_last_score == 5.0

    t1.evaluate_target(increasing_score, palomar, t_ref=t_ref)
    assert t1.get_last_score("palomar") == 1.0
    palomar_t1_last_score_tuple = t1.get_last_score("palomar", return_time=True)
    assert palomar_t1_last_score_tuple[0] == 1.0
    assert palomar_t1_last_score_tuple[1].mjd == 60000.0

    for ii in range(1, 3):
        t_eval = t_ref + ii * u.day
        t1.evaluate_target(increasing_score, palomar, t_ref=t_eval)
    assert t1.get_last_score("palomar") == 3.0
