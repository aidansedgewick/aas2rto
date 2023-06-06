import os
import yaml

import numpy as np

import pytest

from astroplan import Observer

from astropy import units as u
from astropy.time import Time

from dk154_targets import query_managers
from dk154_targets import Target, TargetSelector

from dk154_targets import paths


# @pytest.fixture()
def build_test_config():
    config = {
        "observatories": {
            "lasilla": "lasilla",
            "astrolab": {"lat": 54.75, "lon": -1.55, "height": 20},
        },
        "query_managers": {
            "test_qm1": {
                "username": "test_user",
                "token": "abcdef",
            },
            "test_qm2": {
                "use": False,  # Expect this NOT to appear.
                "username": "test_user",
                "server": "123.456.789.12",
            },
            "test_qm3": {
                "use": True,
                "user": "test_user",
                "password": "super_secret",
            },
        },
    }
    return config


def build_test_target_list():
    return [
        Target(
            "t1",
            ra=30.0,
            dec=10.0,
        ),
        Target(
            "t2",
            ra=60.0,
            dec=10.0,
        ),
        Target(
            "t3",
            ra=90.0,
            dec=10.0,
        ),
    ]


def test__selector_init():
    empty_config = {}
    selector = TargetSelector(empty_config)

    assert len(selector.selector_config) == 0
    assert len(selector.query_manager_config) == 0
    assert len(selector.observatories) == 1
    assert set(selector.observatories.keys()) == set(["no_observatory"])
    assert len(selector.target_lookup) == 0


def test__from_config():
    test_config_path = paths.test_path / "test_data/test_config.yaml"

    test_config = build_test_config()
    if test_config_path.exists():
        os.remove(test_config_path)
    with open(test_config_path, "w") as f:
        yaml.dump(test_config, f)
    # test_data_path = paths.test_path / "ts_config_test_data_path"
    # if test_data_path.exists():
    #    os.remove(test_data_path)
    # assert not test_data_path.exists()

    selector = TargetSelector.from_config(test_config_path)

    ## Test observatories
    assert len(selector.observatories) == 3
    assert selector.observatories["no_observatory"] is None
    assert isinstance(selector.observatories["lasilla"], Observer)
    assert np.isclose(selector.observatories["lasilla"].location.lon, -70.73 * u.deg)
    assert np.isclose(selector.observatories["astrolab"].location.lat, 54.75 * u.deg)

    ## Test paths
    # assert selector.data_path.name == "ts_config_test_data"
    # assert selector.outputs_path.name == "ts_config_outputs"

    ## Test query managers init
    assert len(selector.query_managers) == 2
    assert set(selector.query_managers.keys()) == set(["test_qm1", "test_qm3"])
    qm1 = selector.query_managers["test_qm1"]
    assert isinstance(qm1, query_managers.GenericQueryManager)

    # Clean up...
    os.remove(test_config_path)
    assert not test_config_path.exists()


def test__add_target():
    t1 = Target("t1", ra=10.0, dec=30.0)

    selector = TargetSelector({})


def test__perform_all_qm_tasks():
    class TestObject:
        def __init__(self, t_ref=None):
            self.t_ref = t_ref

    class TestQueryManager:
        def __init__(self, target_lookup, data_path=None):
            self.target_lookup = target_lookup

        def perform_all_tasks(self, t_ref=None):
            ii = len(self.target_lookup)
            target_name = f"test_{ii:03d}"
            self.target_lookup[target_name] = TestObject(t_ref=t_ref)

    selector = TargetSelector({})
    selector.query_managers["test_selector"] = TestQueryManager(selector.target_lookup)

    t_test = Time("2023-02-24T00:00:00.0", format="isot")

    assert len(selector.target_lookup) == 0
    selector.perform_query_manager_tasks(t_ref=t_test)
    assert len(selector.target_lookup) == 1
    assert np.isclose(
        selector.target_lookup["test_000"].t_ref.jd,
        2460000.0,
    )

    for ii in range(1, 5):
        t_ref = t_test + ii * u.day
        selector.perform_query_manager_tasks(t_ref=t_ref)

    assert len(selector.target_lookup) == 5
    assert set(selector.target_lookup.keys()) == set(
        ["test_000", "test_001", "test_002", "test_003", "test_004"]
    )


def test__compute_observatory_nights():
    config = build_test_config()
    selector = TargetSelector(config)

    time_atol = 0.014  # 20 minutes

    target_list = build_test_target_list()
    for target in target_list:
        selector.add_target(target)
    assert len(selector.target_lookup) == 3

    # Test is 'now' as the start of the night if now is nighttime?
    t_ref = Time("2023-02-25T00:00:00", format="isot")  # 2023-02-24-noon=2460000.
    selector.compute_observatory_nights(t_ref=t_ref)

    exp_astrolab_night_start = Time(2460000.5, format="jd")
    exp_astrolab_night_end = Time(2460000.71389, format="jd")  # from internet...
    assert np.isclose(
        target_list[0].observatory_night["astrolab"][0].jd - 2_460_000.0,
        exp_astrolab_night_start.jd - 2_460_000.0,
        atol=time_atol,
    )
    assert np.isclose(
        target_list[0].observatory_night["astrolab"][1].jd - 2_460_000.0,
        exp_astrolab_night_end.jd - 2_460_000.0,
        atol=time_atol,
    )

    # Test the night is correctly updated.
    t_ref = Time("2023-02-25T12:00:00", format="isot")  # jd=2460000.
    selector.compute_observatory_nights(t_ref=t_ref)

    exp_astrolab_night_start = Time(2460001.31458, format="jd")
    exp_astrolab_night_end = Time(2460001.71111, format="jd")  # from internet...
    assert np.isclose(
        target_list[0].observatory_night["astrolab"][0].jd - 2_460_000.0,
        exp_astrolab_night_start.jd - 2_460_000.0,
        atol=time_atol,
    )
    assert np.isclose(
        target_list[0].observatory_night["astrolab"][1].jd - 2_460_000.0,
        exp_astrolab_night_end.jd - 2_460_000.0,
        atol=time_atol,
    )

    exp_lasilla_night_start = Time(2460001.53472, format="jd")
    exp_lasilla_night_end = Time(2460001.87917, format="jd")
    assert np.isclose(
        target_list[0].observatory_night["lasilla"][0].jd - 2_460_000.0,
        exp_lasilla_night_start.jd - 2_460_000.0,
        atol=time_atol,
    )
    assert np.isclose(
        target_list[0].observatory_night["lasilla"][1].jd - 2_460_000.0,
        exp_lasilla_night_end.jd - 2_460_000.0,
        atol=time_atol,
    )


def test__initial_target_check():
    config = {
        "observatories": {"palomar": "palomar"},
    }

    def reject_high_ra_targets(target, obs):
        score = 80 - target.ra
        if target.ra > 80.0:
            score = -np.inf
        return score

    target_list = build_test_target_list()

    selector = TargetSelector(config)
    for target in target_list:
        selector.add_target(target)

    t_ref = Time("2023-02-25T00:00:00", format="isot")  # mjd=60000.

    new_objectIds = selector.new_target_initial_check(
        reject_high_ra_targets, t_ref=t_ref
    )
    assert set(new_objectIds) == set(["t1", "t2", "t3"])
    assert np.isclose(target_list[0].get_last_score(), 50.0)
    assert np.isclose(target_list[1].get_last_score(), 20.0)
    assert target_list[2].get_last_score() == -np.inf

    assert target_list[0].get_last_score("palomar") is None
    assert target_list[1].get_last_score("palomar") is None
    assert target_list[2].get_last_score("palomar") is None

    t4 = Target("t4", ra=30.0, dec=40.0)
    t5 = Target("t5", ra=120.0, dec=40.0)
    selector.add_target(t4)
    selector.add_target(t5)

    new_objectIds_2 = selector.new_target_initial_check(
        reject_high_ra_targets, t_ref=t_ref
    )
    assert set(new_objectIds_2) == set(["t4", "t5"])


def test__remove_bad_targets():
    config = {
        "observatories": {"palomar": "palomar"},
    }

    def reject_high_ra_targets(target, obs):
        score = 80 - target.ra
        if target.ra > 80.0:
            score = -np.inf
        return score

    target_list = build_test_target_list()

    selector = TargetSelector(config)
    for target in target_list:
        selector.add_target(target)

    t_ref = Time("2023-02-25T00:00:00", format="isot")  # mjd=60000.

    new_objectIds = selector.new_target_initial_check(
        reject_high_ra_targets, t_ref=t_ref
    )
    assert set(new_objectIds) == set(["t1", "t2", "t3"])
    rejected_targets = selector.reject_bad_targets()
    assert len(rejected_targets) == 1
    assert isinstance(rejected_targets[0], Target)
    assert rejected_targets[0].objectId == "t3"
