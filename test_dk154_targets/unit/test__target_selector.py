import os
import pytest
import yaml

import numpy as np

import pandas as pd

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


@pytest.fixture
def test_target_list():
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
    test_config_path = paths.test_data_path / "test_config.yaml"

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

    ## Test query managers init
    assert len(selector.query_managers) == 2
    assert set(selector.query_managers.keys()) == set(["test_qm1", "test_qm3"])
    qm1 = selector.query_managers["test_qm1"]
    assert isinstance(qm1, query_managers.GenericQueryManager)

    # Clean up...
    os.remove(test_config_path)
    assert not test_config_path.exists()


# def test__process_paths():
#     tests_path = paths.base_path / "test_dk154_targets"
#     expected_base = paths.test_data_path / "test_base/further"
#     expected_data = paths.test_data_path / "test_base/further/data"
#     expected_outputs = paths.test_data_path / "test_base/further/outputs"

#     if expected_base.exists():
#         expected_base.rmdir()
#         expected_base.parent.rmdir()
#     assert not expected_base.exists()
#     if expected_data.exists():
#         expected_data.rmdir()
#     assert not expected_base.exists()
#     if expected_outputs.exists():
#         expected_outputs.rmdir()
#     assert not expected_outputs.exists()

#     config1 = {"paths": {}}


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


def test__compute_observatory_info(test_target_list):
    config = build_test_config()
    selector = TargetSelector(config)

    time_atol = 0.014  # 20 minutes

    for target in test_target_list:
        selector.add_target(target)
    assert len(selector.target_lookup) == 3

    # Test is 'now' as the start of the night if now is nighttime?
    t_ref = Time("2023-02-25T00:00:00", format="isot")  # 2023-02-24-noon=2460000.
    selector.compute_observatory_info(t_ref=t_ref)

    exp_astrolab_sunset = Time(2460000.5, format="jd")
    astrolab_sunset = test_target_list[0].observatory_info["astrolab"].sunset
    assert np.isclose(
        astrolab_sunset.mjd - 60000.0, exp_astrolab_sunset.mjd - 60000, atol=time_atol
    )
    exp_astrolab_sunrise = Time(2460000.71389, format="jd")  # from internet...
    astrolab_sunrise = test_target_list[0].observatory_info["astrolab"].sunrise
    assert np.isclose(
        astrolab_sunrise.mjd - 2_460_000.0,
        exp_astrolab_sunrise.mjd - 2_460_000.0,
        atol=time_atol,
    )

    # Test the night is correctly updated.
    t_ref = Time("2023-02-25T12:00:00", format="isot")  # jd=2460000.
    selector.compute_observatory_info(t_ref=t_ref)

    exp_astrolab_sunset = Time(2460001.31458, format="jd")
    astrolab_sunset = test_target_list[0].observatory_info["astrolab"].sunset
    assert np.isclose(
        astrolab_sunset.mjd - 60000.0, exp_astrolab_sunset.mjd - 60000.0, atol=time_atol
    )
    exp_astrolab_sunrise = Time(2460001.71111, format="jd")  # from internet...
    astrolab_sunrise = test_target_list[0].observatory_info["astrolab"].sunrise
    assert np.isclose(
        astrolab_sunrise.mjd - 60000.0,
        exp_astrolab_sunrise.mjd - 60000.0,
        atol=time_atol,
    )

    exp_lasilla_sunset = Time(2460001.53472, format="jd")
    exp_lasilla_sunrise = Time(2460001.87917, format="jd")
    lasilla_sunset = test_target_list[0].observatory_info["lasilla"].sunset
    lasilla_sunrise = test_target_list[0].observatory_info["lasilla"].sunrise
    assert np.isclose(
        lasilla_sunset.mjd - 60000.0, exp_lasilla_sunset.mjd - 60000.0, atol=time_atol
    )
    assert np.isclose(
        lasilla_sunrise.mjd - 60000.0, exp_lasilla_sunrise.mjd - 60000.0, atol=time_atol
    )


def test__observatory_info_no_crash_on_tonight_failure():
    config = {
        "observatories": {
            "SPT": dict(lat=-89.9, lon=90.0, height=2300.0),
            "NPT": dict(lat=89.9, lon=0.0),
            "greenwich": "greenwich",
        }
    }
    selector = TargetSelector(config)
    t1 = Target("test1", ra=5.0, dec=0)
    selector.add_target(t1)

    t_ref = Time("2022-12-21T12:00:00")  # Noon UT, solstice
    SPT = selector.observatories["SPT"]  # real - permanent day!
    NPT = selector.observatories["NPT"]  # fictional - permanent night!
    greenwich = selector.observatories["greenwich"]  # long, but finite, night.

    # Try the actual compute
    selector.compute_observatory_info(t_ref=t_ref)

    assert isinstance(SPT, Observer)
    assert isinstance(NPT, Observer)
    assert isinstance(greenwich, Observer)

    # Check it would have raised an error
    with pytest.raises(TypeError):
        SPT.tonight(t_ref)
    assert t1.observatory_info["SPT"].sunset is None
    assert t1.observatory_info["SPT"].sunrise is None

    NPT_tonight = NPT.tonight(t_ref)
    assert isinstance(NPT_tonight[0], Time)
    assert isinstance(NPT_tonight[0].jd, float)
    assert isinstance(NPT_tonight[1], Time)
    assert not isinstance(NPT_tonight[1].jd, float)
    assert not NPT_tonight[1] is None
    NPT_sunset = t1.observatory_info["NPT"].sunset
    assert np.isclose(NPT_sunset.mjd - 60000.0, t_ref.mjd - 60000.0)
    NPT_sunrise = t1.observatory_info["NPT"].sunrise
    assert NPT_sunrise is None  # Correctly set as None.

    greenwich_tonight = greenwich.tonight(t_ref)
    assert isinstance(greenwich_tonight[0], Time)
    assert isinstance(greenwich_tonight[0].jd, float)
    assert isinstance(greenwich_tonight[1], Time)
    assert isinstance(greenwich_tonight[1].jd, float)


def test__initial_target_check(test_target_list):
    config = {
        "observatories": {"palomar": "palomar"},
    }

    def reject_high_ra_targets(target, obs):
        score = 80 - target.ra
        if target.ra > 80.0:
            score = -np.inf
        return score

    selector = TargetSelector(config)
    for target in test_target_list:
        selector.add_target(target)

    t_ref = Time("2023-02-25T00:00:00", format="isot")  # mjd=60000.

    new_objectIds = selector.new_target_initial_check(
        reject_high_ra_targets, t_ref=t_ref
    )
    assert set(new_objectIds) == set(["t1", "t2", "t3"])
    assert np.isclose(test_target_list[0].get_last_score(), 50.0)
    assert np.isclose(test_target_list[1].get_last_score(), 20.0)
    assert test_target_list[2].get_last_score() == -np.inf

    assert test_target_list[0].get_last_score("palomar") is None
    assert test_target_list[1].get_last_score("palomar") is None
    assert test_target_list[2].get_last_score("palomar") is None

    t4 = Target("t4", ra=30.0, dec=40.0)
    t5 = Target("t5", ra=120.0, dec=40.0)
    selector.add_target(t4)
    selector.add_target(t5)

    new_objectIds_2 = selector.new_target_initial_check(
        reject_high_ra_targets, t_ref=t_ref
    )
    assert set(new_objectIds_2) == set(["t4", "t5"])


def test__evaluate_all_targets(test_target_list):
    config = {
        "observatories": {"palomar": "palomar"},
    }

    def test_score_with_obs(target, obs):
        factor = 2.0 if obs else 1.0
        return target.ra * factor

    selector = TargetSelector(config)
    for target in test_target_list:
        selector.add_target(target)
    t_ref = Time("2023-02-25T00:00:00", format="isot")  # mjd=60000.
    assert set(selector.observatories.keys()) == set(["no_observatory", "palomar"])
    assert set(test_target_list[0].score_history.keys()) == set(["no_observatory"])

    selector.evaluate_all_targets(test_score_with_obs, t_ref=t_ref)
    t1_no_obs = selector.target_lookup["t1"].get_last_score()
    t1_palomar = selector.target_lookup["t1"].get_last_score("palomar")
    assert np.isclose(t1_no_obs, 30.0)
    assert np.isclose(t1_palomar, 60.0)
    t2_no_obs = selector.target_lookup["t2"].get_last_score()
    t2_palomar = selector.target_lookup["t2"].get_last_score("palomar")
    assert np.isclose(t2_no_obs, 60.0)
    assert np.isclose(t2_palomar, 120.0)
    t3_no_obs = selector.target_lookup["t3"].get_last_score()
    t3_palomar = selector.target_lookup["t3"].get_last_score("palomar")
    assert np.isclose(t3_no_obs, 90.0)
    assert np.isclose(t3_palomar, 180.0)


def test__remove_bad_targets(test_target_list):
    config = {
        "observatories": {"palomar": "palomar"},
    }

    def reject_high_ra_targets(target, obs):
        score = 80 - target.ra
        if target.ra > 80.0:
            score = -np.inf
        return score

    selector = TargetSelector(config)
    for target in test_target_list:
        selector.add_target(target)

    t_ref = Time("2023-02-25T00:00:00", format="isot")  # mjd=60000.

    new_objectIds = selector.new_target_initial_check(
        reject_high_ra_targets, t_ref=t_ref
    )
    assert set(new_objectIds) == set(["t1", "t2", "t3"])
    rejected_targets = selector.remove_bad_targets()
    assert len(rejected_targets) == 1
    assert isinstance(rejected_targets[0], Target)
    assert rejected_targets[0].objectId == "t3"

    # Make sure even bad target_of_opportunities aren't rejected.
    t_opp = Target("t_opp", ra=190.0, dec=30.0)
    t_opp.target_of_opportunity = True

    selector.add_target(t_opp)
    new_objectIds_2 = selector.new_target_initial_check(
        reject_high_ra_targets, t_ref=t_ref
    )
    assert len(new_objectIds_2) == 1
    assert "t_opp" in new_objectIds_2
    selector.remove_bad_targets()
    assert "t_opp" in selector.target_lookup

    last_score = t_opp.get_last_score()
    assert not np.isfinite(last_score)


def test__compile_target_lightcurves(test_target_list):
    t1_data = pd.DataFrame(
        dict(
            mjd=np.arange(60000.0, 60005.0, 1.0),
            magpsf=[19.0, 18.0, 17.0, 16.0, 15.0],
            sigmapsf=[0.05] * 5,
            fid=[1, 2, 2, 2, 1],
        )
    )
    test_target_list[0].alerce_data.lightcurve = t1_data

    t2_data = pd.DataFrame(
        dict(
            mjd=np.arange(60000.1, 60005.1, 1.0),
            magpsf=[19.1, 18.1, 17.1, 16.1, 15.1],
            sigmapsf=[0.1] * 5,
            fid=[2, 2, 2, 1, 1],
        )
    )
    test_target_list[1].alerce_data.lightcurve = t2_data

    t3_data = pd.DataFrame(
        dict(
            mjd=np.arange(60000.2, 60005.2, 1.0),
            magpsf=[19.2, 18.2, 17.2, 16.2, 15.2],
            sigmapsf=[0.2] * 5,
            fid=[1, 2, 1, 1, 2],
        )
    )
    test_target_list[2].alerce_data.lightcurve = t3_data

    selector = TargetSelector({})
    for target in test_target_list:
        selector.add_target(target)
    assert all([t.compiled_lightcurve is None for t in test_target_list])

    t_ref = Time(60003.8, format="mjd")  # cut off the last datapoint
    selector.compile_target_lightcurves(t_ref=t_ref)

    assert all(["jd" in t.compiled_lightcurve.columns for t in test_target_list])
    assert all([len(t.compiled_lightcurve) == 4 for t in test_target_list])
    expected_t1_values = [19.0, 18.0, 17.0, 16.0]
    assert np.allclose(
        selector.target_lookup["t1"].compiled_lightcurve["mag"], expected_t1_values
    )
    expected_t2_values = [19.1, 18.1, 17.1, 16.1]
    assert np.allclose(
        selector.target_lookup["t2"].compiled_lightcurve["mag"], expected_t2_values
    )
    expected_t3_values = [19.2, 18.2, 17.2, 16.2]
    assert np.allclose(
        selector.target_lookup["t3"].compiled_lightcurve["mag"], expected_t3_values
    )


def test__opp_target_loading():
    expected_t_opp_path = paths.test_data_path / "t_opp"
    if expected_t_opp_path.exists():
        for filepath in expected_t_opp_path.glob("*.yaml"):
            os.remove(filepath)
        expected_t_opp_path.rmdir()
    assert not expected_t_opp_path.exists()

    config = {
        "paths": {
            "project_path": str(paths.test_data_path),
            "opp_targets_path": f"$project_path/t_opp",
        }
    }
    selector = TargetSelector(config)

    assert expected_t_opp_path.exists()
    assert expected_t_opp_path.is_dir()

    expected_t_opp_path.mkdir(exist_ok=True, parents=True)

    t1_yaml = expected_t_opp_path / "test1_opp.yaml"
    t2_yaml = expected_t_opp_path / "test2_opp.yaml"
    t3_yaml = expected_t_opp_path / "test3_opp.yaml"
    t4_yaml = expected_t_opp_path / "test4_opp.yaml"

    for fpath in [t1_yaml, t2_yaml, t3_yaml, t4_yaml]:
        if fpath.exists():
            os.remove(fpath)

    t1_dat = dict(objectId="test1", ra=45.0, dec=20.0)
    with open(t1_yaml, "w+") as f:
        yaml.dump(t1_dat, f)
    t2_dat = dict(objectId="test2", ra=45.0, dec=25.0)
    with open(t2_yaml, "w+") as f:
        yaml.dump(t2_dat, f)
    t3_dat = dict(objectId="test3", ra=45.0, dec=30.0, base_score=1000.0)
    with open(t3_yaml, "w+") as f:
        yaml.dump(t3_dat, f)

    for fpath in [t1_yaml, t2_yaml, t3_yaml]:
        assert fpath.exists()

    t3 = Target("test3", ra=45.0, dec=30.0, base_score=100.0)
    selector.add_target(t3)
    assert set(selector.target_lookup.keys()) == set(["test3"])

    # Try to add three targets - but one of them already exists.
    successful, existing, failed = selector.check_for_targets_of_opportunity()
    assert set(selector.target_lookup.keys()) == set(["test1", "test2", "test3"])

    assert set(successful) == set([t1_yaml, t2_yaml, t3_yaml])
    assert set(existing) == set([t3_yaml])
    assert set(failed) == set()

    assert not any([t1_yaml.exists(), t2_yaml.exists(), t3_yaml.exists()])

    # Now try to add one that should fail.
    t4_dat = dict(blah_blah="test4", ra=22.0, dec=10.0)
    with open(t4_yaml, "w+") as f:
        yaml.dump(t4_dat, f)

    successful, existing, failed = selector.check_for_targets_of_opportunity()
    assert set(selector.target_lookup.keys()) == set(["test1", "test2", "test3"])

    assert set(successful) == set()
    assert set(existing) == set()
    assert set(failed) == set([t4_yaml])

    assert t4_yaml.exists()
    os.remove(t4_yaml)
    expected_t_opp_path.rmdir()
    assert not expected_t_opp_path.exists()


def test__reset_updated_target_flags(test_target_list):
    selector = TargetSelector({})
    for target in test_target_list:
        selector.add_target(target)

    selector.target_lookup["t1"].updated = True
    selector.target_lookup["t2"].updated = True
    selector.target_lookup["t3"].updated = True

    for target in test_target_list:
        assert target.updated
    for objectId, target in selector.target_lookup.items():
        assert target.updated

    selector.reset_updated_target_flags()
    for target in test_target_list:
        assert not target.updated
    for objectId, target in selector.target_lookup.items():
        assert not target.updated
