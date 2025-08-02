import json
import yaml
from typing import List

import pytest

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from astropy.coordinates import EarthLocation
from astropy.time import Time

from astroplan import Observer

from aas2rto import messengers
from aas2rto import query_managers
from aas2rto import utils
from aas2rto.exc import (
    MissingKeysWarning,
    UnexpectedKeysWarning,
)
from aas2rto.target import Target, TargetData, UnknownObservatoryWarning
from aas2rto.target_selector import TargetSelector
from aas2rto.target_lookup import TargetLookup


@pytest.fixture
def selector_parameters():
    return {"sleep_time": 10.0}


@pytest.fixture
def fink_config():
    return {
        "object_queries": ["test_class"],
        "query_parameters": {"object_query_interval": 2.0},
    }


@pytest.fixture
def atlas_config():
    return {"token": "my_token", "project_identifier": "my_identifier"}


@pytest.fixture
def query_managers_config(fink_config, atlas_config):
    return {
        "fink": fink_config,
        "atlas": atlas_config,
    }


@pytest.fixture
def observatories_config():
    return {
        "lasilla": "lasilla",
        "ucph": {"lat": 55.68, "lon": 12.57},
    }


@pytest.fixture
def slack_config():
    return {"token": "some_token", "channel_id": 1234}


@pytest.fixture
def telegram_config():
    return {
        "token": "some_token",
        "users": {1001: "user_a", 1002: "user_b"},
        "sudoers": {2001: "sudo_a"},
        "users_file": None,
    }


@pytest.fixture
def messengers_config(slack_config, telegram_config):
    return {"slack": slack_config, "telegram": telegram_config}


@pytest.fixture
def selector_config(selector_parameters, tmp_path):
    return {
        "selector_parameters": selector_parameters,
        "paths": {"base_path": tmp_path},
    }


@pytest.fixture
def selector(selector_config):
    return TargetSelector(selector_config)


@pytest.fixture
def target_list() -> List[Target]:
    return [
        Target("T101", ra=15.0, dec=30.0),
        Target("T102", ra=30.0, dec=30.0),
        Target("T103", ra=45.0, dec=30.0),
        Target("T104", ra=60.0, dec=30.0),
        Target("T105", ra=75.0, dec=30.0),
        Target("T106", ra=90.0, dec=30.0),
    ]


@pytest.fixture
def mock_lc_rows():
    return [
        (60000.0, 19.5, 0.2, "badqual", "ztfg"),
        (60001.0, 19.0, 0.1, "valid", "ztfg"),
        (60002.0, 18.5, 0.1, "valid", "ztfg"),
        (60003.0, 18.0, 0.1, "valid", "ztfg"),
    ]


@pytest.fixture
def mock_lc(mock_lc_rows):
    return pd.DataFrame(mock_lc_rows, columns="mjd mag magerr tag band".split())


@pytest.fixture
def selector_with_targets(selector_config, observatories_config, target_list, mock_lc):
    selector_config["observatories"] = observatories_config

    selector = TargetSelector(selector_config)

    for ii, target in enumerate(target_list):
        lc = mock_lc.copy()
        lc.loc[:, "mag"] = lc["mag"] + ii * 0.2
        lc.loc[:, "mjd"] = lc["mjd"] + ii * 2.0
        # Last row will be:  T101  T102  T103  T104  T105  T106
        #              mag:  18.0  18.2  18.4  18.6  18.8  19.0
        #              mjd: 60003 60005 60007 60009 60011 60013

        target.target_data["ztf"] = TargetData(lightcurve=lc, include_badqual=False)
        selector.add_target(target)

    assert len(selector.target_lookup) == 6
    for target_id, target in selector.target_lookup.items():
        assert len(target.target_data["ztf"].lightcurve) == 4
    return selector


@pytest.fixture
def extra_targets(mock_lc):
    target_list = [
        Target("T107", ra=105.0, dec=45.0),
        Target("T108", ra=120.0, dec=45.0),
        Target("T109", ra=135.0, dec=45.0),
        Target("T110", ra=150.0, dec=45.0),
    ]
    for ii, target in enumerate(target_list):
        lc = mock_lc.copy()
        lc.loc[:, "mag"] = lc["mag"] + (ii + 6) * 0.2
        lc.loc[:, "mjd"] = lc["mjd"] + (ii + 6) * 2.0
        # Last row will be:  T107  T108  T109  T110
        #              mag:  19.2  19.4  19.6  19.8
        #              mjd: 60012 60014 60016 60018
        target.target_data["ztf"] = TargetData(lightcurve=lc)
    return target_list


def basic_lc_compiler(target, t_ref):
    detections = target.target_data["test_data"].detections.copy()
    badqual = target.target_data["test_data"].badqual.copy()
    limits = target.target_data["test_data"].non_detections.copy()

    detections.loc[:, "flux"] = 3631.0 * 10 ** (-detections["mag"] * 0.4)
    detections.loc[:, "tag"] = "valid"
    badqual.loc[:, "tag"] = "is_badqual"  # Can definitely check these have changed.
    limits.loc[:, "tag"] = "a_limit"
    lc = pd.concat([limits, badqual, detections], ignore_index=True)
    lc.loc[:, "band"] = "ztf-w"
    lc.loc[:, "jd"] = Time(lc["mjd"], format="mjd").jd
    return lc


@pytest.fixture
def test_target():
    t = Target("T101", ra=45.0, dec=60.0)
    return t


@pytest.fixture
def test_observer():
    location = EarthLocation(lat=55.6802, lon=12.5724, height=0.0)
    return Observer(location, name="ucph")


class Test__SelectorInit:

    def test__selector_normal_behaviour(self, selector_parameters, tmp_path):
        selector_config = {
            "selector_parameters": selector_parameters,
            "paths": {"base_path": tmp_path},
        }

        selector = TargetSelector(selector_config)

        assert isinstance(selector.selector_parameters, dict)
        sel_param = selector.selector_parameters
        assert np.isclose(sel_param["sleep_time"], 10.0)  # The only value we changed
        assert sel_param["unranked_value"] == 9999
        assert sel_param["minimum_score"] == 0.0
        assert sel_param["retained_recovery_files"] == 5
        assert isinstance(sel_param["skip_tasks"], list)
        assert set(sel_param["skip_tasks"]) == set()
        assert sel_param["lazy_modeling"]
        assert sel_param["lazy_plotting"]
        assert np.isclose(sel_param["plotting_interval"], 0.25)
        assert sel_param["write_comments"] is True

        assert isinstance(selector.query_managers_config, dict)
        assert isinstance(selector.observatories_config, dict)
        assert isinstance(selector.messengers_config, dict)
        assert isinstance(selector.paths_config, dict)

        assert isinstance(selector.target_lookup, TargetLookup)
        assert set(selector.target_lookup.keys()) == set()

        assert isinstance(selector.paths, dict)
        expected_path_keys = [
            "base_path",
            "project_path",
            "data_path",
            "outputs_path",
            "opp_targets_path",
            "scratch_path",
            "lc_scratch_path",
            "vis_scratch_path",
            "comments_path",
            "rejected_targets_path",
            "existing_targets_path",
        ]
        assert set(selector.paths.keys()) == set(expected_path_keys)
        assert selector.base_path == tmp_path
        for path_key in expected_path_keys:
            assert selector.paths[path_key].exists()

        assert isinstance(selector.query_managers, dict)
        assert set(selector.query_managers.keys()) == set()

        assert isinstance(selector.observatories, dict)
        assert set(selector.observatories.keys()) == set(["no_observatory"])
        assert selector.observatories["no_observatory"] is None

        assert isinstance(selector.messengers, dict)
        assert set(selector.messengers) == set()
        assert selector.slack_messenger is None
        assert selector.telegram_messenger is None

    def test__paths_are_processed(self, selector_parameters, tmp_path):
        sel_config = {"selector_parameters": selector_parameters}
        sel_config["paths"] = dict(
            base_path=tmp_path, extra_path=tmp_path / "some_extra"
        )
        selector = TargetSelector(sel_config)

        assert isinstance(selector.paths, dict)
        assert "extra_path" in selector.paths.keys()
        assert selector.paths["extra_path"] == tmp_path / "some_extra"
        assert selector.paths["extra_path"].exists()

    def test__placeholder_paths_are_processed(
        self, selector_config, selector_parameters, tmp_path
    ):
        selector_config = {
            "selector_parameters": selector_parameters,
            "paths": {
                "base_path": tmp_path,
                "project_path": tmp_path / "project_test",
                "cool_path": "$project_path/cool",
                "subcool_path": "$cool_path/subcool",
            },
        }

        selector = TargetSelector(selector_config)
        assert selector.paths["base_path"] == tmp_path
        assert selector.paths["project_path"] == tmp_path / "project_test"
        assert selector.paths["cool_path"] == tmp_path / "project_test/cool"
        assert selector.paths["subcool_path"] == tmp_path / "project_test/cool/subcool"

    def test__selector_init_qms(self, selector_config, query_managers_config):
        selector_config["query_managers"] = query_managers_config

        selector = TargetSelector(selector_config)

        assert isinstance(selector.query_managers, dict)
        assert set(selector.query_managers.keys()) == set(["fink", "atlas"])
        assert isinstance(
            selector.query_managers["fink"], query_managers.FinkQueryManager
        )
        assert isinstance(
            selector.query_managers["atlas"], query_managers.AtlasQueryManager
        )

    def test__no_expction_for_unknown_qm(self, selector_config):
        selector_config["query_managers"] = {
            "new_qm": {"cool_config": 1.0, "user": "me"}
        }

        selector = TargetSelector(selector_config)

        assert set(selector.query_managers.keys()) == set()  # But no "new_qm"

    def test__qm_use_False_is_respected(self, selector_config, query_managers_config):
        query_managers_config["fink"]["use"] = False
        selector_config["query_managers"] = query_managers_config

        selector = TargetSelector(selector_config)

        assert set(selector.query_managers_config.keys()) == set(["fink", "atlas"])
        assert set(selector.query_managers.keys()) == set(
            ["atlas"]
        )  # fink is not included!

    def test__obs_from_astropy_of_site(self, selector_config):
        selector_config["observatories"] = {
            "lasilla": "lasilla",
            "a_point": "Apache Point",
        }

        selector = TargetSelector(selector_config)

        assert set(selector.observatories.keys()) == set(
            ["no_observatory", "lasilla", "a_point"]
        )

        assert isinstance(selector.observatories["lasilla"], Observer)
        assert isinstance(selector.observatories["a_point"], Observer)

        assert selector.observatories["lasilla"].name == "lasilla"
        assert selector.observatories["a_point"].name == "a_point"  # our name
        assert not selector.observatories["a_point"].name == "Apache Point"

    def test__obs_from_lon_lat(self, selector_config):
        selector_config["observatories"] = {
            "astrolab": dict(lat=54.7670, lon=-1.5741, height=20),
            "nassau": {"lat": 25.0777, "lon": -77.3405},  # No height
            "puntas_arenas": {"lat": -53.16, "lon": -70.91},  # Both negative
            "ucph": dict(lat=55.68, lon=12.57),
        }

        selector = TargetSelector(selector_config)

        assert set(selector.observatories.keys()) == set(
            ["no_observatory", "astrolab", "nassau", "puntas_arenas", "ucph"]
        )

        assert isinstance(selector.observatories["astrolab"], Observer)
        assert isinstance(selector.observatories["nassau"], Observer)
        assert isinstance(selector.observatories["puntas_arenas"], Observer)
        assert isinstance(selector.observatories["ucph"], Observer)

        # Are they correct?
        astrolab_nassau = utils.haversine(
            selector.observatories["astrolab"].location,
            selector.observatories["nassau"].location,
        )
        assert np.isclose(astrolab_nassau, 6856.45, rtol=0.01)

    def test__init_messengers(self, selector_config, messengers_config):
        selector_config["messengers"] = messengers_config

        selector = TargetSelector(selector_config)

        assert set(selector.messengers.keys()) == set(["slack", "telegram"])

        assert isinstance(selector.slack_messenger, messengers.SlackMessenger)
        assert isinstance(selector.telegram_messenger, messengers.TelegramMessenger)

    def test__msgr_use_false_respected(self, selector_config, messengers_config):
        messengers_config["slack"]["use"] = False
        selector_config["messengers"] = messengers_config

        selector = TargetSelector(selector_config)

        assert set(selector.messengers_config.keys()) == set(["slack", "telegram"])
        assert set(selector.messengers.keys()) == set(["telegram"])  # slack not used!
        assert selector.slack_messenger is None

    def test__init_from_config(
        self,
        selector_config,
        query_managers_config,
        observatories_config,
    ):
        selector_config["selector_parameters"]["project_name"] = "fun_project"
        selector_config["query_managers"] = query_managers_config
        selector_config["observatories"] = observatories_config
        selector_config["paths"].update({"cool_path": "cool"})

        config_path = selector_config["paths"]["base_path"] / "test_config.yaml"

        selector_config["paths"] = {
            k: str(v) for k, v in selector_config["paths"].items()
        }  # In order to dump...

        with open(config_path, "w+") as f:
            yaml.dump(selector_config, f)

        assert config_path.exists()

        selector = TargetSelector.from_config(config_path)

        assert isinstance(selector, TargetSelector)

        assert np.isclose(selector.selector_parameters["sleep_time"], 10.0)

        assert set(selector.query_managers.keys()) == set(["fink", "atlas"])

        assert set(selector.observatories.keys()) == set(
            ["no_observatory", "lasilla", "ucph"]
        )


class Test__TargetsOfOpportunity:
    def test__add_target_of_opportunity(self, selector: TargetSelector):
        t_ref = Time(60000.0, format="mjd")

        t_opp_path = selector.opp_targets_path
        assert t_opp_path.exists()

        T101_data = {"target_id": "T101", "ra": 45.0, "dec": 30.0, "base_score": 1000.0}
        T101_file = t_opp_path / "target101.yaml"
        assert not T101_file.exists()

        with open(T101_file, "w+") as f:
            yaml.dump(T101_data, f)
        assert T101_file.exists()

        assert len(selector.target_lookup) == 0

        success, existing, failed = selector.check_for_targets_of_opportunity(
            t_ref=t_ref
        )

        assert set([f.name for f in success]) == set(["target101.yaml"])
        assert set(existing) == set()
        assert set(failed) == set()

        assert len(selector.target_lookup) == 1
        assert set(selector.target_lookup.keys()) == set(["T101"])

        target = selector.target_lookup["T101"]
        assert target.target_id == "T101"
        assert target.target_of_opportunity
        assert np.isclose(target.ra, 45.0)
        assert np.isclose(target.dec, 30.0)
        assert np.isclose(target.base_score, 1000.0)
        assert np.isclose(target.creation_time.mjd, 60000.0)

    def test__skip_existing_targets(self, selector: TargetSelector):
        t_ref = Time(60000.0, format="mjd")

        T101_in = Target("T101", ra=45.0, dec=30.0)
        assert T101_in.target_of_opportunity is False
        assert np.isclose(T101_in.base_score, 1.0)
        selector.add_target(T101_in)

        t_opp_path = selector.opp_targets_path
        assert t_opp_path.exists()

        T101_data = {"target_id": "T101", "ra": 45.0, "dec": 30.0, "base_score": 1000.0}
        T101_file = t_opp_path / "target101.yaml"
        assert not T101_file.exists()
        with open(T101_file, "w+") as f:
            yaml.dump(T101_data, f)
        assert T101_file.exists()

        T102_data = {"target_id": "T102", "ra": 90.0, "dec": 30.0, "base_score": 1000.0}
        T102_file = t_opp_path / "target102.yaml"
        assert not T102_file.exists()
        with open(T102_file, "w+") as f:
            yaml.dump(T102_data, f)
        assert T102_file.exists()

        assert len(selector.target_lookup) == 1

        success, existing, failed = selector.check_for_targets_of_opportunity()

        assert set([f.name for f in success]) == set(["target102.yaml"])
        assert set([f.name for f in existing]) == set(["target101.yaml"])
        assert set(failed) == set()

        assert not T101_file.exists()
        assert not T102_file.exists()

        assert set(selector.target_lookup.keys()) == set(["T101", "T102"])

        T101 = selector.target_lookup["T101"]
        assert np.isclose(T101.base_score, 1000.0)
        assert T101.target_of_opportunity is True  # updated!


def basic_lc_compiler(target: Target, t_ref: Time):
    if "ztf" not in target.target_data.keys():
        return None
    lc = target.target_data["ztf"].detections.copy()
    lc["flux"] = 3631.0 * 10 ** (-0.4 * lc["mag"])
    lc["fluxerr"] = lc["flux"] * lc["magerr"]
    return lc


class Test__CompileTargetLightcurves:
    def test__compiled_lightcurves(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        for target_id, target in selector.target_lookup.items():
            assert target.compiled_lightcurve is None

        compiled, not_compiled = selector.compile_target_lightcurves(
            lightcurve_compiler=basic_lc_compiler
        )

        assert set(compiled) == set(["T101", "T102", "T103", "T104", "T105", "T106"])
        assert set(not_compiled) == set()
        for target_id, target in selector.target_lookup.items():
            assert isinstance(target.compiled_lightcurve, pd.DataFrame)
            assert set(target.compiled_lightcurve.columns)
            assert len(target.compiled_lightcurve) == 3

    def test__no_exc_raised_for_not_compiled(
        self, selector_with_targets: TargetSelector
    ):
        selector = selector_with_targets

        selector.target_lookup["T105"].target_data.pop("ztf")
        assert "ztf" not in selector.target_lookup["T105"].target_data
        selector.target_lookup["T106"].target_data.pop("ztf")
        assert "ztf" not in selector.target_lookup["T106"].target_data

        compiled, not_compiled = selector.compile_target_lightcurves(
            lightcurve_compiler=basic_lc_compiler
        )

        assert set(compiled) == set(["T101", "T102", "T103", "T104"])
        assert set(not_compiled) == set(["T105", "T106"])


def basic_scoring(target: Target, t_ref: Time):
    factors = []
    comms = []
    rej_comms = []
    reject = False
    exclude = False

    detections = target.target_data["ztf"].detections
    # for each T:  T101  T102  T103  T104  T105  T106  T107  T108  T109  T110
    #   last mag:  18.0  18.2  18.4  18.6  18.8  19.0  19.2  19.4  19.6  19.8
    #   last mjd: 60003 60005 60007 60009 60011 60013
    # and factor:   2.0   1.8   1.6   1.4   1.2   1.0   0.8   0.6   0.4   0.2

    last_mag = detections["mag"].iloc[-1]
    mag_factor = 20 - last_mag
    factors.append(mag_factor)
    comms.append(f"mag_factor={mag_factor:.1f}")

    if last_mag > 19.5:
        exclude = True  # excludes T109 T110

    delta_t = t_ref.mjd - detections["mjd"].iloc[-1]
    if delta_t > 20.0:
        rej_comms.append(f"REJECT: target {target.target_id} is old")
        reject = True  # @mjd=600

    score = target.base_score * np.prod(factors)
    if exclude:
        score = -1.0
    if reject:
        score = -np.inf

    comms.extend(rej_comms)

    return score, comms


def basic_scoring_no_comms(target: Target, t_ref: Time):
    score, _ = basic_scoring(target, t_ref=t_ref)
    return score


def basic_obs_scoring(target, observatory, t_ref):
    obs_name = observatory.name
    comms = [f"obs_factor=0.5 fixed ({obs_name})"]
    return 0.5, comms


def scoring_will_raise_error(target: Target, t_ref: Time):
    raise ValueError()


def scoring_bad_return(target: Target, t_ref: Time):
    return 10.0, ["some_comments"], "extra_junk"


class ScoringClass:
    __name__ = "scoring_class"

    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier

    def __call__(self, target: Target, t_ref: Time):
        score, comms = basic_scoring(target, t_ref)
        return score * self.multiplier, comms


class ScoringClassNoName:
    def __call__(self, target: Target, t_ref: Time):
        return 10.0


class Test__EvaluateTargets:

    def test__wrapper_eval_target_sci_score(
        self, selector_with_targets: TargetSelector
    ):
        selector = selector_with_targets
        t_ref = Time(60000.0, format="mjd")
        T101 = selector.target_lookup["T101"]

        assert set(T101.score_history.keys()) == set(["no_observatory"])
        assert len(T101.score_history["no_observatory"]) == 0  # There are no scores

        score, comms = selector._evaluate_target_science_score(
            basic_scoring, T101, t_ref=t_ref
        )
        assert np.isclose(score, 2.00)
        assert set(comms) == set(["mag_factor=2.0"])

        assert (
            len(T101.score_history["no_observatory"]) == 0
        )  # No scores attached to the target.

    def test__wrapper_eval_sci_score_no_comms(
        self, selector_with_targets: TargetSelector
    ):
        selector = selector_with_targets
        t_ref = Time(60000.0, format="mjd")
        T101 = selector.target_lookup["T101"]

        assert set(T101.score_history.keys()) == set(["no_observatory"])
        assert len(T101.score_history["no_observatory"]) == 0  # There are no scores

        score, comms = selector._evaluate_target_science_score(
            basic_scoring_no_comms, T101, t_ref=t_ref
        )
        assert np.isclose(score, 2.00)  # The same as before...
        assert len(comms) == 1
        assert "no score_comments provided" in comms[0]
        # assert set(rej_comms) == set(["no reject_comments provided"])
        assert len(T101.score_history["no_observatory"]) == 0

    def test__wrapper_eval_obs_score(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        t_ref = Time(60000.0, format="mjd")
        T101 = selector.target_lookup["T101"]
        lasilla = selector.observatories["lasilla"]

    def test__eval_single_target_sci_only(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        t_ref = Time(60000.0, format="mjd")
        T101 = selector.target_lookup["T101"]

        assert set(T101.score_history.keys()) == set(["no_observatory"])
        assert len(T101.score_history["no_observatory"]) == 0  # There are no scores

        selector.evaluate_single_target(T101, basic_scoring, t_ref=t_ref)
        assert np.isclose(T101.get_last_score(), 2.00)  # Score now attached to target.

        assert "lasilla" in selector.observatories
        with pytest.warns(UnknownObservatoryWarning):
            assert T101.get_last_score("lasilla") is None
            assert T101.get_last_score("ucph") is None

    def test__eval_single_target_sci_and_obs(
        self, selector_with_targets: TargetSelector
    ):
        selector = selector_with_targets
        t_ref = Time(60000.0, format="mjd")
        T101 = selector.target_lookup["T101"]

        assert set(T101.score_history.keys()) == set(["no_observatory"])
        assert len(T101.score_history["no_observatory"]) == 0  # There are no scores

        selector.evaluate_single_target(
            T101,
            basic_scoring,
            observatory_scoring_function=basic_obs_scoring,
            t_ref=t_ref,
        )

        assert set(T101.score_history.keys()) == set(
            ["no_observatory", "lasilla", "ucph"]
        )
        assert np.isclose(T101.get_last_score(), 2.00)
        assert np.isclose(T101.get_last_score("lasilla"), 1.00)
        assert np.isclose(T101.get_last_score("ucph"), 1.00)

    def test__no_crash_on_bad_function(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        t_ref = Time(60000.0, format="mjd")
        T101 = selector.target_lookup["T101"]

        with pytest.raises(ValueError):
            tt = Target("tt", ra=180.0, dec=0.0)
            scoring_will_raise_error(tt, t_ref=t_ref)

        selector._evaluate_target_science_score(
            scoring_will_raise_error, T101, t_ref=t_ref
        )  # Correctly catches the errors!

    def test__evaluate_targets(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        lasilla = selector.observatories["lasilla"]
        ucph = selector.observatories["ucph"]
        t_ref = Time(60000.0, format="mjd")

        selector.evaluate_targets(
            basic_scoring,
            observatory_scoring_function=basic_obs_scoring,
            t_ref=t_ref,
        )

        t_lookup = selector.target_lookup
        assert np.isclose(t_lookup["T101"].get_last_score(), 2.00)
        assert np.isclose(t_lookup["T101"].get_last_score(lasilla), 1.00)
        assert np.isclose(t_lookup["T101"].get_last_score("lasilla"), 1.00)
        assert np.isclose(t_lookup["T101"].get_last_score(ucph), 1.00)
        assert np.isclose(t_lookup["T101"].get_last_score("ucph"), 1.00)

        assert np.isclose(t_lookup["T103"].get_last_score(), 1.60)
        assert np.isclose(t_lookup["T103"].get_last_score(lasilla), 0.80)
        assert np.isclose(t_lookup["T103"].get_last_score("lasilla"), 0.80)
        assert np.isclose(t_lookup["T103"].get_last_score(ucph), 0.80)
        assert np.isclose(t_lookup["T103"].get_last_score("ucph"), 0.80)

        assert set(t_lookup["T101"].score_comments.keys()) == set(
            ["no_observatory", "lasilla", "ucph"]
        )
        assert set(t_lookup["T101"].score_comments["no_observatory"]) == set(
            ["mag_factor=2.0"]
        )
        assert set(t_lookup["T101"].score_comments["lasilla"]) == set(
            ["obs_factor=0.5 fixed (lasilla)"]
        )
        assert set(t_lookup["T101"].score_comments["ucph"]) == set(
            ["obs_factor=0.5 fixed (ucph)"]
        )

    def test__eval_all_targets_with_class(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        t_lookup = selector.target_lookup
        t_ref = Time(60000.0, format="mjd")

        scoring_func = ScoringClass(multiplier=10.0)

        selector.evaluate_targets(
            scoring_func,
            observatory_scoring_function=basic_obs_scoring,
            t_ref=t_ref,
        )

        assert np.isclose(t_lookup["T101"].get_last_score(), 20.00)
        assert np.isclose(t_lookup["T101"].get_last_score("lasilla"), 10.00)
        assert np.isclose(t_lookup["T101"].get_last_score("ucph"), 10.00)
        assert np.isclose(t_lookup["T102"].get_last_score(), 18.00)
        assert np.isclose(t_lookup["T102"].get_last_score("lasilla"), 9.00)
        assert np.isclose(t_lookup["T102"].get_last_score("ucph"), 9.00)
        assert np.isclose(t_lookup["T103"].get_last_score(), 16.00)
        assert np.isclose(t_lookup["T103"].get_last_score("lasilla"), 8.00)
        assert np.isclose(t_lookup["T103"].get_last_score("ucph"), 8.00)

    def test__new_target_initial_check(
        self, selector_with_targets: TargetSelector, extra_targets
    ):
        selector = selector_with_targets
        t_ref = Time(60000.0, format="mjd")
        t_future = Time(60010.0, format="mjd")
        t_lookup = selector.target_lookup

        scored = selector.new_target_initial_check(basic_scoring, t_ref=t_ref)
        assert set(scored) == set(["T101", "T102", "T103", "T104", "T105", "T106"])
        assert np.isclose(
            t_lookup["T101"].get_last_score(return_time=True)[1].mjd, 60000.0
        )
        assert np.isclose(
            t_lookup["T102"].get_last_score(return_time=True)[1].mjd, 60000.0
        )

        assert "lasilla" in selector.observatories
        assert "ucph" in selector.observatories
        with pytest.warns(UnknownObservatoryWarning):
            assert t_lookup["T101"].get_last_score("lasilla") is None  # Not scored
            assert t_lookup["T101"].get_last_score("ucph") is None  # Not scored at obs.

        for target in extra_targets:
            selector.add_target(target)

        new_scored = selector.new_target_initial_check(basic_scoring, t_ref=t_future)
        assert set(new_scored) == set(["T107", "T108", "T109", "T110"])
        assert np.isclose(t_lookup["T107"].get_last_score(), 0.80)
        assert np.isclose(t_lookup["T108"].get_last_score(), 0.60)
        assert np.isclose(t_lookup["T109"].get_last_score(), -1.0)  # fainter than 19.5
        assert np.isclose(t_lookup["T109"].get_last_score(), -1.0)  # fainter than 19.5

        assert np.isclose(
            t_lookup["T101"].get_last_score(return_time=True)[1].mjd,
            60000.0,  # The same
        )
        assert np.isclose(
            t_lookup["T102"].get_last_score(return_time=True)[1].mjd,
            60000.0,  # The same
        )
        assert np.isclose(
            t_lookup["T107"].get_last_score(return_time=True)[1].mjd, 60010.0  # new
        )
        assert np.isclose(
            t_lookup["T108"].get_last_score(return_time=True)[1].mjd, 60010.0
        )
        assert np.isclose(
            t_lookup["T109"].get_last_score(return_time=True)[1].mjd, 60010.0
        )
        assert np.isclose(
            t_lookup["T110"].get_last_score(return_time=True)[1].mjd, 60010.0
        )

    def test__remove_rejected_targets(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets  # rename for ease
        t_lookup = selector.target_lookup  # rename for ease

        t1 = Time(60026.0, format="mjd")  # should remove T101, T102
        selector.evaluate_targets(basic_scoring, t_ref=t1)

        assert not np.isfinite(t_lookup["T101"].get_last_score())
        assert not np.isfinite(t_lookup["T102"].get_last_score())
        assert np.isclose(t_lookup["T103"].get_last_score(), 1.60)
        assert np.isclose(t_lookup["T104"].get_last_score(), 1.40)
        assert np.isclose(t_lookup["T105"].get_last_score(), 1.20)
        assert np.isclose(t_lookup["T106"].get_last_score(), 1.00)

        # Not testing comms yet...
        removed = selector.remove_rejected_targets(t_ref=t1, write_comments=False)

        assert set([t.target_id for t in removed]) == set(["T101", "T102"])
        assert set(t_lookup.keys()) == set(["T103", "T104", "T105", "T106"])

        t2 = Time(60028.0, format="mjd")  # Remove T103
        selector.evaluate_targets(basic_scoring, t_ref=t2)

        assert not np.isfinite(t_lookup["T103"].get_last_score())

        # Don't test comms yet
        removed = selector.remove_rejected_targets(t_ref=t2, write_comments=False)

        assert set([t.target_id for t in removed]) == set(["T103"])
        assert set(t_lookup.keys()) == set(["T104", "T105", "T106"])


class BasicModel:
    def __init__(self, mag, mag_offset=0.0):
        self.mag = mag + mag_offset


def basic_modeler(target: Target, t_ref: Time = None):
    detections = target.target_data["ztf"].detections
    return BasicModel(detections["mag"].iloc[-1])


class BasicModelBuilder:
    def __init__(self, mag_offset=0.0):
        self.mag_offset = mag_offset

    def __call__(self, target: Target, t_ref: Time = None):
        detections = target.target_data["ztf"].detections
        return BasicModel(detections["mag"].iloc[-1], mag_offset=self.mag_offset)


class AnotherModel:
    def __init__(self, mag):
        self.flux = 3631.0 * 10 ** (-0.4 * mag)


def other_modeler(target: Target, t_ref: Time = None):
    detections = target.target_data["ztf"].detections
    if detections["mag"].iloc[-1] > 18.7:
        raise ValueError()  # Uh-oh, we've failed! for T105, T106
    return AnotherModel(detections["mag"].iloc[-1])


class Test__SelectorBuildModels:

    def test__build_models_func(self, selector_with_targets):
        selector = selector_with_targets
        t_lookup = selector.target_lookup
        t_ref = Time(60000.0, format="mjd")

        selector.build_target_models(basic_modeler, t_ref=t_ref)

        for target_id, target in t_lookup.items():
            assert set(target.models.keys()) == set(["basic_modeler"])
            assert isinstance(target.models["basic_modeler"], BasicModel)
            assert np.isclose(target.models_t_ref["basic_modeler"].mjd, 60000.0)

        assert np.isclose(t_lookup["T101"].models["basic_modeler"].mag, 18.0)
        assert np.isclose(t_lookup["T102"].models["basic_modeler"].mag, 18.2)
        assert np.isclose(t_lookup["T103"].models["basic_modeler"].mag, 18.4)
        assert np.isclose(t_lookup["T104"].models["basic_modeler"].mag, 18.6)
        assert np.isclose(t_lookup["T105"].models["basic_modeler"].mag, 18.8)
        assert np.isclose(t_lookup["T106"].models["basic_modeler"].mag, 19.0)

    def test__no_exception_if_fitting_fails(
        self, selector_with_targets: TargetSelector
    ):
        selector = selector_with_targets
        t_lookup = selector.target_lookup
        t_ref = Time(60000.0, format="mjd")

        with pytest.raises(ValueError):
            other_modeler(t_lookup["T105"])

        with pytest.raises(ValueError):
            other_modeler(t_lookup["T106"])

        selector.build_target_models(other_modeler)

        for target_id, target in t_lookup.items():
            assert set(target.models.keys()) == set(["other_modeler"])

        assert isinstance(t_lookup["T101"].models["other_modeler"], AnotherModel)
        assert np.isclose(t_lookup["T101"].models["other_modeler"].flux, 2.2910061e-04)
        assert isinstance(t_lookup["T102"].models["other_modeler"], AnotherModel)
        assert np.isclose(t_lookup["T102"].models["other_modeler"].flux, 1.9055759e-04)
        assert isinstance(t_lookup["T103"].models["other_modeler"], AnotherModel)
        assert np.isclose(t_lookup["T103"].models["other_modeler"].flux, 1.5849890e-04)
        assert isinstance(t_lookup["T104"].models["other_modeler"], AnotherModel)
        assert np.isclose(t_lookup["T104"].models["other_modeler"].flux, 1.3183364e-04)
        assert t_lookup["T105"].models["other_modeler"] is None
        assert t_lookup["T106"].models["other_modeler"] is None

    def test__several_modeling_func(self, selector_with_targets):
        selector = selector_with_targets
        t_lookup = Time(60000.0, format="mjd")


class Test__WriteTargetComments:

    def test__write_comms_in_evaluate(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        t_ref = Time(60000.0, format="mjd")

        selector.evaluate_targets(basic_scoring, t_ref=t_ref)

        selector.write_target_comments()

        expected_comms_names = [f"T{n}_comments.txt" for n in range(101, 107)]
        existing_comms_names = [f.name for f in selector.comments_path.glob("*")]
        assert set(existing_comms_names) == set(expected_comms_names)

        exp_T101_comms_path = selector.comments_path / "T101_comments.txt"
        assert exp_T101_comms_path.exists()
        with open(exp_T101_comms_path) as f_comms:
            T101_comms = " ".join(line for line in f_comms.readlines())
        assert "mag_factor=2.0" in T101_comms
        assert "obs_factor=0.5 fixed (lasilla)"
        assert "obs_factor=0.5 fixed (ucph)"

    def test__write_comments_reject(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        t_ref = Time(60026.0, format="mjd")

        selector.evaluate_targets(basic_scoring, t_ref=t_ref)
        rejected = selector.remove_rejected_targets(t_ref=t_ref)

        assert set([t.target_id for t in rejected]) == set(["T101", "T102"])

        expected_comms_names = ["T101_comments.txt", "T102_comments.txt"]
        rejected_comms_names = [
            f.name for f in selector.rejected_targets_path.glob("*")
        ]
        normal_comms_names = [f.name for f in selector.comments_path.glob("*")]
        assert set(rejected_comms_names) == set(expected_comms_names)
        assert set(normal_comms_names) == set(expected_comms_names)

        exp_T101_rej_comms_path = selector.rejected_targets_path / "T101_comments.txt"
        assert exp_T101_rej_comms_path.exists()
        with open(exp_T101_rej_comms_path) as f:
            comms = " ".join(line for line in f.readlines())
        assert "target T101 is old" in comms


def return_blank_figure(target, t_ref=None):
    fig = plt.Figure(figsize=(2, 2))
    ax = fig.add_axes(111)
    return fig


class Test__Plotting:

    def test__plot_target_lightcurves(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        t_ref = Time(60015.0, format="mjd")

        for target_id in ["T103", "T104", "T105", "T106"]:
            selector.target_lookup.pop(target_id)
            # Don't make more plots than necessary.

        selector.plot_target_lightcurves()

        T101 = selector.target_lookup["T101"]
        exp_T101_fig_path = (
            selector.base_path / "projects/default/scratch/lc/T101_lc.png"
        )
        exp_T101_fig_path.exists()

    def test__plot_lc_accepts_function(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        t_ref = Time(60015.0, format="mjd")

        selector.plot_target_lightcurves(
            plotting_function=return_blank_figure, t_ref=t_ref
        )

        T101 = selector.target_lookup["T101"]
        exp_T101_fig_path = (
            selector.base_path / "projects/default/scratch/lc/T101_lc.png"
        )
        assert exp_T101_fig_path.exists()

    def test__lc_lazy_plotting_respected(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        t_lookup = selector.target_lookup

        assert np.isclose(selector.selector_parameters["plotting_interval"], 0.25)

        for target_id in ["T103", "T104", "T105", "T106"]:
            t_lookup.pop(target_id)
            # Don't make more plots than necessary.

        T101 = selector.target_lookup["T101"]
        T102 = selector.target_lookup["T102"]

        # ======
        # Make plots for the first time.
        # ======
        t_ref = Time.now()
        plotted, skipped = selector.plot_target_lightcurves(
            plotting_function=return_blank_figure, t_ref=t_ref, lazy=True
        )

        assert set(plotted) == set(["T101", "T102"])
        assert set(skipped) == set()

        T101_exp_fig_path = selector.lc_scratch_path / f"T101_lc.png"
        assert T101_exp_fig_path.exists()
        # assert np.isclose(T101.lc_fig_t_ref.mjd, t_ref.mjd, rtol=1e-8)
        T102_exp_fig_path = selector.lc_scratch_path / f"T102_lc.png"
        assert T102_exp_fig_path.exists()
        # assert np.isclose(T102.lc_fig_t_ref.mjd, t_ref.mjd, rtol=1e-8)

        # ======
        # Make sure updated targets have plots recreated.
        # ======
        T101.updated = True
        t_f1 = Time(t_ref.mjd + 0.2, format="mjd")
        plotted, skipped = selector.plot_target_lightcurves(
            plotting_function=return_blank_figure, t_ref=t_f1, lazy=True
        )
        assert set(plotted) == set(["T101"])
        assert set(skipped) == set(["T102"])

        # Check that T101 was updated...
        # assert np.isclose(T101.lc_fig_t_ref.mjd, t_f1.mjd, rtol=1e-8)
        # assert not np.isclose(T101.lc_fig_t_ref.mjd, t_ref.mjd, rtol=1e-8)
        # The second line checks rtol small enough! dt/t = 0.2/60000.0 is small!

        # But T102 wasn't
        # assert not np.isclose(T102.lc_fig_t_ref.mjd, t_f1.mjd, rtol=1e-8)
        # assert np.isclose(T102.lc_fig_t_ref.mjd, t_ref.mjd, rtol=1e-8)

    def test__get_output_plots_path(self, selector_with_targets):
        selector = selector_with_targets

        pl_dir = selector.get_output_plots_path("test_obs")
        assert pl_dir == selector.base_path / "projects/default/outputs/plots/test_obs"
        assert pl_dir.exists()

    def test__get_ranked_list_path(self, selector_with_targets):
        selector = selector_with_targets

        lists_path = selector.get_ranked_list_path("no_obs")

        exp_list_path = (
            selector.base_path / "projects/default/outputs/ranked_lists/no_obs.csv"
        )
        assert lists_path == exp_list_path
        assert lists_path.parent.exists()


def mod_basic_scoring(target, t_ref):
    exclude = False

    detections = target.target_data["ztf"].detections
    last_mag = detections["mag"].iloc[-1]
    mag_factor = 20 - last_mag
    if last_mag > 18.9:
        exclude = True  # Will exlude T106

    delta_t = t_ref.mjd - detections["mjd"].iloc[-1]
    if delta_t > 11:
        exclude = True  # will exclude T101

    score = target.base_score * mag_factor
    if exclude:
        score = -1.0
    return score


class Test__Ranking:

    def test__rank_for_obs(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        t_lookup = selector.target_lookup
        t_ref = Time(60015.0, format="mjd")  # Note this is LATER than before!

        selector.evaluate_targets(
            mod_basic_scoring,
            observatory_scoring_function=basic_obs_scoring,
            t_ref=t_ref,
        )

        ranked_list = selector.build_ranked_target_list_at_observatory(
            None, plots=False, t_ref=t_ref
        )

        assert set(ranked_list.columns) == set("target_id score ra dec ranking".split())
        assert len(ranked_list) == 4
        assert set(ranked_list["target_id"]) == set(["T102", "T103", "T104", "T105"])

        assert ranked_list.iloc[0].target_id == "T102"
        assert ranked_list.iloc[0].ranking == 1
        assert np.isclose(ranked_list.iloc[0].score, 1.8)
        assert ranked_list.iloc[1].target_id == "T103"
        assert ranked_list.iloc[1].ranking == 2
        assert np.isclose(ranked_list.iloc[1].score, 1.6)
        assert ranked_list.iloc[2].target_id == "T104"
        assert ranked_list.iloc[2].ranking == 3
        assert np.isclose(ranked_list.iloc[2].score, 1.4)
        assert ranked_list.iloc[3].target_id == "T105"
        assert ranked_list.iloc[3].ranking == 4
        assert np.isclose(ranked_list.iloc[3].score, 1.2)

        # Check rank-history correctly recorded
        T101 = t_lookup["T101"]
        assert len(T101.rank_history["no_observatory"]) == 1
        assert T101.rank_history["no_observatory"][0][0] == 9999  # default unranked
        assert np.isclose(T101.get_last_score(), -1.0)
        assert np.isclose(T101.rank_history["no_observatory"][0][1].mjd, 60015.0)

        T102 = t_lookup["T102"]
        assert len(T102.rank_history["no_observatory"]) == 1  # default unranked
        assert T102.rank_history["no_observatory"][0][0] == 1
        assert np.isclose(T102.get_last_score(), 1.80)
        assert np.isclose(T102.rank_history["no_observatory"][0][1].mjd, 60015.0)

        T106 = t_lookup["T106"]
        assert len(T106.rank_history["no_observatory"]) == 1  # default unranked
        assert T106.rank_history["no_observatory"][0][0] == 9999
        assert np.isclose(T106.get_last_score(), -1.0)
        assert np.isclose(T106.rank_history["no_observatory"][0][1].mjd, 60015.0)

        exp_ranked_list_path = selector.outputs_path / "ranked_lists/no_observatory.csv"
        assert exp_ranked_list_path.exists()

        result = pd.read_csv(exp_ranked_list_path)
        assert len(result) == 4
        assert set(result.columns) == set("target_id score ra dec ranking".split())

    def test__rank_all_obs(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        t_lookup = selector.target_lookup
        t_ref = Time(60015.0, format="mjd")

        selector.evaluate_targets(
            mod_basic_scoring,
            observatory_scoring_function=basic_obs_scoring,
            t_ref=t_ref,
        )

        selector.build_ranked_target_lists(t_ref=t_ref, plots=False)

        no_obs_list_file = selector.outputs_path / "ranked_lists/no_observatory.csv"
        lasilla_list_file = selector.outputs_path / "ranked_lists/lasilla.csv"
        ucph_list_file = selector.outputs_path / "ranked_lists/ucph.csv"

        assert no_obs_list_file.exists()
        assert lasilla_list_file.exists()
        assert ucph_list_file.exists()

        T101 = selector.target_lookup["T101"]
        assert T101.rank_history["no_observatory"][0][0] == 9999  # Default unranked
        assert T101.rank_history["lasilla"][0][0] == 9999
        assert T101.rank_history["ucph"][0][0] == 9999

        T102 = selector.target_lookup["T102"]
        assert T102.rank_history["no_observatory"][0][0] == 1
        assert T102.rank_history["lasilla"][0][0] == 1
        assert T102.rank_history["ucph"][0][0] == 1

        T105 = selector.target_lookup["T105"]
        assert T105.rank_history["no_observatory"][0][0] == 4
        assert T105.rank_history["lasilla"][0][0] == 4
        assert T105.rank_history["ucph"][0][0] == 4

        T106 = selector.target_lookup["T106"]
        assert T106.rank_history["no_observatory"][0][0] == 9999  # Default unranked
        assert T106.rank_history["lasilla"][0][0] == 9999
        assert T106.rank_history["ucph"][0][0] == 9999

        for target_id in ["T101", "T102", "T103", "T104", "T105", "T106"]:
            target = t_lookup[target_id]
            for obs in ["no_observatory", "lasilla", "ucph"]:
                assert len(target.rank_history[obs]) == 1
                assert np.isclose(target.rank_history[obs][0][1].mjd, 60015.0)

    def test__rank_all_gets_plots(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        t_ref = Time(60015.0, format="mjd")

        # Don't make more plots than necessary.
        selector.target_lookup.pop("T104")
        selector.target_lookup.pop("T105")
        selector.observatories.pop("ucph")

        selector.evaluate_targets(
            mod_basic_scoring,
            observatory_scoring_function=basic_obs_scoring,
            t_ref=t_ref,
        )
        selector.plot_target_lightcurves(
            plotting_function=return_blank_figure, t_ref=t_ref
        )
        selector.plot_target_visibilities(t_ref=t_ref)

        selector.build_ranked_target_lists(t_ref=t_ref, plots=True)

        no_obs_list_file = selector.outputs_path / "ranked_lists/no_observatory.csv"
        lasilla_list_file = selector.outputs_path / "ranked_lists/lasilla.csv"

        assert no_obs_list_file.exists()
        assert lasilla_list_file.exists()

        plots_path = selector.outputs_path / "plots"
        assert plots_path.exists()

        no_obs_path = plots_path / "no_observatory"
        assert no_obs_path.exists()
        lc_fig_01 = no_obs_path / "001_T102_lc.png"
        lc_fig_02 = no_obs_path / "002_T103_lc.png"
        # lc_fig_03 = no_obs_path / "003_T104_lc.png"
        # lc_fig_04 = no_obs_path / "004_T105_lc.png"
        exp_no_obs_files = [lc_fig_01, lc_fig_02]
        for pl_f in exp_no_obs_files:
            assert pl_f.exists()
        assert set(no_obs_path.glob("*lc.png")) == set(exp_no_obs_files)

        lasilla_path = plots_path / "lasilla"
        lc_fig_01 = lasilla_path / "001_T102_lc.png"
        lc_fig_02 = lasilla_path / "002_T103_lc.png"
        # lc_fig_03 = lasilla_path / "003_T104_lc.png"
        # lc_fig_04 = lasilla_path / "004_T105_lc.png"
        exp_lasilla_files = [lc_fig_01, lc_fig_02]
        for pl_f in exp_lasilla_files:
            assert pl_f.exists()
        assert set(lasilla_path.glob("*lc.png")) == set(exp_lasilla_files)
        vis_fig_01 = lasilla_path / "001_T102_vis.png"
        vis_fig_02 = lasilla_path / "002_T103_vis.png"
        # vis_fig_03 = lasilla_path / "003_T104_vis.png"
        # vis_fig_04 = lasilla_path / "004_T105_vis.png"
        exp_lasilla_vis_files = [vis_fig_01, vis_fig_02]
        for pl_f in exp_lasilla_vis_files:
            assert pl_f.exists()
        assert set(lasilla_path.glob("*vis.png")) == set(exp_lasilla_vis_files)


class Test__ResettingTasks:

    def test__reset_updated_targets(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets

        selector.target_lookup["T101"].updated = True
        selector.target_lookup["T101"].send_updates = True
        selector.target_lookup["T101"].update_messages = ["some update"]

        selector.reset_updated_targets()

        assert selector.target_lookup["T101"].updated is False
        assert selector.target_lookup["T101"].send_updates is False
        assert len(selector.target_lookup["T101"].update_messages) == 0

    def test__clear_output_plots(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets

        no_obs_figs_path = selector.get_output_plots_path("no_observatory")
        lasilla_figs_path = selector.get_output_plots_path("lasilla")

        fig = plt.Figure()
        for ii in range(4):
            fig.savefig(no_obs_figs_path / f"test_{ii}_lc.png")
            fig.savefig(lasilla_figs_path / f"test_{ii}_lc.png")
            fig.savefig(lasilla_figs_path / f"test_{ii}_vis.png")

        assert len([f for f in no_obs_figs_path.glob("*.png")]) == 4
        assert len([f for f in lasilla_figs_path.glob("*.png")]) == 8  # w/ vis figs

        # Clear everything!
        selector.clear_output_plots()

        assert len([f for f in no_obs_figs_path.glob("*.png")]) == 0
        assert len([f for f in lasilla_figs_path.glob("*.png")]) == 0  # w/ vis figs
        # Everything is gone!


class Test__MessagingTasks:
    def test__perform_messaging_tasks(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        t_lookup = selector.target_lookup
        t_ref = Time(60000.0, format="jd")

        # T101 has messages and a score
        t_lookup["T101"].update_messages = ["an update"]
        t_lookup["T101"].score_history["no_observatory"].append((10.0, t_ref))
        t_lookup["T101"].updated = True
        # T102 has messages and score but not updated
        t_lookup["T102"].update_messages = ["an update"]
        t_lookup["T102"].score_history["no_observatory"].append((10.0, t_ref))
        t_lookup["T102"].updated = False
        # T103 has score and updated but no messages
        t_lookup["T103"].update_messages = []
        t_lookup["T103"].score_history["no_observatory"].append((10.0, t_ref))
        t_lookup["T103"].updated = True
        # T104 has messages and updated but exclude score
        t_lookup["T104"].update_messages = ["an update"]
        t_lookup["T104"].score_history["no_observatory"].append((-1.0, t_ref))
        t_lookup["T104"].updated = True
        # T105 has messages and updated but no score
        t_lookup["T105"].update_messages = ["an update"]
        pass
        t_lookup["T105"].updated = True
        # T106 has all, and two messages
        t_lookup["T106"].update_messages = ["an update", "another_update"]
        t_lookup["T106"].score_history["no_observatory"].append((10.0, t_ref))
        t_lookup["T106"].updated = True

        sent, skipped, no_updates = selector.perform_messaging_tasks()

        assert set(sent) == set(["T101", "T106"])
        assert set(skipped) == set(["T102", "T104", "T105"])
        assert set(no_updates) == set(["T103"])


class Test__ExisitingTargets:

    def test__write_exisitng_targets(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        t_ref = Time(60000.0, format="mjd")

        selector.write_existing_target_list(t_ref=t_ref)

        exp_targets_path = (
            selector.project_path / "existing_targets/recover_230225_000000.json"
        )
        assert exp_targets_path.exists()

        with open(exp_targets_path) as f:
            result = json.load(f)

        assert isinstance(result, dict)

        assert set(result.keys()) == set(
            ["T101", "T102", "T103", "T104", "T105", "T106"]
        )

        assert np.isclose(result["T101"]["ra"], 15.0)
        assert np.isclose(result["T102"]["ra"], 30.0)
        assert np.isclose(result["T103"]["ra"], 45.0)
        assert np.isclose(result["T104"]["ra"], 60.0)
        assert np.isclose(result["T105"]["ra"], 75.0)
        assert np.isclose(result["T106"]["ra"], 90.0)

        assert np.allclose([data_ii["dec"] for t_id, data_ii in result.items()], 30.0)

    def test__no_targets_to_write(self, selector):
        assert len(selector.target_lookup) == 0
        t_ref = Time(60000.0, format="mjd")

        selector.write_existing_target_list(t_ref=t_ref)

        existing_names = [f.name for f in selector.existing_targets_path.glob("*.json")]
        assert set(existing_names) == set()  # No targets written!

    def test__write_removes_old_files(
        self, selector_with_targets: TargetSelector, extra_targets
    ):
        selector = selector_with_targets
        selector.selector_parameters["retained_recovery_files"] = 2

        existing_names = [f.name for f in selector.existing_targets_path.glob("*.json")]
        assert set(existing_names) == set()

        t1 = Time(60000.0, format="mjd")
        t2 = Time(60001.0, format="mjd")
        t3 = Time(60002.0, format="mjd")

        selector.write_existing_target_list(t_ref=t1)

        existing_names = [f.name for f in selector.existing_targets_path.glob("*.json")]
        expected_names = ["recover_230225_000000.json"]
        assert set(existing_names) == set(expected_names)

        selector.write_existing_target_list(t_ref=t2)

        existing_names = [f.name for f in selector.existing_targets_path.glob("*.json")]
        expected_names = ["recover_230225_000000.json", "recover_230226_000000.json"]
        assert set(existing_names) == set(expected_names)

        for target in extra_targets:
            target.base_score = 1000.0
            selector.add_target(target)
        selector.write_existing_target_list(t_ref=t3)

        existing_names = [f.name for f in selector.existing_targets_path.glob("*.json")]
        expected_names = ["recover_230226_000000.json", "recover_230227_000000.json"]
        # 25th Feb file has been removed!
        assert set(existing_names) == set(expected_names)

        with open(selector.existing_targets_path / "recover_230227_000000.json") as f:
            result = json.load(f)

        assert set(result["T101"].keys()) == set(
            ["target_id", "ra", "dec", "base_score", "alt_ids"]
        )

        exp_target_ids = [f"T{n}" for n in range(101, 111)]
        assert set(result.keys()) == set(exp_target_ids)

        assert np.isclose(result["T106"]["base_score"], 1.0)
        assert np.isclose(result["T107"]["base_score"], 1000.0)
        assert np.isclose(result["T108"]["base_score"], 1000.0)

    def test__read_latest_existing_target_file(self, selector: TargetSelector):
        df = pd.DataFrame(
            [
                ("T201", 90.0, 45.0, 50.0),
                ("T202", 180.0, 45.0, 100.0),
                ("T203", 270.0, 45.0, 200.0),
            ],
            columns="target_id ra dec base_score".split(),
        )
        df.set_index("target_id", inplace=True, drop=False)

        early_df = df.iloc[:-1]
        assert set(early_df["target_id"]) == set(["T201", "T202"])

        with open(selector.existing_targets_path / "recover_230225.json", "w+") as f:
            json.dump(early_df.to_dict("index"), f)
        with open(selector.existing_targets_path / "recover_230226.json", "w+") as f:
            json.dump(df.to_dict("index"), f)

        selector.recover_existing_targets()

        assert set(selector.target_lookup.keys()) == set(["T201", "T202", "T203"])
        assert np.isclose(selector.target_lookup["T201"].base_score, 50.0)
        assert np.isclose(selector.target_lookup["T201"].ra, 90.0)
        assert np.isclose(selector.target_lookup["T201"].dec, 45.0)
        assert np.isclose(selector.target_lookup["T202"].base_score, 100.0)
        assert np.isclose(selector.target_lookup["T203"].base_score, 200.0)

    def test__read_existing_targets_named_file(self, selector: TargetSelector):
        df = pd.DataFrame(
            [
                ("T201", 90.0, 45.0, 50.0),
                ("T202", 180.0, 45.0, 100.0),
                ("T203", 270.0, 45.0, 200.0),
            ],
            columns="target_id ra dec base_score".split(),
        )
        df.set_index("target_id", inplace=True, drop=False)

        early_df = df.iloc[:-1]
        assert set(early_df["target_id"]) == set(["T201", "T202"])
        with open(selector.existing_targets_path / "recover_230225.json", "w+") as f:
            json.dump(early_df.to_dict("index"), f)
        with open(selector.existing_targets_path / "recover_230226.json", "w+") as f:
            json.dump(df.to_dict("index"), f)

        recovery_file = selector.existing_targets_path / "recover_230225.json"

        selector.recover_existing_targets(existing_targets_file=recovery_file)

        assert set(selector.target_lookup.keys()) == set(["T201", "T202"])

    def test__write_score_histories(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets

        t1 = Time(60000.0, format="mjd")
        t2 = Time(60001.0, format="mjd")
        t3 = Time(60002.0, format="mjd")

        lasilla = selector.observatories["lasilla"]

        T101 = selector.target_lookup["T101"]
        T101.update_score_history(75.0, None, t_ref=t1)  # Def check sort by mjd
        T101.update_score_history(15.0, None, t_ref=t3)  # Wrong order!
        T101.update_score_history(20.0, None, t_ref=t2)
        T101.update_score_history(10.0, "lasilla", t_ref=t1)  # str
        T101.update_score_history(20.0, lasilla, t_ref=t2)  # astroplan.Observer
        T101.update_score_history(0.1, "ucph", t_ref=t3)

        selector.write_score_histories()

        exp_history_path = selector.existing_targets_path / "score_history/T101.csv"
        assert exp_history_path.exists()

        result = pd.read_csv(exp_history_path)
        assert set(result.columns) == set(["observatory", "mjd", "score"])
        assert result.iloc[0].observatory == "lasilla"
        assert result.iloc[1].observatory == "lasilla"
        assert result.iloc[2].observatory == "no_observatory"
        assert np.isclose(result.iloc[2].score, 75.0)
        assert result.iloc[3].observatory == "no_observatory"
        assert np.isclose(result.iloc[3].score, 20.0)
        assert result.iloc[4].observatory == "no_observatory"
        assert np.isclose(result.iloc[4].score, 15.0)
        assert result.iloc[5].observatory == "ucph"
        assert np.isclose(result.iloc[5].score, 0.1)

    def test__write_rank_history(self, selector_with_targets):
        selector = selector_with_targets

        t1 = Time(60000.0, format="mjd")
        t2 = Time(60001.0, format="mjd")
        t3 = Time(60002.0, format="mjd")

        lasilla = selector.observatories["lasilla"]

        T101 = selector.target_lookup["T101"]
        T101.update_rank_history(4, None, t_ref=t1)  # check sort by mjd!
        T101.update_rank_history(1, None, t_ref=t3)  # Wrong order!
        T101.update_rank_history(10, None, t_ref=t2)
        T101.update_rank_history(5, "lasilla", t_ref=t1)  # str
        T101.update_rank_history(6, lasilla, t_ref=t2)  # astroplan.Observer
        T101.update_rank_history(4, "ucph", t_ref=t3)

        selector.write_rank_histories()

        exp_history_path = selector.existing_targets_path / "rank_history/T101.csv"
        assert exp_history_path.exists()

        result = pd.read_csv(exp_history_path)
        assert set(result.columns) == set(["observatory", "mjd", "ranking"])
        assert result.iloc[0].observatory == "lasilla"
        assert result.iloc[1].observatory == "lasilla"
        assert result.iloc[2].observatory == "no_observatory"
        assert result.iloc[2].ranking == 4
        assert result.iloc[3].observatory == "no_observatory"
        assert result.iloc[3].ranking == 10
        assert result.iloc[4].observatory == "no_observatory"
        assert result.iloc[4].ranking == 1  # they were sorted by mjd NOT ranking.

    def test__recover_score_history(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets

        df = pd.DataFrame(
            [
                ("no_observatory", 60000.0, 1.0),
                ("no_observatory", 60001.0, 10.0),
                ("no_observatory", 60002.0, 8.0),
                ("no_observatory", 60003.0, 3.0),
                ("lasilla", 60002.0, -1.0),
                ("lasilla", 60003.0, 3.0),
                ("lasilla", 60004.0, 6.0),
            ],
            columns="observatory mjd score".split(),
        )

        hist_path = selector.get_score_history_file("T101")
        df.to_csv(hist_path, index=False)

        T101 = selector.target_lookup["T101"]

        selector.recover_score_history(T101)

        assert set(T101.score_history.keys()) == set(["no_observatory", "lasilla"])
        assert len(T101.score_history["no_observatory"]) == 4
        assert np.isclose(T101.score_history["no_observatory"][0][0], 1.0)
        assert isinstance(T101.score_history["no_observatory"][0][1], Time)
        assert np.isclose(T101.score_history["no_observatory"][0][1].mjd, 60000.0)
        assert np.isclose(T101.score_history["no_observatory"][1][0], 10.0)
        assert isinstance(T101.score_history["no_observatory"][1][1], Time)
        assert np.isclose(T101.score_history["no_observatory"][1][1].mjd, 60001.0)
        assert np.isclose(T101.score_history["no_observatory"][2][0], 8.0)
        assert isinstance(T101.score_history["no_observatory"][2][1], Time)
        assert np.isclose(T101.score_history["no_observatory"][2][1].mjd, 60002.0)
        assert np.isclose(T101.score_history["no_observatory"][3][0], 3.0)
        assert isinstance(T101.score_history["no_observatory"][3][1], Time)
        assert np.isclose(T101.score_history["no_observatory"][3][1].mjd, 60003.0)
        assert len(T101.score_history["lasilla"]) == 3
        assert np.isclose(T101.score_history["lasilla"][0][0], -1.0)
        assert isinstance(T101.score_history["lasilla"][0][1], Time)
        assert np.isclose(T101.score_history["lasilla"][0][1].mjd, 60002.0)
        assert np.isclose(T101.score_history["lasilla"][1][0], 3.0)
        assert isinstance(T101.score_history["lasilla"][1][1], Time)
        assert np.isclose(T101.score_history["lasilla"][1][1].mjd, 60003.0)
        assert np.isclose(T101.score_history["lasilla"][2][0], 6.0)
        assert isinstance(T101.score_history["lasilla"][2][1], Time)
        assert np.isclose(T101.score_history["lasilla"][2][1].mjd, 60004.0)


class NewQueryManager:
    name = "new_qm"

    def __init__(self):
        self.tasks_performed = False

    def perform_all_tasks(self, startup=False, t_ref=None):

        self.startup = startup
        self.tasks_performed = True


class Test__LoopingFunctions:

    def test__perform_iteration(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets
        selector.observatories.pop("ucph")
        selector.query_managers["new_qm"] = NewQueryManager()
        t_ref = Time(60015.0, format="mjd")  # MJD = 23-03-12

        T101 = selector.target_lookup["T101"]
        T102 = selector.target_lookup["T102"]
        T103 = selector.target_lookup["T103"]
        T104 = selector.target_lookup["T104"]
        T105 = selector.target_lookup["T105"]
        T106 = selector.target_lookup["T106"]

        T101.updated = True
        T101.models["basic_modeler"] = None  # Updated but previous model failed.

        T102.updated = False
        T102.models["basic_modeler"] = None  # Not updated but previous model failed.

        T103.updated = True
        # No models at all - should attempt modeling here regardless of updated.

        T104.updated = False  # Same for T105, T106
        # Should have models

        for target_id, target in selector.target_lookup.items():
            assert target.compiled_lightcurve is None

        # ACTION
        selector.perform_iteration(
            scoring_function=mod_basic_scoring,
            observatory_scoring_function=basic_obs_scoring,
            modeling_function=basic_modeler,
            lightcurve_compiler=basic_lc_compiler,
            t_ref=t_ref,
        )
        assert selector.query_managers["new_qm"].tasks_performed is True
        assert selector.query_managers["new_qm"].startup is False

        for target_id, target in selector.target_lookup.items():
            # Scored once by pre-check, and then once in evaluate
            assert len(target.score_history["no_observatory"]) == 2
            assert len(target.score_history["lasilla"]) == 1

            assert target.updated is False
            assert len(target.compiled_lightcurve) == 3

        for target in [T101, T103, T104, T105, T106]:
            assert isinstance(target.models["basic_modeler"], BasicModel)
        assert T102.models["basic_modeler"] is None

        lists_path = selector.outputs_path / "ranked_lists"
        assert lists_path.exists()
        exp_lists = [lists_path / "no_observatory.csv", lists_path / "lasilla.csv"]
        assert set([f for f in lists_path.glob("*")]) == set(exp_lists)

        no_obs_plots_path = selector.outputs_path / "plots/no_observatory"
        assert no_obs_plots_path.exists()
        assert (
            len([f for f in no_obs_plots_path.glob("*.png")]) == 4
        )  # T101, T106 are excluded
        lasilla_plots_path = selector.outputs_path / "plots/lasilla"
        assert len([f for f in lasilla_plots_path.glob("*.png")]) == 8
        # 8 because 4 lc fig and 4 observing charts (whereas none for no_obs!)

        exp_recovery_path = (
            selector.existing_targets_path / "recover_230312_000000.json"
        )
        assert exp_recovery_path.exists()  # MJD = 60015.

        # score_hist_path = selector.existing_targets_path / "score_history"
        # assert score_hist_path.exists()
        # exp_score_hist = [
        #     score_hist_path / f"{target_id}.csv"
        #     for target_id in "T101 T102 T103 T104 T105 T106".split()
        # ]
        # assert set([f for f in score_hist_path.glob("*")]) == set(exp_score_hist)

        # rank_hist_path = selector.existing_targets_path / "rank_history"
        # assert rank_hist_path.exists()
        # exp_rank_hist = [
        #     rank_hist_path / f"{target_id}.csv"
        #     for target_id in "T101 T102 T103 T104 T105 T106".split()
        # ]
        # assert set([f for f in rank_hist_path.glob("*")]) == set(exp_rank_hist)

    def test__exc_on_bad_skip_tasks(self, selector_with_targets: TargetSelector):
        selector = selector_with_targets

        skip_tasks = ["bad_task"]

        with pytest.raises(ValueError):
            with pytest.warns(UnexpectedKeysWarning):
                selector.perform_iteration(basic_scoring, skip_tasks=skip_tasks)
