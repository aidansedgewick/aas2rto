import pytest
import yaml
from pathlib import Path
from typing import NoReturn

import numpy as np

import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.time import Time

import matplotlib.pyplot as plt

from aas2rto.exc import UnexpectedKeysError
from aas2rto.messaging.messaging_manager import MessagingManager
from aas2rto.modeling.modeling_manager import ModelingManager
from aas2rto.path_manager import PathManager
from aas2rto.observatory.observatory_manager import ObservatoryManager
from aas2rto.outputs.outputs_manager import OutputsManager
from aas2rto.query_managers.primary import PrimaryQueryManager
from aas2rto.recovery.recovery_manager import RecoveryManager
from aas2rto.scoring.scoring_manager import ScoringManager
from aas2rto.target import Target
from aas2rto.target_data import TargetData
from aas2rto.target_lookup import TargetLookup
from aas2rto.target_selector import TargetSelector


###===== Some mock objects for tests =====###


def scoring_func(target: Target, t_ref: Time):
    if target.target_id == "T_reject":
        return -np.inf
    return 1.0


class BasicModel:
    def __init__(self, value: float):
        self.value = value


def basic_model(target: Target, t_ref: Time):
    if target.compiled_lightcurve is not None:
        return BasicModel(target.compiled_lightcurve["mag"].iloc[-1])
    return None


class ExampleQueryManager:
    name = "example"

    def __init__(self, target_lookup: TargetLookup):
        self.target_lookup = target_lookup

    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        coord = SkyCoord(ra=15.0, dec=0.0, unit="deg")
        new_target = Target("T300", coord, source="other_source", t_ref=t_ref)
        self.target_lookup.add_target(new_target)
        print(self.target_lookup.keys())


###===== Some fixtures =====###


@pytest.fixture
def empty_aas2rto_config(tmp_path: Path):
    return {"paths": {"base_path": tmp_path}}


@pytest.fixture
def basic_tselector(
    empty_aas2rto_config: dict, basic_target: Target, ztf_td: TargetData
):
    # Can't use premade 'tlookup', as tselector creates target_lookup in __init__

    ts = TargetSelector(empty_aas2rto_config)

    basic_target.target_data["fink_ztf"] = ztf_td
    ts.target_lookup.add_target(basic_target)
    return ts


@pytest.fixture
def iter_tselector(
    empty_aas2rto_config: dict, basic_target: Target, ztf_td: TargetData
):
    # Can't use premade 'tlookup', as tselector creates target_lookup in __init__

    ts = TargetSelector(empty_aas2rto_config)

    qm = ExampleQueryManager(ts.target_lookup)
    ts.primary_query_manager.query_managers["example_qm"] = qm

    basic_target.target_data["fink_ztf"] = ztf_td
    ts.target_lookup.add_target(basic_target)

    alt_ids = {"other_src": "AAAA"}
    target_to_combine = Target("T_comb", basic_target.coord, alt_ids=alt_ids)
    ts.target_lookup.add_target(target_to_combine)

    target_to_reject = Target("T_reject", SkyCoord(ra=0.0, dec=0.0, unit="deg"))
    ts.target_lookup.add_target(target_to_reject)

    return ts


###===== Actual tests start here =====###


class Test__SelectorInit:

    def test__init_from_config(self, empty_aas2rto_config: dict):
        # Act
        selector = TargetSelector(empty_aas2rto_config)

        # Assert
        assert isinstance(selector, TargetSelector)

        assert isinstance(selector.path_manager, PathManager)
        assert isinstance(selector.observatory_manager, ObservatoryManager)
        assert isinstance(selector.primary_query_manager, PrimaryQueryManager)
        assert isinstance(selector.recovery_manager, RecoveryManager)
        assert isinstance(selector.modeling_manager, ModelingManager)
        assert isinstance(selector.messaging_manager, MessagingManager)

    def test__unexpected_config_section(self, empty_aas2rto_config: dict):
        # Arrange
        empty_aas2rto_config["unknown"] = {"blah": 100.0}

        # Act
        with pytest.raises(UnexpectedKeysError):
            selector = TargetSelector(empty_aas2rto_config)


class Test__FromConfig:
    def test__from_cfg_class_method(self, empty_aas2rto_config: dict, tmp_path: Path):
        # Arrange
        config_path = tmp_path / "config.yaml"
        empty_aas2rto_config["paths"]["base_path"] = str(tmp_path)
        # YAML can't dump type path - convert to str.
        with open(config_path, "w+") as f:
            yaml.dump(empty_aas2rto_config, f)

        # Arrange
        sel = TargetSelector.from_config(config_path)

        # Assert
        isinstance(sel, TargetSelector)


class Test__OppTargets:
    def test__opp_targets(
        self, target_config_example: dict, basic_tselector: TargetSelector
    ):
        # Arrange
        t_opp_path = (
            basic_tselector.path_manager.lookup["opp_targets"] / "test_target.yaml"
        )
        with open(t_opp_path, "w+") as f:
            yaml.dump(target_config_example, f)

        # Act
        loaded, failed = basic_tselector.load_targets_of_opportunity()

        # Assert
        assert set(basic_tselector.target_lookup.keys()) == set(["T00", "T99"])
        assert not t_opp_path.exists()

        T99 = basic_tselector.target_lookup["T99"]
        assert np.isclose(T99.base_score, 100.0)

        assert set([f.name for f in loaded]) == set(["test_target.yaml"])
        assert set(failed) == set()

    def test__malformed_config(
        self, target_config_example: dict, basic_tselector: TargetSelector
    ):
        # Arrange
        target_config_example.pop("ra")
        t_opp_path = (
            basic_tselector.path_manager.lookup["opp_targets"] / "test_target.yaml"
        )
        with open(t_opp_path, "w+") as f:
            yaml.dump(target_config_example, f)

        # Act
        loaded, failed = basic_tselector.load_targets_of_opportunity()

        # Assert
        assert set(loaded) == set()
        assert set([f.name for f in failed]) == set(["test_target.yaml"])


class Test__CompileLightcurves:

    @pytest.fixture(autouse=True)
    def _remove_empty_tmp_dirs(self, remove_tmp_dirs: NoReturn):
        pass  # remove_tmp_dirs defined in unit/contfest.py, executed with autouse=True

    def test__compile_lcs(self, basic_tselector: TargetSelector):
        # Arrange
        T00 = basic_tselector.target_lookup["T00"]
        assert T00.compiled_lightcurve is None

        # Act
        compiled, failed = basic_tselector.compile_target_lightcurves()

        # Assert
        assert isinstance(T00.compiled_lightcurve, pd.DataFrame)
        assert set(compiled) == set(["T00"])

    def test__lazy_skip_compiled_not_updated(self, basic_tselector: TargetSelector):
        # Arrange
        T00 = basic_tselector.target_lookup["T00"]
        T00.updated = False
        basic_tselector.compile_target_lightcurves()

        # Act
        compiled, failed = basic_tselector.compile_target_lightcurves()

        # Assert
        assert set(compiled) == set()

    def test__recompiled_updated(self, basic_tselector: TargetSelector):
        # Arrange
        T00 = basic_tselector.target_lookup["T00"]
        basic_tselector.compile_target_lightcurves()
        T00.updated = True

        # Act
        compiled, failed = basic_tselector.compile_target_lightcurves()

        # Assert
        assert set(compiled) == set(["T00"])


class Test__PerformIteration:
    def test__basic_iter(self, iter_tselector: TargetSelector, t_fixed: Time):
        # Act
        iter_tselector.perform_iteration(scoring_func)
        # Effectively checking for typos here...

    def test__complex_iter(self, iter_tselector: TargetSelector, t_fixed: Time):
        # Arrange
        # boring checks here
        exp_init_targets = ["T00", "T_reject", "T_comb"]
        assert set(iter_tselector.target_lookup.keys()) == set(exp_init_targets)
        T00 = iter_tselector.target_lookup["T00"]
        assert set(T00.models.keys()) == set()
        assert len(T00.science_score_history) == 0

        # Act
        iter_tselector.perform_iteration(
            scoring_func,
            modeling_function=basic_model,
            t_ref=t_fixed,
        )

        # Assert
        assert set(iter_tselector.target_lookup.keys()) == set(["T00", "T300"])
        # T_comb is combined into T00, T_reject is removed! T300 added by QM
        assert set(T00.models.keys()) == set(["basic_model"])
        assert isinstance(T00.models["basic_model"], BasicModel)

        assert len(T00.science_score_history) == 2
        # One from init check, one from "real" score
        assert T00.alt_ids["other_src"] == "AAAA"

    def test__invalid_skip_tasks(self, iter_tselector: TargetSelector):
        # Act
        with pytest.raises(UnexpectedKeysError):
            iter_tselector.perform_iteration(scoring_func, skip_tasks=["bad_task"])

    def test__skip_qms(self, iter_tselector: TargetSelector):
        # Act
        iter_tselector.perform_iteration(scoring_func, skip_tasks=["qm_tasks"])

        # Assert
        assert set(iter_tselector.target_lookup.keys()) == set(["T00"])

        # no new target T300 from QM...

    def test__skip_precheck(self, iter_tselector: TargetSelector):
        # Act
        iter_tselector.perform_iteration(scoring_func, skip_tasks=["pre_check"])
        T00 = iter_tselector.target_lookup["T00"]

        # Assert
        assert len(T00.science_score_history) == 1  # No pre-check


class Test__StartCommand:
    def test__start_command(self, iter_tselector: TargetSelector):
        # Act
        iter_tselector.start(scoring_func, iterations=0)
        # Basically a typo check - not a good test...
