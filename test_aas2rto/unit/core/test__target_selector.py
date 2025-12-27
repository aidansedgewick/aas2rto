import pytest
import yaml
from pathlib import Path
from typing import NoReturn

import numpy as np

import pandas as pd

from astropy.time import Time

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
from aas2rto.target_selector import TargetSelector


###===== Some fixtures =====###


@pytest.fixture
def empty_aas2rto_config(tmp_path: Path):
    return {"paths": {"base_path": tmp_path}}


@pytest.fixture
def tselector(empty_aas2rto_config: dict, basic_target: Target, ztf_td: TargetData):
    # Can't use premade 'tlookup', as tselector creates target_lookup in __init__

    ts = TargetSelector(empty_aas2rto_config)
    basic_target.target_data["fink_ztf"] = ztf_td
    ts.target_lookup.add_target(basic_target)
    return ts


###===== Some mock objects for tests =====###


def scoring_func(target: Target, t_ref: Time):
    return 1.0


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
    def test__opp_targets(self, target_config_example: dict, tselector: TargetSelector):
        # Arrange
        t_opp_path = tselector.path_manager.lookup["opp_targets"] / "test_target.yaml"
        with open(t_opp_path, "w+") as f:
            yaml.dump(target_config_example, f)

        # Act
        loaded, failed = tselector.load_targets_of_opportunity()

        # Assert
        assert set(tselector.target_lookup.keys()) == set(["T00", "T99"])
        assert not t_opp_path.exists()

        T99 = tselector.target_lookup["T99"]
        assert np.isclose(T99.base_score, 100.0)

        assert set([f.name for f in loaded]) == set(["test_target.yaml"])
        assert set(failed) == set()

    def test__malformed_config(
        self, target_config_example: dict, tselector: TargetSelector
    ):
        # Arrange
        target_config_example.pop("ra")
        t_opp_path = tselector.path_manager.lookup["opp_targets"] / "test_target.yaml"
        with open(t_opp_path, "w+") as f:
            yaml.dump(target_config_example, f)

        # Act
        loaded, failed = tselector.load_targets_of_opportunity()

        # Assert
        assert set(loaded) == set()
        assert set([f.name for f in failed]) == set(["test_target.yaml"])


class Test__CompileLightcurves:

    @pytest.fixture(autouse=True)
    def _remove_empty_tmp_dirs(self, remove_tmp_dirs: NoReturn):
        pass  # remove_tmp_dirs defined in unit/contfest.py, executed with autouse=True

    def test__compile_lcs(self, tselector: TargetSelector):
        # Arrange
        T00 = tselector.target_lookup["T00"]
        assert T00.compiled_lightcurve is None

        # Act
        compiled, failed = tselector.compile_target_lightcurves()

        # Assert
        assert isinstance(T00.compiled_lightcurve, pd.DataFrame)
        assert set(compiled) == set(["T00"])

    def test__lazy_skip_compiled_not_updated(self, tselector: TargetSelector):
        # Arrange
        T00 = tselector.target_lookup["T00"]
        T00.updated = False
        tselector.compile_target_lightcurves()

        # Act
        compiled, failed = tselector.compile_target_lightcurves()

        # Assert
        assert set(compiled) == set()

    def test__recompiled_updated(self, tselector: TargetSelector):
        # Arrange
        T00 = tselector.target_lookup["T00"]
        tselector.compile_target_lightcurves()
        T00.updated = True

        # Act
        compiled, failed = tselector.compile_target_lightcurves()

        # Assert
        assert set(compiled) == set(["T00"])


class Test__PerformIteration:
    def test__iter(self, tselector: TargetSelector):

        # Act
        tselector.perform_iteration(scoring_func)
        # Effectively checking for typos here...
