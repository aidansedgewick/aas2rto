import pytest
from pathlib import Path

import numpy as np

from astropy import units as u
from astropy.time import Time

from aas2rto.exc import UnexpectedKeysWarning
from aas2rto.modeling.modeling_manager import (
    ModelingManager,
    ModelingResult,
    modeling_wrapper,
    pool_modeling_wrapper,
)
from aas2rto.path_manager import PathManager
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup


@pytest.fixture
def basic_config():
    return {}


# ===== define some basic classes for testing modeling ===== #


class MockModel:
    def __init__(self, x=1.0):
        self.x = x


def mock_model(target: Target, t_ref: Time):
    if target.coord.dec.deg > 15.0:
        raise ValueError("dec is too high")
    return MockModel(x=target.base_score)


class MockModeling:
    __name__ = "class_mock_model"

    def __init__(self, m=1.0):
        self.m = m

    def __call__(self, target: Target, t_ref: Time):
        if target.coord.dec.deg > 15.0:
            raise ValueError("dec is too high")
        return MockModel(x=self.m * target.base_score)


class OtherMockModel:
    def __init__(self, y=1.0):
        self.y = y


def other_mock_model(target: Target, t_ref: Time):
    return OtherMockModel()


# ===== actual tests start here ===== #


class Test_ModelingResult:
    def test__init(self):
        # Act
        res = ModelingResult(
            target_id="T00", model=MockModel(), success=True, reason="good reason"
        )

        # Assert
        assert res.target_id == "T00"
        assert res.model, MockModel
        assert res.success
        assert res.reason == "good reason"

    def test__init_with_model_None(self):
        # Act
        res = ModelingResult(target_id="T00", model=None, success=False, reason="fail")

        # Assert
        assert res.target_id == "T00"
        assert res.model is None
        assert res.success is False
        assert res.reason == "fail"

    def test__init_with_missing_params(self):
        # Act
        with pytest.raises(TypeError):
            # This is effectively just testing behaviour of a @dataclass class...
            res = ModelingResult()


class Test__ModelMgrInit:
    def test__init(
        self, basic_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Act
        m_manager = ModelingManager(basic_config, tlookup, path_mgr)

        # Assert
        assert set(m_manager.config.keys()) == set(["lazy_modeling", "ncpu"])

    def test__bad_key_warns(
        self, basic_config: dict, tlookup: TargetLookup, path_mgr: PathManager
    ):
        # Arrange
        basic_config["bad_key"] = 100.0

        # Act
        with pytest.warns(UnexpectedKeysWarning):
            m_manager = ModelingManager(basic_config, tlookup, path_mgr)


class Test__BuildModels:
    def test__build_with_func(self, modeling_mgr: ModelingManager, t_fixed: Time):
        # Act
        modeling_mgr.build_target_models(mock_model, t_fixed)

        # Assert
        tl = modeling_mgr.target_lookup
        assert set(tl["T00"].models.keys()) == set(["mock_model"])
        assert isinstance(tl["T00"].models["mock_model"], MockModel)

        assert set(tl["T00"].models_t_ref.keys()) == set(["mock_model"])
        assert np.isclose(tl["T00"].models_t_ref["mock_model"] - 60000.0, 0.0)
        # Model should fail for dec > 15.0 - but still add key=None
        assert set(tl["T01"].models.keys()) == set(["mock_model"])
        assert tl["T01"].models["mock_model"] is None

    def test__build_with_inst(self, modeling_mgr: ModelingManager, t_fixed: Time):
        # Arrange
        inst_func = MockModeling(m=10.0)

        # Act
        modeling_mgr.build_target_models(inst_func, t_fixed)

        # Assert
        tl = modeling_mgr.target_lookup
        assert set(tl["T00"].models.keys()) == set(["class_mock_model"])
        assert isinstance(tl["T00"].models["class_mock_model"], MockModel)
        assert np.isclose(tl["T00"].models["class_mock_model"].x, 10.0)

        assert set(tl["T01"].models.keys()) == set(["class_mock_model"])
        assert tl["T01"].models["class_mock_model"] is None

    def test__build_two_funcs(self, modeling_mgr: ModelingManager, t_fixed: Time):
        # Arrange
        mfuncs = [mock_model, other_mock_model]

        # Act
        modeling_mgr.build_target_models(mfuncs, t_fixed)

        # Assert
        tl = modeling_mgr.target_lookup
        assert set(tl["T00"].models.keys()) == set(["mock_model", "other_mock_model"])
        assert isinstance(tl["T00"].models["mock_model"], MockModel)
        assert isinstance(tl["T00"].models["other_mock_model"], OtherMockModel)

        assert set(tl["T01"].models.keys()) == set(["mock_model", "other_mock_model"])
        assert tl["T01"].models["mock_model"] is None
        assert isinstance(tl["T01"].models["other_mock_model"], OtherMockModel)

    def test__inst_no_name_warns(self, modeling_mgr: ModelingManager, t_fixed: Time):
        # Arrange
        class ModelShouldWarn:
            def __call__(self, target: Target, t_ref: Time):
                return MockModel

        should_warn_inst = ModelShouldWarn()

        # Act
        with pytest.warns(UserWarning):
            modeling_mgr.build_target_models(should_warn_inst, t_fixed)

        # Assert
        tl = modeling_mgr.target_lookup
        assert set(tl["T00"].models.keys()) == set(["ModelShouldWarn"])

    def test__lazy_skips_non_updated(
        self, modeling_mgr: ModelingManager, t_fixed: Time
    ):
        # Arrange
        tl = modeling_mgr.target_lookup
        modeling_mgr.build_target_models(mock_model, t_fixed)
        tl["T00"].updated = False
        tl["T01"].updated = True
        t_later = t_fixed + 1.0 * u.day

        # Act
        modeling_mgr.build_target_models(mock_model, t_ref=t_later)

        # Assert
        assert np.isclose(tl["T00"].models_t_ref["mock_model"] - 60000.0, 0.0)
        assert np.isclose(tl["T01"].models_t_ref["mock_model"] - 60001.0, 0.0)

    def test__lazy_builds_missing(self, modeling_mgr: ModelingManager, t_fixed: Time):
        # Arrange
        tl = modeling_mgr.target_lookup
        modeling_mgr.build_target_models(mock_model, t_fixed)
        tl["T00"].updated = False
        tl["T01"].models = {}
        t_later = t_fixed + 1.0 * u.day

        # Act
        modeling_mgr.build_target_models(mock_model, t_ref=t_later)

        # Assert
        assert np.isclose(tl["T00"].models_t_ref["mock_model"] - 60000.0, 0.0)
        assert np.isclose(tl["T01"].models_t_ref["mock_model"] - 60001.0, 0.0)

    def test__non_lazy_rebuilds(self, modeling_mgr: ModelingManager, t_fixed: Time):
        # Arrange
        modeling_mgr.config["lazy_modeling"] = False
        tl = modeling_mgr.target_lookup
        modeling_mgr.build_target_models(mock_model, t_fixed)
        tl.reset_updated_targets()  # targets AREN'T updates, should be modeled anyway.
        t_later = t_fixed + 1.0 * u.day

        # Act
        modeling_mgr.build_target_models(mock_model, t_ref=t_later)

        # Assert
        assert np.isclose(tl["T00"].models_t_ref["mock_model"] - 60001.0, 0.0)
        assert np.isclose(tl["T01"].models_t_ref["mock_model"] - 60001.0, 0.0)


class Test__BuildModelsPool:

    def test__build_with_pool(self, modeling_mgr: ModelingManager, t_fixed: Time):
        # Arrange
        modeling_mgr.config["ncpu"] = 1

        # Act
        modeling_mgr.build_target_models(mock_model, t_fixed)

        # Assert
        tl = modeling_mgr.target_lookup
        assert set(tl["T00"].models.keys()) == set(["mock_model"])
