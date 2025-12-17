import pytest

import numpy as np

from astropy.time import Time

from aas2rto.scoring.example_functions import (
    constant_score,
    base_score,
    latest_flux,
)
from aas2rto.target import Target
from aas2rto.target_data import TargetData


class Test__ExampleScoringFunctions:
    def test__basic(self, basic_target: Target, t_fixed: Time):
        # Act
        score = constant_score(basic_target, t_fixed)

        # Assert
        assert np.isclose(score, 1.0)

    def test__base_score(self, basic_target: Target, t_fixed: Time):
        # Arrange
        basic_target.base_score = 20.0

        # Act
        score, comms = base_score(basic_target, t_fixed)

        # Assert
        assert np.isclose(score, 20.0)


class Test__LatestFlux:
    def test__no_ztf_data(self, basic_target: Target, t_fixed: Time):
        # Arrange
        assert "fink_ztf" not in basic_target.target_data.keys()

        # Act
        score, comms = latest_flux(basic_target, t_fixed)

        # Assert
        assert np.isclose(score, -1.0)
        assert len(comms) == 1

    def test__with_ztf_data(
        self, basic_target: Target, ztf_td: TargetData, t_fixed: Time
    ):
        # Arrange
        basic_target.target_data["fink_ztf"] = ztf_td

        # Act
        score, comms = latest_flux(basic_target, t_fixed)

        # Assert
        exp_score = 3631.0 * 10 ** (-0.4 * 19.0) * 1e6  # flux in uJy
        assert np.isclose(score, exp_score)
