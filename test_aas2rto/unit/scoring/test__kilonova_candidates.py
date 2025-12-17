import pytest

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from aas2rto.scoring.kn_candidates import KilonovaDiscReject
from aas2rto.target import Target
from aas2rto.target_data import TargetData


@pytest.fixture
def kscore():
    return KilonovaDiscReject()


class Test__ScoreClassInit:

    def test__init(self):
        # Act
        sc = KilonovaDiscReject()

    def test__with_param(self):
        # Act
        sc = KilonovaDiscReject(min_b=10.0)

        # Assert
        assert np.isclose(sc.min_b, 10.0)


class Test__UsefulTarget:
    def test__useful_target(
        self,
        basic_target: Target,
        kscore: KilonovaDiscReject,
        t_fixed: Time,
        ztf_td: TargetData,
    ):
        # Arrange
        basic_target.target_data["fink_ztf"] = ztf_td  # has (l,b) = (276.0, +60.0)

        # Act
        score, comms = kscore(basic_target, t_fixed)

        # Assert
        assert np.isclose(score, 91.20, rtol=0.01)
        comm_str = " ".join(comms)
        assert "[uJy] from mag" in comm_str

    def test__reject_old_useful_target(
        self,
        basic_target: Target,
        kscore: KilonovaDiscReject,
        t_fixed: Time,
        ztf_td: TargetData,
    ):
        # Arrange
        basic_target.target_data["fink_ztf"] = ztf_td  # has (l,b) = (276.0, +60.0)
        t_future = t_fixed + 30 * u.day

        # Act
        score, comms = kscore(basic_target, t_future)

        # Assert
        assert not np.isfinite(score)
        comm_str = " ".join(comms)
        assert "REJECT: too long since first detection:" in comm_str

    def test__exclude_with_no_ztf_data(
        self,
        basic_target: Target,
        kscore: KilonovaDiscReject,
        t_fixed: Time,
    ):
        # Act
        score, comms = kscore(basic_target, t_fixed)

        # Assert
        assert np.isclose(score, -1.0)
        comm_str = " ".join(comms)
        assert "no data from" in comm_str


class Test__RejectTargets:
    def test__reject_bulge(
        self,
        kscore: KilonovaDiscReject,
        ztf_td: TargetData,
        t_fixed: Time,
    ):
        # Arrange
        coord = SkyCoord(frame="galactic", l=0.0, b=8.0, unit="deg")
        t = Target("Tbulge", coord)
        t.target_data["fink_ztf"] = ztf_td

        # Act
        score, comms = kscore(t, t_fixed)

        # Assert
        assert not np.isfinite(score)
        comm_str = " ".join(comms)
        assert "too close to MW centre" in comm_str
        assert "too close to MW disc" not in comm_str

    def test__reject_disk(
        self,
        kscore: KilonovaDiscReject,
        ztf_td: TargetData,
        t_fixed: Time,
    ):
        # Arrange
        coord = SkyCoord(frame="galactic", l=180.0, b=0.0, unit="deg")
        t = Target("Tbulge", coord)
        t.target_data["fink_ztf"] = ztf_td

        # Act
        score, comms = kscore(t, t_fixed)

        # Assert
        assert not np.isfinite(score)
        comm_str = " ".join(comms)
        assert "too close to MW centre" not in comm_str
        assert "too close to MW disc" in comm_str

    def test__reject_bulge_and_disk(
        self,
        kscore: KilonovaDiscReject,
        ztf_td: TargetData,
        t_fixed: Time,
    ):
        # Arrange
        coord = SkyCoord(frame="galactic", l=0.0, b=0.0, unit="deg")
        t = Target("Tbulge", coord)
        t.target_data["fink_ztf"] = ztf_td

        # Act
        score, comms = kscore(t, t_fixed)

        # Assert
        assert not np.isfinite(score)
        comm_str = " ".join(comms)
        assert "too close to MW centre" in comm_str
        assert "too close to MW disc" in comm_str
