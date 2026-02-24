import pytest

import numpy as np

from astropy.time import Time


@pytest.fixture
def mock_ztf_cutouts():
    return dict(
        cutoutScience=dict(stampData=np.random.normal(0.0, 0.1, (10, 10))),
        cutoutDifference=dict(stampData=np.random.normal(0.0, 0.1, (10, 10))),
        cutoutTemplate=dict(stampData=np.random.normal(0.0, 0.1, (10, 10))),
    )


@pytest.fixture
def ztf_alert_base(fink_alert_extras: dict, mock_ztf_cutouts: dict, t_fixed: Time):
    return {
        "objectId": "ZTF00abc",  # the target_id
        "candid": 0,  # the alert_id
        "candidate": {
            "ra": 180.0,
            "dec": -30.0,
            "jd": t_fixed.jd,
            "magpsf": 20.0,
            "sigpsf": 0.1,
            "fid": 1,  # 1=ztf-g, 2=ztf-r
        },
        **mock_ztf_cutouts,
    }


@pytest.fixture
def mock_lsst_cutouts():
    return dict(
        cutoutScience=np.random.normal(0.0, 0.1, (10, 10)),
        cutoutDifference=np.random.normal(0.0, 0.1, (10, 10)),
        cutoutTemplate=np.random.normal(0.0, 0.1, (10, 10)),
    )


@pytest.fixture
def lsst_alert_base(mock_lsst_cutouts: dict, t_fixed: Time):
    return {
        "diaObject": {
            "diaObjectId": 1000,
            "ra": 180.0,
            "dec": -30.0,
            "nDiaSources": 2,  # Number of prev. detections
        },
        "diaSource": {
            "diaSourceId": 1234,  # the alert_id
            "ra": 180.0,
            "dec": -30.0,
            "psfFlux": 5000.0,  # approx mag 22.15
            "psfFluxErr": 500.0,
            "midpointMjdTai": t_fixed.mjd,
            **mock_lsst_cutouts,
        },
    }
