import gzip
import io
import pytest
from typing import NoReturn

import numpy as np

from astropy.io import fits


@pytest.fixture
def fink_alert_extras():
    return {
        "cdsxmatch": 1.0,
        "rf_snia_vs_nonia": 1.0,
        "snn_snia_vs_nonia": 1.0,
        "snn_sn_vs_all": 1.0,
        "mulens": 0.1,
        "roid": 0.1,
        "nalerthist": 4,
        "rf_kn_vs_nonkn": 0.3,
    }
