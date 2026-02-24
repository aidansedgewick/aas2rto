import gzip
import io
import pytest
from typing import NoReturn

import numpy as np

from astropy.io import fits


@pytest.fixture
def patch_fink_readstamp(monkeypatch: pytest.MonkeyPatch) -> NoReturn:

    def mock_readstamp(data, return_type="array"):
        return data

    monkeypatch.setattr(
        "aas2rto.query_managers.fink.fink_ztf.readstamp", mock_readstamp
    )


@pytest.fixture
def fink_stamp_bytes() -> bytes:
    data = np.arange(100).reshape(10, 10)
    hdu = fits.PrimaryHDU(data=data)
    stream = io.BytesIO()
    hdu.writeto(stream)
    stream.seek(0)
    return gzip.compress(stream.read())


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
