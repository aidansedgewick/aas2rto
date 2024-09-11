import pytest

import numpy as np

from aas2rto.query_managers.tns import TnsQueryManager
from aas2rto.utils import UnexpectedKeysWarning


@pytest.fixture
def query_parameters():
    return {"query_interval": 0.5}


@pytest.fixture
def tns_config():
    return {"user": "test", "uid": 1234}


@pytest.fixture
def target_lookup():
    return {}


class Test__QmInit:

    def test__tns_qm_init(self, tns_config, target_lookup, tmp_path):
        tns_qm = TnsQueryManager(tns_config, target_lookup, parent_path=tmp_path)

        assert isinstance(tns_qm.tns_config, dict)
        assert isinstance(tns_qm.target_lookup, dict)

        assert tns_qm.recent_coordinate_searches == set()
        assert tns_qm.query_results is None

        assert isinstance(tns_qm.tns_headers, dict)
        assert set(tns_qm.tns_headers.keys()) == set(["User-Agent"])

        assert isinstance(tns_qm.query_parameters, dict)
        assert np.isclose(tns_qm.query_parameters["query_interval"], 0.125)

        assert tns_qm.parent_path == tmp_path
        assert tns_qm.data_path == tmp_path / "tns"

        assert tns_qm.parameters_path == tmp_path / "tns/parameters"
        assert tns_qm.query_results_path == tmp_path / "tns/query_results"

    def test__tns_qm_bad_config_warns(self, tns_config, target_lookup, tmp_path):

        tns_config["bad_kw"] = "aaagh"

        with pytest.warns(UnexpectedKeysWarning):
            tns_qm = TnsQueryManager(tns_config, target_lookup, parent_path=tmp_path)

    def test__tns_fails_no_user(self, tns_config, target_lookup, tmp_path):
        tns_config.pop("user")
        assert "user" not in tns_config

        with pytest.raises(ValueError):
            tns_qm = TnsQueryManager(tns_config, target_lookup, parent_path=tmp_path)

    def test__tns_fails_no_uid(self, tns_config, target_lookup, tmp_path):
        tns_config.pop("uid")
        assert "uid" not in tns_config

        with pytest.raises(ValueError):
            tns_qm = TnsQueryManager(tns_config, target_lookup, parent_path=tmp_path)
