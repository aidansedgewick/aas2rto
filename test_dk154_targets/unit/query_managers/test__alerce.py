import os
import pytest

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.time import Time

from dk154_targets.query_managers.alerce import (
    AlerceQueryManager,
    target_from_alerce_lightcurve,
)

from dk154_targets import paths


@pytest.fixture
def basic_config():
    return {"query_parameters": {"n_objects": 100, "max_earliest_lookback": 100.0}}


@pytest.fixture
def query_pattern():
    return {
        "classifier": "lc_classifier_top",
        "class": "Transient",
        "probability": 0.8,
    }


# def test__target_from_alerce():
#    target_from_alerce_lightcurve


@pytest.fixture
def query_results_rows():
    return [
        ("ZTF1000", 20, 50, 60000.0),
        ("ZTF1002", 15, 40, 60000.0),
        ("ZTF1001", 10, 30, 60000.0),  # we'll ignore this one when reading in
        ("ZTF1001", 12, 30, 60000.3),  # ...as this one's updated already.
    ]


@pytest.fixture
def query_results_df(query_results_rows):
    return pd.DataFrame(
        query_results_rows, columns="oid ndethistory ncovhistory lastmjd".split()
    )


@pytest.fixture
def page_results_rows_list():
    return [
        [
            ("ZTF100", 10, 20),
            ("ZTF101", 11, 21),
            ("ZTF102", 12, 22),
            ("ZTF103", 13, 23),
            ("ZTF104", 14, 24),
        ],
        [
            ("ZTF200", 10, 20),
            ("ZTF201", 11, 21),
            ("ZTF202", 12, 22),
            ("ZTF203", 13, 23),
            ("ZTF204", 14, 24),
        ],
        [
            ("ZTF300", 10, 20),
            ("ZTF301", 11, 21),
            ("ZTF302", 12, 22),
        ],
    ]


@pytest.fixture
def page_results_df_list(page_results_rows_list):
    return [
        pd.DataFrame(rows, columns="oid ndethistory ncovhistory".split())
        for rows in page_results_rows_list
    ]


class Test__AlerceQueryManager:
    exp_alerce_path = paths.test_data_path / "alerce"
    exp_lightcurves_path = paths.test_data_path / "alerce/lightcurves"
    exp_alerts_path = paths.test_data_path / "alerce/alerts"
    exp_query_results_path = paths.test_data_path / "alerce/query_results"
    exp_probabilities_path = paths.test_data_path / "alerce/probabilities"
    exp_magstats_path = paths.test_data_path / "alerce/magstats"
    exp_cutouts_path = paths.test_data_path / "alerce/cutouts"

    @classmethod
    def _clear_test_directories(cls):
        for path in [
            cls.exp_lightcurves_path,
            cls.exp_alerts_path,
            cls.exp_query_results_path,
            cls.exp_probabilities_path,
            cls.exp_magstats_path,
            cls.exp_cutouts_path,
        ]:
            if path.exists():
                for filepath in path.glob("*.csv"):
                    os.remove(filepath)
                for filepath in path.glob("*.json"):
                    os.remove(filepath)
                path.rmdir()
        if cls.exp_alerce_path.exists():
            cls.exp_alerce_path.rmdir()

    def test__alerce_qm_init(self):
        self._clear_test_directories()
        assert not self.exp_alerce_path.exists()

        empty_config = {}

        qm1 = AlerceQueryManager(empty_config, {}, data_path=paths.test_data_path)
        assert qm1.parent_data_path == self.exp_alerce_path
        assert self.exp_alerce_path.exists()
        assert self.exp_lightcurves_path.exists()
        assert self.exp_query_results_path.exists()
        assert self.exp_probabilities_path.exists()
        assert self.exp_magstats_path.exists()
        assert self.exp_cutouts_path.exists()

        self._clear_test_directories()
        assert not self.exp_alerce_path.exists()

        # Test create_paths=False works
        qm2 = AlerceQueryManager(
            empty_config, {}, data_path=paths.test_data_path, create_paths=False
        )
        assert not self.exp_alerce_path.exists()

    def test__alerce_get_query_results_file(self):
        pass

    def test__init_uses_config(self, basic_config):
        qm2 = AlerceQueryManager(
            basic_config, {}, data_path=paths.test_data_path, create_paths=False
        )
        assert np.isclose(
            qm2.default_query_parameters["max_earliest_lookback"], 70.0
        )  # default is different!
        assert not self.exp_alerce_path.exists()
        assert qm2.query_parameters["n_objects"] == 100
        assert np.isclose(qm2.query_parameters["max_earliest_lookback"], 100.0)
        assert np.isclose(qm2.query_parameters["max_latest_lookback"], 30.0)

    def test__get_query_data(self, basic_config, query_pattern: dict):
        qm = AlerceQueryManager(
            basic_config, {}, data_path=paths.test_data_path, create_paths=False
        )

        t_ref = Time("2023-02-25T12:00:00", format="isot")
        query_data = qm.prepare_query_data(query_pattern, page=10, t_ref=t_ref)
        assert qm.default_query_parameters["n_objects"] == 25
        assert np.isclose(query_data["firstmjd"], 59900.0)
        assert np.isclose(query_data["lastmjd"], 59970.0)
        assert query_data["page_size"] == 100

    def test__read_existing_queries(
        self, query_pattern: dict, query_results_df: pd.DataFrame
    ):
        config = {"object_queries": {"test_query": query_pattern}}

        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)
        assert self.exp_query_results_path.exists()

        fake_results_path = paths.test_data_path / "alerce/query_results/test_query.csv"
        query_results_df.to_csv(fake_results_path, index=False)

        qm.query_new_targets()
        assert len(qm.query_results.keys()) == 0
        assert len(qm.query_updates.keys()) == 1
        assert "test_query" in qm.query_updates
        assert set(qm.query_updates["test_query"].index) == set(
            ["ZTF1000", "ZTF1001", "ZTF1002"]
        )
        assert (
            qm.query_updates["test_query"].loc["ZTF1001"]["ndethistory"] == 12
        )  # latest
        assert fake_results_path.exists()  # has NOT been deleted.

        os.remove(fake_results_path)
        assert not fake_results_path.exists()
        self._clear_test_directories()

    def test__query_and_collate_read_existing(self, page_results_df_list):
        config = {
            "query_parameters": {"n_objects": 5},
        }
        self._clear_test_directories()

        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)
        assert self.exp_query_results_path.exists()

        page1_path = paths.test_data_path / "alerce/query_results/test_query_001.csv"
        page2_path = paths.test_data_path / "alerce/query_results/test_query_002.csv"
        page3_path = paths.test_data_path / "alerce/query_results/test_query_003.csv"
        assert not page1_path.exists()
        assert not page2_path.exists()
        assert not page3_path.exists()

        # write the fake data to files.
        assert len(page_results_df_list) == 3
        page_results_df_list[0].to_csv(page1_path, index=False)
        page_results_df_list[1].to_csv(page2_path, index=False)
        page_results_df_list[2].to_csv(page3_path, index=False)

        t_ref = Time.now()
        query_results = qm.query_and_collate_pages("test_query", {}, t_ref=t_ref)
        assert len(query_results) == 13
        assert not page1_path.exists()
        assert not page2_path.exists()
        assert not page3_path.exists()
        self._clear_test_directories()

    def test__perform_magstats_query(self):
        empty_config = {}
        exp_magstats_file = paths.test_data_path / "alerce/magstats/test1.csv"

        self._clear_test_directories()
        qm = AlerceQueryManager(empty_config, {}, data_path=paths.test_data_path)

        assert self.exp_alerce_path.exists()
        assert self.exp_magstats_path.exists()
        assert not exp_magstats_file.exists()

        t_now = Time.now()
        t_ref = t_now + 1 * u.min

        rows = [(1, 60000.0), (2, 60000.1)]
        test_magstats = pd.DataFrame(rows, columns="fid lastmjd".split())
        oid_list = ["test1"]
        result = qm.perform_magstats_queries(
            oid_list, t_ref=t_ref, test_input=test_magstats
        )
        assert exp_magstats_file.exists()
        assert result == (1, 0, 0)
        self._clear_test_directories()

    def test__perform_magstats_query_read_existing(self):
        empty_config = {}
        data_path = paths.test_data_path
        exp_alerce_path = data_path / "alerce"
        exp_magstats_path = data_path / "alerce/magstats"

        self._clear_test_directories()
        qm = AlerceQueryManager(empty_config, {}, data_path=data_path)
        assert exp_alerce_path.exists()
        assert exp_magstats_path.exists()

        t_now = Time.now()
        rows = [(1, 60000.0), (2, 60000.1)]
        test_magstats = pd.DataFrame(rows, columns="fid lastmjd".split())
        test_magstats_file = exp_magstats_path / "test1.csv"
        test_magstats.to_csv(test_magstats_file, index=False)
        t_ref = t_now + 1 * u.min

        oid_list = ["test1"]
        result = qm.perform_magstats_queries(oid_list, t_ref=t_ref)
        assert result == (0, 1, 0)
