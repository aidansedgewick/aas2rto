import os
import pytest

import numpy as np

import pandas as pd

from dk154_targets.query_managers.exc import BadKafkaConfigError
from dk154_targets.query_managers.lasair import (
    LasairQueryManager,
    target_from_lasair_lightcurve,
)

from dk154_targets import paths


@pytest.fixture
def client_config():
    return {"token": "abcdef_secret"}


@pytest.fixture
def kafka_config():
    return {
        "host": "https://host.com",
        "group_id": "test_1234",
        "topics": ["a_topic_str"],
    }


@pytest.fixture
def query_parameters():
    return {}


@pytest.fixture
def example_config(client_config, kafka_config, query_parameters):
    return {
        "client_config": client_config,
        "kafka_config": kafka_config,
        "query_parameters": query_parameters,
    }


class Test__LasairQueryManager:
    exp_lasair_path = paths.test_data_path / "lasair"
    exp_lightcurves_path = paths.test_data_path / "lasair/lightcurves"
    exp_alerts_path = paths.test_data_path / "lasair/alerts"
    exp_query_results_path = paths.test_data_path / "lasair/query_results"
    exp_probabilities_path = paths.test_data_path / "lasair/probabilities"
    exp_magstats_path = paths.test_data_path / "lasair/magstats"
    exp_cutouts_path = paths.test_data_path / "lasair/cutouts"

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
        if cls.exp_lasair_path.exists():
            cls.exp_lasair_path.rmdir()

    def test__lasair_qm_init(self):
        empty_config = {}

        data_path = paths.test_data_path
        self._clear_test_directories()
        qm = LasairQueryManager(empty_config, {}, data_path=paths.test_data_path)

        assert isinstance(qm.lasair_config, dict)
        assert len(qm.lasair_config) == 0

        assert qm.kafka_config is None

        assert qm.parent_data_path == self.exp_lasair_path
        assert qm.parent_data_path.exists()
        assert self.exp_lasair_path.exists()  # the same as above...
        assert self.exp_lightcurves_path.exists()
        assert self.exp_alerts_path.exists()
        # assert exp_cutouts_path.exists()

        self._clear_test_directories()
        assert not self.exp_lasair_path.exists()

        qm = LasairQueryManager(
            empty_config, {}, data_path=data_path, create_paths=False
        )
        assert not qm.parent_data_path.exists()
        assert not self.exp_lasair_path.exists()  # don't create as appropriate.

    def test__lasair_uses_config(self, example_config):
        example_config["kafka_config"]["n_alerts"] = 20
        example_config["query_parameters"]["interval"] = 3.0

        qm = LasairQueryManager(
            example_config, {}, data_path=paths.test_data_path, create_paths=False
        )
        assert isinstance(qm.client_config, dict)
        assert qm.client_config["token"] == "abcdef_secret"

        assert isinstance(qm.kafka_config, dict)
        assert qm.kafka_config["host"] == "https://host.com"
        assert qm.kafka_config["group_id"] == "test_1234"
        assert isinstance(qm.kafka_config["topics"], list)
        assert set(qm.kafka_config["topics"]) == set(["a_topic_str"])
        assert qm.kafka_config["n_alerts"] == 20
        assert qm.default_kafka_parameters["n_alerts"] == 10  # haven't changed default
        assert np.isclose(qm.kafka_config["timeout"], 10.0)  # correctly read default
        assert np.isclose(qm.default_kafka_parameters["timeout"], 10.0)

        assert np.isclose(qm.default_query_parameters["interval"], 2.0)
        assert np.isclose(qm.query_parameters["interval"], 3.0)  # correctly updated

        assert qm.default_query_parameters["max_failed_queries"] == 10
        assert qm.query_parameters["max_failed_queries"] == 10  # correctly read

    def test__topics_converted_to_list(self, kafka_config):
        kafka_config["topics"] = "a_topic_str"
        config = {"kafka_config": kafka_config}

        self._clear_test_directories()
        qm = LasairQueryManager(
            config, {}, data_path=paths.test_data_path, create_paths=False
        )
        assert not qm.parent_data_path.exists()

        assert qm.kafka_config["host"] == "https://host.com"
        assert qm.kafka_config["group_id"] == "test_1234"
        assert isinstance(qm.kafka_config["topics"], list)
        assert len(qm.kafka_config["topics"]) == 1
        assert set(qm.kafka_config["topics"]) == set(["a_topic_str"])

    def test__error_on_kafka_config_missing_host(self, example_config: dict):
        # Test failure without host
        config1 = example_config.copy()
        config1["kafka_config"].pop("host")
        assert "host" not in config1["kafka_config"]
        assert "group_id" in config1["kafka_config"]
        assert "topics" in config1["kafka_config"]
        with pytest.raises(BadKafkaConfigError):
            qm = LasairQueryManager(
                config1, {}, data_path=paths.test_data_path, create_paths=False
            )

    def test__error_on_kafka_config_missing_host(self, example_config: dict):
        # Test failure without group_id
        config2 = example_config.copy()
        config2["kafka_config"].pop("group_id")
        assert "host" in config2["kafka_config"]
        assert "group_id" not in config2["kafka_config"]
        assert "topics" in config2["kafka_config"]
        with pytest.raises(BadKafkaConfigError):
            qm = LasairQueryManager(
                config2, {}, data_path=paths.test_data_path, create_paths=False
            )

    def test__error_on_kafka_config_missing_host(self, example_config: dict):
        # Test failure without topics
        config3 = example_config.copy()
        config3["kafka_config"].pop("topics")
        assert "host" in config3["kafka_config"]
        assert "group_id" in config3["kafka_config"]
        assert "topics" not in config3["kafka_config"]
        with pytest.raises(BadKafkaConfigError):
            qm = LasairQueryManager(
                config3, {}, data_path=paths.test_data_path, create_paths=False
            )
