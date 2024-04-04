import copy
import os

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import pytest

from dk154_targets import Target
from dk154_targets import TargetData
from dk154_targets.exc import (
    BadKafkaConfigError,
    MissingObjectIdError,
    MissingCoordinatesError,
    MissingKeysWarning,
    UnexpectedKeysWarning,
)
from dk154_targets.query_managers import fink
from dk154_targets.query_managers.fink import (
    combine_fink_detections_non_detections,
    process_fink_lightcurve,
    target_from_fink_lightcurve,
    target_from_fink_alert,
    process_fink_query_results,
    get_updates_from_query_results,
    target_from_fink_query_row,
    FinkQueryManager,
    FinkQuery,
)


@pytest.fixture
def fink_lc_rows():
    """
    candid, jd, objectId, mag, magerr, diffmaglim, ra, dec, tag
    """
    return [
        (-1, 2460001.5, "ZTF00abc", np.nan, np.nan, 17.0, np.nan, np.nan, "upperlim"),
        (-1, 2460002.5, "ZTF00abc", np.nan, np.nan, 17.0, np.nan, np.nan, "upperlim"),
        (-1, 2460003.5, "ZTF00abc", np.nan, np.nan, 17.0, np.nan, np.nan, "upperlim"),
        (-1, 2460004.5, "ZTF00abc", 18.5, 0.5, 19.0, np.nan, np.nan, "badquality"),
        (
            23000_10000_20000_5005,
            2460005.5,
            "ZTF00abc",
            18.2,
            0.1,
            19.0,
            30.0,
            45.0,
            "valid",
        ),
        (
            23000_10000_20000_5006,
            2460006.5,
            "ZTF00abc",
            18.3,
            0.1,
            19.5,
            30.0,
            45.0,
            "valid",
        ),
        (
            23000_10000_20000_5007,
            2460007.5,
            "ZTF00abc",
            18.0,
            0.1,
            19.5,
            30.0,
            45.0,
            "valid",
        ),
    ]


@pytest.fixture
def fink_lc(fink_lc_rows):
    """
    A fake lightcurve. Only use relevant columns...
    """

    return pd.DataFrame(
        fink_lc_rows,
        columns="candid jd objectId mag magerr diffmaglim ra dec tag".split(),
    )


@pytest.fixture
def fink_alert():
    return dict(objectId="ZTF00abc", ra=30.0, dec=45.0)


@pytest.fixture
def fink_query_rows():
    """
    candid, objectId, ra, dec, ndethist
    note that objectId and ndethist are scrambled.

    """
    return [
        (23000_10000_20000_6004, "ZTF00abc", 30.0, 45.0, 9),
        (23000_10000_20000_6002, "ZTF00abc", 30.0, 45.0, 7),
        (23000_10000_20000_6001, "ZTF00abc", 30.0, 45.0, 6),
        (23000_10000_20000_6003, "ZTF00abc", 30.0, 45.0, 8),
        (23000_10000_20000_6021, "ZTF02xyz", 20.0, 20.0, 5),
        (23000_10000_20000_6011, "ZTF01ijk", 15.0, 20.0, 5),
        (23000_10000_20000_6012, "ZTF01ijk", 15.0, 20.0, 6),
    ]


@pytest.fixture
def fink_query_results(fink_query_rows):
    return pd.DataFrame(
        fink_query_rows, columns="candid objectId ra dec ndethist".split()
    )


@pytest.fixture
def fink_updated_query_rows():
    """
    Compare with fink_query_rows.
    00 is not present, 01 has not been updated, 02 has been update, 03 is new.
    """
    return [
        (23000_10000_20000_6021, "ZTF02xyz", 20.0, 20.0, 5),
        (23000_10000_20000_6022, "ZTF02xyz", 20.0, 20.0, 6),
        (23000_10000_20000_6023, "ZTF02xyz", 20.0, 20.0, 7),
        (23000_10000_20000_6011, "ZTF01ijk", 15.0, 20.0, 5),
        (23000_10000_20000_6012, "ZTF01ijk", 15.0, 20.0, 6),
        (23000_10000_20000_6031, "ZTF03lmn", 25.0, 20.0, 4),
    ]


@pytest.fixture
def fink_updated_query_results(fink_updated_query_rows):
    return pd.DataFrame(
        fink_updated_query_rows, columns="candid objectId ra dec ndethist".split()
    )


@pytest.fixture
def kafka_config():
    return {
        "username": "test_user",
        "group_id": "test_group",
        "bootstrap.servers": "123.456.789.01",
        "topics": ["test_alert_stream"],
    }


@pytest.fixture
def query_parameters():
    return {"object_query_interval": 0.5}


@pytest.fixture
def fink_config(kafka_config, query_parameters):
    return {
        "kafka_config": kafka_config,
        "query_parameters": query_parameters,
        "object_queries": "interesting_topic",
    }


@pytest.fixture
def target_lookup():
    return {}


@pytest.fixture
def fink_qm(fink_config, target_lookup, tmp_path):
    return FinkQueryManager(
        fink_config, target_lookup, parent_path=tmp_path, create_paths=False
    )


@pytest.fixture
def alert_base():
    return dict(
        objectId="ZTF00abc",
        cdsxmatch="Unknown",
        rf_snia_vs_nonia=0.0,
        snn_snia_vs_nonia=0.0,
        snn_sn_vs_all=0.0,
        mulens=0.0,
        roid=3,
        nalerthist=1,
        rf_kn_vs_nonkn=0.0,
    )


@pytest.fixture
def alert_list(alert_base):
    l = []
    for ii in range(15):
        mjd = Time(60010.0 + ii, format="mjd")  # mjd60010 = 7-mar-23
        candidate = {}
        candidate["magpsf"] = 19.0 - 0.1 * ii
        candidate["jd"] = mjd.jd
        candidate["ra"] = 45.0
        candidate["dec"] = 60.0
        alert = copy.deepcopy(alert_base)
        alert["candidate"] = candidate
        alert["candid"] = 23000_10000_20000_5010 + (ii + 1)
        alert["timestamp"] = mjd.strftime("%y%m%d_%H%M%S")
        l.append(alert)
    return l


@pytest.fixture()
def polled_alerts(alert_list):
    return [("interesting_topic", alert, "key_") for alert in alert_list]


class MockConsumer:
    alert_list = None  # TODO: how to get alert_list into class definition?!

    def __init__(self, topics, config):
        self.index = 0

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def poll(self, timeout=None):
        if self.alert_list is None:
            raise ValueError("you should set mock_consumer.alert_list")

        if self.index > len(self.alert_list):
            return (None, None, None)
        result = self.alert_list[self.index]
        self.index = self.index + 1
        return ("interesting_topic", result, "key_")


@pytest.fixture
def target_list():
    return [
        Target("ZTF00abc", ra=30.0, dec=45.0),
        Target("ZTF01ijk", ra=45.0, dec=45.0),
        Target("ZTF02xyz", ra=60.0, dec=45.0),
    ]


class Test__CombineFinkDetectionsNonDetections:
    pass


class Test__ProcessFinkLightcurve:
    def test__normal_behaviour(self, fink_lc):
        result = process_fink_lightcurve(fink_lc)
        assert len(result) == 7

        ulim = result[result["tag"] == "upperlim"]
        badqual = result[result["tag"] == "badquality"]
        valid = result[result["tag"] == "valid"]

        assert len(ulim == 4)
        assert len(badqual == 1)
        assert len(valid == 3)

        assert all(ulim["candid"] == -1)
        assert valid.iloc[0]["candid"] == 2300010000200005005

    def test__date_cols_added_for_empty_df(self):
        empty_df = pd.DataFrame()
        assert empty_df.empty

        result = process_fink_lightcurve(empty_df)
        assert len(result) == 0
        assert "jd" in result.columns
        assert "mjd" in result.columns

    def test__result_is_sorted(self, fink_lc):
        scramble_idx = [5, 0, 3, 2, 6, 1, 4]
        assert len(scramble_idx) == len(fink_lc)  # Enough numbers
        assert set(scramble_idx) == set(range(len(fink_lc)))  # Not missed any numbers

        # Scramble the lc to test it gets correctly sorted
        scramble_lc = fink_lc.iloc[scramble_idx, :]
        scramble_lc.reset_index(inplace=True)
        assert not np.all(
            scramble_lc.iloc[:-1].jd.values < scramble_lc.iloc[1:].jd.values
        )
        assert scramble_lc.iloc[0].candid == 23000_10000_20000_5006
        assert np.isclose(scramble_lc.iloc[1].jd, 2460001.5)
        assert scramble_lc.iloc[2].tag == "badquality"
        assert np.isclose(scramble_lc.iloc[6].jd, 2460005.5)

        result = process_fink_lightcurve(scramble_lc)
        assert np.all(result["jd"].values[:-1] < result["jd"].values[1:])


class Test__TargetFromFinkAlert:
    def test__normal_behaviour(self, fink_alert):
        result = target_from_fink_alert(fink_alert)
        assert isinstance(result, Target)
        assert result.objectId == "ZTF00abc"
        assert isinstance(result.coord, SkyCoord)
        assert np.isclose(result.ra, 30.0)
        assert np.isclose(result.dec, 45.0)

    def test__missing_objectId_error(self, fink_alert):
        _ = fink_alert.pop("objectId")
        assert "objectId" not in fink_alert

        with pytest.raises(MissingObjectIdError):
            result = target_from_fink_alert(fink_alert)

    def test__missing_coordinates_error(self, fink_alert):
        ra = fink_alert.pop("ra")
        assert "ra" not in fink_alert
        assert "dec" in fink_alert
        with pytest.raises(MissingCoordinatesError):
            res = target_from_fink_alert(fink_alert)

        fink_alert["ra"] = ra
        dec = fink_alert.pop("dec")
        assert "ra" in fink_alert
        assert "dec" not in fink_alert
        with pytest.raises(MissingCoordinatesError):
            res = target_from_fink_alert(fink_alert)


class Test__TargetFromFinkLightcurve:
    def test__normal_behaviour(self, fink_lc):
        result = target_from_fink_lightcurve(fink_lc)
        assert isinstance(result, Target)
        assert result.objectId == "ZTF00abc"
        assert np.isclose(result.ra, 30.0)
        assert np.isclose(result.dec, 45.0)

        assert isinstance(result.coord, SkyCoord)

        assert "fink" in result.target_data
        assert isinstance(result.target_data["fink"], TargetData)
        assert isinstance(result.target_data["fink"].lightcurve, pd.DataFrame)
        assert len(result.target_data["fink"].lightcurve) == 7

        assert result.updated

    def test__missing_objectId_error(self, fink_lc):
        fink_lc.drop("objectId", axis=1, inplace=True)
        assert "objectId" not in fink_lc.columns

        with pytest.raises(MissingObjectIdError):
            result = target_from_fink_lightcurve(fink_lc)

    def test__ambiguous_objectId(self, fink_lc):
        fink_lc.iloc[1, 2] = "ZTF99xyz"
        assert set(fink_lc["objectId"].unique()) == set(["ZTF00abc", "ZTF99xyz"])

        result = target_from_fink_lightcurve(fink_lc)
        assert result.objectId == "ZTF00abc"

    def test__missing_coordinates_error(self, fink_lc):
        fink_lc.drop("ra", axis=1, inplace=True)
        assert "ra" not in fink_lc.columns

        with pytest.raises(MissingCoordinatesError):
            res = target_from_fink_lightcurve(fink_lc)


class Test__ProcessFinkQueryResults:
    def test__normal_behaviour(self, fink_query_results):
        result = process_fink_query_results(fink_query_results)

        assert "objectId" in result.columns

        assert len(result) == 3
        assert result["objectId"].iloc[0] == "ZTF00abc"
        assert result["candid"].iloc[0] == 23000_10000_20000_6004
        assert result["ndethist"].iloc[0] == 9

        assert result["objectId"].iloc[1] == "ZTF01ijk"
        assert result["candid"].iloc[1] == 23000_10000_20000_6012
        assert result["ndethist"].iloc[1] == 6

        assert result["objectId"].iloc[2] == "ZTF02xyz"
        assert result["candid"].iloc[2] == 23000_10000_20000_6021
        assert result["ndethist"].iloc[2] == 5


class Test__GetUpdatesFromQueryResults:
    def test__existing_results_is_none(self, fink_updated_query_results):
        updated_results = process_fink_query_results(fink_updated_query_results)
        assert len(updated_results) == 3

        existing_results = None
        updates_from_query = get_updates_from_query_results(
            existing_results, updated_results
        )

        assert len(updates_from_query) == 3
        assert set(updates_from_query["objectId"]) == set(
            ["ZTF01ijk", "ZTF02xyz", "ZTF03lmn"]
        )

        assert updates_from_query.iloc[0].objectId == "ZTF01ijk"
        assert updates_from_query.iloc[0].candid == 23000_10000_20000_6012
        assert updates_from_query.iloc[0].ndethist == 6

        assert updates_from_query.iloc[1].objectId == "ZTF02xyz"
        assert updates_from_query.iloc[1].candid == 23000_10000_20000_6023
        assert updates_from_query.iloc[1].ndethist == 7

        assert updates_from_query.iloc[2].objectId == "ZTF03lmn"
        assert updates_from_query.iloc[2].candid == 23000_10000_20000_6031
        assert updates_from_query.iloc[2].ndethist == 4

    def test__updated_results_is_none(self, fink_query_results):
        existing_results = process_fink_query_results(fink_query_results)
        updated_results = None

        updates_from_query = get_updates_from_query_results(
            existing_results, updated_results
        )

        assert updates_from_query.empty
        assert set(updates_from_query.columns) == set(["objectId", "ndethist"])

    def test__both_not_none(self, fink_query_results, fink_updated_query_results):
        existing_results = process_fink_query_results(fink_query_results)
        updated_results = process_fink_query_results(fink_updated_query_results)

        updates_from_query = get_updates_from_query_results(
            existing_results, updated_results
        )
        assert len(updates_from_query) == 2
        assert set(updates_from_query["objectId"]) == set(["ZTF02xyz", "ZTF03lmn"])

        assert updates_from_query.iloc[0].objectId == "ZTF02xyz"
        assert updates_from_query.iloc[0].candid == 23000_10000_20000_6023
        assert updates_from_query.iloc[0].ndethist == 7

        assert updates_from_query.iloc[1].objectId == "ZTF03lmn"
        assert updates_from_query.iloc[1].candid == 23000_10000_20000_6031
        assert updates_from_query.iloc[1].ndethist == 4

    def test__existing_and_updated_results_are_identical(self, fink_query_results):
        existing_results = process_fink_query_results(fink_query_results)
        updated_results = process_fink_query_results(fink_query_results)

        assert existing_results.equals(updated_results)

        updates_from_query = get_updates_from_query_results(
            existing_results, updated_results
        )
        assert updates_from_query.empty
        assert set(updates_from_query.columns) == set(["objectId", "ndethist"])


class Test__TargetFromFinkQueryRow:
    def test__normal_behaviour(self, fink_query_results):
        processed_results = process_fink_query_results(fink_query_results)
        row = processed_results.iloc[0]

        result = target_from_fink_query_row(row)
        assert isinstance(result, Target)
        assert result.objectId == "ZTF00abc"
        assert np.isclose(result.ra, 30.0)
        assert np.isclose(result.dec, 45.0)

    def test__missing_coordinate_error(self, fink_query_results):
        processed_results = process_fink_query_results(fink_query_results)
        processed_results.drop("ra", axis=1, inplace=True)

        with pytest.raises(MissingCoordinatesError):
            res = target_from_fink_query_row(processed_results.iloc[0])


class Test__FinkQueryManagerInit:
    def test__fink_qm_init(self, fink_config, target_lookup, tmp_path):
        qm = FinkQueryManager(fink_config, target_lookup, parent_path=tmp_path)

        assert isinstance(qm.fink_config, dict)
        assert "query_parameters" in qm.fink_config
        assert "kafka_config" in qm.fink_config

        assert isinstance(qm.target_lookup, dict)

        assert isinstance(qm.kafka_config, dict)
        expected_kws = [
            "username",
            "group_id",
            "bootstrap.servers",
            "topics",
            "n_alerts",
            "timeout",
        ]
        assert all([kw in qm.kafka_config for kw in expected_kws])
        assert qm.kafka_config["n_alerts"] == 10  # Default correctly read

        assert isinstance(qm.object_queries, list)
        assert len(qm.object_queries) == 1
        assert qm.object_queries

        assert np.isclose(qm.query_parameters["object_query_interval"], 0.5)
        expected_kws = [
            "object_query_lookback",
            "object_query_timestep",
            "object_query_interval",
            "lightcurve_update_interval",
            "max_failed_queries",
            "max_total_query_time",
        ]
        assert all(
            [kw in qm.query_parameters for kw in expected_kws]
        )  # Defaults correctly added

        assert qm.data_path == tmp_path / "fink"
        assert qm.data_path.exists()  # Path correctly created
        assert qm.lightcurves_path == tmp_path / "fink/lightcurves"
        assert qm.lightcurves_path.exists()

    def test__no_fink_kafka_config(self, target_lookup, tmp_path):
        config = {}
        qm = FinkQueryManager(
            config, target_lookup, parent_path=tmp_path, create_paths=False
        )
        assert qm.kafka_config is None

    def test__bad_fink_kafka_config(self, kafka_config, target_lookup, tmp_path):
        bad_kafka_config = copy.deepcopy(kafka_config)
        _ = bad_kafka_config.pop("username")
        assert set(bad_kafka_config.keys()) == set(
            ["group_id", "bootstrap.servers", "topics"]
        )
        config = {"kafka_config": bad_kafka_config}
        with pytest.warns(MissingKeysWarning):
            with pytest.raises(BadKafkaConfigError):
                qm = FinkQueryManager(
                    config, target_lookup, parent_path=tmp_path, create_paths=False
                )

        bad_kafka_config = copy.deepcopy(kafka_config)
        _ = bad_kafka_config.pop("group_id")
        assert set(bad_kafka_config.keys()) == set(
            ["username", "bootstrap.servers", "topics"]
        )
        config = {"kafka_config": bad_kafka_config}
        with pytest.warns(MissingKeysWarning):
            with pytest.raises(BadKafkaConfigError):
                qm = FinkQueryManager(
                    config, target_lookup, parent_path=tmp_path, create_paths=False
                )

        bad_kafka_config = copy.deepcopy(kafka_config)
        _ = bad_kafka_config.pop("bootstrap.servers")
        assert set(bad_kafka_config.keys()) == set(["username", "group_id", "topics"])
        config = {"kafka_config": bad_kafka_config}
        with pytest.warns(MissingKeysWarning):
            with pytest.raises(BadKafkaConfigError):
                qm = FinkQueryManager(
                    config, target_lookup, parent_path=tmp_path, create_paths=False
                )

        bad_kafka_config = copy.deepcopy(kafka_config)
        _ = bad_kafka_config.pop("topics")
        assert set(bad_kafka_config.keys()) == set(
            ["username", "group_id", "bootstrap.servers"]
        )
        config = {"kafka_config": bad_kafka_config}
        with pytest.warns(MissingKeysWarning):
            with pytest.raises(BadKafkaConfigError):
                qm = FinkQueryManager(
                    config, target_lookup, parent_path=tmp_path, create_paths=False
                )

    def test__warning_on_fink_config_keys(self, fink_config, target_lookup, tmp_path):
        fink_config["blah"] = 0

        with pytest.warns(UnexpectedKeysWarning):
            fink_qm = FinkQueryManager(
                fink_config, target_lookup, parent_path=tmp_path, create_paths=False
            )

    def test__warning_on_fink_query_parameters(
        self, fink_config, target_lookup, tmp_path
    ):
        fink_config["query_parameters"]["blah"] = 0

        with pytest.warns(UnexpectedKeysWarning):
            qm = FinkQueryManager(
                fink_config, target_lookup, parent_path=tmp_path, create_paths=False
            )


class Test__FinkAlertStreams:
    def test__fink_mock_consumer_behaviour(self, alert_list, monkeypatch):
        assert MockConsumer.alert_list is None
        with monkeypatch.context() as m:
            m.setattr(MockConsumer, "alert_list", alert_list)
            assert len(MockConsumer.alert_list) == 15
            with MockConsumer([], []) as mc:
                assert mc.index == 0
                poll_result = mc.poll()
                assert isinstance(poll_result, tuple)
                assert len(poll_result) == 3
                assert mc.index == 1
                assert np.isclose(poll_result[1]["candidate"]["magpsf"], 19.0)
        assert MockConsumer.alert_list is None  # Back to normal.

    def test__fink_listen_for_alerts(self, fink_qm, alert_list, monkeypatch):
        assert MockConsumer.alert_list is None
        assert hasattr(fink.AlertConsumer, "consume")  # Before patch
        with monkeypatch.context() as m:
            # Patch the alert list onto the MockConsumer
            m.setattr(MockConsumer, "alert_list", alert_list)
            assert len(MockConsumer.alert_list) == 15  # is patched!

            # Patch the MockConsumer into the fink module
            m.setattr("dk154_targets.query_managers.fink.AlertConsumer", MockConsumer)
            assert not hasattr(fink.AlertConsumer, "consume")  # is patched!

            alerts = fink_qm.listen_for_alerts()
            assert len(alerts) == 10
            print(alerts[0])
            assert isinstance(alerts[0], tuple)
            assert len(alerts[0]) == 3
            for ii in range(10):
                assert alerts[ii][0] == "interesting_topic"
            # Alerts start at [...]_5011
            assert alerts[0][1]["candid"] == 23000_10000_20000_5011
            assert alerts[1][1]["candid"] == 23000_10000_20000_5012
            assert alerts[5][1]["candid"] == 23000_10000_20000_5016
            assert alerts[9][1]["candid"] == 23000_10000_20000_5020
        assert MockConsumer.alert_list is None
        assert hasattr(fink.AlertConsumer, "consume")  # Back to normal.

    def test__process_fink_alerts(self, fink_qm, polled_alerts):
        # Arrange
        fink_qm.create_paths()
        assert fink_qm.alerts_path.exists()
        exp_alerts_dir = fink_qm.alerts_path / "ZTF00abc"
        assert not exp_alerts_dir.exists()

        # Action
        processed_alerts = fink_qm.process_alerts(polled_alerts)

        # Assert
        assert exp_alerts_dir.exists()
        assert len([p for p in exp_alerts_dir.glob("*.json")]) == 15
        exp_alert_file = exp_alerts_dir / "2300010000200005011.json"
        assert exp_alert_file.exists()

        assert processed_alerts[0]["topic"] == "interesting_topic"
        assert processed_alerts[0]["tag"] == "valid"
        assert "cdsxmatch" in processed_alerts[0]

    def test__new_targets_from_fink_alerts(self, fink_qm, polled_alerts):
        # Arrange
        processed_alerts = fink_qm.process_alerts(polled_alerts, save_alerts=False)
        t_ref = Time(60030.0, format="jd")
        assert "ZTF00abc" not in fink_qm.target_lookup

        # Acion
        added, existing = fink_qm.new_targets_from_alerts(processed_alerts, t_ref=t_ref)

        # Assert
        assert len(added) == 1
        assert len(existing) == 14
        assert set(existing) == set(["ZTF00abc"])


class Test__FinkObjectQueries:
    def test__no_existing_results(self, fink_qm, fink_query_results, monkeypatch):
        fink_qm.create_paths()
        assert set(fink_qm.object_queries) == set(["interesting_topic"])

        assert len(fink_qm.query_results) == 0
        assert "interesting_topic" not in fink_qm.query_results

        exp_query_results_path = (
            fink_qm.data_path / "query_results/interesting_topic.csv"
        )
        assert exp_query_results_path.parent.exists()
        assert not exp_query_results_path.exists()

        def mock_query(*args, **kwargs):
            return fink_query_results

        with monkeypatch.context() as m:
            m.setattr(
                "dk154_targets.query_managers.fink.FinkQuery.query_and_collate_latests",
                mock_query,
            )
            object_updates = fink_qm.query_for_object_updates()

        assert len(object_updates) == 3

        assert object_updates["objectId"].iloc[0] == "ZTF00abc"
        assert object_updates["candid"].iloc[0] == 23000_10000_20000_6004
        assert object_updates["ndethist"].iloc[0] == 9

        assert object_updates["objectId"].iloc[1] == "ZTF01ijk"
        assert object_updates["candid"].iloc[1] == 23000_10000_20000_6012
        assert object_updates["ndethist"].iloc[1] == 6

        assert object_updates["objectId"].iloc[2] == "ZTF02xyz"
        assert object_updates["candid"].iloc[2] == 23000_10000_20000_6021
        assert object_updates["ndethist"].iloc[2] == 5

        assert isinstance(fink_qm.query_results["interesting_topic"], pd.DataFrame)
        assert exp_query_results_path.exists()

    def test__no_fink_results_write_empty_file(self, fink_qm, monkeypatch):
        fink_qm.create_paths()

        def mock_query(*args, **kwargs):
            return None

        exp_query_results_path = (
            fink_qm.data_path / "query_results/interesting_topic.csv"
        )
        assert exp_query_results_path.parent.exists()
        assert not exp_query_results_path.exists()

        with monkeypatch.context() as m:
            m.setattr(
                "dk154_targets.query_managers.fink.FinkQuery.query_and_collate_latests",
                mock_query,
            )
            object_updates = fink_qm.query_for_object_updates()

        topic_results = fink_qm.query_results["interesting_topic"]
        assert isinstance(topic_results, pd.DataFrame)
        assert topic_results.empty
        assert set(object_updates.columns) == set(["objectId", "ndethist"])

        assert object_updates.empty
        assert set(object_updates.columns) == set(["objectId", "ndethist"])

        recovered_df = pd.read_csv(exp_query_results_path)
        assert recovered_df.empty
        assert set(recovered_df.columns) == set(["objectId", "ndethist"])

    def test__many_topics(
        self, fink_qm, fink_query_results, fink_updated_query_results, monkeypatch
    ):
        fink_qm.create_paths()
        assert len(fink_qm.query_results) == 0  # Nothing there yet...

        fink_qm.object_queries = ["topic00", "topic01", "topic02"]  # Change this.

        def mock_query(fink_class, **kwargs):
            if fink_class == "topic00":
                return fink_query_results
            if fink_class == "topic01":
                return fink_updated_query_results
            return None

        with monkeypatch.context() as m:
            m.setattr(
                "dk154_targets.query_managers.fink.FinkQuery.query_and_collate_latests",
                mock_query,
            )
            object_updates = fink_qm.query_for_object_updates()

        assert len(object_updates) == 4
        assert set(object_updates["objectId"]) == set(
            ["ZTF00abc", "ZTF01ijk", "ZTF02xyz", "ZTF03lmn"]
        )

        topic00_results = fink_qm.query_results["topic00"]
        assert isinstance(topic00_results, pd.DataFrame)
        assert len(topic00_results) == 3
        assert set(topic00_results["objectId"]) == set(
            ["ZTF00abc", "ZTF01ijk", "ZTF02xyz"]
        )
        # Should be the same as the results from process query_results
        assert topic00_results["candid"].iloc[0] == 23000_10000_20000_6004
        assert topic00_results["candid"].iloc[1] == 23000_10000_20000_6012
        assert topic00_results["candid"].iloc[2] == 23000_10000_20000_6021
        exp_topic00_results_path = fink_qm.data_path / "query_results/topic00.csv"
        assert exp_topic00_results_path.exists()

        topic01_results = fink_qm.query_results["topic01"]
        assert isinstance(topic01_results, pd.DataFrame)
        assert len(topic01_results) == 3
        assert set(topic01_results["objectId"]) == set(
            ["ZTF01ijk", "ZTF02xyz", "ZTF03lmn"]
        )
        # Should be the same as the results from process query_results
        assert topic01_results["candid"].iloc[0] == 23000_10000_20000_6012
        assert topic01_results["candid"].iloc[1] == 23000_10000_20000_6023
        assert topic01_results["candid"].iloc[2] == 23000_10000_20000_6031
        exp_topic01_results_path = fink_qm.data_path / "query_results/topic01.csv"
        assert exp_topic01_results_path.exists()

        topic02_results = fink_qm.query_results["topic02"]
        assert topic02_results.empty
        assert set(topic02_results.columns) == set(["objectId", "ndethist"])
        exp_topic02_results_path = fink_qm.data_path / "query_results/topic02.csv"
        assert exp_topic02_results_path.exists()
        topic02_recover = pd.read_csv(exp_topic02_results_path)
        assert topic02_recover.empty
        assert set(topic02_recover.columns) == set(["objectId", "ndethist"])

    def test__reads_existing_results(self, fink_qm, fink_query_results, monkeypatch):
        fink_qm.create_paths()
        assert len(fink_qm.query_results) == 0

        query_results_path = fink_qm.get_query_results_file("interesting_topic")
        fink_query_results.to_csv(query_results_path, index=False)

        def mock_query(*args, **kwargs):
            return None

        with monkeypatch.context() as m:
            m.setattr(
                "dk154_targets.query_managers.fink.FinkQuery.query_and_collate_latests",
                mock_query,
            )
            object_updates = fink_qm.query_for_object_updates()

        assert object_updates.empty  # sort of irrelevant
        stored_results = fink_qm.query_results["interesting_topic"]

        assert len(stored_results) == 3
        assert set(stored_results["objectId"]) == set(
            ["ZTF00abc", "ZTF01ijk", "ZTF02xyz"]
        )

    def test__does_not_requery_if_recent_file(
        self, fink_qm, fink_query_results, monkeypatch
    ):
        fink_qm.create_paths()

        query_results_path = fink_qm.get_query_results_file("interesting_topic")
        fink_query_results.to_csv(query_results_path, index=False)
        t_write = Time.now()

        def mock_query(*args, **kwargs):
            raise ValueError("should not have queried!")

        with monkeypatch.context() as m:
            t_ref = Time(t_write.jd + 0.1, format="jd")
            m.setattr(
                "dk154_targets.query_managers.fink.FinkQuery.query_and_collate_latests",
                mock_query,
            )
            object_updates = fink_qm.query_for_object_updates(t_ref=t_ref)

    def test__new_targets_from_updates(self, fink_qm, fink_query_results, monkeypatch):
        object_updates = process_fink_query_results(fink_query_results)

        assert len(fink_qm.target_lookup) == 0

        fink_qm.new_targets_from_object_updates(object_updates)

        assert len(fink_qm.target_lookup) == 3
        assert set(fink_qm.target_lookup.keys()) == set(
            ["ZTF00abc", "ZTF01ijk", "ZTF02xyz"]
        )

        t1 = fink_qm.target_lookup["ZTF00abc"]
        assert isinstance(t1, Target)
        assert np.isclose(t1.ra, 30.0)


class Test__FinkLightcurveQueries:
    def test__get_lightcurves_to_query(self, fink_qm, target_list, fink_lc):
        fink_qm.create_paths()

        lc_file = fink_qm.get_lightcurve_file("ZTF00abc")
        fink_lc.to_csv(lc_file, index=False)
        t_write = Time.now()

        for target in target_list:
            fink_qm.add_target(target)

        # Do we ignore the one which hasa recent lightcurve?
        assert fink_qm.query_parameters["lightcurve_update_interval"] > 0.25
        t_ref = Time(t_write.jd + 0.25, format="jd")
        lcs_to_query = fink_qm.get_lightcurves_to_query(t_ref=t_ref)
        assert set(lcs_to_query) == set(["ZTF01ijk", "ZTF02xyz"])

        # Does the existing one get re-queried if it's too old?
        assert fink_qm.query_parameters["lightcurve_update_interval"] < 3.0
        t_future = Time(t_write.jd + 3.0, format="jd")
        lcs_to_query = fink_qm.get_lightcurves_to_query(t_ref=t_future)
        assert set(lcs_to_query) == set(["ZTF00abc", "ZTF01ijk", "ZTF02xyz"])

    def test__perform_lightcurve_queries(
        self, fink_qm, fink_lc, target_list, monkeypatch
    ):
        fink_qm.create_paths()

        def mock_query(objectId, **kwrags):
            if objectId == "ZTF00abc":
                return fink_lc
            if objectId == "ZTF01ijk":
                return pd.DataFrame(columns="jd mjd".split())
            else:
                raise Exception

        objectId_list = [t.objectId for t in target_list]

        with monkeypatch.context() as m:
            m.setattr(
                "dk154_targets.query_managers.fink.FinkQuery.query_objects", mock_query
            )
            success, failed = fink_qm.perform_lightcurve_queries(objectId_list)

        assert set(success) == set(["ZTF00abc", "ZTF01ijk"])
        assert set(failed) == set(["ZTF02xyz"])

        exp_ZTF00abc_lc_file = fink_qm.data_path / "lightcurves/ZTF00abc.csv"
        assert exp_ZTF00abc_lc_file.exists()
        ZTF00abc_lc = pd.read_csv(exp_ZTF00abc_lc_file)
        assert len(ZTF00abc_lc) == 7

        exp_ZTF01ijk_lc_file = fink_qm.data_path / "lightcurves/ZTF01ijk.csv"
        assert exp_ZTF01ijk_lc_file.exists()
        ZTF01ijk_lc = pd.read_csv(exp_ZTF01ijk_lc_file)
        assert ZTF01ijk_lc.empty

        exp_ZTF02xyz_lc_file = fink_qm.data_path / "lightcurves/ZTF02xyz.csv"
        assert not exp_ZTF02xyz_lc_file.exists()

    def test__load_single_lightcurve(self, fink_qm, fink_lc):
        fink_qm.create_paths()

        fink_lc_path = fink_qm.get_lightcurve_file("ZTF00abc")
        assert not fink_lc_path.exists()

        fink_lc.to_csv(fink_lc_path, index=False)
        assert fink_lc_path.exists()

        lc = fink_qm.load_single_lightcurve("ZTF00abc")
        assert len(lc) == 7

        missing_lc_path = fink_qm.get_lightcurve_file("ZTF01ijk")
        assert not missing_lc_path.exists()
        missing_lc = fink_qm.load_single_lightcurve("ZTF01ijk")
        assert missing_lc is None

    def test__load_target_lightcurves(self, fink_qm, target_list, fink_lc):
        fink_qm.create_paths()

        for target in target_list:
            fink_qm.add_target(target)

        lc_path = fink_qm.get_lightcurve_file("ZTF00abc")
        fink_lc.to_csv(lc_path, index=False)

        loaded, missing = fink_qm.load_target_lightcurves()

        assert set(loaded) == set(["ZTF00abc"])
        assert set(missing) == set(["ZTF01ijk", "ZTF02xyz"])

    def test__load_missing_alerts(self, fink_qm, fink_lc, target_list, polled_alerts):
        fink_qm.create_paths()
        fink_qm.add_target(target_list[0])

        processed_alerts = fink_qm.process_alerts(polled_alerts[:5])

        # Modify lc slightly
        more_alerts = pd.DataFrame(processed_alerts[:2])
        print((more_alerts))
        fink_lc = pd.concat([fink_lc, more_alerts], ignore_index=True)
        assert len(fink_lc) == 9
        assert 2300010000200005011 in fink_lc["candid"].values
        assert fink_lc.iloc[7].candid == 2300010000200005011
        assert fink_lc.iloc[8].candid == 2300010000200005012

        lc_path = fink_qm.get_lightcurve_file("ZTF00abc")
        fink_lc.to_csv(lc_path, index=False)

        fink_qm.load_target_lightcurves()

        assert len(processed_alerts) == 5

        exp_alert00_path = (
            fink_qm.data_path / "alerts/ZTF00abc/2300010000200005011.json"
        )
        exp_alert01_path = (
            fink_qm.data_path / "alerts/ZTF00abc/2300010000200005012.json"
        )
        exp_alert02_path = (
            fink_qm.data_path / "alerts/ZTF00abc/2300010000200005013.json"
        )
        exp_alert03_path = (
            fink_qm.data_path / "alerts/ZTF00abc/2300010000200005014.json"
        )
        exp_alert04_path = (
            fink_qm.data_path / "alerts/ZTF00abc/2300010000200005015.json"
        )

        assert exp_alert00_path.exists()
        assert exp_alert01_path.exists()
        assert exp_alert02_path.exists()
        assert exp_alert03_path.exists()
        assert exp_alert04_path.exists()

        loaded_alerts = fink_qm.load_missing_alerts("ZTF00abc")
        assert len(loaded_alerts) == 3
        assert isinstance(loaded_alerts[0], dict)
        loaded_candids = [a["candid"] for a in loaded_alerts]
        exp_candids = [
            23000_10000_20000_5013,
            23000_10000_20000_5014,
            23000_10000_20000_5015,
        ]

        assert set(loaded_candids) == set(exp_candids)
