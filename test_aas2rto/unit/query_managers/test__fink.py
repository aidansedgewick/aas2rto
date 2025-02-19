import copy
import json
import os
import pickle
import requests

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import pytest

from aas2rto.target import Target, TargetData
from aas2rto.target_lookup import TargetLookup
from aas2rto.exc import (
    BadKafkaConfigError,
    MissingTargetIdError,
    MissingCoordinatesError,
    MissingKeysWarning,
    UnexpectedKeysWarning,
)
from aas2rto.query_managers import fink
from aas2rto.query_managers.fink import (
    # combine_fink_detections_non_detections,
    process_fink_lightcurve,
    target_from_fink_lightcurve,
    target_from_fink_alert,
    process_fink_query_results,
    get_updates_from_query_results,
    target_from_fink_query_row,
    FinkQueryManager,
    FinkQuery,
    FinkQueryError,
)


@pytest.fixture
def fink_lc_rows():
    """
    candid, jd, objectId, mag, magerr, diffmaglim, ra, dec, tag
    """

    base_id = 23000_10000_20000_5000  # integer NOT str.
    return [
        (-1, 2460001.5, "ZTF00abc", np.nan, np.nan, 17.0, np.nan, np.nan, "upperlim"),
        (-1, 2460002.5, "ZTF00abc", np.nan, np.nan, 17.0, np.nan, np.nan, "upperlim"),
        (-1, 2460003.5, "ZTF00abc", np.nan, np.nan, 17.0, np.nan, np.nan, "upperlim"),
        (-1, 2460004.5, "ZTF00abc", 18.5, 0.5, 19.0, np.nan, np.nan, "badquality"),
        (base_id + 5, 2460005.5, "ZTF00abc", 18.2, 0.1, 19.0, 30.0, 45.0, "valid"),
        (base_id + 6, 2460006.5, "ZTF00abc", 18.3, 0.1, 19.5, 30.0, 45.0, "valid"),
        (base_id + 7, 2460007.5, "ZTF00abc", 18.0, 0.1, 19.5, 30.0, 45.0, "valid"),
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
        "group.id": "test_group",
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
    return TargetLookup()


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
        tag="valid",
    )


@pytest.fixture
def alert_list(alert_base):
    """
    Make list of fake alerts. each one is a little brighter than the last.
    """

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


def gen_mock_cutouts(filepath):
    cutouts = {
        "science": np.random.uniform(0, 1, (100, 100)),
        "difference": np.random.uniform(0, 1, (100, 100)),
        "template": np.random.uniform(0, 1, (100, 100)),
    }
    with open(filepath, "wb+") as f:
        pickle.dump(cutouts, f)


def create_mock_consumer(alerts):
    """choose to do this as function which creates classes rather than simpler python closure,
    as the Consumer needs to have poll() method...
    is it better to
    """

    class MockConsumer:
        alert_list = alerts

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

    return MockConsumer


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
        assert result.target_id == "ZTF00abc"

        assert set(result.alt_ids.keys()) == set(["fink", "ztf"])
        assert result.alt_ids["fink"] == "ZTF00abc"

        assert isinstance(result.coord, SkyCoord)

        assert np.isclose(result.ra, 30.0)
        assert np.isclose(result.dec, 45.0)

    def test__alert_missing_objectId_error(self, fink_alert):
        _ = fink_alert.pop("objectId")
        assert "objectId" not in fink_alert

        with pytest.raises(MissingTargetIdError):
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
        assert result.target_id == "ZTF00abc"
        assert np.isclose(result.ra, 30.0)
        assert np.isclose(result.dec, 45.0)

        assert set(result.alt_ids.keys()) == set(["fink", "ztf"])
        assert result.alt_ids["fink"] == "ZTF00abc"
        assert result.alt_ids["ztf"] == "ZTF00abc"

        assert isinstance(result.coord, SkyCoord)

        assert "fink" in result.target_data
        assert isinstance(result.target_data["fink"], TargetData)
        assert isinstance(result.target_data["fink"].lightcurve, pd.DataFrame)
        assert len(result.target_data["fink"].lightcurve) == 7

        assert result.updated

    def test__missing_objectId_error(self, fink_lc):
        fink_lc.drop("objectId", axis=1, inplace=True)
        assert "objectId" not in fink_lc.columns

        with pytest.raises(MissingTargetIdError):
            result = target_from_fink_lightcurve(fink_lc)

    def test__ambiguous_objectId(self, fink_lc):
        fink_lc.iloc[1, 2] = "ZTF99xyz"
        assert set(fink_lc["objectId"].unique()) == set(["ZTF00abc", "ZTF99xyz"])

        result = target_from_fink_lightcurve(fink_lc)
        assert result.target_id == "ZTF00abc"

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
        assert result.target_id == "ZTF00abc"
        assert np.isclose(result.ra, 30.0)
        assert np.isclose(result.dec, 45.0)

        assert set(result.alt_ids.keys()) == set(["fink", "ztf"])
        assert result.alt_ids["fink"] == "ZTF00abc"

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

        assert isinstance(qm.target_lookup, TargetLookup)

        assert isinstance(qm.kafka_config, dict)
        expected_kws = [
            "username",
            "group.id",
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
            ["group.id", "bootstrap.servers", "topics"]
        )
        config = {"kafka_config": bad_kafka_config}
        with pytest.warns(MissingKeysWarning):
            with pytest.raises(BadKafkaConfigError):
                qm = FinkQueryManager(
                    config, target_lookup, parent_path=tmp_path, create_paths=False
                )

        bad_kafka_config = copy.deepcopy(kafka_config)
        _ = bad_kafka_config.pop("group.id")
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
        assert set(bad_kafka_config.keys()) == set(["username", "group.id", "topics"])
        config = {"kafka_config": bad_kafka_config}
        with pytest.warns(MissingKeysWarning):
            with pytest.raises(BadKafkaConfigError):
                qm = FinkQueryManager(
                    config, target_lookup, parent_path=tmp_path, create_paths=False
                )

        bad_kafka_config = copy.deepcopy(kafka_config)
        _ = bad_kafka_config.pop("topics")
        assert set(bad_kafka_config.keys()) == set(
            ["username", "group.id", "bootstrap.servers"]
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

        # with alerts as None...
        MCNone = create_mock_consumer(None)
        assert MCNone.alert_list is None

        MockConsumer = create_mock_consumer(alert_list)
        with MockConsumer([], []) as mc:
            assert mc.index == 0
            poll_result = mc.poll()
            assert isinstance(poll_result, tuple)
            assert len(poll_result) == 3
            assert mc.index == 1
            assert np.isclose(poll_result[1]["candidate"]["magpsf"], 19.0)

    def test__fink_listen_for_alerts(self, fink_qm, alert_list, monkeypatch):
        MockConsumer = create_mock_consumer(alert_list)
        # MockConsumer should be class rather than func, as needs .poll() method.

        assert hasattr(fink.AlertConsumer, "consume")  # Before patch
        with monkeypatch.context() as m:
            # Patch the MockConsumer into the fink module
            m.setattr("aas2rto.query_managers.fink.AlertConsumer", MockConsumer)
            assert not hasattr(fink.AlertConsumer, "consume")  # is patched!
            assert len(fink.AlertConsumer.alert_list) == 15  # is patched!

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
                "aas2rto.query_managers.fink.FinkQuery.query_and_collate_latests",
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
                "aas2rto.query_managers.fink.FinkQuery.query_and_collate_latests",
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
                "aas2rto.query_managers.fink.FinkQuery.query_and_collate_latests",
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
                "aas2rto.query_managers.fink.FinkQuery.query_and_collate_latests",
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
                "aas2rto.query_managers.fink.FinkQuery.query_and_collate_latests",
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
        self, fink_qm: FinkQueryManager, fink_lc: pd.DataFrame, target_list, monkeypatch
    ):
        fink_qm.create_paths()

        def mock_query(objectId, **kwargs):
            # objectId because this mocks FINK's internals
            objectId_list = objectId.split(",")
            if "ZTF00abc" in objectId_list:
                return fink_lc.copy()
            else:
                return pd.DataFrame()

        fink_id_list = [t.target_id for t in target_list]

        with monkeypatch.context() as m:
            m.setattr("aas2rto.query_managers.fink.FinkQuery.query_objects", mock_query)
            success, failed = fink_qm.perform_lightcurve_queries(
                fink_id_list, chunk_size=2
            )

        assert set(success) == set(["ZTF00abc", "ZTF01ijk", "ZTF02xyz"])
        assert set(failed) == set()

        exp_ZTF00abc_lc_file = fink_qm.data_path / "lightcurves/ZTF00abc.csv"
        assert exp_ZTF00abc_lc_file.exists()
        ZTF00abc_lc = pd.read_csv(exp_ZTF00abc_lc_file)
        assert len(ZTF00abc_lc) == 7

        exp_ZTF01ijk_lc_file = fink_qm.data_path / "lightcurves/ZTF01ijk.csv"
        assert exp_ZTF01ijk_lc_file.exists()
        ZTF01ijk_lc = pd.read_csv(exp_ZTF01ijk_lc_file)
        assert ZTF01ijk_lc.empty

        exp_ZTF02xyz_lc_file = fink_qm.data_path / "lightcurves/ZTF02xyz.csv"
        assert exp_ZTF02xyz_lc_file.exists()
        ZTF02xyz_lc = pd.read_csv(exp_ZTF01ijk_lc_file)
        assert ZTF02xyz_lc.empty

    def test__query_fail_does_not_crash(
        self, fink_qm: FinkQueryManager, fink_lc: pd.DataFrame, target_list, monkeypatch
    ):
        fink_qm.create_paths()

        def mock_query(objectId, **kwargs):
            # objectId because this mocks FINK server's internals.
            objectId_list = objectId.split(",")
            if all([oId.startswith("ZTF") for oId in objectId_list]):
                if "ZTF00abc" in objectId_list:
                    return fink_lc.copy()
                else:
                    return pd.DataFrame()
            raise ValueError

        fink_id_list = [t.target_id for t in target_list] + ["CAND_01", "CAND_02"]

        with monkeypatch.context() as m:
            m.setattr("aas2rto.query_managers.fink.FinkQuery.query_objects", mock_query)
            success, failed = fink_qm.perform_lightcurve_queries(
                fink_id_list, chunk_size=2
            )
            # will have chunks (ZTF00, ZTF01), (ZTF02, CAND_01), (CAND_02)

        assert set(success) == set(["ZTF00abc", "ZTF01ijk"])
        assert set(failed) == set(["ZTF02xyz", "CAND_01", "CAND_02"])

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

    def test__load_single_lightcurve_no_crash_on_empty_file(self, fink_qm, fink_lc):
        fink_qm.create_paths()

        fink_lc_filepath = fink_qm.get_lightcurve_file("ZTF01ijk")
        with open(fink_lc_filepath, "w+") as f:
            f.writelines([])

        assert fink_lc_filepath.exists()

        with pytest.raises(pd.errors.EmptyDataError):
            pd.read_csv(fink_lc_filepath)

        lc = fink_qm.load_single_lightcurve("ZTF01ijk")
        assert lc is None

    def test__load_target_lightcurves(
        self, fink_qm: FinkQueryManager, target_list, fink_lc
    ):
        fink_qm.create_paths()

        for target in target_list:
            fink_qm.add_target(target)

        lc_path = fink_qm.get_lightcurve_file("ZTF00abc")
        fink_lc.to_csv(lc_path, index=False)

        loaded, missing = fink_qm.load_target_lightcurves()

        assert set(loaded) == set(["ZTF00abc"])
        assert set(missing) == set(["ZTF01ijk", "ZTF02xyz"])


class Test__AlertsIntegrated:
    def test__load_missing_alerts(
        self, fink_qm: FinkQueryManager, fink_lc, target_list, polled_alerts
    ):
        fink_qm.create_paths()
        fink_qm.add_target(target_list[0])

        processed_alerts = fink_qm.process_alerts(polled_alerts[:5])

        # Modify lc slightly, so we can test only the missing ones are read in.
        more_alerts = pd.DataFrame(processed_alerts[:2])
        fink_lc = pd.concat([fink_lc, more_alerts], ignore_index=True)
        assert len(fink_lc) == 9
        assert 2300010000200005011 in fink_lc["candid"].values
        assert fink_lc.iloc[7].candid == 2300010000200005011
        assert fink_lc.iloc[8].candid == 2300010000200005012

        lc_path = fink_qm.get_lightcurve_file("ZTF00abc")
        fink_lc.to_csv(lc_path, index=False)

        fink_qm.load_target_lightcurves()

        assert len(processed_alerts) == 5

        # We expect all five alerts have been written during processing, but
        # only expect 5013, 5014, 5015 to be re-read.
        alert_path = fink_qm.data_path / "alerts/ZTF00abc/"
        exp_alert00_path = alert_path / "2300010000200005011.json"
        exp_alert01_path = alert_path / "2300010000200005012.json"
        exp_alert02_path = alert_path / "2300010000200005013.json"
        exp_alert03_path = alert_path / "2300010000200005014.json"
        exp_alert04_path = alert_path / "2300010000200005015.json"

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

    def test__integrate_missing_alerts(
        self, fink_qm: FinkQueryManager, fink_lc, target_list, polled_alerts, tmp_path
    ):

        fink_qm.create_paths()

        T1 = target_list[0]
        for target in target_list:
            fink_qm.add_target(target)

        lc_path = fink_qm.get_lightcurve_file("ZTF00abc")
        print(fink_lc)
        fink_lc.to_csv(lc_path, index=False)
        fink_qm.load_target_lightcurves()  # loads the target_data.

        # As a reminder...
        fink_data = T1.target_data["fink"]
        assert len(fink_data.lightcurve) == 7
        assert len(fink_data.non_detections) == 3  # with candid == -1
        assert len(fink_data.badqual) == 1  # with candid == -1
        assert len(fink_data.detections) == 3  # candid 5004, 5005, 5006
        candid_base = 23000_10000_20000_0000
        exp_candids = candid_base + np.array([5005, 5006, 5007])
        assert set(fink_data.detections["candid"].values) == set(exp_candids)

        processed_alerts = fink_qm.process_alerts(polled_alerts[:5])
        integ = fink_qm.integrate_alerts(save_lightcurves=True)  # This is by default.

        assert set(integ) == set(["ZTF00abc"])

        fink_data = T1.target_data["fink"]
        assert len(fink_data.lightcurve) == 12  # 7 + 5 new alerts
        assert len(fink_data.non_detections) == 3
        assert len(fink_data.badqual) == 1
        assert len(fink_data.detections) == 8  # 3 + 5

        new_exp_candids = candid_base + np.array(
            [5005, 5006, 5007, 5011, 5012, 5013, 5014, 5015]
        )
        assert set(fink_data.detections["candid"].values) == set(new_exp_candids)

        # check updated lightcurve has been saved.
        lc_load = pd.read_csv(lc_path)
        assert len(lc_load) == 12


class Test__Cutouts:
    def test__load_cutouts_(self, fink_qm: FinkQueryManager, target_list, fink_lc):
        fink_qm.create_paths()

        T1 = target_list[0]
        T1.target_data["fink"] = TargetData(lightcurve=fink_lc)
        T2 = target_list[1]
        T2.target_data["fink"] = TargetData(lightcurve=fink_lc)
        T2.target_data["fink"].meta["cutouts_candid"] = 23000_10000_20000_5006
        T3 = target_list[2]
        T3.target_data["fink"] = TargetData(lightcurve=fink_lc)
        T3.target_data["fink"].meta["cutouts_candid"] = 23000_10000_20000_5007

        assert set(T1.target_data["fink"].cutouts.keys()) == set()
        assert set(T2.target_data["fink"].cutouts.keys()) == set()

        for target in target_list:
            fink_qm.add_target(target)

        T1_cutouts_filepath = fink_qm.get_cutouts_file(
            "ZTF00abc", 23000_10000_20000_5007  # T1 has no cutouts_candid yet!
        )
        T1_cutouts = gen_mock_cutouts(T1_cutouts_filepath)
        assert T1_cutouts_filepath.exists()

        T2_cutouts_filepath = fink_qm.get_cutouts_file(
            "ZTF01ijk", 23000_10000_20000_5007  # Current is 2nd last row, this is last.
        )
        T2_cutouts = gen_mock_cutouts(T2_cutouts_filepath)
        assert T2_cutouts_filepath.exists()

        T3_cutouts_filepath = fink_qm.get_cutouts_file(
            "ZTF02xyz", 23000_10000_20000_5007  # Current is already == this.
        )
        T3_cutouts = gen_mock_cutouts(T3_cutouts_filepath)
        assert T3_cutouts_filepath.exists()

        loaded_cutouts = fink_qm.load_cutouts()

        assert set(loaded_cutouts) == set(["ZTF00abc", "ZTF01ijk"])

        exp_keys = set(["science", "difference", "template"])
        assert set(T1.target_data["fink"].cutouts) == exp_keys
        assert set(T2.target_data["fink"].cutouts) == exp_keys
        assert set(T3.target_data["fink"].cutouts) == set()

        assert T1.target_data["fink"].meta["cutouts_candid"] == 23000_10000_20000_5007
        assert T2.target_data["fink"].meta["cutouts_candid"] == 23000_10000_20000_5007
        assert T3.target_data["fink"].meta["cutouts_candid"] == 23000_10000_20000_5007


class Test__PerformAllTasks:

    def test__perform_all_tasks(
        self,
        fink_qm: FinkQueryManager,
        alert_list,
        fink_lc,
        fink_query_results,
        monkeypatch,
    ):
        t_ref = Time(60050.0, format="mjd")

        fink_qm.create_paths()

        assert set(fink_qm.target_lookup.keys()) == set()
        assert fink_qm.kafka_config["n_alerts"] == 10

        MockConsumer = create_mock_consumer(alert_list)

        def mock_lc_query(objectId, **kwargs):
            # objectId as arg name because this mocks FINK server's internals.
            objectId_list = objectId.split(",")
            if "ZTF00abc" in objectId_list:
                return fink_lc.copy()
            else:
                return pd.DataFrame()

        def mock_latests_query(*args, **kwargs):
            return fink_query_results

        with monkeypatch.context() as m:
            m.setattr("aas2rto.query_managers.fink.AlertConsumer", MockConsumer)
            m.setattr(
                "aas2rto.query_managers.fink.FinkQuery.query_objects",
                mock_lc_query,
            )
            m.setattr(
                "aas2rto.query_managers.fink.FinkQuery.query_and_collate_latests",
                mock_latests_query,
            )

            fink_qm.perform_all_tasks(t_ref=t_ref)

        assert set(fink_qm.target_lookup.keys()) == set(
            ["ZTF00abc", "ZTF01ijk", "ZTF02xyz"]
        )

        exp_results_file = fink_qm.get_query_results_file("interesting_topic")
        assert exp_results_file.exists()

        T1_alerts_dir = fink_qm.alerts_path / "ZTF00abc"
        T1_alerts = list(T1_alerts_dir.glob("*.json"))
        assert len(T1_alerts) == 10  # see kafka config["n_alerts"]

        exp_lc_path = fink_qm.get_lightcurve_file("ZTF00abc")
        assert exp_lc_path.exists()

        T1 = fink_qm.target_lookup["ZTF00abc"]

        assert (
            len(T1.target_data["fink"].lightcurve) == 7 + 10
        )  # initial length + all alerts.


class Test__ApplyMessengerUpdates:

    def test__normal_behaviour(self, fink_qm: FinkQueryManager, target_list):

        for target in target_list:
            fink_qm.add_target(target)

        for objectId, target in fink_qm.target_lookup.items():
            assert not target.updated

        alerts = [
            dict(objectId="ZTF00abc", jd=2_460_000, topic="topic_01"),
            dict(objectId="ZTF00abc", jd=2_460_001, topic="topic_02"),  # new jd
            dict(objectId="ZTF01ijk", jd=2_460_000, topic="topic_01"),
        ]

        fink_qm.apply_messenger_updates(alerts)

        assert fink_qm.target_lookup["ZTF00abc"].updated is True
        assert fink_qm.target_lookup["ZTF01ijk"].updated is True
        assert fink_qm.target_lookup["ZTF02xyz"].updated is False

        assert len(fink_qm.target_lookup["ZTF00abc"].update_messages) == 2
        assert len(fink_qm.target_lookup["ZTF01ijk"].update_messages) == 1
        assert len(fink_qm.target_lookup["ZTF02xyz"].update_messages) == 0


@pytest.fixture
def mock_data(fink_lc):
    data = [{f"i:{k}": v for k, v in row.items()} for ii, row in fink_lc.iterrows()]
    return data


@pytest.fixture
def mock_json_data(mock_data):
    return json.dumps(mock_data)


class Elapsed:
    def __init__(self, elapsed):
        self.elapsed = elapsed

    def total_seconds(self):
        return self.elapsed


class MockResponse:
    def __init__(self, data: str, status_code, elapsed=5.0):
        self.content = data.encode("utf-8")
        self.status_code = status_code = status_code
        self.elapsed = Elapsed(elapsed)


class Test__FinkQuery:

    def test__init(self):
        fq = FinkQuery()

    def test__fix_dict_keys(self):

        d = {"v:param_01": 0, "i:param_02": 10, "param_03": 100}

        FinkQuery.fix_dict_keys_inplace(d)

        assert set(d.keys()) == set(["param_01", "param_02", "param_03"])
        assert d["param_01"] == 0  # params are unchanged.
        assert d["param_02"] == 10
        assert d["param_03"] == 100

    def test__process_kwargs(self):

        d = {"param_01_": 1, "param_02": 10}

        d_mod = FinkQuery.process_kwargs(**d)
        assert set(d_mod.keys()) == set(["param_01", "param_02"])

    def test__process_data(self, mock_data):

        df_keys = "candid jd objectId mag magerr diffmaglim ra dec tag".split()
        exp_keys = [f"i:{k}" for k in df_keys]
        assert set(mock_data[0].keys()) == set(exp_keys)

        data_01 = FinkQuery.process_data(mock_data, fix_keys=False, return_df=True)
        assert isinstance(data_01, pd.DataFrame)
        assert set(data_01.columns) == set(exp_keys)

        data_02 = FinkQuery.process_data(mock_data, fix_keys=True, return_df=False)
        assert isinstance(data_02, list)
        assert isinstance(data_02[0], dict)
        assert set(data_02[0].keys()) == set(df_keys)

        data_03 = FinkQuery.process_data(mock_data, fix_keys=True, return_df=True)

    def test__process_response(self, mock_json_data):

        res = MockResponse(mock_json_data, 200)

        result = FinkQuery.process_response(res)

        df_keys = "candid jd objectId mag magerr diffmaglim ra dec tag".split()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 7
        assert set(result.columns) == set(df_keys)

    def test__process_response_bad_status(self, mock_json_data):

        res = MockResponse(mock_json_data, 404)

        with pytest.raises(FinkQueryError):
            result = FinkQuery.process_response(res)


class MockQueryLatests:
    __name__ = "mock_query_latests"

    def __init__(self, data: list):
        self.data = data
        self.ii = 0

    def __call__(self, *args, **kwargs):
        result = self.data[self.ii]
        self.ii = self.ii + 1
        return result


class Test__FinkQueryLatests:
    def test__query_and_collate_latests(
        self, fink_query_results, fink_updated_query_results, monkeypatch
    ):

        assert len(fink_query_results) == 7
        assert len(fink_updated_query_results) == 6
        assert set(fink_query_results["objectId"]) == set(
            ["ZTF00abc", "ZTF01ijk", "ZTF02xyz"]
        )
        assert set(fink_updated_query_results["objectId"]) == set(
            ["ZTF01ijk", "ZTF02xyz", "ZTF03lmn"]
        )

        mock_query_latests = MockQueryLatests(
            [fink_query_results, fink_updated_query_results, None]
        )

        with monkeypatch.context() as m:
            m.setattr(
                "aas2rto.query_managers.fink.FinkQuery.query_latests",
                mock_query_latests,
            )
            assert FinkQuery.query_latests.__name__ == "mock_query_latests"

            results = FinkQuery.query_and_collate_latests("some_class")

        assert len(results) == 13
