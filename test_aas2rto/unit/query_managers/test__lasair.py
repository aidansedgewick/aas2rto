import copy
import json

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time


import pytest

from aas2rto.target import Target, TargetData

from aas2rto.query_managers import lasair
from aas2rto.query_managers.lasair import (
    process_lasair_lightcurve,
    target_from_lasair_lightcurve,
    target_from_lasair_alert,
    LasairQueryManager,
)
from aas2rto.exc import (
    BadKafkaConfigError,
    UnexpectedKeysWarning,
    MissingKeysWarning,
)


@pytest.fixture
def lasair_nondet_rows():
    """
    candid, jd, objectId, mag, magerr, diffmaglim, ra, dec, tag
    """

    data_rows = [
        (2460001.5, 1, 17.0),
        (2460002.5, 2, 17.0),
        (2460003.5, 3, 17.0),
    ]
    keys = "jd fid diffmaglim".split()

    rows = []
    for data in data_rows:
        row = {k: v for k, v in zip(keys, data)}
        rows.append(row)
    return rows


@pytest.fixture
def lasair_det_rows():

    base_id = 23000_10000_20000_5000  # integer NOT str
    data_rows = [
        # candid      jd       magpsf sigmapsf magnr ra dec  fid
        (base_id + 4, 2460004.5, 18.5, 0.5, 19.0, 30.0, 45.0, 1),
        (base_id + 5, 2460005.5, 18.2, 0.1, 19.0, 30.0, 45.0, 1),
        (base_id + 6, 2460006.5, 18.3, 0.1, 19.5, 30.0, 45.0, 2),
        (base_id + 7, 2460007.5, 18.0, 0.1, 19.5, 30.0, 45.0, 1),
    ]
    keys = "candid jd magpsf sigmapsf magnr ra dec fid".split()

    rows = []
    for data in data_rows:
        row = {k: v for k, v in zip(keys, data)}
        rows.append(row)
    return rows


@pytest.fixture
def lasair_lc(lasair_nondet_rows, lasair_det_rows):
    lasair_rows = lasair_det_rows + lasair_nondet_rows
    lasair_lc = {"objectId": "ZTF00abc", "candidates": lasair_rows}
    return lasair_lc


@pytest.fixture
def kafka_config():
    return {
        "host": "test_server.ac.uk:8000",
        "group_id": "test_group",
        "topics": ["interesting_topic"],
    }


@pytest.fixture
def query_parameters():
    return {}


@pytest.fixture
def object_queries():
    return {}


@pytest.fixture
def lasair_config(kafka_config, query_parameters, object_queries):
    return {
        "kafka_config": kafka_config,
        "query_parameters": query_parameters,
        "object_queries": object_queries,
        "client_token": "test_token",
    }


@pytest.fixture
def lasair_qm(lasair_config, target_lookup, tmp_path):
    return LasairQueryManager(
        lasair_config, target_lookup, parent_path=tmp_path, create_paths=False
    )


@pytest.fixture
def alert_base():
    return dict(objectId="ZTF00abc", cdsxmatch="Unknown")


@pytest.fixture
def alert_list(alert_base):
    """
    Make list of fake alerts. each one is a little brighter than the last.
    """

    l = []
    for ii in range(15):
        t_ref = Time(60010.0 + ii, format="mjd")  # mjd60010 = 7-mar-23
        alert = copy.deepcopy(alert_base)
        alert["candid"] = 23000_10000_20000_5010 + (ii + 1)
        alert["timestamp"] = t_ref.strftime("%y%m%d_%H%M%S")
        alert["magpsf"] = 19.0 - 0.1 * ii
        alert["jd"] = t_ref.jd
        alert["ra"] = 30.0
        alert["dec"] = 45.0
        l.append(alert)
    return l


@pytest.fixture
def target_lookup():
    return {}


@pytest.fixture
def lasair_qm(lasair_config, target_lookup, tmp_path):
    return LasairQueryManager(
        lasair_config, target_lookup, parent_path=tmp_path, create_paths=False
    )


@pytest.fixture
def target_list():
    return [
        Target("ZTF00abc", ra=30.0, dec=45.0),
        Target("ZTF01ijk", ra=45.0, dec=45.0),
        Target("ZTF02xyz", ra=60.0, dec=45.0),
    ]


class MockMessage:
    def __init__(self, alert):
        self.alert = alert

    def value(self):
        return json.dumps(self.alert)  # for

    def error(self):
        return False


def create_mock_consumer(alerts):
    class MockConsumer:
        alert_list = alerts
        test_attr = True

        def __init__(self, host, group_id, topic):
            self.index = 0

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            pass

        def poll(self, timeout=None):
            if self.alert_list is None:
                raise ValueError("you should set mock_consumer.alert_list")

            if self.index >= len(self.alert_list):
                return None
            result = self.alert_list[self.index]
            self.index = self.index + 1
            return MockMessage(result)

    return MockConsumer


def create_mock_client(lc):
    class MockClient:
        lightcurve = lc

        def __init__(self, token):
            self.token = token

        def lightcurves(self, objectId_list):
            query_return = []
            for objectId in objectId_list:
                lc = copy.deepcopy(self.lightcurve)
                lc["objectId"] = objectId
                query_return.append(lc)
            return query_return

    return MockClient


class Test__ProcessLasairLightcurve:

    def test__normal_behaviour(self, lasair_lc):
        t_ref = Time(60000.0, format="mjd")

        processed_lc = process_lasair_lightcurve(lasair_lc, t_ref=t_ref)

        assert isinstance(processed_lc, pd.DataFrame)

        assert len(processed_lc) == 7
        assert all(processed_lc["jd"].values[:-1] < processed_lc["jd"].values[1:])

        assert processed_lc.iloc[0].candid == -1
        assert processed_lc.iloc[3].candid == 23000_10000_20000_5004

        assert len(processed_lc[processed_lc["tag"] == "valid"]) == 4
        assert len(processed_lc[processed_lc["tag"] == "upperlim"]) == 3

        exp_existing_cols = (
            "candid jd magpsf sigmapsf magnr ra dec fid diffmaglim".split()
        )
        exp_new_cols = "objectId mjd tag candid_str".split()

        assert set(processed_lc.columns) == set(exp_existing_cols + exp_new_cols)

    def test__only_detections(self, lasair_det_rows):

        lc = {"objectId": "ZTF00abc", "candidates": lasair_det_rows}

        proc_lc = process_lasair_lightcurve(lc)
        assert len(proc_lc) == 4

    def test__date_col_added_empty_lc(self, lasair_lc):
        lasair_lc["candidates"] = []

        proc_lc = process_lasair_lightcurve(lasair_lc)
        assert proc_lc.empty
        exp_columns = "jd mjd objectId".split()
        assert set(proc_lc.columns) == set(exp_columns)

    def test__is_sorted(self, lasair_lc):

        scramble_idx = [5, 4, 1, 0, 3, 2, 6]

        lasair_lc["candidates"] = [lasair_lc["candidates"][ii] for ii in scramble_idx]
        assert len(lasair_lc["candidates"]) == 7  # Nothing modified.

        proc_lc = process_lasair_lightcurve(lasair_lc)

        assert all(proc_lc["jd"].values[:-1] < proc_lc["jd"].values[1:])
        assert len(proc_lc) == 7


class Test__TargetFromLightcurve:

    def test__normal_behaviour(self, lasair_lc):
        lc = process_lasair_lightcurve(lasair_lc)

        target = target_from_lasair_lightcurve(lc)
        assert isinstance(target, Target)

        assert np.isclose(target.ra, 30.0)
        assert np.isclose(target.dec, 45.0)

        assert "lasair" in target.target_data
        lasair_data = target.target_data["lasair"]
        assert isinstance(lasair_data, TargetData)

        assert isinstance(lasair_data.lightcurve, pd.DataFrame)
        assert len(lasair_data.lightcurve) == 7

    def test__unprocessed_lightcurve(self, lasair_lc):

        target = target_from_lasair_lightcurve(lasair_lc)

        assert isinstance(target, Target)

        assert "lasair" in target.target_data
        lasair_data = target.target_data["lasair"]
        assert isinstance(lasair_data, TargetData)

        assert isinstance(lasair_data.lightcurve, pd.DataFrame)
        assert len(lasair_data.lightcurve) == 7


class Test__TargetFromAlert:
    def test__target_from_alert(self, alert_list):

        target = target_from_lasair_alert(alert_list[0])

        assert isinstance(target, Target)
        assert np.isclose(target.ra, 30.0)
        assert np.isclose(target.dec, 45.0)

        assert isinstance(target.coord, SkyCoord)

    def test__target_from_alert_no_coords(self, alert_list):

        alert = alert_list[0]
        _ = alert.pop("ra")
        _ = alert.pop("dec")

        target = target_from_lasair_alert(alert)
        assert isinstance(target, Target)

        assert target.ra is None
        assert target.dec is None
        assert target.coord is None


class Test__LasairQueryManagerInit:

    def test__lasair_qm_init(self, lasair_config, target_lookup, tmp_path):
        qm = LasairQueryManager(lasair_config, target_lookup, parent_path=tmp_path)

        assert isinstance(qm.lasair_config, dict)
        assert "query_parameters" in qm.lasair_config
        assert "kafka_config" in qm.lasair_config

        assert isinstance(qm.target_lookup, dict)

        assert qm.client_token == "test_token"

        assert isinstance(qm.kafka_config, dict)

        expected_query_kws = [
            "object_query_interval",
            "max_failed_queries",
            "lightcurve_update_interval",
        ]
        assert set(qm.query_parameters.keys()) == set(expected_query_kws)

    def test__lasair_qm_no_kafka(self, target_lookup, tmp_path):
        config = {}

        qm = LasairQueryManager(
            config, target_lookup, parent_path=tmp_path, create_paths=False
        )
        assert isinstance(qm, LasairQueryManager)

    # def test__warn_on_missing_config(self, lasair_config, target_lookup, tmp_path):
    #    pass

    def test__warn_on_bad_lasair_query_params(
        self, lasair_config, target_lookup, tmp_path
    ):
        lasair_config["query_parameters"]["bad_parameter"] = 1000
        with pytest.warns(UnexpectedKeysWarning):
            qm = LasairQueryManager(
                lasair_config, target_lookup, parent_path=tmp_path, create_paths=False
            )

    def test__bad_fink_kafka_config(self, kafka_config, target_lookup, tmp_path):
        bad_kafka_config = copy.deepcopy(kafka_config)
        _ = bad_kafka_config.pop("group_id")
        assert set(bad_kafka_config.keys()) == set(["host", "topics"])
        config = {"kafka_config": bad_kafka_config, "client_token": 1234}
        with pytest.warns(MissingKeysWarning):
            with pytest.raises(BadKafkaConfigError):
                qm = LasairQueryManager(
                    config, target_lookup, parent_path=tmp_path, create_paths=False
                )

        bad_kafka_config = copy.deepcopy(kafka_config)
        _ = bad_kafka_config.pop("host")
        assert set(bad_kafka_config.keys()) == set(["group_id", "topics"])
        config = {"kafka_config": bad_kafka_config, "client_token": 1234}
        with pytest.warns(MissingKeysWarning):
            with pytest.raises(BadKafkaConfigError):
                qm = LasairQueryManager(
                    config, target_lookup, parent_path=tmp_path, create_paths=False
                )

        bad_kafka_config = copy.deepcopy(kafka_config)
        _ = bad_kafka_config.pop("topics")
        assert set(bad_kafka_config.keys()) == set(["host", "group_id"])
        config = {"kafka_config": bad_kafka_config, "client_token": 1234}
        with pytest.warns(MissingKeysWarning):
            with pytest.raises(BadKafkaConfigError):
                qm = LasairQueryManager(
                    config, target_lookup, parent_path=tmp_path, create_paths=False
                )


class Test__LasairAlertStreams:
    def test__lasair_mock_consumer_behaviour(self, alert_list, monkeypatch):

        MockConsumer = create_mock_consumer(alert_list)
        with monkeypatch.context() as m:
            assert len(MockConsumer.alert_list) == 15
            with MockConsumer("localhost:8080", "test_123", "topic_001") as mc:
                assert mc.index == 0
                poll_result = mc.poll()
                assert isinstance(poll_result, MockMessage)
                assert isinstance(poll_result.alert, dict)
                assert mc.index == 1  # Consumer has moved on an index!
                assert np.isclose(poll_result.alert["magpsf"], 19.0)

    def test__lasair_listen_for_alerts(
        self, lasair_qm: LasairQueryManager, alert_list, monkeypatch
    ):

        MockConsumer = create_mock_consumer(alert_list)
        assert not hasattr(lasair.lasair_consumer, "test_attr")  # Before patch!
        with monkeypatch.context() as m:
            # Patch the alert list onto the MockConsumer

            # Patch the MockConsumer into the lasair module
            m.setattr(
                "aas2rto.query_managers.lasair.lasair_consumer", MockConsumer
            )
            assert hasattr(lasair.lasair_consumer, "test_attr")  # is patched!

            alerts = lasair_qm.listen_for_alerts()
            assert len(alerts) == 10  # Listen only for the correct number of alerts.
            print(alerts[0])
            assert isinstance(alerts[0], dict)
            for alert_ii in alerts:
                assert alert_ii["topic"] == "interesting_topic"
            # Alerts start at [...]_5011
            assert alerts[0]["candid"] == 23000_10000_20000_5011
            assert alerts[1]["candid"] == 23000_10000_20000_5012
            assert alerts[5]["candid"] == 23000_10000_20000_5016
            assert alerts[9]["candid"] == 23000_10000_20000_5020
        assert not hasattr(lasair.lasair_consumer, "test_attr")  # Back to normal.

    def test__break_after_none(
        self, lasair_qm: LasairQueryManager, alert_list, monkeypatch
    ):

        MockConsumer = create_mock_consumer(alert_list[:5])
        assert not hasattr(lasair.lasair_consumer, "test_attr")  # Before patch!

        with monkeypatch.context() as m:
            # Patch the alert list onto the MockConsumer
            assert len(MockConsumer.alert_list) == 5  # is patched!

            # Patch the MockConsumer into the lasair module
            m.setattr(
                "aas2rto.query_managers.lasair.lasair_consumer", MockConsumer
            )
            assert hasattr(lasair.lasair_consumer, "test_attr")  # is patched!

            alerts = lasair_qm.listen_for_alerts()
            assert len(alerts) == 5

    def test__process_alerts(self, lasair_qm: LasairQueryManager, alert_list):

        lasair_qm.create_paths()

        t_ref = Time(60000.0, format="mjd")

        processed_alerts = lasair_qm.process_alerts(
            alert_list[:5], t_ref=t_ref, save_alerts=True
        )

        for alert in processed_alerts:
            assert "alert_timestamp" in alert.keys()
        alert_dir = lasair_qm.parent_path / "lasair/alerts/ZTF00abc"
        assert alert_dir.exists()
        for candid_ext in [11, 12, 13, 14, 15]:
            candid = 23000_10000_20000_5000 + candid_ext
            alert_path = alert_dir / f"{candid}.json"
            assert alert_path.exists()

    def test__new_targets_from_lasair_alerst(
        self, lasair_qm: LasairQueryManager, alert_list
    ):

        assert len(lasair_qm.target_lookup) == 0

        mod_alerts = alert_list[:5]

        for ii, alert in enumerate(mod_alerts[:-1]):
            alert["objectId"] = f"ZTF00abc_{ii:02d}"
        mod_alerts[-1]["objectId"] = f"ZTF00abc_00"  # same as first one.

        added, existing = lasair_qm.new_targets_from_alerts(mod_alerts)
        assert len(added) == 4
        assert set(added) == set(
            ["ZTF00abc_00", "ZTF00abc_01", "ZTF00abc_02", "ZTF00abc_03"]
        )
        assert len(existing) == 1
        assert set(existing) == set(["ZTF00abc_00"])

        assert len(lasair_qm.target_lookup) == 4
        assert set(lasair_qm.target_lookup.keys()) == set(
            ["ZTF00abc_00", "ZTF00abc_01", "ZTF00abc_02", "ZTF00abc_03"]
        )


class Test__LasairLightcurveQueries:
    def test__get_lightcurves_to_query(
        self, lasair_qm: LasairQueryManager, target_list
    ):
        lasair_qm.create_paths()
        for target in target_list:
            lasair_qm.add_target(target)

        to_query = lasair_qm.get_lightcurves_to_query()
        assert len(to_query) == 3
        assert set(to_query) == set(["ZTF00abc", "ZTF01ijk", "ZTF02xyz"])

    def test__to_query_skips_recent_lcs(
        self, lasair_qm: LasairQueryManager, target_list, lasair_lc
    ):
        lasair_qm.create_paths()

        for target in target_list:
            lasair_qm.add_target(target)

        test_lc = process_lasair_lightcurve(lasair_lc)
        lc_filepath = lasair_qm.get_lightcurve_file("ZTF00abc")
        test_lc.to_csv(lc_filepath)

        assert lc_filepath.exists()

        to_query = lasair_qm.get_lightcurves_to_query()
        assert set(to_query) == set(["ZTF01ijk", "ZTF02xyz"])

    def test__perform_lightcurve_queries(
        self, lasair_qm: LasairQueryManager, lasair_lc, monkeypatch
    ):
        lasair_qm.create_paths()

        MockClient = create_mock_client(lasair_lc)

        objectId_list = ["ZTF00abc", "ZTF01ijk"]

        with monkeypatch.context() as m:
            m.setattr("aas2rto.query_managers.lasair.lasair_client", MockClient)
            success, failed = lasair_qm.perform_lightcurve_queries(objectId_list)

        assert set(success) == set(objectId_list)

        T1_exp_filepath = lasair_qm.get_lightcurve_file("ZTF00abc")
        assert T1_exp_filepath.exists()

        T2_exp_filepath = lasair_qm.get_lightcurve_file("ZTF01ijk")
        assert T2_exp_filepath.exists()
