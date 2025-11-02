import copy
import json
import pickle
import pytest
import time
from collections.abc import Iterator
from pathlib import Path
from typing import NewType

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from fink_client.consumer import AlertConsumer

from aas2rto.exc import UnexpectedKeysWarning
from aas2rto.query_managers.fink.fink_base import (
    FinkAlert,
    FinkBaseQueryManager,
    BadKafkaConfigError,
    updates_from_classifier_queries,
)
from aas2rto.query_managers.fink.fink_queries import FinkBaseQuery
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup


##===== Some classes to help testing here =====##

AlertStream = NewType("AlertStream", Iterator[FinkAlert])


def target_from_mock_alert(processed_alert: dict, t_ref: Time = None):
    target_id = processed_alert["target_id"]
    coord = SkyCoord(ra=processed_alert["ra"], dec=processed_alert["dec"], unit="deg")
    return Target(target_id, coord, source="fink_cool", t_ref=t_ref)


def process_single_mock_alert(
    alert_data: FinkAlert,
    t_ref: Time = None,
):
    topic, alert, key = alert_data
    processed_alert = copy.deepcopy(alert["candidate"])
    processed_alert["target_id"] = alert["target_id"]
    processed_alert["alert_id"] = alert["alert_id"]
    processed_alert["topic"] = topic
    return processed_alert


class MockFinkQuery:
    objects = None
    latests_query_and_collate = None


class FinkExampleQM(FinkBaseQueryManager):
    name = "fink_cool"
    id_resolving_order = ("cool_survey", "fink_cool")
    fink_query = MockFinkQuery
    target_id_key = "target_id"
    alert_id_key = "alert_id"

    def add_target_from_alert(self, processed_alert: dict, t_ref: Time = None):
        target_id = processed_alert["target_id"]
        if target_id in self.target_lookup:
            return None
        target = target_from_mock_alert(processed_alert, t_ref=t_ref)
        self.target_lookup.add_target(target)
        return target

    def process_single_alert(self, alert_data: FinkAlert, t_ref=None):
        return process_single_mock_alert(alert_data, t_ref=t_ref)

    def apply_updates_from_alert(self, processed_alert: dict, t_ref: Time = None):
        fink_id = processed_alert["target_id"]
        alert_id = processed_alert["alert_id"]
        target = self.target_lookup[fink_id]
        target.updated = True
        target.update_messages.append(f"{fink_id} alert with alert_id {alert_id}")

    def add_target_from_record(self, record: dict):
        fink_id = record["target_id"]
        exising_target = self.target_lookup.get(fink_id, None)
        if isinstance(exising_target, Target):
            return None
        coord = SkyCoord(record["ra"], record["dec"], unit="deg")
        target = Target(fink_id, coord, source="cool_survey")
        self.target_lookup.add_target(target)
        return target

    def process_fink_lightcurve(self, unprocessed_lc: pd.DataFrame):
        unprocessed_lc.insert(0, "new_col", 100.0)
        return unprocessed_lc

    def load_missing_alerts_for_target(self, fink_id: str):
        target = self.target_lookup.get(fink_id, None)
        if target is None:
            return None
        if target.target_id == "T01":
            target.update_messages.append("alerts integrated!")
            return target
        return None


class FinkBadQM(FinkBaseQueryManager):
    name = "fink_fail"
    id_resolving_order = ("fail", "fink_fail")


##===== Now define pytest fixtures =====##


@pytest.fixture
def alert_base():
    return {
        "target_id": "T101",
        "candidate": {"ra": 60.0, "dec": -45.0},
        "cutouts": {"diff": np.random.uniform(0.0, 1.0, (10, 10))},
    }


@pytest.fixture
def mock_alert_list(alert_base: dict, t_fixed: Time) -> list[FinkAlert]:
    alert_list = []
    mjd = t_fixed.mjd
    for ii in range(10):
        # need deep copy, otherwise nested dicts are just references to the same dict!
        alert = copy.deepcopy(alert_base)
        alert["alert_id"] = 1000 + ii
        alert["candidate"]["mjd"] = mjd + ii
        alert_list.append(("cool_sne", alert, "some_key"))
    return alert_list


@pytest.fixture
def mock_alert_stream(mock_alert_list: list[FinkAlert]) -> AlertStream:
    """return a Generator, so that an item is returned every time next() is called"""

    def alert_stream():
        for alert in mock_alert_list:
            yield alert

    return alert_stream()


@pytest.fixture
def lc_fink(lc_pandas: pd.DataFrame):
    lc = lc_pandas.copy()
    lc["jd"] = Time(lc["mjd"], format="mjd").jd
    lc.drop("mjd", axis=1, inplace=True)
    lc.rename({"candid": "alert_id"}, axis=1, inplace=True)
    lc.loc[:, "target_id"] = "T00"
    return lc


@pytest.fixture
def empty_classifier_results():
    return pd.DataFrame(columns="target_id alert_id lastdate".split())


@pytest.fixture
def mock_existing_classifier_results():
    date = "2025-02-25 12:00:00.000"
    rows = [
        ("T202", 240.0, 0.0, 202_1, date),
        ("T201", 210.0, 0.0, 201_1, date),
    ]
    return pd.DataFrame(rows, columns="target_id ra dec alert_id lastdate".split())


@pytest.fixture
def mock_new_classifier_results():
    old_date = "2025-02-25 12:00:00.000"
    new_date = "2025-02-26 12:00:00.000"
    rows = [
        ("T203", 210.0, 0.0, 203_1, new_date),  # brand new
        ("T201", 210.0, 0.0, 201_1, old_date),  # same as before
        ("T202", 240.0, 0.0, 202_2, new_date),  # updated from before: date & alert_id
    ]
    return pd.DataFrame(rows, columns="target_id ra dec alert_id lastdate".split())


@pytest.fixture
def patched_qm(
    fink_config: dict,
    tlookup: TargetLookup,
    tmp_path: Path,
    mock_alert_stream: AlertStream,
    lc_fink: pd.DataFrame,
    mock_new_classifier_results: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    def consumer_init(self, topics, *args, **kwargs):
        self.topics = topics  # Stop consumer trying to ping real FINK KAFKA servers.

    def consumer_exit(self, *args, **kwargs):
        pass  # AlertConsumer._conusumer() is set in __init__, and closed in __exit__

    def mock_poll(self, *args, **kwargs):
        if self.topics[0] == "cool_sne":
            try:
                return next(mock_alert_stream)
            except StopIteration as e:
                return (None, None, None)
        else:
            return (None, None, None)

    monkeypatch.setattr("fink_client.consumer.AlertConsumer.__init__", consumer_init)
    monkeypatch.setattr("fink_client.consumer.AlertConsumer.__exit__", consumer_exit)
    monkeypatch.setattr("fink_client.consumer.AlertConsumer.poll", mock_poll)

    def mock_objects(*args, **kwargs):
        target_id = kwargs.get("target_id", None)
        if target_id is None:
            raise ValueError("No 'target_id' in 'objects' query payload!")
        if "T00" in target_id:
            return lc_fink.copy()
        if "T_fail" in target_id:
            raise ValueError("failing, as asked!")
        if "T_sleep" in target_id:
            time.sleep(0.4)
        return pd.DataFrame(columns="target_id mjd alert_id".split())

    def mock_latests(*args, **kwargs):
        # need to check class_ with "_", as patched FQ does not properly prepare kwargs
        fink_class = kwargs.get("class_", None)
        if fink_class is None:
            raise ValueError("No 'class_' in 'latests' query payload!")
        if fink_class == "cool_sne":
            return mock_new_classifier_results
        return pd.DataFrame(columns="target_id lastdate fink_class".split())

    qm = FinkExampleQM(fink_config, tlookup, parent_path=tmp_path)
    qm.target_lookup["T00"].alt_ids["cool_survey"] = "T00"
    qm.target_lookup["T01"].alt_ids["cool_survey"] = "T01"

    monkeypatch.setattr(qm.fink_query, "objects", mock_objects)
    monkeypatch.setattr(qm.fink_query, "latests_query_and_collate", mock_latests)
    return qm


##===== Test helper objects work as designed! =====##


class Test__HelperFunctions:
    def test_process_mock_alert(self, mock_alert_list: list):
        # Act
        processed_alert = process_single_mock_alert(mock_alert_list[0])

        # Assert
        assert isinstance(processed_alert, dict)
        assert processed_alert["target_id"] == "T101"
        assert processed_alert["alert_id"] == 1000
        assert np.isclose(processed_alert["mjd"] - 60000.0, 0.0)

    def test__target_from_alert(self, mock_alert_list: list, t_fixed: Time):
        # Arrange
        processed_alert = process_single_mock_alert(mock_alert_list[0])

        # Act
        target = target_from_mock_alert(processed_alert, t_ref=t_fixed)

        # Assert
        assert isinstance(target, Target)
        assert target.target_id == "T101"

    def test__mock_alert_stream(self, mock_alert_stream: AlertStream):
        # Act
        alert = next(mock_alert_stream)

        # Assert
        assert isinstance(alert, tuple)
        assert len(alert) == 3
        assert alert[0] == "cool_sne"
        assert isinstance(alert[1], dict)
        assert alert[1]["target_id"] == "T101"  # etc.
        assert alert[1]["alert_id"] == 1000

    def test__mock_poll(self, mock_alert_stream: AlertStream):
        # Arrange
        def mock_poll():
            return next(mock_alert_stream)

        # Act
        alert = mock_poll()

        # Assert
        assert isinstance(alert, tuple)
        assert len(alert) == 3
        assert alert[1]["target_id"] == "T101"  # etc.

    def test__consumer_is_patched(self, patched_qm: FinkBaseQueryManager):
        # Act
        with AlertConsumer(["cool_sne"], {}) as consumer:
            alert_data = consumer.poll()

        # Assert
        assert isinstance(alert_data, tuple)
        assert alert_data[0] == "cool_sne"

    def test__fq_objects_is_patched(self, patched_qm: FinkBaseQueryManager):
        # Act
        lc = patched_qm.fink_query.objects(target_id="T00")

        # Assert
        assert len(lc) == 14  # etc.

    def test__fq_objects_empty_for_bad_target(self, patched_qm: FinkBaseQueryManager):
        # Act
        lc = patched_qm.fink_query.objects(target_id="T01")

        # Assert
        assert len(lc) == 0

    def test__fq_raise_on_request(self, patched_qm: FinkBaseQueryManager):
        # Act
        with pytest.raises(ValueError):
            lc = patched_qm.fink_query.objects(target_id="T_fail")

    def test__fq_objects_no_target_id_raises(self, patched_qm: FinkBaseQueryManager):
        # Act
        with pytest.raises(ValueError):
            lc = patched_qm.fink_query.objects()

    def test__fq_latests_is_patched(self, patched_qm: FinkBaseQueryManager):
        # Act
        results = patched_qm.fink_query.latests_query_and_collate(class_="cool_sne")

        # Assert
        assert len(results) == 3

    def test__fq_latests_empty_for_bad_class(self, patched_qm: FinkBaseQueryManager):
        # Act
        results = patched_qm.fink_query.latests_query_and_collate(class_="boring_sne")

        # Assert
        assert len(results) == 0
        assert set(results.columns) == set("target_id lastdate fink_class".split())

    def test__fq_latests_missing_class_raises(self, patched_qm: FinkBaseQueryManager):
        # Act
        with pytest.raises(ValueError):
            lc = patched_qm.fink_query.latests_query_and_collate()


class Test__ExampleQMFunctions:
    def test__add_target_from_alert(
        self, patched_qm: FinkBaseQueryManager, mock_alert_list: list, t_fixed: Time
    ):
        # Arrange
        processed_alert = process_single_mock_alert(mock_alert_list[0])

        # Act
        target = patched_qm.add_target_from_alert(processed_alert)

        # Assert
        assert isinstance(target, Target)

        assert set(patched_qm.target_lookup.keys()) == set(["T00", "T01", "T101"])

    def test__no_add_existing_target(
        self, patched_qm: FinkBaseQueryManager, mock_alert_list: list, t_fixed: Time
    ):
        # Arrange
        processed_alert = process_single_mock_alert(mock_alert_list[0])
        coord = SkyCoord(ra=120.0, dec=-60.0, unit="deg")
        t101 = Target("T101", coord)  # different coord! to check alert not added
        patched_qm.target_lookup.add_target(t101)

        # Act
        added_target = patched_qm.add_target_from_alert(processed_alert)

        # Assert
        assert added_target is None

        assert np.isclose(patched_qm.target_lookup["T101"].coord.ra.deg, 120.0)

    def test__process_alerts(
        self, patched_qm: FinkBaseQueryManager, mock_alert_list: list[FinkAlert]
    ):
        # Act
        processed_alert = patched_qm.process_single_alert(mock_alert_list[0])

        # Assert
        assert isinstance(processed_alert, dict)


##===== Actual testing starts here! =====##


class Test__FinkBaseQMInit:
    def test__fink_qm_init(
        self, fink_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Act
        qm = FinkExampleQM(fink_config, tlookup, parent_path=tmp_path)

        # Assert
        exp_directories = "lightcurves alerts query_results cutouts".split()
        assert set(qm.paths_lookup.keys()) == set(exp_directories)

        assert qm.paths_lookup["alerts"] == tmp_path / "fink_cool/alerts"
        assert qm.paths_lookup["alerts"].exists()
        assert qm.paths_lookup["cutouts"] == tmp_path / "fink_cool/cutouts"
        assert qm.paths_lookup["cutouts"].exists()
        assert qm.paths_lookup["lightcurves"] == tmp_path / "fink_cool/lightcurves"
        assert qm.paths_lookup["lightcurves"].exists()
        assert qm.paths_lookup["query_results"] == tmp_path / "fink_cool/query_results"
        assert qm.paths_lookup["query_results"].exists()

    def test__bad_arg_raises(
        self, fink_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Arrange
        fink_config["blah"] = 100.0

        # Act
        with pytest.raises(UnexpectedKeysWarning):
            qm = FinkBaseQueryManager(fink_config, tlookup, parent_path=tmp_path)


class Test__KafkaConfig:
    def test__no_kafka_is_ok(
        self, fink_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Arrange
        fink_config.pop("kafka")

        # Act
        qm = FinkExampleQM(fink_config, tlookup, parent_path=tmp_path)

        # Assert
        qm.kafka_config = None

    def test__missing_kafka_subkey_raises(
        self, fink_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Arrange
        fink_config["kafka"].pop("topics")

        # Act
        with pytest.raises(BadKafkaConfigError):
            qm = FinkExampleQM(fink_config, tlookup, parent_path=tmp_path)

    def test__str_topic_converted(
        self, fink_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Arrange
        fink_config["kafka"]["topics"] = "cool_sne"

        # Act
        qm = FinkExampleQM(fink_config, tlookup, parent_path=tmp_path)

        # Assert
        assert isinstance(qm.kafka_config["topics"], list)
        assert set(qm.kafka_config["topics"]) == set(["cool_sne"])

    def test__no_kafka_skip_alerts(
        self, fink_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Arrange
        fink_config.pop("kafka")
        qm = FinkExampleQM(fink_config, tlookup, parent_path=tmp_path)

        # Act
        alerts = qm.listen_for_alerts()

        # Assert
        assert len(alerts) == 0


class Test__PathMethods:
    def test__lc_path(self, patched_qm: FinkBaseQueryManager, tmp_path: Path):
        # Act
        lc_path = patched_qm.get_lightcurve_filepath("test_id")

        # Assert
        assert lc_path == tmp_path / "fink_cool/lightcurves/test_id.csv"

    def test__alert_path(self, patched_qm: FinkBaseQueryManager, tmp_path: Path):
        # Act
        lc_path = patched_qm.get_alert_filepath("test_id", "000")

        # Assert
        assert lc_path == tmp_path / "fink_cool/alerts/test_id/000.json"

    def test__cutouts_path(self, patched_qm: FinkBaseQueryManager, tmp_path: Path):
        # Act
        lc_path = patched_qm.get_cutouts_filepath("test_id", "000")

        # Assert
        assert lc_path == tmp_path / "fink_cool/cutouts/test_id/000.json"


class Test__MissingFunctionsRaise:
    def test__no_id_resolving_order_raises(self, tlookup: TargetLookup, tmp_path: Path):
        # Arrange
        class NoIdOrderFinkQM(FinkBaseQueryManager):
            name = "no_order"

        # Act
        with pytest.raises(NotImplementedError):
            qm = NoIdOrderFinkQM({}, tlookup, parent_path=tmp_path)

    def test__no_target_from_alert_raises(
        self, mock_alert_list: list, tlookup: TargetLookup, tmp_path: Path
    ):
        # Arrange
        alert = process_single_mock_alert(mock_alert_list[0])
        qm = FinkBadQM({}, tlookup, parent_path=tmp_path)

        # Act
        with pytest.raises(NotImplementedError):
            qm.add_target_from_alert(alert)

    def test__no_apply_updates_raises(
        self,
        mock_alert_list: list,
        tlookup: TargetLookup,
        t_fixed: Time,
        tmp_path: Path,
    ):
        # Arrange
        alert = process_single_mock_alert(mock_alert_list[0])
        target = target_from_mock_alert(alert, t_ref=t_fixed)
        qm = FinkBadQM({}, tlookup, parent_path=tmp_path)

        # Act
        with pytest.raises(NotImplementedError):
            qm.apply_updates_from_alert(alert)

    def test__no_process_alert_raises(
        self,
        mock_alert_list: list,
        tlookup: TargetLookup,
        t_fixed: Time,
        tmp_path: Path,
    ):
        # Arrange
        qm = FinkBadQM({}, tlookup, parent_path=tmp_path)

        # Act
        with pytest.raises(NotImplementedError):
            qm.process_single_alert(mock_alert_list[0])


class Test__ListenForAlerts:
    def test__listen_for_alerts(self, patched_qm: FinkBaseQueryManager, t_fixed: Time):
        # Act
        alerts = patched_qm.listen_for_alerts()

        # Assert
        len(alerts) == 10
        assert isinstance(alerts[0], dict)
        exp_keys = "target_id alert_id topic ra dec mjd".split()
        assert set(alerts[0].keys()) == set(exp_keys)
        assert alerts[0]["alert_id"] == 1000
        assert alerts[0]["target_id"] == "T101"
        assert np.isclose(alerts[0]["mjd"] - 60000.0, 0.0)

    def test__stop_after_nalerts(self, patched_qm: FinkBaseQueryManager):
        # Arrange
        patched_qm.config["n_alerts"] = 5

        # Act
        alerts = patched_qm.listen_for_alerts()

        # Assert
        assert len(alerts) == 5

    def test__no_relevant_alerts_breaks(self, patched_qm: FinkBaseQueryManager):
        # Arrange
        patched_qm.kafka_config["topics"] = ["boring_sne"]

        # Act
        processed_alerts = patched_qm.listen_for_alerts()

        # Assert
        assert isinstance(processed_alerts, list)
        assert len(processed_alerts) == 0


class Test__TargetsFromAlerts:
    def test__new_targets(self, patched_qm: FinkBaseQueryManager, t_fixed: Time):
        # Arrange
        processed_alerts = patched_qm.listen_for_alerts()

        # Act
        new_targets = patched_qm.new_targets_from_alerts(
            processed_alerts, t_ref=t_fixed
        )

        # Assert
        assert set(new_targets) == set(["T101"])

    def test__existing_targets_not_added(
        self, patched_qm: FinkBaseQueryManager, t_fixed: Time
    ):
        # Arrange
        processed_alerts = patched_qm.listen_for_alerts()
        coord = SkyCoord(ra=120.0, dec=-60.0, unit="deg")
        t101 = Target("T101", coord)  # different coord! to check alert not added
        patched_qm.target_lookup.add_target(t101)

        # Act
        new_targets = patched_qm.new_targets_from_alerts(processed_alerts)

        # Assert
        assert set(new_targets) == set()


class Test__ApplyUpdateMessages:
    def test__apply_updates(self, patched_qm: FinkBaseQueryManager):
        # Arrange
        basic_alert = {"target_id": "T00", "alert_id": 100}
        assert not patched_qm.target_lookup["T00"].updated
        assert set(patched_qm.target_lookup["T00"].update_messages) == set()

        # Act
        patched_qm.apply_update_messages([basic_alert])

        # Assert
        assert patched_qm.target_lookup["T00"].updated
        exp_msg = ["T00 alert with alert_id 100"]
        assert set(patched_qm.target_lookup["T00"].update_messages) == set(exp_msg)


class Test__GetFinkIdFromTarget:
    def test__get_fink_id(self, patched_qm: FinkBaseQueryManager):
        # Arrange
        target = Target("T901", SkyCoord(ra=0.0, dec=0.0, unit="deg"))
        target.alt_ids["cool_survey"] = "COOL_000"
        target.alt_ids["fink_cool"] = "FINK_COOL_000"
        patched_qm.target_lookup.add_target(target)

        # Act
        fink_id = patched_qm.get_fink_id_from_target(target)

        # Assert
        assert fink_id == "COOL_000"

    def test__none_from_no_alt_id(self, patched_qm: FinkBaseQueryManager):
        # Arrange
        target = Target("T901", SkyCoord(ra=0.0, dec=0.0, unit="deg"))
        patched_qm.target_lookup.add_target(target)

        # Act
        fink_id = patched_qm.get_fink_id_from_target(target)

        # Assert
        assert fink_id is None


class Test__UpdatesFromClassifierQueries:
    def test__no_existing(
        self,
        empty_classifier_results: pd.DataFrame,
        mock_new_classifier_results: pd.DataFrame,
    ):
        # Act
        updates = updates_from_classifier_queries(
            empty_classifier_results, mock_new_classifier_results, id_key="target_id"
        )

        # Assert
        assert isinstance(updates, pd.DataFrame)
        assert len(updates) == 3

        assert updates.iloc[0]["target_id"] == "T201"  # results are sorted correctly.
        assert updates.iloc[1]["target_id"] == "T202"
        assert updates.iloc[2]["target_id"] == "T203"

    def test__keep_updates_only(
        self,
        mock_existing_classifier_results: pd.DataFrame,
        mock_new_classifier_results: pd.DataFrame,
    ):
        # Act
        updates = updates_from_classifier_queries(
            mock_existing_classifier_results,
            mock_new_classifier_results,
            id_key="target_id",
        )

        # Assert
        assert isinstance(updates, pd.DataFrame)
        assert len(updates) == 2

        assert updates.iloc[0]["target_id"] == "T202"
        assert updates.iloc[0]["alert_id"] == 2022  # NEW row is returned

        assert updates.iloc[1]["target_id"] == "T203"

    def test__no_updated_rows_return_empty(
        self, mock_new_classifier_results: pd.DataFrame
    ):
        # Act
        updates = updates_from_classifier_queries(
            mock_new_classifier_results, mock_new_classifier_results, id_key="target_id"
        )  # use identical rows!

        # Assert
        assert isinstance(updates, pd.DataFrame)
        assert len(updates) == 0
        exp_cols = "target_id ra dec alert_id lastdate".split()
        assert set(updates.columns) == set(exp_cols)


class Test__FinkClassifierQueries:
    def test__no_existing(self, patched_qm: FinkBaseQueryManager):
        # Act
        updates = patched_qm.fink_classifier_queries()

        # Assert
        # the updates we got returned
        assert len(updates) == 3
        assert all([isinstance(u, dict) for u in updates])

        assert updates[0]["target_id"] == "T201"
        assert updates[1]["target_id"] == "T202"
        assert updates[2]["target_id"] == "T203"

        # the "combined" old and new results that were written out.
        exp_file = patched_qm.get_query_results_filepath("cool_sne")
        assert exp_file.exists()
        combined = pd.read_csv(exp_file)
        assert len(combined) == 3
        assert combined["target_id"].iloc[0] == "T201"
        assert combined["alert_id"].iloc[0] == 201_1
        assert combined["target_id"].iloc[1] == "T202"
        assert combined["alert_id"].iloc[1] == 202_2  # it's the new row.
        assert combined["target_id"].iloc[2] == "T203"
        assert combined["alert_id"].iloc[2] == 203_1

    def test__with_existing(
        self,
        patched_qm: FinkBaseQueryManager,
        mock_existing_classifier_results: pd.DataFrame,
    ):
        # Arrange
        existing_file = patched_qm.get_query_results_filepath("cool_sne")
        mock_existing_classifier_results.to_csv(existing_file, index=False)
        patched_qm.config["query_interval"] = -1.0  # so we definitely do the query

        # Act
        updates = patched_qm.fink_classifier_queries()

        # Assert
        assert len(updates) == 2  # we only return the updates
        assert all([isinstance(u, dict) for u in updates])
        assert updates[0]["target_id"] == "T202"
        assert updates[0]["alert_id"] == 202_2  # the NEW alert_id
        assert updates[1]["target_id"] == "T203"

        #
        exp_file = patched_qm.get_query_results_filepath("cool_sne")
        assert exp_file.exists()
        combined = pd.read_csv(exp_file)

        assert len(combined) == 3  # ...but we save everything!
        assert combined["target_id"].iloc[0] == "T201"
        assert combined["alert_id"].iloc[0] == 201_1
        assert combined["target_id"].iloc[1] == "T202"
        assert combined["alert_id"].iloc[1] == 202_2
        assert combined["target_id"].iloc[2] == "T203"
        assert combined["alert_id"].iloc[2] == 203_1

    def test__skip_recent_query(
        self,
        patched_qm: FinkBaseQueryManager,
        mock_existing_classifier_results: pd.DataFrame,
    ):
        # Arrange
        existing_file = patched_qm.get_query_results_filepath("cool_sne")
        mock_existing_classifier_results.to_csv(existing_file, index=False)
        patched_qm.config["query_interval"] = 1.0  # Should SKIP this time!

        # Act
        updates = patched_qm.fink_classifier_queries()

        # Assert
        assert len(updates) == 0

    def test__no_results_writes_empty_file(self, patched_qm: FinkBaseQueryManager):
        # Arrange
        patched_qm.config["fink_classes"] = ["boring_sne"]

        # Act
        updates = patched_qm.fink_classifier_queries()

        # Assert
        assert len(updates) == 0

        exp_path = patched_qm.get_query_results_filepath("boring_sne")
        combined = pd.read_csv(exp_path)

        assert len(combined) == 0
        assert set(combined.columns) == set(["target_id", "lastdate"])


class Test__NewTargetsFromQueryRecords:
    def test__new_targets(self, patched_qm: FinkBaseQueryManager):
        # Arrange
        query_records = patched_qm.fink_classifier_queries()

        # Act
        new_targets = patched_qm.new_targets_from_query_records(query_records)

        # Assert
        assert set(new_targets) == set(["T201", "T202", "T203"])
        assert "T201" in patched_qm.target_lookup
        assert "T202" in patched_qm.target_lookup
        assert "T203" in patched_qm.target_lookup


class Test__GetLCsToQuery:
    def test__skip_recent(
        self, patched_qm: FinkBaseQueryManager, lc_fink: pd.DataFrame
    ):
        # Arrange
        lc_filepath = patched_qm.get_lightcurve_filepath("T00")
        lc_fink.to_csv(lc_filepath, index=False)

        # Act
        to_query = patched_qm.get_lightcurves_to_query()

        # Assert
        assert set(to_query) == set(["T01"])

    def test__include_old(
        self, patched_qm: FinkBaseQueryManager, lc_fink: pd.DataFrame
    ):
        # Arrange
        lc_filepath = patched_qm.get_lightcurve_filepath("T00")
        lc_fink.to_csv(lc_filepath, index=False)
        t_future = Time.now() + 1.1 * u.day
        patched_qm.config["lightcurve_update_interval"] = 1.0

        # Act
        to_query = patched_qm.get_lightcurves_to_query(t_ref=t_future)

        # Assert
        assert set(to_query) == set(["T00", "T01"])


class Test__QueryLCs:
    def test__query_all(self, patched_qm: FinkBaseQueryManager):
        # Act
        success, missing, failed = patched_qm.query_lightcurves()

        # Assert
        assert set(success) == set(["T00"])
        assert set(missing) == set(["T01"])
        assert set(failed) == set()

        exp_lc_file = patched_qm.get_lightcurve_filepath("T00")
        assert exp_lc_file.exists()

        loaded_lc = pd.read_csv(exp_lc_file)
        assert len(loaded_lc) == 14
        assert "new_col" in loaded_lc.columns  # processing func is correctly called.

    def test__failing_queries_no_raise(self, patched_qm: FinkBaseQueryManager):
        # Arrange
        patched_qm.config["lightcurve_chunk_size"] = 1

        # Act
        success, missing, failed = patched_qm.query_lightcurves(
            fink_id_list=["T00", "T_fail"]
        )

        # Assert
        assert set(success) == set(["T00"])
        assert set(missing) == set()
        assert set(failed) == set(["T_fail"])

    def test__all_missing_counted(self, patched_qm: FinkBaseQueryManager):
        # Act
        success, missing, failed = patched_qm.query_lightcurves(
            fink_id_list=["T01", "T02", "T03"]
        )

        # Assert
        len(missing) == 3
        set(missing) == set(["T01", "T02", "T03"])

    def test__quit_after_n_failed(self, patched_qm: FinkBaseQueryManager):
        # Arrange
        patched_qm.config["lightcurve_chunk_size"] = 1
        patched_qm.config["max_failed_queries"] = 3
        id_list = "T_fail_01 T_fail_02 T_fail_03 T_fail_04 T_fail_05".split()

        # Act
        success, missing, failed = patched_qm.query_lightcurves(fink_id_list=id_list)

        # Assert
        assert set(success) == set()
        assert set(missing) == set()
        assert len(failed) == 3
        assert set(failed) == set(["T_fail_01", "T_fail_02", "T_fail_03"])

    def test__quit_after_slow_query(self, patched_qm: FinkBaseQueryManager):
        # Arrange
        t_start = time.perf_counter()
        _, _, _ = patched_qm.query_lightcurves(fink_id_list=["T00"])
        t_stop = time.perf_counter()
        assert t_stop - t_start < 0.3  # Check we wouldn't fail this anyway.
        patched_qm.config["lightcurve_chunk_size"] = 1
        patched_qm.config["max_query_time"] = 0.3

        # Act
        success, missing, failed = patched_qm.query_lightcurves(
            fink_id_list=["T_sleep", "T01"]
        )

        # Assert
        assert set(missing) == set(["T_sleep"])
        # didn't make it to T01 -- quit after 0.3 sec


class Test__IntegrateAlerts:
    def test__no_existing_lc(self, patched_qm: FinkBaseQueryManager):
        # this is a very simple test, as all the logic should be in the subclass...
        # Act
        modified_targets = patched_qm.integrate_alerts()

        # Assert
        assert set(modified_targets) == set(["T01"])


class Test__PerformAllTasks:
    def test__startup(self, patched_qm: FinkBaseQueryManager, t_fixed: Time):
        # Act
        patched_qm.perform_all_tasks(iteration=0, t_ref=t_fixed)

        # Assert
        assert set(patched_qm.target_lookup.keys()) == set(["T00", "T01"])

        T00 = patched_qm.target_lookup["T00"]
        assert "fink_cool" not in T00.target_data.keys()
        T01 = patched_qm.target_lookup["T01"]
        assert set(T01.update_messages) == set(["alerts integrated!"])

    def test__non_startup(self, patched_qm: FinkBaseQueryManager, t_fixed: Time):
        # Act
        patched_qm.perform_all_tasks(-1, t_ref=t_fixed)

        # Assert
        exp_targets = "T00 T01 T101 T201 T202 T203".split()
        assert set(patched_qm.target_lookup.keys()) == set(exp_targets)
