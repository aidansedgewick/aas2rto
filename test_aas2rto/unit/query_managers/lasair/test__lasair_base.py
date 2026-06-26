import copy
import json
import pytest
from typing import Iterator, NewType, NoReturn
from pathlib import Path

import numpy as np

from astropy.time import Time

import confluent_kafka
from confluent_kafka.cimpl import Message

from aas2rto.target_lookup import TargetLookup
from aas2rto.exc import BadKafkaConfigError, UnexpectedKeysError
from aas2rto.query_managers.lasair.lasair_base import LasairBaseQueryManager

import lasair

##===== Some mock classes to help testing here =====##

LasairAlertStream = NewType("LasairAlertStream", Iterator[Message])


class MockConfluentConsumer:
    def __init__(self, settings: dict, **kwargs):
        self.settings = settings
        self._topics = settings.get("topic")

    def subscribe(self, topics, **kwargs):
        self._topics = topics

    def close(self):
        pass


class LasairExampleQM(LasairBaseQueryManager):
    name = "cool_lasair"
    id_resolving_order = ("cool_lasair", "lasair")
    target_id_key = "target_id"
    alert_id_key = "alert_id"

    def process_single_alert(self, alert: dict, topic_keys: dict, t_ref: Time = None):

        target_id_key = topic_keys.get("target_id")

        alert_filepath = self.get_alert_filepath()
        return process_single_alert(alert, t_ref=t_ref)

    def load_single_lightcurve(self, target_id: str, t_ref: Time):
        pass

    def new_target_from_alert(
        self, processed_alert: dict, topic_keys: dict, t_ref=None
    ):
        pass

    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        pass


##===== Define fixtures here =====##


@pytest.fixture
def lasair_config() -> dict:

    cool_sne = {
        "target_id": "cool_id",
        "alert_id": "obs_id",
        "ra": "ra",
        "decl": "dec",
    }
    kafka_config = {
        "server": "lasair-blah.org",
        "group_id": "pytest_123",
        "topics": {
            "cool_sne": cool_sne,
        },
    }
    return {
        "client_token": "example_token",
        "kafka": kafka_config,
    }


@pytest.fixture
def alert_base() -> dict:
    return {
        "cool_id": "T101",
        "ra": 60.0,
        "dec": -45.0,
    }


@pytest.fixture
def mock_alert_list(alert_base: dict, t_fixed: Time) -> list[Message]:
    alert_list = []
    mjd = t_fixed.mjd
    for ii in range(10):
        # need deep copy, otherwise nested dicts are just references to the same dict!
        alert = copy.deepcopy(alert_base)  # Nested objects
        alert["obs_id"] = 1000 + ii
        alert["mjd"] = mjd + ii
        alert["band"] = "r"

        # confluent docs say instantiating Message as mock/tests is ok.
        value = json.dumps(alert).encode("utf-8")
        message = Message(topic="cool_sne", value=value, error=None)
        alert_list.append(message)
    return alert_list


@pytest.fixture
def mock_alert_stream(mock_alert_list: list[Message]) -> LasairAlertStream:
    """returns a Genrator, so an item is returend every time next() is called"""

    def alert_stream():
        for alert in mock_alert_list:
            yield alert

    return alert_stream()  # () - ie. return already primed generator


@pytest.fixture
def patch_consumer_and_poll(
    mock_alert_stream: LasairAlertStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mock_poll(self, timeout=10.0, **kwargs):
        topic = self.consumer._topics[0]
        if topic == "cool_sne":
            try:
                return next(mock_alert_stream)
            except StopIteration:
                return None
        else:
            return Message(topic=topic, error="Only mock alerts for 'cool_sne'!")

    # Mock CONFLUENT consumer, so that the real lasair_consumer __init__ is called,
    # so we can spot bugs in the parameters we pass to lasair_consumer.
    monkeypatch.setattr("confluent_kafka.Consumer", MockConfluentConsumer)
    monkeypatch.setattr("lasair.lasair_consumer.poll", mock_poll)
    return


@pytest.fixture
def patched_qm(
    lasair_config: dict,
    tlookup: TargetLookup,
    tmp_path: Path,
    patch_consumer_and_poll: None,
):
    qm = LasairExampleQM(lasair_config, tlookup, parent_path=tmp_path)
    return qm


##===== Test helper objects work as designed! =====##


class Test__MockAlertStream:
    def test__alert_from_list(self, mock_alert_list: list[Message]):
        # Act
        message = mock_alert_list[0]

        # Assert
        assert isinstance(message, Message)
        assert isinstance(message.value(), bytes)
        assert message.topic() == "cool_sne"

    def test__alert_from_list_decodes(self, mock_alert_list: list[Message]):
        # Act
        message = mock_alert_list[0]
        alert = json.loads(message.value())

        print(message.timestamp())

        # Assert
        assert isinstance(alert, dict)
        exp_keys = ["cool_id", "ra", "dec", "obs_id", "band", "mjd"]
        assert set(alert.keys()) == set(exp_keys)

    def test__mock_alert_stream_yields(self, mock_alert_stream: LasairAlertStream):
        # Act
        message = next(mock_alert_stream)

        # Assert
        assert isinstance(message, Message)

    def test__mock_alert_raise_StopIteration(
        self, mock_alert_stream: LasairAlertStream
    ):
        # Act
        for ii in range(10):
            _ = next(mock_alert_stream)

        with pytest.raises(StopIteration):
            alert = next(mock_alert_stream)


class Test__ConsumerPatching:
    def test__confluent_patched(self, patch_consumer_and_poll: None):
        # This is a boring test...

        # Act
        consumer = confluent_kafka.Consumer({"blah": 100.0})

        # Assert
        assert isinstance(consumer, MockConfluentConsumer)
        assert consumer._topics is None

    def test__lasair_consumer_patched(self, patch_consumer_and_poll: None):
        # Act
        lcons = lasair.lasair_consumer("lasair-blah.org", "test_group", "cool_sne")

        # Assert
        assert isinstance(lcons, lasair.lasair_consumer)
        assert isinstance(lcons.consumer, MockConfluentConsumer)

        # Lasair consumer calls subscribe with .subscribe(['topic']) - 1-list
        assert isinstance(lcons.consumer._topics, list)
        assert lcons.consumer._topics[0] == "cool_sne"

    def test__lasair_poll_patched(self, patch_consumer_and_poll: None):
        # Arrange
        lcons = lasair.lasair_consumer("lasair-blah.org", "test_group", "cool_sne")

        # Act
        message = lcons.poll()

        # Assert
        assert isinstance(message, Message)


##===== Tests start here! =====##


class Test__InitLasairQM:
    def init_config(self, lasair_config: dict, tlookup: TargetLookup, tmp_path: Path):

        # Act
        qm = LasairExampleQM(lasair_config, tlookup, parent_path=tmp_path)

        # Assert
        exp_directories = "alerts cutouts lightcurves sherlock".split()
        assert set(qm.paths_lookup.keys()) == set(exp_directories)

        assert qm.paths_lookup["alerts"] == tmp_path / "lasair_cool/alerts"
        assert qm.paths_lookup["alerts"].exists()
        assert qm.paths_lookup["cutouts"] == tmp_path / "lasair_cool/cutouts"
        assert qm.paths_lookup["cutouts"].exists()
        assert qm.paths_lookup["lightcurves"] == tmp_path / "lasair_cool/lightcurves"
        assert qm.paths_lookup["lightcurves"].exists()
        assert qm.paths_lookup["sherlock"] == tmp_path / "lasair_cool/sherlock"
        assert qm.paths_lookup["sherlock"].exists()

    def test__bad_arg(self, lasair_config: dict, tlookup: TargetLookup, tmp_path: Path):
        # Arrange
        lasair_config["blah"] = 100.0

        # Act
        with pytest.raises(UnexpectedKeysError):
            qm = LasairExampleQM(lasair_config, tlookup, parent_path=tmp_path)


class Test__KafkaConfig:
    def test__no_kafka_is_ok(
        self, lasair_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Arrange
        lasair_config.pop("kafka")

        # Act
        qm = LasairExampleQM(lasair_config, tlookup, parent_path=tmp_path)

        # Assert
        qm.kafka_config = None

    def test__missing_kafka_subkey_raises(
        self, lasair_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Arrange
        lasair_config["kafka"].pop("topics")

        # Act
        with pytest.raises(BadKafkaConfigError):
            qm = LasairExampleQM(lasair_config, tlookup, parent_path=tmp_path)

    def test__str_topic_converted(
        self, lasair_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Arrange
        lasair_config["kafka"]["topics"] = "cool_sne"

        # Act
        qm = LasairExampleQM(lasair_config, tlookup, parent_path=tmp_path)

        # Assert
        assert isinstance(qm.kafka_config["topics"], dict)
        assert set(qm.kafka_config["topics"].keys()) == set(["cool_sne"])
        assert set(qm.kafka_config["topics"]["cool_sne"].keys()) == set(
            ["target_id", "alert_id", "ra", "decl"]
        )

    def test__list_topics_converted(
        self, lasair_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Arrange
        lasair_config["kafka"]["topics"] = ["cool_sne", "other_topic"]

        # Act
        qm = LasairExampleQM(lasair_config, tlookup, parent_path=tmp_path)

        # Assert
        assert isinstance(qm.kafka_config["topics"], dict)
        assert set(qm.kafka_config["topics"].keys()) == set(["cool_sne", "other_topic"])

    def test__missing_topic_key_added(
        self, lasair_config: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Arrange
        lasair_config["kafka"]["topics"]["other_topic"] = {"target_id": "wowow"}

        # Act
        qm = LasairExampleQM(lasair_config, tlookup, parent_path=tmp_path)

        # Assert
        exp_topic_keys = set(["target_id", "alert_id", "ra", "decl"])
        assert set(qm.kafka_config["topics"]["other_topic"].keys()) == exp_topic_keys

    def test__topic_bad_key_raises(
        self, lasair_config: dict, tlookup: TargetLookup, tmp_path
    ):
        # Arrange
        lasair_config["kafka"]["topics"]["other_topic"] = {"aaaaa": "ok"}

        # Act
        with pytest.raises(BadKafkaConfigError):
            qm = LasairExampleQM(lasair_config, tlookup, parent_path=tmp_path)


class Test__PathMethods:
    def test__alert_path(self, patched_qm: LasairBaseQueryManager, tmp_path: Path):
        # Act
        alert_path = patched_qm.get_alert_directory("T00")

        # Assert
        assert alert_path == tmp_path / "cool_lasair/alerts/T00"
        assert alert_path.is_dir()
        assert alert_path.exists()

    def test__alert_filepath(self, patched_qm: LasairBaseQueryManager, tmp_path: Path):
        # Act
        alert_path = patched_qm.get_alert_filepath("T00", 1000)

        # Assert
        assert alert_path == tmp_path / "cool_lasair/alerts/T00/1000.json"

    def test__cutouts_path(self, patched_qm: LasairBaseQueryManager, tmp_path: Path):
        # Act
        cutouts_path = patched_qm.get_cutouts_directory("T00")

        # Assert
        assert cutouts_path == tmp_path / "cool_lasair/cutouts/T00"
        assert cutouts_path.is_dir()
        assert cutouts_path.exists()

    def test__cutouts_filepath(
        self, patched_qm: LasairBaseQueryManager, tmp_path: Path
    ):
        # Act
        cutouts_path = patched_qm.get_cutouts_filepath("T00", 1000)

        # Assert
        assert cutouts_path == tmp_path / "cool_lasair/cutouts/T00/1000.pkl"

    def test__lightcurve_filepath(
        self, patched_qm: LasairBaseQueryManager, tmp_path: Path
    ):
        # Act
        lightcurve_filepath = patched_qm.get_lightcurve_filepath("T00")

        # Assert
        assert lightcurve_filepath == tmp_path / "cool_lasair/lightcurves/T00.csv"


class Test__ListenForAlerts:
    def test__listen_for_alerts(
        self, patched_qm: LasairBaseQueryManager, t_fixed: Time
    ):
        # Act
        alerts = patched_qm.listen_for_alerts()
