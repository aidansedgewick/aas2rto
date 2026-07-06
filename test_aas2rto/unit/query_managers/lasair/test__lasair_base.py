import copy
import json
import pytest
import time
from typing import Iterator, NewType, NoReturn
from pathlib import Path

import numpy as np

import pandas as pd


from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import confluent_kafka
from confluent_kafka.cimpl import Message

from aas2rto.exc import BadKafkaConfigError, UnexpectedKeysError
from aas2rto.query_managers.lasair import lasair_base  # Mainly for patching
from aas2rto.query_managers.lasair.lasair_base import LasairBaseQueryManager
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup

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

    def poll(self):
        pass  # Will be patched in 'patched_qm' def.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class LasairExampleQM(LasairBaseQueryManager):
    name = "lasair_cool"
    id_resolving_order = ("cool_survey", "lasair_cool")
    target_id_key = "cool_id"
    alert_id_key = "obs_id"
    ra_key = "ra"
    dec_key = "dec"
    lightcurve_key = "lightcurve"
    object_data_key = "object_data"
    lasair_client_endpoint = "lasair-blah.org/api"  # Never contact the real lasair!

    def new_target_from_alert(self, processed_alert: dict, t_ref=None):
        target_id = processed_alert[self.target_id_key]
        ra = processed_alert.get("ra", None)
        dec = processed_alert.get("dec", None)

        if ra is not None and dec is not None:
            coord = SkyCoord(ra=ra, dec=dec, unit=u.deg)
        else:
            coord = None

        return Target(target_id, coord, alt_ids={"cool_survey": target_id})

    def process_lasair_object_data(self, target_data: dict, t_ref: Time = None):

        lasair_id = target_data[self.target_id_key]

        # LC
        lc_records = target_data[self.lightcurve_key]
        lc_filepath = self.get_lightcurve_filepath(lasair_id)
        lc = pd.DataFrame(lc_records)
        lc.to_csv(lc_filepath, index=False)

        # Basic data
        object_data = target_data["object_data"]
        object_data_filepath = self.get_object_data_filepath(lasair_id)
        with open(object_data_filepath, "w+") as f:
            json.dump(object_data, f)

        # Image_urls
        image_urls = pd.DataFrame(target_data["image_urls"])
        image_urls_filepath = self.get_cutout_urls_filepath(lasair_id)
        image_urls.to_csv(image_urls_filepath, index=False)

    def apply_updates_from_alert(self, processed_alert: dict, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        lasair_id = processed_alert[self.target_id_key]

        target = self.target_lookup.get(lasair_id, None)

        t_str = t_ref.strftime("%Y-%m-%d %H:%M")
        target.info_messages.append(f"alert at {t_str}")
        target.updated = True


##===== Define fixtures here =====##


@pytest.fixture
def cool_sne_topic_keys():
    return {
        "id_key": "cool_id",
        "ra_key": "alert_ra",
        "dec_key": "alert_dec",
    }


@pytest.fixture
def lasair_config(cool_sne_topic_keys: dict) -> dict:
    kafka_config = {
        "bootstrap.servers": "lasair-blah.org",
        "group.id": "pytest_123",
        "topics": {
            "cool_sne": cool_sne_topic_keys,
        },
    }
    return {
        "client_token": "example_token",
        "kafka": kafka_config,
    }


@pytest.fixture
def lc_lasair(valid_det_lc_pandas: pd.DataFrame) -> pd.DataFrame:
    lc = valid_det_lc_pandas.copy()
    return lc


@pytest.fixture
def alert_base() -> dict:
    return {
        "cool_id": "T101",  # Specific. NOT target_id: Lasair alerts have custom keys.
        "alert_ra": 60.0,
        "alert_dec": -45.0,
    }


@pytest.fixture
def alert_no_coords(t_fixed: Time) -> dict:
    alert = {
        "cool_id": "T101",
        "band": "r",
        "mjd": t_fixed.mjd,
    }

    t_alert = t_fixed
    timestamp_ms = int(t_alert.unix * 1000)
    alert_timestamp = (1, timestamp_ms)

    value = json.dumps(alert).encode("utf-8")
    return Message(
        topic="sne_no_coords", value=value, error=None, timestamp=alert_timestamp
    )


@pytest.fixture
def alert_with_lightcurve(alert_base: dict, lc_lasair: pd.DataFrame, t_fixed: Time):
    alert = copy.deepcopy(alert_base)
    alert["band"] = "r"
    alert["lightcurve"] = lc_lasair.to_dict("records")

    t_alert = t_fixed
    timestamp_ms = int(t_alert.unix * 1000)
    alert_timestamp = (1, timestamp_ms)

    value = json.dumps(alert).encode("utf-8")
    return Message(
        topic="sne_with_lc",
        value=value,
        error=None,
        timestamp=alert_timestamp,
    )


@pytest.fixture
def mock_alert_list(alert_base: dict, t_fixed: Time) -> list[Message]:
    alert_list = []
    for ii in range(10):
        # need deep copy, otherwise nested dicts are just references to the same dict!
        alert = copy.deepcopy(alert_base)  # Nested objects # unix in millisec.
        alert["band"] = "r"

        t_alert = t_fixed + ii * u.day
        alert["mjd"] = t_fixed.mjd
        alert["obs_id"] = 1000 + ii
        timestamp_ms = int(t_alert.unix * 1000)
        message_timestamp = (1, timestamp_ms)

        # confluent docs say instantiating Message as mock/tests is ok.
        value = json.dumps(alert).encode("utf-8")
        message = Message(
            topic="cool_sne",
            value=value,
            error=None,
            timestamp=message_timestamp,
        )
        alert_list.append(message)
    return alert_list


@pytest.fixture
def mock_alert_stream(mock_alert_list: list[Message]) -> LasairAlertStream:
    """returns a Genrator, so an item is returned every time next() is called"""

    def alert_stream():
        for alert in mock_alert_list:
            yield alert

    return alert_stream()  # () - ie. return already primed generator


@pytest.fixture
def patch_consumer_and_poll_method(
    mock_alert_stream: LasairAlertStream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mock_poll(self, timeout=10.0, **kwargs):
        topic = self._topics[0]
        if topic == "cool_sne":
            try:
                return next(mock_alert_stream)
            except StopIteration:
                return None
        else:
            return Message(topic=topic, error=f"No known topic '{topic}'")

    # At the minute we're using Confluent Consumer directly -- not lasair consumer.
    monkeypatch.setattr(MockConfluentConsumer, "poll", mock_poll)
    monkeypatch.setattr("confluent_kafka.Consumer", MockConfluentConsumer)
    monkeypatch.setattr(lasair_base, "Consumer", MockConfluentConsumer)
    return


@pytest.fixture
def lasair_object_data(lc_lasair: pd.DataFrame):
    lightcurve = lc_lasair.to_dict("records")
    image_urls = []
    for rec in lightcurve:
        obs_id = rec["obs_id"]
        rec_urls = {x: f"{x}_{obs_id}" for x in ["Science", "Difference", "Template"]}
        rec_urls["obs_id"] = obs_id
        rec_urls["band"] = rec["band"]
        rec_urls["mjd"] = rec["mjd"]
        image_urls.append(rec_urls)

    return {
        "object_data": {"ra": 60.0, "dec": -45.0},
        "lightcurve": lightcurve,
        "image_urls": image_urls,
    }


@pytest.fixture
def patch_client_endpoints(lasair_object_data: dict, monkeypatch: pytest.MonkeyPatch):

    def mock_object_endpoint(self, target_id: str, **kwargs):

        data = copy.deepcopy(lasair_object_data)
        data["cool_id"] = target_id
        if target_id in ["T00", "T01"]:
            return data
        if target_id.startswith("T_sleep"):
            time.sleep(0.4)
            return data
        else:
            raise lasair.lasair.LasairError(f"No data for {target_id}!")

    monkeypatch.setattr(lasair.lasair_client, "object", mock_object_endpoint)


@pytest.fixture
def patch_request_cutout(monkeypatch: pytest.MonkeyPatch):
    # TODO: Would be better to try to find a way to patch requests.get
    def mock_request_cutout(*args, **kwargs):
        return np.random.normal(0, 1, (10, 10))

    monkeypatch.setattr(lasair_base, "request_cutout", mock_request_cutout)


@pytest.fixture
def patched_qm(
    lasair_config: dict,
    tlookup: TargetLookup,
    tmp_path: Path,
    patch_consumer_and_poll_method: None,
    patch_client_endpoints: None,
    patch_request_cutout: None,
):
    qm = LasairExampleQM(lasair_config, tlookup, parent_path=tmp_path)
    qm.target_lookup["T00"].alt_ids["cool_survey"] = "T00"
    qm.target_lookup["T01"].alt_ids["cool_survey"] = "T01"
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

        # Assert
        assert isinstance(alert, dict)
        exp_keys = ["cool_id", "alert_ra", "alert_dec", "obs_id", "band", "mjd"]
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
    def test__confluent_patched(self, patch_consumer_and_poll_method: None):
        # This is a boring test...

        # Act
        consumer = lasair_base.Consumer({"blah": 100.0})

        # Assert
        assert isinstance(consumer, MockConfluentConsumer)
        assert consumer._topics is None

    # def test__lasair_consumer_patched(self, patch_consumer_and_poll_method: None):
    #     # Act
    #     lcons = lasair.lasair_consumer("lasair-blah.org", "test_group", "cool_sne")

    #     # Assert
    #     assert isinstance(lcons, lasair.lasair_consumer)
    #     assert isinstance(lcons.consumer, MockConfluentConsumer)

    #     # Lasair consumer calls subscribe with .subscribe(['topic']) - 1-list
    #     assert isinstance(lcons.consumer._topics, list)
    #     assert lcons.consumer._topics[0] == "cool_sne"

    def test__confluent_poll_patched(self, patch_consumer_and_poll_method: None):
        # Arrange
        settings = {}
        consumer = lasair_base.Consumer(settings)  # as will used in the lasair module.
        consumer.subscribe(["cool_sne"])

        # Act
        message = consumer.poll()

        # Assert
        assert isinstance(message, Message)


class Test__AddTargetsFromAlerts:
    def test__new_target_from_alert(
        self,
        patched_qm: LasairBaseQueryManager,
        mock_alert_list: list[Message],
        cool_sne_topic_keys: dict,
    ):
        # Arrange
        alert = mock_alert_list[0]
        processed_alert = patched_qm.process_single_alert(alert, cool_sne_topic_keys)

        # Act
        target = patched_qm.new_target_from_alert(processed_alert)

        # Assert
        assert isinstance(target, Target)

    def test__missing_coords_does_not_raise(
        self,
        patched_qm: LasairBaseQueryManager,
        alert_no_coords: Message,
        cool_sne_topic_keys: dict,
    ):
        # Arrange
        processed_alert = patched_qm.process_single_alert(
            alert_no_coords, cool_sne_topic_keys
        )

        # Act
        target = patched_qm.new_target_from_alert(processed_alert)

        # Assert
        assert isinstance(target, Target)
        assert target.target_id == "T101"
        assert target.coord is None


class Test__ClientPatched:
    def test__object_request(self, patched_qm: LasairBaseQueryManager):
        # Act
        data = patched_qm.lasair_client.object("T00")

        # Assert
        assert isinstance(data, dict)

    def test__bad_object_request_raises(self, patched_qm: LasairBaseQueryManager):
        # Act
        with pytest.raises(lasair.lasair.LasairError):
            data = patched_qm.lasair_client.object("T_bad_target")


class Test__ProcessObjectData:
    def test__process_data(
        self,
        patched_qm: LasairBaseQueryManager,
        lasair_object_data: dict,
        tmp_path: Path,
    ):
        # Arrange
        lasair_object_data["cool_id"] = "T00"

        # Act
        patched_qm.process_lasair_object_data(lasair_object_data)

        # Assert
        exp_lc_path = tmp_path / "lasair_cool/lightcurves/T00.csv"
        assert exp_lc_path.exists()
        recovered_lc = pd.read_csv(exp_lc_path)
        assert len(recovered_lc) == 6

        exp_data_path = tmp_path / "lasair_cool/object_data/T00.json"
        assert exp_data_path.exists()
        with open(exp_data_path) as f:
            recovered_data = json.load(f)
        assert set(recovered_data.keys()) == set(["ra", "dec"])

        exp_cutout_urls_path = (
            tmp_path / "lasair_cool/cutouts/cutout_urls/T00_cutout_urls.csv"
        )
        assert exp_cutout_urls_path.exists()
        recovered_urls = pd.read_csv(exp_cutout_urls_path)
        assert len(recovered_urls) == 6
        exp_cols = ["obs_id", "Science", "Difference", "Template", "band", "mjd"]
        assert set(recovered_urls.columns) == set(exp_cols)


class Test__AddMessages:
    def test__add_message(self, patched_qm: LasairBaseQueryManager, t_fixed: Time):
        # Act
        patched_qm.apply_updates_from_alert({"cool_id": "T00"}, t_ref=t_fixed)

        # Assert
        assert len(patched_qm.target_lookup["T00"].info_messages) == 1


class Test__RequestCutoutsPatched:
    def test__request_cutout_patched(self, patch_request_cutout: None):
        # Act
        cutouts = lasair_base.request_cutout("NONE_EXISTANT_URL")

        # Assert
        assert cutouts.shape == (10, 10)


##===== Tests start here! =====##


class Test__InitLasairQM:
    def init_config(self, lasair_config: dict, tlookup: TargetLookup, tmp_path: Path):

        # Act
        qm = LasairExampleQM(lasair_config, tlookup, parent_path=tmp_path)

        # Assert
        exp_directories = "alerts cutouts lightcurves sherlock forced_photom".split()
        assert set(qm.paths_lookup.keys()) == set(exp_directories)

        assert qm.paths_lookup["alerts"] == tmp_path / "lasair_cool/alerts"
        assert qm.paths_lookup["alerts"].exists()
        assert qm.paths_lookup["cutouts"] == tmp_path / "lasair_cool/cutouts"
        assert qm.paths_lookup["cutouts"].exists()
        assert qm.paths_lookup["lightcurves"] == tmp_path / "lasair_cool/lightcurves"
        assert qm.paths_lookup["lightcurves"].exists()
        exp_fp_path = tmp_path / "lasair_cool/forced_photom"
        assert qm.paths_lookup["forced_photom"] == exp_fp_path
        assert qm.paths_lookup["forced_photom"].exists()
        exp_sherlock_path = tmp_path / "lasair_cool/sherlock_context"
        assert qm.paths_lookup["sherlock_context"] == exp_sherlock_path
        assert qm.paths_lookup["sherlock"].exists()

    def test__bad_arg(self, lasair_config: dict, tlookup: TargetLookup, tmp_path: Path):
        # Arrange
        lasair_config["blah"] = 100.0

        # Act
        with pytest.raises(UnexpectedKeysError):
            qm = LasairExampleQM(lasair_config, tlookup, parent_path=tmp_path)

    def test__default_topic_keys_added(
        self, lasair_config, tlookup: TargetLookup, tmp_path: Path
    ):
        # Act
        qm = LasairExampleQM(lasair_config, tlookup, tmp_path)

        assert qm.target_id_key == "cool_id"
        assert qm.ra_key == "ra"
        assert qm.dec_key == "dec"
        exp_keys = ["id_key", "ra_key", "dec_key"]
        assert set(qm.default_topic_keys.keys()) == set(exp_keys)


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

        sne_keys = qm.kafka_config["topics"]["cool_sne"].keys()
        exp_keys = ["id_key", "ra_key", "dec_key"]
        assert set(sne_keys) == set(exp_keys)  # Check DEFAULTS are added.

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
        lasair_config["kafka"]["topics"]["other_topic"] = {"id_key": "wowow"}

        # Act
        qm = LasairExampleQM(lasair_config, tlookup, parent_path=tmp_path)

        # Assert
        exp_topic_keys = set(["id_key", "ra_key", "dec_key"])
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
        alert_path = patched_qm.get_alert_directory("T101", "test_topic")

        # Assert
        assert alert_path == tmp_path / "lasair_cool/alerts/test_topic/T101"
        assert alert_path.is_dir()
        assert alert_path.exists()

    def test__alert_filepath(self, patched_qm: LasairBaseQueryManager, tmp_path: Path):
        # Act
        alert_path = patched_qm.get_alert_filepath("T101", "test_topic", 1000)

        # Assert
        assert alert_path == tmp_path / "lasair_cool/alerts/test_topic/T101/1000.json"

    def test__cutouts_path(self, patched_qm: LasairBaseQueryManager, tmp_path: Path):
        # Act
        cutouts_path = patched_qm.get_cutouts_directory("T00")

        # Assert
        assert cutouts_path == tmp_path / "lasair_cool/cutouts/T00"
        assert cutouts_path.is_dir()
        assert cutouts_path.exists()

    def test__cutouts_filepath(
        self, patched_qm: LasairBaseQueryManager, tmp_path: Path
    ):
        # Act
        cutouts_path = patched_qm.get_cutouts_filepath("T00", 1000)

        # Assert
        assert cutouts_path == tmp_path / "lasair_cool/cutouts/T00/1000.pkl"

    def test__cutout_urls_filepath(
        self, patched_qm: LasairBaseQueryManager, tmp_path: Path
    ):
        # Act
        cutout_urls_filepath = patched_qm.get_cutout_urls_filepath("T00")

        # Assert
        exp_filepath = tmp_path / "lasair_cool/cutouts/cutout_urls/T00_cutout_urls.csv"
        assert cutout_urls_filepath == exp_filepath

    def test__lightcurve_filepath(
        self, patched_qm: LasairBaseQueryManager, tmp_path: Path
    ):
        # Act
        lightcurve_filepath = patched_qm.get_lightcurve_filepath("T00")

        # Assert
        assert lightcurve_filepath == tmp_path / "lasair_cool/lightcurves/T00.csv"


class Test__ProcessSingleAlert:
    def test__process_alert_from_list(
        self,
        patched_qm: LasairBaseQueryManager,
        mock_alert_list: list[Message],
        cool_sne_topic_keys: dict,
        tmp_path: Path,
        t_fixed: Time,
    ):
        # Arrange
        alert = mock_alert_list[0]
        val = json.loads(alert.value())
        assert "ra" not in val and "dec" not in val  # Just a reminder...
        assert "alert_ra" in val and "alert_dec" in val

        # Act
        processed_alert = patched_qm.process_single_alert(
            alert, cool_sne_topic_keys, t_ref=t_fixed
        )

        # Assert
        assert isinstance(processed_alert, dict)

        assert "ra" in processed_alert  # Standardized keys are added!
        assert "dec" in processed_alert

        assert processed_alert["topic"] == "cool_sne"  # topic also added!

        fstem = "20230225_000000"  # t_fixed as a Ymd_HMS
        exp_alert_filepath = tmp_path / f"lasair_cool/alerts/cool_sne/T101/{fstem}.json"
        assert exp_alert_filepath.exists()

        exp_lc_filepath = tmp_path / f"lasair_cool/lightcurves/T101.csv"
        assert not exp_lc_filepath.exists()  # No LC should not write LC.

    def test__with_lightcurve(
        self,
        patched_qm: LasairBaseQueryManager,
        alert_with_lightcurve: Message,
        cool_sne_topic_keys: dict,
        t_fixed: Time,
        tmp_path: Path,
    ):

        # Act
        processed_alert = patched_qm.process_single_alert(
            alert_with_lightcurve, cool_sne_topic_keys, t_ref=t_fixed
        )

        # Assert
        assert isinstance(processed_alert, dict)

        fstem = "20230225_000000"  # t_fixed as a Ymd_HMS
        exp_alert_filepath = (
            tmp_path / f"lasair_cool/alerts/sne_with_lc/T101/{fstem}.json"
        )
        assert exp_alert_filepath.exists()

        exp_lc_filepath = tmp_path / f"lasair_cool/lightcurves/T101.csv"
        assert exp_lc_filepath.exists()  # LC should exist!

    def test__with_missing_coords(
        self,
        patched_qm: LasairBaseQueryManager,
        alert_no_coords: Message,
        cool_sne_topic_keys: dict,
    ):
        # Arrange
        processed_alert = patched_qm.process_single_alert(
            alert_no_coords, cool_sne_topic_keys
        )

        # Assert
        assert isinstance(processed_alert, dict)
        assert "ra" not in processed_alert
        assert "dec" not in processed_alert


class Test__ListenForAlerts:
    def test__listen_for_alerts(
        self, patched_qm: LasairBaseQueryManager, t_fixed: Time
    ):
        # Act
        alerts = patched_qm.listen_for_alerts()

        # Assert
        assert len(alerts) == 10

    def test__polling_breaks_early(
        self, patched_qm: LasairBaseQueryManager, t_fixed: Time
    ):
        # Arrange
        patched_qm.config["n_alerts"] = 5

        # Act
        alerts = patched_qm.listen_for_alerts()

        # Assert
        assert len(alerts) == 5

    def test__bad_topic_does_not_crash(
        self, patched_qm: LasairBaseQueryManager, t_fixed: Time
    ):
        # Arrange
        patched_qm.kafka_config["topics"] = {"bad_topic": patched_qm.default_topic_keys}

        # Act
        alerts = patched_qm.listen_for_alerts()

        # Assert
        assert len(alerts) == 0


class Test__QueryObjectsEndpoint:
    def test__query_objects(self, patched_qm: LasairBaseQueryManager, tmp_path: Path):
        # Act
        success, missing, failed = patched_qm.query_target_data(["T00"])

        # Assert
        assert set(success) == set(["T00"])
        assert set(missing) == set()
        assert set(failed) == set()

        exp_lc_path = tmp_path / "lasair_cool/lightcurves/T00.csv"
        assert exp_lc_path.exists()

        exp_data_path = tmp_path / "lasair_cool/object_data/T00.json"
        assert exp_data_path.exists()

        exp_cutout_urls_path = (
            tmp_path / "lasair_cool/cutouts/cutout_urls/T00_cutout_urls.csv"
        )
        assert exp_cutout_urls_path.exists()

    def test__quit_after_bad_queries(
        self, patched_qm: LasairBaseQueryManager, tmp_path: Path
    ):
        # Arrange
        patched_qm.config["max_failed_queries"] = 3

        # Act
        object_list = ["T_fail_01", "T_fail_02", "T_fail_03", "T_fail_04"]
        success, missing, failed = patched_qm.query_target_data(
            lasair_id_list=object_list
        )

        # Assert
        assert set(success) == set()
        assert set(missing) == set()
        assert set(failed) == set(["T_fail_01", "T_fail_02", "T_fail_03"])

    def test__quit_after_slow_quieries(
        self, patched_qm: LasairBaseQueryManager, tmp_path: Path
    ):
        # Arrange
        patched_qm.config["max_query_time"] = 0.5

        # Act
        object_list = ["T_sleep01", "T_sleep02", "T_sleep03"]
        success, missing, failed = patched_qm.query_target_data(
            lasair_id_list=object_list
        )

        # Assert
        assert set(success) == set(["T_sleep01", "T_sleep02"])
        assert set(missing) == set()
        assert set(failed) == set()  # The last one didn't fail, it just didn't run


class Test__AddNewTargetsFromAlerts:
    def test__new_targets(self, patched_qm: LasairBaseQueryManager):
        # Arrange
        assert set(patched_qm.target_lookup.keys()) == set(["T00", "T01"])  # Reminder
        processed_alerts = patched_qm.listen_for_alerts()
        assert len(processed_alerts) == 10

        # Act
        targets_added = patched_qm.add_targets_from_alerts(processed_alerts)

        # Assert
        assert set(targets_added) == set(["T101"])
        exp_target_ids = ["T00", "T01", "T101"]
        assert set(patched_qm.target_lookup.keys()) == set(exp_target_ids)

    def test__skip_existing(self, patched_qm: LasairBaseQueryManager):
        # Arrange
        processed_alerts = patched_qm.listen_for_alerts()
        new_targets = patched_qm.add_targets_from_alerts(processed_alerts)

        # Act
        next_new_targets = patched_qm.add_targets_from_alerts(processed_alerts)

        # Assert
        assert set(next_new_targets) == set()  # Don't re-add targets!


class Test__QueryObjectData:
    def test__no_query_irrelevant_targets(self, patched_qm: LasairBaseQueryManager):
        # Arrange
        T99 = Target("T99", coord=None, alt_ids={"other_survey": "T99"})
        patched_qm.target_lookup.add_target(T99)

        # Act
        success, missing, failed = patched_qm.query_target_data()

        # Assert
        assert set(success) == set(["T00", "T01"])
        assert set(missing) == set()
        assert set(failed) == set()  # ie. "T00", "T01" (without) relv. id are SKIPPED.


class Test__AddMissingInfo:
    def test__add_missing_info(self, patched_qm: LasairBaseQueryManager):
        # Arrange
        patched_qm.target_lookup["T00"].coord = None
        patched_qm.query_target_data()  # Dump the information into a file

        # Act
        modified = patched_qm.load_missing_coord_information()

        # Assert
        assert set(modified) == set(["T00"])
        assert isinstance(patched_qm.target_lookup["T00"].coord, SkyCoord)


class Test__LoadSingleLightcurve:
    def test__load_lc(
        self, patched_qm: LasairBaseQueryManager, lc_lasair: pd.DataFrame
    ):
        # Arrange
        lc_filepath = patched_qm.get_lightcurve_filepath("T00")
        lc_lasair.to_csv(lc_filepath, index=False)

        # Act
        loaded = patched_qm.load_single_lightcurve(target_id="T00")

        # Assert
        assert isinstance(loaded, pd.DataFrame)

    def test__missing_lc_no_fail(self, patched_qm: LasairBaseQueryManager):
        # Act
        lc = patched_qm.load_single_lightcurve(target_id="T00")

    def test__no_relevant_id(self, patched_qm: LasairBaseQueryManager):
        # Arrange
        T01 = patched_qm.target_lookup["T01"]
        T01.alt_ids.pop("cool_survey")
        assert patched_qm.get_relevant_id_from_target(T01) is None

        # Act
        lc = patched_qm.load_single_lightcurve("T01")

        # Assert
        assert lc is None

    def test__bad_target_no_error(self, patched_qm: LasairBaseQueryManager):
        # Arrange
        assert "T02" not in patched_qm.target_lookup

        # Act
        lc = patched_qm.load_single_lightcurve("T02")

        # Assert
        assert lc is None


class Test__LoadLightcurves:
    def test__load_lightcurves(
        self, patched_qm: LasairBaseQueryManager, tmp_path: Path
    ):
        # not the most useful test: this is a super() method...

        # Arrange
        tl = patched_qm.target_lookup
        assert "lasair_cool" not in tl["T00"].target_data  # A reminder...
        patched_qm.query_target_data()

        # Act
        patched_qm.load_target_lightcurves()

        # Assert
        assert "lasair_cool" in tl["T00"].target_data
        assert len(tl["T00"].target_data["lasair_cool"].lightcurve) == 6


class Test__QueryTargetCutouts:
    def test__query_target_cutouts(
        self, patched_qm: LasairBaseQueryManager, tmp_path: Path
    ):
        # Arrange
        patched_qm.query_target_data()  # Make sure the img_urls are written
        patched_qm.config["n_cutouts"] = 2

        # Act
        success, missing, failed = patched_qm.query_latest_target_cutouts("T00")

        # Assert
        exp_cutouts_dir = tmp_path / "lasair_cool/cutouts/T00"
        assert exp_cutouts_dir.exists()

        found_cutout_pkls = list(exp_cutouts_dir.glob("*.pkl"))
        assert len(found_cutout_pkls) == 2

        pkl_stems = [f.stem for f in found_cutout_pkls]
        exp_stems = ["1000200030004004", "1000200030004005"]  # ie. the last 2.
        assert set(pkl_stems) == set(exp_stems)

        exp_requested_tags = ["T00-1000200030004004", "T00-1000200030004005"]
        assert set(success) == set(exp_requested_tags)

    def test__skip_existing_cutouts(
        self, patched_qm: LasairBaseQueryManager, tmp_path: Path
    ):
        # Arrange
        patched_qm.query_target_data()
        patched_qm.config["n_cutouts"] = 2
        cutout_fpath = patched_qm.get_cutouts_filepath("T00", 1000_2000_3000_4004)
        cutout_fpath.touch()  # Make sure it exists.

        # Act
        success, missing, failed = patched_qm.query_latest_target_cutouts("T00")

        # Assert
        exp_cutouts_dir = tmp_path / "lasair_cool/cutouts/T00"
        assert exp_cutouts_dir.exists()

        found_cutout_pkls = list(exp_cutouts_dir.glob("*.pkl"))
        assert len(found_cutout_pkls) == 2

        pkl_stems = [f.stem for f in found_cutout_pkls]
        exp_stems = ["1000200030004004", "1000200030004005"]  # ie. the last 2.
        assert set(pkl_stems) == set(exp_stems)

        exp_requested_tags = ["T00-1000200030004005"]  # Skipped pre-existing ...04
        assert set(success) == set(exp_requested_tags)

    def test__no_fail_too_many_cutout_requests(
        self, patched_qm: LasairBaseQueryManager, tmp_path: Path
    ):
        # Arrange
        patched_qm.query_target_data()
        patched_qm.config["n_cutouts"] = 10

        # Act
        success, missing, failed = patched_qm.query_latest_target_cutouts("T00")

        id0 = 1000_2000_3000_4000
        exp_success = [f"T00-{id0+ii}" for ii in [0, 1, 2, 3, 4, 5]]
        assert set(success) == set(exp_success)
        assert set(missing) == set()
        assert set(failed) == set()

    def test__missing_cutout_urls_file(
        self, patched_qm: LasairBaseQueryManager, tmp_path
    ):
        # Arrange
        urls_fpath = patched_qm.get_cutout_urls_filepath("T00")
        assert not urls_fpath.exists()

        # Act
        success, missing, failed = patched_qm.query_latest_target_cutouts("T00")

        # Assert
        assert set(success) == set()
        assert set(missing) == set(["T00"])
        assert set(failed) == set()


class Test__QueryLatestCutouts:
    def test__query_cutouts(self, patched_qm: LasairBaseQueryManager, tmp_path: Path):
        # Arrange
        patched_qm.query_target_data()
        patched_qm.config["n_cutouts"] = 3  # per-target.

        # Act
        success, missing, failed = patched_qm.query_latest_cutouts()

        # Assert
        exp_success_tags = [
            "T00-1000200030004003",  # Both target T00 and T01 have
            "T00-1000200030004004",  # the same LC/img-urls for testing.
            "T00-1000200030004005",
            "T01-1000200030004003",
            "T01-1000200030004004",
            "T01-1000200030004005",
        ]
        assert set(success) == set(exp_success_tags)


# class Test__LoadCutouts:
#     def test__perform_all_tasks(self, patched_qm: LasairBaseQueryManager, tmp_path: Path):
#         pass


class Test__PerformAllTasks:
    def test__perform_all_tasks_normal(
        self, patched_qm: LasairBaseQueryManager, tmp_path: Path
    ):
        # Arrange
        tl = patched_qm.target_lookup
        assert set(tl.keys()) == set(["T00", "T01"])  # Just a reminder...

        # Act
        patched_qm.perform_all_tasks(iteration=1)  # Do query steps.

        # Assert
        assert set(tl.keys()) == set(["T00", "T01", "T101"])  # Alerts added
        lightcurve_file = patched_qm.get_lightcurve_filepath("T00")
        assert lightcurve_file.exists()

        assert len(tl["T00"].info_messages) == 0
        assert len(tl["T101"].info_messages) == 10  #

    def test__perform_all_tasks_iter0(
        self, patched_qm: LasairBaseQueryManager, tmp_path: Path
    ):
        # Arrange
        tl = patched_qm.target_lookup
        # Write a "weird" lightcurve that is overwritten, if query_objects() called
        mock_lc = pd.DataFrame(np.arange(9).reshape(3, 3), columns="a b c".split())
        T00_lc_filepath = patched_qm.get_lightcurve_filepath("T00")
        mock_lc.to_csv(T00_lc_filepath, index=False)

        # Act
        patched_qm.perform_all_tasks(iteration=0)

        # Assert
        assert set(tl.keys()) == set(["T00", "T01"])  # None added from alerts!

        loaded_lc = pd.read_csv(T00_lc_filepath)
        exp_cols = set(["a", "b", "c"])  # Check LC is loaded (but not overwritten!)
        assert set(loaded_lc.columns) == exp_cols
        assert set(tl["T00"].target_data["lasair_cool"].lightcurve.columns) == exp_cols
