import pytest
from pathlib import Path

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from aas2rto import paths
from aas2rto.exc import UnknownTargetWarning
from aas2rto.query_managers.base import (
    BaseQueryManager,
    LightcurveQueryManager,
    KafkaQueryManager,
)
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup


class NoNameQM(BaseQueryManager):

    def __init__(self, config: dict, target_lookup: TargetLookup):
        pass

    def perform_all_tasks(self):
        pass


class NoPerformTasksQM(BaseQueryManager):
    name = "no_perform_tasks"

    def __init__(self, config: dict, target_lookup: TargetLookup):
        pass


class CoolQM(BaseQueryManager):
    name = "cool"

    # def __init__(
    #     self, config: dict, target_lookup: TargetLookup, parent_path: Path = None
    # ):
    #     self.config = config
    #     self.target_lookup = target_lookup
    #     # do NOT process paths here

    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        pass

    # def load_single_lightcurve(self, target_id, t_ref=None):
    #    return


class CoolLightcurveQM(LightcurveQueryManager):
    name = "cool_lc"
    id_resolving_order = ("src01", "src02")

    # def __init__(
    #     self, config: dict, target_lookup: TargetLookup, parent_path: Path = None
    # ):
    #     self.config = config
    #     self.target_lookup = target_lookup
    #     # do NOT process paths here

    def get_lightcurve_filepath(self, id_: str):
        pass  # Must implement, else init fails with ABC - see load_single_lightcurve...

    def load_single_lightcurve(self, target_id, t_ref=None):
        pass  # Must implement, else init fails with ABC error - patch in fixture...

    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        pass


class BadLCQM(LightcurveQueryManager):
    name = "bad_lc"

    def __init__(
        self, config: dict, target_lookup: TargetLookup, parent_path: Path = None
    ):
        pass

    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        pass


class CoolKafkaQM(KafkaQueryManager):
    name = "cool_kafka"
    target_id_key = "target_id"

    # def __init__(
    #     self, config: dict, target_lookup: TargetLookup, parent_path: Path = None
    # ):
    #     self.config = config
    #     self.target_lookup = target_lookup
    #     # do NOT process paths here

    def listen_for_alerts(self):
        pass

    def new_target_from_alert(self, processed_alert, t_ref=None):
        target_id = processed_alert["target_id"]
        coord = SkyCoord(processed_alert["ra"], processed_alert["dec"], unit=u.deg)
        return Target(target_id, coord)

    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        pass


##===== Fixtures =====##


@pytest.fixture
def cool_qm(tlookup: TargetLookup, tmp_path: Path):
    return CoolQM({}, tlookup, tmp_path)


@pytest.fixture
def cool_lc_qm(
    ztf_lc: pd.DataFrame,
    tlookup: TargetLookup,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    def mock_load_lc(target_id: str, t_ref=None):
        if target_id == "T00":
            return ztf_lc
        if target_id == "T01":
            return pd.DataFrame(columns="mjd mag magerr tag".split())
        return None

    qm = CoolLightcurveQM({}, tlookup, tmp_path)
    monkeypatch.setattr(qm, "load_single_lightcurve", mock_load_lc)
    return qm


@pytest.fixture
def mock_alert_list(t_fixed: Time) -> list[dict]:
    alert_list = []
    for ii in range(5):
        alert = {
            "target_id": f"T{101 + ii}",  # T101, T102, ...
            "ra": 30.0 + ii * 30.0,  # 30.0, 60.0, 90.0, 120.0, 150.0
            "dec": -45.0,
        }
        kafka_alert = alert  # no fancy needed here
        alert_list.append(kafka_alert)
    return alert_list


@pytest.fixture
def cool_kafka_qm(
    tlookup: TargetLookup,
    mock_alert_list: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    def mock_listen():
        return mock_alert_list

    qm = CoolKafkaQM({}, tlookup, tmp_path)
    monkeypatch.setattr(qm, "listen_for_alerts", mock_listen)
    return qm


##===== Actual tests start here =====##


class Test__MissingABCMethodsFails:

    def test__no_name_fails(self):
        # Act
        with pytest.raises(TypeError):
            qm = NoNameQM({}, TargetLookup(), None)

    def test__no_perform_fails(self):
        # Act
        with pytest.raises(TypeError):
            qm = NoPerformTasksQM({}, TargetLookup(), None)


class Test__InitQM:
    def test__init_qm(self, tmp_path: Path):
        # Act
        qm = CoolQM({}, TargetLookup(), parent_path=tmp_path)

        # Assert
        assert isinstance(qm, BaseQueryManager)


class Test__ProcessPaths:

    def test__with_parent_path(self, cool_qm: BaseQueryManager, tmp_path: Path):
        # Act
        cool_qm.process_paths(parent_path=tmp_path, directories=["some_data"])

        # Assert
        assert set(cool_qm.paths_lookup.keys()) == set(["some_data"])

        exp_lc_path = tmp_path / "cool/some_data"  # NAME is prepended!
        assert cool_qm.paths_lookup["some_data"] == exp_lc_path
        assert cool_qm.paths_lookup["some_data"].exists()
        assert hasattr(cool_qm, "some_data_path")

    def test__with_data_path(self, cool_qm: BaseQueryManager, tmp_path: Path):
        # Act
        cool_qm.process_paths(data_path=tmp_path, directories=["some_data"])

        # Assert
        assert cool_qm.paths_lookup["some_data"] == tmp_path / "some_data"

    def test__data_or_parent_required(self, cool_qm: BaseQueryManager):

        # Act
        with pytest.raises(ValueError):
            cool_qm.process_paths(directories=["some_data"])

    def test__data_and_parent_ignores(self, cool_qm: BaseQueryManager, tmp_path: Path):
        # Act
        cool_qm.process_paths(
            data_path=tmp_path / "some/path", parent_path=tmp_path / "to_ignore"
        )

        # Assert
        assert cool_qm.data_path == tmp_path / "some/path"
        assert cool_qm.parent_path == tmp_path / "some"

    def test__no_create(self, cool_qm: BaseQueryManager, tmp_path: Path):
        # Act
        cool_qm.process_paths(
            parent_path=tmp_path, create_paths=False, directories=["some_data"]
        )

        # Assert
        assert set(cool_qm.paths_lookup.keys()) == set(["some_data"])

        exp_lc_path = tmp_path / "cool/some_data"  # NAME is prepended!
        assert cool_qm.paths_lookup["some_data"] == exp_lc_path
        assert not cool_qm.paths_lookup["some_data"].exists()
        assert hasattr(cool_qm, "some_data_path")


class Test__ClearStaleFiles:

    def test__stale_files(self, cool_qm: BaseQueryManager, tmp_path: Path):
        # Arrange
        cool_qm.process_paths(parent_path=tmp_path, directories=["some_data"])

        data_dir = tmp_path / "cool/some_data"
        data_001 = data_dir / "data_001.csv"
        data_001.touch()
        subdir_A = data_dir / "dir_A"
        subdir_A.mkdir(exist_ok=True)
        data_A01 = subdir_A / "data_A01.csv"
        data_A01.touch()
        data_A02 = subdir_A / "data_A02.csv"
        data_A02.touch()

        t_future = Time.now() + 1.0 * u.day

        # Act
        cool_qm.clear_stale_files(t_ref=t_future, stale_age=0.5)

        # Assert
        assert data_dir.exists()
        assert not data_001.exists()
        assert not subdir_A.exists()
        assert not data_A01.exists()
        assert not data_A02.exists()

    def test__keep_fresh_files(self, cool_qm: BaseQueryManager, tmp_path: Path):
        # Arrange
        cool_qm.process_paths(parent_path=tmp_path, directories=["some_data"])

        data_dir = tmp_path / "cool/some_data"
        data_001 = data_dir / "data_001.csv"
        data_001.touch()
        subdir_A = data_dir / "dir_A"
        subdir_A.mkdir(exist_ok=True)
        data_A01 = subdir_A / "data_A01.csv"
        data_A01.touch()
        data_A02 = subdir_A / "data_A02.csv"
        data_A02.touch()

        t_future = Time.now() + 1.0 * u.day

        # Act
        cool_qm.clear_stale_files(t_ref=t_future, stale_age=1.5)

        # Assert
        assert data_dir.exists()
        assert data_001.exists()
        assert subdir_A.exists()
        assert data_A01.exists()
        assert data_A02.exists()


class Test__BadLightcurveQM:
    def test__no_single_raises(self):
        # Act
        with pytest.raises(TypeError):
            qm = BadLCQM(config={}, target_lookup={})


class Test__IdResolvingOrder:
    def test__take_preferred(
        self, cool_lc_qm: LightcurveQueryManager, basic_target: Target
    ):
        # Arrange
        assert cool_lc_qm.id_resolving_order == ("src01", "src02")  # reminder

        # Act
        relevant_name = cool_lc_qm.get_relevant_id_from_target(basic_target)

        # Assert
        assert relevant_name == "T00"

    def test__take_next_best(
        self, cool_lc_qm: LightcurveQueryManager, basic_target: Target
    ):
        # Arrange
        assert cool_lc_qm.id_resolving_order == ("src01", "src02")  # reminder
        basic_target.alt_ids.pop("src01")  # so that src02 is the only available...

        # Act
        relevant_name = cool_lc_qm.get_relevant_id_from_target(basic_target)

        # Assert
        assert relevant_name == "target_A"

    def test__no_relevant_name_returns_none(
        self, cool_lc_qm: LightcurveQueryManager, basic_target: Target
    ):
        # Arrange
        assert cool_lc_qm.id_resolving_order == ("src01", "src02")  # reminder
        basic_target.alt_ids = {}  # now neither src01 OR src02 exists!

        # Act
        relevant_name = cool_lc_qm.get_relevant_id_from_target(basic_target)

        # Assert
        assert relevant_name is None


class Test__LoadLCs:
    def test__load_lcs(self, cool_lc_qm: LightcurveQueryManager):
        # Arrange
        T00 = cool_lc_qm.target_lookup["T00"]
        assert set(T00.target_data.keys()) == set(["src01"])  # just a reminder...

        # Act
        loaded, skipped, missing = cool_lc_qm.load_target_lightcurves()

        # Assert
        assert set(T00.target_data.keys()) == set(["src01", "cool_lc"])
        assert len(T00.target_data["cool_lc"].lightcurve) == 14
        assert not T00.updated

        assert set(loaded) == set(["T00"])
        assert set(skipped) == set()
        assert set(missing) == set(["T01"])

    def test__load_lcs_flag_new_targets(self, cool_lc_qm: LightcurveQueryManager):
        # Arrange
        T00 = cool_lc_qm.target_lookup["T00"]

        # Act
        loaded, skipped, missing = cool_lc_qm.load_target_lightcurves(
            only_flag_updated=False
        )

        # Assert
        assert set(T00.target_data.keys()) == set(["src01", "cool_lc"])
        assert len(T00.target_data["cool_lc"].lightcurve) == 14
        assert T00.updated

    def test__identical_lc_not_flagged(self, cool_lc_qm: LightcurveQueryManager):
        # Arrange
        T00 = cool_lc_qm.target_lookup["T00"]
        cool_lc_qm.load_target_lightcurves()

        # Act
        loaded, skipped, missing = cool_lc_qm.load_target_lightcurves()

        # Assert
        assert set(loaded) == set()
        assert set(skipped) == set(["T00"])  # didn't do anything to T00
        assert set(missing) == set(["T01"])
        assert not T00.updated

    def test__load_named_targets_only(self, cool_lc_qm: LightcurveQueryManager):
        # Act
        loaded, skipped, missing = cool_lc_qm.load_target_lightcurves(id_list=["T00"])

        assert set(loaded) == set(["T00"])
        assert set(skipped) == set()
        assert set(missing) == set()

    def test__missing_target_warns(self, cool_lc_qm: LightcurveQueryManager):
        # Act
        with pytest.warns(UnknownTargetWarning):
            loaded, skipped, missing = cool_lc_qm.load_target_lightcurves(
                id_list=["Txx"]
            )

        assert set(loaded) == set()
        assert set(skipped) == set()
        assert set(missing) == set(["Txx"])


class Test__KafkaQMMethods:
    def test__listen_for_alerts(self, cool_kafka_qm: KafkaQueryManager):
        # Act
        alerts = cool_kafka_qm.listen_for_alerts()  # boring test!

        # Assert
        assert len(alerts) == 5

    def test__new_targets_for_alerts(
        self, cool_kafka_qm: KafkaQueryManager, mock_alert_list: list[dict]
    ):
        # Arrange
        tl = cool_kafka_qm.target_lookup
        assert set(tl.keys()) == set(["T00", "T01"])

        # Act
        targets_added = cool_kafka_qm.add_targets_from_alerts(mock_alert_list)

        # Assert
        assert set(targets_added) == set("T101 T102 T103 T104 T105".split())

        # also check that the new targets are in the tlookup...
        assert set(tl.keys()) == set("T00 T01 T101 T102 T103 T104 T105".split())
