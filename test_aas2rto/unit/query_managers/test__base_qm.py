import pytest
from pathlib import Path

import pandas as pd

from astropy import units as u
from astropy.time import Time

from aas2rto import paths
from aas2rto.exc import UnknownTargetWarning
from aas2rto.query_managers.base import BaseQueryManager
from aas2rto.target_lookup import TargetLookup


class NoNameQM(BaseQueryManager):

    def __init__(self, target_lookup: TargetLookup):
        self.target_lookup = target_lookup

    def perform_all_tasks(self):
        pass


class NoPerformTasksQM(BaseQueryManager):
    name = "no_perform_tasks"

    def __init__(self, target_lookup: TargetLookup):
        self.target_lookup = target_lookup


class CoolQM(BaseQueryManager):
    name = "cool"

    def __init__(
        self, config: dict, target_lookup: TargetLookup, parent_path: Path = None
    ):
        self.config = config
        self.target_lookup = target_lookup
        # do NOT process paths here

    def perform_all_tasks(self, iteration: int, t_ref: Time = None):
        pass

    # def load_single_lightcurve(self, target_id, t_ref=None):
    #    return


##===== Fixtures =====##


@pytest.fixture
def cool_qm(
    lc_ztf: pd.DataFrame, tlookup: TargetLookup, monkeypatch: pytest.MonkeyPatch
):
    def mock_load_lc(target_id: str, t_ref=None):
        if target_id == "T00":
            return lc_ztf
        if target_id == "T01":
            return pd.DataFrame(columns="mjd mag magerr tag".split())
        return None

    qm = CoolQM({}, tlookup)
    monkeypatch.setattr(qm, "load_single_lightcurve", mock_load_lc)
    return qm


##===== Actual tests start here =====##


class Test__MissingABCMethodsFails:

    def test__no_name_fails(self):
        # Act
        with pytest.raises(TypeError):
            qm = NoNameQM(target_lookup={})

    def test__no_perform_fails(self):
        # Act
        with pytest.raises(TypeError):
            qm = NoPerformTasksQM({})


class Test__InitQM:
    def test__init_qm(self, tmp_path: Path):
        # Act
        qm = CoolQM({}, {}, parent_path=tmp_path)

        # Assert
        assert isinstance(qm, BaseQueryManager)


class Test__ProcessPaths:

    def test__with_parent_path(self, cool_qm: BaseQueryManager, tmp_path: Path):
        # Act
        cool_qm.process_paths(parent_path=tmp_path, directories=["lightcurves"])

        # Assert
        assert set(cool_qm.paths_lookup.keys()) == set(["lightcurves"])

        exp_lc_path = tmp_path / "cool/lightcurves"  # NAME is prepended!
        assert cool_qm.paths_lookup["lightcurves"] == exp_lc_path
        assert cool_qm.paths_lookup["lightcurves"].exists()
        assert hasattr(cool_qm, "lightcurves_path")

    def test__with_data_path(self, cool_qm: BaseQueryManager, tmp_path: Path):
        # Act
        cool_qm.process_paths(data_path=tmp_path, directories=["lightcurves"])

        # Assert
        assert cool_qm.paths_lookup["lightcurves"] == tmp_path / "lightcurves"

    def test__data_or_parent_required(self, cool_qm: BaseQueryManager):

        # Act
        with pytest.raises(ValueError):
            cool_qm.process_paths(directories=["lightcurves"])

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
            parent_path=tmp_path, create_paths=False, directories=["lightcurves"]
        )

        # Assert
        assert set(cool_qm.paths_lookup.keys()) == set(["lightcurves"])

        exp_lc_path = tmp_path / "cool/lightcurves"  # NAME is prepended!
        assert cool_qm.paths_lookup["lightcurves"] == exp_lc_path
        assert not cool_qm.paths_lookup["lightcurves"].exists()
        assert hasattr(cool_qm, "lightcurves_path")


class Test__LoadLCs:
    def test__load_lcs(self, cool_qm: BaseQueryManager):
        # Arrange
        T00 = cool_qm.target_lookup["T00"]
        assert set(T00.target_data.keys()) == set(["src01"])  # just a reminder...

        # Act
        loaded, skipped, missing = cool_qm.load_target_lightcurves()

        # Assert
        assert set(T00.target_data.keys()) == set(["src01", "cool"])
        assert len(T00.target_data["cool"].lightcurve) == 14
        assert not T00.updated

        assert set(loaded) == set(["T00"])
        assert set(skipped) == set()
        assert set(missing) == set(["T01"])

    def test__load_lcs_flag_new_targets(self, cool_qm: BaseQueryManager):
        # Arrange
        T00 = cool_qm.target_lookup["T00"]

        # Act
        loaded, skipped, missing = cool_qm.load_target_lightcurves(
            only_flag_updated=False
        )

        # Assert
        assert set(T00.target_data.keys()) == set(["src01", "cool"])
        assert len(T00.target_data["cool"].lightcurve) == 14
        assert T00.updated

    def test__identical_lc_not_flagged(self, cool_qm: BaseQueryManager):
        # Arrange
        T00 = cool_qm.target_lookup["T00"]
        cool_qm.load_target_lightcurves()

        # Act
        loaded, skipped, missing = cool_qm.load_target_lightcurves()

        # Assert
        assert set(loaded) == set()
        assert set(skipped) == set(["T00"])  # didn't do anything to T00
        assert set(missing) == set(["T01"])
        assert not T00.updated

    def test__load_named_targets_only(self, cool_qm: BaseQueryManager):
        # Act
        loaded, skipped, missing = cool_qm.load_target_lightcurves(id_list=["T00"])

        assert set(loaded) == set(["T00"])
        assert set(skipped) == set()
        assert set(missing) == set()

    def test__missing_target_warns(self, cool_qm: BaseQueryManager):
        # Act
        with pytest.warns(UnknownTargetWarning):
            loaded, skipped, missing = cool_qm.load_target_lightcurves(id_list=["Txx"])

        assert set(loaded) == set()
        assert set(skipped) == set()
        assert set(missing) == set(["Txx"])

    def test__no_load_single_method(self, tlookup: TargetLookup):
        # Arrange
        qm = CoolQM({}, tlookup)

        # Act
        with pytest.raises(NotImplementedError):
            qm.load_target_lightcurves()


class Test__ClearStaleFiles:

    def test__stale_files(self, cool_qm: BaseQueryManager, tmp_path: Path):
        # Arrange
        cool_qm.process_paths(parent_path=tmp_path, directories=["lightcurves"])

        lc_dir = tmp_path / "cool/lightcurves"
        lc_001 = lc_dir / "lc_001.csv"
        lc_001.touch()
        dir_A = lc_dir / "dir_A"
        dir_A.mkdir(exist_ok=True)
        lc_A01 = dir_A / "lc_A01.csv"
        lc_A01.touch()
        lc_A02 = dir_A / "lc_A02.csv"
        lc_A02.touch()

        t_future = Time.now() + 1.0 * u.day

        # Act
        cool_qm.clear_stale_files(t_ref=t_future, stale_age=0.5)

        # Assert
        assert lc_dir.exists()
        assert not lc_001.exists()
        assert not dir_A.exists()
        assert not lc_A01.exists()
        assert not lc_A02.exists()

    def test__stale_files(self, cool_qm: BaseQueryManager, tmp_path: Path):
        # Arrange
        cool_qm.process_paths(parent_path=tmp_path, directories=["lightcurves"])

        lc_dir = tmp_path / "cool/lightcurves"
        lc_001 = lc_dir / "lc_001.csv"
        lc_001.touch()
        dir_A = lc_dir / "dir_A"
        dir_A.mkdir(exist_ok=True)
        lc_A01 = dir_A / "lc_A01.csv"
        lc_A01.touch()
        lc_A02 = dir_A / "lc_A02.csv"
        lc_A02.touch()

        t_future = Time.now() + 1.0 * u.day

        # Act
        cool_qm.clear_stale_files(t_ref=t_future, stale_age=1.5)

        # Assert
        assert lc_dir.exists()
        assert lc_001.exists()
        assert dir_A.exists()
        assert lc_A01.exists()
        assert lc_A02.exists()
