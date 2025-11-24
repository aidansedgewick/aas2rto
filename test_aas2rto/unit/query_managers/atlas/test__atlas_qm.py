import os
import pytest
from pathlib import Path
from typing import NoReturn

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from aas2rto.exc import UnknownTargetWarning
from aas2rto.lightcurve_compilers import prepare_atlas_data
from aas2rto.query_managers.atlas.atlas import (
    AtlasCredentialError,
    AtlasQueryManager,
    process_atlas_lightcurve,
)
from aas2rto.query_managers.atlas.atlas_query import AtlasQuery
from aas2rto.target import Target
from aas2rto.target_data import TargetData
from aas2rto.target_lookup import TargetLookup

from aas2rto.exc import UnexpectedKeysError, UnexpectedKeysWarning


@pytest.fixture
def atlas_qm(
    atlas_config: dict,
    tlookup: TargetLookup,
    tmp_path: Path,
    atlas_td: TargetData,
    t_fixed: Time,
):
    data = {"mjd": 60000.0, "mag": 19.0, "tag": "valid"}

    tlookup["T00"].compiled_lightcurve = prepare_atlas_data(atlas_td)
    tlookup["T00"].update_science_score_history(1.0, t_fixed)

    extra_t01 = Target("T_throttle_01", SkyCoord(ra=90.0, dec=0.0, unit="deg"))
    extra_t02 = Target("T_sleep_01", SkyCoord(ra=90.0, dec=0.0, unit="deg"))
    tlookup.add_target_list([extra_t01, extra_t02])

    return AtlasQueryManager(atlas_config, tlookup, tmp_path)


@pytest.fixture
def t_write(t_fixed: Time):
    return t_fixed + 4.0 * u.day


@pytest.fixture
def write_atlas_lc(
    atlas_qm: AtlasQueryManager, lc_atlas: pd.DataFrame, t_write: Time
) -> NoReturn:
    fpath = atlas_qm.get_lightcurve_filepath("T00")
    lc_atlas.to_csv(fpath, index=False)
    os.utime(fpath, (t_write.unix, t_write.unix))
    return None


class Test__ProcessAtlasLC:
    def test__full_lc(self, lc_atlas: pd.DataFrame):
        # Act
        processed_lc = process_atlas_lightcurve(lc_atlas)

        # Assert
        assert "mjd" in processed_lc.columns
        assert "jd" in processed_lc.columns

    def test__empty_lc(self):
        # Arrange
        empty_lc = AtlasQuery.get_empty_lightcurve(return_type="pandas")

        # Act
        processed_lc = process_atlas_lightcurve(empty_lc)

        # Assert
        assert "mjd" in processed_lc.columns
        assert "jd" in processed_lc.columns


class Test__InitAtlasQM:
    # Do NOT remove empty filetree in this class - want to check empty dirs are written.

    def test__init_with_token(self, tlookup: TargetLookup, tmp_path: Path):
        # Arrange
        config = {"credentials": {"token": 1234}}

        # Act
        qm = AtlasQueryManager(config, tlookup, tmp_path)

        # Assert
        assert qm.token == 1234
        assert qm.atlas_query.headers.keys() == set(["Authorization", "Accept"])

        exp_path = tmp_path / "atlas"
        assert exp_path.exists()

    def test__init_with_usrpwd(self, tlookup: TargetLookup, tmp_path: Path):
        # Arrange
        config = {"credentials": {"username": "user", "password": "shh_its_a_secret"}}

        # Act
        qm = AtlasQueryManager(config, tlookup, tmp_path)

        # Assert
        qm.token == 1234  # This also tests that the function is properly mocked...

    def test__usrpwd_ignorred(self, tlookup: TargetLookup, tmp_path: Path):
        # Arrange
        cred_config = {"token": 5678, "username": "user", "password": "pass"}
        config = {"credentials": cred_config}  # usr/pass combo would give token=1234

        # Act
        qm = AtlasQueryManager(config, tlookup, tmp_path)

        # Assert
        assert qm.token == 5678

    def test__no_credentials_raises(self, tlookup: TargetLookup, tmp_path: Path):
        # Arrange
        config = {}  # Missing 'credentials' kw

        # Act
        with pytest.raises(AtlasCredentialError):
            qm = AtlasQueryManager(config, tlookup, tmp_path)

    def test__bad_credentials_raises(self, tlookup: TargetLookup, tmp_path: Path):
        # Arrange
        config = {"credentials": {"blah": 100}}  # missing all relevant keys

        # Act
        with pytest.raises(AtlasCredentialError):
            qm = AtlasQueryManager(config, tlookup, tmp_path)

    def test__bad_config_raises(
        self, atlas_credentials: dict, tlookup: TargetLookup, tmp_path: Path
    ):
        # Arrange
        config = {"blah": 100.0, "credentials": atlas_credentials}

        # Act
        with pytest.raises(UnexpectedKeysError):
            qm = AtlasQueryManager(config, tlookup, tmp_path)


class Test__GetLCFilePath:

    @pytest.fixture(autouse=True)
    def _remove_empty_tmp_dirs(self, remove_tmp_dirs: NoReturn):
        pass

    def test__get_lc_filepath(self, atlas_qm: AtlasQueryManager, tmp_path: Path):
        # Act
        fpath = atlas_qm.get_lightcurve_filepath("T00")

        # Assert
        exp_path = tmp_path / "atlas/lightcurves/T00.csv"
        assert fpath == exp_path


class Test__RecoverQueryData:

    @pytest.fixture(autouse=True)
    def _remove_empty_tmp_dirs(self, remove_tmp_dirs: NoReturn):
        pass

    def test__recover_query_data(self, atlas_qm: AtlasQueryManager):
        # Act
        mock_url = "http://atlas.mock/T_ready00"
        status, lc = atlas_qm.recover_query_data("T_ready00", mock_url)

        # Assert
        assert status == 200
        assert isinstance(lc, pd.DataFrame)
        assert len(lc) == 12  # from test__aas2rto/conftest.py

        exp_lc_path = atlas_qm.get_lightcurve_filepath("T_ready00")
        assert exp_lc_path.exists()

    def test__recover_no_data(self, atlas_qm: AtlasQueryManager):
        # Act
        status, lc = atlas_qm.recover_query_data("T_no_data", "T_no_data")

        # Assert
        assert status == 200
        exp_path = atlas_qm.get_lightcurve_filepath("T_no_data")
        assert exp_path.exists()

        assert isinstance(lc, pd.DataFrame)
        assert lc.empty

    def test__empty_no_overwrite_existing(
        self, atlas_qm: AtlasQueryManager, lc_atlas: pd.DataFrame
    ):
        # Arrange
        lc_filepath = atlas_qm.get_lightcurve_filepath("T_no_data")
        lc_atlas.to_csv(lc_filepath, index=False)

        # Act
        status, lc = atlas_qm.recover_query_data("T_no_data", "T_no_data")

        # Assert
        assert status == 200
        assert len(lc) == 12  # read and return the existing non-empty

        loaded_lc = pd.read_csv(lc_filepath)
        assert len(loaded_lc) == 12  # the file was not overwritten by the empty one.


class Test__RecoverExistingQueries:

    @pytest.fixture(autouse=True)
    def _remove_empty_tmp_dirs(self, remove_tmp_dirs: NoReturn):
        pass

    def test__recoved_existing(self, atlas_qm: AtlasQueryManager):
        # Act
        finished, ongoing, errors = atlas_qm.recover_existing_queries()

        # Assert
        assert set(finished) == set(["T_ready00", "T_no_data00"])
        # We skipped T901 bc. it has another project_identifier
        assert set(ongoing) == set(["T00", "T01"])
        assert set(errors) == set(["T_bad_query00"])

        # Check that the ongoing queries are saved.
        assert set(atlas_qm.submitted_queries.keys()) == set(["T00", "T01"])

    def test__break_after_long_query(self, atlas_qm: AtlasQueryManager):
        # Arrange
        atlas_qm.config["max_query_time"] = 0.1
        # there is a sleep 0.2 in mock_page_resp

        # Act
        finished, ongoing, errors = atlas_qm.recover_existing_queries()

        # Assert
        exp_targets = ["T_ready00", "T_no_data00"]
        assert set(finished) == set(exp_targets)
        assert set(ongoing) == set(["T00"])  # break before return T01 from pg2
        # We skipped T901 bc. it has another project_identifier
        assert set(errors) == set(["T_bad_query00"])


class Test__SelectQueryCandidates:

    @pytest.fixture(autouse=True)
    def _remove_empty_tmp_dirs(self, remove_tmp_dirs: NoReturn):
        pass

    def test__select_one_candiate(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Arrange
        t_later = t_fixed + 5.0 * u.day

        # Act
        candidates = atlas_qm.select_query_candidates(t_ref=t_later)

        # Assert
        assert set(candidates) == set(["T00"])
        # T00 has detections, not too old, no existing LC, has a valid score

    def test__select_old_lc(
        self, write_atlas_lc: NoReturn, atlas_qm: AtlasQueryManager, t_write: Time
    ):
        # Arrange
        atlas_qm.config["query_lightcurve_age"] = 2.0
        t_later = t_write + 3.0 * u.day  # QM updates old LCs after 2 days.

        # Act
        candidates = atlas_qm.select_query_candidates(t_ref=t_later)

        # Assert
        assert set(candidates) == set(["T00"])

    def test__skip_new_lc(
        self, write_atlas_lc: NoReturn, atlas_qm: AtlasQueryManager, t_write: Time
    ):
        # Arrange
        atlas_qm.config["query_lightcurve_age"] = 2.0
        t_later = t_write + 1.0 * u.day  # QM updates old LCs after 2 days.

        # Act
        candidates = atlas_qm.select_query_candidates(t_ref=t_later)

        # Assert
        assert set(candidates) == set()

    def test__skip_faint_mag(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Arrange
        t_later = t_fixed + 5.0 * u.day
        atlas_qm.config["query_faint_limit"] = 15.0

        # Act
        candidates = atlas_qm.select_query_candidates(t_ref=t_later)

        # Assert
        assert set(candidates) == set()

    def test__skip_stale_lcs(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Arrange
        t_later = t_fixed + 8.0 * u.day
        atlas_qm.config["query_stale_limit"] = 1.0  # should come to 60004

        # Act
        candidates = atlas_qm.select_query_candidates(t_ref=t_later)

        # Assert
        assert set(candidates) == set()

    def test__skip_no_score(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Arrange
        t_later = t_fixed + 5.0 * u.day
        atlas_qm.target_lookup["T00"].science_score_history = []

        # Act
        candidates = atlas_qm.select_query_candidates(t_ref=t_later)

        # Assert
        assert set(candidates) == set()

    def test__skip_bad_last_score(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Arrange
        t_later = t_fixed + 5.0 * u.day
        atlas_qm.target_lookup["T00"].update_science_score_history(-1.0, t_fixed)

        # Act
        candidates = atlas_qm.select_query_candidates(t_ref=t_later)

        # Assert
        assert set(candidates) == set()

    def test__skip_no_detections(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Arrange
        t_later = t_fixed + 5.0 * u.day
        atlas_qm.target_lookup["T00"].compiled_lightcurve = None

        # Act
        candidates = atlas_qm.select_query_candidates(t_ref=t_later)

        # Assert
        assert set(candidates) == set()


class Test__SubmitQueryAndHelpers:

    @pytest.fixture(autouse=True)
    def _remove_empty_tmp_dirs(self, remove_tmp_dirs: NoReturn):
        pass  # remove_tmp_dirs defined in unit/contfest.py, executed with autouse=True

    def test__get_atlas_query_comment(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Act
        comm = atlas_qm.get_atlas_query_comment("T00")

        # Assert
        assert comm == "T00:test"

    def test__prep_payload(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Act
        payload = atlas_qm.prepare_query_payload(atlas_qm.target_lookup["T00"])

        # Assert
        assert isinstance(payload, dict)
        exp_keys = "ra dec mjd_min mjd_max send_email comment".split()
        assert set(payload.keys()) == set(exp_keys)

    def test__submit(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Arrange
        T00 = atlas_qm.target_lookup["T00"]

        # Act
        resp = atlas_qm.submit_query(T00)

        # Assert
        assert resp.__class__.__name__ == "MockPostResponse"
        assert resp.status_code == 201

    def test__submit_thottles(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Arrange
        T00 = atlas_qm.target_lookup["T00"]

        # Act
        resp = atlas_qm.submit_query(T00)

        # Arrange


class Test__SubmitQueryies:

    @pytest.fixture(autouse=True)
    def _remove_empty_tmp_dirs(self, remove_tmp_dirs: NoReturn):
        pass  # remove_tmp_dirs defined in unit/contfest.py, executed with autouse=True

    def test__submit_query(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Act
        submitted, throttled = atlas_qm.submit_new_queries(["T00"], t_ref=t_fixed)

        # Assert
        assert set(submitted) == set(["T00"])
        assert set(atlas_qm.submitted_queries.keys()) == set(["T00"])

        exp_url = "http://atlas.mock/T00"
        assert atlas_qm.submitted_queries["T00"] == exp_url  # testing patch here...

    def test__local_throttle_quits(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Arrange
        tlist = ["T00", "T01"]
        atlas_qm.config["max_submitted"] = 1

        # Act
        submitted, throttled = atlas_qm.submit_new_queries(tlist, t_ref=t_fixed)

        # Asset
        assert set(submitted) == set(["T00"])
        assert set(throttled) == set(["T01"])

        assert atlas_qm.local_throttled
        assert not atlas_qm.server_throttled

    def test__no_resubmit(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Arrange
        submitted1, throttled1 = atlas_qm.submit_new_queries(["T00"], t_ref=t_fixed)

        # Act
        submitted2, throttled2 = atlas_qm.submit_new_queries(["T00"], t_ref=t_fixed)

        # Assert
        assert set(submitted2) == set()  # didn't put T00 in a second time!!

    def test__server_throttle_quits(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Arrange
        tlist = ["T00", "T_throttle_01", "T01"]

        # Act
        submitted, throttled = atlas_qm.submit_new_queries(tlist, t_ref=t_fixed)

        # Assert
        assert set(submitted) == set(["T00"])
        assert set(throttled) == set(["T_throttle_01", "T01"])

        assert not atlas_qm.local_throttled
        assert atlas_qm.server_throttled

    def test__long_query_quits(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Arrange
        tlist = ["T_sleep_01", "T00"]
        atlas_qm.config["max_query_time"] = 0.1
        # "T_sleep" makes patched AtlasQuery sleep for 0.2sec

        # Act
        submitted, throttled = atlas_qm.submit_new_queries(tlist, t_ref=t_fixed)

        # Assert
        assert set(submitted) == set(["T_sleep_01"])

        assert not atlas_qm.local_throttled
        assert not atlas_qm.server_throttled

    def test__missing_target_warns(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Arrange
        with pytest.warns(UnknownTargetWarning):
            submitted, throttled = atlas_qm.submit_new_queries(["T_missing"])


class Test__ResubmitThrottled:

    @pytest.fixture(autouse=True)
    def _remove_empty_tmp_dirs(self, remove_tmp_dirs: NoReturn):
        pass  # remove_tmp_dirs defined in unit/contfest.py, executed with autouse=True

    def test__resubmit(self, atlas_qm: AtlasQueryManager, t_fixed: Time):
        # Arrange
        atlas_qm.throttled_queries = ["T00"]

        # Act
        submitted, throttled = atlas_qm.retry_throttled_queries()

        # Assert
        assert set(submitted) == set(["T00"])
        assert set(throttled) == set()

        assert set(atlas_qm.submitted_queries.keys()) == set(["T00"])
        assert atlas_qm.submitted_queries["T00"] == "http://atlas.mock/T00"
        assert set(atlas_qm.throttled_queries) == set()


class Test__LoadLCs:

    def test__load_single_lc(self, atlas_qm: AtlasQueryManager, lc_atlas: pd.DataFrame):
        # Arrange
        lc_filepath = atlas_qm.get_lightcurve_filepath("T00")
        lc_atlas.to_csv(lc_filepath, index=False)
        atlas_qm.config["alt_id_priority"] = ("src01", "src02")

        # Act
        lc = atlas_qm.load_single_lightcurve("T00")

        # Assert
        assert len(lc) == 12

    def test__load_from_alt_source(
        self, atlas_qm: AtlasQueryManager, lc_atlas: pd.DataFrame
    ):
        # Arrange
        lc_filepath = atlas_qm.get_lightcurve_filepath("target_A")
        lc_atlas.to_csv(lc_filepath, index=False)
        atlas_qm.config["alt_id_priority"] = ("src01", "src02")
        # T00 has src02: target_A - see unit/conftest.py

        # Act
        lc = atlas_qm.load_single_lightcurve("T00")

        # Assert
        assert len(lc) == 12
        # LCs NOT applied in load_single_lightcurve - no need to check target_data

    def test__load_empty_returns_none(self, atlas_qm: AtlasQueryManager):
        # Arrange
        empty_lc = AtlasQuery.get_empty_lightcurve(return_type="pandas")
        lc_filepath = atlas_qm.get_lightcurve_filepath("T00")
        empty_lc.to_csv(lc_filepath, index=False)

        # Act
        loaded_lc = atlas_qm.load_single_lightcurve("T00")

        # Assert
        assert loaded_lc is None

    def test__superclass_load_lightcurves(
        self, atlas_qm: AtlasQueryManager, lc_atlas: pd.DataFrame, t_fixed: Time
    ):
        # Arrange
        lc_filepath = atlas_qm.get_lightcurve_filepath("T00")
        lc_atlas.to_csv(lc_filepath, index=False)
        atlas_qm.config["alt_id_priority"] = ("src01", "src02")

        # Act
        atlas_qm.load_target_lightcurves(t_ref=t_fixed)

        # Assert
        T00 = atlas_qm.target_lookup["T00"]
        assert "atlas" in T00.target_data.keys()
        assert len(T00.target_data["atlas"].lightcurve) == 12


class Test__RenameLCs:
    def test__rename_to_alt_src(
        self, atlas_qm: AtlasQueryManager, lc_atlas: pd.DataFrame
    ):
        # Arrange
        T00_lc_filepath = atlas_qm.get_lightcurve_filepath("T00")
        lc_atlas.to_csv(T00_lc_filepath, index=False)
        atlas_qm.config["alt_id_priority"] = ("src03", "src02", "src01")
        # T00: src03 doesn't exist, src02 = target_A, src01 = T00

        # Act
        renamed, skipped = atlas_qm.update_lightcurve_filenames()

        # Arrange
        assert set(renamed) == set(["T00"])
        assert set(skipped) == set()

        new_exp_filepath = atlas_qm.get_lightcurve_filepath("target_A")
        assert new_exp_filepath.exists()
        assert not T00_lc_filepath.exists()

    def test__skip_already_renamed(
        self, atlas_qm: AtlasQueryManager, lc_atlas: pd.DataFrame
    ):
        # Arrange
        T00_lc_filepath = atlas_qm.get_lightcurve_filepath("target_A")
        lc_atlas.to_csv(T00_lc_filepath, index=False)
        atlas_qm.config["alt_id_priority"] = ("src03", "src02", "src01")
        # T00: src03 doesn't exist, src02 = target_A, src01 = T00

        # Act
        renamed, skipped = atlas_qm.update_lightcurve_filenames()

        # Arrange
        assert set(renamed) == set()
        assert set(skipped) == set(["T00"])

        new_exp_filepath = atlas_qm.get_lightcurve_filepath("target_A")
        assert new_exp_filepath.exists()


class Test__PerformAllTasks:
    def test__iter_zero(
        self, atlas_qm: AtlasQueryManager, lc_atlas: pd.DataFrame, t_fixed: Time
    ):
        # Arrange
        T01_lc_filepath = atlas_qm.get_lightcurve_filepath("T01")
        lc_atlas.to_csv(T01_lc_filepath, index=False)
        atlas_qm.config["alt_id_priority"] = ("src01", "src02")

        # Act
        atlas_qm.perform_all_tasks(iteration=0, t_ref=t_fixed)

        # Assert
        assert set(atlas_qm.submitted_queries) == set()
        assert set(atlas_qm.throttled_queries) == set()

        T00 = atlas_qm.target_lookup["T00"]
        assert "atlas" not in T00.target_data.keys()  # there's no written LC.

        # but T01 has it's LC loaded
        T01 = atlas_qm.target_lookup["T01"]
        assert "atlas" in T01.target_data.keys()

    def test__normal_iter(
        self, atlas_qm: AtlasQueryManager, lc_atlas: pd.DataFrame, t_fixed: Time
    ):
        # Arrange
        T01_lc_filepath = atlas_qm.get_lightcurve_filepath("T01")
        lc_atlas.to_csv(T01_lc_filepath, index=False)
        atlas_qm.config["alt_id_priority"] = ("src01", "src02")

        # Act
        atlas_qm.perform_all_tasks(iteration=1, t_ref=t_fixed)

        # Assert
        # submitted queries is correctly filled
        assert set(atlas_qm.submitted_queries) == set(["T00", "T01"])

        T_ready00_lc_path = atlas_qm.get_lightcurve_filepath("T_ready00")
        assert T_ready00_lc_path.exists()
        T_ready00_lc = pd.read_csv(T_ready00_lc_path)
        assert len(T_ready00_lc) == 12

        T_no_data00_lc_path = atlas_qm.get_lightcurve_filepath("T_no_data00")
        assert T_no_data00_lc_path.exists()
        T_no_data00_lc = pd.read_csv(T_no_data00_lc_path)
        assert T_no_data00_lc.empty

        T01 = atlas_qm.target_lookup["T01"]
        assert "atlas" in T01.target_data.keys()
