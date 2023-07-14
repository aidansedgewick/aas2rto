import os
import pytest
import warnings

import numpy as np

import pandas as pd

from alerce.core import Alerce

from astropy import units as u
from astropy.io import fits
from astropy.time import Time

from dk154_targets import Target
from dk154_targets.query_managers.alerce import (
    AlerceQueryManager,
    target_from_alerce_lightcurve,
    target_from_alerce_query_row,
    process_alerce_lightcurve,
    process_alerce_query_results,
)

from dk154_targets import paths


@pytest.fixture
def basic_config():
    return {"query_parameters": {"n_objects": 100, "lookback": 100.0}}


@pytest.fixture
def query_pattern():
    return {
        "classifier": "lc_classifier_top",
        "class": "Transient",
        "probability": 0.8,
    }


# def test__target_from_alerce():
#    target_from_alerce_lightcurve


@pytest.fixture
def query_results_rows():
    return [
        ("ZTF1000", 20, 50, 30.0, 45.0, 60000.0),
        ("ZTF1002", 15, 40, 31.0, 45.0, 60000.0),
        ("ZTF1001", 10, 30, 32.0, 45.0, 60000.0),  # we'll ignore this one on read.
        ("ZTF1001", 12, 30, 32.0, 45.0, 60000.3),  # ...as this one's updated already.
    ]


@pytest.fixture
def query_results_df(query_results_rows):
    return pd.DataFrame(
        query_results_rows,
        columns="oid ndethist ncovhist meanra meandec lastmjd".split(),
    )


@pytest.fixture
def page_results_rows_list():
    return [
        [
            ("ZTF100", 10, 20, 30.0, 45.0, 60000.0),
            ("ZTF101", 11, 21, 31.0, 45.0, 60000.0),
            ("ZTF102", 12, 22, 32.0, 45.0, 60000.0),
            ("ZTF103", 13, 23, 33.0, 45.0, 60000.0),
            ("ZTF104", 14, 24, 34.0, 45.0, 60000.0),
        ],
        [
            ("ZTF200", 10, 20, 60.0, 45.0, 60000.0),
            ("ZTF201", 11, 21, 61.0, 45.0, 60000.0),
            ("ZTF202", 12, 22, 62.0, 45.0, 60000.0),
            ("ZTF203", 13, 23, 63.0, 45.0, 60000.0),
            ("ZTF204", 14, 24, 64.0, 45.0, 60000.0),
        ],
        [
            ("ZTF300", 10, 20, 91.0, 45.0, 60000.0),
            ("ZTF301", 11, 21, 92.0, 45.0, 60000.0),
            ("ZTF302", 12, 22, 93.0, 45.0, 60000.0),
        ],
    ]


@pytest.fixture
def page_results_df_list(page_results_rows_list):
    columns = "oid ndethist ncovhist meanra meandec lastmjd".split()
    return [pd.DataFrame(rows, columns=columns) for rows in page_results_rows_list]


@pytest.fixture
def magstats_results():
    return pd.DataFrame([(1, 60000.0), (2, 60000.1)], columns="fid lastmjd".split())


@pytest.fixture
def alerce_det_rows():
    return [
        (60004.0, 23000_10000_20000_6001, 18.0, 0.3, 19.0, 1, 30.0, 45.0, True),
        (60006.0, 23000_10000_20000_6003, 18.4, 0.1, 19.0, 2, 30.0, 45.0, False),
        (60005.0, 23000_10000_20000_6002, 18.2, 0.1, 19.0, 2, 30.0, 45.0, False),
        (60007.0, 23000_10000_20000_6004, 18.6, 0.1, 19.0, 1, 30.0, 45.0, False),
    ]  # NOTE that rows 2 AND 3 are mixed up in order!


@pytest.fixture
def alerce_det_df(alerce_det_rows):
    return pd.DataFrame(
        alerce_det_rows,
        columns="mjd candid magpsf sigmapsf diffmaglim fid ra dec dubious".split(),
    )


@pytest.fixture
def alerce_ndet_rows():
    return [
        (60000.0, 1, 17.7),
        (60001.0, 2, 17.6),
        (60002.0, 1, 17.5),
        (60003.0, 2, 17.9),
    ]


@pytest.fixture
def alerce_ndet_df(alerce_ndet_rows):
    return pd.DataFrame(alerce_ndet_rows, columns="mjd fid diffmaglim".split())


@pytest.fixture
def alerce_lc(alerce_det_df, alerce_ndet_df):
    return process_alerce_lightcurve(alerce_det_df, alerce_ndet_df)


def generate_cutouts():
    cutouts = {
        k: np.random.random((60, 60)) for k in "science template difference".split()
    }
    hdul = fits.HDUList()
    for k, data in cutouts.items():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Keyword name 'STAMP_TYPE' is greater than 8 characters"
            )
            header = fits.Header({"STAMP_TYPE": k})
        hdu = fits.ImageHDU(data=data, header=header)
        hdul.append(hdu)
    return hdul


class MockBroker:
    def __init__(
        self,
        page_results_df_list=None,
        magstats_results=None,
        detections=None,
        non_detections=None,
    ):
        self.fake_results_list = page_results_df_list
        self.magstats_results = magstats_results
        self.detections = detections
        self.non_detections = non_detections
        self.ii = 0

    def query_objects(self, **kwargs):
        page = self.fake_results_list[self.ii]
        self.ii = self.ii + 1
        return page

    def query_magstats(self, oid, **kwargs):
        return self.magstats_results

    def query_detections(self, oid, **kwargs):
        return self.detections

    def query_non_detections(self, oid, **kwargs):
        return self.non_detections

    def get_stamps(self, oid, **candid):
        return generate_cutouts()


@pytest.fixture
def mock_broker(page_results_df_list, magstats_results, alerce_det_df, alerce_ndet_df):
    return MockBroker(
        page_results_df_list=page_results_df_list,
        magstats_results=magstats_results,
        detections=alerce_det_df,
        non_detections=alerce_ndet_df,
    )


class FailingMockBroker:
    def __init__(self):
        pass

    def query_objects(self, *args, **kwargs):
        raise ValueError

    def query_magstats(self, *args, **kwargs):
        raise ValueError

    def query_detections(self, *args, **kwargs):
        raise ValueError

    def query_non_detections(self, *args, **kwargs):
        raise ValueError

    def get_stamps(self, *args, **kwargs):
        raise ValueError


@pytest.fixture
def failing_mock_broker():
    return FailingMockBroker()


def test__process_alerce_lightcurve(alerce_det_df, alerce_ndet_df):
    alerce_lc = process_alerce_lightcurve(alerce_det_df, alerce_ndet_df)

    assert len(alerce_lc) == 8
    assert "tag" in alerce_lc.columns
    ulim = alerce_lc.query("tag=='upperlim'")
    assert len(ulim) == 4
    assert all(ulim["candid"] == 0)

    bq = alerce_lc.query("tag=='badquality'")
    assert len(bq) == 1
    assert bq.iloc[0].candid == 23000_10000_20000_6001

    det = alerce_lc.query("tag=='valid'")
    assert det.iloc[0].candid == 23000_10000_20000_6002  # Correctly sorted.
    assert det.iloc[1].candid == 23000_10000_20000_6003  # again
    assert det.iloc[2].candid == 23000_10000_20000_6004  # again


def test__target_from_alerce_query_row(query_results_df):
    query_results_df
    query_results_df.set_index("oid", inplace=True)

    t1 = target_from_alerce_query_row(
        query_results_df.index[0], query_results_df.iloc[0]
    )
    assert t1.objectId == "ZTF1000"
    print(t1.ra)
    assert np.isclose(t1.ra, 30.0)
    assert np.isclose(t1.dec, 45.0)


def test__target_from_alerce_query_row_bad_input(query_results_df):
    t1 = target_from_alerce_query_row("test1", {"meanra": 40.0, "meandec": 60.0})
    assert isinstance(t1, Target)
    assert np.isclose(t1.ra, 40.0)
    assert np.isclose(t1.dec, 60.0)

    row = pd.Series({"blah": 20})
    t1 = target_from_alerce_query_row("test2", row)
    assert t1 is None

    with pytest.raises(ValueError):
        t_failed = target_from_alerce_query_row("test1", query_results_df)


def test__target_from_alerce_lightcurve(alerce_lc):
    t1 = target_from_alerce_lightcurve("t1", alerce_lc)

    assert t1.objectId == "t1"
    assert np.isclose(t1.ra, 30.0)
    assert np.isclose(t1.dec, 45.0)

    assert isinstance(t1, Target)
    assert len(t1.alerce_data.non_detections) == 4
    assert len(t1.alerce_data.detections) == 4


class Test__AlerceQueryManager:
    exp_alerce_path = paths.test_data_path / "alerce"
    exp_lightcurves_path = paths.test_data_path / "alerce/lightcurves"
    exp_alerts_path = paths.test_data_path / "alerce/alerts"
    exp_query_results_path = paths.test_data_path / "alerce/query_results"
    exp_probabilities_path = paths.test_data_path / "alerce/probabilities"
    exp_parameters_path = paths.test_data_path / "alerce/parameters"
    exp_magstats_path = paths.test_data_path / "alerce/magstats"
    exp_cutouts_path = paths.test_data_path / "alerce/cutouts"

    @classmethod
    def _clear_test_directories(cls, extensions=("csv", "json", "fits")):
        for path in [
            cls.exp_lightcurves_path,
            cls.exp_alerts_path,
            cls.exp_query_results_path,
            cls.exp_probabilities_path,
            cls.exp_parameters_path,
            cls.exp_magstats_path,
            cls.exp_cutouts_path,
        ]:
            if path.exists():
                for subpath in path.glob("*"):
                    if subpath.is_dir():
                        for ext in extensions:
                            for filepath in subpath.glob(f"*.{ext}"):
                                os.remove(filepath)
                        subpath.rmdir()

                for ext in extensions:
                    for filepath in path.glob(f"*.{ext}"):
                        os.remove(filepath)
                path.rmdir()
        if cls.exp_alerce_path.exists():
            cls.exp_alerce_path.rmdir()

    def test__alerce_qm_init(self):
        self._clear_test_directories()
        assert not self.exp_alerce_path.exists()

        empty_config = {}

        qm1 = AlerceQueryManager(empty_config, {}, data_path=paths.test_data_path)
        assert qm1.parent_data_path == self.exp_alerce_path
        assert self.exp_alerce_path.exists()
        assert self.exp_lightcurves_path.exists()
        assert self.exp_query_results_path.exists()
        assert self.exp_probabilities_path.exists()
        assert self.exp_magstats_path.exists()
        assert self.exp_cutouts_path.exists()

        self._clear_test_directories()
        assert not self.exp_alerce_path.exists()

        # Test create_paths=False works
        qm2 = AlerceQueryManager(
            empty_config, {}, data_path=paths.test_data_path, create_paths=False
        )
        assert not self.exp_alerce_path.exists()

    def test__alerce_get_query_results_file(self, basic_config):
        exp_query_results_file = (
            paths.test_data_path / "alerce/query_results/test_query.csv"
        )
        qm = AlerceQueryManager(
            basic_config, {}, data_path=paths.test_data_path, create_paths=False
        )
        query_results_file = qm.get_query_results_file("test_query")
        assert query_results_file == query_results_file

    def test__init_uses_config(self, basic_config):
        qm2 = AlerceQueryManager(
            basic_config, {}, data_path=paths.test_data_path, create_paths=False
        )
        assert np.isclose(
            qm2.default_query_parameters["lookback"], 30.0
        )  # default is different!
        assert not self.exp_alerce_path.exists()
        assert qm2.query_parameters["n_objects"] == 100
        assert np.isclose(qm2.query_parameters["lookback"], 100.0)
        assert np.isclose(qm2.query_parameters["update"], 2.0)

    def test__get_query_data(self, basic_config, query_pattern: dict):
        qm = AlerceQueryManager(
            basic_config, {}, data_path=paths.test_data_path, create_paths=False
        )

        t_ref = Time("2023-02-25T00:00:00", format="isot")
        query_data = qm.prepare_query_data(query_pattern, page=10, t_ref=t_ref)
        assert qm.default_query_parameters["n_objects"] == 25
        assert np.isclose(query_data["firstmjd"], 59900.0)
        assert np.isclose(query_data["lastmjd"], 60000.0)
        assert query_data["page_size"] == 100

    def test__query_and_collate_read_existing(self, page_results_df_list):
        config = {"query_parameters": {"n_objects": 5}}
        self._clear_test_directories()

        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)
        assert self.exp_query_results_path.exists()

        page1_path = paths.test_data_path / "alerce/query_results/test_query_001.csv"
        page2_path = paths.test_data_path / "alerce/query_results/test_query_002.csv"
        page3_path = paths.test_data_path / "alerce/query_results/test_query_003.csv"
        assert not page1_path.exists()
        assert not page2_path.exists()
        assert not page3_path.exists()

        # write the fake data to files.
        assert len(page_results_df_list) == 3
        page_results_df_list[0].to_csv(page1_path, index=False)
        page_results_df_list[1].to_csv(page2_path, index=False)
        page_results_df_list[2].to_csv(page3_path, index=False)

        t_ref = Time.now()
        query_results = qm.query_and_collate_pages("test_query", {}, t_ref=t_ref)
        assert len(query_results) == 13
        assert not page1_path.exists()
        assert not page2_path.exists()
        assert not page3_path.exists()
        self._clear_test_directories()

    def test__query_and_collate_fetch_new(self, mock_broker, monkeypatch):
        config = {"query_parameters": {"n_objects": 5}}
        self._clear_test_directories()
        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)
        assert self.exp_query_results_path.exists()
        t_ref = Time(60000.0, format="mjd")

        query_name = "some_query"

        page1_path = paths.test_data_path / f"alerce/query_results/{query_name}_001.csv"
        page2_path = paths.test_data_path / f"alerce/query_results/{query_name}_002.csv"
        page3_path = paths.test_data_path / f"alerce/query_results/{query_name}_003.csv"
        page4_path = paths.test_data_path / f"alerce/query_results/{query_name}_004.csv"

        assert isinstance(qm.alerce_broker, Alerce)
        with monkeypatch.context() as m:
            m.setattr(qm, "alerce_broker", mock_broker)
            print(qm.alerce_broker)
            assert isinstance(qm.alerce_broker, MockBroker)  # different!!
            query_results = qm.query_and_collate_pages(
                query_name, {}, t_ref, delete_pages=False
            )
            assert len(query_results) == 13

            assert page1_path.exists()
            assert page2_path.exists()
            assert page3_path.exists()
            assert not page4_path.exists()
            os.remove(page1_path)
            os.remove(page2_path)
            os.remove(page3_path)

            # check normal behaviour doesn't keep page results.
            query_results = qm.query_and_collate_pages(query_name, {}, t_ref)
            assert not page1_path.exists()
            assert not page2_path.exists()
            assert not page3_path.exists()

        assert isinstance(qm.alerce_broker, Alerce)  # back to normal!

    def test__read_existing_queries(
        self, query_pattern: dict, query_results_df: pd.DataFrame
    ):
        config = {"object_queries": {"test_query": query_pattern}}

        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)
        assert self.exp_query_results_path.exists()

        fake_results_path = paths.test_data_path / "alerce/query_results/test_query.csv"
        query_results_df.to_csv(fake_results_path, index=False)

        qm.query_for_updates()
        assert len(qm.query_results.keys()) == 0
        assert len(qm.query_updates.keys()) == 1
        assert "test_query" in qm.query_updates
        assert set(qm.query_updates["test_query"].index) == set(
            ["ZTF1000", "ZTF1001", "ZTF1002"]
        )
        assert (
            qm.query_updates["test_query"].loc["ZTF1001"]["ndethist"] == 12
        )  # pick the LATEST one.
        assert fake_results_path.exists()  # has NOT been deleted.

        os.remove(fake_results_path)
        assert not fake_results_path.exists()
        self._clear_test_directories()

    def test__query_for_new_targets(self, mock_broker, monkeypatch):
        config = {
            "object_queries": {"query_test": {}},
            "query_parameters": {"n_objects": 5},
        }
        self._clear_test_directories()
        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)

        exp_query_results_file = (
            paths.test_data_path / "alerce/query_results/query_test.csv"
        )

        assert isinstance(qm.alerce_broker, Alerce)  # Normal
        with monkeypatch.context() as m:
            m.setattr(qm, "alerce_broker", mock_broker)
            assert isinstance(qm.alerce_broker, MockBroker)  # Different!

            qm.query_for_updates()
            assert len(qm.query_updates["query_test"]) == 13
            assert exp_query_results_file.exists()

    def test__get_magstats_to_query(self, magstats_results):
        self._clear_test_directories()
        config = {"query_parameters": {"interval": 2.0}}
        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)

        exp_magstats1_file = paths.test_data_path / "alerce/magstats/test1.csv"
        assert not exp_magstats1_file.exists()

        magstats_results.to_csv(exp_magstats1_file, index=False)
        t_now = Time.now()
        input_list = ["test1", "test2", "test3"]
        to_query = qm.get_magstats_to_query(oid_list=input_list, t_ref=t_now)
        assert set(to_query) == set(["test2", "test3"])

        t_ref = t_now + 2.1 * u.day
        to_query = qm.get_magstats_to_query(input_list, t_ref=t_ref)
        assert set(to_query) == set(["test1", "test2", "test3"])  # t1 is old

        test1 = Target("test1", ra=30.0, dec=45.0)
        qm.target_lookup[test1.objectId] = test1
        to_query = qm.get_magstats_to_query(t_ref=t_ref)  # with no input list
        assert set(to_query) == set(["test1"])

    def test__perform_magstats_query(self, mock_broker, monkeypatch):
        self._clear_test_directories()
        empty_config = {}
        qm = AlerceQueryManager(empty_config, {}, data_path=paths.test_data_path)

        exp_magstats_file = paths.test_data_path / "alerce/magstats/test1.csv"

        assert isinstance(qm.alerce_broker, Alerce)
        with monkeypatch.context() as m:
            m.setattr(qm, "alerce_broker", mock_broker)
            assert isinstance(qm.alerce_broker, MockBroker)
            assert not hasattr(qm.alerce_broker, "ztf_url")

            success, failed = qm.perform_magstats_queries(["test1"])
            assert set(success) == set(["test1"])
            assert exp_magstats_file.exists()
            magstats = pd.read_csv(exp_magstats_file)
            assert all(magstats.fid.values == np.array([1, 2]))
            assert np.allclose(magstats.lastmjd, [60000.0, 60000.1])
        assert isinstance(qm.alerce_broker, Alerce)

        self._clear_test_directories()

    def test__perform_magstats_query_break_after_failures(
        self, failing_mock_broker, monkeypatch
    ):
        self._clear_test_directories()

        empty_config = {"query_parameters": {"max_failed_queries": 3}}
        qm = AlerceQueryManager(empty_config, {}, data_path=paths.test_data_path)

        assert isinstance(qm.alerce_broker, Alerce)
        with monkeypatch.context() as m:
            m.setattr(qm, "alerce_broker", failing_mock_broker)
            assert isinstance(qm.alerce_broker, FailingMockBroker)
            success, failed = qm.perform_magstats_queries(
                ["t1", "t2", "t3", "t4", "t5"]
            )
            assert set(failed) == set(["t1", "t2", "t3"])
        assert isinstance(qm.alerce_broker, Alerce)
        self._clear_test_directories()

    def test__new_targets_from_updates_single_query(self, query_results_df):
        self._clear_test_directories()

        raw_query_results = query_results_df.iloc[[0, 2]]  # ZTF1000 and 'old' ZTF1001
        query_results = process_alerce_query_results(raw_query_results)
        assert set(query_results.index) == set(["ZTF1000", "ZTF1001"])

        query_updates = process_alerce_query_results(query_results_df)
        assert len(query_updates) == 3

        empty_config = {}
        qm = AlerceQueryManager(empty_config, {}, data_path=paths.test_data_path)
        qm.query_results["test_query"] = query_results  # Existing
        qm.query_updates["test_query"] = query_updates

        target_updates = qm.target_updates_from_query_results()
        assert len(target_updates) == 2
        assert set(target_updates.index) == set(["ZTF1002", "ZTF1001"])

        assert np.isclose(target_updates.loc["ZTF1001"].lastmjd - 60000.0, 0.3)
        self._clear_test_directories()

    def test__new_targets_from_updates(self, query_results_df):
        self._clear_test_directories()

        query_updates = process_alerce_query_results(query_results_df)
        assert len(query_updates) == 3

        config = {}
        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)

        qm.new_targets_from_updates(query_updates)

        assert len(qm.target_lookup) == 3
        assert set(qm.target_lookup.keys()) == set("ZTF1000 ZTF1001 ZTF1002".split())
        self._clear_test_directories()

    def test__new_targets_from_updates_bad_input(self, query_results_df):
        self._clear_test_directories()

        query_results_df.drop(["meanra"], axis=1, inplace=True)
        query_updates = process_alerce_query_results(query_results_df)

        config = {}
        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)
        qm.new_targets_from_updates(query_updates)

        assert len(qm.target_lookup) == 0

    def test__get_lightcurves_to_query(self, alerce_lc):
        self._clear_test_directories()

        config = {}
        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)
        test1 = Target("test1", ra=30.0, dec=45.0)
        test2 = Target("test2", ra=60.0, dec=45.0)
        qm.target_lookup[test1.objectId] = test1
        qm.target_lookup[test2.objectId] = test2

        exp_test1_lc_file = paths.test_data_path / "alerce/lightcurves/test1.csv"
        alerce_lc.to_csv(exp_test1_lc_file, index=False)
        t_now = Time.now()

        to_query = qm.get_lightcurves_to_query(require_magstats=False)
        assert set(to_query) == set(["test2"])

        t_ref = t_now + 2.1 * u.day
        to_query = qm.get_lightcurves_to_query(t_ref=t_ref, require_magstats=False)
        assert set(to_query) == set(["test1", "test2"])  # test1 is old now

        to_query = qm.get_lightcurves_to_query(
            ["test1", "test3"], require_magstats=False
        )
        assert set(to_query) == set(["test3"])
        self._clear_test_directories()

    def test__perform_lightcurve_queries(self, mock_broker, monkeypatch):
        self._clear_test_directories()

        exp_test1_lc_file = paths.test_data_path / "alerce/lightcurves/test1.csv"
        exp_test2_lc_file = paths.test_data_path / "alerce/lightcurves/test2.csv"
        assert not exp_test1_lc_file.exists()
        assert not exp_test2_lc_file.exists()

        config = {}
        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)

        assert isinstance(qm.alerce_broker, Alerce)
        with monkeypatch.context() as m:
            m.setattr(qm, "alerce_broker", mock_broker)
            assert isinstance(qm.alerce_broker, MockBroker)  # different!

            qm.perform_lightcurve_queries(["test1", "test2"])

        assert isinstance(qm.alerce_broker, Alerce)

        assert exp_test1_lc_file.exists()
        assert exp_test2_lc_file.exists()

        lc1 = pd.read_csv(exp_test1_lc_file)
        assert len(lc1) == 8
        assert "tag" in lc1.columns
        self._clear_test_directories()

    def test__query_lightcurves_breaks_after_failures(
        self, failing_mock_broker, monkeypatch
    ):
        self._clear_test_directories()

        config = {"query_parameters": {"max_failed_queries": 3}}
        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)

        assert isinstance(qm.alerce_broker, Alerce)
        with monkeypatch.context() as m:
            m.setattr(qm, "alerce_broker", failing_mock_broker)  # all calls will fail
            assert isinstance(qm.alerce_broker, FailingMockBroker)

            input_list = "t1 t2 t3 t4 t5 t6".split()
            success, failed = qm.perform_lightcurve_queries(input_list)
            assert set(failed) == set(["t1", "t2", "t3"])

    def test__load_lightcurve_queries(self, alerce_lc):
        self._clear_test_directories()

        config = {}
        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)

        exp_test1_lc_file = paths.test_data_path / "alerce/lightcurves/test1.csv"
        exp_test2_lc_file = paths.test_data_path / "alerce/lightcurves/test2.csv"
        exp_test3_lc_file = paths.test_data_path / "alerce/lightcurves/test3.csv"

        alerce_lc.to_csv(exp_test1_lc_file, index=False)
        alerce_lc.to_csv(exp_test2_lc_file, index=False)
        alerce_lc.to_csv(exp_test3_lc_file, index=False)

        test1 = Target("test1", ra=30.0, dec=45.0)
        test1.alerce_data.add_lightcurve(alerce_lc)
        test2 = Target("test2", ra=60.0, dec=45.0)
        test2.alerce_data.add_lightcurve(alerce_lc.copy().iloc[:-1])
        assert len(test2.alerce_data.detections) == 3

        test3 = Target("test3", ra=90.0, dec=45.0)
        assert test3.alerce_data.detections is None
        test4 = Target("test4", ra=120.0, dec=45.0)

        qm.target_lookup[test1.objectId] = test1
        qm.target_lookup[test2.objectId] = test2
        qm.target_lookup[test3.objectId] = test3
        qm.target_lookup[test4.objectId] = test4

        input_list = "test1 test2 test3 test4".split()  # not needed....
        loaded, missing = qm.load_target_lightcurves()
        assert set(loaded) == set(["test2", "test3"])
        assert set(missing) == set(["test4"])

        assert len(test2.alerce_data.detections) == 4  # updated!
        assert len(test3.alerce_data.detections) == 4  # added!

    def test__get_cutouts_to_query(self, alerce_lc):
        self._clear_test_directories()

        config = {}
        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)

        t2_cutouts = generate_cutouts()
        candid = 23000_10000_20000_6004
        exp_t2cutouts_file = paths.test_data_path / f"alerce/cutouts/t2/{candid}.fits"
        exp_t2cutouts_file.parent.mkdir(parents=False, exist_ok=False)
        t2_cutouts.writeto(exp_t2cutouts_file)

        t3_cutouts = generate_cutouts()
        candid = 23000_10000_20000_6004
        exp_t3cutouts_file = paths.test_data_path / f"alerce/cutouts/t3/{candid}.fits"
        exp_t3cutouts_file.parent.mkdir(parents=False, exist_ok=False)
        t3_cutouts.writeto(exp_t3cutouts_file)

        t1 = Target("t1", ra=30.0, dec=45.0)
        t1.alerce_data.add_lightcurve(alerce_lc)
        t2 = Target("t2", ra=60.0, dec=45.0)
        # No detections - should skip, even though cutouts exist!
        t3 = Target("t3", ra=90.0, dec=45.0)
        t3.alerce_data.add_lightcurve(alerce_lc)

        qm.target_lookup[t1.objectId] = t1
        qm.target_lookup[t2.objectId] = t2
        qm.target_lookup[t3.objectId] = t3

        t_now = Time.now()
        to_query = qm.get_cutouts_to_query()

        assert set(to_query) == set(["t1"])

        t_ref = t_now + 2.1 * u.day
        to_query = qm.get_cutouts_to_query(t_ref=t_ref)
        assert set(to_query) == set(["t1", "t3"])  # Now t3 cuouts are old.

        self._clear_test_directories()

    def test__perform_cutout_queries(self, alerce_lc, mock_broker, monkeypatch):
        self._clear_test_directories()

        config = {}
        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)

        t1 = Target("t1", ra=30.0, dec=45.0)
        t2 = Target("t2", ra=60.0, dec=45.0)
        t1.alerce_data.add_lightcurve(alerce_lc)
        t2.alerce_data.add_lightcurve(alerce_lc)

        qm.target_lookup[t1.objectId] = t1
        qm.target_lookup[t2.objectId] = t2

        candid = 23000_10000_20000_6004
        exp_t1_cutouts_path = paths.test_data_path / f"alerce/cutouts/t1/{candid}.fits"
        exp_t2_cutouts_path = paths.test_data_path / f"alerce/cutouts/t2/{candid}.fits"
        assert not exp_t1_cutouts_path.exists()
        assert not exp_t2_cutouts_path.exists()

        assert isinstance(qm.alerce_broker, Alerce)
        with monkeypatch.context() as m:
            m.setattr(qm, "alerce_broker", mock_broker)
            assert isinstance(qm.alerce_broker, MockBroker)
            success, failed = qm.perform_cutouts_queries(["t1", "t2", "t3"])
            assert set(success) == set(["t1", "t2"])
            assert set(failed) == set(["t3"])  # t3 is not in target lookup!
        assert isinstance(qm.alerce_broker, Alerce)

        assert exp_t1_cutouts_path.exists()
        assert exp_t2_cutouts_path.exists()

        assert t1.alerce_data.meta["cutouts_candid"] == candid
        assert t2.alerce_data.meta["cutouts_candid"] == candid
        self._clear_test_directories()

    def test__perform_cutouts_query_break_after_failures(
        self, failing_mock_broker, monkeypatch
    ):
        self._clear_test_directories()

        empty_config = {"query_parameters": {"max_failed_queries": 3}}
        qm = AlerceQueryManager(empty_config, {}, data_path=paths.test_data_path)

        for oid in ["t1", "t2", "t3", "t4", "t5"]:
            t = Target(oid, ra=30.0, dec=45.0)
            qm.target_lookup[t.objectId] = t

        assert isinstance(qm.alerce_broker, Alerce)
        with monkeypatch.context() as m:
            m.setattr(qm, "alerce_broker", failing_mock_broker)
            assert isinstance(qm.alerce_broker, FailingMockBroker)
            success, failed = qm.perform_cutouts_queries(["t1", "t2", "t3", "t4", "t5"])
            assert set(failed) == set(["t1", "t2", "t3"])
        assert isinstance(qm.alerce_broker, Alerce)
        self._clear_test_directories()

    def test__load_cutouts(self, alerce_lc):
        self._clear_test_directories()

        config = {}
        qm = AlerceQueryManager(config, {}, data_path=paths.test_data_path)

        cutouts_dir = paths.test_data_path / "alerce/cutouts"
        cutout_paths = [
            cutouts_dir / "t1/2300010000200006003.fits",
            cutouts_dir / "t1/2300010000200006004.fits",
            cutouts_dir / "t2/2300010000200006003.fits",
            # no 6004 for t2
            cutouts_dir / "t3/2300010000200006003.fits",
            cutouts_dir / "t3/2300010000200006004.fits",
        ]

        for p in cutout_paths:
            cutouts = generate_cutouts()
            p.parent.mkdir(exist_ok=True, parents=False)
            cutouts.writeto(p)

        t1 = Target("t1", ra=30.0, dec=45.0)
        t1.alerce_data.add_lightcurve(alerce_lc)
        t2 = Target("t2", ra=60.0, dec=45.0)
        t2.alerce_data.add_lightcurve(alerce_lc)
        t3 = Target("t3", ra=90.0, dec=45.0)
        t3.alerce_data.add_lightcurve(alerce_lc)
        with fits.open(cutouts_dir / "t3/2300010000200006004.fits") as hdul:
            for hdu in hdul:
                imtype = hdu.header["STAMP_TYPE"]
                t3.alerce_data.cutouts[imtype] = hdu.data
        t3.alerce_data.meta["cutouts_candid"] = 23000_10000_20000_6004
        assert t3.alerce_data.cutouts["science"].shape == (60, 60)
        assert t3.alerce_data.cutouts["template"].shape == (60, 60)
        assert t3.alerce_data.cutouts["difference"].shape == (60, 60)

        t4 = Target("t4", ra=120.0, dec=45.0)
        t4.alerce_data.add_lightcurve(alerce_lc)

        qm.target_lookup[t1.objectId] = t1
        qm.target_lookup[t2.objectId] = t2
        qm.target_lookup[t3.objectId] = t3
        qm.target_lookup[t4.objectId] = t4

        loaded, missing = qm.load_cutouts()
        assert set(loaded) == set(["t1", "t2"])
        assert set(missing) == set(["t4"])

        assert t1.alerce_data.meta.get("cutouts_candid", None) == 23000_10000_20000_6004
        assert t2.alerce_data.meta.get("cutouts_candid", None) == 23000_10000_20000_6003
        assert t3.alerce_data.meta.get("cutouts_candid", None) == 23000_10000_20000_6004
        assert t4.alerce_data.meta.get("cutouts_candid", None) is None

        assert t1.alerce_data.cutouts["science"].shape == (60, 60)
        assert t1.alerce_data.cutouts["template"].shape == (60, 60)
        assert t1.alerce_data.cutouts["difference"].shape == (60, 60)

        assert t2.alerce_data.cutouts["science"].shape == (60, 60)
        assert t2.alerce_data.cutouts["template"].shape == (60, 60)
        assert t2.alerce_data.cutouts["difference"].shape == (60, 60)

        self._clear_test_directories()
