import copy
import json
import os
import pytest

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.time import Time

from dk154_targets import Target
from dk154_targets.query_managers.exc import BadKafkaConfigError, MissingObjectIdError
from dk154_targets.query_managers import fink
from dk154_targets.query_managers.fink import (
    FinkQueryManager,
    FinkQuery,
    process_fink_lightcurve,
    target_from_fink_lightcurve,
)

from dk154_targets import paths


@pytest.fixture
def det_rows():
    return [
        (23000_10000_20000_5005, 2460005.5, 18.0, 0.1, 19.0, 30.0, 45.0),
        (23000_10000_20000_5006, 2460006.5, 18.0, 0.1, 19.5, 30.0, 45.0),
        (23000_10000_20000_5007, 2460007.5, 18.0, 0.1, 19.5, 30.0, 45.0),
    ]


@pytest.fixture
def det_df(det_rows):
    return pd.DataFrame(
        det_rows, columns="candid jd mag magerr diffmaglim ra dec".split()
    )


@pytest.fixture
def ndet_rows():
    return [
        (np.nan, 2460001.5, np.nan, np.nan, 17.0, np.nan, np.nan, "upperlim"),
        (np.nan, 2460002.5, np.nan, np.nan, 17.0, np.nan, np.nan, "upperlim"),
        (np.nan, 2460003.5, np.nan, np.nan, 17.0, np.nan, np.nan, "upperlim"),
        (np.nan, 2460004.5, 18.5, 0.5, 19.0, np.nan, np.nan, "badquality"),
        (23000_10000_20000_5005, 2460005.5, 18.2, 0.1, 19.0, 30.0, 45.0, "valid"),
        (23000_10000_20000_5006, 2460006.5, 18.3, 0.1, 19.5, 30.0, 45.0, "valid"),
        (23000_10000_20000_5007, 2460007.5, 18.0, 0.1, 19.5, 30.0, 45.0, "valid"),
    ]


@pytest.fixture
def ndet_df(ndet_rows):
    return pd.DataFrame(
        ndet_rows, columns="candid jd mag magerr diffmaglim ra dec tag".split()
    )


@pytest.fixture()
def fink_lc(det_df, ndet_df):
    return process_fink_lightcurve(det_df, ndet_df)


@pytest.fixture
def fink_target(fink_lc):
    return target_from_fink_lightcurve(fink_lc, "ZTF23abcfake")


def test__process_fink_lightcurve(det_df, ndet_df):
    assert "tag" not in det_df.columns

    lc = process_fink_lightcurve(det_df, ndet_df)
    assert len(lc) == 7
    valid = lc.query("tag=='valid'")
    assert set(valid["candid"]) == set(
        [23000_10000_20000_5005, 23000_10000_20000_5006, 23000_10000_20000_5007]
    )
    ulim = lc.query("tag=='upperlim'")
    assert all(ulim["candid"] == 0)


def test__error_on_mismatching_inputs(det_df, ndet_df):
    # test that the checks for sensible input raise ValueError
    bad_det_df = det_df.iloc[:-1]
    assert len(bad_det_df) == 2
    assert set(bad_det_df["candid"]) == set(
        [23000_10000_20000_5005, 23000_10000_20000_5006]
    )
    with pytest.raises(ValueError):
        bad_lc = process_fink_lightcurve(bad_det_df, ndet_df)


def test__error_on_ndet_candid(det_df, ndet_df):
    # test that the non-detections with candid raises an error.
    bad_ndet = ndet_df
    bad_ndet.loc[0, "candid"] = 23000_10000_20000_9990
    with pytest.raises(ValueError):
        bad_lc = process_fink_lightcurve(det_df, bad_ndet)


def test__target_from_fink_lightcurve(fink_lc):
    target = target_from_fink_lightcurve(fink_lc, "test1")
    assert target.objectId == "test1"
    assert np.isclose(target.ra, 30.0)
    assert np.isclose(target.dec, 45.0)
    assert target.updated

    assert len(target.fink_data.lightcurve) == 7
    assert "mjd" in target.fink_data.detections.columns

    assert len(target.fink_data.detections) == 4
    assert len(target.fink_data.non_detections) == 3

    assert np.allclose(
        target.fink_data.detections["jd"] - 2460000.5, [4.0, 5.0, 6.0, 7.0]
    )

    with pytest.raises(TypeError):
        bad_target = target_from_fink_lightcurve(fink_lc)


class Test__FinkQueryManager:
    exp_fink_path = paths.test_data_path / "fink"
    exp_lightcurves_path = paths.test_data_path / "fink/lightcurves"
    exp_alerts_path = paths.test_data_path / "fink/alerts"
    exp_query_results_path = paths.test_data_path / "fink/query_results"
    exp_probabilities_path = paths.test_data_path / "fink/probabilities"
    exp_magstats_path = paths.test_data_path / "fink/magstats"
    exp_cutouts_path = paths.test_data_path / "fink/cutouts"

    @classmethod
    def _clear_test_directories(cls, extensions=("csv", "json")):
        for path in [
            cls.exp_lightcurves_path,
            cls.exp_alerts_path,
            cls.exp_query_results_path,
            cls.exp_probabilities_path,
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
        if cls.exp_fink_path.exists():
            cls.exp_fink_path.rmdir()

    @pytest.fixture
    def kafka_config(self):
        return {
            "username": "test_user",
            "group_id": "test_1234",
            "bootstrap.servers": "123.456.789.0",
            "topics": ["ztf_sso_ztf_alerts"],
            "n_alerts": 10,
        }

    @pytest.fixture
    def query_parameters(self):
        return {}

    @pytest.fixture
    def example_config(self, kafka_config, query_parameters):
        return {"kafka_config": kafka_config, "query_parameters": query_parameters}

    @pytest.fixture
    def fink_qm(self, example_config):
        return FinkQueryManager(
            example_config, {}, data_path=paths.test_data_path, create_paths=False
        )

    @pytest.fixture
    def fake_candidate_base(self):
        return dict(ra=30.0, dec=45.0)

    @pytest.fixture
    def fake_alert_base(self):
        return dict(
            objectId="ZTF23abcfake",
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
    def fake_alert_list(self, fake_candidate_base, fake_alert_base):
        alert_list = []
        for ii in range(15):
            mjd = Time(60010.0 + ii, format="mjd")  # mjd60010 = 7-mar-23
            candidate = copy.deepcopy(fake_candidate_base)
            candidate["magpsf"] = 19.0 - 0.1 * ii
            candidate["jd"] = mjd.jd
            alert = copy.deepcopy(fake_alert_base)
            alert["candidate"] = candidate
            alert["candid"] = 23000_10000_20000_1000 + (ii + 1)
            alert["timestamp"] = mjd.strftime("%y%m%d %H%M%S")
            alert_list.append(alert)
        return alert_list

    @pytest.fixture
    def fake_alert_results(self, fake_alert_list):
        results = [("interesting_topic", alert, "key_") for alert in fake_alert_list]
        return results

    def test__init_fink_qm(self):
        empty_config = {}

        self._clear_test_directories()
        qm = FinkQueryManager(empty_config, {}, data_path=paths.test_data_path)

        assert isinstance(qm.fink_config, dict)
        assert len(qm.fink_config) == 0

        assert qm.kafka_config is None

        assert qm.parent_data_path == self.exp_fink_path
        assert self.exp_fink_path.exists()
        assert self.exp_lightcurves_path.exists()
        assert self.exp_alerts_path.exists()
        assert self.exp_query_results_path.exists()
        assert self.exp_probabilities_path.exists()
        assert self.exp_magstats_path.exists()
        assert self.exp_cutouts_path.exists()

        self._clear_test_directories()
        assert not self.exp_fink_path.exists()

        qm = FinkQueryManager(
            empty_config, {}, data_path=paths.test_data_path, create_paths=False
        )
        assert qm.parent_data_path == self.exp_fink_path
        assert not self.exp_fink_path.exists()  # Not created as appropriate

    def test__uses_config(self, example_config):
        example_config["kafka_config"]["n_alerts"] = 25
        example_config["query_parameters"]["max_failed_queries"] = 20

        self._clear_test_directories()
        qm = FinkQueryManager(
            example_config, {}, data_path=paths.test_data_path, create_paths=False
        )
        assert qm.parent_data_path == self.exp_fink_path
        assert not self.exp_fink_path.exists()

        assert qm.default_kafka_parameters["n_alerts"] == 10  # Default remains
        assert qm.kafka_config["n_alerts"] == 25  # Correctly set

        assert qm.default_query_parameters["max_failed_queries"] == 10
        assert qm.query_parameters["max_failed_queries"] == 20
        assert np.isclose(qm.query_parameters["interval"], 2.0)
        assert np.isclose(qm.default_query_parameters["interval"], 2.0)  # Read default

    def test__string_topics_converted_to_list(self, example_config):
        # is ordinarily a list...
        assert isinstance(example_config["kafka_config"]["topics"], list)
        example_config["kafka_config"]["topics"] = "my_new_topic"  # ...now string!
        qm = FinkQueryManager(
            example_config, {}, data_path=paths.test_data_path, create_paths=False
        )

        assert isinstance(qm.kafka_config["topics"], list)  # back to list!
        assert qm.kafka_config["topics"][0] == "my_new_topic"

    def test__error_on_kafka_config_missing_username(self, example_config):
        example_config["kafka_config"].pop("username")
        assert "username" not in example_config["kafka_config"]
        assert "group_id" in example_config["kafka_config"]
        assert "bootstrap.servers" in example_config["kafka_config"]
        assert "topics" in example_config["kafka_config"]
        with pytest.raises(BadKafkaConfigError):
            qm = FinkQueryManager(
                example_config, {}, data_path=paths.test_data_path, create_paths=False
            )

    def test__error_on_kafka_config_missing_group_id(self, example_config):
        example_config["kafka_config"].pop("group_id")
        assert "username" in example_config["kafka_config"]
        assert "group_id" not in example_config["kafka_config"]
        assert "bootstrap.servers" in example_config["kafka_config"]
        assert "topics" in example_config["kafka_config"]
        with pytest.raises(BadKafkaConfigError):
            qm = FinkQueryManager(
                example_config, {}, data_path=paths.test_data_path, create_paths=False
            )

    def test__error_on_kafka_config_missing_servers(self, example_config):
        example_config["kafka_config"].pop("bootstrap")
        assert "username" in example_config["kafka_config"]
        assert "group_id" in example_config["kafka_config"]
        assert "bootstrap.servers" not in example_config["kafka_config"]
        assert "topics" in example_config["kafka_config"]
        with pytest.raises(BadKafkaConfigError):
            qm = FinkQueryManager(
                example_config, {}, data_path=paths.test_data_path, create_paths=False
            )

    def test__error_on_kafka_config_missing_servers(self, example_config):
        example_config["kafka_config"].pop("topics")
        assert "username" in example_config["kafka_config"]
        assert "group_id" in example_config["kafka_config"]
        assert "bootstrap.servers" in example_config["kafka_config"]
        assert "topics" not in example_config["kafka_config"]
        with pytest.raises(BadKafkaConfigError):
            qm = FinkQueryManager(
                example_config, {}, data_path=paths.test_data_path, create_paths=False
            )

    def test__listen_for_alerts(self, fink_qm, fake_alert_list, monkeypatch):
        """
        this test feels EXTREMELY JANKY. TODO fix!!!
        """

        class MockConsumer:
            def __init__(self, topics, config):
                self.fake_alert_list = fake_alert_list
                self.index = 0

            def __enter__(self):
                return self

            def __exit__(self, type, value, traceback):
                pass

            def poll(self, timeout=None):
                if self.index > len(self.fake_alert_list):
                    return (None, None, None)
                result = self.fake_alert_list[self.index]
                self.index = self.index + 1
                return ("interesting_topic", result, "key_")

        # Test MockConsumer behaviour
        with MockConsumer([], []) as mc:
            assert mc.index == 0
            poll_result = mc.poll()
            assert isinstance(poll_result, tuple)
            assert poll_result[1]["candid"] == 23000_10000_20000_1001

        assert hasattr(fink.AlertConsumer, "consume")  # it definitely does
        with monkeypatch.context() as m:
            m.setattr("dk154_targets.query_managers.fink.AlertConsumer", MockConsumer)
            assert not hasattr(fink.AlertConsumer, "consume")  # is "patched"
            alerts = fink_qm.listen_for_alerts()
            assert len(alerts) == 10
            print(alerts[0])
            assert isinstance(alerts[0], tuple)
            assert len(alerts[0]) == 3
            for ii in range(10):
                assert alerts[ii][0] == "interesting_topic"
            assert alerts[0][1]["candid"] == 23000_10000_20000_1001
            assert alerts[1][1]["candid"] == 23000_10000_20000_1002
            assert alerts[5][1]["candid"] == 23000_10000_20000_1006
            assert alerts[9][1]["candid"] == 23000_10000_20000_1010
        assert hasattr(fink.AlertConsumer, "consume")  # back to normal.

    def test__break_on_null_alert(self, example_config, monkeypatch):
        class MockNullConsumer:
            def __init__(self, topics, config):
                pass

            def __enter__(self):
                return self

            def __exit__(self, type, value, traceback):
                pass

            def poll(self, timeout=None):
                return (None, None, None)

        with monkeypatch.context() as m:
            m.setattr(
                "dk154_targets.query_managers.fink.AlertConsumer", MockNullConsumer
            )
            assert not hasattr(fink.AlertConsumer, "consume")  # is "patched"!
            example_config["kafka_config"]["n_alerts"] = 10
            qm = FinkQueryManager(
                example_config, {}, data_path=paths.test_data_path, create_paths=False
            )
            alerts = qm.listen_for_alerts()
            assert len(alerts) == 0
        assert hasattr(fink.AlertConsumer, "consume")  # back to normal.

    def test__process_alerts(self, fink_qm, fake_alert_results):
        processed_alerts = fink_qm.process_alerts(
            fake_alert_results, save_alerts=False, save_cutouts=False
        )
        assert len(processed_alerts) == 15
        for ii, alert in enumerate(processed_alerts):
            assert "objectId" in alert
            assert "ra" in alert
            assert "dec" in alert
            assert "candidate" not in alert  # have correctly removed...
            assert alert["tag"] == "valid"

        print(processed_alerts)
        assert processed_alerts[0]["candid"] == 23000_10000_20000_1001
        assert processed_alerts[1]["candid"] == 23000_10000_20000_1002
        assert processed_alerts[2]["candid"] == 23000_10000_20000_1003
        assert processed_alerts[8]["candid"] == 23000_10000_20000_1009
        assert processed_alerts[10]["candid"] == 23000_10000_20000_1011
        assert processed_alerts[14]["candid"] == 23000_10000_20000_1015

        exp_alert_dir = paths.test_data_path / "fink/alerts/ZTF23abcfake"
        exp_alert1_path = exp_alert_dir / "2300010000200001001.json"
        assert not exp_alert1_path.exists()

    def test__save_alerts(self, example_config, fake_alert_results):
        self._clear_test_directories()
        qm = FinkQueryManager(
            example_config, {}, data_path=paths.test_data_path, create_paths=True
        )
        assert self.exp_fink_path.exists()
        qm.process_alerts(fake_alert_results[:5], save_alerts=True, save_cutouts=False)
        exp_alert_dir = paths.test_data_path / "fink/alerts/ZTF23abcfake"
        assert exp_alert_dir.exists()
        assert exp_alert_dir.is_dir()

        exp_alert1_path = exp_alert_dir / "2300010000200001001.json"
        assert exp_alert1_path.exists()
        exp_alert2_path = exp_alert_dir / "2300010000200001002.json"
        assert exp_alert2_path.exists()
        exp_alert3_path = exp_alert_dir / "2300010000200001003.json"
        assert exp_alert3_path.exists()
        exp_alert4_path = exp_alert_dir / "2300010000200001004.json"
        assert exp_alert4_path.exists()
        exp_alert5_path = exp_alert_dir / "2300010000200001005.json"
        assert exp_alert5_path.exists()

        self._clear_test_directories()
        assert not exp_alert1_path.exists()
        assert not exp_alert_dir.exists()
        assert not self.exp_alerts_path.exists()
        assert not self.exp_fink_path.exists()

    def test__get_lightcurves_to_query(self, fink_qm, fink_lc):
        self._clear_test_directories()
        fink_qm.process_paths(data_path=paths.test_data_path, create_paths=True)
        assert self.exp_fink_path.exists()
        assert self.exp_lightcurves_path.exists()

        target1 = Target("test1", ra=30, dec=30)
        fink_qm.target_lookup["test1"] = target1  # In target lookup
        exp_target1_lc_file = paths.test_data_path / "fink/lightcurves/test1.csv"
        fink_lc.to_csv(exp_target1_lc_file, index=False)  # Exists

        target2 = Target("test2", ra=40, dec=60)
        fink_qm.target_lookup["test2"] = target2  # In target lookup
        exp_target2_lc_file = paths.test_data_path / "fink/lightcurves/test2.csv"
        # File does not exit

        target3 = Target("test3", ra=60, dec=10)
        # Not in target lookup
        exp_target3_lc_file = paths.test_data_path / "fink/lightcurves/test3.csv"
        fink_lc.to_csv(exp_target3_lc_file, index=False)

        target3 = Target("test4", ra=60, dec=10)
        # Not in target lookup
        exp_target4_lc_file = paths.test_data_path / "fink/lightcurves/test4.csv"
        # Does not exist

        t_now = Time.now()
        t_plus1 = t_now + 1 * u.day
        result = fink_qm.get_lightcurves_to_query(t_ref=t_plus1)
        assert set(result) == set(["test2"])  # only pick the missing lc

        t_plus3 = t_now + 3 * u.day
        result = fink_qm.get_lightcurves_to_query(t_ref=t_plus3)
        assert set(result) == set(["test1", "test2"])
        self._clear_test_directories()

    def test__perform_lightcurve_queries(self, fink_qm, det_df, ndet_df, monkeypatch):
        class MockFinkQuery:
            def __init__(self):
                pass

            @classmethod
            def query_objects(cls, withupperlim=False, **kwargs):
                if withupperlim:
                    return ndet_df
                return det_df

        self._clear_test_directories()
        fink_qm.process_paths(data_path=paths.test_data_path, create_paths=True)
        assert self.exp_lightcurves_path.exists()

        assert fink.FinkQuery.api_url == "https://fink-portal.org/api/v1"
        with monkeypatch.context() as m:
            m.setattr("dk154_targets.query_managers.fink.FinkQuery", MockFinkQuery)
            assert not hasattr(fink.FinkQuery, "api_url")
            input_list = ["ZTF9999"]
            exp_lc_file = paths.test_data_path / "fink/lightcurves/ZTF9999.csv"
            assert not exp_lc_file.exists()
            successful = fink_qm.perform_lightcurve_queries(input_list)
            assert set(input_list) == set(successful)  # boring
            assert exp_lc_file.exists()
            lc_result = pd.read_csv(exp_lc_file)
            assert len(lc_result) == 7
            valid = lc_result.query("tag=='valid'")
            assert len(valid) == 3
            exp_candids = np.array(
                [2300010000200005005, 2300010000200005006, 2300010000200005007]
            )
            assert all(valid["candid"].values == exp_candids)
            ulim = lc_result.query("tag=='upperlim'")
            assert all(ulim["candid"] == 0)
            assert len(ulim) == 3
            assert len(lc_result.query("tag=='badquality'")) == 1

        assert fink.FinkQuery.api_url == "https://fink-portal.org/api/v1"
        self._clear_test_directories()

    def test__load_target_lightcurves(self, fink_qm, fink_lc):
        fink_qm.process_paths(data_path=paths.test_data_path, create_paths=True)

        fink_lc.sort_values("jd", ascending=True, inplace=True)
        assert all(fink_lc["jd"].values[:-1] - fink_lc["jd"].values[1:] < 0.0)
        # reverse it to see if it's correctly sorted on load.

        test1 = Target("test1", ra=30.0, dec=30.0)
        test1.fink_data.add_lightcurve(fink_lc)
        assert len(test1.fink_data.detections) == 4  # include badquality.
        fink_qm.target_lookup["test1"] = test1
        exp_test1_path = paths.test_data_path / "fink/lightcurves/test1.csv"
        fink_lc.to_csv(exp_test1_path, index=False)
        # No useful updates for test1 - shouldn't be read.

        test2 = Target("test2", ra=45.0, dec=60.0)
        trunc_lc = fink_lc.copy()[:-1]
        assert len(trunc_lc) == 6
        test2.fink_data.add_lightcurve(trunc_lc)
        assert len(test2.fink_data.detections) == 3  # includes bad qual.
        fink_qm.target_lookup["test2"] = test2
        exp_test2_path = paths.test_data_path / "fink/lightcurves/test2.csv"
        fink_lc.to_csv(exp_test2_path, index=False)
        # test2 will have useful new data.

        # test3 not in target_lookup.
        exp_test3_path = paths.test_data_path / "fink/lightcurves/test3.csv"
        fink_lc.to_csv(exp_test3_path, index=False)

        test4 = Target("test4", ra=90.0, dec=30.0)
        fink_qm.target_lookup["test4"] = test4
        # test4 lc does not exist

        # test5 not in target lookup
        # test5 lc does not exist

        input_list = ["test1", "test2", "test3", "test4", "test5"]
        successful, missing = fink_qm.load_target_lightcurves(input_list)
        assert set(successful) == set(["test2", "test3"])
        assert set(missing) == set(["test4", "test5"])

        self._clear_test_directories()

    def test__load_alerts(self, fink_qm, fink_lc, fake_alert_results):
        self._clear_test_directories()
        fink_qm.process_paths(data_path=paths.test_data_path, create_paths=True)

        test1 = Target("ZTF23abcfake", ra=30.0, dec=45.0)
        test1.fink_data.add_lightcurve(fink_lc)
        fink_qm.target_lookup[test1.objectId] = test1

        last_row = fink_lc.iloc[-1]
        test1_dir = paths.test_data_path / "fink/alerts/ZTF23abcfake"
        test1_dir.mkdir(exist_ok=True, parents=True)
        exp_last_candid_alert_file = test1_dir / f"{last_row.candid}.json"
        with open(exp_last_candid_alert_file, "w+") as f:
            alert_dict = last_row.to_dict()
            json.dump(alert_dict, f)
        assert exp_last_candid_alert_file.exists()

        extra_alerts = [fake_alert_results[ii] for ii in [0, 9, 10]]  # 3 is enough.
        assert set([a[1]["candid"] for a in extra_alerts]) == set(
            [23000_10000_20000_1001, 23000_10000_20000_1010, 23000_10000_20000_1011]
        )
        fink_qm.process_alerts(extra_alerts)  # dump them to the correct places
        exp_alert_0_file = test1_dir / "2300010000200001001.json"
        assert exp_alert_0_file.exists()
        exp_alert_9_file = test1_dir / "2300010000200001010.json"
        assert exp_alert_9_file.exists()
        exp_alert_10_file = test1_dir / "2300010000200001011.json"
        assert exp_alert_10_file.exists()

        loaded_alerts = fink_qm.load_missing_alerts("ZTF23abcfake")
        assert len(loaded_alerts) == 3
        assert set([a["candid"] for a in loaded_alerts]) == set(
            [23000_10000_20000_1001, 23000_10000_20000_1010, 23000_10000_20000_1011]
        )
        self._clear_test_directories()

    def test__integrate_alerts(self, fink_qm, fink_lc, fake_alert_results):
        self._clear_test_directories()
        fink_qm.process_paths(data_path=paths.test_data_path, create_paths=True)
        processed_alerts = fink_qm.process_alerts(fake_alert_results)
        assert len(processed_alerts) == 15

        target = Target("ZTF23abcfake", ra=30.0, dec=45.0)
        target.fink_data.add_lightcurve(fink_lc)
        assert len(target.fink_data.lightcurve) == 7
        fink_qm.target_lookup[target.objectId] = target

        last_row = fink_lc.iloc[-1]
        test1_dir = paths.test_data_path / "fink/alerts/ZTF23abcfake"
        test1_dir.mkdir(exist_ok=True, parents=True)
        exp_last_candid_alert_file = test1_dir / f"{last_row.candid}.json"
        with open(exp_last_candid_alert_file, "w+") as f:
            alert_dict = last_row.to_dict()
            json.dump(alert_dict, f)
        assert exp_last_candid_alert_file.exists()

        exp_target_lc_file = paths.test_data_path / "fink/lightcurves/ZTF23abcfake.csv"
        assert not exp_target_lc_file.exists()

        fink_qm.integrate_alerts()
        assert len(target.fink_data.lightcurve) == (7 + 15)
        assert len(target.fink_data.detections) == 19  # 4 + 15

        # assert target.


class Test__FinkQuery:
    def test__init(self):
        # Doesn't break on init...
        fq = FinkQuery()
        assert fq.api_url == "https://fink-portal.org/api/v1"
