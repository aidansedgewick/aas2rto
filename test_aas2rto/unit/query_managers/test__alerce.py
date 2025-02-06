import os
import pytest

import numpy as np

import pandas as pd

from astropy.time import Time

from aas2rto.target import Target, TargetData
from aas2rto.target_lookup import TargetLookup
from aas2rto.query_managers.alerce import (
    AlerceQueryManager,
    combine_alerce_detections_non_detections,
    target_from_alerce_query_row,
)

from aas2rto.exc import UnexpectedKeysWarning


@pytest.fixture
def alerce_config(queries_config):
    return {"object_queries": queries_config}


@pytest.fixture
def queries_config():
    return {
        "high_prob_cool_sn": {"classifier": "cool_sn", "probability": 0.9},
        "weird_sn": {"classifier": "weird_sn"},
    }


@pytest.fixture
def target_lookup():
    return TargetLookup()


@pytest.fixture
def alerce_qm(alerce_config, target_lookup, tmp_path):
    return AlerceQueryManager(
        alerce_config, target_lookup, parent_path=tmp_path, create_paths=False
    )


@pytest.fixture
def query_results():
    rows = [
        ("cool_sn", "ZTF00abc", 5, 0.92, 60000.0, 15.0, 30.0),  # 0
        ("cool_sn", "ZTF00def", 5, 0.96, 60000.0, 30.0, 30.0),  # 1
        ("cool_sn", "ZTF00hij", 5, 0.94, 60000.0, 45.0, 30.0),  # 2
        ("cool_sn", "ZTF00klm", 5, 0.85, 60000.0, 60.0, 30.0),  # 3
        ("cool_sn", "ZTF00nop", 5, 0.98, 60000.0, 75.0, 30.0),  # 4
        ("weird_sn", "ZTF01abc", 5, 0.97, 60000.0, 90.0, 30.0),  # 5
        ("weird_sn", "ZTF01def", 5, 0.93, 60000.0, 105.0, 30.0),  # 6
        ("weird_sn", "ZTF01ghi", 5, 0.85, 60000.0, 120.0, 30.0),  # 7
    ]
    columns = "classifier oid ndet probability lastmjd meanra meandec".split()
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def alerce_detections():
    candid_base = 23000_10000_20000_0000
    rows = [
        (60010.0, candid_base + 1, 1, 20.0, 18.5, 0.1, 15.0, 30.0, False),
        (60011.0, candid_base + 2, 2, 20.1, 18.4, 0.3, 15.0, 30.0, True),
        (60012.0, candid_base + 3, 2, 20.2, 18.3, 0.1, 15.0, 30.0, False),
        (60013.0, candid_base + 4, 1, 20.3, 18.2, 0.3, 15.0, 30.0, True),
        (60014.0, candid_base + 5, 2, 20.4, 18.1, 0.1, 15.0, 30.0, False),
        (60015.0, candid_base + 6, 1, 20.5, 18.0, 0.1, 15.0, 30.0, False),
    ]
    columns = "mjd candid fid diffmaglim magpsf sigmapsf ra dec dubious".split()
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def alerce_non_detections():
    rows = [
        (60005.0, 1, 18.0),
        (60007.0, 2, 18.0),
        (60009.0, 1, 18.0),
    ]
    return pd.DataFrame(rows, columns="mjd fid diffmaglim".split())


def mock_query_func_from_results(query_results, t_ref=None):
    """
    choose to do this as a closure as simpler th
    """

    def mock_query(**pattern):
        results = query_results.copy()
        classifier = pattern.get("classifier", None)
        if classifier and "classifier" in results.columns:
            results = results[results["classifier"] == classifier]
        prob = pattern.get("probability", None)
        if prob and "probability" in results.columns:
            results = results[results["probability"] > prob]

        n = pattern.get("page_size", None)
        if n is None:
            raise ValueError("pattern must have page_size")
        page = pattern.get("page", 1)

        idx = page - 1
        page_results = results.iloc[idx * n : (idx + 1) * n]

        page_results["page"] = page
        return page_results

    return mock_query


def test__mock_query_function(query_results):
    mock_query = mock_query_func_from_results(query_results)
    t_ref = Time(60000.0, format="mjd")

    res01 = mock_query(**{"page_size": 25, "page": 1})
    assert len(res01) == 8

    res02 = mock_query(**{"page_size": 25, "probability": 0.95, "page": 1})
    assert set(res02["oid"]) == set("ZTF00def ZTF00nop ZTF01abc".split())

    pattern = {"page_size": 3, "classifier": "cool_sn", "probability": 0.9}
    res03 = mock_query(**pattern, page=1)
    assert len(res03) == 3
    assert set(res03["oid"]) == set("ZTF00abc ZTF00def ZTF00hij".split())

    res04 = mock_query(**pattern, page=2)
    assert len(res04) == 1
    assert set(res04["oid"]) == set(["ZTF00nop"])


class Test__CombineAlerceDetNonDet:

    def test__normal_behaviour(
        self, alerce_detections: pd.DataFrame, alerce_non_detections: pd.DataFrame
    ):
        lc = combine_alerce_detections_non_detections(
            alerce_detections, alerce_non_detections
        )

        assert isinstance(lc, pd.DataFrame)
        assert len(lc) == 9

        assert len(lc[lc["tag"] == "valid"]) == 4
        assert len(lc[lc["tag"] == "dubious"]) == 2
        assert len(lc[lc["tag"] == "nondet"]) == 3

        assert all(lc[lc["tag"] == "nondet"]["candid"] == -1)


class Test__TargetFromAlerceQueryRow:
    def test__normal_behaviour(self, query_results):

        row = query_results.iloc[0]

        assert "meanra" in row
        assert "meandec" in row
        assert "oid" in row

        target = target_from_alerce_query_row(row["oid"], row)

        assert isinstance(target, Target)
        assert target.target_id == "ZTF00abc"
        assert np.isclose(target.ra, 15.0)
        assert np.isclose(target.dec, 30.0)
        assert "alerce" in target.alt_ids
        assert "ztf" in target.alt_ids

    def test__accepts_dict(self, query_results):

        data = query_results.iloc[0].to_dict()
        assert isinstance(data, dict)

        target = target_from_alerce_query_row(data["oid"], data)

        assert isinstance(target, Target)
        assert target.target_id == "ZTF00abc"
        assert np.isclose(target.ra, 15.0)
        assert np.isclose(target.dec, 30.0)
        assert "alerce" in target.alt_ids
        assert "ztf" in target.alt_ids

    def test__return_none_no_coords(self, query_results):
        row = query_results.iloc[0]

        data = row.to_dict()
        data.pop("meanra")
        result_1 = target_from_alerce_query_row("no_ra", data)
        assert result_1 is None

        data = row.to_dict()
        data.pop("meandec")
        result_2 = target_from_alerce_query_row("no_dec", data)
        assert result_2 is None

    def test__raise_error_if_dataframe(self, query_results):

        data = query_results[:2]
        assert len(data) == 2

        with pytest.raises(ValueError):
            result = target_from_alerce_query_row("should_fail", data)


class Test__AlerceQMIinit:

    def test__alerce_qm_init(self, alerce_config, target_lookup, tmp_path):
        qm = AlerceQueryManager(alerce_config, target_lookup, parent_path=tmp_path)

        assert isinstance(qm.alerce_config, dict)

        assert isinstance(qm.object_queries, dict)
        assert isinstance(qm.query_parameters, dict)
        assert isinstance(qm.query_results, dict)

        assert isinstance(qm.target_lookup, TargetLookup)

        assert qm.parent_path == tmp_path
        assert qm.data_path == tmp_path / "alerce"

        assert qm.data_path.exists()

        assert qm.lightcurves_path == tmp_path / "alerce/lightcurves"
        assert qm.lightcurves_path.exists()
        assert qm.cutouts_path == tmp_path / "alerce/cutouts"
        assert qm.cutouts_path.exists()

    def test__create_paths_false(self, alerce_config, target_lookup, tmp_path):

        qm = AlerceQueryManager(
            alerce_config, target_lookup, parent_path=tmp_path, create_paths=False
        )

        assert qm.data_path == tmp_path / "alerce"
        assert not qm.data_path.exists()

    def test__alerce_qm_bad_config_keys(self, alerce_config, target_lookup):

        alerce_config["blah"] = 10

        with pytest.warns(UnexpectedKeysWarning):
            qm = AlerceQueryManager(alerce_config, target_lookup, create_paths=False)


class Test__QueryAndCollate:

    def test__prepare_query_data(self, alerce_qm):

        t_ref = Time(60000.0, format="mjd")

        pattern = alerce_qm.object_queries["high_prob_cool_sn"]
        assert set(pattern.keys()) == set("classifier probability".split())

        result = alerce_qm.prepare_query_data(pattern, page=0, t_ref=t_ref)

        assert isinstance(result, dict)

        assert set(result.keys()) == set(
            "classifier probability page page_size firstmjd".split()
        )

    def test__query_and_collate(
        self, alerce_qm: AlerceQueryManager, query_results, monkeypatch
    ):

        alerce_qm.query_parameters["n_objects"] = 3

        t_ref = Time(60000.0, format="mjd")

        # get a function which can query
        mock_query = mock_query_func_from_results(query_results)

        query_pattern = alerce_qm.object_queries["high_prob_cool_sn"]

        assert alerce_qm.alerce_broker.query_objects.__name__ == "query_objects"

        with monkeypatch.context() as m:
            # patch the mock_query function onto the existing alerce_qm.alerce_broker instance
            m.setattr(alerce_qm.alerce_broker, "query_objects", mock_query)
            assert alerce_qm.alerce_broker.query_objects.__name__ == "mock_query"  # ok!

            results = alerce_qm.query_and_collate_pages(query_pattern)

        assert len(results) == 4
        assert isinstance(results, pd.DataFrame)
        exp_columns = (
            "classifier oid probability ndet lastmjd page meanra meandec".split()
        )
        assert set(results.columns) == set(exp_columns)

        assert set(results["classifier"].values) == set(["cool_sn"])
        exp_all_ids = "ZTF00abc ZTF00def ZTF00hij ZTF00nop".split()
        assert set(results["oid"]) == set(exp_all_ids)

        assert len(results[results["page"] == 1]) == 3
        exp_page1_ids = "ZTF00abc ZTF00def ZTF00hij".split()
        assert set(results[results["page"] == 1]["oid"]) == set(exp_page1_ids)

        assert len(results[results["page"] == 2]) == 1
        assert set(results[results["page"] == 2]["oid"]) == set("ZTF00nop".split())


class Test__QueryForObjectUpdates:
    def test__no_existing_results(
        self, alerce_qm: AlerceQueryManager, query_results: pd.DataFrame, monkeypatch
    ):
        alerce_qm.create_paths()

        mock_query = mock_query_func_from_results(query_results)

        exp_cool_sn_path = alerce_qm.query_results_path / "high_prob_cool_sn.csv"
        exp_weird_sn_path = alerce_qm.query_results_path / "weird_sn.csv"

        assert not exp_cool_sn_path.exists()
        assert not exp_weird_sn_path.exists()

        with monkeypatch.context() as m:
            # patch the mock_query function onto the existing alerce_qm.alerce_broker instance
            m.setattr(alerce_qm.alerce_broker, "query_objects", mock_query)
            assert alerce_qm.alerce_broker.query_objects.__name__ == "mock_query"  # ok!

            results = alerce_qm.query_for_object_updates()

        # query results have been written
        assert exp_cool_sn_path.exists()
        assert exp_weird_sn_path.exists()

        assert len(results) == 7  # 4 high_prob_cool_sn + all 3 weird_sn

        cool_sn_df = pd.read_csv(exp_cool_sn_path)
        assert len(cool_sn_df) == 4
        weird_sn_df = pd.read_csv(exp_weird_sn_path)
        assert len(weird_sn_df) == 3

        assert set(alerce_qm.query_results)

    def test__load_existing_queries(
        self, alerce_qm: AlerceQueryManager, query_results: pd.DataFrame
    ):

        alerce_qm.create_paths()

        mock_query = mock_query_func_from_results(query_results)

        for query_name, query_pattern in alerce_qm.object_queries.items():
            results = mock_query(**query_pattern, page_size=10)
            query_results_path = alerce_qm.get_query_results_file(query_name)
            results.to_csv(query_results_path, index=False)

        assert set(alerce_qm.query_results.keys()) == set()

        t_now = Time.now()
        to_requery = alerce_qm.load_existing_query_results(t_ref=t_now)
        assert set(to_requery) == set()

        # what about when the queries are old?

        t_future = Time(t_now.jd + 3.0, format="jd")
        to_requery = alerce_qm.load_existing_query_results(t_ref=t_future)
        assert set(to_requery) == set(["high_prob_cool_sn", "weird_sn"])

    def test__merge_existing_queries(
        self, alerce_qm: AlerceQueryManager, query_results: pd.DataFrame, monkeypatch
    ):
        """
        Test that new rows from query are correcly added to set of existing queries.
        """

        alerce_qm.create_paths()

        new_results = query_results.copy()
        new_results["ndet"].iloc[0] = 6
        new_results["ndet"].iloc[1] = 6
        new_results["ndet"].iloc[5] = 6
        new_results["lastmjd"].iloc[0] = 60001.0
        new_results["lastmjd"].iloc[1] = 60001.0
        new_results["lastmjd"].iloc[5] = 60001.0

        ndet_is_6 = new_results[new_results["ndet"] == 6]
        assert len(ndet_is_6) == 3

        # Write the old queries to load in, to test what is updated...
        old_results = query_results.iloc[:-1]  # cut off the last row
        mock_query_old_results = mock_query_func_from_results(old_results)

        for query_name, query_pattern in alerce_qm.object_queries.items():
            results = mock_query_old_results(**query_pattern, page_size=10)
            alerce_qm.query_results[query_name] = results

        mock_query = mock_query_func_from_results(new_results)

        with monkeypatch.context() as m:
            m.setattr(alerce_qm.alerce_broker, "query_objects", mock_query)
            results = alerce_qm.query_for_object_updates()

        assert len(results) == 4  # Return updated ones.
        assert set(results["oid"]) == set(
            ["ZTF00abc", "ZTF00def", "ZTF01abc", "ZTF01ghi"]
        )


class Test__QueryLightcurves:

    def test__normal_behaviour(
        self,
        alerce_qm: AlerceQueryManager,
        alerce_detections: pd.DataFrame,
        alerce_non_detections: pd.DataFrame,
        monkeypatch,
    ):

        alerce_qm.create_paths()

        def mock_lc_query(object_ids):
            det_dict = alerce_detections.to_dict("records")
            non_det_dict = alerce_non_detections.to_dict("records")
            return {"detections": det_dict, "non_detections": non_det_dict}

        exp_lc_path = alerce_qm.lightcurves_path / "ZTF00abc.csv"
        assert not exp_lc_path.exists()

        with monkeypatch.context() as m:
            # patch the alerce_broker with the mock function!
            m.setattr(alerce_qm.alerce_broker, "query_lightcurve", mock_lc_query)
            assert alerce_qm.alerce_broker.query_lightcurve.__name__ == "mock_lc_query"

            success, failed = alerce_qm.perform_lightcurve_queries(["ZTF00abc"])

        assert set(success) == set(["ZTF00abc"])

        assert alerce_qm.alerce_broker.query_lightcurve.__name__ == "query_lightcurve"

        assert exp_lc_path.exists()
        result = pd.read_csv(exp_lc_path)

        assert len(result) == 9
