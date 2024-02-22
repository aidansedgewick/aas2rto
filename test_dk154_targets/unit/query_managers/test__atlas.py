import os

import pytest

import numpy as np

import pandas as pd

from astropy.time import Time

from dk154_targets import Target

from dk154_targets.query_managers.atlas import (
    process_atlas_lightcurve,
    get_empty_atlas_lightcurve,
    AtlasQueryManager,
    AtlasQuery,
)

@pytest.fixture
def atlas_lc_rows():
    return [
        (60000.0, 20.0, 0.1, 19.0, "c"),  # below 5 sig
        (60000.5, 19.9, 0.1, 21.0, "o"),
        (60001.0, -19.8, 0.1, 21.0, "c"),  # negative mag
        (60001.5, -19.7, 0.1, 21.0, "o"),  # negative mag
        (60002.0, 19.6, 0.1, 21.0, "o"),
        (60002.5, 19.5, 0.1, 21.0, "c"),
        (60003.0, 19.4, 0.1, 21.0, "c"),
        (60003.5, 19.3, 0.1, 19.0, "o"),  # below 5 sig
        (60004.0, 19.2, 0.1, 21.0, "o"),
        (60004.5, 19.1, 0.1, 21.0, "o"),  #
    ]
    
@pytest.fixture
def atlas_lc(atlas_lc_rows):
    return pd.DataFrame(
        atlas_lc_rows, columns="MJD m dm mag5sig F".split()
    )

@pytest.fixture
def atlas_config():
    return {
        "query_parameters": {
            "requests_timeout": 60.,
        },
        "token": "token_goes_here",
        "project_identifier": "test_id",
    }

@pytest.fixture
def target_lookup():
    return {}

@pytest.fixture
def atlas_qm(atlas_config, target_lookup, tmp_path):
    return AtlasQueryManager(
        atlas_config, target_lookup, parent_path=tmp_path, create_paths=False
    )

@pytest.fixture
def mock_query_results_00():
    return [
        dict(comment="T000:test_id", url="url_for_T000"),
        dict(comment="T001:test_id", url="url_for_T001"),
        dict(comment="T002:another_id", url="url_for_T002"), # will ignore
        dict(comment="T003:test_id", url="url_for_T003"),
        dict(comment="T004:test_id", url="url_for_T004"),
    ]

@pytest.fixture
def mock_query_response_00(mock_query_results_00):
    return dict(
        results=mock_query_results_00, next="goto_task_01",
    )

@pytest.fixture
def mock_query_results_01():
    return [
        dict(comment="T005:another_id", url="url_for_T005"), # will ignore
        dict(comment="T006:another_id", url="url_for_T006"), # will ignore
        dict(comment="T007:test_id", url="url_for_T007"),
        dict(comment="T008:test_id", url="url_for_T008"),
        dict(comment="T009", url="url_for_T009"),
        dict(url="url_for_T010"),
    ]

@pytest.fixture
def mock_query_response_01(mock_query_results_01):
    return dict(
        results=mock_query_results_01, next=None,
    )

@pytest.fixture
def target_list():
    return [
        Target(f"T101", ra=10.0, dec=30.),
        Target(f"T102", ra=20.0, dec=30.),
        Target(f"T103", ra=30.0, dec=30.),
        Target(f"T104", ra=40.0, dec=30.),
        Target(f"T105", ra=50.0, dec=30.),
        Target(f"T106", ra=60.0, dec=30.),
        Target(f"T107", ra=70.0, dec=30.),
        Target(f"T108", ra=80.0, dec=30.),
        Target(f"T109", ra=90.0, dec=30.),
        Target(f"T110", ra=100.0, dec=30.),
        Target(f"T111", ra=110.0, dec=30.),
    ]

@pytest.fixture
def fake_query_data():
    return dict(
        ra=30., dec=45., mjd_min=60000., mjd_max=60030., send_email=False, comment="TestComment"
    )

@pytest.fixture
def mock_finished_response(atlas_lc):
    return 

class MockAtlasResponse:
    def __init__(self, data):
        self.data = data
        self.reason = "some_reason"
        self.comment = self.data["comment"]

    def json(self):
        return dict(url=f"url_for_{self.comment}")

class MockAtlasSubmitted(MockAtlasResponse):
    status_code = 201

class MockAtlasThrottled(MockAtlasResponse):
    status_code = 429

def status_from_objectId(objectId):
    ii = int(objectId[-1])
    if ii % 2 == 0:
        return 200 # Finished query
    return 201 # Ongoing query

def test__status_from_objectId():
    assert status_from_objectId("T000") == 200
    assert status_from_objectId("T001") == 201
    assert status_from_objectId("T002") == 200
    assert status_from_objectId("T003") == 201


# For testing actually getting a lightcurve response.

@pytest.fixture
def atlas_lc_serialized(atlas_lc):
    lines = ["###" + "\t".join([col for col in atlas_lc.columns])]
    for ii, row in atlas_lc.iterrows():
        lines.append("\t".join([str(row[col]) for col in atlas_lc.columns]))
    return "\n".join(lines)

@pytest.fixture
def mock_response_data_finished():
    return {
        "finishtimestamp": 63000.0,
        "result_url": "url_of_lightcurve"
    }
    
@pytest.fixture
def mock_response_data_submitted():
    return {}
    
@pytest.fixture
def mock_response_data_no_data():
    return {
        "finishtimestamp": 60300.0,
        "error_msg": "No data returned"
    } 

class MockPhotometryResponse:
    def __init__(self, text):
        self.text = text
    
class MockRequestsResponse:
    def __init__(self, data):
        self.data = data
        
    def json(self):
        return self.data



### ========================= Testing starts here ========================== ###


class Test__ProcessAtlasLightcurve:

    def test__normal_behaviour(self, atlas_lc):
        processed = process_atlas_lightcurve(atlas_lc)
        assert set(processed.columns) == set(["mjd", "jd", "m", "dm", "mag5sig", "F"])
        
        
class Test__AtlasQueryManagerInit:

    def test__atlas_qm_init(self, atlas_config, target_lookup, tmp_path):
        atlas_qm = AtlasQueryManager(
            atlas_config, target_lookup, parent_path=tmp_path, create_paths=True
        )

        assert np.isclose(atlas_qm.query_parameters["lightcurve_query_lookback"], 30.0)
        assert np.isclose(atlas_qm.query_parameters["requests_timeout"], 60.0)

        assert isinstance(atlas_qm.target_lookup, dict)

        assert isinstance(atlas_qm.atlas_headers, dict)
        assert set(atlas_qm.atlas_headers.keys()) == set(["Authorization", "Accept"])
        assert atlas_qm.atlas_headers["Authorization"] == "Token token_goes_here"
        assert atlas_qm.atlas_headers["Accept"] == "application/json"

        assert isinstance(atlas_qm.submitted_queries, dict)
        assert isinstance(atlas_qm.throttled_queries, list)

        assert atlas_qm.data_path == tmp_path / "atlas"
        assert atlas_qm.data_path.exists()
        assert atlas_qm.lightcurves_path == tmp_path / "atlas/lightcurves"

    def test__fails_with_no_token(self, atlas_config, target_lookup, tmp_path):
        token = atlas_config.pop("token")

        assert "token" not in atlas_config

        with pytest.raises(ValueError):
            atlas_qm = AtlasQueryManager(
                atlas_config, target_lookup, parent_path=tmp_path
            )

    def test__fails_with_bad_config_param(self, atlas_config, target_lookup, tmp_path):
        atlas_config["bad_kw"] = "my_name"
        with pytest.raises(ValueError):
            qm = AtlasQueryManager(
                atlas_config, target_lookup, parent_path=tmp_path, create_paths=False
            )

    def test__fails_with_bad_query_param(self, atlas_config, target_lookup, tmp_path):
        atlas_config["query_parameters"]["blah"] = 100
        with pytest.raises(ValueError):
            qm = AtlasQueryManager(
                atlas_config, target_lookup, parent_path=tmp_path, create_paths=False
            )



class Test__AtlasRecoverExistingQueries:

    def test__atlas_query_comment(self, atlas_qm):
        assert atlas_qm.project_identifier == "test_id"
        comm1 = atlas_qm.get_atlas_query_comment("T000")
        assert comm1 == "T000:test_id"

    def test__recover_finished_queries(
        self, atlas_qm, mock_query_response_00, mock_query_response_01, monkeypatch
    ):

        def mock_get_existing_queries(url, **kwargs):
            if url == AtlasQuery.atlas_default_queue_url:
                return mock_query_response_00
            if url == "goto_task_01":
                return mock_query_response_01
            raise ValueError("Shouldn't have made it here")

        def mock_recover_query_data(self, objectId, *args, **kwargs):
            # Test which returns 200 (finished) if oId is even, 201 (submitted) if oId is odd.
            return status_from_objectId(objectId)

        # There are no existing queries. It's as if we've just started AAS2RTO.
        assert len(atlas_qm.submitted_queries) == 0
        assert len(atlas_qm.throttled_queries) == 0

        with monkeypatch.context() as m:
            m.setattr(
                "dk154_targets.query_managers.atlas.AtlasQuery.get_existing_queries",
                mock_get_existing_queries
            )
            m.setattr(
                "dk154_targets.query_managers.atlas.AtlasQueryManager.recover_query_data",
                 mock_recover_query_data
            ) # Not ready to test downloading a lightcurve at this point.
            finished, ongoing = atlas_qm.recover_finished_queries(delete_finished_queries=False)

        assert set(finished) == set(["T000", "T004", "T008"]) # 
        assert set(ongoing) == set(["T001", "T003", "T007"])
        # We should ignore "T002", "T005", "T006" as they belong to a different project.

        assert set(atlas_qm.submitted_queries.keys()) == set(["T001", "T003", "T007"])
        assert atlas_qm.submitted_queries["T001"] == "url_for_T001"
        assert atlas_qm.submitted_queries["T003"] == "url_for_T003"
        assert atlas_qm.submitted_queries["T007"] == "url_for_T007"
        
        assert "T000" not in atlas_qm.submitted_queries # It's finished.
        assert "T000" not in atlas_qm.submitted_queries # It's finished.
        
        # T005 would be in submitted if it belonged to our project, "test_id"...
        assert "T005" not in atlas_qm.submitted_queries # ...but it belongs to "another_id".
        
    def test__no_crash_if_query_fails(self, atlas_qm, mock_query_response_00, monkeypatch):
        
        def mock_get_existing_queries(*args, **kwargs):
            raise Exception("This should be caught!")
            
        with monkeypatch.context() as m:
            m.setattr(
                "dk154_targets.query_managers.atlas.AtlasQuery.get_existing_queries", 
                mock_get_existing_queries
            )
            finished, ongoing = atlas_qm.recover_finished_queries()
            
        assert len(finished) == 0
        assert len(ongoing) == 0
            

class Test__MockAtlasResponse:

    def test__submitted_init_correctly(self, fake_query_data):
        res = MockAtlasSubmitted(fake_query_data)
        assert issubclass(MockAtlasSubmitted, MockAtlasResponse)
        assert res.status_code == 201
        assert res.comment == "TestComment"
        assert res.reason == "some_reason"
        assert hasattr(res, "json")
        
    def test__throttled_init_correctly(self, fake_query_data):
        res = MockAtlasThrottled(fake_query_data)
        assert issubclass(MockAtlasResponse, MockAtlasResponse)
        assert res.status_code == 429
        assert res.comment == "TestComment"
        assert res.reason == "some_reason"
        assert hasattr(res, "json")
            
    def test__json_call(self, fake_query_data):
        res = MockAtlasResponse(fake_query_data)
        res_json = res.json()
        assert res.json()["url"] == "url_for_TestComment"
        
    def test__submitted_json_call(self, fake_query_data):       
        res = MockAtlasSubmitted(fake_query_data)
        res_json = res.json()
        assert res_json["url"] == "url_for_TestComment" # pre-called
        assert res.json()["url"] == "url_for_TestComment" # after call
        
    def test__throttled_json_call(self, fake_query_data):
        res = MockAtlasThrottled(fake_query_data)
        res_json = res.json()
        assert res_json["url"] == "url_for_TestComment" # pre-called
        assert res.json()["url"] == "url_for_TestComment" # after call

class Test__SubmitAtlasQuery:

    def test__atlas_prepare_query_data(self, atlas_qm, target_list):
        t_submit = Time(60040., format="mjd")

        query_data = atlas_qm.prepare_query_data(target_list[0], t_ref=t_submit)
        assert isinstance(query_data, dict)
        assert np.isclose(query_data["ra"], 10.)
        assert np.isclose(query_data["dec"], 30.)
        assert np.isclose(query_data["mjd_min"], 60010.0)
        assert np.isclose(query_data["mjd_max"], 60039.997) # has -1e-3
        assert query_data["send_email"] is False
        assert query_data["comment"] == "T101:test_id"

    def test__atlas_submit_query_successful(self, atlas_qm, target_list, monkeypatch):

        def mock_atlas_query_submitted(data, **kwargs):
            # response has url which is "url_for_ + the target comment.
            return MockAtlasSubmitted(data)
            
        with monkeypatch.context() as m:
            # AtlasQM.submit_query() calls AtlasQuery.atlas_query - an API call...
            # patch so it doesn't make this call to the outside world.
            m.setattr(
                "dk154_targets.query_managers.atlas.AtlasQuery.atlas_query", 
                mock_atlas_query_submitted
            )
            query_status = atlas_qm.submit_query(target_list[0])
            
        assert query_status == 201
        assert set(atlas_qm.submitted_queries.keys()) == set(["T101"])
        assert atlas_qm.submitted_queries["T101"] == "url_for_T101:test_id"
        assert len(atlas_qm.throttled_queries) == 0
                
    def test__atlas_submit_query_throttled(self, atlas_qm, target_list, monkeypatch):
        
        def mock_atlas_query_throttled(data, **kwargs):
            return MockAtlasThrottled(data)
            
        with monkeypatch.context() as m:
            # AtlasQM.submit_query() calls AtlasQuery.atlas_query - an API call...
            # patch so it doesn't make this call to the outside world.
            m.setattr(
                "dk154_targets.query_managers.atlas.AtlasQuery.atlas_query", 
                mock_atlas_query_throttled
            )
            query_status = atlas_qm.submit_query(target_list[0])
        assert query_status == 429
        assert set(atlas_qm.throttled_queries) == set(["T101"])
        assert len(atlas_qm.submitted_queries) == 0
        
    def test__no_throttling_submit_new_queries(self, atlas_qm, target_list, monkeypatch):
        for target in target_list:
            atlas_qm.add_target(target)            
        assert len(atlas_qm.target_lookup) == 11
        assert atlas_qm.query_parameters["max_submitted"] == 25 # We'll not stop ourselves.
        
        def mock_atlas_query(data, **kwargs):
            # Fake call to Atlas always returns OK.
            return MockAtlasSubmitted(data)
            
        with monkeypatch.context() as m:        
            # AQM.submit_new_queries() calls AQM.submit_query() calls API. See above.    
            m.setattr(
                "dk154_targets.query_managers.atlas.AtlasQuery.atlas_query", 
                mock_atlas_query
            )
            
            objectId_list = [t.objectId for t in target_list]
            submitted, throttled = atlas_qm.submit_new_queries(objectId_list)
            
        assert len(submitted) == 11
        exp_submitted = [
            "T101", "T102", "T103", "T104", "T105", "T106", "T107", "T108", "T109", "T110", "T111"
        ]        
        assert set(submitted) == set(exp_submitted)
        
        assert len(throttled) == 0
        assert atlas_qm.local_throttled is False
        assert atlas_qm.server_throttled is False
        
            

    def test__local_throttling_submit_new_queries(self, atlas_qm, target_list, monkeypatch):
        
        for target in target_list:
            atlas_qm.add_target(target)            
            
        atlas_qm.query_parameters["max_submitted"] = 5
        assert len(atlas_qm.target_lookup) == 11 # We'll definitely stop ourselves here.   
        
        def mock_atlas_query(data, **kwargs):
            return MockAtlasSubmitted(data)
        
        with monkeypatch.context() as m:
            # AQM.submit_new_queries() calls AQM.submit_query() calls API. See above.    
            m.setattr(
                "dk154_targets.query_managers.atlas.AtlasQuery.atlas_query", 
                mock_atlas_query
            )       
            
            objectId_list = [t.objectId for t in target_list]
            submitted, throttled = atlas_qm.submit_new_queries(objectId_list)
        
        assert atlas_qm.local_throttled is True # We stopped ourselves.
        assert atlas_qm.server_throttled is False
        
        assert len(submitted) == 5
        assert len(throttled) == 6
        exp_submitted = ["T101", "T102", "T103", "T104", "T105"]
        assert set(submitted) == set(exp_submitted)
        exp_throttled = ["T106", "T107", "T108", "T109", "T110", "T111"]
        assert set(throttled) == set(exp_throttled)
        
        assert isinstance(atlas_qm.submitted_queries, dict)
        assert set(atlas_qm.submitted_queries.keys()) == set(exp_submitted)
        assert atlas_qm.submitted_queries["T101"] == "url_for_T101:test_id"
        assert atlas_qm.submitted_queries["T102"] == "url_for_T102:test_id"
        assert atlas_qm.submitted_queries["T103"] == "url_for_T103:test_id"
        assert atlas_qm.submitted_queries["T104"] == "url_for_T104:test_id"
        assert atlas_qm.submitted_queries["T105"] == "url_for_T105:test_id"
        
        assert isinstance(atlas_qm.throttled_queries, list)
        assert set(atlas_qm.throttled_queries) == set(exp_throttled)
        
        
    def test__server_throttling_new_queries(self, atlas_qm, target_list, monkeypatch):
        
        for target in target_list:
            atlas_qm.add_target(target)
        assert len(atlas_qm.target_lookup) == 11 # We'll not stop ourselves here...
        assert atlas_qm.query_parameters["max_submitted"] == 25
            
        def mock_atlas_query(data, **kwargs):
            if data["ra"] < 45.0:
                # Targets T101, T102, T103, T104
                return MockAtlasSubmitted(data)
            return MockAtlasThrottled(data) # Throttled for T105 and up.
            
        with monkeypatch.context() as m:
            # AQM.submit_new_queries() calls AQM.submit_query() calls API. See above.    
            m.setattr(
                "dk154_targets.query_managers.atlas.AtlasQuery.atlas_query", 
                mock_atlas_query
            )        
            
            objectId_list = [t.objectId for t in target_list]
            submitted, throttled = atlas_qm.submit_new_queries(objectId_list)
            
        assert atlas_qm.local_throttled is False
        assert atlas_qm.server_throttled is True # Atlas stopped us! D:

        assert len(submitted) == 4
        assert set(atlas_qm.submitted_queries.keys()) == set(["T101", "T102", "T103", "T104"])
        
        assert len(throttled) == 7
        exp_throttled = ["T105", "T106", "T107", "T108", "T109", "T110", "T111"]
        assert set(atlas_qm.throttled_queries) == set(exp_throttled)

    def test__throttled_flags_are_reset(self, atlas_qm):
        atlas_qm.local_throttled = True
        atlas_qm.server_throttled = True
        
        atlas_qm.submit_new_queries([])
        assert atlas_qm.local_throttled == False
        assert atlas_qm.local_throttled == False

    def test__retry_throttled_queries(self, atlas_qm, target_list, monkeypatch):
        for target in target_list:
            atlas_qm.add_target(target)

        atlas_qm.query_parameters["max_submitted"] = 2

        def mock_atlas_query(data, **kwargs):
            print("we are mocking a submit", data["comment"])
            return MockAtlasSubmitted(data)
            
        ## Set it up so there are 4 objects to retry. We expect 2 will make it.
        atlas_qm.throttled_queries.extend(["T101", "T102", "T103", "T104"])
        assert len(atlas_qm.throttled_queries) == 4
        for objectId in atlas_qm.throttled_queries:
            assert objectId in atlas_qm.target_lookup
        assert len(atlas_qm.submitted_queries) == 0                                
            
        with monkeypatch.context() as m:
            m.setattr(
                "dk154_targets.query_managers.atlas.AtlasQuery.atlas_query", 
                mock_atlas_query
            )     

            atlas_qm.retry_throttled_queries()
               
        assert len(atlas_qm.submitted_queries) == 2
        assert set(atlas_qm.submitted_queries.keys()) == set(["T101", "T102"])
        assert atlas_qm.submitted_queries["T101"] == "url_for_T101:test_id"
        assert atlas_qm.submitted_queries["T102"] == "url_for_T102:test_id"
        
        atlas_qm.local_throttled is True
        atlas_qm.server_throttled is False
        assert len(atlas_qm.throttled_queries) == 2
        assert set(atlas_qm.throttled_queries) == set(["T103", "T104"])
        

class Test__RecoverQueryData:
        
    def test__serialized_lc_unserializes_correcly(self, atlas_lc, atlas_lc_serialized):
        
        mock_response = MockPhotometryResponse(atlas_lc_serialized)
        processed = AtlasQuery.process_response(mock_response)
        assert isinstance(processed, pd.DataFrame)
        assert set(processed.columns) == set(["MJD", "m", "dm", "mag5sig", "F"])
        assert len(processed) == 10
        
        assert pd.api.types.is_float_dtype(processed["MJD"])
        assert pd.api.types.is_float_dtype(processed["m"])
        assert pd.api.types.is_float_dtype(processed["dm"])
        assert pd.api.types.is_float_dtype(processed["mag5sig"])
        assert pd.api.types.is_string_dtype(processed["F"])
        
        assert np.isclose(processed.iloc[0].MJD, 60000.0)
        assert processed.iloc[5].F == "c"
        
        exp_F_vals = "c o c o o c c o o o".split()
        assert np.all(processed["F"].values == np.array(exp_F_vals))
        
    def test__recover_finished(
        self, atlas_qm, mock_response_data_finished, atlas_lc_serialized, monkeypatch
    ):
        atlas_qm.create_paths()
        
        def mock_get(self, task_url, **kwargs):
            if task_url == "url_of_lightcurve":
                return MockPhotometryResponse(atlas_lc_serialized)
            return MockRequestsResponse(mock_response_data_finished)
            
        exp_lightcurve_file = atlas_qm.data_path / "lightcurves/T1000.csv"
        assert not exp_lightcurve_file.exists()
            
        with monkeypatch.context() as m:
            m.setattr("requests.Session.get", mock_get)
            
            status_code = atlas_qm.recover_query_data("T1000", "any_url")
            
        assert status_code == 200
        assert exp_lightcurve_file.exists()        
        recovered_lc = pd.read_csv(exp_lightcurve_file)
        assert len(recovered_lc) == 10
        assert set(recovered_lc.columns) == set(["mjd", "jd", "m", "dm", "mag5sig", "F"])
        # It has been correctly "processed", so "MJD" -> "mjd". also added "jd"
        

    def test__recover_submitted(
        self, atlas_qm, mock_response_data_submitted, monkeypatch
    ):
        atlas_qm.create_paths()
        
        def mock_get(self, task_url, **kwargs):
            return MockRequestsResponse(mock_response_data_submitted)
            
        with monkeypatch.context() as m:
            m.setattr("requests.Session.get", mock_get)
            
            status_code = atlas_qm.recover_query_data("T1001", "any_url")
        
        assert status_code == 201
        
    def test__recover_no_data(
        self, atlas_qm, mock_response_data_no_data, monkeypatch
    ):
        atlas_qm.create_paths()
        
        def mock_get(self, task_url, **kwargs):
            return MockRequestsResponse(mock_response_data_no_data)
            
        exp_lightcurve_file = atlas_qm.data_path / "lightcurves/T1002.csv"
        assert not exp_lightcurve_file.exists()
            
        with monkeypatch.context() as m:
            m.setattr("requests.Session.get", mock_get)
            
            status_code = atlas_qm.recover_query_data("T1002", "any_url")
                        
        assert status_code == 200
        
        assert exp_lightcurve_file.exists()
        recovered_lc = pd.read_csv(exp_lightcurve_file)
        assert recovered_lc.empty
        exp_cols = "mjd jd m dm uJy duJy F err chi/N RA Dec x y maj min phi apfit mag5sig Sky Obs".split()
        assert set(recovered_lc.columns) == set(exp_cols)


    def test__no_data_does_not_overwrite(
        self, atlas_qm, atlas_lc, mock_response_data_no_data, monkeypatch
    ):
        atlas_qm.create_paths()
        
        def mock_get(self, task_url, **kwargs):
            return MockRequestsResponse(mock_response_data_no_data)
        
        exp_lightcurve_file = atlas_qm.data_path / "lightcurves/T1003.csv"
        atlas_lc.to_csv(exp_lightcurve_file, index=False)
        
        with monkeypatch.context() as m:
            m.setattr("requests.Session.get", mock_get)
            
            status_code = atlas_qm.recover_query_data("T1003", "any_url")

        assert status_code == 200
        assert exp_lightcurve_file.exists()        
        recovered_lc = pd.read_csv(exp_lightcurve_file)
        assert len(recovered_lc) == 10
        assert set(recovered_lc.columns) == set(["MJD", "m", "dm", "mag5sig", "F"])

    def test__select_query_candidates(self, atlas_qm, target_list, atlas_lc):
        atlas_qm.create_paths()
        
        t_now = Time.now()
        for ii, target in enumerate(target_list[:8]):
            assert target.get_last_score() is None
            if ii <=3:
                # T101, T102, T103, T104
                target.update_score_history(ii+1, None, t_ref=t_now)
            if 3 < ii <= 5:
                # T105, T106
                target.update_score_history(-1, None, t_ref=t_now)
            # ignore T107, T108
            atlas_qm.add_target(target)
        assert atlas_qm.target_lookup["T104"].get_last_score() == 4
        assert atlas_qm.target_lookup["T105"].get_last_score() == -1
        assert atlas_qm.target_lookup["T107"].get_last_score() is None
        
            
        # If no targets have recent lightcurve...
        # expect "T104", "T103", "T102", "T101", "T104", "T105" -- the last two should be missing.

        query_candidates = atlas_qm.select_query_candidates(t_ref=t_now)
        assert len(query_candidates) == 6
        assert query_candidates.values[0] == "T104"
        assert query_candidates.values[1] == "T103"
        assert query_candidates.values[2] == "T102"
        assert query_candidates.values[3] == "T101"
        assert query_candidates.values[4] in ["T105", "T106"] # Ordering of duplicates is random?!
        assert query_candidates.values[5] in ["T105", "T106"] # TODO fix this!!
        
        # If some targets have recent lightcurves, ignore them!
        atlas_lc.to_csv(atlas_qm.get_lightcurve_file("T103"), index=False)
        atlas_lc.to_csv(atlas_qm.get_lightcurve_file("T105"), index=False)
        
        t_now = Time.now()
        query_candidates = atlas_qm.select_query_candidates(t_ref=t_now)
        assert len(query_candidates) == 4
        assert query_candidates.values[0] == "T104"
        assert query_candidates.values[1] == "T102"
        assert query_candidates.values[2] == "T101"
        assert query_candidates.values[3] == "T106"
        
        # But back to normal for old lightcurves.
        t_future = Time(t_now.jd + 3.0, format="jd")
        query_candidates = atlas_qm.select_query_candidates(t_ref=t_future)
        assert len(query_candidates) == 6
        exp_objectIds = ["T104", "T103", "T102", "T101", "T104", "T105", "T106"]
        assert set(query_candidates.values) == set(exp_objectIds)

class Test__LoadTargetLightcurves():

    def test__load_lightcurve(self, atlas_qm, atlas_lc, target_list):
        
        atlas_qm.create_paths()
        
        for target in target_list[:5]:
            atlas_qm.add_target(target)
        assert len(atlas_qm.target_lookup) == 5
            
        processed_lc = process_atlas_lightcurve(atlas_lc)
        empty_lc = get_empty_atlas_lightcurve()
                
        atlas_qm.target_lookup["T101"].atlas_data.add_lightcurve(atlas_lc) # Full lightcurve
        atlas_qm.target_lookup["T102"].atlas_data.add_lightcurve(atlas_lc.iloc[:7]) # Partial
        atlas_qm.target_lookup["T103"].atlas_data.add_lightcurve(atlas_lc) # Full lightcurve
        
        assert len(atlas_qm.target_lookup["T101"].atlas_data.lightcurve) == 10
        assert len(atlas_qm.target_lookup["T102"].atlas_data.lightcurve) == 7
        assert len(atlas_qm.target_lookup["T103"].atlas_data.lightcurve) == 10
        assert atlas_qm.target_lookup["T104"].atlas_data.lightcurve is None
        assert atlas_qm.target_lookup["T105"].atlas_data.lightcurve is None
        
        atlas_lc.to_csv(atlas_qm.get_lightcurve_file("T101"), index=False)
        atlas_lc.to_csv(atlas_qm.get_lightcurve_file("T102"), index=False)
        empty_lc.to_csv(atlas_qm.get_lightcurve_file("T103"), index=False)
        # Nothing for 104
        # Nothing for 105
        
        for objectId, target in atlas_qm.target_lookup.items():
            assert not target.updated
        
        loaded, missing = atlas_qm.load_target_lightcurves()

        assert atlas_qm.target_lookup["T101"].updated is False
        assert atlas_qm.target_lookup["T102"].updated is True
        assert atlas_qm.target_lookup["T103"].updated is False
        assert atlas_qm.target_lookup["T104"].updated is False
        assert atlas_qm.target_lookup["T105"].updated is False

        assert len(loaded) == 1
        assert set(loaded) == set(["T102"])

        assert len(missing) == 2
        assert set(missing) == set(["T104", "T105"])
        
        

class Test__AtlasQuery:

    def test__init_does_nothing(self):
        aq = AtlasQuery()
        assert aq.atlas_base_url == "https://fallingstar-data.com/forcedphot"
        










