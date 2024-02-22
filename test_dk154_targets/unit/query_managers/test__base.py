from dk154_targets.query_managers.base import BaseQueryManager

import pytest

class NewQueryManager(BaseQueryManager):    
    name = "new_qm"
    
    def __init__(
        self, config, target_lookup, data_path=None, parent_path=None, create_paths=True
    ):
        self.config = config
        self.target_lookup = target_lookup
        
        self.process_paths(
            data_path=data_path, parent_path=parent_path, create_paths=create_paths
        )
    
    def perform_all_tasks(self):
        test_file = self.data_path / "test_file.csv"
        with open(test_file, "w+") as f:
            f.write("some_data")
            
class MissingNameQueryManager(BaseQueryManager):
    
    def __init__(
        self, config, target_lookup, data_path=None, parent_path=None, create_paths=True       
    ):
        self.config = config
        self.target_lookup=target_lookup
        
        self.process_paths(data_path=data_path, parent_path=parent_path, create_paths=True)
        
    def perform_all_tasks(self):
        return
        
class MissingTasksQueryManager(BaseQueryManager):

    name = "missing_tasks"
    
    def __init__(
        self, config, target_lookup, data_path=None, parent_path=None, create_paths=True       
    ):
        self.config = config
        self.target_lookup=target_lookup
        
        self.process_paths(data_path=data_path, parent_path=parent_path, create_paths=True)


@pytest.fixture
def config():
    return {}

@pytest.fixture
def target_lookup():
    return {}

class Test__ValidSubclassQueryManager:

    def test__parent_path(self, config, target_lookup, tmp_path):
                
        qm = NewQueryManager(config, target_lookup, parent_path=tmp_path)
        qm.perform_all_tasks()
        
        assert qm.parent_path == tmp_path
        assert qm.data_path == tmp_path / "new_qm"
                
        assert qm.data_path.stem == "new_qm"
        assert qm.data_path.exists()
        
        assert qm.lightcurves_path == tmp_path / "new_qm/lightcurves"
        assert qm.lightcurves_path.exists()
                
        expected_test_file = tmp_path / "new_qm/test_file.csv"
        assert expected_test_file.exists()
        with open(expected_test_file) as f:
            data = f.readline()
        assert data == "some_data"
        
    def test__data_path(self, config, target_lookup, tmp_path):
        data_path = tmp_path / "data_goes_here"
        qm = NewQueryManager(config, target_lookup, data_path=data_path)
        assert qm.data_path == tmp_path / "data_goes_here"
        assert qm.parent_path == tmp_path       
        
        
    def test__create_paths_false(self, config, target_lookup, tmp_path):
        qm = NewQueryManager(config, target_lookup, parent_path=tmp_path, create_paths=False)
        assert qm.data_path.stem == "new_qm"
        assert qm.lightcurves_path == tmp_path / "new_qm/lightcurves"
        assert not qm.lightcurves_path.exists()
        
        
    def test__file_convenience_functions(self, config, target_lookup, tmp_path):
        qm = NewQueryManager(config, target_lookup, parent_path=tmp_path)
        
        exp_query_results_file = tmp_path / "new_qm/query_results/test_query.csv"
        assert qm.get_query_results_file("test_query") == exp_query_results_file
        
        exp_alert_dir = tmp_path / "new_qm/alerts/ZTF00abc"
        assert qm.get_alert_dir("ZTF00abc") == exp_alert_dir
        
        exp_alert_file = tmp_path / "new_qm/alerts/ZTF00abc/2400010000200001001.json"
        assert qm.get_alert_file("ZTF00abc", 24000_10000_20000_1001) == exp_alert_file
        assert exp_alert_file.parent.exists()
        
        exp_magstats_file = tmp_path / "new_qm/magstats/ZTF00abc.csv"
        assert qm.get_magstats_file("ZTF00abc") == exp_magstats_file
        
        exp_lightcurve_file = tmp_path / "new_qm/lightcurves/ZTF00abc.csv"
        assert qm.get_lightcurve_file("ZTF00abc") == exp_lightcurve_file
        
        exp_cutouts_dir = tmp_path / "new_qm/cutouts/ZTF00abc"
        assert qm.get_cutouts_dir("ZTF00abc") == exp_cutouts_dir
        
        exp_cutouts_file = tmp_path / "new_qm/cutouts/ZTF00abc/1234510000200001001.pkl"
        assert qm.get_cutouts_file("ZTF00abc", 12345_10000_20000_1001) == exp_cutouts_file
        assert exp_cutouts_file.parent.exists()
        
        exp_parameters_file = tmp_path / "new_qm/parameters/ZTF00abc.pkl"
        assert qm.get_parameters_file("ZTF00abc") == exp_parameters_file
        
class Test__InvalidSubclassQueryManager:
        
    def test__missing_name_raises_error(self, config, target_lookup):
        
        with pytest.raises(TypeError):
            qm = MissingNameQueryManager(config, target_lookup)
            
    def test__missing_process_all_tasks_raises_error(self, config, target_lookup):
        with pytest.raises(TypeError):
            qm = MissingTasksQueryManager(self, config, target_lookup)
    
        
        
        
        
        
        
        
        
        
        
        
