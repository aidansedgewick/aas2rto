import os
import pytest

from dk154_targets.query_managers import BaseQueryManager

from dk154_targets import paths


class ExampleQueryManager(BaseQueryManager):
    name = "example_qm"

    def perform_all_tasks(self):
        pass


class Test__BaseQueryManager:
    @classmethod
    def _clear_test_directories(cls, name):
        exp_data_path = paths.test_data_path / name
        exp_lightcurves_path = paths.test_data_path / f"{name}/lightcurves"
        exp_alerts_path = paths.test_data_path / f"{name}/alerts"
        exp_query_results_path = paths.test_data_path / f"{name}/query_results"
        exp_probabilities_path = paths.test_data_path / f"{name}/probabilities"
        exp_parameters_path = paths.test_data_path / f"{name}/parameters"
        exp_magstats_path = paths.test_data_path / f"{name}/magstats"
        exp_cutouts_path = paths.test_data_path / f"{name}/cutouts"

        for path in [
            exp_lightcurves_path,
            exp_alerts_path,
            exp_query_results_path,
            exp_probabilities_path,
            exp_parameters_path,
            exp_magstats_path,
            exp_cutouts_path,
        ]:
            if path.exists():
                for filepath in path.glob("*.csv"):
                    os.remove(filepath)
                for filepath in path.glob("*.json"):
                    os.remove(filepath)
                path.rmdir()
        if exp_data_path.exists():
            exp_data_path.rmdir()

    def test__inherit(self):
        qm = ExampleQueryManager()

        class FailingTestQueryManager(BaseQueryManager):
            pass  # no name attribute.

        with pytest.raises(TypeError):
            qm = FailingTestQueryManager()

    def test__process_paths(self):
        self._clear_test_directories("example_qm")

        parent_data_path = paths.test_data_path / "example_qm"
        assert not parent_data_path.exists()

        qm = ExampleQueryManager()
        qm.process_paths(data_path=paths.test_data_path, create_paths=True)

        exp_data_path = paths.test_data_path / "example_qm"
        exp_lightcurves_path = paths.test_data_path / "example_qm/lightcurves"
        exp_alerts_path = paths.test_data_path / "example_qm/alerts"
        exp_query_results_path = paths.test_data_path / "example_qm/query_results"
        exp_probabilities_path = paths.test_data_path / "example_qm/probabilities"
        exp_magstats_path = paths.test_data_path / "example_qm/magstats"
        exp_cutouts_path = paths.test_data_path / "example_qm/cutouts"
        assert exp_data_path.exists()
        assert exp_lightcurves_path.exists()
        assert exp_alerts_path.exists()
        assert exp_query_results_path.exists()
        assert exp_probabilities_path.exists()
        assert exp_magstats_path.exists()
        assert exp_cutouts_path.exists()

        self._clear_test_directories("example_qm")
        assert not exp_data_path.exists()
        assert not exp_lightcurves_path.exists()
        assert not exp_alerts_path.exists()
        assert not exp_query_results_path.exists()
        assert not exp_probabilities_path.exists()
        assert not exp_magstats_path.exists()
        assert not exp_cutouts_path.exists()

    def test__get_files(self):
        qm = ExampleQueryManager()
        qm.process_paths(data_path=paths.test_data_path, create_paths=False)

        exp_query_results_file = (
            paths.test_data_path / "example_qm/query_results/test_query.csv"
        )
        assert qm.get_query_results_file("test_query") == exp_query_results_file

        objectId = "ZTF23testobj"
        candid = 23000_10000_20000_1234

        exp_lc_file = paths.test_data_path / "example_qm/lightcurves/ZTF23testobj.csv"
        assert qm.get_lightcurve_file(objectId) == exp_lc_file

        exp_cutouts_dir = paths.test_data_path / "example_qm/cutouts/ZTF23testobj"
        if exp_cutouts_dir.exists():
            exp_cutouts_dir.rmdir()
        assert qm.get_cutouts_dir(objectId) == exp_cutouts_dir
        exp_cutout_file = (
            paths.test_data_path
            / "example_qm/cutouts/ZTF23testobj/2300010000200001234.pkl"
        )
        assert qm.get_cutouts_file(objectId, candid, mkdir=False) == exp_cutout_file
        assert not exp_cutouts_dir.exists()

        exp_alerts_dir = paths.test_data_path / "example_qm/alerts/ZTF23testobj"
        if exp_alerts_dir.exists():
            exp_alerts_dir.rmdir()
        assert qm.get_alert_dir(objectId) == exp_alerts_dir
        exp_alert_file = (
            paths.test_data_path
            / "example_qm/alerts/ZTF23testobj/2300010000200001234.json"
        )
        assert qm.get_alert_file(objectId, candid, mkdir=False) == exp_alert_file
        assert not exp_alerts_dir.exists()
