import abc

from pathlib import Path

from dk154_targets import paths


class BaseQueryManager(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def perform_all_tasks(self):
        raise NotImplementedError

    def process_paths(self, data_path: Path = None, create_paths=True):
        if data_path is None:
            data_path = paths.base_path / paths.default_data_dir
        self.parent_data_path = data_path / self.name
        self.lightcurves_path = self.parent_data_path / "lightcurves"
        self.alerts_path = self.parent_data_path / "alerts"
        self.probabilities_path = self.parent_data_path / "probabilities"
        self.parameters_path = self.parent_data_path / "parameters"
        self.magstats_path = self.parent_data_path / "magstats"
        self.query_results_path = self.parent_data_path / "query_results"
        self.cutouts_path = self.parent_data_path / "cutouts"
        if create_paths:
            self.parent_data_path.mkdir(exist_ok=True, parents=True)
            self.lightcurves_path.mkdir(exist_ok=True, parents=True)
            self.alerts_path.mkdir(exist_ok=True, parents=True)
            self.probabilities_path.mkdir(exist_ok=True, parents=True)
            self.parameters_path.mkdir(exist_ok=True, parents=True)
            self.magstats_path.mkdir(exist_ok=True, parents=True)
            self.query_results_path.mkdir(exist_ok=True, parents=True)
            self.cutouts_path.mkdir(exist_ok=True, parents=True)

    def get_query_results_file(self, query_name, fmt="csv") -> Path:
        return self.query_results_path / f"{query_name}.{fmt}"

    def get_alert_dir(self, objectId) -> Path:
        return self.alerts_path / f"{objectId}"

    def get_alert_file(self, objectId, candid, fmt="json", mkdir=True) -> Path:
        alert_dir = self.get_alert_dir(objectId)
        if mkdir:
            alert_dir.mkdir(exist_ok=True, parents=True)
        return alert_dir / f"{candid}.{fmt}"

    def get_magstats_dir(self, objectId) -> Path:
        return self.magstats_path  # / f"{objectId}"

    def get_magstats_file(self, objectId) -> Path:
        magstats_dir = self.get_magstats_dir(objectId)
        return magstats_dir / f"{objectId}.csv"

    def get_lightcurve_file(self, objectId, fmt="csv") -> Path:
        return self.lightcurves_path / f"{objectId}.{fmt}"

    def get_cutouts_dir(self, objectId) -> Path:
        return self.cutouts_path / f"{objectId}"

    def get_cutouts_file(self, objectId, candid, fmt="pkl", mkdir=True) -> Path:
        cutouts_dir = self.get_cutouts_dir(objectId)
        if mkdir:
            cutouts_dir.mkdir(exist_ok=True, parents=True)
        return cutouts_dir / f"{candid}.{fmt}"
