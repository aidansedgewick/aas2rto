import abc

from pathlib import Path

from dk154_targets.target import TargetData
from dk154_targets import paths

from dk154_targets import utils


EXPECTED_DIRECTORIES = [
    "lightcurves",
    "alerts",
    "probabilities",
    "parameters",
    "magstats",
    "query_results",
    "cutouts",
    "photometry",
]


class BaseQueryManager(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def perform_all_tasks(self):
        raise NotImplementedError

    def add_target(self, target):
        if target.objectId in self.target_lookup:
            raise ValueError(
                f"{self.name}: obj {target.objectId} already in target_lookup"
            )
        self.target_lookup[target.objectId] = target

    def process_paths(
        self,
        data_path: Path = None,
        parent_path: Path = None,
        create_paths: bool = True,
        # paths: list = EXPECTED_DIRECTORIES,
    ):
        """
        If data path is None
        """
        if data_path is None:
            if parent_path is None:
                parent_path = paths.base_path / paths.default_data_dir
            parent_path = Path(parent_path)
            data_path = parent_path / self.name
        else:
            data_path = Path(data_path)
            parent_path = data_path.parent

        self.parent_path = parent_path
        self.data_path = data_path

        # utils.check_unexpected_config_keys(
        #    dirs, EXPECTED_DIRECTORIES, name=f"{self.name} qm __init__(paths)"
        # )

        self.lightcurves_path = self.data_path / "lightcurves"
        self.alerts_path = self.data_path / "alerts"
        self.probabilities_path = self.data_path / "probabilities"
        self.parameters_path = self.data_path / "parameters"
        self.magstats_path = self.data_path / "magstats"
        self.query_results_path = self.data_path / "query_results"
        self.cutouts_path = self.data_path / "cutouts"
        if create_paths:
            self.create_paths()

    def create_paths(self):
        self.data_path.mkdir(exist_ok=True, parents=True)
        self.lightcurves_path.mkdir(exist_ok=True, parents=True)
        self.alerts_path.mkdir(exist_ok=True, parents=True)
        self.probabilities_path.mkdir(exist_ok=True, parents=True)
        self.parameters_path.mkdir(exist_ok=True, parents=True)
        self.magstats_path.mkdir(exist_ok=True, parents=True)
        self.query_results_path.mkdir(exist_ok=True, parents=True)
        self.cutouts_path.mkdir(exist_ok=True, parents=True)

    def init_missing_target_data(self):
        for objectId, target in self.target_lookup.items():
            qm_data = target.target_data.get(self.name, None)
            if qm_data is None:
                assert self.name not in target.target_data
                target.target_data[self.name] = TargetData()

    def get_query_results_file(self, query_name, fmt="csv") -> Path:
        return self.query_results_path / f"{query_name}.{fmt}"

    def get_alert_dir(self, objectId) -> Path:
        return self.alerts_path / f"{objectId}"

    def get_alert_file(self, objectId, candid, fmt="json", mkdir=True) -> Path:
        alert_dir = self.get_alert_dir(objectId)
        if mkdir:
            alert_dir.mkdir(exist_ok=True, parents=True)
        return alert_dir / f"{candid}.{fmt}"

    def get_magstats_file(self, objectId) -> Path:
        return self.magstats_path / f"{objectId}.csv"

    def get_lightcurve_file(self, objectId, fmt="csv") -> Path:
        return self.lightcurves_path / f"{objectId}.{fmt}"

    def get_probabilities_file(self, objectId, fmt="csv") -> Path:
        return self.probabilities_path / f"{objectId}.{fmt}"

    def get_cutouts_dir(self, objectId) -> Path:
        return self.cutouts_path / f"{objectId}"

    def get_cutouts_file(self, objectId, candid, fmt="pkl", mkdir=True) -> Path:
        cutouts_dir = self.get_cutouts_dir(objectId)
        if mkdir:
            cutouts_dir.mkdir(exist_ok=True, parents=True)
        return cutouts_dir / f"{candid}.{fmt}"

    def get_parameters_file(self, objectId, fmt="pkl") -> Path:
        return self.parameters_path / f"{objectId}.{fmt}"
