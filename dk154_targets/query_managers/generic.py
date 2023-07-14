import logging

from dk154_targets.query_managers.base import BaseQueryManager

from dk154_targets import paths

logger = logging.getLogger(__name__)


class UsingGenericWarning(Warning):
    pass


class GenericQueryManager(BaseQueryManager):
    name = "generic"

    def __init__(self, qm_config, target_lookup, data_path=None, create_paths=True):
        self.config = qm_config
        self.data_path = paths.base_path / paths.default_data_dir
        self.target_lookup = target_lookup

        self.data_path = data_path

        pass

    def perform_all_tasks():
        logger.info("performing no tasks")
        pass
