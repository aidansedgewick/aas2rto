import logging

from .base import BaseQueryManager

from dk154_targets import paths

logger = logging.getLogger(__name__)

class UsingGenericWarning(Warning):
    pass

class GenericQueryManager(BaseQueryManager):

    def __init__(self, qm_config, target_lookup, data_path=None):
        self.config = qm_config
        self.data_path = paths.default_data_path
        self.target_lookup = target_lookup

        pass

    def perform_all_tasks():
        logger.info("performing no tasks")
        pass