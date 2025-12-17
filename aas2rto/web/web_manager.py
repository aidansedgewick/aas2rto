from logging import getLogger

from aas2rto import utils
from aas2rto.outputs.outputs_manager import OutputsManager
from aas2rto.path_manager import PathManager
from aas2rto.web.static_pages_manager import StaticPagesManager

logger = getLogger(__name__.split(".")[-1])

WEB_MANAGER_CLS_LOOKUP = {
    "static_pages": StaticPagesManager,
}


class WebManager:

    def __init__(
        self, config: dict, outputs_manager: OutputsManager, path_manager: PathManager
    ):
        self.config = config
        self.outputs_manager = outputs_manager
        self.path_manager = path_manager

        self.managers = {}

    def init_web_managers(self):

        utils.check_unexpected_config_keys(
            self.config, WEB_MANAGER_CLS_LOOKUP, name="web", raise_exc=True
        )
        for manager_name, manager_config in self.config.items():
            use = manager_config.pop("use", True)
            if not use:
                logger.info(f"{manager_name} has use={use} (False - like) - skip init!")

            cls = WEB_MANAGER_CLS_LOOKUP.get(manager_name)
            manager = cls(manager_config)
            self.managers[manager_name] = manager

    def perform_all_web_tasks(self):
        for manager_name, manager in self.managers.items():
            logger.info(f"perform {manager_name} tasks")
            manager.perform_all_tasks()
