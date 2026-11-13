import logging
import logging.config
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

from aas2rto import paths

logger = logging.getLogger("logging_utils")


class AAS2RTOLoggerFilter(logging.Filter):
    def __init__(self, project_root: Path = paths.BASE_PATH):
        super().__init__()
        self.project_root = Path(project_root).resolve()

    def filter(self, record):
        record_path = Path(record.pathname).resolve()
        emit = record_path.is_relative_to(self.project_root)
        return emit


AAS2RTO_FILTER_CONFIG = {
    "()": AAS2RTOLoggerFilter,
    "project_root": paths.BASE_PATH,
}

SIMPLE_FORMATTER = {
    "format": "%(asctime)s %(name)s %(levelname)s: %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
}


def _get_logging_config(
    logs_path: Path = None,
    filename: str = "log.out",
    use_file_logger: bool = True,
    stream_level=logging.INFO,
    debug_file: bool = False,
):

    console_handler = {
        "class": logging.StreamHandler,
        "level": stream_level,
        "formatter": "simple",
        "filters": ["aas2rto_filter"],
    }

    handlers_config = {"console": console_handler}

    if use_file_logger:
        if logs_path is None:
            raise ValueError("Must provide 'logs_path' if use_file_logger=True")

        log_filepath = Path(logs_path) / filename
        debug_log_filepath = Path(logs_path) / "debug" / filename

        info_file_handler = {
            "class": TimedRotatingFileHandler,
            "filename": log_filepath,
            "level": logging.INFO,
            "formatter": "simple",
            "backupCount": 21,
            "filters": ["aas2rto_filter"],
            "when": "midnight",
        }
        debug_file_handler = {
            "class": TimedRotatingFileHandler,
            "filename": debug_log_filepath,
            "level": logging.DEBUG,
            "backupCount": 3,
            "formatter": "simple",
            "filters": ["aas2rto_filter"],
            "when": "midnight",
        }

        handlers_config["info_file"] = info_file_handler
        if debug_file:
            debug_log_filepath.parent.mkdir(exist_ok=True)
            handlers_config["debug_file"] = debug_file_handler

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"aas2rto_filter": AAS2RTO_FILTER_CONFIG},
        "formatters": {"simple": SIMPLE_FORMATTER},
        "handlers": handlers_config,
        "root": {
            "level": logging.DEBUG,
            "handlers": list(handlers_config.keys()),
        },
    }


def setup_stream_logging(stream_level=logging.INFO):
    logging_config = _get_logging_config(
        stream_level=stream_level,
        use_file_logger=False,
    )
    logging.config.dictConfig(logging_config)


def setup_file_logging(
    logs_path: Path,
    filename: str = "out.log",
    stream_level=logging.INFO,
    debug_file: bool = True,
):
    logs_path = Path(logs_path)

    logger.info(f"logs now also written to:\n    {logs_path}")

    logging_config = _get_logging_config(
        logs_path=logs_path,
        filename=filename,
        use_file_logger=True,
        stream_level=stream_level,
        debug_file=debug_file,
    )
    logging.config.dictConfig(logging_config)
