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


def _get_stream_logging_config(stream_level=logging.INFO):
    CONSOLE_HANDLER = {
        "class": logging.StreamHandler,
        "level": stream_level,
        "formatter": "simple",
        "filters": ["aas2rto_filter"],
    }
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"aas2rto_filter": AAS2RTO_FILTER_CONFIG},
        "formatters": {"simple": SIMPLE_FORMATTER},
        "handlers": {"console": CONSOLE_HANDLER},
        "root": {
            "level": logging.DEBUG,
            "handers": ["console"],
        },
    }


def _get_file_logging_config(
    log_path: Path,
    filename: str = "log.out",
    stream_level=logging.INFO,
    debug_file: bool = False,
):
    log_filepath = Path(log_path) / filename
    debug_log_filepath = Path(log_path) / "debug" / filename

    CONSOLE_HANDLER = {
        "class": logging.StreamHandler,
        "level": stream_level,
        "formatter": "simple",
        "filters": ["aas2rto_filter"],
    }
    INFO_FILE_HANDLER = {
        "class": TimedRotatingFileHandler,
        "filename": log_filepath,
        "level": logging.INFO,
        "formatter": "simple",
        "backupCount": 21,
        "filters": ["aas2rto_filter"],
        "when": "midnight",
    }
    DEBUG_FILE_HANDLER = {
        "class": TimedRotatingFileHandler,
        "filename": debug_log_filepath,
        "level": logging.DEBUG,
        "backupCount": 3,
        "formatter": "simple",
        "filters": ["aas2rto_filter"],
        "when": "midnight",
    }

    HANDLERS_CONFIG = {
        "console": CONSOLE_HANDLER,
        "info_file": INFO_FILE_HANDLER,
    }
    if debug_file:
        debug_log_filepath.parent.mkdir(exist_ok=True)
        HANDLERS_CONFIG["debug_file"] = DEBUG_FILE_HANDLER

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "aas2rto_filter": AAS2RTO_FILTER_CONFIG,
        },
        "formatters": {"simple": SIMPLE_FORMATTER},
        "handlers": HANDLERS_CONFIG,
        "root": {
            "level": logging.DEBUG,
            "handlers": list(HANDLERS_CONFIG.keys()),
        },
    }


def setup_stream_logging(stream_level=logging.INFO):
    logging_config = _get_stream_logging_config(stream_level=stream_level)
    logging.config.dictConfig(logging_config)
    logger.info("here's a log")


def setup_file_logging(
    logs_path: Path,
    filename: str = "out.log",
    stream_level=logging.INFO,
    debug_file: bool = True,
):
    logs_path = Path(logs_path)

    logger.info(f"logs now written to:\n{logs_path}")

    logging_config = _get_file_logging_config(
        logs_path, filename=filename, stream_level=stream_level, debug_file=debug_file
    )
    logging.config.dictConfig(logging_config)
