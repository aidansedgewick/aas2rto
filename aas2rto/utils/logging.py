import logging
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

SIMPLE_FORMATTER = "%(asctime)s %(name)s %(levelname)s: %(message)s"


def _get_init_logging_config():
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "aas2rto_filter": AAS2RTO_FILTER_CONFIG,
        },
        "formatters": {
            "simple": {
                "format": SIMPLE_FORMATTER,
            }
        },
        "handlers": {
            "console": {
                "class": logging.StreamHandler,
                "level": logging.DEBUG,
                "formatter": "simple",
                "filters": ["aas2rto_filter"],
            },
        },
        "root": {
            "level": logging.DEBUG,
            "handers": ["console"],
        },
    }


def _get_file_logging_config(
    log_path: Path, filepath: str = log.out, stream_level=logging.INFO
):
    log_filepath = log_path / filepath
    debug_log_filepath = log_path / "debug" / filepath

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "aas2rto_filter": AAS2RTO_FILTER_CONFIG,
        },
        "formatters": {
            "simple": {
                "format": SIMPLE_FORMATTER,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": logging.StreamHandler,
                "level": stream_level,
                "formatter": "simple",
                "filters": ["aas2rto_filter"],
            },
            "info_file": {
                "class": TimedRotatingFileHandler,
                "filename": log_filepath,
                "level": logging.INFO,
                "formatter": "simple",
                "filters": ["aas2rto_filter"],
                "when": "midnight",
            },
            "debug_file": {
                "class": TimedRotatingFileHandler,
                "filename": debug_log_filepath,
                "level": logging.DEBUG,
                "backupCount": 3,
                "formatter": "simple",
                "filters": ["aas2rto_filter"],
                "when": "midnight",
            },
        },
        "root": {
            "level": logging.DEBUG,
            "handlers": ["console", "info_file", "debug_file"],
        },
    }


def setup_initial_logging():
    logging_config = _get_init_logging_config()
    print("set loggers!")
    logging.config.dictConfig(logging_config)
    logger.info("here's a log")


def setup_file_logging(
    logs_path: Path,
    filename: str = "out.log",
    stream_level=logging.INFO,
    file_level=logging.DEBUG,
):
    logs_path = Path(logs_path)
    log_filepath = logs_path / filename

    logger.info(f"logs now written to:\n{log_filepath}")

    logging_config = _get_file_logging_config(
        log_filepath, stream_level=stream_level, file_level=file_level
    )
    logging.config.dictConfig(logging_config)
