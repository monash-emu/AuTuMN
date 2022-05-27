import logging
import logging.config
import socket
import os

from autumn.core.utils.runs import read_run_id
from autumn.core.project import Project, get_project


def get_project_from_run_id(run_id: str) -> Project:
    app_name, region_name, _, _ = read_run_id(run_id)
    return get_project(app_name, region_name)


def set_logging_config(verbose: bool, chain="main", log_path="log", task="task"):
    old_factory = logging.getLogRecordFactory()
    if chain != "main":
        chain = f"chain-{chain}"

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.chain = chain
        record.host = socket.gethostname()
        return record

    logging.setLogRecordFactory(record_factory)

    log_format = "%(asctime)s %(host)s [%(chain)s] %(levelname)s %(message)s"
    logfile = os.path.join(log_path, f"{task}-{chain}.log")
    root_logger = {"level": "INFO", "handlers": ["file"]}
    handlers = {
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": logfile,
            "formatter": "app",
            "encoding": "utf-8",
        }
    }
    if verbose:
        root_logger["handlers"].append("stream")
        handlers["stream"] = {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "app",
        }

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "root": root_logger,
            "handlers": handlers,
            "formatters": {
                "app": {
                    "format": log_format,
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
        }
    )
