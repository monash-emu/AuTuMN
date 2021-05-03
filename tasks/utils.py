import logging
import logging.config
import socket

from apps import covid_19, tuberculosis, tuberculosis_strains
from utils.runs import read_run_id

APP_MAP = {
    "covid_19": covid_19,
    "tuberculosis": tuberculosis,
    "tuberculosis_strains": tuberculosis_strains,
}


def get_app_region(run_id: str):
    app_name, region_name, _, _ = read_run_id(run_id)
    app_module = APP_MAP[app_name]
    return app_module.app.get_region(region_name)


def set_logging_config(verbose: bool, chain="main"):
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
    logfile = f"log/task-{chain}.log"
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
