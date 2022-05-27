import logging

logger = logging.getLogger(__name__)


def run_stub(**kwargs):
    for k, v in kwargs.items():
        msg = f"Key: {k} Value: {v}"
        logger.info(msg)
