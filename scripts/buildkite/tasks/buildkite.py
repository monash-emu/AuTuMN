import logging
import pprint
import subprocess as sp

import yaml

logger = logging.getLogger(__name__)


def get_metadata(key: str):
    """Read in a Buildkite metadata key value pair"""
    logger.info("Fetching Buildkite metadata %s", key)
    cmd = f"buildkite-agent meta-data get {key}"
    proc = sp.run(cmd, shell=True, check=True, stdout=sp.PIPE, encoding="utf-8")
    stdout = proc.stdout.strip() if proc.stdout else ""
    stderr = proc.stderr.strip() if proc.stderr else ""
    if stderr:
        logger.info("stderr for metadata fetch: %s", stderr)
    if not stdout:
        raise ValueError(f"No stdout returned for metadata key: {key}")

    return stdout


def trigger_pipeline(pipeline_data: dict):
    """Trigger a downstream pipeline run"""
    data_str = pprint.pformat(pipeline_data, indent=2)
    logger.info("Triggering Buildkite pipeline: %s", data_str)
    stdin = yaml.dump(pipeline_data)
    cmd = f"buildkite-agent pipeline upload"
    proc = sp.run(cmd, shell=True, check=True, stdin=stdin, stdout=sp.PIPE, encoding="utf-8")
    stdout = proc.stdout.strip() if proc.stdout else ""
    stderr = proc.stderr.strip() if proc.stderr else ""
    if stderr:
        logger.info("stderr for metadata fetch: %s", stderr)
    if not stdout:
        raise ValueError(f"No stdout returned for metadata key: {key}")

    return stdout

