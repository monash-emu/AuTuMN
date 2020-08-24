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


def _trigger_pipeline(pipeline_data: dict):
    """Trigger a downstream pipeline run"""
    data_str = pprint.pformat(pipeline_data, indent=2)
    logger.info("Triggering Buildkite pipeline:\n%s", data_str)
    yaml_str = yaml.dump(pipeline_data)
    cmd = f"buildkite-agent pipeline upload"
    proc = sp.run(cmd, shell=True, check=True, input=yaml_str, stdout=sp.PIPE, encoding="utf-8")
    stdout = proc.stdout.strip() if proc.stdout else ""
    stderr = proc.stderr.strip() if proc.stderr else ""
    if stdout:
        logger.info("stdout for trigger pipeline: %s", stdout)
    if stderr:
        logger.info("stderr for trigger pipeline: %s", stderr)


def trigger_pipeline(label: str, target: str, msg: str, run_id: str, params):
    pipeline_data = {
        "steps": [
            {
                "label": label,
                "trigger": target,
                "async": True,
                "build": {
                    "message": msg,
                    "commit": params.buildkite.commit,
                    "branch": params.buildkite.branch,
                    "env": {"RUN_ID": run_id},
                },
            }
        ]
    }
    _trigger_pipeline(pipeline_data)
