import logging
import pprint
import os
import subprocess as sp

import yaml

from autumn.constants import DATA_PATH


logger = logging.getLogger(__name__)

PIPELINE_PATH = os.path.join(DATA_PATH, "buildkite")


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


def trigger_pipeline(label: str, target: str, msg: str, env: dict = {}, meta: dict = {}):
    pipeline_data = {
        "steps": [
            {
                "label": label,
                "trigger": target,
                "async": True,
                "build": {
                    "message": msg,
                    "commit": os.environ["BUILDKITE_COMMIT"],
                    "branch": os.environ["BUILDKITE_BRANCH"],
                    "env": env,
                    "meta_data": meta,
                },
            }
        ]
    }
    _trigger_pipeline(pipeline_data)


class Pipeline:
    def __init__(self, key, steps):
        self.steps = steps
        self.path = os.path.join(PIPELINE_PATH, f"{key}.yml")

    def save(self):
        with open(self.path, "w") as f:
            yaml.dump(self.to_dict(), f)

    def to_dict(self):
        return {"steps": [s.to_dict() for s in self.steps]}


class CommandStep:
    def __init__(self, key: str, command: str, depends_on=None, allow_dependency_failure=False):
        self.key = key
        self.depends_on = depends_on
        self.command = command
        self.allow_dependency_failure = allow_dependency_failure

    @property
    def label(self):
        return self.key.replace("-", " ").title()

    def to_dict(self):
        return {
            "command": self.command,
            "key": self.key,
            "label": self.label,
        }
        if self.depends_on is not None:
            input_dict["depends_on"] = self.depends_on.key
            input_dict["allow_dependency_failure"] = self.allow_dependency_failure


class InputStep:
    def __init__(self, key: str, run_condition: str, fields):
        self.key = key
        self.run_condition = run_condition
        self.fields = fields

    @property
    def label(self):
        return self.key.replace("-", " ").title()

    def to_dict(self):
        input_dict = {
            "block": self.label,
            "key": self.key,
            "fields": [f.to_dict() for f in self.fields],
        }
        if self.run_condition:
            input_dict["if"] = self.run_condition

        return input_dict


class BaseInputField:
    def __init__(self, key: str, hint: str, type, default=None):
        self.key = key
        self.hint = hint
        self.default = default
        self.type = type

    def to_dict(self):
        input_dict = {
            "key": self.key,
            "hint": self.hint,
            "required": True,
        }
        if self.default is not None:
            input_dict["default"] = self.default

        return input_dict

    def get_value(self):
        val_str = get_metadata(self.key)
        return self.type(val_str)


class TextInputField(BaseInputField):
    def __init__(self, text: str, *args, **kwargs):
        self.text = text
        super().__init__(*args, **kwargs)

    def to_dict(self):
        return {
            **super().to_dict(),
            "text": self.text,
        }


class SelectInputField(BaseInputField):
    def __init__(self, select: str, options: list, *args, **kwargs):
        self.select = select
        self.options = options
        if not callable(options):
            assert all([type(o) is dict and "label" in o and "value" in o for o in options])
        super().__init__(*args, **kwargs)

    def to_dict(self):
        return {
            **super().to_dict(),
            "select": self.select,
            "options": self.options() if callable(self.options) else self.options,
        }
