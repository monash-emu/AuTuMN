import logging
import os
import pprint
import subprocess as sp

import yaml

from autumn.settings import DATA_PATH

logger = logging.getLogger(__name__)

PIPELINE_PATH = os.path.join(DATA_PATH, "buildkite")


def get_metadata(key: str, required=True):
    """Read in a Buildkite metadata key value pair"""
    logger.info("Fetching Buildkite metadata %s", key)
    cmd = f"buildkite-agent meta-data get {key}"
    proc = sp.run(cmd, shell=True, check=False, stdout=sp.PIPE, encoding="utf-8")
    stdout = proc.stdout.strip() if proc.stdout else ""
    stderr = proc.stderr.strip() if proc.stderr else ""
    if stderr:
        logger.info("stderr for metadata fetch: %s", stderr)
    if not stdout:
        if required:
            raise ValueError(f"No stdout returned for metadata key: {key}")
        else:
            return None

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


def trigger_pipeline(label: str, target: str, msg: str, env: dict = None, meta: dict = None):
    env = env or {}
    meta = meta or {}
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
        cmd_dict = {
            "command": self.command,
            "key": self.key,
            "label": self.label,
        }
        if self.depends_on is not None:
            cmd_dict["depends_on"] = self.depends_on.key
            cmd_dict["allow_dependency_failure"] = self.allow_dependency_failure

        return cmd_dict


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
    def __init__(self, key: str, title: str, hint: str, type, default=None, required=True):
        self.key = key
        self.hint = hint
        self.default = default
        self.type = type
        self.title = title
        self.required = required
        self._value = None

    def to_dict(self):
        input_dict = {
            "key": self.key,
            "hint": self.hint,
            "required": self.required,
        }
        if self.default is not None:
            input_dict["default"] = str(self.default)

        return input_dict

    def get_value(self):
        if self._value is not None:
            return self._value

        val_str = get_metadata(self.key, self.required)
        self._value = self.type(val_str)
        return self._value


class TextInputField(BaseInputField):
    def to_dict(self):
        return {
            **super().to_dict(),
            "text": self.title,
        }


class SelectInputField(BaseInputField):
    def __init__(self, options: list, *args, **kwargs):
        self.options = options
        if not callable(options):
            assert all([type(o) is dict and "label" in o and "value" in o for o in options])
        super().__init__(*args, **kwargs)

    def to_dict(self):
        return {
            **super().to_dict(),
            "select": self.title,
            "options": self.options() if callable(self.options) else self.options,
        }


class BooleanInputField(BaseInputField):
    def to_dict(self):
        return {
            **super().to_dict(),
            "select": self.title,
            "options": [{"label": "Yes", "value": "yes"}, {"label": "No", "value": "no"}],
        }

    def get_option(self, value: bool):
        return "yes" if value else "no"

    def get_value(self):
        if self._value is not None:
            return self._value

        val_str = get_metadata(self.key)
        val_bool = val_str == "yes"
        self._value = self.type(val_bool)
        return self._value
