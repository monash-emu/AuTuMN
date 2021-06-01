from autumn.remote.buildkite.buildkite import CommandStep, InputStep, Pipeline

from .calibrate import (
    branch_field,
    chains_field,
    runtime_field,
    spot_field,
    trigger_field,
)
from .full import burn_in_field, sample_size_field

fields = [
    chains_field,
    branch_field,
    runtime_field,
    burn_in_field,
    sample_size_field,
    trigger_field,
    spot_field,
]
input_step = InputStep(
    key="calibration-settings",
    run_condition=None,
    fields=fields,
)
trigger_step = CommandStep(key="run-triggers", command="./scripts/buildkite.sh trigger philippines")
pipeline = Pipeline(key="trigger-philippines", steps=[input_step, trigger_step])
