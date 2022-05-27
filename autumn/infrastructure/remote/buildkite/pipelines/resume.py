from autumn.infrastructure.remote.buildkite.buildkite import (
    BooleanInputField,
    CommandStep,
    InputStep,
    Pipeline,
    TextInputField,
)

from .calibrate import burn_in_field, sample_size_field, trigger_field, chains_field, runtime_field

run_id_field = TextInputField(
    key="run-id",
    title="Existing calibration run-id",
    hint="Which calibration run should be resumed?",
    type=str,
)

fields = [
    run_id_field,
    chains_field,
    runtime_field,
    burn_in_field,
    sample_size_field,
    trigger_field
]
input_step = InputStep(
    key="resume_calibration_settings", run_condition='build.env("SKIP_INPUT") == null', fields=fields
)
resume_calibration_step = CommandStep(
    key="resume_calibration",
    command="./scripts/buildkite.sh resume",
)
website_step = CommandStep(
    key="update-website",
    command="./scripts/website/deploy.sh",
    depends_on=resume_calibration_step,
    allow_dependency_failure=True,
)
steps = [
    input_step,
    resume_calibration_step,
    website_step,
]
pipeline = Pipeline(key="resume_calibration", steps=steps)
