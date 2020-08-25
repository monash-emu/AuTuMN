from remote.buildkite.buildkite import (
    Pipeline,
    CommandStep,
    InputStep,
    TextInputField,
)


run_id_field = TextInputField(
    key="run-id",
    title="Calibration run name",
    hint="Which calibration run should be used?",
    type=str,
)
fields = [run_id_field]
input_step = InputStep(
    key="powerbi-processing-settings",
    run_condition='build.env("SKIP_INPUT") == null',
    fields=fields,
)
powerbi_step = CommandStep(key="run-powerbi", command="./scripts/buildkite.sh powerbi",)
website_step = CommandStep(
    key="update-website",
    command="./scripts/website/deploy.sh",
    depends_on=powerbi_step,
    allow_dependency_failure=True,
)
steps = [
    input_step,
    powerbi_step,
    website_step,
]
pipeline = Pipeline(key="powerbi", steps=steps)
