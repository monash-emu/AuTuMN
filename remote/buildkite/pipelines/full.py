from remote.buildkite.buildkite import (
    Pipeline,
    CommandStep,
    InputStep,
    TextInputField,
    SelectInputField,
)


run_id_field = TextInputField(
    key="run-id",
    text="Calibration run name",
    hint="Which calibration run should be used?",
    type=str,
)
burn_in_field = TextInputField(
    key="burn-in",
    text="Burn-in",
    hint="How many MCMC iterations should be burned?",
    type=int,
    default=50,
)
use_latest_code_field = SelectInputField(
    key="use-latest-code",
    select="Use latest code for model run",
    hint="Should this task use the same Git commit as the calibration, or use the latest code instead?",
    type=bool,
    options=[{"label": "Yes", "value": "yes"}, {"label": "No", "value": ""}],
    default="",
)
trigger_field = SelectInputField(
    key="trigger-downstream",
    select="Trigger full model run",
    hint="Should this task trigger a full model run when it is done?",
    type=bool,
    options=[{"label": "Yes", "value": "yes"}, {"label": "No", "value": ""}],
    default="yes",
)
fields = [
    run_id_field,
    burn_in_field,
    use_latest_code_field,
    trigger_field,
]
input_step = InputStep(
    key="full-model-run-settings", run_condition='build.env("SKIP_INPUT") == null', fields=fields
)
powerbi_step = CommandStep(key="run-full", command="./scripts/buildkite.sh full",)
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
pipeline = Pipeline(key="run-full", steps=steps)
