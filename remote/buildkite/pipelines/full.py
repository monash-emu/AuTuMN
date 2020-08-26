from remote.buildkite.buildkite import (
    Pipeline,
    CommandStep,
    InputStep,
    TextInputField,
    BooleanInputField,
    SelectInputField,
)


run_id_field = TextInputField(
    key="run-id",
    title="Calibration run name",
    hint="Which calibration run should be used?",
    type=str,
)
burn_in_field = TextInputField(
    key="burn-in",
    title="Burn-in",
    hint="How many MCMC iterations should be burned?",
    type=int,
    default=50,
)
use_latest_code_field = SelectInputField(
    key="use-latest-code",
    title="Use latest code for model run",
    hint="Should this task use the same Git commit as the calibration, or use the latest code instead?",
    type=bool,
    options=[{"label": "Yes", "value": "yes"}, {"label": "No", "value": ""}],
    default="",
)
trigger_field = BooleanInputField(
    key="trigger-downstream",
    title="Trigger full model run",
    hint="Should this task trigger a full model run when it is done?",
    type=bool,
    default="yes",
)
spot_field = BooleanInputField(
    key="spot-instance",
    title="Use spot instances",
    hint="Is 1/3 of the price but sometimes randomly fails.",
    default="yes",
    type=bool,
)
fields = [
    run_id_field,
    burn_in_field,
    use_latest_code_field,
    trigger_field,
    spot_field,
]
input_step = InputStep(
    key="full-model-run-settings", run_condition='build.env("SKIP_INPUT") == null', fields=fields
)
full_model_run_step = CommandStep(key="run-full", command="./scripts/buildkite.sh full",)
website_step = CommandStep(
    key="update-website",
    command="./scripts/website/deploy.sh",
    depends_on=full_model_run_step,
    allow_dependency_failure=True,
)
steps = [
    input_step,
    full_model_run_step,
    website_step,
]
pipeline = Pipeline(key="run-full", steps=steps)
