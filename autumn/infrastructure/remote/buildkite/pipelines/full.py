from autumn.infrastructure.remote.buildkite.buildkite import (
    BooleanInputField,
    CommandStep,
    InputStep,
    Pipeline,
    TextInputField,
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
    default=500,
)
sample_size_field = TextInputField(
    key="sample-size",
    title="Sample size",
    hint="How many accepted runs per chain should be sampled for uncertainty calcs?",
    type=int,
    default=100,
)
commit_field = TextInputField(
    key="commit",
    title="Specify commit SHA",
    hint="Specify git commit, or leave as default to use calibration commit",
    type=str,
    default="use_original_commit",
)
trigger_field = BooleanInputField(
    key="trigger-downstream",
    title="Trigger PowerBI post processing job",
    hint="Should this task trigger a PowerBI post processing job when it is done?",
    type=bool,
    default="yes",
)

fields = [
    run_id_field,
    burn_in_field,
    sample_size_field,
    commit_field,
    trigger_field,
]
input_step = InputStep(
    key="full-model-run-settings", run_condition='build.env("SKIP_INPUT") == null', fields=fields
)
full_model_run_step = CommandStep(
    key="run-full",
    command="./scripts/buildkite.sh full",
)
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
