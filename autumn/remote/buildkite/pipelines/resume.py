from autumn.remote.buildkite.buildkite import (
    BooleanInputField,
    CommandStep,
    InputStep,
    Pipeline,
    TextInputField,
)

run_id_field = TextInputField(
    key="run-id",
    title="Existing calibration run-id",
    hint="Which calibration run should be resumed?",
    type=str,
)
chains_field = TextInputField(
    key="num-chains",
    title="Number of MCMC chains",
    hint="How many MCMC chains do you want to run?",
    default=7,
    type=int,
)
branch_field = TextInputField(
    key="mcmc-branch",
    title="Model git branch name",
    hint="Which git branch do you want to use to run the model?",
    default="master",
    type=str,
)
runtime_field = TextInputField(
    key="mcmc-runtime",
    title="Runtime",
    hint="How many hours should the model run for?",
    default=0.5,
    type=lambda s: int(float(s) * 3600),
)
use_latest_code_field = BooleanInputField(
    key="use-latest-code",
    title="Use latest code for model run",
    hint="Should this task use most recent push (HEAD)? If no, the calibration commit will be used",
    type=bool,
    default="no",
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
    chains_field,
    runtime_field,
    use_latest_code_field,
    branch_field,
    spot_field,
]
input_step = InputStep(
    key="resume-calibration-settings", run_condition='build.env("SKIP_INPUT") == null', fields=fields
)
resume_calibration_step = CommandStep(
    key="resume-calibration",
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
pipeline = Pipeline(key="resume-calibration", steps=steps)
