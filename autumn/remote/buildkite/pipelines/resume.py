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
runtime_field = TextInputField(
    key="mcmc-runtime",
    title="Runtime",
    hint="How many hours should the model run for?",
    default=0.5,
    type=lambda s: int(float(s) * 3600),
)
fields = [
    run_id_field,
    chains_field,
    runtime_field,
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
