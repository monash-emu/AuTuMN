from remote.buildkite.buildkite import (
    Pipeline,
    CommandStep,
    InputStep,
    TextInputField,
    SelectInputField,
)


input_step
calibrate_step = CommandStep(key="run-calibration", command="./scripts/buildkite.sh calibrate",)
website_step = CommandStep(
    key="update-website",
    command="./scripts/website/deploy.sh",
    depends_on=calibrate_step,
    allow_dependency_failure=True,
)


steps = [
    input_step,
    calibrate_step,
    website_step,
]

pipeline = Pipeline(path="scripts/buildkite/pipelines/calibrate.yml", steps=steps)
