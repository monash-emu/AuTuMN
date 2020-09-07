from remote.buildkite.buildkite import (
    Pipeline,
    CommandStep,
    InputStep,
    TextInputField,
    BooleanInputField,
)


commit_field = TextInputField(
    key="run-commit",
    title="Git commit hash for Victorian calibrations",
    hint="Enter the Git commit hash for the model run you want to ingest",
    default="",
    type=str,
)
spot_field = BooleanInputField(
    key="spot-instance",
    title="Use spot instances",
    hint="Is 1/3 of the price but sometimes randomly fails.",
    default="yes",
    type=bool,
)
fields = [commit_field, spot_field]
input_step = InputStep(key="dhhs-settings", run_condition=None, fields=fields)
dhhs_step = CommandStep(key="run-dhhs", command="./scripts/buildkite.sh dhhs",)
website_step = CommandStep(
    key="update-website",
    command="./scripts/website/deploy.sh",
    depends_on=dhhs_step,
    allow_dependency_failure=True,
)
steps = [
    input_step,
    dhhs_step,
    website_step,
]
pipeline = Pipeline(key="dhhs", steps=steps)
