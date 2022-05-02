from autumn.remote.buildkite.buildkite import (
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
urunid_field = TextInputField(
    key="urunid",
    title="Candidate run id",
    hint="Which candidate run should be used? (format: chain_run)",
    default="mle",
    type=str,
)
commit_field = TextInputField(
    key="commit",
    title="Specify commit SHA",
    hint="Specify git commit, or leave as default to use calibration commit",
    type=str,
    default="use_original_commit",
)

fields = [run_id_field, urunid_field, commit_field]
input_step = InputStep(
    key="powerbi-processing-settings",
    run_condition='build.env("SKIP_INPUT") == null',
    fields=fields,
)
powerbi_step = CommandStep(
    key="run-powerbi",
    command="./scripts/buildkite.sh powerbi",
)
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
