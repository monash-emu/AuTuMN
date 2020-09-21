from remote.buildkite.buildkite import (
    Pipeline,
    CommandStep,
    InputStep,
    TextInputField,
    SelectInputField,
    BooleanInputField,
)
from .full import burn_in_field


def get_region_options():
    """Dynamically fetch region options from COVID app"""
    from apps import covid_19, tuberculosis

    covid_options = [
        {"label": n.replace("-", " ").title() + " (COVID)", "value": f"covid_19:{n}"}
        for n in covid_19.app.region_names
    ]
    tb_options = [
        {"label": n.replace("-", " ").title() + " (TB)", "value": f"tuberculosis:{n}"}
        for n in tuberculosis.app.region_names
    ]
    return covid_options + tb_options


model_field = SelectInputField(
    key="model-name",
    hint="Which model do you want to run?",
    title="Model Region",
    options=get_region_options,
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
trigger_field = BooleanInputField(
    key="trigger-downstream",
    title="Trigger full model run",
    hint="Should this task trigger a full model run when it is done?",
    default="yes",
    type=bool,
)
spot_field = BooleanInputField(
    key="spot-instance",
    title="Use spot instances",
    hint="Is 1/3 of the price but sometimes randomly fails.",
    default="yes",
    type=bool,
)

fields = [
    model_field,
    chains_field,
    branch_field,
    runtime_field,
    burn_in_field,
    trigger_field,
    spot_field,
]
input_step = InputStep(
    key="calibration-settings", run_condition='build.env("SKIP_INPUT") == null', fields=fields
)
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
pipeline = Pipeline(key="calibrate", steps=steps)
