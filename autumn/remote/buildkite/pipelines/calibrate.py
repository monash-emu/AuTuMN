from autumn.remote.buildkite.buildkite import (
    BooleanInputField,
    CommandStep,
    InputStep,
    Pipeline,
    SelectInputField,
    TextInputField,
)

from .full import burn_in_field, sample_size_field
from autumn.tools.registry import get_registered_model_names, get_registered_project_names


def get_region_options():
    """Dynamically fetch region options from COVID app"""
    options = []
    for model_name in get_registered_model_names():
        for region_name in get_registered_project_names(model_name):
            o = {
                "label": region_name.replace("-", " ").title() + f" ({model_name})",
                "value": f"{model_name}:{region_name}",
            }
            options.append(o)

    return options


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
    default=8,
    type=int,
)
commit_field = TextInputField(
    key="commit",
    title="Model git commit SHA",
    hint="Which git commit do you want to use to run the model?",
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

fields = [
    model_field,
    chains_field,
    commit_field,
    runtime_field,
    burn_in_field,
    sample_size_field,
    trigger_field,
]
input_step = InputStep(
    key="calibration-settings", run_condition='build.env("SKIP_INPUT") == null', fields=fields
)
calibrate_step = CommandStep(
    key="run-calibration",
    command="./scripts/buildkite.sh calibrate",
)
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
