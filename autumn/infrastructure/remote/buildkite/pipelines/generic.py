from autumn.infrastructure.remote.buildkite.buildkite import (
    BooleanInputField,
    CommandStep,
    InputStep,
    Pipeline,
    TextInputField,
)

model_field = TextInputField(
    key="model-name",
    hint="Which project do you want to run?",
    title="Project (model:region)",
    type=str,
)
task_field = TextInputField(
    key="task-key",
    hint="Which task do you want to run?",
    title="Task key",
    type=str,
)
cores_field = TextInputField(
    key="num-cores",
    title="Number of CPU cores",
    hint="How many CPU cores do you require?",
    default=1,
    type=int,
)
commit_field = TextInputField(
    key="commit",
    title="Model git commit SHA",
    hint="Which git commit do you want to use to run the model?",
    type=str,
)
kwargs_field = TextInputField(
    key="kwargs",
    title="Additional task arguments",
    hint="Enter a JSON string for the kwargs dict",
    type=str,
    required=False
)


fields = [
    commit_field,
    task_field,
    model_field,    
    kwargs_field,
    cores_field,
]

input_step = InputStep(
    key="task-settings", run_condition='build.env("SKIP_INPUT") == null', fields=fields
)
run_step = CommandStep(
    key="run-generic",
    command="./scripts/buildkite.sh generic",
)

steps = [
    input_step,
    run_step,
]
pipeline = Pipeline(key="generic", steps=steps)
