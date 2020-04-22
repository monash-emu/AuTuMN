from copy import deepcopy

from summer.model.strat_model import StratifiedModel
from autumn.tool_kit import schema_builder as sb

from .requested_outputs import RequestedOutput

# FIXME - This data representation can be improved... somehow.
validate_post_process_config = sb.build_validator(
    # Outputs to be generated
    # Eg. ["prevXinfectiousXamongXage_10Xstrain_sensitive", "distribution_of_strataXstrain"]
    requested_outputs=sb.List(str),
    # Constants to multiply the generated outputs by.
    # Eg. {"prevXinfectiousXamongXage_10Xstrain_sensitive": 1.0e5}
    multipliers=sb.DictGeneric(str, float),
    # List of compartment, stratification pairs used to generate some more requested outputs
    # Eg. [["infectious", "location"], ["latent", "location"]]
    collated_combos=sb.List(sb.List(str)),
)


def post_process(model: StratifiedModel, post_process_config: dict, add_defaults=True):
    """
    Derive generated outputs from a model after the model has run.
    Returns a dict of generated outputs.
    """
    validate_post_process_config(post_process_config)

    # Read config.
    config = deepcopy(post_process_config)
    requested_outputs = config["requested_outputs"]
    multipliers = config.get("multipliers", {})
    collated_combos = config.get("collated_combos", [])

    # Automatically add some basic generated outputs.
    if add_defaults:
        for stratum in model.all_stratifications.keys():
            requested_outputs.append(f"distribution_of_strataX{stratum}")

    # Generate more outputs prevalence outputs using 'collated_combos'
    for compartment, stratification in collated_combos:
        strata = model.all_stratifications[stratification]
        for stratum in strata:
            requested_output = f"prevX{compartment}XamongX{stratification}_{stratum}"
            requested_outputs.append(requested_output)

    # Convert any string-based requested outputs into a requested output class instance.
    requested_outputs = [RequestedOutput.from_str(o) for o in requested_outputs]
    requested_outputs = [o for o in requested_outputs if o.is_valid(model)]

    # Check that the model has actually been run
    if model.outputs is None:
        raise ValueError("The model needs to be run before post-processing")

    # Calculated generated outputs
    generated_outputs = {}
    for output in requested_outputs:
        output_str = output.to_str()
        time_idxs = range(len(model.times))
        generated_outputs[output_str] = output.calculate_output(model, multipliers, time_idxs)

    return generated_outputs
