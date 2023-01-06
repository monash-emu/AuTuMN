from pathlib import Path

from summer2 import CompartmentalModel
from summer2.experimental.model_builder import ModelBuilder

from autumn.core.project import Params #, build_rel_path

from .parameters import Parameters

base_params = Params(
    str(Path(__file__).parent.resolve() / "params.yml"),
    validator=lambda d: Parameters(**d),
    validate=False,
)


def build_model(params: dict, ret_builder=False) -> CompartmentalModel:

    builder = ModelBuilder(params, Parameters)
    params = builder.params

    base_compartments = ["S", "I", "R"]
    time_params = params.time

    # Create the model object
    model = CompartmentalModel(
        times=(time_params['start'], time_params['end']),
        compartments=base_compartments,
        infectious_compartments=["I"],
        timestep=time_params['step'],
    )
   
    # Initial compartment sizes 
    init_pop = {
        "S": params.pop_size - params.infection_seed,
        "I": params.infection_seed
    }
    model.set_initial_population(init_pop)

    # Transmission flow
    model.add_infection_frequency_flow(
        name="infection",
        contact_rate=params.contact_rate,
        source="S",
        dest="I",
    )

    # Recovery flow
    model.add_transition_flow(
        name="recovery",
        fractional_rate=params.recovery_rate,
        source="I",
        dest="R",
    )

    # Track incidence
    model.request_output_for_flow(name="incidence", flow_name="infection")

    # Update builder with finalised model
    builder.set_model(model)     

    # Return model object (and builder if requested)
    if ret_builder:
        return model, builder
    else:
        return model