from pathlib import Path

from summer2 import CompartmentalModel
from summer2.parameters import Parameter

from autumn.core.project import Params 

base_params = Params(
    str(Path(__file__).parent.resolve() / "params.yml")
)


def build_model(config: dict, ret_build=False) -> CompartmentalModel:

    base_compartments = ["S", "I", "R"]
    time_config = config['time']

    # Create the model object
    model = CompartmentalModel(
        times=(time_config['start'], time_config['end']),
        compartments=base_compartments,
        infectious_compartments=["I"],
        timestep=time_config['step'],
    )
   
    # Initial compartment sizes 
    init_pop = {
        "S": config['pop_size'] - config['infection_seed'],
        "I": config['infection_seed']
    }
    model.set_initial_population(init_pop)

    # Transmission flow
    model.add_infection_frequency_flow(
        name="infection",
        contact_rate=Parameter('contact_rate'),
        source="S",
        dest="I",
    )

    # Recovery flow
    model.add_transition_flow(
        name="recovery",
        fractional_rate=config['recovery_rate'],
        source="I",
        dest="R",
    )

    # Track incidence
    model.request_output_for_flow(name="incidence", flow_name="infection")
   
    return model