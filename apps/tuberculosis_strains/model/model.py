from copy import deepcopy
from summer.model import StratifiedModel
from autumn.constants import Compartment, BirthApproach, Flow
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from autumn.tool_kit.demography import set_model_time_variant_birth_rate

from apps.tuberculosis_strains.model import preprocess, outputs



def build_model(params: dict) -> StratifiedModel:
    """
    Build the master function to run a tuberculosis model
    """

    # Define model times.
    time = params["time"]
    integration_times = get_model_times_from_inputs(
        round(time["start"]), time["end"], time["step"], time["critical_ranges"]
    )

    # Define model compartments.
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EARLY_LATENT,
        Compartment.LATE_LATENT,
        Compartment.INFECTIOUS,
        Compartment.ON_TREATMENT,
        Compartment.RECOVERED,
    ]
    infectious_comps = [
        Compartment.INFECTIOUS,
        Compartment.ON_TREATMENT,
    ]

    # Define initial conditions - 1 infectious person.
    init_conditions = {
        Compartment.INFECTIOUS: params["initial_infectious_population"],
    }


    # Generate derived parameters
    params = preprocess.derived_params.get_derived_params(params)


    # Define inter-compartmental flows.
    flows = deepcopy(preprocess.flows.DEFAULT_FLOWS)

    # Create the model.
    tb_model = StratifiedModel(
        times=integration_times,
        compartment_names=compartments,
        initial_conditions=init_conditions,
        parameters=params,
        requested_flows=flows,
        infectious_compartments=infectious_comps,
        birth_approach=BirthApproach.ADD_CRUDE,
        entry_compartment=Compartment.SUSCEPTIBLE,
        starting_population=int(params["start_population_size"]),
    )

    # Register derived output functions, which are calculations based on the model's compartment values or flows.
    # These are calculated after the model is run.
    outputs.get_all_derived_output_functions(
        params["calculated_outputs"], params["outputs_stratification"], tb_model
    )

    return tb_model
