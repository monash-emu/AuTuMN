from copy import deepcopy

from summer.model import StratifiedModel
from autumn.constants import Compartment, BirthApproach, Flow
from autumn.tool_kit.scenarios import get_model_times_from_inputs

from . import preprocess, outputs
from .validate import validate_params


def build_model(params: dict) -> StratifiedModel:
    """
    Build the master function to run a simple SIR model
    """
    # Define model compartments.
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EARLY_LATENT,
        Compartment.LATE_LATENT,
        Compartment.INFECTIOUS,
        Compartment.RECOVERED,
    ]
    infectious_comps = [
        Compartment.INFECTIOUS,
    ]

    # Define inter-compartmental flows.
    flows = deepcopy(preprocess.flows.DEFAULT_FLOWS)

    # Define model times.
    integration_times = get_model_times_from_inputs(
        round(params["start_time"]), params["end_time"], params["time_step"]
    )

    # Define initial conditions - 1 infectious person.
    init_conditions = {Compartment.INFECTIOUS: 1}

    # Create the model.
    sir_model = StratifiedModel(
        times=integration_times,
        compartment_names=compartments,
        initial_conditions=init_conditions,
        parameters=params,
        requested_flows=flows,
        infectious_compartments=infectious_comps,
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment=Compartment.SUSCEPTIBLE,
        starting_population=1000000,
    )

    # Register derived output functions
    # These functions calculate 'derived' outputs of interest, based on the
    # model's compartment values. These are calculated after the model is run.
    # This is not strictly necessary in this simple model, but becomes useful
    # when the compartments are stratified.
    func_outputs = {
        "prevalence_infectious": outputs.calculate_prevalence_infectious,
        "prevalence_susceptible": outputs.calculate_prevalence_susceptible,
    }
    sir_model.add_function_derived_outputs(func_outputs)

    return sir_model

