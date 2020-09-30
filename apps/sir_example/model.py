from summer.model import StratifiedModel
from autumn.constants import Compartment, BirthApproach, Flow
from autumn.tool_kit.scenarios import get_model_times_from_inputs


def build_model(params: dict) -> StratifiedModel:
    """
    Build the master function to run a simple SIR model
    """
    # Define model compartments.
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.INFECTIOUS,
        Compartment.RECOVERED,
    ]

    # Define inter-compartmental flows.
    flows = [
        # Infection flow.
        {
            "type": Flow.INFECTION_FREQUENCY,
            "parameter": "contact_rate",
            "origin": Compartment.SUSCEPTIBLE,
            "to": Compartment.INFECTIOUS,
        },
        # Recovery flow.
        {
            "type": Flow.STANDARD,
            "parameter": "recovery_rate",
            "origin": Compartment.INFECTIOUS,
            "to": Compartment.RECOVERED,
        },
    ]
    # Define model times.
    integration_times = get_model_times_from_inputs(
        round(params["time.start"]), params["end_time"], params["time_step"]
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
        infectious_compartments=[Compartment.INFECTIOUS],
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
        "prevalence_infectious": calculate_prevalence_infectious,
        "prevalence_susceptible": calculate_prevalence_susceptible,
    }
    sir_model.add_function_derived_outputs(func_outputs)

    return sir_model


def calculate_prevalence_susceptible(time_idx, model, compartment_values, derived_outputs):
    """
    Calculate the total number of susceptible people at each time-step.
    """
    prevalence_susceiptible = 0
    for i, comp in enumerate(model.compartment_names):
        is_susceiptible = comp.has_name(Compartment.SUSCEPTIBLE)
        if is_susceiptible:
            prevalence_susceiptible += compartment_values[i]

    return prevalence_susceiptible


def calculate_prevalence_infectious(time_idx, model, compartment_values, derived_outputs):
    """
    Calculate the total number of infectious people at each time-step.
    """
    prevalence_infectious = 0
    for i, comp in enumerate(model.compartment_names):
        is_infectious = comp.has_name(Compartment.INFECTIOUS)
        if is_infectious:
            prevalence_infectious += compartment_values[i]

    return prevalence_infectious
