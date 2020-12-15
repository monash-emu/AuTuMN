from summer2 import CompartmentalModel
from autumn.constants import Compartment


def build_model(params: dict) -> CompartmentalModel:
    """
    Build the master function to run a simple SIR model
    """
    # Define model compartments.
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.INFECTIOUS,
        Compartment.RECOVERED,
    ]

    time = params["time"]
    model = CompartmentalModel(
        times=[time["start"], time["end"]],
        compartments=compartments,
        infectious_compartments=[Compartment.INFECTIOUS],
        timestep=time["step"],
    )
    model.set_initial_population(
        {
            Compartment.INFECTIOUS: 1,
            Compartment.SUSCEPTIBLE: 999999,
        }
    )
    # Add flows
    model.add_infection_frequency_flow(
        "infection", params["contact_rate"], Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS
    )
    model.add_fractional_flow(
        "recovery", params["recovery_rate"], Compartment.INFECTIOUS, Compartment.RECOVERED
    )

    # Request derived outputs
    model.request_output_for_compartments(
        "prevalence_susceptible", compartments=[Compartment.SUSCEPTIBLE]
    )
    model.request_output_for_compartments(
        "prevalence_infectious", compartments=[Compartment.INFECTIOUS]
    )
    return model
