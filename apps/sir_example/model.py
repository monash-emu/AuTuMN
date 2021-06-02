from summer import CompartmentalModel


def build_model(params: dict) -> CompartmentalModel:
    """
    Build the master function to run a simple SIR model
    """
    # Define model compartments.
    compartments = ["S", "I", "R"]

    time = params["time"]
    model = CompartmentalModel(
        times=[time["start"], time["end"]],
        compartments=compartments,
        infectious_compartments=["I"],
        timestep=time["step"],
    )
    model.set_initial_population(
        {
            "I": 1,
            "S": 999999,
        }
    )
    # Add flows
    model.add_infection_frequency_flow("infection", params["contact_rate"], "S", "I")
    model.add_transition_flow("recovery", params["recovery_rate"], "I", "R")

    # Request derived outputs
    model.request_output_for_compartments("prevalence_susceptible", compartments=["S"])
    model.request_output_for_compartments("prevalence_infectious", compartments=["I"])
    return model
