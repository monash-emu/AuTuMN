"""
A slightly more complicated version of the SIR model, with richer outputs.

Create a SIR compartmental model for some fictional disease.
It has four compartments (**S**, **E**, **I**, **R**) and **I** is considered "infectious".
The model will run from **1990** to **2025**.
There will be a starting population of **1000** people, with **10** of them infectious.
"""
from summer2 import CompartmentalModel
from summer2.examples.utils import plot_timeseries


def build_model() -> CompartmentalModel:
    """
    Returns the SEIR model, ready to run.
    """
    model = CompartmentalModel(
        times=[1990, 2025],
        compartments=["S", "E", "I", "R"],
        infectious_compartments=["I"],
        timestep=0.1,
    )
    # Add 1000 people to the compartments with 100 infectious.
    model.set_initial_population(distribution={"S": 990, "I": 10})

    # Add flows between the compartments.
    # People are born susceptible.
    model.add_crude_birth_flow(name="crude_birth", birth_rate=0.03, dest="S")
    # Susceptible people get infected and become exposed, although they are not yet infectious.
    model.add_infection_frequency_flow(name="infection", contact_rate=2, source="S", dest="E")
    # Exposed people take 4 years, on average, to become infectious.
    model.add_sojourn_flow(name="progression", sojourn_time=2, source="E", dest="I")
    # Infectious people take 2 years, on average, to recover.
    model.add_sojourn_flow(name="recovery", sojourn_time=2, source="I", dest="R")
    # Add a universal death flow, which is automatically applied to all compartments.
    model.add_universal_death_flows(base_name="universal_death", death_rate=0.02)
    # Add an infection-specific death flow to the I compartment.
    model.add_death_flow(name="infection_death", death_rate=0.05, source="I")

    # Add derived output requests so that we can track flow rates and compartment sizes over time.
    model.request_output_for_flow(name="incidence", flow_name="infection")
    model.request_output_for_flow(name="progression", flow_name="progression")
    model.request_output_for_flow(name="recovery", flow_name="recovery")
    model.request_output_for_flow(name="infection_death", flow_name="infection_death")
    model.request_cumulative_output(name="infection_death_cum", source="infection_death")
    model.request_cumulative_output(name="recovered_cum", source="recovery")
    model.request_cumulative_output(name="infected_cum", source="incidence")
    model.request_output_for_compartments(
        name="count_infectious", compartments=["I"], save_results=False
    )
    model.request_output_for_compartments(
        name="count_not_infectious", compartments=["S", "E", "I"], save_results=False
    )
    model.request_aggregate_output(
        name="total_population",
        sources=["count_infectious", "count_not_infectious"],
        save_results=False,
    )
    model.request_function_output(
        name="prev_infectious",
        sources=["count_infectious", "total_population"],
        func=lambda count, total: count / total,
    )
    model.request_function_output(
        name="prev_not_infectious",
        sources=["count_not_infectious", "total_population"],
        func=lambda count, total: count / total,
    )
    return model


def plot_outputs(model: CompartmentalModel):
    """
    Run the model and plot the results.
    """
    model.run()
    plot_timeseries(
        title="Compartment sizes",
        times=model.times,
        values={str(c): v for c, v in zip(model.compartments, model.outputs.T)},
    )
    plot_timeseries(
        title="Flows",
        times=model.times,
        values={
            "Incidence": model.derived_outputs["incidence"],
            "Progression": model.derived_outputs["progression"],
            "Recovery": model.derived_outputs["recovery"],
            "Infection Deaths": model.derived_outputs["infection_death"],
        },
    )
    plot_timeseries(
        title="Cumulative flows",
        times=model.times,
        values={
            "Infection Deaths": model.derived_outputs["infection_death_cum"],
            "Recovery": model.derived_outputs["recovered_cum"],
            "Infections": model.derived_outputs["infected_cum"],
        },
    )
    plot_timeseries(
        title="Prevalence of disease",
        times=model.times,
        values={
            "Prevalence infected": model.derived_outputs["prev_infectious"],
            "Prevalence not infected": model.derived_outputs["prev_not_infectious"],
        },
    )
