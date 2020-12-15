"""
A basic example showing a minimal model.

Create a SIR compartmental model for some fictional disease.
It has three compartments (**S**, **I**, **R**) and **I** is considered "infectious".
The model will run from **1990** to **2025**.
There will be a starting population of **1000** people, with **10** of them infectious.
"""
from summer2 import CompartmentalModel
from summer2.examples.utils import plot_timeseries


def build_model() -> CompartmentalModel:
    """
    Returns the SIR model, ready to run.
    """
    model = CompartmentalModel(
        times=[1990, 2025],
        compartments=["S", "I", "R"],
        infectious_compartments=["I"],
        timestep=0.1,
    )
    # Add 1000 people to the compartments with 100 infectious.
    model.set_initial_population(distribution={"S": 990, "I": 10})

    # Add flows between the compartments.
    # Susceptible people get infected.
    model.add_infection_frequency_flow(name="infection", contact_rate=2, source="S", dest="I")
    # Infectious people take 2 years, on average, to recover.
    model.add_sojourn_flow(name="recovery", sojourn_time=2, source="I", dest="R")
    # Add an infection-specific death flow to the I compartment.
    model.add_death_flow(name="infection_death", death_rate=0.05, source="I")

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
