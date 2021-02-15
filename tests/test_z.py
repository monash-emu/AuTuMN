from summer2 import CompartmentalModel


def test_z():
    model = CompartmentalModel(
        times=[0, 10], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.add_crude_birth_flow("births", 0.02, "S")
    model.add_universal_death_flows("deaths", 0.01)
    model.add_infection_frequency_flow("infection", 2, "S", "I")
    model.add_death_flow("infect_death", 0.4, "I")
    model.add_fractional_flow("recovery", 0.2, "I", "R")
    model.run_stochastic()
    import numpy as np

    print("\n", model.outputs)