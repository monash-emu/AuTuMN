"""
Basic test to ensure that CompartmentalModel produces reasonable outputs with no stratifications.
Results modelled in Excel using the Euler method using a 1e-6 timestep. Some results have been tweaked slightly.
"""
import numpy as np
from numpy.testing import assert_allclose

from summer2 import CompartmentalModel, AgeStratification


def test_model__with_static_dynamics__expect_no_change():
    """
    Ensure that a model with two compartments and no internal dynamics results in no change.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.run()
    # Expect that no one has moved from sucsceptible to infections at any point in time
    expected_outputs = np.array(
        [
            [990, 10, 0],  # Initial conditions
            [990, 10, 0],
            [990, 10, 0],
            [990, 10, 0],
            [990, 10, 0],
            [990, 10, 0],
        ]
    )
    assert_allclose(model.outputs, expected_outputs, atol=0.1, verbose=True)


def test_model__with_birth_rate__expect_pop_increase():
    """
    Ensure that a model with two compartments and only birth rate dynamics results in more people.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 100, "I": 100})
    # Add some babies at ~2 babies / 100 / year.
    model.add_crude_birth_flow("births", 0.02, "S")
    model.run()
    # Expect that we have more people in the population per year
    expected_outputs = np.array(
        [
            [100.0, 100, 0],  # Initial conditions
            [104.0, 100, 0],
            [108.2, 100, 0],
            [112.4, 100, 0],
            [116.7, 100, 0],
            [121.0, 100, 0],
        ]
    )
    assert_allclose(model.outputs, expected_outputs, atol=0.1, verbose=True)


def test_model__with_death_rate__expect_pop_decrease():
    """
    Ensure that a model with two compartments and only death rate dynamics results in fewer people.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 100, "I": 100})
    # Add some dying at ~2 people / 100 / year.
    model.add_universal_death_flows("deaths", 0.02)
    model.run()
    # Expect that we have fewer people in the population per year
    expected_outputs = np.array(
        [
            [100.0, 100.0, 0],  # Initial conditions
            [98.0, 98.0, 0],
            [96.1, 96.1, 0],
            [94.2, 94.2, 0],
            [92.3, 92.3, 0],
            [90.5, 90.5, 0],
        ]
    )
    assert_allclose(model.outputs, expected_outputs, atol=0.1, verbose=True)


def test_model__with_birth_and_death_rate__expect_pop_static_overall():

    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 100, "I": 100})
    # Add some babies at ~2 babies / 100 / year.
    model.add_crude_birth_flow("births", 0.02, "S")
    # Add some dying at ~2 people / 100 / year.
    model.add_universal_death_flows("deaths", 0.02)
    model.run()
    expected_outputs = np.array(
        [
            [100.0, 100.0, 0],  # Initial conditions
            [102.0, 98.0, 0],
            [104.0, 96.0, 0],
            [105.8, 94.2, 0],  # Tweaked.
            [107.7, 92.3, 0],  # Tweaked.
            [109.5, 90.5, 0],  # Tweaked.
        ]
    )
    assert_allclose(model.outputs, expected_outputs, atol=0.1, verbose=True)


def test_model__with_birth_and_death_rate_replace_deaths__expect_pop_static_overall():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 100, "I": 100})
    model.add_replacement_birth_flow("births", "S")
    # Add some dying at ~2 people / 100 / year.
    model.add_universal_death_flows("deaths", 0.02)
    model.run()
    expected_outputs = np.array(
        [
            [100.0, 100.0, 0],  # Initial conditions
            [102.0, 98.0, 0],
            [104.0, 96.0, 0],
            [105.8, 94.2, 0],  # Tweaked.
            [107.7, 92.3, 0],  # Tweaked.
            [109.5, 90.5, 0],  # Tweaked.
        ]
    )
    assert_allclose(model.outputs, expected_outputs, atol=0.1, verbose=True)


def test_model__with_higher_birth_than_and_death_rate__expect_pop_increase():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 100, "I": 100})
    # Add some babies at ~10 babies / 100 / year.
    model.add_crude_birth_flow("births", 0.1, "S")
    # Add some dying at ~2 people / 100 / year.
    model.add_universal_death_flows("deaths", 0.02)
    model.run()
    expected_outputs = np.array(
        [
            [100.0, 100.0, 0],  # Initial conditions
            [118.6, 98.0, 0],  # Tweaked ~0.1
            [138.6, 96.1, 0],  # Tweaked ~0.4
            [160.1, 94.2, 0],  # Tweaked ~0.9
            [183.1, 92.3, 0],  # Tweaked ~1.7
            [207.9, 90.5, 0],  # Tweaked ~2.7
        ]
    )
    assert_allclose(model.outputs, expected_outputs, atol=0.1, verbose=True)


def test_model__with_recovery_rate__expect_all_recover():
    """
    Ensure that a model with three compartments and only recovery dynamics
    results in (almost) everybody recovering.
    """
    # Set up a model with 100 people, all infectious.
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"I": 100})
    # Add recovery dynamics.
    model.add_fractional_flow("recovery", 1, "I", "R")
    model.run()
    # Expect that almost everyone recovers
    expected_outputs = np.array(
        [
            [0.00, 100.00, 0.00],  # Initial conditions
            [0.00, 36.79, 63.21],
            [0.00, 13.53, 86.47],
            [0.00, 4.98, 95.02],
            [0.00, 1.83, 98.17],
            [0.00, 0.67, 99.33],
        ]
    )

    assert_allclose(model.outputs, expected_outputs, atol=0.1, verbose=True)


def test_model__with_infect_death_rate__expect_infected_pop_decrease():
    """
    Ensure that a model with two compartments and only infected death rate dynamics
    results in fewer infected people, but no change to susceptible pop.
    """
    # Set up a model with 100 people, all infectious.
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 50, "I": 50})
    # Add some dying at ~2 people / 100 / year.
    model.add_death_flow("infect_death", 0.02, "I")
    model.run()
    expected_outputs = np.array(
        [
            [50.00, 50.00, 0],  # Initial conditions
            [50.00, 49.01, 0],
            [50.00, 48.04, 0],
            [50.00, 47.09, 0],
            [50.00, 46.16, 0],
            [50.00, 45.24, 0],
        ]
    )
    assert_allclose(model.outputs, expected_outputs, atol=0.1, verbose=True)


def test_model__with_no_infected__expect_no_change():
    """
    Ensure that if no one has the disease, then no one gets the disease in the future.
    """
    # Set up a model with 100 people, all susceptible, transmission highly likely, but no one is infected.
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 100, "I": 0})
    model.add_infection_frequency_flow("infection", 10, "S", "I")
    model.run()
    # Expect that no one has moved from sucsceptible to infections at any point in time
    expected_outputs = np.array(
        [
            [100.0, 0.0, 0],  # Initial conditions
            [100.0, 0.0, 0],
            [100.0, 0.0, 0],
            [100.0, 0.0, 0],
            [100.0, 0.0, 0],
            [100.0, 0.0, 0],
        ]
    )
    assert_allclose(model.outputs, expected_outputs, atol=0.1, verbose=True)


def test_model__with_infection_frequency__expect_all_infected():
    """
    Ensure that a model with two compartments and one-way internal dynamics results in all infected.
    """
    # Set up a model with 100 people, all susceptible execept 1 infected.
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 99, "I": 1})
    model.add_infection_frequency_flow("infection", 3, "S", "I")
    model.run()
    # Expect that everyone gets infected eventually.
    expected_outputs = np.array(
        [
            [99.00, 1.00, 0],  # Initial conditions
            [83.13, 16.87, 0],
            [19.70, 80.30, 0],
            [1.21, 98.79, 0],
            [0.06, 99.94, 0],
            [0.00, 100.00, 0],
        ]
    )
    assert_allclose(model.outputs, expected_outputs, atol=0.1, verbose=True)


def test_model__with_infection_density__expect_all_infected():
    """
    Ensure that a model with two compartments and one-way internal dynamics results in all infected.
    """
    # Set up a model with 100 people, all susceptible execept 1 infected.
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 99, "I": 1})
    model.add_infection_density_flow("infection", 0.03, "S", "I")
    model.run()
    # Expect that everyone gets infected eventually.
    expected_outputs = np.array(
        [
            [99.00, 1.00, 0],  # Initial conditions
            [83.13, 16.87, 0],
            [19.70, 80.30, 0],
            [1.21, 98.79, 0],
            [0.06, 99.94, 0],
            [0.00, 100.00, 0],
        ]
    )
    assert_allclose(model.outputs, expected_outputs, atol=0.1, verbose=True)


def test_model__with_complex_dynamics__expect_correct_outputs():
    """
    Ensure that a model with the "full suite" of TB dynamics produces correct results:
        - 5 compartments
        - birth rate +  universal death rate
        - standard inter-compartment flows
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "EL", "LL", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 900, "I": 100})
    model.add_crude_birth_flow("births", 0.02, "S")
    model.add_universal_death_flows("universal_deaths", 0.02)
    model.add_infection_frequency_flow("infection", 14, "S", "EL")
    model.add_infection_frequency_flow("reinfection", 14, "R", "EL")
    model.add_infection_frequency_flow("regression", 3, "LL", "EL")
    model.add_fractional_flow("early_progression", 2, "EL", "I")
    model.add_fractional_flow("stabilisation", 3, "EL", "LL")
    model.add_fractional_flow("early_progression", 1, "LL", "I")
    model.add_death_flow("infect_death", 0.4, "I")
    model.add_fractional_flow("recovery", 0.2, "I", "R")
    model.add_fractional_flow("case_detection", 1, "I", "R")
    model.run()
    # Expect that the results are consistent, nothing crazy happens.
    # These results were not independently calculated, so this is more of an "acceptance test".
    expected_outputs = np.array(
        [
            [900.0, 0.0, 0.0, 100.0, 0.0],
            [66.1, 203.8, 274.2, 307.2, 75.3],
            [2.9, 150.9, 220.5, 345.3, 69.4],
            [2.2, 127.3, 175.6, 297.0, 58.1],
            [1.8, 106.4, 145.6, 248.8, 48.5],
            [1.5, 88.8, 121.5, 207.8, 40.5],
        ]
    )
    assert_allclose(model.outputs, expected_outputs, atol=0.2, verbose=True)


def test_strat_model__with_age__expect_ageing():
    """
    Ensure that a module with age stratification produces ageing flows,
    and the correct output.
    """
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": 1000, "I": 0})
    strat = AgeStratification("age", [0, 5, 15, 60], ["S", "I"])
    model.stratify_with(strat)
    # Run the model for 5 years.
    model.run()

    # Expect everyone to generally get older, but no one should die or get sick
    expected_arr = np.array(
        [
            [250.0, 250.0, 250.0, 250.0, 0.0, 0.0, 0.0, 0.0],
            [204.7, 269.3, 270.3, 255.8, 0.0, 0.0, 0.0, 0.0],
            [167.6, 278.9, 291.5, 262.0, 0.0, 0.0, 0.0, 0.0],
            [137.2, 281.2, 312.8, 268.7, 0.0, 0.0, 0.0, 0.0],
            [112.3, 278.1, 333.7, 275.9, 0.0, 0.0, 0.0, 0.0],
            [92.0, 270.9, 353.5, 283.6, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)


def test_strat_model__with_age_and_starting_proportion__expect_ageing():
    """
    Ensure that a module with age stratification and starting proporptions
    produces ageing flows, and the correct output.
    """
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": 1000, "I": 0})
    strat = AgeStratification("age", [0, 5, 15, 60], ["S", "I"])
    strat.set_population_split({"0": 0.8, "5": 0.1, "15": 0.1, "60": 0})
    model.stratify_with(strat)
    # Run the model for 5 years.
    model.run()

    # Expect everyone to generally get older, but no one should die or get sick.
    # Expect initial distribution of ages to be set according to "requested_proportions".
    expected_arr = np.array(
        [
            [800.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [655.0, 228.3, 114.4, 2.4, 0.0, 0.0, 0.0, 0.0],
            [536.2, 319.4, 139.2, 5.2, 0.0, 0.0, 0.0, 0.0],
            [439.0, 381.3, 171.1, 8.6, 0.0, 0.0, 0.0, 0.0],
            [359.5, 420.6, 207.1, 12.8, 0.0, 0.0, 0.0, 0.0],
            [294.3, 442.5, 245.4, 17.8, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)
