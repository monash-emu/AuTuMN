"""
Tests for the 'strains' feature of SUMMER.

Strains allows for multiple concurrent infections, which have different properties.

- all infected compartments are stratified into strains (all, not just diseased or infectious, etc)
- assume that a person can only have one strain (simplifying assumption)
- strains can have different infectiousness, mortality rates, etc (set via flow adjustment)
- strains can progress from one to another (inter-strain flows)
- each strain has a different force of infection calculation
- any strain stratification you must be applied to all infected compartments

Force of infection:

- we have multiple infectious populations (one for each strain)
- people infected by a particular strain get that strain
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from summer2 import (
    CompartmentalModel,
    Stratification,
    StrainStratification,
    Compartment as C,
    adjust,
)


def test_model__with_two_symmetric_stratifications():
    """
    Adding two strata with the same properties should yield the exact same infection dynamics and outputs as having no strata at all.
    This does not test strains directly, but if this doesn't work then further testing is pointless.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 900, "I": 100})
    model.add_infection_frequency_flow("infection", 0.2, "S", "I")
    model.add_sojourn_flow("recovery", 10, "I", "R")

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    model._prepare_time_step(0, model.initial_population)

    # Check infectiousness multipliers
    susceptible = model.compartments[0]
    infectious = model.compartments[1]
    assert model._get_infection_density_multiplier(susceptible, infectious) == 100.0
    assert model._get_infection_frequency_multiplier(susceptible, infectious) == 0.1
    model.run()

    # Create a stratified model where the two non-strain strata are symmetric
    stratified_model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    stratified_model.set_initial_population(distribution={"S": 900, "I": 100})
    stratified_model.add_infection_frequency_flow("infection", 0.2, "S", "I")
    stratified_model.add_sojourn_flow("recovery", 10, "I", "R")
    strat = Stratification("clinical", ["home", "hospital"], ["I"])
    stratified_model.stratify_with(strat)
    stratified_model.run()

    # Ensure stratified model has the same results as the unstratified model.
    merged_outputs = np.zeros_like(model.outputs)
    merged_outputs[:, 0] = stratified_model.outputs[:, 0]
    merged_outputs[:, 1] = stratified_model.outputs[:, 1] + stratified_model.outputs[:, 2]
    merged_outputs[:, 2] = stratified_model.outputs[:, 3]
    assert_allclose(merged_outputs, model.outputs, atol=0.01, rtol=0.01, verbose=True)


def test_strains__with_two_symmetric_strains():
    """
    Adding two strains with the same properties should yield the same infection dynamics and outputs as having no strains at all.
    We expect the force of infection for each strain to be 1/2 of the unstratified model,
    but the stratification process will not apply the usual conservation fraction to the fan out flows.
    """
    # Create an unstratified model
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 900, "I": 100})
    model.add_infection_frequency_flow("infection", 0.2, "S", "I")
    model.add_sojourn_flow("recovery", 10, "I", "R")

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    model._prepare_time_step(0, model.initial_population)
    # Check infectiousness multipliers
    susceptible = model.compartments[0]
    infectious = model.compartments[1]
    assert model._get_infection_density_multiplier(susceptible, infectious) == 100.0
    assert model._get_infection_frequency_multiplier(susceptible, infectious) == 0.1
    model.run()

    # Create a stratified model where the two strain strata are symmetric
    strain_model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    strain_model.set_initial_population(distribution={"S": 900, "I": 100})
    strain_model.add_infection_frequency_flow("infection", 0.2, "S", "I")
    strain_model.add_sojourn_flow("recovery", 10, "I", "R")
    strat = StrainStratification("strain", ["a", "b"], ["I"])
    strain_model.stratify_with(strat)
    strain_model.run()

    # Ensure stratified model has the same results as the unstratified model.
    merged_outputs = np.zeros_like(model.outputs)
    merged_outputs[:, 0] = strain_model.outputs[:, 0]
    merged_outputs[:, 1] = strain_model.outputs[:, 1] + strain_model.outputs[:, 2]
    merged_outputs[:, 2] = strain_model.outputs[:, 3]
    assert_allclose(merged_outputs, model.outputs, atol=0.01, rtol=0.01, verbose=True)


def test_strain__with_infectious_multipliers():
    """
    Test infectious multiplier and flow rate calculations for
    3 strains which have different infectiousness levels.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 900, "I": 100})
    contact_rate = 0.2
    model.add_infection_frequency_flow("infection", contact_rate, "S", "I")
    strat = StrainStratification("strain", ["a", "b", "c"], ["I"])
    strat.set_population_split(
        {
            "a": 0.7,  # 70 people
            "b": 0.2,  # 20 people
            "c": 0.1,  # 10 people
        }
    )
    strat.add_infectiousness_adjustments(
        "I",
        {
            "a": adjust.Multiply(0.5),  # 0.5x as infectious
            "b": adjust.Multiply(3),  # 3x as infectious
            "c": adjust.Multiply(2),  # 2x as infectious
        },
    )
    model.stratify_with(strat)

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    assert_array_equal(model._compartment_infectiousness["a"], np.array([0, 0.5, 0, 0, 0]))
    assert_array_equal(model._compartment_infectiousness["b"], np.array([0, 0, 3, 0, 0]))
    assert_array_equal(model._compartment_infectiousness["c"], np.array([0, 0, 0, 2, 0]))
    assert model._category_lookup == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    assert_array_equal(model._category_matrix, np.array([[1, 1, 1, 1, 1]]))

    # Do pre-iteration force of infection calcs
    model._prepare_time_step(0, model.initial_population)
    assert_array_equal(model._category_populations, np.array([[1000]]))
    assert_array_equal(model._infection_density["a"], np.array([[70 * 0.5]]))
    assert_array_equal(model._infection_density["b"], np.array([[20 * 3]]))
    assert_array_equal(model._infection_density["c"], np.array([[10 * 2]]))
    assert_array_equal(model._infection_frequency["a"], np.array([[70 * 0.5 / 1000]]))
    assert_array_equal(model._infection_frequency["b"], np.array([[20 * 3 / 1000]]))
    assert_array_equal(model._infection_frequency["c"], np.array([[10 * 2 / 1000]]))

    # Get multipliers
    susceptible = model.compartments[0]
    infectious_a = model.compartments[1]
    infectious_b = model.compartments[2]
    infectious_c = model.compartments[3]
    assert model._get_infection_density_multiplier(susceptible, infectious_a) == 70 * 0.5
    assert model._get_infection_density_multiplier(susceptible, infectious_b) == 20 * 3
    assert model._get_infection_density_multiplier(susceptible, infectious_c) == 10 * 2
    assert model._get_infection_frequency_multiplier(susceptible, infectious_a) == 70 * 0.5 / 1000
    assert model._get_infection_frequency_multiplier(susceptible, infectious_b) == 20 * 3 / 1000
    assert model._get_infection_frequency_multiplier(susceptible, infectious_c) == 10 * 2 / 1000

    # Get infection flow rates
    flow_rates = model._get_flow_rates(model.initial_population, 0)
    sus_pop = 900
    flow_to_a = sus_pop * contact_rate * (70 * 0.5 / 1000)
    flow_to_b = sus_pop * contact_rate * (20 * 3 / 1000)
    flow_to_c = sus_pop * contact_rate * (10 * 2 / 1000)
    expected_flow_rates = np.array(
        [-flow_to_a - flow_to_b - flow_to_c, flow_to_a, flow_to_b, flow_to_c, 0.0]
    )
    assert_allclose(expected_flow_rates, flow_rates, verbose=True)


def test_strain__with_flow_adjustments():
    """
    Test infectious multiplier and flow rate calculations for
    3 strains which have different flow adjustments.

    These flow adjustments would correspond to some physical process that we're modelling,
    and they should be effectively the same as applying infectiousness multipliers.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 900, "I": 100})
    contact_rate = 0.2
    model.add_infection_frequency_flow("infection", contact_rate, "S", "I")
    strat = StrainStratification("strain", ["a", "b", "c"], ["I"])
    strat.set_population_split(
        {
            "a": 0.7,  # 70 people
            "b": 0.2,  # 20 people
            "c": 0.1,  # 10 people
        }
    )
    strat.add_flow_adjustments(
        "infection",
        {
            "a": adjust.Multiply(0.5),  # 0.5x as susceptible
            "b": adjust.Multiply(3),  # 3x as susceptible
            "c": adjust.Multiply(2),  # 2x as susceptible
        },
    )
    model.stratify_with(strat)

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    assert_array_equal(model._compartment_infectiousness["a"], np.array([0, 1, 0, 0, 0]))
    assert_array_equal(model._compartment_infectiousness["b"], np.array([0, 0, 1, 0, 0]))
    assert_array_equal(model._compartment_infectiousness["c"], np.array([0, 0, 0, 1, 0]))
    assert model._category_lookup == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    assert_array_equal(model._category_matrix, np.array([[1, 1, 1, 1, 1]]))

    # Do pre-iteration force of infection calcs
    model._prepare_time_step(0, model.initial_population)
    assert_array_equal(model._category_populations, np.array([[1000]]))
    assert_array_equal(model._infection_density["a"], np.array([[70]]))
    assert_array_equal(model._infection_density["b"], np.array([[20]]))
    assert_array_equal(model._infection_density["c"], np.array([[10]]))
    assert_array_equal(model._infection_frequency["a"], np.array([[70 / 1000]]))
    assert_array_equal(model._infection_frequency["b"], np.array([[20 / 1000]]))
    assert_array_equal(model._infection_frequency["c"], np.array([[10 / 1000]]))

    # Get multipliers
    susceptible = model.compartments[0]
    infectious_a = model.compartments[1]
    infectious_b = model.compartments[2]
    infectious_c = model.compartments[3]
    assert model._get_infection_density_multiplier(susceptible, infectious_a) == 70
    assert model._get_infection_density_multiplier(susceptible, infectious_b) == 20
    assert model._get_infection_density_multiplier(susceptible, infectious_c) == 10
    assert model._get_infection_frequency_multiplier(susceptible, infectious_a) == 70 / 1000
    assert model._get_infection_frequency_multiplier(susceptible, infectious_b) == 20 / 1000
    assert model._get_infection_frequency_multiplier(susceptible, infectious_c) == 10 / 1000

    # Get infection flow rates
    flow_rates = model._get_flow_rates(model.initial_population, 0)
    sus_pop = 900
    flow_to_a = sus_pop * contact_rate * (70 * 0.5 / 1000)
    flow_to_b = sus_pop * contact_rate * (20 * 3 / 1000)
    flow_to_c = sus_pop * contact_rate * (10 * 2 / 1000)
    expected_flow_rates = np.array(
        [-flow_to_a - flow_to_b - flow_to_c, flow_to_a, flow_to_b, flow_to_c, 0.0]
    )
    assert_allclose(expected_flow_rates, flow_rates, verbose=True)


def test_strain__with_infectious_multipliers_and_heterogeneous_mixing():
    """
    Test infectious multiplier and flow rate calculations for
    3 strains which have different infectiousness levels plus a seperate
    stratification which has a mixing matrix.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 900, "I": 100})
    contact_rate = 0.2
    model.add_infection_frequency_flow("infection", contact_rate, "S", "I")

    age_strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    age_strat.set_population_split(
        {
            "child": 0.6,  # 600 people
            "adult": 0.4,  # 400 people
        }
    )
    # Higher mixing among adults or children,
    # than between adults or children.
    age_strat.set_mixing_matrix(np.array([[1.5, 0.5], [0.5, 1.5]]))
    model.stratify_with(age_strat)

    strain_strat = StrainStratification("strain", ["a", "b", "c"], ["I"])
    strain_strat.set_population_split(
        {
            "a": 0.7,  # 70 people
            "b": 0.2,  # 20 people
            "c": 0.1,  # 10 people
        }
    )
    strain_strat.add_infectiousness_adjustments(
        "I",
        {
            "a": adjust.Multiply(0.5),  # 0.5x as susceptible
            "b": adjust.Multiply(3),  # 3x as susceptible
            "c": adjust.Multiply(2),  # 2x as susceptible
        },
    )
    model.stratify_with(strain_strat)

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    assert_array_equal(
        model._compartment_infectiousness["a"], np.array([0, 0, 0.5, 0, 0, 0.5, 0, 0, 0, 0])
    )
    assert_array_equal(
        model._compartment_infectiousness["b"], np.array([0, 0, 0, 3, 0, 0, 3, 0, 0, 0])
    )
    assert_array_equal(
        model._compartment_infectiousness["c"], np.array([0, 0, 0, 0, 2, 0, 0, 2, 0, 0])
    )
    # 0 for child, 1 for adult
    assert model._category_lookup == {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 0, 9: 1}
    assert_array_equal(
        model._category_matrix,
        np.array(
            [
                [1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
            ]
        ),
    )

    # Do pre-iteration force of infection calcs
    model._prepare_time_step(0, model.initial_population)
    assert_array_equal(model._category_populations, np.array([[600], [400]]))
    assert_array_equal(
        model._infection_density["a"],
        np.array([[0.5 * (42 * 1.5 + 28 * 0.5)], [0.5 * (42 * 0.5 + 28 * 1.5)]]),
    )
    assert_array_equal(
        model._infection_density["b"],
        np.array(
            [
                [3 * (12 * 1.5 + 8 * 0.5)],
                [3 * (8 * 1.5 + 12 * 0.5)],
            ]
        ),
    )
    assert_array_equal(
        model._infection_density["c"],
        np.array([[2 * (6 * 1.5 + 4 * 0.5)], [2 * (4 * 1.5 + 6 * 0.5)]]),
    )
    assert_array_equal(
        model._infection_frequency["a"],
        np.array(
            [
                [0.5 * ((42 / 600) * 1.5 + (28 / 400) * 0.5)],
                [0.5 * ((42 / 600) * 0.5 + (28 / 400) * 1.5)],
            ]
        ),
    )
    assert_array_equal(
        model._infection_frequency["b"],
        np.array(
            [
                [3 * ((12 / 600) * 1.5 + (8 / 400) * 0.5)],
                [3 * ((8 / 400) * 1.5 + (12 / 600) * 0.5)],
            ]
        ),
    )
    assert_array_equal(
        model._infection_frequency["c"],
        np.array(
            [[2 * ((6 / 600) * 1.5 + (4 / 400) * 0.5)], [2 * ((4 / 400) * 1.5 + (6 / 600) * 0.5)]]
        ),
    )

    # Get multipliers
    sus_child = model.compartments[0]
    sus_adult = model.compartments[1]
    inf_child_a = model.compartments[2]
    inf_child_b = model.compartments[3]
    inf_child_c = model.compartments[4]
    inf_adult_a = model.compartments[5]
    inf_adult_b = model.compartments[6]
    inf_adult_c = model.compartments[7]
    density = model._get_infection_density_multiplier
    freq = model._get_infection_frequency_multiplier
    assert density(sus_child, inf_child_a) == 0.5 * (42 * 1.5 + 28 * 0.5)
    assert density(sus_adult, inf_adult_a) == 0.5 * (42 * 0.5 + 28 * 1.5)
    assert density(sus_child, inf_child_b) == 3 * (12 * 1.5 + 8 * 0.5)
    assert density(sus_adult, inf_adult_b) == 3 * (8 * 1.5 + 12 * 0.5)
    assert density(sus_child, inf_child_c) == 2 * (6 * 1.5 + 4 * 0.5)
    assert density(sus_adult, inf_adult_c) == 2 * (4 * 1.5 + 6 * 0.5)
    assert freq(sus_child, inf_child_a) == 0.5 * ((42 / 600) * 1.5 + (28 / 400) * 0.5)
    assert freq(sus_adult, inf_adult_a) == 0.5 * ((42 / 600) * 0.5 + (28 / 400) * 1.5)
    assert freq(sus_child, inf_child_b) == 3 * ((12 / 600) * 1.5 + (8 / 400) * 0.5)
    assert freq(sus_adult, inf_adult_b) == 3 * ((8 / 400) * 1.5 + (12 / 600) * 0.5)
    assert freq(sus_child, inf_child_c) == 2 * ((6 / 600) * 1.5 + (4 / 400) * 0.5)
    assert freq(sus_adult, inf_adult_c) == 2 * ((4 / 400) * 1.5 + (6 / 600) * 0.5)

    # Get infection flow rates
    flow_to_inf_child_a = 540 * contact_rate * freq(sus_child, inf_child_a)
    flow_to_inf_adult_a = 360 * contact_rate * freq(sus_adult, inf_adult_a)
    flow_to_inf_child_b = 540 * contact_rate * freq(sus_child, inf_child_b)
    flow_to_inf_adult_b = 360 * contact_rate * freq(sus_adult, inf_adult_b)
    flow_to_inf_child_c = 540 * contact_rate * freq(sus_child, inf_child_c)
    flow_to_inf_adult_c = 360 * contact_rate * freq(sus_adult, inf_adult_c)
    expected_flow_rates = np.array(
        [
            -flow_to_inf_child_a - flow_to_inf_child_b - flow_to_inf_child_c,
            -flow_to_inf_adult_a - flow_to_inf_adult_b - flow_to_inf_adult_c,
            flow_to_inf_child_a,
            flow_to_inf_child_b,
            flow_to_inf_child_c,
            flow_to_inf_adult_a,
            flow_to_inf_adult_b,
            flow_to_inf_adult_c,
            0.0,
            0.0,
        ]
    )
    flow_rates = model._get_flow_rates(model.initial_population, 0)
    assert_allclose(expected_flow_rates, flow_rates, verbose=True)
