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


TODO
    - flow adjustments
    - with other stratifications
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from summer.model import StratifiedModel
from summer.constants import BirthApproach, Flow


MODEL_KWARGS = {
    "times": np.array([0.0, 1, 2, 3, 4, 5]),
    "compartment_names": ["S", "I", "R"],
    "initial_conditions": {"I": 100},
    "parameters": {},
    "requested_flows": [],
    "starting_population": 1000,
    "infectious_compartments": ["I"],
    "birth_approach": BirthApproach.NO_BIRTH,
    "entry_compartment": "S",
}


def test_model__with_two_symmetric_stratifications():
    """
    Adding two strata with the same properties should yield the exact same infection dynamics and outputs as having no strata at all.
    This does not test strains directly, but if this doesn't work then further testing is pointless.
    """
    params = {
        "contact_rate": 0.2,
        "recovery_rate": 0.1,
    }
    flows = (
        {
            "type": Flow.INFECTION_FREQUENCY,
            "parameter": "contact_rate",
            "origin": "S",
            "to": "I",
        },
        {
            "type": Flow.STANDARD,
            "parameter": "recovery_rate",
            "origin": "I",
            "to": "R",
        },
    )
    # Create an unstratified model
    kwargs = {**MODEL_KWARGS, "parameters": params, "requested_flows": flows}
    model = StratifiedModel(**kwargs)
    # Do pre-run force of infection calcs.
    model.prepare_to_run()
    model.prepare_time_step(0, model.compartment_values)
    # Check infectiousness multipliers
    susceptible = model.compartment_names[0]
    infectious = model.compartment_names[1]
    assert model.get_infection_density_multipier(susceptible, infectious) == 100.0
    assert model.get_infection_frequency_multipier(susceptible, infectious) == 0.1
    model.run_model()
    # Create a stratified model where the two strains are symmetric
    strat_model = StratifiedModel(**kwargs)
    strat_model.stratify(
        stratification_name="clinical",
        strata_request=["home", "hospital"],
        compartments_to_stratify=["I"],
        # Use defaults - equally split compartments, flows, etc.
        comp_split_props={},
        flow_adjustments={},
        infectiousness_adjustments={},
        mixing_matrix=None,
    )
    strat_model.run_model()
    merged_outputs = np.zeros_like(model.outputs)
    merged_outputs[:, 0] = strat_model.outputs[:, 0]
    merged_outputs[:, 1] = strat_model.outputs[:, 1] + strat_model.outputs[:, 2]
    merged_outputs[:, 2] = strat_model.outputs[:, 3]
    assert_allclose(merged_outputs, model.outputs, atol=0.01, rtol=0.01, verbose=True)


def test_strains__with_two_symmetric_strains():
    """
    Adding two strains with the same properties should yield the same infection dynamics and outputs as having no strains at all.
    We expect the force of infection for each strain to be 1/2 of the unstratified model,
    but the stratification process will not apply the usual conservation fraction to the fan out flows.
    """
    params = {
        "contact_rate": 0.2,
        "recovery_rate": 0.1,
    }
    flows = (
        {
            "type": Flow.INFECTION_FREQUENCY,
            "parameter": "contact_rate",
            "origin": "S",
            "to": "I",
        },
        {
            "type": Flow.STANDARD,
            "parameter": "recovery_rate",
            "origin": "I",
            "to": "R",
        },
    )
    # Create an unstratified model
    kwargs = {**MODEL_KWARGS, "parameters": params, "requested_flows": flows}
    model = StratifiedModel(**kwargs)
    # Do pre-run force of infection calcs.
    model.prepare_to_run()
    model.prepare_time_step(0, model.compartment_values)
    # Check infectiousness multipliers
    susceptible = model.compartment_names[0]
    infectious = model.compartment_names[1]
    assert model.get_infection_density_multipier(susceptible, infectious) == 100.0
    assert model.get_infection_frequency_multipier(susceptible, infectious) == 0.1
    model.run_model()
    # Create a stratified model where the two strains are symmetric
    strain_model = StratifiedModel(**kwargs)
    strain_model.stratify(
        stratification_name="strain",
        strata_request=["a", "b"],
        compartments_to_stratify=["I"],
        # Use defaults - equally split compartments, flows, etc.
        comp_split_props={},
        flow_adjustments={},
        infectiousness_adjustments={},
        mixing_matrix=None,
    )
    strain_model.run_model()
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
    contact_rate = 0.2
    params = {
        "contact_rate": contact_rate,
    }
    flows = (
        {
            "type": Flow.INFECTION_FREQUENCY,
            "parameter": "contact_rate",
            "origin": "S",
            "to": "I",
        },
    )
    kwargs = {**MODEL_KWARGS, "parameters": params, "requested_flows": flows}
    model = StratifiedModel(**kwargs)
    model.stratify(
        stratification_name="strain",
        strata_request=["a", "b", "c"],
        compartments_to_stratify=["I"],
        comp_split_props={
            "a": 0.7,  # 70 people
            "b": 0.2,  # 20 people
            "c": 0.1,  # 10 people
        },
        infectiousness_adjustments={
            "a": 0.5,  # 0.5x as infectious
            "b": 3,  # 3x as infectious
            "c": 2,  # 2x as infectious
        },
        mixing_matrix=None,
    )
    # Do pre-run force of infection calcs.
    model.prepare_to_run()
    assert_array_equal(model.compartment_infectiousness["a"], np.array([0, 0.5, 0, 0, 0]))
    assert_array_equal(model.compartment_infectiousness["b"], np.array([0, 0, 3, 0, 0]))
    assert_array_equal(model.compartment_infectiousness["c"], np.array([0, 0, 0, 2, 0]))
    assert model.category_lookup == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    assert_array_equal(model.category_matrix, np.array([[1, 1, 1, 1, 1]]))

    # Do pre-iteration force of infection calcs
    model.prepare_time_step(0, model.compartment_values)
    assert_array_equal(model.category_populations, np.array([[1000]]))
    assert_array_equal(model.infection_density["a"], np.array([[70 * 0.5]]))
    assert_array_equal(model.infection_density["b"], np.array([[20 * 3]]))
    assert_array_equal(model.infection_density["c"], np.array([[10 * 2]]))
    assert_array_equal(model.infection_frequency["a"], np.array([[70 * 0.5 / 1000]]))
    assert_array_equal(model.infection_frequency["b"], np.array([[20 * 3 / 1000]]))
    assert_array_equal(model.infection_frequency["c"], np.array([[10 * 2 / 1000]]))

    # Get multipliers
    susceptible = model.compartment_names[0]
    infectious_a = model.compartment_names[1]
    infectious_b = model.compartment_names[2]
    infectious_c = model.compartment_names[3]
    assert model.get_infection_density_multipier(susceptible, infectious_a) == 70 * 0.5
    assert model.get_infection_density_multipier(susceptible, infectious_b) == 20 * 3
    assert model.get_infection_density_multipier(susceptible, infectious_c) == 10 * 2
    assert model.get_infection_frequency_multipier(susceptible, infectious_a) == 70 * 0.5 / 1000
    assert model.get_infection_frequency_multipier(susceptible, infectious_b) == 20 * 3 / 1000
    assert model.get_infection_frequency_multipier(susceptible, infectious_c) == 10 * 2 / 1000

    # Get infection flow rates
    flow_rates = model.get_flow_rates(model.compartment_values, 0)
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
    contact_rate = 0.2
    params = {
        "contact_rate": contact_rate,
    }
    flows = (
        {
            "type": Flow.INFECTION_FREQUENCY,
            "parameter": "contact_rate",
            "origin": "S",
            "to": "I",
        },
    )
    kwargs = {**MODEL_KWARGS, "parameters": params, "requested_flows": flows}
    model = StratifiedModel(**kwargs)
    model.stratify(
        stratification_name="strain",
        strata_request=["a", "b", "c"],
        compartments_to_stratify=["I"],
        comp_split_props={
            "a": 0.7,  # 70 people
            "b": 0.2,  # 20 people
            "c": 0.1,  # 10 people
        },
        flow_adjustments={
            "contact_rate": {
                "a": 0.5,  # 0.5x as infectious
                "b": 3,  # 3x as infectious
                "c": 2,  # 2x as infectious
            }
        },
        mixing_matrix=None,
    )
    # Do pre-run force of infection calcs.
    model.prepare_to_run()
    assert_array_equal(model.compartment_infectiousness["a"], np.array([0, 1, 0, 0, 0]))
    assert_array_equal(model.compartment_infectiousness["b"], np.array([0, 0, 1, 0, 0]))
    assert_array_equal(model.compartment_infectiousness["c"], np.array([0, 0, 0, 1, 0]))
    assert model.category_lookup == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    assert_array_equal(model.category_matrix, np.array([[1, 1, 1, 1, 1]]))

    # Do pre-iteration force of infection calcs
    model.prepare_time_step(0, model.compartment_values)
    assert_array_equal(model.category_populations, np.array([[1000]]))
    assert_array_equal(model.infection_density["a"], np.array([[70]]))
    assert_array_equal(model.infection_density["b"], np.array([[20]]))
    assert_array_equal(model.infection_density["c"], np.array([[10]]))
    assert_array_equal(model.infection_frequency["a"], np.array([[70 / 1000]]))
    assert_array_equal(model.infection_frequency["b"], np.array([[20 / 1000]]))
    assert_array_equal(model.infection_frequency["c"], np.array([[10 / 1000]]))

    # Get multipliers
    susceptible = model.compartment_names[0]
    infectious_a = model.compartment_names[1]
    infectious_b = model.compartment_names[2]
    infectious_c = model.compartment_names[3]
    assert model.get_infection_density_multipier(susceptible, infectious_a) == 70
    assert model.get_infection_density_multipier(susceptible, infectious_b) == 20
    assert model.get_infection_density_multipier(susceptible, infectious_c) == 10
    assert model.get_infection_frequency_multipier(susceptible, infectious_a) == 70 / 1000
    assert model.get_infection_frequency_multipier(susceptible, infectious_b) == 20 / 1000
    assert model.get_infection_frequency_multipier(susceptible, infectious_c) == 10 / 1000

    # Get infection flow rates
    flow_rates = model.get_flow_rates(model.compartment_values, 0)
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
    contact_rate = 0.2
    params = {
        "contact_rate": contact_rate,
    }
    flows = (
        {
            "type": Flow.INFECTION_FREQUENCY,
            "parameter": "contact_rate",
            "origin": "S",
            "to": "I",
        },
    )
    kwargs = {**MODEL_KWARGS, "parameters": params, "requested_flows": flows}
    model = StratifiedModel(**kwargs)
    model.stratify(
        stratification_name="agegroup",
        strata_request=["child", "adult"],
        compartments_to_stratify=["S", "I", "R"],
        comp_split_props={
            "child": 0.6,  # 600 people
            "adult": 0.4,  # 400 people
        },
        # Higher mixing among adults or children,
        # than between adults or children.
        mixing_matrix=np.array([[1.5, 0.5], [0.5, 1.5]]),
    )
    model.stratify(
        stratification_name="strain",
        strata_request=["a", "b", "c"],
        compartments_to_stratify=["I"],
        comp_split_props={
            "a": 0.7,  # 70 people
            "b": 0.2,  # 20 people
            "c": 0.1,  # 10 people
        },
        infectiousness_adjustments={
            "a": 0.5,  # 0.5x as infectious
            "b": 3,  # 3x as infectious
            "c": 2,  # 2x as infectious
        },
        mixing_matrix=None,
    )
    # Do pre-run force of infection calcs.
    model.prepare_to_run()
    assert_array_equal(
        model.compartment_infectiousness["a"], np.array([0, 0, 0.5, 0, 0, 0.5, 0, 0, 0, 0])
    )
    assert_array_equal(
        model.compartment_infectiousness["b"], np.array([0, 0, 0, 3, 0, 0, 3, 0, 0, 0])
    )
    assert_array_equal(
        model.compartment_infectiousness["c"], np.array([0, 0, 0, 0, 2, 0, 0, 2, 0, 0])
    )
    # 0 for child, 1 for adult
    assert model.category_lookup == {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 0, 9: 1}
    assert_array_equal(
        model.category_matrix,
        np.array(
            [
                [1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
            ]
        ),
    )

    # Do pre-iteration force of infection calcs
    model.prepare_time_step(0, model.compartment_values)
    assert_array_equal(model.category_populations, np.array([[600], [400]]))
    assert_array_equal(
        model.infection_density["a"],
        np.array([[0.5 * (42 * 1.5 + 28 * 0.5)], [0.5 * (42 * 0.5 + 28 * 1.5)]]),
    )
    assert_array_equal(
        model.infection_density["b"],
        np.array(
            [
                [3 * (12 * 1.5 + 8 * 0.5)],
                [3 * (8 * 1.5 + 12 * 0.5)],
            ]
        ),
    )
    assert_array_equal(
        model.infection_density["c"],
        np.array([[2 * (6 * 1.5 + 4 * 0.5)], [2 * (4 * 1.5 + 6 * 0.5)]]),
    )
    assert_array_equal(
        model.infection_frequency["a"],
        np.array(
            [
                [0.5 * ((42 / 600) * 1.5 + (28 / 400) * 0.5)],
                [0.5 * ((42 / 600) * 0.5 + (28 / 400) * 1.5)],
            ]
        ),
    )
    assert_array_equal(
        model.infection_frequency["b"],
        np.array(
            [
                [3 * ((12 / 600) * 1.5 + (8 / 400) * 0.5)],
                [3 * ((8 / 400) * 1.5 + (12 / 600) * 0.5)],
            ]
        ),
    )
    assert_array_equal(
        model.infection_frequency["c"],
        np.array(
            [[2 * ((6 / 600) * 1.5 + (4 / 400) * 0.5)], [2 * ((4 / 400) * 1.5 + (6 / 600) * 0.5)]]
        ),
    )

    # Get multipliers
    sus_child = model.compartment_names[0]
    sus_adult = model.compartment_names[1]
    inf_child_a = model.compartment_names[2]
    inf_child_b = model.compartment_names[3]
    inf_child_c = model.compartment_names[4]
    inf_adult_a = model.compartment_names[5]
    inf_adult_b = model.compartment_names[6]
    inf_adult_c = model.compartment_names[7]
    density = model.get_infection_density_multipier
    freq = model.get_infection_frequency_multipier
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
    flow_rates = model.get_flow_rates(model.compartment_values, 0)
    assert_allclose(expected_flow_rates, flow_rates, verbose=True)
