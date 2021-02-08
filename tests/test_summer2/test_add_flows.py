"""
Basic test of all the flows that CompartmentalModel provides, with no stratifications.
Ensure that the model produces the correct flow rates when run.
"""
import pytest

import numpy as np
from numpy.testing import assert_array_equal

from summer2.model import CompartmentalModel


def test_apply_flows__with_no_flows():
    """
    Expect no flow to occur because there are no flows.
    """
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": 990, "I": 10})
    model._prepare_to_run()
    actual_flow_rates = model._get_flow_rates(model.initial_population, 0)
    expected_flow_rates = np.array([0, 0])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


@pytest.mark.parametrize(
    "inf_pop, sus_pop, exp_flow", [(10, 990, 99), (500, 500, 50), (0, 1000, 100), (1000, 0, 0)]
)
def test_apply_flows__with_fractional_flow__expect_flows_applied(inf_pop, sus_pop, exp_flow):
    """
    Expect a flow to occur proportional to the compartment size and parameter.
    """
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": sus_pop, "I": inf_pop})
    model.add_fractional_flow("deliberately_infected", 0.1, "S", "I")
    model._prepare_to_run()
    actual_flow_rates = model._get_flow_rates(model.initial_population, 0)
    # Expect sus_pop * 0.1 = exp_flow
    expected_flow_rates = np.array([-exp_flow, exp_flow])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


@pytest.mark.parametrize("inf_pop, exp_flow", [(120, 12), (500, 50), (0, 0)])
def test_apply_flows__with_sojourn_flow__expect_flows_applied(inf_pop, exp_flow):
    """
    Expect a flow to occur proportional to the compartment size and parameter.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"I": inf_pop})
    model.add_sojourn_flow("recovery", 10, "I", "R")
    model._prepare_to_run()
    actual_flow_rates = model._get_flow_rates(model.initial_population, 0)
    # Expect exp_flow = inf_pop * () exp_flow
    expected_flow_rates = np.array([0, -exp_flow, exp_flow])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


@pytest.mark.parametrize(
    "inf_pop, sus_pop, exp_flow", [(10, 990, 109), (500, 500, 550), (0, 1000, 100), (1000, 0, 1000)]
)
def test_apply_flows__with_function_flow__expect_flows_applied(inf_pop, sus_pop, exp_flow):
    """
    Expect a flow to occur proportional to the result of `get_flow_rate`.
    """

    def get_flow_rate(flow, comps, comp_vals, flows, flow_rates, time):
        _, i_pop, _ = comp_vals
        i_flow, _ = flow_rates
        return i_pop + i_flow

    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": sus_pop, "I": inf_pop})
    model.add_fractional_flow("infection", 0.1, "S", "I")
    model.add_function_flow("treatment", get_flow_rate, "I", "S")
    model._prepare_to_run()
    actual_flow_rates = model._get_flow_rates(model.initial_population, 0)
    expected_infected = sus_pop * 0.1
    expected_flow_rates = np.array(
        [
            exp_flow - expected_infected,
            expected_infected - exp_flow,
            0,
        ]
    )

    assert_array_equal(actual_flow_rates, expected_flow_rates)


@pytest.mark.parametrize(
    "inf_pop, sus_pop, exp_flow", [(10, 990, 198), (500, 500, 5000), (0, 1000, 0), (1000, 0, 0)]
)
def test_apply_flows__with_infection_frequency(inf_pop, sus_pop, exp_flow):
    """
    Use infection frequency, expect infection multiplier to be proportional
    to the proprotion of infectious to total pop.
    """
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": sus_pop, "I": inf_pop})
    model.add_infection_frequency_flow("infection", 20, "S", "I")
    model._prepare_to_run()
    actual_flow_rates = model._get_flow_rates(model.initial_population, 0)
    # Expect sus_pop * 20 * (inf_pop / 1000) = exp_flow
    expected_flow_rates = np.array([-exp_flow, exp_flow])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


@pytest.mark.parametrize(
    "inf_pop, sus_pop, exp_flow", [(10, 990, 198), (500, 500, 5000), (0, 1000, 0), (1000, 0, 0)]
)
def test_apply_flows__with_infection_density(inf_pop, sus_pop, exp_flow):
    """
    Use infection density, expect infection multiplier to be proportional
    to the infectious pop.
    """
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": sus_pop, "I": inf_pop})
    model.add_infection_density_flow("infection", 0.02, "S", "I")
    model._prepare_to_run()
    actual_flow_rates = model._get_flow_rates(model.initial_population, 0)
    # Expect 0.2 * sus_pop * inf_pop = exp_flow
    expected_flow_rates = np.array([-exp_flow, exp_flow])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


@pytest.mark.parametrize("inf_pop, exp_flow", [(1000, 100), (990, 99), (50, 5), (0, 0)])
def test_apply_infect_death_flows(inf_pop, exp_flow):
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"I": inf_pop})
    model.add_death_flow("infection_death", 0.1, "I")
    model._prepare_to_run()
    actual_flow_rates = model._get_flow_rates(model.initial_population, 0)
    # Expect 0.1 * inf_pop = exp_flow
    expected_flow_rates = np.array([0, -exp_flow])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


def test_apply_univeral_death_flow():
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.add_universal_death_flows("universal_death", 0.1)
    model._prepare_to_run()
    actual_flow_rates = model._get_flow_rates(model.initial_population, 0)
    expected_flow_rates = np.array([-99, -1])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


@pytest.mark.parametrize("birth_rate, exp_flow", [[0.0035, 3.5], [0, 0]])
def test_apply_crude_birth_rate_flow(birth_rate, exp_flow):
    """
    Expect births proportional to the total population and birth rate when
    the birth approach is "crude birth rate".
    """
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.add_crude_birth_flow("births", birth_rate, "S")
    model._prepare_to_run()
    actual_flow_rates = model._get_flow_rates(model.initial_population, 0)
    # Expect birth_rate * total_population = exp_flow
    expected_flow_rates = np.array([exp_flow, 0])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


def test_apply_replace_death_birth_flow():
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": 900, "I": 100})
    model.add_death_flow("infection_death", 0.1, "I")
    model.add_universal_death_flows("universal_death", 0.05)
    model.add_replacement_birth_flow("births", "S")
    model._prepare_to_run()
    actual_flow_rates = model._get_flow_rates(model.initial_population, 0)
    # Expect 10 people to die and 10 to be born
    exp_i_flow_rate = -0.1 * 100 - 0.05 * 100
    exp_s_flow_rate = -exp_i_flow_rate  # N.B births + deaths in the S compartment should balance.
    expected_flow_rates = np.array([exp_s_flow_rate, exp_i_flow_rate])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


def test_apply_many_flows():
    """
    Expect multiple flows to operate independently and produce the correct final flow rate.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 900, "I": 100})
    model.add_death_flow("infection_death", 0.1, "I")
    model.add_universal_death_flows("universal_death", 0.1)
    model.add_infection_frequency_flow("infection", 0.2, "S", "I")
    model.add_sojourn_flow("recovery", 10, "I", "R")
    model.add_fractional_flow("vaccination", 0.1, "S", "R")
    model.add_crude_birth_flow("births", 0.1, "S")
    model._prepare_to_run()
    actual_flow_rates = model._get_flow_rates(model.initial_population, 0)

    # Expect the effects of all these flows to be linearly superimposed.
    infect_death_flows = np.array([0, -10, 0])
    universal_death_flows = np.array([-90, -10, 0])
    infected = 900 * 0.2 * (100 / 1000)
    infection_flows = np.array([-infected, infected, 0])
    recovery_flows = np.array([0, -10, 10])
    vaccination_flows = np.array([-90, 0, 90])
    birth_flows = np.array([100, 0, 0])
    expected_flow_rates = (
        infect_death_flows
        + universal_death_flows
        + infection_flows
        + recovery_flows
        + vaccination_flows
        + birth_flows
    )
    assert_array_equal(actual_flow_rates, expected_flow_rates)
