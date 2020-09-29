"""
Ensure that the EpiModel model produces the correct flow rates and outputs when run.
"""
import pytest

import numpy as np

from summer.model import EpiModel, StratifiedModel
from summer.constants import (
    Compartment,
    Flow,
    BirthApproach,
    IntegrationType,
)

MODEL_KWARGS = {
    "times": np.array([0.0, 1, 2, 3, 4, 5]),
    "compartment_names": [Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
    "initial_conditions": {Compartment.EARLY_INFECTIOUS: 10},
    "parameters": {},
    "requested_flows": [],
    "starting_population": 1000,
    "infectious_compartments": [Compartment.EARLY_INFECTIOUS],
    "birth_approach": BirthApproach.NO_BIRTH,
    "entry_compartment": Compartment.SUSCEPTIBLE,
}


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_apply_flows__with_no_flows(ModelClass):
    """
    Expect no flow to occur because there are no flows.
    """

    model_kwargs = {
        **MODEL_KWARGS,
        "parameters": {},
        "requested_flows": [],
    }
    model = ModelClass(**model_kwargs)
    model.prepare_to_run()
    model.prepare_time_step(0, model.compartment_values)
    flow_rates = np.array([1, 2], dtype=np.float)
    new_rates = model.apply_transition_flows(flow_rates, model.compartment_values, 0)
    new_rates = model.apply_exit_flows(new_rates, model.compartment_values, 0)
    new_rates = model.apply_entry_flows(new_rates, model.compartment_values, 0)
    assert (new_rates == np.array([1, 2], dtype=np.float)).all()


@pytest.mark.parametrize(
    "inf_pop, sus_pop, exp_flow", [(10, 990, 99), (500, 500, 50), (0, 1000, 100), (1000, 0, 0)]
)
@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_apply_flows__with_standard_flow__expect_flows_applied(
    ModelClass, inf_pop, sus_pop, exp_flow
):
    """
    Expect a flow to occur proportional to the compartment size and parameter.
    """
    model_kwargs = {
        **MODEL_KWARGS,
        "parameters": {"a_param": 0.1},
        "initial_conditions": {
            Compartment.EARLY_INFECTIOUS: inf_pop,
            Compartment.SUSCEPTIBLE: sus_pop,
        },
        "requested_flows": [
            {
                "type": Flow.STANDARD,
                "parameter": "a_param",
                "origin": Compartment.SUSCEPTIBLE,
                "to": Compartment.EARLY_INFECTIOUS,
            }
        ],
    }
    model = ModelClass(**model_kwargs)
    model.prepare_to_run()
    model.prepare_time_step(0, model.compartment_values)
    flow_rates = np.array([1, 2], dtype=np.float)

    new_rates = model.apply_transition_flows(flow_rates, model.compartment_values, 0)
    new_rates = model.apply_exit_flows(new_rates, model.compartment_values, 0)
    new_rates = model.apply_entry_flows(new_rates, model.compartment_values, 0)

    # Expect sus_pop * 0.1 = exp_flow
    assert new_rates.tolist() == [1 - exp_flow, 2 + exp_flow]


@pytest.mark.parametrize(
    "inf_pop, sus_pop, exp_flow", [(10, 990, 198), (500, 500, 5000), (0, 1000, 0), (1000, 0, 0)]
)
@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_apply_flows__with_infection_frequency(ModelClass, inf_pop, sus_pop, exp_flow):
    """
    Use infection frequency, expect infection multiplier to be proportional
    to the proprotion of infectious to total pop.
    """
    model_kwargs = {
        **MODEL_KWARGS,
        "parameters": {"contact_rate": 20},
        "initial_conditions": {
            Compartment.EARLY_INFECTIOUS: inf_pop,
            Compartment.SUSCEPTIBLE: sus_pop,
        },
        "requested_flows": [
            {
                "type": Flow.INFECTION_FREQUENCY,
                "parameter": "contact_rate",
                "origin": Compartment.SUSCEPTIBLE,
                "to": Compartment.EARLY_INFECTIOUS,
            }
        ],
    }
    model = ModelClass(**model_kwargs)
    model.prepare_to_run()
    model.prepare_time_step(0, model.compartment_values)
    flow_rates = np.array([1, 2], dtype=np.float)

    new_rates = model.apply_transition_flows(flow_rates, model.compartment_values, 0)
    new_rates = model.apply_exit_flows(new_rates, model.compartment_values, 0)
    new_rates = model.apply_entry_flows(new_rates, model.compartment_values, 0)

    # Expect sus_pop * 20 * (inf_pop / 1000) = exp_flow
    assert new_rates.tolist() == [1 - exp_flow, 2 + exp_flow]


@pytest.mark.parametrize(
    "inf_pop, sus_pop, exp_flow", [(10, 990, 198), (500, 500, 5000), (0, 1000, 0), (1000, 0, 0)]
)
@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_apply_flows__with_infection_density(ModelClass, inf_pop, sus_pop, exp_flow):
    """
    Use infection density, expect infection multiplier to be proportional
    to the infectious pop.
    """
    model_kwargs = {
        **MODEL_KWARGS,
        "parameters": {"contact_rate": 0.02},
        "initial_conditions": {Compartment.EARLY_INFECTIOUS: inf_pop},
        "requested_flows": [
            {
                "type": Flow.INFECTION_DENSITY,
                "parameter": "contact_rate",
                "origin": Compartment.SUSCEPTIBLE,
                "to": Compartment.EARLY_INFECTIOUS,
            }
        ],
    }
    model = ModelClass(**model_kwargs)
    model.prepare_to_run()
    model.prepare_time_step(0, model.compartment_values)
    flow_rates = np.array([1, 2], dtype=np.float)

    new_rates = model.apply_transition_flows(flow_rates, model.compartment_values, 0)
    new_rates = model.apply_exit_flows(new_rates, model.compartment_values, 0)
    new_rates = model.apply_entry_flows(new_rates, model.compartment_values, 0)

    # Expect 0.2 * sus_pop * inf_pop = exp_flow
    assert new_rates.tolist() == [1 - exp_flow, 2 + exp_flow]


@pytest.mark.parametrize("inf_pop, exp_flow", [(1000, 100), (990, 99), (50, 5), (0, 0)])
@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_apply_infect_death_flows(ModelClass, inf_pop, exp_flow):
    model_kwargs = {
        **MODEL_KWARGS,
        "parameters": {"infect_death": 0.1},
        "initial_conditions": {Compartment.EARLY_INFECTIOUS: inf_pop},
        "requested_flows": [
            {
                "type": Flow.DEATH,
                "parameter": "infect_death",
                "origin": Compartment.EARLY_INFECTIOUS,
            }
        ],
    }
    model = ModelClass(**model_kwargs)
    model.prepare_to_run()
    model.prepare_time_step(0, model.compartment_values)
    flow_rates = np.array([1, 2], dtype=np.float)

    new_rates = model.apply_transition_flows(flow_rates, model.compartment_values, 0)
    new_rates = model.apply_exit_flows(new_rates, model.compartment_values, 0)
    new_rates = model.apply_entry_flows(new_rates, model.compartment_values, 0)

    # Expect 0.1 * inf_pop = exp_flow
    assert new_rates.tolist() == [1, 2 - exp_flow]
    assert model.total_deaths == exp_flow


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_apply_univeral_death_flow(ModelClass):
    model_kwargs = {
        **MODEL_KWARGS,
        "parameters": {"universal_death_rate": 0.1},
        "requested_flows": [],
    }
    model = ModelClass(**model_kwargs)
    model.prepare_to_run()
    model.prepare_time_step(0, model.compartment_values)
    assert model.compartment_values.tolist() == [990, 10]
    flow_rates = np.array([1, 2], dtype=np.float)
    new_rates = model.apply_exit_flows(flow_rates, model.compartment_values, 0)
    assert new_rates.tolist() == [-98, 1]
    assert model.total_deaths == 100


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_apply_univeral_death_flow__with_no_death_rate(ModelClass):
    model_kwargs = {
        **MODEL_KWARGS,
        "parameters": {"universal_death_rate": 0},
        "requested_flows": [],
    }
    model = ModelClass(**model_kwargs)
    model.prepare_to_run()
    model.prepare_time_step(0, model.compartment_values)
    assert model.compartment_values.tolist() == [990, 10]
    flow_rates = np.array([1, 2], dtype=np.float)
    new_rates = model.apply_exit_flows(flow_rates, model.compartment_values, 0)
    assert new_rates.tolist() == [1, 2]
    assert model.total_deaths == 0


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model_apply_birth_rate__with_no_birth_approach__expect_no_births(ModelClass):
    """
    Expect no births when a no birth approach is used.
    """
    model_kwargs = {**MODEL_KWARGS, "birth_approach": BirthApproach.NO_BIRTH}
    model = ModelClass(**model_kwargs)
    model.prepare_to_run()
    model.prepare_time_step(0, model.compartment_values)
    flow_rates = np.array([1, 2], dtype=np.float)
    new_rates = model.apply_entry_flows(flow_rates, model.compartment_values, 0)
    assert new_rates.tolist() == [1, 2]


@pytest.mark.parametrize("birth_rate, exp_flow", [[0.0035, 3.5], [0, 0]])
@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model_apply_birth_rate__with_crude_birth_rate__expect_births(
    ModelClass, birth_rate, exp_flow
):
    """
    Expect births proportional to the total population and birth rate when
    the birth approach is "crude birth rate".
    """
    params = {"crude_birth_rate": birth_rate}
    model_kwargs = {
        **MODEL_KWARGS,
        "birth_approach": BirthApproach.ADD_CRUDE,
        "parameters": params,
    }
    model = EpiModel(**model_kwargs)
    model.prepare_to_run()
    flow_rates = np.array([1, 2], dtype=np.float)
    new_rates = model.apply_entry_flows(flow_rates, model.compartment_values, 0)
    assert new_rates.tolist() == [1 + exp_flow, 2]


@pytest.mark.parametrize("total_deaths", [1, 123, 0])
@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model_apply_birth_rate__with_replace_deaths__expect_births(ModelClass, total_deaths):
    """
    Expect births proportional to the tracked deaths when birth approach is "replace deaths".
    """
    model_kwargs = {**MODEL_KWARGS, "birth_approach": BirthApproach.REPLACE_DEATHS}
    model = EpiModel(**model_kwargs)
    model.prepare_to_run()
    model.total_deaths = total_deaths
    flow_rates = np.array([1, 2], dtype=np.float)
    new_rates = model.apply_entry_flows(flow_rates, model.compartment_values, 0)
    assert new_rates.tolist() == [1 + total_deaths, 2]
