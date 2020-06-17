"""
Ensure that the EpiModel model produces the correct flow rates and outputs when run.
"""
import pytest

from summer.model import EpiModel
from summer.constants import (
    Compartment,
    Flow,
    BirthApproach,
    Stratification,
    IntegrationType,
)

MODEL_KWARGS = {
    "times": [2000, 2001, 2002, 2003, 2004, 2005],
    "compartment_types": [Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
    "initial_conditions": {Compartment.EARLY_INFECTIOUS: 10},
    "parameters": {},
    "requested_flows": [],
    "starting_population": 1000,
}


PARAM_VARS = "flows,params,flow_rates,expected_new_rates"
PARAM_VALS = [
    # No transition flows implemented, expect no flows.
    [[], {}, [1, 2], [1, 2]],
    # Add standard flow where 30% of infected people recover back to susceptible.
    [
        [
            {
                "type": Flow.STANDARD,
                "parameter": "recover_rate",
                "origin": Compartment.EARLY_INFECTIOUS,
                "to": Compartment.SUSCEPTIBLE,
            }
        ],
        {"recover_rate": 0.3},
        [1, 2],
        [4, -1],
    ],
    # Add a flow with a custom flow func
    [
        [
            {
                "type": Flow.CUSTOM,
                "parameter": "recover_rate",
                "origin": Compartment.EARLY_INFECTIOUS,
                "to": Compartment.SUSCEPTIBLE,
                # Add a function that halves the rate, but uses total population.
                "function": lambda model, flow_idx, time, compartments: 0.5 * sum(compartments),
            }
        ],
        {"recover_rate": 0.3},
        [1, 2],
        [151, -148],
    ],
    # Use infection frequency, expect infection multiplier to be proportional
    # to the proprotion of infectious to total pop.
    [
        [
            {
                "type": Flow.INFECTION_FREQUENCY,
                "parameter": "contact_rate",
                "origin": Compartment.SUSCEPTIBLE,
                "to": Compartment.EARLY_INFECTIOUS,
            }
        ],
        {"contact_rate": 10},
        [1, 2],
        [-98, 101],
    ],
    # Use infection density, expect infection multiplier to be proportional
    # to the infectious pop.
    [
        [
            {
                "type": Flow.INFECTION_DENSITY,
                "parameter": "contact_rate",
                "origin": Compartment.SUSCEPTIBLE,
                "to": Compartment.EARLY_INFECTIOUS,
            }
        ],
        {"contact_rate": 0.01},
        [1, 2],
        [-98, 101],
    ],
]


@pytest.mark.parametrize(PARAM_VARS, PARAM_VALS)
def test_apply_transition_flows(flows, params, flow_rates, expected_new_rates):
    """
    Ensure user-specified compartment transition flows are applied correctly.
    """
    model_kwargs = {
        **MODEL_KWARGS,
        "parameters": params,
        "requested_flows": flows,
    }
    model = EpiModel(**model_kwargs)
    model.prepare_to_run()
    model.update_tracked_quantities(model.compartment_values)
    new_rates = model.apply_transition_flows(flow_rates, model.compartment_values, 2000)
    assert new_rates == expected_new_rates


PARAM_VARS = "flows,params,flow_rates,expected_new_rates,expect_deaths"
PARAM_VALS = [
    # No death flows implemented, expect no deaths.
    [[], {}, [1, 2], [1, 2], 0,],
    # Add flow where 50% of infectious people die.
    [
        [
            {
                "type": Flow.COMPARTMENT_DEATH,
                "parameter": "infect_death",
                "origin": Compartment.EARLY_INFECTIOUS,
            }
        ],
        {"infect_death": 0.5},
        [1, 2],
        [1, -3],
        5,
    ],
    # Add flow where 20% of susceptible people die.
    [
        [
            {
                "type": Flow.COMPARTMENT_DEATH,
                "parameter": "sus_death",
                "origin": Compartment.SUSCEPTIBLE,
            }
        ],
        {"sus_death": 0.2},
        [1, 2],
        [-197, 2],
        198,
    ],
]


@pytest.mark.parametrize(PARAM_VARS, PARAM_VALS)
def test_apply_compartment_death_flows(
    flows, params, flow_rates, expected_new_rates, expect_deaths
):
    """
    Ensure user-specified compartment death flows are applied correctly.
    """
    model_kwargs = {
        **MODEL_KWARGS,
        "birth_approach": BirthApproach.REPLACE_DEATHS,
        "parameters": params,
        "requested_flows": flows,
    }
    model = EpiModel(**model_kwargs)
    model.prepare_to_run()
    new_rates = model.apply_compartment_death_flows(flow_rates, model.compartment_values, 2000)
    assert new_rates == expected_new_rates
    assert model.tracked_quantities["total_deaths"] == expect_deaths


PARAM_VARS = "death_rate,flow_rates,expected_new_rates,expect_deaths"
PARAM_VALS = [
    # Positive death rate, expect deaths in all compartments.
    [0.02, [1, 2], [-18.8, 1.8], 20],
    # Zero death rate, expect no deaths.
    [0, [1, 2], [1, 2], 0],
]


@pytest.mark.parametrize(PARAM_VARS, PARAM_VALS)
def test_epi_model_apply_universal_death_flow(
    death_rate, flow_rates, expected_new_rates, expect_deaths
):
    """
    Ensure EpiModel apply universal death rates to kill all compartments in proportion to death rate.
    """
    model_kwargs = {
        **MODEL_KWARGS,
        "birth_approach": BirthApproach.REPLACE_DEATHS,
        "parameters": {"universal_death_rate": death_rate,},
    }
    model = EpiModel(**model_kwargs)
    model.prepare_to_run()
    new_rates = model.apply_universal_death_flow(flow_rates, model.compartment_values, 2000)
    assert new_rates == expected_new_rates
    assert model.tracked_quantities["total_deaths"] == expect_deaths


def test_epi_model_apply_change_rates():
    """
    Ensure EpiModel apply change rates makes no change to compartment flows.
    """
    model = EpiModel(**MODEL_KWARGS)
    model.prepare_to_run()
    flow_rates = [1, 2]
    new_rates = model.apply_change_rates(flow_rates, model.compartment_values, 2000)
    assert new_rates == flow_rates


def test_epi_model_apply_birth_rate__with_no_birth_approach__expect_no_births():
    """
    Expect no births when a no birth approach is used.
    """
    model_kwargs = {**MODEL_KWARGS, "birth_approach": BirthApproach.NO_BIRTH}
    model = EpiModel(**model_kwargs)
    model.prepare_to_run()
    flow_rates = [1, 2]
    new_rates = model.apply_birth_rate(flow_rates, model.compartment_values, 2000)
    assert new_rates == flow_rates


PARAM_VARS = "birth_rate,flow_rates,expected_new_rates"
PARAM_VALS = [
    # Positive birth rate, expect births in entry compartment.
    [0.0035, [1, 2], [4.5, 2]],
    # Zero birth rate, expect no births.
    [0, [1, 2], [1, 2]],
]


@pytest.mark.parametrize(PARAM_VARS, PARAM_VALS)
def test_epi_model_apply_birth_rate__with_crude_birth_rate__expect_births(
    birth_rate, flow_rates, expected_new_rates
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
    new_rates = model.apply_birth_rate(flow_rates, model.compartment_values, 2000)
    assert new_rates == expected_new_rates


PARAM_VARS = "total_deaths,flow_rates,expected_new_rates"
PARAM_VALS = [
    # Some deaths, expect proportional births in entry compartment,
    [4, [1, 2], [5, 2]],
    # No deaths, expect no births.
    [0, [1, 2], [1, 2]],
]


@pytest.mark.parametrize(PARAM_VARS, PARAM_VALS)
def test_epi_model_apply_birth_rate__with_replace_deaths__expect_births(
    total_deaths, flow_rates, expected_new_rates
):
    """
    Expect births proportional to the tracked deaths when birth approach is "replace deaths".
    """
    model_kwargs = {**MODEL_KWARGS, "birth_approach": BirthApproach.REPLACE_DEATHS}
    model = EpiModel(**model_kwargs)
    model.prepare_to_run()
    model.tracked_quantities["total_deaths"] = total_deaths
    new_rates = model.apply_birth_rate(flow_rates, model.compartment_values, 2000)
    assert new_rates == expected_new_rates
