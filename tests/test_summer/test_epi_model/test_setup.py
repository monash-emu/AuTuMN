"""
Test that basic setup of the EpiModel and StratifiedModel works.
"""
from unittest import mock
from copy import deepcopy

import pytest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

from summer.model.utils.validation import ValidationException
from summer.model import EpiModel, StratifiedModel
from summer.constants import Compartment, Flow, BirthApproach, Stratification, IntegrationType


@pytest.mark.parametrize(
    "ModelClass", [EpiModel, StratifiedModel],
)
def test_setup_default_parameters(ModelClass):
    """
    Ensure default params are set correctly on model setup.
    """
    base_kwargs = {
        "times": _get_integration_times(2000, 2005, 1),
        "compartment_types": [Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS],
        "initial_conditions": {},
        "parameters": {},
        "requested_flows": [],
        "starting_population": 100,
    }

    # Expect univseral death rate to be set by default
    model = ModelClass(**deepcopy(base_kwargs))
    assert model.parameters == {
        "universal_death_rate": 0,
    }

    # Expect univseral death rate to be overidden
    kwargs = {**deepcopy(base_kwargs), "parameters": {"universal_death_rate": 1234}}
    model = ModelClass(**kwargs)
    assert model.parameters == {"universal_death_rate": 1234}

    # Expect crude birth rate to be set
    kwargs = {**deepcopy(base_kwargs), "birth_approach": BirthApproach.ADD_CRUDE}
    model = ModelClass(**kwargs)
    assert model.parameters == {
        "universal_death_rate": 0,
        "crude_birth_rate": 0,
    }

    # Expect crude birth rate to be overidden
    kwargs = {
        **deepcopy(base_kwargs),
        "birth_approach": BirthApproach.ADD_CRUDE,
        "parameters": {"crude_birth_rate": 123},
    }
    model = ModelClass(**kwargs)
    assert model.parameters == {
        "universal_death_rate": 0,
        "crude_birth_rate": 123,
    }

    # Expect death rate to be tracked
    kwargs = {**deepcopy(base_kwargs), "birth_approach": BirthApproach.REPLACE_DEATHS}
    model = ModelClass(**kwargs)
    assert model.parameters == {"universal_death_rate": 0}
    assert model.tracked_quantities == {
        "total_deaths": 0,
    }


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_model_flow_setup(ModelClass):
    """
    Ensure model flows are setup with:
        - death flows
        - transition flows
        - transition flows wih custom functions
    """
    mock_func_1 = mock.Mock()
    mock_func_2 = mock.Mock()
    parameters = {
        "infect_death": 1,
        "recovery": 2,
        "slip_on_banana_peel_rate": 3,
        "divine_blessing": 4,
        "cursed": 5,
    }
    requested_flows = [
        # Try some death flows
        {
            "type": Flow.COMPARTMENT_DEATH,
            "parameter": "infect_death",
            "origin": Compartment.INFECTIOUS,
        },
        {
            "type": Flow.COMPARTMENT_DEATH,
            "parameter": "slip_on_banana_peel_rate",
            "origin": Compartment.SUSCEPTIBLE,
        },
        # Try a standard flow
        {
            "type": Flow.STANDARD,
            "parameter": "recovery",
            "origin": Compartment.INFECTIOUS,
            "to": Compartment.SUSCEPTIBLE,
        },
        # Try some custom flows
        {
            "type": Flow.CUSTOM,
            "parameter": "divine_blessing",
            "origin": Compartment.INFECTIOUS,
            "to": Compartment.SUSCEPTIBLE,
            "function": mock_func_1,
        },
        {
            "type": Flow.CUSTOM,
            "parameter": "cursed",
            "origin": Compartment.SUSCEPTIBLE,
            "to": Compartment.INFECTIOUS,
            "function": mock_func_2,
        },
    ]
    model = ModelClass(
        times=_get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS],
        initial_conditions={Compartment.SUSCEPTIBLE: 10, Compartment.INFECTIOUS: 20},
        parameters=parameters,
        requested_flows=requested_flows,
        starting_population=100,
    )
    # Check that death flows were set up properly
    expected_columns = ["type", "parameter", "origin", "implement"]
    expected_data = [
        ["compartment_death", "infect_death", "infectious", 0],
        ["compartment_death", "slip_on_banana_peel_rate", "susceptible", 0],
    ]
    expected_df = pd.DataFrame(expected_data, columns=expected_columns).astype(object)
    assert_frame_equal(expected_df, model.death_flows)
    # Check that transition flows were set up properly
    expected_columns = ["type", "parameter", "origin", "to", "implement", "strain", "force_index"]
    expected_data = [
        ["standard_flows", "recovery", "infectious", "susceptible", 0, None, None],
        ["customised_flows", "divine_blessing", "infectious", "susceptible", 0, None, None],
        ["customised_flows", "cursed", "susceptible", "infectious", 0, None, None],
    ]
    expected_df = pd.DataFrame(expected_data, columns=expected_columns).astype(object)
    assert_frame_equal(expected_df, model.transition_flows)
    # Ensure custom functions were stored.
    assert model.customised_flow_functions[1] == mock_func_1
    assert model.customised_flow_functions[2] == mock_func_2


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_model_compartment_init__with_remainder__expect_correct_allocation(ModelClass):
    """
    Ensure model compartments are set up correctly when there are left over people
    in the population from the intiial conditions setup.
    """
    model = ModelClass(
        times=_get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS, "empty"],
        initial_conditions={Compartment.SUSCEPTIBLE: 10, Compartment.INFECTIOUS: 20},
        parameters={},
        requested_flows=[],
        starting_population=100,
    )
    assert model.compartment_values == [80, 20, 0]


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_model_compartment_init__with_no_remainder__expect_correct_allocation(ModelClass):
    """
    Ensure model compartments are set up correctly when there are no left over people
    in the population from the intitial conditions setup.
    """
    model = ModelClass(
        times=_get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS, "empty"],
        initial_conditions={Compartment.SUSCEPTIBLE: 30, Compartment.INFECTIOUS: 70},
        parameters={},
        requested_flows=[],
        starting_population=100,
    )
    assert model.compartment_values == [30, 70, 0]


bad_inputs = [
    {"starting_population": "this should be an integer"},
    {"times": "this should be a list"},
    {"birth_approach": 0},  # Should be a string
    {"verbose": "this should be a bool"},
    {"derived_output_functions": "this should be a dict"},
    # Infectious compartment not in compartment types
    {"infectious_compartment": ("D",)},
    # Invalid birth approach
    {"birth_approach": "not_a_valid_approach"},
    # Times out of order (seems kind of arbitrary?)
    {"times": [2, 34, 5, 1]},
    # Output connections has wrong keys
    {"output_connections": {"foo": {"bar": 1}}},
    # Initial condition compartment not in compartment types
    {
        "compartment_types": [Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS],
        "initial_conditions": {"this is wrong": 20},
    },
    # Initial condition population exceeds starting pop.
    {"initial_conditions": {Compartment.SUSCEPTIBLE: 99999}, "starting_population": 100,},
]
test_params = []
for bad_input in bad_inputs:
    for model_class in [EpiModel, StratifiedModel]:
        test_params.append([model_class, bad_input])


@pytest.mark.parametrize("ModelClass,bad_input", test_params)
def test_model_input_validation__with_bad_inputs__expect_error(ModelClass, bad_input):
    """
    Ensure bad input types raises a type error.
    """
    inputs = {
        "times": _get_integration_times(2000, 2005, 1),
        "compartment_types": [Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS],
        "initial_conditions": {Compartment.SUSCEPTIBLE: 20},
        "parameters": {},
        "requested_flows": [],
        "starting_population": 100,
        **bad_input,
    }
    with pytest.raises(ValidationException):
        ModelClass(**inputs)


def _get_integration_times(start_year: int, end_year: int, time_step: int):
    """
    Get a list of timesteps from start_year to end_year, spaced by time_step.
    """
    n_iter = int(round((end_year - start_year) / time_step)) + 1
    return np.linspace(start_year, end_year, n_iter).tolist()
