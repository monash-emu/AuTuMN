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
from summer.constants import (
    Compartment,
    Flow,
    BirthApproach,
    IntegrationType,
)
from summer import flow


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_setup_default_parameters(ModelClass):
    """
    Ensure default params are set correctly on model setup.
    """
    base_kwargs = {
        "times": _get_integration_times(2000, 2005, 1),
        "compartment_names": [Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        "initial_conditions": {},
        "parameters": {},
        "requested_flows": [],
        "starting_population": 100,
        "infectious_compartments": [Compartment.EARLY_INFECTIOUS],
        "birth_approach": BirthApproach.NO_BIRTH,
        "entry_compartment": Compartment.SUSCEPTIBLE,
    }

    # Expect univseral death rate to be set by default
    model = ModelClass(**deepcopy(base_kwargs))
    assert model.parameters == {
        "universal_death_rate": 0,
        "crude_birth_rate": 0,
    }

    # Expect univseral death rate to be overidden
    kwargs = {**deepcopy(base_kwargs), "parameters": {"universal_death_rate": 1234}}
    model = ModelClass(**kwargs)
    assert model.parameters == {
        "universal_death_rate": 1234,
        "crude_birth_rate": 0,
    }

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
    assert model.parameters == {
        "universal_death_rate": 0,
        "crude_birth_rate": 0,
    }


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_model_flow_setup(ModelClass):
    """
    Ensure model flows are setup with:
        - death flows
        - transition flows
        - transition flows wih custom functions
    """
    parameters = {
        "infect_death": 1,
        "recovery": "recovery",
        "slip_on_banana_peel_rate": 3,
        "divine_blessing": 4,
        "cursed": 5,
    }
    requested_flows = [
        # Try some death flows
        {"type": Flow.DEATH, "parameter": "infect_death", "origin": Compartment.EARLY_INFECTIOUS,},
        {
            "type": Flow.DEATH,
            "parameter": "slip_on_banana_peel_rate",
            "origin": Compartment.SUSCEPTIBLE,
        },
        # Try a standard flow
        {
            "type": Flow.STANDARD,
            "parameter": "recovery",
            "origin": Compartment.EARLY_INFECTIOUS,
            "to": Compartment.SUSCEPTIBLE,
        },
    ]
    model = ModelClass(
        times=_get_integration_times(2000, 2005, 1),
        compartment_names=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        initial_conditions={Compartment.SUSCEPTIBLE: 10, Compartment.EARLY_INFECTIOUS: 20,},
        parameters=parameters,
        requested_flows=requested_flows,
        birth_approach=BirthApproach.NO_BIRTH,
        starting_population=100,
        infectious_compartments=[Compartment.EARLY_INFECTIOUS],
        entry_compartment=Compartment.SUSCEPTIBLE,
    )
    model.time_variants["recovery"] = lambda t: 2 * t
    # Check that flows were set up properly
    assert model.flows[0].type == Flow.DEATH
    assert model.flows[0].source == "infectious"
    assert model.flows[0].param_name == "infect_death"
    assert model.flows[0].get_weight_value(0) == 1
    assert model.flows[0].get_weight_value(1) == 1

    assert model.flows[1].type == Flow.DEATH
    assert model.flows[1].source == "susceptible"
    assert model.flows[1].param_name == "slip_on_banana_peel_rate"
    assert model.flows[1].get_weight_value(0) == 3
    assert model.flows[1].get_weight_value(1) == 3

    assert model.flows[2].type == Flow.STANDARD
    assert model.flows[2].source == "infectious"
    assert model.flows[2].dest == "susceptible"
    assert model.flows[2].param_name == "recovery"
    assert model.flows[2].get_weight_value(0) == 0
    assert model.flows[2].get_weight_value(1) == 2
    assert model.flows[2].get_weight_value(2) == 4


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_model_compartment_init__with_remainder__expect_correct_allocation(ModelClass):
    """
    Ensure model compartments are set up correctly when there are left over people
    in the population from the intiial conditions setup.
    """
    model = ModelClass(
        times=_get_integration_times(2000, 2005, 1),
        compartment_names=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS, "empty"],
        initial_conditions={Compartment.SUSCEPTIBLE: 10, Compartment.EARLY_INFECTIOUS: 20},
        parameters={},
        requested_flows=[],
        birth_approach=BirthApproach.NO_BIRTH,
        starting_population=100,
        infectious_compartments=[Compartment.EARLY_INFECTIOUS],
        entry_compartment=Compartment.SUSCEPTIBLE,
    )
    assert model.compartment_values.tolist() == [80, 20, 0]


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_model_compartment_init__with_remainder__expect_correct_allocation_to_alt_entry_comp(
    ModelClass,
):
    """
    Ensure model compartments are set up correctly when there are left over people
    in the population from the intiial conditions setup.
    """
    model = ModelClass(
        times=_get_integration_times(2000, 2005, 1),
        compartment_names=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS, "entry"],
        initial_conditions={Compartment.SUSCEPTIBLE: 10, Compartment.EARLY_INFECTIOUS: 20},
        parameters={},
        requested_flows=[],
        birth_approach=BirthApproach.NO_BIRTH,
        starting_population=100,
        infectious_compartments=[Compartment.EARLY_INFECTIOUS],
        entry_compartment="entry",
    )
    assert model.compartment_values.tolist() == [10, 20, 70]


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_model_compartment_init__with_no_remainder__expect_correct_allocation(ModelClass,):
    """
    Ensure model compartments are set up correctly when there are no left over people
    in the population from the intitial conditions setup.
    """
    model = ModelClass(
        times=_get_integration_times(2000, 2005, 1),
        compartment_names=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS, "empty",],
        initial_conditions={Compartment.SUSCEPTIBLE: 30, Compartment.EARLY_INFECTIOUS: 70,},
        parameters={},
        requested_flows=[],
        birth_approach=BirthApproach.NO_BIRTH,
        starting_population=100,
        infectious_compartments=[Compartment.EARLY_INFECTIOUS],
        entry_compartment=Compartment.SUSCEPTIBLE,
    )
    assert model.compartment_values.tolist() == [30, 70, 0]


bad_inputs = [
    {"starting_population": "this should be an integer"},
    {"birth_approach": 0},  # Should be a string
    # Infectious compartment not in compartment types
    {"infectious_compartments": ("D",)},
    # Invalid birth approach
    {"birth_approach": "not_a_valid_approach"},
    # Initial condition compartment not in compartment types
    {
        "compartment_names": [Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
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
        "compartment_names": [Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        "initial_conditions": {Compartment.SUSCEPTIBLE: 20},
        "parameters": {},
        "requested_flows": [],
        "starting_population": 100,
        "infectious_compartments": [Compartment.EARLY_INFECTIOUS],
        "birth_approach": BirthApproach.NO_BIRTH,
        "entry_compartment": Compartment.SUSCEPTIBLE,
        **bad_input,
    }
    with pytest.raises(ValidationException):
        ModelClass(**inputs)


bad_time_inputs = [
    # Times not numpy array
    "this should be a Numpy array",
    [1.0, 2.0, 3.0, 4.0],
    # Times wrong data type
    np.array([1, 2, 3, 4]),
    # Times out of order (seems kind of arbitrary?)
    np.array([2.0, 34.0, 5.0, 1.0]),
]
time_test_params = []
for bad_input in bad_time_inputs:
    for model_class in [EpiModel, StratifiedModel]:
        time_test_params.append([model_class, bad_input])


@pytest.mark.parametrize("ModelClass,bad_input", time_test_params)
def test_model_input_validation__with_bad_time_inputs__expect_error(ModelClass, bad_input):
    """
    Ensure bad input types raises a type error.
    """
    inputs = {
        "compartment_names": [Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        "initial_conditions": {Compartment.SUSCEPTIBLE: 20},
        "parameters": {},
        "requested_flows": [],
        "starting_population": 100,
        "infectious_compartments": [Compartment.EARLY_INFECTIOUS],
        "birth_approach": BirthApproach.NO_BIRTH,
        "entry_compartment": Compartment.SUSCEPTIBLE,
        "times": time_test_params,
    }
    with pytest.raises(AssertionError):
        ModelClass(**inputs)


def _get_integration_times(start_year: int, end_year: int, time_step: int):
    """
    Get a list of timesteps from start_year to end_year, spaced by time_step.
    """
    n_iter = int(round((end_year - start_year) / time_step)) + 1
    return np.linspace(start_year, end_year, n_iter)
