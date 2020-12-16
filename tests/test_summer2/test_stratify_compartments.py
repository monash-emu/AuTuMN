"""
Test applying a Stratification to a CompartmentalModel via stratify_with  updates compartments correctly.
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from summer2 import Stratification, CompartmentalModel, Compartment


def test_stratify__single__validate_compartments():
    """
    Ensure stratifying a model correctly adjusts the model compartments.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population({"S": 900, "I": 90, "R": 10})
    # Compartments exist
    assert model.compartments == [Compartment("S"), Compartment("I"), Compartment("R")]
    # Each compartment knows its index
    assert [c.idx for c in model.compartments] == list(range(len(model.compartments)))
    # Compartments have the correct population
    assert_array_equal(model.initial_population, np.array([900, 90, 10]))

    # Stratify the model
    strat = Stratification(name="age", strata=["child", "adult"], compartments=["S", "I", "R"])
    model.stratify_with(strat)
    assert model._stratifications == [strat]

    # Ensure compartments are stratified correctly
    assert [c.idx for c in model.compartments] == list(range(len(model.compartments)))
    assert model.compartments == [
        Compartment("S", {"age": "child"}),
        Compartment("S", {"age": "adult"}),
        Compartment("I", {"age": "child"}),
        Compartment("I", {"age": "adult"}),
        Compartment("R", {"age": "child"}),
        Compartment("R", {"age": "adult"}),
    ]
    expected_pop_arr = np.array([450, 450, 45, 45, 5, 5])
    assert_array_equal(model.initial_population, expected_pop_arr)


def test_stratify__single_with_pop_split__validate_compartments():
    """
    Ensure stratifying a model correctly adjusts the model compartments.
    Also the population split should be applied.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population({"S": 900, "I": 90, "R": 10})
    # Compartments exist
    assert model.compartments == [Compartment("S"), Compartment("I"), Compartment("R")]
    # Each compartment knows its index
    assert [c.idx for c in model.compartments] == list(range(len(model.compartments)))
    # Compartments have the correct population
    assert_array_equal(model.initial_population, np.array([900, 90, 10]))

    # Stratify the model
    strat = Stratification(name="age", strata=["child", "adult"], compartments=["S", "I", "R"])
    strat.set_population_split({"child": 0.8, "adult": 0.2})
    model.stratify_with(strat)
    assert model._stratifications == [strat]

    # Ensure compartments are stratified correctly
    assert [c.idx for c in model.compartments] == list(range(len(model.compartments)))
    assert model.compartments == [
        Compartment("S", {"age": "child"}),
        Compartment("S", {"age": "adult"}),
        Compartment("I", {"age": "child"}),
        Compartment("I", {"age": "adult"}),
        Compartment("R", {"age": "child"}),
        Compartment("R", {"age": "adult"}),
    ]
    expected_pop_arr = np.array([720, 180, 72, 18, 8, 2])
    assert_array_equal(model.initial_population, expected_pop_arr)


def test_stratify__single_with_split_and_partial__validate_compartments():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population({"S": 900, "I": 90, "R": 10})
    # Compartments exist
    assert model.compartments == [Compartment("S"), Compartment("I"), Compartment("R")]
    # Each compartment knows its index
    assert [c.idx for c in model.compartments] == list(range(len(model.compartments)))
    # Compartments have the correct population
    assert_array_equal(model.initial_population, np.array([900, 90, 10]))

    # Stratify the model
    strat = Stratification(name="age", strata=["child", "adult"], compartments=["S", "R"])
    strat.set_population_split({"child": 0.8, "adult": 0.2})
    model.stratify_with(strat)
    assert model._stratifications == [strat]

    # Ensure compartments are stratified correctly
    assert [c.idx for c in model.compartments] == list(range(len(model.compartments)))
    assert model.compartments == [
        Compartment("S", {"age": "child"}),
        Compartment("S", {"age": "adult"}),
        Compartment("I"),
        Compartment("R", {"age": "child"}),
        Compartment("R", {"age": "adult"}),
    ]
    expected_pop_arr = np.array([720, 180, 90, 8, 2])
    assert_array_equal(model.initial_population, expected_pop_arr)


def test_stratify__double_with_split_and_partial__validate_compartments():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population({"S": 900, "I": 90, "R": 10})
    # Compartments exist
    assert model.compartments == [Compartment("S"), Compartment("I"), Compartment("R")]
    # Each compartment knows its index
    assert [c.idx for c in model.compartments] == list(range(len(model.compartments)))
    # Compartments have the correct population
    assert_array_equal(model.initial_population, np.array([900, 90, 10]))

    # Stratify the model
    age_strat = Stratification(name="age", strata=["child", "adult"], compartments=["S", "R"])
    age_strat.set_population_split({"child": 0.8, "adult": 0.2})
    model.stratify_with(age_strat)
    assert model._stratifications == [age_strat]

    # Ensure compartments are stratified correctly
    assert [c.idx for c in model.compartments] == list(range(len(model.compartments)))
    assert model.compartments == [
        Compartment("S", {"age": "child"}),
        Compartment("S", {"age": "adult"}),
        Compartment("I"),
        Compartment("R", {"age": "child"}),
        Compartment("R", {"age": "adult"}),
    ]
    expected_pop_arr = np.array([720, 180, 90, 8, 2])
    assert_array_equal(model.initial_population, expected_pop_arr)

    # Stratify the model again!
    loc_strat = Stratification(
        name="location", strata=["urban", "rural", "alpine"], compartments=["S", "I"]
    )
    loc_strat.set_population_split({"urban": 0.7, "rural": 0.2, "alpine": 0.1})
    model.stratify_with(loc_strat)
    assert model._stratifications == [age_strat, loc_strat]

    # Ensure compartments are stratified correctly
    assert [c.idx for c in model.compartments] == list(range(len(model.compartments)))
    assert model.compartments == [
        Compartment("S", {"age": "child", "location": "urban"}),
        Compartment("S", {"age": "child", "location": "rural"}),
        Compartment("S", {"age": "child", "location": "alpine"}),
        Compartment("S", {"age": "adult", "location": "urban"}),
        Compartment("S", {"age": "adult", "location": "rural"}),
        Compartment("S", {"age": "adult", "location": "alpine"}),
        Compartment("I", {"location": "urban"}),
        Compartment("I", {"location": "rural"}),
        Compartment("I", {"location": "alpine"}),
        Compartment("R", {"age": "child"}),
        Compartment("R", {"age": "adult"}),
    ]
    expected_pop_arr = np.array([504, 144, 72, 126, 36, 18, 63, 18, 9, 8, 2])
    assert_allclose(model.initial_population, expected_pop_arr, atol=1e-9, rtol=0)
