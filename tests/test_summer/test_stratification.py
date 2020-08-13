import pytest
import numpy as np
from numpy.testing import assert_array_equal

from summer.compartment import Compartment
from summer.constants import FlowAdjustment
from summer.stratification import (
    Stratification,
    get_stratified_compartment_names,
    get_stratified_compartment_values,
)


def test_stratification_with_basic_setup():
    """
    Ensure we can create a simple stratification.
    """
    strat = Stratification(
        name="foo",
        strata=[1, "2", "bar", "baz"],
        compartments=["sus", "inf", "rec"],
        comp_split_props={},
        flow_adjustments={},
    )
    assert strat.name == "foo"
    assert strat.compartments == [Compartment("sus"), Compartment("inf"), Compartment("rec")]
    assert strat.strata == ["1", "2", "bar", "baz"]
    assert strat.comp_split_props == {"1": 0.25, "2": 0.25, "bar": 0.25, "baz": 0.25}
    assert strat.flow_adjustments == {}


def test_stratification_with_compartment_split_proportions():
    """
    Ensure we can create a stratification that parses compartment splits.
    """
    strat = Stratification(
        name="foo",
        strata=["1", "2", "3"],
        compartments=["sus", "inf", "rec"],
        comp_split_props={"1": 0.2, "2": 0.2},
        flow_adjustments={},
    )
    assert strat.comp_split_props == {
        "1": 0.2,
        "2": 0.2,
        "3": 0.6,
    }


def test_stratification_with_flow_adjustments():
    """
    Ensure we can create a stratification that parses flow adjustments.
    """
    strat = Stratification(
        name="foo",
        strata=["1", "2", "3"],
        compartments=["sus", "inf", "rec"],
        comp_split_props={},
        flow_adjustments={
            "infect_death": {"2": 0.5},
            "infect_deathXlocation_work": {"1": 0.9},
            "contact_rateXage_10": {"1": 0.5, "2W": 0.2},
            "contact_rateXage_20": {"1": 0.5, "3": "some_function"},
        },
    )
    assert strat.flow_adjustments == {
        "infect_death": [
            {"strata": {}, "adjustments": {"2": (FlowAdjustment.MULTIPLY, 0.5)}},
            {"strata": {"location": "work"}, "adjustments": {"1": (FlowAdjustment.MULTIPLY, 0.9)}},
        ],
        "contact_rate": [
            {
                "strata": {"age": "10"},
                "adjustments": {
                    "1": (FlowAdjustment.MULTIPLY, 0.5),
                    "2": (FlowAdjustment.OVERWRITE, 0.2),
                },
            },
            {
                "strata": {"age": "20"},
                "adjustments": {
                    "1": (FlowAdjustment.MULTIPLY, 0.5),
                    "3": (FlowAdjustment.COMPOSE, "some_function"),
                },
            },
        ],
    }


def test_get_stratified_compartment_names__with_no_extisting_strat():
    strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S", "I", "R"],
        comp_split_props={},
        flow_adjustments={},
    )
    comp_names = [Compartment("S"), Compartment("I"), Compartment("R")]
    strat_comp_names = get_stratified_compartment_names(strat, comp_names)
    assert strat_comp_names == [
        Compartment("S", strat_names=["age"], strat_values={"age": "0"}),
        Compartment("S", strat_names=["age"], strat_values={"age": "10"}),
        Compartment("S", strat_names=["age"], strat_values={"age": "20"}),
        Compartment("I", strat_names=["age"], strat_values={"age": "0"}),
        Compartment("I", strat_names=["age"], strat_values={"age": "10"}),
        Compartment("I", strat_names=["age"], strat_values={"age": "20"}),
        Compartment("R", strat_names=["age"], strat_values={"age": "0"}),
        Compartment("R", strat_names=["age"], strat_values={"age": "10"}),
        Compartment("R", strat_names=["age"], strat_values={"age": "20"}),
    ]


def test_get_stratified_compartment_names__with_no_extisting_strat_and_subset_only():
    strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S"],
        comp_split_props={},
        flow_adjustments={},
    )
    comp_names = [Compartment("S"), Compartment("I"), Compartment("R")]
    strat_comp_names = get_stratified_compartment_names(strat, comp_names)
    assert strat_comp_names == [
        Compartment("S", strat_names=["age"], strat_values={"age": "0"}),
        Compartment("S", strat_names=["age"], strat_values={"age": "10"}),
        Compartment("S", strat_names=["age"], strat_values={"age": "20"}),
        Compartment("I"),
        Compartment("R"),
    ]


def test_get_stratified_compartment_names__with_extisting_strat():
    age_strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S", "I", "R"],
        comp_split_props={},
        flow_adjustments={},
    )
    comp_names = [Compartment("S"), Compartment("I"), Compartment("R")]
    age_comp_names = get_stratified_compartment_names(age_strat, comp_names)
    loc_strat = Stratification(
        name="location",
        strata=["rural", "urban"],
        compartments=["S", "I", "R"],
        comp_split_props={},
        flow_adjustments={},
    )
    loc_comp_names = get_stratified_compartment_names(loc_strat, age_comp_names)
    assert loc_comp_names == [
        Compartment("S", ["age", "location"], {"age": "0", "location": "rural"}),
        Compartment("S", ["age", "location"], {"age": "0", "location": "urban"}),
        Compartment("S", ["age", "location"], {"age": "10", "location": "rural"}),
        Compartment("S", ["age", "location"], {"age": "10", "location": "urban"}),
        Compartment("S", ["age", "location"], {"age": "20", "location": "rural"}),
        Compartment("S", ["age", "location"], {"age": "20", "location": "urban"}),
        Compartment("I", ["age", "location"], {"age": "0", "location": "rural"}),
        Compartment("I", ["age", "location"], {"age": "0", "location": "urban"}),
        Compartment("I", ["age", "location"], {"age": "10", "location": "rural"}),
        Compartment("I", ["age", "location"], {"age": "10", "location": "urban"}),
        Compartment("I", ["age", "location"], {"age": "20", "location": "rural"}),
        Compartment("I", ["age", "location"], {"age": "20", "location": "urban"}),
        Compartment("R", ["age", "location"], {"age": "0", "location": "rural"}),
        Compartment("R", ["age", "location"], {"age": "0", "location": "urban"}),
        Compartment("R", ["age", "location"], {"age": "10", "location": "rural"}),
        Compartment("R", ["age", "location"], {"age": "10", "location": "urban"}),
        Compartment("R", ["age", "location"], {"age": "20", "location": "rural"}),
        Compartment("R", ["age", "location"], {"age": "20", "location": "urban"}),
    ]


def test_get_stratified_compartment_values__with_no_extisting_strat():
    strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S", "I", "R"],
        comp_split_props={"0": 0.25, "10": 0.5, "20": 0.25},
        flow_adjustments={},
    )
    comp_names = [Compartment("S"), Compartment("I"), Compartment("R")]
    comp_values = np.array([1000.0, 100.0, 0.0])
    new_comp_values = get_stratified_compartment_values(strat, comp_names, comp_values)
    expected_arr = np.array([250, 500.0, 250.0, 25.0, 50.0, 25.0, 0.0, 0.0, 0.0])
    assert_array_equal(expected_arr, new_comp_values)


def test_get_stratified_compartment_values__with_subset_stratified():
    strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S"],
        comp_split_props={"0": 0.25, "10": 0.5, "20": 0.25},
        flow_adjustments={},
    )
    comp_names = [Compartment("S"), Compartment("I"), Compartment("R")]
    comp_values = np.array([1000.0, 100.0, 0.0])
    new_comp_values = get_stratified_compartment_values(strat, comp_names, comp_values)
    expected_arr = np.array([250.0, 500.0, 250.0, 100.0, 0.0])
    assert_array_equal(expected_arr, new_comp_values)


def test_get_stratified_compartment_values__with_extisting_strat():
    """
    Stratify compartments for the second time, expect that compartments
    are are split according to proportions and old compartments are removed.
    """
    comp_values = np.array([250.0, 500.0, 250.0, 25.0, 50.0, 25.0, 0.0, 0.0, 0.0])
    comp_names = [
        Compartment("S", strat_names=["age"], strat_values={"age": "0"}),
        Compartment("S", strat_names=["age"], strat_values={"age": "10"}),
        Compartment("S", strat_names=["age"], strat_values={"age": "20"}),
        Compartment("I", strat_names=["age"], strat_values={"age": "0"}),
        Compartment("I", strat_names=["age"], strat_values={"age": "10"}),
        Compartment("I", strat_names=["age"], strat_values={"age": "20"}),
        Compartment("R", strat_names=["age"], strat_values={"age": "0"}),
        Compartment("R", strat_names=["age"], strat_values={"age": "10"}),
        Compartment("R", strat_names=["age"], strat_values={"age": "20"}),
    ]
    strat = Stratification(
        name="location",
        strata=["rural", "urban"],
        compartments=["S", "I", "R"],
        comp_split_props={"rural": 0.1, "urban": 0.9},
        flow_adjustments={},
    )
    new_comp_values = get_stratified_compartment_values(strat, comp_names, comp_values)
    expected_arr = np.array(
        [
            25,
            225.0,
            50.0,
            450.0,
            25.0,
            225.0,
            2.5,
            22.5,
            5.0,
            45.0,
            2.5,
            22.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert_array_equal(expected_arr, new_comp_values)


@pytest.mark.parametrize(
    "param_name, stratum, comp, adjustment",
    [
        [
            "contact_rate",
            "0",
            Compartment("S", strat_names=["age"], strat_values={"age": "10"}),
            None,
        ],
        [
            "contact_rate",
            "1",
            Compartment("S", strat_names=["age"], strat_values={"age": "10"}),
            (FlowAdjustment.MULTIPLY, 0.5),
        ],
        [
            "contact_rate",
            "1",
            Compartment(
                "S", strat_names=["age", "location"], strat_values={"age": "10", "location": "work"}
            ),
            None,
        ],
        [
            "contact_rate",
            "2",
            Compartment("S", strat_names=["age"], strat_values={"age": "10"}),
            (FlowAdjustment.OVERWRITE, 0.2),
        ],
        [
            "contact_rate",
            "1",
            Compartment("S", strat_names=["age"], strat_values={"age": "20"}),
            (FlowAdjustment.MULTIPLY, 0.6),
        ],
        [
            "contact_rate",
            "3",
            Compartment("S", strat_names=["age"], strat_values={"age": "20"}),
            (FlowAdjustment.COMPOSE, "some_function"),
        ],
        ["infect_death", "2", Compartment("S"), (FlowAdjustment.MULTIPLY, 0.5),],
        [
            "infect_death",
            "1",
            Compartment("S", strat_names=["location"], strat_values={"location": "work"}),
            (FlowAdjustment.MULTIPLY, 0.9),
        ],
    ],
)
def test_get_flow_adjustment(param_name, stratum, comp, adjustment):
    """
    Ensure we can create a stratification that parses flow adjustments.
    """
    strat = Stratification(
        name="foo",
        strata=["1", "2", "3"],
        compartments=["sus", "inf", "rec"],
        comp_split_props={},
        flow_adjustments={},
    )
    strat.flow_adjustments = {
        "infect_death": [
            {"strata": {}, "adjustments": {"2": (FlowAdjustment.MULTIPLY, 0.5)}},
            {"strata": {"location": "work"}, "adjustments": {"1": (FlowAdjustment.MULTIPLY, 0.9)}},
        ],
        "contact_rate": [
            {
                "strata": {"age": "10"},
                "adjustments": {
                    "1": (FlowAdjustment.MULTIPLY, 0.5),
                    "2": (FlowAdjustment.OVERWRITE, 0.2),
                },
            },
            {
                "strata": {"age": "20"},
                "adjustments": {
                    "1": (FlowAdjustment.MULTIPLY, 0.6),
                    "3": (FlowAdjustment.COMPOSE, "some_function"),
                },
            },
        ],
    }
    actual_adj = strat.get_flow_adjustment(comp, stratum, param_name)
    assert actual_adj == adjustment
