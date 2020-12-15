"""
Unit tests for the Stratification model.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from summer2 import (
    adjust,
    Stratification,
    AgeStratification,
    StrainStratification,
    Compartment,
)


def test_create_stratification():
    strat = Stratification(name="location", strata=["rural", "urban"], compartments=["S", "I", "R"])
    assert strat.name == "location"
    assert strat.compartments == [Compartment("S"), Compartment("I"), Compartment("R")]
    assert strat.strata == ["rural", "urban"]
    assert strat.population_split == {"rural": 0.5, "urban": 0.5}
    assert strat.flow_adjustments == {}
    assert strat.infectiousness_adjustments == {}
    assert strat.mixing_matrix is None
    assert not strat.is_ageing()
    assert not strat.is_strain()


def test_create_strain_stratification():
    strat = StrainStratification(
        name="strain", strata=["mild", "horrible"], compartments=["S", "I", "R"]
    )
    assert not strat.is_ageing()
    assert strat.is_strain()


def test_create_age_stratification():
    strat = AgeStratification(name="age", strata=["0", "5", "10"], compartments=["S", "I", "R"])
    assert strat.is_ageing()
    assert not strat.is_strain()

    # Fails coz non integer stratum
    with pytest.raises(AssertionError):
        AgeStratification(name="age", strata=["0", "hello", "10"], compartments=["S", "I", "R"])

    # Fails coz no zero age
    with pytest.raises(AssertionError):
        AgeStratification(name="age", strata=["1", "5", "10"], compartments=["S", "I", "R"])


def test_create_stratification__with_pop_split():
    strat = Stratification(name="location", strata=["rural", "urban"], compartments=["S", "I", "R"])
    assert strat.population_split == {"rural": 0.5, "urban": 0.5}
    # Works
    strat.set_population_split({"rural": 0.2, "urban": 0.8})
    assert strat.population_split == {"rural": 0.2, "urban": 0.8}

    # Fails coz missing a key
    with pytest.raises(AssertionError):
        strat.set_population_split({"urban": 1})

    # Fails coz doesn't sum to 1
    with pytest.raises(AssertionError):
        strat.set_population_split({"urban": 0.2, "rural": 0.3})

    # Fails coz contains negative number
    with pytest.raises(AssertionError):
        strat.set_population_split({"urban": -2, "rural": 3})


def test_create_stratification__with_flow_adjustments():
    strat = Stratification(name="location", strata=["rural", "urban"], compartments=["S", "I", "R"])
    assert strat.flow_adjustments == {}

    # Fail coz not all strata specified
    with pytest.raises(AssertionError):
        strat.add_flow_adjustments(
            flow_name="recovery",
            adjustments={"rural": adjust.Multiply(1.2)},
        )

    # Fail coz an incorrect strata specified
    with pytest.raises(AssertionError):
        strat.add_flow_adjustments(
            flow_name="recovery",
            adjustments={
                "rural": adjust.Multiply(1.2),
                "urban": adjust.Multiply(0.8),
                "alpine": adjust.Multiply(1.1),
            },
        )

    strat.add_flow_adjustments(
        flow_name="recovery",
        adjustments={"rural": adjust.Multiply(1.2), "urban": adjust.Multiply(0.8)},
    )
    assert strat.flow_adjustments["recovery"]["rural"]._is_equal(adjust.Multiply(1.2))
    assert strat.flow_adjustments["recovery"]["urban"]._is_equal(adjust.Multiply(0.8))

    # Fail coz we just did this.
    with pytest.raises(AssertionError):
        strat.add_flow_adjustments(
            flow_name="recovery",
            adjustments={"rural": adjust.Multiply(1.2), "urban": adjust.Multiply(0.8)},
        )

    def urban_infection_adjustment(t):
        return 2 * t

    strat.add_flow_adjustments(
        flow_name="infection",
        adjustments={
            "rural": adjust.Multiply(urban_infection_adjustment),
            "urban": None,
        },
    )
    assert strat.flow_adjustments["infection"]["rural"]._is_equal(
        adjust.Multiply(urban_infection_adjustment)
    )
    assert strat.flow_adjustments["infection"]["urban"] is None


def test_create_stratification__with_infectiousness_adjustments():
    strat = Stratification(name="location", strata=["rural", "urban"], compartments=["S", "I", "R"])
    assert strat.infectiousness_adjustments == {}

    # Fail coz not all strata specified
    with pytest.raises(AssertionError):
        strat.add_infectiousness_adjustments(
            compartment_name="S",
            adjustments={"rural": adjust.Multiply(1.2)},
        )

    # Fail coz an incorrect strata specified
    with pytest.raises(AssertionError):
        strat.add_infectiousness_adjustments(
            compartment_name="S",
            adjustments={
                "rural": adjust.Multiply(1.2),
                "urban": adjust.Multiply(0.8),
                "alpine": adjust.Multiply(1.1),
            },
        )

    # Fail coz a time-varying function was used (not allowed!)
    with pytest.raises(AssertionError):
        strat.add_infectiousness_adjustments(
            compartment_name="S",
            adjustments={
                "rural": adjust.Multiply(1.2),
                "urban": adjust.Multiply(lambda t: 2),
            },
        )

    strat.add_infectiousness_adjustments(
        compartment_name="S",
        adjustments={
            "rural": adjust.Multiply(1.2),
            "urban": adjust.Multiply(2),
        },
    )

    assert strat.infectiousness_adjustments["S"]["rural"]._is_equal(adjust.Multiply(1.2))
    assert strat.infectiousness_adjustments["S"]["urban"]._is_equal(adjust.Multiply(2))

    # Fail coz we just did this
    with pytest.raises(AssertionError):
        strat.add_infectiousness_adjustments(
            compartment_name="S",
            adjustments={
                "rural": adjust.Multiply(1.2),
                "urban": adjust.Multiply(2),
            },
        )

    strat.add_infectiousness_adjustments(
        compartment_name="I",
        adjustments={
            "rural": adjust.Multiply(1.2),
            "urban": None,
        },
    )

    assert strat.infectiousness_adjustments["I"]["rural"]._is_equal(adjust.Multiply(1.2))
    assert strat.infectiousness_adjustments["I"]["urban"] is None


def test_stratify_compartments__with_no_extisting_strat():
    strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S", "I", "R"],
    )
    comps = [Compartment("S"), Compartment("I"), Compartment("R")]
    strat_comps = strat._stratify_compartments(comps)
    assert strat_comps == [
        Compartment("S", {"age": "0"}),
        Compartment("S", {"age": "10"}),
        Compartment("S", {"age": "20"}),
        Compartment("I", {"age": "0"}),
        Compartment("I", {"age": "10"}),
        Compartment("I", {"age": "20"}),
        Compartment("R", {"age": "0"}),
        Compartment("R", {"age": "10"}),
        Compartment("R", {"age": "20"}),
    ]


def test_stratify_compartments__with_no_extisting_strat_and_subset_only():
    strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S"],
    )
    comps = [Compartment("S"), Compartment("I"), Compartment("R")]
    strat_comps = strat._stratify_compartments(comps)
    assert strat_comps == [
        Compartment("S", {"age": "0"}),
        Compartment("S", {"age": "10"}),
        Compartment("S", {"age": "20"}),
        Compartment("I"),
        Compartment("R"),
    ]


def test_stratify_compartments__with_extisting_strat():
    age_strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S", "I", "R"],
    )
    comps = [Compartment("S"), Compartment("I"), Compartment("R")]
    age_comps = age_strat._stratify_compartments(comps)
    loc_strat = Stratification(
        name="location",
        strata=["rural", "urban"],
        compartments=["S", "I", "R"],
    )
    loc_comps = loc_strat._stratify_compartments(age_comps)
    assert loc_comps == [
        Compartment("S", {"age": "0", "location": "rural"}),
        Compartment("S", {"age": "0", "location": "urban"}),
        Compartment("S", {"age": "10", "location": "rural"}),
        Compartment("S", {"age": "10", "location": "urban"}),
        Compartment("S", {"age": "20", "location": "rural"}),
        Compartment("S", {"age": "20", "location": "urban"}),
        Compartment("I", {"age": "0", "location": "rural"}),
        Compartment("I", {"age": "0", "location": "urban"}),
        Compartment("I", {"age": "10", "location": "rural"}),
        Compartment("I", {"age": "10", "location": "urban"}),
        Compartment("I", {"age": "20", "location": "rural"}),
        Compartment("I", {"age": "20", "location": "urban"}),
        Compartment("R", {"age": "0", "location": "rural"}),
        Compartment("R", {"age": "0", "location": "urban"}),
        Compartment("R", {"age": "10", "location": "rural"}),
        Compartment("R", {"age": "10", "location": "urban"}),
        Compartment("R", {"age": "20", "location": "rural"}),
        Compartment("R", {"age": "20", "location": "urban"}),
    ]


def test_stratify_compartment_values__with_no_extisting_strat():
    strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S", "I", "R"],
    )
    strat.set_population_split({"0": 0.25, "10": 0.5, "20": 0.25})
    comps = [Compartment("S"), Compartment("I"), Compartment("R")]

    comp_values = np.array([1000.0, 100.0, 0.0])
    new_comp_values = strat._stratify_compartment_values(comps, comp_values)
    expected_arr = np.array([250, 500.0, 250.0, 25.0, 50.0, 25.0, 0.0, 0.0, 0.0])
    assert_array_equal(expected_arr, new_comp_values)


def test_stratify_compartment_values__with_subset_stratified():
    strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S"],
    )
    strat.set_population_split({"0": 0.25, "10": 0.5, "20": 0.25})
    comps = [Compartment("S"), Compartment("I"), Compartment("R")]
    comp_values = np.array([1000.0, 100.0, 0.0])
    new_comp_values = strat._stratify_compartment_values(comps, comp_values)
    expected_arr = np.array([250.0, 500.0, 250.0, 100.0, 0.0])
    assert_array_equal(expected_arr, new_comp_values)


def test_stratify_compartment_values__with_extisting_strat():
    """
    Stratify compartments for the second time, expect that compartments
    are are split according to proportions and old compartments are removed.
    """
    comp_values = np.array([250.0, 500.0, 250.0, 25.0, 50.0, 25.0, 0.0, 0.0, 0.0])
    comps = [
        Compartment("S", {"age": "0"}),
        Compartment("S", {"age": "10"}),
        Compartment("S", {"age": "20"}),
        Compartment("I", {"age": "0"}),
        Compartment("I", {"age": "10"}),
        Compartment("I", {"age": "20"}),
        Compartment("R", {"age": "0"}),
        Compartment("R", {"age": "10"}),
        Compartment("R", {"age": "20"}),
    ]
    strat = Stratification(
        name="location",
        strata=["rural", "urban"],
        compartments=["S", "I", "R"],
    )
    strat.set_population_split({"rural": 0.1, "urban": 0.9})
    new_comp_values = strat._stratify_compartment_values(comps, comp_values)
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
