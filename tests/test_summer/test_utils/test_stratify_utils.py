import pytest

from summer.model.utils import get_all_proportions, get_stratified_compartments


PROPS_TEST_CASES = [
    # Test no change
    [
        ["foo", "bar", "baz"],
        {"foo": 0.3, "bar": 0.3, "baz": 0.4},
        {"foo": 0.3, "bar": 0.3, "baz": 0.4},
    ],
    # Test no props
    [["foo", "bar"], {}, {"foo": 0.5, "bar": 0.5,},],
    # Test some missing
    [["foo", "bar", "baz"], {"baz": 0.4}, {"foo": 0.3, "bar": 0.3, "baz": 0.4},],
]


@pytest.mark.parametrize("names, props, expected_props", PROPS_TEST_CASES)
def test_get_all_proportions(names, props, expected_props):
    """
    Ensure get_all_proportions adjusts proportions correctly.
    """
    actual_props = get_all_proportions(names, props)
    assert actual_props == expected_props


def test_get_stratified_compartments__with_no_extisting_strat():
    """
    Stratify compartments for the first time, expect that compartments
    are are split according to proportions and old compartments are removed.
    """
    actual_to_add, actual_to_remove = get_stratified_compartments(
        stratification_name="age",
        strata_names=["0", "10", "20"],
        stratified_compartments=["S", "I", "R"],
        split_proportions={"0": 0.25, "10": 0.5, "20": 0.25,},
        current_names=["S", "I", "R"],
        current_values=[1000, 100, 0],
    )
    assert actual_to_add == {
        "SXage_0": 250,
        "SXage_10": 500,
        "SXage_20": 250,
        "IXage_0": 25,
        "IXage_10": 50,
        "IXage_20": 25,
        "RXage_0": 0,
        "RXage_10": 0,
        "RXage_20": 0,
    }
    assert actual_to_remove == ["S", "I", "R"]


def test_get_stratified_compartments__with_subset_stratified():
    """
    Stratify subset of compartments for the first time, expect that compartments
    are are split according to proportions and old compartments are removed.
    Also only the "S" compartment should be stratified.
    """
    actual_to_add, actual_to_remove = get_stratified_compartments(
        stratification_name="age",
        strata_names=["0", "10", "20"],
        stratified_compartments=["S"],
        split_proportions={"0": 0.25, "10": 0.5, "20": 0.25,},
        current_names=["S", "I", "R"],
        current_values=[1000, 100, 0],
    )
    assert actual_to_add == {
        "SXage_0": 250,
        "SXage_10": 500,
        "SXage_20": 250,
    }
    assert actual_to_remove == ["S"]


def test_get_stratified_compartments__with_extisting_strat():
    """
    Stratify compartments for the second time, expect that compartments
    are are split according to proportions and old compartments are removed.
    """
    actual_to_add, actual_to_remove = get_stratified_compartments(
        stratification_name="location",
        strata_names=["rural", "urban"],
        stratified_compartments=["S", "I", "R"],
        split_proportions={"rural": 0.1, "urban": 0.9},
        current_names=[
            "SXage_0",
            "SXage_10",
            "SXage_20",
            "IXage_0",
            "IXage_10",
            "IXage_20",
            "RXage_0",
            "RXage_10",
            "RXage_20",
        ],
        current_values=[250, 500, 250, 25, 50, 25, 0, 0, 0],
    )
    assert actual_to_add == {
        "SXage_0Xlocation_rural": 25,
        "SXage_0Xlocation_urban": 225,
        "SXage_10Xlocation_rural": 50,
        "SXage_10Xlocation_urban": 450,
        "SXage_20Xlocation_rural": 25,
        "SXage_20Xlocation_urban": 225,
        "IXage_0Xlocation_rural": 2.5,
        "IXage_0Xlocation_urban": 22.5,
        "IXage_10Xlocation_rural": 5,
        "IXage_10Xlocation_urban": 45,
        "IXage_20Xlocation_rural": 2.5,
        "IXage_20Xlocation_urban": 22.5,
        "RXage_0Xlocation_rural": 0,
        "RXage_0Xlocation_urban": 0,
        "RXage_10Xlocation_rural": 0,
        "RXage_10Xlocation_urban": 0,
        "RXage_20Xlocation_rural": 0,
        "RXage_20Xlocation_urban": 0,
    }
    assert actual_to_remove == [
        "SXage_0",
        "SXage_10",
        "SXage_20",
        "IXage_0",
        "IXage_10",
        "IXage_20",
        "RXage_0",
        "RXage_10",
        "RXage_20",
    ]
