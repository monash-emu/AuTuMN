import pytest

from summer.constants import Flow
from summer.model.utils import (
    get_all_proportions,
    get_stratified_compartments,
    parse_param_adjustment_overwrite,
    create_ageing_flows,
    stratify_transition_flows,
    stratify_entry_flows,
)


def test_stratify_stratify_entry_flows__with_age_strata():
    param_updates, time_variant_updates = stratify_entry_flows(
        stratification_name="age",
        strata_names=["0", "10", "20"],
        entry_proportions={"0": 0.2, "10": 0.8, "20": 0},
        time_variant_funcs={},
    )
    assert time_variant_updates == {}
    # Just ignore any provided proprotions
    assert param_updates == {
        "entry_fractionXage_0": 1,
        "entry_fractionXage_10": 0,
        "entry_fractionXage_20": 0,
    }


def test_stratify_stratify_entry_flows__with_entry_proprotions():
    param_updates, time_variant_updates = stratify_entry_flows(
        stratification_name="test",
        strata_names=["foo", "bar"],
        entry_proportions={"foo": 0.2, "bar": 0.8},
        time_variant_funcs={},
    )
    assert time_variant_updates == {}
    assert param_updates == {
        "entry_fractionXtest_foo": 0.2,
        "entry_fractionXtest_bar": 0.8,
    }


def test_stratify_stratify_entry_flows__with_unspecified_entry_proprotions():
    param_updates, time_variant_updates = stratify_entry_flows(
        stratification_name="test",
        strata_names=["foo", "bar"],
        entry_proportions={"foo": 0.2},
        time_variant_funcs={},
    )
    assert time_variant_updates == {}
    # FIXME: I'm not saying that this is a good idea, it's just the current behaviour
    assert param_updates == {
        "entry_fractionXtest_foo": 0.2,
        "entry_fractionXtest_bar": 0.5,
    }


def test_stratify_stratify_entry_flows__with_time_varying_entry_proprotions():
    param_updates, time_variant_updates = stratify_entry_flows(
        stratification_name="test",
        strata_names=["foo", "bar"],
        entry_proportions={"foo": 0.2, "bar": "plus_one"},
        time_variant_funcs={"plus_one": lambda x: x + 1},
    )
    assert time_variant_updates["entry_fractionXtest_bar"](1) == 2
    assert param_updates == {
        "entry_fractionXtest_foo": 0.2,
    }


def test_stratify_transition_flows__with_adjustments():
    """
    Ensure correct flows and params are returned when adjustments are used
    """
    flows = [
        {
            "type": "standard_flows",
            "parameter": "flow_0",
            "origin": "susceptible",
            "to": "infectious",
            "implement": 0,
            "strain": None,
        },
        {
            "type": "standard_flows",
            "parameter": "flow_1",
            "origin": "infectious",
            "to": "susceptible",
            "implement": 0,
            "strain": None,
        },
    ]
    params = {
        "flow_0": 0.5,
        "flow_1": 0.1,
    }
    adjustments = {"flow_0": {"foo": 0.2, "bar": 0.7,}}
    compartments_to_stratify = ["susceptible", "infectious"]
    (
        new_flows,
        overwritten_parameter_adjustment_names,
        param_updates,
        adaptation_function_updates,
    ) = stratify_transition_flows(
        "test",
        strata_names=["foo", "bar"],
        adjustment_requests=adjustments,
        compartments_to_stratify=compartments_to_stratify,
        transition_flows=flows,
        implement_count=1,
    )
    assert adaptation_function_updates == {}
    assert overwritten_parameter_adjustment_names == []
    assert param_updates == {"flow_0Xtest_foo": 0.2, "flow_0Xtest_bar": 0.7}
    assert new_flows == [
        {
            "type": "standard_flows",
            "parameter": "flow_0Xtest_foo",
            "origin": "susceptibleXtest_foo",
            "to": "infectiousXtest_foo",
            "implement": 1,
            "strain": None,
        },
        {
            "type": "standard_flows",
            "parameter": "flow_0Xtest_bar",
            "origin": "susceptibleXtest_bar",
            "to": "infectiousXtest_bar",
            "implement": 1,
            "strain": None,
        },
        {
            "type": "standard_flows",
            "parameter": "flow_1",
            "origin": "infectiousXtest_foo",
            "to": "susceptibleXtest_foo",
            "implement": 1,
            "strain": None,
        },
        {
            "type": "standard_flows",
            "parameter": "flow_1",
            "origin": "infectiousXtest_bar",
            "to": "susceptibleXtest_bar",
            "implement": 1,
            "strain": None,
        },
    ]


def test_stratify_transition_flows__with_adjustments_and_partial_strat():
    """
    Ensure correct flows and params are returned when adjustments are used
    but only one compartment is stratified.
    """
    flows = [
        {
            "type": "standard_flows",
            "parameter": "flow_0",
            "origin": "susceptible",
            "to": "infectious",
            "implement": 0,
            "strain": None,
        },
        {
            "type": "standard_flows",
            "parameter": "flow_1",
            "origin": "infectious",
            "to": "susceptible",
            "implement": 0,
            "strain": None,
        },
    ]
    params = {
        "flow_0": 0.5,
        "flow_1": 0.1,
    }
    adjustments = {"flow_0": {"foo": 0.2, "bar": 0.7,}}
    compartments_to_stratify = ["susceptible"]
    (
        new_flows,
        overwritten_parameter_adjustment_names,
        param_updates,
        adaptation_function_updates,
    ) = stratify_transition_flows(
        "test",
        strata_names=["foo", "bar"],
        adjustment_requests=adjustments,
        compartments_to_stratify=compartments_to_stratify,
        transition_flows=flows,
        implement_count=1,
    )
    assert adaptation_function_updates["flow_1Xtest_foo"](1, 0) == 0.5
    assert adaptation_function_updates["flow_1Xtest_bar"](1, 0) == 0.5
    assert overwritten_parameter_adjustment_names == []
    assert param_updates == {
        "flow_0Xtest_foo": 0.2,
        "flow_0Xtest_bar": 0.7,
        "flow_1Xtest_foo": 0.5,
        "flow_1Xtest_bar": 0.5,
    }
    assert new_flows == [
        {
            "type": "standard_flows",
            "parameter": "flow_0Xtest_foo",
            "origin": "susceptibleXtest_foo",
            "to": "infectious",
            "implement": 1,
            "strain": None,
        },
        {
            "type": "standard_flows",
            "parameter": "flow_0Xtest_bar",
            "origin": "susceptibleXtest_bar",
            "to": "infectious",
            "implement": 1,
            "strain": None,
        },
        {
            "type": "standard_flows",
            "parameter": "flow_1Xtest_foo",
            "origin": "infectious",
            "to": "susceptibleXtest_foo",
            "implement": 1,
            "strain": None,
        },
        {
            "type": "standard_flows",
            "parameter": "flow_1Xtest_bar",
            "origin": "infectious",
            "to": "susceptibleXtest_bar",
            "implement": 1,
            "strain": None,
        },
    ]


def test_stratify_transition_flows__with_custom_flow_funcs():
    """
    Ensure correct flows and params are returned when adjustments are used
    """
    flows = [
        {
            "type": "customised_flows",
            "parameter": "flow_0",
            "origin": "susceptible",
            "to": "infectious",
            "implement": 0,
            "strain": None,
        },
        {
            "type": "customised_flows",
            "parameter": "flow_1",
            "origin": "infectious",
            "to": "susceptible",
            "implement": 0,
            "strain": None,
        },
        {
            "type": "customised_flows",
            "parameter": "flow_2",
            "origin": "comp_0",
            "to": "comp_1",
            "implement": 0,
            "strain": None,
        },
    ]
    params = {"flow_0": 0.5, "flow_1": 0.1, "flow_2": 1.9}
    adjustments = {}
    compartments_to_stratify = ["susceptible"]
    (
        new_flows,
        overwritten_parameter_adjustment_names,
        param_updates,
        adaptation_function_updates,
    ) = stratify_transition_flows(
        "test",
        strata_names=["foo", "bar"],
        adjustment_requests=adjustments,
        compartments_to_stratify=compartments_to_stratify,
        transition_flows=flows,
        implement_count=1,
    )
    assert adaptation_function_updates["flow_1Xtest_foo"](1, 0) == 0.5
    assert adaptation_function_updates["flow_1Xtest_bar"](1, 0) == 0.5
    assert adaptation_function_updates["flow_0Xtest_foo"](1, 0) == 0.5
    assert adaptation_function_updates["flow_0Xtest_bar"](1, 0) == 0.5
    assert overwritten_parameter_adjustment_names == []
    assert param_updates == {
        "flow_0Xtest_foo": 0.5,
        "flow_0Xtest_bar": 0.5,
        "flow_1Xtest_foo": 0.5,
        "flow_1Xtest_bar": 0.5,
    }
    assert new_flows == [
        {
            "type": "customised_flows",
            "parameter": "flow_0Xtest_foo",
            "origin": "susceptibleXtest_foo",
            "to": "infectious",
            "implement": 1,
            "strain": None,
        },
        {
            "type": "customised_flows",
            "parameter": "flow_0Xtest_bar",
            "origin": "susceptibleXtest_bar",
            "to": "infectious",
            "implement": 1,
            "strain": None,
        },
        {
            "type": "customised_flows",
            "parameter": "flow_1Xtest_foo",
            "origin": "infectious",
            "to": "susceptibleXtest_foo",
            "implement": 1,
            "strain": None,
        },
        {
            "type": "customised_flows",
            "parameter": "flow_1Xtest_bar",
            "origin": "infectious",
            "to": "susceptibleXtest_bar",
            "implement": 1,
            "strain": None,
        },
        {
            "type": "customised_flows",
            "parameter": "flow_2",
            "origin": "comp_0",
            "to": "comp_1",
            "implement": 1,
            "strain": None,
        },
    ]


def test_create_ageing_flows():
    """
    Ensure ageing flows are created with correct flow rate
    """
    ages = ["0", "10", "15", "20", "60"]
    compartments = ["S", "I", "R"]
    implement_count = 2
    params, flows = create_ageing_flows(ages, compartments, implement_count)
    assert params == {
        "ageing0to10": 0.1,
        "ageing10to15": 0.2,
        "ageing15to20": 0.2,
        "ageing20to60": 0.025,
    }
    assert flows == [
        {
            "type": Flow.STANDARD,
            "parameter": "ageing0to10",
            "origin": "SXage_0",
            "to": "SXage_10",
            "implement": 2,
        },
        {
            "type": Flow.STANDARD,
            "parameter": "ageing0to10",
            "origin": "IXage_0",
            "to": "IXage_10",
            "implement": 2,
        },
        {
            "type": Flow.STANDARD,
            "parameter": "ageing0to10",
            "origin": "RXage_0",
            "to": "RXage_10",
            "implement": 2,
        },
        {
            "type": Flow.STANDARD,
            "parameter": "ageing10to15",
            "origin": "SXage_10",
            "to": "SXage_15",
            "implement": 2,
        },
        {
            "type": Flow.STANDARD,
            "parameter": "ageing10to15",
            "origin": "IXage_10",
            "to": "IXage_15",
            "implement": 2,
        },
        {
            "type": Flow.STANDARD,
            "parameter": "ageing10to15",
            "origin": "RXage_10",
            "to": "RXage_15",
            "implement": 2,
        },
        {
            "type": Flow.STANDARD,
            "parameter": "ageing15to20",
            "origin": "SXage_15",
            "to": "SXage_20",
            "implement": 2,
        },
        {
            "type": Flow.STANDARD,
            "parameter": "ageing15to20",
            "origin": "IXage_15",
            "to": "IXage_20",
            "implement": 2,
        },
        {
            "type": Flow.STANDARD,
            "parameter": "ageing15to20",
            "origin": "RXage_15",
            "to": "RXage_20",
            "implement": 2,
        },
        {
            "type": Flow.STANDARD,
            "parameter": "ageing20to60",
            "origin": "SXage_20",
            "to": "SXage_60",
            "implement": 2,
        },
        {
            "type": Flow.STANDARD,
            "parameter": "ageing20to60",
            "origin": "IXage_20",
            "to": "IXage_60",
            "implement": 2,
        },
        {
            "type": Flow.STANDARD,
            "parameter": "ageing20to60",
            "origin": f"RXage_20",
            "to": f"RXage_60",
            "implement": 2,
        },
    ]


def test_parse_param_adjustment_overwrite__with_no_overwrite():
    adjustment_requests = {"contact_rate": {"work": 0.5, "home": 0.2,}}
    strata_names = ["work", "home"]
    assert parse_param_adjustment_overwrite(strata_names, adjustment_requests) == {
        "contact_rate": {"work": 0.5, "home": 0.2},
        "overwrite": [],
    }


def test_parse_param_adjustment_overwrite___with_overwrite():
    adjustment_requests = {"contact_rate": {"work": 0.5, "homeW": 0.2,}}
    strata_names = ["work", "home"]
    assert parse_param_adjustment_overwrite(strata_names, adjustment_requests) == {
        "contact_rate": {"work": 0.5, "home": 0.2, "overwrite": ["home"]},
        "overwrite": [],
    }


PROPS_TEST_CASES = [
    # Test no change
    [
        ["foo", "bar", "baz"],
        {"foo": 0.3, "bar": 0.3, "baz": 0.4},
        {"foo": 0.3, "bar": 0.3, "baz": 0.4},
    ],
    # Test no props
    [["foo", "bar"], {}, {"foo": 0.5, "bar": 0.5},],
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
        split_proportions={"0": 0.25, "10": 0.5, "20": 0.25},
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
        split_proportions={"0": 0.25, "10": 0.5, "20": 0.25},
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
