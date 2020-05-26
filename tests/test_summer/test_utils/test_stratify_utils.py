import pytest

from summer.constants import Flow
from summer.model.utils import (
    get_all_proportions,
    get_stratified_compartments,
    parse_param_adjustment_overwrite,
    create_ageing_flows,
    stratify_transition_flows,
)


def test_stratify_transition_flows():
    flows = [
        {
            "type": "standard_flows",
            "parameter": "flow_0",
            "origin": "susceptible",
            "to": "infectious",
            "implement": 0,
            "strain": None,
            "force_index": None,
        },
        {
            "type": "standard_flows",
            "parameter": "flow_1",
            "origin": "infectious",
            "to": "susceptible",
            "implement": 0,
            "strain": None,
            "force_index": None,
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


PARAM_VARS = "flows, custom_func, params, adjustment, comps, expected_flows, expected_custom_func, expected_params"
PARAM_VALS = [
    # Start with two transition flows and two parameters,
    # Apply 2x strata and 1x parameter adjustment
    # Apply strata to all compartments
    # Expect 4 new transition flows and 2 new parameters.
    [
        # Starting flow
        [
            ["standard_flows", "flow_0", "susceptible", "infectious", 0, None, None],
            ["standard_flows", "flow_1", "infectious", "susceptible", 0, None, None],
        ],
        # Starting custom flow funcs
        {},
        # Starting params
        {"flow_0": 0.5, "flow_1": 0.1,},
        # Adjustments
        {"flow_0": {"foo": 0.2, "bar": 0.7,}},
        # Compartments to stratify
        ["susceptible", "infectious"],
        # Expected flows
        [
            ["standard_flows", "flow_0", "susceptible", "infectious", 0, None, None],
            ["standard_flows", "flow_1", "infectious", "susceptible", 0, None, None],
            [
                "standard_flows",
                "flow_0Xtest_foo",
                "susceptibleXtest_foo",
                "infectiousXtest_foo",
                1,
                None,
                None,
            ],
            [
                "standard_flows",
                "flow_0Xtest_bar",
                "susceptibleXtest_bar",
                "infectiousXtest_bar",
                1,
                None,
                None,
            ],
            [
                "standard_flows",
                "flow_1",
                "infectiousXtest_foo",
                "susceptibleXtest_foo",
                1,
                None,
                None,
            ],
            [
                "standard_flows",
                "flow_1",
                "infectiousXtest_bar",
                "susceptibleXtest_bar",
                1,
                None,
                None,
            ],
        ],
        # Expected custom funcs
        {},
        # Expected params
        {"flow_0": 0.5, "flow_1": 0.1, "flow_0Xtest_foo": 0.2, "flow_0Xtest_bar": 0.7},
    ],
    # Same as above but only stratify one compartment
    [
        # Starting flows
        [
            ["standard_flows", "flow_0", "susceptible", "infectious", 0, None, None],
            ["standard_flows", "flow_1", "infectious", "susceptible", 0, None, None],
        ],
        # Starting custom flow funcs
        {},
        # Starting params
        {"flow_0": 0.5, "flow_1": 0.1,},
        # Adjustments
        {"flow_0": {"foo": 0.2, "bar": 0.7,}},
        # Compartments to stratify
        ["susceptible"],
        # Expected flows
        [
            ["standard_flows", "flow_0", "susceptible", "infectious", 0, None, None],
            ["standard_flows", "flow_1", "infectious", "susceptible", 0, None, None],
            [
                "standard_flows",
                "flow_0Xtest_foo",
                "susceptibleXtest_foo",
                "infectious",
                1,
                None,
                None,
            ],
            [
                "standard_flows",
                "flow_0Xtest_bar",
                "susceptibleXtest_bar",
                "infectious",
                1,
                None,
                None,
            ],
            [
                "standard_flows",
                "flow_1Xtest_foo",
                "infectious",
                "susceptibleXtest_foo",
                1,
                None,
                None,
            ],
            [
                "standard_flows",
                "flow_1Xtest_bar",
                "infectious",
                "susceptibleXtest_bar",
                1,
                None,
                None,
            ],
        ],
        # Expected custom funcs
        {},
        # Expected params
        {
            "flow_0": 0.5,
            "flow_1": 0.1,
            "flow_0Xtest_foo": 0.2,
            "flow_0Xtest_bar": 0.7,
            "flow_1Xtest_foo": 0.5,
            "flow_1Xtest_bar": 0.5,
        },
    ],
    # Expect custom flow funcs to be updated
    [
        # Starting flow
        [
            ["customised_flows", "flow_0", "susceptible", "infectious", 0, None, None],
            ["customised_flows", "flow_1", "infectious", "susceptible", 0, None, None],
            ["customised_flows", "flow_2", "comp_0", "comp_1", 0, None, None],
        ],
        # Starting custom flow funcs
        {0: "func_for_flow_0", 1: "func_for_flow_1", 2: "func_for_flow_2"},
        # Starting params
        {"flow_0": 0.5, "flow_1": 0.1, "flow_2": 1.9},
        # Adjustments
        {},
        # Compartments to stratify
        ["susceptible"],
        # Expected flows
        [
            ["customised_flows", "flow_0", "susceptible", "infectious", 0, None, None],
            ["customised_flows", "flow_1", "infectious", "susceptible", 0, None, None],
            ["customised_flows", "flow_2", "comp_0", "comp_1", 0, None, None],
            ["customised_flows", "flow_0", "susceptibleXtest_foo", "infectious", 1, None, None,],
            ["customised_flows", "flow_0", "susceptibleXtest_bar", "infectious", 1, None, None,],
            [
                "customised_flows",
                "flow_1Xtest_foo",
                "infectious",
                "susceptibleXtest_foo",
                1,
                None,
                None,
            ],
            [
                "customised_flows",
                "flow_1Xtest_bar",
                "infectious",
                "susceptibleXtest_bar",
                1,
                None,
                None,
            ],
            ["customised_flows", "flow_2", "comp_0", "comp_1", 1, None, None],
        ],
        # Expected custom funcs
        {
            0: "func_for_flow_0",
            1: "func_for_flow_1",
            2: "func_for_flow_2",
            3: "func_for_flow_0",
            4: "func_for_flow_0",
            5: "func_for_flow_1",
            6: "func_for_flow_1",
            7: "func_for_flow_2",
        },
        # Expected params
        {
            "flow_0": 0.5,
            "flow_1": 0.1,
            "flow_1Xtest_foo": 0.5,
            "flow_1Xtest_bar": 0.5,
            "flow_2": 1.9,
        },
    ],
]


# @pytest.mark.parametrize(PARAM_VARS, PARAM_VALS)
# def test_stratify_transition_flows(
#     flows,
#     custom_func,
#     params,
#     adjustment,
#     comps,
#     expected_flows,
#     expected_custom_func,
#     expected_params,
# ):
#     """
#     Ensure that `stratify_compartments` splits up transition flows correctly.
#     """
#     model = StratifiedModel(**_get_model_kwargs())
#     cols = ["type", "parameter", "origin", "to", "implement", "strain", "force_index"]
#     model.transition_flows = pd.DataFrame(flows, columns=cols).astype(object)
#     model.parameters = params
#     strat_name = "test"
#     strata_names = ["foo", "bar"]
#     model.customised_flow_functions = custom_func
#     model.all_stratifications = {strat_name: strata_names}
#     model.stratify_compartments(strat_name, strata_names, {"foo": 0.5, "bar": 0.5}, comps)
#     model.stratify_transition_flows(strat_name, strata_names, adjustment, comps)
#     # Check flows df stratified
#     expected_flows_df = pd.DataFrame(expected_flows, columns=cols).astype(object)
#     assert_frame_equal(expected_flows_df, model.transition_flows)
#     # Check custom flow func is updated
#     assert model.customised_flow_functions == expected_custom_func
#     # Check params are stratified
#     for k, v in expected_params.items():
#         assert model.parameters[k] == v


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
