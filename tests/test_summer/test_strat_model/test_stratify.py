"""
Test the results of stratifying a model.
"""
import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal

from summer.model import StratifiedModel
from summer.constants import Compartment, Flow, BirthApproach, Stratification, IntegrationType


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


@pytest.mark.parametrize(PARAM_VARS, PARAM_VALS)
def test_stratify_transition_flows(
    flows,
    custom_func,
    params,
    adjustment,
    comps,
    expected_flows,
    expected_custom_func,
    expected_params,
):
    """
    Ensure that `stratify_compartments` splits up transition flows correctly.
    """
    model = StratifiedModel(**_get_model_kwargs())
    cols = ["type", "parameter", "origin", "to", "implement", "strain", "force_index"]
    model.transition_flows = pd.DataFrame(flows, columns=cols).astype(object)
    model.parameters = params
    strat_name = "test"
    strata_names = ["foo", "bar"]
    model.customised_flow_functions = custom_func
    model.all_stratifications = {strat_name: strata_names}
    model.stratify_compartments(strat_name, strata_names, {"foo": 0.5, "bar": 0.5}, comps)
    model.stratify_transition_flows(strat_name, strata_names, adjustment, comps)
    # Check flows df stratified
    expected_flows_df = pd.DataFrame(expected_flows, columns=cols).astype(object)
    assert_frame_equal(expected_flows_df, model.transition_flows)
    # Check custom flow func is updated
    assert model.customised_flow_functions == expected_custom_func
    # Check params are stratified
    for k, v in expected_params.items():
        assert model.parameters[k] == v


PARAM_VARS = "back_one, include_change, all_stratifications, flows, expected_idxs"
PARAM_VALS = [
    # Test default case - expect all flow idxs.
    [
        0,
        False,
        {},
        [
            ["standard_flows", "flow_name_0", "comp_0", "comp_1", 0, None, None],
            ["standard_flows", "flow_name_1", "comp_1", "comp_0", 0, None, None],
        ],
        [0, 1],
    ],
    # Test default case, but 1 flow has wrong implement - expect all flow idxs except that one.
    [
        0,
        False,
        {},
        [
            ["standard_flows", "flow_name_0", "comp_0", "comp_1", 1, None, None],
            ["standard_flows", "flow_name_1", "comp_1", "comp_0", 0, None, None],
        ],
        [1],
    ],
    # Test default case, but 1 strats added yet with back 1 - expect all flow idxs.
    [
        1,
        False,
        {"age": [0, 5, 15, 50]},
        [
            ["standard_flows", "flow_name_0", "comp_0", "comp_1", 0, None, None],
            ["standard_flows", "flow_name_1", "comp_1", "comp_0", 0, None, None],
        ],
        [0, 1],
    ],
    # Test default case, but 1 strats added yet no back 1 - expect no flow idxs.
    [
        0,
        False,
        {"age": [0, 5, 15, 50]},
        [
            ["standard_flows", "flow_name_0", "comp_0", "comp_1", 0, None, None],
            ["standard_flows", "flow_name_1", "comp_1", "comp_0", 0, None, None],
        ],
        [],
    ],
    # Test default case, but with a strata change flow - expect all flow idxs except strata change flow.
    [
        0,
        False,
        {},
        [
            ["standard_flows", "flow_name_0", "comp_0", "comp_1", 0, None, None],
            ["strata_change", "flow_name_0", "comp_0", "comp_1", 0, None, None],
            ["standard_flows", "flow_name_1", "comp_1", "comp_0", 0, None, None],
        ],
        [0, 2],
    ],
    # Test default case, but with a strata change flow and 'include change' - expect all flow idxs.
    [
        0,
        True,
        {},
        [
            ["standard_flows", "flow_name_0", "comp_0", "comp_1", 0, None, None],
            ["strata_change", "flow_name_0", "comp_0", "comp_1", 0, None, None],
            ["standard_flows", "flow_name_1", "comp_1", "comp_0", 0, None, None],
        ],
        [0, 1, 2],
    ],
]


@pytest.mark.parametrize(PARAM_VARS, PARAM_VALS)
def test_find_transition_indices_to_implement(
    back_one, include_change, all_stratifications, flows, expected_idxs
):
    """
    Ensure `find_transition_indices_to_implement` returns the correct list of indices.
    """
    model = StratifiedModel(**_get_model_kwargs())
    cols = ["type", "parameter", "origin", "to", "implement", "strain", "force_index"]
    model.transition_flows = pd.DataFrame(flows, columns=cols).astype(object)
    model.all_stratifications = all_stratifications
    actual_idxs = model.find_transition_indices_to_implement(back_one, include_change)
    assert expected_idxs == actual_idxs


PARAM_VARS = "strata,proportions,to_stratify,expected_names,expected_values"
PARAM_VALS = [
    # Use 2 strata, expect 2x new compartments, strata split evenly.
    [
        ["foo", "bar"],
        {"foo": 0.5, "bar": 0.5},
        ["susceptible", "infectious"],
        [
            "susceptibleXtest_foo",
            "susceptibleXtest_bar",
            "infectiousXtest_foo",
            "infectiousXtest_bar",
        ],
        [450, 450, 50, 50],
    ],
    # Use 2 strata, expect 2x new compartments, strata split unevenly.
    [
        ["foo", "bar"],
        {"foo": 0.1, "bar": 0.9},
        ["susceptible", "infectious"],
        [
            "susceptibleXtest_foo",
            "susceptibleXtest_bar",
            "infectiousXtest_foo",
            "infectiousXtest_bar",
        ],
        [90, 810, 10, 90],
    ],
    # Use 2 strata, don't stratify infectious.
    [
        ["foo", "bar"],
        {"foo": 0.5, "bar": 0.5},
        ["susceptible"],
        ["infectious", "susceptibleXtest_foo", "susceptibleXtest_bar"],
        [100, 450, 450],
    ],
]


@pytest.mark.parametrize(PARAM_VARS, PARAM_VALS)
def test_stratify_compartments(strata, proportions, to_stratify, expected_names, expected_values):
    """
    Ensure that `stratify_compartments` splits up compartment names and values correctly.
    """
    model = StratifiedModel(**_get_model_kwargs())
    model.stratify_compartments("test", strata, proportions, to_stratify)
    assert model.compartment_names == expected_names
    assert model.compartment_values == expected_values


PARAM_VARS = "age_strata,expected_flows,expected_ageing"
PARAM_VALS = [
    # Test simple age split, expect ageing to be proprotional to bracket width.
    [
        [0, 50],
        [
            [
                "standard_flows",
                "ageing0to50",
                "susceptibleXage_0",
                "susceptibleXage_50",
                0,
                None,
                None,
            ],
            [
                "standard_flows",
                "ageing0to50",
                "infectiousXage_0",
                "infectiousXage_50",
                0,
                None,
                None,
            ],
        ],
        {"ageing0to50": 1 / 50},
    ],
    # Test typical age split, expect ageing to be proprotional to bracket width.
    [
        [0, 5, 15, 50],
        [
            [
                "standard_flows",
                "ageing0to5",
                "susceptibleXage_0",
                "susceptibleXage_5",
                0,
                None,
                None,
            ],
            ["standard_flows", "ageing0to5", "infectiousXage_0", "infectiousXage_5", 0, None, None],
            [
                "standard_flows",
                "ageing5to15",
                "susceptibleXage_5",
                "susceptibleXage_15",
                0,
                None,
                None,
            ],
            [
                "standard_flows",
                "ageing5to15",
                "infectiousXage_5",
                "infectiousXage_15",
                0,
                None,
                None,
            ],
            [
                "standard_flows",
                "ageing15to50",
                "susceptibleXage_15",
                "susceptibleXage_50",
                0,
                None,
                None,
            ],
            [
                "standard_flows",
                "ageing15to50",
                "infectiousXage_15",
                "infectiousXage_50",
                0,
                None,
                None,
            ],
        ],
        {"ageing0to5": 1 / 5, "ageing5to15": 1 / 10, "ageing15to50": 1 / 35,},
    ],
]


@pytest.mark.parametrize(PARAM_VARS, PARAM_VALS)
def test_set_ageing_rates(age_strata, expected_flows, expected_ageing):
    """
    Ensure that `set_ageing_rates` adds ageing flows to the transition flows dataframe
    """
    model = StratifiedModel(**_get_model_kwargs())
    cols = ["type", "parameter", "origin", "to", "implement", "strain", "force_index"]
    # Ensure there are no initial flows
    initial_df = pd.DataFrame([], columns=cols).astype(object)
    assert_frame_equal(initial_df, model.transition_flows)
    # Set ageing rates
    model.set_ageing_rates(age_strata)

    # Check ageing flows are set
    expected_df = pd.DataFrame(expected_flows, columns=cols).astype(object)
    assert_frame_equal(expected_df, model.transition_flows)
    # Check ageing params are set
    for k, v in expected_ageing.items():
        assert model.parameters[k] == v


def _get_model_kwargs(**kwargs):
    return {
        "times": [2000, 2001, 2002, 2003, 2004, 2005],
        "compartment_types": [Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS],
        "initial_conditions": {Compartment.INFECTIOUS: 100},
        "parameters": {},
        "requested_flows": [],
        "starting_population": 1000,
        **kwargs,
    }
