"""
Test the results of stratifying a model.
"""
import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal

from summer.model import StratifiedModel
from summer.constants import Compartment


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
        "compartment_types": [Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        "initial_conditions": {Compartment.EARLY_INFECTIOUS: 100},
        "parameters": {},
        "requested_flows": [],
        "starting_population": 1000,
        **kwargs,
    }
