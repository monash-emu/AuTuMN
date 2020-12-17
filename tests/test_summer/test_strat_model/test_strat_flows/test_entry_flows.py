import numpy as np

from summer.flow.base import BaseEntryFlow
from summer.compartment import Compartment
from summer.stratification import Stratification
from summer.constants import FlowAdjustment


class EntryFlow(BaseEntryFlow):
    """Basic entry flow used to test BaseEntryFlow stratification."""

    type = "entry"

    def __init__(
        self,
        dest,
        param_name,
        param_func,
        adjustments=[],
    ):
        self.dest = dest
        self.param_name = param_name
        self.param_func = param_func
        self.adjustments = adjustments

    def get_net_flow(self, compartment_values, time):
        return 1

    def copy(self, **kwargs):
        return EntryFlow(**kwargs)

    def __repr__(self):
        return ""


def _get_param_value(name, time):
    return 2 * time


def test_entry_flow_stratify__when_not_applicable():
    strat = Stratification(
        name="location",
        strata=["1", "2", "3"],
        compartments=["recovery"],  # infect compartment not included
        comp_split_props={},
        flow_adjustments={},
    )
    flow = EntryFlow(
        dest=Compartment("infect"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[],
    )
    new_flows = flow.stratify(strat)
    assert new_flows == [flow]


def test_entry_flow_stratify__with_no_flow_adjustments():
    strat = Stratification(
        name="location",
        strata=["1", "2"],
        compartments=["infect", "recovery"],
        comp_split_props={},
        flow_adjustments={},
    )
    flow = EntryFlow(
        dest=Compartment("infect"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[],
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2

    assert new_flows[0].param_name == "foo"
    assert new_flows[0].param_func == _get_param_value
    assert new_flows[0].adjustments == [(FlowAdjustment.MULTIPLY, 0.5)]
    assert new_flows[0].dest == Compartment(
        "infect", strat_names=["location"], strat_values={"location": "1"}
    )
    assert new_flows[1].param_name == "foo"
    assert new_flows[1].param_func == _get_param_value
    assert new_flows[1].adjustments == [(FlowAdjustment.MULTIPLY, 0.5)]
    assert new_flows[1].dest == Compartment(
        "infect", strat_names=["location"], strat_values={"location": "2"}
    )


def test_entry_flow_stratify_with_adjustments():
    strat = Stratification(
        name="location",
        strata=["1", "2"],
        compartments=["infect", "recovery"],
        comp_split_props={},
        flow_adjustments={},
    )
    strat.flow_adjustments = {
        "foo": [{"strata": {}, "adjustments": {"1": (FlowAdjustment.MULTIPLY, 0.1)}}]
    }
    flow = EntryFlow(
        dest=Compartment("infect"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[(FlowAdjustment.OVERWRITE, 0.2)],
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2

    assert new_flows[0].param_name == "foo"
    assert new_flows[0].param_func == _get_param_value
    assert new_flows[0].adjustments == [
        (FlowAdjustment.OVERWRITE, 0.2),
        (FlowAdjustment.MULTIPLY, 0.1),
    ]
    assert new_flows[0].dest == Compartment(
        "infect", strat_names=["location"], strat_values={"location": "1"}
    )
    assert new_flows[1].param_name == "foo"
    assert new_flows[1].param_func == _get_param_value
    assert new_flows[1].adjustments == [
        (FlowAdjustment.OVERWRITE, 0.2),
        (FlowAdjustment.MULTIPLY, 0.5),
    ]
    assert new_flows[1].dest == Compartment(
        "infect", strat_names=["location"], strat_values={"location": "2"}
    )


def test_entry_flow_stratify_with_ageing():
    strat = Stratification(
        name="age",
        strata=["0", "1", "2"],
        compartments=["infect", "recovery"],
        comp_split_props={},
        flow_adjustments={},
    )
    # Expect these to be ignored in favour of birth specific adjustments.
    strat.flow_adjustments = {
        "foo": [{"strata": {}, "adjustments": {"1": (FlowAdjustment.MULTIPLY, 0.1)}}]
    }
    flow = EntryFlow(
        dest=Compartment("infect"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[(FlowAdjustment.OVERWRITE, 0.2)],
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 3

    assert new_flows[0].param_name == "foo"
    assert new_flows[0].param_func == _get_param_value
    assert new_flows[0].adjustments == [
        (FlowAdjustment.OVERWRITE, 0.2),
        (FlowAdjustment.MULTIPLY, 1),
    ]
    assert new_flows[0].dest == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "0"}
    )
    assert new_flows[1].param_name == "foo"
    assert new_flows[1].param_func == _get_param_value
    assert new_flows[1].adjustments == [
        (FlowAdjustment.OVERWRITE, 0.2),
        (FlowAdjustment.OVERWRITE, 0),
    ]
    assert new_flows[1].dest == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "1"}
    )
    assert new_flows[2].param_name == "foo"
    assert new_flows[2].param_func == _get_param_value
    assert new_flows[2].adjustments == [
        (FlowAdjustment.OVERWRITE, 0.2),
        (FlowAdjustment.OVERWRITE, 0),
    ]
    assert new_flows[2].dest == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "2"}
    )
