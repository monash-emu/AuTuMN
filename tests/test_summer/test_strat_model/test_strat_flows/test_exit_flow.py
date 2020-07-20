import numpy as np

from summer.flow.base import BaseExitFlow
from summer.compartment import Compartment
from summer.stratification import Stratification
from summer.constants import FlowAdjustment


class ExitFlow(BaseExitFlow):
    """Basic exit flow used to test BaseExitFlow stratification."""

    type = "exit"

    def __init__(
        self, source, param_name, param_func, adjustments=[],
    ):
        self.source = source
        self.param_name = param_name
        self.param_func = param_func
        self.adjustments = adjustments

    def get_net_flow(self, compartment_values, time):
        return 1

    def copy(self, **kwargs):
        return ExitFlow(**kwargs)

    def __repr__(self):
        return ""


def _get_param_value(name, time):
    return 2 * time


def test_exit_flow_stratify__when_no_compartment_match():
    strat = Stratification(
        name="age",
        strata=["1", "2", "3"],
        compartments=["recovery"],  # infect compartment not included
        comp_split_props={},
        flow_adjustments={},
    )
    flow = ExitFlow(
        source=Compartment("infect"), param_name="foo", param_func=_get_param_value, adjustments=[],
    )
    new_flows = flow.stratify(strat)
    assert new_flows == [flow]


def test_exit_flow_stratify__with_no_flow_adjustments():
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["infect", "recovery"],
        comp_split_props={},
        flow_adjustments={},
    )
    flow = ExitFlow(
        source=Compartment("infect"), param_name="foo", param_func=_get_param_value, adjustments=[],
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2

    assert new_flows[0].param_name == "foo"
    assert new_flows[0].param_func == _get_param_value
    assert new_flows[0].adjustments == []
    assert new_flows[0].source == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "1"}
    )
    assert new_flows[1].param_name == "foo"
    assert new_flows[1].param_func == _get_param_value
    assert new_flows[1].adjustments == []
    assert new_flows[1].source == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "2"}
    )


def test_exit_flow_stratify_with_flow_adjustments():
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["infect", "recovery"],
        comp_split_props={},
        flow_adjustments={},
    )
    strat.flow_adjustments = {
        "foo": [{"strata": {}, "adjustments": {"1": (FlowAdjustment.MULTIPLY, 0.1)}}]
    }
    flow = ExitFlow(
        source=Compartment("infect"),
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
    assert new_flows[0].source == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "1"}
    )
    assert new_flows[1].param_name == "foo"
    assert new_flows[1].param_func == _get_param_value
    assert new_flows[1].adjustments == [(FlowAdjustment.OVERWRITE, 0.2)]
    assert new_flows[1].source == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "2"}
    )
