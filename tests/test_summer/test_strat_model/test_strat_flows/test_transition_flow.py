import numpy as np

from summer.flow.base import BaseTransitionFlow
from summer.compartment import Compartment
from summer.stratification import Stratification
from summer.constants import FlowAdjustment


class TransitionFlow(BaseTransitionFlow):
    """Basic transition flow used to test BaseTransitionFlow stratification."""

    type = "transition"

    def __init__(
        self, source, dest, param_name, param_func, adjustments=[],
    ):
        self.source = source
        self.dest = dest
        self.param_name = param_name
        self.param_func = param_func
        self.adjustments = adjustments

    def get_net_flow(self, compartment_values, time):
        return 1

    def copy(self, **kwargs):
        return TransitionFlow(**kwargs)

    def __repr__(self):
        return ""


def _get_param_value(name, time):
    return 2 * time


def test_transition_flow_stratify_with_no_matching_compartments():
    strat = Stratification(
        name="age",
        strata=["1", "2", "3"],
        compartments=["recovery"],  # Flow compartments not included
        comp_split_props={},
        flow_adjustments={},
    )
    flow = TransitionFlow(
        source=Compartment("infect"),
        dest=Compartment("happy"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[],
    )
    new_flows = flow.stratify(strat)
    assert new_flows == [flow]


def test_transition_flow_stratify_source_and_dest():
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["infect", "happy", "recovery"],
        comp_split_props={},
        flow_adjustments={},
    )
    flow = TransitionFlow(
        source=Compartment("infect"),
        dest=Compartment("happy"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[],
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2

    assert new_flows[0].param_name == "foo"
    assert new_flows[0].param_func == _get_param_value
    assert new_flows[0].adjustments == []
    assert new_flows[0].source == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "1"}
    )
    assert new_flows[0].dest == Compartment("happy", strat_names=["age"], strat_values={"age": "1"})

    assert new_flows[1].param_name == "foo"
    assert new_flows[1].param_func == _get_param_value
    assert new_flows[1].adjustments == []
    assert new_flows[1].source == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "2"}
    )
    assert new_flows[1].dest == Compartment("happy", strat_names=["age"], strat_values={"age": "2"})


def test_transition_flow_stratify_source_but_not_dest():
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["infect", "recovery"],
        comp_split_props={},
        flow_adjustments={},
    )
    flow = TransitionFlow(
        source=Compartment("infect"),
        dest=Compartment("happy"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[],
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2

    assert new_flows[0].param_name == "foo"
    assert new_flows[0].param_func == _get_param_value
    assert new_flows[0].adjustments == []
    assert new_flows[0].source == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "1"}
    )
    assert new_flows[0].dest == Compartment("happy", strat_names=[], strat_values={})
    assert new_flows[1].param_name == "foo"
    assert new_flows[1].param_func == _get_param_value
    assert new_flows[1].adjustments == []
    assert new_flows[1].source == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "2"}
    )
    assert new_flows[1].dest == Compartment("happy", strat_names=[], strat_values={})


def test_transition_flow_stratify_dest_but_not_source():
    """
    Ensure flow is adjusted to account for fan out.
    """
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["happy", "recovery"],
        comp_split_props={},
        flow_adjustments={},
    )
    flow = TransitionFlow(
        source=Compartment("infect"),
        dest=Compartment("happy"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[],
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2

    assert new_flows[0].param_name == "foo"
    assert new_flows[0].param_func == _get_param_value
    assert new_flows[0].adjustments == [(FlowAdjustment.MULTIPLY, 0.5)]
    assert new_flows[0].source == Compartment("infect", strat_names=[], strat_values={})
    assert new_flows[0].dest == Compartment("happy", strat_names=["age"], strat_values={"age": "1"})

    assert new_flows[1].param_name == "foo"
    assert new_flows[1].param_func == _get_param_value
    assert new_flows[1].adjustments == [(FlowAdjustment.MULTIPLY, 0.5)]
    assert new_flows[1].source == Compartment("infect", strat_names=[], strat_values={})
    assert new_flows[1].dest == Compartment("happy", strat_names=["age"], strat_values={"age": "2"})


def test_transition_flow_stratify_source_and_dest__with_flow_adjustments():
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["infect", "happy", "recovery"],
        comp_split_props={},
        flow_adjustments={},
    )
    strat.flow_adjustments = {
        "foo": [{"strata": {}, "adjustments": {"1": (FlowAdjustment.MULTIPLY, 0.1)}}]
    }
    flow = TransitionFlow(
        source=Compartment("infect"),
        dest=Compartment("happy"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[],
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2

    assert new_flows[0].param_name == "foo"
    assert new_flows[0].param_func == _get_param_value
    assert new_flows[0].adjustments == [(FlowAdjustment.MULTIPLY, 0.1)]
    assert new_flows[0].source == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "1"}
    )
    assert new_flows[0].dest == Compartment("happy", strat_names=["age"], strat_values={"age": "1"})

    assert new_flows[1].param_name == "foo"
    assert new_flows[1].param_func == _get_param_value
    assert new_flows[1].adjustments == []
    assert new_flows[1].source == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "2"}
    )
    assert new_flows[1].dest == Compartment("happy", strat_names=["age"], strat_values={"age": "2"})


def test_transition_flow_stratify_source_but_not_dest__with_flow_adjustments():
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["infect", "recovery"],
        comp_split_props={},
        flow_adjustments={},
    )
    flow = TransitionFlow(
        source=Compartment("infect"),
        dest=Compartment("happy"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[],
    )
    strat.flow_adjustments = {
        "foo": [{"strata": {}, "adjustments": {"1": (FlowAdjustment.MULTIPLY, 0.1)}}]
    }
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2

    assert new_flows[0].param_name == "foo"
    assert new_flows[0].param_func == _get_param_value
    assert new_flows[0].adjustments == [(FlowAdjustment.MULTIPLY, 0.1)]
    assert new_flows[0].source == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "1"}
    )
    assert new_flows[0].dest == Compartment("happy", strat_names=[], strat_values={})
    assert new_flows[1].param_name == "foo"
    assert new_flows[1].param_func == _get_param_value
    assert new_flows[1].adjustments == []
    assert new_flows[1].source == Compartment(
        "infect", strat_names=["age"], strat_values={"age": "2"}
    )
    assert new_flows[1].dest == Compartment("happy", strat_names=[], strat_values={})


def test_transition_flow_stratify_dest_but_not_source__with_flow_adjustments():
    """
    Ensure flow is adjusted to account for fan out.
    """
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["happy", "recovery"],
        comp_split_props={},
        flow_adjustments={},
    )
    strat.flow_adjustments = {
        "foo": [{"strata": {}, "adjustments": {"1": (FlowAdjustment.MULTIPLY, 0.1)}}]
    }
    flow = TransitionFlow(
        source=Compartment("infect"),
        dest=Compartment("happy"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[],
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2

    assert new_flows[0].param_name == "foo"
    assert new_flows[0].param_func == _get_param_value
    assert new_flows[0].adjustments == [(FlowAdjustment.MULTIPLY, 0.1)]
    assert new_flows[0].source == Compartment("infect", strat_names=[], strat_values={})
    assert new_flows[0].dest == Compartment("happy", strat_names=["age"], strat_values={"age": "1"})

    assert new_flows[1].param_name == "foo"
    assert new_flows[1].param_func == _get_param_value
    assert new_flows[1].adjustments == [(FlowAdjustment.MULTIPLY, 0.5)]
    assert new_flows[1].source == Compartment("infect", strat_names=[], strat_values={})
    assert new_flows[1].dest == Compartment("happy", strat_names=["age"], strat_values={"age": "2"})
