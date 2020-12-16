from summer2 import adjust, Compartment, Stratification
from summer2.flows import BaseTransitionFlow


class TransitionFlow(BaseTransitionFlow):
    """Basic transition flow used to test BaseTransitionFlow stratification."""

    def get_net_flow(self, compartment_values, time):
        return 1


def test_transition_flow_stratify_with_no_matching_compartments():
    flow = TransitionFlow(
        name="flow",
        source=Compartment("S"),
        dest=Compartment("I"),
        param=2,
        adjustments=[],
    )
    strat = Stratification(
        name="location",
        strata=["1", "2", "3"],
        compartments=["R"],
    )
    new_flows = flow.stratify(strat)
    assert new_flows == [flow]


def test_transition_flow_stratify_source_and_dest():
    """
    Ensure two parallel flows created, no adjustments required.
    """
    flow = TransitionFlow(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=2,
        adjustments=[],
    )
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["S", "I", "R"],
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2
    assert new_flows[0]._is_equal(
        TransitionFlow(
            name="flow",
            param=2,
            source=Compartment("I", {"age": "1"}),
            dest=Compartment("R", {"age": "1"}),
            adjustments=[],
        )
    )
    assert new_flows[1]._is_equal(
        TransitionFlow(
            name="flow",
            param=2,
            source=Compartment("I", {"age": "2"}),
            dest=Compartment("R", {"age": "2"}),
            adjustments=[],
        )
    )


def test_transition_flow_stratify_source_but_not_dest():
    """
    Ensure two parallel flows created, no adjustments required.
    """
    flow = TransitionFlow(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=2,
        adjustments=[],
    )
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["S", "I"],
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2
    assert new_flows[0]._is_equal(
        TransitionFlow(
            name="flow",
            param=2,
            source=Compartment("I", {"age": "1"}),
            dest=Compartment("R"),
            adjustments=[],
        )
    )
    assert new_flows[1]._is_equal(
        TransitionFlow(
            name="flow",
            param=2,
            source=Compartment("I", {"age": "2"}),
            dest=Compartment("R"),
            adjustments=[],
        )
    )


def test_transition_flow_stratify_dest_but_not_source():
    """
    Ensure two new flows are created and automatically adjusted to account for fan out.
    """
    flow = TransitionFlow(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=2,
        adjustments=[],
    )
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["S", "R"],
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2
    assert new_flows[0]._is_equal(
        TransitionFlow(
            name="flow",
            param=2,
            source=Compartment("I"),
            dest=Compartment("R", {"age": "1"}),
            adjustments=[adjust.Multiply(0.5)],
        )
    )
    assert new_flows[1]._is_equal(
        TransitionFlow(
            name="flow",
            param=2,
            source=Compartment("I"),
            dest=Compartment("R", {"age": "2"}),
            adjustments=[adjust.Multiply(0.5)],
        )
    )


def test_transition_flow_stratify_source_and_dest__with_flow_adjustments():
    flow = TransitionFlow(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=2,
        adjustments=[adjust.Multiply(0.1)],
    )
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["S", "I", "R"],
    )
    strat.add_flow_adjustments(
        "flow",
        {
            "1": adjust.Multiply(0.2),
            "2": adjust.Multiply(0.3),
        },
    )

    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2
    assert new_flows[0]._is_equal(
        TransitionFlow(
            name="flow",
            param=2,
            source=Compartment("I", {"age": "1"}),
            dest=Compartment("R", {"age": "1"}),
            adjustments=[adjust.Multiply(0.1), adjust.Multiply(0.2)],
        )
    )
    assert new_flows[1]._is_equal(
        TransitionFlow(
            name="flow",
            param=2,
            source=Compartment("I", {"age": "2"}),
            dest=Compartment("R", {"age": "2"}),
            adjustments=[adjust.Multiply(0.1), adjust.Multiply(0.3)],
        )
    )


def test_transition_flow_stratify_source_but_not_dest__with_flow_adjustments():
    flow = TransitionFlow(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=2,
        adjustments=[adjust.Multiply(0.1)],
    )
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["S", "I"],
    )
    strat.add_flow_adjustments(
        "flow",
        {
            "1": adjust.Multiply(0.2),
            "2": adjust.Multiply(0.3),
        },
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2
    assert new_flows[0]._is_equal(
        TransitionFlow(
            name="flow",
            param=2,
            source=Compartment("I", {"age": "1"}),
            dest=Compartment("R"),
            adjustments=[adjust.Multiply(0.1), adjust.Multiply(0.2)],
        )
    )
    assert new_flows[1]._is_equal(
        TransitionFlow(
            name="flow",
            param=2,
            source=Compartment("I", {"age": "2"}),
            dest=Compartment("R"),
            adjustments=[adjust.Multiply(0.1), adjust.Multiply(0.3)],
        )
    )


def test_transition_flow_stratify_dest_but_not_source__with_flow_adjustments():
    """
    Ensure flow is adjusted to account for fan out.
    """
    flow = TransitionFlow(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=2,
        adjustments=[adjust.Multiply(0.1)],
    )
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["S", "R"],
    )
    strat.add_flow_adjustments(
        "flow",
        {
            "1": adjust.Multiply(0.2),
            "2": adjust.Multiply(0.3),
        },
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2
    assert new_flows[0]._is_equal(
        TransitionFlow(
            name="flow",
            param=2,
            source=Compartment("I"),
            dest=Compartment("R", {"age": "1"}),
            adjustments=[adjust.Multiply(0.1), adjust.Multiply(0.2)],
        )
    )
    assert new_flows[1]._is_equal(
        TransitionFlow(
            name="flow",
            param=2,
            source=Compartment("I"),
            dest=Compartment("R", {"age": "2"}),
            adjustments=[adjust.Multiply(0.1), adjust.Multiply(0.3)],
        )
    )
