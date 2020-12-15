import pytest
from summer2.flows import BaseExitFlow
from summer2 import Compartment, Stratification, adjust


class ExitFlow(BaseExitFlow):
    """Basic exit flow used to test BaseExitFlow stratification."""

    def get_net_flow(self, compartment_values, time):
        return 1


def test_exit_flow_stratify__when_no_compartment_match():
    flow = ExitFlow(
        name="flow",
        source=Compartment("I"),
        param=2,
        adjustments=[],
    )
    strat = Stratification(
        name="location",
        strata=["1", "2", "3"],
        compartments=["R"],
    )

    # Expect no stratification because compartment not being stratified.
    new_flows = flow.stratify(strat)

    assert new_flows == [flow]


def test_exit_flow_stratify__with_no_flow_adjustments():
    flow = ExitFlow(
        name="flow",
        source=Compartment("I"),
        param=2,
        adjustments=[],
    )
    strat = Stratification(
        name="age",
        strata=["1", "2", "3"],
        compartments=["I", "R"],
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 3

    assert new_flows[0]._is_equal(
        ExitFlow(
            name="flow",
            param=2,
            source=Compartment("I", {"age": "1"}),
            adjustments=[],
        )
    )
    assert new_flows[1]._is_equal(
        ExitFlow(
            name="flow",
            param=2,
            source=Compartment("I", {"age": "2"}),
            adjustments=[],
        )
    )
    assert new_flows[2]._is_equal(
        ExitFlow(
            name="flow",
            param=2,
            source=Compartment("I", {"age": "3"}),
            adjustments=[],
        )
    )


def test_exit_flow_stratify_with_flow_adjustments():
    flow = ExitFlow(
        name="flow",
        source=Compartment("I"),
        param=2,
        adjustments=[adjust.Overwrite(0.2)],
    )
    strat = Stratification(
        name="age",
        strata=["1", "2"],
        compartments=["I", "R"],
    )
    strat.add_flow_adjustments(
        "flow",
        {
            "1": adjust.Multiply(0.1),
            "2": None,
        },
    )
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 2
    assert new_flows[0]._is_equal(
        ExitFlow(
            name="flow",
            param=2,
            source=Compartment("I", {"age": "1"}),
            adjustments=[adjust.Overwrite(0.2), adjust.Multiply(0.1)],
        )
    )
    assert new_flows[1]._is_equal(
        ExitFlow(
            name="flow",
            param=2,
            source=Compartment("I", {"age": "2"}),
            adjustments=[adjust.Overwrite(0.2)],
        )
    )
