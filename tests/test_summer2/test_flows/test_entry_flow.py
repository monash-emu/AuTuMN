import pytest
from summer2.flows import BaseEntryFlow
from summer2 import Compartment, AgeStratification, Stratification, adjust


class EntryFlow(BaseEntryFlow):
    """Basic entry flow used to test BaseEntryFlow stratification."""

    def get_net_flow(self, compartment_values, time):
        return 1


def test_entry_flow_stratify__when_not_applicable():
    flow = EntryFlow(
        name="flow",
        dest=Compartment("I"),
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


def test_entry_flow_stratify__with_no_flow_adjustments():
    flow = EntryFlow(
        name="flow",
        dest=Compartment("I"),
        param=2,
        adjustments=[],
    )
    strat = Stratification(
        name="location",
        strata=["1", "2"],
        compartments=["I", "R"],
    )

    new_flows = flow.stratify(strat)

    assert len(new_flows) == 2
    # Both flows has 50% flow adjustment applied to conserve inflows of people.
    assert new_flows[0]._is_equal(
        EntryFlow(
            name="flow",
            param=2,
            dest=Compartment("I", {"location": "1"}),
            adjustments=[adjust.Multiply(0.5)],
        )
    )
    assert new_flows[1]._is_equal(
        EntryFlow(
            name="flow",
            param=2,
            dest=Compartment("I", {"location": "2"}),
            adjustments=[adjust.Multiply(0.5)],
        )
    )


def test_entry_flow_stratify_with_adjustments():
    flow = EntryFlow(
        name="flow",
        dest=Compartment("I"),
        param=2,
        adjustments=[adjust.Overwrite(0.2)],
    )
    strat = Stratification(
        name="location",
        strata=["1", "2"],
        compartments=["I", "R"],
    )
    strat.add_flow_adjustments("flow", {"1": adjust.Multiply(0.1), "2": adjust.Multiply(0.3)})

    new_flows = flow.stratify(strat)

    assert len(new_flows) == 2

    assert new_flows[0]._is_equal(
        EntryFlow(
            name="flow",
            param=2,
            dest=Compartment("I", {"location": "1"}),
            adjustments=[adjust.Overwrite(0.2), adjust.Multiply(0.1)],
        )
    )
    assert new_flows[1]._is_equal(
        EntryFlow(
            name="flow",
            param=2,
            dest=Compartment("I", {"location": "2"}),
            adjustments=[adjust.Overwrite(0.2), adjust.Multiply(0.3)],
        )
    )


def test_entry_flow_stratify_with_ageing():
    strat = AgeStratification(
        name="age",
        strata=["0", "1", "2"],
        compartments=["I", "R"],
    )
    flow = EntryFlow(
        name="birth",
        dest=Compartment("I"),
        param=2,
        adjustments=[adjust.Overwrite(0.2)],
    )

    flow._is_birth_flow = False  # Not marked as a birth flow!
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 3  # So the birth flow rules don't apply.

    flow._is_birth_flow = True  # Marked as a birth flow.
    new_flows = flow.stratify(strat)
    assert len(new_flows) == 1  # So the birth flow rules apply.
    # Only age 0 babies get born.
    assert new_flows[0]._is_equal(
        EntryFlow(
            name="birth",
            param=2,
            dest=Compartment("I", {"age": "0"}),
            adjustments=[adjust.Overwrite(0.2)],
        )
    )

    # Expect this to fail coz you can't adjust birth flows for age stratifications.
    strat.add_flow_adjustments("birth", {"0": adjust.Multiply(0.1), "1": None, "2": None})
    with pytest.raises(AssertionError):
        flow.stratify(strat)
