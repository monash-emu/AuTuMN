import pytest

from summer2.adjust import Overwrite, Multiply
from summer2 import Compartment
from summer2.flows import BaseEntryFlow


class EntryFlow(BaseEntryFlow):
    def get_net_flow(self, compartment_values, time):
        return 1


def some_func(t):
    return t + 1


ADJUST_TESTS = (
    # Empty list should not change.
    ([], []),
    # Single Overwrite should not change
    ([Overwrite(0.3)], [Overwrite(0.3)]),
    ([Overwrite(some_func)], [Overwrite(some_func)]),
    # Single Multiply should not change
    ([Multiply(0.3)], [Multiply(0.3)]),
    ([Multiply(some_func)], [Multiply(some_func)]),
    # Most recent overwrite goes 1st
    ([Multiply(0.1), Overwrite(0.3), Multiply(0.3)], [Overwrite(0.3), Multiply(0.3)]),
    (
        [Overwrite(0.1), Multiply(0.1), Overwrite(0.2), Overwrite(0.3), Multiply(0.3)],
        [Overwrite(0.3), Multiply(0.3)],
    ),
    # Constant Multiplies are collated
    (
        [Multiply(0.1), Multiply(0.3), Multiply(2)],
        [Multiply(0.06)],
    ),
    (
        [Multiply(0.1), Multiply(0.3), Multiply(some_func), Multiply(2)],
        [Multiply(0.06), Multiply(some_func)],
    ),
    # All optimizations at once
    (
        [
            Multiply(5),
            Overwrite(2),
            Overwrite(3),
            Multiply(0.1),
            Multiply(0.3),
            Multiply(some_func),
            Multiply(2),
            Multiply(some_func),
        ],
        [
            Overwrite(3),
            Multiply(0.06),
            Multiply(some_func),
            Multiply(some_func),
        ],
    ),
)


@pytest.mark.parametrize("adjustments,optimized", ADJUST_TESTS)
def test_optimize_adjustments(adjustments, optimized):
    flow = EntryFlow("flow", Compartment("S"), 1, adjustments)
    flow.optimize_adjustments()
    assert len(flow.adjustments) == len(optimized)
    assert all([a._is_equal(o) for a, o in zip(flow.adjustments, optimized)])
