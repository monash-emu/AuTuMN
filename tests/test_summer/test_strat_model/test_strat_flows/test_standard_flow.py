import numpy as np

from summer.flow import StandardFlow
from summer.compartment import Compartment
from summer.stratification import Stratification
from summer.constants import FlowAdjustment


def test_standard_flow_get_net_flow():
    flow = StandardFlow(
        source=Compartment("infect"),
        dest=Compartment("happy"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[],
    )
    flow.source.idx = 1
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 2 * 3 * 7


def test_standard_flow_get_net_flow_with_adjust():
    flow = StandardFlow(
        source=Compartment("infect"),
        dest=Compartment("happy"),
        param_name="foo",
        param_func=_get_param_value,
        adjustments=[(FlowAdjustment.MULTIPLY, 13)],
    )
    flow.source.idx = 2
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 2 * 5 * 7 * 13


def _get_param_value(name, time):
    return 2 * time
