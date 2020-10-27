import numpy as np

from summer.flow import CrudeBirthFlow, ReplacementBirthFlow
from summer.compartment import Compartment
from summer.stratification import Stratification
from summer.constants import FlowAdjustment


def test_crude_birth_flow_get_net_flow():
    flow = CrudeBirthFlow(
        dest=Compartment("susceptible"),
        param_name="crude_birth_rate",
        param_func=_get_param_value,
        adjustments=[],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 0.1 * 7 * (1 + 3 + 5)


def test_crude_birth_flow_get_net_flow_with_adjust():
    flow = CrudeBirthFlow(
        dest=Compartment("susceptible"),
        param_name="crude_birth_rate",
        param_func=_get_param_value,
        adjustments=[(FlowAdjustment.MULTIPLY, 13)],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 0.1 * 7 * 13 * (1 + 3 + 5)


def test_replace_deaths_birth_flow_get_net_flow():
    flow = ReplacementBirthFlow(
        dest=Compartment("susceptible"),
        get_total_deaths=_get_total_deaths,
        adjustments=[],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 23


def test_replace_deaths_birth_flow_get_net_flow_with_adjust():
    flow = ReplacementBirthFlow(
        dest=Compartment("susceptible"),
        get_total_deaths=_get_total_deaths,
        adjustments=[(FlowAdjustment.MULTIPLY, 13)],
    )
    flow.dest.idx = 2
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 23 * 13


def _get_param_value(name, time):
    return 0.1 * time


def _get_total_deaths():
    return 23
