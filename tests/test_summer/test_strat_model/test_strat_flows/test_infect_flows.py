import pytest
import numpy as np

from summer.flow import InfectionDensityFlow, InfectionFrequencyFlow
from summer.compartment import Compartment
from summer.stratification import Stratification
from summer.constants import FlowAdjustment


@pytest.mark.parametrize("FlowClass", [InfectionDensityFlow, InfectionFrequencyFlow])
def test_infect_get_net_flow(FlowClass):
    flow = FlowClass(
        source=Compartment("infect"),
        dest=Compartment("happy"),
        param_name="foo",
        param_func=_get_param_value,
        find_infectious_multiplier=find_infectious_multiplier,
        adjustments=[],
    )
    flow.source.idx = 1
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 2 * 3 * 7 * 23


@pytest.mark.parametrize("FlowClass", [InfectionDensityFlow, InfectionFrequencyFlow])
def test_infect_density_get_net_flow_with_adjust(FlowClass):
    flow = FlowClass(
        source=Compartment("infect"),
        dest=Compartment("happy"),
        param_name="foo",
        param_func=_get_param_value,
        find_infectious_multiplier=find_infectious_multiplier,
        adjustments=[(FlowAdjustment.MULTIPLY, 13)],
    )
    flow.source.idx = 2
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 2 * 5 * 7 * 13 * 23


def find_infectious_multiplier(comp):
    return 23


def _get_param_value(name, time):
    return 2 * time
