import pytest
import numpy as np

from summer2 import adjust, Compartment
from summer2.flows import (
    CrudeBirthFlow,
    ReplacementBirthFlow,
    ImportFlow,
    DeathFlow,
    FractionalFlow,
    FunctionFlow,
    SojournFlow,
    InfectionDensityFlow,
    InfectionFrequencyFlow,
)


def test_function_flow_get_net_flow():
    def get_flow_rate(flow, comps, comp_vals, flows, flow_rates, time):
        i_pop = sum([comp_vals[c.idx] for c in comps if c.name == "I"])
        s_flows = sum([flow_rates[c.idx] for c in comps if c.name == "S"])
        return s_flows * i_pop * time

    flow = FunctionFlow(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=get_flow_rate,
        adjustments=[],
    )
    flow.source.idx = 1
    compartments = [Compartment("S"), Compartment("I"), Compartment("R")]
    for i, c in enumerate(compartments):
        c.idx = i

    compartment_values = np.array([1, 3, 5])
    flow_rates = np.array([2, 7, 13])
    net_flow = flow.get_net_flow(compartments, compartment_values, [flow], flow_rates, 7)
    assert net_flow == 2 * 3 * 7


def test_function_flow_get_net_flow_with_adjust():
    def get_flow_rate(flow, comps, comp_vals, flows, flow_rates, time):
        i_pop = sum([comp_vals[c.idx] for c in comps if c.name == "I"])
        s_flows = sum([flow_rates[c.idx] for c in comps if c.name == "S"])
        return s_flows * i_pop * time

    flow = FunctionFlow(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=get_flow_rate,
        adjustments=[adjust.Multiply(13)],
    )
    flow.source.idx = 1
    compartments = [Compartment("S"), Compartment("I"), Compartment("R")]
    for i, c in enumerate(compartments):
        c.idx = i

    compartment_values = np.array([1, 3, 5])
    flow_rates = np.array([2, 7, 13])
    net_flow = flow.get_net_flow(compartments, compartment_values, [flow], flow_rates, 7)
    assert net_flow == 2 * 3 * 7 * 13


def test_fractional_flow_get_net_flow():
    flow = FractionalFlow(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=lambda t: 2 * t,
        adjustments=[],
    )
    flow.source.idx = 1
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 2 * 3 * 7


def test_fractional_flow_get_net_flow_with_adjust():
    flow = FractionalFlow(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=lambda t: 2 * t,
        adjustments=[adjust.Multiply(13)],
    )
    flow.source.idx = 2
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 2 * 5 * 7 * 13


def test_sojourn_flow_get_net_flow():
    flow = SojournFlow(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=lambda t: 2 * t,
        adjustments=[],
    )
    flow.source.idx = 1
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 3 / (2 * 7)


def test_sojourn_flow_get_net_flow_with_adjust():
    flow = SojournFlow(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=lambda t: 2 * t,
        adjustments=[adjust.Multiply(13)],
    )
    flow.source.idx = 2
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 5 / (2 * 13 * 7)


def test_import_flow_get_net_flow():
    flow = ImportFlow(
        name="flow",
        dest=Compartment("S"),
        param=lambda t: 0.1 * t,
        adjustments=[],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 0.1 * 7


def test_import_flow_get_net_flow_with_adjust():
    flow = ImportFlow(
        name="flow",
        dest=Compartment("S"),
        param=lambda t: 0.1 * t,
        adjustments=[adjust.Multiply(13)],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 0.1 * 7 * 13


def test_crude_birth_flow_get_net_flow():
    flow = CrudeBirthFlow(
        name="flow",
        dest=Compartment("S"),
        param=lambda t: 0.1 * t,
        adjustments=[],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 0.1 * 7 * (1 + 3 + 5)


def test_crude_birth_flow_get_net_flow_with_adjust():
    flow = CrudeBirthFlow(
        name="flow",
        dest=Compartment("S"),
        param=lambda t: 0.1 * t,
        adjustments=[adjust.Multiply(13)],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 0.1 * 7 * (1 + 3 + 5) * 13


def test_replace_deaths_birth_flow_get_net_flow():
    flow = ReplacementBirthFlow(
        name="flow",
        dest=Compartment("S"),
        param=lambda t: 23,
        adjustments=[],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 23


def test_replace_deaths_birth_flow_get_net_flow_with_adjust():
    flow = ReplacementBirthFlow(
        name="flow",
        dest=Compartment("S"),
        param=lambda t: 23,
        adjustments=[adjust.Multiply(13)],
    )
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 23 * 13


def test_death_flow_get_net_flow():
    flow = DeathFlow(
        name="flow",
        source=Compartment("I"),
        param=lambda t: 2 * t,
        adjustments=[],
    )
    flow.source.idx = 1
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 2 * 3 * 7


def test_death_flow_get_net_flow_with_adjust():
    flow = DeathFlow(
        name="flow",
        source=Compartment("I"),
        param=lambda t: 2 * t,
        adjustments=[adjust.Multiply(13)],
    )
    flow.source.idx = 2
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 2 * 5 * 7 * 13


@pytest.mark.parametrize("FlowClass", [InfectionDensityFlow, InfectionFrequencyFlow])
def test_infection_get_net_flow(FlowClass):
    flow = FlowClass(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=lambda t: 2 * t,
        find_infectious_multiplier=lambda s, d: 23,
        adjustments=[],
    )
    flow.source.idx = 1
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 2 * 3 * 7 * 23


@pytest.mark.parametrize("FlowClass", [InfectionDensityFlow, InfectionFrequencyFlow])
def test_infection_get_net_flow_with_adjust(FlowClass):
    flow = FlowClass(
        name="flow",
        source=Compartment("I"),
        dest=Compartment("R"),
        param=lambda t: 2 * t,
        find_infectious_multiplier=lambda s, d: 23,
        adjustments=[adjust.Multiply(13)],
    )
    flow.source.idx = 1
    vals = np.array([1, 3, 5])
    net_flow = flow.get_net_flow(vals, 7)
    assert net_flow == 2 * 3 * 7 * 23 * 13
