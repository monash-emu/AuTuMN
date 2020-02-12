"""
Standardised flow functions
"""
from autumn.constants import Flow, Compartment


def add_standard_latency_flows(list_of_flows):
    """
    Adds our standard latency flows to the list of flows to be implemented in the model
    """
    list_of_flows += LATENCY_FLOWS
    return list_of_flows


def add_standard_natural_history_flows(list_of_flows):
    """
    Adds our standard natural history to the list of flows to be implemented in the model
    """
    list_of_flows += NATURAL_HISTORY_FLOWS
    return list_of_flows


def add_standard_infection_flows(list_of_flows):
    """
    Adds our standard infection processes to the list of flows to be implemented in the model
    """
    list_of_flows += INFECTION_FLOWS
    return list_of_flows


def add_density_infection_flows(list_of_flows):
    """
    Adds our standard infection processes to the list of flows to be implemented in the model
    """
    list_of_flows += DENSITY_INFECTION_FLOWS
    return list_of_flows


LATENCY_FLOWS = [
    {
        "type": Flow.STANDARD,
        "parameter": "early_progression",
        "origin": Compartment.EARLY_LATENT,
        "to": Compartment.INFECTIOUS,
    },
    {
        "type": Flow.STANDARD,
        "parameter": "stabilisation",
        "origin": Compartment.EARLY_LATENT,
        "to": Compartment.LATE_LATENT,
    },
    {
        "type": Flow.STANDARD,
        "parameter": "late_progression",
        "origin": Compartment.LATE_LATENT,
        "to": Compartment.INFECTIOUS,
    },
]

NATURAL_HISTORY_FLOWS = [
    {
        "type": Flow.STANDARD,
        "parameter": "recovery",
        "origin": Compartment.INFECTIOUS,
        "to": Compartment.RECOVERED,
    },
    {"type": Flow.COMPARTMENT_DEATH, "parameter": "infect_death", "origin": Compartment.INFECTIOUS},
]

INFECTION_FLOWS = [
    {
        "type": Flow.INFECTION_FREQUENCY,
        "parameter": "contact_rate",
        "origin": Compartment.SUSCEPTIBLE,
        "to": Compartment.EARLY_LATENT,
    },
    {
        "type": Flow.INFECTION_FREQUENCY,
        "parameter": "contact_rate_recovered",
        "origin": Compartment.RECOVERED,
        "to": Compartment.EARLY_LATENT,
    },
    {
        "type": Flow.INFECTION_FREQUENCY,
        "parameter": "contact_rate_infected",
        "origin": Compartment.LATE_LATENT,
        "to": Compartment.EARLY_LATENT,
    },
]

DENSITY_INFECTION_FLOWS = [
    {
        "type": Flow.INFECTION_DENSITY,
        "parameter": "contact_rate",
        "origin": Compartment.SUSCEPTIBLE,
        "to": Compartment.EARLY_LATENT,
    },
    {
        "type": Flow.INFECTION_DENSITY,
        "parameter": "contact_rate_recovered",
        "origin": Compartment.RECOVERED,
        "to": Compartment.EARLY_LATENT,
    },
    {
        "type": Flow.INFECTION_DENSITY,
        "parameter": "contact_rate_infected",
        "origin": Compartment.LATE_LATENT,
        "to": Compartment.EARLY_LATENT,
    },
]
