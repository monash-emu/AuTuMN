"""
Standardised flow functions
"""
from autumn.constants import Flow, Compartment


def add_standard_latency_flows(list_of_flows):
    """
    adds our standard latency flows to the list of flows to be implemented in the model

    :param list_of_flows: list
        existing flows for implementation in the model
    :return: list_of_flows: list
        list of flows updated to include the standard latency flows
    """
    list_of_flows += [
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
    return list_of_flows


def add_standard_natural_history_flows(list_of_flows):
    """
    adds our standard natural history to the list of flows to be implemented in the model

    :param list_of_flows: list
        existing flows for implementation in the model
    :return: list_of_flows: list
        list of flows updated to include the standard latency flows
    """
    list_of_flows += [
        {
            "type": Flow.STANDARD,
            "parameter": "recovery",
            "origin": Compartment.INFECTIOUS,
            "to": Compartment.RECOVERED,
        },
        {
            "type": Flow.COMPARTMENT_DEATH,
            "parameter": "infect_death",
            "origin": Compartment.INFECTIOUS,
        },
    ]
    return list_of_flows


def add_standard_infection_flows(list_of_flows):
    """
    adds our standard infection processes to the list of flows to be implemented in the model

    :param list_of_flows: list
        existing flows for implementation in the model
    :return: list_of_flows: list
        list of flows updated to include the standard infection processes
    """
    list_of_flows += [
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
    return list_of_flows


def add_density_infection_flows(list_of_flows):
    """
    adds our standard infection processes to the list of flows to be implemented in the model

    :param list_of_flows: list
        existing flows for implementation in the model
    :return: list_of_flows: list
        list of flows updated to include the standard infection processes
    """
    list_of_flows += [
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
    return list_of_flows
