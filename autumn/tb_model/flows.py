"""
Standardised flow functions
"""
from autumn.constants import Flow, Compartment


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

ACF_LTBI_FLOWS = [
    {
        "type": "standard_flows",
        "parameter": "acf_ltbi_rate",
        "origin": Compartment.LATE_LATENT,
        "to": "ltbi_treated",
    }
]


def add_case_detection(list_of_flows):
    list_of_flows += [
        {
            "type": "standard_flows",
            "parameter": "case_detection",
            "origin": "infectious",
            "to": "recovered",
        }
    ]
    return list_of_flows


def add_latency_progression(list_of_flows):
    list_of_flows += [
        {
            "type": "infection_frequency",
            "parameter": "contact_rate_ltbi_treated",
            "origin": "ltbi_treated",
            "to": "early_latent",
        }
    ]
    return list_of_flows


def add_acf(list_of_flows):
    list_of_flows += [
        {
            "type": "standard_flows",
            "parameter": "acf_rate",
            "origin": "infectious",
            "to": "recovered",
        }
    ]
    return list_of_flows


def add_acf_ltbi(list_of_flows):
    """
    Add standard flows for ACF linked to LTBI
    """
    list_of_flows += ACF_LTBI_FLOWS
    return list_of_flows


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


