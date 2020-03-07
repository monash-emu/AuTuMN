"""
Standardised flow functions
"""
from autumn.constants import Flow, Compartment

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
    {
        "type": Flow.COMPARTMENT_DEATH,
        "parameter": "infect_death",
        "origin": Compartment.INFECTIOUS
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

ACF_FLOWS = [
    {
        "type": Flow.STANDARD,
        "parameter": "acf_rate",
        "origin": Compartment.INFECTIOUS,
        "to": Compartment.RECOVERED,
    }
]

ACF_LTBI_FLOWS = [
    {
        "type": Flow.STANDARD,
        "parameter": "acf_ltbi_rate",
        "origin": Compartment.LATE_LATENT,
        "to": "ltbi_treated",
    }
]

CASE_DETECTION_FLOWS = [
    {
        "type": Flow.STANDARD,
        "parameter": "case_detection",
        "origin": Compartment.INFECTIOUS,
        "to": Compartment.RECOVERED,
    }
]

LATENCY_REINFECTION = [
    {
        "type": Flow.INFECTION_FREQUENCY,
        "parameter": "contact_rate_ltbi_treated",
        "origin": "ltbi_treated",
        "to": Compartment.EARLY_LATENT,
    }
]


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


def add_latency_progression(list_of_flows):
    """
    Adds standard latency progression flows
    """
    list_of_flows += LATENCY_REINFECTION
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


def add_case_detection(list_of_flows):
    """
    Adds standard passive (DOTS-based) case detection flows
    """
    list_of_flows += CASE_DETECTION_FLOWS
    return list_of_flows


def add_acf(list_of_flows):
    """
    Adds active case finding flows
    """
    list_of_flows += ACF_FLOWS
    return list_of_flows


def add_acf_ltbi(list_of_flows):
    """
    Adds standard flows for ACF linked to LTBI
    """
    list_of_flows += ACF_LTBI_FLOWS
    return list_of_flows


def get_incidence_connections():
    return {
        "incidence_early":
            {"origin": Compartment.EARLY_LATENT,
             "to": Compartment.INFECTIOUS},
        "incidence_late":
            {"origin": Compartment.LATE_LATENT,
             "to": Compartment.INFECTIOUS},
    }

