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
        "parameter": "contact_rate_late_latent",
        "origin": Compartment.LATE_LATENT,
        "to": Compartment.EARLY_LATENT,
    },
]

LATENCY_FLOWS = [
    {
        "type": Flow.STANDARD,
        "parameter": "early_progression",
        "origin": Compartment.EARLY_LATENT,
        "to": Compartment.EARLY_INFECTIOUS,
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
        "to": Compartment.EARLY_INFECTIOUS,
    },
]

NATURAL_HISTORY_FLOWS = [
    {
        "type": Flow.STANDARD,
        "parameter": "recovery",
        "origin": Compartment.EARLY_INFECTIOUS,
        "to": Compartment.RECOVERED,
    },
    {
        "type": Flow.COMPARTMENT_DEATH,
        "parameter": "infect_death",
        "origin": Compartment.EARLY_INFECTIOUS,
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
        "parameter": "contact_rate_late_latent",
        "origin": Compartment.LATE_LATENT,
        "to": Compartment.EARLY_LATENT,
    },
]

ACF_FLOWS = [
    {
        "type": Flow.STANDARD,
        "parameter": "acf_rate",
        "origin": Compartment.EARLY_INFECTIOUS,
        "to": Compartment.ON_TREATMENT,
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
    {"type": Flow.STANDARD, "parameter": "case_detection", "origin": Compartment.EARLY_INFECTIOUS,}
]

LATENCY_REINFECTION = [
    {
        "type": Flow.INFECTION_FREQUENCY,
        "parameter": "contact_rate_ltbi_treated",
        "origin": "ltbi_treated",
        "to": Compartment.EARLY_LATENT,
    }
]

TREATMENT_FLOWS = [
    {
        "type": Flow.STANDARD,
        "parameter": "treatment_success",
        "origin": Compartment.ON_TREATMENT,
        "to": Compartment.RECOVERED,
    },
    {
        "type": Flow.STANDARD,
        "parameter": "treatment_nonsuccess",
        "origin": Compartment.ON_TREATMENT,
        "to": Compartment.EARLY_INFECTIOUS,
    },
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


def add_case_detection(list_of_flows, available_compartments):
    """
    Adds standard passive (DOTS-based) case detection flows
    """
    case_detection_flows = CASE_DETECTION_FLOWS
    case_detection_flows[0].update(
        {
            "to": Compartment.ON_TREATMENT
            if Compartment.ON_TREATMENT in available_compartments
            else Compartment.RECOVERED
        }
    )
    list_of_flows += CASE_DETECTION_FLOWS
    return list_of_flows


def add_treatment_flows(list_of_flows):
    list_of_flows += TREATMENT_FLOWS
    return list_of_flows


def add_acf(list_of_flows, available_compartments):
    """
    Adds active case finding flows
    """
    acf_flows = ACF_FLOWS
    acf_flows[0].update(
        {
            "to": Compartment.ON_TREATMENT
            if Compartment.ON_TREATMENT in available_compartments
            else Compartment.RECOVERED
        }
    )
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
        "incidence_early": {
            "origin": Compartment.EARLY_LATENT,
            "to": Compartment.EARLY_INFECTIOUS,
            "origin_condition": "",
            "to_condition": "",
        },
        "incidence_late": {
            "origin": Compartment.LATE_LATENT,
            "to": Compartment.EARLY_INFECTIOUS,
            "origin_condition": "",
            "to_condition": "",
        },
    }


def get_notifications_connections():
    return {
        "notifications": {
            "origin": Compartment.EARLY_INFECTIOUS,
            "to": Compartment.ON_TREATMENT,
            "origin_condition": "",
            "to_condition": "",
        }
    }
