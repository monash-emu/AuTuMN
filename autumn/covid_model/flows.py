from autumn.constants import Flow, Compartment

INFECTION_FLOWS = [
    {
        'type': Flow.INFECTION_FREQUENCY,
        'parameter': 'contact_rate',
        'origin': Compartment.SUSCEPTIBLE,
        'to': Compartment.EXPOSED
    }
]

PROGRESSION_FLOWS = [
    {
        'type': Flow.STANDARD,
        'parameter': 'progression',
        'origin': Compartment.EXPOSED,
        'to': Compartment.INFECTIOUS,
    }
]

RECOVERY_FLOWS = [
    {
        'type': Flow.STANDARD,
        'parameter': 'recovery',
        'origin': Compartment.INFECTIOUS,
        'to': Compartment.RECOVERED,
    }
]


def add_progression_flows(list_of_flows):
    list_of_flows += PROGRESSION_FLOWS
    return list_of_flows


def add_recovery_flows(list_of_flows):
    list_of_flows += RECOVERY_FLOWS
    return list_of_flows


def add_infection_flows(list_of_flows):
    list_of_flows += INFECTION_FLOWS
    return list_of_flows
