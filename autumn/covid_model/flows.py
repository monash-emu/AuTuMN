from autumn.constants import Flow, Compartment
import copy

INFECTION_FLOWS = [
    {
        'type': Flow.INFECTION_FREQUENCY,
        'parameter': 'contact_rate',
        'origin': Compartment.SUSCEPTIBLE,
    }
]

WITHIN_EXPOSED_FLOWS = [
    {
        'type': Flow.STANDARD,
        'parameter': 'within_exposed',
    }
]


PROGRESSION_FLOWS = [
    {
        'type': Flow.STANDARD,
        'to': Compartment.INFECTIOUS
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


def add_progression_flows(list_of_flows, n_exposed):
    progression_flows = PROGRESSION_FLOWS
    progression_flows[0]['origin'] = \
        Compartment.EXPOSED + '_' + str(n_exposed) if n_exposed > 0 else Compartment.EXPOSED
    progression_flows[0]['parameter'] = 'within_exposed' if n_exposed > 0 else 'within_exposed'
    list_of_flows += PROGRESSION_FLOWS
    return list_of_flows


def add_recovery_flows(list_of_flows):
    list_of_flows += RECOVERY_FLOWS
    return list_of_flows


def add_infection_flows(list_of_flows, n_exposed):
    infection_flows = INFECTION_FLOWS
    infection_flows[0]['to'] = Compartment.EXPOSED + '_1' if n_exposed > 0 else Compartment.EXPOSED
    list_of_flows += infection_flows
    return list_of_flows


def add_within_exposed_flows(list_of_flows, n_exposed):
    if n_exposed > 1:
        for i_flow in range(1, n_exposed):
            within_exposed_flows = copy.deepcopy(WITHIN_EXPOSED_FLOWS)
            within_exposed_flows[0]['origin'] = \
                Compartment.EXPOSED + '_' + str(i_flow)
            within_exposed_flows[0]['to'] = \
                Compartment.EXPOSED + '_' + str(i_flow + 1)
            list_of_flows += within_exposed_flows
    return list_of_flows



