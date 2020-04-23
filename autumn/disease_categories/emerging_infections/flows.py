from autumn.constants import Flow, Compartment


def add_infection_flows(working_flows, n_exposed):
    """
    Add standard infection flows for transition from susceptible to exposed through infection
    """
    working_flows += [{
        'type': Flow.INFECTION_FREQUENCY,
        'parameter': 'contact_rate',
        'origin': Compartment.SUSCEPTIBLE,
        'to': Compartment.EXPOSED + '_1' if n_exposed > 1 else Compartment.EXPOSED
    }]
    return working_flows


def add_transition_flows(list_of_flows, n_origin, n_to, origin_compartment, to_compartment, parameter_name):
    """
    Add flow from end of sequential exposed compartments to start of presymptomatic compartments
    """
    list_of_flows += [{
        'type': Flow.STANDARD,
        'origin': origin_compartment + '_' + str(n_origin) if n_origin > 1 else origin_compartment,
        'to': to_compartment + '_1' if n_to > 1 else to_compartment,
        'parameter': parameter_name
    }]
    return list_of_flows


def add_recovery_flows(working_flows, n_late):
    """
    Add standard recovery flows
    Differs from the transition flows in that the recovered compartment is never duplicated
    """
    working_flows += [{
        'type': Flow.STANDARD,
        'parameter': 'within_late',
        'to': Compartment.RECOVERED,
        'origin': Compartment.LATE_INFECTIOUS + '_' + str(n_late) if
        n_late > 1 else Compartment.LATE_INFECTIOUS
    }]
    return working_flows


def add_sequential_compartment_flows(working_flows, n_compartments, compartment_name):
    """
    Add standard flows for progression through any sequential compartments
    """
    for i_flow in range(1, n_compartments):
        working_flows += [{
            'type': Flow.STANDARD,
            'parameter': 'within_' + compartment_name,
            'origin': compartment_name + '_' + str(i_flow),
            'to': compartment_name + '_' + str(i_flow + 1)
        }]
    return working_flows


def add_infection_death_flows(working_flows, n_infectious):
    """
    Add infection-related death flows to the infectious compartments
    """
    for i_comp in range(n_infectious):
        working_flows.append({
            'type': Flow.COMPARTMENT_DEATH,
            'parameter': 'infect_death',
            'origin': Compartment.LATE_INFECTIOUS + '_' + str(i_comp + 1) if
            n_infectious > 1 else
            Compartment.LATE_INFECTIOUS
        })
    return working_flows
