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


def add_transition_flows(list_of_flows, n_sequential, origin_compartment, to_compartment, parameter_name):
    """
    Add flow from end of sequential exposed compartments to start of presymptomatic compartments
    """
    list_of_flows += [{
        'type': Flow.STANDARD,
        'origin': origin_compartment + '_' + str(n_sequential) if n_sequential > 1 else origin_compartment,
        'to': to_compartment + '_1' if n_sequential > 1 else to_compartment,
        'parameter': parameter_name
    }]
    return list_of_flows


def add_recovery_flows(working_flows, n_infectious):
    """
    Add standard recovery flows
    Differs from the transition flows in that the recovered compartment is never duplicated
    """
    working_flows += [{
        'type': Flow.STANDARD,
        'parameter': 'within_infectious',
        'to': Compartment.RECOVERED,
        'origin': Compartment.INFECTIOUS + '_' + str(n_infectious) if n_infectious > 1 else Compartment.INFECTIOUS
    }]
    return working_flows


def add_sequential_compartment_flows(working_flows, n_compartments, compartment_name):
    """
    Add standard flows for progression through sequential compartments
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
            'origin': Compartment.INFECTIOUS + '_' + str(i_comp + 1) if
            n_infectious > 1 else
            Compartment.INFECTIOUS
        })
    return working_flows


def multiply_flow_value_for_multiple_compartments(model_parameters, compartment_name, parameter_name):
    """
    Multiply the progression rate through the compartments placed in series by the number of compartments, so that the
    average sojourn time in the group of compartment remains the same.
    """
    model_parameters['within_' + compartment_name] = \
        model_parameters[parameter_name] * \
        float(model_parameters['n_compartment_repeats'])
    return model_parameters
