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

WITHIN_INFECTIOUS_FLOWS = [
    {
        'type': Flow.STANDARD,
        'parameter': 'within_infectious',
    }
]

PROGRESSION_FLOWS = [
    {
        'type': Flow.STANDARD,
    }
]

RECOVERY_FLOWS = [
    {
        'type': Flow.STANDARD,
        'parameter': 'within_infectious',
        'to': Compartment.RECOVERED,
    }
]


def add_progression_flows(list_of_flows, n_exposed, n_infectious, infectious_compartment_name, parameter_name):
    progression_flows = copy.deepcopy(PROGRESSION_FLOWS)
    progression_flows[0]['origin'] = \
        Compartment.EXPOSED + '_' + str(n_exposed) if n_exposed > 1 else Compartment.EXPOSED
    progression_flows[0]['to'] = \
        infectious_compartment_name + '_1' if n_infectious > 1 else infectious_compartment_name
    progression_flows[0]['parameter'] = parameter_name
    list_of_flows += progression_flows
    return list_of_flows


def add_recovery_flows(list_of_flows, n_infectious, infectious_compartment_name):
    recovery_flows = copy.deepcopy(RECOVERY_FLOWS)
    recovery_flows[0]['origin'] = \
        infectious_compartment_name + '_' + str(n_infectious) if n_infectious > 1 else infectious_compartment_name
    list_of_flows += recovery_flows
    return list_of_flows


def add_infection_flows(list_of_flows, n_exposed):
    infection_flows = INFECTION_FLOWS
    infection_flows[0]['to'] = Compartment.EXPOSED + '_1' if n_exposed > 1 else Compartment.EXPOSED
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


def add_within_infectious_flows(list_of_flows, n_infectious, infectious_compartment_name):
    if n_infectious > 1:
        for i_flow in range(1, n_infectious):
            within_infectious_flows = copy.deepcopy(WITHIN_INFECTIOUS_FLOWS)
            within_infectious_flows[0]['origin'] = \
                infectious_compartment_name + '_' + str(i_flow)
            within_infectious_flows[0]['to'] = \
                infectious_compartment_name + '_' + str(i_flow + 1)
            list_of_flows += within_infectious_flows
    return list_of_flows


def replicate_compartment(n_replications, current_compartments, compartment_stem, infectious_seed=0.):
    """
    Implement n compartments of a certain type
    Also returns the names of the infectious compartments and the intial population evenly distributed across the
    replicated infectious compartments
    """
    if n_replications == 1:
        current_compartments += [compartment_stem]
        infectious_compartments = [compartment_stem]
        init_pop = {
            compartment_stem: infectious_seed,
        }
    else:
        infectious_compartments, init_pop = [], {}
        for i_infectious in range(n_replications):
            current_compartments += [compartment_stem + '_' + str(i_infectious + 1)]
            infectious_compartments += [compartment_stem + '_' + str(i_infectious + 1)]
            init_pop[compartment_stem + '_' + str(i_infectious + 1)] = \
                infectious_seed / float(n_replications)
    return current_compartments, infectious_compartments, init_pop


def multiply_flow_value_for_multiple_compartments(model_parameters, compartment_name, parameter_name):
    model_parameters['within_' + compartment_name] = \
        model_parameters[parameter_name] * \
        float(model_parameters['n_' + compartment_name + '_compartments'])
    return model_parameters
