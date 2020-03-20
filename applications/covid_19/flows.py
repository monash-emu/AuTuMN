from autumn.constants import Flow, Compartment
import copy

INFECTION_FLOWS = [
    {
        'type': Flow.INFECTION_FREQUENCY,
        'parameter': 'contact_rate',
        'origin': Compartment.SUSCEPTIBLE,
    }
]

DEATH_FLOWS = [
    {
        'type': Flow.COMPARTMENT_DEATH,
        'parameter': 'infect_death',
    }
]

WITHIN_EXPOSED_FLOWS = [
    {
        'type': Flow.STANDARD,
        'parameter': 'within_exposed',
    }
]

WITHIN_PRESYMPT_FLOWS = [
    {
        'type': Flow.STANDARD,
        'parameter': 'within_presympt',
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


def add_to_presympt_flows(list_of_flows, n_exposed, n_presympt, presympt_compartment_name, parameter_name):
    progression_flows = copy.deepcopy(PROGRESSION_FLOWS)
    progression_flows[0]['origin'] = \
        Compartment.EXPOSED + '_' + str(n_exposed) if n_exposed > 1 else Compartment.EXPOSED
    progression_flows[0]['to'] = \
        presympt_compartment_name + '_1' if n_presympt > 1 else presympt_compartment_name
    progression_flows[0]['parameter'] = parameter_name
    list_of_flows += progression_flows
    return list_of_flows


def add_to_infectious_flows(list_of_flows, n_presympt, n_infectious, infectious_compartment_name, parameter_name):
    progression_flows = copy.deepcopy(PROGRESSION_FLOWS)
    progression_flows[0]['origin'] = \
        'presympt_' + str(n_presympt) if n_presympt > 1 else 'presympt'
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
    infection_flows = copy.deepcopy(INFECTION_FLOWS)
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


def add_within_presympt_flows(list_of_flows, n_presympt, presympt_compartment_name):
    if n_presympt > 1:
        for i_flow in range(1, n_presympt):
            within_presympt_flows = copy.deepcopy(WITHIN_PRESYMPT_FLOWS)
            within_presympt_flows[0]['origin'] = \
                presympt_compartment_name + '_' + str(i_flow)
            within_presympt_flows[0]['to'] = \
                presympt_compartment_name + '_' + str(i_flow + 1)
            list_of_flows += within_presympt_flows
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


def replicate_compartment(
        n_replications,
        current_compartments,
        compartment_stem,
        infectious_compartments,
        initial_populations,
        infectious=False,
        infectious_seed=0.,
):
    """
    Implement n compartments of a certain type
    Also returns the names of the infectious compartments and the intial population evenly distributed across the
    replicated infectious compartments
    """

    # Add the compartment names to the working list of compartments
    compartments_to_add = \
        [compartment_stem] if \
            n_replications == 1 else \
            [compartment_stem + '_' + str(i_comp + 1) for i_comp in range(n_replications)]

    # Add the compartment names to the working list of infectious compartments, if the compartment is infectious
    infectious_compartments_to_add = \
        compartments_to_add if infectious else []

    # Add the infectious population to the initial conditions
    if infectious_seed == 0.:
        init_pop = {}
    elif n_replications == 1:
        init_pop = {compartment_stem: infectious_seed}
    else:
        init_pop = \
            {compartment_stem + '_' + str(i_infectious + 1): infectious_seed / float(n_replications) for
             i_infectious in range(n_replications)}
    initial_populations.update(init_pop)

    return current_compartments + compartments_to_add, \
           infectious_compartments + infectious_compartments_to_add, \
           initial_populations


def multiply_flow_value_for_multiple_compartments(model_parameters, compartment_name, parameter_name):
    """
    Multiply the progression rate through the compartments placed in series by the number of compartments, so that the
    average sojourn time in the group of compartment remains the same.
    """
    model_parameters['within_' + compartment_name] = \
        model_parameters[parameter_name] * \
        float(model_parameters['n_compartment_repeats'])
    return model_parameters


def add_infection_death_flows(list_of_flows, n_infectious):
    death_flows = copy.deepcopy(DEATH_FLOWS)
    if n_infectious > 1:
        for i_comp in range(n_infectious):
            death_flows[0]['origin'] = Compartment.INFECTIOUS + '_' + str(i_comp + 1)
    else:
        death_flows[0]['origin'] = Compartment.INFECTIOUS
    list_of_flows += death_flows
    return list_of_flows
