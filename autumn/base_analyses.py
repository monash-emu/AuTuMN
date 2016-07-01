
import random
import base_analyses

def capitalise_first_letter(old_string):

    """
    Really simple method to capitalise the first character of a string

    Args:
        old_string: The string to be capitalised

    Returns:
        new_string: The capitalised string

    """

    new_string = ''
    for i in range(len(old_string)):
        if i == 0:
            new_string += old_string[i].upper()
        else:
            new_string += old_string[i]

    return new_string

def introduce_model(models, model_name):

    return 'Initialising model for ' + capitalise_first_letter(
        models[model_name].inputs['model_constants']['country']) + ' with key "' + model_name + '".'

def describe_model(models, model_name):

    model = models[model_name]

    returned_string = 'Model "' + model_name + '" has the following attributes:\n'
    if model.inputs['model_constants']['n_organs'][0] <= 1:
        returned_string += 'unstratified by organ involvement,\n'
    else:
        returned_string += str(model.inputs['model_constants']['n_organs'][0]) + ' organ strata,\n'
    if model.inputs['model_constants']['n_comorbidities'][0] <= 1:
        returned_string += 'unstratified by comorbidities, \n'
    else:
        returned_string += str(model.inputs['model_constants']['n_comorbidities'][0]) + ' comorbidity strata'
    if model.inputs['model_constants']['n_strains'][0] <= 1:
        returned_string += 'single strain model.'
    else:
        returned_string += str(model.inputs['model_constants']['n_strains'][0]) + ' circulating strains.'

    return returned_string


def sum_over_compartments(model, compartment_types):

    """
    General method to sum sets of compartments
    Args:
        compartment_types: List of the compartments to be summed over

    Returns:
        summed_soln: Dictionary of lists for the sums of each compartment
        summed_denominator: List of the denominator values
    """

    summed_soln = {}
    summed_denominator \
        = [0] * len(random.sample(model.compartment_soln.items(), 1)[0][1])
    for compartment_type in compartment_types:
        summed_soln[compartment_type] \
            = [0] * len(random.sample(model.compartment_soln.items(), 1)[0][1])
        for label in model.labels:
            if compartment_type in label:
                summed_soln[compartment_type] = [
                    a + b
                    for a, b
                    in zip(
                        summed_soln[compartment_type],
                        model.compartment_soln[label])]
                summed_denominator += model.compartment_soln[label]
    return summed_soln, summed_denominator


def get_fraction_soln(numerator_labels, numerators, denominator):

    """
    General method for calculating the proportion of a subgroup of the population
    in each compartment type
    Args:
        numerator_labels: Labels of numerator compartments
        numerators: Lists of values of each numerator
        denominator: List of values for the denominator

    Returns:
        Fractions of the denominator in each numerator
    """

    fraction = {}

    # Just to avoid warnings, replace any zeros in the denominators with small values
    # (numerators will still be zero, so all fractions should be zero)
    for i in range(len(denominator)):
        if denominator[i] == 0.:
            denominator[i] = 1E-3

    for label in numerator_labels:
        fraction[label] = [
            v / t
            for v, t
            in zip(
                numerators[label],
                denominator)]
    return fraction


def sum_over_compartments_bycategory(model, compartment_types, categories):

    # Waiting for Bosco's input, so won't fully comment yet
    summed_soln = {}
    # HELP BOSCO
    # The following line of code works, but I'm sure this isn't the best approach:
    summed_denominator \
        = [0] * len(random.sample(model.compartment_soln.items(), 1)[0][1])
    compartment_types_bycategory = []
    # HELP BOSCO
    # I think there is probably a more elegant way to do the following, but perhaps not.
    # Also, it could possibly be better generalised. That is, rather than insisting that
    # strain applies to all compartments except for the susceptible, it might be possible
    # to say that strain applies to all compartments except for those that have any
    # strain in their label.
    if categories == 'strain':
        working_categories = model.strains
    elif categories == 'organ':
        working_categories = model.organ_status
    for compartment_type in compartment_types:
        if (categories == 'strain' and 'susceptible' in compartment_type) \
                or (categories == 'organ' and
                        ('susceptible' in compartment_type or 'latent' in compartment_type)):
            summed_soln[compartment_type] \
                = [0] * len(random.sample(model.compartment_soln.items(), 1)[0][1])
            for label in model.labels:
                if compartment_type in label:
                    summed_soln[compartment_type] = [
                        a + b
                        for a, b
                        in zip(
                            summed_soln[compartment_type],
                            model.compartment_soln[label])]
                    summed_denominator += model.compartment_soln[label]
                if compartment_type in label \
                        and compartment_type not in compartment_types_bycategory:
                    compartment_types_bycategory.append(compartment_type)
        else:
            for working_category in working_categories:
                compartment_types_bycategory.append(compartment_type + working_category)
                summed_soln[compartment_type + working_category] \
                    = [0] * len(random.sample(model.compartment_soln.items(), 1)[0][1])
                for label in model.labels:
                    if compartment_type in label and working_category in label:
                        summed_soln[compartment_type + working_category] = [
                            a + b
                            for a, b
                            in zip(
                                summed_soln[compartment_type + working_category],
                                model.compartment_soln[label])]
                        summed_denominator += model.compartment_soln[label]

    return summed_soln, summed_denominator, compartment_types_bycategory


def find_fractions(model):

    # All compartmental disease stages
    dictionary_of_classifications = {
        'compartment_types':
            ['susceptible_fully',
             'susceptible_vac',
             'susceptible_treated',
             'latent_early',
             'latent_late',
             'active',
             'detect',
             'missed',
             'treatment_infect',
             'treatment_noninfect'],
        'broad_compartment_types':
            ['susceptible',
             'latent',
             'active',
             'missed',
             'treatment']
    }
    if model.is_lowquality:
        dictionary_of_classifications['compartment_types'] += ['lowquality']
        dictionary_of_classifications['broad_compartment_types'] += ['lowquality']

    # The following was previously the additional diagnostics code in model.py
    subgroup_solns = {}
    subgroup_fractions = {}
    for category in dictionary_of_classifications:
        subgroup_solns[category], compartment_type_denominator \
            = sum_over_compartments(model, dictionary_of_classifications[category])
        subgroup_fractions[category] \
            = get_fraction_soln(
            dictionary_of_classifications[category],
            subgroup_solns[category],
            compartment_type_denominator)
        for strata in ['strain', 'organ']:
            subgroup_solns[category + strata], compartment_type_bystrain_denominator, compartment_types_bystrain \
                = sum_over_compartments_bycategory(model, dictionary_of_classifications[category], strata)
            subgroup_fractions[category + strata] \
                = get_fraction_soln(
                compartment_types_bystrain,
                subgroup_solns[category + strata],
                compartment_type_bystrain_denominator)

    return subgroup_solns, subgroup_fractions


def calculate_additional_diagnostics(model):

    """
    Calculate fractions and populations within subgroups of the full population
    """

    groups = {
        'ever_infected': ['susceptible_treated', 'latent', 'active', 'missed', 'lowquality', 'detect', 'treatment'],
        'infected': ['latent', 'active', 'missed', 'lowquality', 'detect', 'treatment'],
        'active': ['active', 'missed', 'detect', 'lowquality', 'treatment'],
        'infectious': ['active', 'missed', 'lowquality', 'detect', 'treatment_infect'],
        'identified': ['detect', 'treatment'],
        'treatment': ['treatment_infect', 'treatment_noninfect']}

    subgroup_solns = {}
    subgroup_fractions = {}

    for key in groups:
        subgroup_solns[key], compartment_denominator \
            = sum_over_compartments(model, groups[key])
        subgroup_fractions[key] \
            = get_fraction_soln(
            groups[key],
            subgroup_solns[key],
            compartment_denominator)

    return subgroup_solns, subgroup_fractions
