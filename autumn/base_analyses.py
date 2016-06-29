
import random


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

