
import random
from matplotlib import pyplot, patches
from scipy import exp
import outputs
import cPickle as pickle
from numpy import isfinite


"""
Note that this module is intended only to contain stand-alone functions for use by multiple other modules.
Object-oriented structures are not intended to be kept here.
"""


def is_parameter_value_valid(parameter):

    """
    Determine whether a number (typically a parameter value) is finite and positive.
    """

    return isfinite(parameter) and parameter > 0.


def find_scenario_string_from_number(scenario):

    """
    Find a string to represent a scenario from it's number (or None in the case of baseline).

    Args:
        scenario: The scenario value or None for baseline
        return: The string representing the scenario
    """

    if scenario is None:
        scenario_name = 'baseline'
    else:
        scenario_name = 'scenario_' + str(scenario)

    return scenario_name


def find_scenario_number_from_string(scenario):

    """
    Reverse of the above method. Not currently used, but likely to be needed.

    Args:
        scenario: Scenario string
    Returns:
        scenario_number: The scenario number or None for baseline
    """

    if scenario == 'baseline':
        scenario_number = None
    else:
        scenario_number = int(scenario[9:])

    return scenario_number


def capitalise_first_letter(old_string):

    """
    Really simple method to capitalise the first character of a string.

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


def replace_underscore_with_space(original_string):

    """
    Another really simple method to remove underscores and replace with spaces for titles of plots.

    Args:
        original_string: String with underscores
    Returns:
        replaced_string: String with underscores replaced
    """

    replaced_string = ''
    for i in range(len(original_string)):
        if original_string[i] == '_':
            replaced_string += ' '
        else:
            replaced_string += original_string[i]

    return replaced_string


def capitalise_and_remove_underscore(original_string):

    """
    Combine the previous two methods used to create titles.

    Args:
        original_string: String to be modified
    Return:
        revised string
    """

    return capitalise_first_letter(replace_underscore_with_space(original_string))


def adjust_country_name(country_name):

    """
    Currently very simple method to convert one country's name into that used by the GTB Report. However, likely to
    need to expand this as we work with more countries.

    Args:
        country_name: String for the original country name
    Returns:
        adjusted_country_name: Adjusted string
    """

    adjusted_country_name = country_name
    if country_name == 'Philippines':
        adjusted_country_name = country_name + ' (the)'
    return adjusted_country_name


def find_title_from_dictionary(name):

    """
    Function to store nicer strings for plot titles in a dictionary and extract
    for scale-up functions (initially, although could be used more widely).

    Args:
        name: The scale-up function (or other string for conversion)

    Returns:
        String for title of plots
    """

    dictionary_of_names = {
        'program_prop_vaccination':
            'Vaccination coverage',
        'program_prop_treatment_success':
            'Treatment success rate',
        'program_prop_xpert':
            'GeneXpert coverage',
        'program_prop_detect':
            'Case detection rate',
        'program_prop_treatment_death':
            'Death rate on treatment',
        'program_prop_algorithm_sensitivity':
            'Diagnostic algorithm sensitivity',
        'program_prop_ipt':
            'All ages IPT coverage',
        'program_prop_shortcourse_mdr':
            'Short-course MDR-TB regimen',
        'econ_program_unitcost_ipt':
            'IPT unit cost',
        'program_cost_vaccination':
            'Vaccination program cost (?dummy)',
        'econ_program_unitcost_vaccination':
            'Unit cost of BCG',
        'econ_program_totalcost_ipt':
            'IPT program cost',
        'econ_program_totalcost_vaccination':
            'Vaccination program cost',
        'demo_rate_birth':
            'Birth rate',
        'demo_life_expectancy':
            'Life expectancy',
        'econ_inflation':
            'Inflation rate',
        'econ_cpi':
            'Consumer price index',
        'program_timeperiod_await_treatment_smearpos':
            'Smear-positive',
        'program_timeperiod_await_treatment_smearneg':
            'Smear-negative',
        'program_timeperiod_await_treatment_extrapul':
            'Extrapulmonary',
        'program_prop_':
            'programmatic time-variant',
        'econ':
            'economic',
        'econ_':
            'economic',
        'demo':
            'demographic',
        'demo_':
            'demographic',
        'program_other':
            'unclassified',
        'start_time':
            ' from start of model run',
        'early_time':
            ' from start of model run',  # This is a bit of a fib
        'recent_time':
            ' over recent years',
        '_diabetes':
            'Diabetes',
        '_hiv':
            'HIV',
        '_nocomorb':
            'No risk group',
        'age':
            'age group',
        'comorbidity':
            'risk group',
        'program_prop_xpertacf':
            'Xpert-based ACF coverage',
        'program_prop_smearacf':
            'Smear-based ACF coverage',
        'program_prop_treatment_support':
            'Treatment support coverage',
        'program_prop_ipt_age0to5':
            'IPT in under 5s coverage',
        'program_prop_ipt_age5to15':
            'IPT in 5 to 15s coverage',
        'program_timeperiod':
            'programmatic time to treatment',
        'program_prop_engage_lowquality':
            'Engage low-quality sector',
        'engage_lowquality':
            'Engage low-quality sector',
        'program_prop_xpertacf_prison':
            'Prison ACF',
        'program_prop_xpertacf_indigenous':
            'Indigenous ACF',
        'program_prop_xpertacf_urbanpoor':
            'Urban poor ACF',
        'program_prop_xpertacf_ruralpoor':
            'Rural poor ACF',
        'epi_':
            'epidemiological',
        'epi_prop_smearneg':
            'Proportion smear-negative',
        'epi_prop_smearpos':
            'Proportion smear-positive',
        'xpertacf':
            'ACF using Xpert',
        'vaccination':
            'BCG vaccination',
        'smearacf':
            'ACF using sputum smear',
        'treatment_support':
            'Treatment support',
        'xpert':
            'Xpert replaces smear',
        'shortcourse_mdr':
            'Short-course MDR-TB regimen',
        'baseline':
            'Baseline scenario',
        'ipt_age5to15':
            'IPT in ages 5 to 15',
        'ipt_age0to5':
            'IPT in under 5s',
        'plot_start_time':
            'plotting start time',
        'early_time':
            'start of model run',
        'program_prop_decentralisation':
            'Decentralisation coverage',
        'epi_rr_diabetes':
            'Diabetes progression relative risk',
        'tb_n_contact':
            'Effective contact rate',
        'tb_prop_early_progression':
            'Early progression proportion',
        'tb_prop_amplification':
            'Amplification proportion',
        'tb_rate_late_progression':
            'Late progression rate',
        'tb_prop_casefatality_untreated':
            'Untreated case fatality',
        'tb_prop_casefatality_untreated_smearpos':
            'Untreated smear-positive case fatality',
        'tb_prop_casefatality_untreated_smearneg':
            'Untreated smear-negative case fatality',
        'decentralisation':
            'Decentralisation',
        'ipt':
            'IPT'
    }

    if name in dictionary_of_names:
        return dictionary_of_names[name]
    elif 'scenario' in name:
        return capitalise_first_letter(replace_underscore_with_space(name))
    else:
        return name


def find_string_from_starting_letters(string_to_analyse, string_start_to_find):

    """
    Possibly overly complicated function to find a string referring to an age group or population sub-group from
    the entire string for the compartment.

    Args:
        string_to_analyse: The full string for the compartment
        string_start_to_find: The beginning of the string of interest (e.g. 'age')
    Returns:
        result_string: The string of interest to be extracted
        stem: The remaining start of the string to be analysed
    """

    # Find the position of the string
    string_position = string_to_analyse.find(string_start_to_find)

    # Find the position of all the underscores in the string
    underscores = [pos for pos, char in enumerate(string_to_analyse) if char == '_']

    # Find the underscore at the start of the string of interest's position in the list of underscores
    for i, position in enumerate(underscores):
        if position == string_position:
            string_underscore_index = i

    # If the string of interest is at the end of the string
    if string_position == underscores[-1]:
        result_string = string_to_analyse[string_position:]

    # Otherwise if more string follows the age string
    else:
        result_string = string_to_analyse[string_position: underscores[string_underscore_index + 1]]

    stem = string_to_analyse[:string_position]

    return result_string, stem


def interrogate_age_string(age_string):

    """
    Take a string referring to an age group and find it's upper and lower limits.

    Args:
        age_string: The string to be analysed
    Returns:
        limits: List of the lower and upper limits
        dict_limits: Dictionary of the lower and upper limits
    """

    # Check the age string sta
    assert age_string[:4] == '_age', 'Age string does not begin with "_age".'

    # Extract the part of the string that actually refers to the ages
    ages = age_string[4:]

    # Find the lower age limit
    lower_age_limit = ''
    for i, letter in enumerate(ages):
        if letter.isdigit():
            lower_age_limit += letter
        else:
            break
    remaining_string = ages[len(lower_age_limit):]
    lower_age_limit = float(lower_age_limit)

    # Find the upper age limit
    if remaining_string == 'up':
        upper_age_limit = float('inf')
    elif remaining_string[:2] == 'to':
        upper_age_limit = float(remaining_string[2:])
    else:
        raise NameError('Age string incorrectly specified')

    limits = [lower_age_limit, upper_age_limit]
    dict_limits = {age_string: limits}

    return limits, dict_limits


def find_age_limits_directly_from_string(param_or_compartment):

    """
    Simple function to quickly grab the age limits from a string containing a standardised age string by combining the
    two previous functions.

    Args:
        param_or_compartment: String for parameter or compartment that contains a standardised age string

    Returns:
        age_limits: A list containing the upper and lower bounds of the age limit from the string being interrogated
    """

    age_string, _ = find_string_from_starting_letters(param_or_compartment, '_age')
    age_limits, _ = interrogate_age_string(age_string)

    return age_limits


def find_age_breakpoints_from_dicts(age_dict):

    breakpoints_with_repetition = []
    breakpoints = []

    # Add all age breakpoints to a temporary list that allows repetition
    for key in age_dict:
        for i in age_dict[key]:
            breakpoints_with_repetition += [i]

    # Check there is a lowest and highest age group
    assert 0. in breakpoints_with_repetition, 'No age group goes to zero'
    assert float('inf') in breakpoints_with_repetition, 'No age group goes to infinity'

    # Add the actual breakpoints once each
    for breakpoint in breakpoints_with_repetition:
        if breakpoint != 0. and breakpoint < 1E10 and breakpoint not in breakpoints:
            breakpoints += [breakpoint]

    return breakpoints


def estimate_prop_of_population_in_agegroup(age_limits, life_expectancy):

    """
    Function to estimate the proportion of the population that should be in a specific age
    group, assuming model equilibrium and absence of TB effects (which are false, of course).
    Args:
        age_limits: Two element string of the upper and lower limit of the age group.
        life_expectancy: Float specifying the life expectancy.

    Returns:
        estimated_prop_in_agegroup: Estimate of the proportion of the population in the age group.
    """

    estimated_prop_in_agegroup = exp(-age_limits[0] * (1. / life_expectancy)) \
                                 - exp(-age_limits[1] * (1. / life_expectancy))

    return estimated_prop_in_agegroup


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

    # Just to avoid warnings, replace any zeros in the denominators with small values
    # (numerators will still be zero, so all fractions should be zero)
    for i in range(len(denominator)):
        if denominator[i] == 0.:
            denominator[i] = 1E-3

    fraction = {}
    for label in numerator_labels:
        fraction[label] = [v / t for v, t in zip(numerators[label], denominator)]
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
            = get_fraction_soln(dictionary_of_classifications[category],
                                subgroup_solns[category],
                                compartment_type_denominator)
        for strata in ['strain', 'organ']:
            subgroup_solns[category + strata], compartment_type_bystrain_denominator, compartment_types_bystrain \
                = sum_over_compartments_bycategory(model, dictionary_of_classifications[category], strata)
            subgroup_fractions[category + strata] \
                = get_fraction_soln(compartment_types_bystrain,
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


def get_agegroups_from_breakpoints(breakpoints):

    """
    This function consolidates get_strat_from_breakpoints from Romain's age_strat
    module and define_age_structure from James' model.py method into one function
    that can return either a dictionary or a list for the model stratification. (One
    reason for using this approach rather than Romain's is that the lists need to be
    ordered for model.py.)

    Args:
        breakpoints: The age group cut-offs.

    Returns:
        agegroups: List of the strings describing the age groups only.
        agegroups_dict: List with strings of agegroups as keys with values being
            lists of the lower and upper age cut-off for that age group.

    """

    # Initialise
    agegroups = []
    agegroups_dict = {}

    if len(breakpoints) > 0:
        for i in range(len(breakpoints)):

            # The first age-group
            if i == 0:
                agegroup_string = '_age0to' + str(int(breakpoints[i]))
                agegroups_dict[agegroup_string] = [0.,
                                                   float(breakpoints[i])]

            # Middle age-groups
            else:
                agegroup_string = '_age' + str(int(breakpoints[i - 1])) + 'to' + str(int(breakpoints[i]))
                agegroups_dict[agegroup_string] = [float(breakpoints[i - 1]),
                                                   float(breakpoints[i])]
            agegroups += [agegroup_string]

        # Last age-group
        agegroup_string = '_age' + str(int(breakpoints[-1])) + 'up'
        agegroups_dict[agegroup_string] = [float(breakpoints[-1]),
                                           float('inf')]
        agegroups += [agegroup_string]

    # If no age groups
    else:
        # List consisting of one empty string required for methods that iterate over strains
        agegroups += ['']

    return agegroups, agegroups_dict


def turn_strat_into_label(stratum):

    if 'up' in stratum:
        label = stratum[4: -2] + ' and up'
    elif 'to' in stratum:
        to_index = stratum.find('to')
        label = stratum[4: to_index] + ' to ' + stratum[to_index+2:]
    elif stratum == '':
        label = 'All ages'
    else:
        label = ''

    return label


def adapt_params_to_stratification(data_breakpoints,
                                   model_breakpoints,
                                   data_param_vals,
                                   assumed_max_params=100.,
                                   parameter_name='',
                                   whether_to_plot=False):

    """
    Create a new set of parameters associated to the model stratification given parameter values that are known for
    another stratification.

    Args:
        data_breakpoints: tuple defining the breakpoints used in data.
        model_breakpoints: tuple defining the breakpoints used in the model.
        data_param_vals: dictionary containing the parameter values associated with each category defined by data_breakpoints
                         format example: {'_age0to5': 0.0, '_age5to15': 0.5, '_age15up': 1.0}
        assumed_max_params: the assumed maximal value for the parameter (exemple, age: 100 yo).

    Returns:
        dictionary containing the parameter values associated with each category defined by model_breakpoints
    """

    data_strat_list, data_strat = get_agegroups_from_breakpoints(data_breakpoints)
    model_strat_list, model_strat = get_agegroups_from_breakpoints(model_breakpoints)

    assert data_param_vals.viewkeys() == data_strat.viewkeys()

    model_param_vals = {}
    for new_name, new_range in model_strat.iteritems():
        new_low, new_up = new_range[0], new_range[1]
        considered_old_cats = []
        for old_name, old_range in data_strat.iteritems():
            if (old_range[0] <= new_low <= old_range[1]) or (old_range[0] <= new_up <= old_range[1]):
                considered_old_cats.append(old_name)
        beta = 0.  # store the new value for the parameter
        for old_name in considered_old_cats:
            alpha = data_param_vals[old_name]
            # calculate the weight to be affected to alpha (w = w_right - w_left)
            w_left = max(new_low, data_strat[old_name][0])
            if (data_strat[old_name][1] == float('inf')) and (new_up == float('inf')):
                w_right = assumed_max_params
                new_up = assumed_max_params
            else:
                w_right = min(new_up, data_strat[old_name][1])

            beta += alpha * (w_right - w_left)
        beta = beta / (new_up - new_low)
        model_param_vals[new_name] = beta

    report_age_specific_parameter_calculations(parameter_name, model_param_vals)

    # Convert data into list with same order as the ordered strat_lists
    data_value_list = []
    for i in data_strat_list:
        data_value_list += [data_param_vals[i]]
    model_value_list = []
    for i in model_strat_list:
        model_value_list += [model_param_vals[i]]

    if whether_to_plot:
        outputs.plot_comparative_age_parameters(data_strat_list,
                                                data_value_list,
                                                model_value_list,
                                                model_strat_list,
                                                parameter_name)

    return(model_param_vals)


def report_age_specific_parameter_calculations(parameter_name, model_param_vals):

    # Function to report the age-specific parameter calculations

    print('For parameter "' + replace_underscore_with_space(parameter_name[:-4]) + '":')
    for age_param in model_param_vals:
        limits, _ = interrogate_age_string(age_param)
        if limits[1] != float('inf'):
            lower_limit = ' from ' + str(int(limits[0]))
            upper_limit = ' to ' + str(int(limits[1]))
        else:
            lower_limit = ' aged ' + str(int(limits[0]))
            upper_limit = ' and up'
        print('\tthe parameter value for the age group' + lower_limit + upper_limit
              + ' has been estimated as ' + str(model_param_vals[age_param]))


def indices(a, func):

    """
    Returns the indices of a which verify a condition defined by a lambda function
        example: year = indices(self.model.times, lambda x: x >= 2003)[0]  returns the smallest index where x >=2003

    """

    return [i for (i, val) in enumerate(a) if func(val)]


def find_first_list_element_above_value(list, value):

    """
    Simple method to return the index of the first element of a list that is greater than a specified value.
    Args:
        list: The list of floats.
        value: The value that the element must be greater than.

    """

    index = next(x[0] for x in enumerate(list) if x[1] > value)

    return index


def find_first_list_element_at_least_value(list, value):

    """
    Simple method to return the index of the first element of a list that is greater than a specified value.
    Args:
        list: The list of floats.
        value: The value that the element must be greater than.

    """

    index = next(x[0] for x in enumerate(list) if x[1] >= value)

    return index


def pickle_save(object, file):

    with open(file, 'wb') as output:
        pickle.dump(object, output)


def pickle_load(file):

    with open(file, 'rb') as input:
        loaded_object = pickle.load(input)
    return loaded_object


def prepare_denominator(list_to_prepare):

    """
    Method to safely divide a list of numbers while ignoring zero denominators.

    Args:
        list_to_prepare: The list to be used as a denominator.

    Returns:
        The list with zeros replaced with small numbers.
    """

    return [list_to_prepare[i] if list_to_prepare[i] > 0. else 1e-10 for i in range(len(list_to_prepare))]


def is_upper_age_limit_at_or_below(compartment_string, age_value):

    """
    Return boolean for whether the upper limit of the age string is below a certain value. Expected to be used for
    determining whether an age-group is entirely paediatric.

    Args:
        compartment_string: The compartment string to analyse.
        age_value: The age to compare against.

    Returns:
        Boolean for whether the upper limit of age-group is below age_value
    """

    return interrogate_age_string(find_string_from_starting_letters(compartment_string, '_age')[0])[0][1] <= age_value


# * * * * * * * * * * * * * * * * * * * * * *
#                   Test age-stratification

if __name__ == "__main__":

    data_breaks = [5., 15.]
    model_breaks = [2., 7., 20.]

    data_param_vals = {'_age0to5': 0.5,
                       '_age5to15': 0.1,
                       '_age15up': 1.0}

    model_param = adapt_params_to_stratification(data_breaks,
                                                 model_breaks,
                                                 data_param_vals,
                                                 parameter_name='test parameter')
    print(model_param)




