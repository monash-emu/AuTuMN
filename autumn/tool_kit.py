
from six.moves import cPickle as pickle
import random
from scipy import exp
#import cPickle as pickle
import numpy
import json


'''
Note that this module is intended only to contain stand-alone functions for use by multiple other modules.
Object-oriented structures are not intended to be kept here.
'''


''' general data manipulation functions '''


def extract_dict_to_ordered_key_lists(dictionary, data_name, key_name='times'):
    """
    Create a dictionary with each element lists, one giving the "times" that the list element refers to and the others
    giving the data content that these times refer to - maintaining the order that the keys were originally in.

    Args:
        dictionary: The dictionary containing the data to be extracted (N.B. Assumption is that the keys refer to times)
        data_name: The key of interest within the input dictionary
        key_name: String to assign to the dictionary element that is the sorted list of keys of the input dictionary
    Returns:
        Dictionary containing the extracted lists with keys 'times' and "key_string"
    """

    return {key_name: sorted(dictionary.keys()), data_name: [dictionary[time] for time in sorted(dictionary.keys())]}


def increment_dictionary_with_dictionary(dict_a, dict_b):
    """
    Function to add the values from one dictionary to the values from the same keys in another.

    Args:
        dict_a: First dictionary to be added
        dict_b: Second dictionary to be added
    Return:
        The combined dictionary
    """
    # converting dict_a.items() and dict_b.items() to list for python 3
    return dict(list(dict_a.items()) + list(dict_b.items()) + [(k, dict_a[k] + dict_b[k]) for k in set(dict_b) & set(dict_a)])


def indices(a, func):
    """
    Returns the indices of a which verify a condition defined by a lambda function.

    Args:
        a: The list to be interrogated
        func: The function to be applied to the list
    Returns:
        List of the indices of the list that satisfy the function
    Example:
        year = indices(self.model.times, lambda x: x >= 2003)[0]  returns the smallest index where x >=2003
    """

    return [i for (i, val) in enumerate(a) if func(val)]


def find_first_list_element_above(list, value):
    """
    Simple method to return the index of the first element of a list that is greater than a specified value.

    Args:
        list: List of floats
        value: The value that the element must be greater than
    """
    if isinstance(list, numpy.ndarray):
        if len(list.shape) > 1:   # multi-dimensional array. We jsut want to keep the last row.
            list = list[-1, ]

    return next(x[0] for x in enumerate(list) if x[1] > value)


def find_list_element_equal_to(list, value):
    """
    Find the list element exactly equal to a specific value.

    Args:
        list: The list to search through
        value: Value being searched for
    """
    if isinstance(list, numpy.ndarray):
        if len(list.shape) > 1:   # multi-dimensional array. We jsut want to keep the last row.
            list = list[-1, ]

    if value in list:
        return next(x[0] for x in enumerate(list) if x[1] == value)
    else:
        return None

def find_first_list_element_at_least(list_to_search, value):
    """
    Simple method to return the index of the first element of a list that is greater than a specified value.

    Args:
        list_to_search: The list of floats
        value: The value that the element must be greater than
    """
    if isinstance(list_to_search, numpy.ndarray):
        if len(list_to_search.shape) > 1:   # multi-dimensional array. We jsut want to keep the last row.
            list_to_search = list_to_search[-1, ]

    if max(list_to_search) >= value:
        return next(x[0] for x in enumerate(list_to_search) if x[1] >= value)
    else:
        return None


def prepare_denominator(list_to_prepare):
    """
    Method to safely divide a list of numbers while ignoring zero denominators.

    Args:
        list_to_prepare: The list to be used as a denominator
    Returns:
        The list with zeros replaced with small numbers
    """

    return [list_to_prepare[i] if list_to_prepare[i] > 0. else 1e-10 for i in range(len(list_to_prepare))]


def find_common_elements(list_1, list_2):
    """
    Simple method to find the intersection of two lists.

    Args:
        list_1 and list_2: The two lists
    Returns:
        intersection: The common elements of the two lists
    """

    return [i for i in list_1 if i in list_2]


def find_common_elements_multiple_lists(list_of_lists):
    """
    Simple method to find the common elements of any number of lists

    Args:
        list_of_lists: A list whose elements are all the lists we want to find the
            intersection of.

    Returns:
        intersection: Common elements of all lists.
    """

    intersection = list_of_lists[0]
    for i in range(1, len(list_of_lists)):
        intersection = find_common_elements(intersection, list_of_lists[i])
    return intersection


def combine_two_lists_no_duplicate(list_1, list_2):
    """
    Method to combine two lists, drop one copy of the elements present in both and return a list comprised of the
    elements present in either list - but with only one copy of each.

    Args:
        list_1: First list
        list_2: Second list
    Returns:
        The combined list, as described above
    """

    additional_unique_elements_list_2 = [i for i in list_2 if i not in list_1]
    return list_1 + additional_unique_elements_list_2


def calculate_proportion_dict(data, relevant_keys, percent=False, floor=0., underscore=True):
    """
    General method to calculate proportions from absolute values provided as dictionaries.

    Args:
        data: Dictionary containing the absolute values
        relevant_keys: The keys of data from which proportions are to be calculated (generally a list of strings)
        floor: Minimum allowable value
        percent: Boolean describing whether the method should return the output as a percent or proportion
        underscore: Just whether the string stem has an underscore or not
    Returns:
        proportions: A dictionary of the resulting proportions
    """

    string_stem = 'prop_' if underscore else 'prop'

    # calculate multiplier for percentages if requested, otherwise leave as one
    multiplier = 1e2 if percent else 1.

    # create a list of the years that are common to all keys within data
    lists_of_years = []
    for i in range(len(relevant_keys)):
        lists_of_years.append(data[relevant_keys[i]].keys())
    common_years = find_common_elements_multiple_lists(lists_of_years)

    # calculate the denominator by summing the values for which proportions have been requested
    denominator = {}
    for i in common_years:
        denominator[i] = 0.
        for j in relevant_keys:
            denominator[i] += data[j][i]

    # calculate the proportions
    proportions = {}
    for j in relevant_keys:
        proportions[string_stem + j] = {}
        for i in common_years:
            if denominator[i] > floor:
                proportions[string_stem + j][i] = data[j][i] / denominator[i] * multiplier
    return proportions


def calculate_proportion_list(numerator, denominator):
    """
    Simple method to return each list element from numerator divided by each list element from denominator.

    Args:
        numerator: List of numerator values
        denominator: List of denominator values
    """

    assert len(numerator) == len(denominator), 'Attempted to divide list elements from lists of differing lengths'
    return [i / j for i, j in zip(numerator, prepare_denominator(denominator))]


def remove_specific_key(dictionary, key):
    """
    Remove a specific named key from a dictionary.

    Args:
        dictionary: The dictionary to have a key removed
        key: The key to be removed
    Returns:
        dictionary: The dictionary with the key removed
    """

    if key in dictionary:
        del dictionary[key]
    return dictionary


def remove_nans(dictionary):
    """
    Takes a dictionary and removes all of the elements for which the value is nan.

    Args:
        dictionary: Should typically be the dictionary of programmatic values, usually
                    with time in years as the key.
    Returns:
        dictionary: The dictionary with the nans removed.
    """

    nan_indices = []
    for i in dictionary:
        if type(dictionary[i]) == float and numpy.isnan(dictionary[i]):
            nan_indices += [i]
    for i in nan_indices:
        del dictionary[i]
    return dictionary


def label_intersects_tags(label, tags):
    """
    Primarily for use in force of infection calculation to determine whether a compartment is infectious.

    Args:
        label: Generally a compartment label.
        tags: Tag for whether label is to be counted.
    Returns:
        Boolean for whether any of the tags are in the label.
    """

    for tag in tags:
        if tag in label: return True
    return False


def apply_odds_ratio_to_proportion(proportion, odds_ratio):
    """
    Use an odds ratio to adjust a proportion.

    Starts from the premise that the odds associated with the original proportion (p1) = p1 / (1 - p1)
    and similarly, that the odds associated with the adjusted proportion (p2) = p2 / (1 - p2)
    We want to multiply the odds associated with p1 by a certain odds ratio.
    That, is we need to solve the following equation for p2:
        p1 / (1 - p1) * OR = p2 / (1 - p2)
    By simple algebra, the solution to this is:
        p2 = p1 * OR / (p1 * (OR - 1) + 1)

    Args:
        proportion: The original proportion (p1 in the description above)
        odds_ratio: The odds ratio to adjust by
    Returns:
        The adjusted proportion
    """

    return proportion * odds_ratio / (proportion * (odds_ratio - 1.) + 1.)


def increase_parameter_closer_to_value(old_value, target_value, coverage):
    """
    Simple but commonly used calculation for interventions. Acts to increment from the original or baseline value
    closer to the target or intervention value according to the coverage of the intervention being implemented.

    Args:
        old_value: Baseline or original value to be incremented
        target_value: Target value or value at full intervention coverage
        coverage: Intervention coverage or proportion of the intervention value to apply
    """

    return old_value + (target_value - old_value) * coverage if old_value < target_value else old_value


def decrease_parameter_closer_to_value(old_value, target_value, coverage):
    """
    Simple but commonly used calculation for interventions. Acts to decrement from the original or baseline value
    closer to the target or intervention value according to the coverage of the intervention being implemented.

    Args:
        old_value: Baseline or original value to be decremented
        target_value: Target value or value at full intervention coverage
        coverage: Intervention coverage or proportion of the intervention value to apply
    """

    return old_value - (old_value - target_value) * coverage if old_value > target_value else old_value


def force_list_to_length(list, length):
    """
    Force a list to be a certain list in order that it can be stacked with other lists to make an array when working out
    epidemiological outputs.
    No longer used because was for use in model_runner to make all the new data outputs the same length as the first.
    However, now the approach is to make all the data outputs the same length as the longest one, by sticking zeros on
    at the start where necessary.

    Args:
        list: The list that needs its length adapted
        length: The desired length as an integer
    Returns:
        The list in its adapted form, either unchanged, truncated to desired length or with zeroes added to the start
    """

    if len(list) == length:
        return list
    elif len(list) > length:
        return list[-length:]
    elif len(list) < length:
        return [0.] * (length - len(list)) + list


def is_all_same_value(a_list, test_val):
    """
    Simple method to find whether all values in list are equal to a particular value.

    Args:
        a_list: The list being interrogated
        test_val: The value to compare the elements of the list against
    """

    for val in a_list:
        if val != test_val: return False
    return True


def replace_specified_value(a_list, new_val, old_value):
    """
    Replace all elements of a list that are a certain value with a new value specified in the inputs.

    Args:
         a_list: The list being modified
         new_val: The value to insert into the list
         old_value: The value of the list to be replaced
    """

    return [new_val if val == old_value else val for val in a_list]


def elementwise_list_addition(increment, list_to_increment):
    """
    Simple method to element-wise increment a list by the values in another list of the same length.
    """

    if not list_to_increment:
        return increment
    assert len(increment) == len(list_to_increment), 'Attempted to add two lists of different lengths'
    return [sum(x) for x in zip(list_to_increment, increment)]


def elementwise_list_division(numerator, denominator, percentage=False):
    """
    Simple method to element-wise divide a list by the values in another list of the same length.
    """

    assert len(numerator) == len(denominator), 'Attempted to divide two lists of different lengths'
    percentage_multiplier = 100. if percentage else 1.
    return [n / d * percentage_multiplier for n, d in zip(numerator, denominator)]


def join_zero_array_to_left(number_of_zeros, array_to_extend):
    """
    Quick method to concatenate a list of zeros on to the left of an existing array.

    Args:
        number_of_zeros: Number of zeros to add on
        array_to_extend: The original array
    Returns:
        The new numpy array with the zeros joined on
    """

    zeros = numpy.zeros(number_of_zeros) if type(array_to_extend) == list or array_to_extend.ndim == 1 \
        else numpy.zeros((array_to_extend.shape[0], number_of_zeros))
    return numpy.hstack((zeros, array_to_extend))


def are_all_values_the_same(dictionary_to_evaluate, keys_of_interest=(0, 1)):
    """
    Work out whether all values in a dictionary have the same value over a range of integer-valued keyed. If the key is
    absent from the dictionary or the key isn't an integer, it doesn't matter and the loop continues.

    Args:
        dictionary_to_evaluate: The dictionary to loop through
        keys_of_interest: List of the starting and finishing element for integers to loop through
    Return:
        Boolean value for the condition described above
    """

    value = None
    for i in range(keys_of_interest[0], keys_of_interest[1]):
        if i not in dictionary_to_evaluate:
            continue
        elif not value:
            value = dictionary_to_evaluate[i]
        elif dictionary_to_evaluate[i] == value:
            continue
        else:
            return False
    return True


def are_strings_in_subdict(mapper, subdict, strings_of_interest, string_for_mapper):
    """
    Search an element of a sub-dictionary to determine whether a particular string is present in it. Mostly for use in
    calculating outputs in the model runner module.

    Args:
        mapper: Dictionary describing the purpose of each element of the subdict
        subdict: The sub-dictionary, which has one element being the string of interest to search
        strings_of_interest: The strings being searched for
        string_for_mapper: The string to be fed to the mapper to find which element of subdict we're after
    """

    if string_for_mapper not in mapper:
        return False
    else:
        for string in strings_of_interest:
            if string not in subdict[mapper[string_for_mapper]]: return False
        return True


def apply_weighting(object_to_be_weighted, weights):
    """
    Replicate the components of object_to_be_weighted using the corresponding weights to define the number of replicates.
    Args:
        object_to_be_weighted: could be a list or an array
        weights: a list of integers
    Returns: the transformed object
    """
    zipped = zip(object_to_be_weighted, weights)
    weighted_object = []
    for item in zipped:
        for j in range(item[1]):
            weighted_object.append(item[0])
    return weighted_object


''' scenario name manipulation '''


def find_scenario_string_from_number(scenario):
    """
    Find a string to represent a scenario from it's number (or None in the case of baseline).

    Args:
        scenario: The scenario value or None for baseline
    Returns:
        The string representing the scenario
    """

    if scenario == 0:
        return 'baseline'
    elif scenario == 16:
        return 'no transmission'
    else:
        return 'scenario_' + str(scenario)


def find_scenario_number_from_string(scenario):
    """
    Reverse of the above method. Not currently used, but likely to be needed.

    Args:
        scenario: Scenario string
    Returns:
        scenario_number: The scenario number or None for baseline
    """

    # strip of the manual if being used in model_runner
    if 'manual_' in scenario:
        scenario_string = scenario.replace('manual_', '')
    elif 'uncertainty_' in scenario:
        scenario_string = scenario.replace('uncertainty_', '')
    else:
        scenario_string = scenario

    # find number or zero for baseline scenario
    if scenario_string == 'baseline':
        scenario_number = 0
    else:
        scenario_number = int(scenario_string[9:])
    return scenario_number


''' string manipulation '''


def capitalise_first_letter(old_string):
    """
    Really simple method to capitalise the first character of a string.

    Args:
        old_string: The string to be capitalised
    Returns:
        new_string: The capitalised string
    """

    if len(old_string) > 0:
        return old_string[0].upper() + old_string[1:]
    else:
        return old_string


def replace_underscore_with_space(original_string):
    """
    Another really simple method to remove underscores and replace with spaces for titles of plots.

    Args:
        original_string: String with underscores
    Returns:
        replaced_string: String with underscores replaced
    """

    return original_string.replace('_', ' ')


def capitalise_and_remove_underscore(original_string):
    """
    Combine the previous two methods used to create titles.

    Args:
        original_string: String to be modified
    Return:
        revised string
    """

    return capitalise_first_letter(replace_underscore_with_space(original_string))


def adjust_country_name(country, purpose):
    """
    Use a dictionary to adapt the basic country name to the name used in specific spreadsheets.

    Args:
        country: String for the original country name
        purpose: The "purpose", which is the string to index the second tier of the country_name_adaptations dictionary
    Returns:
        adjusted_country_name: Adjusted string
    """

    country_name_adaptations =\
        {'Kyrgyzstan': {'demographic': 'Kyrgyz Republic'},
         'Moldova': {'tb': 'Republic of Moldova'},
         'Philippines': {'tb': 'Philippines (the)'}}
    if country in country_name_adaptations:
        if purpose in country_name_adaptations[country]:
            return country_name_adaptations[country][purpose]
    return country


def find_title_from_dictionary(working_string, forward=True, capital_first_letter=True, country=None):
    """
    Master function to convert between strings used in the code and ones to present to the user - either through the GUI
    or for creating figures in the outputs module. Now goes in both directions.

    Args:
        working_string: AuTuMN's name for the boolean quantity
        forward: Whether to go from code string to GUI string or the other way
        capital_first_letter: Boolean for whether to capitalise the first letter of the string
        country: Which country is simulated
    Returns:
        The converted string
    """

    for starting_string_to_trim in ['int_prop_', 'program_prop_', 'econ_program_', 'is_', 'plot_option_']:
        if working_string.startswith(starting_string_to_trim):
            working_string = working_string[len(starting_string_to_trim):]

    if working_string[:4] == '_age' and 'up' in working_string:
        return working_string[4: 6] + ' and above'
    elif working_string[:7] == '_age0to':
        return working_string[7:] + ' and under'
    elif working_string[:4] == '_age':
        return working_string[4: working_string.index('to')] + ' to ' + working_string[working_string.index('to') + 2:]

    conversion_dictionary \
        = {'vaccination':
               'BCG vaccination',
           'treatment_success':
               'treatment success rate',
           'treatment_new_death':
               'death on treatment, new cases',
           'treatment_treated_death':
               'death on treatment, previously treated',
           'ipt_age0to5':
               'IPT coverage in under-5s',
           'ipt_age5to15':
               'IPT coverage in ages 5 to 15',
           'ipt':
               'IPT coverage, all ages',
           'treatment_new_success':
               'treatment success, new cases',
           'treatment_treated_success':
               'treatment success, previously treated',
           'detect':
               'case detection rate',
           'treatment_death':
               'death rate on treatment',
           'algorithm_sensitivity':
               'diagnostic algorithm sensitivity',
           'shortcourse_mdr':
               'short-course MDR-TB regimen',
           'unitcost_ipt':
               'IPT unit cost',
           'program_cost_vaccination':
               'vaccination program cost (?dummy)',
           'unitcost_vaccination':
               'unit cost of BCG',
           'totalcost_ipt':
               'IPT program cost',
           'totalcost_vaccination':
               'vaccination program cost',
           'demo_rate_birth':
               'birth rate',
           'demo_life_expectancy':
               'life expectancy',
           'rate_force':
               'force of infection',
           'econ_inflation':
               'inflation rate',
           'econ_cpi':
               'consumer price index',
           'program_timeperiod_await_treatment_smearpos':
               'smear-positive',
           'program_timeperiod_await_treatment_smearneg':
               'smear-negative',
           'program_timeperiod_await_treatment_extrapul':
               'extrapulmonary',
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
               'run starting time',
           'recent_time':
               ' over recent years',
           '_diabetes':
               'diabetes',
           'riskgroup_prop_diabetes':
               'diabetes prevalence',
           'riskgroup_prop_prison':
               'prison population',
           'riskgroup_prop_ruralpoor':
               'Roma population',
           '_hiv':
               'HIV',
           '_nocomorb':
               'general population',
           'age':
               'age group',
           'comorbidity':
                'risk group',
           'smearacf':
                'smear-based ACF',
           'treatment_support':
                'treatment support',
           'program_timeperiod':
                'programmatic time to treatment',
           'program_prop_engage_lowquality':
                'engage low-quality sector',
           'engage_lowquality':
                'engage low-quality sector',
           'xpertacf':
                'Xpert-based ACF',
           'xpertacf_prison':
                'Xpert ACF, prisons',
           'xpertacf_indigenous':
                'Xpert ACF, indigenous',
           'xpertacf_urbanpoor':
                'Xpert ACF, urban poor',
           'xpertacf_ruralpoor':
                'Xpert ACF, rural poor',
           'epi_':
                'epidemiological',
           'epi_prop_smear':
                'organ status',
           'epi_prop_smearneg':
                'Proportion smear-negative',
           'epi_prop_smearpos':
                'Proportion smear-positive',
            'xpert':
                'Xpert replaces smear',
            'baseline':
                'baseline scenario',
            'plot_start_time':
                'plotting start time',
            'early_time':
                'start of model run',
            'program_prop_decentralisation':
                'decentralisation coverage',
            'tb_n_contact':
                'effective contact rate',
            'tb_prop_early_progression':
                'early progression proportion',
            'tb_prop_amplification':
                'amplification proportion',
            'tb_rate_late_progression':
                'late progression rate',
            'tb_prop_casefatality_untreated':
                'untreated case fatality',
            'tb_prop_casefatality_untreated_smearpos':
                'untreated smear-positive case fatality',
            'tb_prop_casefatality_untreated_smearneg':
                'untreated smear-negative case fatality',
            'decentralisation':
                'decentralisation',
            '_prison':
                'prisoners',
            '_ruralpoor':
                'rural poor',
            '_dorm':
                'dorm',
            '_urbanpoor':
                'urban poor',
            '_norisk':
                'general population',
            '_ds':
                'DS-TB',
            '_mdr':
                'MDR-TB',
            'bulgaria_improve_dst':
                'improve DST (Bulgaria implementation)',
            'food_voucher_ds':
                'food vouchers, DS-TB',
            'food_voucher_mdr':
                'food vouchers, MDR-TB',
            'program_prop_firstline_dst':
                'DST availability',
            'int_prop_firstline_dst':
                'DST availability',
            'program_prop_treatment_success_mdr':
                'treatment success MDR-TB',
            'program_prop_treatment_success_ds':
                'treatment success DS-TB',
            'program_prop_treatment_death_mdr':
                'death on treatment MDR-TB',
            'program_prop_treatment_death_ds':
                'death on treatment DS-TB',
            'program_prop_treatment':
                'treatment outcome',
            'program_timeperiod_':
                'waiting period',
            'misc':
                'miscellaneous',
            'tb_multiplier_treated_protection':
                'relative susceptibility after treatment',
            'tb_timeperiod_activeuntreated':
                'duration active untreated',
            'beta_2_2':
                'beta, params: 2, 2',
            'program_prop_treatment_support_relative':
                'Treatment support',
            'program_prop_vaccination':
                'BCG vaccination',
            'program_prop_treatment_ds_new_death':
                'death on treatment DS-TB, new cases',
            'program_prop_treatment_ds_new_success':
                'treatment success DS-TB, new cases',
            'program_prop_treatment_ds_treated_death':
                'death on treatment DS-TB, previously treated',
            'program_prop_treatment_ds_treated_success':
                'treatment success DS-TB, previously treated',
            'program_prop_treatment_mdr_new_success':
                'treatment success MDR-TB',
            'program_prop_treatment_mdr_treated_death':
                'death on treatment MDR-TB',
            'program_prop_treatment_inappropriate_new_death':
                'death on treatment MDR-TB on inappropriate regimen',
            'program_prop_treatment_inappropriate_new_success':
                'treatment success MDR-TB on inappropriate regimen',
            'program_prop_nonsuccess_new_death':
                'death on treatment, new cases',
            'program_prop_nonsuccess_treated_death':
                'death on treatment, previously treated',
            'program_prop_nonsuccess_ds_new_death':
                'death on treatment DS-TB, new cases',
            'program_prop_nonsuccess_ds_treated_death':
                'death on treatment DS-TB, previously treated',
            'program_prop_nonsuccess_mdr_new_death':
                'death on treatment MDR-TB',
            'program_prop_nonsuccess_inappropriate_death':
                'death on treatment MDR-TB on inappropriate regimen',
            'write_uncertainty_outcome_params':
                'record parameters',
           'output_spreadsheets':
                'write to spreadsheets',
           'output_documents':
                'write to documents',
           'output_by_scenario':
                'output by scenario',
           'output_horizontally':
                'write horizontally',
           'output_epi_plots':
                'plot outcomes',
           'output_compartment_populations':
                'plot compartment sizes',
           'output_by_subgroups':
                'plot outcomes by sub-groups',
           'output_age_fractions':
                'plot proportions by age',
           'output_flow_diagram':
                'draw flow diagram',
           'output_scaleups':
                'plot scale-up functions',
           'output_plot_economics':
                'plot economics graphs',
           'output_param_plots':
                'plot parameter progression',
           'output_likelihood_plot':
                'plot log likelihoods over runs',
           'riskgroup_diabetes':
                'type II diabetes',
           'riskgroup_hiv':
                'HIV',
           'riskgroup_prison':
                'prison',
           'riskgroup_urbanpoor':
                'urban poor',
           'riskgroup_ruralpoor':
                'rural poor',
           'riskgroup_dorm':
                'dorm',
           'riskgroup_indigenous':
                'indigenous',
           'lowquality':
                'low quality care',
           'amplification':
                'resistance amplification',
           'timevariant_organs':
                'time-variant organ status',
           'misassignment':
                'strain mis-assignment',
           'vary_detection_by_organ':
                'vary case detection by organ status',
           'organ_strata':
                'number of organ strata',
           'strains':
                'number of strains',
           'vary_force_infection_by_riskgroup':
                'heterogeneous mixing',
           'treatment_history':
                'treatment history',
           'vary_detection_by_riskgroup':
                'vary detection by risk group',
           'include_relapse_in_ds_outcomes':
                'include relapse treatment outcomes (DS-TB)',
           'include_hiv_treatment_outcomes':
                'include HIV treatment outcomes',
           'adjust_population':
                'adjust population to target',
           'shortcourse_improves_outcomes':
                'short course MDR improves outcomes',
           'vars_two_panels':
               'Plot scale-up functions on two panels',
           'nonsuccess_new_death':
               'death among non-success outcomes, new cases',
           'nonsuccess_treated_death':
               'death among non-success outcomes, previously treated',
           'incidence_mdr':
               'MDR-TB incidence',
           'mortality_mdr':
               'MDR-TB mortality',
           'prevalence_mdr':
               'MDR-TB prevalence',
           'perc_incidence_mdr':
               'MDR-TB percentage incidence',
           'mdr-tb-related':
               'MDR-TB-related',
           'main':
               '',
           'by_riskgroups':
               ' by risk group',
           'by_agegroups':
               ' by age group'}

    list_of_code_strings = []
    list_of_interface_strings = []
    for key, value in conversion_dictionary.items():
        list_of_code_strings.append(key)
        list_of_interface_strings.append(value)

    if forward and working_string in list_of_code_strings:
        converted_string = list_of_interface_strings[list_of_code_strings.index(working_string)]
    elif working_string in list_of_interface_strings:
        converted_string = list_of_code_strings[list_of_interface_strings.index(working_string)]
    else:
        converted_string = replace_underscore_with_space(working_string)

    if country == 'bulgaria':
        converted_string = converted_string.replace('ruralpoor', 'Roma')
    converted_string = converted_string.replace('norisk', 'no risk')

    return capitalise_first_letter(converted_string) if capital_first_letter else converted_string


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

    # find the position of the string
    string_position = string_to_analyse.find(string_start_to_find)

    # find the position of all the underscores in the string
    underscores = [pos for pos, char in enumerate(string_to_analyse) if char == '_']

    # find the underscore at the start of the string of interest's position in the list of underscores
    for i, position in enumerate(underscores):
        if position == string_position:
            string_underscore_index = i

    # if the string of interest is at the end of the string
    if string_position == underscores[-1]:
        result_string = string_to_analyse[string_position:]

    # otherwise if more string follows the age string
    else:
        result_string = string_to_analyse[string_position: underscores[string_underscore_index + 1]]

    stem = string_to_analyse[:string_position]

    return result_string, stem


'''  age string manipulation '''


def interrogate_age_string(age_string):
    """
    Take a string referring to an age group and find it's upper and lower limits.

    Args:
        age_string: The string to be analysed
    Returns:
        limits: List of the lower and upper limits
        dict_limits: Dictionary of the lower and upper limits
    """

    # check the age string starts with the string to indicate age
    assert age_string[:4] == '_age', 'Age string does not begin with "_age".'

    # extract the part of the string that actually refers to the ages
    ages = age_string[4:]

    # find the lower age limit
    lower_age_limit = ''
    for i, letter in enumerate(ages):
        if letter.isdigit():
            lower_age_limit += letter
        else:
            break
    remaining_string = ages[len(lower_age_limit):]
    lower_age_limit = float(lower_age_limit)

    # find the upper age limit
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
    """
    Convert a dictionary of age groups back into the list of breakpoints.
    """

    breakpoints_with_repetition = []
    breakpoints = []

    # add all age breakpoints to a temporary list that allows repetition
    for key in age_dict:
        for i in age_dict[key]:
            breakpoints_with_repetition += [i]

    # check there is a lowest and highest age group
    assert 0. in breakpoints_with_repetition, 'No age group goes to zero'
    assert float('inf') in breakpoints_with_repetition, 'No age group goes to infinity'

    # add the actual breakpoints once each
    for breakpoint in breakpoints_with_repetition:
        if breakpoint != 0. and breakpoint < 1e10 and breakpoint not in breakpoints:
            breakpoints += [breakpoint]

    return breakpoints


def estimate_prop_of_population_in_agegroup(age_limits, life_expectancy):
    """
    Function to estimate the proportion of the population that should be in a specific age group, assuming model
    equilibrium and absence of TB effects (which are false assumptions, of course).

    Args:
        age_limits: Two element string of the upper and lower limit of the age group.
        life_expectancy: Float specifying the life expectancy.
    Returns:
        Estimate of the proportion of the population in the age group.
    """

    return exp(-age_limits[0] * (1. / life_expectancy)) - exp(-age_limits[1] * (1. / life_expectancy))


def get_agegroups_from_breakpoints(breakpoints):
    """
    This function consolidates get_strat_from_breakpoints from Romain's age_strat module and define_age_structure from
    James' model.py method into one function that can return either a dictionary or a list for the model stratification.
    (One reason for using this approach rather than Romain's is that the lists need to be ordered for model.py.)

    Args:
        breakpoints: The age group cut-offs.
    Returns:
        agegroups: List of the strings describing the age groups only.
        agegroups_dict: List with strings of agegroups as keys with values being
            lists of the lower and upper age cut-off for that age group.
    """

    # initialise
    agegroups = []
    agegroups_dict = {}

    if len(breakpoints) > 0:
        for i in range(len(breakpoints)):

            # the first age-group
            if i == 0:
                agegroup_string = '_age0to' + str(int(breakpoints[i]))
                agegroups_dict[agegroup_string] = [0., float(breakpoints[i])]

            # middle age-groups
            else:
                agegroup_string = '_age' + str(int(breakpoints[i - 1])) + 'to' + str(int(breakpoints[i]))
                agegroups_dict[agegroup_string] = [float(breakpoints[i - 1]),
                                                   float(breakpoints[i])]
            agegroups += [agegroup_string]

        # last age-group
        agegroup_string = '_age' + str(int(breakpoints[-1])) + 'up'
        agegroups_dict[agegroup_string] = [float(breakpoints[-1]),
                                           float('inf')]
        agegroups += [agegroup_string]

    # if no age groups
    else:
        # list consisting of one empty string required for methods that iterate over strains
        agegroups += ['']

    return agegroups, agegroups_dict


def turn_strat_into_label(stratum):
    """
    Convert age stratification string into a string that describes it more clearly.

    Args:
        stratum: String used in the model
    Returns:
        label: String that can be used in plotting
    """

    if 'up' in stratum:
        return stratum[4: -2] + ' and up'
    elif 'to' in stratum:
        to_index = stratum.find('to')
        return stratum[4: to_index] + ' to ' + stratum[to_index+2:]
    elif stratum == '':
        return 'All ages'
    else:
        return ''


def report_age_specific_parameter_calculations(parameter_name, model_param_vals):
    """
    Function to report the age-specific parameter calculations.
    """

    message = 'For parameter "' + replace_underscore_with_space(parameter_name[:-4]) + '":\n'
    for age_param in model_param_vals:
        limits, _ = interrogate_age_string(age_param)
        if limits[1] != float('inf'):
            lower_limit = ' from ' + str(int(limits[0]))
            upper_limit = ' to ' + str(int(limits[1]))
        else:
            lower_limit = ' aged ' + str(int(limits[0]))
            upper_limit = ' and up'
        message += '\tthe parameter value for the age group' + lower_limit + upper_limit \
              + ' has been estimated as ' + str(model_param_vals[age_param]) + '\n'
    return message


def is_upper_age_limit_at_or_below(compartment_string, age_value):
    """
    Return boolean for whether the upper limit of the age string is below a certain value. Expected to be used for
    determining whether an age-group is entirely paediatric.

    Args:
        compartment_string: The compartment string to analyse
        age_value: The age to compare against
    Returns:
        Boolean for whether the upper limit of age-group is below age_value
    """

    return interrogate_age_string(find_string_from_starting_letters(compartment_string, '_age')[0])[0][
               1] <= age_value


def adapt_params_to_stratification(data_breakpoints, model_breakpoints, data_param_vals, assumed_max_params=100.,
                                   parameter_name='', gui_console_fn=None):
    """
    Create a new set of parameters associated to the model stratification given parameter values that are known for
    another stratification.

    Args:
        data_breakpoints: Tuple defining the breakpoints used in data.
        model_breakpoints: Tuple defining the breakpoints used in the model.
        data_param_vals: Dictionary containing the parameter values associated with each category defined by
            data_breakpoints format example: {'_age0to5': 0.0, '_age5to15': 0.5, '_age15up': 1.0}
        assumed_max_params: The assumed maximal value for the parameter (example, age: 100 yo).
    Returns:
        Dictionary containing the parameter values associated with each category defined by model_breakpoints
    """

    data_strat_list, data_strat = get_agegroups_from_breakpoints(data_breakpoints)
    model_strat_list, model_strat = get_agegroups_from_breakpoints(model_breakpoints)

    assert data_param_vals.keys() == data_strat.keys()

    model_param_vals = {}
    for new_name, new_range in model_strat.items():
        new_low, new_up = new_range[0], new_range[1]
        considered_old_cats = []
        for old_name, old_range in data_strat.items():
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

    message = report_age_specific_parameter_calculations(parameter_name, model_param_vals)
    if gui_console_fn:
        gui_console_fn('console', {'message': message})
    else:
        print(message)

    # convert data into list with same order as the ordered strat_lists
    data_value_list = []
    for i in data_strat_list:
        data_value_list += [data_param_vals[i]]
    model_value_list = []
    for i in model_strat_list:
        model_value_list += [model_param_vals[i]]
    return model_param_vals


''' output interrogation functions '''


def sum_over_compartments(model, compartment_types):
    """
    General method to sum sets of compartments.

    Args:
        compartment_types: List of the compartments to be summed over
    Returns:
        summed_soln: Dictionary of lists for the sums of each compartment
        summed_denominator: List of the denominator values
    """

    summed_soln = {}
    summed_denominator = [0] * len(random.sample(model.compartment_soln.items(), 1)[0][1])
    for compartment_type in compartment_types:
        summed_soln[compartment_type] = [0] * len(random.sample(model.compartment_soln.items(), 1)[0][1])
        for label in model.labels:
            if compartment_type in label:
                summed_soln[compartment_type] \
                    = [i + j for i, j in zip(summed_soln[compartment_type], model.compartment_soln[label])]
                summed_denominator += model.compartment_soln[label]
    return summed_soln, summed_denominator


def get_fraction_soln(numerator_labels, numerators, denominator):
    """
    General method for calculating the proportion of a subgroup of the population in each compartment type.

    Args:
        numerator_labels: Labels of numerator compartments
        numerators: Lists of values of each numerator
        denominator: List of values for the denominator
    Returns:
        Fractions of the denominator in each numerator
    """

    # just to avoid warnings, replace any zeros in the denominators with small values
    # (numerators will still be zero, so all fractions should be zero)
    for i in range(len(denominator)):
        if denominator[i] == 0.:
            denominator[i] = 1e-3

    fraction = {}
    for label in numerator_labels:
        fraction[label] = [v / t for v, t in zip(numerators[label], denominator)]
    return fraction


def sum_over_compartments_bycategory(model, compartment_types, categories):

    summed_soln = {}
    summed_denominator \
        = [0] * len(random.sample(model.compartment_soln.items(), 1)[0][1])
    compartment_types_bycategory = []
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
                    summed_soln[compartment_type] \
                        = [i + j for i, j in zip(summed_soln[compartment_type], model.compartment_soln[label])]
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
                        summed_soln[compartment_type + working_category] \
                            = [i + j for i, j in zip(summed_soln[compartment_type + working_category],
                                                     model.compartment_soln[label])]
                        summed_denominator += model.compartment_soln[label]
    return summed_soln, summed_denominator, compartment_types_bycategory


def find_fractions(model):

    # all compartmental disease stages
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

    # the following was previously the additional diagnostics code in model.py
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


''' pickling functions '''


def pickle_save(object, file):
    """
    Save an object in pickle format.

    Args:
        object: The object to be saved
        file: The filename to save the data to
    """

    with open(file, 'wb') as output:
        pickle.dump(object, output)


def pickle_load(file):
    """
    Load an object previously saved in pickle format.

    Args:
        file: Filename storing the object to be loaded
    """

    with open(file, 'rb') as input:
        loaded_object = pickle.load(input)
    return loaded_object


''' Json functions '''


def json_save(object, file):
    """
    Save an object in json format.

    Args:
        object: The object to be saved
        file: The filename to save the data to
    """

    with open(file, 'wb') as output:
        json.dump(object, output)


def json_load(file):
    """
    Load an object previously saved in json format.

    Args:
        file: Filename storing the object to be loaded
    """

    with open(file, 'rb') as input:
        loaded_object = json.loads(input)
    return loaded_object

