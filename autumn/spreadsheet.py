# -*- coding: utf-8 -*-

from __future__ import print_function
from xlrd import open_workbook # For opening Excel workbooks
from numpy import nan
import numpy
import os
import datetime
import copy


spreadsheet_start_realtime = datetime.datetime.now()


"""
Import model inputs from Excel spreadsheet 
"""

###############################################################
#  General functions for use by readers below

def is_all_same_value(a_list, test_val):
    for val in a_list:
        if val != test_val:
            return False
    return True


def replace_blanks(a_list, new_val, blank):
    return [new_val if val == blank else val for val in a_list]


def parse_year_data(these_data, blank, endcolumn):
    these_data = replace_blanks(these_data, nan, blank)
    assumption_val = these_data[-1]
    year_vals = these_data[: endcolumn]
    if is_all_same_value(year_vals, nan):
        return [assumption_val] 
    else:
        # skip "OR" and assumption col
        return year_vals


def adjust_country_name(country_name):

    adjusted_country_name = country_name
    if country_name == u'Philippines':
        adjusted_country_name = country_name + u' (the)'
    return adjusted_country_name


def calculate_proportion(data, indices, percent):

    """
    Calculate the proportions of patients within subgroups

    Args:
        data: The main data structure containing all the data for that country
        indices: A list of the dictionary elements of data that are to be summed
            and then the proportions calculated

    Returns:
        proportions: A dictionary containing the proportions by indices
    """

    multiplier = 1.
    if percent:
        multiplier = 100.

    # Calculate totals for denominator
    denominator = []
    for j in range(len(indices)):
        for i in range(len(data[indices[0]])):
            # Start list from first index
            if j == 0:
                denominator += [data[indices[j]][i]]
            # Add other indices
            else:
                denominator[i] += data[indices[j]][i]

    # Calculate proportions
    proportions = {}
    for j in range(len(indices)):
        proportions['prop_' + indices[j]] = []
        for i in range(len(data[indices[0]])):

            # Avoid division by zero errors and nans
            if type(denominator[i]) == float and denominator[i] > 0.:
                proportions['prop_' + indices[j]] \
                    += [data[indices[j]][i]
                        / denominator[i] * multiplier]
            else:
                proportions['prop_' + indices[j]] += [float('nan')]

    return proportions


def remove_nans(program):

    """
    Takes a dictionary and removes all of the elements for which the value is nan

    Args:
        program: Should typically be the dictionary of programmatic values, usually
                    with time in years as the key.

    Returns:
        program: The dictionary with the nans removed.

    """

    nan_indices = []
    for i in program:
        if type(program[i]) == float and numpy.isnan(program[i]):
            nan_indices += [i]
    for i in nan_indices:
        del program[i]

    return program


def remove_specific_key(dict, key):

    """
    Remove a specific named key from a dictionary
    Args:
        dict: The dictionary to have a key removed
        key: The key to be removed

    Returns:
        dict: The dictionary with the key removed

    """

    if key in dict:
        del dict[key]

    return dict


def add_starting_zero(program, data):

    """
    Add a zero at the starting time for the model run

    Args:
        program: The program to have a zero added

    Returns:
        program: The program with the zero added

    """

    program[int(data['attributes']['start_time'])] = 0.

    return program


def convert_dictionary_of_lists_to_dictionary_of_dictionaries(lists_to_process):

    dict = {}
    for list in lists_to_process:
        # If it's actually numeric data and not just a column that's of not much use
        # (not sure how to generalise this - clearly it should be generalised)
        if u'source' not in list and u'year' not in list and u'iso' not in list and u'who' not in list:
            dict[list] = {}
            for i in range(len(lists_to_process['year'])):
                dict[list][int(lists_to_process['year'][i])] = lists_to_process[list][i]
            # Remove nans
            dict[list] = remove_nans(dict[list])
    return dict



###############################################################
#  Readers

class BcgCoverageSheetReader:

    """
    Reader for the WHO/UNICEF BCG coverage data
    Now only reads the rows relevant to the country in question to speed things up
    Therefore, creates a single dictionary with years keys and coverage as a float
    for BCG coverage over
    """

    def __init__(self, country_to_read):
        self.data = {}  # Empty dictionary to contain the data that is read
        self.tab_name = 'BCG'  # Tab of spreadsheet to be read
        self.key = 'bcg'  # String that defines the data type in this file
        self.filename = 'xls/who_unicef_bcg_coverage.xlsx'  # Filename
        self.start_row = 0  # First row to be read in
        self.column_for_keys = 2  # Column that keys come from
        self.horizontal = True  # Orientation of spreadsheet
        self.country_to_read = country_to_read  # Country being read
        self.start_col = 4
        self.first_cell = u'Region'

    def parse_row(self, row):

        if row[0] == self.first_cell:
            self.parlist = parse_year_data(row, '', len(row))
        elif row[self.column_for_keys] == adjust_country_name(self.country_to_read)\
                or self.key == 'time_variants':
            for i in range(self.start_col, len(row)):
                if type(row[i]) == float:
                    self.data[int(self.parlist[i])] = \
                        row[i]

    def get_data(self):
        return self.data


class BirthRateReader:

    """
    Reader for the WHO/UNICEF BCG coverage data
    Same structure and approach as for BcgCoverageSheetReader above
    """

    def __init__(self, country_to_read):
        self.data = {}
        self.tab_name = 'Data'
        self.key = 'rate_birth'
        self.parlist = []
        self.filename = 'xls/world_bank_crude_birth_rate.xlsx'
        self.start_row = 0
        self.column_for_keys = 2
        self.horizontal = True
        self.country_to_read = country_to_read

    def parse_row(self, row):

        if row[0] == u'Series Name':
            for i in range(len(row)):
                self.parlist += \
                    [row[i][:4]]
        elif row[self.column_for_keys] == self.country_to_read:
            for i in range(4, len(row)):
                if type(row[i]) == float:
                    self.data[int(self.parlist[i])] = \
                        row[i]

    def get_data(self):
        return self.data


class LifeExpectancyReader:

    def __init__(self, country_to_read):
        self.data = {}
        self.tab_name = 'Data'
        self.key = 'life_expectancy'
        self.parlist = []
        self.filename = 'xls/world_bank_life_expectancy.xlsx'
        self.start_row = 3
        self.column_for_keys = 0
        self.horizontal = True
        self.country_to_read = country_to_read

    def parse_row(self, row):

        if row[0] == u'Country Name':
            self.parlist += row
        elif row[self.column_for_keys] == self.country_to_read:
            for i in range(4, len(row)):
                if type(row[i]) == float:
                    self.data[int(self.parlist[i])] = \
                        row[i]

    def get_data(self):
        return self.data


class FixedParametersReader:

    def __init__(self):
        self.data = {}
        self.tab_name = 'fixed_params'
        self.key = 'parameters'
        self.parlist = []
        self.filename = 'xls/universal_constants.xlsx'
        self.start_row = 1
        self.column_for_keys = 0
        self.horizontal = True
        self.parameter_dictionary_keys = []

    def parse_row(self, row):

        self.data[row[0]] = row[1]

    def get_data(self):
        return self.data


class ParametersReader(FixedParametersReader):

    def __init__(self, country_to_read):
        self.data = {}
        self.tab_name = 'miscellaneous_constants'
        self.key = 'miscellaneous'
        self.parlist = []
        self.filename = 'xls/programs_' + country_to_read.lower() + '.xlsx'
        self.start_row = 1
        self.column_for_keys = 0
        self.horizontal = True
        self.parameter_dictionary_keys = []


class ModelAttributesReader(FixedParametersReader):

    def __init__(self, country_to_read):
        self.data = {}
        self.data['start_compartments'] = {}
        self.tab_name = 'model_attributes'
        self.key = 'attributes'
        self.parlist = []
        self.filename = 'xls/programs_' + country_to_read.lower() + '.xlsx'
        self.start_row = 1
        self.column_for_keys = 0
        self.horizontal = True
        self.parameter_dictionary_keys = []

    def parse_row(self, row):

        # For the calendar year times
        if u'time' in row[0] or u'smoothness' in row[0]:
            self.data[row[0]] = float(row[1])

        elif u'fitting' in row[0]:
            self.data[row[0]] = int(row[1])

        # For the integration
        elif u'integration' in row[0]:
            self.data[row[0]] = row[1]

        # For the model stratifications
        elif u'n_' in row[0] or u'age_breakpoints' in row[0] or u'scenarios' in row[0]:
            self.data[row[0]] = []
            for i in range(1, len(row)):
                if not row[i] == '':
                    self.data[row[0]] += [int(row[i])]

        # For optional elaborations
        elif u'is_' in row[0]:
            self.data[row[0]] = []
            for i in range(1, len(row)):
                if not row[i] == '':
                    self.data[row[0]] += [bool(row[i])]

        # For starting compartments
        elif u'susceptible' in row[0] or u'active' in row[0]:
            self.data['start_compartments'][row[0]] = float(row[1])


class ProgramReader:

    def __init__(self, country_to_read):
        self.data = {}
        self.tab_name = 'time_variants'
        self.key = 'time_variants'
        self.filename = 'xls/programs_' + country_to_read.lower() + '.xlsx'
        self.start_row = 0
        self.column_for_keys = 0
        self.horizontal = True
        self.country_to_read = country_to_read
        self.start_col = 1
        self.first_cell = u'program'

    def parse_row(self, row):

        if row[0] == self.first_cell:
            self.parlist = parse_year_data(row, '', len(row))
        else:
            self.data[row[0]] = {}
            for i in range(self.start_col, len(row)):
                parlist_item_string = str(self.parlist[i])
                if ('19' in parlist_item_string or '20' in parlist_item_string) and row[i] != '':
                    self.data[row[0]][int(self.parlist[i])] = \
                        row[i]
                elif row[i] != '':
                    self.data[row[0]][self.parlist[i]] = \
                        row[i]

    def get_data(self):
        return self.data


class GlobalTbReportReader:

    def __init__(self, country_to_read):
        self.data = {}
        self.tab_name = 'TB_burden_countries_2016-04-19'
        self.key = 'tb'
        self.parlist = []
        self.filename = 'xls/gtb_data.xlsx'
        self.start_row = 1
        self.horizontal = False
        self.start_column = 0
        self.indices = []
        self.country_to_read = country_to_read

    def parse_col(self, col):

        if self.key in ['notifications', 'laboratories', 'outcomes']:
            col = replace_blanks(col, nan, '')

        # If it's the country column (the first one)
        if col[0] == u'country':

            # Find the indices for the country in question
            for i in range(len(col)):
                if col[i] == self.country_to_read:
                    self.indices += [i]

        # All other columns
        else:
            self.data[col[0]] = []
            for i in self.indices:
                self.data[col[0]] += [col[i]]

    def get_data(self):
        return self.data


class NotificationsReader(GlobalTbReportReader):

    def __init__(self, country_to_read):
        self.data = {}
        self.tab_name = 'TB_notifications_2016-04-20'
        self.key = 'notifications'
        self.parlist = []
        self.filename = 'xls/notifications_data.xlsx'
        self.start_row = 1
        self.horizontal = False
        self.start_column = 0
        self.start_row = 1
        self.indices = []
        self.country_to_read = country_to_read


class TreatmentOutcomesReader(GlobalTbReportReader):

    def __init__(self, country_to_read):
        self.data = {}
        self.tab_name = 'TB_outcomes_2016-04-21'
        self.key = 'outcomes'
        self.parlist = []
        self.filename = 'xls/outcome_data.xlsx'
        self.start_row = 1
        self.horizontal = False
        self.start_column = 0
        self.start_row = 1
        self.indices = []
        self.country_to_read = country_to_read


class MdrReportReader:

    def __init__(self, country_to_read):
        self.data = {}
        self.tab_name = 'MDR-TB_burden_estimates_2016-04'
        self.key = 'mdr'
        self.parlist = []
        self.filename = 'xls/mdr_data.xlsx'
        self.start_row = 0
        self.column_for_keys = 0
        self.horizontal = True
        self.dictionary_keys = []
        self.country_to_read = country_to_read

    def parse_row(self, row):

        # Create the list to turn in to dictionary keys later
        if row[0] == u'country':
            self.dictionary_keys += row

        # Populate when country to read is encountered
        elif row[0] == self.country_to_read:
            for i in range(len(self.dictionary_keys)):
                self.data[self.dictionary_keys[i]] = row[i]

    def get_data(self):
        return self.data


class LaboratoriesReader(GlobalTbReportReader):

    def __init__(self, country_to_read):
        self.data = {}
        self.tab_name = 'TB_laboratories_2016-04-21'
        self.key = 'laboratories'
        self.parlist = []
        self.filename = 'xls/laboratories_data.xlsx'
        self.start_row = 1
        self.horizontal = False
        self.start_column = 0
        self.indices = []
        self.country_to_read = country_to_read


class StrategyReader(MdrReportReader):

    def __init__(self, country_to_read):
        self.data = {}
        self.tab_name = 'TB_strategy_2016-04-21'
        self.key = 'strategy'
        self.parlist = []
        self.filename = 'xls/strategy_data.xlsx'
        self.start_row = 0
        self.column_for_keys = 0
        self.horizontal = True
        self.dictionary_keys = []
        self.country_to_read = country_to_read


###############################################################
#  Master scripts


def read_xls_with_sheet_readers(sheet_readers=[]):

    """
    Runs the individual readers to gather all the data from the sheets

    Args:
        sheet_readers: The sheet readers that were previously collated into a list

    Returns:
        All the data for reading as a single object
    """

    result = {}
    for reader in sheet_readers:

        # Check that the spreadsheet to be read exists
        try:
            workbook = open_workbook(reader.filename)
        except:
            raise Exception('Failed to open spreadsheet: %s' % reader.filename)
        #print("Reading sheet \"{}\"".format(reader.tab_name))
        sheet = workbook.sheet_by_name(reader.tab_name)

        # Read in the direction that the reader expects (either horizontal or vertical)
        if reader.horizontal:
            for i_row in range(reader.start_row, sheet.nrows):
                reader.parse_row(sheet.row_values(i_row))
        else:
            for i_col in range(reader.start_column, sheet.ncols):
                reader.parse_col(sheet.col_values(i_col))
        result[reader.key] = reader.get_data()

    return result


def read_input_data_xls(from_test, sheets_to_read, country):

    """
    Compile sheet readers into a list according to which ones have
    been selected.
    Note that most readers now take the country in question as an input,
    while only the fixed parameters sheet reader does not.

    Args:
        from_test: Whether being called from the directory above
        sheets_to_read: A list containing the strings that are also the
            'keys' attribute of the reader
        country: Country being read for

    Returns:
        A single data structure containing all the data to be read
            (by calling the read_xls_with_sheet_readers method)
    """

    sheet_readers = []

    if 'bcg' in sheets_to_read:
        sheet_readers.append(BcgCoverageSheetReader(country))
    if 'rate_birth' in sheets_to_read:
        sheet_readers.append(BirthRateReader(country))
    if 'life_expectancy' in sheets_to_read:
        sheet_readers.append(LifeExpectancyReader(country))
    if 'attributes' in sheets_to_read:
        sheet_readers.append(ModelAttributesReader(country))
    if 'parameters' in sheets_to_read:
        sheet_readers.append(FixedParametersReader())
    if 'miscellaneous' in sheets_to_read:
        sheet_readers.append(ParametersReader(country))
    if 'time_variants' in sheets_to_read:
        sheet_readers.append(ProgramReader(country))
    if 'tb' in sheets_to_read:
        sheet_readers.append(GlobalTbReportReader(country))
    if 'notifications' in sheets_to_read:
        sheet_readers.append(NotificationsReader(country))
    if 'outcomes' in sheets_to_read:
        sheet_readers.append(TreatmentOutcomesReader(country))
    if 'mdr' in sheets_to_read:
        sheet_readers.append(MdrReportReader(country))
    if 'laboratories' in sheets_to_read:
        sheet_readers.append(LaboratoriesReader(country))
    if 'strategy' in sheets_to_read:
        sheet_readers.append(StrategyReader(country))

    # If being run from the directory above
    if from_test:
        for reader in sheet_readers:
            reader.filename = os.path.join('autumn/', reader.filename)

    return read_xls_with_sheet_readers(sheet_readers)


def read_and_process_data(from_test, keys_of_sheets_to_read, country):

    """
    Runs the main data reading function and performs a few tidying tasks.

    Args:
        from_test, sheets_to_read, country: Are just passed on to the reading function.

    Returns:
        data: The adapted and processed data structure.
    """

    # First just get the main data object from the reading function
    data = read_input_data_xls(from_test, keys_of_sheets_to_read, country)

    # Calculate proportions that are smear-positive, smear-negative or extra-pulmonary
    # and add them to the data object's notification dictionary
    organs = [u'new_sp', u'new_sn', u'new_ep']
    data['notifications'].update(calculate_proportion(data['notifications'], organs, False))

    # Combine loaded data with data from spreadsheets for vaccination and case detection
    # Now with spreadsheet inputs over-riding GTB loaded data
    if data['time_variants']['program_prop_vaccination'][u'load_data'] == 'yes':
        for i in data['bcg']:
            # If not already loaded through the inputs spreadsheet
            if i not in data['time_variants']['program_prop_vaccination']:
                data['time_variants']['program_prop_vaccination'][i] = data['bcg'][i]

    # As above, now for case detection
    if data['time_variants']['program_prop_detect'][u'load_data'] == 'yes':
        for i in range(len(data['tb']['year'])):
            # If not already loaded through the inputs spreadsheet
            if data['tb']['year'][i] not in data['time_variants']['program_prop_detect']:
                data['time_variants']['program_prop_detect'][int(data['tb']['year'][i])] \
                    = data['tb']['c_cdr'][i]

    # Calculate proportions of patients with each outcome for DS-TB
    treatment_outcomes = [u'new_sp_cmplt', u'new_sp_cur', u'new_sp_def', u'new_sp_died', u'new_sp_fail']
    data['outcomes'].update(calculate_proportion(data['outcomes'], treatment_outcomes, True))

    # Calculate treatment success as cure plus completion
    data['outcomes'][u'prop_new_sp_success'] = []
    for i in range(len(data['outcomes'][u'prop_new_sp_cmplt'])):
        data['outcomes'][u'prop_new_sp_success'] \
            += [data['outcomes'][u'prop_new_sp_cmplt'][i] + data['outcomes'][u'prop_new_sp_cur'][i]]

    # Add the treatment success and death data to the program dictionary
    if data['time_variants']['program_prop_treatment_success'][u'load_data'] == 'yes':
        for i in range(len(data['outcomes'][u'prop_new_sp_success'])):
            if int(data['outcomes']['year'][i]) not in data['time_variants']['program_prop_treatment_success']:
                data['time_variants']['program_prop_treatment_success'][int(data['outcomes']['year'][i])] \
                    = data['outcomes'][u'prop_new_sp_success'][i]
    if data['time_variants']['program_prop_treatment_death'][u'load_data'] == 'yes':
        for i in range(len(data['outcomes'][u'prop_new_sp_died'])):
            if int(data['outcomes']['year'][i]) not in data['time_variants']['program_prop_treatment_death']:
                data['time_variants']['program_prop_treatment_death'][int(data['outcomes']['year'][i])] \
                    = data['outcomes'][u'prop_new_sp_died'][i]

    # Populate program dictionaries from epi ones
    for demo_parameter in ['life_expectancy', 'rate_birth']:
        if data['time_variants']['demo_' + demo_parameter][u'load_data'] == u'yes':
            for i in data[demo_parameter]:
                if i not in data['time_variants']['demo_' + demo_parameter]:
                    data['time_variants']['demo_' + demo_parameter][i] = data[demo_parameter][i]

    # Populate smear-positive and smear-negative proportion dictionaries to time-variant dictionary
    if data['time_variants']['tb_prop_smearpos'][u'load_data'] == u'yes':
        for i in range(len(data['notifications'][u'prop_new_sp'])):
            if int(data['notifications']['year'][i]) not in data['time_variants']['tb_prop_smearpos']:
                data['time_variants']['tb_prop_smearpos'][(int(data['notifications']['year'][i]))] = \
                    data['notifications'][u'prop_new_sp'][i]
    if data['time_variants']['tb_prop_smearneg'][u'load_data'] == u'yes':
        for i in range(len(data['notifications'][u'prop_new_sn'])):
            if int(data['notifications']['year'][i]) not in data['time_variants']['tb_prop_smearneg']:
                data['time_variants']['tb_prop_smearneg'][(int(data['notifications']['year'][i]))] = \
                    data['notifications'][u'prop_new_sn'][i]

    # Treatment outcomes
    # The aim is now to have data for success and death, as default can be derived from these
    # in the model module.

    # Calculate proportions of each outcome for MDR and XDR-TB
    # Outcomes for MDR and XDR in the GTB data
    mdr_xdr_treatment_outcomes = [u'succ', u'fail', u'died', u'lost']
    mdr_treatment_outcomes = []
    xdr_treatment_outcomes = []
    for i in range(len(mdr_xdr_treatment_outcomes)):
        mdr_treatment_outcomes += [u'mdr_' + mdr_xdr_treatment_outcomes[i]]
        xdr_treatment_outcomes += [u'xdr_' + mdr_xdr_treatment_outcomes[i]]
    data['outcomes'].update(calculate_proportion(data['outcomes'], mdr_treatment_outcomes, True))
    data['outcomes'].update(calculate_proportion(data['outcomes'], xdr_treatment_outcomes, True))

    # Duplicate DS-TB outcomes for single strain models
    for outcome in [u'_success', u'_death']:
        data['time_variants'][u'program_prop_treatment' + outcome + '_ds'] \
            = copy.copy(data['time_variants'][u'program_prop_treatment' + outcome])

    # Populate MDR and XDR data from outcomes dictionary into program dictionary
    for strain in [u'_mdr', u'_xdr']:
        for i in range(len(data['outcomes'][u'prop' + strain + u'_succ'])):
            # If there's a GTB value
            if not numpy.isnan(data['outcomes'][u'prop' + strain + u'_succ'][i]):
                # If there isn't already a value from the inputs spreadsheet
                if data['outcomes'][u'year'][i] \
                        not in data['time_variants'][u'program_prop_treatment_success' + strain]:
                    # Populate with value from report
                    data['time_variants'][u'program_prop_treatment_success' + strain][data['outcomes'][u'year'][i]] \
                        = data['outcomes'][u'prop' + strain + u'_succ'][i]
                # If there isn't already a value from the inputs spreadsheet
                if data['outcomes'][u'year'][i] \
                        not in data['time_variants'][u'program_prop_treatment_death' + strain]:
                    # Populate with value from report
                    data['time_variants'][u'program_prop_treatment_death' + strain][data['outcomes'][u'year'][i]] \
                        = data['outcomes'][u'prop' + strain + u'_died'][i]

    # Probably temporary code to assign the same treatment outcomes to XDR-TB as for inappropriate
    for outcome in [u'_success', u'_death']:
        data['time_variants'][u'program_prop_treatment' + outcome + u'_inappropriate'] \
            = copy.copy(data['time_variants'][u'program_prop_treatment' + outcome + u'_xdr'])

    # Final rounds of tidying programmatic data
    for program in data['time_variants']:

        # Add zero at starting time for model run to all programs that are proportions
        if u'program_prop' in program:
            data['time_variants'][program] = add_starting_zero(data['time_variants'][program], data)

        # Remove the load_data keys, as they have now been used
        data['time_variants'][program] = remove_specific_key(data['time_variants'][program], u'load_data')

        # Remove dictionary keys for which values are nan
        data['time_variants'][program] = remove_nans(data['time_variants'][program])

    # Convert list formats to dictionaries
    # This should be done above in the data readers, but as subsequent code depends on the lists,
    # I don't want to kill the lists yet. However, we now have consistent
    for data_item in ['tb', 'notifications', 'outcomes']:
        data[data_item + '_dict'] = convert_dictionary_of_lists_to_dictionary_of_dictionaries(data[data_item])

    return data


if __name__ == "__main__":

    country = u'Fiji'

    keys_of_sheets_to_read = [
        'bcg', 'rate_birth', 'life_expectancy', 'attributes', 'parameters', 'miscellaneous', 'time_variants', 'tb',
        'notifications', 'outcomes']
    data = read_and_process_data(False, keys_of_sheets_to_read, country)

    print("Time elapsed in running script is " + str(datetime.datetime.now() - spreadsheet_start_realtime))


