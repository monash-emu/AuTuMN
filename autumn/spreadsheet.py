# -*- coding: utf-8 -*-

from __future__ import print_function
from xlrd import open_workbook # For opening Excel workbooks
from numpy import nan
import numpy
import copy
import os
import datetime


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


def adjust_country_name(country_name, data_item):

    adjusted_country_name = country_name
    if country_name == u'Philippines' and data_item == 'bcg':
        adjusted_country_name = country_name + u' (the)'
    return adjusted_country_name


def calculate_proportion(data, indices):

    """
    Calculate the proportions of patients within subgroups

    Args:
        data: The main data structure containing all the data for that country
        indices: A list of the dictionary elements of data that are to be summed
            and then the proportions calculated

    Returns:
        proportions: A dictionary containing the proportions by indices
    """

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
            proportions['prop_' + indices[j]] \
                += [data[indices[j]][i]
                    / denominator[i]]

    return proportions


###############################################################
#  Readers

class BcgCoverageSheetReader():

    """
    Reader for the WHO/UNICEF BCG coverage data
    Now only reads the rows relevant to the country in question to speed things up
    Therefore, creates a single dictionary with years keys and coverage as a float
    for BCG coverage over
    """

    def __init__(self, country_to_read):
        self.data = {}  # Empty dictionary to contain the data that is read
        self.i_par = 0  # Starting row
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
        elif row[self.column_for_keys] == self.country_to_read\
                or self.key == 'programs':
            for i in range(self.start_col, len(row)):
                if type(row[i]) == float:
                    self.data[int(self.parlist[i])] = \
                        row[i]
        self.i_par += 1

    def get_data(self):
        return self.data


class BirthRateReader():

    """
    Reader for the WHO/UNICEF BCG coverage data
    Same structure and approach as for BcgCoverageSheetReader above
    """

    def __init__(self, country_to_read):
        self.data = {}
        self.i_par = 0
        self.tab_name = 'Data'
        self.key = 'birth_rate'
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
        self.i_par += 1

    def get_data(self):
        return self.data


class LifeExpectancyReader():

    def __init__(self, country_to_read):
        self.data = {}
        self.i_par = 0
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
        self.i_par += 1

    def get_data(self):
        return self.data


class FixedParametersReader():

    def __init__(self):
        self.data = {}
        self.i_par = 0
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
        self.i_par += 1

    def get_data(self):
        return self.data


class ParametersReader(FixedParametersReader):

    def __init__(self, country_to_read):
        self.data = {}
        self.i_par = 0
        self.tab_name = 'miscellaneous_constants'
        self.key = 'miscellaneous'
        self.parlist = []
        self.filename = 'xls/programs_' + country_to_read.lower() + '.xlsx'
        self.start_row = 1
        self.column_for_keys = 0
        self.horizontal = True
        self.parameter_dictionary_keys = []


class ProgramReader():

    def __init__(self, country_to_read):
        self.data = {}
        self.i_par = 0
        self.tab_name = 'programs'
        self.key = 'programs'
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
                if i == 1:
                    self.data[row[0]][self.parlist[i]] = \
                        row[i]
                elif type(row[i]) == float:
                    self.data[row[0]][int(self.parlist[i])] = \
                        row[i]

        self.i_par += 1

    def get_data(self):
        return self.data


class GlobalTbReportReader():

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
        self.i_par = -1
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


class MdrReportReader():

    def __init__(self, country_to_read):
        self.data = {}
        self.i_par = 0
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
        self.i_par += 1

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
        self.i_par = 0
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
    if 'birth_rate' in sheets_to_read:
        sheet_readers.append(BirthRateReader(country))
    if 'life_expectancy' in sheets_to_read:
        sheet_readers.append(LifeExpectancyReader(country))
    if 'parameters' in sheets_to_read:
        sheet_readers.append(FixedParametersReader())
    if 'miscellaneous' in sheets_to_read:
        sheet_readers.append(ParametersReader(country))
    if 'programs' in sheets_to_read:
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


def read_and_process_data(from_test, keys_of_sheets_to_read, country, start_time):

    """
    Runs the main data reading function and performs a few tidying tasks.
    Args:
        from_test:
        sheets_to_read:
        country:

    Returns:

    """

    data = read_input_data_xls(from_test, keys_of_sheets_to_read, country)

    # Calculate proportions that are smear-positive, smear-negative or extra-pulmonary
    # and add them to the data object
    organs = [u'new_sp', u'new_sn', u'new_ep']
    data['notifications'].update(calculate_proportion(data['notifications'], organs))

    # Combine loaded data with data from spreadsheets where applicable
    if data['programs']['program_prop_vaccination'][u'load_data'] == 'yes':
        data['programs']['program_prop_vaccination'].update(data['bcg'])
    if data['programs']['program_prop_detect'][u'load_data'] == 'yes':
        for i in range(len(data['tb']['year'])):
            data['programs']['program_prop_detect'][int(data['tb']['year'][i])] \
                = data['tb']['c_cdr'][i]
    print()
    for program in data['programs']:
        data['programs'][program][start_time] = 0.


    # Get rid of the load_data item from the program dictionaries
    for program in data['programs']:
        del data['programs'][program][u'load_data']

    return data


if __name__ == "__main__":

    country = u'Fiji'

    start_time = 1850

    keys_of_sheets_to_read = [
        'bcg',
        'birth_rate',
        'life_expectancy',
        'parameters',
        'miscellaneous',
        'programs',
        'tb',
        'notifications',
        'outcomes',
        'mdr',
        'laboratories',
        'strategy']

    data = read_and_process_data(False, keys_of_sheets_to_read, country, start_time)

    print("Time elapsed in running script is " + str(datetime.datetime.now() - spreadsheet_start_realtime))

