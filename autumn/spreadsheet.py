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
#  General functions

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


def calculate_proportion(country_data, numerator, denominators):

    """

    Calculate the proportions of patients within subgroups

    Args:
        country_data: The main data structure containing all the data for that country
        numerator: The key indexing the current working numerator dictionary, whose proportion is to be found
        denominators: All keys of all the dictionaries that contribute to the denominator

    Returns:
        proportion: A dictionary containing the proportions contained within the current numerator dictionary

    """
    proportion = {}
    for i in country_data[numerator]:
        total = 0
        for j in denominators:
            total += country_data[j][i]
        result = country_data[numerator][i] / total
        if not numpy.isnan(result):
            proportion[i] = result
    return proportion

###############################################################
#  Readers

class BcgCoverageSheetReader():

    """
    Reader for the WHO/UNICEF BCG coverage data
    Now only reads the rows relevant to the country in question to speed things up
    Therefore, creates a single dictionary with years keys and coverage as a float
    for BCG coverage over
    """

    def __init__(self, country):
        self.data = {}
        self.par = None
        self.i_par = -1
        self.tab_name = 'BCG'
        self.key = 'bcg'
        self.parlist = []
        self.filename = 'xls/who_unicef_bcg_coverage.xlsx'
        self.start_row = 0
        self.create_parlist = False
        self.column_for_keys = 2
        self.horizontal = True
        self.country = country

    def parse_row(self, row):

        self.i_par += 1
        if row[2] == u'Cname':
            self.parlist += \
                parse_year_data(row, '', len(row))
        elif row[2] == self.country:
            for i in range(4, len(row[4:])):
                if type(row[i]) == float:
                    self.data[self.parlist[i]] = \
                        row[i]

    def get_data(self):
        return self.data


class BirthRateReader():

    def __init__(self):
        self.data = {}
        self.par = None
        self.i_par = -1
        self.tab_name = 'Data'
        self.key = 'birth_rate'
        self.parlist = []
        self.filename = 'xls/world_bank_crude_birth_rate.xlsx'
        self.start_row = 0
        self.create_parlist = True
        self.column_for_keys = 2
        self.horizontal = True

    def parse_row(self, row):

        self.i_par += 1
        self.par = self.parlist[self.i_par]
        if row[2] == u'Country Name':  # Year
            self.data[self.par] = []
            for i in range(4, len(row)):
                self.data[self.par] += [int(row[i][:4])]
        elif row[2] == '':  # Blank lines at the end
            return
        else:  # Data
            self.data[self.par] =\
                parse_year_data(row[4:], u'..', -1)

    def get_data(self):
        return self.data


class LifeExpectancyReader():

    def __init__(self):
        self.data = {}
        self.par = None
        self.i_par = -1
        self.tab_name = 'Data'
        self.key = 'life_expectancy'
        self.parlist = []
        self.filename = 'xls/world_bank_life_expectancy.xlsx'
        self.start_row = 3
        self.create_parlist = True
        self.column_for_keys = 0
        self.horizontal = True

    def parse_row(self, row):

        self.i_par += 1
        self.par = self.parlist[self.i_par]
        if row[0] == u'Country Name':  # Year
            self.data[self.par] = []
            for i in range(4, len(row)):
                self.data[self.par] += [int(row[i])]
        else:  # Data
            self.data[self.par] =\
                parse_year_data(row[4:], u'', -1)

    def get_data(self):
        return self.data


class FixedParametersReader():

    def __init__(self):
        self.data = {}
        self.par = None
        self.i_par = -1
        self.tab_name = 'fixed_params'
        self.key = 'params'
        self.parlist = []
        self.filename = 'xls/universal_constants.xlsx'
        self.start_row = 0
        self.create_parlist = True
        self.column_for_keys = 0
        self.horizontal = True
        self.parameter_dictionary_keys = []

    def parse_row(self, row):

        self.i_par += 1
        self.par = self.parlist[self.i_par]

        if self.i_par > 0:
            self.data[row[0]] = row[1]

    def get_data(self):
        return self.data


class ParametersReader(FixedParametersReader):

    def __init__(self):
        self.data = {}
        self.par = None
        self.i_par = -1
        self.tab_name = 'miscellaneous_constants'
        self.key = 'miscellaneous'
        self.parlist = []
        self.filename = 'xls/programs_fiji.xlsx'
        self.start_row = 0
        self.create_parlist = True
        self.column_for_keys = 0
        self.horizontal = True
        self.parameter_dictionary_keys = []


class GlobalTbReportReader():

    def __init__(self):
        self.data = {}
        self.par = None
        self.i_par = -1
        self.tab_name = 'TB_burden_countries_2016-04-19'
        self.key = 'tb'
        self.parlist = []
        self.filename = 'xls/TB_burden_countries_2016-04-19.xlsx'
        self.start_row = 1
        self.create_parlist = True
        self.horizontal = False
        self.start_column = 0
        self.indices = {}

    def parse_col(self, col):

        if self.key == 'notifications':
            col = replace_blanks(col, nan, '')

        # If it's the country column (the first one)
        if col[0] == u'country':

            # Cycle through the rest of the country column
            for i in range(1, len(col)):

                # Create a list of indices for the countries
                # when parsing the first column
                self.indices[i] = col[i]

                # Add a dictionary for that country
                if col[i] not in self.data:
                    self.data[col[i]] = {}

        # All other columns
        else:

            # Ignore the first row
            for i in range(1, len(col)):

                # The data item
                item_to_add = col[i]

                # Add a dictionary key if that country hasn't encountered the field yet
                if col[0] not in self.data[self.indices[i]]:
                    self.data[self.indices[i]][col[0]] = []

                # Add the item to the dictionary
                self.data[self.indices[i]][col[0]] += [item_to_add]

    def get_data(self):

        return self.data


class MdrReportReader():

    def __init__(self):
        self.data = []
        self.par = None
        self.i_par = -1
        self.tab_name = 'MDR-TB_burden_estimates_2016-04'
        self.key = 'mdr'
        self.parlist = []
        self.filename = 'xls/MDR-TB_burden_estimates_2016-04-20.xlsx'
        self.start_row = 0
        self.create_parlist = True
        self.column_for_keys = 0
        self.horizontal = True
        self.country_dictionary_keys = []

    def parse_row(self, row):

        self.i_par += 1
        self.par = self.parlist[self.i_par]

        if row[0] == u'country':
            for i in range(len(row)):
                self.country_dictionary_keys += [row[i]]
        else:
            self.data += [{}]
            for i in range(len(row)):
                self.data[self.i_par - 1][self.country_dictionary_keys[i]] = row[i]

    def get_data(self):
        return self.data


class NotificationsReader(GlobalTbReportReader):

    def __init__(self):
        self.data = {}
        self.par = None
        self.i_par = -1
        self.tab_name = 'TB_notifications_2016-04-20'
        self.key = 'notifications'
        self.parlist = []
        self.filename = 'xls/TB_notifications_2016-04-20.xlsx'
        self.start_row = 1
        self.create_parlist = True
        self.horizontal = False
        self.start_column = 0
        self.start_row = 1
        self.indices = {}


class LaboratoriesReader(GlobalTbReportReader):

    def __init__(self):
        self.data = {}
        self.par = None
        self.i_par = -1
        self.tab_name = 'TB_laboratories_2016-04-21'
        self.key = 'laboratories'
        self.parlist = []
        self.filename = 'xls/TB_laboratories_2016-04-21.xlsx'
        self.start_row = 1
        self.create_parlist = True
        self.horizontal = False
        self.start_column = 0
        self.start_row = 1
        self.indices = {}


class TreatmentOutcomesReader(GlobalTbReportReader):

    def __init__(self):
        self.data = {}
        self.par = None
        self.i_par = -1
        self.tab_name = 'TB_outcomes_2016-04-21'
        self.key = 'outcomes'
        self.parlist = []
        self.filename = 'xls/TB_outcomes_2016-04-21.xlsx'
        self.start_row = 1
        self.create_parlist = True
        self.horizontal = False
        self.start_column = 0
        self.start_row = 1
        self.indices = {}


class StrategyReader(MdrReportReader):

    def __init__(self):
        self.data = []
        self.par = None
        self.i_par = -1
        self.tab_name = 'TB_strategy_2016-04-21'
        self.key = 'strategy'
        self.parlist = []
        self.filename = 'xls/TB_strategy_2016-04-21.xlsx'
        self.start_row = 0
        self.create_parlist = True
        self.column_for_keys = 0
        self.horizontal = True
        self.country_dictionary_keys = []


###############################################################
#  Master scripts


def read_xls_with_sheet_readers(sheet_readers=[]):

    result = {}
    for reader in sheet_readers:
        try:
            workbook = open_workbook(reader.filename)
        except:
            raise Exception('Failed to open spreadsheet: %s' % reader.filename)
        #print("Reading sheet \"{}\"".format(reader.tab_name))
        sheet = workbook.sheet_by_name(reader.tab_name)
        if reader.horizontal:
            for i_row in range(reader.start_row, sheet.nrows):
                if reader.create_parlist:
                    key = [sheet.row_values(i_row)[reader.column_for_keys]]
                    if key == [u'Cname'] or key == [u'Country Name']:
                        key = [u'year']
                    reader.parlist += key
                reader.parse_row(sheet.row_values(i_row))
        else:
            for i_col in range(reader.start_column, sheet.ncols):
                reader.parse_col(sheet.col_values(i_col))
        result[reader.key] = reader.get_data()

    return result


def read_input_data_xls(from_test, sheets_to_read, country):

    sheet_readers = []

    if 'parameters' in sheets_to_read:
        sheet_readers.append(FixedParametersReader())
    if 'miscellaneous' in sheets_to_read:
        sheet_readers.append(ParametersReader())
    if 'bcg' in sheets_to_read:
        sheet_readers.append(BcgCoverageSheetReader(country))
    if 'birth_rate' in sheets_to_read:
        sheet_readers.append(BirthRateReader())
    if 'life_expectancy' in sheets_to_read:
        sheet_readers.append(LifeExpectancyReader())
    if 'tb' in sheets_to_read:
        sheet_readers.append(GlobalTbReportReader())
    if 'mdr' in sheets_to_read:
        sheet_readers.append(MdrReportReader())
    if 'notifications' in sheets_to_read:
        sheet_readers.append(NotificationsReader())
    if 'lab' in sheets_to_read:
        sheet_readers.append(LaboratoriesReader())
    if 'outcomes' in sheets_to_read:
        sheet_readers.append(TreatmentOutcomesReader())
    if 'strategy' in sheets_to_read:
        sheet_readers.append(StrategyReader())

    if from_test:
        for reader in sheet_readers:
            reader.filename = os.path.join('autumn/', reader.filename)

    return read_xls_with_sheet_readers(sheet_readers)


def get_country_data(spreadsheet, data, data_item, country_name):

    # Call function to adjust country name
    adjusted_country_name = adjust_country_name(country_name, data_item)

    # Initialise an empty dictionary for the data field being extracted
    country_data_field = {}

    # If it's a Global TB Report data field (Afghanistan is arbitrary)
    if data_item in data[spreadsheet][u'Afghanistan'] or data_item in data['outcomes'][u'Afghanistan']:
        if data_item in data[spreadsheet][u'Afghanistan']:
            gtb_sheet = spreadsheet
        elif data_item in data['outcomes'][u'Afghanistan']:
            gtb_sheet = 'outcomes'

        for i in range(len(data[gtb_sheet][adjusted_country_name][u'year'])):
            country_data_field[data[gtb_sheet][adjusted_country_name][u'year'][i]] = \
                data[gtb_sheet][adjusted_country_name][data_item][i]

    elif data_item == 'bcg':
        country_data_field[data_item] = data[data_item]

    # For the other spreadsheets
    else:
        for i in range(len(data[data_item][adjusted_country_name])):
            if not numpy.isnan(data[data_item][adjusted_country_name][i]):
                country_data_field[data[data_item]['year'][i]] = data[data_item][adjusted_country_name][i]

    return country_data_field





if __name__ == "__main__":
    import json
    country = u'Algeria'
    data = read_input_data_xls(False, ['bcg',
                                       'birth_rate', 'life_expectancy',
                                       'tb', 'outcomes', 'notifications',
                                       'parameters', 'miscellaneous'],
                               country)
                               # , 'mdr', 'lab', 'strategy'])
    # I suspect the next line of code was causing the problems with GitHub desktop
    # failing to create commits, so commented out:
    # open('spreadsheet.out.txt', 'w').write(json.dumps(data, indent=2))
    country_data = {}
    for data_item in ['birth_rate', 'life_expectancy', 'bcg', u'c_cdr', u'c_new_tsr']:
        country_data[data_item] = get_country_data('tb', data, data_item, country)
    for data_item in [u'new_sp', u'new_sn', u'new_ep']:
        country_data[data_item] = get_country_data('notifications', data, data_item, country)

    # Calculate proportions that are smear-positive, smear-negative or extra-pulmonary
    organs = [u'new_sp', u'new_sn', u'new_ep']
    for organ in organs:
        country_data[u'prop_' + organ] =\
            calculate_proportion(country_data, organ, organs)

    print(country_data)
    print("Time elapsed in running script is " + str(datetime.datetime.now() - spreadsheet_start_realtime))

