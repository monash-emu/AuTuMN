# -*- coding: utf-8 -*-

from __future__ import print_function
from xlrd import open_workbook
from numpy import nan
import numpy
import os
import tool_kit


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


def parse_year_data(year_data, blank, endcolumn):
    """
    Code to parse rows of data that are years.

    Args:
        year_data: The row to parse
        blank: A value for blanks to be ignored
        endcolumn: Column to end at
    """

    year_data = replace_specified_value(year_data, nan, blank)
    assumption_val = year_data[-1]
    year_vals = year_data[:endcolumn]
    if is_all_same_value(year_vals, nan):
        return [assumption_val] 
    else:
        return year_vals


#######################################
###  Individual spreadsheet readers ###
#######################################

class BcgCoverageSheetReader:
    """
    Reader for the WHO/UNICEF BCG coverage data.
    Creates a single dictionary with years keys and coverage as a float representing BCG coverage.
    Comments to each method and attribute apply to the other sheet readers below as well.
    """

    def __init__(self, country_to_read):

        self.country_to_read = country_to_read  # country being read
        self.data = {}  # empty dictionary to contain the data that is read
        self.tab_name = 'BCG'  # tab of spreadsheet to be read
        self.key = 'bcg'  # string that defines the data type in this file
        self.filename = 'xls/who_unicef_bcg_coverage.xlsx'  # filename
        self.start_row = 0  # first row to be read in
        self.start_col = 4
        self.column_for_keys = 2  # column that keys come from
        self.horizontal = True  # spreadsheet orientation
        self.first_cell = 'WHO_REGION'

    def parse_row(self, row):

        # parse first row
        if row[0] == self.first_cell:
            self.parlist = parse_year_data(row, '', len(row))
            for i in range(len(self.parlist)):
                self.parlist[i] = str(self.parlist[i])

        # subsequent rows
        elif row[self.column_for_keys] == tool_kit.adjust_country_name(self.country_to_read):
            for i in range(self.start_col, len(row)):
                if type(row[i]) == float:
                    self.data[int(self.parlist[i])] = row[i]

    def get_data(self):

        # simply return the data that has been collected
        return self.data


class BirthRateReader:
    """
    Reader for the WHO/UNICEF BCG coverage data. Same structure and approach as for BcgCoverageSheetReader above.
    """

    def __init__(self, country_to_read):

        self.country_to_read = country_to_read
        self.data = {}
        self.tab_name = 'Data'
        self.key = 'rate_birth'
        self.parlist = []
        self.filename = 'xls/world_bank_crude_birth_rate.xlsx'
        self.start_row = 0
        self.column_for_keys = 2
        self.horizontal = True

    def parse_row(self, row):

        if row[0] == 'Series Name':
            for i in range(len(row)):
                self.parlist += [row[i][:4]]
        elif row[self.column_for_keys] == self.country_to_read:
            for i in range(4, len(row)):
                if type(row[i]) == float:
                    self.data[int(self.parlist[i])] = row[i]

    def get_data(self):

        return self.data


class LifeExpectancyReader:

    def __init__(self, country_to_read):

        self.country_to_read = country_to_read
        self.data = {}
        self.tab_name = 'Data'
        self.key = 'life_expectancy'
        self.parlist = []
        self.filename = 'xls/world_bank_life_expectancy.xlsx'
        self.start_row = 3
        self.column_for_keys = 0
        self.horizontal = True

    def parse_row(self, row):

        if row[0] == 'Country Name':
            self.parlist += row
        elif row[self.column_for_keys] == self.country_to_read:
            for i in range(4, len(row)):
                if type(row[i]) == float:
                    self.data[int(self.parlist[i])] = row[i]

    def get_data(self):

        return self.data


class ControlPanelReader:

    def __init__(self):

        self.tab_name = 'control_panel'
        self.key = 'control_panel'
        self.filename = 'xls/control_panel.xlsx'
        self.general_program_intialisations()
        self.start_row = 0
        self.data['start_compartments'] = {}

    def general_program_intialisations(self):

        self.data = {}
        self.parlist = []
        self.start_row = 1
        self.column_for_keys = 0
        self.horizontal = True
        self.parameter_dictionary_keys = []

    def parse_row(self, row):

        if row[0] != 'age_breakpoints' and row[1] == '':
            pass

        # for the country to be analysed
        elif row[0] == 'country':
            self.data[str(row[0])] = str(row[1])

        # integration method
        elif row[0] == 'integration':
            self.data[str(row[0])] = str(row[1])

        # stratifications
        elif row[0][:2] == 'n_':
            self.data[str(row[0])] = int(row[1])

        # for the calendar year times
        elif 'time' in row[0] or 'smoothness' in row[0]:
            self.data[str(row[0])] = float(row[1])

        # fitting approach
        elif 'fitting' in row[0]:
            self.data[str(row[0])] = int(row[1])

        # all instructions around outputs, plotting and spreadsheet/document writing
        elif 'output_' in row[0]:
            self.data[str(row[0])] = bool(row[1])

        # conditionals for model
        elif row[0][:3] == 'is_' or row[0][:11] == 'comorbidity':
            self.data[str(row[0])] = bool(row[1])

        # don't set scenarios through the default sheet
        elif row[0] == 'scenarios_to_run':
            self.data[str(row[0])] = [None]
            if self.key == 'control_panel':
                for i in range(1, len(row)):
                    if not row[i] == '': self.data[str(row[0])] += [int(row[i])]

        # age breakpoints (arguably should just be an empty list always)
        elif row[0] == 'age_breakpoints':
            self.data[str(row[0])] = []
            for i in range(1, len(row)):
                if not row[i] == '': self.data[str(row[0])] += [int(row[i])]

        # parameters values
        else:
            self.data[str(row[0])] = row[1]

        # uncertainty parameters
        # not sure why this if statement needs to be split exactly, but huge bugs seem to occur if it isn't
        if len(row) >= 4:
            # if an entry present in second column and it is a constant parameter that can be modified in uncertainty
            if row[2] != '' and ('tb_' in row[0] or 'program_' in row[0]):
                uncertainty_dict = {'point': row[1],
                                    'lower': row[2],
                                    'upper': row[3]}
                self.data[str(row[0]) + '_uncertainty'] = uncertainty_dict

    def get_data(self):

        return self.data


class FixedParametersReader(ControlPanelReader):

    def __init__(self):

        self.tab_name = 'constants'
        self.key = 'default_constants'
        self.filename = 'xls/data_default.xlsx'
        self.general_program_intialisations()


class CountryParametersReader(ControlPanelReader):

    def __init__(self, country_to_read):

        self.tab_name = 'constants'
        self.key = 'country_constants'
        self.filename = 'xls/data_' + country_to_read.lower() + '.xlsx'
        self.general_program_intialisations()


class DefaultProgramReader:

    def __init__(self):

        self.filename = 'xls/data_default.xlsx'
        self.key = 'default_programs'
        self.general_program_intialisations()

    def general_program_intialisations(self):

        self.data = {}
        self.tab_name = 'time_variants'
        self.start_row = 0
        self.column_for_keys = 0
        self.horizontal = True
        self.start_col = 1
        self.first_cell = 'program'

    def parse_row(self, row):

        if row[0] == self.first_cell:
            self.parlist = parse_year_data(row, '', len(row))
        else:
            self.data[str(row[0])] = {}
            for i in range(self.start_col, len(row)):
                parlist_item_string = str(self.parlist[i])
                if ('19' in parlist_item_string or '20' in parlist_item_string) and row[i] != '':
                    self.data[row[0]][int(self.parlist[i])] = \
                        row[i]
                elif row[i] != '':
                    self.data[str(row[0])][str(self.parlist[i])] = row[i]

    def get_data(self):

        return self.data


class CountryProgramReader(DefaultProgramReader):

    def __init__(self, country_to_read):

        self.filename = 'xls/data_' + country_to_read.lower() + '.xlsx'
        self.country_to_read = country_to_read
        self.key = 'country_programs'
        self.general_program_intialisations()


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

        col = replace_specified_value(col, nan, '')

        # if it's the country column (the first one)
        if col[0] == 'country':

            # find the indices for the country in question
            for i in range(len(col)):
                if col[i] == self.country_to_read: self.indices += [i]

        elif 'iso' in col[0] or 'g_who' in col[0] or 'source' in col[0]:
            pass

        elif col[0] == 'year':
            self.year_indices = {}
            for i in self.indices:
                self.year_indices[int(col[i])] = i

        # all other columns
        else:
            self.data[str(col[0])] = {}
            for year in self.year_indices:
                if not numpy.isnan(col[self.year_indices[year]]):
                    self.data[col[0]][year] = col[self.year_indices[year]]

    def get_data(self):

        return self.data


class NotificationsReader(GlobalTbReportReader):

    def __init__(self, country_to_read):

        self.data = {}
        self.tab_name = 'TB_notifications_2016-12-22'
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
        self.tab_name = 'MDR_RR_TB_burden_estimates_2016'
        self.key = 'mdr'
        self.parlist = []
        self.filename = 'xls/mdr_data.xlsx'
        self.start_row = 0
        self.horizontal = True
        self.dictionary_keys = []
        self.country_to_read = country_to_read

    def parse_row(self, row):

        # create the list to turn in to dictionary keys later
        if row[0] == 'country': self.dictionary_keys += row

        # populate when country to read is encountered
        elif row[0] == self.country_to_read:
            for i in range(len(self.dictionary_keys)):
                self.data[self.dictionary_keys[i]] = row[i]

    def get_data(self):

        return self.data


class LaboratoriesReader(GlobalTbReportReader):

    def __init__(self, country_to_read):

        self.data = {}
        self.tab_name = 'TB_laboratories_2016-12-22'
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
        self.tab_name = 'TB_policies_services_2016-12-22'
        self.key = 'strategy'
        self.parlist = []
        self.filename = 'xls/strategy_data.xlsx'
        self.start_row = 0
        self.column_for_keys = 0
        self.horizontal = True
        self.dictionary_keys = []
        self.country_to_read = country_to_read


class DiabetesReportReader:

    def __init__(self, country_to_read):

        self.data = {}
        self.tab_name = 'DM estimates 2015'
        self.key = 'diabetes'
        self.parlist = []
        self.filename = 'xls/diabetes_internationaldiabetesfederation.xlsx'
        self.start_row = 2
        self.column_for_keys = 0
        self.horizontal = True
        self.dictionary_keys = []
        self.country_to_read = country_to_read

    def parse_row(self, row):

        # create the list to turn in to dictionary keys later
        if row[0] == u'Country/territory':
            self.dictionary_keys += row

        # populate when country to read is encountered
        elif row[0] == self.country_to_read:
            for i in range(len(self.dictionary_keys)):
                if self.dictionary_keys[i][:28] == u'Diabetes national prevalence':
                    self.data['comorb_prop_diabetes'] = float(row[i][:row[i].find('\n')]) / 1E2

    def get_data(self):

        return self.data


######################
### Master scripts ###
######################


def read_xls_with_sheet_readers(sheet_readers):
    """
    Runs the individual readers to gather all the data from the sheets

    Args:
        sheet_readers: The sheet readers that were previously collated into a list
    Returns:
        All the data for reading as a single object
    """

    result = {}
    for reader in sheet_readers:

        try:

            # check that the spreadsheet to be read exists
            workbook = open_workbook(reader.filename)

        except:

            # if sheet unavailable, print error message
            print('Unable to open country spreadsheet')

        else:

            # if the workbook is available, read the sheet in question
            sheet = workbook.sheet_by_name(reader.tab_name)

            # read in the direction that the reader expects (either horizontal or vertical)
            if reader.horizontal:
                for i_row in range(reader.start_row, sheet.nrows):
                    reader.parse_row(sheet.row_values(i_row))
            else:
                for i_col in range(reader.start_column, sheet.ncols):
                    reader.parse_col(sheet.col_values(i_col))
            result[reader.key] = reader.get_data()

    return result


def read_input_data_xls(from_test, sheets_to_read, country=None):
    """
    Compile sheet readers into a list according to which ones have been selected.
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
    if 'control_panel' in sheets_to_read:
        sheet_readers.append(ControlPanelReader())
    if 'default_constants' in sheets_to_read:
        sheet_readers.append(FixedParametersReader())
    if 'country_constants' in sheets_to_read:
        sheet_readers.append(CountryParametersReader(country))
    if 'default_programs' in sheets_to_read:
        sheet_readers.append(DefaultProgramReader())
    if 'country_programs' in sheets_to_read:
        sheet_readers.append(CountryProgramReader(country))
    if 'tb' in sheets_to_read:
        if country == 'Moldova':
            sheet_readers.append(GlobalTbReportReader('Republic of Moldova'))
        else:
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
    if 'diabetes' in sheets_to_read:
        sheet_readers.append(DiabetesReportReader(country))

    # if being run from the directory above
    if from_test:
        for reader in sheet_readers:
            reader.filename = os.path.join('autumn/', reader.filename)

    return read_xls_with_sheet_readers(sheet_readers)


