

from xlrd import open_workbook
from numpy import nan
import numpy
import os
import tool_kit


''' static functions '''


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
    assumption_val, year_vals = year_data[-1], year_data[:endcolumn]
    if is_all_same_value(year_vals, nan):
        return [assumption_val]
    else:
        return year_vals


'''  individual spreadsheet readers '''


class MasterReader:
    """
    Reader for the WHO/UNICEF BCG coverage data.
    Creates a single dictionary with years keys and coverage as a float representing BCG coverage.
    Comments to each method and attribute apply to the other sheet readers below as well.
    """

    def __init__(self, country_to_read, purpose):

        self.key = purpose  # string that defines the data type in this file
        self.country_to_read = country_to_read  # country being read
        self.data = {}  # empty dictionary to contain the data that is read
        tab_dictionary = {'bcg': 'BCG',
                          'rate_birth': 'Data',
                          'life_expectancy': 'Data',
                          'tb': 'TB_burden_countries_2016-04-19',
                          'default_constants': 'constants',
                          'country_constants': 'constants',
                          'default_programs': 'time_variants',
                          'country_programs': 'time_variants',
                          'notifications': 'TB_notifications_2016-12-22',
                          'outcomes': 'TB_outcomes_2016-04-21',
                          'laboratories': 'TB_laboratories_2016-12-22',
                          'strategy': 'TB_policies_services_2016-12-22'}
        self.tab_name = tab_dictionary[purpose]
        filenames = {'bcg': 'xls/who_unicef_bcg_coverage.xlsx',
                     'rate_birth': 'xls/world_bank_crude_birth_rate.xlsx',
                     'life_expectancy': 'xls/world_bank_life_expectancy.xlsx',
                     'tb': 'xls/gtb_data.xlsx',
                     'default_constants': 'xls/data_default.xlsx',
                     'country_constants': 'xls/data_' + country_to_read.lower() + '.xlsx',
                     'default_programs': 'xls/data_default.xlsx',
                     'country_programs': 'xls/data_' + country_to_read.lower() + '.xlsx',
                     'notifications': 'xls/notifications_data.xlsx',
                     'outcomes': 'xls/outcome_data.xlsx',
                     'laboratores': 'xls/laboratories_data.xlsx',
                     'strategy': 'xls/strategy_data.xlsx'}
        self.filename = filenames[purpose]
        start_rows = {'bcg': 0,
                      'rate_birth': 0,
                      'life_expectancy': 3,
                      'tb': 1,
                      'default_constants': 1,
                      'country_constants': 1,
                      'default_programs': 0,
                      'country_programs': 0,
                      'notifications': 1,
                      'outcomes': 1,
                      'laboratories': 1,
                      'strategy': 0}
        self.start_row = start_rows[purpose]
        start_cols = {'bcg': 4,
                      'rate_birth': 'n/a',
                      'life_expectancy': 'n/a',
                      'tb': 0,
                      'default_constants': 'n/a',
                      'country_constants': 'n/a',
                      'default_programs': 1,
                      'country_programs': 1,
                      'notifications': 0,
                      'outcomes': 0,
                      'laboratories': 0,
                      'strategy': 'n/a'}
        self.start_col = start_cols[purpose]
        columns_for_keys = {'bcg': 2,
                            'rate_birth': 2,
                            'life_expectancy': 0,
                            'tb': 'n/a',
                            'default_constants': 0,
                            'country_constants': 0,
                            'default_programs': 0,
                            'country_programs': 0,
                            'notifications': 'n/a',
                            'outcomes': 'n/a',
                            'laboratories': 'n/a',
                            'strategy': 0}
        self.column_for_keys = columns_for_keys[purpose]
        first_cells = {'bcg': 'WHO_REGION',
                       'rate_birth': 'n/a',
                       'life_expectancy': 'n/a',
                       'tb': 'n/a',
                       'default_constants': 'n/a',
                       'country_constants': 'n/a',
                       'default_programs': 'program',
                       'country_programs': 'program',
                       'notifications': 'n/a',
                       'outcomes': 'n/a',
                       'laboratories': 'n/a',
                       'strategy': 'n/a'}
        self.first_cell = first_cells[purpose]
        self.horizontal = True  # spreadsheet orientation
        if self.key in ['tb', 'notifications', 'outcomes', 'laboratories']: self.horizontal = False

        self.indices = []
        self.parlist = []

    def parse_row(self, row):

        if self.key == 'bcg':

            # first row
            if row[0] == self.first_cell:
                self.parlist = parse_year_data(row, '', len(row))
                for i in range(len(self.parlist)): self.parlist[i] = str(self.parlist[i])

            # subsequent rows
            elif row[self.column_for_keys] \
                    == tool_kit.adjust_country_name(self.country_to_read, 'tb'):
                for i in range(self.start_col, len(row)):
                    if type(row[i]) == float: self.data[int(self.parlist[i])] = row[i]

        elif self.key == 'rate_birth':

            if row[0] == 'Series Name':
                for i in range(len(row)):
                    self.parlist += [row[i][:4]]
            elif row[self.column_for_keys] == self.country_to_read:
                for i in range(4, len(row)):
                    if type(row[i]) == float:
                        self.data[int(self.parlist[i])] = row[i]

        elif self.key == 'life_expectancy':

            if row[0] == 'Country Name':
                self.parlist += row
            elif row[self.column_for_keys] == self.country_to_read:
                for i in range(4, len(row)):
                    if type(row[i]) == float:
                        self.data[int(self.parlist[i])] = row[i]

        elif self.key == 'default_constants' or self.key == 'country_constants':

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
                if row[2] != '':
                    uncertainty_dict = {'point': row[1],
                                        'lower': row[2],
                                        'upper': row[3]}
                    self.data[str(row[0]) + '_uncertainty'] = uncertainty_dict

        elif self.key == 'default_programs' or self.key == 'country_programs':

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

        elif self.key == 'strategy':

            # create the list to turn in to dictionary keys later
            if row[0] == 'country':
                self.dictionary_keys += row

            # populate when country to read is encountered
            elif row[0] == self.country_to_read:
                for i in range(len(self.dictionary_keys)):
                    self.data[self.dictionary_keys[i]] = row[i]

    def parse_col(self, col):

        if self.key == 'tb' or self.key == 'notifications' or self.key == 'outcomes' or self.key == 'laboratories'\
                or self.key == 'strategy':

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


class GlobalTbReportReader:

    def __init__(self, country_to_read):

        self.data = {}
        self.tab_name = 'TB_burden_countries_2016-04-19'
        self.key = 'tb'
        self.parlist = []
        self.filename = 'xls/gtb_data.xlsx'
        self.start_row = 1
        self.horizontal = False
        self.start_col = 0
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
        self.country_to_read = tool_kit.adjust_country_name(country_to_read, 'tb')

    def parse_row(self, row):

        # create the list to turn in to dictionary keys later
        if row[0] == 'country': self.dictionary_keys += row

        # populate when country to read is encountered
        elif row[0] == self.country_to_read:
            for i in range(len(self.dictionary_keys)):
                self.data[self.dictionary_keys[i]] = row[i]


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
        self.country_to_read = tool_kit.adjust_country_name(country_to_read, 'tb')


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
        self.country_to_read = tool_kit.adjust_country_name(country_to_read, 'tb')

    def parse_row(self, row):

        # create the list to turn in to dictionary keys later
        if row[0] == u'Country/territory':
            self.dictionary_keys += row

        # populate when country to read is encountered
        elif row[0] == self.country_to_read:
            for i in range(len(self.dictionary_keys)):
                if self.dictionary_keys[i][:28] == u'Diabetes national prevalence':
                    self.data['comorb_prop_diabetes'] = float(row[i][:row[i].find('\n')]) / 1E2


''' master functions to call readers '''


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

        # check that the spreadsheet to be read exists
        try:
            print('Reading file', os.getcwd(), reader.filename)
            workbook = open_workbook(reader.filename)

        # if sheet unavailable, report error
        except:
            print('Unable to open country spreadsheet')

        # if the workbook is available, read the sheet in question
        else:
            sheet = workbook.sheet_by_name(reader.tab_name)
            if reader.horizontal:
                for i_row in range(reader.start_row, sheet.nrows): reader.parse_row(sheet.row_values(i_row))
            else:
                for i_col in range(reader.start_col, sheet.ncols): reader.parse_col(sheet.col_values(i_col))
            result[reader.key] = reader.data

    return result


def read_input_data_xls(from_test, sheets_to_read, country):
    """
    Compile sheet readers into a list according to which ones have been selected.
    Note that most readers now take the country in question as an input, while only the fixed parameters sheet reader
    does not.

    Args:
        from_test: Whether being called from the directory above
        sheets_to_read: A list containing the strings that are also the 'keys' attribute of the reader
        country: Country being read for
    Returns:
        A single data structure containing all the data to be read (by calling the read_xls_with_sheet_readers method)
    """

    # add sheet readers as required
    sheet_readers = []
    if 'default_constants' in sheets_to_read:
        sheet_readers.append(MasterReader(country, 'default_constants'))
    if 'bcg' in sheets_to_read:
        sheet_readers.append(MasterReader(country, 'bcg'))
    if 'rate_birth' in sheets_to_read:
        sheet_readers.append(MasterReader(tool_kit.adjust_country_name(country, 'demographic'), 'rate_birth'))
    if 'life_expectancy' in sheets_to_read:
        sheet_readers.append(MasterReader(tool_kit.adjust_country_name(country, 'demographic'), 'life_expectancy'))
    if 'country_constants' in sheets_to_read:
        sheet_readers.append(MasterReader(country, 'country_constants'))
    if 'default_programs' in sheets_to_read:
        sheet_readers.append(MasterReader(country, 'default_programs'))
    if 'country_programs' in sheets_to_read:
        sheet_readers.append(MasterReader(country, 'country_programs'))
    if 'tb' in sheets_to_read:
        sheet_readers.append(MasterReader(tool_kit.adjust_country_name(country, 'tb'), 'tb'))
    if 'notifications' in sheets_to_read:
        sheet_readers.append(MasterReader(tool_kit.adjust_country_name(country, 'tb'), 'notifications'))
    if 'outcomes' in sheets_to_read:
        sheet_readers.append(MasterReader(tool_kit.adjust_country_name(country, 'tb'), 'outcomes'))
    if 'mdr' in sheets_to_read:
        sheet_readers.append(MdrReportReader(country))
    if 'laboratories' in sheets_to_read:
        sheet_readers.append(MasterReader(country), 'laboratories')
    if 'strategy' in sheets_to_read:
        sheet_readers.append(MasterReader(country), 'strategy')
    if 'diabetes' in sheets_to_read: sheet_readers.append(DiabetesReportReader(country))

    # if being run from the directory above
    if from_test:
        for reader in sheet_readers: reader.filename = os.path.join('autumn/', reader.filename)

    # return data
    return read_xls_with_sheet_readers(sheet_readers)


