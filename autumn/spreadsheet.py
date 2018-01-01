

from xlrd import open_workbook
from numpy import nan
import numpy
import os
import tool_kit


''' static functions '''


def parse_year_data(year_data, blank, end_column):
    """
    Code to parse rows of data that are years.

    Args:
        year_data: The row to parse
        blank: A value for blanks to be ignored
        end_column: Column to end at
    """

    year_data = tool_kit.replace_specified_value(year_data, nan, blank)
    assumption_val, year_vals = year_data[-1], year_data[:end_column]
    if tool_kit.is_all_same_value(year_vals, nan):
        return [assumption_val]
    else:
        return year_vals


def read_input_data_xls(from_test, sheets_to_read, country):
    """
    Compile sheet readers into a list according to which ones have been selected.

    Args:
        from_test: Whether being called from the directory above
        sheets_to_read: A list containing the strings that are also the 'keys' attribute of the reader
        country: Country being read
    Returns:
        A single data structure containing all the data to be read (by calling the read_xls_with_sheet_readers method)
    """

    # sheets that have been checked and should definitely work being read in
    available_sheets \
        = ['default_constants', 'country_constants', 'default_programs', 'country_programs', 'bcg_2014', 'bcg_2015',
           'bcg_2016', 'rate_birth_2014', 'rate_birth_2015', 'life_expectancy_2014', 'life_expectancy_2015',
           'notifications_2014', 'notifications_2015', 'notifications_2016', 'outcomes_2013', 'outcomes_2015',
           'mdr_2014', 'mdr_2015', 'mdr_2016', 'laboratories_2014', 'laboratories_2015', 'laboratories_2016',
           'strategy_2014', 'strategy_2015', 'strategy_2016', 'diabetes', 'gtb_2015', 'gtb_2016', 'latent_2016',
           'tb_hiv_2016']

    # compile list of sheets to read and read them
    final_sheets_to_read = tool_kit.find_common_elements(sheets_to_read, available_sheets)
    data_read_from_sheets = {}
    for sheet_name in final_sheets_to_read:
        sheet_reader = SpreadsheetReader(country, sheet_name, from_test)
        data_read_from_sheets[sheet_reader.revised_purpose] = sheet_reader.read_data()

    # update 2016 data with 2015 data where available
    for gtb_key in data_read_from_sheets['gtb_2016']:
        if gtb_key in data_read_from_sheets['gtb']:
            data_read_from_sheets['gtb'][gtb_key].update(data_read_from_sheets['gtb_2016'][gtb_key])
        else:
            data_read_from_sheets['gtb'][gtb_key] = data_read_from_sheets['gtb_2016'][gtb_key]

    return data_read_from_sheets


''' spreadsheet reader object '''


class SpreadsheetReader:
    """
    The master spreadsheet reader that now subsumes all of the previous readers (which had been structured as one
    sheet reader per spreadsheet to be read. Now the consistently required data is indexed from dictionaries in
    instantiation before the row or column reading methods are called as required with if/elif statements to use the
    appropriate reading approach.
    """

    def __init__(self, country_to_read, purpose, from_test):
        """
        Use the "purpose" input to index static dictionaries to set basic reader characteristics.

        Args:
            country_to_read: The adapted country name
            purpose: String that defines the spreadsheet type to be read
        """

        self.indices, self.parlist, self.dictionary_keys, self.data, self.year_indices, country_adjustment_types, \
            self.data_read_from_sheets, self.purpose, vertical_sheets, tb_adjustment_countries \
            = [], [], [], {}, {}, {}, {}, purpose, [], []

        # set basic sheet characteristics, either directly or in short loops by years
        specific_sheet_names \
            = {'default_constants': 'xls/data_default.xlsx',
               'country_constants': 'xls/data_' + country_to_read.lower() + '.xlsx',
               'default_programs': 'xls/data_default.xlsx',
               'country_programs': 'xls/data_' + country_to_read.lower() + '.xlsx'}
        specific_tab_names \
            = {'default_constants': 'constants',
               'country_constants': 'constants',
               'default_programs': 'time_variants',
               'country_programs': 'time_variants'}
        sheets_starting_row_one \
            = ['default_constants', 'country_constants', 'notifications_2015', 'outcomes_2013']
        start_rows \
            = {'diabetes': 2,
               'rate_birth_2015': 3}
        start_cols \
            = {'default_programs': 1,
               'country_programs': 1,
               'rate_birth_2015': 5}
        columns_for_keys \
            = {'rate_birth_2014': 2}
        first_cells \
            = {'rate_birth_2014': 'Series Name',
               'rate_birth_2015': 'Country Name',
               'default_constants': 'program',
               'country_constants': 'program',
               'default_programs': 'program',
               'country_programs': 'program',
               'diabetes': u'Country/territory'}
        for year in range(2015, 2017):
            gtb_string = 'gtb_' + str(year)
            sheets_starting_row_one.append(gtb_string)
            first_cells.update({gtb_string: 'country'})
            vertical_sheets.append(gtb_string)
            tb_adjustment_countries.append(gtb_string)
        for year in range(2014, 2017):
            laboratories_string = 'laboratories_' + str(year)
            sheets_starting_row_one.append(laboratories_string)
            vertical_sheets.append(laboratories_string)
            bcg_string = 'bcg_' + str(year)
            start_cols.update({bcg_string: 4})
            columns_for_keys.update({bcg_string: 2})
            first_cells.update({bcg_string: 'WHO_REGION'})
            notifications_string = 'notifications_' + str(year)
            vertical_sheets.append(notifications_string)
            tb_adjustment_countries.append(notifications_string)
        for year in range(2014, 2016):
            life_expectancy_string = 'life_expectancy_' + str(year)
            start_rows.update({life_expectancy_string: 3})
            first_cells.update({life_expectancy_string: 'Country Name'})
            rate_birth_string = 'rate_birth_' + str(year)
            country_adjustment_types.update({life_expectancy_string: 'demographic'})
            country_adjustment_types.update({rate_birth_string: 'demographic'})
        for year in range(2013, 2016, 2):
            outcomes_string = 'outcomes_' + str(year)
            vertical_sheets.append(outcomes_string)
            tb_adjustment_countries.append(outcomes_string)

        if purpose in country_adjustment_types:
            country_adjustment = country_adjustment_types[purpose]
        elif purpose in tb_adjustment_countries:
            country_adjustment = 'tb'
        else:
            country_adjustment = ''
        self.country_to_read = tool_kit.adjust_country_name(country_to_read, country_adjustment)
        filename = specific_sheet_names[purpose] if purpose in specific_sheet_names else 'xls/' + purpose + '.xlsx'
        self.filename = os.path.join('autumn/', filename) if from_test else filename  # if from directory above
        self.tab_name = specific_tab_names[purpose] if purpose in specific_tab_names else purpose
        if purpose in sheets_starting_row_one:
            self.start_row = 1
        elif purpose in start_rows:
            self.start_row = start_rows[purpose]
        else:
            self.start_row = 0
        self.start_col = start_cols[purpose] if purpose in start_cols else 0
        self.column_for_keys = columns_for_keys[purpose] if purpose in columns_for_keys else 0
        self.first_cell = first_cells[purpose] if purpose in first_cells else 'country'
        self.horizontal = False if self.purpose in vertical_sheets else True
        self.revised_purpose = self.purpose[:-5] \
            if self.purpose[-5:-2] == '_20' and self.purpose != 'gtb_2016' else self.purpose

    def read_data(self):
        """
        Read the data from the spreadsheet being read.
        """

        # check that the spreadsheet to be read exists
        try:
            print('Reading file ' + self.filename)
            workbook = open_workbook(self.filename)

        # if sheet unavailable, warn of issue
        except:
            print('Unable to open spreadsheet ' + self.filename)
            return

        # read the sheet according to reading orientation
        else:
            return self.read_data_by_line(workbook)

    def read_data_by_line(self, workbook):
        """
        Short function to determine whether to read horizontally or vertically.

        Arg:
            workbook: The Excel workbook for interrogation
        """

        sheet = workbook.sheet_by_name(self.tab_name)
        if self.horizontal:
            for i_row in range(self.start_row, sheet.nrows): self.parse_row(sheet.row_values(i_row))
        else:
            for i_col in range(self.start_col, sheet.ncols): self.parse_col(sheet.col_values(i_col))
        return self.data

    def parse_row(self, row):
        """
        Method to read rows of the spreadsheets for sheets that read horizontally. Several different spreadsheet readers
        use this method, so there is a conditional loop to determine which approach to use according to the sheet name.
        Several of these approaches involve first determining the dictionary keys (as a list) from the first row
        (identified by first_cell) and then reading values in for each key.

        Args:
            row: The row of data being interpreted
        """

        # vaccination sheet
        if self.purpose == 'bcg_2016':
            if row[0] == self.first_cell:
                self.parlist = parse_year_data(row, '', len(row))
                for i in range(len(self.parlist)): self.parlist[i] = str(self.parlist[i])
            elif row[self.column_for_keys] == self.country_to_read:
                for i in range(self.start_col, len(row)):
                    if type(row[i]) == float: self.data[int(self.parlist[i])] = row[i]

        # demographics
        elif self.purpose in ['rate_birth_2014', 'rate_birth_2015', 'life_expectancy_2014', 'life_expectancy_2015']:
            if row[0] == self.first_cell:
                for i in range(len(row)): self.parlist.append(row[i][:4])
            elif row[self.column_for_keys] == self.country_to_read:
                for i in range(self.start_row, len(row)):
                    if type(row[i]) == float: self.data[int(self.parlist[i])] = row[i]

        # constants
        elif self.purpose in ['default_constants', 'country_constants']:

            # age breakpoints
            if row[0] == 'age_breakpoints':
                self.data[str(row[0])] = []
                for i in range(1, len(row)):
                    if row[i]: self.data[str(row[0])].append(int(row[i]))

            # empty entries
            elif row[1] == '':
                pass

            # country or integration method
            elif row[0] in ['country', 'integration']:
                self.data[str(row[0])] = str(row[1])

            # model strata or fitting method
            elif row[0][:2] == 'n_' or 'fitting' in row[0]:
                self.data[str(row[0])] = int(row[1])

            # for the calendar year times
            elif 'time' in row[0] or 'smoothness' in row[0]:
                self.data[str(row[0])] = float(row[1])

            # all instructions around outputs, plotting and spreadsheet/document writing
            elif 'output_' in row[0] or row[0][:3] == 'is_' or row[0][:12] == 'comorbidity_':
                self.data[str(row[0])] = bool(row[1])

            # parameter values
            else:
                self.data[str(row[0])] = row[1]

            # uncertainty parameters, which must have an entry present in the third column
            # no idea why this if statement needs to be split and written like this, but huge bugs occur if it isn't
            if len(row) > 3:
                if row[2] != '':
                    self.data[str(row[0]) + '_uncertainty'] = {'point': row[1], 'lower': row[2], 'upper': row[3]}

        # time-variant programs and interventions
        elif self.purpose in ['default_programs', 'country_programs']:
            if row[0] == self.first_cell:
                self.parlist = parse_year_data(row, '', len(row))
            else:
                self.data[str(row[0])] = {}
                for i in range(self.start_col, len(row)):
                    parlist_item_string = str(self.parlist[i])
                    if ('19' in parlist_item_string or '20' in parlist_item_string) and row[i] != '':
                        self.data[row[0]][int(self.parlist[i])] = row[i]
                    elif row[i] != '':
                        self.data[str(row[0])][str(self.parlist[i])] = row[i]

        # diabetes
        elif self.purpose == 'diabetes':
            if row[0] == self.first_cell:
                self.dictionary_keys = row
            elif row[0] == self.country_to_read:
                for i in range(len(self.dictionary_keys)):
                    if self.dictionary_keys[i][:28] == u'Diabetes national prevalence':
                        self.data['comorb_prop_diabetes'] = float(row[i][:row[i].find('\n')]) / 1e2

        # other sheets, such as strategy and mdr
        else:
            if row[0] == self.first_cell:
                self.dictionary_keys = row
            elif row[0] == self.country_to_read:
                for i in range(len(self.dictionary_keys)): self.data[self.dictionary_keys[i]] = row[i]

    def parse_col(self, col):
        """
        Read columns of data. Note that all the vertically oriented spreadsheets currently run off the same reading
        method, so there is not need for an outer if/elif statement to determine which approach to use - this is likely
        to change.

        Args:
            col: The column of data being interpreted
        """

        col = tool_kit.replace_specified_value(col, nan, '')

        # country column (the first one)
        if col[0] == self.first_cell:

            # find the indices for the country
            for i in range(len(col)):
                if col[i] == self.country_to_read: self.indices.append(i)

        # skip some irrelevant columns
        elif 'iso' in col[0] or 'g_who' in col[0] or 'source' in col[0]:
            pass

        # year column
        elif col[0] == 'year':
            self.year_indices = {int(col[i]): i for i in self.indices}

        # all other columns
        else:
            self.data[str(col[0])] = {}
            for year in self.year_indices:
                if not numpy.isnan(col[self.year_indices[year]]): self.data[col[0]][year] = col[self.year_indices[year]]

