
import os
import glob
import datetime
import autumn.model
import autumn.plotting
from autumn.spreadsheet import read_input_data_xls
import numpy as np
import openpyxl as xl
import tool_kit
from docx import Document


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


class Project:

    def __init__(self, country, inputs):

        """
        Initialises an object of class Project, that will contain all the information (data + outputs) for writing a
        report for a country
        Args:
            models: dictionary such as: models = {'baseline': model, 'scenario_1': model_1,  ...}
        """

        self.country = country.lower()
        self.name = 'test_' + self.country
        self.scenarios = []
        self.models = {}
        self.full_output_dict = {}
        self.integer_output_dict = {}
        self.inputs = inputs

    #################################
    # General methods for use below #
    #################################

    def find_or_make_directory(self):

        out_dir_project = os.path.join('projects', self.name)
        if not os.path.isdir(out_dir_project):
            os.makedirs(out_dir_project)
        return out_dir_project

    def make_path(self, filename):

        # Sort out directory if not already sorted
        out_dir_project = os.path.join(filename, self.name)
        if not os.path.isdir(out_dir_project):
            os.makedirs(out_dir_project)
        return out_dir_project

    #########################################
    # Methods to collect data for later use #
    #########################################

    def create_output_dicts(self, outputs=['incidence', 'mortality', 'prevalence', 'notifications']):

        """
        Works through all the methods to this object that are required to populate the output dictionaries.
        First the "full" ones with all time point included, then the abbreviated ones.

        Args:
            outputs: The outputs to be populated to the dictionaries
        """

        self.create_full_output_dict(outputs)
        self.add_full_economics_dict()
        self.extract_integer_dict()

    def create_full_output_dict(self, outputs):

        """
        Creates a dictionary for each requested output at every time point in that model's times attribute
        """

        for scenario in self.scenarios:
            self.full_output_dict[scenario] = {}
            for label in outputs:
                times = self.models[scenario].times
                solution = self.models[scenario].get_var_soln(label)
                self.full_output_dict[scenario][label] = dict(zip(times, solution))

    def add_full_economics_dict(self):

        """
        Creates an economics dictionary structure that mirrors that of the epi dictionaries and adds
        this to the main outputs (epi) dictionary
        """

        for model in self.models:
            economics_dict = {}
            for intervention in self.models[model].costs:
                if intervention != 'cost_times':
                    economics_dict['cost_' + intervention] = {}
                    for t in range(len(self.models[model].costs['cost_times'])):
                        economics_dict['cost_' + intervention][self.models[model].costs['cost_times'][t]] \
                            = self.models[model].costs[intervention]['raw_cost'][t]
            self.full_output_dict[model].update(economics_dict)

    def extract_integer_dict(self):

        """
        Extracts a dictionary from full_output_dict with only integer years, using the first time value greater than
        the integer year in question.
        """

        for model in self.models:
            self.integer_output_dict[model] = {}
            for output in self.full_output_dict[model]:
                self.integer_output_dict[model][output] = {}
                times = self.full_output_dict[model][output].keys()
                times.sort()
                start = np.floor(times[0])
                finish = np.floor(times[-1])
                float_years = np.linspace(start, finish, finish - start + 1.)
                for year in float_years:
                    key = [t for t in times if t >= year][0]
                    self.integer_output_dict[model][output][int(key)] \
                        = self.full_output_dict[model][output][key]

    #################################################
    # Methods for outputting to Office applications #
    #################################################

    def write_spreadsheets(self):

        """
        Determine whether to write to spreadsheets by scenario or by output
        """

        if self.inputs.model_constants['output_spreadsheets']:
            if self.inputs.model_constants['output_by_scenario']:
                self.write_xls_by_scenario(horizontal=True, minimum=2015, maximum=2040, step=5)
            else:
                self.write_xls_by_output(horizontal=True, minimum=2015, maximum=2040, step=5)

    def write_xls_by_output(self, horizontal, minimum=None, maximum=None, step=None):

        # Find directory to write to
        out_dir_project = self.find_or_make_directory()

        # Write a new file for each output
        outputs = self.integer_output_dict['baseline'].keys()
        for output in outputs:

            # Write a new file for each epidemiological indicator
            for scenario in self.integer_output_dict.keys():

                # Make filename
                path = os.path.join(out_dir_project, output)
                path += ".xlsx"

                # Get active sheet
                wb = xl.Workbook()
                sheet = wb.active
                sheet.title = output

                # Find years to write
                years = self.find_years_to_write(scenario, output, minimum, maximum, step)

                # Write data
                if horizontal:
                    self.write_horizontally_by_scenario(sheet, output, years)
                else:
                    self.write_vertically_by_scenario(sheet, output, years)

            # Save workbook
            wb.save(path)

    def write_xls_by_scenario(self, horizontal, minimum=None, maximum=None, step=None):

        # Find directory to write to
        out_dir_project = self.find_or_make_directory()

        # Write a new file for each scenario
        scenarios = self.integer_output_dict.keys()
        for scenario in scenarios:

            # Make filename
            path = os.path.join(out_dir_project, scenario)
            path += '.xlsx'

            # Get active sheet
            wb = xl.Workbook()
            sheet = wb.active
            sheet.title = scenario

            for output in self.integer_output_dict['baseline'].keys():

                # Find years to write
                years = self.find_years_to_write(scenario, output, minimum, maximum, step)

                # Write data
                if horizontal:
                    self.write_horizontally_by_output(sheet, scenario, years)
                else:
                    self.write_vertically_by_output(sheet, scenario, years)

            # Save workbook
            wb.save(path)

    def write_scenario_dict_word(self,
                               indicator='incidence',
                               minimum=None, maximum=None, step=None):

        # Sort out directory if not already sorted
        out_dir_project = self.make_path('projects')

        # Initialise document
        path = os.path.join(out_dir_project, 'scenarios')
        path += ".docx"
        document = Document()
        table = document.add_table(rows=1, cols=len(self.scenarios)+1)

        # Write headers
        header_cells = table.rows[0].cells
        header_cells[0].text = 'Year'
        for scenario_no, scenario in enumerate(self.scenarios):
            header_cells[scenario_no+1].text = tool_kit.capitalise_first_letter(scenario)

        # Find years to write
        years = self.find_years_to_write(indicator, minimum, maximum, step)
        for year in years:

            # Add row to table
            row_cells = table.add_row().cells

            # Write a new file for each epidemiological indicator
            for scenario_no, scenario in enumerate(self.scenarios):
                row_cells[0].text = str(year)
                text_to_write = '%.2f' % self.output_dict[scenario][indicator][year]
                row_cells[scenario_no+1].text = text_to_write

        # Save document
        document.save(path)

    def write_output_dict_word(self,
                               indicators_to_tabulate=['incidence', 'prevalence', 'mortality', 'notifications'],
                               minimum=None, maximum=None, step=None):

        # Sort out directory if not already sorted
        out_dir_project = self.make_path('projects')

        # Find outputs
        outputs = []
        for output in self.output_dict['baseline'].keys():
            if output in indicators_to_tabulate:
                outputs += [output]

        # Initialise document
        path = os.path.join(out_dir_project, 'results')
        path += ".docx"
        document = Document()
        table = document.add_table(rows=1, cols=len(outputs)+1)

        # Write headers
        header_cells = table.rows[0].cells
        header_cells[0].text = 'Year'
        for output_no, output in enumerate(outputs):
            header_cells[output_no+1].text = tool_kit.capitalise_first_letter(output)

        # Find years to write
        years = self.find_years_to_write(outputs[0], minimum, maximum, step)
        for year in years:

            # Add row to table
            row_cells = table.add_row().cells

            # Write a new file for each epidemiological indicator
            for output_no, output in enumerate(outputs):
                row_cells[0].text = str(year)
                indicator = '%.2f' % self.output_dict['baseline'][output][year]
                row_cells[output_no+1].text = indicator

        # Save document
        document.save(path)

    def find_years_to_write(self, scenario, output, minimum=0, maximum=3000, step=1):

        requested_years = range(minimum, maximum, step)
        years = []
        for y in self.integer_output_dict[scenario][output].keys():
            if y in requested_years:
                years += [y]
        return years

    def write_horizontally_by_scenario(self, sheet, output, years):

        sheet.cell(row=1, column=1).value = 'Year'

        col = 1
        for y in years:
            col += 1
            sheet.cell(row=1, column=col).value = y

        r = 1
        for scenario in self.scenarios:
            r += 1
            sheet.cell(row=r, column=1).value = \
                tool_kit.replace_underscore_with_space(
                    tool_kit.capitalise_first_letter(scenario))
            col = 1
            for y in years:
                col += 1
                if y in self.integer_output_dict[scenario][output]:
                    sheet.cell(row=r, column=col).value \
                        = self.integer_output_dict[scenario][output][y]

    def write_horizontally_by_output(self, sheet, scenario, years):

        sheet.cell(row=1, column=1).value = 'Year'

        col = 1
        for y in years:
            col += 1
            sheet.cell(row=1, column=col).value = y

        r = 1
        for output in self.integer_output_dict['baseline'].keys():
            r += 1
            sheet.cell(row=r, column=1).value = \
                tool_kit.replace_underscore_with_space(
                    tool_kit.capitalise_first_letter(output))
            col = 1
            for y in years:
                col += 1
                if y in self.integer_output_dict[scenario][output]:
                    sheet.cell(row=r, column=col).value \
                        = self.integer_output_dict[scenario][output][y]

    def write_vertically_by_scenario(self, sheet, output, years):

        sheet.cell(row=1, column=1).value = 'Year'

        row = 1
        for y in years:
            row += 1
            sheet.cell(row=row, column=1).value = y

        col = 1
        for scenario in self.scenarios:
            col += 1
            sheet.cell(row=1, column=col).value = \
                tool_kit.replace_underscore_with_space(
                    tool_kit.capitalise_first_letter(scenario))
            row = 1
            for y in years:
                row += 1
                if y in self.integer_output_dict[scenario][output]:
                    sheet.cell(row=row, column=col).value = self.integer_output_dict[scenario][output][y]

    def write_vertically_by_output(self, sheet, scenario, years):

        sheet.cell(row=1, column=1).value = 'Year'

        row = 1
        for y in years:
            row += 1
            sheet.cell(row=row, column=1).value = y

        col = 1
        for output in self.integer_output_dict['baseline'].keys():
            col += 1
            sheet.cell(row=1, column=col).value = \
                tool_kit.replace_underscore_with_space(
                    tool_kit.capitalise_first_letter(output))
            row = 1
            for y in years:
                row += 1
                if y in self.integer_output_dict[scenario][output]:
                    sheet.cell(row=row, column=col).value \
                        = self.integer_output_dict[scenario][output][y]

    def create_output_dict(self, model_name):

        """
        This the old version that I'm trying to make obsolete

        Create a dictionary with the main model outputs

        Args:
            model_name: a model instance after integration
        Returns:
            output_dict: a dictionary with the different outputs
        """


        outputs = ['incidence', 'mortality', 'prevalence', 'notifications']
        self.output_dict[model_name] = {}
        times = np.linspace(self.models[model_name].inputs.model_constants['start_time'],
                            self.models[model_name].inputs.model_constants['scenario_end_time'],
                            num=(1 + self.models[model_name].inputs.model_constants['scenario_end_time'] \
                                 - self.models[model_name].inputs.model_constants['start_time']))

        for label in outputs:
            self.output_dict[model_name][label] = {}
            solution = self.models[model_name].get_var_soln(label)
            for time in times:
                year_index = indices(self.models[model_name].times, lambda x: x >= time)[0]
                self.output_dict[model_name][label][int(round(time))] = solution[year_index]




