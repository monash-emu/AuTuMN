
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

    def __init__(self):

        """
        Initialises an object of class Project, that will contain all the information (data + outputs) for writing a
        report for a country
        Args:
            models: dictionary such as: models = {'baseline': model, 'scenario_1': model_1,  ...}
        """

        self.country = ''
        self.name = 'project_test'
        self.scenarios = []
        self.models = {}
        self.full_output_dict = {}
        self.integer_output_dict = {}

    def write_output_dict_xls(self, model_name, horizontal, minimum=None, maximum=None, step=None):

        out_dir_project = os.path.join('projects', self.name)
        if not os.path.isdir(out_dir_project):
            os.makedirs(out_dir_project)

        outputs = self.output_dict[model_name].keys()

        # Write a new file for each epidemiological indicator
        for output in outputs:

            # Get filename
            path = os.path.join(out_dir_project, output)
            path += ".xlsx"

            # Get active sheet
            wb = xl.Workbook()
            sheet = wb.active
            sheet.title = 'model_outputs'

            # Find years to write
            years = self.find_years_to_write(output, minimum, maximum, step)

            # Write data
            if horizontal:
                self.write_horizontally(sheet, output, years)
            else:
                self.write_vertically(sheet, output, years)

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

    def make_path(self, filename):

        # Sort out directory if not already sorted
        out_dir_project = os.path.join(filename, self.name)
        if not os.path.isdir(out_dir_project):
            os.makedirs(out_dir_project)
        return out_dir_project

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

    def find_years_to_write(self, output, minimum, maximum, step):

        # Determine years requested
        if minimum is None:
            minimum = 0
        if maximum is None:
            maximum = 3000
        if step is None:
            requested_years = range(minimum, maximum)
        else:
            requested_years = range(minimum, maximum, step)

        # Find years to write
        years = []
        for y in self.output_dict['baseline'][output].keys():
            if y in requested_years:
                years += [y]
        return years

    def write_horizontally(self, sheet, output, years):

        sheet.cell(row=1, column=1).value = 'Year'

        col = 1
        for y in years:
            col += 1
            sheet.cell(row=1, column=col).value = y

        r = 1
        for sc in self.scenarios:
            r += 1
            sheet.cell(row=r, column=1).value = \
                tool_kit.replace_underscore_with_space(
                    tool_kit.capitalise_first_letter(sc))
            col = 1
            for y in years:
                col += 1
                if y in self.output_dict[sc][output]:
                    sheet.cell(row=r, column=col).value = self.output_dict[sc][output][y]

    def write_vertically(self, sheet, output, years):

        sheet.cell(row=1, column=1).value = 'Year'

        row = 1
        for y in years:
            row += 1
            sheet.cell(row=row, column=1).value = y

        col = 1
        for sc in self.scenarios:
            col += 1
            sheet.cell(row=1, column=col).value = \
                tool_kit.replace_underscore_with_space(
                    tool_kit.capitalise_first_letter(sc))
            row = 1
            for y in years:
                row += 1
                if y in self.output_dict[sc][output]:
                    sheet.cell(row=row, column=col).value = self.output_dict[sc][output][y]

    def create_output_dict(self, model_name):

        """
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

    def create_output_dicts(self, outputs=['incidence', 'mortality', 'prevalence', 'notifications']):

        self.create_full_output_dict(outputs)
        self.extract_integer_dict(outputs)

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

    def extract_integer_dict(self, outputs):

        """
        Extracts a dictionary from full_output_dict with only integer years, using the first time value greater than
        the integer year in question.

        Args:
            outputs: Model outputs that were previously extracted (should be the same as for create_full_output_dict)
        """

        for scenario in self.scenarios:
            self.integer_output_dict[scenario] = {}
            for output in outputs:
                self.integer_output_dict[scenario][output] = {}
                times = self.full_output_dict[scenario][output].keys()
                times.sort()
                start = np.floor(times[0])
                finish = np.floor(times[-1])
                float_years = np.linspace(start, finish, finish - start + 1.)
                for year in float_years:
                    key = [t for t in times if t >= year][0]
                    self.integer_output_dict[scenario][output][int(key)] \
                        = self.full_output_dict[scenario][output][key]

    def add_economics_outputs_to_dict(self, model_name):

        econ_outputs = ['discounted_inflated_cost']
        times = np.linspace(self.models[model_name].inputs.model_constants['recent_time'],
                            self.models[model_name].inputs.model_constants['scenario_end_time'],
                            num=(1 + self.models[model_name].inputs.model_constants['scenario_end_time'] \
                                 - self.models[model_name].inputs.model_constants['recent_time']))

