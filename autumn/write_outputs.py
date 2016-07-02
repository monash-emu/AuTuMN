
import os
import glob
import datetime
import autumn.model
import autumn.plotting
from autumn.spreadsheet import read_and_process_data, read_input_data_xls
import numpy as np
import openpyxl as xl
import base_analyses
from docx import Document

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def create_output_dict(model):

    """
    Create a dictionary with the main model outputs
    Args:
        model: a model instance after integration
    Returns: a dictionary with the different outputs
    """

    outputs = ['incidence', 'mortality', 'prevalence', 'notifications']
    output_dict = {}
    times = np.linspace(model.start_time, model.end_time, num=(1 + model.end_time - model.start_time))

    for label in outputs:
        output_dict[label] = {}
        solution = model.get_var_soln(label)
        for time in times:
            year_index = indices(model.times, lambda x: x >= time)[0]
            output_dict[label][int(round(time))] = solution[year_index]

    return output_dict

class Project():

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
        self.output_dict = {}

    def write_output_dict_xls(self, horizontal, minimum=None, maximum=None, step=None):

        out_dir_project = os.path.join('projects', self.name)
        if not os.path.isdir(out_dir_project):
            os.makedirs(out_dir_project)

        outputs = self.output_dict['baseline'].keys()

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

    def write_output_dict_word(self, minimum=None, maximum=None, step=None):

        # Sort out directory if not already sorted
        out_dir_project = os.path.join('projects', self.name)
        if not os.path.isdir(out_dir_project):
            os.makedirs(out_dir_project)

        # Find outputs
        outputs = self.output_dict['baseline'].keys()

        # Initialise document
        path = os.path.join(out_dir_project, 'results')
        path += ".docx"
        document = Document()
        table = document.add_table(rows=1, cols=4)

        # Write headers
        header_cells = table.rows[0].cells
        header_cells[0].text = 'Year'
        for output_no, output in enumerate(outputs):
            header_cells[output_no].text = base_analyses.capitalise_first_letter(output)

        # Find years to write
        years = self.find_years_to_write('incidence', minimum, maximum, step)
        for year in years:

            # Add row to table
            row_cells = table.add_row().cells

            # Write a new file for each epidemiological indicator
            for output_no, output in enumerate(outputs):
                row_cells[0].text = str(year)
                indicator = '%.2f' % self.output_dict['baseline'][output][year]
                row_cells[output_no].text = indicator

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
                base_analyses.replace_underscore_with_space(
                    base_analyses.capitalise_first_letter(sc))
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
                base_analyses.replace_underscore_with_space(
                    base_analyses.capitalise_first_letter(sc))
            row = 1
            for y in years:
                row += 1
                if y in self.output_dict[sc][output]:
                    sheet.cell(row=row, column=col).value = self.output_dict[sc][output][y]


if __name__ == "__main__":

    scenario = None
    country = read_input_data_xls(False, ['attributes'])['attributes'][u'country']
    print(country)
    data = read_and_process_data(country,
                                 keys_of_sheets_to_read=['bcg', 'rate_birth', 'life_expectancy', 'attributes', 'parameters',
                                  'country_constants', 'time_variants', 'tb', 'notifications', 'outcomes'],
                                 from_test=False)

    is_additional_diagnostics = data['attributes']['is_additional_diagnostics'][0]
    n_organs = data['attributes']['n_organs'][0]
    n_strains = data['attributes']['n_strains'][0]
    n_comorbidities = data['attributes']['n_comorbidities'][0]
    is_quality = data['attributes']['is_lowquality'][0]
    is_amplification = data['attributes']['is_amplification'][0]
    is_misassignment = data['attributes']['is_misassignment'][0]
    model = autumn.model.ConsolidatedModel(
        n_organs,
        n_strains,
        n_comorbidities,
        is_quality,  # Low quality care
        is_amplification,  # Amplification
        is_misassignment,  # Misassignment by strain
        is_additional_diagnostics,
        scenario,  # Scenario to run
        data)

    for key, value in data['parameters'].items():
        model.set_parameter(key, value)
    for key, value in data['country_constants'].items():
        model.set_parameter(key, value)

    model.integrate()

    dict = create_output_dict(model)

    print(dict)


