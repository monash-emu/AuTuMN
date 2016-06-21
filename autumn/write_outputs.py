import os
import glob
import datetime
import autumn.model
import autumn.plotting
from autumn.spreadsheet import read_and_process_data, read_input_data_xls
import numpy as np
import openpyxl as xl

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
            indice_year = indices(model.times, lambda x: x >= time)[0]
            output_dict[label][time] = solution[indice_year]

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

        # for sc in self.scenarios:
        #     self.output_dict[sc] = create_output_dict(self.models[sc])

    def write_output_dict_xls(self):
        out_dir_project = os.path.join('projects', self.name)
        if not os.path.isdir(out_dir_project):
            os.makedirs(out_dir_project)

        scenario_name = {None: 'baseline'}
        for i in range(10)[1:-1]:
            scenario_name[i] = 'scenario_' + str(i)

        outputs = self.output_dict[self.scenarios[0]].keys()

        for output in outputs: # write a new file
            path = os.path.join(out_dir_project, output)
            path += ".xlsx"

            wb = xl.Workbook()
            sheet = wb.active
            sheet.title = 'model_outputs'

            sheet.cell(row=1, column=1).value = 'year'
            years = self.output_dict[self.scenarios[0]][output].keys()
            col = 1
            for y in years:
                col += 1
                sheet.cell(row=1, column=col).value = y

            r = 1
            for sc in self.scenarios:
                r += 1
                sheet.cell(row=r, column=1).value = scenario_name[sc]
                col = 1
                for y in years:
                    col += 1
                    sheet.cell(row=r, column=col).value = self.output_dict[sc][output][y]

            wb.save(path)



if __name__ == "__main__":

    scenario = None
    country = read_input_data_xls(False, ['attributes'])['attributes'][u'country']
    print(country)
    data = read_and_process_data(False,
                                 ['bcg', 'rate_birth', 'life_expectancy', 'attributes', 'parameters',
                                  'country_constants', 'time_variants', 'tb', 'notifications', 'outcomes'],
                                 country)

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


