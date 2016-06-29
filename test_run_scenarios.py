

import os
import glob
import datetime
import autumn.model
import autumn.plotting
from autumn.spreadsheet import read_and_process_data, read_input_data_xls
import autumn.write_outputs as w_o

# Start timer
start_realtime = datetime.datetime.now()

# Import the data
country = read_input_data_xls(True, ['control_panel'])['control_panel']['country']
data = read_and_process_data(True,
                             ['bcg', 'rate_birth', 'life_expectancy', 'control_panel',
                              'default_parameters',
                              'tb', 'notifications', 'outcomes',
                              'country_constants', 'default_constants',
                              'country_economics', 'default_economics',
                              'country_programs', 'default_programs'],
                             country)

# A few basic preliminaries
out_dir = 'fullmodel_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

project = w_o.Project()
project.country = country
project.name = 'project_test' # this name will be used as a directory to store all the output files

# Note that it takes about one hour to run all of the possible model structures,
# so perhaps don't do that - and longer if running multiple scenarios
for scenario in data['model_constants']['scenarios_to_run']:
    project.scenarios.append(scenario)

    n_organs = data['model_constants']['n_organs'][0]
    n_strains =  data['model_constants']['n_strains'][0]
    n_comorbidities = data['model_constants']['n_comorbidities'][0]
    is_quality = data['model_constants']['is_lowquality'][0]
    is_amplification = data['model_constants']['is_amplification'][0]
    is_misassignment = data['model_constants']['is_misassignment'][0]
    if (is_misassignment and not is_amplification) \
            or (n_strains <= 1 and (is_amplification or is_misassignment)):
        pass
    else:
        base = os.path.join(out_dir, country + '_scenarios')

        model = autumn.model.ConsolidatedModel(
            n_organs,
            n_strains,
            n_comorbidities,
            is_quality,  # Low quality care
            is_amplification,  # Amplification
            is_misassignment,  # Misassignment by strain
            scenario,  # Scenario to run
            data)
        print(str(n_organs) + " organ(s),   " +
              str(n_strains) + " strain(s),   " +
              str(n_comorbidities) + " comorbidity(ies),   " +
              "Low quality? " + str(is_quality) + ",   " +
              "Amplification? " + str(is_amplification) + ",   " +
              "Misassignment? " + str(is_misassignment) + ".")

        model.integrate()

        project.models[scenario] = model # Store the model in the object 'project'
        project.output_dict[scenario] = w_o.create_output_dict(model) # store simplified outputs

        # Only make a flow-diagram if the model isn't overly complex
        if n_organs + n_strains + n_comorbidities <= 5:
            model.make_graph(base + '.workflow')

        autumn.plotting.plot_outputs_against_gtb(
            model, ["incidence", "mortality", "prevalence", "notifications"],
            data['model_constants']['recent_time'],
            'scenario_end_time',
            base + '.rate_outputs_gtb_recent.png',
            country,
            scenario=scenario,
            figure_number=1)

        autumn.plotting.plot_classified_scaleups(model, base)

pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)


project.write_output_dict_xls()

print("Time elapsed in running script is " + str(datetime.datetime.now() - start_realtime))

