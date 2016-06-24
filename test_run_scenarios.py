

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
country = read_input_data_xls(True, ['attributes'])['attributes'][u'country']
data = read_and_process_data(True,
                             ['bcg', 'rate_birth', 'life_expectancy', 'attributes', 'parameters',
                              'country_constants', 'time_variants', 'tb', 'notifications', 'outcomes'],
                             country)

# A few basic preliminaries
out_dir = 'fullmodel_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

is_additional_diagnostics = data['attributes']['is_additional_diagnostics'][0]

project = w_o.Project()
project.country = country
project.name = 'project_test' # this name will be used as a directory to store all the output files

# Note that it takes about one hour to run all of the possible model structures,
# so perhaps don't do that - and longer if running multiple scenarios
for scenario in data['attributes']['scenarios_to_run'] + [None]:
    project.scenarios.append(scenario)

    n_organs = data['attributes']['n_organs'][0]
    n_strains =  data['attributes']['n_strains'][0]
    n_comorbidities = data['attributes']['n_comorbidities'][0]
    is_quality = data['attributes']['is_lowquality'][0]
    is_amplification = data['attributes']['is_amplification'][0]
    is_misassignment = data['attributes']['is_misassignment'][0]
    if (is_misassignment and not is_amplification) \
            or (n_strains <= 1 and (is_amplification or is_misassignment)):
        pass
    else:
        name = 'model%d' % n_organs
        base = os.path.join(out_dir, name)

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
            data['attributes']['recent_time'],
            'scenario_end_time',
            base + '.rate_outputs_gtb_recent.png',
            country,
            scenario=scenario,
            figure_number=1)

        # autumn.plotting.plot_outputs_against_gtb(
        #     model, ["incidence", "mortality", "prevalence", "notifications"],
        #     data['country_constants']['start_time'],
        #     'scenario_end_time',
        #     base + '.rate_outputs_gtb_start.png',
        #     country,
        #     scenario=scenario,
        #     figure_number=11)

        # Classify scale-up functions for plotting
        classified_scaleups = {'program_prop': [],
                               'program_other': [],
                               'birth': [],
                               'non_program': []}
        for fn in model.scaleup_fns:
            if 'program_prop' in fn:
                classified_scaleups['program_prop'] += [fn]
            elif 'program' in fn:
                classified_scaleups['program_other'] += [fn]
            elif 'demo_rate_birth' in fn:
                classified_scaleups['birth'] += [fn]
            else:
                classified_scaleups['non_program'] += [fn]

        # Plot them from the start of the model and from "recent_time"
        # Enumerations are to make sure each plot goes on a separate figure
        # and to tell the loop to come back to the relevant figure at each iteration
        for i, classification in enumerate(classified_scaleups):
            if len(classified_scaleups[classification]) > 0:
                for j, start_time in enumerate(['start_', 'recent_']):
                    autumn.plotting.plot_all_scaleup_fns_against_data(model,
                                                                      classified_scaleups[classification],
                                                                      base + '.' + classification + '_scaleups_' + start_time + 'time.png',
                                                                      start_time + 'time',
                                                                      'scenario_end_time',
                                                                      classification,
                                                                      country,
                                                                      scenario=scenario,
                                                                      figure_number= i + j*5 + 2)
                    if classification == 'program_prop':
                        autumn.plotting.plot_scaleup_fns(model,
                                                         classified_scaleups[classification],
                                                         base + '.' + classification + 'scaleups_' + start_time + '.png',
                                                         start_time + 'time',
                                                         'scenario_end_time',
                                                         classification,
                                                         country,
                                                         figure_number= i + j*4 + 12)

pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)


project.write_output_dict_xls()

print("Time elapsed in running script is " + str(datetime.datetime.now() - start_realtime))

