

import os
import glob
import datetime
import autumn.model
import autumn.base_analyses
import autumn.plotting
from autumn.spreadsheet import read_and_process_data, read_input_data_xls

# Start timer
start_realtime = datetime.datetime.now()

# Import the data
country = read_input_data_xls(True, ['control_panel'])['control_panel']['country']
inputs = read_and_process_data(country, from_test=True)

# A few basic preliminaries
out_dir = 'fullmodel_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# At this point, I'm leaving the model attributes elements that follow as lists,
# as it may be useful iterate over several model structures in the future, although I'm not sure
# this will be needed.
n_organs = inputs['model_constants']['n_organs'][0]
n_strains = inputs['model_constants']['n_strains'][0]
n_comorbidities = inputs['model_constants']['n_comorbidities'][0]
is_quality = inputs['model_constants']['is_lowquality'][0]
is_amplification = inputs['model_constants']['is_amplification'][0]
is_misassignment = inputs['model_constants']['is_misassignment'][0]
if (is_misassignment and not is_amplification) \
        or (n_strains <= 1 and (is_amplification or is_misassignment)):
    pass
else:
    base = os.path.join(out_dir, country + '_baseline')

    models = {}
    for n, scenario in enumerate(inputs['model_constants']['scenarios_to_run']):
        if scenario is None:
            model_name = 'baseline'
        else:
            model_name = 'scenario_' + str(scenario)

        if scenario == inputs['model_constants']['scenarios_to_run'][-1]:
            final = True
        else:
            final = False

        models[model_name] = autumn.model.ConsolidatedModel(
            n_organs,
            n_strains,
            n_comorbidities,
            is_quality,  # Low quality care
            is_amplification,  # Amplification
            is_misassignment,  # Misassignment by strain
            scenario,  # Scenario to run
            inputs)
        if n == 0:
            print(autumn.base_analyses.introduce_model(models, model_name))

        if scenario is not None:
            scenario_start_time_index = models['baseline'].find_time_index(inputs['model_constants']['scenario_start_time'])
            models[model_name].start_time = models['baseline'].times[scenario_start_time_index]
            models[model_name].loaded_compartments = models['baseline'].load_state(scenario_start_time_index)

        print('Running model "' + model_name + '".')
        if n == 0:
            print(autumn.base_analyses.describe_model(models, model_name))
        models[model_name].integrate()

        print("Time elapsed so far is " + str(datetime.datetime.now() - start_realtime))

        autumn.plotting.plot_outputs_against_gtb(
            models[model_name], ["incidence", "mortality", "prevalence", "notifications"],
            inputs['model_constants']['recent_time'],
            'scenario_end_time',
            base + '_rate_outputs_gtb.png',
            country,
            scenario=scenario,
            figure_number=11,
            final_run=final)

    # Only make a flow-diagram if the model isn't overly complex
    if n_organs + n_strains + n_comorbidities <= 5:
        models['baseline'].make_graph(base + '.workflow')

    # Plot over subgroups
    subgroup_solns, subgroup_fractions = autumn.base_analyses.find_fractions(models['baseline'])
    for i, category in enumerate(subgroup_fractions):
        autumn.plotting.plot_fractions(
            models['baseline'], subgroup_fractions[category], models['baseline'].inputs['model_constants']['recent_time'],
            'strain', base + '_fraction_' + category + '.png', figure_number=30+i)

    autumn.plotting.plot_classified_scaleups(models['baseline'], base)

    subgroup_solns, subgroup_fractions = \
        autumn.base_analyses.calculate_additional_diagnostics(models['baseline'])


    # if n_strains >= 2:
    #     autumn.plotting.plot_outputs(
    #         models['baseline'], ["incidence_ds", "incidence_mdr", "mortality_ds", "mortality_mdr", "prevalence_ds", "prevalence_mdr",
    #                 "notifications_ds", "notifications_mdr"],
    #         data['model_constants']['recent_time'], base + '.rate_bystrain_outputs_recent.png')
    #     autumn.plotting.plot_outputs(
    #         models['baseline'], ["incidence_ds", "incidence_mdr", "mortality_ds", "mortality_mdr", "prevalence_ds", "prevalence_mdr",
    #                 "notifications_ds", "notifications_mdr"],
    #         start_time, base + '.rate_outputs.png')
    #     autumn.plotting.plot_outputs(
    #         models['baseline'], ["incidence_mdr", "prevalence_mdr", "mortality_mdr"],
    #         data['model_constants']['start_time'], base + '.mdr_outputs.png')
    #     autumn.plotting.plot_outputs(
    #         models['baseline'], ["incidence_mdr", "prevalence_mdr", "mortality_mdr"],
    #         data['model_constants']['recent_time'], base + '.mdr_outputs_recent.png')
    #     autumn.plotting.plot_outputs(
    #         models['baseline'], ["proportion_mdr"],
    #         data['model_constants']['start_time'], base + '.mdr_proportion_recent.png')

pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)

print("Time elapsed in running script is " + str(datetime.datetime.now() - start_realtime))


