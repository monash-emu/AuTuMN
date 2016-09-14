
import os
import glob
import datetime
import autumn.model
import autumn.tool_kit
import autumn.outputs as outputs
import autumn.data_processing

# Start timer
start_realtime = datetime.datetime.now()

# Import the data
inputs = autumn.data_processing.Inputs(True)
inputs.read_and_load_data()

print('Data have been loaded.')
print('Time elapsed so far is ' + str(datetime.datetime.now() - start_realtime) + '\n')

# A few basic preliminaries - which will be disposed of once outputs are fully object-oriented
out_dir = 'fullmodel_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

out_dir_pickles = 'pickles'
if not os.path.isdir(out_dir_pickles):
    os.makedirs(out_dir_pickles)

project = outputs.Project(inputs.country, inputs)
base = os.path.join(out_dir, inputs.country + '_baseline')

models = {}
for scenario in inputs.model_constants['scenarios_to_run']:

    # Name and initialise model
    scenario_name = autumn.tool_kit.find_scenario_string_from_number(scenario)
    models[scenario_name] = autumn.model.ConsolidatedModel(scenario, inputs)

    # Create an outputs object for use later
    project.scenarios.append(scenario_name)

    # Introduce model at first run
    autumn.tool_kit.introduce_model(models, scenario_name)

    if scenario is None:
        models[scenario_name].start_time = inputs.model_constants['start_time']
    else:
        scenario_start_time_index = \
            models['baseline'].find_time_index(inputs.model_constants['recent_time'])
        models[scenario_name].start_time = \
            models['baseline'].times[scenario_start_time_index]
        models[scenario_name].loaded_compartments = \
            models['baseline'].load_state(scenario_start_time_index)

    # Describe model
    print('Running model "' + scenario_name + '".')
    autumn.tool_kit.describe_model(models, scenario_name)

    models[scenario_name].integrate()

    print('Time elapsed to completion of integration is ' + str(datetime.datetime.now() - start_realtime))

    project.models[scenario_name] = models[scenario_name]

#   ______ Run uncertainty ____
if inputs.model_constants['output_uncertainty']:
    project.models['baseline'].run_uncertainty()

# Write to spreadsheets
project.prepare_for_outputs()  # Store simplified outputs

# Plot proportions of population
if inputs.model_constants['output_comorbidity_fractions']:
    autumn.outputs.plot_stratified_populations(models['baseline'],
                                               png=base + '_comorbidity_fraction.png',
                                               age_or_comorbidity='comorbidity',
                                               start_time='early_time')

# Plot proportions of population
if inputs.model_constants['output_age_fractions']:
    autumn.outputs.plot_stratified_populations(models['baseline'],
                                               png=base + '_age_fraction.png',
                                               age_or_comorbidity='age',
                                               start_time='early_time')

pngs = glob.glob(os.path.join(out_dir, '*png'))

project.write_spreadsheets()
project.write_documents()
project.run_plotting()

# Added to test total cost plotting - need install of Pandas library for DataFrames (Eike)
# project.plot_intervention_costs_by_scenario(2016, 2025)
#project.plot_intervention_costs_by_scenario(2016, 2025, plot_options= {'interventions': ['xpert', 'treatment_support', 'smearacf', 'xpertacf', 'ipt_age0to5', 'ipt_age5to15']})

autumn.outputs.open_pngs(pngs)

print('Time elapsed in running script is ' + str(datetime.datetime.now() - start_realtime))