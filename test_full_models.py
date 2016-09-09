
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

    # Create Boolean for uncertainty for this run to save re-typing the multi-factorial condition statement
    uncertainty_this_run = False
    if (inputs.model_constants['output_uncertainty'] and scenario is None) \
            or inputs.model_constants['output_uncertainty_all_scenarios']:

        # Generally only run uncertainty if this on the baseline scenario, unless specified otherwise
        uncertainty_this_run = True

    if scenario is None:
        models[scenario_name].start_time = inputs.model_constants['start_time']
    else:
        scenario_start_time_index = \
            models['baseline'].find_time_index(inputs.model_constants['recent_time'])
        models[scenario_name].start_time = \
            models['baseline'].times[scenario_start_time_index]
        models[scenario_name].loaded_compartments = \
            models['baseline'].load_state(scenario_start_time_index)

        for count_run in range(len(models['baseline'].model_shelf)):
            new_model = autumn.model.ConsolidatedModel(scenario, inputs)
            new_model.start_time = models['baseline'].model_shelf[count_run].times[scenario_start_time_index]
            new_model.loaded_compartments = models['baseline'].model_shelf[count_run].load_state(scenario_start_time_index)
            new_model.integrate()
            models[scenario_name].model_shelf.append(new_model)

    # Describe model
    print('Running model "' + scenario_name + '".')
    autumn.tool_kit.describe_model(models, scenario_name)

    if uncertainty_this_run:
        models[scenario_name].run_uncertainty()

    # Integrate the rigid parameterization in any case. Indeed, we still need the central estimates for uncertainty
    models[scenario_name].integrate()

    print('Time elapsed to completion of integration is ' + str(datetime.datetime.now() - start_realtime))

    if inputs.model_constants['output_by_age'] and scenario == inputs.model_constants['scenarios_to_run'][-1]:
        autumn.outputs.plot_outputs_by_age(
            models[scenario_name],
            inputs.model_constants['recent_time'],
            'scenario_end_time',
            base + '_age_outputs_gtb.png',
            inputs.country,
            scenario=scenario,
            figure_number=21)

    project.models[scenario_name] = []
    if inputs.model_constants['output_uncertainty']:
        project.model_shelf_uncertainty[scenario_name] = models[scenario_name].model_shelf
    project.models[scenario_name] = models[scenario_name]

# Write to spreadsheets
project.create_output_dicts()  # Store simplified outputs

# Make a flow-diagram
if inputs.model_constants['output_flow_diagram']:
    models['baseline'].make_graph(base + '.workflow')

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
project.plot_intervention_costs_by_scenario(2016, 2025)
#project.plot_intervention_costs_by_scenario(2016, 2025, plot_options= {'interventions': ['xpert', 'treatment_support', 'smearacf', 'xpertacf', 'ipt_age0to5', 'ipt_age5to15']})

autumn.outputs.open_pngs(pngs)

print('Time elapsed in running script is ' + str(datetime.datetime.now() - start_realtime))