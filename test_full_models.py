

import os
import glob
import datetime
import autumn.model
import autumn.tool_kit
import autumn.plotting
from autumn.spreadsheet import read_input_data_xls
import autumn.write_outputs as w_o
import autumn.data_processing

# Start timer
start_realtime = datetime.datetime.now()

# Import the data
country = read_input_data_xls(True, ['control_panel'])['control_panel']['country']

inputs = autumn.data_processing.Inputs(True)
inputs.read_and_load_data()

print('Data have been loaded.')
print('Time elapsed so far is ' + str(datetime.datetime.now() - start_realtime) + '\n')

# A few basic preliminaries
out_dir = 'fullmodel_graphs'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

if inputs.model_constants['output_spreadsheets']:
    project = w_o.Project()
    project.country = country
    project.name = 'project_test'  # This name will be used as a directory to store all the output files

# At this point, I'm leaving the model attributes elements that follow as lists,
# as it may be useful iterate over several model structures in the future, although I'm not sure
# this will be needed.
n_strains = inputs.model_constants['n_strains'][0]
is_amplification = inputs.model_constants['is_amplification'][0]
is_misassignment = inputs.model_constants['is_misassignment'][0]
if (is_misassignment and not is_amplification) \
        or (inputs.model_constants['n_strains'][0] <= 1 and (is_amplification or is_misassignment)):
    pass
else:
    base = os.path.join(out_dir, country + '_baseline')

    models = {}
    for n, scenario in enumerate(inputs.model_constants['scenarios_to_run']):

        if scenario is None:
            model_name = 'baseline'
        else:
            model_name = 'scenario_' + str(scenario)

        if scenario == inputs.model_constants['scenarios_to_run'][-1]:
            final = True
        else:
            final = False

        if inputs.model_constants['output_spreadsheets']:
            project.scenarios.append(model_name)

        models[model_name] = autumn.model.ConsolidatedModel(
            inputs.model_constants['n_organs'][0],
            n_strains,
            inputs.model_constants['is_lowquality'][0],  # Low quality care
            is_amplification,  # Amplification
            is_misassignment,  # Misassignment by strain
            scenario,  # Scenario to run
            inputs)
        if n == 0:
            print(autumn.tool_kit.introduce_model(models, model_name))

        if scenario is not None:
            scenario_start_time_index = \
                models['baseline'].find_time_index(inputs.model_constants['scenario_start_time'])
            models[model_name].start_time = \
                models['baseline'].times[scenario_start_time_index]
            models[model_name].loaded_compartments = \
                models['baseline'].load_state(scenario_start_time_index)

        print('Running model "' + model_name + '".')
        if n == 0:
            print(autumn.tool_kit.describe_model(models, model_name))
        models[model_name].integrate()

        if inputs.model_constants['output_spreadsheets']:
            project.models[model_name] = \
                models[model_name]  # Store the model in the object 'project'
            project.output_dict[model_name] = \
                w_o.create_output_dict(models[model_name])  # Store simplified outputs

        print('Time elapsed so far is ' + str(datetime.datetime.now() - start_realtime))

        autumn.plotting.plot_outputs_against_gtb(
            models[model_name], ['incidence', 'mortality', 'prevalence', 'notifications'],
            inputs.model_constants['recent_time'],
            'scenario_end_time',
            base + '_outputs_gtb.png',
            country,
            scenario=scenario,
            figure_number=31,
            final_run=final)

        if inputs.model_constants['output_by_age']:
            autumn.plotting.plot_outputs_by_age(
                models[model_name],
                inputs.model_constants['recent_time'],
                'scenario_end_time',
                base + '_age_outputs_gtb.png',
                country,
                scenario=scenario,
                figure_number=21,
                final_run=final)

    # Make a flow-diagram
    if inputs.model_constants['output_flow_diagram']:
        models['baseline'].make_graph(base + '.workflow')

    # Plot over subgroups
    if inputs.model_constants['output_fractions']:
        subgroup_solns, subgroup_fractions = autumn.tool_kit.find_fractions(models['baseline'])
        for i, category in enumerate(subgroup_fractions):
            autumn.plotting.plot_fractions(
                models['baseline'],
                subgroup_fractions[category],
                models['baseline'].inputs.model_constants['recent_time'],
                'strain', base + '_fraction_' + category + '.png',
                figure_number=30+i)

    # Plot proportions of population
    if inputs.model_constants['output_comorbidity_fractions']:
        autumn.plotting.plot_stratified_populations(models['baseline'],
                                                    png=base + '_comorbidity_fraction.png',
                                                    age_or_comorbidity='comorbidity',
                                                    start_time='early_time')

    # Plot proportions of population
    if inputs.model_constants['output_age_fractions']:
        autumn.plotting.plot_stratified_populations(models['baseline'],
                                                    png=base + '_age_fraction.png',
                                                    age_or_comorbidity='age',
                                                    start_time='early_time')


    if inputs.model_constants['output_scaleups']:
        autumn.plotting.plot_classified_scaleups(models['baseline'], base)

pngs = glob.glob(os.path.join(out_dir, '*png'))
autumn.plotting.open_pngs(pngs)

if inputs.model_constants['output_spreadsheets']:
    project.write_output_dict_xls(horizontal=True, minimum=2015, maximum=2040, step=5)
    project.write_scenario_dict_word('incidence', minimum=2019, maximum=2040, step=5)

print('Time elapsed in running script is ' + str(datetime.datetime.now() - start_realtime))

