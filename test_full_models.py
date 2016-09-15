

import datetime
import autumn.model
import autumn.tool_kit
import autumn.outputs as outputs
import autumn.data_processing
import os

# Start timer
start_realtime = datetime.datetime.now()

# Import data
inputs = autumn.data_processing.Inputs(True)
inputs.read_and_load_data()

print('Data have been loaded.')
print('Time elapsed so far is ' + str(datetime.datetime.now() - start_realtime) + '\n')

project = outputs.Project(inputs.country, inputs)

models = {}
for scenario in inputs.model_constants['scenarios_to_run']:

    # Name and initialise model
    scenario_name = autumn.tool_kit.find_scenario_string_from_number(scenario)
    models[scenario_name] = autumn.model.ConsolidatedModel(scenario, inputs)

    # Create an outputs object for use later
    project.scenarios.append(scenario_name)

    # Introduce model at first run
    autumn.tool_kit.introduce_model(models, scenario_name)

    # Sort out times for scenario runs
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

    # Integrate and add result to outputs object
    models[scenario_name].integrate()
    print('Time elapsed to completion of integration is ' + str(datetime.datetime.now() - start_realtime))
    project.models[scenario_name] = models[scenario_name]

#   ______ Run uncertainty ____
if inputs.model_constants['output_uncertainty']:
    # Prepare directory for eventual pickling
    out_dir = 'pickles'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    filename = os.path.join(out_dir, 'uncertainty.pkl')
    if project.models['baseline'].pickle_uncertainty == 'read':    # we don't run uncertainty but load a saved simulation
        project.models['baseline'].uncertainty_results = autumn.tool_kit.pickle_load(filename)
        print "Uncertainty results loaded from previous simulation"
    else:   # we need to run uncertainty
        project.models['baseline'].run_uncertainty()
    if project.models['baseline'].pickle_uncertainty == 'write':
        autumn.tool_kit.pickle_save(project.models['baseline'].uncertainty_results, filename)
        print "Uncertainty results written on the disk"

    project.rearrange_uncertainty()

# Work through outputs
project.master_outputs_runner()
print('Time elapsed in running script is ' + str(datetime.datetime.now() - start_realtime))