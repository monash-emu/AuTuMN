

import datetime
import autumn.model
import autumn.tool_kit
import autumn.outputs as outputs
import autumn.data_processing

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
    project.models['baseline'].run_uncertainty()

# Work through outputs
project.master_outputs_runner()
print('Time elapsed in running script is ' + str(datetime.datetime.now() - start_realtime))