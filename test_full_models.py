

import datetime
import autumn.model_runner

# Start timer
start_realtime = datetime.datetime.now()

# Import data
inputs = autumn.data_processing.Inputs(True)

print('Data have been loaded.')
print('Time elapsed so far is ' + str(datetime.datetime.now() - start_realtime) + '\n')

model_runner = autumn.model_runner.ModelRunnerNew(inputs)
model_runner.run_scenarios()
model_runner.project.master_outputs_runner()
model_runner.run_uncertainty()

print('Time elapsed in running script is ' + str(datetime.datetime.now() - start_realtime))

