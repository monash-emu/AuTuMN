
import datetime
import autumn.model_runner
import autumn.outputs

# Start timer
start_realtime = datetime.datetime.now()

# Run everything
model_runner = autumn.model_runner.ModelRunner()
model_runner.run_scenarios()
model_runner.master_uncertainty()
project = autumn.outputs.Project(model_runner)
project.master_outputs_runner()

print('Time elapsed in running script is ' + str(datetime.datetime.now() - start_realtime))

