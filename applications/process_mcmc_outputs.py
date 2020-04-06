from autumn.tb_model import load_calibration_from_db, create_mcmc_outputs
import os
import yaml
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

APPLICATION = 'marshall_islands'
PATH_TO_CALIBRATION_FILE = 'marshall_islands/rmi_calibration.py'
PATH_TO_MCMC_DATABASES = 'marshall_islands/mcmc_outputs/first_calibration_3_4_2020'
N_BURNED_ITERATIONS = 280
from applications.marshall_islands.rmi_calibration import TARGET_OUTPUTS


outputs_path = os.path.join(FILE_DIR, APPLICATION, 'outputs.yml')
with open(outputs_path, 'r') as yaml_file:
    output_options = yaml.safe_load(yaml_file)

scenario_params = {
}
scenario_list = list(scenario_params.keys())
if 0 not in scenario_list:
    scenario_list = [0] + scenario_list

models = load_calibration_from_db(PATH_TO_MCMC_DATABASES, n_burned_per_chain=N_BURNED_ITERATIONS)
scenario_list = [i + 1 for i in range(len(models))]

req_outputs = output_options['req_outputs']
targets_to_plot = output_options['targets_to_plot']

# automatically add some targets based on calibration targets
for calib_target in TARGET_OUTPUTS:
    if 'cis' in calib_target:
        targets_to_plot[calib_target['output_key']] = {
            "times": calib_target['years'], "values":[[calib_target['values'][i], calib_target['cis'][i][0], calib_target['cis'][i][1]] for i in range(len(calib_target['values']))]
        }
    else:
        targets_to_plot[calib_target['output_key']] = {
            "times": calib_target['years'], "values":[[calib_target['values'][i]] for i in range(len(calib_target['values']))]
        }

 #  {'output_key': 'prevXinfectiousXamongXlocation_ebeye', 'years': [2017.0], 'values': [755.0], 'cis': [(620.0, 894.0)]}

# "prevXinfectiousXamong": {"times": [2015], "values": [[757.0, 620.0, 894.0]]},


for target in targets_to_plot.keys():
    if target not in req_outputs and target[0:5] == "prevX":
        req_outputs.append(target)

multipliers = output_options['req_multipliers']

ymax = output_options['ymax']

translations = output_options['translation_dictionary']

create_mcmc_outputs(
    models,
    req_outputs=req_outputs,
    out_dir=PATH_TO_MCMC_DATABASES,
    targets_to_plot=targets_to_plot,
    req_multipliers=multipliers,
    translation_dictionary=translations,
    scenario_list=scenario_list,
    ymax=ymax,
    plot_start_time=1940,
    # outputs_to_plot_by_stratum=PLOTTED_STRATIFIED_PREVALENCE_OUTPUTS,
    )
