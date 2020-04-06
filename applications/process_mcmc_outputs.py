from autumn.tb_model import load_calibration_from_db, create_mcmc_outputs

PATH_TO_MCMC_DATABASES = 'marshall_islands/mcmc_outputs/first_calibration_3_4_2020'
N_BURNED_ITERATIONS = 0

scenario_params = {
}
scenario_list = list(scenario_params.keys())
if 0 not in scenario_list:
    scenario_list = [0] + scenario_list

models = load_calibration_from_db(PATH_TO_MCMC_DATABASES, n_burned_per_chain=N_BURNED_ITERATIONS)
scenario_list = [i + 1 for i in range(len(models))]

req_outputs = []

targets_to_plot = {}

for target in targets_to_plot.keys():
    if target not in req_outputs and target[0:5] == "prevX":
        req_outputs.append(target)

multipliers = {}

ymax = {}

translations = {}

create_mcmc_outputs(
    models,
    req_outputs=req_outputs,
    out_dir="mcmc_outputs",
    targets_to_plot=targets_to_plot,
    req_multipliers=multipliers,
    translation_dictionary=translations,
    scenario_list=scenario_list,
    ymax=ymax,
    plot_start_time=1940,
    # outputs_to_plot_by_stratum=PLOTTED_STRATIFIED_PREVALENCE_OUTPUTS,
    )
