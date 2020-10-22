import pandas as pd
import yaml
from apps.covid_19.mixing_optimisation.mixing_opti import (
    MODES, CONFIGS, OBJECTIVES, N_DECISION_VARS, build_params_for_phases_2_and_3
)
from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS

"""
Create scenarios for each combination of mode, config and objective
"""
SCENARIO_MAPPING = {
}
_sc_idx = 1
for _mode in MODES:
    for _config in CONFIGS:
        for _objective in OBJECTIVES:
            SCENARIO_MAPPING[_sc_idx] = {
                'mode': _mode,
                'config': _config,
                'objective': _objective,
            }
            _sc_idx += 1
SCENARIO_MAPPING[_sc_idx + 1] = {'mode': None, 'config': None, 'objective': None}  # extra scenario for unmitigated

"""
Reading optimisation outputs from csv file
"""


def read_opti_outputs():
    df = pd.read_csv("opti_outputs.csv")
    return df


def read_decision_vars(opti_outputs_df, country, mode, config, objective):
    mask = (opti_outputs_df['country'] == country) &\
           (opti_outputs_df['mode'] == mode) &\
           (opti_outputs_df['config'] == config) &\
           (opti_outputs_df['objective'] == objective)
    df = opti_outputs_df[mask]

    return [float(df[f"best_x{i}"]) for i in range(N_DECISION_VARS[mode])]


"""
Automatically write yml files for scenarios
"""


def drop_yml_scenario_file(country, sc_idx, decision_vars, final_mixing=1.):
    country_folder_name = country.replace("-", "_")
    # read settings associated with scenario sc_idx
    if sc_idx == max(list(SCENARIO_MAPPING.keys())):  # this is the unmitigated scenario
        config = CONFIGS[0]  # does not matter but needs to be defined
        mode = MODES[0]
    else:  # this is an optimised scenario
        config = SCENARIO_MAPPING[_sc_idx]['config']
        mode = SCENARIO_MAPPING[_sc_idx]['mode']

    sc_params = build_params_for_phases_2_and_3(decision_vars, config, mode, final_mixing)
    sc_params['parent'] = f"apps/covid_19/regions/{country_folder_name}/params/default.yml"

    param_file_path = f"../params/regions/{country_folder_name}/params/scenario-{sc_idx}.yml"
    with open(param_file_path, "w") as f:
        yaml.dump(sc_params, f)


def write_all_scenario_yml_files_from_outputs():
    opti_outputs_df = read_opti_outputs()

    for country in OPTI_REGIONS:
        for sc_idx, settings in SCENARIO_MAPPING.items():
            if sc_idx == max(list(SCENARIO_MAPPING.keys())):  # this is the unmitigated scenario
                decision_vars = [1.] * N_DECISION_VARS[MODES[0]]
            else:  # this is an optimised scenario
                decision_vars = read_decision_vars(
                    opti_outputs_df, country, settings['mode'], settings['config'], settings['objective']
                )
            drop_yml_scenario_file(country, sc_idx, decision_vars)


