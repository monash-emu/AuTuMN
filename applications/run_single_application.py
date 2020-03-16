"""
Build and run any AuTuMN model, storing the outputs
"""

import os
from datetime import datetime
import yaml

from summer_py.constants import IntegrationType
import summer_py.post_processing as post_proc
from autumn.outputs.outputs import Outputs, OutputPlotter, collate_compartment_across_stratification, collate_prevalence

from autumn.tool_kit.timer import Timer
from autumn.tool_kit import run_multi_scenario
from autumn.tool_kit.utils import make_directory_if_absent, record_parameter_request, record_run_metadata
from autumn.tb_model import store_run_models
from autumn import constants
from autumn.demography.ageing import add_agegroup_breaks

from applications.marshall_islands.rmi_model import build_rmi_model
from applications.covid_19.covid_model import build_covid_model


FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# Settings for the hand-coded Euler method.
# It's not clear whether this produces reliable results, but is likely faster than odeint
# (~60s at 0.1 step, ~20s at 0.3 step)
EULER_KWARGS = {
    'integration_type': IntegrationType.EULER,
    'solver_args': {'step_size': 0.3},
}
# Settings for the hand-coded Runge-Kutta method, which is more accurate, but slower than Euler.
# It's not clear whether this produces reliable results, but it can be faster than odeint
# (~230s at 0.1 step, ~70s at 0.3 step, ~50s at 0.5 step)
RUNGE_KUTTA_KWARGS = {
    'integration_type': IntegrationType.RUNGE_KUTTA,
    'solver_args': {'step_size': 0.5},
}
# Settings for the SciPy odeint solver - this can get stuck on some ODE types and take a long time (~230s).
ODEINT_KWARGS = {
    'integration_type': IntegrationType.ODE_INT,
}
# ODE solver settings to use when running the model.
SOLVER_KWARGS = ODEINT_KWARGS


def run_model(application):

    # Load user information for parameters and outputs from YAML files
    params_path = os.path.join(FILE_DIR, application, 'params.yml')
    outputs_path = os.path.join(FILE_DIR, application, 'outputs.yml')
    with open(params_path, 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)
    with open(outputs_path, 'r') as yaml_file:
        output_options = yaml.safe_load(yaml_file)

    # If agegroup breaks specified in default, add these to the agegroup stratification
    params = add_agegroup_breaks(params)

    # Run the model
    if application == 'marshall_islands':
        model_function = build_rmi_model
    elif application == 'covid_19':
        model_function = build_covid_model

    output_options = collate_prevalence(output_options)

    for i_combination in range(len(output_options['output_combinations_to_collate'])):
        output_options = \
            collate_compartment_across_stratification(
                output_options,
                output_options['output_combinations_to_collate'][i_combination][0],
                output_options['output_combinations_to_collate'][i_combination][1],
                params['default']['all_stratifications'][output_options['output_combinations_to_collate'][i_combination][1]]
            )

    # Ensure project folder exists
    project_dir = os.path.join(constants.DATA_PATH, application)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir, exist_ok=True)

    # Include user input if requested
    run_name, run_description = 'manual-calibration', ''

    # Create output data folder
    timestamp = datetime.now().strftime('%d-%m-%Y--%H-%M-%S')
    output_directory = os.path.join(project_dir, f'{run_name}-{timestamp}')
    make_directory_if_absent(output_directory, run_name, timestamp)

    # Determine where to save model outputs
    output_db_path = os.path.join(output_directory, 'outputs.db')
    plot_path = os.path.join(output_directory, 'plots')

    # Save parameter requests and metadata
    record_parameter_request(output_directory, params)
    record_run_metadata(output_directory, run_name, run_description, timestamp)

    # Prepare scenario data
    scenario_params = params['scenarios']
    scenario_list = [0, *scenario_params.keys()]

    with Timer('Running model scenarios'):
        models = run_multi_scenario(
            scenario_params, params['scenario_start'], model_function, run_kwargs=SOLVER_KWARGS,
        )

    # Post-process and save model outputs
    with Timer('Processing model outputs'):
        store_run_models(models, scenarios=scenario_list, database_name=output_db_path)
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)

        pps = []
        for scenario_index in range(len(models)):

            # Automatically add some basic outputs
            if hasattr(models[scenario_index], 'all_stratifications'):
                for group in models[scenario_index].all_stratifications.keys():
                    output_options['req_outputs'].append(
                        'distribution_of_strataX' + group)

            pps.append(
                post_proc.PostProcessing(
                    models[scenario_index],
                    requested_outputs=output_options['req_outputs'],
                    scenario_number=scenario_list[scenario_index],
                    requested_times={},
                    multipliers=output_options['req_multipliers'],
                    ymax=output_options['ymax'],
                )
            )

    with Timer('Creating model outputs'):
        outputs = Outputs(
            models,
            pps,
            output_options,
            output_options['targets_to_plot'],
            plot_path,
            output_options['translation_dictionary'],
            plot_start_time=output_options['plot_start_time']
        )

        # outputs.plot_requested_outputs()
        # for output in output_options['outputs_to_plot_by_stratum']:
        #     for sc_index in range(len(models)):
        #         outputs.plot_outputs_by_stratum(output, sc_index=sc_index)

        # New approach to plotting outputs, intended to be more general
        outputs_plotter = OutputPlotter(models, pps, output_options, plot_path)
        outputs_plotter.run_input_plots()

if __name__ == '__main__':
    run_model('covid_19')
