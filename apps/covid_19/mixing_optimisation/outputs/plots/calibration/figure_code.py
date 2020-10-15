from autumn.tool_kit.params import load_targets
from autumn import db
from autumn.plots.calibration.plots import get_posterior

from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS
import matplotlib.pyplot as plt
from numpy import random, mean, quantile


# --------------  Load outputs form databases
def get_calibration_outputs(country):
    calib_dirpath = f"../../pbi_databases/calibration/{country}/"
    mcmc_tables = db.load.load_mcmc_tables(calib_dirpath)
    mcmc_params = db.load.load_mcmc_params_tables(calib_dirpath)

    return mcmc_tables, mcmc_params


def get_all_countries_calibration_outputs():
    calibration_outputs = {}
    for country in OPTI_REGIONS:
        mcmc_tables, mcmc_params = get_calibration_outputs(country)
        calibration_outputs[country] = {
            "mcmc_tables": mcmc_tables, "mcmc_params": mcmc_params
        }

    return calibration_outputs


def get_targets(region_name):
    return load_targets("covid_19", region_name)


# --------------  Make figure with posterior estimates

param_info = {
    'contact_rate': {'name': 'transmission prob.', 'range': [0.02, 0.06]},
    'time.start': {'name': 'model start time', 'range': [0., 40.]},
    'sojourn.compartment_periods_calculated.exposed.total_period': {'name': 'incubation time', 'range': [3., 7.]},
    'sojourn.compartment_periods_calculated.active.total_period': {'name': 'time infectious', 'range': [4., 10.]},
    'infection_fatality.multiplier': {'name': 'IFR multiplier', 'range': [.8, 1.2]},
    'case_detection.maximum_gradient': {'name': 'detection (shape)', 'range': [.05, .1]},
    'case_detection.max_change_time': {'name': 'detection (inflection)', 'range': [100., 250.]},
    'case_detection.end_value': {'name': 'detection (prop_final)', 'range': [.10, .90]},
    'case_detection.start_value': {'name': 'detection (prop_start)', 'range': [0., .10]},
    'icu_prop': {'name': 'prop ICU among hosp.', 'range': [.15, .20]},
    'compartment_periods.hospital_late': {'name': 'hopital duration', 'range': [17.7, 20.4]},
    'compartment_periods.icu_late': {'name': 'time in ICU', 'range': [9., 13.]},
    'clinical_stratification.props.hospital.multiplier': {'name': 'hosp. prop. multiplier', 'range': [.75, 1.25]},
    'symptomatic_props_multiplier': {'name': 'sympt. prop. multiplier', 'range': [.6, 1.4]},
    'mobility.microdistancing.behaviour.parameters.c': {'name': 'microdist. (inflection)', 'range': [80, 130]},
    'mobility.microdistancing.behaviour.parameters.upper_asymptote': {'name': 'microdist. (final)', 'range': [.25, .6]},
    'mobility.microdistancing.behaviour_adjuster.parameters.c': {'name': 'microdist. wane (inflection)', 'range': [150, 250]},
    'mobility.microdistancing.behaviour_adjuster.parameters.sigma': {'name': 'microdist. wane (final)', 'range': [.5, 1]},
}


def make_posterior_ranges_figure(calibration_outputs):
    param_values = {}
    for country, outputs in calibration_outputs.items():
        if len(outputs["mcmc_params"]) == 0:
            continue
        param_values[country] = {}
        all_param_names = [p for p in outputs["mcmc_params"][0]["name"].unique().tolist() if 'dispersion_param' not in p]
        for param_name in all_param_names:
            param_values[country][param_name] = get_posterior(
                outputs["mcmc_params"], outputs["mcmc_tables"], param_name, 0
            )[param_name].tolist()
    plot_param_posteriors(param_values)


def plot_param_posteriors(param_values):
    n_panels = len(param_values['belgium'])
    country_list = OPTI_REGIONS[::-1]
    n_col = 4
    n_row = int(n_panels // n_col)
    if n_col * n_row < n_panels:
        n_row += 1

    fig, axs = plt.subplots(n_row, n_col, figsize=(11, 12))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.4)

    i_col = -1
    i_row = 0
    for param_name in list(param_values['belgium'].keys()):
        i_col += 1
        if i_col >= n_col:
            i_col = 0
            i_row += 1
        h = 0
        for country in country_list:
            h += 1
            # find mean and CI
            values = param_values[country][param_name]
            point_estimate = mean(values)
            low_95, low_50, med, up_50, up_95 = quantile(values, q=(.025, .25, .5, .75, .975))

            axs[i_row, i_col].plot([low_95, up_95], [h, h], linewidth=1, color='black')
            axs[i_row, i_col].plot([low_50, up_50], [h, h], linewidth=3, color='steelblue')
            axs[i_row, i_col].plot([point_estimate], [h], marker='o', color='crimson', markersize=5)

        axs[i_row, i_col].plot([0], [0])
        axs[i_row, i_col].set_ylim((0.5, 6.5))

        axs[i_row, i_col].set_title(param_info[param_name]['name'], fontsize=10.5)
        axs[i_row, i_col].set_xlim(param_info[param_name]['range'])

        # Format x-ticks if requested
        if "xticks" in param_info[param_name]:
            axs[i_row, i_col].set_xticks(param_info[param_name]['xticks'])
            axs[i_row, i_col].set_xticklabels(param_info[param_name]['xlabels'])

        if "multiplier" in param_info[param_name]:
            axs[i_row, i_col].set_xlabel(param_info[param_name]["multiplier"], labelpad=-7.5)

        # Set y-ticks and yticks-labels
        if i_col == 0:
            axs[i_row, i_col].set_yticks([1, 2, 3, 4, 5, 6])
            axs[i_row, i_col].set_yticklabels([c.title().replace("United-Kingdom", "UK") for c in country_list])
        else:
            axs[i_row, i_col].set_yticks([])

    # Leave blank axis for remaining panels
    for i_col_blank in range(i_col + 1, n_col):
        axs[i_row, i_col_blank].axis("off")

    plt.tight_layout()
    plt.savefig("figures/param_posteriors.pdf")


# --------------  Make figure with posterior estimates
