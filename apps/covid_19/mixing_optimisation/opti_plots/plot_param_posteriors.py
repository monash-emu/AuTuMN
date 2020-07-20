import pandas as pd
from matplotlib import pyplot as plt

import os
from autumn.db.database import Database
from autumn.tool_kit.uncertainty import export_compartment_size, collect_iteration_weights, collect_all_mcmc_output_tables
from numpy import random, mean, quantile

from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS
from apps.covid_19.mixing_optimisation.utils import get_list_of_ifr_priors_from_pollan


param_info = {
    'contact_rate': {'name': 'contact rate', 'range': [0., 40.]},
    'start_time': {'name': 'model start time', 'range': [0., 40.]},
    'npi_effectiveness.other_locations': {'name': 'alpha', 'range': [0.5, 1.]},
    'compartment_periods_calculated.incubation.total_period': {'name': 'incubation time', 'range': [3., 7.]},
    'compartment_periods_calculated.total_infectious.total_period"': {'name': 'time infectious', 'range': [5., 10.]},
    'tv_detection_b': {'name': 'detection (shape)', 'range': [.05, .1]},
    'tv_detection_c': {'name': 'detection (inflection)', 'range': [70., 110.]},
    'prop_detected_among_symptomatic': {'name': 'detection (prop_final)', 'range': [.10, .90]},
    'icu_prop': {'name': 'prop ICU among hosp.', 'range': [.10, .30]},
    'compartment_periods.hospital_late': {'name': 'hopital duration', 'range': [4., 12.]},
    'compartment_periods.icu_late': {'name': 'time in ICU', 'range': [4., 12.]},
    'hospital_props_multiplier': {'name': 'hosp. prop. multiplier', 'range': [.5, 2.0]},
    'microdistancing.parameters.c': {'name': 'microdistancing (inflection)', 'range': [90, 152]},
    'microdistancing.parameters.sigma': {'name': 'microdistancing (final)', 'range': [.6, 1.]},
}

pollan_priors = get_list_of_ifr_priors_from_pollan()
for i in range(9):
    age_index = pollan_priors[i]["param_name"].split("(")[1].split(")")[0]
    age_group = str(10*int(age_index)) + "-" + str(10*int(age_index) + 9)
    if age_index == "8":
        age_group = "80+"
    param_info[pollan_priors[i]["param_name"]] = {'name': 'IFR age ' + age_group,
                                                  'range': pollan_priors[i]["distri_params"],
                                                  'xticks': pollan_priors[i]["distri_params"],
                                                  'xlabels': ["{:.2e}".format(b).split("e")[0] for b in pollan_priors[i]["distri_params"]],
                                                  'multiplier': "(.e" + "{:.2e}".format(pollan_priors[i]["distri_params"][0]).split("e")[1] + ")"
                                                  },
    if age_index == '0':
        param_info[pollan_priors[i]["param_name"]][0]['multiplier'] = "(.e-06)"
        param_info[pollan_priors[i]["param_name"]][0]['xlabels'][1] = \
            str(10. * float(param_info[pollan_priors[i]["param_name"]][0]['xlabels'][1]))

    if age_index == '2':
        param_info[pollan_priors[i]["param_name"]][0]['multiplier'] = "(.e-05)"
        param_info[pollan_priors[i]["param_name"]][0]['xlabels'][1] = \
            str(10. * float(param_info[pollan_priors[i]["param_name"]][0]['xlabels'][1]))

    if age_index == '8':
        param_info[pollan_priors[i]["param_name"]][0]['multiplier'] = "(.e-02)"
        param_info[pollan_priors[i]["param_name"]][0]['xlabels'][1] = \
            str(10. * float(param_info[pollan_priors[i]["param_name"]][0]['xlabels'][1]))


def get_param_values_by_country(country, calibration_folder_name, burn_in=0):

    calibration_output_path = "../../../../data/outputs/calibrate/covid_19/" + country + "/" + calibration_folder_name
    mcmc_tables, _, _ = collect_all_mcmc_output_tables(
        calibration_output_path
    )
    weights = collect_iteration_weights(mcmc_tables, burn_in)

    dodged_columns = ["idx", "Scenario", "loglikelihood", "accept", "tv_detection_sigma"]
    dodged_columns += [c for c in mcmc_tables[0].columns if "dispersion_param" in c]
    param_list = [c for c in mcmc_tables[0].columns if c not in dodged_columns]

    country_param_values = {}
    for param_name in param_list:
        values = []
        for chain_index in range(len(mcmc_tables)):
            for run_id, w in weights[chain_index].items():
                mask = mcmc_tables[0]['idx'] == run_id
                values += [float(mcmc_tables[0][mask][param_name])] * w
        country_param_values[param_name] = values

    return country_param_values


def get_all_param_values(calibration_folder_name):
    param_values = {}
    for country in OPTI_REGIONS:
        param_values[country] = get_param_values_by_country(country, calibration_folder_name)

    return param_values


def plot_param_posteriors(param_values, param_info={}):
    n_panels = len(param_values['belgium'])

    n_col = 4
    n_row = int(n_panels // n_col)
    if n_col * n_row < n_panels:
        n_row += 1

    fig, axs = plt.subplots(n_row, n_col, figsize=(11, 15))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.4)

    i_col = -1
    i_row = 0
    for param_name in list(param_values['belgium'].keys()):
        if "infection_fatality" in param_name:  # reformat
            param_info[param_name] = param_info[param_name][0]

        i_col += 1
        if i_col >= n_col:
            i_col = 0
            i_row += 1

        h = 0

        for country in OPTI_REGIONS:
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

        axs[i_row, i_col].set_title(param_info[param_name]['name'])
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
            axs[i_row, i_col].set_yticklabels([c.title().replace("United-Kingdom", "UK") for c in OPTI_REGIONS])
        else:
            axs[i_row, i_col].set_yticks([])

    # Leave blank axis for remaining panels
    for i_col_blank in range(i_col + 1, n_col):
        axs[i_row, i_col_blank].axis("off")

    plt.savefig("figures/param_posgteriors.png", dpi=300)


if __name__ == "__main__":
    param_values = get_all_param_values('Revised-2020-07-18')

    # dummy data to start with
    # param_values = {}
    # for c in OPTI_REGIONS:
    #     param_values[c] = {}
    #     for i in range(23):
    #         param_values[c][list(param_info.keys())[i]] = random.random(50)

    plot_param_posteriors(param_values, param_info)

