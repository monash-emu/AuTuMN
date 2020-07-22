import pandas as pd
from matplotlib import pyplot as plt

import os
from autumn.db.database import Database
from autumn.tool_kit.uncertainty import export_compartment_size, collect_iteration_weights, collect_all_mcmc_output_tables
from numpy import random, mean, quantile
import numpy as np

from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS
from apps.covid_19.mixing_optimisation.utils import get_list_of_ifr_priors_from_pollan

from autumn.curve import scale_up_function, tanh_based_scaleup

import yaml


param_info = {
    'contact_rate': {'name': 'contact rate', 'range': [0.025, 0.08]},
    'start_time': {'name': 'model start time', 'range': [0., 40.]},
    'npi_effectiveness.other_locations': {'name': 'alpha', 'range': [0.5, 1.]},
    'compartment_periods_calculated.incubation.total_period': {'name': 'incubation time', 'range': [3., 7.]},
    'compartment_periods_calculated.total_infectious.total_period': {'name': 'time infectious', 'range': [5., 10.]},
    'time_variant_detection.maximum_gradient': {'name': 'detection (shape)', 'range': [.05, .1]},
    'time_variant_detection.max_change_time': {'name': 'detection (inflection)', 'range': [70., 160.]},
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

burn_in_by_country = {
    "france": 2000,
    "belgium": 1500,
    "spain": 1500,
    "italy": 2000,
    "sweden": 1500,
    "united-kingdom": 1500,
}


def get_param_values_by_country(country, calibration_folder_name, burn_in=0):

    calibration_output_path = "../../../../data/outputs/calibrate/covid_19/" + country + "/" + calibration_folder_name
    mcmc_tables, _, _ = collect_all_mcmc_output_tables(
        calibration_output_path
    )
    weights = collect_iteration_weights(mcmc_tables, burn_in)

    dodged_columns = ["idx", "Scenario", "loglikelihood", "accept"]
    dodged_columns += [c for c in mcmc_tables[0].columns if "dispersion_param" in c]
    param_list = [c for c in mcmc_tables[0].columns if c not in dodged_columns]

    country_param_values = {}
    for param_name in param_list:
        values = []
        for chain_index in range(len(mcmc_tables)):
            for run_id, w in weights[chain_index].items():
                mask = mcmc_tables[0]['idx'] == run_id
                try:
                    values += [float(mcmc_tables[0][mask][param_name])] * w
                except:
                    print()
        country_param_values[param_name] = values

    return country_param_values


def get_all_param_values(calibration_folder_name):
    param_values = {}
    for country in OPTI_REGIONS:
        param_values[country] = get_param_values_by_country(country, calibration_folder_name, burn_in_by_country[country])

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

    plt.savefig("figures/param_posteriors.png", dpi=300)


def get_country_posterior_detection_percentiles(country_param_values):

    calculated_times = range(213)
    store_matrix = np.zeros((len(calculated_times), len(country_param_values['start_time'])))

    for i in range(len(country_param_values['start_time'])):
        if 'tv_detection_sigma' in country_param_values:
            sigma = country_param_values['tv_detection_sigma'][i]
        else:
            sigma = 0.
        my_func = tanh_based_scaleup(country_param_values['time_variant_detection.maximum_gradient'][i],
                                     country_param_values['time_variant_detection.max_change_time'][i],
                                     sigma)
        detect_vals = [country_param_values['prop_detected_among_symptomatic'][i] * my_func(float(t)) for t in calculated_times]
        store_matrix[:, i] = detect_vals

    perc = np.percentile(store_matrix, [2.5, 25, 50, 75, 97.5], axis=1)
    calculated_times = np.array([calculated_times])
    perc = np.concatenate((calculated_times, perc))

    return perc


def get_all_posterior_detection_percentiles(param_values):
    for country in OPTI_REGIONS:
        print(country)
        country_perc = get_country_posterior_detection_percentiles(param_values[country])
        file_path_ = os.path.join('dumped_files', 'dumped_detection_percentiles_' + country + '.npy')
        with open(file_path_, "wb") as f:
            np.save(f, country_perc)


def plot_detection_posteriors():

    # load percentiles previously dumped
    percentiles = {}

    x_ticks = [32, 61, 92, 122, 153, 183]
    x_ticks_labels = ["Feb 1", "Mar 1", "Apr 1", "May 1", "Jun 1", "Jul 1"]

    for country in OPTI_REGIONS:
        file_path_ = os.path.join('dumped_files', 'dumped_detection_percentiles_' + country + '.npy')
        with open(file_path_, "rb") as f:
            percentiles[country] = np.load(f)

    n_panels = len(OPTI_REGIONS)

    n_col = 3
    n_row = int(n_panels // n_col)
    if n_col * n_row < n_panels:
        n_row += 1

    fig, axs = plt.subplots(n_row, n_col, figsize=(11, 7))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=.3)
    plt.style.use("ggplot")

    i_col = -1
    i_row = 0

    for country in OPTI_REGIONS:
        i_col += 1
        if i_col >= n_col:
            i_col = 0
            i_row += 1

        with plt.style.context('ggplot'):

            times = percentiles[country][0, :]
            median = percentiles[country][3, :]
            low_95 = percentiles[country][1, :]
            up_95 = percentiles[country][5, :]
            low_50 = percentiles[country][2, :]
            up_50 = percentiles[country][4, :]

            axs[i_row, i_col].fill_between(times, low_95, up_95, facecolor="lightsteelblue")
            axs[i_row, i_col].fill_between(times, low_50, up_50, facecolor="cornflowerblue")
            axs[i_row, i_col].plot(times, median, color="navy")

            axs[i_row, i_col].set_ylim((0., 1.))

            c_title = country.title() if country != "united-kingdom" else "United Kingdom"

            axs[i_row, i_col].set_title(c_title)
            axs[i_row, i_col].set_xticks(x_ticks)
            axs[i_row, i_col].set_xticklabels(x_ticks_labels)
            axs[i_row, i_col].set_xlim((45, 212))

            if i_col == 0:
                axs[i_row, i_col].set_ylabel("proportion of symptomatic detected")
            else:
                axs[i_row, i_col].set_yticks([])

    plt.savefig("figures/detection_posteriors.png")


if __name__ == "__main__":
    # param_values = get_all_param_values('Revised-2020-07-18')
    # file_path = os.path.join('dumped_files', 'dumped_dict_param_posteriors.yml')
    # with open(file_path, "w") as f:
    #     yaml.dump(param_values, f)

    # with open('dumped_dict_param_posteriors.yml', "r") as yaml_file:
    #     param_values = yaml.safe_load(yaml_file)
    #
    # detection_percentiles = get_all_posterior_detection_percentiles(param_values)

    # plot_param_posteriors(param_values, param_info)

    plot_detection_posteriors()
