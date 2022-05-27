import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, quantile

from autumn.projects.covid_19.mixing_optimisation.constants import COUNTRY_TITLES, OPTI_REGIONS
from autumn.projects.covid_19.mixing_optimisation.serosurvey_by_age.survey_data import (
    get_serosurvey_data,
)
from autumn.projects.covid_19.mixing_optimisation.utils import get_prior_distributions_for_opti
from autumn.coreimport db, plots
from autumn.model_features.curve.tanh import tanh_based_scaleup
from autumn.coreplots.calibration.plots import get_posterior, get_posterior_best_chain
from autumn.coreplots.uncertainty.plots import plot_timeseries_with_uncertainty
from autumn.utils.params import load_targets


# --------------  Load outputs from databases
def get_calibration_outputs(country):
    calib_dirpath = f"../../pbi_databases/calibration/{country}/"
    mcmc_tables = db.load.load_mcmc_tables(calib_dirpath)
    mcmc_params = db.load.load_mcmc_params_tables(calib_dirpath)
    uncertainty_df = db.load.load_uncertainty_table(calib_dirpath)

    return mcmc_tables, mcmc_params, uncertainty_df


def get_all_countries_calibration_outputs():
    calibration_outputs = {}
    for country in OPTI_REGIONS:
        mcmc_tables, mcmc_params, uncertainty_df = get_calibration_outputs(country)
        calibration_outputs[country] = {
            "mcmc_tables": mcmc_tables,
            "mcmc_params": mcmc_params,
            "uncertainty_df": uncertainty_df,
        }

    return calibration_outputs


def get_targets(region_name):
    return load_targets("covid_19", region_name)


def get_parameter_values(calibration_outputs, best_chain_only=False):
    param_values = {}
    for country, outputs in calibration_outputs.items():
        if len(outputs["mcmc_params"]) == 0:
            continue
        param_values[country] = {}
        all_param_names = [
            p
            for p in outputs["mcmc_params"][0]["name"].unique().tolist()
            if "dispersion_param" not in p
        ]
        for param_name in all_param_names:
            if best_chain_only:
                param_values[country][param_name] = get_posterior_best_chain(
                    outputs["mcmc_params"], outputs["mcmc_tables"], param_name, 0
                )[param_name].tolist()
            else:
                param_values[country][param_name] = get_posterior(
                    outputs["mcmc_params"], outputs["mcmc_tables"], param_name, 0
                )[param_name].tolist()
    return param_values


# --------------  Make figure with posterior estimates

# range is now automatically loaded
param_info = {
    "contact_rate": {"name": "transmission prob.", "range": [0.02, 0.06]},
    "time.start": {"name": "model start time", "range": [0.0, 40.0]},
    "sojourn.compartment_periods_calculated.exposed.total_period": {
        "name": "incubation time",
        "range": [2.5, 7.0],
    },
    "sojourn.compartment_periods_calculated.active.total_period": {
        "name": "time infectious",
        "range": [4.0, 10.0],
    },
    "sojourn.compartment_periods.hospital_late": {
        "name": "time in hospital (non-ICU)",
        "range": [17.7, 20.4],
    },
    "sojourn.compartment_periods.icu_late": {"name": "time in ICU", "range": [9.0, 13.0]},
    "infection_fatality.multiplier": {"name": "IFR multiplier", "range": [0.8, 1.2]},
    "case_detection.shape": {"name": "detection (shape)", "range": [0.05, 0.1]},
    "case_detection.inflection_time": {"name": "detection (inflection)", "range": [100.0, 250.0]},
    "case_detection.end_asymptote": {"name": "detection (prop_final)", "range": [0.10, 0.90]},
    "case_detection.start_asymptote": {"name": "detection (prop_start)", "range": [0.0, 0.10]},
    "icu_prop": {"name": "prop ICU among hosp.", "range": [0.15, 0.20]},
    "compartment_periods.hospital_late": {"name": "hopital duration", "range": [17.7, 20.4]},
    "compartment_periods.icu_late": {"name": "time in ICU", "range": [9.0, 13.0]},
    "clinical_stratification.props.hospital.multiplier": {
        "name": "hosp. prop. multiplier",
        "range": [0.6, 1.4],
    },
    "clinical_stratification.props.symptomatic.multiplier": {
        "name": "sympt. prop. multiplier",
        "range": [0.6, 1.4],
    },
    "mobility.microdistancing.behaviour.parameters.inflection_time": {
        "name": "microdist. (inflection)",
        "range": [80, 130],
    },
    "mobility.microdistancing.behaviour.parameters.end_asymptote": {
        "name": "microdist. (final)",
        "range": [0.25, 0.6],
    },
    "mobility.microdistancing.behaviour_adjuster.parameters.inflection_time": {
        "name": "microdist. wane (inflection)",
        "range": [150, 250],
    },
    "mobility.microdistancing.behaviour_adjuster.parameters.start_asymptote": {
        "name": "microdist. wane (final)",
        "range": [0.5, 1],
    },
    "elderly_mixing_reduction.relative_reduction": {
        "name": "elderly mixing reduction",
        "range": [0, 0.5],
    },
}


def make_posterior_ranges_figure(param_values):
    n_panels = len(param_values["belgium"])
    country_list = OPTI_REGIONS[::-1]
    n_col = 4
    n_row = int(n_panels // n_col)
    if n_col * n_row < n_panels:
        n_row += 1

    fig, axs = plt.subplots(n_row, n_col, figsize=(11, 12))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

    # load priors to set x range
    prior_list = get_prior_distributions_for_opti()

    i_col = -1
    i_row = 0
    for param_name in list(param_values["belgium"].keys()):
        i_col += 1
        if i_col >= n_col:
            i_col = 0
            i_row += 1
        h = 0
        for country in country_list:
            h += 1
            # find mean and CI
            if param_name in param_values[country]:
                values = param_values[country][param_name]
                point_estimate = mean(values)
                low_95, low_50, med, up_50, up_95 = quantile(
                    values, q=(0.025, 0.25, 0.5, 0.75, 0.975)
                )

                axs[i_row, i_col].plot([low_95, up_95], [h, h], linewidth=1, color="black")
                axs[i_row, i_col].plot([low_50, up_50], [h, h], linewidth=3, color="steelblue")
                axs[i_row, i_col].plot(
                    [point_estimate], [h], marker="o", color="crimson", markersize=5
                )

        axs[i_row, i_col].plot([0], [0])
        axs[i_row, i_col].set_ylim((0.5, 6.5))

        axs[i_row, i_col].set_title(param_info[param_name]["name"], fontsize=10.5)

        x_range = param_info[param_name]["range"]
        _prior = [p for p in prior_list if p["param_name"] == param_name]
        if len(_prior) > 0:
            prior = _prior[0]
            if prior["distribution"] == "uniform":
                x_range = prior["distri_params"]
        range_w = x_range[1] - x_range[0]
        buffer = 0.1 * range_w
        x_range = [x_range[0] - buffer, x_range[1] + buffer]

        axs[i_row, i_col].set_xlim(x_range)

        # Format x-ticks if requested
        if "xticks" in param_info[param_name]:
            axs[i_row, i_col].set_xticks(param_info[param_name]["xticks"])
            axs[i_row, i_col].set_xticklabels(param_info[param_name]["xlabels"])

        if "multiplier" in param_info[param_name]:
            axs[i_row, i_col].set_xlabel(param_info[param_name]["multiplier"], labelpad=-7.5)

        # Set y-ticks and yticks-labels
        if i_col == 0:
            axs[i_row, i_col].set_yticks([1, 2, 3, 4, 5, 6])
            axs[i_row, i_col].set_yticklabels(
                [c.title().replace("United-Kingdom", "UK") for c in country_list]
            )
        else:
            axs[i_row, i_col].set_yticks([])

        axs[i_row, i_col].grid(False, axis="y")

    # Leave blank axis for remaining panels
    for i_col_blank in range(i_col + 1, n_col):
        axs[i_row, i_col_blank].axis("off")

    plt.tight_layout()
    plt.savefig("figures/param_posteriors.pdf")


def plot_parameter_traces(param_values_by_chain, max_n_iter=2500):
    param_names = list(param_values_by_chain["belgium"].keys())
    n_rows = len(param_names) + 1
    n_cols = len(OPTI_REGIONS) + 1

    w, h = 6, 2
    title_w, title_h = 4, 2
    fig = plt.figure(
        constrained_layout=True,
        figsize=(title_w + w * len(OPTI_REGIONS), title_h + h * len(param_names)),
    )  # (w, h)
    widths = [title_w] + [w] * len(OPTI_REGIONS)
    heights = [title_h] + [h] * len(param_names)
    spec = fig.add_gridspec(ncols=n_cols, nrows=n_rows, width_ratios=widths, height_ratios=heights)

    # load priors to set y range
    prior_list = get_prior_distributions_for_opti()
    for i_country, country in enumerate(OPTI_REGIONS):
        ax = fig.add_subplot(spec[0, i_country + 1])
        ax.text(
            0.5,
            0.2,
            COUNTRY_TITLES[country],
            fontsize=23,
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")
        for i_param, param_name in enumerate(param_names):
            if param_name in param_values_by_chain[country]:

                ax = fig.add_subplot(spec[i_param + 1, i_country + 1])

                if i_param < len(param_names) - 1:
                    ax.axes.get_xaxis().set_visible(False)

                param_values = param_values_by_chain[country][param_name]
                n_iterations = min(max_n_iter, len(param_values))
                param_values = param_values[:n_iterations]

                ax.plot(range(n_iterations), param_values, "-", color="royalblue")

                ax.grid(False, axis="x")

                y_range = param_info[param_name]["range"]
                _prior = [p for p in prior_list if p["param_name"] == param_name]
                if len(_prior) > 0:
                    prior = _prior[0]
                    if prior["distribution"] == "uniform":
                        y_range = prior["distri_params"]

                range_w = y_range[1] - y_range[0]
                buffer = 0.1 * range_w
                y_range = [y_range[0] - buffer, y_range[1] + buffer]

                ax.set_ylim(y_range)

            if i_country == 0:
                ax = fig.add_subplot(spec[i_param + 1, 0])
                ax.text(
                    0.5,
                    0.2,
                    param_info[param_name]["name"],
                    fontsize=20,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                ax.axis("off")

    plt.tight_layout()
    plt.savefig("figures/param_traces.pdf")


# --------------  Make figure with posterior time-variant detection
def get_all_posterior_detection_percentiles(param_values):
    for country in OPTI_REGIONS:
        country_perc = get_country_posterior_detection_percentiles(param_values[country])
        file_path_ = os.path.join(
            "dumped_files", "dumped_detection_percentiles_" + country + ".npy"
        )
        with open(file_path_, "wb") as f:
            np.save(f, country_perc)


def get_country_posterior_detection_percentiles(country_param_values):

    calculated_times = list(range(300))[30:]
    store_matrix = np.zeros((len(calculated_times), len(country_param_values["time.start"])))

    for i in range(len(country_param_values["time.start"])):
        if "case_detection.start_asymptote" in country_param_values:
            start_asymptote = country_param_values["case_detection.start_asymptote"][i]
        else:
            start_asymptote = 0.0
        my_func = tanh_based_scaleup(
            country_param_values["case_detection.shape"][i],
            country_param_values["case_detection.inflection_time"][i],
            start_asymptote,
            country_param_values["case_detection.end_asymptote"][i],
        )
        detect_vals = [my_func(float(t)) for t in calculated_times]
        store_matrix[:, i] = detect_vals

    perc = np.percentile(store_matrix, [2.5, 25, 50, 75, 97.5], axis=1)
    calculated_times = np.array([calculated_times])
    perc = np.concatenate((calculated_times, perc))

    return perc


def plot_posterior_detection():

    # load percentiles previously dumped
    percentiles = {}

    x_ticks = [61, 122, 183, 245, 306]
    x_ticks_labels = ["Mar 1", "May 1", "Jul 1", "Sep 1", "Nov 1"]

    for country in OPTI_REGIONS:
        file_path_ = os.path.join(
            "dumped_files", "dumped_detection_percentiles_" + country + ".npy"
        )
        with open(file_path_, "rb") as f:
            percentiles[country] = np.load(f)

    n_panels = len(OPTI_REGIONS)

    n_col = 3
    n_row = int(n_panels // n_col)
    if n_col * n_row < n_panels:
        n_row += 1

    fig, axs = plt.subplots(n_row, n_col, figsize=(13, 7))
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.5)

    i_col = -1
    i_row = 0
    for country in OPTI_REGIONS:
        i_col += 1
        if i_col >= n_col:
            i_col = 0
            i_row += 1

        times = percentiles[country][0, :]
        median = percentiles[country][3, :]
        low_95 = percentiles[country][1, :]
        up_95 = percentiles[country][5, :]
        low_50 = percentiles[country][2, :]
        up_50 = percentiles[country][4, :]

        axs[i_row, i_col].fill_between(times, low_95, up_95, facecolor="lightsteelblue")
        axs[i_row, i_col].fill_between(times, low_50, up_50, facecolor="cornflowerblue")
        axs[i_row, i_col].plot(times, median, color="navy")

        axs[i_row, i_col].set_ylim((0.0, 1.0))

        c_title = country.title() if country != "united-kingdom" else "United Kingdom"

        axs[i_row, i_col].set_title(c_title)
        axs[i_row, i_col].set_xticks(x_ticks)
        axs[i_row, i_col].set_xticklabels(x_ticks_labels)
        axs[i_row, i_col].set_xlim((45, 300))

        if i_col == 0:
            axs[i_row, i_col].set_ylabel("proportion of symptomatic\ndetected")

    plt.tight_layout()
    plt.savefig("figures/detection_posteriors.pdf")


# --------------  Make figure with fits to population-level data
SEROSURVEYS = {
    "belgium": [
        {"time_window": [90.0, 96], "value": 0.029, "ci": [0.023, 0.034]},  # 30 mar  5 Apr
        {"time_window": [111, 117], "value": 0.06, "ci": [0.051, 0.071]},  # 20 -26 Apr
        {"time_window": [139, 146], "value": 0.069, "ci": [0.059, 0.080]},  # 18 -25 May
        {"time_window": [160, 165], "value": 0.055, "ci": [0.047, 0.065]},  # 8 - 13 jun
        {"time_window": [181, 185], "value": 0.045, "ci": [0.037, 0.054]},  # 29 Jun -3 Jul
    ],
    "france": [
        {"time_window": [85.0, 98], "value": 0.0271},
    ],
    "italy": [
        {"time_window": [146.0, 197], "value": 0.0250},  # 25 may 15 jul
    ],
    "spain": [
        {"time_window": [118, 132], "value": 0.05, "ci": [0.047, 0.054]},
        {"time_window": [139, 153], "value": 0.052, "ci": [0.049, 0.055]},
        {"time_window": [160, 174], "value": 0.052, "ci": [0.049, 0.055]},
    ],
    "sweden": [
        {"time_window": [122, 152], "value": 0.108, "ci": [0.079, 0.137]},
    ],
    "united-kingdom": [
        {"time_window": [118, 124], "value": 0.0710},
        {"time_window": [172, 195], "value": 0.06, "ci": [0.058, 0.061]},
    ],
}


def make_calibration_fits_figure(calibration_outputs, seroprevalence=False):

    if not seroprevalence:
        countries_per_row = 1
        n_target_outputs = 4
        show_title = True
        lab_fontsize = 13
        target_outputs = {}
        for country in OPTI_REGIONS:
            targets = get_targets(country)
            target_outputs[country] = ["notifications"]
            hospital_target = [t for t in list(targets.keys()) if "hospital" in t or "icu" in t][0]
            target_outputs[country].append(hospital_target)
            target_outputs[country].append("infection_deaths")
            target_outputs[country].append("proportion_seropositive")
    else:
        show_title = False
        lab_fontsize = 15
        countries_per_row = 3
        n_target_outputs = 1
        target_outputs = {}
        for country in OPTI_REGIONS:
            target_outputs[country] = ["proportion_seropositive"]

    n_countries_per_col = int(6 / countries_per_row)

    width = n_target_outputs * countries_per_row * 5.2
    height = n_countries_per_col * 5
    fig = plt.figure(constrained_layout=True, figsize=(width, height))  # (w, h)
    widths = [5] * (countries_per_row * n_target_outputs)
    heights = [1, 6] * n_countries_per_col
    spec = fig.add_gridspec(
        ncols=countries_per_row * n_target_outputs,
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
    )

    text_size = 23
    i_row = 1
    i_col = 0

    ordered_countries = ["italy", "united-kingdom", "france", "belgium", "spain", "sweden"]
    for country in ordered_countries:
        # write country name
        ax = fig.add_subplot(spec[i_row - 1, i_col : i_col + n_target_outputs])
        ax.text(
            0.5,
            0.2,
            COUNTRY_TITLES[country],
            fontsize=text_size,
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")
        for output in target_outputs[country]:
            show_ylab = (i_col == 0) and seroprevalence

            ax = fig.add_subplot(spec[i_row, i_col])
            country_targets = get_targets(country)
            targets = {k: v for k, v in country_targets.items() if v["output_key"] == output}
            plot_timeseries_with_uncertainty(
                None,
                calibration_outputs[country]["uncertainty_df"],
                output,
                [0],
                targets,
                False,
                40,
                276,
                ax,
                n_xticks=None,
                title_font_size=16,
                label_font_size=lab_fontsize,
                requested_x_ticks=[61, 122, 183, 245],
                show_title=show_title,
                show_ylab=show_ylab,
                add_targets=(not seroprevalence),
            )
            # for s in SEROSURVEYS[country]:
            #     if "ci" in s:
            #         rect = patches.Rectangle(
            #             (s["time_window"][0], s["ci"][0]),
            #             s["time_window"][1] - s["time_window"][0],
            #             s["ci"][1] - s["ci"][0],
            #             linewidth=0,
            #             facecolor="gold",
            #             alpha=0.4,
            #         )
            #         rect.set_zorder(1)
            #         ax.add_patch(rect)
            #     ax.plot(s["time_window"], [s["value"]] * 2, linewidth=1.5, color="black")

            if output == "proportion_seropositive":
                ax.set_ylim((0, 0.20))
                ax.set_title("percentage ever infected")

            i_col += 1
            if i_col == countries_per_row * n_target_outputs:
                i_col = 0
                i_row += 2

    # Add vertical separator line
    if not seroprevalence and countries_per_row == 2:
        line = plt.Line2D((0.5, 0.5), (-0.1, 1.9), color="grey", linewidth=1)
        fig.add_artist(line)

    filename = "figures/model_fits"
    if seroprevalence:
        filename += "_seroprevalence"
    plt.savefig(filename + ".pdf")


# --------------  Make figure with fits to age-specific seroprevalence
def make_all_sero_by_age_fits_figures(calibration_outputs):
    # get all age-specific targets
    sero_data = get_serosurvey_data()
    country_list = [c for c in OPTI_REGIONS if c in sero_data]

    for region in country_list:
        make_sero_by_age_fits_figure(
            calibration_outputs[region]["uncertainty_df"], region, sero_data[region]
        )


def make_sero_by_age_fits_figure(uncertainty_df, region, sero_data_by_age):

    fig = plots.uncertainty.plots.plot_seroprevalence_by_age_against_targets(
        None, uncertainty_df, 0, sero_data_by_age, 3
    )
    plt.savefig(f"figures/seroprevalence_by_age/sero_by_age_{region}.pdf")
