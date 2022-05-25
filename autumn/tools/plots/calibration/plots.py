"""
Calibration plots
"""
import logging
import os
from math import log
from random import choices
from typing import List

import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot
from scipy import stats

from autumn.tools import db
from autumn.runners.calibration.utils import calculate_prior, raise_error_unsupported_prior
from autumn.tools.plots.plotter import COLOR_THEME, Plotter
from autumn.tools.plots.utils import (
    REF_DATE,
    _plot_targets_to_axis,
    change_xaxis_to_date,
    get_plot_text_dict,
    split_mcmc_outputs_by_chain,
)
from autumn.tools.utils.utils import flatten_list
from autumn.runners.calibration.diagnostics import calculate_effective_sample_size, calculate_r_hat

logger = logging.getLogger(__name__)


# FIXME: Should really have some tests around it
def collate_acceptance_ratios(acceptance_list):
    """
    Collate the running proportion of all runs that have been accepted from an MCMC chain.
    """

    count, n_total, ratios = 0, 0, []
    for n_accept in acceptance_list:
        n_total += 1
        if n_accept:
            count += 1
        ratios.append(count / n_total)
    return ratios


def get_epi_params(mcmc_params, strings_to_ignore=("dispersion_param",)):
    """
    Extract only the epidemiological parameters, ignoring the ones that were only used to tune proposal distributions,
    which end in dispersion_param.
    """

    return [
        param
        for param in mcmc_params[0].loc[:, "name"].unique().tolist()
        if not any(string in param for string in strings_to_ignore)
    ]


def find_shortest_chain_length(mcmc_tables):
    """
    Find the length of the shortest chain from the MCMC tables.
    """

    return int(min([table["run"].iloc[-1] for table in mcmc_tables]))


"""
Parameter diagnostics
"""


def plot_acceptance_ratio(
    plotter: Plotter,
    mcmc_tables: List[pd.DataFrame],
    burn_in: int,
    label_font_size=6,
    dpi_request=300,
):
    """
    Plot the progressive acceptance ratio over iterations.
    """

    fig, axis, _, _, _, _ = plotter.get_figure()
    full_df = db.load.append_tables(mcmc_tables)
    # Chain index starts at 0
    n_chains = max(full_df["chain"]) + 1
    for chain in range(n_chains):
        chain_mask = full_df["chain"] == chain
        chain_df = full_df[chain_mask]
        ratios = collate_acceptance_ratios(chain_df["accept"])

        # Plot
        axis.plot(ratios, alpha=0.8, linewidth=0.7)

        # Add vertical line for burn-in point
        if burn_in > 0:
            axis.axvline(x=burn_in, color=COLOR_THEME[1], linestyle="dotted")

    axis.set_title("acceptance ratio", fontsize=label_font_size)
    axis.set_xlabel("iterations", fontsize=label_font_size)
    axis.set_ylim(bottom=0.0)
    pyplot.setp(axis.get_yticklabels(), fontsize=label_font_size)
    pyplot.setp(axis.get_xticklabels(), fontsize=label_font_size)
    plotter.save_figure(fig, filename=f"acceptance_ratio", dpi_request=dpi_request)


def plot_prior(i: int, prior_dict: dict, path: str):
    if prior_dict["distribution"] == "lognormal":
        logger.error("Cannot plot prior distributions for lognormal.")
        return

    fig, ax = pyplot.subplots()
    x_range = workout_plot_x_range(prior_dict)
    x_values = np.linspace(x_range[0], x_range[1], num=1000)
    y_values = [calculate_prior(prior_dict, x, log=False) for x in x_values]
    zeros = [0.0 for i in x_values]
    pyplot.fill_between(x_values, y_values, zeros, color="cornflowerblue")

    if "distri_mean" in prior_dict:
        pyplot.axvline(
            x=prior_dict["distri_mean"],
            ymin=0,
            ymax=100 * max(y_values),
            linewidth=1,
            color="red",
        )
    if "distri_ci" in prior_dict:
        pyplot.axvline(
            x=prior_dict["distri_ci"][0],
            ymin=0,
            ymax=100 * max(y_values),
            linewidth=0.7,
            color="red",
        )
        pyplot.axvline(
            x=prior_dict["distri_ci"][1],
            ymin=0,
            ymax=100 * max(y_values),
            linewidth=0.7,
            color="red",
        )

    pyplot.xlabel(prior_dict["param_name"])
    pyplot.ylabel("prior PDF")

    # place a text box in upper left corner to indicate the prior details
    props = dict(boxstyle="round", facecolor="dimgray", alpha=0.5)
    textstr = (
        prior_dict["distribution"]
        + "\n("
        + str(round(float(prior_dict["distri_params"][0]), 3))
        + ", "
        + str(round(float(prior_dict["distri_params"][1]), 3))
        + ")"
    )
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    pyplot.tight_layout()
    filename = os.path.join(path, prior_dict["param_name"] + ".png")
    pyplot.savefig(filename)


def workout_plot_x_range(prior_dict):
    if prior_dict["distribution"] == "uniform":
        x_range = prior_dict["distri_params"]
    elif prior_dict["distribution"] == "beta":
        a = prior_dict["distri_params"][0]
        b = prior_dict["distri_params"][1]
        x_range = stats.beta.ppf([0.005, 0.995], a, b)
    elif prior_dict["distribution"] == "gamma":
        shape = prior_dict["distri_params"][0]
        scale = prior_dict["distri_params"][1]
        x_range = stats.gamma.ppf([0.005, 0.995], shape, 0.0, scale)
    elif prior_dict["distribution"] == "trunc_normal":
        lower = prior_dict["trunc_range"][0]
        upper = prior_dict["trunc_range"][1]
        x_range = [lower - 0.05 * (upper - lower), upper + 0.05 * (upper - lower)]

        mean = prior_dict["distri_params"][0]
        sd = prior_dict["distri_params"][1]
        x_range = stats.norm.ppf([0.005, 0.995], mean, sd)

        if lower != -np.inf:
            x_range[0] = lower - 0.05 * (x_range[1] - lower)
        if upper != np.inf:
            x_range[1] = upper + 0.05 * (upper - x_range[0])

    else:
        raise_error_unsupported_prior(prior_dict["distribution"])

    return x_range


def plot_mcmc_parameter_trace(
    plotter: Plotter, mcmc_params: List[pd.DataFrame], burn_in: int, param_name: str
):
    """
    Plot the prameter traces for each MCMC run.
    """
    fig, axis, _, _, _, _ = plotter.get_figure()
    for idx, table_df in enumerate(mcmc_params):
        param_mask = table_df["name"] == param_name
        param_df = table_df[param_mask]
        axis.plot(param_df["run"], param_df["value"], alpha=0.8, linewidth=0.7)
        if burn_in > 0:
            axis.axvline(x=burn_in, color=COLOR_THEME[1], linestyle="dotted")

    axis.set_ylabel(param_name)
    axis.set_xlabel("MCMC iterations")
    plotter.save_figure(fig, filename=f"{param_name}-traces", title_text=param_name)


def plot_autocorrelation(
    plotter: Plotter,
    mcmc_params: List[pd.DataFrame],
    mcmc_tables: List[pd.DataFrame],
    burn_in: int,
    param_name: str,
):
    """
    Plot each chain's autocorrelogram.
    """
    fig, axis, _, n_rows, n_cols, indices = plotter.get_figure()

    for idx, table_df in enumerate(mcmc_params):
        # Retrieve the full chain's posterior distribution
        posterior_chain = get_posterior([table_df], [mcmc_tables[idx]], param_name, burn_in)
        # Plot autocorrelogram
        pd.plotting.autocorrelation_plot(posterior_chain, ax=axis)

    # Plot details
    axis.set_ylabel(f"autocorrelation")
    axis.set_xlabel("lag")
    axis.set_title(param_name)

    # Save figure
    plotter.save_figure(fig, filename=f"{param_name}-autocorrelation", title_text="")


def make_ess_table(mcmc_params: List[pd.DataFrame], mcmc_tables: List[pd.DataFrame], burn_in: int):
    """
    Creates a table containing the effective sample size for each parameter and for each chain
    :return: a pandas dataframe
    """
    param_options = mcmc_params[0]["name"].unique().tolist()

    index_names = param_options + ["average ESS"]
    col_names = [f"chain_{i}" for i in range(len(mcmc_tables))]
    ess_table = pd.DataFrame(0, index=index_names, columns=col_names)

    for i_chain, table_df in enumerate(mcmc_params):
        for param_name in param_options:
            posterior_chain = get_posterior([table_df], [mcmc_tables[i_chain]], param_name, burn_in)
            ess = calculate_effective_sample_size(posterior_chain)
            ess_table[f"chain_{i_chain}"][param_name] = ess

        ess_table[f"chain_{i_chain}"]["average ESS"] = ess_table[f"chain_{i_chain}"][
            0 : len(param_options)
        ].mean()

    ess_table["total_ESS"] = ess_table.sum(numeric_only=True, axis=1)

    return ess_table


def calculate_r_hats(mcmc_params: List[pd.DataFrame], mcmc_tables: List[pd.DataFrame], burn_in: int):
    """
    Calculates the R_hat statistic for all parameters
    :return: a dictionary
    """

    param_options = mcmc_params[0].columns.tolist()
    chain_idx = mcmc_tables[0].chain.unique()
    r_hats = {}
    for param_name in param_options:
        posterior_chains = {}
        for chain_id in chain_idx:
            mask =  mcmc_tables[0].chain == chain_id
            param_vals = mcmc_params[0][mask][param_name].to_list()
            weights = mcmc_tables[0][mask].weight.to_list()
            posterior_chains[chain_id] = flatten_list([[param_vals[i]] * w for i, w in enumerate(weights)])

        r_hats[param_name] = calculate_r_hat(posterior_chains)

    return r_hats


def plot_effective_sample_size(
    plotter: Plotter, mcmc_params: List[pd.DataFrame], mcmc_tables: List[pd.DataFrame], burn_in: int
):
    fig, axis, _, n_rows, n_cols, indices = plotter.get_figure()

    ess_table = make_ess_table(mcmc_params, mcmc_tables, burn_in)
    colours = COLOR_THEME[1 : len(mcmc_tables) + 1] + [
        "black"
    ]  # AuTuMN color scheme and black for the sum of ESS
    ess_table.plot.barh(ax=axis, color=colours)

    axis.tick_params(axis="y", labelrotation=45)
    axis.xaxis.set_tick_params(labeltop="on", top="on")
    axis.set_xlabel("effective sample size")
    axis.set_title("Effective sample size")

    # figure height
    h = (len(ess_table.index)) * 1.5
    fig.set_figheight(h)

    # Save figure
    plotter.save_figure(fig, filename=f"effective_sample_size", title_text="")


def plot_parallel_coordinates_flat(
    plotter: Plotter,
    mcmc_params: List[pd.DataFrame],
    mcmc_tables: List[pd.DataFrame],
    priors,
    map_only=False,
):

    fig, axis, _, n_rows, n_cols, indices = plotter.get_figure()

    # select parameter values for all chains
    parameters = [
        param
        for param in mcmc_params[0].loc[:, "name"].unique().tolist()
        if "dispersion_param" not in param
    ]
    combined_mcmc_df = merge_and_pivot_mcmc_parameters_loglike(
        mcmc_tables, mcmc_params, parameters, n_samples_per_chain=10, map_only=map_only
    )
    labels = {}
    for param in parameters:
        labels[param] = get_plot_text_dict(param)

    # drop some columns
    combined_mcmc_df = combined_mcmc_df[parameters + ["fitness", "chain"]]

    # rescale the parameters
    for var in parameters + ["fitness"]:
        a = min(combined_mcmc_df[var])
        b = max(combined_mcmc_df[var])
        if priors:
            if priors != [None]:
                prior = [p for p in priors if p["param_name"] == var]
                if len(prior) > 0:
                    prior = prior[0]
                    x_range = workout_plot_x_range(prior)
                    a = x_range[0]
                    b = x_range[1]

        if b > a:
            combined_mcmc_df[var] = (combined_mcmc_df[var] - a) / (b - a)

    # set chain colours
    chain_ids = combined_mcmc_df["chain"].unique()
    colours = [COLOR_THEME[i + 1] for i in chain_ids]

    # main plot
    _lw = 1.0 if map_only else 0.4
    pd.plotting.parallel_coordinates(combined_mcmc_df, "chain", ax=axis, color=colours, lw=_lw)
    axis.set_ylabel("normalised value (rel. to prior range)")
    axis.tick_params(axis="x", labelrotation=20)

    # figure width
    w = len(parameters) * 2
    fig.set_figwidth(w)

    if map_only:
        plotter.save_figure(fig, filename="parallel_map", title_text="MAP per chain")
    else:
        plotter.save_figure(fig, filename="parallel_coord", title_text="parallel coordinates")


def plot_multiple_param_traces(
    plotter: Plotter,
    mcmc_params: List[pd.DataFrame],
    mcmc_tables: List[pd.DataFrame],
    burn_in: int,
    title_font_size: int,
    label_font_size: int,
    capitalise_first_letter: bool,
    dpi_request: int,
    optional_param_request=None,
    file_name="all_traces",
    x_ticks_on=True,
):

    # Except not the dispersion parameters - only the epidemiological ones
    parameters = [
        param
        for param in mcmc_params[0].loc[:, "name"].unique().tolist()
        if "dispersion_param" not in param
    ]
    params_to_plot = optional_param_request if optional_param_request else parameters

    fig, axes, _, n_rows, n_cols, indices = plotter.get_figure(
        len(params_to_plot), share_xaxis=True, share_yaxis=False
    )

    # split MCMC tables by chain
    mcmc_params_list, mcmc_tables_list = split_mcmc_outputs_by_chain(mcmc_params, mcmc_tables)

    for i in range(n_rows * n_cols):
        if len(params_to_plot) <= 2:
            axis = axes[i]
        else:
            axis = axes[indices[i][0], indices[i][1]]
        if i < len(params_to_plot):
            param_name = params_to_plot[i]

            for i_chain in range(len(mcmc_params_list)):
                param_values = get_posterior([mcmc_params_list[i_chain]], [mcmc_tables_list[i_chain]], param_name, burn_in)
                axis.plot(param_values, alpha=0.8, linewidth=0.3)

            axis.set_title(
                get_plot_text_dict(param_name, capitalise_first_letter=capitalise_first_letter),
                fontsize=title_font_size,
            )

            if not x_ticks_on:
                axis.set_xticks([])
            elif not indices or indices[i][0] == n_rows - 1:
                x_label = "Iterations" if capitalise_first_letter else "iterations"
                axis.set_xlabel(x_label, fontsize=label_font_size)
            pyplot.setp(axis.get_yticklabels(), fontsize=label_font_size)
            pyplot.setp(axis.get_xticklabels(), fontsize=label_font_size)

            if burn_in > 0:
                axis.axvline(x=burn_in, color=COLOR_THEME[1], linestyle="dotted")

        else:
            axis.axis("off")

    fig.tight_layout()
    plotter.save_figure(fig, filename=file_name, dpi_request=dpi_request)


def plot_loglikelihood_trace(plotter: Plotter, mcmc_tables: List[pd.DataFrame], burn_in=0, posterior=False):
    """
    Plot the loglikelihood traces for each MCMC run.
    """
    fig, axis, _, _, _, _ = plotter.get_figure()

    variable_key = "ap_loglikelihood" if posterior else "loglikelihood"

    if len(mcmc_tables) == 1:  # there may be multiple chains within a single dataframe
        table_df = mcmc_tables[0]
        accept_mask = table_df["accept"] == 1
        chain_idx = list(table_df["chain"].unique())
        for chain_id in chain_idx:
            chain_mask = table_df["chain"] == chain_id
            masked_df = table_df[accept_mask][chain_mask]
            axis.plot(masked_df["run"], masked_df[variable_key], alpha=0.8, linewidth=0.7)
    else:  # there is one chain per dataframe
        for idx, table_df in enumerate(mcmc_tables):
            accept_mask = table_df["accept"] == 1
            if posterior:
                table_df[accept_mask].loglikelihood.plot.line(ax=axis, alpha=0.8, linewidth=0.7)
            else:
                table_df[accept_mask].ap_loglikelihood.plot.line(ax=axis, alpha=0.8, linewidth=0.7)
    title = "Posterior Loglikelihood" if posterior else "Loglikelihood"
    axis.set_ylabel(title)
    axis.set_xlabel("Metropolis iterations")

    if burn_in:
        axis.axvline(x=burn_in, color=COLOR_THEME[1], linestyle="dotted")
        y_min = min(table_df.loglikelihood[burn_in:])
        y_max = max(table_df.loglikelihood[burn_in:])
        axis.set_ylim((y_min - 0.2 * (y_max - y_min), y_max + 0.2 * (y_max - y_min)))

    plotter.save_figure(fig, filename=f"{variable_key}-traces", title_text=f"{variable_key}-traces")


def plot_loglikelihood_boxplots(plotter: Plotter, mcmc_tables: List[pd.DataFrame]):
    fig, axis, _, _, _, _ = plotter.get_figure()
    if len(mcmc_tables) > 1:
        df = pd.concat(mcmc_tables)
    else:
        df = mcmc_tables[0]

    df["-log(-loglikelihood)"] = [-log(-v) for v in df["loglikelihood"]]
    df.boxplot(column=["-log(-loglikelihood)"], by="chain", ax=axis)
    plotter.save_figure(fig, filename="loglikelihood-boxplots", title_text="")


def plot_burn_in(plotter: Plotter, num_iters: int, burn_in: int):
    """
    Plot the trade off been num iters and burn-in for MCMC runs.
    """
    fig, axis, _, _, _, _ = plotter.get_figure()

    def floor(n):
        val = num_iters - n
        return val if val > 0 else 0

    values = [floor(i) for i in range(num_iters)]

    fig, axis, _, _, _, _ = plotter.get_figure()

    axis.plot(values, color=COLOR_THEME[0])
    axis.set_ylabel("Number iters after burn-in")
    axis.set_xlabel("Burn-in size")
    axis.set_ylim(bottom=-5, top=num_iters)
    axis.set_xlim(left=0, right=num_iters)
    axis.axvline(x=burn_in, color=COLOR_THEME[1], linestyle="dotted")
    axis.axhline(y=num_iters - burn_in, color=COLOR_THEME[1], linestyle="dotted")
    plotter.save_figure(fig, filename="burn-in", title_text="burn-in")


def get_posterior(mcmc_params, mcmc_tables, param_name, burn_in=0):
    weighted_vals = []
    for param_df, run_df in zip(mcmc_params, mcmc_tables):
        table_df = param_df.merge(run_df, on=["run", "chain"])
        param_mask = (table_df["name"] == param_name) & (table_df["run"] > burn_in)
        unweighted_vals = table_df[param_mask].value
        weights = table_df[param_mask].weight
        for v, w in zip(unweighted_vals, weights):
            weighted_vals += [v] * w

    return pd.DataFrame(weighted_vals, columns=[param_name])


def get_posterior_best_chain(mcmc_params, mcmc_tables, param_name, burn_in=0):

    weighted_vals = []
    for param_df, run_df in zip(mcmc_params, mcmc_tables):
        table_df = param_df.merge(run_df, on=["run", "chain"])
        param_mask = (table_df["name"] == param_name) & (table_df["run"] > burn_in)
        table_df = table_df[param_mask]
        best_chain = int(table_df.loc[table_df["loglikelihood"].idxmax()]["chain"])
        mask = table_df["chain"] == best_chain
        weighted_vals = []
        unweighted_vals = table_df[mask].value
        weights = table_df[mask].weight
        for v, w in zip(unweighted_vals, weights):
            weighted_vals += [v] * w

    return pd.DataFrame(weighted_vals, columns=[param_name])


def plot_posterior(
    plotter: Plotter,
    mcmc_params: List[pd.DataFrame],
    mcmc_tables: List[pd.DataFrame],
    burn_in: int,
    param_name: str,
    num_bins: int,
    priors,
):
    """
    Plots the posterior distribution of a given parameter in a histogram.
    """
    vals_df = get_posterior(mcmc_params, mcmc_tables, param_name, burn_in)
    fig, axis, _, _, _, _ = plotter.get_figure()
    vals_df.hist(bins=num_bins, ax=axis, density=True)

    if priors:
        if priors != [None]:
            prior = [p for p in priors if p["param_name"] == param_name]
            if len(prior) > 0:
                prior = prior[0]
                x_range = workout_plot_x_range(prior)
                x_values = np.linspace(x_range[0], x_range[1], num=1000)
                y_values = [calculate_prior(prior, x, log=False) for x in x_values]

                # Plot the prior
                axis.plot(x_values, y_values)

    plotter.save_figure(fig, filename=f"{param_name}-posterior", title_text="")


def plot_multiple_posteriors(
    plotter: Plotter,
    mcmc_params: List[pd.DataFrame],
    mcmc_tables: List[pd.DataFrame],
    burn_in: int,
    num_bins: int,
    title_font_size: int,
    label_font_size: int,
    capitalise_first_letter: bool,
    dpi_request: int,
    priors: list,
    parameters: list,
    file_name="all_posteriors",
):
    """
    Plots the posterior distribution of a given parameter in a histogram.
    """

    # Except not the dispersion parameters - only the epidemiological ones
    fig, axes, _, n_rows, n_cols, indices = plotter.get_figure(len(parameters))

    for i in range(n_rows * n_cols):

        if len(parameters) <= 2:
            axis = axes[i]
        else:
            axis = axes[indices[i][0], indices[i][1]]

        if i < len(parameters):
            param_name = parameters[i]
            vals_df = get_posterior(mcmc_params, mcmc_tables, param_name, burn_in)

            for i, prior in enumerate(priors):
                if prior["param_name"] == param_name:
                    prior = priors[i]
                    break
            x_range = workout_plot_x_range(prior)
            x_values = np.linspace(x_range[0], x_range[1], num=1000)
            y_values = [calculate_prior(prior, x, log=False) for x in x_values]

            # Plot histograms
            vals_df.hist(bins=num_bins, ax=axis, density=True)

            # Plot the prior
            axis.plot(x_values, y_values)

            axis.set_title(
                get_plot_text_dict(param_name, capitalise_first_letter=capitalise_first_letter),
                fontsize=title_font_size,
            )
            pyplot.setp(axis.get_yticklabels(), fontsize=label_font_size)
            pyplot.setp(axis.get_xticklabels(), fontsize=label_font_size)

        else:
            axis.axis("off")

    fig.tight_layout()
    plotter.save_figure(fig, filename=file_name, dpi_request=dpi_request)


def plot_param_vs_loglike(mcmc_tables, mcmc_params, param_name, burn_in, axis, posterior=False):
    for mcmc_df, param_df in zip(mcmc_tables, mcmc_params):
        df = param_df.merge(mcmc_df, on=["run", "chain"])
        mask = (df["accept"] == 1) & (df["name"] == param_name) & (df["run"] > burn_in)
        df = df[mask]
        param_values = df["value"]
        var_key = "ap_loglikelihood" if posterior else "loglikelihood"
        loglikelihood_values = df[var_key]

        # apply transformation to improve readability
        m = max(loglikelihood_values) + 0.01
        trans_loglikelihood_values = [-log(-v + m) for v in df[var_key]]
        axis.plot(param_values, trans_loglikelihood_values, ".")


def plot_parallel_coordinates(
    plotter: Plotter,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
):
    parameters = [
        param
        for param in mcmc_params[0].loc[:, "name"].unique().tolist()
        if "dispersion_param" not in param
    ]

    target_n_lines = 500.0
    n_samples = int(target_n_lines / len(mcmc_tables))
    combined_mcmc_df = merge_and_pivot_mcmc_parameters_loglike(
        mcmc_tables, mcmc_params, parameters, n_samples_per_chain=n_samples
    )
    w = len(parameters) * 200
    h = 800
    labels = {}
    for param in parameters:
        labels[param] = get_plot_text_dict(param)
    figure = px.parallel_coordinates(
        combined_mcmc_df,
        color="fitness",
        dimensions=parameters,
        labels=labels,
        color_continuous_scale=px.colors.diverging.Tealrose,
        height=h,
        width=w,
    )
    figure.show()


def merge_and_pivot_mcmc_parameters_loglike(
    mcmc_tables, mcmc_params, parameters, n_samples_per_chain=None, map_only=False
):

    combined_mcmc_df = None
    for mcmc_df, param_df in zip(mcmc_tables, mcmc_params):
        mask = mcmc_df["accept"] == 1
        mcmc_df = mcmc_df[mask]
        if not map_only:
            n_iter = (
                len(mcmc_df.index)
                if n_samples_per_chain is None
                else min(n_samples_per_chain, len(mcmc_df.index))
            )
            mcmc_df = mcmc_df.iloc[-n_iter:]
        else:
            mcmc_df = mcmc_df[mcmc_df.ap_loglikelihood == max(mcmc_df.ap_loglikelihood)]
        for param in parameters:
            param_vals = []
            for c, r in zip(mcmc_df["chain"], mcmc_df["run"]):
                mask1 = param_df["chain"] == c
                mask2 = param_df["name"] == param
                mask3 = param_df["run"] == r
                param_vals.append(param_df[mask1][mask2][mask3]["value"].iloc[0])
            mcmc_df[param] = param_vals

        if combined_mcmc_df is None:
            combined_mcmc_df = mcmc_df
        else:
            combined_mcmc_df = combined_mcmc_df.append(mcmc_df)

    combined_mcmc_df["fitness"] = [-log(-v) for v in combined_mcmc_df["loglikelihood"]]
    return combined_mcmc_df


def plot_loglikelihood_surface(
    plotter: Plotter,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    param_1,
    param_2,
):
    combined_mcmc_df = merge_and_pivot_mcmc_parameters_loglike(
        mcmc_tables, mcmc_params, [param_1, param_2]
    )

    fig = px.scatter_3d(combined_mcmc_df, x=param_1, y=param_2, z="fitness", color="chain")
    fig.show()


def plot_single_param_loglike(
    plotter: Plotter,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    burn_in: int,
    param_name: str,
    posterior=False,
):
    """
    Plots the loglikelihood against parameter values.
    """


    fig, axis, _, _, _, _ = plotter.get_figure()
    plot_param_vs_loglike(mcmc_tables, mcmc_params, param_name, burn_in, axis, posterior)
    axis.set_xlabel(param_name)
    axis.set_ylabel("-log(-loglikelihood)")

    filename = f"likelihood-against-{param_name}"
    if posterior:
        filename = "posterior-" + filename

    plotter.save_figure(
        fig,
        filename=filename,
        title_text=param_name,
    )


def plot_param_vs_param_by_chain(
    plotter: Plotter,
    mcmc_params: List[pd.DataFrame],
    parameters: list,
    label_font_size: int,
    label_chars: int,
    dpi_request: int,
):
    """
    Plot the parameter traces for each MCMC chain with separate colouring.
    """

    fig, axes, _, _, _, _ = plotter.get_figure(n_panels=len(parameters) ** 2)

    for chain in range(len(mcmc_params)):
        for x_idx, x_param_name in enumerate(parameters):
            x_param_mask = mcmc_params[chain]["name"] == x_param_name
            for y_idx, y_param_name in enumerate(parameters):
                y_param_mask = mcmc_params[chain]["name"] == y_param_name

                # Get axis and turn off ticks
                axis = axes[x_idx, y_idx]
                axis.xaxis.set_ticks([])
                axis.yaxis.set_ticks([])

                # Plot
                if x_idx > y_idx:
                    axis.scatter(
                        mcmc_params[chain][x_param_mask]["value"].to_list(),
                        mcmc_params[chain][y_param_mask]["value"].to_list(),
                        alpha=0.5,
                        s=0.1,
                    )
                elif x_idx == y_idx:
                    axis.hist(mcmc_params[chain][x_param_mask]["value"].to_list())
                else:
                    axis.axis("off")

                # Set labels
                if y_idx == 0:
                    axis.set_ylabel(
                        x_param_name[:label_chars], rotation=0, fontsize=label_font_size
                    )
                if x_idx == len(parameters) - 1:
                    axis.set_xlabel(y_param_name[:label_chars], fontsize=label_font_size)

    # Save
    plotter.save_figure(fig, filename="parameter_correlation_matrix", dpi_request=dpi_request)


def plot_param_vs_param(
        plotter: Plotter,
        mcmc_params: List[pd.DataFrame],
        mcmc_tables: List[pd.DataFrame],
        parameters: list,
        burn_in: int,
        style: str,
        bins: int,
        label_font_size: int,
        label_chars: int,
        dpi_request: int,
        label_param_string=True,
        show_ticks=False,
        file_name="parameter_correlation_matrix",
        tight_layout=False,
        short_label=False,
):
    """
    Plot the parameter correlation matrices for each parameter combination.
    """

    # split tables by chain
    mcmc_params_list, mcmc_tables_list = split_mcmc_outputs_by_chain(mcmc_params, mcmc_tables)

    # Prelims
    fig, axes, _, _, _, _ = plotter.get_figure(n_panels=len(parameters) ** 2)
    row_data, col_data = {}, {}
    if tight_layout:
        fig.tight_layout(pad=0.)

    # Get x and y data separately and collate up over the chains
    for row_idx, row_param_name in enumerate(parameters):
        row_data[row_param_name] = []
        for i_chain in range(len(mcmc_params_list)):
            values = get_posterior([mcmc_params_list[i_chain]], [mcmc_tables_list[i_chain]], row_param_name, burn_in)
            row_data[row_param_name] += values[row_param_name].to_list()
    for col_idx, col_param_name in enumerate(parameters):
        col_data[col_param_name] = []
        for i_chain in range(len(mcmc_params_list)):
            values = get_posterior([mcmc_params_list[i_chain]], [mcmc_tables_list[i_chain]], col_param_name, burn_in)
            col_data[col_param_name] += values[col_param_name].to_list()

    # Loop over parameter combinations
    for row_idx, row_param_name in enumerate(parameters):
        for col_idx, col_param_name in enumerate(parameters):

            axis = axes[row_idx, col_idx]
            if not show_ticks:
                axis.xaxis.set_ticks([])
                axis.yaxis.set_ticks([])
            else:
                axis.tick_params(labelsize=4)

            # Plot
            if row_idx > col_idx:
                if style == "Scatter":
                    axis.scatter(
                        col_data[col_param_name],
                        row_data[row_param_name],
                        alpha=0.5,
                        s=0.1,
                        color="k",
                    )
                elif style == "KDE":
                    sns.kdeplot(
                        col_data[col_param_name],
                        row_data[row_param_name],
                        ax=axis,
                        shade=True,
                        levels=5,
                        lw=1.0,
                    )
                else:
                    axis.hist2d(col_data[col_param_name], row_data[row_param_name], bins=bins)
            elif row_idx == col_idx:
                axis.hist(
                    row_data[row_param_name],
                    color=[0.2, 0.2, 0.6] if style == "Shade" else "k",
                    bins=bins,
                )
                axis.yaxis.set_ticks([])
            else:
                axis.axis("off")

            # Axis labels (these have to be reversed for some reason)
            x_param_label = col_param_name if label_param_string else str(col_idx + 1)
            y_param_label = row_param_name if label_param_string else str(row_idx + 1)
            if row_idx == len(parameters) - 1:
                axis.set_xlabel(
                    get_plot_text_dict(x_param_label, get_short_text=short_label), fontsize=label_font_size, labelpad=3,
                )
            if col_idx == 0:
                axis.set_ylabel(
                    get_plot_text_dict(y_param_label, get_short_text=short_label), fontsize=label_font_size
                )

    # Save
    plotter.save_figure(fig, filename=file_name, dpi_request=dpi_request)


def plot_all_params_vs_loglike(
    plotter: Plotter,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    burn_in: int,
    title_font_size: int,
    label_font_size: int,
    capitalise_first_letter: bool,
    dpi_request: int,
    posterior: bool,
):

    # Except not the dispersion parameters - only the epidemiological ones
    parameters = [
        param
        for param in mcmc_params[0].loc[:, "name"].unique().tolist()
        if "dispersion_param" not in param
    ]

    fig, axes, _, n_rows, n_cols, indices = plotter.get_figure(
        len(parameters), share_xaxis=False, share_yaxis=True
    )

    for i in range(n_rows * n_cols):
        axis = axes[indices[i][0], indices[i][1]]

        if i < len(parameters):
            param_name = parameters[i]
            plot_param_vs_loglike(mcmc_tables, mcmc_params, param_name, burn_in, axis, posterior)
            axis.set_title(
                get_plot_text_dict(param_name, capitalise_first_letter=capitalise_first_letter),
                fontsize=title_font_size,
            )
            if indices[i][0] == n_rows - 1:
                x_label = "Iterations" if capitalise_first_letter else "iterations"
                axis.set_xlabel(x_label, fontsize=label_font_size)
            pyplot.setp(axis.get_yticklabels(), fontsize=label_font_size)
            pyplot.setp(axis.get_xticklabels(), fontsize=label_font_size)

        else:
            axis.axis("off")

    fig.tight_layout()
    plotter.save_figure(fig, filename=f"all_posteriors", dpi_request=dpi_request)


def sample_outputs_for_calibration_fit(
    output_name: str,
    mcmc_tables: List[pd.DataFrame],
    do_tables: List[pd.DataFrame],
    burn_in: int,
):
    assert len(mcmc_tables) == len(do_tables)
    mcmc_df = db.load.append_tables(mcmc_tables)
    do_df = db.load.append_tables(do_tables)

    # Determine max chain length, throw away first half of that if no burn-in request
    discard_point = mcmc_df["run"].max() // 2 if burn_in == 0 else burn_in
    mcmc_df = mcmc_df[mcmc_df["run"] >= discard_point]

    # Choose runs with probability proportional to their weights.
    weights = mcmc_df["weight"].tolist()
    run_choices = list(zip(mcmc_df["chain"].tolist(), mcmc_df["run"].tolist()))
    num_chosen = 20 * len(mcmc_tables)
    chosen_runs = choices(run_choices, weights=weights, k=num_chosen)

    outputs = []
    for chain, run in chosen_runs:
        mask = (do_df["run"] == run) & (do_df["chain"] == chain)
        times = do_df[mask]["times"]
        values = do_df[mask][output_name]
        outputs.append([times, values])

    # Find MLE run
    mle_df = db.process.find_mle_run(mcmc_df)
    run = mle_df["run"].iloc[0]
    chain = mle_df["chain"].iloc[0]
    # Automatically use the MLE run as the last chosen run
    mask = (do_df["run"] == run) & (do_df["chain"] == chain)
    times = do_df[mask]["times"]
    values = do_df[mask][output_name]
    outputs.append([times, values])

    return outputs


def plot_multi_output_single_run(
    plotter: Plotter,
    mcmc_tables: List[pd.DataFrame],
    calib_dir_path,
    chosen_outputs: List[str],
    selected_scenarios: List[int],
    run_id: str,
    x_low: int,
    x_up: int,
    is_legend: bool,
    n_xticks: int,
    title_font_size: int,
    label_font_size: int,
    dpi_request=300,
    xaxis_date=True
):

    # Except not the dispersion parameters - only the epidemiological ones
    fig, axes, _, n_rows, n_cols, indices = plotter.get_figure(len(chosen_outputs))

    for i in range(n_rows * n_cols):

        if len(chosen_outputs) == 1:
            axis = axes
        elif len(chosen_outputs) == 2:
            axis = axes[i]
        else:
            axis = axes[indices[i][0], indices[i][1]]

        if i < len(chosen_outputs):
            output_name = chosen_outputs[i]
            derived_output_tables = db.load.load_derived_output_tables(calib_dir_path, column=output_name)

            for scenario in selected_scenarios:

                times, values = get_output_from_run_id(output_name, mcmc_tables, derived_output_tables, run_id, scenario)

                # Plot the prior
                linestyle = 'dotted' if scenario == 0 else 'solid'
                axis.plot(times, values, color=COLOR_THEME[scenario], linestyle=linestyle)

            axis.set_title(
                get_plot_text_dict(output_name),
                fontsize=title_font_size,
            )
            pyplot.setp(axis.get_yticklabels(), fontsize=label_font_size)
            pyplot.setp(axis.get_xticklabels(), fontsize=label_font_size)

            if xaxis_date:
                change_xaxis_to_date(axis, REF_DATE)

            axis.set_xlim((x_low, x_up))

        else:
            axis.axis("off")

    file_name = "multi_scenario_single_run"
    fig.tight_layout()
    plotter.save_figure(fig, filename=file_name, dpi_request=dpi_request)


def get_output_from_run_id(
    output_name: str,
    mcmc_tables: List[pd.DataFrame],
    do_tables: List[pd.DataFrame],
    run_id: str,
    scenario: int,
):
    assert len(mcmc_tables) == len(do_tables)
    mcmc_df = db.load.append_tables(mcmc_tables)
    do_df = db.load.append_tables(do_tables)

    if run_id in ["mle", "MLE"]:
        # Find MLE run
        mle_df = db.process.find_mle_run(mcmc_df)
        i_run = mle_df["run"].iloc[0]
        chain_id = mle_df["chain"].iloc[0]
    else:
        chain_id, i_run = run_id.split("_")

    mask = (do_df["run"] == i_run) & (do_df["chain"] == chain_id) & (do_df["scenario"] == scenario)
    times = do_df[mask]["times"]
    values = do_df[mask][output_name]

    return [times, values]


def plot_calibration(axis, output, outputs, targets, is_logscale, ref_date=REF_DATE):
    # Track the maximum value being plotted
    label_font_size = 8

    max_value = 0.0
    for times, values in outputs:
        axis.plot(times, values)
        if len(values) > 0:
            max_value = max(values) if max(values) > max_value else max_value

    # Mark the MLE run with a dotted line
    axis.plot(outputs[-1][0], outputs[-1][1], linestyle=(0, (1, 3)), color="black", linewidth=3)

    # Add plot targets
    output_config = {"output_key": output, "values": [], "times": []}
    for t in targets.values():
        if t["output_key"] == output:
            output_config = t

    values = output_config["values"]
    target_times = output_config["times"]

    n_pre = len([t for t in target_times if t < times.iloc[0]])
    values = values[n_pre:]
    target_times = target_times[n_pre:]

    _plot_targets_to_axis(axis, values, target_times, on_uncertainty_plot=False)

    # Find upper limit for y-axis
    if values:
        upper_buffer = 2.0
        max_target = max(values)
        upper_ylim = (
            max_value if max_value < max_target * upper_buffer else max_target * upper_buffer
        )
    else:
        upper_ylim = max_value

    if is_logscale:
        axis.set_yscale("log")
    else:
        axis.set_ylim([0.0, upper_ylim])

    # Sort out x-axis
    if output == "proportion_seropositive":
        axis.yaxis.set_major_formatter(mtick.PercentFormatter(1, symbol="", decimals=2))
        axis.set_ylabel("percentage", fontsize=label_font_size)
    axis.tick_params(axis="x", labelsize=label_font_size)
    axis.tick_params(axis="y", labelsize=label_font_size)

    change_xaxis_to_date(axis, ref_date, rotation=0)
    axis.set_xlim([times.iloc[0], times.iloc[-1]])

    return axis


def plot_calibration_fit(
    plotter: Plotter,
    output_name: str,
    outputs: list,
    targets,
    is_logscale=False,
):
    fig, axis, _, _, _, _ = plotter.get_figure()
    plot_calibration(axis, output_name, outputs, targets, is_logscale)
    if is_logscale:
        filename = f"calibration-fit-{output_name}-logscale"
        title_text = f"{output_name} (logscale)"
    else:
        filename = f"calibration-fit-{output_name}"
        title_text = f"{output_name}"
    plotter.save_figure(fig, filename=filename, title_text=title_text)


def plot_cdr_curves(
    plotter: Plotter,
    times,
    detected_proportion,
    end_date,
    rotation,
    start_date=1.0,
    alpha=1.0,
    line_width=0.7,
):
    """
    Plot a single set of CDR curves to a one-panel figure
    """
    fig, axis, _, _, _, _ = plotter.get_figure()
    axis = plot_cdr_to_axis(axis, times, detected_proportion, alpha=alpha, line_width=line_width)
    axis.set_ylabel("proportion symptomatic cases passively detected", fontsize=10)
    tidy_cdr_axis(axis, rotation, start_date, end_date)
    plotter.save_figure(fig, filename=f"cdr_curves")


def plot_multi_cdr_curves(
    plotter: Plotter, times, detected_proportions, start_date, end_date, rotation, regions
):
    """
    Plot multiple sets of CDR curves onto a multi-panel figure
    """
    fig, axes, _, n_rows, n_cols, indices = plotter.get_figure(n_panels=len(regions))

    # Loop over models and plot
    for i_region in range(n_rows * n_cols):
        axis = axes[indices[i_region][0], indices[i_region][1]]
        if i_region < len(regions):
            axis = plot_cdr_to_axis(axis, times, detected_proportions[i_region])
            tidy_cdr_axis(axis, rotation, start_date, end_date)
            axis.set_title(get_plot_text_dict(regions[i_region]))
        else:
            axis.axis("off")

    fig.tight_layout()
    plotter.save_figure(fig, filename=f"multi_cdr_curves")


def plot_cdr_to_axis(axis, times, detected_proportions, alpha=1.0, line_width=0.7):
    """
    Plot a set of CDR curves to an axis
    """

    for i_curve in range(len(detected_proportions)):
        axis.plot(
            times,
            [detected_proportions[i_curve](i_time) for i_time in times],
            color="k",
            linewidth=line_width,
            alpha=alpha,
        )
    return axis


def tidy_cdr_axis(axis, rotation, start_date, end_date):
    """
    Tidy up a plot axis in the same way for both the two previous figures
    """
    change_xaxis_to_date(axis, ref_date=REF_DATE, rotation=rotation)
    axis.set_xlim([start_date, end_date])
    axis.set_ylim([0.0, 1.0])
    return axis


def plot_multi_fit(
    plotter: Plotter,
    output_names: list,
    outputs: dict,
    targets,
    is_logscale=False,
    title_font_size=8,
    label_font_size=8,
    dpi_request=300,
    capitalise_first_letter=False,
):

    fig, axes, _, n_rows, n_cols, indices = plotter.get_figure(len(output_names), share_xaxis=True)

    for i_output in range(n_rows * n_cols):
        if i_output < len(output_names):
            output = output_names[i_output]
            axis = plot_calibration(
                axes[indices[i_output][0], indices[i_output][1]],
                output,
                outputs[output],
                targets,
                is_logscale,
            )
            change_xaxis_to_date(axis, REF_DATE, rotation=0)
            axis.set_title(
                get_plot_text_dict(output, capitalise_first_letter=capitalise_first_letter),
                fontsize=title_font_size,
            )
            filename = f"calibration-fit-{output}"
        else:
            axes[indices[i_output][0], indices[i_output][1]].axis("off")

    fig.tight_layout()
    plotter.save_figure(fig, filename=filename, dpi_request=dpi_request)


def _overwrite_non_accepted_mcmc_runs(mcmc_tables: List[pd.DataFrame], column_name: str):
    """
    Count non-accepted rows in a MCMC trace as the last accepted row.
    Modifies mcmc_tables in-place.
    """
    for table_df in mcmc_tables:
        prev_val = None
        for idx in range(len(table_df)):
            if table_df.at[idx, "accept"] == 1:
                prev_val = table_df.at[idx, column_name]
            else:
                table_df.at[idx, column_name] = prev_val
