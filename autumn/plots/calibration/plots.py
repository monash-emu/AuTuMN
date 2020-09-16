"""
Calibration plots
"""
import os
import copy
import logging
from math import log
from typing import List, Tuple, Callable
from random import choices

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot
from scipy import stats

from autumn import db
from autumn.calibration.utils import calculate_prior, raise_error_unsupported_prior
from autumn.plots.plotter import Plotter, COLOR_THEME

logger = logging.getLogger(__name__)


def plot_acceptance_ratio(plotter: Plotter, mcmc_tables: List[pd.DataFrame]):
    """
    Plot the prameter traces for each MCMC run.
    """
    fig, axis, _, _, _ = plotter.get_figure()
    mcmc_df = db.process.append_tables(mcmc_tables)
    chains = mcmc_df["chain"].unique().tolist()
    for chain in chains:
        df = mcmc_df[mcmc_df["chain"] == chain]
        count = 0
        total = 0
        ratios = []
        for accept in df["accept"]:
            total += 1
            if accept:
                count += 1

            ratios.append(count / total)

        axis.plot(ratios, alpha=0.8, linewidth=0.7)

    axis.set_ylabel("Acceptance Ratio")
    axis.set_xlabel("MCMC iterations")
    plotter.save_figure(fig, filename=f"acceptance_ratio", title_text=f"Acceptance Ratio")


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
            x=prior_dict["distri_mean"], ymin=0, ymax=100 * max(y_values), linewidth=1, color="red",
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


def plot_mcmc_parameter_trace(plotter: Plotter, mcmc_params: List[pd.DataFrame], param_name: str):
    """
    Plot the prameter traces for each MCMC run.
    """
    fig, axis, _, _, _ = plotter.get_figure()
    for idx, table_df in enumerate(mcmc_params):
        param_mask = table_df["name"] == param_name
        param_df = table_df[param_mask]
        axis.plot(param_df["run"], param_df["value"], alpha=0.8, linewidth=0.7)

    axis.set_ylabel(param_name)
    axis.set_xlabel("MCMC iterations")
    plotter.save_figure(fig, filename=f"{param_name}-traces", title_text=f"{param_name}-traces")


def plot_loglikelihood_trace(plotter: Plotter, mcmc_tables: List[pd.DataFrame], burn_in=0):
    """
    Plot the loglikelihood traces for each MCMC run.
    """
    fig, axis, _, _, _ = plotter.get_figure()

    for idx, table_df in enumerate(mcmc_tables):
        accept_mask = table_df["accept"] == 1
        table_df[accept_mask].loglikelihood.plot.line(ax=axis, alpha=0.8, linewidth=0.7)

    axis.set_ylabel("Loglikelihood")
    axis.set_xlabel("MCMC iterations")

    if burn_in:
        axis.axvline(x=burn_in, color=COLOR_THEME[1], linestyle="dotted")
        y_min = min(table_df.loglikelihood[burn_in:])
        y_max = max(table_df.loglikelihood[burn_in:])
        axis.set_ylim((y_min - 0.2 * (y_max - y_min), y_max + 0.2 * (y_max - y_min)))

    plotter.save_figure(fig, filename="loglikelihood-traces", title_text="loglikelihood-traces")


def plot_burn_in(plotter: Plotter, num_iters: int, burn_in: int):
    """
    Plot the trade off been num iters and burn-in for MCMC runs.
    """
    fig, axis, _, _, _ = plotter.get_figure()

    def floor(n):
        val = num_iters - n
        return val if val > 0 else 0

    values = [floor(i) for i in range(num_iters)]

    fig, axis, _, _, _ = plotter.get_figure()

    axis.plot(values, color=COLOR_THEME[0])
    axis.set_ylabel("Number iters after burn-in")
    axis.set_xlabel("Burn-in size")
    axis.set_ylim(bottom=-5, top=num_iters)
    axis.set_xlim(left=0, right=num_iters)
    axis.axvline(x=burn_in, color=COLOR_THEME[1], linestyle="dotted")
    axis.axhline(y=num_iters - burn_in, color=COLOR_THEME[1], linestyle="dotted")
    plotter.save_figure(fig, filename="burn-in", title_text="burn-in")


def plot_posterior(
    plotter: Plotter, mcmc_params: List[pd.DataFrame], param_name: str, num_bins: int
):
    """
    Plots the posterior distribution of a given parameter in a histogram.
    """
    vals_df = None
    for table_df in mcmc_params:
        param_mask = table_df["name"] == param_name
        table_vals = table_df[param_mask].value
        if vals_df is not None:
            vals_df = vals_df.append(table_vals)
        else:
            vals_df = table_vals

    fig, axis, _, _, _ = plotter.get_figure()
    vals_df.hist(bins=num_bins, ax=axis)
    plotter.save_figure(
        fig, filename=f"{param_name}-posterior", title_text=f"{param_name} posterior"
    )


def plot_loglikelihood_vs_parameter(
    plotter: Plotter,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    param_name: str,
):
    """
    Plots the loglikelihood against parameter values.
    """
    fig, axis, _, _, _ = plotter.get_figure()
    for mcmc_df, param_df in zip(mcmc_tables, mcmc_params):
        df = param_df.merge(mcmc_df, on=["run", "chain"])
        mask = (df["accept"] == 1) & (df["name"] == param_name)
        df = df[mask]
        param_values = df["value"]
        loglikelihood_values = [-log(-v) for v in df["loglikelihood"]]
        axis.plot(param_values, loglikelihood_values, ".")

    axis.set_xlabel(param_name)
    axis.set_ylabel("-log(-loglikelihood)")
    plotter.save_figure(
        fig,
        filename=f"likelihood-against-{param_name}",
        title_text=f"likelihood against {param_name}",
    )


def sample_outputs_for_calibration_fit(
    output_name: str, mcmc_tables: List[pd.DataFrame], do_tables: List[pd.DataFrame],
):
    assert len(mcmc_tables) == len(do_tables)
    mcmc_df = db.process.append_tables(mcmc_tables)
    do_df = db.process.append_tables(do_tables)

    # Determine max chain length, throw away first half of that
    max_run = mcmc_df["run"].max()
    half_max = max_run // 2
    mcmc_df = mcmc_df[mcmc_df["run"] >= half_max]

    # Choose runs with probability proprotional to their weights.
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


def plot_calibration_fit(
    plotter: Plotter, output_name: str, outputs: list, targets, is_logscale=False,
):
    fig, axis, _, _, _ = plotter.get_figure()

    # Track the maximum value being plotted
    max_value = 0.0

    for times, values in outputs:
        axis.plot(times, values)
        max_value = max(values) if max(values) > max_value else max_value

    # Mark the MLE run with a dotted line
    axis.plot(outputs[-1][0], outputs[-1][1], linestyle=(0, (1, 3)), color="black", linewidth=3)

    # Add plot targets
    output_config = {"output_key": output_name, "values": [], "times": []}
    for t in targets.values():
        if t["output_key"] == output_name:
            output_config = t

    values = output_config["values"]
    times = output_config["times"]
    _plot_targets_to_axis(axis, values, times, on_uncertainty_plot=False)

    # Find upper limit for y-axis
    if values:
        upper_buffer = 2.0
        max_target = max(values)
        upper_ylim = (
            max_value if max_value < max_target * upper_buffer else max_target * upper_buffer
        )
    else:
        upper_ylim = max_value

    # Plot outputs
    axis.set_xlabel("time")
    axis.set_ylabel(output_name)
    if is_logscale:
        axis.set_yscale("log")
    else:
        axis.set_ylim([0.0, upper_ylim])

    if is_logscale:
        filename = f"calibration-fit-{output_name}-logscale"
        title_text = f"Calibration fit for {output_name} (logscale)"
    else:
        filename = f"calibration-fit-{output_name}"
        title_text = f"Calibration fit for {output_name}"

    plotter.save_figure(fig, filename=filename, title_text=title_text)


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


def _plot_targets_to_axis(axis, values: List[float], times: List[int], on_uncertainty_plot=False):
    """
    Plot output value calibration targets as points on the axis.
    """
    assert len(times) == len(values), "Targets have inconsistent length"
    # Plot a single point estimate
    if on_uncertainty_plot:
        axis.scatter(times, values, marker="o", color="black", s=10)
    else:
        axis.scatter(times, values, marker="o", color="red", s=30, zorder=999)
        axis.scatter(times, values, marker="o", color="white", s=10, zorder=999)
