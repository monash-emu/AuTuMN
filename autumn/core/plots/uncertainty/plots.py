"""
Plotting projection uncertainty.
"""
import datetime
import logging
from math import ceil
from typing import List

import matplotlib.ticker as mtick
import pandas as pd
from matplotlib import pyplot
from numpy import mean

from autumn.core.plots.plotter import Plotter
from autumn.core.plots.utils import (
    ALPHAS,
    COLORS,
    REF_DATE,
    _apply_transparency,
    _plot_targets_to_axis,
    change_xaxis_to_date,
    get_plot_text_dict,
    add_vertical_lines_to_plot,
    add_horizontal_lines_to_plot,
)

logger = logging.getLogger(__name__)


def plot_timeseries_with_uncertainty(

        plotter: Plotter,
        uncertainty_df: pd.DataFrame,
        output_name: str,
        scenario_idxs: List[int],
        targets: dict,
        is_logscale=False,
        x_low=0.0,
        x_up=1e6,
        axis=None,
        n_xticks=None,
        ref_date=REF_DATE,
        add_targets=True,
        overlay_uncertainty=False,
        title_font_size=12,
        label_font_size=10,
        dpi_request=300,
        capitalise_first_letter=False,
        legend=True,
        requested_x_ticks=None,
        show_title=True,
        ylab=None,
        x_axis_to_date=True,
        start_quantile=0,
        sc_colors=None,
        custom_title=None,
        vlines={},
        hlines={},

):
    """
    Plots the uncertainty timeseries for one or more scenarios.
    Also plots any calibration targets that are provided.
    """

    single_panel = axis is None
    if single_panel:
        fig, axis, _, _, _, _ = plotter.get_figure()

    n_scenarios_to_plot = len(scenario_idxs)

    if sc_colors is None:
        n_scenarios_to_plot = min([len(scenario_idxs), len(COLORS)])
        colors = _apply_transparency(COLORS[:n_scenarios_to_plot], ALPHAS[:n_scenarios_to_plot])
    else:
        colors = sc_colors

    # If we can map particular scenario IDs to particular colours, we should try to do so
    # use_index_color_map = True

    # max_sc_idx = max(scenario_idxs[:n_scenarios_to_plot])
    # if max_sc_idx >= len(colors):
    #    use_index_color_map = False

    # Plot each scenario on a single axis
    data_to_return = {}
    for i, scenario_idx in enumerate(scenario_idxs[:n_scenarios_to_plot]):

        # +++
        # Retain these commented blocks just so we can see the original intentions
        # when we come to refactor...
        # if sc_colors is None:
        #    if scenario_idx < len(colors):
        #        scenario_colors = colors[scenario_idx]
        #    else:
        #        scenario_colors = colors[-1]
        # else:
        #    scenario_colors = sc_colors[i]

        # if use_index_color_map:
        #    scenerio_colors = colors[scenario_idx]
        # else:
        #    scenario_colors = colors[i]

        # FIXME:
        # This basically ignores all the other colour mapping functionality, but
        # works reliably; this module really requires a refactor, so it's not worth
        # trying to fix the other bits right now...

        #if scenario_idx !=0:
            scenario_colors = colors[i]

            times, quantiles = _plot_uncertainty(
                axis,
                uncertainty_df,
                output_name,
                scenario_idx,
                x_up,
                x_low,
                scenario_colors,
                overlay_uncertainty=overlay_uncertainty,
                start_quantile=start_quantile,
                zorder=i + 1,
            )

            data_to_return[scenario_idx] = pd.DataFrame.from_dict(quantiles)
            data_to_return[scenario_idx].insert(0, "days from 31/12/2019", times)



    # Add plot targets
    if add_targets:
        values, times = _get_target_values(targets, output_name)
        trunc_values = [v for (v, t) in zip(values, times) if x_low <= t <= x_up]
        trunc_times = [t for (v, t) in zip(values, times) if x_low <= t <= x_up]
        _plot_targets_to_axis(axis, trunc_values, trunc_times, on_uncertainty_plot=True)

    # Sort out x-axis
    if x_axis_to_date:
        change_xaxis_to_date(axis, ref_date, rotation=0)
    axis.tick_params(axis="x", labelsize=label_font_size)
    axis.tick_params(axis="y", labelsize=label_font_size)

    # Add lines with marking text to plots
    add_vertical_lines_to_plot(axis, vlines)
    add_horizontal_lines_to_plot(axis, hlines)

    if output_name == "proportion_seropositive":
        axis.yaxis.set_major_formatter(mtick.PercentFormatter(1, symbol=""))
    if show_title:
        title = custom_title if custom_title else get_plot_text_dict(output_name)
        axis.set_title(title, fontsize=title_font_size)
    if output_name == "proportion_seropositive":
        axis.set_title("recovered percentage", fontsize=title_font_size)

    if requested_x_ticks is not None:
        pyplot.xticks(requested_x_ticks)
    elif n_xticks is not None:
        pyplot.locator_params(axis="x", nbins=n_xticks)

    if is_logscale:
        axis.set_yscale("log")
    elif not (output_name.startswith("rel_diff") or output_name.startswith("abs_diff")):
        axis.set_ylim(ymin=0)

    if ylab is not None:
        axis.set_ylabel(ylab, fontsize=label_font_size)

    if legend:
        pyplot.legend(labels=scenario_idxs)

    if single_panel:
        idx_str = "-".join(map(str, scenario_idxs))
        filename = f"uncertainty-{output_name}-{idx_str}"
        plotter.save_figure(fig, filename=filename, dpi_request=dpi_request)

    return data_to_return


def _plot_uncertainty(
    axis,
    uncertainty_df: pd.DataFrame,
    output_name: str,
    scenario_idx: int,
    x_up: float,
    x_low: float,
    colors: List[str],
    overlay_uncertainty=True,
    start_quantile=0,
    zorder=1,
    linestyle="solid",
    linewidth=1,
):
    """Plots the uncertainty values in the provided dataframe to an axis"""
    mask = (
        (uncertainty_df["type"] == output_name)
        & (uncertainty_df["scenario"] == scenario_idx)
        & (uncertainty_df["time"] <= x_up)
        & (uncertainty_df["time"] >= x_low)
    )
    df = uncertainty_df[mask]
    times = df.time.unique()[1:]
    quantiles = {}
    quantile_vals = df["quantile"].unique().tolist()
    for q in quantile_vals:
        mask = df["quantile"] == q
        quantiles[q] = df[mask]["value"].tolist()[1:]
    q_keys = sorted([float(k) for k in quantiles.keys()])
    num_quantiles = len(q_keys)
    half_length = num_quantiles // 2
    if overlay_uncertainty:
        for i in range(start_quantile, half_length):
            color = colors[i - start_quantile]
            start_key = q_keys[i]
            end_key = q_keys[-(i + 1)]
            axis.fill_between(
                times, quantiles[start_key], quantiles[end_key], facecolor=color, zorder=zorder
            )

    if num_quantiles % 2:
        q_key = q_keys[half_length]
        axis.plot(times, quantiles[q_key], color=colors[3], zorder=zorder, linestyle=linestyle, linewidth=linewidth)

    return times, quantiles


def plot_multi_output_timeseries_with_uncertainty(
        plotter: Plotter,
        uncertainty_df: pd.DataFrame,
        output_names: str,
        scenarios: list,
        all_targets: dict,
        is_logscale=False,
        x_low=0.0,
        x_up=2000.0,
        n_xticks=None,
        title_font_size=12,
        label_font_size=10,
        file_name="multi_uncertainty",
        max_y_values=(),
        custom_titles=None,
        custom_sup_title=None,
        multi_panel_vlines=(),
        multi_panel_hlines=(),
        overlay_uncertainty=False,
        is_legend=True,
):
    pyplot.style.use("ggplot")
    if len(output_names) * len(scenarios) == 0:
        return
    # pyplot.rcParams.update({'font.size': 15})

    max_n_col = 2
    n_panels = len(output_names)
    n_cols = min(max_n_col, n_panels)
    n_rows = ceil(n_panels / max_n_col)

    fig = pyplot.figure(
        constrained_layout=True,
        figsize=(n_cols * 7, n_rows * 5),  # (w, h)
    )
    spec = fig.add_gridspec(ncols=n_cols, nrows=n_rows)

    i_col, i_row, axes = 0, 0, []
    for i_out, output_name in enumerate(output_names):
        targets = {k: v for k, v in all_targets.items() if v["output_key"] == output_name}

        axes.append(fig.add_subplot(spec[i_row, i_col]))

        assert len(max_y_values) in (0, len(output_names)), "Wrong number of y-values submitted"
        if max_y_values:
            axes[i_out].set_ylim(top=max_y_values[i_out])

        custom_title = custom_titles[i_out] if custom_titles else None
        if multi_panel_vlines:
            assert len(multi_panel_vlines) == len(
                output_names
            ), "Wrong number of vertical line groups submitted for requested panels/outputs"
            vlines = multi_panel_vlines[i_out]
        else:
            vlines = {}

        if multi_panel_hlines:
            assert len(multi_panel_hlines) == len(
                output_names
            ), "Wrong number of horizontal line groups submitted for requested panels/outputs"
            hlines = multi_panel_hlines[i_out]
        else:
            hlines = {}

        plot_timeseries_with_uncertainty(
            plotter,
            uncertainty_df,
            output_name,
            scenarios,
            targets,
            is_logscale,
            x_low,
            x_up,
            axes[i_out],
            n_xticks,
            title_font_size=title_font_size,
            label_font_size=label_font_size,
            custom_title=custom_title,
            vlines=vlines,
            hlines=hlines,
            overlay_uncertainty=overlay_uncertainty,
            legend=is_legend,
        )
        i_col += 1
        if i_col == max_n_col:
            i_col = 0
            i_row += 1

    if custom_sup_title:
        fig.suptitle(custom_sup_title)
    plotter.save_figure(fig, filename=file_name, title_text="")

    # out_dir = "apps/tuberculosis/regions/marshall_islands/figures/calibration_targets/"
    # filename = out_dir + "targets"
    # pyplot.savefig(filename + ".pdf")


def plot_multicountry_timeseries_with_uncertainty(
    plotter: Plotter,
    uncertainty_df: list,
    output_name: str,
    scenarios: list,
    all_targets: dict,
    regions: dict,
    is_logscale=False,
    x_low=0.0,
    x_up=2000.0,
    n_xticks=None,
    title_font_size=12,
    label_font_size=10,
):

    pyplot.style.use("ggplot")
    max_n_col = 2
    n_panels = len(regions)
    n_cols = min(max_n_col, n_panels)
    n_rows = ceil(n_panels / max_n_col)

    fig = pyplot.figure(constrained_layout=True, figsize=(n_cols * 7, n_rows * 5))  # (w, h)
    spec = fig.add_gridspec(ncols=n_cols, nrows=n_rows)

    i_row, i_col = 0, 0
    for i_region, region in regions.items():
        targets = {k: v for k, v in all_targets[i_region].items() if v["output_key"] == output_name}
        ax = fig.add_subplot(spec[i_row, i_col])
        plot_timeseries_with_uncertainty(
            plotter,
            uncertainty_df[i_region],
            output_name,
            scenarios,
            targets,
            is_logscale,
            x_low,
            x_up,
            ax,
            n_xticks,
            title_font_size=title_font_size,
            label_font_size=label_font_size,
        )

        # Uncomment the following code for the custom titles for the Philippines application plot
        # if i_region == 0:
        #     ax.set_title("MHS incorporated", fontsize=title_font_size)
        # elif i_region == 1:
        #     ax.set_title("MHS not incorporated", fontsize=title_font_size)
        ax.set_title(region, fontsize=title_font_size)
        i_col += 1
        if i_col == max_n_col:
            i_col = 0
            i_row += 1

    plotter.save_figure(fig, filename="multi_uncertainty", subdir="outputs", title_text="")


def plot_age_seroprev_to_axis(
    uncertainty_df,
    scenario_id,
    time,
    axis,
    requested_quantiles,
    ref_date,
    name,
    add_date_as_title=True,
    add_ylabel=True,
    credible_range=95,
):

    mask = (uncertainty_df["scenario"] == scenario_id) & (uncertainty_df["time"] == time)
    df = uncertainty_df[mask]
    quantile_vals = df["quantile"].unique().tolist()
    seroprevalence_by_age = {}
    sero_outputs = [
        output
        for output in df["type"].unique().tolist()
        if "proportion_seropositiveXagegroup_" in output
    ]

    max_value = -10.0
    if len(sero_outputs) == 0:
        axis.text(0.0, 0.5, "Age-specific seroprevalence outputs are not available for this run")
    else:
        for output in sero_outputs:
            output_mask = df["type"] == output
            age = output.split("proportion_seropositiveXagegroup_")[1]
            seroprevalence_by_age[age] = {}
            for q in quantile_vals:
                q_mask = df["quantile"] == q
                seroprevalence_by_age[age][q] = [
                    100.0 * v for v in df[output_mask][q_mask]["value"].tolist()
                ]

        q_keys = requested_quantiles if requested_quantiles else sorted(quantile_vals)
        num_quantiles = len(q_keys)
        half_length = num_quantiles // 2

        x_positions = [float(i) + 2.5 for i in seroprevalence_by_age.keys()]

        lower_q_key = (100.0 - credible_range) / 100.0 / 2.0
        upper_q_key = 1.0 - lower_q_key

        for i, age in enumerate(list(seroprevalence_by_age.keys())):

            axis.plot(
                [x_positions[i], x_positions[i]],
                [seroprevalence_by_age[age][lower_q_key], seroprevalence_by_age[age][upper_q_key]],
                "-",
                color="black",
                lw=1.0,
            )
            max_value = max(max_value, seroprevalence_by_age[age][upper_q_key][0])

            if num_quantiles % 2:
                q_key = q_keys[half_length]
                label = None if i > 0 else "model"
                axis.plot(
                    x_positions[i],
                    seroprevalence_by_age[age][q_key],
                    "o",
                    color="black",
                    markersize=4,
                    label=label,
                )

        axis.xaxis.set_ticks(x_positions)
        axis.set_xticklabels([str(i) for i in range(0, 80, 5)], fontsize=10, rotation=90)

        axis.set_xlabel("age (years)", fontsize=13)
        axis.set_ylim(bottom=0.0)
        if add_ylabel:
            axis.set_ylabel("% previously infected", fontsize=13)
        _date = ref_date + datetime.timedelta(days=time)
        if add_date_as_title:
            axis.set_title(f'{name} {_date.strftime("%d/%m/%Y")}', fontsize=15)

    return axis, max_value, df, seroprevalence_by_age


def plot_seroprevalence_by_age(
    plotter: Plotter,
    uncertainty_df: pd.DataFrame,
    scenario_id: int,
    time: float,
    ref_date=REF_DATE,
    axis=None,
    name="",
    requested_quantiles=None,
):
    single_panel = axis is None
    if single_panel:
        fig, axis, _, _, _, _ = plotter.get_figure()

    axis, max_value, df, seroprevalence_by_age = plot_age_seroprev_to_axis(
        uncertainty_df, scenario_id, time, axis, requested_quantiles, ref_date, name
    )
    if single_panel:
        plotter.save_figure(fig, filename="sero_by_age", subdir="outputs", title_text="")

    overall_seropos_estimates = df[df["type"] == "proportion_seropositive"][
        ["quantile", "value"]
    ].set_index("quantile")

    return max_value, seroprevalence_by_age, overall_seropos_estimates


def plot_vic_seroprevalences(
    plotter: Plotter,
    uncertainty_df: pd.DataFrame,
    scenario_id: int,
    time: float,
    ref_date=REF_DATE,
    name="",
    requested_quantiles=None,
    credible_range=50,
):

    fig, axes, _, _, _, _ = plotter.get_figure(n_panels=2, share_yaxis="all")
    cluster_axis, age_axis = axes
    mask = (uncertainty_df["scenario"] == scenario_id) & (uncertainty_df["time"] == time)
    df = uncertainty_df[mask]
    quantile_vals = df["quantile"].unique().tolist()
    seroprevalence_by_cluster = {}
    sero_outputs = [
        output
        for output in df["type"].unique().tolist()
        if "proportion_seropositiveXcluster_" in output
    ]

    max_value = -10.0
    if len(sero_outputs) == 0:
        cluster_axis.text(
            0.0, 0.5, "Cluster-specific seroprevalence outputs are not available for this run"
        )
    else:
        for output in sero_outputs:
            output_mask = df["type"] == output
            cluster = output.split("proportion_seropositiveXcluster_")[1]
            seroprevalence_by_cluster[cluster] = {}
            for q in quantile_vals:
                q_mask = df["quantile"] == q
                seroprevalence_by_cluster[cluster][q] = [
                    100.0 * v for v in df[output_mask][q_mask]["value"].tolist()
                ]
        q_keys = requested_quantiles if requested_quantiles else sorted(quantile_vals)
        num_quantiles = len(q_keys)
        half_length = num_quantiles // 2

        cluster_names = [
            get_plot_text_dict(i.split("proportion_seropositiveXcluster_")[1]) for i in sero_outputs
        ]

        lower_q_key = (100.0 - credible_range) / 100.0 / 2.0
        upper_q_key = 1.0 - lower_q_key

        x_positions = range(len(seroprevalence_by_cluster))

        for i, cluster in enumerate(list(seroprevalence_by_cluster.keys())):
            cluster_axis.plot(
                [x_positions[i], x_positions[i]],
                [
                    seroprevalence_by_cluster[cluster][lower_q_key],
                    seroprevalence_by_cluster[cluster][upper_q_key],
                ],
                "-",
                color="black",
                lw=1.0,
            )
            cluster_axis.set_xticklabels(cluster_names, fontsize=10, rotation=90)
            max_value = max(max_value, seroprevalence_by_cluster[cluster][upper_q_key][0])

            if num_quantiles % 2:
                q_key = q_keys[half_length]
                label = None if i > 0 else "model"
                cluster_axis.plot(
                    x_positions[i],
                    seroprevalence_by_cluster[cluster][q_key],
                    "o",
                    color="black",
                    markersize=4,
                    label=label,
                )

        cluster_axis.xaxis.set_ticks(x_positions)

        cluster_axis.set_ylim(bottom=0.0)
        cluster_axis.set_ylabel("% previously infected", fontsize=13)
        _date = ref_date + datetime.timedelta(days=time)

    axis, max_value, df, seroprevalence_by_age = plot_age_seroprev_to_axis(
        uncertainty_df,
        scenario_id,
        time,
        age_axis,
        requested_quantiles,
        ref_date,
        name,
        add_date_as_title=False,
        add_ylabel=False,
    )

    plotter.save_figure(fig, filename="sero_by_cluster", subdir="outputs", title_text="")

    overall_seropos_estimates = df[df["type"] == "proportion_seropositive"][
        ["quantile", "value"]
    ].set_index("quantile")

    return max_value, seroprevalence_by_cluster, overall_seropos_estimates


def plot_seroprevalence_by_age_against_targets(
    plotter, uncertainty_df, selected_scenario, serosurvey_data, n_columns
):
    n_surveys = len(serosurvey_data)
    n_rows = ceil(n_surveys / n_columns)

    with pyplot.style.context("ggplot"):
        fig = pyplot.figure(constrained_layout=True, figsize=(n_columns * 7, n_rows * 5))  # (w, h)
        spec = fig.add_gridspec(ncols=n_columns, nrows=n_rows)

        i_row = 0
        i_col = 0
        for survey in serosurvey_data:
            # plot model outputs
            midpoint_time = int(mean(survey["time_range"]))
            ax = fig.add_subplot(spec[i_row, i_col])
            s_name = "" if not "survey_name" in survey else survey["survey_name"]
            max_value, _, _ = plot_seroprevalence_by_age(
                plotter, uncertainty_df, selected_scenario, time=midpoint_time, axis=ax, name=s_name
            )

            # add data
            shift = 0.5
            max_prev = -10.0
            for i, measure in enumerate(survey["measures"]):
                mid_age = mean(measure["age_range"])
                ax.plot(
                    [mid_age + shift, mid_age + shift],
                    [measure["ci"][0], measure["ci"][1]],
                    "-",
                    color="red",
                    lw=1.0,
                )
                label = None if i > 0 else "data"
                ax.plot(mid_age + shift, measure["central"], "o", color="red", ms=4, label=label)
                # ax.axvline(x=measure["age_range"][0], linestyle="--", color='grey', lw=.5)
                max_prev = max(max_prev, measure["ci"][1])

            if i_col + i_row == 0:
                ax.legend(loc="upper right", fontsize=12, facecolor="white")

            ax.set_ylim((0.0, max(max_prev * 1.3, max_value * 1.3)))

            i_col += 1
            if i_col == n_columns:
                i_row += 1
                i_col = 0

        if plotter is not None:
            plotter.save_figure(fig, filename="multi_sero_by_age", subdir="outputs", title_text="")
        else:
            return fig


def _get_target_values(targets: dict, output_name: str):
    """Pulls out values for a given target"""
    output_config = {"values": [], "times": []}
    for t in targets.values():
        if t["output_key"] == output_name:
            output_config = t

    values = output_config["values"]
    times = output_config["times"]
    return values, times
