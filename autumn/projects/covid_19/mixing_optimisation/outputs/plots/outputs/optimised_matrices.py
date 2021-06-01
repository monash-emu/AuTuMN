import copy
import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot

from apps import covid_19
from autumn.projects.covid_19.mixing_optimisation.constants import OPTI_REGIONS, PHASE_2_START_TIME
from autumn.projects.covid_19.mixing_optimisation.mixing_opti import (
    DURATIONS,
    MODES,
    OBJECTIVES,
    build_params_for_phases_2_and_3,
)
from autumn.projects.covid_19.mixing_optimisation.write_scenarios import (
    read_decision_vars,
    read_opti_outputs,
)
from autumn.utils.params import merge_dicts
from autumn.settings import BASE_PATH

FIGURE_PATH = os.path.join(
    BASE_PATH,
    "apps",
    "covid_19",
    "mixing_optimisation",
    "outputs",
    "plots",
    "outputs",
    "figures",
    "optimised_matrices",
)


def main():
    # Reset pyplot style
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.pyplot.style.use("ggplot")
    opti_output_filename = "opti_outputs.csv"
    opti_outputs_df = read_opti_outputs(opti_output_filename)

    for mode in MODES:
        print(mode)
        optimised_matrices = {}
        original_matrices = {}
        for country in OPTI_REGIONS:
            print(country)
            original_matrices[country] = get_optimised_mixing_matrix(
                country, [1] * 16, MODES[0], DURATIONS[0]
            )
            optimised_matrices[country] = {}
            for duration in DURATIONS:
                print(duration)
                optimised_matrices[country][duration] = {}
                for objective in OBJECTIVES:
                    print(objective)
                    decision_vars = read_decision_vars(
                        opti_outputs_df, country, mode, duration, objective
                    )
                    optimised_matrices[country][duration][objective] = get_optimised_mixing_matrix(
                        country, decision_vars, mode, duration
                    )

        plot_multicountry_matrices(original_matrices, optimised_matrices, mode)


def get_optimised_mixing_matrix(country, decision_vars, mode, duration):
    if decision_vars is None:
        return np.ones((16, 16))

    app_region = covid_19.app.get_region(country)
    build_model = app_region.build_model
    params = copy.deepcopy(app_region.params)

    # Create and run scenario 1
    sc_1_params_update = build_params_for_phases_2_and_3(
        decision_vars, params["default"]["elderly_mixing_reduction"], duration, mode
    )
    sc_1_params = merge_dicts(sc_1_params_update, params["default"])
    params["scenarios"][1] = sc_1_params
    scenario_1 = Scenario(build_model, idx=0, params=params)

    print(
        "WARNING: Make sure to comment the line of code where integration is called in scenairos.py !!!!!!!!!!!!!!"
    )
    scenario_1.run()

    sc_1_model = scenario_1.model
    time_matrix_called = PHASE_2_START_TIME + 30  # could be any time during Phase 2

    return sc_1_model.get_mixing_matrix(time_matrix_called)


def plot_mixing_matrix_opti(matrix, axis, fig, vmax, include_legend):

    axis.matshow(matrix)
    vmin = 0.0
    # vmax_all = np.amax(self.model_runner.model_diagnostics[scenario]['contact_matrices'][key]['all'])
    # vmax = np.amax(self.model_runner.model_diagnostics[scenario]['contact_matrices'][key][location])

    pyplot.gca().xaxis.tick_bottom()

    with pyplot.style.context(("seaborn-dark")):
        im = axis.imshow(matrix.transpose(), cmap="hot", origin="lower", vmin=vmin, vmax=vmax)

        if include_legend:
            cbar = fig.colorbar(im, ax=axis)
            cbar.ax.set_ylabel("n contacts per day", rotation=90, va="bottom", fontsize=15)

            cbar.ax.yaxis.set_label_coords(4.0, 0.5)

        ticks = [i - 0.5 for i in range(16)]
        labs = [str(int(i * 5.0)) for i in range(16)]

        axis.set_xticks(ticks)
        axis.set_xticklabels(labs)
        axis.set_xlabel("index age")

        axis.set_yticks(ticks)
        axis.set_yticklabels(labs)
        axis.set_ylabel("contact age")


def plot_multicountry_matrices(original_matrices, optimised_matrices, mode, include_config=True):

    fig_h = 23 if not include_config else 25
    heights = [1, 6, 6, 6, 6, 6, 6] if not include_config else [2, 1, 6, 6, 6, 6, 6, 6]
    pyplot.rcParams["axes.grid"] = False

    fig = pyplot.figure(constrained_layout=True, figsize=(22, fig_h))  # (w, h)

    widths = [1, 5, 5, 5, 5, 5]
    spec = fig.add_gridspec(
        ncols=6, nrows=len(heights), width_ratios=widths, height_ratios=heights, hspace=0, wspace=0
    )

    output_names = [
        "unmitigated",
        "optimising deaths (6 mo.)",
        "optimising deaths (12 mo.)",
        "optimising YLLs (6 mo.)",
        "optimising YLLs (6 mo.)",
    ]

    countries = ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    text_size = 23

    for i, country in enumerate(countries):
        i_grid = i + 2 if include_config else i + 1
        j = 0
        vmax = np.max(original_matrices[country])
        for objective in ["original", "deaths", "yoll"]:
            for duration in DURATIONS:
                j += 1
                if objective == "original":
                    matrix = original_matrices[country]
                else:
                    matrix = optimised_matrices[country][duration][objective]
                ax = fig.add_subplot(spec[i_grid, j])
                include_legend = True if j == 5 else False
                plot_mixing_matrix_opti(matrix, ax, fig, vmax, include_legend)

                if i == 0:
                    i_grid_output_labs = 1 if include_config else 0
                    ax = fig.add_subplot(spec[i_grid_output_labs, j])
                    ax.text(
                        0.5,
                        0.5,
                        output_names[j - 1],
                        fontsize=text_size,
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                    ax.axis("off")

                if objective == "original":
                    break

        ax = fig.add_subplot(spec[i_grid, 0])
        ax.text(
            0.5,
            0.5,
            country_names[i],
            rotation=90,
            fontsize=text_size,
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")

    if include_config:
        ax = fig.add_subplot(spec[0, :])
        config_label = "Optimisation by " + mode[3:]
        ax.text(
            0.0,
            0.5,
            config_label,
            fontsize=text_size,
            horizontalalignment="left",
            verticalalignment="center",
        )
        ax.axis("off")

    pyplot.rcParams["font.family"] = "Times New Roman"

    filename = "optimal_matrices_" + mode
    png_path = os.path.join(FIGURE_PATH, f"{filename}.png")
    pdf_path = os.path.join(FIGURE_PATH, f"{filename}.pdf")

    pyplot.savefig(png_path, dpi=300)
    pyplot.savefig(pdf_path)


if __name__ == "__main__":
    main()
