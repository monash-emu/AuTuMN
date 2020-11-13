import streamlit as st
from autumn import plots
from dash.dashboards.calibration_results.plots import get_uncertainty_data
from math import ceil
import matplotlib.pyplot as pyplot
from autumn.tool_kit.params import load_params
from autumn.tool_kit.scenarios import get_model_times_from_inputs

PLOT_FUNCS = {}


def multi_country_cdr(
    plotter, calib_dir_path, mcmc_tables, mcmc_params, targets, app_name, region_name
):
    """
    Code taken directly from the fit calibration file at this stage.
    """

    # Set up interface
    fig, axes, _, n_rows, n_cols, indices = plotter.get_figure(len(region_name), share_xaxis=True)

    # This should remain fixed
    param_name = "testing_to_detection.assumed_cdr_parameter"
    end_date = st.sidebar.slider("End date", 1, 365, 275)
    samples = st.sidebar.slider("Samples", 1, 200, 10)
    label_rotation = st.sidebar.slider("Label rotation", 0, 90, 0)

    # Get data for plotting
    for i_region in range(len(region_name)):
        region = region_name[i_region].replace("-", "_")

        # Extract parameters relevant to this function
        params = load_params(app_name, region)
        default_params = params["default"]

        iso3 = default_params["country"]["iso3"]
        testing_year = default_params["population"]["year"]
        assumed_tests_parameter = default_params["testing_to_detection"]["assumed_tests_parameter"]
        smoothing_period = default_params["testing_to_detection"]["smoothing_period"]
        agegroup_params = default_params["age_stratification"]
        time_params = default_params["time"]

        # Derive times and age group breaks as the model does
        times = get_model_times_from_inputs(
            time_params["start"], time_params["end"], time_params["step"]
        )
        agegroup_strata = [
            str(s) for s in range(0, agegroup_params["max_age"], agegroup_params["age_step_size"])
        ]

        st.write(times)
        st.write(agegroup_strata)


PLOT_FUNCS["Multi-country CDR"] = multi_country_cdr
