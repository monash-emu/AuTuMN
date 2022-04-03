import random

import streamlit as st

from autumn.models.covid_19.detection import find_cdr_function_from_test_data
from autumn.tools import inputs, plots
from autumn.utils.params import load_params

PLOT_FUNCS = {}


def multi_country_cdr(
    plotter, calib_dir_path, mcmc_tables, mcmc_params, targets, app_name, region_name
):
    """
    Code taken directly from the fit calibration file at this stage.
    """

    from dash.dashboards.calibration_results.plots import get_cdr_constants

    param_name = "testing_to_detection.assumed_cdr_parameter"
    start_date = st.sidebar.slider("Start date", 1, 365, 1)
    end_date = st.sidebar.slider("End date", 1, 365, 275)
    samples = st.sidebar.slider("Samples", 1, 200, 10)
    label_rotation = st.sidebar.slider("Label rotation", 0, 90, 0)
    detected_proportions = []

    # Get data for plotting
    for i_region in range(len(region_name)):

        # Extract parameters relevant to this function
        region = region_name[i_region].replace("-", "_")
        params = load_params(app_name, region)
        (
            iso3,
            testing_year,
            assumed_tests_parameter,
            smoothing_period,
            agegroup_params,
            time_params,
            times,
            agegroup_strata,
        ) = get_cdr_constants(params["default"])
        pop_region = params["default"]["population"]["region"]
        pop_year = params["default"]["population"]["year"]

        # Collate parameters into one structure
        testing_to_detection_values = []
        for i_chain in range(len(mcmc_params)):
            param_mask = mcmc_params[i_chain][0]["name"] == param_name
            testing_to_detection_values += mcmc_params[i_chain][0]["value"][param_mask].tolist()
        sampled_test_to_detect_vals = random.sample(testing_to_detection_values, samples)

        # Get CDR function - needs to be done outside of autumn, because it is importing from the apps
        testing_year = 2020 if iso3 == "AUS" else pop_year
        testing_pops = inputs.get_population_by_agegroup(
            agegroup_strata, iso3, pop_region, year=testing_year
        )

        detected_proportions.append([])
        for assumed_cdr_parameter in sampled_test_to_detect_vals:
            detected_proportions[i_region].append(
                find_cdr_function_from_test_data(
                    assumed_tests_parameter,
                    assumed_cdr_parameter,
                    smoothing_period,
                    iso3,
                    testing_pops,
                    subregion=testing_region,
                )
            )
    plots.calibration.plots.plot_multi_cdr_curves(
        plotter, times, detected_proportions, start_date, end_date, label_rotation, region_name
    )


PLOT_FUNCS["Multi-country CDR"] = multi_country_cdr
