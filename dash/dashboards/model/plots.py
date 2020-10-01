from datetime import datetime

import streamlit as st

from autumn.plots.plotter import StreamlitPlotter
from autumn.tool_kit.model_register import AppRegion
from autumn.tool_kit.scenarios import Scenario, get_model_times_from_inputs
from autumn.tool_kit.params import load_params
from autumn import plots
from autumn.inputs import get_mobility_data

from apps.covid_19.model.preprocess.mixing_matrix.adjust_location import (
    LocationMixingAdjustment,
    LOCATIONS,
)

BASE_DATE = datetime(2020, 1, 1, 0, 0, 0)
PLOT_FUNCS = {}


def plot_flow_params(
    plotter: StreamlitPlotter, app: AppRegion,
):
    # Assume a COVID model
    model = app.build_model(app.params["default"])
    param_names = sorted(list({f.param_name for f in model.flows}))
    param_name = st.sidebar.selectbox("Select parameter", param_names)
    flows = [f for f in model.flows if f.param_name == param_name]
    is_logscale = st.sidebar.checkbox("Log scale")
    flow_funcs = [f.get_weight_value for f in flows]
    plots.model.plots.plot_time_varying_input(
        plotter, f"flow-params-{param_name}", flow_funcs, model.times, is_logscale
    )

    init_dict = {}
    for f in flows:
        f_name = ""
        src = getattr(f, "source", None)
        dest = getattr(f, "dest", None)
        if src:
            f_name += f"from {src}"
        if dest:
            f_name += f" to {dest}"

        f_name = f_name.strip()
        init_dict[f_name] = f.get_weight_value(0)

    st.write("Values at start time:")
    st.write(init_dict)


PLOT_FUNCS["Flow weights"] = plot_flow_params


def plot_dynamic_inputs(
    plotter: StreamlitPlotter, app: AppRegion,
):
    # Assume a COVID model
    model = app.build_model(app.params["default"])
    tvs = model.time_variants
    tv_options = sorted(list(tvs.keys()))
    tv_key = st.sidebar.selectbox("Select function", tv_options)
    is_logscale = st.sidebar.checkbox("Log scale")
    tv_func = tvs[tv_key]
    plots.model.plots.plot_time_varying_input(plotter, tv_key, tv_func, model.times, is_logscale)


PLOT_FUNCS["Time variant functions"] = plot_dynamic_inputs


def plot_location_mixing(plotter: StreamlitPlotter, app: AppRegion):
    if not app.app_name == "covid_19":
        # Assume a COVID model
        st.write("This only works for COVID-19 models :(")
        return

    params = app.params["default"]
    mixing = params.get("mixing")
    if not mixing:
        st.write("This model does not have location based mixing")
        return

    start_time = params["time"]["start"]
    end_time = params["time"]["end"]
    time_step = params["time"]["step"]
    times = get_model_times_from_inputs(round(start_time), end_time, time_step,)

    loc_key = st.sidebar.selectbox("Select location", LOCATIONS)
    is_logscale = st.sidebar.checkbox("Log scale")

    country_iso3 = params["iso3"]
    region = params["mobility_region"]
    microdistancing = params["microdistancing"]
    npi_effectiveness_params = params["npi_effectiveness"]
    google_mobility_locations = params["google_mobility_locations"]
    smooth_google_data = params.get("smooth_google_data")
    microdistancing_locations = ["home", "other_locations", "school", "work"]
    adjust = LocationMixingAdjustment(
        country_iso3,
        region,
        mixing,
        npi_effectiveness_params,
        google_mobility_locations,
        microdistancing,
        smooth_google_data,
        microdistancing_locations,
    )
    if adjust.microdistancing_function and loc_key in microdistancing_locations:
        loc_func = lambda t: adjust.microdistancing_function(t) * adjust.loc_adj_funcs[loc_key](t)
    elif loc_key in adjust.loc_adj_funcs:
        loc_func = lambda t: adjust.loc_adj_funcs[loc_key](t)
    else:
        loc_func = lambda t: 1

    plots.model.plots.plot_time_varying_input(plotter, loc_key, loc_func, times, is_logscale)


PLOT_FUNCS["Dynamic location mixing"] = plot_location_mixing


def plot_mobility_raw(
    plotter: StreamlitPlotter, app: AppRegion,
):
    params = app.params["default"]
    values, days = get_mobility_data(
        params["iso3"], params["mobility_region"], BASE_DATE, params["google_mobility_locations"],
    )
    options = list(params["google_mobility_locations"].keys())
    loc_key = st.sidebar.selectbox("Select location", options)
    values_lookup = {days[i]: values[loc_key][i] for i in range(len(days))}
    loc_func = lambda t: values_lookup[t]
    plots.model.plots.plot_time_varying_input(plotter, loc_key, loc_func, days, is_logscale=False)


PLOT_FUNCS["Google Mobility Raw"] = plot_mobility_raw


def plot_model_targets(
    plotter: StreamlitPlotter, app: AppRegion,
):
    # Assume a COVID model
    scenario = Scenario(app.build_model, idx=0, params=app.params)
    with st.spinner("Running model..."):
        scenario.run()

    target_name_lookup = {t["title"]: t for t in app.targets.values()}
    title_options = sorted(list(target_name_lookup.keys()))
    title = st.sidebar.selectbox("Select a target", title_options)
    target = target_name_lookup[title]
    is_logscale = st.sidebar.checkbox("Log scale")
    plots.model.plots.plot_outputs_single(plotter, scenario, target, is_logscale)


def plot_model_multi_targets(
    plotter: StreamlitPlotter, app: AppRegion,
):
    # Assume a COVID model
    scenario = Scenario(app.build_model, idx=0, params=app.params)
    with st.spinner("Running model..."):
        scenario.run()
    is_logscale = st.sidebar.checkbox("Log scale")
    target_name_lookup = {t["title"]: t for t in app.targets.values()}
    titles = [
        "Population size",
        "TB prevalence (/100,000)",
        "Notifications",
    ]  # FIXME: Could come from a multi-selector
    target_list = [target_name_lookup[title] for title in titles]
    plots.model.plots.plot_multi_targets(plotter, scenario, target_list, is_logscale)


PLOT_FUNCS["Calibration targets"] = plot_model_targets


PLOT_FUNCS["Calibration multi-targets"] = plot_model_multi_targets
