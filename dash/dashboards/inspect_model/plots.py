from datetime import datetime

import streamlit as st

from autumn import plots
from autumn.plots.plotter import StreamlitPlotter
from autumn.tool_kit.model_register import AppRegion
from autumn.tool_kit.scenarios import Scenario, get_model_times_from_inputs
from autumn.inputs import get_mobility_data


BASE_DATE = datetime(2020, 1, 1, 0, 0, 0)
PLOT_FUNCS = {}

iso3_map = {
    "barwon-south-west": "AUS",
    "belgium": "BEL",
    "calabarzon": "PHL",
    "central-visayas": "PHL",
    "france": "FRA",
    "gippsland": "AUS",
    "grampians": "AUS",
    "hume": "AUS",
    "italy": "ITA",
    "loddon-mallee": "AUS",
    "malaysia": "MYS",
    "manila": "PHL",
    "north-metro": "AUS",
    "philippines": "PHL",
    "sabah": "MYS",
    "south-east-metro": "AUS",
    "south-metro": "AUS",
    "spain": "ESP",
    "sweden": "SWE",
    "united-kingdom": "GBR",
    "victoria": "AUS",
    "west-metro": "AUS",
}

sub_region_map = {
    "barwon-south-west": "BARWON_SOUTH_WEST",
    "calabarzon": "Calabarzon",
    "central-visayas": "Central Visayas",
    "gippsland": "GIPPSLAND",
    "grampians": "GRAMPIANS",
    "hume": "HUME",
    "loddon-mallee": "LODDON_MALLEE",
    "manila": "Metro Manila",
    "north-metro": "NORTH_METRO",
    "sabah": "Sabah",
    "south-east-metro": "SOUTH_METRO",
    "south-metro": "SOUTH_METRO",
    "victoria": "Victoria",
    "west-metro": "WEST_METRO",
}


def plot_age_distribution(
    plotter: StreamlitPlotter, app: AppRegion,
):

    iso3 = iso3_map[app.region_name]

    if app.region_name in sub_region_map.keys():
        sub_region = sub_region_map[app.region_name]
    else:
        sub_region = None
    plots.model.plots.plot_age_distribution(plotter, sub_region, iso3)


PLOT_FUNCS["Age distribution"] = plot_age_distribution


def plot_mixing_matrix(
    plotter: StreamlitPlotter, app: AppRegion,
):
    iso3 = app.params["default"]["country"]["iso3"]
    param_names = sorted(list(("all_locations", "home", "other_locations", "school", "work")))
    param_name = st.sidebar.selectbox("Select parameter", param_names)
    plots.model.plots.plot_mixing_matrix(plotter, param_name, iso3)


PLOT_FUNCS["Mixing matrix"] = plot_mixing_matrix


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
        params["country"]["iso3"],
        params["mobility"]["region"],
        BASE_DATE,
        params["mobility"]["google_mobility_locations"],
    )
    options = list(params["mobility"]["google_mobility_locations"].keys())
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

    is_logscale = st.sidebar.checkbox("Log scale")

    target_name_lookup = {t["title"]: t for t in app.targets.values()}
    title_options = sorted(list(target_name_lookup.keys()))
    titles = st.multiselect("Select outputs", title_options)
    target_list = [target_name_lookup[title] for title in titles]
    if len(target_list) > 0:
        with st.spinner("Running model..."):
            scenario.run()
    plots.model.plots.plot_multi_targets(plotter, scenario, target_list, is_logscale)


PLOT_FUNCS["Calibration targets"] = plot_model_targets


PLOT_FUNCS["Calibration multi-targets"] = plot_model_multi_targets
