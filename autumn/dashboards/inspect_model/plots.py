from datetime import datetime

import streamlit as st
from matplotlib import pyplot

from autumn import plots
from autumn.tools.inputs import get_mobility_data
from autumn.tools.plots.plotter import StreamlitPlotter
from autumn.dashboards.dashboards.inspect_model.flow_graph import plot_flow_graph

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


def plot_flow_params(plotter: StreamlitPlotter, app: AppRegion):
    # Assume a COVID model
    model = app.build_model(app.params["default"])
    flow_names = sorted(list({f.name for f in model._flows}))

    flow_name = st.sidebar.selectbox("Select flow", flow_names)
    flows = [f for f in model._flows if f.name == flow_name]
    is_logscale = st.sidebar.checkbox("Log scale")
    flow_funcs = [f.get_weight_value for f in flows]
    plots.model.plots.plot_time_varying_input(
        plotter, f"flow-weights-{flow_name}", flow_funcs, model.times, is_logscale
    )
    t = st.slider("Time", min_value=0, max_value=int(model.times[-1]))
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
        init_dict[f_name] = f.get_weight_value(t)

    st.write("Values at start time:")
    st.write(init_dict)


PLOT_FUNCS["Flow weights"] = plot_flow_params


PLOT_FUNCS["Flow graph"] = plot_flow_graph


def plot_multi_age_distribution(plotter: StreamlitPlotter, app: AppRegion):

    iso3 = iso3_map[app.region_name]

    if app.region_name is "philippines":
        sub_region = [None, "Metro Manila", "Calabarzon", "Central Visayas"]
        plots.model.plots.plot_multi_age_distribution(plotter, sub_region, iso3)
    else:
        st.write("This region has no sub  regions")


PLOT_FUNCS["Multi age distribution"] = plot_multi_age_distribution


def plot_age_distribution(plotter: StreamlitPlotter, app: AppRegion):

    iso3 = iso3_map[app.region_name]

    if app.region_name in sub_region_map.keys():
        sub_region = sub_region_map[app.region_name]
    else:
        sub_region = None
    plots.model.plots.plot_age_distribution(plotter, sub_region, iso3)


PLOT_FUNCS["Age distribution"] = plot_age_distribution


def plot_dynamic_mixing_matrix(plotter: StreamlitPlotter, app: AppRegion):
    model = app.build_model(app.params["default"])
    t = st.slider("Time", min_value=0, max_value=int(model.times[-1]))
    mixing_matrix = model._get_mixing_matrix(t)
    fig, _, _, _, _, _ = plotter.get_figure()
    pyplot.imshow(mixing_matrix, cmap="hot", interpolation="none", extent=[0, 80, 80, 0])
    plotter.save_figure(fig, filename="mixing-matrix", title_text="Mixing matrix")
    st.write(mixing_matrix)


PLOT_FUNCS["Dynamic mixing matrix"] = plot_dynamic_mixing_matrix


def plot_mixing_matrix(plotter: StreamlitPlotter, app: AppRegion):
    iso3 = app.params["default"]["country"]["iso3"]
    param_names = sorted(list(("all_locations", "home", "other_locations", "school", "work")))
    param_name = st.sidebar.selectbox("Select parameter", param_names)
    plots.model.plots.plot_mixing_matrix(plotter, param_name, iso3)


PLOT_FUNCS["Static mixing matrix"] = plot_mixing_matrix


def plot_all_mixing_matrices(plotter: StreamlitPlotter, app: AppRegion):
    iso3 = app.params["default"]["country"]["iso3"]
    plots.model.plots.plot_mixing_matrix_2(plotter, iso3)


PLOT_FUNCS["All mixing matrices"] = plot_all_mixing_matrices


def plot_dynamic_inputs(plotter: StreamlitPlotter, app: AppRegion):
    # Assume a COVID model
    model = app.build_model(app.params["default"])
    tvs = model.time_variants
    tv_options = sorted(list(tvs.keys()))
    tv_key = st.sidebar.selectbox("Select function", tv_options)
    is_logscale = st.sidebar.checkbox("Log scale")
    tv_func = tvs[tv_key]
    plots.model.plots.plot_time_varying_input(plotter, tv_key, tv_func, model.times, is_logscale)


PLOT_FUNCS["Time variant functions"] = plot_dynamic_inputs


def plot_mobility_raw(plotter: StreamlitPlotter, app: AppRegion):
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


def plot_multilocation_mobility(plotter: StreamlitPlotter, app: AppRegion):
    params = app.params["default"]
    values, days = get_mobility_data(
        params["country"]["iso3"],
        params["mobility"]["region"],
        BASE_DATE,
        params["mobility"]["google_mobility_locations"],
    )
    plots.model.plots.plot_time_varying_multi_input(plotter, values, days, is_logscale=False)


PLOT_FUNCS["Google Mobility multi-location"] = plot_multilocation_mobility


def plot_model_targets(plotter: StreamlitPlotter, app: AppRegion):
    # Assume a COVID model
    scenario = Scenario(app.build_model, idx=0, params=app.params)
    with st.spinner("Running model..."):
        scenario.run()

    target_name_lookup = {t["title"]: t for t in app.targets.values()}
    title_options = sorted(list(target_name_lookup.keys()))
    title = st.sidebar.selectbox("Select a target", title_options)
    target = target_name_lookup[title]
    is_logscale = st.sidebar.checkbox("Log scale")
    xaxis_date = app.app_name == "covid_19"
    plots.model.plots.plot_outputs_single(
        plotter,
        scenario,
        target,
        is_logscale,
        xaxis_date=xaxis_date,
    )


def plot_model_multi_targets(plotter: StreamlitPlotter, app: AppRegion):
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
