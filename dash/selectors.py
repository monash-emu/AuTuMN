"""
Streamlit selectors.
"""
import os
from datetime import datetime
from typing import List, Tuple

import streamlit as st
import pandas as pd

from autumn import constants
from autumn.tool_kit import Scenario
from summer.model import StratifiedModel


def get_original_compartments(model: StratifiedModel) -> List[str]:
    return list(set([str(c).split("X")[0] for c in model.compartment_names]))


def single_compartment(model: StratifiedModel) -> str:
    """
    Allows the user to select a compartment given a model.
    Returns a compartment name.
    """
    original_compartments = get_original_compartments(model)
    compartment_choice = st.selectbox("compartment", original_compartments)

    matching_compartments = [c for c in model.compartment_names if compartment_choice in c]
    for strat_name, strata in model.all_stratifications.items():
        if any([strat_name in c for c in matching_compartments]):
            chosen_strata = st.selectbox(strat_name, strata)
            key = f"{strat_name}_{chosen_strata}"
            matching_compartments = [
                c for c in matching_compartments if any([key == p for p in c.split("X")])
            ]

    assert len(matching_compartments) == 1
    return matching_compartments[0]


def multi_compartment(model: StratifiedModel) -> List[str]:
    """
    Allows the user to select multiple compartments given a model.
    Returns a list of compartment names.
    """

    # Choose compartments to aggregate
    original_compartments = get_original_compartments(model)
    options = ["All"] + original_compartments
    compartment_choices = st.multiselect("compartments", options, default=["All"])
    chosen_compartments = "All" if "All" in compartment_choices else compartment_choices

    # Choose strata to aggregate
    chosen_strata = {}
    for strat_name, strata in model.all_stratifications.items():
        options = ["All"] + strata
        choices = st.multiselect(strat_name, options, default=["All"])
        chosen_strata[strat_name] = "All" if "All" in choices else choices

    # Figure out which compartment names we just chose.
    chosen_compartment_names = []
    for compartment_name in model.compartment_names:
        parts = str(compartment_name).split("X")
        compartment, strata_strs = parts[0], parts[1:]
        # Check that this compartment was selected
        if chosen_compartments == "All":
            is_accepted = True
        else:
            is_accepted = compartment in chosen_compartments

        # Figure out which strata were selected
        comp_strata = {}
        for strata_str in strata_strs:
            strat_parts = strata_str.split("_")
            comp_strata[strat_parts[0]] = "_".join(strat_parts[1:])

        # Check that each strata was selected
        for strat_name, chosen_stratas in chosen_strata.items():
            if chosen_stratas != "All":
                is_accepted = is_accepted and comp_strata[strat_name] in chosen_stratas

        if is_accepted:
            chosen_compartment_names.append(compartment_name)

    return chosen_compartment_names


def app_name(run_type) -> Tuple[str, str]:
    """
    Selector for users to choose which app they want to select
    from the model output data directory.
    Returns app dir and path to app dir
    """
    if run_type == "model":
        apps = sorted(
            [
                d
                for d in os.listdir(constants.APPS_PATH)
                if os.path.isdir(os.path.join(constants.APPS_PATH, d))
                and d not in ["cli", "legacy", "__pycache__"]
            ]
        )
        chosen_dirname = st.sidebar.selectbox("Select app", apps)
        return chosen_dirname, os.path.join(constants.APPS_PATH, chosen_dirname)
    else:
        run_outputs_path = os.path.join(constants.OUTPUT_DATA_PATH, run_type)
        if not os.path.exists(run_outputs_path):
            return None, None

        apps = os.listdir(run_outputs_path)
        chosen_dirname = st.sidebar.selectbox("Select app", apps)
        return chosen_dirname, os.path.join(run_outputs_path, chosen_dirname)


def output_region_name(app_output_path: str, name: str) -> Tuple[str, str]:
    """
    Selector for users to choose which parameter set they want to select
    for a given application
    Returns param set dir and path to param set dir
    """
    param_sets = os.listdir(app_output_path)
    if not param_sets:
        return None, None

    chosen_param_set = st.sidebar.selectbox(name, param_sets)
    return chosen_param_set, os.path.join(app_output_path, chosen_param_set)


def app_region_name(app_name: str) -> Tuple[str, str]:
    """
    Selector for users to choose which parameter set they want to select
    for a given application
    Returns param set dir and path to param set dir
    """
    region_path = os.path.join(constants.APPS_PATH, app_name, "regions")
    regions = [r.replace("_", "-") for r in os.listdir(region_path)]
    if not regions:
        return None, None

    chosen_region = st.sidebar.selectbox("Select app region", regions)
    return chosen_region, os.path.join(region_path, chosen_region.replace("-", "_"))


def scenarios(scenarios: List[Scenario], include_all=True) -> List[Scenario]:
    """
    Get user to select the scenario that they want.
    User may select "All", which includes all Scenarios.
    Returns a list of Scenarios.
    """
    if include_all:
        options = ["All", "Baseline"]
        offset = 1
    else:
        options = ["Baseline"]
        offset = 0

    for s in scenarios[1:]:
        options.append(f"Scenario {s.idx}")

    scenario_name = st.sidebar.selectbox("Select scenario", options)
    if scenario_name == "All":
        return scenarios
    else:
        idx = options.index(scenario_name) - offset
        return [scenarios[idx]]


def calibration_run(param_set_dirpath: str, name: str) -> str:
    """
    Allows a user to select what model run they want, given an app
    Returns the directory name selected.
    """
    # Read model runs from filesystem
    model_run_dirs = os.listdir(param_set_dirpath)

    # Parse model run folder names
    model_runs = []
    for dirname in model_run_dirs:
        run_datetime = datetime.strptime(dirname, "%Y-%m-%d")
        model_runs.append([run_datetime, dirname])

    # Sort model runs by date
    model_runs = reversed(sorted(model_runs, key=lambda i: i[0]))

    # Create labels for the select box.
    labels = []
    model_run_dir_lookup = {}
    for run_datetime, dirname in model_runs:
        run_datestr = run_datetime.strftime("%d %b at %I%p")
        model_run_dir_lookup[run_datestr] = dirname
        labels.append(run_datestr)

    label = st.sidebar.selectbox(name, labels)
    if not label:
        return None, None
    else:
        dirname = model_run_dir_lookup[label]
        dirpath = os.path.join(param_set_dirpath, dirname)
        return dirname, dirpath


def model_run(param_set_dirpath: str) -> Tuple[str, str]:
    """
    Allows a user to select what model run they want, given an app
    Returns the directory name selected.
    """
    # Read model runs from filesystem
    model_run_dirs = list(reversed(sorted(os.listdir(param_set_dirpath))))

    # Parse model run folder names
    model_runs = []
    for dirname in model_run_dirs:
        run_datetime = datetime.strptime(dirname, "%Y-%m-%d--%H-%M-%S")
        model_runs.append(run_datetime)

    # Create labels for the select box.
    labels = []
    model_run_dir_lookup = {}
    for idx, run_datetime in enumerate(model_runs):
        label = run_datetime.strftime("%d %b at %I:%M%p %Ss ")
        model_run_dir_lookup[label] = idx
        labels.append(label)

    label = st.sidebar.selectbox("Select app model run", labels)
    if not label:
        return None, None
    else:
        idx = model_run_dir_lookup[label]
        dirname = model_run_dirs[idx]
        dirpath = os.path.join(param_set_dirpath, dirname)
        return dirname, dirpath


def burn_in(mcmc_tables: List[pd.DataFrame]):
    """
    Slider for selecting how much burn in we should apply to an MCMC trace.
    """
    min_length = min([len(t) for t in mcmc_tables])
    return st.sidebar.slider("Burn-in", 0, min_length, 0)


def parameter(mcmc_params: pd.DataFrame):
    """
    Drop down for selecting parameters
    """
    options = mcmc_params["name"].unique().tolist()
    return st.sidebar.selectbox("Select parameter", options)


def create_standard_plotting_sidebar():
    title_font_size = st.sidebar.slider("Title font size", 1, 15, 8)
    label_font_size = st.sidebar.slider("Label font size", 1, 15, 8)
    dpi_request = st.sidebar.slider("DPI", 50, 2000, 300)
    capitalise_first_letter = st.sidebar.checkbox("Title start capital")
    return title_font_size, label_font_size, dpi_request, capitalise_first_letter


def create_xrange_selector(x_min, x_max):
    x_min = st.sidebar.slider("Plot start time", x_min, x_max, x_min)
    x_max = st.sidebar.slider("Plot end time", x_min, x_max, x_max)
    return x_min, x_max
