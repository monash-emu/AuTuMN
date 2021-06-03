"""
Streamlit selectors.
"""
import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import streamlit as st
from summer.legacy.model import StratifiedModel
from summer import CompartmentalModel

from autumn.settings import OUTPUT_DATA_PATH
from autumn import settings


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


from autumn.tools.registry import get_registered_model_names, get_registered_project_names
from autumn.tools.project import Project, get_project


def project() -> Project:
    model = model_name()
    if not model:
        st.write("No model has been selected.")
        return

    region = region_name(model)
    if not region:
        st.write("No region has been selected.")
        return

    try:
        project = get_project(model, region, reload=True)
    except AssertionError:
        st.write(f"Cannot find a project for {model} {region}.")
        return

    return project


def region_name(model: str) -> str:
    """
    Selector for users to choose which model they want to select.
    """
    regions = get_registered_project_names(model)
    region = st.sidebar.selectbox("Select region", regions)
    return region


def model_name() -> str:
    """
    Selector for users to choose which model they want to select.
    """
    models = get_registered_model_names()
    model = st.sidebar.selectbox("Select model ", models)
    return model


def output_region_name(app_output_path: str, name: str, default_index=0) -> Tuple[str, str]:
    """
    Selector for users to choose which parameter set they want to select
    for a given application
    Returns param set dir and path to param set dir
    """
    param_sets = os.listdir(app_output_path)
    if not param_sets:
        return None, None

    chosen_param_set = st.sidebar.selectbox(name, param_sets, default_index)
    return chosen_param_set, os.path.join(app_output_path, chosen_param_set)


def app_region_name(app_name: str) -> Tuple[str, str]:
    """
    Selector for users to choose which parameter set they want to select
    for a given application
    Returns param set dir and path to param set dir
    """
    region_path = os.path.join(settings.APPS_PATH, app_name, "regions")
    regions = [r.replace("_", "-") for r in os.listdir(region_path)]
    if not regions:
        return None, None

    chosen_region = st.sidebar.selectbox("Select app region", regions)
    return chosen_region, os.path.join(region_path, chosen_region.replace("-", "_"))


def scenarios(models: List[CompartmentalModel], include_all=True) -> List[CompartmentalModel]:
    """
    Get user to select the scenario that they want.
    User may select "All", which includes all Scenarios.
    Returns a list of modelled scenarios.
    """
    if include_all:
        options = ["All", "Baseline"]
        offset = 1
    else:
        options = ["Baseline"]
        offset = 0

    for i in range(1, len(models) + 2):
        options.append(f"Scenario {i}")

    scenario_name = st.sidebar.selectbox("Select scenario", options)
    if scenario_name == "All":
        return models
    else:
        idx = options.index(scenario_name) - offset
        return [models[idx]]


def calibration_path(project: Project) -> str:
    """
    Allows a user to select what model run they want, given an app
    Returns the directory name selected.
    """
    project_calib_dir = os.path.join(
        OUTPUT_DATA_PATH, "calibrate", project.model_name, project.region_name
    )
    if not os.path.exists(project_calib_dir):
        return

    # Read model runs from filesystem
    model_run_dirs = os.listdir(project_calib_dir)

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

    label = st.sidebar.selectbox("Calibration run", labels)
    if not label:
        return None
    else:
        dirname = model_run_dir_lookup[label]
        dirpath = os.path.join(project_calib_dir, dirname)
        return dirpath


def model_run_path(project: Project, multi_country_run_number=None) -> str:
    """
    Allows a user to select what model run they want, given an app
    Returns the directory name selected.
    """
    # Read model runs from filesystem
    model_run_dir = os.path.join(OUTPUT_DATA_PATH, "run", project.model_name, project.region_name)
    if not os.path.exists(model_run_dir):
        return

    model_run_dirs = list(reversed(sorted(os.listdir(model_run_dir))))
    if not model_run_dirs:
        return

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

    if multi_country_run_number is None:
        label = st.sidebar.selectbox("Select app model run", labels)
    else:
        label = st.sidebar.selectbox(f"Select app model run {multi_country_run_number}", labels)
    if not label:
        return None
    else:
        idx = model_run_dir_lookup[label]
        dirname = model_run_dirs[idx]
        dirpath = os.path.join(model_run_dir, dirname)
        return dirpath


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
