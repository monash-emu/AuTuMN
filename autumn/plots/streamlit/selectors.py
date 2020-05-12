"""
Streamlit selectors.
"""
import os
from datetime import datetime
from typing import List, Tuple

import streamlit as st

from autumn import constants
from autumn.tool_kit import Scenario
from summer.model import StratifiedModel


def get_original_compartments(model: StratifiedModel) -> List[str]:
    return list(set([c.split("X")[0] for c in model.compartment_names]))


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
    compartment_choices = st.multiselect(
        "compartments", ["All"] + original_compartments, default="All"
    )
    chosen_compartments = "All" if "All" in compartment_choices else compartment_choices

    # Choose strata to aggregate
    chosen_strata = {}
    for strat_name, strata in model.all_stratifications.items():
        options = ["All"] + strata
        choices = st.multiselect(strat_name, options, default="All")
        chosen_strata[strat_name] = "All" if "All" in choices else choices

    # Figure out which compartment names we just chose.
    chosen_compartment_names = []
    for compartment_name in model.compartment_names:
        parts = compartment_name.split("X")
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


def app() -> Tuple[str, str]:
    """
    Selector for users to choose which app they want to select
    from the model output data directory.
    Returns app dir and path to app dir
    """
    APP_DIRNAMES = ["covid_", "marshall_islands", "mongolia", "dummy", "sir_example"]
    paths = {}
    dirnames = []
    for dirname in os.listdir(constants.DATA_PATH):
        dirpath = os.path.join(constants.DATA_PATH, dirname)
        is_valid = os.path.isdir(dirpath) and any([dirname.startswith(n) for n in APP_DIRNAMES])
        if is_valid:
            dirnames.append(dirname)
            paths[dirname] = dirpath

    chosen_dirname = st.sidebar.selectbox("Select app", dirnames)
    return chosen_dirname, paths[chosen_dirname]


def scenario(scenarios: List[Scenario], include_all=True) -> List[Scenario]:
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


def calibration_run(app_dirpath: str) -> str:
    """
    Allows a user to select what model run they want, given an app 
    Returns the directory name selected.
    """
    # Read model runs from filesystem
    model_run_dirs = [f for f in reversed(os.listdir(app_dirpath)) if f.startswith("calibration")]
    # Parse model run folder names
    model_runs = []
    for dirname in model_run_dirs:
        datestr = "-".join(dirname.split("-")[-3:])
        run_name = " ".join(dirname.split("-")[:-3]).title()
        run_datetime = datetime.strptime(datestr, "%d-%m-%Y")
        model_runs.append([run_datetime, run_name, dirname])

    # Sort model runs by date
    model_runs = reversed(sorted(model_runs, key=lambda i: i[0]))

    # Create labels for the select box.
    labels = []
    model_run_dir_lookup = {}
    for run_datetime, run_name, dirname in model_runs:
        run_datestr = run_datetime.strftime("%d %b at %I:%M%p ")
        label = f'{run_datestr} "{run_name}"'
        model_run_dir_lookup[label] = dirname
        labels.append(label)

    label = st.sidebar.selectbox("Select app calibration run", labels)
    if not label:
        return None, None
    else:
        dirname = model_run_dir_lookup[label]
        dirpath = os.path.join(app_dirpath, dirname)
        return dirname, dirpath


def model_run(app_dirpath: str) -> Tuple[str, str]:
    """
    Allows a user to select what model run they want, given an app 
    Returns the directory name selected.
    """
    # Read model runs from filesystem
    model_run_dirs = [
        f for f in reversed(os.listdir(app_dirpath)) if not f.startswith("calibration")
    ]
    # Parse model run folder names
    model_runs = []
    for dirname in model_run_dirs:
        datestr = "-".join(dirname.split("-")[-7:])
        run_name = " ".join(dirname.split("-")[:-7]).title()
        run_datetime = datetime.strptime(datestr, "%d-%m-%Y--%H-%M-%S")
        model_runs.append([run_datetime, run_name, dirname])

    # Sort model runs by date
    model_runs = reversed(sorted(model_runs, key=lambda i: i[0]))

    # Create labels for the select box.
    labels = []
    model_run_dir_lookup = {}
    for run_datetime, run_name, dirname in model_runs:
        run_datestr = run_datetime.strftime("%d %b at %I:%M%p ")
        label = f'{run_datestr} "{run_name}"'
        model_run_dir_lookup[label] = dirname
        labels.append(label)

    label = st.sidebar.selectbox("Select app model run", labels)
    if not label:
        return None, None
    else:
        dirname = model_run_dir_lookup[label]
        dirpath = os.path.join(app_dirpath, dirname)
        return dirname, dirpath
