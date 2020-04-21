"""
Post processing and storing/loading data from the output database.
"""
import os

import numpy
import pandas as pd
import summer_py.post_processing as post_proc
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

from ..constants import Compartment
from ..db import Database
from .loaded_model import LoadedModel
from autumn.tool_kit.utils import find_first_list_element_above, element_wise_list_summation
from autumn.tb_model.flows import get_incidence_connections, get_notifications_connections


def create_request_stratified_incidence(requested_stratifications, strata_dict):
    """
    Create derived outputs for disaggregated incidence
    """
    out_connections = get_incidence_connections()
    for stratification in requested_stratifications:
        for stratum in strata_dict[stratification]:
            for stage in ["early", "late"]:
                out_connections["incidence_" + stage + "X" + stratification + "_" + stratum] = {
                    "origin": stage + "_latent",
                    "to": Compartment.INFECTIOUS,
                    "to_condition": stratification + "_" + stratum,
                }
    return out_connections


def create_request_stratified_notifications(requested_stratifications, strata_dict):
    """
    Create derived outputs for disaggregated notifications
    """
    out_connections = get_notifications_connections()
    for stratification in requested_stratifications:
        for stratum in strata_dict[stratification]:
            out_connections["notificationsX" + stratification + "_" + stratum] = {
                "origin": Compartment.INFECTIOUS,
                "to": Compartment.ON_TREATMENT,
                "origin_condition": "",
                "to_condition": stratification + "_" + stratum,
            }
    return out_connections

def create_output_connections_for_incidence_by_stratum(
    all_compartment_names, infectious_compartment_name=Compartment.INFECTIOUS
):
    """
    Automatically create output connections for fully disaggregated incidence outputs

    :param all_compartment_names: full list of model compartment names
    :param infectious_compartment_name: the name used for the active TB compartment
    :return: a dictionary containing incidence output connections
    """
    out_connections = {}
    for compartment in all_compartment_names:
        if infectious_compartment_name in compartment:
            stratum = compartment.split(infectious_compartment_name)[1]
            for stage in ["early", "late"]:
                out_connections["incidence_" + stage + stratum] = {
                    "origin": stage + "_latent",
                    "to": infectious_compartment_name,
                    "origin_condition": "",
                    "to_condition": stratum,
                }
    return out_connections


def list_all_strata_for_mortality(
    all_compartment_names, infectious_compartment_name=Compartment.INFECTIOUS
):
    """
    Automatically lists all combinations of population subgroups to request disaggregated mortality outputs

    :param all_compartment_names: full list of model compartment names
    :param infectious_compartment_name: the name used for the active TB compartment
    :return: a tuple designed to be passed as death_output_categories argument to the model
    """
    death_output_categories = []
    for compartment in all_compartment_names:
        if infectious_compartment_name in compartment:
            stratum = compartment.split(infectious_compartment_name)[1]
            categories_to_append = tuple(stratum.split("X")[1:])
            categories_to_append = [
                i_category for i_category in categories_to_append if i_category != ""
            ]
            death_output_categories.append(categories_to_append)
    death_output_categories.append(())

    return tuple(death_output_categories)


# FIXME: Not used, need to get back into use so we can load and re-plot old models.
def load_model_scenario(scenario_idx: int, database_path: str):
    """
    Load a model's outputs and derived outputs, returns a LoadedModel
    filled with this data,
    """
    out_database = Database(database_name=database_path)
    outputs = out_database.db_query(
        table_name="outputs", conditions=[f"Scenario='S_{scenario_idx}'"]
    )
    derived_outputs = out_database.db_query(
        table_name="derived_outputs", conditions=[f"Scenario='S_{scenario_idx}'"]
    )
    kwargs = {"outputs": outputs.to_dict(), "derived_outputs": derived_outputs.to_dict()}
    return LoadedModel(**kwargs)


def load_calibration_from_db(database_directory, n_burned_per_chain=0):
    """
    Load all model runs stored in multiple databases found in database_directory
    :param database_directory: path to databases location
    :return: list of models
    """
    # list all databases
    db_names = os.listdir(database_directory + "/")
    db_names = [s for s in db_names if s[-3:] == ".db"]

    models = []
    n_loaded_iter = 0
    for db_name in db_names:
        out_database = Database(database_name=database_directory + "/" + db_name)

        # find accepted run indices
        res = out_database.db_query(table_name="mcmc_run", column="idx", conditions=["accept=1"])
        run_ids = list(res.to_dict()["idx"].values())
        # find weights to associate with the accepted runs
        accept = out_database.db_query(table_name="mcmc_run", column="accept")
        accept = accept["accept"].tolist()
        one_indices = [i for i, val in enumerate(accept) if val == 1]
        one_indices.append(len(accept))  # add extra index for counting
        weights = [one_indices[j + 1] - one_indices[j] for j in range(len(one_indices) - 1)]

        # burn fist iterations
        cum_sum = numpy.cumsum(weights).tolist()
        if cum_sum[-1] <= n_burned_per_chain:
            continue
        retained_indices = [i for i, c in enumerate(cum_sum) if c > n_burned_per_chain]
        run_ids = run_ids[retained_indices[0] :]
        previous_cum_sum = cum_sum[retained_indices[0] - 1] if retained_indices[0] > 0 else 0
        weights[retained_indices[0]] = weights[retained_indices[0]] - (
            n_burned_per_chain - previous_cum_sum
        )
        weights = weights[retained_indices[0] :]
        n_loaded_iter += sum(weights)
        for i, run_id in enumerate(run_ids):
            outputs = out_database.db_query(
                table_name="outputs", conditions=["idx='" + str(run_id) + "'"]
            )
            output_dict = outputs.to_dict()

            if out_database.engine.dialect.has_table(out_database.engine, "derived_outputs"):
                derived_outputs = out_database.db_query(
                    table_name="derived_outputs", conditions=["idx='" + str(run_id) + "'"]
                )

                derived_outputs_dict = derived_outputs.to_dict()
            else:
                derived_outputs_dict = None
            model_info_dict = {
                "db_name": db_name,
                "run_id": run_id,
                "model": LoadedModel(output_dict, derived_outputs_dict),
                "weight": weights[i],
            }
            models.append(model_info_dict)

    print("MCMC runs loaded.")
    print("Number of loaded iterations after burn-in: " + str(n_loaded_iter))
    return models


def store_tb_database(
    outputs,
    table_name="outputs",
    scenario=0,
    run_idx=0,
    times=None,
    database_name="../databases/outputs.db",
    append=True,
):
    """
    store outputs from the model in sql database for use in producing outputs later
    """

    if times:
        outputs.insert(0, column="times", value=times)
    if table_name != "mcmc_run_info":
        outputs.insert(0, column="idx", value="run_" + str(run_idx))
        outputs.insert(1, column="Scenario", value="S_" + str(scenario))
    engine = create_engine("sqlite:///" + database_name, echo=False)
    if table_name == "functions":
        outputs.to_sql(
            table_name, con=engine, if_exists="replace", index=False, dtype={"cdr_values": float()}
        )
    elif append:
        outputs.to_sql(table_name, con=engine, if_exists="append", index=False)
    else:
        outputs.to_sql(table_name, con=engine, if_exists="replace", index=False)


def store_run_models(models, scenarios, database_name="../databases/outputs.db"):
    for i, model in enumerate(models):
        output_df = pd.DataFrame(model.outputs, columns=model.compartment_names)
        derived_output_df = pd.DataFrame.from_dict(model.derived_outputs)
        pbi_outputs = _unpivot_outputs(model)
        store_tb_database(
            pbi_outputs,
            table_name="pbi_scenario_" + str(scenarios[i]),
            database_name=database_name,
            scenario=scenarios[i],
        )
        store_tb_database(
            derived_output_df,
            scenario=scenarios[i],
            table_name="derived_outputs",
            database_name=database_name,
        )
        store_tb_database(
            output_df,
            scenario=scenarios[i],
            times=model.times,
            database_name=database_name,
            append=True,
        )


def get_post_processing_results(
    models, req_outputs, req_multipliers, outputs_to_plot_by_stratum, scenario_list, req_times, ymax
):
    pps = []
    for scenario_index in range(len(models)):

        # automatically add some basic outputs
        if hasattr(models[scenario_index], "all_stratifications"):
            for group in models[scenario_index].all_stratifications.keys():
                req_outputs.append("distribution_of_strataX" + group)
                for output in outputs_to_plot_by_stratum:
                    for stratum in models[scenario_index].all_stratifications[group]:
                        req_outputs.append(output + "XamongX" + group + "_" + stratum)

            if "strain" in models[scenario_index].all_stratifications.keys():
                req_outputs.append("prevXinfectiousXstrain_mdrXamongXinfectious")

        for output in req_outputs:
            if (
                output[0:15] == "prevXinfectious"
                and output != "prevXinfectiousXstrain_mdrXamongXinfectious"
            ):
                req_multipliers[output] = 1.0e5
            elif output[0:11] == "prevXlatent":
                req_multipliers[output] = 1.0e2

        pps.append(
            post_proc.PostProcessing(
                models[scenario_index],
                requested_outputs=req_outputs,
                scenario_number=scenario_list[scenario_index],
                requested_times=req_times,
                multipliers=req_multipliers,
                ymax=ymax,
            )
        )
        return pps


def get_ebeye_baseline_data():
    """
    Load in baseline data from spreadsheets provided by CDC - to create dictionary of data frames
    """
    baseline_data = {}
    for island in ["majuro", "ebeye", "otherislands"]:

        # Read data in
        file_path = os.path.abspath(__file__)
        final_path = os.path.join(
            "\\".join(file_path.split("\\")[:-3]),
            "applications",
            "marshall_islands",
            "rmi_specific_data",
            "baseline_" + island + ".csv",
        )
        baseline_data[island] = pd.read_csv(final_path, header=3)

        # Fix column names
        age_cols = {"Unnamed: 0": "year", "Age": "subgroup"}
        n_chars = [0] * 2 + ([1] * 2 + [2] * 3) * 3
        extra_string = [""] * 7 + ["no"] * 5 + ["unknown"] * 5
        for i_column in range(2, len(baseline_data[island].columns) - 1):
            age_cols.update(
                {
                    baseline_data[island]
                    .columns[i_column]: baseline_data[island]
                    .columns[i_column][0 : n_chars[i_column]]
                    + "_"
                    + extra_string[i_column]
                    + "diabetes"
                }
            )
        baseline_data[island].rename(columns=age_cols, inplace=True)
        baseline_data[island].drop(columns=["Total"], inplace=True)
    return baseline_data


def compare_marshall_notifications(
    models,
    req_outputs,
    req_times={},
    req_multipliers={},
    ymax={},
    out_dir="outputs_tes",
    targets_to_plot={},
    translation_dictionary={},
    scenario_list=[],
    plot_start_time=1990,
    outputs_to_plot_by_stratum=["prevXinfectious", "prevXlatent"],
    comparison_times=[2012.0, 2017.0],
):
    """
    Targeted function to produce plot outputs that are specific to the Marshall Islands application.
    Produces a comparison of the notification rates in the model to those in the inputs provided by CDC.
    """

    # Prepare figure
    plt.style.use("ggplot")
    fig = plt.figure()

    # Gather data
    pps = get_post_processing_results(
        models,
        req_outputs,
        req_multipliers,
        outputs_to_plot_by_stratum,
        scenario_list,
        req_times,
        ymax,
    )
    start_time_index = find_first_list_element_above(
        pps[0].derived_outputs["times"], comparison_times[0]
    )
    end_time_index = find_first_list_element_above(
        pps[0].derived_outputs["times"], comparison_times[1]
    )
    stratifications = {
        stratification: models[0].all_stratifications[stratification]
        for stratification in models[0].all_stratifications
    }

    age_groups = models[0].all_stratifications["age"]
    locations = models[0].all_stratifications["location"]
    organs = models[0].all_stratifications["organ"]
    diabetes = models[0].all_stratifications["diabetes"]

    ages = [float(age) for age in age_groups]
    baseline_data = get_ebeye_baseline_data()

    # Sum notifications over organ and diabetes status.
    summed_notifications = sum_notifications_over_organ_diabetes(
        pps[0].derived_outputs, locations, age_groups, organs, diabetes
    )

    # Plot by location
    for i_loc, location in enumerate(locations):
        location_notifications_by_age = [
            sum(summed_notifications[location][age_group][start_time_index:end_time_index])
            for age_group in age_groups
        ]

        # Collate real data
        real_notifications = [
            baseline_data[location]
            .loc[
                1:5,
                [
                    age_group + "_diabetes",
                    age_group + "_nodiabetes",
                    age_group + "_unknowndiabetes",
                ],
            ]
            .sum(0)
            .sum()
            for age_group in age_groups
        ]

        # Prepare plot
        axis = fig.add_subplot(2, 2, i_loc + 1)
        axis.scatter(ages, location_notifications_by_age, color="r")
        axis.scatter(ages, real_notifications, color="k")
        axis.set_title(location)

        # Tidy x-axis
        axis.set_xlim(-5.0, max(ages) + 5.0)
        axis.set_xlabel("age groups")
        axis.set_xticks(ages)
        axis.set_xticklabels(age_groups)

        # Tidy y-axis
        axis.set_ylim(0.0, max(location_notifications_by_age + real_notifications) * 1.2)
        axis.set_ylabel("notifications")

    # Save
    fig.suptitle(
        "Notifications by location, from %s to %s"
        % tuple([str(round(time)) for time in comparison_times])
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(os.path.join(out_dir, "notification_comparisons.png"))


def create_mcmc_outputs(
    mcmc_models,
    req_outputs,
    req_times={},
    req_multipliers={},
    ymax={},
    out_dir="outputs_tes",
    targets_to_plot={},
    translation_dictionary={},
    scenario_list=[],
    plot_start_time=1990,
):
    """similar to create_multi_scenario_outputs but using MCMC outputs"""
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for output in req_outputs:
        if (
            output[0:15] == "prevXinfectious"
            and output != "prevXinfectiousXstrain_mdrXamongXinfectious"
        ):
            req_multipliers[output] = 1.0e5
        elif output[0:11] == "prevXlatent":
            req_multipliers[output] = 1.0e2

    pps = []
    for scenario_index in range(len(mcmc_models)):
        pps.append(
            post_proc.PostProcessing(
                mcmc_models[scenario_index]["model"],
                requested_outputs=req_outputs,
                scenario_number=scenario_list[scenario_index],
                requested_times=req_times,
                multipliers=req_multipliers,
                ymax=ymax,
            )
        )

    mcmc_weights = [mcmc_models[i]["weight"] for i in range(len(mcmc_models))]
    outputs = Outputs(
        mcmc_models,
        pps,
        output_options={},
        targets_to_plot=targets_to_plot,
        out_dir=out_dir,
        translation_dict=translation_dictionary,
        mcmc_weights=mcmc_weights,
        plot_start_time=plot_start_time,
    )
    outputs.plot_requested_outputs()


def _unpivot_outputs(model):
    """
    take outputs in the form they come out of the model object and convert them into a "long", "melted" or "unpiovted"
    format in order to more easily plug to PowerBI
    """
    output_df = pd.DataFrame(model.outputs, columns=model.compartment_names)
    output_df["times"] = model.times
    output_df = output_df.melt("times")

    # Make compartment column
    def get_compartment_name(row):
        return row.variable.split("X")[0]

    output_df["compartment"] = output_df.apply(get_compartment_name, axis=1)

    # Map compartment names to strata names
    # Eg.
    #   from susceptibleXage_0Xdiabetes_diabeticXlocation_majuro
    #   to {
    #       "age": "age_0",
    #       "diabetes": "diabetes_diabetic",
    #       "location": "location_majuro"
    #   }
    compartment_to_column_map = {}
    strata_names = list(model.all_stratifications.keys())
    for compartment_name in model.compartment_names:
        compartment_to_column_map[compartment_name] = {}
        compartment_stratas = compartment_name.split("X")[1:]
        for compartment_strata in compartment_stratas:
            for strata_name in strata_names:
                if compartment_strata.startswith(strata_name):
                    compartment_to_column_map[compartment_name][strata_name] = compartment_strata

    def get_strata_names(strata_name):
        def _get_strata_names(row):
            compartment_name = row["variable"]
            try:
                return compartment_to_column_map[compartment_name][strata_name]
            except KeyError:
                return ""

        return _get_strata_names

    for strata_name in strata_names:
        output_df[strata_name] = output_df.apply(get_strata_names(strata_name), axis=1)

    output_df = output_df.drop(columns="variable")
    return output_df


def plot_time_variant_param(function, time_span, title=""):
    times = numpy.linspace(time_span[0], time_span[1], 1000)
    y = [function(t) for t in times]
    plt.plot(times, y)
    plt.title(title)
    plt.show()


def sum_notifications_over_organ_diabetes(derived_outputs, locations, age_groups, organs, diabetes):
    """
    Quite a specific function to sum the fully disaggregated notifications according to two specific modelled strata.
    Currently only intended to be used for the RMI application.
    """
    n_times = len(derived_outputs["times"])
    summed_notifications = {}
    for i_loc, location in enumerate(locations):
        summed_notifications[location] = {}
        for age_group in age_groups:
            summed_notifications[location][age_group] = [0.0] * n_times
            for organ in organs:
                for diabetic in diabetes:
                    summed_notifications[location][age_group] = element_wise_list_summation(
                        summed_notifications[location][age_group],
                        derived_outputs[
                            "notificationsXage_"
                            + age_group
                            + "Xdiabetes_"
                            + diabetic
                            + "Xorgan_"
                            + organ
                            + "Xlocation_"
                            + location
                        ],
                    )
    return summed_notifications
