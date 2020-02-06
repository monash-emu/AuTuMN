"""
Post processing and storing/loading data from the output database.
"""
import os
import re

import numpy
import pandas as pd
import summer_py.post_processing as post_proc
from sqlalchemy import create_engine
from summer_py.outputs import Outputs

from ..constants import Compartment
from ..db import Database
from .dummy_model import DummyModel


def add_combined_incidence(derived_outputs, outputs):
    columns_to_add = {}
    comp_names = outputs.drop(outputs.columns[[0, 1, 2]], axis=1).columns
    for column_name in derived_outputs.columns:
        if column_name[0:15] == "incidence_early":
            incidence_late_name = "incidence_late" + column_name[15:]
            new_output = "incidence" + column_name[15:]
            absolute_incidence = derived_outputs[column_name] + derived_outputs[incidence_late_name]

            # work out total stratum population
            if column_name == "incidence_early":  # we need the total population
                stratum_compartments = comp_names
            else:  # we may need a subgroup population
                stratification_name = column_name[15:].split("_")[0]
                if all(stratification_name in c for c in comp_names):
                    stratum_compartments = [c for c in comp_names if column_name[15:] in c]
                else:
                    stratum_compartments = comp_names

            stratum_population = outputs[stratum_compartments].sum(axis=1)
            columns_to_add[new_output] = absolute_incidence / stratum_population * 1.0e5

    for key, val in columns_to_add.items():
        derived_outputs[key] = val

    return derived_outputs


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
            death_output_categories.append(tuple(stratum.split("X")[1:]))
    return tuple(death_output_categories)


def load_model_scenario(scenario_name, database_name):
    out_database = Database(database_name="databases/" + database_name)
    outputs = out_database.db_query(
        table_name="outputs", conditions=["Scenario='S_" + scenario_name + "'"]
    )
    derived_outputs = out_database.db_query(
        table_name="derived_outputs", conditions=["Scenario='S_" + scenario_name + "'"]
    )
    derived_outputs = add_combined_incidence(derived_outputs, outputs)
    return {"outputs": outputs.to_dict(), "derived_outputs": derived_outputs.to_dict()}


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
            derived_outputs = out_database.db_query(
                table_name="derived_outputs", conditions=["idx='" + str(run_id) + "'"]
            )
            derived_outputs = add_combined_incidence(derived_outputs, outputs)

            derived_outputs_dict = derived_outputs.to_dict()
            model_info_dict = {
                "db_name": db_name,
                "run_id": run_id,
                "model": DummyModel(output_dict, derived_outputs_dict),
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


def create_multi_scenario_outputs(
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
):
    """
    process and generate plots for several scenarios
    :param models: a list of run models
    :param req_outputs. See PostProcessing class
    :param req_times. See PostProcessing class
    :param req_multipliers. See PostProcessing class
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    pps = []
    for scenario_index in range(len(models)):

        # automatically add some basic outputs
        if hasattr(models[scenario_index], "all_stratifications"):
            for group in models[scenario_index].all_stratifications.keys():
                req_outputs.append("distribution_of_strataX" + group)
                for stratum in models[scenario_index].all_stratifications[group]:
                    req_outputs.append("prevXinfectiousXamongX" + group + "_" + stratum)
                    req_outputs.append("prevXlatentXamongX" + group + "_" + stratum)

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

    outputs = Outputs(
        pps, targets_to_plot, out_dir, translation_dictionary, plot_start_time=plot_start_time
    )
    outputs.plot_requested_outputs()

    for req_output in ["prevXinfectious", "prevXlatent"]:
        for sc_index in range(len(models)):
            outputs.plot_outputs_by_stratum(req_output, sc_index=sc_index)


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
        pps,
        targets_to_plot,
        out_dir,
        translation_dictionary,
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
