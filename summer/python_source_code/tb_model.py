
from python_source_code.summer_model import *
from python_source_code.db import InputDB
import matplotlib.pyplot
import os
import numpy
import scipy.integrate
import copy
from python_source_code.curve import scale_up_function
import pandas as pd


def provide_aggregated_latency_parameters():
    """
    function to add the latency parameters estimated by Ragonnet et al from our paper in Epidemics to the existing
    parameter dictionary
    """
    return {"early_progression": 1.1e-3, "stabilisation": 1.0e-2, "late_progression": 5.5e-6}


def provide_age_specific_latency_parameters():
    """
    simply list out all the latency progression parameters from Ragonnet et al as a dictionary
    """
    return {"early_progression": {0: 6.6e-3, 5: 2.7e-3, 15: 2.7e-4},
            "stabilisation": {0: 1.2e-2, 5: 1.2e-2, 15: 5.4e-3},
            "late_progression": {0: 1.9e-11, 5: 6.4e-6, 15: 3.3e-6}}


def get_adapted_age_parameters(
        age_breakpoints_, parameter_names=("early_progression", "stabilisation", "late_progression")):
    """
    get age-specific parameters adapted to any specification of age breakpoints
    """
    adapted_parameter_dict = {}
    for parameter in parameter_names:
        adapted_parameter_dict[parameter] = \
            add_w_to_param_names(
                change_parameter_unit(
                    get_parameter_dict_from_function(
                        create_step_function_from_dict(
                            provide_age_specific_latency_parameters()[parameter]), age_breakpoints_)))
    return adapted_parameter_dict


def add_standard_latency_flows(flows_):
    """
    adds our standard latency flows to the list of flows to be implemented in the model
    """
    flows_ += [
        {"type": "standard_flows", "parameter": "early_progression", "origin": "early_latent", "to": "infectious"},
        {"type": "standard_flows", "parameter": "stabilisation", "origin": "early_latent", "to": "late_latent"},
        {"type": "standard_flows", "parameter": "late_progression", "origin": "late_latent", "to": "infectious"}]
    return flows_


def convert_competing_proportion_to_rate(competing_flows):
    """
    convert a proportion to a rate dependent on the other flows coming out of a compartment
    """
    return lambda proportion: proportion * competing_flows / (1.0 - proportion)


def return_function_of_function(inner_function, outer_function):
    """
    general method to return a chained function from two functions
    """
    return lambda value: outer_function(inner_function(value))


def unpivot_outputs(model_object):
    """
    take outputs in the form they come out of the model object and convert them into a "long", "melted" or "unpiovted"
    format in order to more easily plug to PowerBI
    """
    output_dataframe = pd.DataFrame(model_object.outputs, columns=model_object.compartment_names)
    output_dataframe["times"] = model_object.times
    output_dataframe = output_dataframe.melt("times")
    for n_stratification in range(len(model_object.strata) + 1):
        column_name = "compartment" if n_stratification == 0 else model_object.strata[n_stratification - 1]
        output_dataframe[column_name] = \
            output_dataframe.apply(lambda row: row.variable.split("X")[n_stratification], axis=1)
        if n_stratification > 0:
            output_dataframe[column_name] = \
                output_dataframe.apply(lambda row: row[column_name].split("_")[1], axis=1)
    return output_dataframe.drop(columns="variable")

# loading time-variant case detection rate
input_database = InputDB(report=True)
res_cdr = input_database.db_query("gtb_2015", column="c_cdr", is_filter="country", value="Mongolia")
res_cdr_year = input_database.db_query("gtb_2015", column="year", is_filter="country", value="Mongolia")


def build_working_tb_model(tb_n_contact, cdr_adjustment=0.6, start_time=1800.):
    """
    current working tb model with some characteristics of mongolia applied at present
    """
    times_ = numpy.linspace(start_time, 2020.0, 201).tolist()

    # set basic parameters, flows and times, except for latency flows and parameters, then functionally add latency
    case_fatality_rate = 0.4
    untreated_disease_duration = 3.0
    parameters = \
        {"beta": tb_n_contact,
         "recovery": case_fatality_rate / untreated_disease_duration,
         "infect_death": (1.0 - case_fatality_rate) / untreated_disease_duration,
         "universal_death_rate": 1.0 / 50.0,
         "case_detection": 0.0}
    parameters.update(change_parameter_unit(provide_aggregated_latency_parameters()))

    flows = [{"type": "infection_frequency", "parameter": "beta", "origin": "susceptible", "to": "early_latent"},
             {"type": "infection_frequency", "parameter": "beta", "origin": "recovered", "to": "early_latent"},
             {"type": "standard_flows", "parameter": "recovery", "origin": "infectious", "to": "recovered"},
             {"type": "compartment_death", "parameter": "infect_death", "origin": "infectious"}]
    flows = add_standard_latency_flows(flows)

    tb_model_ = StratifiedModel(
        times_, ["susceptible", "early_latent", "late_latent", "infectious", "recovered"], {"infectious": 1e-3},
        parameters, flows, birth_approach="replace_deaths")

    tb_model_.add_transition_flow(
        {"type": "standard_flows", "parameter": "case_detection", "origin": "infectious", "to": "recovered"})


    # add scaling case detection rate
    cdr_adjustment_factor = cdr_adjustment
    cdr_mongolia = res_cdr["c_cdr"].values / 1e2 * cdr_adjustment_factor
    cdr_mongolia = numpy.concatenate(([0.0], cdr_mongolia))

    cdr_mongolia_year = res_cdr_year["year"].values
    cdr_mongolia_year = numpy.concatenate(([1950.], cdr_mongolia_year))
    cdr_scaleup = scale_up_function(cdr_mongolia_year, cdr_mongolia, smoothness=0.2, method=5)
    prop_to_rate = convert_competing_proportion_to_rate(1.0 / untreated_disease_duration)
    detect_rate = return_function_of_function(cdr_scaleup, prop_to_rate)
    tb_model_.time_variants["case_detection"] = detect_rate

    # store scaling functions in database if required
    # function_dataframe = pd.DataFrame(times)
    # function_dataframe["cdr_values"] = [cdr_scaleup(t) for t in times]
    # store_database(function_dataframe, table_name="functions")

    # add age stratification
    age_breakpoints = [0, 6, 13, 15]
    age_infectiousness = get_parameter_dict_from_function(logistic_scaling_function(15.0), age_breakpoints)
    tb_model_.stratify("age", age_breakpoints, [],
                       adjustment_requests=get_adapted_age_parameters(age_breakpoints),
                       infectiousness_adjustments=age_infectiousness,
                       verbose=False)

    # tb_model_.stratify("smear",
    #                   ["smearpos", "smearneg", "extrapul"],
    #                   ["infectious"], adjustment_requests=[], report=False)

    return tb_model_


if __name__ == "__main__":

    tb_model = build_working_tb_model(40.0)

    # create_flowchart(tb_model, name="mongolia_flowchart")
    # tb_model.transition_flows.to_csv("transitions.csv")

    tb_model.run_model()

    # get outputs
    infectious_population = tb_model.get_total_compartment_size(["infectious"])

    # print statements to enable crude manual calibration
    # time_2016 = [i for i in range(len(tb_model.times)) if tb_model.times[i] > 2016.][0]
    # print(time_2016)
    # print(infectious_population[time_2016] * 1e5)
    # print(cdr_mongolia)

    # output the results into a format that will be easily loadable into PowerBI
    # pbi_outputs = unpivot_outputs(tb_model)
    # store_database(pbi_outputs, table_name="pbi_outputs")

    # easy enough to output a flow diagram if needed:
    # create_flowchart(tb_model)

    # output some basic quantities if not bothered with the PowerBI bells and whistles
    # tb_model.plot_compartment_size(["early_latent", "late_latent"])
    # tb_model.plot_compartment_size(["infectious"], 1e5)

    # store outputs into database
    # tb_model.store_database()
    #
    # matplotlib.pyplot.plot(times, infectious_population * 1e5)
    # matplotlib.pyplot.xlim((1980., 2020.))
    # matplotlib.pyplot.ylim((0.0, 1e3))
    # matplotlib.pyplot.show()
    # matplotlib.pyplot.savefig("mongolia_cdr_output")



