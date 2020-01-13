import re
from summer_py.summer_model import *
from autumn_from_summer.db import InputDB
import matplotlib.pyplot
import numpy
import copy
from autumn_from_summer.curve import scale_up_function
import pandas as pd
from autumn_from_summer.db import get_bcg_coverage, get_crude_birth_rate, get_pop_mortality_functions
import summer_py.post_processing as post_proc
from summer_py.outputs import Outputs
import json
from autumn_from_summer.tool_kit import *
from summer_py.parameter_processing import *

def load_model_scenario(scenario_name, database_name):
    out_database = InputDB(database_name="databases/" + database_name)
    res = out_database.db_query(table_name='outputs', conditions=["Scenario='S_"+scenario_name+"'"])
    return res.to_dict()


def load_calibration_from_db(database_name):
    out_database = InputDB(database_name='databases/' + database_name)
    res = out_database.db_query(table_name='mcmc_run', column='idx')
    run_ids = list(res.to_dict()['idx'].values())
    models = []
    for run_id in run_ids:
        res = out_database.db_query(table_name='outputs', conditions=["idx='"+ str(run_id) +"'"])
        model_dict = res.to_dict()
        models.append(DummyModel(model_dict))
    return models

def scale_relative_risks_for_equivalence(proportions, relative_risks):
    """
    :param proportions: dictionary
    :param relative_risks: dictionary
    :return: dictionary
    """
    new_reference_deno = 0.
    for stratum in proportions.keys():
        new_reference_deno += proportions[stratum] * relative_risks[stratum]
    new_reference = 1. / new_reference_deno
    for stratum in relative_risks.keys():
        relative_risks[stratum] *= new_reference
    return relative_risks


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
        age_breakpoints, parameter_names=("early_progression", "stabilisation", "late_progression")):
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
                            provide_age_specific_latency_parameters()[parameter]), age_breakpoints), 365.251))
    return adapted_parameter_dict


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


# temporary fix for store database, need to move to tb_model
def store_tb_database(outputs, table_name="outputs", scenario=0, run_idx=0, times=None, database_name="../databases/outputs.db", append=True):
    """
    store outputs from the model in sql database for use in producing outputs later
    """

    if times:
        outputs.insert(0, column='times', value=times)
    if table_name != 'mcmc_run_info':
        outputs.insert(0, column='idx', value='run_' + str(run_idx))
        outputs.insert(1, column='Scenario', value='S_' + str(scenario))
    engine = create_engine("sqlite:///"+ database_name, echo=False)
    if table_name == "functions":
        outputs.to_sql(table_name, con=engine, if_exists="replace", index=False, dtype={"cdr_values": FLOAT()})
    elif append:
        outputs.to_sql(table_name, con=engine, if_exists="append",  index=False)
    else:
        outputs.to_sql(table_name, con=engine, if_exists="replace", index=False)


def find_match(row, column_name):
    """
    method to return a matching item in row
    """
    regex = re.compile(r'.*' + column_name + r'.*')
    row_list = row.variable.split("X")
    match_item = list(filter(regex.search, row_list))
    if len(match_item) > 0:
        result = match_item[0]
    else:
        result = ''
    return result


def unpivot_outputs(model_object):
    """
    take outputs in the form they come out of the model object and convert them into a "long", "melted" or "unpiovted"
    format in order to more easily plug to PowerBI
    """
    output_dataframe = pd.DataFrame(model_object.outputs, columns=model_object.compartment_names)
    output_dataframe["times"] = model_object.times
    output_dataframe = output_dataframe.melt("times")
    for n_stratification in range(len(model_object.all_stratifications.keys()) + 1):
        column_name = "compartment" if n_stratification == 0 else list(model_object.all_stratifications.keys())[n_stratification - 1]
        if n_stratification == 0:
            output_dataframe[column_name] = \
                 output_dataframe.apply(lambda row: row.variable.split("X")[n_stratification], axis=1)
        if n_stratification > 0:
            output_dataframe[column_name] = \
                output_dataframe.apply(lambda row: find_match(row,column_name), axis=1)
    return output_dataframe.drop(columns="variable")


def store_run_models(models, scenarios, database_name="../databases/outputs.db"):
    for i, model in enumerate(models):
        output_df = pd.DataFrame(model.outputs, columns=model.compartment_names)
        derived_output_df = pd.DataFrame.from_dict(model.derived_outputs)
        pbi_outputs = unpivot_outputs(model)
        store_tb_database(pbi_outputs, table_name='pbi_scenario_' + str(scenarios[i]),  database_name=database_name)
        store_tb_database(derived_output_df, scenario=scenarios[i], table_name='derived_outputs', database_name=database_name)
        store_tb_database(output_df, scenario=scenarios[i], times=model.times, database_name=database_name, append=True)


"""
standardised flow functions
"""


def add_standard_latency_flows(list_of_flows):
    """
    adds our standard latency flows to the list of flows to be implemented in the model

    :param list_of_flows: list
        existing flows for implementation in the model
    :return: list_of_flows: list
        list of flows updated to include the standard latency flows
    """
    list_of_flows += [
        {"type": "standard_flows", "parameter": "early_progression", "origin": "early_latent", "to": "infectious"},
        {"type": "standard_flows", "parameter": "stabilisation", "origin": "early_latent", "to": "late_latent"},
        {"type": "standard_flows", "parameter": "late_progression", "origin": "late_latent", "to": "infectious"}]
    return list_of_flows


def add_standard_natural_history_flows(list_of_flows):
    """
    adds our standard natural history to the list of flows to be implemented in the model

    :param list_of_flows: list
        existing flows for implementation in the model
    :return: list_of_flows: list
        list of flows updated to include the standard latency flows
    """
    list_of_flows += [
        {"type": "standard_flows", "parameter": "recovery", "origin": "infectious", "to": "recovered"},
        {"type": "compartment_death", "parameter": "infect_death", "origin": "infectious"}]
    return list_of_flows


def add_standard_infection_flows(list_of_flows):
    """
    adds our standard infection processes to the list of flows to be implemented in the model

    :param list_of_flows: list
        existing flows for implementation in the model
    :return: list_of_flows: list
        list of flows updated to include the standard infection processes
    """
    list_of_flows += [
        {"type": "infection_frequency", "parameter": "contact_rate", "origin": "susceptible", "to": "early_latent"},
        {"type": "infection_frequency", "parameter": "contact_rate_recovered", "origin": "recovered", "to": "early_latent"},
        {"type": "infection_frequency", "parameter": "contact_rate_infected", "origin": "late_latent", "to": "early_latent"}
    ]
    return list_of_flows


def get_bcg_functions(_tb_model, _input_database, _country_iso3, start_year=1955):
    """
    function to obtain the bcg coverage from the input database and add the appropriate functions to the tb model

    :param _tb_model: StratifiedModel class
        SUMMER model object to be assigned bcg vaccination coverage functions
    :param _input_database: sql database
        database containing the TB data to extract the bcg coverage from
    :param _country_iso3: string
        iso3 code for country of interest
    :param start_year: int
        year in which bcg vaccination was assumed to have started at a significant programmatic level for the country
    :return: StratifiedModel class
        SUMMER model object with bcg vaccination functions added
    """

    # create dictionary of data
    bcg_coverage = get_bcg_coverage(_input_database, _country_iso3)
    bcg_coverage[start_year] = 0.0

    # fit function
    bcg_coverage_function = scale_up_function(bcg_coverage.keys(), bcg_coverage.values(), smoothness=0.2, method=5)

    # add to model object and return
    _tb_model.time_variants["bcg_coverage"] = bcg_coverage_function
    _tb_model.time_variants["bcg_coverage_complement"] = lambda value: 1.0 - bcg_coverage_function(value)
    return _tb_model


def get_birth_rate_functions(_tb_model, _input_database, _country_iso3):
    """
    add crude birth rate function to existing epidemiological model

    :param _tb_model: EpiModel or StratifiedModel class
        SUMMER model object to be assigned bcg vaccination coverage functions
    :param _input_database: sql database
        database containing the TB data to extract the bcg coverage from
    :param _country_iso3: string
        iso3 code for country of interest
    :return: EpiModel or StratifiedModel class
        SUMMER model object with birth rate function added
    """
    crude_birth_rate_data = get_crude_birth_rate(_input_database, _country_iso3)
    if _country_iso3 == 'MNG':  # provisional patch
        for year in crude_birth_rate_data.keys():
            if year > 1990.:
                crude_birth_rate_data[year] = 0.04

    _tb_model.time_variants["crude_birth_rate"] = \
        scale_up_function(crude_birth_rate_data.keys(), crude_birth_rate_data.values(), smoothness=0.2, method=5)
    return _tb_model


"""
main model construction functions
"""


def build_working_tb_model(tb_n_contact, country_iso3, cdr_adjustment=0.6, start_time=1800.):
    """
    current working tb model with some characteristics of mongolia applied at present
    """
    input_database = InputDB()

    integration_times = numpy.linspace(start_time, 2020.0, 201).tolist()

    # set basic parameters, flows and times, then functionally add latency
    case_fatality_rate = 0.4
    untreated_disease_duration = 3.0
    parameters = \
        {"contact_rate": tb_n_contact,
         "recovery": case_fatality_rate / untreated_disease_duration,
         "infect_death": (1.0 - case_fatality_rate) / untreated_disease_duration,
         "universal_death_rate": 1.0 / 50.0,
         "case_detection": 0.0,
         "crude_birth_rate": 20.0 / 1e3}
    parameters.update(change_parameter_unit(provide_aggregated_latency_parameters(), 365.251))

    # sequentially add groups of flows
    flows = add_standard_infection_flows([])
    flows = add_standard_latency_flows(flows)
    flows = add_standard_natural_history_flows(flows)

    # compartments
    compartments = ["susceptible", "early_latent", "late_latent", "infectious", "recovered"]

    # define model
    _tb_model = StratifiedModel(
        integration_times, compartments, {"infectious": 1e-3}, parameters, flows, birth_approach="add_crude_birth_rate")

    # add crude birth rate from un estimates
    _tb_model = get_birth_rate_functions(_tb_model, input_database, country_iso3)

    # add case detection process to basic model
    _tb_model.add_transition_flow(
        {"type": "standard_flows", "parameter": "case_detection", "origin": "infectious", "to": "recovered"})

    # create_flowchart(_tb_model, name="unstratified")

    # prepare age stratification
    age_breakpoints = [5, 15]
    age_infectiousness = get_parameter_dict_from_function(logistic_scaling_function(15.0), age_breakpoints)
    age_params = get_adapted_age_parameters(age_breakpoints)
    age_params.update(split_age_parameter(age_breakpoints, "contact_rate"))

    # test age stratification
    # age_only_model = copy.deepcopy(_tb_model)
    # age_only_model.stratify("age", copy.deepcopy(age_breakpoints), [], {},
    #                         adjustment_requests=age_params,
    #                         infectiousness_adjustments=age_infectiousness,
    #                         verbose=False)
    # create_flowchart(age_only_model, name="stratified_by_age")

    bcg_wane = create_sloping_step_function(15.0, 0.3, 30.0, 1.0)
    age_bcg_efficacy_dict = get_parameter_dict_from_function(lambda value: bcg_wane(value), age_breakpoints)
    bcg_efficacy = substratify_parameter("contact_rate", "vaccinated", age_bcg_efficacy_dict, age_breakpoints)

    pop_morts = get_pop_mortality_functions(input_database, age_breakpoints, country_iso_code=country_iso3)
    for age_break in age_breakpoints:
        _tb_model.time_variants["universal_death_rateXage_" + str(age_break)] = pop_morts[age_break]
    age_params["universal_death_rate"] = {"5W": "universal_death_rateXage_5"}

    # stratify the actual model by age
    _tb_model.stratify("age", age_breakpoints, [], {},
                       adjustment_requests=age_params,
                       infectiousness_adjustments=age_infectiousness,
                       verbose=False)

    # get bcg coverage function
    _tb_model = get_bcg_functions(_tb_model, input_database, country_iso3)

    # stratify by vaccination status
    _tb_model.stratify("bcg", ["vaccinated", "unvaccinated"], ["susceptible"],
                       requested_proportions={"vaccinated": 0.0},
                       entry_proportions={"vaccinated": "bcg_coverage",
                                          "unvaccinated": "bcg_coverage_complement"},
                       adjustment_requests=bcg_efficacy,
                       verbose=False)

    # create_flowchart(_tb_model, name="stratified_by_age_vaccination")

    # loading time-variant case detection rate
    input_database = InputDB()
    res = input_database.db_query("gtb_2015", column="c_cdr", is_filter="country", value="Mongolia")

    # add scaling case detection rate
    cdr_adjustment_factor = cdr_adjustment
    cdr_mongolia = res["c_cdr"].values / 1e2 * cdr_adjustment_factor
    cdr_mongolia = numpy.concatenate(([0.0], cdr_mongolia))
    res = input_database.db_query("gtb_2015", column="year", is_filter="country", value="Mongolia")
    cdr_mongolia_year = res["year"].values
    cdr_mongolia_year = numpy.concatenate(([1950.], cdr_mongolia_year))
    cdr_scaleup = scale_up_function(cdr_mongolia_year, cdr_mongolia, smoothness=0.2, method=5)
    prop_to_rate = convert_competing_proportion_to_rate(1.0 / untreated_disease_duration)
    detect_rate = return_function_of_function(cdr_scaleup, prop_to_rate)
    _tb_model.time_variants["case_detection"] = detect_rate

    # store scaling functions in database if required
    # function_dataframe = pd.DataFrame(times)
    # function_dataframe["cdr_values"] = [cdr_scaleup(t) for t in times]
    # store_database(function_dataframe, table_name="functions")

    # test strain stratification
    # strain_only_model = copy.deepcopy(_tb_model)
    # strain_only_model.stratify("strain", ["ds", "mdr"], ["early_latent", "late_latent", "infectious"], {},
    #                            verbose=False)
    # create_flowchart(strain_only_model, name="stratified_by_strain")

    # test organ stratification
    # organ_only_model = copy.deepcopy(_tb_model)
    # organ_only_model.stratify("smear",
    #                           ["smearpos", "smearneg", "extrapul"],
    #                           ["infectious"], adjustment_requests={}, verbose=False, requested_proportions={})
    # create_flowchart(organ_only_model, name="stratified_by_organ")

    # test risk stratification
    # risk_only_model = copy.deepcopy(_tb_model)
    # risk_only_model.stratify("risk",
    #                          ["urban", "urbanpoor", "ruralpoor"], [], requested_proportions={},
    #                          adjustment_requests={}, verbose=False)
    # create_flowchart(risk_only_model, name="stratified_by_risk")

    # _tb_model.stratify("strain", ["ds", "mdr"], ["early_latent", "late_latent", "infectious"], {},
    #                    verbose=False)
    # _tb_model.stratify("smear",
    #                    ["smearpos", "smearneg", "extrapul"],
    #                    ["infectious"],
    #                    infectiousness_adjustments={"smearneg": 0.24, "extrapul": 0.0},
    #                    verbose=False, requested_proportions={})
    # _tb_model.stratify("risk",
    #                    ["urban", "urbanpoor", "ruralpoor"], [], requested_proportions={},
    #                    adjustment_requests=[], verbose=False)
    return _tb_model


def create_multi_scenario_outputs(models, req_outputs, req_times={}, req_multipliers={}, out_dir='outputs_tes',
                                  targets_to_plot={}, translation_dictionary={}, scenario_list=[]):
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
                req_outputs.append('distribution_of_strataX' + group)
                for stratum in models[scenario_index].all_stratifications[group]:
                    req_outputs.append('prevXinfectiousXamongX' + group + '_' + stratum)
                    req_outputs.append('prevXlatentXamongX' + group + '_' + stratum)

            if "strain" in models[scenario_index].all_stratifications.keys():
                req_outputs.append('prevXinfectiousXstrain_mdrXamongXinfectious')

        for output in req_outputs:
            if output[0:15] == 'prevXinfectious' and output != 'prevXinfectiousXstrain_mdrXamongXinfectious':
                req_multipliers[output] = 1.e5
            elif output[0:11] == 'prevXlatent':
                req_multipliers[output] = 1.e2

        pps.append(post_proc.PostProcessing(models[scenario_index], requested_outputs=req_outputs,
                                            scenario_number=scenario_list[scenario_index],
                                            requested_times=req_times,
                                            multipliers=req_multipliers))

    outputs = Outputs(pps, targets_to_plot, out_dir, translation_dictionary)
    outputs.plot_requested_outputs()

    for req_output in ['prevXinfectious', 'prevXlatent']:
        for sc_index in range(len(models)):
            outputs.plot_outputs_by_stratum(req_output, sc_index=sc_index)



class DummyModel:
    def __init__(self, model_dict):
        self.compartment_names = [name for name in model_dict.keys() if name not in ['idx', 'Scenario', 'times']]

        self.outputs = numpy.column_stack([list(column.values()) for name, column in model_dict.items()if
                                           name not in ['idx', 'Scenario', 'times']])
        self.times = list(model_dict['times'].values())

if __name__ == "__main__":

    for country in ["MNG"]:

        tb_model = build_working_tb_model(40.0, country)

        # create_flowchart(tb_model, name="mongolia_flowchart")
        # tb_model.transition_flows.to_csv("transitions.csv")

        # tb_model.run_model()

    # get outputs
    # infectious_population = tb_model.get_total_compartment_size(["infectious"])

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
    # matplotlib.pyplot.plot(numpy.linspace(1800., 2020.0, 201).tolist(), infectious_population * 1e5)
    # matplotlib.pyplot.xlim((1980., 2020.))
    # matplotlib.pyplot.ylim((0.0, 1e3))
    # matplotlib.pyplot.show()
    # matplotlib.pyplot.savefig("mongolia_cdr_output")



