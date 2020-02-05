"""
Test an example TB model end-to-end.
"""
import os

import pytest
import numpy
from summer_py.summer_model import (
    StratifiedModel,
    split_age_parameter,
    create_sloping_step_function,
)
from summer_py.parameter_processing import (
    get_parameter_dict_from_function,
    logistic_scaling_function,
)

from autumn import constants
from autumn.constants import Compartment, Flow
from autumn.db import Database
from autumn.tool_kit import utils
from autumn.tb_model import (
    provide_aggregated_latency_parameters,
    get_adapted_age_parameters,
    convert_competing_proportion_to_rate,
    add_standard_latency_flows,
    add_standard_natural_history_flows,
    add_standard_infection_flows,
    get_birth_rate_functions,
)

COUNTRY_ISO3 = "MNG"  # Mongolia
CONTACT_RATE = 40.0
RR_TRANSMISSION_RECOVERED = 1
RR_TRANSMISSION_INFECTED = 0.21
CDR_ADJUSTMENT = 0.6
START_TIME = 1800
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")


@pytest.mark.xfail(reason="Function 'substratify_parameter' doesn't exist anymore.")
def test_example_model_for_regressions():
    """
    Ensure the example model produces the same results and builds and runs without crashing.
    """
    input_database = Database(INPUT_DB_PATH)
    integration_times = numpy.linspace(START_TIME, 2020.0, 201).tolist()

    # Set basic parameters, flows and times, then functionally add latency
    case_fatality_rate = 0.4
    untreated_disease_duration = 3.0
    parameters = {
        "contact_rate": CONTACT_RATE,
        "contact_rate_recovered": CONTACT_RATE * RR_TRANSMISSION_RECOVERED,
        "contact_rate_infected": CONTACT_RATE * RR_TRANSMISSION_INFECTED,
        "recovery": case_fatality_rate / untreated_disease_duration,
        "infect_death": (1.0 - case_fatality_rate) / untreated_disease_duration,
        "universal_death_rate": 1.0 / 50.0,
        "case_detection": 0.0,
        "crude_birth_rate": 20.0 / 1e3,
    }
    latency_params = utils.change_parameter_unit(provide_aggregated_latency_parameters(), 365.251)
    parameters.update(latency_params)

    # Sequentially add groups of flows
    flows = add_standard_infection_flows([])
    flows = add_standard_latency_flows(flows)
    flows = add_standard_natural_history_flows(flows)

    # Model compartments
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EARLY_LATENT,
        Compartment.LATE_LATENT,
        Compartment.INFECTIOUS,
        Compartment.RECOVERED,
    ]

    # Define model
    tb_model = StratifiedModel(
        integration_times,
        compartments,
        {"infectious": 1e-3},
        parameters,
        flows,
        birth_approach="add_crude_birth_rate",
    )

    # Add crude birth rate from un estimates
    tb_model = get_birth_rate_functions(tb_model, input_database, COUNTRY_ISO3)

    # Add case detection process to basic model
    tb_model.add_transition_flow(
        {
            "type": Flow.STANDARD,
            "parameter": "case_detection",
            "origin": Compartment.INFECTIOUS,
            "to": Compartment.RECOVERED,
        }
    )

    # Prepare age stratification
    age_breakpoints = [5, 15]
    age_infectiousness = get_parameter_dict_from_function(
        logistic_scaling_function(15.0), age_breakpoints
    )
    age_params = get_adapted_age_parameters(age_breakpoints)
    age_params.update(split_age_parameter(age_breakpoints, "contact_rate"))

    bcg_wane = create_sloping_step_function(15.0, 0.3, 30.0, 1.0)
    age_bcg_efficacy_dict = get_parameter_dict_from_function(
        lambda value: bcg_wane(value), age_breakpoints
    )
    bcg_efficacy = substratify_parameter(
        "contact_rate", "vaccinated", age_bcg_efficacy_dict, age_breakpoints
    )

    pop_morts = get_pop_mortality_functions(
        input_database, age_breakpoints, country_iso_code=COUNTRY_ISO3
    )
    for age_break in age_breakpoints:
        tb_model.time_variants["universal_death_rateXage_" + str(age_break)] = pop_morts[age_break]

    age_params["universal_death_rate"] = {"5W": "universal_death_rateXage_5"}

    # Stratify the model by age
    tb_model.stratify(
        "age",
        age_breakpoints,
        [],
        {},
        adjustment_requests=age_params,
        infectiousness_adjustments=age_infectiousness,
        verbose=False,
    )

    # Get bcg coverage function
    tb_model = get_bcg_functions(tb_model, input_database, COUNTRY_ISO3)

    # Stratify by vaccination status
    tb_model.stratify(
        "bcg",
        ["vaccinated", "unvaccinated"],
        ["susceptible"],
        requested_proportions={"vaccinated": 0.0},
        entry_proportions={"vaccinated": "bcg_coverage", "unvaccinated": "bcg_coverage_complement"},
        adjustment_requests=bcg_efficacy,
        verbose=False,
    )

    # Load time-variant case detection rate
    input_database = Database(INPUT_DB_PATH)
    res = input_database.db_query("gtb_2015", column="c_cdr", is_filter="country", value="Mongolia")

    # Sdd scaling case detection rate
    cdr_adjustment_factor = CDR_ADJUSTMENT
    cdr_mongolia = res["c_cdr"].values / 1e2 * cdr_adjustment_factor
    cdr_mongolia = numpy.concatenate(([0.0], cdr_mongolia))
    res = input_database.db_query("gtb_2015", column="year", is_filter="country", value="Mongolia")
    cdr_mongolia_year = res["year"].values
    cdr_mongolia_year = numpy.concatenate(([1950.0], cdr_mongolia_year))
    cdr_scaleup = scale_up_function(cdr_mongolia_year, cdr_mongolia, smoothness=0.2, method=5)
    prop_to_rate = convert_competing_proportion_to_rate(1.0 / untreated_disease_duration)
    detect_rate = utils.return_function_of_function(cdr_scaleup, prop_to_rate)
    tb_model.time_variants["case_detection"] = detect_rate

    # Run the model
    tb_model.run_model()
    import pdb

    pdb.set_trace()
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
