from autumn_from_summer.tb_model import *
import summer_py.post_processing as post_proc
from summer_py.outputs import Outputs


def build_model_for_calibration(start_time=1800.):
    input_database = InputDB()

    integration_times = numpy.linspace(start_time, 2020.0, 201).tolist()

    # set basic parameters, flows and times, then functionally add latency
    case_fatality_rate = 0.4
    untreated_disease_duration = 3.0
    parameters = \
        {"contact_rate": 20.,
         "recovery": case_fatality_rate / untreated_disease_duration,
         "infect_death": (1.0 - case_fatality_rate) / untreated_disease_duration,
         "universal_death_rate": 1.0 / 50.0,
         "case_detection": 0.,
         "dr_amplification": 100.,  # high value for testing
         "crude_birth_rate": 20.0 / 1e3}
    parameters.update(change_parameter_unit(provide_aggregated_latency_parameters(), 365.251))

    # sequentially add groups of flows
    flows = add_standard_infection_flows([])
    flows = add_standard_latency_flows(flows)
    flows = add_standard_natural_history_flows(flows)

    # compartments
    compartments = ["susceptible", "early_latent", "late_latent", "infectious", "recovered"]

    # define model     #replace_deaths
    _tb_model = StratifiedModel(
        integration_times, compartments, {"infectious": 1e-3}, parameters, flows, birth_approach="replace_deaths")

     # add case detection process to basic model
    _tb_model.add_transition_flow(
        {"type": "standard_flows", "parameter": "case_detection", "origin": "infectious", "to": "recovered"})

    _tb_model.stratify("strain", ["ds", "mdr"], ["early_latent", "late_latent", "infectious"], verbose=False,
                       requested_proportions={"mdr": 0.0000001})

    _tb_model.transition_flows.to_csv("temp_flows.csv")

    _tb_model.add_transition_flow(
        {"type": "standard_flows", "parameter": "dr_amplification",
         "origin": "infectiousXstrain_ds", "to": "infectiousXstrain_mdr",
         "implement": len(_tb_model.all_stratifications)})   # not working

    print(len(_tb_model.all_stratifications))

    _tb_model.transition_flows.to_csv("temp_flows_2.csv")
    create_flowchart(_tb_model)

    # age stratification
    age_breakpoints = [5, 15]
    age_infectiousness = get_parameter_dict_from_function(logistic_scaling_function(15.0), age_breakpoints)
    age_params = get_adapted_age_parameters(age_breakpoints)
    age_params.update(split_age_parameter(age_breakpoints, "contact_rate"))

    _tb_model.stratify("age", copy.deepcopy(age_breakpoints), [], {}, adjustment_requests=age_params,
                       infectiousness_adjustments=age_infectiousness, verbose=False)


     # get bcg coverage function
    _tb_model = get_bcg_functions(_tb_model, input_database, 'MNG')

    # stratify by vaccination status
    bcg_wane = create_sloping_step_function(15.0, 0.3, 30.0, 1.0)
    age_bcg_efficacy_dict = get_parameter_dict_from_function(lambda value: bcg_wane(value), age_breakpoints)
    bcg_efficacy = substratify_parameter("contact_rate", "vaccinated", age_bcg_efficacy_dict, age_breakpoints)
    _tb_model.stratify("bcg", ["vaccinated", "unvaccinated"], ["susceptible"],
                       requested_proportions={"vaccinated": 0.0},
                       entry_proportions={"vaccinated": "bcg_coverage",
                                          "unvaccinated": "bcg_coverage_complement"},
                       adjustment_requests=bcg_efficacy,
                       verbose=False)

    # # loading time-variant case detection rate
    # input_database = InputDB()
    # res = input_database.db_query("gtb_2015", column="c_cdr", is_filter="country", value="Mongolia")

    # # add scaling case detection rate
    # cdr_adjustment_factor = 0.
    # cdr_mongolia = res["c_cdr"].values / 1e2 * cdr_adjustment_factor
    # cdr_mongolia = numpy.concatenate(([0.0], cdr_mongolia))
    # res = input_database.db_query("gtb_2015", column="year", is_filter="country", value="Mongolia")
    # cdr_mongolia_year = res["year"].values
    # cdr_mongolia_year = numpy.concatenate(([1950.], cdr_mongolia_year))
    # cdr_scaleup = scale_up_function(cdr_mongolia_year, cdr_mongolia, smoothness=0.2, method=5)
    # prop_to_rate = convert_competing_proportion_to_rate(1.0 / untreated_disease_duration)
    # detect_rate = return_function_of_function(cdr_scaleup, prop_to_rate)
    #
    # _tb_model.time_variants["case_detection"] = detect_rate

    # create_flowchart(_tb_model)

    # _tb_model.stratify("smear", ["smearpos", "smearneg", "extrapul"], ["infectious"],
    #                    adjustment_requests={}, verbose=False, requested_proportions={})

    return _tb_model


if __name__ == "__main__":
    mongolia_model = build_model_for_calibration()
    mongolia_model.run_model()

    req_outputs = ['distribution_of_strataXage',
                   'distribution_of_strataXstrain',
                   #'distribution_of_strataXsmear',
                   'prevXinfectiousXamong',
                   'prevXlatentXamong',
                   'prevXlatentXamongXage_5']

    req_multipliers = {'prevXinfectiousXamong': 1.e5}

    pp = post_proc.PostProcessing(mongolia_model, req_outputs, multipliers=req_multipliers)

    # generate outputs
    outputs = Outputs(pp)
    outputs.plot_requested_outputs()
