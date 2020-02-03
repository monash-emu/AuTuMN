import dill

from autumn.tb_model import *
from autumn.tool_kit import (
    initialise_scenario_run,
    change_parameter_unit,
)


def build_timevariant_cdr():
    """
    Interpolate case detection rate
    """
    cdr = {1950.: 0., 1980.: .10, 1990.: .15, 2000.: .20, 2010.: .30, 2015: .33}
    return scale_up_function(cdr.keys(), cdr.values(), smoothness=0.2, method=5)


def build_timevariant_tsr():
    """
    Interpolate treatment success rate
    """
    tsr = {1950.: 0., 1970.: .2, 1994.: .6, 2000.: .85, 2010.: .87, 2016: .9}
    return scale_up_function(tsr.keys(), tsr.values(), smoothness=0.2, method=5)


def build_model(update_params={}):

    stratify_by = ['age', 'strain'] # , 'location']  #, 'location', 'housing']

    # some default parameter values
    external_params = {  # run configuration
                       'start_time': 1800.,
                       'end_time': 2050.,
                       'time_step': 1.,
                       'start_population': 105000000,
                         # base model definition:
                       'contact_rate': 4.6,
                       'rr_transmission_recovered': .21,
                       'rr_transmission_infected': 0.21,
                       'latency_adjustment': 5.,  # used to modify progression rates during calibration
                       'case_fatality_rate': 0.4,
                       'untreated_disease_duration': 3.0,
                         # MDR-TB:
                       'dr_amplification_prop_among_nonsuccess': 0.15,
                       'prop_mdr_detected_as_mdr': 0.5,
                       'mdr_tsr': .6,
                         # adjustments by location and housing type
                       'rr_transmission_ger': 1.8,  # reference: non-ger
                       'rr_transmission_urban': 1.4,  # reference: rural
                       'rr_transmission_province': .9,  # reference: rural
                       # IPT
                       'n_ipt': 0.,
                       'n_ipt_age_0': 0., 'n_ipt_age_5': 0., 'n_ipt_age_10': 0., 'n_ipt_age_15': 0.,
                       'n_ipt_age_30': 0., 'n_ipt_age_50': 0., 'n_ipt_age_60': 0,
                       'ipt_age_0_ct_coverage': 0.17,  # Children contact tracing coverage  .17
                       'ipt_age_5_ct_coverage': 0.,  # Children contact tracing coverage
                       'ipt_age_15_ct_coverage': 0.,  # Children contact tracing coverage
                       'ipt_age_60_ct_coverage': 0.,  # Children contact tracing coverage
                       'yield_contact_ct_tstpos_per_detected_tb': 2.,  # expected number of infections traced per index
                       'ipt_efficacy': .9,   # based on intention-to-treat
                       'ds_ipt_switch': 1.,  # used as a DS-specific multiplier to the coverage defined above
                       'mdr_ipt_switch': .0,  # used as an MDR-specific multiplier to the coverage defined above
                       # Treatment improvement (C-DOTS)
                       'reduction_negative_tx_outcome': 0.,
                         # ACF for risk groups
                       'acf_coverage': 0.,
                       'acf_sensitivity': .8,
                       'acf_ger_switch': 0.,
                       'acf_non-ger_switch': 0.,
                       'acf_rural_switch': 0.,
                       'acf_province_switch': 0.,
                       'acf_urban_switch': 0.
                       }
    # update external_params with new parameter values found in update_params
    external_params.update(update_params)

    model_parameters = \
        {"contact_rate": external_params['contact_rate'],
         "contact_rate_recovered": external_params['contact_rate'] * external_params['rr_transmission_recovered'],
         "contact_rate_infected": external_params['contact_rate'] * external_params['rr_transmission_infected'],
         "recovery": external_params['case_fatality_rate'] / external_params['untreated_disease_duration'],
         "infect_death": (1.0 - external_params['case_fatality_rate']) / external_params['untreated_disease_duration'],
         "universal_death_rate": 1.0 / 50.0,
         "case_detection": 0.,
         "ipt_rate": 0.,
         "n_ipt": external_params['n_ipt'],
         "acf_rate": 0.,
         "dr_amplification": .0,  # high value for testing
         "crude_birth_rate": 20.0 / 1e3}

    input_database = InputDB()
    n_iter = int(round((external_params['end_time'] - external_params['start_time']) / external_params['time_step'])) + 1
    integration_times = numpy.linspace(external_params['start_time'], external_params['end_time'], n_iter).tolist()

    model_parameters.update(change_parameter_unit(provide_aggregated_latency_parameters(), 365.251))

    # sequentially add groups of flows
    flows = add_standard_infection_flows([])
    flows = add_standard_latency_flows(flows)
    flows = add_standard_natural_history_flows(flows)

    # compartments
    compartments = ["susceptible", "early_latent", "late_latent", "infectious", "recovered"]

    # define model     #replace_deaths  add_crude_birth_rate
    if len(stratify_by) > 0:
        _tb_model = StratifiedModel(
            integration_times, compartments, {"infectious": 1e-3}, model_parameters, flows, birth_approach="replace_deaths",
            starting_population=external_params['start_population'])
    else:
        _tb_model = EpiModel(
            integration_times, compartments, {"infectious": 1e-3}, model_parameters, flows, birth_approach="replace_deaths",
            starting_population=external_params['start_population'])

    # add crude birth rate from un estimates
    _tb_model = get_birth_rate_functions(_tb_model, input_database, 'MNG')

    # add case detection process to basic model
    _tb_model.add_transition_flow(
        {"type": "standard_flows", "parameter": "case_detection", "origin": "infectious", "to": "recovered"})

    # Add IPT as a customised flow
    def early_ipt_flow_func(model, n_flow):
        no_age_stratifications = {key: value for key, value in model.all_stratifications.items() if key != 'age'}
        n_pop_subgroups = 1.
        for stratification in no_age_stratifications.keys():
            n_pop_subgroups *= float(len(no_age_stratifications[stratification]))

        index_early_cpt = model.compartment_names.index(model.transition_flows.origin[n_flow])
        index_late_cpt = model.compartment_names.index(model.transition_flows.origin[n_flow].replace('early_latent', 'late_latent'))

        time_step = 0. if len(model.times) < 2 else (model.times[-1] - model.times[-2])

        n_early_latent = model.compartment_values[index_early_cpt]
        n_late_latent = model.compartment_values[index_late_cpt]
        if n_early_latent+n_late_latent > 0.:
            return n_early_latent / (n_early_latent + n_late_latent) / n_pop_subgroups * time_step * external_params['ipt_efficacy']
        else:
            return 0.

    def late_ipt_flow_func(model, n_flow):
        no_age_stratifications = {key: value for key, value in model.all_stratifications.items() if key != 'age'}
        n_pop_subgroups = 1.
        for stratification in no_age_stratifications.keys():
            n_pop_subgroups *= float(len(no_age_stratifications[stratification]))

        index_late_cpt = model.compartment_names.index(model.transition_flows.origin[n_flow])
        index_early_cpt = model.compartment_names.index(model.transition_flows.origin[n_flow].replace('late_latent', 'early_latent'))

        time_step = 0. if len(model.times) < 2 else (model.times[-1] - model.times[-2])

        n_early_latent = model.compartment_values[index_early_cpt]
        n_late_latent = model.compartment_values[index_late_cpt]

        if n_early_latent+n_late_latent > 0.:
            return n_late_latent / (n_early_latent + n_late_latent) / n_pop_subgroups * time_step * external_params['ipt_efficacy']
        else:
            return 0.

    _tb_model.add_transition_flow(
        {"type": "customised_flows", "parameter": "n_ipt", "origin": "early_latent", "to": "recovered",
         "function": early_ipt_flow_func})
    _tb_model.add_transition_flow(
        {"type": "customised_flows", "parameter": "n_ipt", "origin": "late_latent", "to": "recovered",
         "function": late_ipt_flow_func})

    # add ACF flow
    _tb_model.add_transition_flow(
        {"type": "standard_flows", "parameter": "acf_rate", "origin": "infectious", "to": "recovered"})

    # load time-variant case detection rate
    cdr_scaleup = build_timevariant_cdr()
    prop_to_rate = convert_competing_proportion_to_rate(1.0 / external_params['untreated_disease_duration'])
    detect_rate = return_function_of_function(cdr_scaleup, prop_to_rate)

    # load time-variant treatment success rate
    mongolia_tsr = build_timevariant_tsr()

    # create a tb_control_recovery_rate function combining case detection and treatment succes rates
    tb_control_recovery_rate = \
        lambda t: detect_rate(t) *\
                  (mongolia_tsr(t) + external_params['reduction_negative_tx_outcome'] * (1. - mongolia_tsr(t)))

    # initialise ipt_rate function assuming coverage of 1.0 before age stratification
    ipt_rate_function = lambda t: detect_rate(t) * 1.0 *\
                                  external_params['yield_contact_ct_tstpos_per_detected_tb'] * external_params['ipt_efficacy']

    # initialise acf_rate function
    acf_rate_function = lambda t: external_params['acf_coverage'] * external_params['acf_sensitivity'] *\
                                  (mongolia_tsr(t) + external_params['reduction_negative_tx_outcome'] * (1. - mongolia_tsr(t)))

    # assign newly created functions to model parameters
    if len(stratify_by) == 0:
        _tb_model.time_variants["case_detection"] = tb_control_recovery_rate
        _tb_model.time_variants["ipt_rate"] = ipt_rate_function
        _tb_model.time_variants["acf_rate"] = acf_rate_function

    else:
        _tb_model.adaptation_functions["case_detection"] = tb_control_recovery_rate
        _tb_model.parameters["case_detection"] = "case_detection"

        _tb_model.adaptation_functions["ipt_rate"] = ipt_rate_function
        _tb_model.parameters["ipt_rate"] = "ipt_rate"

        _tb_model.adaptation_functions["acf_rate"] = acf_rate_function
        _tb_model.parameters["acf_rate"] = "acf_rate"

    if "strain" in stratify_by:
        mdr_adjustment = external_params['prop_mdr_detected_as_mdr'] * external_params['mdr_tsr'] / .9  # /.9 for last DS TSR

        _tb_model.stratify("strain", ["ds", "mdr"], ["early_latent", "late_latent", "infectious"], verbose=False,
                           requested_proportions={"mdr": 0.},
                           adjustment_requests={
                               'contact_rate': {'ds': 1., 'mdr': 1.},
                               'case_detection': {"mdr": mdr_adjustment},
                               'ipt_rate': {"ds": 1., #external_params['ds_ipt_switch'],
                                            "mdr": external_params['mdr_ipt_switch']}
                           })

        _tb_model.add_transition_flow(
            {"type": "standard_flows", "parameter": "dr_amplification",
             "origin": "infectiousXstrain_ds", "to": "infectiousXstrain_mdr",
             "implement": len(_tb_model.all_stratifications)})

        dr_amplification_rate = \
            lambda t: detect_rate(t) * (1. - mongolia_tsr(t)) *\
                      (1. - external_params['reduction_negative_tx_outcome']) *\
                      external_params['dr_amplification_prop_among_nonsuccess']

        _tb_model.adaptation_functions["dr_amplification"] = dr_amplification_rate
        _tb_model.parameters["dr_amplification"] = "dr_amplification"

    if 'smear' in stratify_by:
        props_smear = {"smearpos": 0.5, "smearneg": 0.25, "extrapul": 0.25}
        _tb_model.stratify("smear", ["smearpos", "smearneg", "extrapul"], ["infectious"],
                           infectiousness_adjustments={"smearpos": 1., "smearneg": 0.25, "extrapul": 0.},
                           verbose=False, requested_proportions=props_smear,
                           entry_proportions=props_smear)

    if "age" in stratify_by:
        age_breakpoints = [0, 5, 10, 15, 30, 50, 60]
        age_infectiousness = get_parameter_dict_from_function(logistic_scaling_function(10.0), age_breakpoints)
        age_params = get_adapted_age_parameters(age_breakpoints)
        age_params.update(split_age_parameter(age_breakpoints, "contact_rate"))

        # adjustment of latency parameters
        for param in ['late_progression']:
            for age_break in age_breakpoints:
                age_params[param][str(age_break) + 'W'] *= external_params['latency_adjustment']

        pop_morts = get_pop_mortality_functions(input_database, age_breakpoints, country_iso_code='MNG')
        age_params["universal_death_rate"] = {}
        for age_break in age_breakpoints:
            _tb_model.time_variants["universal_death_rateXage_" + str(age_break)] = pop_morts[age_break]
            _tb_model.parameters["universal_death_rateXage_" + str(age_break)] = "universal_death_rateXage_" + str(age_break)

            age_params["universal_death_rate"][str(age_break) + 'W'] = "universal_death_rateXage_" + str(age_break)
        _tb_model.parameters["universal_death_rateX"] = 0.

        # age-specific IPT
        ipt_by_age = {'n_ipt': {}}
        for age_break in age_breakpoints:
            ipt_by_age['n_ipt'][str(age_break) + 'W'] = external_params['n_ipt_age_' + str(age_break)]
        age_params.update(ipt_by_age)

        # add BCG effect without stratification assuming constant 100% coverage
        bcg_wane = create_sloping_step_function(15.0, 0.3, 30.0, 1.0)
        age_bcg_efficacy_dict = get_parameter_dict_from_function(lambda value: bcg_wane(value), age_breakpoints)
        age_params.update({'contact_rate': age_bcg_efficacy_dict})

        age_mixing = numpy.ones(len(age_breakpoints)**2).reshape((len(age_breakpoints), len(age_breakpoints)))
        for i in range(len(age_breakpoints)):
            age_mixing[i, i] = 4.
        age_mixing[0, 3] = 3.
        age_mixing[3, 0] = 3.
        age_mixing[1, 3] = 3.
        age_mixing[3, 1] = 3.
        age_mixing[1, 4] = 2.
        age_mixing[4, 1] = 2.

        _tb_model.stratify("age", copy.deepcopy(age_breakpoints), [], {}, adjustment_requests=age_params,
                           infectiousness_adjustments=age_infectiousness, verbose=False, mixing_matrix=age_mixing)

        # patch for IPT to overwrite parameters when ds_ipt has been turned off while we still need some coverage at baseline
        if external_params['ds_ipt_switch'] == 0. and external_params['mdr_ipt_switch'] == 1.:
            _tb_model.parameters['ipt_rateXstrain_dsXage_0'] = 0.17
            for age_break in [5, 15, 60]:
                _tb_model.parameters['ipt_rateXstrain_dsXage_' + str(age_break)] = 0.

    #
    # if 'bcg' in stratify_by:
    #      # get bcg coverage function
    #     _tb_model = get_bcg_functions(_tb_model, input_database, 'MNG')
    #
    #     # stratify by vaccination status
    #     bcg_wane = create_sloping_step_function(15.0, 0.3, 30.0, 1.0)
    #     age_bcg_efficacy_dict = get_parameter_dict_from_function(lambda value: bcg_wane(value), age_breakpoints)
    #     bcg_efficacy = substratify_parameter("contact_rate", "vaccinated", age_bcg_efficacy_dict, age_breakpoints)
    #     _tb_model.stratify("bcg", ["vaccinated", "unvaccinated"], ["susceptible"],
    #                        requested_proportions={"vaccinated": 0.0},
    #                        entry_proportions={"vaccinated": "bcg_coverage",
    #                                           "unvaccinated": "bcg_coverage_complement"},
    #                        adjustment_requests=bcg_efficacy,
    #                        verbose=False)

    if "housing" in stratify_by:
        props_housing = {"ger": .45, "non-ger": .55}
        raw_relative_risks = {"ger": external_params['rr_transmission_ger'], "non-ger": 1.}
        scaled_relative_risks = scale_relative_risks_for_equivalence(props_housing, raw_relative_risks)

        housing_mixing = numpy.ones(4).reshape((2, 2))
        # housing_mixing[0, 0] = 5.
        # housing_mixing[1, 1] = 5.
        housing_adjustments = {}
        for beta_type in ['', '_infected', '_recovered']:
            housing_adjustments['contact_rate' + beta_type] = scaled_relative_risks

        housing_adjustments['acf_rate'] = {}
        for stratum in ['ger', 'non-ger']:
            housing_adjustments['acf_rate'][stratum] = external_params['acf_' + stratum + '_switch']

        _tb_model.stratify("housing", ["ger", "non-ger"], [], requested_proportions=props_housing, verbose=False,
                           adjustment_requests=housing_adjustments,
                           mixing_matrix=housing_mixing,
                           entry_proportions=props_housing
                           )

    if "location" in stratify_by:
        props_location = {"rural": .32, 'province': .16, "urban": .52}
        raw_relative_risks_loc = {"rural": 1., "province":external_params['rr_transmission_province'],
                                  "urban": external_params['rr_transmission_urban']}
        scaled_relative_risks_loc = scale_relative_risks_for_equivalence(props_location, raw_relative_risks_loc)

        location_mixing = numpy.ones(9).reshape((3, 3))
        # location_mixing[0, 0] = 10.
        # location_mixing[1, 1] = 10.
        # location_mixing[2, 2] = 10.

        location_adjustments = {}
        for beta_type in ['', '_infected', '_recovered']:
            location_adjustments['contact_rate' + beta_type] = scaled_relative_risks_loc

        location_adjustments['acf_rate'] = {}
        for stratum in ['rural', 'province', 'urban']:
            location_adjustments['acf_rate'][stratum] = external_params['acf_' + stratum + '_switch']

        _tb_model.stratify("location", ["rural", "province", "urban"], [],
                           requested_proportions=props_location, verbose=False, entry_proportions=props_location,
                           adjustment_requests=location_adjustments,
                           mixing_matrix=location_mixing
                           )

    _tb_model.transition_flows.to_csv("transitions.csv")
    # _tb_model.death_flows.to_csv("deaths.csv")

    return _tb_model


def objective_function(resource_allocation):
    if not os.path.exists('stored_baseline.pickle'):
        run_baseline()
    my_model = run_scenario(resource_allocation)
    out = get_model_output(my_model)
    return out


def run_baseline(store=True):
    baseline_model = build_model()
    print("____________________  Now running Baseline Scenario ")
    baseline_model.run_model()
    if store:
        file_name = "stored_baseline.pickle"
        file_stream = open(file_name, "wb")

        attributes_to_store = ['times', 'compartment_names', 'outputs', 'all_stratifications']

        model_as_dict = {}

        for att in attributes_to_store:
            model_as_dict[att] = getattr(baseline_model, att)

        dill.dump(model_as_dict, file_stream)
        file_stream.close()

    return baseline_model


def run_scenario(scenario_params={'start_time': 2020.}, baseline_was_run=True):
    if 'start_time' not in scenario_params.keys():
        scenario_params['start_time'] = 2020.

    if baseline_was_run:
        file_stream = open('stored_baseline.pickle', "rb")
        baseline_model = dill.load(file_stream)
    else:
        baseline_model = run_baseline()

    scenario_model = initialise_scenario_run(baseline_model,scenario_params, build_model)
    print("Now running scenario model")
    scenario_model.run_model()

    return scenario_model


def get_model_output(run_model):
    req_output = ['prevXinfectiousXamong']
    multipliers = {'prevXinfectiousXamong': 1.e5}
    req_times = {'prevXinfectiousXamong': [2049.]}
    pp = post_proc.PostProcessing(run_model, requested_outputs=req_output, requested_times=req_times,
                                  multipliers=multipliers)
    return pp.generated_outputs['prevXinfectiousXamong'][0]


if __name__ == "__main__":

    # the total number of treatments available is 600,000
    resource_allocation = {
        'n_ipt_age_0': 100000,
        'n_ipt_age_5': 100000,
        'n_ipt_age_10': 100000,
        'n_ipt_age_15': 100000,
        'n_ipt_age_30': 100000,
        'n_ipt_age_50': 100000
    }

    obj = objective_function(resource_allocation)
    print(obj)
