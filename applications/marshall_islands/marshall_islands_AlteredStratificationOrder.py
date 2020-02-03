import os
from datetime import datetime
from time import time
from copy import deepcopy

import numpy
import pandas as pd
from summer_py.summer_model import StratifiedModel, split_age_parameter, create_sloping_step_function, create_flowchart
from summer_py.parameter_processing import get_parameter_dict_from_function, logistic_scaling_function

from autumn import constants
from autumn.curve import scale_up_function
from autumn.db import Database, get_pop_mortality_functions
from autumn.tb_model import (
    add_combined_incidence,
    load_model_scenario,
    load_calibration_from_db,
    provide_aggregated_latency_parameters,
    get_adapted_age_parameters,
    convert_competing_proportion_to_rate,
    return_function_of_function,
    store_run_models,
    add_standard_latency_flows,
    add_standard_natural_history_flows,
    add_standard_infection_flows,
    get_birth_rate_functions,
    create_multi_scenario_outputs,
    DummyModel,
)
from autumn.tool_kit import (
    run_multi_scenario,
    progressive_step_function_maker,
    change_parameter_unit,
)

# Database locations
file_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
OUTPUT_DB_PATH = os.path.join(file_dir, 'databases', f'outputs_{timestamp}.db')
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, 'inputs.db')


def build_rmi_timevariant_cdr(cdr_multiplier):
    cdr = {1950.: 0., 1980.: .10, 1990.: .1, 2000.: .2, 2010.: .3, 2015: .3}
    return scale_up_function(cdr.keys(), [c * cdr_multiplier for c in list(cdr.values())], smoothness=0.2, method=5)

def build_rmi_timevariant_tsr():
    tsr = {1950.: 0., 1970.: .2, 1994.: .6, 2000.: .85, 2010.: .87, 2016: .87}
    return scale_up_function(tsr.keys(), tsr.values(), smoothness=0.2, method=5)


def build_rmi_model(update_params={}):

    # stratify_by = ['location']
    # stratify_by = ['age']
    stratify_by = ['age', 'organ']
    # stratify_by = ['age', 'diabetes', 'organ']
    # stratify_by = ['age', 'diabetes', 'organ', 'location']

    # some default parameter values
    external_params = {  # run configuration
                       'start_time': 1600.,
                       'end_time': 2035.,
                       'time_step': 1.,
                       'start_population': 9000,
                       # base model definition:
                       'contact_rate': 30.,
                       'rr_transmission_recovered': 0.6,
                       'rr_transmission_infected': 0.21,
                       'rr_transmission_ltbi_treated': 0.21,
                       'latency_adjustment': 2.,  # used to modify progression rates during calibration
                       'self_recovery_rate': 0.231,  # this is for smear-positive TB
                       'tb_mortality_rate': 0.389,  # this is for smear-positive TB
                       'prop_smearpos': .1,
                        'cdr_multiplier': 1.1,
                        # diagnostic sensitivity by organ status:
                        'diagnostic_sensitivity_smearpos': 1.,
                        'diagnostic_sensitivity_smearneg': .7,
                        'diagnostic_sensitivity_extrapul': .5,
                         # adjustments by location and diabetes
                       # 'rr_transmission_ebeye': 1.,  # reference: majuro
                       # 'rr_transmission_otherislands': 1.,  # reference: majuro
                       'rr_progression_has_diabetes': 3.11,  # reference: no_diabetes
                       # ACF for intervention groups
                       'acf_coverage': 0.,
                       'acf_sensitivity': .9,
                       'acf_majuro_switch': 0.,
                       'acf_ebeye_switch': 0.,
                       'acf_otherislands_switch': 0.,
                        # LTBI ACF for intervention groups
                       'acf_ltbi_coverage': 0.,
                       'acf_ltbi_sensitivity': .9,
                       'acf_ltbi_efficacy': .85, # higher than ipt_efficacy as higher completion rate
                       'acf_ltbi_majuro_switch': 0.,
                       'acf_ltbi_ebeye_switch': 0.,
                       'acf_ltbi_otherislands_switch': 0.,
                       }
    # update external_params with new parameter values found in update_params
    external_params.update(update_params)

    model_parameters = \
        {"contact_rate": external_params['contact_rate'],
         "contact_rate_recovered": external_params['contact_rate'] * external_params['rr_transmission_recovered'],
         "contact_rate_infected": external_params['contact_rate'] * external_params['rr_transmission_infected'],
         "contact_rate_ltbi_treated": external_params['contact_rate'] * external_params['rr_transmission_ltbi_treated'],
         "recovery": external_params['self_recovery_rate'],
         "infect_death": external_params['tb_mortality_rate'],
         "universal_death_rate": 1.0 / 70.0,
         "case_detection": 0.,
         "ipt_rate": 0.,
         "acf_rate": 0.,
         "acf_ltbi_rate": 0.,
         "crude_birth_rate": 35.0 / 1e3}

    input_database = Database(database_name=INPUT_DB_PATH)
    n_iter = int(round((external_params['end_time'] - external_params['start_time']) / external_params['time_step'])) + 1
    integration_times = numpy.linspace(external_params['start_time'], external_params['end_time'], 4*n_iter).tolist()

    model_parameters.update(change_parameter_unit(provide_aggregated_latency_parameters(), 365.251))

    # sequentially add groups of flows
    flows = add_standard_infection_flows([])
    flows = add_standard_latency_flows(flows)
    flows = add_standard_natural_history_flows(flows)

    # compartments
    compartments = ["susceptible", "early_latent", "late_latent", "infectious", "recovered", "ltbi_treated"]

    # derived output definition
    out_connections = {
        "incidence_early": {"origin": "early_latent", "to": "infectious"},
        "incidence_late": {"origin": "late_latent", "to": "infectious"}
    }

    all_stratifications = {'organ': ['smearpos', 'smearneg', 'extrapul'],
                           'age': ['0', '5', '15', '35', '50'],
                           'location': ['majuro', 'ebeye', 'otherislands'],
                           'diabetes': ['has_diabetes', 'no_diabetes']}

    #  create derived outputs for disaggregated incidence
    for stratification in stratify_by:
        for stratum in all_stratifications[stratification]:
            for stage in ["early", 'late']:
                out_connections["indidence_" + stage + "X" + stratification + "_" + stratum] =\
                    {"origin": stage + "_latent", "to": "infectious", "to_condition": stratification + "_" + stratum}

    # create personalised derived outputs for mortality and notifications
    def mortality_derived_output(model):
        total_deaths = 0.
        for comp_ind in model.infectious_indices['all_strains']:
            infectious_pop = model.compartment_values[comp_ind]
            flow_index = model.death_flows[model.death_flows.origin == model.compartment_names[comp_ind]].index
            param_name = model.death_flows.parameter[flow_index].to_string().split('    ')[1]
            mortality_rate = model.get_parameter_value(param_name, 2019.)
            total_deaths += infectious_pop * mortality_rate
        return total_deaths

    # define model     #replace_deaths  add_crude_birth_rate
    _tb_model = StratifiedModel(
        integration_times, compartments, {"infectious": 1e-3}, model_parameters, flows, birth_approach="add_crude_birth_rate",
        starting_population=external_params['start_population'],
        output_connections=out_connections)

    # add crude birth rate from un estimates
    _tb_model = get_birth_rate_functions(_tb_model, input_database, 'FSM')

    # add case detection process to basic model
    _tb_model.add_transition_flow(
        {"type": "standard_flows", "parameter": "case_detection", "origin": "infectious", "to": "recovered"})

    # add ltbi treated infection flow
    _tb_model.add_transition_flow(
        {"type": "infection_frequency", "parameter": "contact_rate_ltbi_treated", "origin": "ltbi_treated",
         "to": "early_latent"})

    # add ACF flow
    _tb_model.add_transition_flow(
        {"type": "standard_flows", "parameter": "acf_rate", "origin": "infectious", "to": "recovered"})

    # add LTBI ACF flows
    _tb_model.add_transition_flow(
        {"type": "standard_flows", "parameter": "acf_ltbi_rate", "origin": "early_latent", "to": "ltbi_treated"})

    _tb_model.add_transition_flow(
        {"type": "standard_flows", "parameter": "acf_ltbi_rate", "origin": "late_latent", "to": "ltbi_treated"})

    # load time-variant case detection rate
    cdr_scaleup_overall = build_rmi_timevariant_cdr(external_params['cdr_multiplier'])

    # targeted TB prevalence proportions by organ
    prop_smearpos = .1
    prop_smearneg = .6
    prop_extrapul = .3

    # disease duration by organ
    overall_duration = prop_smearpos * 1.6 + 5.3 * (1 - prop_smearpos)
    disease_duration = {'smearpos': 1.6, 'smearneg': 5.3, 'extrapul': 5.3, 'overall': overall_duration}

    # work out the CDR for smear-positive TB
    def cdr_smearpos(time):
        return (cdr_scaleup_overall(time) /
                (prop_smearpos + prop_smearneg * external_params['diagnostic_sensitivity_smearneg'] +
                 prop_extrapul * external_params['diagnostic_sensitivity_extrapul']))

    def cdr_smearneg(time):
        return cdr_smearpos(time) * external_params['diagnostic_sensitivity_smearneg']

    def cdr_extrapul(time):
        return cdr_smearpos(time) * external_params['diagnostic_sensitivity_extrapul']

    cdr_by_organ = {'smearpos': cdr_smearpos, 'smearneg': cdr_smearneg, 'extrapul': cdr_extrapul,
                    'overall': cdr_scaleup_overall}
    detect_rate_by_organ = {}
    for organ in ['smearpos', 'smearneg', 'extrapul', 'overall']:
        prop_to_rate = convert_competing_proportion_to_rate(1.0 / disease_duration[organ])
        detect_rate_by_organ[organ] = return_function_of_function(cdr_by_organ[organ], prop_to_rate)

    # load time-variant treatment success rate
    rmi_tsr = build_rmi_timevariant_tsr()

    # create a treatment success rate function adjusted for treatment support intervention
    tsr_function = lambda t: rmi_tsr(t)

    # tb control recovery rate (detection and treatment) function set for overall if not organ-specific, smearpos otherwise
    if 'organ' not in stratify_by:
        tb_control_recovery_rate = lambda t: tsr_function(t) * detect_rate_by_organ['overall'](t)
    else:
        tb_control_recovery_rate = lambda t: tsr_function(t) * detect_rate_by_organ['smearpos'](t)

    # set acf screening rate using proportion of population reached and duration of intervention
    acf_screening_rate = -numpy.log(1 - .90)/.5

    acf_rate_over_time = progressive_step_function_maker(2018.2, 2018.7, acf_screening_rate, scaling_time_fraction=.3)

    # initialise acf_rate function
    acf_rate_function = lambda t: (acf_rate_over_time(t)) * external_params['acf_sensitivity'] * (rmi_tsr(t))

    acf_ltbi_rate_function = lambda t: (acf_rate_over_time(t)) * external_params['acf_ltbi_sensitivity'] * external_params['acf_ltbi_efficacy']

    # assign newly created functions to model parameters
    if len(stratify_by) == 0:
        _tb_model.time_variants["case_detection"] = tb_control_recovery_rate
        _tb_model.time_variants["acf_rate"] = acf_rate_function
        _tb_model.time_variants["acf_ltbi_rate"] = acf_ltbi_rate_function

    else:
        _tb_model.adaptation_functions["case_detection"] = tb_control_recovery_rate
        _tb_model.parameters["case_detection"] = "case_detection"

        _tb_model.adaptation_functions["acf_rate"] = acf_rate_function
        _tb_model.parameters["acf_rate"] = "acf_rate"

        _tb_model.adaptation_functions["acf_ltbi_rate"] = acf_ltbi_rate_function
        _tb_model.parameters["acf_ltbi_rate"] = "acf_ltbi_rate"

    if "age" in stratify_by:
        age_breakpoints = [0, 5, 15, 35, 50]
        age_infectiousness = get_parameter_dict_from_function(logistic_scaling_function(10.0), age_breakpoints)
        age_params = get_adapted_age_parameters(age_breakpoints)
        age_params.update(split_age_parameter(age_breakpoints, "contact_rate"))

        # adjustment of latency parameters
        for param in ['early_progression', 'late_progression']:
            for age_break in age_breakpoints:
                age_params[param][str(age_break) + 'W'] *= external_params['latency_adjustment']

        pop_morts = get_pop_mortality_functions(input_database, age_breakpoints, country_iso_code='FSM')
        age_params["universal_death_rate"] = {}
        for age_break in age_breakpoints:
            _tb_model.time_variants["universal_death_rateXage_" + str(age_break)] = pop_morts[age_break]
            _tb_model.parameters["universal_death_rateXage_" + str(age_break)] = "universal_death_rateXage_" + str(age_break)

            age_params["universal_death_rate"][str(age_break) + 'W'] = "universal_death_rateXage_" + str(age_break)
        _tb_model.parameters["universal_death_rateX"] = 0.

        # add BCG effect without stratification assuming constant 100% coverage
        bcg_wane = create_sloping_step_function(15.0, 0.3, 30.0, 1.0)
        age_bcg_efficacy_dict = get_parameter_dict_from_function(lambda value: bcg_wane(value), age_breakpoints)
        age_params.update({'contact_rate': age_bcg_efficacy_dict})

        _tb_model.stratify("age", deepcopy(age_breakpoints), [], {}, adjustment_requests=age_params,
                           infectiousness_adjustments=age_infectiousness, verbose=False)

    if 'organ' in stratify_by:
        props_smear = {"smearpos": external_params['prop_smearpos'],
                       "smearneg": 1. - (external_params['prop_smearpos'] + .3),
                       "extrapul": .3}
        mortality_adjustments = {"smearpos": 1., "smearneg": .064, "extrapul": .064}
        recovery_adjustments = {"smearpos": 1., "smearneg": .56, "extrapul": .56}
        diagnostic_sensitivity = {}
        for stratum in ["smearpos", "smearneg", "extrapul"]:
            diagnostic_sensitivity[stratum] = external_params["diagnostic_sensitivity_" + stratum]
        _tb_model.stratify("organ", ["smearpos", "smearneg", "extrapul"], ["infectious"],
                           infectiousness_adjustments={"smearpos": 1., "smearneg": .25, "extrapul": 0.},
                           verbose=False, requested_proportions=props_smear,
                           adjustment_requests={'recovery': recovery_adjustments,
                                                'infect_death': mortality_adjustments,
                                                'case_detection': diagnostic_sensitivity
                                                },
                           entry_proportions=props_smear)

    if 'diabetes' in stratify_by:
        props_diabetes = {'has_diabetes': 0.3, 'no_diabetes': 0.7}
        progression_adjustments = {"has_diabetes": 3.18, "no_diabetes": 1.}

        _tb_model.stratify("diabetes", ["has_diabetes", "no_diabetes"], [],
                           verbose=False, requested_proportions=props_diabetes,
                           adjustment_requests={'late_progressionXage_15': progression_adjustments,
                                                'late_progressionXage_35': progression_adjustments,
                                                'late_progressionXage_50': progression_adjustments,},
                           entry_proportions=props_diabetes)


    _tb_model.transition_flows.to_csv("transitions_age_dm_organ.csv")

    if "location" in stratify_by:
        props_location = {'majuro': .523, 'ebeye': .2, 'otherislands': .277}


        # dummy matrix for mixing by location
        location_mixing = numpy.array([.9, .05, .05,
                                       .05, .9, .05,
                                       .05, .05, .9]).reshape((3, 3))
        location_mixing *= 3.  # adjusted such that heterogeneous mixing yields similar overall burden as homogeneous

        location_adjustments = {}
        location_adjustments['acf_rate'] = {}
        for stratum in ['majuro', 'ebeye', 'otherislands']:
            location_adjustments['acf_rate'][stratum] = external_params['acf_' + stratum + '_switch']

        location_adjustments['acf_ltbi_rate'] = {}
        for stratum in ['majuro', 'ebeye', 'otherislands']:
            location_adjustments['acf_ltbi_rate'][stratum] = external_params['acf_ltbi_' + stratum + '_switch']

        _tb_model.stratify("location", ['majuro', 'ebeye', 'otherislands'], [],
                           requested_proportions=props_location, verbose=False, entry_proportions=props_location,
                           adjustment_requests=location_adjustments,
                           mixing_matrix=location_mixing
                           )

    _tb_model.transition_flows.to_csv("transitions_all.csv")
    _tb_model.death_flows.to_csv("deaths.csv")
    create_flowchart(_tb_model, strata=0, name="rmi_flow_diagram_0")
    create_flowchart(_tb_model, strata=1, name="rmi_flow_diagram_1")
    create_flowchart(_tb_model, strata=2, name="rmi_flow_diagram_2")
    create_flowchart(_tb_model, strata=3, name="rmi_flow_diagram_3")
    # create_flowchart(_tb_model, strata=2, name="rmi_flow_diagram_2")

    return _tb_model


if __name__ == "__main__":

    load_model = False

    scenario_params = {
        1: {'acf_majuro_switch': 1.,
                       'acf_ebeye_switch': 1.,
                       'acf_otherislands_switch': 0.,
                       'acf_ltbi_majuro_switch': 1.,
                       'acf_ltbi_ebeye_switch': 0.,
                       'acf_ltbi_otherislands_switch': 0.}

        }
    scenario_list = [0]
    scenario_list.extend(list(scenario_params.keys()))

    if load_model:
        load_mcmc = True

        if load_mcmc:
            models = load_calibration_from_db('outputs_11_27_2019_14_07_54.db')
            scenario_list = range(len(models))
        else:
            models = []
            scenarios_to_load = scenario_list
            for sc in scenarios_to_load:
                print("Loading model for scenario " + str(sc))
                model_dict = load_model_scenario(str(sc), database_name='outputs_11_27_2019_13_12_43.db')
                models.append(DummyModel(model_dict))
    else:
        t0 = time()
        models = run_multi_scenario(scenario_params, 2000., build_rmi_model)
        # automatically add combined incidence output
        for model in models:
            outputs_df = pd.DataFrame(model.outputs, columns=model.compartment_names)
            derived_outputs_df = pd.DataFrame(model.derived_outputs, columns=model.derived_outputs.keys())
            updated_derived_outputs = add_combined_incidence(derived_outputs_df, outputs_df)
            updated_derived_outputs = updated_derived_outputs.to_dict('list')
            model.derived_outputs = updated_derived_outputs
        store_run_models(models, scenarios=scenario_list, database_name=OUTPUT_DB_PATH)
        delta = time() - t0
        print("Running time: " + str(round(delta, 1)) + " seconds")

    req_outputs = ['prevXinfectiousXamong',
                   'prevXlatentXamong'
                   # 'prevXinfectiousXorgan_smearposXamongXinfectious',
                   # 'prevXinfectiousXorgan_smearnegXamongXinfectious',
                   # 'prevXinfectiousXorgan_extrapulXamongXinfectious',
                   # 'prevXlatentXamong',
                   # 'prevXinfectiousXamongXage_15Xage_60',
                   # 'prevXinfectiousXamongXage_15Xage_60Xhousing_ger',
                   # 'prevXinfectiousXamongXage_15Xage_60Xhousing_non-ger',
                   # 'prevXinfectiousXamongXage_15Xage_60Xlocation_rural',
                   # 'prevXinfectiousXamongXage_15Xage_60Xlocation_province',
                   # 'prevXinfectiousXamongXage_15Xage_60Xlocation_urban',
                   # 'prevXinfectiousXamongXhousing_gerXlocation_urban',
                   # 'prevXlatentXamongXhousing_gerXlocation_urban',
                   #
                   # 'prevXinfectiousXstrain_mdrXamong'
                 ]

    multipliers = {
        'prevXinfectiousXstrain_mdrXamongXinfectious': 100.,
        'prevXinfectiousXstrain_mdrXamong': 1.e5
    }

    targets_to_plot = {
                       }

    translations = {'prevXinfectiousXamong': 'TB prevalence (/100,000)',
                    'prevXinfectiousXamongXage_0': 'TB prevalence among 0-4 y.o. (/100,000)',
                    'prevXinfectiousXamongXage_5': 'TB prevalence among 5-14 y.o. (/100,000)',
                    'prevXinfectiousXamongXage_15': 'TB prevalence among 15-34 y.o. (/100,000)',
                    'prevXinfectiousXamongXage_35': 'TB prevalence among 35-49 y.o. (/100,000)',
                    'prevXinfectiousXamongXage_50': 'TB prevalence among 50+ y.o. (/100,000)',
                    'prevXinfectiousXamongXlocation_majuro': 'TB prevalence in Majuro (/100,000)',
                    'prevXinfectiousXamongXlocation_ebeye': 'TB prevalence in Ebeye (/100,000)',
                    'prevXinfectiousXamongXlocation_rural': 'TB prevalence in other areas (/100,000)',
                    'prevXinfectiousXamongXdiabetes_has_diabetes': 'TB prevalence in diabetics (/100,000)',
                    'prevXinfectiousXamongXdiabetes_no_diabetes': 'TB prevalence in non-diabetics (/100,000)',
                    'prevXlatentXamong': 'Latent TB infection prevalence (%)',
                    'prevXlatentXamongXage_0': 'Latent TB infection prevalence among 0-4 y.o. (%)',
                    'prevXlatentXamongXage_5': 'Latent TB infection prevalence among 5-14 y.o. (%)',
                    'prevXlatentXamongXage_15': 'Latent TB infection prevalence among 15-34 y.o. (%)',
                    'prevXlatentXamongXage_35': 'Latent TB infection prevalence among 35-49 y.o. (%)',
                    'prevXlatentXamongXage_50': 'Latent TB infection prevalence among 50+ y.o. (%)',
                    'prevXlatentXamongXlocation_majuro': 'Latent TB infection prevalence in Majuro (%)',
                    'prevXlatentXamongXlocation_ebeye': 'Latent TB infection prevalence in Ebeye (%)',
                    'prevXlatentXamongXlocation_otherislands': 'Latent TB infection prevalence in other areas (%)',
                    'prevXlatentXamongXdiabetes_has_diabetes': 'Latent TB infection prevalence in diabetics (%)',
                    'prevXlatentXamongXdiabetes_no_diabetes': 'Latent TB infection prevalence in non-diabetics (%)',
                    'age_0': 'Age 0-4',
                    'age_5': 'Age 5-14',
                    'age_15': 'Age 15-34',
                    'age_35': 'Age 35-49',
                    'age_50': 'Age 50+',
                    'location_majuro': 'Majuro',
                    'location_ebeye': 'Ebeye',
                    'location_otherislands': 'Other locations',
                    'diabetes_has_diabetes': 'Diabetes',
                    'diabetes_no_diabetes': 'No Diabetes',
                    }

    create_multi_scenario_outputs(models, req_outputs=req_outputs, out_dir='test_12_26_2', targets_to_plot=targets_to_plot,
                                  req_multipliers=multipliers, translation_dictionary=translations,
                                  scenario_list=scenario_list)
