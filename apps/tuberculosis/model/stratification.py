from apps.tuberculosis.constants import Compartment, OrganStratum
from apps.tuberculosis.model.preprocess.latency import get_adapted_age_parameters
from autumn.inputs import get_death_rates_by_agegroup
from autumn.curve import scale_up_function

from math import log, exp

import itertools


def stratify_by_age(model, params, compartments):

    flow_adjustments = {}

    # age-specific all-causes mortality rate
    death_rates_by_age, death_rate_years = get_death_rates_by_agegroup(params['age_breakpoints'], params['iso3'])
    flow_adjustments['universal_death_rate'] = {}
    for age_group in params['age_breakpoints']:
        name = 'universal_death_rate_' + str(age_group)
        flow_adjustments['universal_death_rate'][str(age_group)] = name
        model.time_variants[name] = scale_up_function(
            death_rate_years, death_rates_by_age[age_group], smoothness=0.2, method=5
        )
        model.parameters[name] = name
    # age-specific latency progresison rates
    if params["override_latency_rates"]:
        flow_adjustments.update(get_adapted_age_parameters(params['age_breakpoints']))

    # age-specific infectiousness
    strata_infectiousness = calculate_age_specific_infectiousness(params['age_breakpoints'],
                                                                  params['age_infectiousness_switch'])

    # age-specific relapse and treatment death
    input_time_variant_tsr = scale_up_function(
        list(params['time_variant_tsr'].keys()),
        list(params['time_variant_tsr'].values()),
        method=4,
    )
    factory_functions = {
        'treatment_death_rate': make_treatment_death_func,
        'relapse_rate': make_relapse_rate_func,
    }
    for param_stem in factory_functions:
        flow_adjustments[param_stem] = {}
        for age_group in params['age_breakpoints']:
            flow_adjustments[param_stem][str(age_group)] = param_stem + '_' + str(age_group)

            model.time_variants[param_stem + '_' + str(age_group)] = factory_functions[param_stem](
                age_group, model, params, input_time_variant_tsr
            )
            model.parameters[param_stem + '_' + str(age_group)] = param_stem + '_' + str(age_group)

    # trigger model stratification
    model.stratify(
        "age",
        params['age_breakpoints'],
        compartments,
        infectiousness_adjustments=strata_infectiousness,
        flow_adjustments=flow_adjustments,
    )


def apply_universal_stratification(model, compartments, stratification_name, stratification_details):
    """
    Stratify all model compartments based on a user-defined stratification request. This stratification can only adjust
    the parameters that are directly implemented in the model. That is, adjustment requests to parameters that are used
    for pre-processing but not directly linked to a flow will have no effect.
    """
    # prepare parameter adjustments
    flow_adjustments = {}
    for param_name, adjustment in stratification_details['adjustments'].items():
        stratified_param_names = get_stratified_param_names(param_name, model.stratifications)
        for stratified_param_name in stratified_param_names:
            flow_adjustments[stratified_param_name] = {}
            for stratum in stratification_details['strata']:
                flow_adjustments[stratified_param_name][stratum] = adjustment

    # apply stratification
    model.stratify(
        stratification_name,
        stratification_details['strata'],
        compartments,
        comp_split_props=stratification_details['proportions'],
        flow_adjustments=flow_adjustments,
    )


def stratify_by_organ(model, params):

    compartments_to_stratify = [
        Compartment.INFECTIOUS,
    ]
    organ_strata = [
        OrganStratum.SMEAR_POSITIVE,
        OrganStratum.SMEAR_NEGATIVE,
        OrganStratum.EXTRAPULMONARY,
    ]

    # Define infectiousness adjustment by organ status
    strata_infectiousness = {}
    for stratum in organ_strata:
        if stratum + "_infect_multiplier" in params:
            strata_infectiousness[stratum] = params[stratum + "_infect_multiplier"]

    # define differential natural history by organ status
    flow_adjustments = {}
    for param_name in ["infect_death_rate", "self_recovery_rate"]:
        stratified_param_names = get_stratified_param_names(param_name, model.stratifications)
        for stratified_param_name in stratified_param_names:
            flow_adjustments[stratified_param_name] = {}
            for organ_stratum in organ_strata:
                flow_adjustments[stratified_param_name][organ_stratum + "W"] = params[param_name + "_dict"][
                    organ_stratum
                ]

    # define differential detection rates by organ status
    stratified_param_names = get_stratified_param_names('detection_rate', model.stratifications)
    for stratified_param_name in stratified_param_names:
        flow_adjustments[stratified_param_name] = {}
        for organ_stratum in organ_strata:
            flow_adjustments[stratified_param_name][organ_stratum + "W"] =\
                params['passive_screening_rate'] * params['passive_screening_sensitivity'][organ_stratum]

    # Adjust the progression rates by organ using the requested incidence proportions
    splitting_proportions = {
        "smear_positive": params["incidence_props_pulmonary"]
        * params["incidence_props_smear_positive_among_pulmonary"],
        "smear_negative": params["incidence_props_pulmonary"]
        * (1.0 - params["incidence_props_smear_positive_among_pulmonary"]),
        "extrapulmonary": 1.0 - params["incidence_props_pulmonary"],
    }
    for stage in ["early", "late"]:
        flow_adjustments[stage + "_activation_rate"] = splitting_proportions

    # trigger model stratification
    model.stratify(
        "organ",
        organ_strata,
        compartments_to_stratify,
        infectiousness_adjustments=strata_infectiousness,
        flow_adjustments=flow_adjustments,
    )


def calculate_age_specific_infectiousness(age_breakpoints, age_infectiousness_switch):
    """
    We assume that infectiousness increases with age
    A sigmoidal function (x -> 1 / (1 + exp(-(x-alpha)))) is used to model a progressive increase  with  age.
    This is the approach used in Ragonnet et al. (BMC Medicine, 2019)
    :param age_breakpoints: model age brealpoints
    :param age_infectiousness_switch: parameter alpha
    :return:
    """
    infectiousness_by_agegroup = {}
    for i, age_low in enumerate(age_breakpoints):
        if i < len(age_breakpoints) - 1:
            age_up = age_breakpoints[i + 1]
            # Calculate the average of the sigmoidal function(x -> 1 / (1 + exp(-(x-alpha)))) between the age bounds
            average_infectiousness = \
                (log(1 + exp(age_up - age_infectiousness_switch)) -
                 log(1 + exp(age_low - age_infectiousness_switch))
                 ) / (age_up - age_low)
        else:  # set infectiousness to 1. for the oldest age group
            average_infectiousness = 1.

        infectiousness_by_agegroup[str(age_low)] = average_infectiousness

    return infectiousness_by_agegroup


def make_treatment_death_func(age_group, model, params, input_time_variant_tsr):
    def treatment_death_func(t):
        tsr_ceiling = params['prop_death_among_negative_tx_outcome'] * params['treatment_recovery_rate'] /\
                     (params['prop_death_among_negative_tx_outcome'] * params['treatment_recovery_rate'] +
                      model.time_variants['universal_death_rate_' + str(age_group)](t))
        tsr = input_time_variant_tsr(t)
        if tsr >= tsr_ceiling:
            return 0.
        else:
            return params['prop_death_among_negative_tx_outcome'] * params['treatment_recovery_rate'] *\
                (1. - tsr) / tsr - model.time_variants['universal_death_rate_' + str(age_group)](t)
    return treatment_death_func


def make_relapse_rate_func(age_group, model, params, input_time_variant_tsr):
    def relapse_rate_func(t):
        tsr_ceiling = params['prop_death_among_negative_tx_outcome'] * params['treatment_recovery_rate'] /\
                     (params['prop_death_among_negative_tx_outcome'] * params['treatment_recovery_rate'] +
                      model.time_variants['universal_death_rate_' + str(age_group)](t))
        tsr = input_time_variant_tsr(t)
        if tsr >= tsr_ceiling:
            mu_t = 0.  # treatment_death_rate
        else:
            mu_t = params['prop_death_among_negative_tx_outcome'] * params['treatment_recovery_rate'] *\
                (1. - tsr) / tsr - model.time_variants['universal_death_rate_' + str(age_group)](t)
        return (model.time_variants['universal_death_rate_' + str(age_group)](t) + mu_t) *\
            (1. / params['prop_death_among_negative_tx_outcome'] - 1.)
    return relapse_rate_func


def get_stratified_param_names(param_name, stratifications):
    stratified_param_names = [param_name]
    all_strata_names = []
    for stratification in stratifications:
        all_strata_names.append([stratification.name + "_" + s for s in stratification.strata])
    for strata_combo in itertools.product(*all_strata_names):
        strata_combo_string = ""
        for strata in strata_combo:
            strata_combo_string += "X" + strata
        stratified_param_names.append(param_name + strata_combo_string)
    return stratified_param_names
