from apps.tuberculosis.constants import Compartment, OrganStratum
from apps.tuberculosis.model.preprocess.latency import (
    get_adapted_age_parameters,
    edit_adjustments_for_diabetes,
)
from autumn.inputs import get_death_rates_by_agegroup
from autumn.inputs.social_mixing.queries import get_mixing_matrix_specific_agegroups
from autumn.curve import scale_up_function, tanh_based_scaleup, make_linear_curve
from summer.model.utils.parameter_processing import get_parameter_dict_from_function
from summer.model import create_sloping_step_function

from math import log, exp
import numpy as np
import itertools


def stratify_by_age(model, params, compartments):

    flow_adjustments = {}

    # age-specific all-causes mortality rate
    death_rates_by_age, death_rate_years = get_death_rates_by_agegroup(
        params["age_breakpoints"], params["iso3"]
    )
    flow_adjustments["universal_death_rate"] = {}
    for age_group in params["age_breakpoints"]:
        name = "universal_death_rate_" + str(age_group)
        flow_adjustments["universal_death_rate"][str(age_group)] = name
        model.time_variants[name] = scale_up_function(
            death_rate_years, death_rates_by_age[age_group], smoothness=0.2, method=5
        )
        model.parameters[name] = name

    # age-specific latency progression rates
    flow_adjustments.update(
        get_adapted_age_parameters(params["age_breakpoints"], params["age_specific_latency"])
    )
    if params["inflate_reactivation_for_diabetes"]:
        flow_adjustments = edit_adjustments_for_diabetes(
            model,
            flow_adjustments,
            params["age_breakpoints"],
            params["extra_params"]["prop_diabetes"],
            params["extra_params"]["rr_progression_diabetes"],
            params["extra_params"]["future_diabetes_multiplier"],
        )

    # age-specific infectiousness
    strata_infectiousness = calculate_age_specific_infectiousness(
        params["age_breakpoints"], params["age_infectiousness_switch"]
    )

    # age-specific treatment recovery, relapse and treatment death rates
    time_variant_tsr = scale_up_function(
        list(params["time_variant_tsr"].keys()),
        list(params["time_variant_tsr"].values()),
        method=4,
    )
    factory_functions = {
        "treatment_recovery_rate": make_treatment_recovery_func,
        "treatment_death_rate": make_treatment_death_func,
        "relapse_rate": make_relapse_rate_func,
    }
    for param_stem in factory_functions:
        flow_adjustments[param_stem] = {}
        for age_group in params["age_breakpoints"]:
            flow_adjustments[param_stem][str(age_group)] = param_stem + "_" + str(age_group)

            model.time_variants[param_stem + "_" + str(age_group)] = factory_functions[param_stem](
                age_group, model, params, time_variant_tsr
            )
            model.parameters[param_stem + "_" + str(age_group)] = param_stem + "_" + str(age_group)

    # get mixing matrix
    mixing_matrix = get_mixing_matrix_specific_agegroups(params["iso3"], params["age_breakpoints"])

    # add BCG effect without stratifying for BCG
    bcg_wane = create_sloping_step_function(15.0, 0.3, 30.0, 1.0)
    bcg_susceptibility_multilier_dict = get_parameter_dict_from_function(
        lambda value: bcg_wane(value), params["age_breakpoints"]
    )
    bcg_coverage_func = scale_up_function(
        list(params["time_variant_bcg_perc"].keys()),
        list(params["time_variant_bcg_perc"].values()),
        method=5,
        bound_low=0,
        bound_up=100,
        smoothness=1.5
    )
    for agegroup, multiplier in bcg_susceptibility_multilier_dict.items():
        if multiplier <  1.0:
            average_age = get_average_age_for_bcg(agegroup, params['age_breakpoints'])
            name = "contact_rate_" + agegroup
            bcg_susceptibility_multilier_dict[agegroup] = name
            model.time_variants[name] = make_bcg_multiplier_func(
                bcg_coverage_func, multiplier, average_age
            )
            model.parameters[name] = name
    flow_adjustments.update({"contact_rate": bcg_susceptibility_multilier_dict})

    # trigger model stratification
    model.stratify(
        "age",
        params["age_breakpoints"],
        compartments,
        infectiousness_adjustments=strata_infectiousness,
        flow_adjustments=flow_adjustments,
        mixing_matrix=mixing_matrix,
    )


def apply_user_defined_stratification(
    model,
    compartments,
    stratification_name,
    stratification_details,
    implement_acf,
    implement_ltbi_screening,
):
    """
    Stratify all model compartments based on a user-defined stratification request. This stratification can only adjust
    the parameters that are directly implemented in the model. That is, adjustment requests to parameters that are used
    for pre-processing but not directly linked to a flow will have no effect.
    """
    # also adjust reinfection contact rates if the primary contact rate is adjusted
    if "contact_rate" in stratification_details["adjustments"]:
        for stage in ["latent", "recovered"]:
            param_name = "contact_rate_from_" + stage
            if param_name not in stratification_details["adjustments"]:
                stratification_details["adjustments"][param_name] = stratification_details[
                    "adjustments"
                ]["contact_rate"]

    # adjust crude birth rate according to the strata proportions
    stratification_details["adjustments"]["crude_birth_rate"] = stratification_details[
        "proportions"
    ]

    # prepare parameter adjustments
    flow_adjustments = {}
    for param_name, adjustment in stratification_details["adjustments"].items():
        stratified_param_names = get_stratified_param_names(param_name, model.stratifications)
        for stratified_param_name in stratified_param_names:
            flow_adjustments[stratified_param_name] = {}
            for stratum in adjustment:
                flow_adjustments[stratified_param_name][stratum] = adjustment[stratum]

    # format mixing matrix
    if "mixing_matrix" in stratification_details:
        mixing_matrix = np.array([row for row in stratification_details["mixing_matrix"]])
    else:
        mixing_matrix = None

    # ACF and preventive treatment interventions
    int_details = {
        "acf": {
            "implement_switch": implement_acf,
            "parameter_name": "acf_detection_rate",
            "sensitivity": model.parameters["acf_screening_sensitivity"],
            "prop_detected_effectively_moving": 1.0,
        },
        "ltbi_screening": {
            "implement_switch": implement_ltbi_screening,
            "parameter_name": "preventive_treatment_rate",
            "sensitivity": model.parameters["ltbi_screening_sensitivity"],
            "prop_detected_effectively_moving": model.parameters["pt_efficacy"],
        },
    }
    for int_type in ["acf", "ltbi_screening"]:
        if int_details[int_type]["implement_switch"]:
            int_adjustments = {}
            for intervention in model.parameters["time_variant_" + int_type]:
                if stratification_name in intervention["stratum_filter"]:
                    param_name = (
                        int_details[int_type]["parameter_name"]
                        + "X"
                        + stratification_name
                        + "_"
                        + stratum
                    )
                    stratum = intervention["stratum_filter"][stratification_name]
                    model.time_variants[param_name] = make_intervention_adjustment_func(
                        intervention["time_variant_screening_rate"],
                        int_details[int_type]["sensitivity"],
                        int_details[int_type]["prop_detected_effectively_moving"],
                    )
                    model.parameters[param_name] = param_name
                    int_adjustments[stratum] = param_name
            if int_adjustments:
                for stratum in stratification_details["strata"]:
                    if stratum not in int_adjustments:
                        int_adjustments[stratum] = 0.0
            stratified_param_names = get_stratified_param_names(
                int_details[int_type]["parameter_name"], model.stratifications
            )
            for stratified_param_name in stratified_param_names:
                flow_adjustments[stratified_param_name] = int_adjustments

    # apply stratification
    model.stratify(
        stratification_name,
        stratification_details["strata"],
        compartments,
        comp_split_props=stratification_details["proportions"],
        flow_adjustments=flow_adjustments,
        mixing_matrix=mixing_matrix,
    )


def stratify_by_organ(model, params):

    compartments_to_stratify = [
        Compartment.INFECTIOUS,
        Compartment.ON_TREATMENT,
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
                organ_stratum_ = (
                    organ_stratum
                    if organ_stratum != OrganStratum.EXTRAPULMONARY
                    else OrganStratum.SMEAR_NEGATIVE
                )
                flow_adjustments[stratified_param_name][organ_stratum + "W"] = params[
                    param_name + "_dict"
                ][organ_stratum_]

    # define differential detection rates by organ status
    screening_rate_func = tanh_based_scaleup(
        params["time_variant_tb_screening_rate"]["maximum_gradient"],
        params["time_variant_tb_screening_rate"]["max_change_time"],
        params["time_variant_tb_screening_rate"]["start_value"],
        params["time_variant_tb_screening_rate"]["end_value"],
    )
    if params["awareness_raising"]:
        awaireness_linear_scaleup = make_linear_curve(
            x_0=params["awareness_raising"]["scale_up_range"][0],
            x_1=params["awareness_raising"]["scale_up_range"][1],
            y_0=1,
            y_1=params["awareness_raising"]["relative_screening_rate"]
        )

        def awaireness_multiplier(t):
            if t <= params["awareness_raising"]["scale_up_range"][0]:
                return 1.
            elif t >= params["awareness_raising"]["scale_up_range"][1]:
                return params["awareness_raising"]["relative_screening_rate"]
            else:
                return awaireness_linear_scaleup(t)
    else:
        awaireness_multiplier = lambda t: 1.

    combined_screening_rate_func = lambda t: screening_rate_func(t) * awaireness_multiplier(t)

    stratified_param_names = get_stratified_param_names("detection_rate", model.stratifications)
    for stratified_param_name in stratified_param_names:
        flow_adjustments[stratified_param_name] = {}
        for organ_stratum in organ_strata:
            flow_adjustments[stratified_param_name][organ_stratum] = (
                stratified_param_name + "_" + organ_stratum
            )
            model.time_variants[stratified_param_name + "_" + organ_stratum] = make_detection_func(
                organ_stratum, params, combined_screening_rate_func
            )
            model.parameters[stratified_param_name + "_" + organ_stratum] = (
                stratified_param_name + "_" + organ_stratum
            )

    # Adjust the progression rates by organ using the requested incidence proportions
    splitting_proportions = {
        "smear_positive": params["incidence_props_pulmonary"]
        * params["incidence_props_smear_positive_among_pulmonary"],
        "smear_negative": params["incidence_props_pulmonary"]
        * (1.0 - params["incidence_props_smear_positive_among_pulmonary"]),
        "extrapulmonary": 1.0 - params["incidence_props_pulmonary"],
    }
    for stage in ["early", "late"]:
        param_stem = stage + "_activation_rate"
        stratified_param_names = get_stratified_param_names(param_stem, model.stratifications)
        for stratified_param_name in stratified_param_names:
            flow_adjustments[stratified_param_name] = splitting_proportions

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
            average_infectiousness = (
                log(1 + exp(age_up - age_infectiousness_switch))
                - log(1 + exp(age_low - age_infectiousness_switch))
            ) / (age_up - age_low)
        else:  # set infectiousness to 1. for the oldest age group
            average_infectiousness = 1.0

        infectiousness_by_agegroup[str(age_low)] = average_infectiousness

    return infectiousness_by_agegroup


def make_bcg_multiplier_func(bcg_coverage_func, multiplier, average_age):
    def bcg_multiplier_func(t):
        return 1.0 - bcg_coverage_func(t - average_age) / 100.0 * (1.0 - multiplier)

    return bcg_multiplier_func


def make_treatment_recovery_func(age_group, model, params, time_variant_tsr):
    def treatment_recovery_func(t):
        return max(
            1 / params["treatment_duration"],
            model.time_variants["universal_death_rate_" + str(age_group)](t)
            / params["prop_death_among_negative_tx_outcome"]
            * (1.0 / (1.0 - time_variant_tsr(t)) - 1.0),
        )

    return treatment_recovery_func


def make_treatment_death_func(age_group, model, params, time_variant_tsr):
    def treatment_death_func(t):
        recovery_rate = max(
            1 / params["treatment_duration"],
            model.time_variants["universal_death_rate_" + str(age_group)](t)
            / params["prop_death_among_negative_tx_outcome"]
            * (1.0 / (1.0 - time_variant_tsr(t)) - 1.0),
        )
        return params["prop_death_among_negative_tx_outcome"] * recovery_rate * (
            1.0 - time_variant_tsr(t)
        ) / time_variant_tsr(t) - model.time_variants["universal_death_rate_" + str(age_group)](t)

    return treatment_death_func


def make_relapse_rate_func(age_group, model, params, time_variant_tsr):
    def relapse_rate_func(t):
        recovery_rate = max(
            1 / params["treatment_duration"],
            model.time_variants["universal_death_rate_" + str(age_group)](t)
            / params["prop_death_among_negative_tx_outcome"]
            * (1.0 / (1.0 - time_variant_tsr(t)) - 1.0),
        )
        return (
            recovery_rate
            * (1.0 / time_variant_tsr(t) - 1.0)
            * (1.0 - params["prop_death_among_negative_tx_outcome"])
        )

    return relapse_rate_func


def make_detection_func(organ_stratum, params, screening_rate_func):
    def detection_func(t):
        return screening_rate_func(t) * params["passive_screening_sensitivity"][organ_stratum]

    return detection_func


def make_intervention_adjustment_func(
    time_variant_screening_rate, sensitivity, prop_detected_effectively_moving
):
    acf_detection_func = scale_up_function(
        list(time_variant_screening_rate.keys()),
        [
            v * sensitivity * prop_detected_effectively_moving
            for v in list(time_variant_screening_rate.values())
        ],
        method=4,
    )
    return acf_detection_func


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


def get_average_age_for_bcg(agegroup, age_breakpoints):
    agegroup_idx = age_breakpoints.index(int(agegroup))
    if agegroup_idx == len(age_breakpoints) - 1:
        # We should normally never be in this situation because the last agegroup is not affected by BCG anyway.
        print("Warning: the agegroup name is being used to represent the average age of the group")
        return float(agegroup)
    else:
        return 0.5 * (age_breakpoints[agegroup_idx] + age_breakpoints[agegroup_idx + 1])
