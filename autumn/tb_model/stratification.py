from copy import deepcopy
from numpy import array

from summer.model.utils.parameter_processing import (
    create_step_function_from_dict,
    get_parameter_dict_from_function,
    logistic_scaling_function,
)
from summer.model import (
    split_age_parameter,
    create_sloping_step_function,
)

from autumn.db import get_pop_mortality_functions
from autumn.tool_kit import (
    change_parameter_unit,
    add_w_to_param_names,
)
from autumn.constants import Compartment
from autumn.tb_model import scale_relative_risks_for_equivalence


def get_adapted_age_parameters(age_breakpoints, AGE_SPECIFIC_LATENCY_PARAMETERS):
    """
    Get age-specific latency parameters adapted to any specification of age breakpoints
    """
    adapted_parameter_dict = {}
    for parameter in ("early_progression", "stabilisation", "late_progression"):
        adapted_parameter_dict[parameter] = add_w_to_param_names(
            change_parameter_unit(
                get_parameter_dict_from_function(
                    create_step_function_from_dict(AGE_SPECIFIC_LATENCY_PARAMETERS[parameter]),
                    age_breakpoints,
                ),
                365.251,
            )
        )
    return adapted_parameter_dict


def stratify_by_age(model_to_stratify, age_specific_latency_parameters, input_database, age_strata):
    # FIXME: This is Marshall Islands specifc - DO NOT USE outside of Marshall Islands
    age_breakpoints = [int(i_break) for i_break in age_strata]
    age_infectiousness = get_parameter_dict_from_function(
        logistic_scaling_function(10.0), age_breakpoints
    )
    age_params = get_adapted_age_parameters(age_breakpoints, age_specific_latency_parameters)
    age_params.update(split_age_parameter(age_breakpoints, "contact_rate"))
    pop_morts = get_pop_mortality_functions(
        input_database,
        age_breakpoints,
        country_iso_code="FSM",
        emigration_value=0.0075,
        emigration_start_time=1990.0,
    )
    age_params["universal_death_rate"] = {}
    for age_break in age_breakpoints:
        model_to_stratify.time_variants["universal_death_rateXage_" + str(age_break)] = pop_morts[
            age_break
        ]
        model_to_stratify.parameters[
            "universal_death_rateXage_" + str(age_break)
        ] = "universal_death_rateXage_" + str(age_break)
        age_params["universal_death_rate"][
            str(age_break) + "W"
        ] = "universal_death_rateXage_" + str(age_break)
    model_to_stratify.parameters["universal_death_rateX"] = 0.0

    # Add BCG effect without stratification assuming constant 100% coverage
    bcg_wane = create_sloping_step_function(15.0, 0.3, 30.0, 1.0)
    age_bcg_efficacy_dict = get_parameter_dict_from_function(
        lambda value: bcg_wane(value), age_breakpoints
    )
    age_params.update({"contact_rate": age_bcg_efficacy_dict})
    model_to_stratify.stratify(
        "age",
        deepcopy(age_breakpoints),
        [],
        {},
        adjustment_requests=age_params,
        infectiousness_adjustments=age_infectiousness,
        verbose=False,
    )
    return model_to_stratify


def stratify_by_diabetes(
    model_to_stratify,
    model_parameters,
    diabetes_strata,
    requested_diabetes_proportions,
    age_specific_prevalence=True,
):
    # FIXME: This is Marshall Islands specifc - DO NOT USE outside of Marshall Islands
    progression_adjustments = {
        "diabetic": model_parameters["rr_progression_diabetic"],
        "nodiabetes": 1.0,
    }
    adjustment_dict = {
        "early_progression": progression_adjustments,
        "late_progression": progression_adjustments,
    }
    if age_specific_prevalence:
        diabetes_target_props = {}
        for age_group in model_parameters["all_stratifications"]["age"]:
            diabetes_target_props.update(
                {"age_" + age_group: {"diabetic": requested_diabetes_proportions[int(age_group)]}}
            )
        diabetes_starting_and_entry_props = {"diabetic": 0.01, "nodiabetes": 0.99}

        model_to_stratify.stratify(
            "diabetes",
            diabetes_strata,
            [],
            verbose=False,
            split_proportions=diabetes_starting_and_entry_props,
            adjustment_requests=adjustment_dict,
            entry_proportions=diabetes_starting_and_entry_props,
            target_props=diabetes_target_props,
        )
    else:
        diabetes_starting_and_entry_props = {
            "diabetic": model_parameters["diabetes_prevalence_adults"],
            "nodiabetes": 1.0 - model_parameters["diabetes_prevalence_adults"],
        }

        # define age-specific adjustment requests for progression if age-stratified model
        if "age" in model_parameters["stratify_by"]:
            adjustment_dict = {}
            for age_break in model_parameters["all_stratifications"]["age"]:
                if int(age_break) >= 15:
                    adjustment_dict["early_progressionXage_" + age_break] = progression_adjustments
                    adjustment_dict["late_progressionXage_" + age_break] = progression_adjustments

        model_to_stratify.stratify(
            "diabetes",
            diabetes_strata,
            [],
            verbose=False,
            split_proportions=diabetes_starting_and_entry_props,
            adjustment_requests=adjustment_dict,
            entry_proportions=diabetes_starting_and_entry_props,
        )

    return model_to_stratify


def stratify_by_organ(model_to_stratify, model_parameters, detect_rate_by_organ, organ_strata):
    # FIXME: This is Marshall Islands specifc - DO NOT USE outside of Marshall Islands
    props_smear = {
        "smearpos": model_parameters["prop_smearpos"],
        "smearneg": 1.0 - (model_parameters["prop_smearpos"] + 0.2),
        "extrapul": 0.2,
    }
    mortality_adjustments = {"smearpos": 1.0, "smearneg": 0.064, "extrapul": 0.064}
    recovery_adjustments = {"smearpos": 1.0, "smearneg": 0.56, "extrapul": 0.56}

    # workout the detection rate adjustment by organ status
    adjustment_smearneg = (
        detect_rate_by_organ["smearneg"](2015.0) / detect_rate_by_organ["smearpos"](2015.0)
        if detect_rate_by_organ["smearpos"](2015.0) > 0.0
        else 1.0
    )
    adjustment_extrapul = (
        detect_rate_by_organ["extrapul"](2015.0) / detect_rate_by_organ["smearpos"](2015.0)
        if detect_rate_by_organ["smearpos"](2015.0) > 0.0
        else 1.0
    )
    model_to_stratify.stratify(
        "organ",
        organ_strata,
        [Compartment.EARLY_INFECTIOUS],
        infectiousness_adjustments={"smearpos": 1.0, "smearneg": 0.25, "extrapul": 0.0},
        verbose=False,
        split_proportions=props_smear,
        adjustment_requests={
            "recovery": recovery_adjustments,
            "infect_death": mortality_adjustments,
            "case_detection": {
                "smearpos": 1.0,
                "smearneg": adjustment_smearneg,
                "extrapul": adjustment_extrapul,
            },
            "early_progression": props_smear,
            "late_progression": props_smear,
        },
    )
    return model_to_stratify


def stratify_by_location(tb_model, model_parameters, location_strata):
    # FIXME: This is Marshall Islands specifc - DO NOT USE outside of Marshall Islands
    props_location = {"majuro": 0.523, "ebeye": 0.2, "otherislands": 0.277}

    raw_relative_risks_loc = {"majuro": 1.0}
    for stratum in ["ebeye", "otherislands"]:
        raw_relative_risks_loc[stratum] = model_parameters["rr_transmission_" + stratum]
    scaled_relative_risks_loc = scale_relative_risks_for_equivalence(
        props_location, raw_relative_risks_loc
    )

    # dummy matrix for mixing by location
    location_mixing = array([0.9, 0.05, 0.05, 0.05, 0.9, 0.05, 0.05, 0.05, 0.9]).reshape((3, 3))
    location_mixing *= (
        3.0  # adjusted such that heterogeneous mixing yields similar overall burden as homogeneous
    )

    location_adjustments = {}
    for beta_type in ["", "_late_latent", "_recovered"]:
        location_adjustments["contact_rate" + beta_type] = scaled_relative_risks_loc

    location_adjustments["case_detection"] = {}
    for stratum in location_strata:
        location_adjustments["case_detection"][stratum] = model_parameters[
            "case_detection_" + stratum + "_multiplier"
        ]

    # location_adjustments["acf_coverage"] = {}
    # for stratum in location_strata:
    #     location_adjustments["acf_coverage"][stratum] = model_parameters[
    #         "acf_" + stratum + "_coverage"
    #         ]
    #
    # location_adjustments["acf_ltbi_coverage"] = {}
    # for stratum in location_strata:
    #     location_adjustments["acf_ltbi_coverage"][stratum] = model_parameters[
    #         "acf_ltbi_" + stratum + "_coverage"
    #         ]

    tb_model.stratify(
        "location",
        location_strata,
        [],
        split_proportions=props_location,
        verbose=False,
        entry_proportions=props_location,
        adjustment_requests=location_adjustments,
        mixing_matrix=location_mixing,
    )
    return tb_model
