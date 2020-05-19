from autumn.tool_kit.utils import (
    find_rates_and_complements_from_ifr,
    repeat_list_elements,
    repeat_list_elements_average_last_two,
    element_wise_list_division,
)
from autumn.constants import Compartment
from autumn.summer_related.parameter_adjustments import (
    adjust_upstream_stratified_parameter,
    split_prop_into_two_subprops,
)
from autumn.curve import scale_up_function, tanh_based_scaleup


def get_raw_clinical_props(params):
    """
    Get the raw proportions of persons dying and progressing to symptomatic, hospital if symptomatic, and to ICU if
    hospitalised - adjusting to required number of age groups.
    """
    return {
        "adjusted_infection_fatality_props": repeat_list_elements_average_last_two(
            params["infection_fatality_props"]
        ),
        "raw_sympt": repeat_list_elements(2, params["symptomatic_props"]),
        "raw_hospital": repeat_list_elements_average_last_two(params["hospital_props"]),
    }


def find_abs_death_props(params, abs_props):
    """
    Calculate the absolute proportion of all patients who should eventually reach hospital death or ICU death.
    """

    # Find IFR that needs to be contributed by ICU and non-ICU hospital deaths\
    hospital_death, icu_death = [], []
    for age_idxgroup, agegroup in enumerate(params["all_stratifications"]["agegroup"]):

        # If IFR for age group is greater than absolute proportion hospitalised, increased hospitalised proportion
        if (
            params["adjusted_infection_fatality_props"][age_idxgroup]
            > abs_props["hospital"][age_idxgroup]
        ):
            abs_props["hospital"][age_idxgroup] = params["adjusted_infection_fatality_props"][
                age_idxgroup
            ]

        # Find the target absolute ICU mortality and the amount left over from IFRs to go to hospital, if any
        target_icu_abs_mort = abs_props["icu"][age_idxgroup] * params["icu_mortality_prop"]
        left_over_mort = (
            params["adjusted_infection_fatality_props"][age_idxgroup] - target_icu_abs_mort
        )

        # If some IFR will be left over for the hospitalised
        if left_over_mort > 0.0:
            hospital_death.append(left_over_mort)
            icu_death.append(target_icu_abs_mort)

        # Otherwise if all IFR taken up by ICU
        else:
            hospital_death.append(0.0)
            icu_death.append(params["adjusted_infection_fatality_props"][age_idxgroup])

    return {"hospital_death": hospital_death, "icu_death": icu_death}


def set_isolation_props(
    _covid_model,
    model_parameters,
    abs_props,
    stratification_adjustments,
    tv_prop_detect_among_sympt,
):
    """
    Set the absolute proportions of new cases isolated and not isolated, and indicate to the model where they should be
    found.
    """
    # need wrapper functions around all time-variant splitting functions to avoid using the final age_idx
    # for all age groups, which is what would happen if the t_v functions were defined within a loop.
    def abs_prop_sympt_non_hosp_wrapper(_age_idx):
        def abs_prop_sympt_non_hosp_func(t):
            return abs_props["sympt"][_age_idx] * (1.0 - tv_prop_detect_among_sympt(t))

        return abs_prop_sympt_non_hosp_func

    # we need to adjust the hospital_props to make sure it remains <= detected proportions
    def adjusted_prop_hospital_among_sympt_wrapper(_age_idx):
        def adjusted_prop_hospital_among_sympt_func(t):
            raw_h_prop = abs_props["sympt"][_age_idx] * model_parameters["raw_hospital"][_age_idx]
            adjusted_h_prop = (
                raw_h_prop
                if tv_prop_detect_among_sympt(t) >= raw_h_prop
                else tv_prop_detect_among_sympt(t)
            )
            return adjusted_h_prop

        return adjusted_prop_hospital_among_sympt_func

    def abs_prop_isolate_wrapper(_age_idx):
        def abs_prop_isolate_func(t):
            return (
                abs_props["sympt"][_age_idx]
                * tv_prop_detect_among_sympt(t)
                * (
                    1.0
                    - adjusted_prop_hospital_among_sympt_wrapper(_age_idx)(t)
                    / tv_prop_detect_among_sympt(t)
                )
            )

        return abs_prop_isolate_func

    def abs_prop_hosp_non_icu_wrapper(_age_idx):
        def abs_prop_hosp_non_icu_func(t):
            return (
                abs_props["sympt"][_age_idx]
                * adjusted_prop_hospital_among_sympt_wrapper(_age_idx)(t)
                * (1.0 - model_parameters["icu_prop"])
            )

        return abs_prop_hosp_non_icu_func

    def abs_prop_icu_wrapper(_age_idx):
        def abs_prop_icu_func(t):
            return (
                abs_props["sympt"][_age_idx]
                * adjusted_prop_hospital_among_sympt_wrapper(_age_idx)(t)
                * model_parameters["icu_prop"]
            )

        return abs_prop_icu_func

    for age_idx, agegroup in enumerate(model_parameters["all_stratifications"]["agegroup"]):
        # pass the functions to summer
        _covid_model.time_variants[
            "prop_sympt_non_hospital_" + agegroup
        ] = abs_prop_sympt_non_hosp_wrapper(age_idx)
        _covid_model.time_variants["prop_sympt_isolate_" + agegroup] = abs_prop_isolate_wrapper(
            age_idx
        )
        _covid_model.time_variants[
            "prop_hospital_non_icu_" + agegroup
        ] = abs_prop_hosp_non_icu_wrapper(age_idx)
        _covid_model.time_variants["prop_icu_" + agegroup] = abs_prop_icu_wrapper(age_idx)

        # define the stratification adjustments to be made
        for clinical_stratum in ["sympt_non_hospital", "sympt_isolate", "hospital_non_icu", "icu"]:
            stratification_adjustments["to_infectiousXagegroup_" + agegroup][clinical_stratum] = (
                "prop_" + clinical_stratum + "_" + agegroup
            )

    return _covid_model, stratification_adjustments


def adjust_infectiousness(model, params, strata, adjustments):
    """
    Sort out all infectiousness adjustments for all compartments of the model.
    """

    # Make adjustment for hospitalisation and ICU admission
    strata_infectiousness = {}
    for stratum in strata:
        if stratum + "_infect_multiplier" in params:
            strata_infectiousness[stratum] = params[stratum + "_infect_multiplier"]

    # Make adjustment for isolation/quarantine
    model.individual_infectiousness_adjustments = [
        [[Compartment.LATE_INFECTIOUS, "clinical_sympt_isolate"], 0.2]
    ]
    return model, adjustments, strata_infectiousness


def stratify_by_clinical(_covid_model, model_parameters, compartments):
    """
    Stratify the infectious compartments of the covid model (not including the pre-symptomatic compartments, which are
    actually infectious)
    """
    all_stratifications = model_parameters["all_stratifications"]
    clinical_strata = model_parameters["clinical_strata"]
    hospital_props = model_parameters["hospital_props"]
    icu_prop = model_parameters["icu_prop"]
    implement_importation = model_parameters["implement_importation"]
    imported_cases_explict = model_parameters["imported_cases_explict"]
    prop_detected_among_symptomatic = model_parameters["prop_detected_among_symptomatic"]
    traveller_quarantine = model_parameters["traveller_quarantine"]
    tv_detection_b = model_parameters["tv_detection_b"]
    tv_detection_c = model_parameters["tv_detection_c"]
    tv_detection_sigma = model_parameters["tv_detection_sigma"]
    within_hospital_early = model_parameters["within_hospital_early"]
    within_icu_early = model_parameters["within_icu_early"]

    for age_idx, h_prop in enumerate(hospital_props):
        # Check that hospital props aren't too big for some reason.
        if h_prop > prop_detected_among_symptomatic:
            # Reduce hospital props for some reason.
            print("Warning: Hospital proportions had to be reduced for age-group " + str(age_idx))
            hospital_props[age_idx] = prop_detected_among_symptomatic

    # Define stratification
    strata_to_implement = clinical_strata
    model_parameters["all_stratifications"]["clinical"] = strata_to_implement
    compartments_to_split = [
        comp
        for comp in compartments
        if comp.startswith(Compartment.EARLY_INFECTIOUS)
        or comp.startswith(Compartment.LATE_INFECTIOUS)
    ]

    # Find unadjusted parameters
    model_parameters.update(get_raw_clinical_props(model_parameters))

    # Find the absolute progression proportions from the requested splits
    abs_props = split_prop_into_two_subprops([1.0] * 16, "", model_parameters["raw_sympt"], "sympt")
    abs_props.update(
        split_prop_into_two_subprops(
            abs_props["sympt"], "sympt", model_parameters["raw_hospital"], "hospital"
        )
    )
    abs_props.update(
        split_prop_into_two_subprops(
            abs_props["hospital"], "hospital", [model_parameters["icu_prop"]] * 16, "icu"
        )
    )

    # Find the absolute proportion dying in hospital and in ICU
    abs_props.update(find_abs_death_props(model_parameters, abs_props))

    # CFR for non-ICU hospitalised patients
    rel_props = {
        "hospital_death": element_wise_list_division(
            abs_props["hospital_death"], abs_props["hospital_non_icu"]
        ),
        "icu_death": element_wise_list_division(abs_props["icu_death"], abs_props["icu"]),
    }

    # Progression rates into the infectious compartment(s)
    fixed_prop_strata = ["non_sympt"]
    stratification_adjustments = adjust_upstream_stratified_parameter(
        "to_infectious",
        fixed_prop_strata,
        "agegroup",
        model_parameters["all_stratifications"]["agegroup"],
        [abs_props[stratum] for stratum in fixed_prop_strata],
    )

    # Set time-variant proportion of sympt_isolate among all symptomatics
    # create a scale-up function converging to 1
    scale_up_multiplier = tanh_based_scaleup(
        model_parameters["tv_detection_b"],
        model_parameters["tv_detection_c"],
        model_parameters["tv_detection_sigma"],
    )
    # use the input parameter 'prop_detected_among_symptomatic', specifying the maximum prop of isolates among all sympt
    _tv_prop_detect_among_sympt = lambda t: model_parameters[
        "prop_detected_among_symptomatic"
    ] * scale_up_multiplier(t)

    # Set isolation rates as absolute proportions
    _covid_model, stratification_adjustments = set_isolation_props(
        _covid_model,
        model_parameters,
        abs_props,
        stratification_adjustments,
        _tv_prop_detect_among_sympt,
    )

    # Calculate death rates and progression rates for hospitalised and ICU patients
    progression_death_rates = {}
    for stratum in ("hospital", "icu"):
        (
            progression_death_rates[stratum + "_infect_death"],
            progression_death_rates[stratum + "_within_late"],
        ) = find_rates_and_complements_from_ifr(
            rel_props[stratum + "_death"],
            1,
            [model_parameters["within_" + stratum + "_late"]] * 16,
        )

    # Death and non-death progression between infectious compartments towards the recovered compartment
    for param in ("within_late", "infect_death"):
        stratification_adjustments.update(
            adjust_upstream_stratified_parameter(
                param,
                strata_to_implement[3:],
                "agegroup",
                model_parameters["all_stratifications"]["agegroup"],
                [
                    progression_death_rates["hospital_" + param],
                    progression_death_rates["icu_" + param],
                ],
                overwrite=True,
            )
        )

    # Over-write rate of progression for early compartments for hospital and ICU
    stratification_adjustments.update(
        {
            "within_infectious": {
                "hospital_non_icuW": model_parameters["within_hospital_early"],
                "icuW": model_parameters["within_icu_early"],
            },
        }
    )

    # Sort out all infectiousness adjustments for entire model here
    _covid_model, stratification_adjustments, strata_infectiousness = adjust_infectiousness(
        _covid_model, model_parameters, strata_to_implement, stratification_adjustments
    )

    # work out time-variant clinical proportions for imported cases accounting for quarantine
    if model_parameters["implement_importation"] and model_parameters["imported_cases_explict"]:
        rep_age_group = (
            "35"  # the clinical split will be defined according to this representative age-group
        )
        tvs = _covid_model.time_variants  # to reduce verbosity

        quarantine_scale_up = scale_up_function(
            model_parameters["traveller_quarantine"]["times"],
            model_parameters["traveller_quarantine"]["values"],
            method=4,
        )

        tv_prop_imported_non_sympt = lambda t: stratification_adjustments[
            "to_infectiousXagegroup_" + rep_age_group
        ]["non_sympt"] * (1.0 - quarantine_scale_up(t))

        tv_prop_imported_sympt_non_hospital = lambda t: tvs[
            stratification_adjustments["to_infectiousXagegroup_" + rep_age_group][
                "sympt_non_hospital"
            ]
        ](t) * (1.0 - quarantine_scale_up(t))

        tv_prop_imported_sympt_isolate = lambda t: tvs[
            stratification_adjustments["to_infectiousXagegroup_" + rep_age_group]["sympt_isolate"]
        ](t) + quarantine_scale_up(t) * (
            tvs[
                stratification_adjustments["to_infectiousXagegroup_" + rep_age_group][
                    "sympt_non_hospital"
                ]
            ](t)
            + stratification_adjustments["to_infectiousXagegroup_" + rep_age_group]["non_sympt"]
        )

        tv_prop_imported_hospital_non_icu = lambda t: tvs[
            stratification_adjustments["to_infectiousXagegroup_" + rep_age_group][
                "hospital_non_icu"
            ]
        ](t)

        tv_prop_imported_icu = lambda t: tvs[
            stratification_adjustments["to_infectiousXagegroup_" + rep_age_group]["icu"]
        ](t)

        _covid_model.time_variants["tv_prop_imported_non_sympt"] = tv_prop_imported_non_sympt
        _covid_model.time_variants[
            "tv_prop_imported_sympt_non_hospital"
        ] = tv_prop_imported_sympt_non_hospital
        _covid_model.time_variants[
            "tv_prop_imported_sympt_isolate"
        ] = tv_prop_imported_sympt_isolate
        _covid_model.time_variants[
            "tv_prop_imported_hospital_non_icu"
        ] = tv_prop_imported_hospital_non_icu
        _covid_model.time_variants["tv_prop_imported_icu"] = tv_prop_imported_icu

        importation_props_by_clinical = {}
        for stratum in strata_to_implement:
            importation_props_by_clinical[stratum] = "tv_prop_imported_" + stratum
    else:
        importation_props_by_clinical = {}

    # Stratify the model using the SUMMER stratification function
    _covid_model.stratify(
        "clinical",
        strata_to_implement,
        compartments_to_split,
        infectiousness_adjustments=strata_infectiousness,
        requested_proportions={
            stratum: 1.0 / len(strata_to_implement) for stratum in strata_to_implement
        },
        adjustment_requests=stratification_adjustments,
        entry_proportions=importation_props_by_clinical,
        verbose=False,
    )
    return _covid_model, model_parameters
