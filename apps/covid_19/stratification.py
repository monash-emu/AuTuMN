from autumn.tool_kit.utils import (
    find_rates_and_complements_from_ifr,
    repeat_list_elements,
    repeat_list_elements_average_last_two,
    element_wise_list_division
)
from autumn.constants import Compartment
from autumn.summer_related.parameter_adjustments import (
    adjust_upstream_stratified_parameter,
    split_prop_into_two_subprops,
)


def get_raw_clinical_props(params):
    """
    Get the raw proportions of persons dying and progressing to symptomatic, hospital if symptomatic, and to ICU if
    hospitalised - adjusting to required number of age groups.
    """
    return {
        "adjusted_infection_fatality_props":
            repeat_list_elements_average_last_two(params["infection_fatality_props"]),
        "raw_sympt":
            repeat_list_elements(2, params["symptomatic_props"]),
        "raw_hospital":
            repeat_list_elements_average_last_two(params["hospital_props"])
    }


def find_abs_death_props(params, abs_props):
    """
    Calculate the absolute proportion of all patients who should eventually reach hospital death or ICU death.
    """

    # Find IFR that needs to be contributed by ICU and non-ICU hospital deaths\
    hospital_death, icu_death = [], []
    for i_agegroup, agegroup in enumerate(params["all_stratifications"]["agegroup"]):

        # If IFR for age group is greater than absolute proportion hospitalised, increased hospitalised proportion
        if params["adjusted_infection_fatality_props"][i_agegroup] > \
                abs_props["hospital"][i_agegroup]:
            abs_props["hospital"][i_agegroup] = params["adjusted_infection_fatality_props"][i_agegroup]

        # Find the target absolute ICU mortality and the amount left over from IFRs to go to hospital, if any
        target_icu_abs_mort = \
            abs_props["icu"][i_agegroup] * \
            params["icu_mortality_prop"]
        left_over_mort = \
            params["adjusted_infection_fatality_props"][i_agegroup] - \
            target_icu_abs_mort

        # If some IFR will be left over for the hospitalised
        if left_over_mort > 0.:
            hospital_death.append(left_over_mort)
            icu_death.append(target_icu_abs_mort)

        # Otherwise if all IFR taken up by ICU
        else:
            hospital_death.append(0.0)
            icu_death.append(params["adjusted_infection_fatality_props"][i_agegroup])

    return {"hospital_death": hospital_death, "icu_death": icu_death}


def set_isolation_props(_covid_model, model_parameters, abs_props, stratification_adjustments):
    """
    Set the absolute proportions of new cases isolated and not isolated, and indicate to the model where they should be
    found.
    """

    # Apply the isolated proportion to the symptomatic non-hospitalised group
    for i_age, agegroup in enumerate(model_parameters["all_stratifications"]["agegroup"]):
        prop_isolated = \
            (abs_props["sympt"][i_age] * model_parameters["prop_isolated_among_symptomatic"] -
             abs_props["hospital"][i_age]) / \
            abs_props["sympt_non_hospital"][i_age]
        stratification_adjustments["to_infectiousXagegroup_" + agegroup]["sympt_isolate"] = \
            abs_props["sympt_non_hospital"][i_age] * \
            prop_isolated
        stratification_adjustments["to_infectiousXagegroup_" + agegroup]["sympt_non_hospital"] = \
            abs_props["sympt_non_hospital"][i_age] * \
            (1.0 - prop_isolated)
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
    model.individual_infectiousness_adjustments = \
        [
            [[Compartment.LATE_INFECTIOUS, "clinical_sympt_isolate"], 0.2]
        ]
    return model, adjustments, strata_infectiousness


def stratify_by_clinical(_covid_model, model_parameters, compartments):
    """
    Stratify the infectious compartments of the covid model (not including the pre-symptomatic compartments, which are
    actually infectious)
    """

    # Define stratification
    strata_to_implement = \
        model_parameters["clinical_strata"]
    model_parameters["all_stratifications"]["clinical"] = \
        strata_to_implement
    compartments_to_split = [
        comp
        for comp in compartments
        if comp.startswith(Compartment.EARLY_INFECTIOUS) or comp.startswith(Compartment.LATE_INFECTIOUS)
    ]

    # Find unadjusted parameters
    model_parameters.update(get_raw_clinical_props(model_parameters))

    # Find the absolute progression proportions from the requested splits
    abs_props = split_prop_into_two_subprops([1.0] * 16, "", model_parameters["raw_sympt"], "sympt")
    abs_props.update(
        split_prop_into_two_subprops(abs_props["sympt"], "sympt", model_parameters["raw_hospital"], "hospital")
    )
    abs_props.update(
        split_prop_into_two_subprops(abs_props["hospital"], "hospital", [model_parameters["icu_prop"]] * 16, "icu")
    )

    # Find the absolute proportion dying in hospital and in ICU
    abs_props.update(find_abs_death_props(model_parameters, abs_props))

    # CFR for non-ICU hospitalised patients
    rel_props = {
        "hospital_death": element_wise_list_division(abs_props["hospital_death"], abs_props["hospital_non_icu"]),
        "icu_death": element_wise_list_division(abs_props["icu_death"], abs_props["icu"])
    }

    # Progression rates into the infectious compartment(s)
    fixed_prop_strata = ["non_sympt", "hospital_non_icu", "icu"]
    stratification_adjustments = adjust_upstream_stratified_parameter(
        "to_infectious",
        fixed_prop_strata,
        "agegroup",
        model_parameters["all_stratifications"]["agegroup"],
        [abs_props[stratum] for stratum in fixed_prop_strata],
    )

    # Set isolation rates as absolute proportions
    _covid_model, stratification_adjustments = \
        set_isolation_props(_covid_model, model_parameters, abs_props, stratification_adjustments)

    # Calculate death rates and progression rates for hospitalised and ICU patients
    progression_death_rates = {}
    for stratum in ("hospital", "icu"):
        (
            progression_death_rates[stratum + "_infect_death"],
            progression_death_rates[stratum + "_within_late"],
        ) = find_rates_and_complements_from_ifr(
            rel_props[stratum + "_death"],
            model_parameters["n_compartment_repeats"][Compartment.LATE_INFECTIOUS],
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
                [progression_death_rates["hospital_" + param], progression_death_rates["icu_" + param]],
                overwrite=True,
            )
        )

    # Over-write rate of progression for early compartments for hospital and ICU
    stratification_adjustments.update(
        {"within_infectious":
             {"hospital_non_icuW":
                  model_parameters['within_hospital_early'],
              "icuW":
                  model_parameters['within_icu_early']},
         }
    )

    # Sort out all infectiousness adjustments for entire model here
    _covid_model, stratification_adjustments, strata_infectiousness = \
        adjust_infectiousness(_covid_model, model_parameters, strata_to_implement, stratification_adjustments)

    # work out clinical proportions for imported cases
    importation_props_by_clinical = {
        "non_sympt": 1. - model_parameters['symptomatic_props_imported'],
        "sympt_non_hospital": model_parameters['symptomatic_props_imported'] *
                              (1. - model_parameters['prop_isolated_among_symptomatic_imported'] -
                               model_parameters['hospital_props_imported']),
        "sympt_isolate": model_parameters['symptomatic_props_imported'] *
                         model_parameters['prop_isolated_among_symptomatic_imported'],
        "hospital_non_icu": model_parameters['hospital_props_imported'] * (1. - model_parameters['icu_prop_imported']),
        "icu": model_parameters['hospital_props_imported'] * model_parameters['icu_prop_imported']
    }

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
