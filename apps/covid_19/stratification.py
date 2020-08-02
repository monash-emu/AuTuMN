import numpy as np

from autumn.tool_kit.utils import (
    find_rates_and_complements_from_ifr,
    repeat_list_elements_average_last_two,
    element_wise_list_division,
)
from autumn.summer_related.parameter_adjustments import adjust_upstream_stratified_parameter
from autumn.curve import scale_up_function
from apps.covid_19.preprocess.mortality import age_specific_ifrs_from_double_exp_model
from autumn.inputs import get_population_by_agegroup

from apps.covid_19.constants import Compartment


def stratify_by_clinical(model, model_parameters, compartments, detected_proportion, symptomatic_props):
    """
    Stratify the infectious compartments of the covid model (not including the pre-symptomatic compartments, which are
    actually infectious)

    - notifications are derived from progress from early to late for some strata
    - the proportion of people moving from presymt to early infectious, conditioned on age group
    - rate of which people flow through these compartments (reciprocal of time, using within_* which is a rate of ppl / day)
    - infectiousness levels adjusted by early/late and for clinical strata
    - we start with an age stratified infection fatality rate
        - 50% of deaths for each age bracket die in ICU
        - the other deaths go to hospital, assume no-one else can die from COVID
        - should we ditch this?

    """
    agegroup_strata = model_parameters["all_stratifications"]["agegroup"]
    all_stratifications = model_parameters["all_stratifications"]
    clinical_strata = model_parameters["clinical_strata"]
    implement_importation = model_parameters["implement_importation"]
    traveller_quarantine = model_parameters["traveller_quarantine"]
    within_hospital_early = model_parameters["within_hospital_early"]
    within_icu_early = model_parameters["within_icu_early"]
    icu_prop = model_parameters["icu_prop"]
    icu_mortality_prop = model_parameters["icu_mortality_prop"]
    infection_fatality_props_10_year = model_parameters["infection_fatality_props"]
    hospital_props = model_parameters["hospital_props"]
    use_raw_mortality_estimates = model_parameters["use_raw_mortality_estimates"]
    ifr_double_exp_model_params = model_parameters["ifr_double_exp_model_params"]

    # apply multiplier to symptomatic proportions
    symptomatic_props = [min([p * model_parameters["symptomatic_props_multiplier"], 1.]) for p in symptomatic_props]

    # apply multiplyer to hospital proportions
    hospital_props = [min([p * model_parameters["hospital_props_multiplier"], 1.]) for p in hospital_props]

    # Define stratification - only stratify infected compartments
    strata_to_implement = clinical_strata
    all_stratifications["clinical"] = strata_to_implement
    compartments_to_split = [
        comp
        for comp in compartments
        if comp.startswith(Compartment.EARLY_ACTIVE)
           or comp.startswith(Compartment.LATE_ACTIVE)
    ]

    # FIXME: Set params to make comparison happy
    model_parameters["infection_fatality_props"] = [
        ifr * adjustment for
        ifr, adjustment in
        zip(infection_fatality_props_10_year, model_parameters["ifr_multipliers"])
    ]

    # Calculate the proportion of 80+ years old among the 75+ population
    elderly_populations = get_population_by_agegroup([0, 75, 80], model_parameters["iso3"], model_parameters["region"])
    prop_over_80 = elderly_populations[2] / sum(elderly_populations[1:])

    # Infection fatality rate by age group.
    if use_raw_mortality_estimates:
        # Data in props used 10 year bands 0-80+, but we want 5 year bands from 0-75+

        # Calculate 75+ age bracket as weighted average between 75-79 and half 80+
        infection_fatality_props = repeat_list_elements_average_last_two(
            infection_fatality_props_10_year, prop_over_80
        )
    else:
        infection_fatality_props = age_specific_ifrs_from_double_exp_model(
            ifr_double_exp_model_params['k'],
            ifr_double_exp_model_params['m'],
            ifr_double_exp_model_params['last_representative_age']
        )

    # Find the absolute progression proportions.
    symptomatic_props_arr = np.array(symptomatic_props)
    hospital_props_arr = np.array(hospital_props)
    # Determine the absolute proportion of presymptomatic who become sympt vs non-sympt.
    sympt, non_sympt = subdivide_props(1, symptomatic_props_arr)
    # Determine the absolute proportion of sympt who become hospitalized vs non-hospitalized.
    sympt_hospital, sympt_non_hospital = subdivide_props(sympt, hospital_props_arr)
    # Determine the absolute proportion of hospitalized who become icu vs non-icu.
    sympt_hospital_icu, sympt_hospital_non_icu = subdivide_props(sympt_hospital, icu_prop)
    abs_props = {
        "sympt": sympt.tolist(),
        "non_sympt": non_sympt.tolist(),
        "hospital": sympt_hospital.tolist(),
        "sympt_non_hospital": sympt_non_hospital.tolist(),  # Over-ridden by a by time-varying proportion later
        "icu": sympt_hospital_icu.tolist(),
        "hospital_non_icu": sympt_hospital_non_icu.tolist(),
    }

    # Calculate the absolute proportion of all patients who should eventually reach hospital death or ICU death.
    # Find IFR that needs to be contributed by ICU and non-ICU hospital deaths
    hospital_death, icu_death = [], []
    for age_idx, agegroup in enumerate(agegroup_strata):
        # If IFR for age group is greater than absolute proportion hospitalised, increased hospitalised proportion
        if infection_fatality_props[age_idx] > abs_props["hospital"][age_idx]:
            abs_props["hospital"][age_idx] = infection_fatality_props[age_idx]

        # Find the target absolute ICU mortality and the amount left over from IFRs to go to hospital, if any
        target_icu_abs_mort = abs_props["icu"][age_idx] * icu_mortality_prop
        left_over_mort = infection_fatality_props[age_idx] - target_icu_abs_mort

        # If enough IFR left over to allow partial mortality for the hospitalised
        if 0.0 < left_over_mort <= abs_props['hospital_non_icu'][age_idx]:
            hospital_death_prop = left_over_mort
            icu_death_prop = target_icu_abs_mort

        # Otherwise if there is too much excess death to fit in hospital, inflate the ICU proportion
        elif left_over_mort > abs_props['hospital_non_icu'][age_idx]:
            hospital_death_prop = abs_props['hospital_non_icu'][age_idx]
            icu_death_prop = infection_fatality_props[age_idx] - abs_props['hospital_non_icu'][age_idx]

        # Otherwise if all IFR taken up by ICU
        else:
            hospital_death_prop = 0.0
            icu_death_prop = infection_fatality_props[age_idx]

        hospital_death.append(hospital_death_prop)
        icu_death.append(icu_death_prop)

    abs_props.update({"hospital_death": hospital_death, "icu_death": icu_death})

    # FIXME: These depend on static variables which have been made time-variant.
    # fatality rate for hospitalised patients
    rel_props = {
        "hospital_death": element_wise_list_division(
            abs_props["hospital_death"], abs_props["hospital_non_icu"]
        ),
        "icu_death": element_wise_list_division(abs_props["icu_death"], abs_props["icu"]),
    }

    # Progression rates into the infectious compartment(s)
    # Define progression rates into non-symptomatic compartments using parameter adjustment.
    stratification_adjustments = {}
    for age_idx, age in enumerate(agegroup_strata):
        key = f"to_infectiousXagegroup_{age}"
        stratification_adjustments[key] = {
            "non_sympt": non_sympt[age_idx],
            "icu": sympt_hospital_icu[age_idx],
            "hospital_non_icu": sympt_hospital_non_icu[age_idx],
        }

    # Set time-varying isolation proportions
    for age_idx, agegroup in enumerate(agegroup_strata):
        # Pass the functions to the model
        tv_props = TimeVaryingProportions(age_idx, abs_props, detected_proportion)
        time_variants = [
            [f"prop_sympt_non_hospital_{agegroup}", tv_props.get_abs_prop_sympt_non_hospital,],
            [f"prop_sympt_isolate_{agegroup}", tv_props.get_abs_prop_isolated],
        ]
        for name, func in time_variants:
            model.time_variants[name] = func

        # Tell the model to use these time varying functions for the stratification adjustments.
        agegroup_adj = stratification_adjustments[f"to_infectiousXagegroup_{agegroup}"]
        agegroup_adj["sympt_non_hospital"] = f"prop_sympt_non_hospital_{agegroup}"
        agegroup_adj["sympt_isolate"] = f"prop_sympt_isolate_{agegroup}"

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
                all_stratifications["agegroup"],
                [
                    progression_death_rates["hospital_" + param],
                    progression_death_rates["icu_" + param],
                ],
                overwrite=True,
            )
        )

    # Over-write rate of progression for early compartments for hospital and ICU
    # FIXME: Ask Romain if he knows why we bother doing this.
    stratification_adjustments.update(
        {
            "within_infectious": {
                "hospital_non_icuW": within_hospital_early,
                "icuW": within_icu_early,
            },
        }
    )

    # Make infectiousness adjustment for asymptomatic patients only
    strata_infectiousness = {}
    for stratum in strata_to_implement:
        if stratum + "_infect_multiplier" in model_parameters:
            strata_infectiousness[stratum] = model_parameters[stratum + "_infect_multiplier"]

    # Make adjustment for isolation/quarantine
    for stratum in strata_to_implement:
        if stratum in model_parameters["late_infect_multiplier"]:
            model.individual_infectiousness_adjustments.append(
                [
                    [Compartment.LATE_ACTIVE, "clinical_" + stratum],
                    model_parameters["late_infect_multiplier"][stratum],
                ]
            )

    # FIXME: Ask Romain about importation
    # Work out time-variant clinical proportions for imported cases accounting for quarantine
    if implement_importation:
        to_infectious_key = "to_infectiousXagegroup_" + str(model_parameters['import_representative_age'])
        tvs = model.time_variants

        # Create scale-up function for quarantine
        quarantine_scale_up = \
            scale_up_function(traveller_quarantine["times"], traveller_quarantine["values"], method=4)

        # Clinical proportions for imported cases stay fixed for the hospitalised and critical groups (last two strata)
        importation_props_by_clinical = \
            {stratum: float(stratification_adjustments[to_infectious_key][stratum]) for
             stratum in strata_to_implement[-2:]}

        # Proportion entering non-symptomatic stratum reduced by the quarantined (and so isolated) proportion
        def tv_prop_imported_non_sympt(t):
            return stratification_adjustments[to_infectious_key]["non_sympt"] * \
                   (1. - quarantine_scale_up(t))

        # Proportion ambulatory also reduced by quarantined proportion due to isolation
        def tv_prop_imported_sympt_non_hospital(t):
            return tvs[stratification_adjustments[to_infectious_key]["sympt_non_hospital"]](t) * \
                   (1. - quarantine_scale_up(t))

        # Proportion isolated includes those that would have been detected anyway and the ones above quarantined
        def tv_prop_imported_sympt_isolate(t):
            return tvs[stratification_adjustments[to_infectious_key]["sympt_isolate"]](t) + \
                   quarantine_scale_up(t) * \
                   (tvs[stratification_adjustments[to_infectious_key]["sympt_non_hospital"]](t) +
                    stratification_adjustments[to_infectious_key]["non_sympt"])

        # Pass time-variant functions to the model object
        model.time_variants["tv_prop_imported_non_sympt"] = \
            tv_prop_imported_non_sympt
        model.time_variants["tv_prop_imported_sympt_non_hospital"] = \
            tv_prop_imported_sympt_non_hospital
        model.time_variants["tv_prop_imported_sympt_isolate"] = \
            tv_prop_imported_sympt_isolate
        for stratum in strata_to_implement[:3]:
            importation_props_by_clinical[stratum] = "tv_prop_imported_" + stratum

    else:
        importation_props_by_clinical = {}

    # Stratify the model using the SUMMER stratification function
    model.stratify(
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


def subdivide_props(base_props: np.ndarray, split_props: np.ndarray):
    """
    Split an array (base_array) of proportions into two arrays (split_arr, complement_arr)
    according to the split proportions provided (split_prop).
    """
    split_arr = base_props * split_props
    complement_arr = base_props * (1 - split_props)
    return split_arr, complement_arr


class TimeVaryingProportions:
    """
    Provides time-varying proportions for a given age group.
    The proportions determine which clinical stratum people transition into when they go
    from being presymptomatic to early infectious.
    """

    def __init__(self, age_idx, abs_props, prop_detect_func):
        self.abs_prop_sympt = abs_props["sympt"][age_idx]
        self.abs_prop_hospital = abs_props["hospital"][age_idx]
        self.prop_detect_among_sympt_func = prop_detect_func

    def get_abs_prop_isolated(self, t):
        """
        Returns the absolute proportion of infected becoming isolated at home.
        Isolated people are those who are detected but not sent to hospital.
        """
        abs_prop_detected = self.abs_prop_sympt * self.prop_detect_among_sympt_func(t)
        abs_prop_isolated = abs_prop_detected - self.abs_prop_hospital
        if abs_prop_isolated < 0:
            # If more people go to hospital than are detected, ignore detection
            # proportion, and assume no one is being isolated.
            abs_prop_isolated = 0

        return abs_prop_isolated

    def get_abs_prop_sympt_non_hospital(self, t):
        """
        Returns the absolute proportion of infected not entering the hospital.
        This also does not count people who are isolated.
        This is only people who are not detected.
        """
        return self.abs_prop_sympt - self.abs_prop_hospital - self.get_abs_prop_isolated(t)
