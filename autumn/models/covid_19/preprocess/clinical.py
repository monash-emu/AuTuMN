import numpy as np
from summer import Overwrite

from autumn.models.covid_19.constants import Clinical, Compartment
from autumn.models.covid_19.model import preprocess
from autumn.models.covid_19.preprocess.case_detection import build_detected_proportion_func
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.tools import inputs
from autumn.tools.utils.utils import (
    apply_odds_ratio_to_proportion,
    repeat_list_elements,
    repeat_list_elements_average_last_two,
)


def get_entry_adjustments(abs_props, get_detected_proportion, early_rate):

    adjustments = {}
    for age_idx, agegroup in enumerate(AGEGROUP_STRATA):

        # Function-based flow rates

        # Get isolated rate for overwriting
        get_abs_prop_isolated = get_abs_prop_isolated_factory(
            age_idx, abs_props, get_detected_proportion
        )

        def isolate_flow_rate(
            t, func=get_abs_prop_isolated
        ):  # Function must be "bound" within loop
            return func(t) * early_rate

        # Get sympt non-hospital rate for overwriting
        get_abs_prop_sympt_non_hospital = get_abs_prop_sympt_non_hospital_factory(
            age_idx, abs_props, get_abs_prop_isolated
        )

        def sympt_non_hosp_rate(
            t, func=get_abs_prop_sympt_non_hospital
        ):  # Function must be "bound" within loop
            return func(t) * early_rate

        # Constant flow rates
        clinical_non_sympt_rate = abs_props[Clinical.NON_SYMPT][age_idx] * early_rate
        clinical_icu_rate = abs_props[Clinical.ICU][age_idx] * early_rate
        hospital_non_icu_rate = abs_props[Clinical.HOSPITAL_NON_ICU][age_idx] * early_rate

        # Age-specific adjustments object
        adjustments[agegroup] = {
            Clinical.NON_SYMPT: Overwrite(clinical_non_sympt_rate),
            Clinical.ICU: Overwrite(clinical_icu_rate),
            Clinical.HOSPITAL_NON_ICU: Overwrite(hospital_non_icu_rate),
            Clinical.SYMPT_NON_HOSPITAL: Overwrite(sympt_non_hosp_rate),
            Clinical.SYMPT_ISOLATE: Overwrite(isolate_flow_rate),
        }

    return adjustments


def get_proportion_symptomatic(clinical_params):
    """
    This is defined 8x10 year bands, 0-70+, which we transform into 16x5 year bands 0-75+.
    """
    return repeat_list_elements(2, clinical_params.props.symptomatic.props)


def get_abs_prop_isolated_factory(age_idx, abs_props, prop_detect_among_sympt_func):
    def get_abs_prop_isolated(t):
        """
        Returns the absolute proportion of infected becoming isolated at home.
        Isolated people are those who are detected but not sent to hospital.
        """
        abs_prop_detected = abs_props["sympt"][age_idx] * prop_detect_among_sympt_func(t)
        abs_prop_isolated = abs_prop_detected - abs_props["hospital"][age_idx]
        if abs_prop_isolated < 0:
            # If more people go to hospital than are detected, ignore detection
            # proportion, and assume no one is being isolated.
            abs_prop_isolated = 0

        return abs_prop_isolated

    return get_abs_prop_isolated


def get_abs_prop_sympt_non_hospital_factory(age_idx, abs_props, get_abs_prop_isolated_func):
    def get_abs_prop_sympt_non_hospital(t):
        """
        Returns the absolute proportion of infected not entering the hospital.
        This also does not count people who are isolated.
        This is only people who are not detected.
        """
        return (
            abs_props["sympt"][age_idx]
            - abs_props["hospital"][age_idx]
            - get_abs_prop_isolated_func(t)
        )

    return get_abs_prop_sympt_non_hospital


def get_absolute_strata_proportions(
    symptomatic_props,
    icu_props,
    hospital_props,
    symptomatic_props_multiplier,
    hospital_props_multiplier,
):
    """
    Returns the proportion of people in each clinical stratum.
    ie: Given all the people people who are infected, what proportion are in each strata?
    Each of these are stratified into 16 age groups 0-75+
    """
    # Apply multiplier to proportions
    hospital_props = [
        apply_odds_ratio_to_proportion(i_prop, hospital_props_multiplier)
        for i_prop in hospital_props
    ]
    symptomatic_props = [
        apply_odds_ratio_to_proportion(i_prop, symptomatic_props_multiplier)
        for i_prop in symptomatic_props
    ]

    # Find the absolute progression proportions.
    symptomatic_props_arr = np.array(symptomatic_props)
    hospital_props_arr = np.array(hospital_props)
    # Determine the absolute proportion of early exposed who become sympt vs non-sympt
    sympt, non_sympt = subdivide_props(1, symptomatic_props_arr)
    # Determine the absolute proportion of sympt who become hospitalized vs non-hospitalized.
    sympt_hospital, sympt_non_hospital = subdivide_props(sympt, hospital_props_arr)
    # Determine the absolute proportion of hospitalized who become icu vs non-icu.
    sympt_hospital_icu, sympt_hospital_non_icu = subdivide_props(sympt_hospital, icu_props)

    return {
        "sympt": sympt,
        "non_sympt": non_sympt,
        "hospital": sympt_hospital,
        "sympt_non_hospital": sympt_non_hospital,  # Over-ridden by a by time-varying proportion later
        Clinical.ICU: sympt_hospital_icu,
        Clinical.HOSPITAL_NON_ICU: sympt_hospital_non_icu,
    }


def get_absolute_death_proportions(abs_props, infection_fatality_props, icu_mortality_prop):
    """
    Calculate death proportions: find where the absolute number of deaths accrue
    Represents the number of people in a strata who die given the total number of people infected.
    """
    NUM_AGE_STRATA = 16
    abs_death_props = {
        Clinical.NON_SYMPT: np.zeros(NUM_AGE_STRATA),
        Clinical.ICU: np.zeros(NUM_AGE_STRATA),
        Clinical.HOSPITAL_NON_ICU: np.zeros(NUM_AGE_STRATA),
    }
    for age_idx in range(NUM_AGE_STRATA):
        age_ifr_props = infection_fatality_props[age_idx]

        # Make sure there are enough asymptomatic and hospitalised proportions to fill the IFR
        thing = (
            abs_props["non_sympt"][age_idx]
            + abs_props[Clinical.HOSPITAL_NON_ICU][age_idx]
            + abs_props[Clinical.ICU][age_idx] * icu_mortality_prop
        )
        age_ifr_props = min(thing, age_ifr_props)

        # Absolute proportion of all patients dying in ICU
        # Maximum ICU mortality allowed
        thing = abs_props[Clinical.ICU][age_idx] * icu_mortality_prop
        abs_death_props[Clinical.ICU][age_idx] = min(thing, age_ifr_props)

        # Absolute proportion of all patients dying in hospital, excluding ICU
        thing = max(
            age_ifr_props
            - abs_death_props[Clinical.ICU][
                age_idx
            ],  # If left over mortality from ICU for hospitalised
            0.0,  # Otherwise zero
        )
        abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx] = min(
            thing,
            # Otherwise fill up hospitalised
            abs_props[Clinical.HOSPITAL_NON_ICU][age_idx],
        )

        # Absolute proportion of all patients dying out of hospital
        thing = (
            age_ifr_props
            - abs_death_props[Clinical.ICU][age_idx]
            - abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx]
        )  # If left over mortality from hospitalised
        abs_death_props[Clinical.NON_SYMPT][age_idx] = max(0.0, thing)  # Otherwise zero

        # Check everything sums up properly
        allowed_rounding_error = 6
        assert (
            round(
                abs_death_props[Clinical.ICU][age_idx]
                + abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx]
                + abs_death_props[Clinical.NON_SYMPT][age_idx],
                allowed_rounding_error,
            )
            == round(age_ifr_props, allowed_rounding_error)
        )
        # Check everything sums up properly
        allowed_rounding_error = 6
        assert (
            round(
                abs_death_props[Clinical.ICU][age_idx]
                + abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx]
                + abs_death_props[Clinical.NON_SYMPT][age_idx],
                allowed_rounding_error,
            )
            == round(age_ifr_props, allowed_rounding_error)
        )

    return abs_death_props


def get_infection_fatality_proportions(
    infection_fatality_props_10_year,
    infection_rate_multiplier,
    iso3,
    pop_region,
    pop_year,
):
    """
    Returns the Proportion of people in age group who die, given the total number of people in that compartment.
    ie: dead / total infected
    """
    if_props_10_year = [
        apply_odds_ratio_to_proportion(i_prop, infection_rate_multiplier)
        for i_prop in infection_fatality_props_10_year
    ]
    # Calculate the proportion of 80+ years old among the 75+ population
    elderly_populations = inputs.get_population_by_agegroup(
        [0, 75, 80], iso3, pop_region, year=pop_year
    )
    prop_over_80 = elderly_populations[2] / sum(elderly_populations[1:])
    # Infection fatality rate by age group.
    # Data in props may have used 10 year bands 0-80+, but we want 5 year bands from 0-75+
    # Calculate 75+ age bracket as weighted average between 75-79 and half 80+
    if len(infection_fatality_props_10_year) == 17:
        last_ifr = if_props_10_year[-1] * prop_over_80 + if_props_10_year[-2] * (1 - prop_over_80)
        ifrs_by_age = if_props_10_year[:-1]
        ifrs_by_age[-1] = last_ifr
    else:
        ifrs_by_age = repeat_list_elements_average_last_two(if_props_10_year, prop_over_80)
    return ifrs_by_age


def subdivide_props(base_props: np.ndarray, split_props: np.ndarray):
    """
    Split an array (base_array) of proportions into two arrays (split_arr, complement_arr)
    according to the split proportions provided (split_prop).
    """
    split_arr = base_props * split_props
    complement_arr = base_props * (1 - split_props)
    return split_arr, complement_arr


def get_ifr_props(adjuster, country, pop, ifr_props, top_bracket_overwrite=None):

    # Proportion of people in age group who die, given the number infected: dead / total infected.
    base_ifr_props = get_infection_fatality_proportions(
        infection_fatality_props_10_year=ifr_props,
        infection_rate_multiplier=adjuster,
        iso3=country.iso3,
        pop_region=pop.region,
        pop_year=pop.year,
    )
    if top_bracket_overwrite:
        base_ifr_props[-1] = top_bracket_overwrite
    return base_ifr_props


def get_sympt_props(symptomatic_adjuster, hospital_adjuster, clinical_params):
    """
    Get the proportion of people in each clinical stratum, relative to total people in compartment.
    """
    symptomatic_props = get_proportion_symptomatic(clinical_params)
    return get_absolute_strata_proportions(
        symptomatic_props=symptomatic_props,
        icu_props=clinical_params.icu_prop,
        hospital_props=clinical_params.props.hospital.props,
        symptomatic_props_multiplier=symptomatic_adjuster,
        hospital_props_multiplier=hospital_adjuster,
    )


def get_progress_adjs(within_hospital_early, within_icu_early):
    return {
        Clinical.NON_SYMPT: None,
        Clinical.ICU: Overwrite(within_icu_early),
        Clinical.HOSPITAL_NON_ICU: Overwrite(within_hospital_early),
        Clinical.SYMPT_NON_HOSPITAL: None,
        Clinical.SYMPT_ISOLATE: None,
    }


def get_rate_adjustments(asympt_rates, hospital_rates, icu_rates):
    """
    Apply adjusted infection death and recovery rates for asymptomatic and hospitalised patients (ICU and non-ICU)
    Death and non-death progression between the infectious compartments towards the recovered compartment
    """
    rate_adjustments = {}
    for i_age, age_group in enumerate(AGEGROUP_STRATA):
        rate_adjustments[age_group] = {
            Clinical.NON_SYMPT: Overwrite(asympt_rates[i_age]),
            Clinical.SYMPT_NON_HOSPITAL: None,
            Clinical.SYMPT_ISOLATE: None,
            Clinical.ICU: Overwrite(icu_rates[i_age]),
            Clinical.HOSPITAL_NON_ICU: Overwrite(hospital_rates[i_age]),
        }
    return rate_adjustments


def get_all_adjs(
    clinical_params,
    country,
    pop,
    ifr_props,
    sojourn,
    testing_to_detection,
    case_detection,
    ifr_adjuster,
    symptomatic_adjuster,
    hospital_adjuster,
    top_bracket_overwrite=None,
):

    """
    Work out all the relevant sojourn times and the associated total rates at which they exit the compartments.
    We assume everyone who dies does so at the end of their time in the "late active" compartment.
    Later, we split the flow rate out of "late active" into a death or recovery flow, based on proportion dying.
    """
    compartment_periods = preprocess.compartments.calc_compartment_periods(sojourn)
    within_asympt_late = 1. / compartment_periods["late_active"]
    within_hospital_early = 1. / sojourn.compartment_periods["hospital_early"]
    within_icu_early = 1. / sojourn.compartment_periods["icu_early"]
    within_hospital_late = 1. / sojourn.compartment_periods["hospital_late"]
    within_icu_late = 1. / sojourn.compartment_periods["icu_late"]

    """
    Process the age-structured IFR parameters.
    """
    infection_fatality_props = get_ifr_props(ifr_adjuster, country, pop, ifr_props, top_bracket_overwrite)

    """
    Work out the proportion of people entering each stratum who die in that stratum (for each age group).
    """

    # The proportion of all people who enter each stratum.
    # Numerator: people entering stratum, denominator: everyone
    abs_props = get_sympt_props(symptomatic_adjuster, hospital_adjuster, clinical_params)

    # The proportion of people entering each stratum who die in that stratum
    # Numerator: deaths in stratum, denominator: everyone
    abs_death_props = get_absolute_death_proportions(
        abs_props, infection_fatality_props, clinical_params.icu_mortality_prop
    )

    # The resulting proportion
    # Numerator: deaths in stratum, denominator: people entering stratum
    relative_death_props = {
        stratum: np.array(abs_death_props[stratum]) / np.array(abs_props[stratum])
        for stratum in (Clinical.HOSPITAL_NON_ICU, Clinical.ICU, Clinical.NON_SYMPT)
    }

    # The survival proportions are the complement of the death proportions.
    asympt_survival_props = 1.0 - relative_death_props[Clinical.NON_SYMPT]
    hospital_survival_props = 1.0 - relative_death_props[Clinical.HOSPITAL_NON_ICU]
    icu_survival_props = 1.0 - relative_death_props[Clinical.ICU]

    """
    Find death and survival rates from death proportions and sojourn times.
    """
    asympt_death_rates = relative_death_props[Clinical.NON_SYMPT] * within_asympt_late
    hospital_death_rates = relative_death_props[Clinical.HOSPITAL_NON_ICU] * within_hospital_late
    icu_death_rates = relative_death_props[Clinical.ICU] * within_icu_late

    asympt_survival_rates = asympt_survival_props * within_asympt_late
    hospital_survival_rates = hospital_survival_props * within_hospital_late
    icu_survival_rates = icu_survival_props * within_icu_late

    """
    Convert to summer parameter overwrite objects.
    """
    death_adjs = get_rate_adjustments(asympt_death_rates, hospital_death_rates, icu_death_rates)
    recovery_adjs = get_rate_adjustments(asympt_survival_rates, hospital_survival_rates, icu_survival_rates)

    """
    Do the entry adjustments.
    """
    get_detected_proportion = build_detected_proportion_func(
        AGEGROUP_STRATA, country, pop, testing_to_detection, case_detection
    )
    entry_adjustments = get_entry_adjustments(
        abs_props, get_detected_proportion, 1. / compartment_periods[Compartment.EARLY_EXPOSED]
    )
    progress_adjs = get_progress_adjs(within_hospital_early, within_icu_early)

    return (
        entry_adjustments,
        death_adjs,
        progress_adjs,
        recovery_adjs,
        abs_props,
        get_detected_proportion,
    )
