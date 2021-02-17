import numpy as np

from autumn import inputs
from autumn.tool_kit.utils import (
    apply_odds_ratio_to_proportion,
    repeat_list_elements,
    repeat_list_elements_average_last_two,
)
from apps.covid_19.model.stratifications.agegroup import AGEGROUP_STRATA
from summer2 import Multiply, Overwrite

from apps.covid_19.model.parameters import Parameters
from apps.covid_19.constants import Clinical


def get_proportion_symptomatic(params: Parameters):
    # This is defined 8x10 year bands, 0-70+, which we transform into 16x5 year bands 0-75+
    return repeat_list_elements(2, params.clinical_stratification.props.symptomatic.props)


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


def get_ifr_props(params, adjuster):

    country = params.country
    pop = params.population

    # Proportion of people in age group who die, given the number infected: dead / total infected.
    return get_infection_fatality_proportions(
        infection_fatality_props_10_year=params.infection_fatality.props,
        infection_rate_multiplier=adjuster,
        iso3=country.iso3,
        pop_region=pop.region,
        pop_year=pop.year,
    )


def get_sympt_props(params, symptomatic_adjuster, hospital_adjuster):

    clinical_params = params.clinical_stratification

    # Get the proportion of people in each clinical stratum, relative to total people in compartment.
    symptomatic_props = get_proportion_symptomatic(params)
    return get_absolute_strata_proportions(
        symptomatic_props=symptomatic_props,
        icu_props=clinical_params.icu_prop,
        hospital_props=clinical_params.props.hospital.props,
        symptomatic_props_multiplier=symptomatic_adjuster,
        hospital_props_multiplier=hospital_adjuster,
    )


def get_relative_death_props(abs_props, abs_death_props):
    # Calculate relative death proportions for each strata / agegroup.
    # This is the number of people in strata / agegroup who die, given the total num people in that strata / agegroup.
    return {
        stratum: np.array(abs_death_props[stratum]) / np.array(abs_props[stratum])
        for stratum in (
            Clinical.HOSPITAL_NON_ICU,
            Clinical.ICU,
            Clinical.NON_SYMPT,
        )
    }


def get_hosp_sojourns(sojourn):
    # Now we want to convert these death proportions into flow rates
    # These flow rates are the death rates for hospitalised patients in ICU and non-ICU
    # We assume everyone who dies does so at the end of their time in the "late active" compartment
    # We split the flow rate out of "late active" into a death or recovery flow, based on the relative death proportion
    within_hospital_late = 1.0 / sojourn.compartment_periods["hospital_late"]
    within_icu_late = 1.0 / sojourn.compartment_periods["icu_late"]

    return within_hospital_late, within_icu_late


def get_hosp_death_rates(relative_death_props, within_hospital_late, within_icu_late):
    hospital_death_rates = relative_death_props[Clinical.HOSPITAL_NON_ICU] * within_hospital_late
    icu_death_rates = relative_death_props[Clinical.ICU] * within_icu_late

    return hospital_death_rates, icu_death_rates


def apply_death_adjustments(hospital_death_rates, icu_death_rates):

    # Apply adjusted infection death rates for hospital patients (ICU and non-ICU)
    # Death and non-death progression between infectious compartments towards the recovered compartment
    death_adjs = {}
    for idx, age_group in enumerate(AGEGROUP_STRATA):
        death_adjs[age_group] = {
            Clinical.NON_SYMPT: None,
            Clinical.SYMPT_NON_HOSPITAL: None,
            Clinical.SYMPT_ISOLATE: None,
            Clinical.HOSPITAL_NON_ICU: Overwrite(hospital_death_rates[idx]),
            Clinical.ICU: Overwrite(icu_death_rates[idx]),
        }
    return death_adjs


def get_entry_adjustments(abs_props, get_detected_proportion):

    adjustments = {}
    for age_idx, agegroup in enumerate(AGEGROUP_STRATA):
        get_abs_prop_isolated = get_abs_prop_isolated_factory(
            age_idx, abs_props, get_detected_proportion
        )
        get_abs_prop_sympt_non_hospital = get_abs_prop_sympt_non_hospital_factory(
            age_idx, abs_props, get_abs_prop_isolated
        )
        adjustments[agegroup] = {
            Clinical.NON_SYMPT: Multiply(abs_props[Clinical.NON_SYMPT][age_idx]),
            Clinical.ICU: Multiply(abs_props[Clinical.ICU][age_idx]),
            Clinical.HOSPITAL_NON_ICU: Multiply(abs_props[Clinical.HOSPITAL_NON_ICU][age_idx]),
            Clinical.SYMPT_NON_HOSPITAL: Multiply(get_abs_prop_sympt_non_hospital),
            Clinical.SYMPT_ISOLATE: Multiply(get_abs_prop_isolated),
        }

    return adjustments
