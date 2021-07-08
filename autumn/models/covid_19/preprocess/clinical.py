import numpy as np
from summer import Overwrite

from autumn.models.covid_19.constants import Clinical, Compartment, CLINICAL_STRATA, DEATH_CLINICAL_STRATA
from autumn.models.covid_19.model import preprocess
from autumn.models.covid_19.preprocess.case_detection import build_detected_proportion_func
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.tools import inputs
from autumn.tools.utils.utils import (
    apply_odds_ratio_to_proportion,
    repeat_list_elements,
)
NUM_AGE_STRATA = 16


"""
General data processing functions.
"""


def apply_odds_ratio_to_multiple_proportions(props, adjuster):
    return [apply_odds_ratio_to_proportion(i_prop, adjuster) for i_prop in props]


def subdivide_props(base_props: np.ndarray, split_props: np.ndarray):
    """
    Split an array (base_array) of proportions into two arrays (split_arr, complement_arr) according to the split
    proportions provided (split_prop).
    """
    split_arr = base_props * split_props
    complement_arr = base_props * (1. - split_props)
    return split_arr, complement_arr


"""
Used in multiple sections.
"""


def get_absolute_strata_proportions(symptomatic_props, icu_props, hospital_props):
    """
    Returns the proportion of people in each clinical stratum.
    ie: Given all the people people who are infected, what proportion are in each strata?
    Each of these are stratified into 16 age groups 0-75+
    """

    # Determine the absolute proportion of early exposed who become symptomatic as opposed to asymptomatic.
    sympt, non_sympt = subdivide_props(1., np.array(symptomatic_props))

    # Determine the absolute proportion of the symptomatics who become hospitalised vs not.
    sympt_hospital, sympt_non_hospital = subdivide_props(sympt, np.array(hospital_props))  # sympt_non_hospital not used

    # Determine the absolute proportion of those hospitalised who go to ICU versus those that don't.
    sympt_hospital_icu, sympt_hospital_non_icu = subdivide_props(sympt_hospital, icu_props)

    return {
        Clinical.NON_SYMPT: non_sympt,
        "sympt": sympt,  # Not a stratum, all those symptomatic
        "hospital": sympt_hospital,  # Not a stratum, all those admitted to hospital
        Clinical.ICU: sympt_hospital_icu,
        Clinical.HOSPITAL_NON_ICU: sympt_hospital_non_icu,
    }


"""
Entry-related functions.
"""


def get_abs_prop_isolated_factory(age_idx, abs_props, prop_detect_among_sympt_func):
    """
    Returns the absolute proportion of infected becoming isolated at home.
    Isolated people are those who are detected but not sent to hospital.
    """

    def get_abs_prop_isolated(time):
        abs_prop_detected = abs_props["sympt"][age_idx] * prop_detect_among_sympt_func(time)
        abs_prop_isolated = abs_prop_detected - abs_props["hospital"][age_idx]

        # If more people go to hospital than are detected, assume that no one is being detected/isolated
        if abs_prop_isolated < 0:
            abs_prop_isolated = 0.

        return abs_prop_isolated

    return get_abs_prop_isolated


def get_abs_prop_sympt_non_hospital_factory(age_idx, abs_props, get_abs_prop_isolated_func):
    """
    Returns the absolute proportion of infected not entering the hospital.
    This also does not count people who are isolated/detected.
    """
    return lambda t: abs_props["sympt"][age_idx] - abs_props["hospital"][age_idx] - get_abs_prop_isolated_func(t)


def get_entry_adjustments(abs_props, get_detected_proportion, early_rate):
    """
    Gather together all the entry adjustments, two of which are functions of time and three are constant over time.
    """
    adjustments = {}
    for age_idx, agegroup in enumerate(AGEGROUP_STRATA):

        # Function-based flow rates

        # Get isolated rate for overwriting
        get_abs_prop_isolated = get_abs_prop_isolated_factory(age_idx, abs_props, get_detected_proportion)

        def isolate_flow_rate(t, func=get_abs_prop_isolated):  # Function must be "bound" within loop
            return func(t) * early_rate

        # Get sympt non-hospital rate for overwriting
        abs_prop_sympt_non_hospital_func = get_abs_prop_sympt_non_hospital_factory(
            age_idx, abs_props, get_abs_prop_isolated
        )

        def sympt_non_hosp_rate(t, func=abs_prop_sympt_non_hospital_func):  # Function must be "bound" within loop
            return func(t) * early_rate

        # Constant flow rates
        clinical_non_sympt_rate = abs_props[Clinical.NON_SYMPT][age_idx] * early_rate
        hospital_non_icu_rate = abs_props[Clinical.HOSPITAL_NON_ICU][age_idx] * early_rate
        clinical_icu_rate = abs_props[Clinical.ICU][age_idx] * early_rate

        # Age-specific adjustments object
        adjustments[agegroup] = {
            Clinical.NON_SYMPT: Overwrite(clinical_non_sympt_rate),
            Clinical.SYMPT_NON_HOSPITAL: Overwrite(sympt_non_hosp_rate),
            Clinical.SYMPT_ISOLATE: Overwrite(isolate_flow_rate),
            Clinical.HOSPITAL_NON_ICU: Overwrite(hospital_non_icu_rate),
            Clinical.ICU: Overwrite(clinical_icu_rate),
        }

    return adjustments


"""
Death/exit-related functions
"""


def convert_ifr_agegroups(raw_ifr_props, iso3, pop_region, pop_year):
    """
    Converts the IFRs from the age groups they were provided in to the ones needed for the model.
    """

    # Work out the proportion of 80+ years old among the 75+ population
    elderly_populations = inputs.get_population_by_agegroup([0, 75, 80], iso3, pop_region, year=pop_year)
    prop_over_80 = elderly_populations[2] / sum(elderly_populations[1:])

    # Calculate 75+ age bracket as weighted average between 75-79 and half 80+
    return [
        *raw_ifr_props[:-2],
        raw_ifr_props[-1] * prop_over_80 + raw_ifr_props[-2] * (1. - prop_over_80)
    ]


def get_absolute_death_proportions(abs_props, infection_fatality_props, icu_mortality_prop):
    """
    Calculate death proportions: find where the absolute number of deaths accrue.
    Represents the number of people in a strata who die given the total number of people infected.
    """
    abs_death_props = {stratum: np.zeros(NUM_AGE_STRATA) for stratum in DEATH_CLINICAL_STRATA}
    for age_idx in range(NUM_AGE_STRATA):
        target_ifr_prop = infection_fatality_props[age_idx]

        # Maximum deaths that could be assigned to each of the death strata
        max_asympt_prop_allowed = abs_props[Clinical.NON_SYMPT][age_idx]
        max_hosp_prop_allowed = abs_props[Clinical.HOSPITAL_NON_ICU][age_idx]
        max_icu_prop_allowed = abs_props[Clinical.ICU][age_idx] * icu_mortality_prop
        max_total_death_prop_allowed = max_asympt_prop_allowed + max_hosp_prop_allowed + max_icu_prop_allowed

        # Make sure there are enough asymptomatic and hospitalised proportions to fill the IFR, discard some if not
        # This would never happen, because the asymptomatic prop would always be far beyond the IFR, but just in case
        ifr_prop = min(max_total_death_prop_allowed, target_ifr_prop)

        # Absolute proportion of all patients dying in ICU
        abs_death_props[Clinical.ICU][age_idx] = min(max_icu_prop_allowed, ifr_prop)

        # Absolute proportion of all patients dying in hospital, excluding ICU
        target_hospital_mortality = max(ifr_prop - abs_death_props[Clinical.ICU][age_idx], 0.)
        abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx] = min(target_hospital_mortality, max_hosp_prop_allowed)

        # Absolute proportion of all patients dying out of hospital
        target_asympt_mortality = \
            ifr_prop - abs_death_props[Clinical.ICU][age_idx] - abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx]
        abs_death_props[Clinical.NON_SYMPT][age_idx] = max(0., target_asympt_mortality)

        # Double-check everything sums up properly - it should be impossible for this to fail, but just being paranoid
        allowed_rounding_error = 6
        total_death_props = sum([abs_death_props[stratum][age_idx] for stratum in DEATH_CLINICAL_STRATA])
        assert round(total_death_props, allowed_rounding_error) == round(ifr_prop, allowed_rounding_error)

    return abs_death_props


def get_rate_adjustments(rates):
    """
    Apply adjusted infection death and recovery rates for asymptomatic and hospitalised patients (ICU and non-ICU)
    Death and non-death progression between the infectious compartments towards the recovered compartment
    """
    rate_adjustments = {}
    for i_age, age_group in enumerate(AGEGROUP_STRATA):
        rate_adjustments[age_group] = {
            stratum: Overwrite(
                rates[stratum][i_age] if stratum in rates else None
            )
            for stratum in CLINICAL_STRATA
        }
    return rate_adjustments


"""
Master function
"""


def get_all_adjustments(
    clinical_params,
    country,
    pop,
    raw_ifr_props,
    sojourn,
    testing_to_detection,
    case_detection,
    ifr_adjuster,
    symptomatic_adjuster,
    hospital_adjuster,
    top_bracket_overwrite=None,
):

    """
    Preliminary processing.
    """

    # Work out the proportions of all infected people entering each clinical stratum.
    symptomatic_props = repeat_list_elements(2, clinical_params.props.symptomatic.props)  # Convert to model age groups

    # Apply odds ratio adjusters to proportions needing to be adjusted.
    hospital_props = apply_odds_ratio_to_multiple_proportions(clinical_params.props.hospital.props, hospital_adjuster)
    adjusted_symptomatic_props = apply_odds_ratio_to_multiple_proportions(symptomatic_props, symptomatic_adjuster)

    # Get the proportions that don't vary over time, not all of these
    abs_props = get_absolute_strata_proportions(adjusted_symptomatic_props, clinical_params.icu_prop, hospital_props)

    # Work out all the relevant sojourn times and the associated total rates at which they exit the compartments.
    # We assume everyone who dies does so at the end of their time in the "late active" compartment.
    # Later, we split the flow rate out of "late active" into a death or recovery flow, based on proportion dying.
    compartment_periods = preprocess.compartments.calc_compartment_periods(sojourn)

    within_early_exposed = 1. / compartment_periods[Compartment.EARLY_EXPOSED]

    within_early_rates = {
        Clinical.HOSPITAL_NON_ICU: 1. / sojourn.compartment_periods["hospital_early"],
        Clinical.ICU: 1. / sojourn.compartment_periods["icu_early"]
    }

    within_late_rates = {
        Clinical.NON_SYMPT: 1. / compartment_periods["late_active"],
        Clinical.HOSPITAL_NON_ICU: 1. / sojourn.compartment_periods["hospital_late"],
        Clinical.ICU: 1. / sojourn.compartment_periods["icu_late"],
    }

    """
    Entry adjustments.
    """
    get_detected_proportion = build_detected_proportion_func(
        AGEGROUP_STRATA, country, pop, testing_to_detection, case_detection
    )
    entry_adjs = get_entry_adjustments(abs_props, get_detected_proportion, within_early_exposed)

    """
    Progress adjustments.
    """
    progress_adjs = {}
    for stratum in CLINICAL_STRATA:
        rate = Overwrite(within_early_rates[stratum]) if stratum in within_early_rates else None
        progress_adjs.update({stratum: rate})

    """
    Death and recovery adjustments.
    
    Process the IFR parameters, i.e. the proportion of people in age group who die, given the number infected
    Numerator: deaths, denominator: all infected
    """

    # Scale raw IFR values according to adjuster parameter
    adjusted_ifr_props = apply_odds_ratio_to_multiple_proportions(raw_ifr_props, ifr_adjuster)

    # Convert from provided age groups to model age groups
    final_ifr_props = convert_ifr_agegroups(adjusted_ifr_props, country.iso3, pop.region, pop.year)

    # Over-write the oldest age bracket, if that's what's being done in the model
    if top_bracket_overwrite:
        final_ifr_props[-1] = top_bracket_overwrite

    """
    Work out the proportion of people entering each stratum who die in that stratum (for each age group).
    """

    # The proportion of all people who enter each stratum.
    # Numerator: people entering stratum, denominator: everyone - is abs_props (already defined)

    # The proportion of people entering each stratum who die in that stratum
    # Numerator: deaths in stratum, denominator: everyone
    abs_death_props = get_absolute_death_proportions(
        abs_props, final_ifr_props, clinical_params.icu_mortality_prop
    )

    # The resulting proportion
    # Numerator: deaths in stratum, denominator: people entering stratum
    relative_death_props = {
        stratum: np.array(abs_death_props[stratum]) / np.array(abs_props[stratum]) for
        stratum in DEATH_CLINICAL_STRATA
    }

    """
    Find death and survival rates from death proportions and sojourn times.
    """
    death_rates = {
        stratum: relative_death_props[stratum] * within_late_rates[stratum]
        for stratum in DEATH_CLINICAL_STRATA
    }
    # The survival proportions are the complement of the death proportions.
    survival_rates = {
        stratum: (1. - relative_death_props[stratum]) * within_late_rates[stratum]
        for stratum in DEATH_CLINICAL_STRATA
    }

    """
    Convert to summer parameter overwrite objects.
    """
    death_adjs = get_rate_adjustments(death_rates)
    recovery_adjs = get_rate_adjustments(survival_rates)

    """
    Return all the adjustments
    """
    return entry_adjs, death_adjs, progress_adjs, recovery_adjs, abs_props, get_detected_proportion
