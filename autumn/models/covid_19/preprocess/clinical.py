import numpy as np

from summer import Overwrite
from summer.adjust import AdjustmentComponent

from autumn.models.covid_19.constants import Clinical, Compartment, CLINICAL_STRATA, DEATH_CLINICAL_STRATA
from autumn.models.covid_19.preprocess import adjusterprocs
from autumn.models.covid_19.model import preprocess
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.tools import inputs
from autumn.tools.utils.utils import apply_odds_ratio_to_multiple_proportions, subdivide_props


NUM_AGE_STRATA = 16
ALLOWED_ROUNDING_ERROR = 6


"""
Used in multiple sections - both entries and deaths.
"""


def get_absolute_strata_proportions(symptomatic_props: list, icu_props: float, hospital_props: list):
    """
    Returns the proportion of people in each clinical stratum.
    ie: Given all the people people who are infected, what proportion are in each strata?
    Each of these are stratified into 16 age groups 0-75+
    """

    # Determine the absolute proportion of early exposed who become symptomatic as opposed to asymptomatic,
    # starting with total proportions of one.
    sympt, non_sympt = subdivide_props(np.array((1.,) * NUM_AGE_STRATA), np.array(symptomatic_props))

    # Determine the absolute proportion of the symptomatics who become hospitalised vs not.
    sympt_hospital, sympt_non_hospital = subdivide_props(sympt, np.array(hospital_props))  # sympt_non_hospital not used

    # Determine the absolute proportion of those hospitalised who go to ICU versus those that don't.
    sympt_hospital_icu, sympt_hospital_non_icu = subdivide_props(sympt_hospital, icu_props)

    return {
        Clinical.NON_SYMPT: non_sympt,
        "sympt": sympt,  # Not a single stratum, all those symptomatic (strata 2 to 5)
        "hospital": sympt_hospital,  # Not a stratum, all those admitted to hospital (strata 4 and 5)
        Clinical.ICU: sympt_hospital_icu,
        Clinical.HOSPITAL_NON_ICU: sympt_hospital_non_icu,
    }


def get_entry_adjustments(abs_props, early_rate):
    """
    Gather together all the entry adjustments, two of which are functions of time and three are constant over time.
    """

    adjustments = {}
    for age_idx, agegroup in enumerate(AGEGROUP_STRATA):
        adjustments[agegroup] = {}

        # Get time-varying symptomatic isolated non-community rate - this function must be "bound" within loop.
        proportion_sympt = abs_props["sympt"][age_idx]
        proportion_hosp = abs_props["hospital"][age_idx]

        proportions = {"proportion_sympt": proportion_sympt, "proportion_hosp": proportion_hosp}

        # AdjustmentComponent contains all data required for the 'isolated' system to compute the values later
        adj_isolated_component = AdjustmentComponent(system="isolated", data=proportions)
        adjustments[agegroup][Clinical.SYMPT_ISOLATE] = adj_isolated_component

        # AdjustmentComponent contains all data required for the 'sympt_non_hosp' system to compute the values later
        adj_sympt_non_hospital_component = AdjustmentComponent(system="sympt_non_hosp", data=proportions)
        adjustments[agegroup][Clinical.SYMPT_NON_HOSPITAL] = adj_sympt_non_hospital_component

        # Constant flow rates
        adjustments[agegroup].update({
            stratum: abs_props[stratum][age_idx] * early_rate for stratum in DEATH_CLINICAL_STRATA
        })

        # Update the summer adjustments object
        adjustments[agegroup] = {
            stratum: Overwrite(adjustments[agegroup][stratum]) for stratum in CLINICAL_STRATA
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

    # Calculate 75+ age bracket as weighted average between 75-79 and 80+
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

        # Maximum deaths that could be assigned to each of the death strata ...
        max_asympt_prop_allowed = abs_props[Clinical.NON_SYMPT][age_idx]
        max_hosp_prop_allowed = abs_props[Clinical.HOSPITAL_NON_ICU][age_idx]
        max_icu_prop_allowed = abs_props[Clinical.ICU][age_idx] * icu_mortality_prop  # Less than the total stratum size

        # ... and maximum overall allowable.
        max_total_death_prop_allowed = max_asympt_prop_allowed + max_hosp_prop_allowed + max_icu_prop_allowed

        # Make sure there are enough asymptomatic and hospitalised proportions to fill the IFR, discard some if not.
        # This would never happen, because the asymptomatic prop would always be far beyond the IFR, but just in case.
        ifr_prop = min(max_total_death_prop_allowed, target_ifr_prop)

        # Absolute proportion of all patients dying in ICU.
        abs_death_props[Clinical.ICU][age_idx] = min(max_icu_prop_allowed, ifr_prop)

        # Absolute proportion of all patients dying in hospital, excluding ICU.
        target_hospital_mortality = max(ifr_prop - abs_death_props[Clinical.ICU][age_idx], 0.)
        abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx] = min(target_hospital_mortality, max_hosp_prop_allowed)

        # Absolute proportion of all patients dying out of hospital.
        target_asympt_mortality = \
            ifr_prop - abs_death_props[Clinical.ICU][age_idx] - abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx]
        abs_death_props[Clinical.NON_SYMPT][age_idx] = max(0., target_asympt_mortality)

        # Double-check everything sums up properly - it should be impossible for this to fail, but just being paranoid.
        total_death_props = sum([abs_death_props[stratum][age_idx] for stratum in DEATH_CLINICAL_STRATA])
        assert round(total_death_props, ALLOWED_ROUNDING_ERROR) == round(ifr_prop, ALLOWED_ROUNDING_ERROR)

    return abs_death_props


def get_rate_adjustments(rates):
    """
    Apply adjusted infection death and recovery rates for asymptomatic and hospitalised patients (ICU and non-ICU)
    Death and non-death progression between the infectious compartments towards the recovered compartment
    """
    rate_adjustments = {}
    for i_age, age_group in enumerate(AGEGROUP_STRATA):
        rate_adjustments[age_group] = {
            stratum: Overwrite(rates[stratum][i_age] if stratum in rates else None)
            for stratum in CLINICAL_STRATA
        }
    return rate_adjustments


"""
Master function
"""


def get_all_adjustments(
        clinical_params, country, pop, raw_ifr_props, sojourn, ifr_adjuster, symptomatic_adjuster, hospital_adjuster,
        top_bracket_overwrite=None,
):

    """
    Preliminary processing.
    """

    # Apply odds ratio adjusters to proportions needing to be adjusted
    hospital_props = apply_odds_ratio_to_multiple_proportions(clinical_params.props.hospital.props, hospital_adjuster)
    adjusted_symptomatic_props = apply_odds_ratio_to_multiple_proportions(
        clinical_params.props.symptomatic.props, symptomatic_adjuster
    )

    # Get the proportions that don't vary over time, not all of these are actually clinical strata
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

    entry_adjs = get_entry_adjustments(abs_props, within_early_exposed)

    # These are the systems that will compute (in a vectorised fashion) the adjustments added using AdjustmentComponents
    adjuster_systems = {
        "isolated": adjusterprocs.AbsPropIsolatedSystem(within_early_exposed),
        "sympt_non_hosp": adjusterprocs.AbsPropSymptNonHospSystem(within_early_exposed)
    }

    """
    Progress adjustments.
    """

    progress_adjs = {}
    for stratum in CLINICAL_STRATA:
        progress_rate = Overwrite(within_early_rates[stratum]) if stratum in within_early_rates else None
        progress_adjs.update({stratum: progress_rate})

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
    Return all the adjustments.
    """

    return entry_adjs, death_adjs, progress_adjs, recovery_adjs, adjuster_systems
