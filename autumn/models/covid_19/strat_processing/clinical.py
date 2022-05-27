import numpy as np
from typing import List, Dict

from summer import Overwrite
from summer.adjust import AdjustmentComponent, AdjustmentSystem

from autumn.models.covid_19.constants import (
    Clinical, Compartment, FIXED_STRATA, INFECTIOUSNESS_ONSET, INFECT_DEATH, PROGRESS, RECOVERY
)
from autumn.models.covid_19.utils import calc_compartment_periods
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.parameters import Country, Population, Sojourn, ClinicalStratification
from autumn.core.inputs.demography.queries import convert_ifr_agegroups
from autumn.core.utils.utils import apply_odds_ratio_to_props, subdivide_props
from autumn.models.covid_19.stratifications.clinical import CLINICAL_STRATA


ALLOWED_ROUNDING_ERROR = 6

"""
Utility function.
"""


def get_rate_adjustments(rates: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """
    This function simply converts the values in nested dictionaries into Overwrite objects for use by the summer model
    object - where applicable, and None where not.

    Args:
        rates: Similarly structured to the output dictionary, but populated with the raw numbers only

    Returns:
        The collated adjustments by age group and then clinical stratum as Overwrite objects or None

    """

    rate_adjustments = {}
    for i_age, age_group in enumerate(AGEGROUP_STRATA):
        rate_adjustments[age_group] = {}
        for stratum in CLINICAL_STRATA:
            adjustment = rates[stratum][i_age] if stratum in rates else None
            rate_adjustments[age_group][stratum] = Overwrite(adjustment)

    return rate_adjustments


"""
Entry-related functions/objects.
"""


def get_abs_prop_isolated(prop_sympt: np.ndarray, prop_hosp: np.ndarray, cdr: float) -> np.ndarray:
    """
    Returns the absolute proportion of infected who will be isolated at home (detected but not hospitalised).
    A floor of zero is placed on this quantity because the CDR could be less than the hospitalised proportion.
    
    Args:
        prop_sympt: Proportion of all infections resulting in symptoms
        prop_hosp: Proportion of symptomatic cases hospitalised
        cdr: The proportion of symptomatic cases detected

    Returns:
        The proportion of all new infections that are isolated (i.e. entering the third clinical stratum)

    """

    target_prop_detected = prop_sympt * cdr
    return np.maximum(0., target_prop_detected - prop_hosp)


class AbsRateIsolatedSystem(AdjustmentSystem):
    """
    Returns the absolute rate of infected becoming isolated at home - the third clinical stratum of the clinical
    stratification (excludes those that are admitted to hospital).
    We need the proportion of all cases that are symptomatic and the proportion of all cases that are hospitalised to
    work this out.

    Attributes:
        early_rate: Rate by which all output is scaled
        prop_sympt: Proportion of all cases that are symptomatic
        prop_hosp: Proportion of all cases that are hospitalised

    """

    def __init__(self, early_rate: float):
        self.early_rate = early_rate
        self.prop_sympt = None
        self.prop_hosp = None

    def prepare_to_run(self, component_data: List[dict]):
        """
        Compile all components into arrays used for faster computation.

        Args:
            component_data: List containing the required data described in Attributes above

        """

        self.prop_sympt = np.empty_like(component_data, dtype=float)
        self.prop_hosp = np.empty_like(component_data, dtype=float)

        for i, component in enumerate(component_data):
            self.prop_sympt[i] = component["proportion_sympt"]
            self.prop_hosp[i] = component["proportion_hosp"]

    def get_weights_at_time(self, time: float, computed_values: dict) -> np.ndarray:
        """
        Get the final value for the absolute proportion isolated needed at run-time.

        Args:
            Standard arguments for this summer method

        Returns:
            The final calculated value

        """

        return get_abs_prop_isolated(self.prop_sympt, self.prop_hosp, computed_values["cdr"]) * self.early_rate


class AbsPropSymptNonHospSystem(AdjustmentSystem):
    """
    Returns the absolute rate of infected becoming symptomatic, including both detected and undetected symptomatic.

    Attributes:
        early_rate: Rate by which all output is scaled

    """

    def __init__(self, early_rate: float):
        self.early_rate = early_rate
        self.prop_sympt = None
        self.prop_hosp = None

    def prepare_to_run(self, component_data: List[dict]):
        """
        Compile all components into arrays used for faster computation.

        Args:
            component_data: List containing the required data described in Attributes above

        """

        self.prop_sympt = np.empty_like(component_data, dtype=float)
        self.prop_hosp = np.empty_like(component_data, dtype=float)

        for i, component in enumerate(component_data):
            self.prop_sympt[i] = component["proportion_sympt"]
            self.prop_hosp[i] = component["proportion_hosp"]

    def get_weights_at_time(self, time: float, computed_values: dict) -> np.ndarray:
        """
        Get the final value for the absolute proportion isolated needed at run-time.
        Here we have to calculate the absolute proportion detected and subtract it from the absolute proportion of
        patients symptomatic but not hospitalised.

        Args:
            Standard arguments for this summer method

        Returns:
            The final calculated value

        """

        prop_isolated = get_abs_prop_isolated(self.prop_sympt, self.prop_hosp, computed_values["cdr"])
        prop_sympt_non_hospital = self.prop_sympt - self.prop_hosp - prop_isolated
        return prop_sympt_non_hospital * self.early_rate


def get_entry_adjustments(abs_props: dict, early_rate: float) -> Dict[str, dict]:
    """
    Gather together all the entry adjustments, two of which are functions of time and three are constant over time.

    Args:
        abs_props: The absolute proportions entering certain categories defined in get_fixed_abs_strata_props above
        early_rate: The rate of transition into the compartments that are split by clinical status

    Returns:
        The adjustments (all as Overwrites) in a summer-ready format

    """

    adj_values, adjustments = {}, {}
    for age_idx, agegroup in enumerate(AGEGROUP_STRATA):
        adj_values[agegroup], adjustments[agegroup] = {}, {}

        # Get time-varying symptomatic isolated non-community rate
        key_props = {"proportion_sympt": abs_props["sympt"][age_idx], "proportion_hosp": abs_props["hospital"][age_idx]}

        # Variable flow rates, AdjustmentComponent contains data for the 'isolated' and 'sympt_non_hosp' systems
        adj_values[agegroup][Clinical.SYMPT_ISOLATE] = AdjustmentComponent(system="isolated", data=key_props)
        adj_values[agegroup][Clinical.SYMPT_NON_HOSPITAL] = AdjustmentComponent(system="sympt_non_hosp", data=key_props)

        # Calculate the constant flow rates
        adj_values[agegroup].update({stratum: abs_props[stratum][age_idx] * early_rate for stratum in FIXED_STRATA})

        # Update the summer adjustments object
        adjustments[agegroup] = {stratum: Overwrite(adj_values[agegroup][stratum]) for stratum in CLINICAL_STRATA}

    return adjustments


def get_fixed_abs_strata_props(
        sympt_props: List[float], icu_prop: float, hosp_props: List[float]
) -> Dict[str, np.ndarray]:
    """
    Returns various proportions relevant to calculating the distribution moving into the clinical stratum.
    Note that this is not for each clinical stratum and that is not what the keys of the returned dictionary are
    (even though there are five of them).
    Two of these are stratified into 16 age groups from zero to 75+.

    Args:
        sympt_props: Proportion of all infections that are symptomatic (by age group)
        icu_prop: Proportion of hospitalised patients who are admitted to ICU (not age-strtified)
        hosp_props: Proportion of symptomatic infections admitted to hospital

    Returns:
        The results of these calculations packaged up in the format expected by the other functions
            (both entry and death/recovery splitting functions)

    """

    # Absolute proportion of early exposed who become symptomatic, rather than asymptomatic
    sympt, non_sympt = subdivide_props(np.array((1.,) * len(AGEGROUP_STRATA)), np.array(sympt_props))

    # Absolute proportion of all infections who become hospitalised (sympt_non_hospital not needed here)
    sympt_hospital, _ = subdivide_props(sympt, np.array(hosp_props))

    # Absolute proportion of those hospitalised who go to ICU versus those that don't
    sympt_hospital_icu, sympt_hospital_non_icu = subdivide_props(sympt_hospital, icu_prop)

    return {
        Clinical.NON_SYMPT: non_sympt,
        "sympt": sympt,  # Not a single stratum, all those symptomatic (strata 2 to 5)
        "hospital": sympt_hospital,  # Not a stratum, all those admitted to hospital (strata 4 and 5)
        Clinical.ICU: sympt_hospital_icu,
        Clinical.HOSPITAL_NON_ICU: sympt_hospital_non_icu,
    }


"""
Recovery and death related.
"""


def get_absolute_death_proportions(abs_props: dict, infection_fatality_props: list, icu_mortality_prop: float) -> dict:
    """
    Calculate death proportions: find where the absolute number of deaths accrue.
    Represents the number of people in a strata who die given the total number of people infected.

    Args:
        abs_props: The proportions calculated by get_fixed_abs_strata_props
        infection_fatality_props: The final IFR proportions after any adjustments have been made
        icu_mortality_prop: The proportion of hospital admissions dying in ICU

    Returns:

    """

    abs_death_props = {stratum: np.zeros(len(AGEGROUP_STRATA)) for stratum in FIXED_STRATA}
    for age_idx in range(len(AGEGROUP_STRATA)):
        target_ifr_prop = infection_fatality_props[age_idx]

        # Maximum deaths that can be assigned to each of the death strata based on the absolute proportions entering...
        max_asympt_prop_allowed = abs_props[Clinical.NON_SYMPT][age_idx]
        max_hosp_prop_allowed = abs_props[Clinical.HOSPITAL_NON_ICU][age_idx]
        max_icu_prop_allowed = abs_props[Clinical.ICU][age_idx] * icu_mortality_prop  # Less than the total stratum size

        # ... and maximum overall allowable
        max_total_death_prop_allowed = max_asympt_prop_allowed + max_hosp_prop_allowed + max_icu_prop_allowed

        # Make sure there are enough asymptomatic and hospitalised proportions to fill the IFR, discard some if not
        # (This would never happen, because the asymptomatic prop would always be far beyond the IFR, but just in case)
        ifr_prop = min(max_total_death_prop_allowed, target_ifr_prop)

        # Absolute proportion of all patients dying in ICU
        abs_death_props[Clinical.ICU][age_idx] = min(max_icu_prop_allowed, ifr_prop)

        # Absolute proportion of all patients dying in hospital, excluding ICU
        target_hospital_mortality = max(ifr_prop - abs_death_props[Clinical.ICU][age_idx], 0.)
        abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx] = min(target_hospital_mortality, max_hosp_prop_allowed)

        # Absolute proportion of all patients dying out of hospital
        abs_death_prop_hosp = abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx]
        target_asympt_mortality = ifr_prop - abs_death_props[Clinical.ICU][age_idx] - abs_death_prop_hosp
        abs_death_props[Clinical.NON_SYMPT][age_idx] = max(0., target_asympt_mortality)

        # Double-check everything sums up properly - it should be impossible for this to fail
        total_death_props = sum([abs_death_props[stratum][age_idx] for stratum in FIXED_STRATA])
        assert round(total_death_props, ALLOWED_ROUNDING_ERROR) == round(ifr_prop, ALLOWED_ROUNDING_ERROR)

    return abs_death_props


"""
Master function.
"""


def get_all_adjustments(
        clinical_params: ClinicalStratification, country: Country, pop: Population, raw_ifr_props: list,
        sojourn: Sojourn, sympt_adjuster: float, hospital_adjuster: float, ifr_adjuster: float,
) -> Dict[str, dict]:
    """
    Get the clinical adjustments either for the direct stratification, or as overwrites for the clinical strata of the
    history or vaccination stratifications.
    Collates these together into one object, which has a logical structure, although not the fully summer-ready
    adjustments structure.

    Args:
        clinical_params: Parameters directly defining the clinical stratification
        country: Country we are working with
        pop: Other country-related parameters
        raw_ifr_props: The IFRs unadjusted for anything being done in calibration or further stratifications
        sojourn: The compartment sojourn times for this model
        sympt_adjuster: Any adjustment to all the age-specific symptomatic proportions
        hospital_adjuster: Any adjustment to all the age-specific proportion of symptomatic persons hospitalised
        ifr_adjuster: Any adjustment to all the age-specific IFRs

    Returns:
        A dictionary structured with first tier of keys being the flows to be modified, the second being the age groups
        (if the flow has differences by agegroup), the second being the clinical stratum (with all five strata
        represented)

    """

    """
    Preliminary processing.
    """

    all_adjustments = {}

    # Apply odds ratio adjusters to proportions needing to be adjusted
    hospital_props = apply_odds_ratio_to_props(clinical_params.props.hospital.props, hospital_adjuster)
    adjusted_symptomatic_props = apply_odds_ratio_to_props(clinical_params.props.symptomatic.props, sympt_adjuster)
    adjusted_ifr_props = apply_odds_ratio_to_props(raw_ifr_props, ifr_adjuster)

    # Get the proportions that are fixed over time, five keys, but these do not map directly to the five clinical strata
    abs_props = get_fixed_abs_strata_props(adjusted_symptomatic_props, clinical_params.icu_prop, hospital_props)

    # Work out all the relevant sojourn times and the associated total rates at which they exit the compartments
    compartment_periods = calc_compartment_periods(sojourn)

    """
    Entry adjustments - complicated process for entering the compartments that are stratified by clinical status.
    (See get_entry_adjustments above for details)
    """

    within_early_exposed = 1. / compartment_periods[Compartment.EARLY_EXPOSED]
    all_adjustments[INFECTIOUSNESS_ONSET] = get_entry_adjustments(abs_props, within_early_exposed)

    """
    Progression adjustments - simpler process that just depends on hospital admission status.
    """

    hosp_rate = 1. / sojourn.compartment_periods["hospital_early"]
    icu_rate = 1. / sojourn.compartment_periods["icu_early"]
    within_early_rates = {Clinical.HOSPITAL_NON_ICU: hosp_rate, Clinical.ICU: icu_rate}
    all_adjustments[PROGRESS] = {}
    for stratum in CLINICAL_STRATA:
        progress_rate = Overwrite(within_early_rates[stratum]) if stratum in within_early_rates else None
        all_adjustments[PROGRESS].update({stratum: progress_rate})

    """
    Death and recovery adjustments - also more complicated to work out the absolute proportion of each age and stratum-
    specific popoulation that will die versus recover.
    (See get_absolute_death_proportions above)
    """

    within_late_rates = {
        Clinical.NON_SYMPT: 1. / compartment_periods["late_active"],
        Clinical.HOSPITAL_NON_ICU: 1. / sojourn.compartment_periods["hospital_late"],
        Clinical.ICU: 1. / sojourn.compartment_periods["icu_late"],
    }

    # Convert from provided age groups to model age groups
    final_ifr_props = convert_ifr_agegroups(adjusted_ifr_props, country.iso3, pop.region, pop.year)

    # The proportion of those entering each stratum who die. Numerator: deaths in stratum, denominator: everyone
    abs_death_props = get_absolute_death_proportions(abs_props, final_ifr_props, clinical_params.icu_mortality_prop)

    # The resulting proportion. Numerator: deaths in stratum, denominator: people entering stratum
    # (This could be over-written here by the probability of death given ICU or hospital admission if preferred)
    rel_death_props = {strat: np.array(abs_death_props[strat]) / np.array(abs_props[strat]) for strat in FIXED_STRATA}

    # Convert to rates and then to summer Overwrite objects
    death_rates = {strat: rel_death_props[strat] * within_late_rates[strat] for strat in FIXED_STRATA}
    survival_rates = {strat: (1. - rel_death_props[strat]) * within_late_rates[strat] for strat in FIXED_STRATA}
    all_adjustments[INFECT_DEATH] = get_rate_adjustments(death_rates)
    all_adjustments[RECOVERY] = get_rate_adjustments(survival_rates)

    return all_adjustments

