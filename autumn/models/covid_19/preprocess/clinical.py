import numpy as np
from typing import Tuple, List

from summer import Overwrite
from summer.adjust import AdjustmentComponent, AdjustmentSystem

from autumn.models.covid_19.constants import Clinical, Compartment, DEATH_CLINICAL_STRATA
from autumn.models.covid_19.model import preprocess
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.parameters import Country, Population, Sojourn
from autumn.tools.inputs.demography.queries import convert_ifr_agegroups
from autumn.tools.utils.utils import apply_odds_ratio_to_multiple_proportions, subdivide_props
from autumn.models.covid_19.constants import INFECTIOUSNESS_ONSET, INFECT_DEATH, PROGRESS, RECOVERY
from autumn.models.covid_19.stratifications.clinical import CLINICAL_STRATA


ALLOWED_ROUNDING_ERROR = 6


def get_abs_prop_isolated(proportion_sympt, proportion_hosp, cdr) -> np.ndarray:
    """
    Returns the absolute proportion of infected becoming isolated at home.
    Isolated people are those who are detected but not sent to hospital.

    Args:
        proportion_sympt ([type]): Float or np.ndarray
        proportion_hosp ([type]): Float or np.ndarray
        cdr ([type]): Float

    Returns:
        [np.ndarray]: Output value
    """

    target_prop_detected = proportion_sympt * cdr
    return np.maximum(0., target_prop_detected - proportion_hosp)


class AbsPropIsolatedSystem(AdjustmentSystem):
    """
    Returns the absolute rate of infected becoming isolated at home.
    Isolated people are those who are detected but not sent to hospital.
    """

    def __init__(self, early_rate: float):
        """
        Initialise the system

        Args:
            early_rate (float): Rate by which all output is scaled
        """
        self.early_rate = early_rate

    def prepare_to_run(self, component_data: List[dict]):
        """
        Compile all components into arrays used for faster computation.

        Args:
            component_data (List[dict]): List containing data specific to individual flow adjusments
        """

        self.proportion_sympt = np.empty_like(component_data, dtype=float)
        self.proportion_hosp = np.empty_like(component_data, dtype=float)

        for i, component in enumerate(component_data):
            self.proportion_sympt[i] = component["proportion_sympt"]
            self.proportion_hosp[i] = component["proportion_hosp"]

    def get_weights_at_time(self, time: float, computed_values: dict) -> np.ndarray:
        cdr = computed_values["cdr"]
        return get_abs_prop_isolated(self.proportion_sympt, self.proportion_hosp, cdr) * self.early_rate


class AbsPropSymptNonHospSystem(AdjustmentSystem):
    """
    Returns the absolute rate of infected becoming symptomatic, including both detected and undetected symptomatic.
    """

    def __init__(self, early_rate: float):
        """
        Initialise the system

        Args:
            early_rate (float): Rate by which all output is scaled
        """
        self.early_rate = early_rate

    def prepare_to_run(self, component_data: List[dict]):
        """
        Compile all components into arrays used for fast computation.

        Args:
            component_data (List[dict]): List containing data specific to individual flow adjustments
        """

        self.proportion_sympt = np.empty_like(component_data, dtype=float)
        self.proportion_hosp = np.empty_like(component_data, dtype=float)

        for i, component in enumerate(component_data):
            self.proportion_sympt[i] = component["proportion_sympt"]
            self.proportion_hosp[i] = component["proportion_hosp"]

    def get_weights_at_time(self, time: float, computed_values: dict) -> np.ndarray:
        cdr = computed_values["cdr"]
        prop_isolated = get_abs_prop_isolated(self.proportion_sympt, self.proportion_hosp, cdr)
        prop_sympt_non_hospital = self.proportion_sympt - self.proportion_hosp - prop_isolated
        return prop_sympt_non_hospital * self.early_rate


def get_rate_adjustments(rates: dict) -> dict:
    """
    Apply adjusted infection death and recovery rates for asymptomatic and hospitalised patients (ICU and non-ICU)
    Death and non-death progression between the infectious compartments towards the recovered compartment
    """

    rate_adjustments = {}
    for i_age, age_group in enumerate(AGEGROUP_STRATA):
        rate_adjustments[age_group] = {
            stratum: Overwrite(rates[stratum][i_age] if stratum in rates else None) for stratum in CLINICAL_STRATA
        }
    return rate_adjustments


def get_absolute_strata_proportions(symptomatic_props: list, icu_props: float, hospital_props: list) -> dict:
    """
    Returns the proportion of people in each clinical stratum.
    ie: Given all the people people who are infected, what proportion are in each strata?
    Each of these are stratified into 16 age groups 0-75+
    """

    # Determine the absolute proportion of early exposed who become symptomatic as opposed to asymptomatic,
    # starting with total proportions of one
    sympt, non_sympt = subdivide_props(np.array((1.,) * len(AGEGROUP_STRATA)), np.array(symptomatic_props))

    # Determine the absolute proportion of the symptomatics who become hospitalised vs not
    sympt_hospital, sympt_non_hospital = subdivide_props(sympt, np.array(hospital_props))  # sympt_non_hospital not used

    # Determine the absolute proportion of those hospitalised who go to ICU versus those that don't
    sympt_hospital_icu, sympt_hospital_non_icu = subdivide_props(sympt_hospital, icu_props)

    return {
        Clinical.NON_SYMPT: non_sympt,
        "sympt": sympt,  # Not a single stratum, all those symptomatic (strata 2 to 5)
        "hospital": sympt_hospital,  # Not a stratum, all those admitted to hospital (strata 4 and 5)
        Clinical.ICU: sympt_hospital_icu,
        Clinical.HOSPITAL_NON_ICU: sympt_hospital_non_icu,
    }


def get_entry_adjustments(abs_props: dict, early_rate: float) -> dict:
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
        constant_strata_adjs = {stratum: abs_props[stratum][age_idx] * early_rate for stratum in DEATH_CLINICAL_STRATA}
        adjustments[agegroup].update(constant_strata_adjs)

        # Update the summer adjustments object
        adjustments[agegroup] = {stratum: Overwrite(adjustments[agegroup][stratum]) for stratum in CLINICAL_STRATA}

    return adjustments


def get_absolute_death_proportions(abs_props: dict, infection_fatality_props: list, icu_mortality_prop: float) -> dict:
    """
    Calculate death proportions: find where the absolute number of deaths accrue.
    Represents the number of people in a strata who die given the total number of people infected.
    """

    abs_death_props = {stratum: np.zeros(len(AGEGROUP_STRATA)) for stratum in DEATH_CLINICAL_STRATA}
    for age_idx in range(len(AGEGROUP_STRATA)):
        target_ifr_prop = infection_fatality_props[age_idx]

        # Maximum deaths that could be assigned to each of the death strata ...
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
        target_asympt_mortality = \
            ifr_prop - abs_death_props[Clinical.ICU][age_idx] - abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx]
        abs_death_props[Clinical.NON_SYMPT][age_idx] = max(0., target_asympt_mortality)

        # Double-check everything sums up properly - it should be impossible for this to fail, but just being paranoid
        total_death_props = sum([abs_death_props[stratum][age_idx] for stratum in DEATH_CLINICAL_STRATA])
        assert round(total_death_props, ALLOWED_ROUNDING_ERROR) == round(ifr_prop, ALLOWED_ROUNDING_ERROR)

    return abs_death_props


def process_ifrs(raw_ifr_props, ifr_adjuster, top_bracket_overwrite, country, pop):

    # Scale raw IFR values according to adjuster parameter
    adjusted_ifr_props = apply_odds_ratio_to_multiple_proportions(raw_ifr_props, ifr_adjuster)

    # Convert from provided age groups to model age groups
    final_ifr_props = convert_ifr_agegroups(adjusted_ifr_props, country.iso3, pop.region, pop.year)

    # Over-write the oldest age bracket, if that's what's being done in the model
    if top_bracket_overwrite:
        final_ifr_props[-1] = top_bracket_overwrite

    return final_ifr_props


"""
Master functions
"""


def get_all_adjustments(
        clinical_params, country: Country, pop: Population, raw_ifr_props: list, sojourn: Sojourn,
        ifr_adjuster: float, symptomatic_adjuster: float, hospital_adjuster: float, top_bracket_overwrite=None,
) -> Tuple[dict, dict, dict, dict, dict]:

    """
    Preliminary processing.
    """

    # Apply odds ratio adjusters to proportions needing to be adjusted
    hospital_props = apply_odds_ratio_to_multiple_proportions(
        clinical_params.props.hospital.props, hospital_adjuster
    )
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
        "isolated": AbsPropIsolatedSystem(within_early_exposed),
        "sympt_non_hosp": AbsPropSymptNonHospSystem(within_early_exposed)
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

    final_ifr_props = process_ifrs(raw_ifr_props, ifr_adjuster, top_bracket_overwrite, country, pop)

    """
    Work out the proportion of people entering each stratum who die in that stratum (for each age group).
    """

    # The proportion of all people who enter each stratum.
    # Numerator: people entering stratum, denominator: everyone - is abs_props (already defined)

    # The proportion of people entering each stratum who die in that stratum
    # Numerator: deaths in stratum, denominator: everyone
    abs_death_props = get_absolute_death_proportions(abs_props, final_ifr_props, clinical_params.icu_mortality_prop)

    # The resulting proportion
    # Numerator: deaths in stratum, denominator: people entering stratum
    relative_death_props = {
        stratum: np.array(abs_death_props[stratum]) / np.array(abs_props[stratum]) for stratum in DEATH_CLINICAL_STRATA
    }

    """
    Find death and survival rates from death proportions and sojourn times.
    """

    death_rates = {
        stratum: relative_death_props[stratum] * within_late_rates[stratum] for stratum in DEATH_CLINICAL_STRATA
    }
    # The survival proportions are the complement of the death proportions
    survival_rates = {
        stratum: (1. - relative_death_props[stratum]) * within_late_rates[stratum] for stratum in DEATH_CLINICAL_STRATA
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


"""
Dumping this here for now from the vaccination preprocess file.
"""


def add_clinical_adjustments_to_strat(
        strat, unaffected_stratum, first_modified_stratum, params, symptomatic_adjuster, hospital_adjuster,
        ifr_adjuster, top_bracket_overwrite, second_modified_stratum=None, second_sympt_adjuster=1.,
        second_hospital_adjuster=1., second_ifr_adjuster=1., second_top_bracket_overwrite=None,
):
    """
    Get all the adjustments in the same way for both the history and vaccination stratifications.
    """

    entry_adjs, death_adjs, progress_adjs, recovery_adjs, _ = get_all_adjustments(
        params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
        params.sojourn, ifr_adjuster, symptomatic_adjuster,
        hospital_adjuster, top_bracket_overwrite,
    )

    # Make these calculations for the one-dose stratum, even if this is being called by the history stratification
    if second_modified_stratum:
        second_entry_adjs, second_death_adjs, second_progress_adjs, second_recovery_adjs, _ = get_all_adjustments(
            params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
            params.sojourn, second_ifr_adjuster, second_sympt_adjuster,
            second_hospital_adjuster, second_top_bracket_overwrite,
        )

    for i_age, agegroup in enumerate(AGEGROUP_STRATA):
        for clinical_stratum in CLINICAL_STRATA:
            relevant_strata = {
                "agegroup": agegroup,
                "clinical": clinical_stratum,
            }

            # Infectiousness onset adjustments *** Must be dest
            infect_onset_adjustments = {
                unaffected_stratum: None,
                first_modified_stratum: entry_adjs[agegroup][clinical_stratum]
            }
            if second_modified_stratum:
                infect_onset_adjustments.update(
                    {second_modified_stratum: second_entry_adjs[agegroup][clinical_stratum]}
                )
            strat.add_flow_adjustments(INFECTIOUSNESS_ONSET, infect_onset_adjustments, dest_strata=relevant_strata)

            # Infect death adjustments *** Must be source
            infect_death_adjustments = {
                unaffected_stratum: None,
                first_modified_stratum: death_adjs[agegroup][clinical_stratum]
            }
            if second_modified_stratum:
                infect_death_adjustments.update(
                    {second_modified_stratum: second_death_adjs[agegroup][clinical_stratum]}
                )
            strat.add_flow_adjustments(INFECT_DEATH, infect_death_adjustments, source_strata=relevant_strata)

            # Progress adjustments *** Either source, dest or both *** Note that this isn't indexed by age group
            progress_adjustments = {
                unaffected_stratum: None,
                first_modified_stratum: progress_adjs[clinical_stratum]
            }
            if second_modified_stratum:
                progress_adjustments.update(
                    {second_modified_stratum: second_progress_adjs[clinical_stratum]}
                )
            strat.add_flow_adjustments(PROGRESS, progress_adjustments, source_strata=relevant_strata)

            # Recovery adjustments *** Must be source
            recovery_adjustments = {
                unaffected_stratum: None,
                first_modified_stratum: recovery_adjs[agegroup][clinical_stratum]
            }
            if second_modified_stratum:
                recovery_adjustments.update(
                    {second_modified_stratum: second_recovery_adjs[agegroup][clinical_stratum]}
                )
            strat.add_flow_adjustments(RECOVERY, recovery_adjustments, source_strata=relevant_strata)

    return strat
