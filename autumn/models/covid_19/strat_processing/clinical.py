import numpy as np
from typing import List, Dict

from summer import Overwrite, Stratification
from summer.adjust import AdjustmentComponent, AdjustmentSystem

from autumn.models.covid_19.constants import Clinical, Compartment, FIXED_STRATA, AGE_CLINICAL_TRANSITIONS
from autumn.models.covid_19.utils import calc_compartment_periods
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.parameters import Country, Population, Sojourn, ClinicalStratification
from autumn.tools.inputs.demography.queries import convert_ifr_agegroups
from autumn.tools.utils.utils import apply_odds_ratio_to_props, subdivide_props
from autumn.models.covid_19.constants import INFECTIOUSNESS_ONSET, INFECT_DEATH, PROGRESS, RECOVERY
from autumn.models.covid_19.stratifications.clinical import CLINICAL_STRATA


ALLOWED_ROUNDING_ERROR = 6


def get_abs_prop_isolated(proportion_sympt: np.ndarray, proportion_hosp: np.ndarray, cdr: float) -> np.ndarray:
    """
    Returns the absolute proportion of infected who will be isolated at home.
    Isolated people are those who are detected but not sent to hospital.
    """

    target_prop_detected = proportion_sympt * cdr
    return np.maximum(0., target_prop_detected - proportion_hosp)


class AbsRateIsolatedSystem(AdjustmentSystem):
    """
    Returns the absolute rate of infected becoming isolated at home - the third clinical stratum of the clinical
    stratification.
    Doesn't include those that are admitted to hospital.
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


def get_rate_adjustments(rates: dict) -> Dict[str, dict]:
    """
    This function just converts the values in nested dictionaries into Overwrite objects for use by the summer model
    object - where applicable, and None where not.
    """

    rate_adjustments = {}
    for i_age, age_group in enumerate(AGEGROUP_STRATA):
        rate_adjustments[age_group] = {}
        for stratum in CLINICAL_STRATA:
            adjustment = Overwrite(rates[stratum][i_age] if stratum in rates else None)
            rate_adjustments[age_group][stratum] = adjustment

    return rate_adjustments


def get_fixed_abs_strata_props(
        symptomatic_props: list, icu_props: float, hospital_props: list
) -> Dict[str, np.ndarray]:
    """
    Returns the proportion of people in each clinical stratum.
    ie: Given all the people people who are infected, what proportion are in each strata?
    Each of these are stratified into 16 age groups 0-75+.
    This is distinct from the adjuster processes above, that track the strata proportions that vary over time (strata 2
    and 3).
    """

    # Absolute proportion of early exposed who become symptomatic, rather than asymptomatic
    sympt, non_sympt = subdivide_props(np.array((1.,) * len(AGEGROUP_STRATA)), np.array(symptomatic_props))

    # Absolute proportion of symptomatics who become hospitalised (sympt_non_hospital not needed here)
    sympt_hospital, _ = subdivide_props(sympt, np.array(hospital_props))

    # Absolute proportion of those hospitalised who go to ICU versus those that don't
    sympt_hospital_icu, sympt_hospital_non_icu = subdivide_props(sympt_hospital, icu_props)

    return {
        Clinical.NON_SYMPT: non_sympt,
        "sympt": sympt,  # Not a single stratum, all those symptomatic (strata 2 to 5)
        "hospital": sympt_hospital,  # Not a stratum, all those admitted to hospital (strata 4 and 5)
        Clinical.ICU: sympt_hospital_icu,
        Clinical.HOSPITAL_NON_ICU: sympt_hospital_non_icu,
    }


def get_entry_adjustments(abs_props: dict, early_rate: float) -> Dict[str, dict]:
    """
    Gather together all the entry adjustments, two of which are functions of time and three are constant over time.
    """

    adjustments = {}
    for age_idx, agegroup in enumerate(AGEGROUP_STRATA):
        adjustments[agegroup] = {}

        # Get time-varying symptomatic isolated non-community rate
        prop_sympt = abs_props["sympt"][age_idx]
        prop_hosp = abs_props["hospital"][age_idx]
        proportions = {"proportion_sympt": prop_sympt, "proportion_hosp": prop_hosp}

        # Variable flow rates, AdjustmentComponent contains data for the 'isolated' and 'sympt_non_hosp' systems
        adj_isolated_component = AdjustmentComponent(system="isolated", data=proportions)
        adjustments[agegroup][Clinical.SYMPT_ISOLATE] = adj_isolated_component
        adj_sympt_non_hospital_component = AdjustmentComponent(system="sympt_non_hosp", data=proportions)
        adjustments[agegroup][Clinical.SYMPT_NON_HOSPITAL] = adj_sympt_non_hospital_component

        # Constant flow rates
        constant_strata_adjs = {stratum: abs_props[stratum][age_idx] * early_rate for stratum in FIXED_STRATA}
        adjustments[agegroup].update(constant_strata_adjs)

        # Update the summer adjustments object
        adjustments[agegroup] = {stratum: Overwrite(adjustments[agegroup][stratum]) for stratum in CLINICAL_STRATA}

    return adjustments


def get_absolute_death_proportions(abs_props: dict, infection_fatality_props: list, icu_mortality_prop: float) -> dict:
    """
    Calculate death proportions: find where the absolute number of deaths accrue.
    Represents the number of people in a strata who die given the total number of people infected.
    """

    abs_death_props = {stratum: np.zeros(len(AGEGROUP_STRATA)) for stratum in FIXED_STRATA}
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

        # Double-check everything sums up properly - it should be impossible for this to fail
        total_death_props = sum([abs_death_props[stratum][age_idx] for stratum in FIXED_STRATA])
        assert round(total_death_props, ALLOWED_ROUNDING_ERROR) == round(ifr_prop, ALLOWED_ROUNDING_ERROR)

    return abs_death_props


def get_all_adjustments(
        clinical_params: ClinicalStratification, country: Country, pop: Population, raw_ifr_props: list,
        sojourn: Sojourn, sympt_adjuster: float, hospital_adjuster: float, ifr_adjuster: float,
) -> Dict[str, dict]:
    """
    Preliminary processing.
    """

    all_adjustments = {}

    # Apply odds ratio adjusters to proportions needing to be adjusted
    hospital_props = apply_odds_ratio_to_props(clinical_params.props.hospital.props, hospital_adjuster)
    adjusted_symptomatic_props = apply_odds_ratio_to_props(clinical_params.props.symptomatic.props, sympt_adjuster)

    # Get the proportions that don't vary over time, not all of these are actually clinical strata
    abs_props = get_fixed_abs_strata_props(adjusted_symptomatic_props, clinical_params.icu_prop, hospital_props)

    # Work out all the relevant sojourn times and the associated total rates at which they exit the compartments
    compartment_periods = calc_compartment_periods(sojourn)

    """
    Entry adjustments.
    """

    within_early_exposed = 1. / compartment_periods[Compartment.EARLY_EXPOSED]
    all_adjustments[INFECTIOUSNESS_ONSET] = get_entry_adjustments(abs_props, within_early_exposed)

    """
    Progression adjustments.
    """

    within_early_rates = {
        Clinical.HOSPITAL_NON_ICU: 1. / sojourn.compartment_periods["hospital_early"],
        Clinical.ICU: 1. / sojourn.compartment_periods["icu_early"]
    }
    all_adjustments[PROGRESS] = {}
    for stratum in CLINICAL_STRATA:
        progress_rate = Overwrite(within_early_rates[stratum]) if stratum in within_early_rates else None
        all_adjustments[PROGRESS].update({stratum: progress_rate})

    """
    Death and recovery adjustments.
    """

    within_late_rates = {
        Clinical.NON_SYMPT: 1. / compartment_periods["late_active"],
        Clinical.HOSPITAL_NON_ICU: 1. / sojourn.compartment_periods["hospital_late"],
        Clinical.ICU: 1. / sojourn.compartment_periods["icu_late"],
    }

    # Scale raw IFR values according to adjuster parameter
    adjusted_ifr_props = apply_odds_ratio_to_props(raw_ifr_props, ifr_adjuster)

    # Convert from provided age groups to model age groups
    final_ifr_props = convert_ifr_agegroups(adjusted_ifr_props, country.iso3, pop.region, pop.year)

    # The proportion of those entering each stratum who die. Numerator: deaths in stratum, denominator: everyone
    abs_death_props = get_absolute_death_proportions(abs_props, final_ifr_props, clinical_params.icu_mortality_prop)

    # The resulting proportion. Numerator: deaths in stratum, denominator: people entering stratum
    # This could be over-written here by the probability of death given ICU or hospital admission if preferred
    rel_death_props = {strat: np.array(abs_death_props[strat]) / np.array(abs_props[strat]) for strat in FIXED_STRATA}

    # Convert to rates and then to summer Overwrite objects
    death_rates = {strat: rel_death_props[strat] * within_late_rates[strat] for strat in FIXED_STRATA}
    survival_rates = {strat: (1. - rel_death_props[strat]) * within_late_rates[strat] for strat in FIXED_STRATA}
    all_adjustments[INFECT_DEATH] = get_rate_adjustments(death_rates)
    all_adjustments[RECOVERY] = get_rate_adjustments(survival_rates)

    return all_adjustments


def get_blank_adjustments_for_strat(transitions: list) -> Dict[str, dict]:
    """
    Provide a blank set of flow adjustments to be populated by the update_adjustments_for_strat function below.

    Args:
        transitions: All the transition flows we will be modifying through the clinical stratification process

    Returns:
        Dictionary of dictionaries of dictionaries of blank dictionaries to be populated later

    """

    flow_adjs = {}
    for agegroup in AGEGROUP_STRATA:
        flow_adjs[agegroup] = {}
        for clinical_stratum in CLINICAL_STRATA:
            flow_adjs[agegroup][clinical_stratum] = {}
            for transition in transitions:
                flow_adjs[agegroup][clinical_stratum][transition] = {}

    return flow_adjs


def update_adjustments_for_strat(strat: str, flow_adjustments: dict, adjustments: dict, voc: str):
    """
    Add the flow adjustments to the blank adjustments created above by get_blank_adjustments_for_strat.

    Args:
        strat: The current stratification that we're modifying here
        flow_adjustments: Tiered dictionary containing the adjustments
        adjustments: Adjustments in the format that they are returned by get_all_adjustments
        voc: The current VoC being considered, the VoC loop being external to this function

    """

    # Loop over the stratifications that affect these flow rates, other than VoC stratification
    for agegroup in AGEGROUP_STRATA:
        for clinical_stratum in CLINICAL_STRATA:

            # *** Note that PROGRESS is not indexed by age group
            modification = {strat: adjustments[PROGRESS][clinical_stratum]}
            flow_adjustments[voc][agegroup][clinical_stratum][PROGRESS].update(modification)

            # ... but the other transition processes are
            for transition in AGE_CLINICAL_TRANSITIONS:
                modification = {strat: adjustments[transition][agegroup][clinical_stratum]}
                flow_adjustments[voc][agegroup][clinical_stratum][transition].update(modification)


def add_clinical_adjustments_to_strat(
        strat: Stratification, flow_adjs: Dict[str, dict], unaffected_stratum: str, vocs: list
):
    """
    Add the clinical adjustments created in update_adjustments_for_strat to a stratification.

    Uses the summer method to the stratification add_flow_adjustments, that will then be applied when the stratify_with
    is called from the model object using this stratification object.

    Note:
        Whether source or dest(ination) is requested is very important and dependent on where the clinical
        stratification splits.

    Args:
        strat: The current stratification that we're modifying here
        flow_adjs: The requested adjustments created in the previous function
        unaffected_stratum: The stratum that isn't affected and takes the default parameters
        vocs: The variants of concern, that may have different severity levels

    """

    # Loop over other stratifications that may affect these parameters, i.e. age group, VoC status and clinical status
    for agegroup in AGEGROUP_STRATA:
        for voc in vocs:
            for clinical_stratum in CLINICAL_STRATA:

                # The other model strata that we want to limit these adjustments to
                working_strata = {"agegroup": agegroup, "clinical": clinical_stratum}
                voc_strat = {"strain": voc} if len(vocs) > 1 else {}
                working_strata.update(voc_strat)

                # * Onset must be dest(ination) because this is the point at which the clinical stratification splits *
                infectious_onset_adjs = flow_adjs[voc][agegroup][clinical_stratum][INFECTIOUSNESS_ONSET]
                infectious_onset_adjs[unaffected_stratum] = None
                strat.add_flow_adjustments(INFECTIOUSNESS_ONSET, infectious_onset_adjs, dest_strata=working_strata)

                # * Progress can be either source, dest(ination) or both, but infect_death and recovery must be source *
                for transition in [PROGRESS, INFECT_DEATH, RECOVERY]:
                    adjs = flow_adjs[voc][agegroup][clinical_stratum][transition]
                    adjs[unaffected_stratum] = None
                    strat.add_flow_adjustments(transition, adjs, source_strata=working_strata)
