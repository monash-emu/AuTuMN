from typing import List, Callable, Tuple, Union, Dict

import numpy as np
import numba

from summer import CompartmentalModel, Multiply, Stratification

from autumn.models.covid_19.constants import (
    Vaccination, INFECTION, DISEASE_COMPARTMENTS, CLINICAL_STRATA, Compartment, AGE_CLINICAL_TRANSITIONS,
    INFECTIOUSNESS_ONSET, INFECT_DEATH, PROGRESS, RECOVERY
)
from autumn.models.covid_19.parameters import Parameters, TimeSeries, VaccEffectiveness, TanhScaleup
from autumn.models.covid_19.parameters import Vaccination as VaccParams
from autumn.settings import COVID_BASE_AGEGROUPS
from autumn.models.covid_19.strat_processing.clinical import get_all_adjustments
from autumn.core.utils.utils import check_list_increasing
from autumn.model_features.curve import tanh_based_scaleup
from autumn.core.inputs.covid_lka.queries import get_lka_vac_coverage
from autumn.core.inputs.covid_mmr.queries import base_mmr_adult_vacc_doses



def find_vacc_strata(is_dosing_active: bool, waning_vacc_immunity: bool, is_boost_delay: bool) -> Tuple[list, str]:
    """
    Work out the vaccination strata to be implemented in the model currently being constructed.

    Args:
        is_dosing_active: Whether one and two doses are being simulated
        waning_vacc_immunity: Whether waning immunity is being simulated
        is_boost_delay: Whether booster doses are being simulated

    Returns:
        The vaccination strata being implemented in this model
        The stratum that waning immunity starts from, which depends on whether dosing is active

    """

    vacc_strata = [Vaccination.UNVACCINATED, Vaccination.ONE_DOSE_ONLY]
    if is_dosing_active:
        vacc_strata.append(Vaccination.VACCINATED)
        wane_origin_stratum = Vaccination.VACCINATED
    else:
        wane_origin_stratum = Vaccination.ONE_DOSE_ONLY
    if waning_vacc_immunity:
        vacc_strata.extend([Vaccination.PART_WANED, Vaccination.WANED])
    if is_boost_delay and waning_vacc_immunity:
        vacc_strata.append(Vaccination.BOOSTED)
    elif is_boost_delay and not waning_vacc_immunity:
        raise ValueError("Boosting not permitted unless waning immunity also implemented")

    return vacc_strata, wane_origin_stratum


def get_blank_adjustments_for_strat(transitions: list) -> Dict[str, dict]:
    """
    Provide a blank set of flow adjustments to be populated by the update_adjustments_for_strat function below.

    Args:
        transitions: All the transition flows we will be modifying through the clinical stratification process

    Returns:
        Dictionary of dictionaries of dictionaries of blank dictionaries to be populated later

    """

    flow_adjs = {}
    for agegroup in COVID_BASE_AGEGROUPS:
        flow_adjs[agegroup] = {}
        for clinical_stratum in CLINICAL_STRATA:
            flow_adjs[agegroup][clinical_stratum] = {}
            for transition in transitions:
                flow_adjs[agegroup][clinical_stratum][transition] = {}

    return flow_adjs


def update_adjustments_for_strat(strat: str, flow_adjustments: dict, adjustments: dict):
    """
    Add the flow adjustments to the blank adjustments created above by get_blank_adjustments_for_strat.

    Args:
        strat: The current stratification that we're modifying here
        flow_adjustments: Tiered dictionary containing the adjustments
        adjustments: Adjustments in the format that they are returned by get_all_adjustments

    """

    # Loop over the stratifications that affect these flow rates, other than VoC stratification
    for agegroup in COVID_BASE_AGEGROUPS:
        for clinical_stratum in CLINICAL_STRATA:

            # *** Note that PROGRESS is not indexed by age group
            modification = {strat: adjustments[PROGRESS][clinical_stratum]}
            flow_adjustments[agegroup][clinical_stratum][PROGRESS].update(modification)

            # ... but the other transition processes are
            for transition in AGE_CLINICAL_TRANSITIONS:
                modification = {strat: adjustments[transition][agegroup][clinical_stratum]}
                flow_adjustments[agegroup][clinical_stratum][transition].update(modification)


def add_clinical_adjustments_to_strat(
        strat: Stratification, flow_adjs: Dict[str, dict], unaffected_stratum: str, vocs: list
):
    """
    Add the clinical adjustments created in update_adjustments_for_strat to a stratification.

    Uses the summer method to the stratification set_flow_adjustments, that will then be applied when the stratify_with
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
    for agegroup in COVID_BASE_AGEGROUPS:
        for voc in vocs:
            for clinical_stratum in CLINICAL_STRATA:

                # The other model strata that we want to limit these adjustments to
                working_strata = {"agegroup": agegroup, "clinical": clinical_stratum}
                voc_strat = {"strain": voc} if len(vocs) > 1 else {}
                working_strata.update(voc_strat)

                # * Onset must be dest(ination) because this is the point at which the clinical stratification splits *
                infectious_onset_adjs = flow_adjs[voc][agegroup][clinical_stratum][INFECTIOUSNESS_ONSET]
                infectious_onset_adjs[unaffected_stratum] = None
                strat.set_flow_adjustments(INFECTIOUSNESS_ONSET, infectious_onset_adjs, dest_strata=working_strata)

                # * Progress can be either source, dest(ination) or both, but infect_death and recovery must be source *
                for transition in [PROGRESS, INFECT_DEATH, RECOVERY]:
                    adjs = flow_adjs[voc][agegroup][clinical_stratum][transition]
                    adjs[unaffected_stratum] = None
                    strat.set_flow_adjustments(transition, adjs, source_strata=working_strata)


def apply_immunity_to_strat(
        stratification: Stratification, params: Parameters, stratified_adjusters: Dict[str, Dict[str, float]],
        unaffected_stratum: str
):
    """
    Apply all the immunity effects to a stratification by immunity (either vaccination or history)

    Args:
        stratification: The current stratification being adjusted
        params: All requested model parameters
        stratified_adjusters: The VoC and outcome-specific adjusters
        unaffected_stratum: The name of the unaffected stratum

    """

    imm_params = getattr(params, stratification.name)
    changed_strata = [strat for strat in stratification.strata if strat != unaffected_stratum]
    infect_efficacy, flow_adjs = {}, {}
    vocs = list(stratified_adjusters.keys())
    for voc in vocs:
        flow_adjs[voc] = get_blank_adjustments_for_strat([PROGRESS, *AGE_CLINICAL_TRANSITIONS])
        for stratum in changed_strata:

            # Collate the effects together
            strat_args = (params, stratum, stratified_adjusters[voc], stratification.name)
            infect_efficacy[stratum], sympt_adj, hosp_adj, ifr_adj = get_stratum_vacc_history_effect(*strat_args)

            # Get the adjustments by clinical status and age group applicable to this VoC and vaccination stratum
            adjs = get_all_adjustments(
                params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
                params.sojourn, sympt_adj, hosp_adj, ifr_adj
            )

            # Get them into the format needed to be applied to the model
            update_adjustments_for_strat(stratum, flow_adjs[voc], adjs)
    add_clinical_adjustments_to_strat(stratification, flow_adjs, unaffected_stratum, vocs)

    # Effect against infection
    infect_adjs = {stratum: Multiply(1. - infect_efficacy[stratum]) for stratum in changed_strata}
    infect_adjs.update({unaffected_stratum: None})
    stratification.set_flow_adjustments(INFECTION, infect_adjs)

    # Vaccination effect against infectiousness
    infectious_adjs = {s: Multiply(1. - getattr(getattr(imm_params, s), "ve_infectiousness")) for s in changed_strata}
    infectious_adjs.update({unaffected_stratum: None})
    for compartment in DISEASE_COMPARTMENTS:
        stratification.add_infectiousness_adjustments(compartment, infectious_adjs)


"""
Parameter processing.
"""


def get_stratum_vacc_history_effect(
        params: Parameters, stratum: str, voc_adjusters: Dict[str, float], stratification: str
):
    """
    Process the vaccination parameters for the vaccination stratum being considered.

    Args:
        params: All the model parameters
        stratum: The vaccination stratum currently being added
        voc_adjusters: The calibration and VoC adjusters to IFR, hospitalisation and symptomatic proportion

    Returns:
        The processed VEs and the modified adjusters
    """

    # Parameters to directly pull out
    stratum_vacc_params = getattr(getattr(params, stratification), stratum)
    raw_effectiveness_keys = ["ve_prop_prevent_infection", "ve_sympt_covid"]
    if stratum_vacc_params.ve_death:
        raw_effectiveness_keys.append("ve_death")
    vacc_effects = {key: getattr(stratum_vacc_params, key) for key in raw_effectiveness_keys}

    # Parameters that need to be processed
    action_args = (vacc_effects["ve_prop_prevent_infection"], vacc_effects["ve_sympt_covid"])
    vacc_effects["infection_efficacy"], severity_effect = find_vaccine_action(*action_args)
    if stratum_vacc_params.ve_hospitalisation:
        hosp_args = (stratum_vacc_params.ve_hospitalisation, vacc_effects["ve_sympt_covid"])
        hospitalisation_effect = get_hosp_given_case_effect(*hosp_args)

    sympt_adjuster = 1.0 - severity_effect

    # Use the standard severity adjustment if no specific request for reducing death
    ifr_adjuster = 1.0 - vacc_effects["ve_death"] if "ve_death" in vacc_effects else 1.0 - severity_effect
    hospital_adjuster = 1.0 - hospitalisation_effect if stratum_vacc_params.ve_hospitalisation else 1.0

    # Apply the calibration adjusters
    ifr_adjuster *= voc_adjusters["ifr"]
    hospital_adjuster *= voc_adjusters["hosp"]
    sympt_adjuster *= voc_adjusters["sympt"]

    return vacc_effects["infection_efficacy"], sympt_adjuster, hospital_adjuster, ifr_adjuster


def get_hosp_given_case_effect(ve_hospitalisation: float, ve_case: float) -> float:
    """
    Calculate the effect of vaccination on hospitalisation in cases.
    Allowable values restricted to effect on hospitalisation being greater than or equal to the effect on developing
    symptomatic Covid (because otherwise hospitalisation would be commoner in breakthrough cases than in unvaccinated
    cases).

    Args:
        ve_hospitalisation: Vaccine effectiveness against hospitalisation (given exposure only)
        ve_case: Vaccine effectiveness against symptomatic Covid (also given exposure)

    Returns:
        VE for hospitalisation in breakthrough symptomatic cases

    """

    msg = "Hospitalisation effect less than the effect on becoming a case"
    assert ve_hospitalisation >= ve_case, msg
    ve_hosp_given_case = 1.0 - (1.0 - ve_hospitalisation) / (1.0 - ve_case)

    # Should be impossible for the following assertion to fail, but anyway
    msg = f"Effect of vaccination on hospitalisation given case: {ve_hosp_given_case}"
    assert 0.0 <= ve_hosp_given_case <= 1.0, msg

    return ve_hosp_given_case


def find_vaccine_action(vacc_prop_prevent_infection: float, overall_efficacy: float) -> Tuple[float, float]:
    """
    Calculate the vaccine efficacy in preventing infection and prevntion of symptomatic disese given infection.

    Args:
        vacc_prop_prevent_infection: The proportion of the observed effectiveness attributable to infection prevention
        overall_efficacy: The observed effectiveness on clinical cases

    Returns:
        VE for preventing infection and for preventing clinical outcomes given infection

    """

    # Infection efficacy follows from how we have defined these
    infection_efficacy = vacc_prop_prevent_infection * overall_efficacy

    # Severity efficacy must be calculated as the risk of severe outcomes given that the person was infected
    if vacc_prop_prevent_infection < 1.0:
        prop_attribute_severity = 1.0 - vacc_prop_prevent_infection
        prop_infect_prevented = 1.0 - vacc_prop_prevent_infection * overall_efficacy
        severity_efficacy = overall_efficacy * prop_attribute_severity / prop_infect_prevented
    else:
        severity_efficacy = 0.0

    msg = f"Infection efficacy not in [0, 1]: {infection_efficacy}"
    assert 0.0 <= infection_efficacy <= 1.0, msg
    msg = f"Severity efficacy not in [0, 1]: {severity_efficacy}"
    assert 0.0 <= severity_efficacy <= 1.0, msg

    return infection_efficacy, severity_efficacy


"""
Progression to second dose.
"""


def get_second_dose_delay_rate(dose_delay_params: Union[float, TanhScaleup]):
    """
    Get the rate of progression from partially to fully vaccinated (i.e. one dose to two doses).
    Currently this supports just two options, but could be extended to reflect the local context as needed.
    The format of the request determines the format of the output.

    Args:
        dose_delay_params: The user request for the rate of transition from first to second dose.

    Returns:
        The rate of transition from the first to second dose in a summer-ready format (either float or function)

    """

    if type(dose_delay_params) == float:
        return 1.0 / dose_delay_params
    else:
        return tanh_based_scaleup(
            shape=dose_delay_params.shape,
            inflection_time=dose_delay_params.inflection_time,
            start_asymptote=1.0 / dose_delay_params.start_asymptote,
            end_asymptote=1.0 / dose_delay_params.end_asymptote,
        )


"""
Provision of first dose, various approaches.
"""


def get_rate_from_coverage_and_duration(coverage_increase: float, duration: float) -> float:
    """
    Find the vaccination rate needed to achieve a certain coverage (of the remaining unvaccinated population) over a
    duration of time.
    Calculated by solving the following equation:
        coverage_increase = 1.0 - exp(-rate * duration)

    Args:
        coverage_increase: The proportion of remaining unvaccinated people vaccinated during the period
        duration: The time period in days over which this vaccination coverage accrues

    Returns:
        The rate needed to achieve this

    """

    assert duration >= 0.0, f"Vaccination roll-out request is negative: {duration}"
    assert 0.0 <= coverage_increase <= 1.0, f"Coverage not in [0, 1]: {coverage_increase}"
    return -np.log(1.0 - coverage_increase) / duration


def get_vacc_roll_out_function_from_coverage(coverage: float, start_time: float, end_time: float) -> Callable:
    """
    Calculate a single time-variant vaccination rate, based on a requested coverage and period for this roll-out.

    Args:
        coverage: Proportion of remaining unvaccinated people to be vaccinated during this period
        start_time: Time at which this period of vaccination starts
        end_time: Time at which this period of vaccination ends

    Returns:
        Function of time that steps up from zero for the vaccination period and then back down again

    """

    # Get vaccination parameters
    duration = end_time - start_time

    # Calculate the vaccination rate from the coverage and the duration of the program
    vaccination_rate = get_rate_from_coverage_and_duration(coverage, duration)

    # Create the function in standard summer format
    def get_vaccination_rate(time, computed_values):
        return vaccination_rate if start_time <= time < end_time else 0.0

    return get_vaccination_rate


def get_eligible_age_groups(age_min: float, age_max: float) -> List:
    """
    Get a list of the model's age groups that are relevant to the requested roll_out_component.

    Args:
        age_min: The minimum value of the requested age range
        age_max: The maximum value of the requested age range

    Returns:
        A list of all the age groups that are relevant to this implementation of vaccination

    """

    eligible_age_groups = []
    for agegroup in COVID_BASE_AGEGROUPS:

        # Either not requested, or requested and meets that age cut-off for min or max
        above_age_min = not age_min or bool(age_min) and float(agegroup) >= age_min
        below_age_max = not age_max or bool(age_max) and float(agegroup) < age_max
        if above_age_min and below_age_max:
            eligible_age_groups.append(agegroup)

    return eligible_age_groups


def apply_vacc_flows(model: CompartmentalModel, age_groups: Union[list, set], vaccination_rate: Union[float, Callable]):
    """
    Add vaccination flows from function or value that has previously been specified - including zero flows for to make
    the derived outputs requests simpler.

    Args:
        model: The model having the vaccination flows applied to it
        age_groups: The age groups to which these flows apply
        vaccination_rate: The vaccination transition rate for this age group, in summer-ready form

    """

    for eligible_age_group in age_groups:
        model.add_transition_flow(
            name="vaccination",
            fractional_rate=vaccination_rate,
            source=Compartment.SUSCEPTIBLE,
            dest=Compartment.SUSCEPTIBLE,
            source_strata={"vaccination": Vaccination.UNVACCINATED, "agegroup": eligible_age_group},
            dest_strata={"vaccination": Vaccination.ONE_DOSE_ONLY, "agegroup": eligible_age_group},
        )


def get_piecewise_vacc_rates(coverage_times: List[int], coverage_values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the per capita vaccination rates over sequential periods of time based on steadily increasing vaccination
    coverage values.

    Args:
        coverage_times: Times at which coverage values are available
        coverage_values: Proportion of the (age-specific) population covered at each coverage time
        vaccination_lag: Time from vaccination to developing immunological protection

    Returns:
        End times of vaccination period, lagged for development of immunity
        Rates of vaccination during each period

    """

    assert len(coverage_times) == len(coverage_values), "Vaccination coverage times and values not equal length"
    starting_coverage = coverage_values[0]

    error = 1e-4  # Some Vic clusters are > 1e-5 based on data provided by the Dept, this value is currently OK
    msg = f"Not starting from zero coverage: {starting_coverage}"
    assert abs(starting_coverage) < error, msg

    # Loop over the vaccination periods, which are one fewer than the length of data submitted
    n_intervals = len(coverage_times) - 1
    vaccination_rates = np.zeros(n_intervals)
    for i_period in range(n_intervals):
        period_start_time, period_end_time = coverage_times[i_period: i_period + 2]
        period_start_coverage, period_end_coverage = coverage_values[i_period: i_period + 2]

        # Duration of the current period
        duration = period_end_time - period_start_time
        assert duration >= 0.0, f"Vaccination roll-out request duration is negative: {duration}"

        # The proportion of the remaining unvaccinated people who will be vaccinated during the current interval
        coverage_increase = (period_end_coverage - period_start_coverage) / (1.0 - period_start_coverage)
        assert 0.0 <= coverage_increase <= 1.0, f"Coverage increase is not in [0, 1]: {coverage_increase}"

        # Get the vaccination rate from the increase in coverage over the period
        vaccination_rates[i_period] = get_rate_from_coverage_and_duration(coverage_increase, duration)

    # Convert to array for use during run-time
    return np.asarray(coverage_times)[1:], vaccination_rates


@numba.jit(nopython=True)
def get_vaccination_rate_jit(end_times, vaccination_rates, time):

    # Identify the index of the first list element greater than the time of interest
    # If there is such an index, return the corresponding vaccination rate
    for end_i, end_t in enumerate(end_times):
        if end_t > time:
            return vaccination_rates[end_i]

    # Return zero if the time is after the last end time
    return 0.0


def get_piecewise_rollout(end_times: np.ndarray, vaccination_rates: np.ndarray) -> Callable:
    """
    Turn the vaccination rates and end times into a piecewise roll-out function.
    coverage function, which are responsible for calculating the rates based on the coverage values at the requested
    points in time.

    Args:
        end_times: Sequence of the times at which the vaccination rate periods end
        vaccination_rates: The per capita rate of vaccination during the period ending at that end time

    Returns:
        Piecewise function in a summer-ready format of the rates of vaccination over time

    """

    assert len(end_times) == len(vaccination_rates), "Number of vaccination periods (end times) and rates differs"

    # Function defined in the standard format for function-based standard transition flows
    def get_vaccination_rate(time, computed_values):
        return get_vaccination_rate_jit(end_times, vaccination_rates, time)

    return get_vaccination_rate


def apply_standard_vacc_coverage(
        model: CompartmentalModel, vacc_lag: float, iso3: str, age_pops: List[int],
        one_dose_vacc_params: VaccEffectiveness, is_baseline: bool
):
    """
    Apply estimates of vaccination coverage, based on first doses provided at several points in time that are be
    obtained from the get_standard_vacc_coverage function to the model.

    Args:
        model: The model that will have these flows added to it
        vacc_lag: The period of time from vaccination to the development of immunity in days
        iso3: The string representing the country, to work out where to get the vaccination data from
        age_pops: The population structure by age group
        one_dose_vacc_params: The parameters relevant to this process
        is_baseline: Whether we are working from the baseline model

    """

    # Rates are likely to differ by age group, but not by other factors
    for agegroup in COVID_BASE_AGEGROUPS:

        # Note this function must return something for every age group to stop outputs calculation crashing
        coverage_times, coverage_values = get_standard_vacc_coverage(iso3, agegroup, age_pops, one_dose_vacc_params)
        lagged_cov_times = [i + vacc_lag for i in coverage_times]

        lagged_cov_times = [0.] + lagged_cov_times  # to avoid the issue that the first coverage value
        # is zero and the second one is positive.
        # So when you create a piecewise function from this, the first value of the piecewise function is
        # the rate in the first period - which is the daily rate when the program starts.

        coverage_values = [0.] + coverage_values
        # Get the vaccination rate function of time from the coverage values
        rollout_period_times, vaccination_rates = get_piecewise_vacc_rates(lagged_cov_times, coverage_values)

        # Vaccination program must commence after model has started
        if is_baseline:
            msg = "Vaccination program starts before model commencement"
            assert rollout_period_times[0] > model.times[0] + vacc_lag, msg

        # Apply the vaccination rate function to the model, using the period end times and the calculated rates
        vacc_rate_func = get_piecewise_rollout(rollout_period_times, vaccination_rates)
        model.add_transition_flow(
            name="vaccination",
            fractional_rate=vacc_rate_func,
            source=Compartment.SUSCEPTIBLE,
            dest=Compartment.SUSCEPTIBLE,
            source_strata={"agegroup": agegroup, "vaccination": Vaccination.UNVACCINATED},
            dest_strata={"agegroup": agegroup, "vaccination": Vaccination.ONE_DOSE_ONLY},
        )


def add_vacc_rollout_requests(model: CompartmentalModel, vacc_params: VaccParams):
    """
    Add the vaccination flows associated with each vaccine roll-out component.
    Loops through each roll-out component, which may apply to some or all age groups, but just adds one rate over one
    period of time.
    Note that this will create one flow object for each request/age group/compartment combination, which may not be
    inefficient.

    Args:
        model: The model object to have the flows added to
        vacc_params: The full set of vaccination-related user requests

    """

    all_eligible_agegroups = []
    for roll_out_component in vacc_params.roll_out_components:

        # Get parameters and lag vaccination times
        coverage_requests = roll_out_component.supply_period_coverage
        start_time = coverage_requests.start_time + vacc_params.lag
        end_time = coverage_requests.end_time + vacc_params.lag
        coverage = coverage_requests.coverage

        # Progressively collate the age groups
        working_agegroups = get_eligible_age_groups(roll_out_component.age_min, roll_out_component.age_max)
        all_eligible_agegroups += working_agegroups

        # Find the rate of vaccination over the period of this request
        vaccination_roll_out_function = get_vacc_roll_out_function_from_coverage(coverage, start_time, end_time)

        # Apply the vaccination flows to the model
        apply_vacc_flows(model, working_agegroups, vaccination_roll_out_function)

    # Add blank flows to make things simpler when we come to tracking the outputs
    ineligible_ages = set(COVID_BASE_AGEGROUPS) - set(all_eligible_agegroups)
    apply_vacc_flows(model, ineligible_ages, 0.0)


def get_standard_vacc_coverage(iso3: str, agegroup: str, age_pops: List[int], one_dose_vacc_params: VaccEffectiveness):
    """
    Implementation of a country-specific roll-out that needs coverage values to be looked up.

    Args:
        iso3: Country identifier
        agegroup: The age group being considered
        age_pops: Population age structure
        one_dose_vacc_params: Parameters governing transition to one dose stratum

    Returns:
        The times at which coverage values are available and their values

    """

    # Use the appropriate function to look up the coverage values
    vac_cov_map = {"MMR": get_myanmar_vacc_coverage, "LKA": get_lka_vac_coverage}
    time_series = vac_cov_map[iso3](agegroup, age_pops, one_dose_vacc_params)

    # Some simple checks
    check_list_increasing(time_series.times)
    check_list_increasing(time_series.values)
    assert all((0.0 <= i_coverage <= 1.0 for i_coverage in time_series.values))

    return time_series.times, time_series.values


def get_myanmar_vacc_coverage(age_group: str, age_pops: List[int], one_dose_vacc_params: VaccEffectiveness):
    """
    Get the age-specific vaccination coverage values from the number of people who have received at least one dose,
    provided by base_mmr_adult_vacc_doses.

    Args:
        age_group: The age group for which coverage values are needed
        age_pops: The population structure by age group
        one_dose_vacc_params: The parameter requests for receipt of first dose

    Returns:
        Time series of coverage values

    """

    times, one_dose_pops = base_mmr_adult_vacc_doses()

    # For the adult population
    if int(age_group) >= 15:

        # Convert doses to coverage
        adult_denominator = sum(age_pops[3:])
        coverage_values = [i_doses / adult_denominator for i_doses in one_dose_pops]

        # Extend with user requests
        if one_dose_vacc_params.coverage:
            times.extend(one_dose_vacc_params.coverage.times)
            coverage_values.extend(one_dose_vacc_params.coverage.values)

    # No vaccination for children
    else:
        coverage_values = [0.0] * len(times)

    return TimeSeries(times=times, values=coverage_values)
