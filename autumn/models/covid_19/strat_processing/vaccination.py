import numpy as np
from typing import List, Callable, Tuple, Union

from summer import CompartmentalModel

from autumn.models.covid_19.constants import (
    VACCINE_ELIGIBLE_COMPARTMENTS,
    Vaccination,
)
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.tools.inputs.covid_au.queries import (
    get_both_vacc_coverage,
    VACC_COVERAGE_START_AGES,
    VACC_COVERAGE_END_AGES,
)
from autumn.tools.utils.utils import find_closest_value_in_list, check_list_increasing
from autumn.models.covid_19.parameters import Vaccination as VaccParams, TimeSeries, VaccEffectiveness
from autumn.tools.curve import tanh_based_scaleup
from autumn.tools.inputs.covid_lka.queries import get_lka_vac_coverage
from autumn.tools.inputs.covid_mmr.queries import base_mmr_vac_doses


def get_second_dose_delay_rate(dose_delay_params):
    """
    Get the rate of progression from partially to fully vaccinated.

    :param dose_delay_params:
    :type dose_delay_params:
    :return:
    :rtype:
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


def get_rate_from_coverage_and_duration(coverage_increase: float, duration: float) -> float:
    """
    Find the vaccination rate needed to achieve a certain coverage (of the remaining unvaccinated population) over a
    duration of time.
    """

    assert duration >= 0.0, f"Vaccination roll-out request is negative: {duration}"
    assert 0.0 <= coverage_increase <= 1.0, f"Coverage not in [0, 1]: {coverage_increase}"
    return -np.log(1.0 - coverage_increase) / duration


def get_vacc_roll_out_function_from_coverage(
    coverage: float, start_time: float, end_time: float, coverage_override: float = None
) -> Callable:
    """
    Calculate the time-variant vaccination rate, based on a requested coverage and roll-out window.
    Return a single stepped function of time.
    """

    # Get vaccination parameters
    coverage = coverage if coverage else coverage_override
    duration = end_time - start_time

    # Calculate the vaccination rate from the coverage and the duration of the program
    vaccination_rate = get_rate_from_coverage_and_duration(coverage, duration)

    # Create the function
    def get_vaccination_rate(time, computed_values):
        return vaccination_rate if start_time <= time < end_time else 0.0

    return get_vaccination_rate


def get_hosp_given_case_effect(vacc_reduce_hosp_given_case: float, case_effect: float) -> float:
    """
    Calculate the effect of vaccination on hospitalisation in cases.
    Note that this calculation is only intended for the situation where the effect on hospitalisation is at least as
    great as the effect on becoming a case.
    """

    msg = "Hospitalisation effect less than the effect on becoming a case"
    assert vacc_reduce_hosp_given_case >= case_effect, msg
    effect = 1.0 - (1.0 - vacc_reduce_hosp_given_case) / (1.0 - case_effect)

    # Should be impossible for the following assertion to fail, but anyway
    assert 0.0 <= effect <= 1.0, f"Effect of vaccination on hospitalisation given case: {effect}"
    return effect


def find_vaccine_action(
    vacc_prop_prevent_infection: float, overall_efficacy: float
) -> Tuple[float, float]:
    """
    Calculating the vaccine efficacy in preventing infection and leading to severe infection.
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


def get_eligible_age_groups(age_min, age_max) -> List:
    """
    Return a list with the model's age groups that are relevant to the requested roll_out_component.
    """

    eligible_age_groups = []
    for agegroup in AGEGROUP_STRATA:

        # Either not requested, or requested and meets that age cut-off for min or max
        above_age_min = not age_min or bool(age_min) and float(agegroup) >= age_min
        below_age_max = not age_max or bool(age_max) and float(agegroup) < age_max
        if above_age_min and below_age_max:
            eligible_age_groups.append(agegroup)

    return eligible_age_groups


def add_vacc_flows(
    model: CompartmentalModel,
    age_groups: Union[list, set],
    vaccination_rate: Union[float, Callable],
    extra_stratum={},
):
    """
    Add vaccination flows from function or value that has previously been specified - including zero flows for to make
    the derived outputs requests simpler.
    """

    for eligible_age_group in age_groups:
        source_strata = {"vaccination": Vaccination.UNVACCINATED, "agegroup": eligible_age_group}
        source_strata.update(extra_stratum)
        dest_strata = {"vaccination": Vaccination.ONE_DOSE_ONLY, "agegroup": eligible_age_group}
        dest_strata.update(extra_stratum)
        for compartment in VACCINE_ELIGIBLE_COMPARTMENTS:
            model.add_transition_flow(
                name="vaccination",
                fractional_rate=vaccination_rate,
                source=compartment,
                dest=compartment,
                source_strata=source_strata,
                dest_strata=dest_strata,
            )


def add_requested_vacc_flows(model: CompartmentalModel, vacc_params: VaccParams):
    """
    Add the vaccination flows associated with a vaccine roll-out component (i.e. a given age-range and supply function).
    Flexible enough to handle various user requests, but will create one flow object for each request/age group/
    compartment combination.
    """

    all_eligible_agegroups = []
    for roll_out_component in vacc_params.roll_out_components:
        coverage_requests = roll_out_component.supply_period_coverage
        working_agegroups = get_eligible_age_groups(
            roll_out_component.age_min, roll_out_component.age_max
        )
        all_eligible_agegroups += working_agegroups

        # Coverage-based vaccination
        vaccination_roll_out_function = get_vacc_roll_out_function_from_coverage(
            coverage_requests.coverage,
            coverage_requests.start_time,
            coverage_requests.end_time,
        )
        add_vacc_flows(model, working_agegroups, vaccination_roll_out_function)

    # Add blank flows to make things simpler when we come to doing the outputs
    ineligible_ages = set(AGEGROUP_STRATA) - set(all_eligible_agegroups)
    add_vacc_flows(model, ineligible_ages, 0.0)


def get_piecewise_vacc_rates(
    coverage_times: List[int], coverage_values: List[float], vaccination_lag: float,
) -> Tuple[np.ndarray, np.ndarray]:
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

    error = 1e-4  # Some Vic clusters are > 1e-5 based on data provided by the Dept, this value is currently OK
    assert len(coverage_times) == len(coverage_values), "Vaccination coverage times and values not equal length"
    starting_coverage = coverage_values[0]
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

    # Lag for immunity and convert to array for use during run-time
    vacc_times = np.asarray(coverage_times) - vaccination_lag
    return vacc_times, vaccination_rates


def get_piecewise_rollout(end_times: np.ndarray, vaccination_rates: np.ndarray) -> Callable:
    """
    Turn the vaccination rates and end times into a piecewise roll-out function.
    Called by the Victoria progressive vaccination coverage function and by the more generalisable standard vacc
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

        # Identify the index of the first list element greater than the time of interest
        idx = sum(end_times < time)

        # Return zero if the time is after the last end time, otherwise take the vaccination rate
        return 0.0 if idx >= len(vaccination_rates) else vaccination_rates[idx]

    return get_vaccination_rate


def add_vic_regional_vacc(
    model: CompartmentalModel, vacc_params: VaccParams, cluster_name: str, model_start_time: float
):
    """
    Apply vaccination to the Victoria regional cluster models.
    """

    # Track all the age groups we have applied vaccination to as we loop over components with different age requests
    all_eligible_agegroups = []
    roll_out_params = vacc_params.roll_out_components[0].vic_supply
    age_breaks = roll_out_params.age_breaks
    for i_age, age_min in enumerate(age_breaks):
        age_max = (
            VACC_COVERAGE_END_AGES[-1] if i_age == len(age_breaks) - 1 else age_breaks[i_age + 1]
        )
        working_agegroups = get_eligible_age_groups(age_min, age_max)
        all_eligible_agegroups += working_agegroups

        # Get the cluster-specific historical vaccination data
        close_enough_age_min = find_closest_value_in_list(VACC_COVERAGE_START_AGES, age_min)
        close_enough_age_max = find_closest_value_in_list(VACC_COVERAGE_END_AGES, age_max)
        coverage_times, coverage_values = get_both_vacc_coverage(
            cluster_name.upper(),
            start_age=close_enough_age_min,
            end_age=close_enough_age_max,
        )

        # The first age group should be adjusted for partial coverage of that age group
        if i_age == 0:
            coverage_values *= (close_enough_age_max - close_enough_age_min) / (age_max - age_min)

        # Get the vaccination rate function of time
        rollout_period_times, vaccination_rates = get_piecewise_vacc_rates(
            coverage_times, coverage_values, vacc_params.lag,
        )

        # Apply the vaccination rate function to the model
        vacc_rate_func = get_piecewise_rollout(rollout_period_times[1:], vaccination_rates)
        add_vacc_flows(model, working_agegroups, vacc_rate_func)

    # Add blank/zero flows to make the output requests simpler
    ineligible_ages = set(AGEGROUP_STRATA) - set(all_eligible_agegroups)
    add_vacc_flows(model, ineligible_ages, 0.0)


def apply_standard_vacc_coverage(
        model: CompartmentalModel, vacc_lag: float, iso3: str, age_pops: List[int],
        one_dose_vacc_params: VaccEffectiveness, scenario_status: bool
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
        scenario_status: Whether we are working from the baseline model

    """

    # Rates are likely to differ by age group, but not by other factors
    for agegroup in AGEGROUP_STRATA:

        # Note this function must return something for every age group to stop outputs calculation crashing
        coverage_times, coverage_values = get_standard_vacc_coverage(iso3, agegroup, age_pops, one_dose_vacc_params)

        # Get the vaccination rate function of time from the coverage values
        rollout_period_times, vaccination_rates = get_piecewise_vacc_rates(coverage_times, coverage_values, vacc_lag)

        # Vaccination program must commence after model has started
        if scenario_status:
            msg = "Vaccination program starts before model commencement"
            assert rollout_period_times[0] > model.times[0] + vacc_lag, msg

        # Apply the vaccination rate function to the model, using the period end times and the calculated rates
        vacc_rate_func = get_piecewise_rollout(rollout_period_times[1:], vaccination_rates)
        for compartment in VACCINE_ELIGIBLE_COMPARTMENTS:
            model.add_transition_flow(
                name="vaccination",
                fractional_rate=vacc_rate_func,
                source=compartment,
                dest=compartment,
                source_strata={"agegroup": agegroup, "vaccination": "unvaccinated"},
                dest_strata={"vaccination": "one_dose"},
            )


def get_stratum_vacc_effect(params, stratum, voc_adjusters):

    # Parameters to directly pull out
    stratum_vacc_params = getattr(params.vaccination, stratum)
    raw_effectiveness_keys = ["ve_prop_prevent_infection", "ve_sympt_covid"]
    if stratum_vacc_params.ve_death:
        raw_effectiveness_keys.append("ve_death")
    vacc_effects = {key: getattr(stratum_vacc_params, key) for key in raw_effectiveness_keys}

    # Parameters that need to be processed
    vacc_effects["infection_efficacy"], severity_effect = find_vaccine_action(
        vacc_effects["ve_prop_prevent_infection"],
        vacc_effects["ve_sympt_covid"],
    )
    if stratum_vacc_params.ve_hospitalisation:
        hospitalisation_effect = get_hosp_given_case_effect(
            stratum_vacc_params.ve_hospitalisation,
            vacc_effects["ve_sympt_covid"],
        )

    sympt_adjuster = 1.0 - severity_effect

    # Use the standard severity adjustment if no specific request for reducing death
    ifr_adjuster = (1.0 - vacc_effects["ve_death"] if "ve_death" in vacc_effects else 1.0 - severity_effect)
    hospital_adjuster = (1.0 - hospitalisation_effect if "ve_hospitalisation" in vacc_effects else 1.0)

    # Apply the calibration adjusters
    ifr_adjuster *= voc_adjusters["ifr"]
    hospital_adjuster *= voc_adjusters["hosp"]
    sympt_adjuster *= voc_adjusters["sympt"]

    return vacc_effects, sympt_adjuster, hospital_adjuster, ifr_adjuster


def get_standard_vacc_coverage(iso3, agegroup, age_pops, one_dose_vacc_params):

    vac_cov_map = {"MMR": get_mmr_vac_coverage, "LKA": get_lka_vac_coverage}

    time_series = vac_cov_map[iso3](agegroup, age_pops, one_dose_vacc_params)

    # A couple of standard checks
    check_list_increasing(time_series.times)
    check_list_increasing(time_series.values)
    assert all((0.0 <= i_coverage <= 1.0 for i_coverage in time_series.values))

    return time_series.times, time_series.values


def get_mmr_vac_coverage(age_group, age_pops, one_dose_vacc_params):

    times, at_least_one_dose = base_mmr_vac_doses()

    # For the adult population
    if int(age_group) >= 15:
        adult_denominator = sum(age_pops[3:])

        # Convert doses to coverage
        coverage_values = [i_doses / adult_denominator for i_doses in at_least_one_dose]

        # Extend with user requests
        if one_dose_vacc_params.coverage:
            times.extend(one_dose_vacc_params.coverage.times)
            coverage_values.extend(one_dose_vacc_params.coverage.values)

    # For the children, no vaccination
    else:
        coverage_values = [0.0] * len(times)

    return TimeSeries(times=times, values=coverage_values)
