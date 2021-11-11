import numpy as np
from typing import List, Callable, Optional, Tuple, Union

from summer import CompartmentalModel

from autumn.models.covid_19.constants import VACCINE_ELIGIBLE_COMPARTMENTS, Vaccination, VACCINATION_STRATA
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.tools.inputs.covid_au.queries import (
    get_both_vacc_coverage, VACC_COVERAGE_START_AGES, VACC_COVERAGE_END_AGES
)
from autumn.tools.utils.utils import find_closest_value_in_list
from autumn.models.covid_19.parameters import Vaccination as VaccParams, RollOutFunc
from autumn.tools.curve import tanh_based_scaleup


def get_second_dose_delay_rate(dose_delay_params):
    """
    Get the rate of progression from partially to fully vaccinated.

    :param dose_delay_params:
    :type dose_delay_params:
    :return:
    :rtype:
    """

    if type(dose_delay_params) == float:
        return 1. / dose_delay_params
    else:
        return tanh_based_scaleup(
            shape=dose_delay_params.shape,
            inflection_time=dose_delay_params.inflection_time,
            start_asymptote=1. / dose_delay_params.start_asymptote,
            end_asymptote=1. / dose_delay_params.end_asymptote,
        )


def get_rate_from_coverage_and_duration(coverage_increase: float, duration: float) -> float:
    """
    Find the vaccination rate needed to achieve a certain coverage (of the remaining unvaccinated population) over a
    duration of time.
    """

    assert duration >= 0., f"Vaccination roll-out request is negative: {duration}"
    assert 0. <= coverage_increase <= 1., f"Coverage not in [0, 1]: {coverage_increase}"
    return -np.log(1. - coverage_increase) / duration


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
        return vaccination_rate if start_time <= time < end_time else 0.

    return get_vaccination_rate


def get_hosp_given_case_effect(vacc_reduce_hosp_given_case: float, case_effect: float) -> float:
    """
    Calculate the effect of vaccination on hospitalisation in cases.
    Note that this calculation is only intended for the situation where the effect on hospitalisation is at least as
    great as the effect on becoming a case.
    """

    msg = "Hospitalisation effect less than the effect on becoming a case"
    assert vacc_reduce_hosp_given_case >= case_effect, msg
    effect = 1. - (1. - vacc_reduce_hosp_given_case) / (1. - case_effect)

    # Should be impossible for the following assertion to fail, but anyway
    assert 0. <= effect <= 1., f"Effect of vaccination on hospitalisation given case: {effect}"
    return effect


def find_vaccine_action(vacc_prop_prevent_infection: float, overall_efficacy: float) -> Tuple[float, float]:
    """
    Calculating the vaccine efficacy in preventing infection and leading to severe infection.
    """

    # Infection efficacy follows from how we have defined these
    infection_efficacy = vacc_prop_prevent_infection * overall_efficacy

    # Severity efficacy must be calculated as the risk of severe outcomes given that the person was infected
    if vacc_prop_prevent_infection < 1.:
        prop_attribute_severity = 1. - vacc_prop_prevent_infection
        prop_infect_prevented = 1. - vacc_prop_prevent_infection * overall_efficacy
        severity_efficacy = overall_efficacy * prop_attribute_severity / prop_infect_prevented
    else:
        severity_efficacy = 0.

    msg = f"Infection efficacy not in [0, 1]: {infection_efficacy}"
    assert 0. <= infection_efficacy <= 1., msg
    msg = f"Severity efficacy not in [0, 1]: {severity_efficacy}"
    assert 0. <= severity_efficacy <= 1., msg

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
        model: CompartmentalModel, age_groups: Union[list, set], vaccination_rate: Union[float, Callable],
        extra_stratum={}
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
        working_agegroups = get_eligible_age_groups(roll_out_component.age_min, roll_out_component.age_max)
        all_eligible_agegroups += working_agegroups

        # Coverage-based vaccination
        vaccination_roll_out_function = get_vacc_roll_out_function_from_coverage(
            coverage_requests.coverage, coverage_requests.start_time, coverage_requests.end_time,
        )
        add_vacc_flows(model, working_agegroups, vaccination_roll_out_function)

    # Add blank flows to make things simpler when we come to doing the outputs
    ineligible_ages = set(AGEGROUP_STRATA) - set(all_eligible_agegroups)
    add_vacc_flows(model, ineligible_ages, 0.)


def get_piecewise_vacc_rates(
        start_time: float, end_time: float, time_intervals: int, coverage_times: List, coverage_values: list,
        vaccination_lag: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a vaccination roll-out rate function of time with values determined by sequentially increasing coverage
    values.
    """

    # Find all the intervals to create the step function over
    rollout_period_times = np.linspace(start_time, end_time, time_intervals + 1)

    # Loop over the periods of time in the step function
    vaccination_rates = np.zeros(time_intervals)
    for i_period in range(time_intervals):
        period_start_time = rollout_period_times[i_period]
        period_end_time = rollout_period_times[i_period + 1]

        # Interpolate for coverage values, starting from zero if this is the first roll-out period
        period_start_coverage = np.interp(period_start_time - vaccination_lag, coverage_times, coverage_values)
        modelled_start_coverage = 0. if i_period == 0. else period_start_coverage
        period_end_coverage = np.interp(period_end_time - vaccination_lag, coverage_times, coverage_values)

        # Find the duration of this period
        duration = period_end_time - period_start_time
        assert duration >= 0., f"Vaccination roll-out request duration is negative: {duration}"

        # The proportion of the remaining unvaccinated people who will be vaccinated
        coverage_increase = (period_end_coverage - modelled_start_coverage) / (1. - modelled_start_coverage)
        assert 0. <= coverage_increase <= 1., f"Coverage increase is not in [0, 1]: {coverage_increase}"

        # Calculate the rate from the increase in coverage
        vaccination_rates[i_period] = get_rate_from_coverage_and_duration(coverage_increase, duration)

    return rollout_period_times, vaccination_rates


def get_piecewise_rollout(start_time: float, end_times: list, vaccination_rates: list) -> Callable:
    """
    Turn the vaccination rates and end times into a piecewise roll-out function.
    """

    def get_vaccination_rate(time, computed_values):
        if time > start_time:
            idx = sum(end_times < time)
            if idx < len(vaccination_rates):
                return vaccination_rates[idx]
        return 0.

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
        age_max = VACC_COVERAGE_END_AGES[-1] if i_age == len(age_breaks) - 1 else age_breaks[i_age + 1]
        working_agegroups = get_eligible_age_groups(age_min, age_max)
        all_eligible_agegroups += working_agegroups

        # Get the cluster-specific historical vaccination data
        close_enough_age_min = find_closest_value_in_list(VACC_COVERAGE_START_AGES, age_min)
        close_enough_age_max = find_closest_value_in_list(VACC_COVERAGE_END_AGES, age_max)
        coverage_times, coverage_values = get_both_vacc_coverage(
            cluster_name.upper(), start_age=close_enough_age_min, end_age=close_enough_age_max,
        )

        # The first age group should be adjusted for partial coverage of that age group
        if i_age == 0:
            coverage_values *= (close_enough_age_max - close_enough_age_min) / (age_max - age_min)

        # Get end times, stopping at end of available data
        end_time = min((max(coverage_times), roll_out_params.end_time))

        # Get the vaccination rate function of time
        rollout_period_times, vaccination_rates = get_piecewise_vacc_rates(
            roll_out_params.start_time, end_time, roll_out_params.time_intervals,
            coverage_times, coverage_values, vacc_params.lag
        )

        # Apply the vaccination rate function to the model
        vacc_rate_func = get_piecewise_rollout(model_start_time, rollout_period_times[1:], vaccination_rates)
        add_vacc_flows(model, working_agegroups, vacc_rate_func)

    # Add blank/zero flows to make the output requests simpler
    ineligible_ages = set(AGEGROUP_STRATA) - set(all_eligible_agegroups)
    add_vacc_flows(model, ineligible_ages, 0.)


def add_vic2021_supermodel_vacc(model: CompartmentalModel, vacc_params, cluster_strata: str):
    """
    *** This appears to be working, but would need to be checked if we went back to using this approach ***
    """

    for roll_out_component in vacc_params.roll_out_components:

        # Work out eligible model age_groups
        eligible_age_groups = get_eligible_age_groups(roll_out_component.age_min, roll_out_component.age_max)

        close_enough_age_min = find_closest_value_in_list(VACC_COVERAGE_START_AGES, roll_out_component.age_min) if \
            roll_out_component.age_min else 0
        close_enough_age_max = find_closest_value_in_list(VACC_COVERAGE_END_AGES, roll_out_component.age_max) if \
            roll_out_component.age_max else 89

        for vic_cluster in cluster_strata:
            cluster_stratum = {"cluster": vic_cluster}

            # Get the cluster-specific historical vaccination numbers
            coverage_times, coverage_values = get_both_vacc_coverage(
                vic_cluster.upper(),
                start_age=close_enough_age_min,
                end_age=close_enough_age_max,
            )

            # Stop at the end of the available data, even if the request is later
            end_time = min((max(coverage_times), roll_out_component.vic_supply.end_time))

            get_vaccination_rate = get_piecewise_vacc_rates(
                roll_out_component.vic_supply.start_time, end_time, roll_out_component.vic_supply.time_intervals,
                coverage_times, coverage_values, vacc_params.lag,
            )
            add_vacc_flows(model, eligible_age_groups, get_vaccination_rate, extra_stratum=cluster_stratum)

            # Add blank flows to make things simpler when we come to doing the outputs
            ineligible_ages = set(AGEGROUP_STRATA) - set(eligible_age_groups)
            add_vacc_flows(model, ineligible_ages, 0., extra_stratum=cluster_stratum)


def get_vacc_effects_by_stratum(symptomatic_adjuster, hospital_adjuster, ifr_adjuster, params):

    vaccination_effects = {}

    # Get vaccination effect parameters in the form needed for the model
    for stratum in VACCINATION_STRATA[1:]:

        # Parameters to directly pull out
        raw_effectiveness_keys = ["ve_prop_prevent_infection", "ve_sympt_covid"]
        stratum_vacc_params = getattr(params.vaccination, stratum)
        if stratum_vacc_params.ve_death:
            raw_effectiveness_keys.append("ve_death")
        vaccination_effects[stratum] = {key: getattr(stratum_vacc_params, key) for key in raw_effectiveness_keys}

        # Parameters that need to be processed
        vaccination_effects[stratum]["infection_efficacy"], severity_effect = find_vaccine_action(
            vaccination_effects[stratum]["ve_prop_prevent_infection"],
            vaccination_effects[stratum]["ve_sympt_covid"],
        )
        if stratum_vacc_params.ve_hospitalisation:
            hospitalisation_effect = get_hosp_given_case_effect(
                stratum_vacc_params.ve_hospitalisation, vaccination_effects[stratum]["ve_sympt_covid"],
            )

        symptomatic_adjuster[stratum] = 1. - severity_effect

        # Use the standard severity adjustment if no specific request for reducing death
        ifr_adjuster[stratum] = 1. - vaccination_effects[stratum]["ve_death"] if \
            "ve_death" in vaccination_effects[stratum] else 1. - severity_effect
        hospital_adjuster[stratum] = 1. - hospitalisation_effect if \
            "ve_hospitalisation" in vaccination_effects[stratum] else 1.

        # Apply the calibration adjusters
        symptomatic_adjuster[stratum] *= params.clinical_stratification.props.symptomatic.multiplier
        ifr_adjuster[stratum] *= params.infection_fatality.multiplier

    return vaccination_effects
