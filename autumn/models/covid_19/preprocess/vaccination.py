import numpy as np
from typing import List, Callable, Optional, Tuple, Union

from summer import CompartmentalModel

from autumn.models.covid_19.constants import VACCINE_ELIGIBLE_COMPARTMENTS, Vaccination
from autumn.tools.curve.scale_up import scale_up_function
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.tools.inputs.covid_au.queries import (
    get_both_vacc_coverage, VACC_COVERAGE_START_AGES, VACC_COVERAGE_END_AGES
)
from autumn.models.covid_19.parameters import Vaccination as VaccParams, RollOutFunc


def find_closest_value_in_list(list_request: List, value_request: int) -> int:
    """
    Find the closest value within one list to the value of interest.
    """

    return min(list_request, key=lambda list_value: abs(list_value - value_request))


def get_vacc_roll_out_function_from_coverage(
        coverage: float, start_time: float, end_time: float, coverage_override: Optional[float]
) -> Callable:
    """
    Calculate the time-variant vaccination rate based on a requested coverage and roll-out window.
    Return a single stepped function of time.
    """

    # Get vaccination parameters
    coverage = coverage if coverage else coverage_override
    duration = end_time - start_time
    assert duration >= 0., f"Vaccination roll-out request is negative: {duration}"
    assert 0. <= coverage <= 1., f"Coverage not in [0, 1]: {coverage}"

    # Calculate the vaccination rate from the coverage and the duration of the program
    vaccination_rate = -np.log(1. - coverage) / duration

    # Create the function
    def get_vaccination_rate(time, computed_values):
        return vaccination_rate if start_time <= time < end_time else 0.

    return get_vaccination_rate


def get_vacc_roll_out_function_from_doses(
    time_variant_supply, compartment_name, eligible_age_group, eligible_age_groups
):
    """
    Work out the number of vaccinated individuals for a given agegroup and compartment name.
    Return a time-variant function in a format that can be used to inform a functional flow.
    """

    def net_flow_func(model, compartments, compartment_values, flows, flow_rates, computed_values, time):
        # work out the proportion of the eligible population that is in the relevant compartment
        # FIXME: we should be able to cache the two lists below. They depend on compartment_name, eligible_age_group
        #  and eligible_age_groups but not on time!
        num_indices, deno_indices = [], []
        for i, compartment in enumerate(compartments):
            if (
                compartment.name in VACCINE_ELIGIBLE_COMPARTMENTS
                and compartment.strata["agegroup"] in eligible_age_groups
                and compartment.strata["vaccination"] == "unvaccinated"
            ):
                deno_indices.append(i)
                if (
                    compartment.name == compartment_name
                    and compartment.strata["agegroup"] == eligible_age_group
                ):
                    num_indices.append(i)

        compartment_size = sum([compartment_values[i] for i in num_indices])
        total_eligible_pop_size = sum([compartment_values[i] for i in deno_indices])
        nb_vaccinated = \
            compartment_size / total_eligible_pop_size * time_variant_supply(time) if \
            total_eligible_pop_size >= 0.1 else 0.

        return max(0, min(nb_vaccinated, compartment_size))

    return net_flow_func


def get_eligible_age_groups(roll_out_component: RollOutFunc) -> List:
    """
    Return a list with the model's age groups that are relevant to the requested roll_out_component.
    """

    eligible_age_groups = []
    for agegroup in AGEGROUP_STRATA:

        # Either not requested, or requested and meets that age cut-off for min or max
        above_age_min = \
            not roll_out_component.age_min or \
            bool(roll_out_component.age_min) and float(agegroup) >= roll_out_component.age_min
        below_age_max = \
            not roll_out_component.age_max or \
            bool(roll_out_component.age_max) and float(agegroup) < roll_out_component.age_max
        if above_age_min and below_age_max:
            eligible_age_groups.append(agegroup)

    return eligible_age_groups


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


def add_vacc_flows(
        model: CompartmentalModel, age_groups: List, vaccination_rate: Union[float, Callable], dest_stratum: str,
        extra_stratum={}):
    """
    Add vaccination flows from function or value that has previously been specified - including zero flows for to make
    the derived outputs requests simpler.
    """

    for eligible_age_group in age_groups:
        source_strata = {"vaccination": Vaccination.UNVACCINATED, "agegroup": eligible_age_group}
        source_strata.update(extra_stratum)
        dest_strata = {"vaccination": dest_stratum, "agegroup": eligible_age_group}
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


def add_requested_vacc_flows(model: CompartmentalModel, vacc_params: VaccParams, one_dose: bool):
    """
    Add the vaccination flows associated with a vaccine roll-out component (i.e. a given age-range and supply function).
    Flexible enough to handle various user requests, but will create one flow object for each request/age group/
    compartment combination.
    """

    all_eligible_agegroups = []
    vacc_dest_stratum = None
    for roll_out_component in vacc_params.roll_out_components:
        working_agegroups = get_eligible_age_groups(roll_out_component)
        all_eligible_agegroups += working_agegroups

        # Find vaccination destination stratum, depending on whether one-dose vaccination stratum is active
        vacc_dest_stratum = Vaccination.ONE_DOSE_ONLY if one_dose else Vaccination.VACCINATED

        # Coverage-based vaccination
        if roll_out_component.supply_period_coverage:
            vaccination_roll_out_function = get_vacc_roll_out_function_from_coverage(
                roll_out_component.supply_period_coverage.coverage,
                roll_out_component.supply_period_coverage.start_time,
                roll_out_component.supply_period_coverage.end_time,
                vacc_params.coverage_override if vacc_params.coverage_override else None
            )
            add_vacc_flows(model, working_agegroups, vaccination_roll_out_function, vacc_dest_stratum)

        # Dose-based vaccination
        else:
            time_variant_supply = scale_up_function(
                roll_out_component.supply_timeseries.times,
                roll_out_component.supply_timeseries.values,
                method=4,
            )
            for eligible_age_group in working_agegroups:
                _source_strata = {"vaccination": Vaccination.UNVACCINATED, "agegroup": eligible_age_group}
                _dest_strata = {"vaccination": vacc_dest_stratum, "agegroup": eligible_age_group}
                for compartment in VACCINE_ELIGIBLE_COMPARTMENTS:

                    # We need to create a functional flow, which depends on the agegroup and the compartment considered
                    vaccination_roll_out_function = get_vacc_roll_out_function_from_doses(
                        time_variant_supply, compartment, eligible_age_group, working_agegroups
                    )
                    model.add_function_flow(
                        name="vaccination",
                        flow_rate_func=vaccination_roll_out_function,
                        source=compartment,
                        dest=compartment,
                        source_strata=_source_strata,
                        dest_strata=_dest_strata,
                    )

    # Add blank flows to make things simpler when we come to doing the outputs
    ineligible_ages = set(AGEGROUP_STRATA) - set(all_eligible_agegroups)
    add_vacc_flows(model, ineligible_ages, 0., vacc_dest_stratum)


def get_piecewise_vacc_func(
        start_time: float, end_time: float, time_intervals: int, coverage_times: List, coverage_values: list,
        vaccination_lag: float) -> Callable:
    """
    Creates a vaccination roll-out rate function of time with values determined by sequentially increasing coverage
    values.
    """

    # Find all the intervals to create the step function over
    rollout_period_times = np.linspace(start_time, end_time, int(time_intervals) + 1)

    # Loop over the periods of time in the step function
    n_periods = len(rollout_period_times) - 1
    vaccination_rates = np.zeros(n_periods)
    for i_period in range(n_periods):
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
        vaccination_rate = -np.log(1. - coverage_increase) / duration
        vaccination_rates[i_period] = vaccination_rate

    # Get the piecewise function
    end_times = rollout_period_times[1:]

    def get_vaccination_rate(time, computed_values):
        if time > start_time:
            idx = sum(end_times < time)
            if idx < len(vaccination_rates):
                return vaccination_rates[idx]
        return 0.

    return get_vaccination_rate


def add_vic_regional_vacc(model: CompartmentalModel, vacc_params: VaccParams, cluster_name: str):
    """
    Apply vaccination to the Victoria regional cluster models.
    """

    # Track all the age groups we have applied vaccination to as we loop over components with different age requests
    all_eligible_agegroups = []
    for agegroup_component in vacc_params.roll_out_components:
        working_agegroups = get_eligible_age_groups(agegroup_component)
        msg = "An age group has been requested multiple times"
        assert not bool(set(all_eligible_agegroups) & set(working_agegroups)), msg
        all_eligible_agegroups += working_agegroups

        # Get the cluster-specific historical vaccination data
        close_enough_age_min = find_closest_value_in_list(VACC_COVERAGE_START_AGES, agegroup_component.age_min) if \
            agegroup_component.age_min else 0
        close_enough_age_max = find_closest_value_in_list(VACC_COVERAGE_END_AGES, agegroup_component.age_max) if \
            agegroup_component.age_max else 89
        coverage_times, coverage_values = get_both_vacc_coverage(
            cluster_name.upper(),
            start_age=close_enough_age_min,
            end_age=close_enough_age_max,
        )

        # Manually adjust according to the proportion of the age band that the group is referring to - should generalise
        adjustment = 0.6 if close_enough_age_min == 12 and close_enough_age_max == 15 else 1.
        coverage_values *= adjustment

        # Stop at the end of the available data, even if the request is later
        end_time = min((max(coverage_times), agegroup_component.vic_supply.end_time))

        # Get the vaccination rate function of time
        get_vaccination_rate = get_piecewise_vacc_func(
            agegroup_component.vic_supply.start_time, end_time, agegroup_component.vic_supply.time_interval,
            coverage_times, coverage_values, vacc_params.lag
        )

        # Apply the vaccination rate function to the model
        add_vacc_flows(model, working_agegroups, get_vaccination_rate, Vaccination.ONE_DOSE_ONLY)

    # Add blank/zero flows to make the output requests simpler
    ineligible_ages = set(AGEGROUP_STRATA) - set(all_eligible_agegroups)
    add_vacc_flows(model, ineligible_ages, 0., Vaccination.ONE_DOSE_ONLY)


def add_vic2021_supermodel_vacc(model: CompartmentalModel, vacc_params, cluster_strata: str):
    """
    *** This appears to be working, but would need to be checked if we went back to using this approach ***
    """

    for roll_out_component in vacc_params.roll_out_components:

        # Work out eligible model age_groups
        eligible_age_groups = get_eligible_age_groups(roll_out_component)

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

            get_vaccination_rate = get_piecewise_vacc_func(
                roll_out_component.vic_supply.start_time, end_time, roll_out_component.vic_supply.time_interval,
                coverage_times, coverage_values, vacc_params.lag,
            )
            add_vacc_flows(
                model, eligible_age_groups, get_vaccination_rate, Vaccination.ONE_DOSE_ONLY,
                extra_stratum=cluster_stratum
            )

            # Add blank flows to make things simpler when we come to doing the outputs
            ineligible_ages = set(AGEGROUP_STRATA) - set(eligible_age_groups)
            add_vacc_flows(model, ineligible_ages, 0., Vaccination.ONE_DOSE_ONLY, extra_stratum=cluster_stratum)
