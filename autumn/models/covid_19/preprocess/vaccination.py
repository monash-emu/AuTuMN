import numpy as np

from autumn.models.covid_19.constants import (
    VACCINE_ELIGIBLE_COMPARTMENTS, Vaccination, INFECTIOUSNESS_ONSET, INFECT_DEATH, PROGRESS, RECOVERY
)
from autumn.tools.curve.scale_up import scale_up_function
from autumn.models.covid_19.stratifications.clinical import CLINICAL_STRATA
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.preprocess.clinical import get_all_adjustments
from autumn.tools.inputs.covid_au.queries import get_both_vacc_coverage, VACC_COVERAGE_START_AGES, VACC_COVERAGE_END_AGES


def get_vacc_roll_out_function_from_coverage(coverage, start_time, end_time, coverage_override=None):
    """
    Calculate the time-variant vaccination rate based on a requested coverage and roll-out window.
    Return the stepped function of time.
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
        return vaccination_rate if start_time < time < end_time else 0.

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


def get_eligible_age_groups(roll_out_component, age_strata):
    """
    Return a list with the model's age groups that are relevant to the requested roll_out_component.
    Also return the ineligible age groups so that we can apply vaccination to them as well to simplify loops later.
    """

    eligible_age_groups, ineligible_age_groups = [], []
    for agegroup in age_strata:

        # Either not requested, or requested and meets that age cut-off for min or max
        above_age_min = \
            not roll_out_component.age_min or \
            bool(roll_out_component.age_min) and float(agegroup) >= roll_out_component.age_min
        below_age_max = \
            not roll_out_component.age_max or \
            bool(roll_out_component.age_max) and float(agegroup) < roll_out_component.age_max
        if above_age_min and below_age_max:
            eligible_age_groups.append(agegroup)
        else:
            ineligible_age_groups.append(agegroup)

    return eligible_age_groups, ineligible_age_groups


def find_vaccine_action(vacc_prop_prevent_infection, overall_efficacy):
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


def add_vaccination_flows(
        model, roll_out_component, age_strata, one_dose, coverage_override=None, vic_cluster=None,
        cluster_stratum={}, vaccination_lag=0.,
):
    """
    Add the vaccination flows associated with a vaccine roll-out component (i.e. a given age-range and supply function)
    """

    # Work out eligible model age_groups
    eligible_age_groups, ineligible_age_groups = get_eligible_age_groups(roll_out_component, age_strata)

    # Find vaccination destination stratum, depending on whether one-dose vaccination stratum is active
    vacc_dest_stratum = Vaccination.ONE_DOSE_ONLY if one_dose else Vaccination.VACCINATED

    # First phase of the Victorian roll-out, informed by vaccination data
    if roll_out_component.vic_supply:

        if roll_out_component.age_min:
            close_enough_age_min = min(VACC_COVERAGE_START_AGES, key=lambda age: abs(age - roll_out_component.age_min))
        else:
            close_enough_age_min = 0

        if roll_out_component.age_max:
            close_enough_age_max = min(VACC_COVERAGE_END_AGES, key=lambda age: abs(age - roll_out_component.age_max))
        else:
            close_enough_age_max = 89

        # Get the cluster-specific historical vaccination numbers
        coverage_times, coverage_values = get_both_vacc_coverage(
            vic_cluster.upper(),
            start_age=close_enough_age_min,
            end_age=close_enough_age_max,
        )

        # Manually adjust according to the proportion of the age band that the group is referring to
        adjustment = 0.6 if close_enough_age_min == 12 and close_enough_age_max == 15 else 1.
        coverage_values *= adjustment

        # Stop at the end of the available data, even if the request is later
        final_time = min((max(coverage_times), roll_out_component.vic_supply.end_time))
        rollout_period_times = np.linspace(
            roll_out_component.vic_supply.start_time,
            final_time,
            int(roll_out_component.vic_supply.time_interval) + 1
        )

        for i_period in range(len(rollout_period_times) - 1):
            period_start_time = rollout_period_times[i_period]
            period_end_time = rollout_period_times[i_period + 1]

            # Interpolate for coverage values, always starting from zero if this is the first roll-out period
            period_start_coverage = np.interp(period_start_time - vaccination_lag, coverage_times, coverage_values)
            modelled_start_coverage = 0. if i_period == 0. else period_start_coverage
            period_end_coverage = np.interp(period_end_time - vaccination_lag, coverage_times, coverage_values)

            # The proportion of the remaining people who will be vaccinated
            coverage_increase = (period_end_coverage - modelled_start_coverage) / (1. - modelled_start_coverage)

            # Make sure we're dealing with reasonably sensible coverage values and place a ceiling just in case
            assert 0. <= coverage_increase <= 1.
            sensible_coverage = min(coverage_increase, 0.96)

            # Create the function
            vaccination_roll_out_function = get_vacc_roll_out_function_from_coverage(
                sensible_coverage, period_start_time, period_end_time,
            )

            # Have to apply within the loop
            for compartment in VACCINE_ELIGIBLE_COMPARTMENTS:
                for eligible_age_group in eligible_age_groups:
                    _source_strata = {"vaccination": Vaccination.UNVACCINATED, "agegroup": eligible_age_group}
                    _source_strata.update(cluster_stratum)
                    _dest_strata = {"vaccination": vacc_dest_stratum, "agegroup": eligible_age_group}
                    _dest_strata.update(cluster_stratum)

                    model.add_transition_flow(
                        name="vaccination",
                        fractional_rate=vaccination_roll_out_function,
                        source=compartment,
                        dest=compartment,
                        source_strata=_source_strata,
                        dest_strata=_dest_strata,
                    )

    # Coverage based vaccination
    elif roll_out_component.supply_period_coverage:
        vaccination_roll_out_function = get_vacc_roll_out_function_from_coverage(
            roll_out_component.supply_period_coverage.coverage,
            roll_out_component.supply_period_coverage.start_time,
            roll_out_component.supply_period_coverage.end_time,
            coverage_override
        )

    # Dose based vaccination
    else:
        time_variant_supply = scale_up_function(
            roll_out_component.supply_timeseries.times,
            roll_out_component.supply_timeseries.values,
            method=4,
        )

    for compartment in VACCINE_ELIGIBLE_COMPARTMENTS:
        for eligible_age_group in eligible_age_groups:
            _source_strata = {"vaccination": Vaccination.UNVACCINATED, "agegroup": eligible_age_group}
            _source_strata.update(cluster_stratum)
            _dest_strata = {"vaccination": vacc_dest_stratum, "agegroup": eligible_age_group}
            _dest_strata.update(cluster_stratum)
            if roll_out_component.supply_period_coverage:

                # The roll-out function is applied as a rate that multiplies the source compartments
                model.add_transition_flow(
                    name="vaccination",
                    fractional_rate=vaccination_roll_out_function,
                    source=compartment,
                    dest=compartment,
                    source_strata=_source_strata,
                    dest_strata=_dest_strata,
                )
            elif roll_out_component.supply_timeseries:
                # We need to create a functional flow, which depends on the agegroup and the compartment considered
                vaccination_roll_out_function = get_vacc_roll_out_function_from_doses(
                    time_variant_supply, compartment, eligible_age_group, eligible_age_groups
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
        for ineligible_age_group in ineligible_age_groups:
            _source_strata = {"vaccination": Vaccination.UNVACCINATED, "agegroup": ineligible_age_group}
            _source_strata.update(cluster_stratum)
            _dest_strata = {"vaccination": Vaccination.ONE_DOSE_ONLY, "agegroup": ineligible_age_group}
            _dest_strata.update(cluster_stratum)
            model.add_transition_flow(
                name="vaccination",
                fractional_rate=0.,
                source=compartment,
                dest=compartment,
                source_strata=_source_strata,
                dest_strata=_dest_strata,
            )


def add_vacc_flows(model, age_groups, vaccination_rate):
    """
    Add blank/zero flows to make things simpler when we come to requesting the outputs.
    """

    for eligible_age_group in age_groups:
        _source_strata = {"vaccination": Vaccination.UNVACCINATED, "agegroup": eligible_age_group}
        _dest_strata = {"vaccination": Vaccination.ONE_DOSE_ONLY, "agegroup": eligible_age_group}
        for compartment in VACCINE_ELIGIBLE_COMPARTMENTS:
            model.add_transition_flow(
                name="vaccination",
                fractional_rate=vaccination_rate,
                source=compartment,
                dest=compartment,
                source_strata=_source_strata,
                dest_strata=_dest_strata,
            )


def find_closest_value_in_list(list_request, value_request):
    return min(list_request, key=lambda list_value: abs(list_value - value_request))


def add_vic_regional_vacc(model, vacc_params, age_strata, one_dose, vic_cluster):

    all_eligible_age_groups = []
    for i_comp, roll_out_component in enumerate(vacc_params.roll_out_components):

        # Work out eligible model age_groups for the current roll-out request
        eligible_age_groups, _ = get_eligible_age_groups(roll_out_component, age_strata)
        msg = "Age group requested multiple times"
        assert not bool(set(all_eligible_age_groups) & set(eligible_age_groups)), msg
        all_eligible_age_groups += eligible_age_groups

        close_enough_age_min = find_closest_value_in_list(VACC_COVERAGE_START_AGES, roll_out_component.age_min) if \
            roll_out_component.age_min else 0
        close_enough_age_max = find_closest_value_in_list(VACC_COVERAGE_END_AGES, roll_out_component.age_max) if \
            roll_out_component.age_max else 89

        # Get the cluster-specific historical vaccination numbers
        coverage_times, coverage_values = get_both_vacc_coverage(
            vic_cluster.upper(),
            start_age=close_enough_age_min,
            end_age=close_enough_age_max,
        )

        # Manually adjust according to the proportion of the age band that the group is referring to
        adjustment = 0.6 if close_enough_age_min == 12 and close_enough_age_max == 15 else 1.
        coverage_values *= adjustment

        # Stop at the end of the available data, even if the request is later
        final_time = min((max(coverage_times), roll_out_component.vic_supply.end_time))
        rollout_period_times = np.linspace(
            roll_out_component.vic_supply.start_time,
            final_time,
            int(roll_out_component.vic_supply.time_interval) + 1
        )

        vaccination_rates = []
        end_times = []

        for i_period in range(len(rollout_period_times) - 1):
            period_start_time = rollout_period_times[i_period]
            period_end_time = rollout_period_times[i_period + 1]

            # Interpolate for coverage values, always starting from zero if this is the first roll-out period
            vaccination_lag = vacc_params.lag
            period_start_coverage = np.interp(period_start_time - vaccination_lag, coverage_times, coverage_values)
            modelled_start_coverage = 0. if i_period == 0. else period_start_coverage
            period_end_coverage = np.interp(period_end_time - vaccination_lag, coverage_times, coverage_values)

            # The proportion of the remaining people who will be vaccinated
            coverage_increase = (period_end_coverage - modelled_start_coverage) / (1. - modelled_start_coverage)
            assert 0. <= coverage_increase <= 1.

            # Create the function - remove this
            vaccination_roll_out_function = get_vacc_roll_out_function_from_coverage(
                coverage_increase, period_start_time, period_end_time,
            )

            end_times.append(period_end_time)

            duration = period_end_time - period_start_time
            assert duration >= 0., f"Vaccination roll-out request duration is negative: {duration}"
            assert 0. <= coverage_increase <= 1., f"Coverage not in [0, 1]: {coverage_increase}"
            vaccination_rate = -np.log(1. - coverage_increase) / duration
            vaccination_rates.append(vaccination_rate)

            # Apply to the model
            add_vacc_flows(model, eligible_age_groups, vaccination_roll_out_function)  # *** need to move this

        def get_vaccination_rate(time):
            if time > roll_out_component.vic_supply.start_time:
                idx = sum([int(end_time < time) for end_time in end_times])
                if idx < len(vaccination_rates):
                    return vaccination_rates[idx]
            return 0.

        for eligible_age_group in eligible_age_groups:
            _source_strata = {"vaccination": Vaccination.UNVACCINATED, "agegroup": eligible_age_group}
            _dest_strata = {"vaccination": Vaccination.ONE_DOSE_ONLY, "agegroup": eligible_age_group}
            for compartment in VACCINE_ELIGIBLE_COMPARTMENTS:
                model.add_transition_flow(
                    name="vaccination",
                    fractional_rate=vaccination_rate,
                    source=compartment,
                    dest=compartment,
                    source_strata=_source_strata,
                    dest_strata=_dest_strata,
                )

    # Add blank/zero flows to make the output requests simpler
    ineligible_ages = set(AGEGROUP_STRATA) - set(all_eligible_age_groups)
    add_vacc_flows(model, ineligible_ages, 0.)
