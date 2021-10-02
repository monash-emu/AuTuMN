import numpy as np

from autumn.models.covid_19.constants import (
    VACCINE_ELIGIBLE_COMPARTMENTS, Vaccination, INFECTIOUSNESS_ONSET, INFECT_DEATH, PROGRESS, RECOVERY, COMPARTMENTS
)
from autumn.tools.curve.scale_up import scale_up_function
from autumn.models.covid_19.stratifications.clinical import CLINICAL_STRATA
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.preprocess.clinical import get_all_adjustments
from autumn.tools.inputs.covid_au.queries import get_dhhs_vaccination_numbers


def get_vacc_roll_out_function_from_coverage(coverage, start_time, end_time, coverage_override=None):
    """
    Calculate the time-variant vaccination rate based on a requested coverage and roll-out window.
    Return the stepped function of time.
    """

    # Get vaccination parameters
    coverage = coverage if coverage else coverage_override
    duration = end_time - start_time
    assert duration >= 0.
    assert 0. <= coverage <= 1.

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


def add_vaccine_infection_and_severity(vacc_prop_prevent_infection, overall_efficacy):
    """
    Calculating the vaccine efficacy in preventing infection and leading to severe infection.
    """

    if vacc_prop_prevent_infection == 1.:
        severity_efficacy = 0.
    else:
        prop_infected = 1. - vacc_prop_prevent_infection
        prop_infect_prevented = 1. - vacc_prop_prevent_infection * overall_efficacy
        severity_efficacy = overall_efficacy * prop_infected / prop_infect_prevented
    infection_efficacy = vacc_prop_prevent_infection * overall_efficacy

    return infection_efficacy, severity_efficacy


def add_clinical_adjustments_to_strat(
        strat, unaffected_stratum, first_modified_stratum, params, symptomatic_adjuster, hospital_adjuster,
        ifr_adjuster, top_bracket_overwrite, second_modified_stratum=None, second_sympt_adjuster=1.,
        second_hospital_adjuster=1., second_ifr_adjuster=1., second_top_bracket_overwrite=None,
):
    """
    Get all the adjustments in the same way for both the history and vaccination stratifications.
    """

    entry_adjs, death_adjs, progress_adjs, recovery_adjs, _, _ = get_all_adjustments(
        params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
        params.sojourn, params.testing_to_detection, ifr_adjuster, symptomatic_adjuster,
        hospital_adjuster, top_bracket_overwrite,
    )

    # Make these calculations for the one-dose stratum, even if this is being called by the history stratification
    second_entry_adjs, second_death_adjs, second_progress_adjs, second_recovery_adjs, _, _ = get_all_adjustments(
        params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
        params.sojourn, params.testing_to_detection, second_ifr_adjuster, second_sympt_adjuster,
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
        cluster_stratum={},
):
    """
    Add the vaccination flows associated with a vaccine roll-out component (i.e. a given age-range and supply function)
    """

    # cluster_stratum = {"cluster": additional_strata} if additional_strata else {}

    # First phase of the Victorian roll-out, informed by vaccination data
    if roll_out_component.vic_supply_to_history:

        # Get the cluster-specific historical vaccination numbers
        coverage = get_dhhs_vaccination_numbers(
            vic_cluster.upper(),
            start_age=roll_out_component.age_min
        )[1].max()

        # Make sure we're dealing with reasonably sensible coverage values and place a ceiling just in case
        assert 0. <= coverage <= 1.
        sensible_coverage = min(coverage, 0.96)

        # Create the function
        vaccination_roll_out_function = get_vacc_roll_out_function_from_coverage(
            sensible_coverage,
            roll_out_component.vic_supply_to_history.start_time,
            roll_out_component.vic_supply_to_history.end_time,
        )

    elif roll_out_component.vic_supply_to_target:

        # Calculate the most recent statewide coverage
        statewide_coverage = get_dhhs_vaccination_numbers(
            start_age=roll_out_component.age_min
        )[1].max()

        # Increase to the end of the simulation period, making sure it is an increase
        coverage = max((roll_out_component.vic_supply_to_target.coverage - statewide_coverage), 0.) / \
                   (1. - statewide_coverage)

        # Make sure we're dealing with reasonably sensible coverage values
        assert 0. <= coverage <= 1.
        sensible_coverage = min(coverage, 0.96)

        # Create the function
        vaccination_roll_out_function = get_vacc_roll_out_function_from_coverage(
            sensible_coverage,
            roll_out_component.vic_supply_to_target.start_time,
            roll_out_component.vic_supply_to_target.end_time,
        )

    elif roll_out_component.vic_supply_region_to_target:

        # Get the cluster-specific historical vaccination numbers
        previous_coverage = get_dhhs_vaccination_numbers(
            vic_cluster.upper(),
            start_age=roll_out_component.age_min
        )[1].max()

        # Increase to the end of the simulation period, making sure it is an increase
        coverage = max((roll_out_component.vic_supply_region_to_target.coverage - previous_coverage), 0.) / \
                   (1. - previous_coverage)

        # Make sure we're dealing with reasonably sensible coverage values
        assert 0. <= coverage <= 1.
        sensible_coverage = min(coverage, 0.96)

        # Create the function
        vaccination_roll_out_function = get_vacc_roll_out_function_from_coverage(
            sensible_coverage,
            roll_out_component.vic_supply_region_to_target.start_time,
            roll_out_component.vic_supply_region_to_target.end_time,
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

    # Work out eligible model age_groups
    eligible_age_groups, ineligible_age_groups = get_eligible_age_groups(roll_out_component, age_strata)

    # Find vaccination destination stratum, depending on whether one-dose vaccination stratum is active
    vacc_dest_stratum = Vaccination.ONE_DOSE_ONLY if one_dose else Vaccination.VACCINATED
    for compartment in VACCINE_ELIGIBLE_COMPARTMENTS:

        for eligible_age_group in eligible_age_groups:
            _source_strata = {"vaccination": Vaccination.UNVACCINATED, "agegroup": eligible_age_group}
            _source_strata.update(cluster_stratum)
            _dest_strata = {"vaccination": vacc_dest_stratum, "agegroup": eligible_age_group}
            _dest_strata.update(cluster_stratum)
            if roll_out_component.supply_period_coverage or \
                    roll_out_component.vic_supply_to_target or \
                    roll_out_component.vic_supply_to_history or \
                    roll_out_component.vic_supply_region_to_target:

                # The roll-out function is applied as a rate that multiplies the source compartments
                model.add_transition_flow(
                    name="vaccination",
                    fractional_rate=vaccination_roll_out_function,
                    source=compartment,
                    dest=compartment,
                    source_strata=_source_strata,
                    dest_strata=_dest_strata,
                )
            else:
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
            _dest_strata = {"vaccination": vacc_dest_stratum, "agegroup": ineligible_age_group}
            _dest_strata.update(cluster_stratum)
            model.add_transition_flow(
                name="vaccination",
                fractional_rate=0.,
                source=compartment,
                dest=compartment,
                source_strata=_source_strata,
                dest_strata=_dest_strata,
            )
