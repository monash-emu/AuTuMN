import numpy as np

from apps.covid_19.constants import Compartment, VACCINE_ELIGIBLE_COMPARTMENTS
from autumn.curve.scale_up import scale_up_function


def get_vacc_roll_out_function_from_coverage(supply_params):
    """
    Work out the time-variant vaccination rate based on a requested coverage and roll-out window.
    Return a function of time.
    """
    # Get vaccination parameters
    coverage = supply_params.coverage
    start_time = supply_params.start_time
    end_time = supply_params.end_time
    duration = end_time - start_time
    assert end_time >= start_time
    assert 0.0 <= coverage <= 1.0

    # Calculate the vaccination rate from the coverage and the duration of the program
    vaccination_rate = -np.log(1.0 - coverage) / duration

    def get_vaccination_rate(time):
        return vaccination_rate if start_time < time < end_time else 0.0

    return get_vaccination_rate


def get_vacc_roll_out_function_from_doses(
        time_variant_supply,
        compartment_name,
        eligible_age_group,
        eligible_age_groups
):
    """
    Work out the number of vaccinated individuals for a given agegroup and compartment name.
    Return a time-variant function in a format that can be used to inform a functional flow.
    """

    def net_flow_func(model, compartments, compartment_values, flows, flow_rates, time):
        # work out the proportion of the eligible population that is in the relevant compartment
        # FIXME: we should be able to cache the two lists below. They depend on compartment_name, eligible_age_group
        #  and eligible_age_groups but not on time!
        num_indices = []
        deno_indices = []
        for i, compartment in enumerate(compartments):
            if compartment.name in VACCINE_ELIGIBLE_COMPARTMENTS and compartment.strata['agegroup'] in eligible_age_groups and compartment.strata['immunity'] == "unvaccinated":
                deno_indices.append(i)
                if compartment.name == compartment_name and compartment.strata['agegroup'] == eligible_age_group:
                    num_indices.append(i)

        compartment_size = sum([compartment_values[i] for i in num_indices])
        total_eligible_pop_size = sum([compartment_values[i] for i in deno_indices])
        if total_eligible_pop_size > .1:
            nb_vaccinated = compartment_size / total_eligible_pop_size * time_variant_supply(time)
        else:
            nb_vaccinated = 0.

        return max(0, min(nb_vaccinated, compartment_size))

    return net_flow_func


def get_eligible_age_groups(roll_out_component, age_strata):
    """
    return a list with the model's age groups that are relevant to the requested roll_out_component
    """
    eligible_age_groups = age_strata
    if roll_out_component.age_min:
        eligible_age_groups = [
            agegroup for agegroup in eligible_age_groups if float(agegroup) >= roll_out_component.age_min
        ]
    if roll_out_component.age_max:
        eligible_age_groups = [
            agegroup for agegroup in eligible_age_groups if float(agegroup) < roll_out_component.age_max
        ]
    return eligible_age_groups


def add_vaccination_flows(model, roll_out_component, age_strata):
    """
    Add the vaccination flows associated with a vaccine roll-out component (i.e. a given age-range and supply function)
    """
    # Is vaccine supply informed by final coverage or daily doses available
    is_coverage = bool(roll_out_component.supply_period_coverage)
    if is_coverage:
        vaccination_roll_out_function = get_vacc_roll_out_function_from_coverage(roll_out_component.supply_period_coverage)
    else:
        time_variant_supply = scale_up_function(
            roll_out_component.supply_timeseries.times,
            roll_out_component.supply_timeseries.values,
            method=4
        )

    # work out eligible model age_groups
    eligible_age_groups = get_eligible_age_groups(roll_out_component, age_strata)

    for eligible_age_group in eligible_age_groups:
        _source_strata = {"immunity": "unvaccinated", "agegroup": eligible_age_group}
        _dest_strata = {"immunity": "vaccinated", "agegroup": eligible_age_group}
        for compartment in VACCINE_ELIGIBLE_COMPARTMENTS:
            if is_coverage:
                # the roll-out function is applied as a rate that multiplies the source compartments
                model.add_fractional_flow(
                    name="vaccination",
                    fractional_rate=vaccination_roll_out_function,
                    source=compartment,
                    dest=compartment,
                    source_strata=_source_strata,
                    dest_strata=_dest_strata,
                )
            else:
                # we need to create a functional flow, which depends on the agegroup and the compartment considered
                vaccination_roll_out_function = get_vacc_roll_out_function_from_doses(
                    time_variant_supply,
                    compartment,
                    eligible_age_group,
                    eligible_age_groups
                )

                model.add_function_flow(
                    name="vaccination",
                    flow_rate_func=vaccination_roll_out_function,
                    source=compartment,
                    dest=compartment,
                    source_strata=_source_strata,
                    dest_strata=_dest_strata,
                )
