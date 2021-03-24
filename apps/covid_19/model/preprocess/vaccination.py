import numpy as np

from apps.covid_19.constants import Compartment, VACCINE_ELIGIBLE_COMPARTMENTS
from autumn.curve.scale_up import scale_up_function


def check_vaccination_params(vacc_params):
    pass  # FIXME: Would be good to check a few things here


def get_vacc_roll_out_function_from_coverage(params):

    # Get vaccination parameters
    coverage = params.coverage
    start_time = params.start_time
    end_time = params.end_time
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

    def net_flow_func(compartments, compartment_values, flows, flow_rates, time):
        # work out the proportion of the eligible population that is in the relevant compartment
        num_indices = []
        deno_indices = []
        for i, compartment in enumerate(compartments):
            if compartment.name in VACCINE_ELIGIBLE_COMPARTMENTS and compartment.strata['agegroup'] in eligible_age_groups and compartment.strata['immunity'] == "unvaccinated":
                deno_indices.append(i)
                if compartment.name == compartment_name and compartment.strata['agegroup'] == eligible_age_group:
                    num_indices.append(i)

        compartment_size = sum([compartment_values[i] for i in num_indices])
        total_eligible_pop_size = sum([compartment_values[i] for i in deno_indices])
        rel_prop = compartment_size / total_eligible_pop_size

        nb_vaccinated = rel_prop * time_variant_supply(time)
        nb_vaccinated = min(nb_vaccinated, compartment_size)

        return nb_vaccinated

    return net_flow_func


def get_eligible_age_groups(roll_out_component, age_strata):
    """
    return a list with the age groups that are relevant to the requested roll_out_component
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
    # Is vaccine supply informed by final coverage or daily doses available
    is_coverage = "coverage" in list(roll_out_component.supply.fields.keys())

    if is_coverage:
        vaccination_roll_out_function = get_vacc_roll_out_function_from_coverage(roll_out_component.supply)
    else:
        time_variant_supply = scale_up_function(
            roll_out_component.supply.times,
            roll_out_component.supply.values,
            method=4
        )

    # work out eligible model age_groups
    if roll_out_component.age_min or roll_out_component.age_max:
        eligible_age_groups = get_eligible_age_groups(roll_out_component, age_strata)
    else:
        eligible_age_groups = ["all"]

    for eligible_age_group in eligible_age_groups:
        if eligible_age_group == "all":
            _source_strata = {"immunity": "unvaccinated"}
            _dest_strata = {"immunity": "vaccinated"}
        else:
            _source_strata = {"immunity": "unvaccinated", "agegroup": eligible_age_group}
            _dest_strata = {"immunity": "vaccinated", "agegroup": eligible_age_group}

        for compartment in VACCINE_ELIGIBLE_COMPARTMENTS:
            if is_coverage:
                model.add_fractional_flow(
                    name="vaccination",
                    fractional_rate=vaccination_roll_out_function,
                    source=compartment,
                    dest=compartment,
                    source_strata=_source_strata,
                    dest_strata=_dest_strata,
                )
            else:
                # we first need to work out the number of vaccinated people from this particular compartment and this age
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
