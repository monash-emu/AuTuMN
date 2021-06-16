from typing import List
from summer import CompartmentalModel

from autumn.tools.curve import tanh_based_scaleup

from .constants import COMPARTMENTS, Compartment, INFECTIOUS_COMPS


def request_outputs(
    model: CompartmentalModel,
    cumulative_start_time: float,
    location_strata: List[str],
    time_variant_tb_screening_rate,
):
    request_compartment_output_plus_locations(
        model, "population_size", COMPARTMENTS, location_strata
    )

    # Percentage latent
    latent_comps = [Compartment.LATE_LATENT, Compartment.EARLY_LATENT]
    request_compartment_output_plus_locations(
        model, "latent_population_size", latent_comps, location_strata, save_results=False
    )
    request_output_func_plus_locations(
        model,
        "percentage_latent",
        calculate_percentage,
        ["latent_population_size", "population_size"],
        location_strata,
    )

    # Deaths
    request_flow_output_plus_locations(
        model, "mortality_infectious", "infect_death", location_strata
    )
    request_flow_output_plus_locations(
        model, "mortality_on_treatment", "treatment_death", location_strata
    )
    request_aggregation_output_plus_locations(
        model,
        "mortality_raw",
        ["mortality_infectious", "mortality_on_treatment"],
        location_strata,
        save_results=False,
    )
    request_output_func_plus_locations(
        model,
        "mortality",
        calculate_per_hundred_thousand,
        ["mortality_raw", "population_size"],
        location_strata,
    )
    model.request_cumulative_output(
        "cumulative_deaths",
        "mortality_raw",
        start_time=cumulative_start_time,
    )

    # Disease incidence
    request_flow_output_plus_locations(
        model, "incidence_early", "early_activation", location_strata
    )
    request_flow_output_plus_locations(model, "incidence_late", "late_activation", location_strata)
    request_aggregation_output_plus_locations(
        model,
        "incidence_raw",
        ["incidence_early", "incidence_late"],
        location_strata,
        save_results=False,
    )
    request_output_func_plus_locations(
        model,
        "incidence",
        calculate_per_hundred_thousand,
        ["incidence_raw", "population_size"],
        location_strata,
    )
    model.request_cumulative_output(
        "cumulative_diseased",
        "incidence_raw",
        start_time=cumulative_start_time,
    )

    # Prevalence infectious
    request_compartment_output_plus_locations(
        model, "infectious_population_size", INFECTIOUS_COMPS, location_strata, save_results=False
    )
    request_output_func_plus_locations(
        model,
        "prevalence_infectious",
        calculate_per_hundred_thousand,
        ["infectious_population_size", "population_size"],
        location_strata,
    )

    # Notifications
    # FIXME: Not seeing same crazy spike
    # TODO: Investigate
    request_flow_output_plus_locations(model, "notifications", "detection", location_strata)

    # Screening rate
    screening_rate_func = tanh_based_scaleup(
        time_variant_tb_screening_rate["shape"],
        time_variant_tb_screening_rate["inflection_time"],
        time_variant_tb_screening_rate["lower_asymptote"],
        time_variant_tb_screening_rate["upper_asymptote"],
    )

    def get_screening_rate():
        return screening_rate_func(model.times)

    model.request_function_output("screening_rate", get_screening_rate, [])


def request_flow_output_plus_locations(model, output_name, flow_name, location_strata):
    model.request_output_for_flow(output_name, flow_name)
    for location_stratum in location_strata:
        loc_output_name = f"{output_name}Xlocation_{location_stratum}"
        model.request_output_for_flow(
            loc_output_name, flow_name, source_strata={"location": location_stratum}
        )


def request_aggregation_output_plus_locations(
    model, output_name, sources, location_strata, save_results=True
):
    model.request_aggregate_output(output_name, sources, save_results=save_results)
    for location_stratum in location_strata:
        # For location-specific mortality calculations
        loc_output_name = f"{output_name}Xlocation_{location_stratum}"
        loc_sources = [f"{s}Xlocation_{location_stratum}" for s in sources]
        model.request_aggregate_output(loc_output_name, loc_sources, save_results=save_results)


def request_output_func_plus_locations(model, output_name, func, sources, location_strata):
    model.request_function_output(output_name, func, sources)
    for location_stratum in location_strata:
        loc_output_name = f"{output_name}Xlocation_{location_stratum}"
        loc_sources = [f"{s}Xlocation_{location_stratum}" for s in sources]
        model.request_function_output(loc_output_name, func, loc_sources)


def calculate_per_hundred_thousand(sub_pop_size, total_pop_size):
    return 1e5 * sub_pop_size / total_pop_size


def calculate_percentage(sub_pop_size, total_pop_size):
    return 100 * sub_pop_size / total_pop_size


def request_compartment_output_plus_locations(
    model, output_name, compartments, location_strata, save_results=True
):
    model.request_output_for_compartments(output_name, compartments, save_results=save_results)
    for location_stratum in location_strata:
        # For location-specific mortality calculations
        loc_output_name = f"{output_name}Xlocation_{location_stratum}"
        model.request_output_for_compartments(
            loc_output_name,
            compartments,
            strata={"location": location_stratum},
            save_results=save_results,
        )
