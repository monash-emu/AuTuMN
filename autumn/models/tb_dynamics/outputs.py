from summer import CompartmentalModel
from autumn.model_features.outputs import OutputsBuilder
import numpy as np
from .constants import BASE_COMPARTMENTS, LATENT_COMPS, INFECTIOUS_COMPS


def request_outputs(
    model: CompartmentalModel, cumulative_start_time: float
):
    output_builder = TbOutputBuilder(model)
    output_builder.request_compartment_output("total_population", BASE_COMPARTMENTS)
    # Latency
    output_builder.request_compartment_output(
        "latent_population_size", LATENT_COMPS, save_results=False
    )
    sources = ["latent_population_size", "total_population"]
    output_builder.request_output_func("percentage_latent", calculate_percentage, sources)

    # Prevalence
    output_builder.request_compartment_output(
        "infectious_population_size", INFECTIOUS_COMPS, save_results=False
    )
    sources = ["infectious_population_size", "total_population"]
    output_builder.request_output_func(
        "prevalence_infectious", calculate_per_hundred_thousand, sources
    )

    # Death
    output_builder.request_flow_output(
        "mortality_infectious_raw", "infect_death", save_results=False
    )
   
    sources = ["mortality_infectious_raw"]
    output_builder.request_aggregation_output(
        "mortality_raw",
        sources,
        save_results=False
    )
    model.request_cumulative_output(
        "cumulative_deaths",
        "mortality_raw",
        start_time=cumulative_start_time,
    )
    # Disease incidence
    output_builder.request_flow_output(
        "incidence_early_raw", "early_activation", save_results=False
    )
    output_builder.request_flow_output("incidence_late_raw", "late_activation", save_results=False)
    sources = ["incidence_early_raw", "incidence_late_raw"]
    output_builder.request_aggregation_output("incidence_raw", sources, save_results=False)
    sources = ["incidence_raw", "total_population"]
    model.request_cumulative_output(
        "cumulative_diseased",
        "incidence_raw",
        start_time=cumulative_start_time,
    )
    # Normalise incidence so that it is per unit time (year), not per timestep
    output_builder.request_normalise_flow_output("incidence_early", "incidence_early_raw")
    output_builder.request_normalise_flow_output("incidence_late", "incidence_late_raw")
    output_builder.request_normalise_flow_output(
        "incidence_norm", "incidence_raw", save_results=False
    )
    sources = ["incidence_norm", "total_population"]
    output_builder.request_output_func("incidence", calculate_per_hundred_thousand, sources)

       # Notifications (normalized to per year)
    output_builder.request_flow_output("passive_notifications_raw", "detection", save_results=False)

    sources = ["passive_notifications_raw"]
    output_builder.request_aggregation_output("notifications_raw", sources, save_results=False)
    output_builder.request_normalise_flow_output("notifications", "notifications_raw")


class TbOutputBuilder(OutputsBuilder):
    """Helps build derived outputs for the TB model"""

    def __init__(self, model) -> None:
        self.model = model

    def request_compartment_output(self, output_name, compartments, save_results=True):
        self.model.request_output_for_compartments(
            output_name, compartments, save_results=save_results
        )

    def request_output_func(self, output_name, func, sources, save_results=True):
        self.model.request_function_output(output_name, func, sources, save_results=save_results)

    def request_flow_output(self, output_name, flow_name, save_results=True):
        self.model.request_output_for_flow(output_name, flow_name, save_results=save_results)

    def request_aggregation_output(self, output_name, sources, save_results=True):
        self.model.request_aggregate_output(output_name, sources, save_results=save_results)

    def _normalise_timestep(self, vals):
        """Normalise flow outputs to be 'per unit time (year)'"""
        return vals / self.model.timestep

    def request_normalise_flow_output(self, output_name, source, save_results=True):
        self.request_output_func(
            output_name, self._normalise_timestep, [source], save_results=save_results
        )


def calculate_percentage(sub_pop_size, total_pop_size):
    return 100 * sub_pop_size / total_pop_size


def calculate_per_hundred_thousand(sub_pop_size, total_pop_size):
    return 1e5 * sub_pop_size / total_pop_size


def calculate_proportion(sub_pop_size, total_pop_size):
    return sub_pop_size / total_pop_size
