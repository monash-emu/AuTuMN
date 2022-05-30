import numpy as np

from typing import List
from summer import CompartmentalModel

from autumn.model_features.curve import tanh_based_scaleup

from .constants import COMPARTMENTS, Compartment, INFECTIOUS_COMPS


def request_outputs(
    model: CompartmentalModel,
    cumulative_start_time: float,
    location_strata: List[str],
    time_variant_tb_screening_rate,
    implement_acf: bool,
    implement_ltbi_screening=False,
    pt_efficacy=1.,
    pt_sae_prop=0.
):
    out = OutputBuilder(model, location_strata)

    # Population
    out.request_compartment_output("population_size", COMPARTMENTS)

    # Percentage latent
    latent_comps = [Compartment.LATE_LATENT, Compartment.EARLY_LATENT]
    out.request_compartment_output("latent_population_size", latent_comps, save_results=False)
    sources = ["latent_population_size", "population_size"]
    out.request_output_func("percentage_latent", calculate_percentage, sources)

    # Deaths
    out.request_flow_output("mortality_infectious_raw", "infect_death", save_results=False)
    out.request_flow_output("mortality_on_treatment_raw", "treatment_death", save_results=False)
    sources = ["mortality_infectious_raw", "mortality_on_treatment_raw"]
    out.request_aggregation_output("mortality_raw", sources, save_results=False)
    model.request_cumulative_output(
        "cumulative_deaths",
        "mortality_raw",
        start_time=cumulative_start_time,
    )

    # Normalise mortality so that it is per unit time (year), not per timestep
    out.request_normalise_flow_output("mortality_infectious", "mortality_infectious_raw")
    out.request_normalise_flow_output("mortality_on_treatment", "mortality_on_treatment_raw")
    out.request_normalise_flow_output("mortality_norm", "mortality_raw", save_results=False)
    sources = ["mortality_norm", "population_size"]
    out.request_output_func("mortality", calculate_per_hundred_thousand, sources)

    # Disease incidence
    out.request_flow_output("incidence_early_raw", "early_activation", save_results=False)
    out.request_flow_output("incidence_late_raw", "late_activation", save_results=False)
    sources = ["incidence_early_raw", "incidence_late_raw"]
    out.request_aggregation_output("incidence_raw", sources, save_results=False)
    sources = ["incidence_raw", "population_size"]
    model.request_cumulative_output(
        "cumulative_diseased",
        "incidence_raw",
        start_time=cumulative_start_time,
    )

    # Normalise incidence so that it is per unit time (year), not per timestep
    out.request_normalise_flow_output("incidence_early", "incidence_early_raw")
    out.request_normalise_flow_output("incidence_late", "incidence_late_raw")
    out.request_normalise_flow_output("incidence_norm", "incidence_raw", save_results=False)
    sources = ["incidence_norm", "population_size"]
    out.request_output_func("incidence", calculate_per_hundred_thousand, sources)

    # Prevalence infectious
    out.request_compartment_output(
        "infectious_population_size", INFECTIOUS_COMPS, save_results=False
    )
    sources = ["infectious_population_size", "population_size"]
    out.request_output_func("prevalence_infectious", calculate_per_hundred_thousand, sources)

    # Notifications (normalized to per year)
    out.request_flow_output("passive_notifications_raw", "detection", save_results=False)
    if implement_acf:
        out.request_flow_output("active_notifications_raw", "acf_detection", save_results=False)
    else:
        null_func = lambda: np.zeros_like(model.times)
        out.request_output_func("active_notifications_raw", null_func, [], save_results=False)

    sources = ["passive_notifications_raw", "active_notifications_raw"]
    out.request_aggregation_output("notifications_raw", sources, save_results=False)
    out.request_normalise_flow_output("notifications", "notifications_raw")

    # Screening rate
    screening_rate_func = tanh_based_scaleup(
        time_variant_tb_screening_rate["shape"],
        time_variant_tb_screening_rate["inflection_time"],
        time_variant_tb_screening_rate["start_asymptote"],
        time_variant_tb_screening_rate["end_asymptote"],
    )

    def get_screening_rate():
        return screening_rate_func(model.times)

    model.request_function_output("screening_rate", get_screening_rate, [])

    # Track cumulative number of preventive treatments provided from 2016
    if implement_ltbi_screening:
        model.request_output_for_flow("pt_early_raw", "preventive_treatment_early", save_results=False)
        model.request_output_for_flow("pt_late_raw", "preventive_treatment_late", save_results=False)
        model.request_aggregate_output("pt_raw", ["pt_early_raw", "pt_late_raw"], save_results=False)

        # so far, the pt flows only include succesfully treated individuals, we need tp adjust for efficacy
        model.request_function_output(
            name="pt",
            func=lambda x: x / pt_efficacy,
            sources=["pt_raw"],
            save_results=False,
        )
        model.request_cumulative_output("cumulative_pt", "pt", start_time=2016., save_results=True)
        model.request_function_output(
            name="cumulative_pt_sae",
            func=lambda x: x * pt_sae_prop,
            sources=["cumulative_pt"],
            save_results=True,
        )
    else:  # just record zeroes if PT not implemented
        for zero_output in ["cumulative_pt", "cumulative_pt_sae"]:
            model.request_function_output(
                name=zero_output,
                func=lambda x: x * 0.,  # uses x * 0 so we copy the size of the source output x
                sources=["incidence"],  # could be any source output
                save_results=True,
            )


class OutputBuilder:
    """Helps build derived outputs for the TB model"""

    def __init__(self, model, location_strata) -> None:
        self.model = model
        self.locs = location_strata

    def _normalise_timestep(self, vals):
        """Normalise flow outputs to be 'per unit time (year)'"""
        return vals / self.model.timestep

    def request_normalise_flow_output(self, output_name, source, save_results=True, stratify_by_loc=True):
        self.request_output_func(
            output_name, self._normalise_timestep, [source], save_results=save_results, stratify_by_loc=stratify_by_loc
        )

    def request_flow_output(self, output_name, flow_name, save_results=True):
        self.model.request_output_for_flow(output_name, flow_name, save_results=save_results)
        for location_stratum in self.locs:
            loc_output_name = f"{output_name}Xlocation_{location_stratum}"
            self.model.request_output_for_flow(
                loc_output_name,
                flow_name,
                source_strata={"location": location_stratum},
                save_results=save_results,
            )

    def request_aggregation_output(self, output_name, sources, save_results=True):
        self.model.request_aggregate_output(output_name, sources, save_results=save_results)
        for location_stratum in self.locs:
            # For location-specific mortality calculations
            loc_output_name = f"{output_name}Xlocation_{location_stratum}"
            loc_sources = [f"{s}Xlocation_{location_stratum}" for s in sources]
            self.model.request_aggregate_output(
                loc_output_name, loc_sources, save_results=save_results
            )

    def request_output_func(self, output_name, func, sources, save_results=True, stratify_by_loc=True):
        self.model.request_function_output(output_name, func, sources, save_results=save_results)
        if stratify_by_loc:
            for location_stratum in self.locs:
                loc_output_name = f"{output_name}Xlocation_{location_stratum}"
                loc_sources = [f"{s}Xlocation_{location_stratum}" for s in sources]
                self.model.request_function_output(
                    loc_output_name, func, loc_sources, save_results=save_results
                )

    def request_compartment_output(self, output_name, compartments, save_results=True):
        self.model.request_output_for_compartments(
            output_name, compartments, save_results=save_results
        )
        for location_stratum in self.locs:
            # For location-specific mortality calculations
            loc_output_name = f"{output_name}Xlocation_{location_stratum}"
            self.model.request_output_for_compartments(
                loc_output_name,
                compartments,
                strata={"location": location_stratum},
                save_results=save_results,
            )


def calculate_per_hundred_thousand(sub_pop_size, total_pop_size):
    return 1e5 * sub_pop_size / total_pop_size


def calculate_percentage(sub_pop_size, total_pop_size):
    return 100 * sub_pop_size / total_pop_size
