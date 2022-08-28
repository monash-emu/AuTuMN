from typing import List
from summer import CompartmentalModel
from autumn.model_features.outputs import OutputsBuilder
from .constants import Compartment, LATENT_COMPS


def request_outputs(
    model: CompartmentalModel,
    compartments: Compartment,
    location_strata):
    output_builder = TbOutputBuilder(model, compartments, location_strata)
    # Latency related
    output_builder.request_compartment_output("latent_population_size", LATENT_COMPS, save_results=False)
    sources = ["latent_population_size", "total_population"]
    output_builder.request_output_func("percentage_latent", calculate_percentage, sources)


class TbOutputBuilder(OutputsBuilder):
    """Helps build derived outputs for the TB model"""

    def __init__(self, model, compartments, location_strata) -> None:
        super().__init__(model, compartments)
        self.locs = location_strata
        print(self.locs)

    def request_compartment_output(self, output_name, compartments, save_results=True):
        self.model.request_output_for_compartments(
            output_name, compartments, save_results=save_results
        )

    def request_output_func(self, output_name, func, sources, save_results=True):
        self.model.request_function_output(output_name, func, sources, save_results=save_results)

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


def calculate_percentage(sub_pop_size, total_pop_size):
    return 100 * sub_pop_size / total_pop_size
