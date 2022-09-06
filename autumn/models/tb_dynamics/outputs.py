from summer import CompartmentalModel
from autumn.model_features.outputs import OutputsBuilder
from .constants import Compartment, LATENT_COMPS


def request_outputs(model: CompartmentalModel, compartments: Compartment):
    output_builder = TbOutputBuilder(model, compartments)
    # Latency related
    output_builder.request_compartment_output("latent_population_size", LATENT_COMPS, save_results=False)
    sources = ["latent_population_size", "total_population"]
    output_builder.request_output_func("percentage_latent", calculate_percentage, sources)


class TbOutputBuilder(OutputsBuilder):
    """Helps build derived outputs for the TB model"""

    #def __init__(self, model) -> None:
    #    self.model = model

    def request_compartment_output(self, output_name, compartments, save_results=True):
        self.model.request_output_for_compartments(
            output_name, compartments, save_results=save_results
        )

    def request_output_func(self, output_name, func, sources, save_results=True):
        self.model.request_function_output(output_name, func, sources, save_results=save_results)


def calculate_percentage(sub_pop_size, total_pop_size):
    return 100 * sub_pop_size / total_pop_size
