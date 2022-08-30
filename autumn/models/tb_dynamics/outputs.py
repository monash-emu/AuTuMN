from summer import CompartmentalModel
from autumn.model_features.outputs import OutputsBuilder
from .constants import Compartment, LATENT_COMPS, INFECTIOUS_COMPS


def request_outputs(model: CompartmentalModel, compartments: Compartment):
    output_builder = TbOutputBuilder(model, compartments)
    
    # Latency
    output_builder.request_compartment_output("latent_population_size", LATENT_COMPS, save_results=False)
    sources = ["latent_population_size", "total_population"]
    output_builder.request_output_func("percentage_latent", calculate_percentage, sources)

    # Prevalence
    output_builder.request_compartment_output(
        "infectious_population_size", INFECTIOUS_COMPS, save_results=False
    )
    sources = ["infectious_population_size", "total_population"]
    output_builder.request_output_func("prevalence_infectious", calculate_per_hundred_thousand, sources)

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

    def request_flow_output(self, output_name, flow_name, save_results=True):
        self.model.request_output_for_flow(output_name, flow_name, save_results=save_results)

    def request_aggregation_output(self, output_name, sources, save_results=True):
        self.model.request_aggregate_output(output_name, sources, save_results=save_results)


def calculate_percentage(sub_pop_size, total_pop_size):
    return 100 * sub_pop_size / total_pop_size

def calculate_per_hundred_thousand(sub_pop_size, total_pop_size):
    return 1e5 * sub_pop_size / total_pop_size

def calculate_proportion(sub_pop_size, total_pop_size):
    return sub_pop_size / total_pop_size
