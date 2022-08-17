from summer import CompartmentalModel

from .constants import BASE_COMPARTMENTS


def request_outputs(model: CompartmentalModel):
    out = TbOutputBuilder(model)

    # Population
    out.request_compartment_output("population_size", BASE_COMPARTMENTS)

    #latency
    latent_comps = [BASE_COMPARTMENTS.LATE_LATENT, BASE_COMPARTMENTS.EARLY_LATENT]
    out.request_compartment_output("latent_population_size", latent_comps, save_results=False)
    sources = ["latent_population_size", "population_size"]
    out.request_output_func("percentage_latent", calculate_percentage, sources)


class TbOutputBuilder:
    """Helps build derived outputs for the TB model"""

    def __init__(self, model) -> None:
        self.model = model


    def request_compartment_output(self, output_name, compartments, save_results=True):
        self.model.request_output_for_compartments(
            output_name, compartments, save_results=save_results
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

def calculate_percentage(sub_pop_size, total_pop_size):
    return 100 * sub_pop_size / total_pop_size

  
        
