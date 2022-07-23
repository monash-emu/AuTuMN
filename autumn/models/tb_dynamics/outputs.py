from summer import CompartmentalModel

from .constants import BASE_COMPARTMENTS


def request_outputs(model: CompartmentalModel):
    out = TbOutputBuilder(model)

    # Population
    out.request_compartment_output("population_size", BASE_COMPARTMENTS)


class TbOutputBuilder:
    """Helps build derived outputs for the TB model"""

    def __init__(self, model) -> None:
        self.model = model

    def _normalise_timestep(self, vals):
        """Normalise flow outputs to be 'per unit time (year)'"""
        return vals / self.model.timestep

    def request_compartment_output(self, output_name, compartments, save_results=True):
        self.model.request_output_for_compartments(
            output_name, compartments, save_results=save_results
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
        
