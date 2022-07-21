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
