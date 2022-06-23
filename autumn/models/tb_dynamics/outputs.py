import numpy as np

from typing import List
from summer import CompartmentalModel

from autumn.model_features.curve import tanh_based_scaleup

from .constants import BASE_COMPARTMENTS, Compartment, INFECTIOUS_COMPS

def request_outputs(
    model: CompartmentalModel,
    location_strata: List[str],
):
    out = TbOutputBuilder(model, location_strata)

    # Population
    out.request_compartment_output("population_size", BASE_COMPARTMENTS)

    # Percentage latent
   

class TbOutputBuilder:
    """Helps build derived outputs for the TB model"""

    def __init__(self, model, location_strata) -> None:
        self.model = model
        self.locs = location_strata

    def _normalise_timestep(self, vals):
        """Normalise flow outputs to be 'per unit time (year)'"""
        return vals / self.model.timestep


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

