import numpy as np
from typing import List
from summer import CompartmentalModel
from autumn.models.tb_dynamics.parameters import Sojourns

from autumn.models.tb_dynamics.parameters import Parameters
from autumn.core.project import Params, build_rel_path
from autumn.model_features.curve import scale_up_function, tanh_based_scaleup
from autumn.core import inputs
from autumn.core.inputs.social_mixing.queries import get_prem_mixing_matrices
from autumn.core.inputs.social_mixing.build_synthetic_matrices import (
    build_synthetic_matrices,
)

from .constants import Compartment, BASE_COMPARTMENTS, INFECTIOUS_COMPS

from .outputs import request_outputs

base_params = Params(
    build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False
)


def build_model(params: dict, build_options: dict = None) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.
    """
    params = Parameters(**params)
    time = params.time
    model = CompartmentalModel(
        times=[time.start, time.end],
        compartments=BASE_COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPS,
        timestep=time.step,
    )
    init_pop = {
        Compartment.INFECTIOUS: params.infectious_seed,
        Compartment.SUSCEPTIBLE: params.start_population_size - params.infectious_seed,
    }
    """Assign the initial population"""
    model.set_initial_population(init_pop)

    birth_rates, years = inputs.get_crude_birth_rate(params.iso3)
    birth_rates = [
        b / 1000.0 for b in birth_rates
    ]  # Birth rates are provided / 1000 population
    crude_birth_rate = scale_up_function(years, birth_rates, smoothness=0.2, method=5)
    model.add_crude_birth_flow(
        "birth",
        crude_birth_rate,
        Compartment.SUSCEPTIBLE,
    )
    # # Infection flows.

    # Derived outputs
    request_outputs(model)

    return model
