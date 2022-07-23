from summer import CompartmentalModel

from autumn.models.tb_dynamics.parameters import Parameters
from autumn.core.project import Params, build_rel_path
from autumn.model_features.curve import scale_up_function
from autumn.core import inputs

from .constants import Compartment, BASE_COMPARTMENTS, INFECTIOUS_COMPS
from .stratifications.age import get_age_strat
from .outputs import request_outputs

base_params = Params(
    build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False
)


def build_model(params: dict, build_options: dict = None) -> CompartmentalModel:
    """Build the compartmental model from the provided parameters.

    Args:
        params (dict): Build the compartmental model from the provided parameters.
        build_options (dict, optional). Defaults to None.

    Returns:
        CompartmentalModel: Returns tb_dynamics Model
    """    
    params = Parameters(**params)
    time = params.time
    model = CompartmentalModel(
        times=[time.start, time.end],
        compartments=BASE_COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPS,
        timestep=time.step,
    )
    init_pop = {Compartment.SUSCEPTIBLE: params.start_population_size}

    """Assign the initial population"""
    model.set_initial_population(init_pop)

    birth_rates, years = inputs.get_crude_birth_rate(params.iso3)
    birth_rates = [(b + 3) / 1000.0  for b in birth_rates]  # Birth rates are provided / 1000 population
    crude_birth_rate = scale_up_function(years, birth_rates, smoothness=0.2, method=5)

    """Add crude birth flow to the model"""
    model.add_crude_birth_flow(
        "birth",
        crude_birth_rate,
        Compartment.SUSCEPTIBLE,
    )

    """Add universal death flow to the model"""
    universal_death_rate = params.crude_death_rate
    model.add_universal_death_flows(
        "universal_death",
        death_rate=universal_death_rate
    )

    """Add Stratification to the model"""
    age_strat  = get_age_strat(params)
    model.stratify_with(age_strat)

    """Generate outputs"""
    request_outputs(model)

    return model
