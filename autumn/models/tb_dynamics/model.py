from summer import CompartmentalModel
import pandas as pd

from autumn.models.tb_dynamics.parameters import Parameters
from autumn.core.project import Params, build_rel_path
from autumn.model_features.curve import scale_up_function
from autumn.core import inputs
from autumn.core.inputs.social_mixing.queries import get_prem_mixing_matrices
from autumn.core.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices

from .constants import Compartment, BASE_COMPARTMENTS, INFECTIOUS_COMPS
from .stratifications.age import get_age_strat
from .outputs import request_outputs

base_params = Params(
    build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False
)


def assign_population(seed: float, total_pop: int, model: CompartmentalModel):
    """
    Assign the starting population to the model according to the user requests and total population of the model.

    Args:
        seed: The starting infectious seed
        total_pop: The total population being modelled
        model: The summer compartmental model object to have its starting population set

    """

    # Split by seed and remainder susceptible
    susceptible = total_pop - seed
    init_pop = {
        Compartment.INFECTIOUS: seed,
        Compartment.SUSCEPTIBLE: susceptible,
    }

    # Assign to the model
    model.set_initial_population(init_pop)


def build_model(params: dict, build_options: dict = None) -> CompartmentalModel:
    """Build the compartmental model from the provided parameters.
    Returns:
        CompartmentalModel: Returns tb_dynamics Model
    """
    params = Parameters(**params)
    time = params.time
    iso3 = params.iso3
    seed = params.infectious_seed
    age_breakpoints = [str(age) for age in params.age_breakpoints]

    model = CompartmentalModel(
        times=(time.start, time.end),
        compartments=BASE_COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPS,
        timestep=time.step,
    )
    age_pops = pd.Series(
        inputs.get_population_by_agegroup(age_breakpoints, iso3, None, time.start),
        index=age_breakpoints,
    )
    assign_population(seed, age_pops.sum(), model)

    # Build age mixing matrix

    if params.age_mixing:
        if params.age_mixing.type == "prem":
            age_mixing_matrices = get_prem_mixing_matrices(
                params.iso3, params.age_breakpoints, None
            )
        elif params.age_mixing.type == "extrapolated":
            age_mixing_matrices = build_synthetic_matrices(
                params.iso3,
                params.age_mixing.source_iso3,
                params.age_breakpoints,
                params.age_mixing.age_adjust,
                requested_locations=["all_locations"],
            )
        else:
            raise Exception("Invalid mixing matrix type specified in parameters")
    else:
        # Default to prem matrices (old model runs)
        age_mixing_matrices = get_prem_mixing_matrices(params.iso3, params.age_breakpoints, None)

    age_mixing_matrix = age_mixing_matrices["all_locations"]
    # convert daily contact rates to yearly rates
    age_mixing_matrix *= 365.25

    # Assign the initial population

    birth_rates, years = inputs.get_crude_birth_rate(params.iso3)
    birth_rates = [b / 1000.0 for b in birth_rates]  # Birth rates are provided / 1000 population
    crude_birth_rate = scale_up_function(years, birth_rates, smoothness=0.2, method=5)

    #Add crude birth flow to the model
    model.add_crude_birth_flow(
        "birth",
        crude_birth_rate,
        Compartment.SUSCEPTIBLE,
    )

    # Add universal death flow to the model
    universal_death_rate = params.crude_death_rate
    model.add_universal_death_flows("universal_death", death_rate=universal_death_rate)

    #Add Age stratification to the model
    age_strat = get_age_strat(
        params.age_breakpoints, iso3, age_pops, age_mixing_matrix, BASE_COMPARTMENTS
    )
    model.stratify_with(age_strat)

    # Generate outputs
    request_outputs(model)

    return model
