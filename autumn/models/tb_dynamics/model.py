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


# def assign_population(seed: float, total_pop: int, model: CompartmentalModel):
#     """
#     Assign the starting population to the model according to the user requests and total population of the model.

#     Args:
#         seed: The starting infectious seed
#         total_pop: The total population being modelled
#         model: The summer compartmental model object to have its starting population set

#     """
#     # Split by seed and remainder susceptible
#     susceptible = total_pop - seed
#     init_pop = {
#         Compartment.INFECTIOUS: seed,
#         Compartment.SUSCEPTIBLE: susceptible,
#     }
#     # Assign to the model
#     model.set_initial_population(init_pop)


def build_model(params: dict, build_options: dict = None) -> CompartmentalModel:
    """Build the compartmental model from the provided parameters.
    Returns:
        CompartmentalModel: Returns tb_dynamics Model
    """
    params = Parameters(**params)
    time = params.time
    iso3 = params.iso3
    seed = params.infectious_seed
    start_population_size = params.start_population_size
    cumulative_start_time = params.cumulative_start_time
    age_breakpoints = [str(age) for age in params.age_breakpoints]

    model = CompartmentalModel(
        times=(time.start, time.end),
        compartments=BASE_COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPS,
        timestep=time.step,
    )
    # age_pops = pd.Series(
    #     inputs.get_population_by_agegroup(age_breakpoints, iso3, None, time.start),
    #     index=age_breakpoints,
    # )
    # Set initial population
    init_pop = {
        Compartment.INFECTIOUS: seed,
        Compartment.SUSCEPTIBLE: start_population_size - seed,
    }
    model.set_initial_population(init_pop)

    contact_rate = params.contact_rate
    contact_rate_latent = params.contact_rate * params.rr_infection_latent
    contact_rate_recovered = params.contact_rate * params.rr_infection_recovered

    birth_rates, years = inputs.get_crude_birth_rate(params.iso3)
    birth_rates = birth_rates / 1000.0  # Birth rates are provided / 1000 population
    crude_birth_rate = scale_up_function(
        years.to_list(), 
        birth_rates.to_list(), 
        smoothness=0.2, 
        method=5
    )

    # Add infection flow.
    model.add_infection_frequency_flow(
        "infection",
        contact_rate,
        Compartment.SUSCEPTIBLE,
        Compartment.EARLY_LATENT,
    )
    model.add_infection_frequency_flow(
        "infection_from_latent",
        contact_rate_latent,
        Compartment.LATE_LATENT,
        Compartment.EARLY_LATENT,
    )
    model.add_infection_frequency_flow(
        "infection_from_recovered",
        contact_rate_recovered,
        Compartment.RECOVERED,
        Compartment.EARLY_LATENT,
    )
    # Add transition flow.
    stabilisation_rate = 1.0 # will be overwritten by stratification
    early_activation_rate = 1.0
    late_activation_rate = 1.0
    model.add_transition_flow(
        "stabilisation",
        stabilisation_rate,
        Compartment.EARLY_LATENT,
        Compartment.LATE_LATENT,
    )
    model.add_transition_flow(
        "early_activation",
        early_activation_rate,
        Compartment.EARLY_LATENT,
        Compartment.INFECTIOUS,
    )
    model.add_transition_flow(
        "late_activation",
        late_activation_rate,
        Compartment.LATE_LATENT,
        Compartment.INFECTIOUS,
    )
    # Add post-diseases flows
    model.add_transition_flow(
        "self_recovery",
        params.self_recovery_rate,
        Compartment.INFECTIOUS,
        Compartment.RECOVERED,
    )


    # Add crude birth flow to the model
    model.add_crude_birth_flow(
        "birth",
        crude_birth_rate,
        Compartment.SUSCEPTIBLE,
    )

    # Add universal death flow to the model
    universal_death_rate = params.crude_death_rate
    model.add_universal_death_flows("universal_death", death_rate=universal_death_rate)

    # Infection death
    model.add_death_flow(
        "infect_death",
        params.infect_death_rate,
        Compartment.INFECTIOUS,
    )

    # Set mixing matrix
    if params.age_mixing:
        age_mixing_matrices = build_synthetic_matrices(
            params.iso3, params.age_mixing.source_iso3, params.age_breakpoints, params.age_mixing.age_adjust,
            requested_locations=["all_locations"]
        )
        age_mixing_matrix = age_mixing_matrices["all_locations"]
        # convert daily contact rates to yearly rates
        age_mixing_matrix *= 365.25
        #Add Age stratification to the model
        age_strat = get_age_strat(
            params = params,
            compartments = BASE_COMPARTMENTS,
            age_mixing_matrix = age_mixing_matrix,
        )
    else:
        age_strat = get_age_strat(
            params = params,
            #age_pops,
            compartments = BASE_COMPARTMENTS,
        )
    model.stratify_with(age_strat)

    # Generate outputs
    request_outputs(
        model,
        cumulative_start_time,
        BASE_COMPARTMENTS)

    return model
