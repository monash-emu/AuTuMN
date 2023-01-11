from typing import List
import pandas as pd

from summer2 import CompartmentalModel
from summer2.experimental.model_builder import ModelBuilder

from jax import numpy as jnp

from autumn.core import inputs
from autumn.core.project import Params, build_rel_path
from autumn.core.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices
from autumn.model_features.jax.random_process import get_random_process
from .outputs import TbOutputsBuilder
from .parameters import Parameters
from .constants import Compartment, FlowName
from .stratifications.age import get_age_strat


from .constants import BASE_COMPARTMENTS, INFECTIOUS_COMPS, LATENT_COMPS

from pathlib import Path


def get_base_params():
    base_params = Params(
        str(Path(__file__).parent.resolve() / "params.yml"),
        validator=lambda d: Parameters(**d),
        validate=False,
    )
    return base_params


def build_model(params: dict, build_options: dict = None, ret_builder=False) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.

    Args:
        params: The validated user-requested parameters
        build_options:

    Returns:
        The "SM-SIR" model, currently being used only for COVID-19

    """

    # Get the parameters and extract some of the more used ones to have simpler names
    builder = ModelBuilder(params, Parameters)
    params = builder.params
    country = params.country
    iso3 = country.iso3
    seed = params.infectious_seed
    start_population_size = params.start_population_size
    #age_breakpoints = [str(age) for age in params.age_breakpoints]
    #age_strat_params = params.age_stratification
    time_params = params.time
    cumulative_start_time = params.cumulative_start_time


    # Create the model object
    model = CompartmentalModel(
        times=(time_params.start, time_params.end),
        compartments=BASE_COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPS,
        timestep=time_params.step,
    )

    """
    Check build options
    """

    # This will be automatically populated by calibration.py if we are running a calibration, but can be manually set
    if build_options:
        validate = build_options.get("enable_validation")
        if validate is not None:
            model.set_validation_enabled(validate)
        idx_cache = build_options.get("derived_outputs_idx_cache")
        if idx_cache:
            model._set_derived_outputs_index_cache(idx_cache)

    """
    Create the total population
    """

    # Get population by age-group
    init_pop = {
        Compartment.INFECTIOUS: seed,
        Compartment.SUSCEPTIBLE: start_population_size - seed,
    }

    # Assign to the model
    model.set_initial_population(init_pop)


    """
    Add intercompartmental flows
    """
    contact_rate = params.contact_rate
    contact_rate_latent = params.contact_rate * params.rr_infection_latent
    contact_rate_recovered = params.contact_rate * params.rr_infection_recovered
    # Add the process of infecting the susceptibles
    model.add_infection_frequency_flow(
        name=FlowName.INFECTION,
        contact_rate=contact_rate,
        source=Compartment.SUSCEPTIBLE,
        dest=Compartment.EARLY_LATENT,
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
        # Infection death
    model.add_death_flow(
        "infect_death",
        params.infect_death_rate,
        Compartment.INFECTIOUS,
    )

    universal_death_rate= 1.0
    model.add_universal_death_flows("universal_death", death_rate=universal_death_rate)


    """
    Apply age stratification
    """
    # Set mixing matrix
    if params.age_mixing:
        age_mixing_matrices = build_synthetic_matrices(
            iso3, params.age_mixing.source_iso3, params.age_breakpoints, params.age_mixing.age_adjust,
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
            compartments = BASE_COMPARTMENTS,
        )
    model.stratify_with(age_strat)

   
    """
    Get the applicable outputs
    """

    outputs_builder = TbOutputsBuilder(model)
    outputs_builder.request_compartment_output("total_population", BASE_COMPARTMENTS)
    # Latency
    outputs_builder.request_compartment_output(
        "latent_population_size", LATENT_COMPS, save_results=False
    )
    sources = ["latent_population_size", "total_population"]
    outputs_builder.request_output_func("percentage_latent", calculate_percentage, sources)

    # Prevalence
    outputs_builder.request_compartment_output(
        "infectious_population_size", INFECTIOUS_COMPS, save_results=False
    )
    sources = ["infectious_population_size", "total_population"]
    outputs_builder.request_output_func(
        "prevalence_infectious", calculate_per_hundred_thousand, sources
    )

    # Death
    outputs_builder.request_flow_output(
        "mortality_infectious_raw", "infect_death", save_results=False
    )
   
    sources = ["mortality_infectious_raw"]
    outputs_builder.request_aggregation_output(
        "mortality_raw",
        sources,
        save_results=False
    )
    model.request_cumulative_output(
        "cumulative_deaths",
        "mortality_raw",
        start_time=cumulative_start_time,
    )
    # Disease incidence
    outputs_builder.request_flow_output(
        "incidence_early_raw", "early_activation", save_results=False
    )
    outputs_builder.request_flow_output("incidence_late_raw", "late_activation", save_results=False)
    sources = ["incidence_early_raw", "incidence_late_raw"]
    outputs_builder.request_aggregation_output("incidence_raw", sources, save_results=False)
    sources = ["incidence_raw", "total_population"]
    model.request_cumulative_output(
        "cumulative_diseased",
        "incidence_raw",
        start_time=cumulative_start_time,
    )
    # Normalise incidence so that it is per unit time (year), not per timestep
    outputs_builder.request_normalise_flow_output("incidence_early", "incidence_early_raw")
    outputs_builder.request_normalise_flow_output("incidence_late", "incidence_late_raw")
    outputs_builder.request_normalise_flow_output(
        "incidence_norm", "incidence_raw", save_results=False
    )
    sources = ["incidence_norm", "total_population"]
    outputs_builder.request_output_func("incidence", calculate_per_hundred_thousand, sources)

    # request extra output to store the number of students*weeks of school missed
 
    builder.set_model(model)
    if ret_builder:
        return model, builder
    else:
        return model

def calculate_percentage(sub_pop_size, total_pop_size):
    return 100 * sub_pop_size / total_pop_size


def calculate_per_hundred_thousand(sub_pop_size, total_pop_size):
    return 1e5 * sub_pop_size / total_pop_size


def calculate_proportion(sub_pop_size, total_pop_size):
    return sub_pop_size / total_pop_size

