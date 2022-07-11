from typing import List
import pandas as pd

from summer import CompartmentalModel

from autumn.core import inputs
from autumn.core.project import Params, build_rel_path
from autumn.model_features.random_process import RandomProcessProc
from autumn.core.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices
from autumn.core.utils.utils import multiply_function_or_constant
from autumn.model_features.computed_values import FunctionWrapper
from autumn.model_features.random_process import get_random_process
from .outputs import SmCovidOutputsBuilder
from .parameters import Parameters, Sojourns, CompartmentSojourn
from .constants import BASE_COMPARTMENTS, Compartment, FlowName
from .stratifications.immunity import (
    get_immunity_strat,
    adjust_susceptible_infection_without_strains,
    set_dynamic_vaccination_flows
)

from autumn.models.sm_sir.stratifications.agegroup import convert_param_agegroups, get_agegroup_strat
from autumn.settings.constants import COVID_BASE_DATETIME

# Base date used to calculate mixing matrix times
base_params = Params(build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False)


def build_model(
        params: dict,
        build_options: dict = None
) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.

    Args:
        params: The validated user-requested parameters
        build_options:

    Returns:
        The "SM-SIR" model, currently being used only for COVID-19

    """

    # Get the parameters and extract some of the more used ones to have simpler names
    params = Parameters(**params)

    country = params.country
    pop = params.population
    iso3 = country.iso3
    region = pop.region
    age_groups = [str(age) for age in params.age_groups]
    age_strat_params = params.age_stratification
    sojourns = params.sojourns
    
    time_params = params.time
    time_to_event_params = params.time_from_onset_to_event

    # Determine the infectious compartment(s)
    infectious_compartments = [Compartment.INFECTIOUS]

    # Create the model object
    model = CompartmentalModel(
        times=(time_params.start, time_params.end),
        compartments=BASE_COMPARTMENTS,
        infectious_compartments=infectious_compartments,
        timestep=time_params.step,
        ref_date=COVID_BASE_DATETIME,
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

    # Get country population by age-group
    age_pops = pd.Series(
        inputs.get_population_by_agegroup(age_groups, iso3, region, pop.year),
        index=age_groups
    )

    # Assign the population to compartments
    susceptible_pop = age_pops.sum() - params.infectious_seed
    init_pop = {
        Compartment.INFECTIOUS: params.infectious_seed,
        Compartment.SUSCEPTIBLE: susceptible_pop,
    }

    # Assign to the model
    model.set_initial_population(init_pop)

    """
    Add intercompartmental flows
    """
    # From latent to active infection
    progression_rate = 1. / sojourns.latent
    model.add_transition_flow(
        name=FlowName.PROGRESSION,
        fractional_rate=progression_rate,
        source=Compartment.LATENT,
        dest=Compartment.INFECTIOUS,
    )

    # Transmission
    infection_dest, infectious_entry_flow = Compartment.LATENT, FlowName.PROGRESSION

    if params.activate_random_process:

        # Store random process as a computed value to make it available as an output
        rp_function, contact_rate = get_random_process(
            params.random_process,
            params.contact_rate
        )
        model.add_computed_value_process(
            "transformed_random_process",
            RandomProcessProc(rp_function)
        )

    else:
        contact_rate = params.contact_rate

    # Add the process of infecting the susceptibles
    model.add_infection_frequency_flow(
        name=FlowName.INFECTION,
        contact_rate=contact_rate,
        source=Compartment.SUSCEPTIBLE,
        dest=infection_dest,
    )

    # Add recovery flow 
    recovery_rate = 1. / sojourns.active
    model.add_transition_flow(
        name=FlowName.RECOVERY,
        fractional_rate=recovery_rate,
        source=Compartment.INFECTIOUS,
        dest=Compartment.RECOVERED,
    )

    """
    Apply age stratification
    """
    suscept_req = age_strat_params.susceptibility
    sympt_req = age_strat_params.prop_symptomatic

    # Preprocess age-specific parameters to match model age bands if requested in this way
    if type(suscept_req) == dict:
        suscept_adjs = convert_param_agegroups(iso3, region, suscept_req, age_groups)
    else:
        suscept_adjs = suscept_req  # In which case it should be None or a float, confirmed in parameter validation

    if type(sympt_req) == dict:
        sympt_props = convert_param_agegroups(iso3, region, sympt_req, age_groups)
        sympt_props.index = sympt_props.index.map(str)  # Change int indices to string to match model format
    else:
        sympt_props = sympt_req  # In which case it should be None or a float

    # Get the age-specific mixing matrices
    mixing_matrices = build_synthetic_matrices(
        iso3,
        params.ref_mixing_iso3,
        [int(age) for age in age_groups],
        True,  # Always age-adjust, could change this to being a parameter
        region
    )

    # Get the actual age stratification now
    age_strat = get_agegroup_strat(
        params,
        age_groups,
        age_pops,
        mixing_matrices,
        BASE_COMPARTMENTS,
        params.is_dynamic_mixing_matrix,
        suscept_adjs,
    )
    model.stratify_with(age_strat)

    """
    Immunity stratification
    """

    # Get the immunity stratification
    vaccine_effects_params = params.vaccine_effects
    immunity_strat = get_immunity_strat(
        BASE_COMPARTMENTS,
    )

    # Adjust infection of susceptibles for immunity status
    adjust_susceptible_infection_without_strains(vaccine_effects_params.ve_infection, immunity_strat)

    # Apply the immunity stratification
    model.stratify_with(immunity_strat)

    # Apply dynamic vaccination flows
    set_dynamic_vaccination_flows(BASE_COMPARTMENTS, model, iso3, age_groups)
    
    """
    Get the applicable outputs
    """
    outputs_builder = SmCovidOutputsBuilder(model, BASE_COMPARTMENTS)
    
    outputs_builder.request_incidence(age_groups, infectious_entry_flow, params.request_incidence_by_age)
    outputs_builder.request_recovered_proportion(BASE_COMPARTMENTS)
    outputs_builder.request_immunity_props(immunity_strat.strata, age_pops, params.request_immune_prop_by_age)

    return model
