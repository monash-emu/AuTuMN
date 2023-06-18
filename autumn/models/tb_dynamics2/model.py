from summer2 import CompartmentalModel
from summer2.experimental.model_builder import ModelBuilder

from autumn.core.inputs.demography.queries import get_crude_birth_rate_series
from autumn.core.project import Params
from .outputs import TbOutputsBuilder
from .parameters import Parameters
from .constants import Compartment
from .stratifications.age import get_age_strat
from autumn.model_features.curve.interpolate import build_static_sigmoidal_multicurve
from summer2.parameters.params import Function
from .stratifications.organ import get_organ_strat
from .stratifications.gender import get_gender_strat

from .constants import BASE_COMPARTMENTS, INFECTIOUS_COMPS, LATENT_COMPS
from summer2.parameters import Time, DerivedOutput

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
        The TB dynamics 2 model
    """

    # Get the parameters and extract some of the more used ones to have simpler names
    builder = ModelBuilder(params, Parameters)
    params = builder.params
    country = params.country
    iso3 = country.iso3
    seed = params.infectious_seed
    start_population_size = params.start_population_size
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

    # Add the process of infecting the susceptibles
    model.add_infection_frequency_flow(
        name="infection",
        contact_rate=contact_rate,
        source=Compartment.SUSCEPTIBLE,
        dest=Compartment.EARLY_LATENT,
    )

    # And those with partial immunity
    model.add_infection_frequency_flow(
        "infection_from_latent",
        contact_rate * params.rr_infection_latent ,
        Compartment.LATE_LATENT,
        Compartment.EARLY_LATENT,
    )
    model.add_infection_frequency_flow(
        "infection_from_recovered",
        contact_rate * params.rr_infection_recovered,
        Compartment.RECOVERED,
        Compartment.EARLY_LATENT,
    )

    # Latency-related flows
    stabilisation_rate = 1.0  
    model.add_transition_flow(
        "stabilisation",
        stabilisation_rate,
        Compartment.EARLY_LATENT,
        Compartment.LATE_LATENT,
    )

    early_activation_rate = 1.0
    model.add_transition_flow(
        "early_activation",
        early_activation_rate,
        Compartment.EARLY_LATENT,
        Compartment.INFECTIOUS,
    )

    model.add_transition_flow(
        "late_activation",
        params.progression_multiplier,  # Set to the adjuster, rather than one
        Compartment.LATE_LATENT,
        Compartment.INFECTIOUS,
    )

    # Add post-diseases flows
    model.add_transition_flow(
        "self_recovery",
        params.self_recovery_rate_dict.unstratified,
        Compartment.INFECTIOUS,
        Compartment.RECOVERED,
    )
  

    detection_rate = 1.0 # later adjusted by organ
    model.add_transition_flow(
        "detection",
        detection_rate,
        Compartment.INFECTIOUS,
        Compartment.ON_TREATMENT,
    )

    #Treatment recovery, releapse, death flows.
    treatment_recovery_rate = 1.0 # will be adjusted later
    model.add_transition_flow(
        "treatment_recovery",
        treatment_recovery_rate,
        Compartment.ON_TREATMENT,
        Compartment.RECOVERED,
    )

    treatment_death_rate = 1.0
    model.add_death_flow(
        "treatment_death",
        treatment_death_rate,
        Compartment.ON_TREATMENT,
    )
    relapse_rate = 1.0
    model.add_transition_flow(
        "relapse",
        relapse_rate,
        Compartment.ON_TREATMENT,
        Compartment.INFECTIOUS,
    )

    # Entry flows
    birth_rates = get_crude_birth_rate_series(iso3)
    birth_rates /= 1000.0  # Birth rates are provided / 1000 population
    tfunc = build_static_sigmoidal_multicurve(birth_rates.index, birth_rates)
    crude_birth_rate = Function(tfunc, [Time])
    model.add_crude_birth_flow(
        "birth",
        crude_birth_rate,
        Compartment.SUSCEPTIBLE,
    )

    # Universal death
    universal_death_rate = 1.0
    model.add_universal_death_flows("universal_death", death_rate=universal_death_rate)

    # Infection death
    model.add_death_flow(
        "infect_death",
        params.infect_death_rate_dict.unstratified,
        Compartment.INFECTIOUS,
    )

    # acf implementation
    implement_acf = len(params.time_variant_screening_rate.keys()) > 0
    if implement_acf:
        # Default value
        acf_detection_rate = 1.0

            # Universal active case funding is applied
        times = list(params.time_variant_screening_rate.keys())
        vals = [
                v * params.acf_screening_sensitivity
                for v in list(params.time_variant_screening_rate.values())
            ]
        acf_detection_rate = Function(build_static_sigmoidal_multicurve(times, vals), [Time])

        model.add_transition_flow(
            "acf_detection",
            acf_detection_rate,
            Compartment.INFECTIOUS,
            Compartment.ON_TREATMENT,
        )

    
    """
    Age stratification
    """

    age_strat = get_age_strat(
        params=params
    )
    model.stratify_with(age_strat)

    """
    Organ stratification
    """
    
    if "organ" in params.stratify_by:
        organ_strat = get_organ_strat(params)
        model.stratify_with(organ_strat)

    """
    Get the applicable outputs
    """

    outputs_builder = TbOutputsBuilder(model)
    outputs_builder.request_compartment_output("total_population", BASE_COMPARTMENTS)
 
    model.request_output_for_compartments(
        "latent_population_size", LATENT_COMPS, save_results=False
    )
    # latency
    model.request_function_output("percentage_latent", 100.0 * DerivedOutput("latent_population_size") / DerivedOutput("total_population"))

    # Prevalence
    model.request_output_for_compartments("infectious_population_size", INFECTIOUS_COMPS, save_results=True)

    model.request_function_output("prevalence_infectious", 1e5 * DerivedOutput("infectious_population_size") / DerivedOutput("total_population"),)

    # Death
    model.request_output_for_flow("mortality_infectious_raw", "infect_death", save_results=True)

    sources = ["mortality_infectious_raw"]
    outputs_builder.request_aggregation_output("mortality_raw", sources, save_results=False)
    model.request_cumulative_output("cumulative_deaths", "mortality_raw", start_time=cumulative_start_time)
    # Disease incidence
    outputs_builder.request_flow_output("incidence_early_raw", "early_activation", save_results=False)
    outputs_builder.request_flow_output("incidence_late_raw", "late_activation", save_results=False)
    sources = ["incidence_early_raw", "incidence_late_raw"]
    outputs_builder.request_aggregation_output("incidence_raw", sources, save_results=False)
    model.request_cumulative_output("cumulative_diseased", "incidence_raw", start_time=cumulative_start_time)
    # Normalise incidence so that it is per unit time (year), not per timestep
    outputs_builder.request_normalise_flow_output("incidence_early", "incidence_early_raw")
    outputs_builder.request_normalise_flow_output("incidence_late", "incidence_late_raw")
    outputs_builder.request_normalise_flow_output("incidence_norm", "incidence_raw", save_results=False)
    model.request_function_output(
        "incidence", 1e5 * DerivedOutput("incidence_norm") / DerivedOutput("total_population")
    )
    outputs_builder.request_flow_output("passive_notifications_raw", "detection", save_results=False)
    # request notifications
    outputs_builder.request_normalise_flow_output("notifications", "passive_notifications_raw")


    builder.set_model(model)
    if ret_builder:
        return model, builder
    else:
        return model

