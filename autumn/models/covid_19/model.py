import copy

from summer import CompartmentalModel

from autumn.tools import inputs
from autumn.tools.project import Params, build_rel_path

from .constants import (
    COMPARTMENTS,
    DISEASE_COMPARTMENTS,
    INFECTIOUS_COMPARTMENTS,
    Compartment,
    Strain,
)
from . import preprocess
from .outputs.standard import request_standard_outputs
from .outputs.victorian import request_victorian_outputs
from .parameters import Parameters
from .preprocess.seasonality import get_seasonal_forcing
from .preprocess.vaccination import add_vaccination_flows
from .preprocess.tracing import trace_function
from .stratifications.agegroup import (
    AGEGROUP_STRATA,
    get_agegroup_strat,
)
from .stratifications.clinical import get_clinical_strat
from .stratifications.cluster import (
    apply_post_cluster_strat_hacks,
    get_cluster_strat,
)
from .stratifications.tracing import get_tracing_strat
from .stratifications.strains import get_strain_strat
from .stratifications.history import get_history_strat
from .stratifications.vaccination import get_vaccination_strat


base_params = Params(
    build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False
)


def build_model(params: dict) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.
    """
    params = Parameters(**params)
    model = CompartmentalModel(
        times=[params.time.start, params.time.end],
        compartments=COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPARTMENTS,
        timestep=params.time.step,
    )

    # Population distribution
    country = params.country
    pop = params.population
    # Time periods calculated from periods (ie "sojourn times")
    compartment_periods = preprocess.compartments.calc_compartment_periods(params.sojourn)
    # Get country population by age-group
    total_pops = inputs.get_population_by_agegroup(
        AGEGROUP_STRATA, country.iso3, pop.region, year=pop.year
    )
    # Distribute infectious seed across infectious split sub-compartments
    total_disease_time = sum([compartment_periods[c] for c in DISEASE_COMPARTMENTS])
    init_pop = {
        c: params.infectious_seed * compartment_periods[c] / total_disease_time
        for c in DISEASE_COMPARTMENTS
    }
    # Assign the remainder starting population to the S compartment
    init_pop[Compartment.SUSCEPTIBLE] = sum(total_pops) - sum(init_pop.values())
    model.set_initial_population(init_pop)

    # Add intercompartmental flows.

    # Use a time-varying, sinusoidal seasonal forcing function or constant value for the contact rate.
    if params.seasonal_force:
        contact_rate = get_seasonal_forcing(
            365.0, 173.0, params.seasonal_force, params.contact_rate
        )
    else:
        # Use a static contact rate.
        contact_rate = params.contact_rate

    model.add_infection_frequency_flow(
        name="infection",
        contact_rate=contact_rate,
        source=Compartment.SUSCEPTIBLE,
        dest=Compartment.EARLY_EXPOSED,
    )
    # Infection progress flows.
    model.add_transition_flow(
        name="infect_onset",
        fractional_rate=1.0 / compartment_periods[Compartment.EARLY_EXPOSED],
        source=Compartment.EARLY_EXPOSED,
        dest=Compartment.LATE_EXPOSED,
    )
    incidence_flow_rate = 1.0 / compartment_periods[Compartment.LATE_EXPOSED]
    model.add_transition_flow(
        name="incidence",
        fractional_rate=incidence_flow_rate,
        source=Compartment.LATE_EXPOSED,
        dest=Compartment.EARLY_ACTIVE,
    )
    model.add_transition_flow(
        name="progress",
        fractional_rate=1.0 / compartment_periods[Compartment.EARLY_ACTIVE],
        source=Compartment.EARLY_ACTIVE,
        dest=Compartment.LATE_ACTIVE,
    )
    # Recovery flows
    model.add_transition_flow(
        name="recovery",
        fractional_rate=1.0 / compartment_periods[Compartment.LATE_ACTIVE],
        source=Compartment.LATE_ACTIVE,
        dest=Compartment.RECOVERED,
    )
    # Infection death
    model.add_death_flow(
        name="infect_death",
        death_rate=0,  # Will be overwritten later in clinical stratification.
        source=Compartment.LATE_ACTIVE,
    )

    # Stratify the model by age group.
    age_strat = get_agegroup_strat(params, total_pops)
    model.stratify_with(age_strat)

    # Stratify the model by clinical status
    clinical_strat = get_clinical_strat(params)
    model.stratify_with(clinical_strat)

    # Contact tracing stratification
    tracing_strat = get_tracing_strat(params)
    model.stratify_with(tracing_strat)

    # Apply the VoC stratification and adjust contact rate for Variant of Concerns.
    if params.voc_emergence:
        voc_start_time = params.voc_emergence.start_time
        voc_entry_rate = params.voc_emergence.entry_rate
        seed_duration = params.voc_emergence.seed_duration
        strain_strat = get_strain_strat(params.voc_emergence.contact_rate_multiplier)
        model.stratify_with(strain_strat)

        # Work out the seeding function and seed the VoC stratum.
        voc_seed = (
            lambda time: voc_entry_rate if 0.0 < time - voc_start_time < seed_duration else 0.0
        )

        model.add_importation_flow(
            "seed_voc",
            voc_seed,
            dest=Compartment.EARLY_ACTIVE,
            dest_strata={"strain": Strain.VARIANT_OF_CONCERN},
        )

    # Infection history stratification
    if params.stratify_by_infection_history:
        history_strat = get_history_strat(params)
        model.stratify_with(history_strat)

        # Waning immunity (if requested)
        # Note that this approach would mean that the recovered in the naive class have actually previously had Covid.
        if params.waning_immunity_duration:
            model.add_transition_flow(
                name="waning_immunity",
                fractional_rate=1.0 / params.waning_immunity_duration,
                source=Compartment.RECOVERED,
                dest=Compartment.SUSCEPTIBLE,
                source_strata={"history": "naive"},
                dest_strata={"history": "experienced"},
            )

    # Stratify by vaccination status
    if params.vaccination:
        vaccination_strat = get_vaccination_strat(params)
        model.stratify_with(vaccination_strat)
        vacc_params = params.vaccination
        for roll_out_component in vacc_params.roll_out_components:
            if vacc_params.coverage_override:
                coverage_override = vacc_params.coverage_override
            else:
                coverage_override = None
            add_vaccination_flows(model, roll_out_component, age_strat.strata, coverage_override)

    # Stratify model by Victorian subregion (used for Victorian cluster model).
    if params.victorian_clusters:
        cluster_strat = get_cluster_strat(params)
        model.stratify_with(cluster_strat)
        apply_post_cluster_strat_hacks(params, model)

    # **** THIS MUST BE THE LAST STRATIFICATION ****
    # Apply the process of contact tracing
    prop_traced = 0.1

    early_exposed_untraced_comps = \
        [comp for comp in model.compartments if comp.is_match(Compartment.EARLY_EXPOSED, {"tracing": "untraced"})]
    early_exposed_traced_comps = \
        [comp for comp in model.compartments if comp.is_match(Compartment.EARLY_EXPOSED, {"tracing": "traced"})]

    for untraced, traced in zip(early_exposed_untraced_comps, early_exposed_traced_comps):
        trace_func = trace_function(prop_traced, incidence_flow_rate, untraced)
        model.add_function_flow(
            "tracing",
            trace_func,
            Compartment.EARLY_EXPOSED,
            Compartment.EARLY_EXPOSED,
            source_strata=untraced.strata,
            dest_strata=traced.strata,
            expected_flow_count=1,
        )

    # Set up derived output functions
    if not params.victorian_clusters:
        request_standard_outputs(model, params)
    else:
        request_victorian_outputs(model, params)

    return model
