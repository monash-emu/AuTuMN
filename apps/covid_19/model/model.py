from summer import CompartmentalModel

from apps.covid_19.constants import (
    COMPARTMENTS,
    DISEASE_COMPARTMENTS,
    INFECTIOUS_COMPARTMENTS,
    Compartment,
    Strain
)
from apps.covid_19.model import preprocess
from apps.covid_19.model.outputs.standard import request_standard_outputs
from apps.covid_19.model.outputs.victorian import request_victorian_outputs
from apps.covid_19.model.parameters import Parameters
from apps.covid_19.model.preprocess.seasonality import get_seasonal_forcing
from apps.covid_19.model.preprocess.vaccination import add_vaccination_flows
from apps.covid_19.model.stratifications.agegroup import (
    AGEGROUP_STRATA,
    get_agegroup_strat,
)
from apps.covid_19.model.stratifications.clinical import get_clinical_strat
from apps.covid_19.model.stratifications.cluster import (
    apply_post_cluster_strat_hacks,
    get_cluster_strat,
)
from apps.covid_19.model.stratifications.strains import get_strain_strat
from apps.covid_19.model.stratifications.history import get_history_strat
from apps.covid_19.model.stratifications.vaccination import get_vaccination_strat
from autumn import inputs
from autumn.curve.scale_up import scale_up_function


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

    # Add intercompartmental flows
    if params.seasonal_force:
        # Use a time-varying, sinusoidal seasonal forcing function for contact rate.
        raw_contact_rate = get_seasonal_forcing(
            365.0, 173.0, params.seasonal_force, params.contact_rate
        )
    else:
        # Use a static contact rate.
        raw_contact_rate = params.contact_rate
    contact_rate = None

    # Adjust contact rate for Variant of Concerns.
    if params.voc_emergence:
        voc_start_time = params.voc_emergence.start_time
        voc_end_time = params.voc_emergence.end_time
        voc_final_prop = params.voc_emergence.final_proportion
        voc_contact_rate_multiplier = params.voc_emergence.contact_rate_multiplier

        # Work out the seeding function for later.
        if params.voc_emergence.dual_strain:
            voc_seed = lambda time: 1. if abs(time - voc_start_time) < 5. else 0.

        # Otherwise adjust the contact rate parameter.
        else:
            voc_multiplier = scale_up_function(
                x=[voc_start_time, voc_end_time],
                y=[
                    1.0,
                    1.0
                    + voc_final_prop
                    * (voc_contact_rate_multiplier - 1.0),
                    ],
                method=4,
            )

            # Calculate the time-varying contact rate.
            contact_rate = \
                lambda time: raw_contact_rate * voc_multiplier(time) if isinstance(raw_contact_rate, float) else \
                lambda time: raw_contact_rate(time) * voc_multiplier(time)

    final_contact_rate = contact_rate if contact_rate else raw_contact_rate
    model.add_infection_frequency_flow(
        name="infection",
        contact_rate=final_contact_rate,
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
    model.add_transition_flow(
        name="incidence",
        fractional_rate=1.0 / compartment_periods[Compartment.LATE_EXPOSED],
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

    # Stratify the model by strain.
    if params.voc_emergence and params.voc_emergence.dual_strain:
        strain_strat = get_strain_strat(params)
        model.stratify_with(strain_strat)

        # Seed the VoC stratum.
        model.add_importation_flow(
            "seed_voc",
            voc_seed,
            dest=Compartment.EARLY_ACTIVE,
            dest_strata={"strain": Strain.VARIANT_OF_CONCERN}
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
                dest_strata={"history": "experienced"}
            )

    # Stratify by vaccination status
    if params.vaccination:
        vaccination_strat = get_vaccination_strat(params)
        model.stratify_with(vaccination_strat)
        vacc_params = params.vaccination
        for roll_out_component in vacc_params.roll_out_components:
            add_vaccination_flows(model, roll_out_component, age_strat.strata)

    # Stratify model by Victorian subregion (used for Victorian cluster model).
    if params.victorian_clusters:
        cluster_strat = get_cluster_strat(params)
        model.stratify_with(cluster_strat)
        apply_post_cluster_strat_hacks(params, model)

    # Set up derived output functions
    if not params.victorian_clusters:
        request_standard_outputs(model, params)
    else:
        request_victorian_outputs(model, params)

    return model
