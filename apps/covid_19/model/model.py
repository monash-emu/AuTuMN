from summer2 import CompartmentalModel

from autumn import inputs
from autumn.environment.seasonality import get_seasonal_forcing

from apps.covid_19.constants import (
    Compartment,
    COMPARTMENTS,
    INFECTIOUS_COMPARTMENTS,
    DISEASE_COMPARTMENTS,
)
from apps.covid_19.model import preprocess
from apps.covid_19.model.parameters import Parameters
from apps.covid_19.model.stratifications.agegroup import get_agegroup_strat, AGEGROUP_STRATA
from apps.covid_19.model.stratifications.cluster import (
    get_cluster_strat,
    apply_post_cluster_strat_hacks,
)
from apps.covid_19.model.stratifications.immunity import get_immunity_strat
from apps.covid_19.model.stratifications.clinical import get_clinical_strat
from apps.covid_19.model.outputs.standard import request_standard_outputs
from apps.covid_19.model.outputs.victorian import request_victorian_outputs

from apps.covid_19.model.preprocess.vaccination import get_vacc_roll_out_function


def apply_vaccination(model, vacc_params):
    vaccination_roll_out_function = \
        get_vacc_roll_out_function(vacc_params.roll_out_function)
    for compartment in [Compartment.SUSCEPTIBLE, Compartment.RECOVERED]:
        model.add_fractional_flow(
            name="vaccination",
            fractional_rate=vaccination_roll_out_function,
            source=compartment,
            dest=compartment,
            source_strata={"immunity": "unvaccinated"},
            dest_strata={"immunity": "vaccinated"},
        )
    return model


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
    model.add_fractional_flow(
        name="infect_onset",
        fractional_rate=1.0 / compartment_periods[Compartment.EARLY_EXPOSED],
        source=Compartment.EARLY_EXPOSED,
        dest=Compartment.LATE_EXPOSED,
    )
    model.add_fractional_flow(
        name="incidence",
        fractional_rate=1.0 / compartment_periods[Compartment.LATE_EXPOSED],
        source=Compartment.LATE_EXPOSED,
        dest=Compartment.EARLY_ACTIVE,
    )
    model.add_fractional_flow(
        name="progress",
        fractional_rate=1.0 / compartment_periods[Compartment.EARLY_ACTIVE],
        source=Compartment.EARLY_ACTIVE,
        dest=Compartment.LATE_ACTIVE,
    )
    # Recovery flows
    model.add_fractional_flow(
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

    if params.waning_immunity_duration is not None:
        # Waning immunity (if requested)
        model.add_fractional_flow(
            name="warning_immunity",
            fractional_rate=1.0 / params.waning_immunity_duration,
            source=Compartment.RECOVERED,
            dest=Compartment.SUSCEPTIBLE,
        )

    # Optionally add an importation flow, where we ship in infected people from overseas.
    if params.importation:
        get_importation_rate = preprocess.importation.build_importation_rate_func(
            params, AGEGROUP_STRATA, total_pops
        )
        # Imported people are infectious (ie. late active).
        model.add_importation_flow(
            name="importation",
            num_imported=get_importation_rate,
            dest=Compartment.LATE_ACTIVE,
        )

    # Stratify the model by age group.
    age_strat = get_agegroup_strat(params, total_pops)
    model.stratify_with(age_strat)

    # Stratify the model by clinical status
    clinical_strat = get_clinical_strat(params)
    model.stratify_with(clinical_strat)

    # Stratify by immunity - which will include vaccination and infection history
    if params.stratify_by_immunity:
        immunity_strat = get_immunity_strat(params)
        model.stratify_with(immunity_strat)
        if params.vaccination:
            model = apply_vaccination(model, params.vaccination)

    # Infection history stratification
    # if params.stratify_by_infection_history:
    # history_strat = get_history_strat(params, compartment_periods)
    # model.stratify_with(history_strat)

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
