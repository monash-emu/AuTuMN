from summer import CompartmentalModel

from apps.covid_19.constants import (
    COMPARTMENTS,
    DISEASE_COMPARTMENTS,
    INFECTIOUS_COMPARTMENTS,
    Compartment,
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
from apps.covid_19.model.stratifications.history import get_history_strat
from apps.covid_19.model.stratifications.immunity import get_immunity_strat
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
        contact_rate = get_seasonal_forcing(
            365.0, 173.0, params.seasonal_force, params.contact_rate
        )
    else:
        # Use a static contact rate.
        contact_rate = params.contact_rate

    # Adjust contact rate for Variant of Concerns
    if params.voc_emmergence:
        voc_multiplier = scale_up_function(
            x=[params.voc_emmergence.start_time, params.voc_emmergence.end_time],
            y=[
                1.0,
                1.0
                + params.voc_emmergence.final_proportion
                * (params.voc_emmergence.contact_rate_multiplier - 1.0),
            ],
            method=4,
        )
        raw_contact_rate = contact_rate
        if isinstance(contact_rate, float):

            def contact_rate(t):
                return raw_contact_rate * voc_multiplier(t)

        else:

            def contact_rate(t):
                return raw_contact_rate(t) * voc_multiplier(t)

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

    # Stratify the model by age group.
    age_strat = get_agegroup_strat(params, total_pops)
    model.stratify_with(age_strat)

    # Stratify the model by clinical status
    clinical_strat = get_clinical_strat(params)
    model.stratify_with(clinical_strat)

    # Infection history stratification
    if params.stratify_by_infection_history:
        history_strat = get_history_strat(params)
        model.stratify_with(history_strat)

    # Waning immunity (if requested)
    if params.waning_immunity_duration is not None:
        model.add_fractional_flow(
            name="waning_immunity",
            fractional_rate=1.0 / params.waning_immunity_duration,
            source=Compartment.RECOVERED,
            dest=Compartment.SUSCEPTIBLE,
            source_strata={"history": "naive"},
            dest_strata={"history": "experienced"}
        )

    # Stratify by immunity - which will include vaccination and infection history
    # if params.stratify_by_immunity:
    #     immunity_strat = get_immunity_strat(params)
    #     model.stratify_with(immunity_strat)
    #     if params.vaccination:
    #         vacc_params = params.vaccination
    #         for roll_out_component in vacc_params.roll_out_components:
    #             add_vaccination_flows(model, roll_out_component, age_strat.strata)

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
