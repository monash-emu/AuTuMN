import numpy as np

from summer2 import CompartmentalModel

from autumn import inputs
from autumn.environment.seasonality import get_seasonal_forcing

from apps.covid_19.constants import (
    Compartment,
    COMPARTMENTS,
    INFECTIOUS_COMPARTMENTS,
    DISEASE_COMPARTMENTS,
)
from apps.covid_19.mixing_optimisation.constants import Region
from apps.covid_19.model import preprocess
from apps.covid_19.model.preprocess.importation import get_all_vic_notifications
from apps.covid_19.model.parameters import Parameters
from apps.covid_19.model.preprocess.case_detection import (
    build_detected_proportion_func,
    get_testing_pop,
)
from apps.covid_19.model.stratifications.agegroup import get_agegroup_strat, AGEGROUP_STRATA
from apps.covid_19.model.stratifications.history import get_history_strat
from apps.covid_19.model.stratifications.cluster import (
    get_cluster_strat,
    apply_post_cluster_strat_hacks,
)
from apps.covid_19.model.stratifications.clinical import (
    get_clinical_strat,
    get_proportion_symptomatic,
)
from apps.covid_19.model.outputs.standard import request_standard_outputs
from apps.covid_19.model.outputs.victorian import request_victorian_outputs


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
    importation = params.importation
    # Get proportion of importations by age. This is used to calculate case detection and used in age stratification.
    importation_props_by_age = (
        importation.props_by_age
        if importation and importation.props_by_age
        else {s: 1.0 / len(AGEGROUP_STRATA) for s in AGEGROUP_STRATA}
    )
    # Get case detection rate function.
    get_detected_proportion = build_detected_proportion_func(
        AGEGROUP_STRATA, country, pop, params.testing_to_detection, params.case_detection
    )
    # Determine how many importations there are, including the undetected and asymptomatic importations
    symptomatic_props = get_proportion_symptomatic(params)
    import_symptomatic_prop = sum(
        [
            import_prop * sympt_prop
            for import_prop, sympt_prop in zip(importation_props_by_age.values(), symptomatic_props)
        ]
    )

    def get_abs_detection_proportion_imported(t):
        # Returns absolute proprotion of imported people who are detected to be infectious.
        return import_symptomatic_prop * get_detected_proportion(t)

    if importation:
        is_region_vic = pop.region and Region.to_name(pop.region) in Region.VICTORIA_SUBREGIONS
        if is_region_vic:
            import_times, importation_data = get_all_vic_notifications(
                excluded_regions=(pop.region,)
            )
            testing_pop, _ = get_testing_pop(AGEGROUP_STRATA, country, pop)
            movement_to_region = (
                sum(total_pops) / sum(testing_pop) * params.importation.movement_prop
            )
            import_cases = [i_cases * movement_to_region for i_cases in importation_data]
        else:
            import_times = params.importation.case_timeseries.times
            import_cases = params.importation.case_timeseries.values

        import_rate_func = preprocess.importation.get_importation_rate_func_as_birth_rates(
            import_times, import_cases, get_abs_detection_proportion_imported
        )

        # Imported people are infectious (ie. late active).
        model.add_importation_flow(
            name="importation",
            num_imported=import_rate_func,
            dest=Compartment.LATE_ACTIVE,
        )

    # Stratify the model by age group.
    age_strat = get_agegroup_strat(params, total_pops, importation_props_by_age)
    model.stratify_with(age_strat)

    # Stratify the model by clinical status
    clinical_strat, abs_props = get_clinical_strat(params)
    model.stratify_with(clinical_strat)

    # Infection history stratification
    if params.stratify_by_infection_history:
        history_strat = get_history_strat(params, abs_props, compartment_periods)
        model.stratify_with(history_strat)

    # Stratify model by Victorian subregion (used for Victorian cluster model).
    if params.victorian_clusters:
        cluster_strat = get_cluster_strat(params)
        model.stratify_with(cluster_strat)
        apply_post_cluster_strat_hacks(params, model)

    # Set up derived output functions
    if not params.victorian_clusters:
        request_standard_outputs(
            model, params, AGEGROUP_STRATA, get_abs_detection_proportion_imported
        )
    else:
        request_victorian_outputs(model, params)

    return model
