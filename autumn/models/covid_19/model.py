from summer import CompartmentalModel
from summer.adjust import Multiply, Overwrite

from autumn.settings.region import Region
from autumn.tools.inputs.social_mixing.queries import get_prem_mixing_matrices
from autumn.tools.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices
from autumn.models.covid_19.constants import Vaccination
from autumn.tools import inputs
from autumn.tools.project import Params, build_rel_path
from autumn.models.covid_19.preprocess.case_detection import CdrProc
from .preprocess.seasonality import get_seasonal_forcing
from .preprocess.testing import find_cdr_function_from_test_data

from .constants import (
    COMPARTMENTS, DISEASE_COMPARTMENTS, INFECTIOUS_COMPARTMENTS, Compartment, Tracing, BASE_DATE, History, INFECTION,
    INFECTIOUSNESS_ONSET, INCIDENCE, PROGRESS, RECOVERY, INFECT_DEATH, VicModelTypes,
)

from . import preprocess
from .outputs.common import request_common_outputs
from .outputs.vaccination import request_vaccination_outputs
from .outputs.strains import request_strain_outputs
from .outputs.tracing import request_tracing_outputs
from .outputs.healthcare import request_healthcare_outputs
from .outputs.history import request_history_outputs, request_recovered_outputs
from .parameters import Parameters
from .preprocess.vaccination import add_vaccination_flows
from .preprocess import tracing
from .preprocess.strains import make_voc_seed_func
from .stratifications.agegroup import AGEGROUP_STRATA, get_agegroup_strat
from .stratifications.clinical import get_clinical_strat
from .stratifications.cluster import apply_post_cluster_strat_hacks, get_cluster_strat
from .stratifications.tracing import get_tracing_strat
from .stratifications.strains import get_strain_strat
from .stratifications.history import get_history_strat
from .stratifications.vaccination import get_vaccination_strat

base_params = Params(
    build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False
)


def build_model(params: dict, build_options: dict = None) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.
    """
    params = Parameters(**params)

    is_region_vic = params.vic_status in (VicModelTypes.VIC_SUPER_2020, VicModelTypes.VIC_SUPER_2021)

    model = CompartmentalModel(
        times=[params.time.start, params.time.end],
        compartments=COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPARTMENTS,
        timestep=params.time.step,
        ref_date=BASE_DATE
    )

    # Check build_options
    # This will be automatically populated by calibration.py if we are running a calibration,
    # but can be manually set if so desired
    if build_options:
        validate = build_options.get("enable_validation")
        if validate is not None:
            model.set_validation_enabled(validate)
        idx_cache = build_options.get("derived_outputs_idx_cache")
        if idx_cache:
            model._set_derived_outputs_index_cache(idx_cache)

    # Time periods calculated from periods (ie "sojourn times")
    compartment_periods = preprocess.compartments.calc_compartment_periods(params.sojourn)

    # Get country population by age-group
    country = params.country
    pop = params.population
    total_pops = inputs.get_population_by_agegroup(
        AGEGROUP_STRATA, country.iso3, pop.region, year=pop.year
    )

    # Distribute infectious seed across infectious split sub-compartments
    total_disease_time = sum([compartment_periods[comp] for comp in DISEASE_COMPARTMENTS])
    init_pop = {
        comp: params.infectious_seed * compartment_periods[comp] / total_disease_time
        for comp in DISEASE_COMPARTMENTS
    }

    # Assign the remainder starting population to the S compartment
    init_pop[Compartment.SUSCEPTIBLE] = sum(total_pops) - sum(init_pop.values())
    model.set_initial_population(init_pop)

    """
    Get input data
    """

    if params.mixing_matrices.type == "prem":
        mixing_matrices = get_prem_mixing_matrices(country.iso3, None, pop.region)
    elif params.mixing_matrices.type == "extrapolated":
        mixing_matrices = build_synthetic_matrices(
            country.iso3, params.mixing_matrices.source_iso3, AGEGROUP_STRATA, params.mixing_matrices.age_adjust,
            pop.region
        )

    """
    Add intercompartmental flows
    """

    # Use a time-varying, sinusoidal seasonal forcing function or constant value for the contact rate
    if params.seasonal_force:
        contact_rate = get_seasonal_forcing(365., 173., params.seasonal_force, params.contact_rate)
    else:
        # Use a static contact rate
        contact_rate = params.contact_rate

    # Infection
    model.add_infection_frequency_flow(
        name=INFECTION,
        contact_rate=contact_rate,
        source=Compartment.SUSCEPTIBLE,
        dest=Compartment.EARLY_EXPOSED,
    )

    # Progression through Covid stages
    infect_onset_rate = 1. / compartment_periods[Compartment.EARLY_EXPOSED]
    model.add_transition_flow(
        name=INFECTIOUSNESS_ONSET,
        fractional_rate=infect_onset_rate,
        source=Compartment.EARLY_EXPOSED,
        dest=Compartment.LATE_EXPOSED,
    )
    incidence_flow_rate = 1. / compartment_periods[Compartment.LATE_EXPOSED]
    model.add_transition_flow(
        name=INCIDENCE,
        fractional_rate=incidence_flow_rate,
        source=Compartment.LATE_EXPOSED,
        dest=Compartment.EARLY_ACTIVE,
    )
    progress_rate = 1. / compartment_periods[Compartment.EARLY_ACTIVE]
    model.add_transition_flow(
        name=PROGRESS,
        fractional_rate=progress_rate,
        source=Compartment.EARLY_ACTIVE,
        dest=Compartment.LATE_ACTIVE,
    )
    # Recovery flows
    recovery_rate = 1. / compartment_periods[Compartment.LATE_ACTIVE]
    model.add_transition_flow(
        name=RECOVERY,
        fractional_rate=recovery_rate,
        source=Compartment.LATE_ACTIVE,
        dest=Compartment.RECOVERED,
    )
    # Infection death
    model.add_death_flow(
        name=INFECT_DEATH,
        death_rate=0.,  # Inconsequential value because it will be overwritten later in clinical stratification
        source=Compartment.LATE_ACTIVE,
    )

    # Stratify the model by age group
    age_strat = get_agegroup_strat(params, total_pops, mixing_matrices)
    model.stratify_with(age_strat)

    # Stratify the model by clinical status
    if pop.region and pop.region.replace("_", "-").lower() in Region.VICTORIA_SUBREGIONS:
        override_test_region = "Victoria"
    else:
        override_test_region = pop.region

    get_detected_proportion = find_cdr_function_from_test_data(
        params.testing_to_detection, country.iso3, override_test_region, pop.year
    )
    clinical_strat, adjustment_systems = get_clinical_strat(params)
    model.stratify_with(clinical_strat)

    # Add the adjuster systems used by the clinical stratification
    for k, v in adjustment_systems.items():
        model.add_adjustment_system(k, v)

    # Register the CDR function as a computed value
    model.add_computed_value_process("cdr", CdrProc(get_detected_proportion))

    # Apply the VoC stratification and adjust contact rate for single/dual Variants of Concern
    if params.voc_emergence:
        voc_params = params.voc_emergence

        # Build and apply stratification
        strain_strat = get_strain_strat(voc_params)
        model.stratify_with(strain_strat)

        # Use importation flows to seed VoC cases
        for voc_name, characteristics in voc_params.items():
            voc_seed_func = make_voc_seed_func(
                characteristics.entry_rate, characteristics.start_time, characteristics.seed_duration
            )
            model.add_importation_flow(
                f"seed_voc_{voc_name}",
                voc_seed_func,
                dest=Compartment.EARLY_EXPOSED,
                dest_strata={"strain": voc_name},
            )

    # Infection history stratification
    if params.stratify_by_infection_history:
        history_strat = get_history_strat(params)
        model.stratify_with(history_strat)

        # Waning immunity (if requested)
        # Note that this approach would mean that the recovered in the naive class have actually previously had Covid
        if params.waning_immunity_duration:
            model.add_transition_flow(
                name="waning_immunity",
                fractional_rate=1. / params.waning_immunity_duration,
                source=Compartment.RECOVERED,
                dest=Compartment.SUSCEPTIBLE,
                source_strata={"history": History.NAIVE},
                dest_strata={"history": History.EXPERIENCED},
            )

    # Stratify model by Victorian sub-region (used for Victorian cluster model)
    if params.vic_status in (VicModelTypes.VIC_SUPER_2020, VicModelTypes.VIC_SUPER_2021):
        cluster_strat = get_cluster_strat(params)
        model.stratify_with(cluster_strat)
        mixing_matrix_function = apply_post_cluster_strat_hacks(params, model, mixing_matrices)

    if params.vic_status == VicModelTypes.VIC_SUPER_2021:
        seed_start_time = params.vic_2021_seeding.seed_time

        for stratum in cluster_strat.strata:
            seed = params.vic_2021_seeding.clusters.__getattribute__(stratum)

            # Function must be bound in loop with optional argument
            def model_seed_func(time, computed_values, seed=seed):
                return seed if seed_start_time < time < seed_start_time + 5. else 0.

            model.add_importation_flow(
                "seed",
                model_seed_func,
                dest=Compartment.EARLY_EXPOSED,
                dest_strata={"cluster": stratum.replace("-", "_")},
            )

    elif params.vic_status == VicModelTypes.VIC_REGION_2021:
        seed_start_time = params.vic_2021_seeding.seed_time
        seed = params.vic_2021_seeding.seed

        def model_seed_func(time, computed_values, seed=seed):
            return seed if seed_start_time < time < seed_start_time + 5. else 0.

        model.add_importation_flow("seed", model_seed_func, dest=Compartment.EARLY_EXPOSED)

    # Contact tracing stratification
    if params.contact_tracing:

        # Stratify the model structure
        tracing_strat = get_tracing_strat(
            params.contact_tracing.quarantine_infect_multiplier,
            params.clinical_stratification.late_infect_multiplier
        )
        model.stratify_with(tracing_strat)

        # Contact tracing processes
        trace_param = tracing.get_tracing_param(
            params.contact_tracing.assumed_trace_prop,
            params.contact_tracing.assumed_prev,
            params.contact_tracing.floor,
        )

        model.add_computed_value_process(
            "prevalence",
            tracing.PrevalenceProc()
        )

        model.add_computed_value_process(
            "prop_detected_traced",
            tracing.PropDetectedTracedProc(
                trace_param,
                params.contact_tracing.floor,
            )
        )

        model.add_computed_value_process(
            "prop_contacts_with_detected_index",
            tracing.PropIndexDetectedProc(
                params.clinical_stratification.non_sympt_infect_multiplier,
                params.clinical_stratification.late_infect_multiplier
            )
        )

        model.add_computed_value_process(
            "traced_flow_rate",
            tracing.TracedFlowRateProc(incidence_flow_rate)
        )

        # Add the transition process to the model
        early_exposed_untraced_comps = \
            [comp for comp in model.compartments if
             comp.is_match(Compartment.EARLY_EXPOSED, {"tracing": Tracing.UNTRACED})]
        early_exposed_traced_comps = \
            [comp for comp in model.compartments if
             comp.is_match(Compartment.EARLY_EXPOSED, {"tracing": Tracing.TRACED})]
        for untraced, traced in zip(early_exposed_untraced_comps, early_exposed_traced_comps):
            model.add_transition_flow(
                "tracing",
                tracing.contact_tracing_func,
                Compartment.EARLY_EXPOSED,
                Compartment.EARLY_EXPOSED,
                source_strata=untraced.strata,
                dest_strata=traced.strata,
                expected_flow_count=1,
            )
            # +++ FIXME: convert this to transition flow with new computed_values aware flow param

    # Stratify by vaccination status
    if params.vaccination:
        vaccination_strat = get_vaccination_strat(params)

        # Was going to delete this, but it is necessary - doesn't make sense to have VoC in an otherwise empty stratum
        if params.voc_emergence:
            for voc_name, characteristics in voc_params.items():
                vaccination_strat.add_flow_adjustments(
                    f"seed_voc_{voc_name}",
                    {
                        Vaccination.VACCINATED: Multiply(1. / 2.),
                        Vaccination.ONE_DOSE_ONLY: Overwrite(0.),
                        Vaccination.UNVACCINATED: Multiply(1. / 2.),
                    }
                )
        model.stratify_with(vaccination_strat)

        # Implement the process of people getting vaccinated
        vacc_params = params.vaccination

        # Vic 2021 code is not generalisable
        if params.vic_status == VicModelTypes.VIC_SUPER_2021:
            for component in vacc_params.roll_out_components:
                for cluster in cluster_strat.strata:
                    add_vaccination_flows(
                        model,
                        component,
                        age_strat.strata,
                        params.vaccination.one_dose,
                        vic_cluster=cluster,
                        cluster_stratum={"cluster": cluster},
                    )
        elif params.vic_status == VicModelTypes.VIC_REGION_2021:
            for i_comp, component in enumerate(vacc_params.roll_out_components):
                add_vaccination_flows(
                    model,
                    component,
                    age_strat.strata,
                    params.vaccination.one_dose,
                    vic_cluster=params.population.region,
                    i_component=i_comp,
                    vaccination_lag=vacc_params.lag,
                )

        else:
            for roll_out_component in vacc_params.roll_out_components:
                coverage_override = vacc_params.coverage_override if vacc_params.coverage_override else None
                add_vaccination_flows(
                    model,
                    roll_out_component,
                    age_strat.strata,
                    params.vaccination.one_dose,
                    coverage_override
                )

        # Add transition from single dose to fully vaccinated
        if params.vaccination.one_dose:
            for compartment in COMPARTMENTS:
                model.add_transition_flow(
                    name="second_dose",
                    fractional_rate=1. / params.vaccination.second_dose_delay,
                    source=compartment,
                    dest=compartment,
                    source_strata={"vaccination": Vaccination.ONE_DOSE_ONLY},
                    dest_strata={"vaccination": Vaccination.VACCINATED},
                )

    # Dive into summer internals to over-write mixing matrix
    if is_region_vic:
        model._mixing_matrices = [mixing_matrix_function]

    # Find the total population, used by multiple output types
    model.request_output_for_compartments(name="_total_population", compartments=COMPARTMENTS, save_results=False)

    # Most standard outputs
    request_common_outputs(model, params, is_region_vic)
    request_healthcare_outputs(model, params.sojourn.compartment_periods, is_region_vic)

    """
    Set up derived output functions
    """

    # Vaccination
    if params.vaccination:
        request_vaccination_outputs(model, params)

    # Strain-related outputs
    if params.voc_emergence:
        request_strain_outputs(model, list(params.voc_emergence.keys()))

    # Proportion of the population previously infected/exposed
    if params.stratify_by_infection_history:
        request_history_outputs(model)
    else:
        request_recovered_outputs(model, is_region_vic)

    # Contact tracing-related outputs
    if params.contact_tracing:
        request_tracing_outputs(model)

    return model
