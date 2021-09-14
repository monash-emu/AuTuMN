from summer import CompartmentalModel
from summer.adjust import Multiply, Overwrite

from autumn.tools.inputs.social_mixing.queries import get_prem_mixing_matrices
from autumn.tools.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices
from autumn.models.covid_19.constants import Vaccination
from autumn.tools import inputs
from autumn.tools.project import Params, build_rel_path
from autumn.models.covid_19.preprocess.case_detection import CdrProc
from autumn.settings.region import Region

from .constants import (
    COMPARTMENTS, DISEASE_COMPARTMENTS, INFECTIOUS_COMPARTMENTS, Compartment, Tracing, BASE_DATE, History, INFECTION,
    INFECTIOUSNESS_ONSET, INCIDENCE, PROGRESS, RECOVERY, INFECT_DEATH,
)

from . import preprocess
from .outputs.vaccination import request_vaccination_outputs
from .outputs.standard import request_standard_outputs
from .outputs.victorian import request_victorian_outputs
from .parameters import Parameters
from .preprocess.vaccination import add_vaccination_flows
from .preprocess import tracing
from .stratifications.agegroup import AGEGROUP_STRATA, get_agegroup_strat
from .stratifications.clinical import get_clinical_strat
from .stratifications.cluster import apply_post_cluster_strat_hacks, get_cluster_strat
from .stratifications.tracing import get_tracing_strat
from .stratifications.strains import get_strain_strat, make_voc_seed_func
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
    is_region_vic = bool(params.victorian_clusters)  # Different structures for Victoria model
    is_region_vic2021 = is_region_vic and params.time.start > 365.  # Probably this can be done better
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
    if params.mixing_matrices:
        if params.mixing_matrices.type == 'prem':
            mixing_matrices = get_prem_mixing_matrices(country.iso3)
        elif params.mixing_matrices.type == 'extrapolated':
            mixing_matrices = build_synthetic_matrices(
                country.iso3, params.mixing_matrices.source_iso3, AGEGROUP_STRATA, params.mixing_matrices.age_adjust,
                pop.region
            )
        else:
            raise Exception("Invalid mixing matrix type specified in parameters")
    else:
        # Default to prem matrices (old model runs)
        mixing_matrices = get_prem_mixing_matrices(country.iso3)

    """
    Add intercompartmental flows
    """

    # Infection
    contact_rate = params.contact_rate
    model.add_infection_frequency_flow(
        name=INFECTION,
        contact_rate=contact_rate,
        source=Compartment.SUSCEPTIBLE,
        dest=Compartment.EARLY_EXPOSED,
    )

    # Progression through Covid stages
    model.add_transition_flow(
        name=INFECTIOUSNESS_ONSET,
        fractional_rate=1. / compartment_periods[Compartment.EARLY_EXPOSED],
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
    model.add_transition_flow(
        name=PROGRESS,
        fractional_rate=1. / compartment_periods[Compartment.EARLY_ACTIVE],
        source=Compartment.EARLY_ACTIVE,
        dest=Compartment.LATE_ACTIVE,
    )
    # Recovery flows
    model.add_transition_flow(
        name=RECOVERY,
        fractional_rate=1. / compartment_periods[Compartment.LATE_ACTIVE],
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
    clinical_strat, get_detected_proportion, adjustment_systems = get_clinical_strat(params)
    model.stratify_with(clinical_strat)

    # Add the adjuster systems used by the clinical stratification
    for k, v in adjustment_systems.items():
        model.add_adjustment_system(k, v)

    # Register the CDR function as derived value
    model.add_computed_value_process(
        "cdr",
        CdrProc(get_detected_proportion)
    )

    # Apply the VoC stratification and adjust contact rate for single/dual Variants of Concern.
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

    # Stratify model by Victorian subregion (used for Victorian cluster model)
    if is_region_vic:
        cluster_strat = get_cluster_strat(params)
        model.stratify_with(cluster_strat)
        mixing_matrix_function = apply_post_cluster_strat_hacks(params, model, mixing_matrices)

    if is_region_vic2021:

        # Seeding well after vaccination commencement
        seed_date = 560

        cluster_seeds = {
            Region.NORTH_METRO: 0.2,
            Region.WEST_METRO: 0.2,
            Region.SOUTH_METRO: 0.2,
            Region.SOUTH_EAST_METRO: 0.02,
            Region.BARWON_SOUTH_WEST: 0.,
            Region.GRAMPIANS: 0.,
            Region.GIPPSLAND: 0.,
            Region.HUME: 0.1,
            Region.LODDON_MALLEE: 0.,
        }

        for stratum in cluster_seeds:

            seed = cluster_seeds[stratum]

            # Function must be bound in loop with optional argument
            def model_seed_func(time, computed_values, seed=seed):
                return seed if seed_date < time < seed_date + 5. else 0.

            model.add_importation_flow(
                "seed",
                model_seed_func,
                dest=Compartment.EARLY_EXPOSED,
                dest_strata={"cluster": stratum.replace("-", "_")},
            )

    # Contact tracing stratification
    if params.contact_tracing:

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

        early_exposed_untraced_comps = \
            [comp for comp in model.compartments if
             comp.is_match(Compartment.EARLY_EXPOSED, {"tracing": Tracing.UNTRACED})]
        early_exposed_traced_comps = \
            [comp for comp in model.compartments if
             comp.is_match(Compartment.EARLY_EXPOSED, {"tracing": Tracing.TRACED})]

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
        if is_region_vic2021:
            for i_component, roll_out_component in enumerate(vacc_params.roll_out_components):
                total_adult_pop = sum(total_pops[3:])
                cluster_adults_pops = {
                    cluster: cluster_strat.population_split[cluster] * total_adult_pop for
                    cluster in cluster_strat.strata
                }
                for cluster in cluster_strat.strata:
                    add_vaccination_flows(
                        model, vacc_params.roll_out_components[i_component], age_strat.strata,
                        params.vaccination.one_dose, i_component + 1, additional_strata={"cluster": cluster},
                        cluster_pop=cluster_adults_pops[cluster], total_pop=total_adult_pop
                    )
        else:
            for roll_out_component in vacc_params.roll_out_components:
                coverage_override = vacc_params.coverage_override if vacc_params.coverage_override else None
                add_vaccination_flows(
                    model, roll_out_component, age_strat.strata, params.vaccination.one_dose,
                    0, coverage_override
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

    # Set up derived output functions
    if is_region_vic:
        request_victorian_outputs(model, params)
    else:
        request_standard_outputs(model, params)

    # Vaccination
    if params.vaccination:
        request_vaccination_outputs(model, params)

    # Dive into summer internals to over-write mixing matrix
    if is_region_vic:
        model._mixing_matrices = [mixing_matrix_function]

    return model
