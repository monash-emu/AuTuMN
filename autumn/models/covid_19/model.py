from summer import CompartmentalModel

from autumn.settings.region import Region
from autumn.tools.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices
from autumn.models.covid_19.constants import Vaccination
from autumn.tools import inputs
from autumn.tools.project import Params, build_rel_path
from autumn.models.covid_19.detection import CdrProc
from .utils import get_seasonal_forcing
from autumn.models.covid_19.detection import find_cdr_function_from_test_data
from autumn.models.covid_19.utils import calc_compartment_periods

from .constants import (
    COMPARTMENTS, DISEASE_COMPARTMENTS, INFECTIOUS_COMPARTMENTS, Compartment, Tracing, BASE_DATE, History, INFECTION,
    INFECTIOUSNESS_ONSET, INCIDENCE, PROGRESS, RECOVERY, INFECT_DEATH, VACCINE_ELIGIBLE_COMPARTMENTS, VACCINATION_STRATA
)
from .outputs.common import CovidOutputsBuilder
from .parameters import Parameters
from .strat_processing.vaccination import add_requested_vacc_flows, add_vic_regional_vacc, apply_standard_vacc_coverage
from .strat_processing import tracing
from .strat_processing.clinical import AbsRateIsolatedSystem, AbsPropSymptNonHospSystem
from .strat_processing.strains import make_voc_seed_func
from .strat_processing.vaccination import get_second_dose_delay_rate
from .stratifications.agegroup import AGEGROUP_STRATA, get_agegroup_strat
from .stratifications.clinical import get_clinical_strat
from .stratifications.tracing import get_tracing_strat
from .stratifications.strains import get_strain_strat
from .stratifications.history import get_history_strat
from .stratifications.vaccination import get_vaccination_strat

base_params = Params(build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False)


def build_model(params: dict, build_options: dict = None) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.
    """

    params = Parameters(**params)

    # Create the model object
    model = CompartmentalModel(
        times=(params.time.start, params.time.end),
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

    """
    Create the total population.
    """

    # Distribute infectious seed across infectious split sub-compartments
    compartment_periods = calc_compartment_periods(params.sojourn)
    total_disease_time = sum([compartment_periods[comp] for comp in DISEASE_COMPARTMENTS])
    init_pop = {
        comp: params.infectious_seed * compartment_periods[comp] / total_disease_time
        for comp in DISEASE_COMPARTMENTS
    }

    # Get country population by age-group
    country = params.country
    pop = params.population
    total_pops = inputs.get_population_by_agegroup(AGEGROUP_STRATA, country.iso3, pop.region, year=pop.year)

    # Assign the remainder starting population to the S compartment
    init_pop[Compartment.SUSCEPTIBLE] = sum(total_pops) - sum(init_pop.values())
    model.set_initial_population(init_pop)

    """
    Add intercompartmental flows.
    """

    # Use a time-varying, sinusoidal seasonal forcing function or constant value for the contact rate
    contact_rate = get_seasonal_forcing(365., 173., params.seasonal_force, params.contact_rate) if \
        params.seasonal_force else params.contact_rate

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

    """
    Age group stratification.
    """

    mixing_matrices = build_synthetic_matrices(
        country.iso3, params.mixing_matrices.source_iso3, AGEGROUP_STRATA, params.mixing_matrices.age_adjust, pop.region
    )

    age_strat = get_agegroup_strat(params, total_pops, mixing_matrices)
    model.stratify_with(age_strat)

    """
    Clinical stratification.
    """

    is_region_vic = pop.region and pop.region.replace("_", "-").lower() in Region.VICTORIA_SUBREGIONS
    override_test_region = "Victoria" if pop.region and is_region_vic else pop.region

    get_detected_proportion = find_cdr_function_from_test_data(
        params.testing_to_detection, country.iso3, override_test_region, pop.year
    )
    clinical_strat = get_clinical_strat(params)
    model.stratify_with(clinical_strat)

    """
    Case detection.
    """

    model.add_computed_value_process("cdr", CdrProc(get_detected_proportion))

    compartment_periods = calc_compartment_periods(params.sojourn)
    within_early_exposed = 1. / compartment_periods[Compartment.EARLY_EXPOSED]
    model.add_adjustment_system("isolated", AbsRateIsolatedSystem(within_early_exposed))
    model.add_adjustment_system("sympt_non_hosp", AbsPropSymptNonHospSystem(within_early_exposed))

    """
    Variants of concern stratification.
    """

    if params.voc_emergence:
        voc_params = params.voc_emergence

        # Build and apply stratification
        strain_strat = get_strain_strat(voc_params)
        model.stratify_with(strain_strat)

        # Use importation flows to seed VoC cases
        for voc_name, voc_values in voc_params.items():
            voc_seed_func = make_voc_seed_func(voc_values.entry_rate, voc_values.start_time, voc_values.seed_duration)
            model.add_importation_flow(
                f"seed_voc_{voc_name}", voc_seed_func, dest=Compartment.EARLY_EXPOSED, dest_strata={"strain": voc_name}
            )

    """
    Contact tracing stratification.
    """

    if params.contact_tracing:

        # Stratify
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
        early_exposed_untraced_comps = [
            comp for comp in model.compartments if
            comp.is_match(Compartment.EARLY_EXPOSED, {"tracing": Tracing.UNTRACED})
        ]
        early_exposed_traced_comps = [
            comp for comp in model.compartments if
            comp.is_match(Compartment.EARLY_EXPOSED, {"tracing": Tracing.TRACED})
        ]
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

    """
    Vaccination status stratification.
    """

    vacc_params = params.vaccination
    if vacc_params:
        dose_delay_params = vacc_params.second_dose_delay
        is_dosing_active = bool(dose_delay_params)  # Presence of parameter determines stratification by dosing
        is_waning_vacc_immunity = vacc_params.vacc_full_effect_duration  # Similarly this one for waning immunity

        # Work out the strata to be implemented
        if not is_dosing_active and not is_waning_vacc_immunity:
            vacc_strata = VACCINATION_STRATA[: 2]
        elif not is_dosing_active and is_waning_vacc_immunity:
            vacc_strata = VACCINATION_STRATA[: 2] + VACCINATION_STRATA[3:]
        elif is_dosing_active and not is_waning_vacc_immunity:
            vacc_strata = VACCINATION_STRATA[: 3]
        elif is_dosing_active and is_waning_vacc_immunity:
            vacc_strata = VACCINATION_STRATA

        # Get the vaccination stratification object
        vaccination_strat = get_vaccination_strat(params, vacc_strata)
        model.stratify_with(vaccination_strat)

        # Victoria vaccination code is not generalisable
        if is_region_vic:
            add_vic_regional_vacc(model, vacc_params, params.population.region, params.time.start)
        elif params.vaccination.standard_supply:
            apply_standard_vacc_coverage(model, vacc_params.lag, params.time.start, params.country.iso3, total_pops)
        else:
            add_requested_vacc_flows(model, vacc_params)

        # Add transition from single dose to fully vaccinated
        if is_dosing_active:
            second_dose_transition = get_second_dose_delay_rate(dose_delay_params)
            for compartment in VACCINE_ELIGIBLE_COMPARTMENTS:
                model.add_transition_flow(
                    name="second_dose",
                    fractional_rate=second_dose_transition,
                    source=compartment,
                    dest=compartment,
                    source_strata={"vaccination": Vaccination.ONE_DOSE_ONLY},
                    dest_strata={"vaccination": Vaccination.VACCINATED},
                )

        # Add the waning immunity progressions through the strata
        if is_waning_vacc_immunity:
            wane_origin_stratum = Vaccination.VACCINATED if is_dosing_active else Vaccination.ONE_DOSE_ONLY
            for compartment in VACCINE_ELIGIBLE_COMPARTMENTS:
                model.add_transition_flow(
                    name="part_wane",
                    fractional_rate=1. / vacc_params.vacc_full_effect_duration,
                    source=compartment,
                    dest=compartment,
                    source_strata={"vaccination": wane_origin_stratum},
                    dest_strata={"vaccination": Vaccination.PART_WANED},
                )
                model.add_transition_flow(
                    name="full_wane",
                    fractional_rate=1. / vacc_params.vacc_part_effect_duration,
                    source=compartment,
                    dest=compartment,
                    source_strata={"vaccination": Vaccination.PART_WANED},
                    dest_strata={"vaccination": Vaccination.WANED},
                )

    """
    Infection history stratification.
    """

    is_waning_immunity = bool(params.waning_immunity_duration)
    if is_waning_immunity:
        history_strat = get_history_strat(params)
        model.stratify_with(history_strat)

        # Waning immunity (if requested)
        # Note that this approach would mean that the recovered in the naive class have actually previously had Covid
        model.add_transition_flow(
            name="waning_immunity",
            fractional_rate=1. / params.waning_immunity_duration,
            source=Compartment.RECOVERED,
            dest=Compartment.SUSCEPTIBLE,
            source_strata={"history": History.NAIVE},
            dest_strata={"history": History.EXPERIENCED},
        )

    """
    Set up derived output functions
    """

    outputs_builder = CovidOutputsBuilder(model, COMPARTMENTS)

    outputs_builder.request_incidence()
    outputs_builder.request_infection()
    outputs_builder.request_notifications(params.contact_tracing, params.cumul_incidence_start_time)
    outputs_builder.request_progression()
    outputs_builder.request_cdr()
    outputs_builder.request_deaths()
    outputs_builder.request_admissions()
    outputs_builder.request_occupancy(params.sojourn.compartment_periods)
    if params.contact_tracing:
        outputs_builder.request_tracing()
    if params.voc_emergence:
        outputs_builder.request_strains(list(params.voc_emergence.keys()))
    if vacc_params:
        outputs_builder.request_vaccination(is_dosing_active, vacc_strata)
        if len(vacc_params.roll_out_components) > 0 and params.vaccination_risk.calculate:
            outputs_builder.request_vacc_aefis(params.vaccination_risk)

    if is_waning_immunity:
        outputs_builder.request_history()
    else:
        outputs_builder.request_recovered()
        outputs_builder.request_extra_recovered()

    return model
