from summer import CompartmentalModel

from autumn.tools.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices
from autumn.models.covid_19.constants import Vaccination
from autumn.tools import inputs
from autumn.tools.project import Params, build_rel_path
from autumn.models.covid_19.detection import find_cdr_function_from_test_data, CdrProc
from autumn.models.covid_19.utils import calc_compartment_periods

from .constants import (
    COMPARTMENTS, COVID_BASE_DATETIME, DISEASE_COMPARTMENTS, INFECTIOUS_COMPARTMENTS, Compartment, Tracing, History, INFECTION,
    INFECTIOUSNESS_ONSET, INCIDENCE, PROGRESS, RECOVERY, INFECT_DEATH
)
from .outputs.common import CovidOutputsBuilder
from .parameters import Parameters
from .strat_processing.vaccination import add_vacc_rollout_requests, apply_standard_vacc_coverage
from .strat_processing import tracing
from .strat_processing.clinical import AbsRateIsolatedSystem, AbsPropSymptNonHospSystem
from .strat_processing.strains import make_voc_seed_func
from .strat_processing.vaccination import get_second_dose_delay_rate, find_vacc_strata
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

    # Get country/region details
    country = params.country
    pop = params.population

    # Create the model object
    model = CompartmentalModel(
        times=(params.time.start, params.time.end),
        compartments=COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPARTMENTS,
        timestep=params.time.step,
        ref_date=COVID_BASE_DATETIME
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
    seed = params.infectious_seed
    init_pop = {comp: seed * compartment_periods[comp] / total_disease_time for comp in DISEASE_COMPARTMENTS}

    # Get country population by age-group
    total_pops = inputs.get_population_by_agegroup(AGEGROUP_STRATA, country.iso3, pop.region, pop.year)

    # Assign the remainder starting population to the S compartment
    init_pop[Compartment.SUSCEPTIBLE] = sum(total_pops) - sum(init_pop.values())
    model.set_initial_population(init_pop)

    """
    Add intercompartmental flows.
    """

    # Infection
    model.add_infection_frequency_flow(
        name=INFECTION,
        contact_rate=params.contact_rate,
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
        dest=Compartment.SUSCEPTIBLE,
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

    mixing_matrices = build_synthetic_matrices(country.iso3, params.ref_mixing_iso3, AGEGROUP_STRATA, True, pop.region)
    age_strat = get_agegroup_strat(params, total_pops, mixing_matrices)
    model.stratify_with(age_strat)

    """
    Variants of concern stratification.
    """

    voc_ifr_effects = {"wild": 1.}
    voc_hosp_effects = {"wild": 1.}
    if params.voc_emergence:
        voc_params = params.voc_emergence

        # Build and apply stratification
        strain_strat = get_strain_strat(voc_params)
        model.stratify_with(strain_strat)

        # Use importation flows to seed VoC cases
        for voc_name, voc_values in voc_params.items():
            voc_seed_func = make_voc_seed_func(voc_values.entry_rate, voc_values.start_time, voc_values.seed_duration)
            model.add_importation_flow(
                f"seed_voc_{voc_name}", voc_seed_func, dest=Compartment.EARLY_EXPOSED, dest_strata={"strain": voc_name}, split_imports=True
            )

        # Get the adjustments to the IFR for each strain (if not requested, will have defaulted to a value of one)
        voc_ifr_effects.update({s: params.voc_emergence[s].ifr_multiplier for s in strain_strat.strata[1:]})
        voc_hosp_effects.update({s: params.voc_emergence[s].hosp_multiplier for s in strain_strat.strata[1:]})

    """
    Clinical stratification.
    """

    stratified_adjusters = {}
    for voc in voc_ifr_effects.keys():
        stratified_adjusters[voc] = {
            "ifr": params.infection_fatality.multiplier * voc_ifr_effects[voc],
            "hosp": params.clinical_stratification.props.hospital.multiplier * voc_hosp_effects[voc],
            "sympt": params.clinical_stratification.props.symptomatic.multiplier,
        }

    get_detected_proportion = find_cdr_function_from_test_data(
        params.testing_to_detection, country.iso3, pop.region, pop.year
    )
    clinical_strat = get_clinical_strat(params, stratified_adjusters)
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

        # Determine the vaccination stratification structure
        dose_delay_params = vacc_params.second_dose_delay
        is_dosing_active = bool(dose_delay_params)  # Presence of parameter determines stratification by dosing
        waning_vacc_immunity = vacc_params.vacc_full_effect_duration  # Similarly this one for waning immunity
        is_boosting = bool(params.vaccination.boost_delay)
        vacc_strata, wane_origin_stratum = find_vacc_strata(is_dosing_active, bool(waning_vacc_immunity), is_boosting)

        # Get the vaccination stratification object
        vaccination_strat = get_vaccination_strat(params, vacc_strata, stratified_adjusters)
        model.stratify_with(vaccination_strat)

        if params.vaccination.standard_supply:
            apply_standard_vacc_coverage(
                model, vacc_params.lag, params.country.iso3, total_pops, params.vaccination.one_dose,
                params.description == "BASELINE"
            )
        else:
            add_vacc_rollout_requests(model, vacc_params)

        # Add transition from single dose to fully vaccinated
        if is_dosing_active:
            second_dose_transition = get_second_dose_delay_rate(dose_delay_params)
            model.add_transition_flow(
                name="second_dose",
                fractional_rate=second_dose_transition,
                source=Compartment.SUSCEPTIBLE,
                dest=Compartment.SUSCEPTIBLE,
                source_strata={"vaccination": Vaccination.ONE_DOSE_ONLY},
                dest_strata={"vaccination": Vaccination.VACCINATED},
            )

        # Add the waning immunity progressions through the strata
        if waning_vacc_immunity:
            model.add_transition_flow(
                name="part_wane",
                fractional_rate=1. / waning_vacc_immunity,
                source=Compartment.SUSCEPTIBLE,
                dest=Compartment.SUSCEPTIBLE,
                source_strata={"vaccination": wane_origin_stratum},
                dest_strata={"vaccination": Vaccination.PART_WANED},
            )
            model.add_transition_flow(
                name="full_wane",
                fractional_rate=1. / vacc_params.vacc_part_effect_duration,
                source=Compartment.SUSCEPTIBLE,
                dest=Compartment.SUSCEPTIBLE,
                source_strata={"vaccination": Vaccination.PART_WANED},
                dest_strata={"vaccination": Vaccination.WANED},
            )

        # Apply the process of boosting to those who have previously received their full course of vaccination only
        if params.vaccination.boost_delay:
            for boost_eligible_stratum in (Vaccination.VACCINATED, Vaccination.WANED):
                model.add_transition_flow(
                    name=f"boost_{boost_eligible_stratum}",
                    fractional_rate=1. / params.vaccination.boost_delay,
                    source=Compartment.SUSCEPTIBLE,
                    dest=Compartment.SUSCEPTIBLE,
                    source_strata={"vaccination": boost_eligible_stratum},
                    dest_strata={"vaccination": Vaccination.BOOSTED},
                )

    """
    Infection history stratification.
    """

    history_strat = get_history_strat(params, stratified_adjusters)
    model.stratify_with(history_strat)

    # Manipulate all the recovery flows by digging into the summer object to make them go to the experienced stratum
    for flow in [f for f in model._flows if f.name == RECOVERY]:
        updated_strata = flow.dest.strata.copy()
        updated_strata["history"] = History.EXPERIENCED
        # Find the destination compartment matching the original (but with updated history)
        new_dest_comp = model.get_matching_compartments(flow.dest.name, updated_strata)
        assert len(new_dest_comp) == 1, f"Multiple compartments match query for {flow.dest}"
        flow.dest = new_dest_comp[0]

    # Apply waning immunity if present
    if params.history.natural_immunity_duration:
        model.add_transition_flow(
            name="waning_immunity",
            fractional_rate=1. / params.history.natural_immunity_duration,
            source=Compartment.SUSCEPTIBLE,
            dest=Compartment.SUSCEPTIBLE,
            source_strata={"history": History.EXPERIENCED},
            dest_strata={"history": History.WANED},
        )

    """
    Set up derived output functions
    """

    outputs_builder = CovidOutputsBuilder(model, COMPARTMENTS, bool(params.contact_tracing))

    outputs_builder.request_incidence()
    outputs_builder.request_infection()
    outputs_builder.request_notifications(params.cumul_incidence_start_time, params.hospital_reporting)
    outputs_builder.request_non_hosp_notifications()
    outputs_builder.request_adult_paeds_notifications()
    outputs_builder.request_cdr()
    outputs_builder.request_deaths()
    outputs_builder.request_admissions()
    outputs_builder.request_occupancy(params.sojourn.compartment_periods)
    outputs_builder.request_experienced()
    if params.contact_tracing:
        outputs_builder.request_tracing()
    if params.voc_emergence:
        outputs_builder.request_strains(list(params.voc_emergence.keys()))
    if vacc_params:
        outputs_builder.request_vaccination(is_dosing_active, vacc_strata)
        if len(vacc_params.roll_out_components) > 0 and params.vaccination_risk.calculate:
            outputs_builder.request_vacc_aefis(params.vaccination_risk)

    return model
