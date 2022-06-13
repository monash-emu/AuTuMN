from typing import List
import pandas as pd

from summer import CompartmentalModel

from autumn.core import inputs
from autumn.core.project import Params, build_rel_path
from autumn.model_features.random_process import RandomProcessProc
from autumn.core.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices
from autumn.core.utils.utils import multiply_function_or_constant
from autumn.model_features.computed_values import FunctionWrapper
from autumn.model_features.random_process import get_random_process
from .detection import get_cdr_func
from .outputs import SmSirOutputsBuilder
from .parameters import Parameters, Sojourns, CompartmentSojourn
from .constants import BASE_COMPARTMENTS, Compartment, FlowName
from .stratifications.agegroup import get_agegroup_strat
from .stratifications.immunity import (
    get_immunity_strat,
    adjust_susceptible_infection_without_strains,
    adjust_susceptible_infection_with_strains,
    adjust_reinfection_without_strains,
    adjust_reinfection_with_strains,
    apply_reported_vacc_coverage,
    apply_reported_vacc_coverage_with_booster,
)
from .stratifications.strains import get_strain_strat, seed_vocs, apply_reinfection_flows_with_strains
from .stratifications.clinical import get_clinical_strat
from autumn.models.sm_sir.stratifications.agegroup import convert_param_agegroups
from autumn.settings.constants import COVID_BASE_DATETIME

# Base date used to calculate mixing matrix times
base_params = Params(build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False)


def get_compartments(
        sojourns: Sojourns
) -> List[str]:
    """
    Find the model compartments that are applicable to the parameter requests, based on the sojourn times structure.
        - Start with just the base SEIR structure
        - Add in a second serial latent compartment if there is a proportion of the latent period early
        - Add in a second serial active compartment if there is a proportion of the active period early

    Args:
        sojourns: User requested sojourn times

    Returns:
        The names of the model compartments to be implemented

    """

    # Make a copy, we really don't want to append to something that's meant to be a constant...
    compartments = BASE_COMPARTMENTS.copy()
    
    if sojourns.latent.proportion_early:
        compartments.append(Compartment.LATENT_LATE)
    if sojourns.active.proportion_early:
        compartments.append(Compartment.INFECTIOUS_LATE)
    if sojourns.recovered:
        compartments.append(Compartment.WANED)

    return compartments


def assign_population(
        seed: float,
        total_pop: int,
        model: CompartmentalModel
):
    """
    Assign the starting population to the model according to the user requests and total population of the model.

    Args:
        seed: The starting infectious seed
        total_pop: The total population being modelled
        model: The summer compartmental model object to have its starting population set

    """

    # Split by seed and remainder susceptible
    susceptible = total_pop - seed
    init_pop = {
        Compartment.INFECTIOUS: seed,
        Compartment.SUSCEPTIBLE: susceptible,
    }

    # Assign to the model
    model.set_initial_population(init_pop)


def add_latent_transitions(
        latent_sojourn_params: CompartmentSojourn,
        model: CompartmentalModel,
):
    """
    Add the transition flows taking people from infection through to infectiousness, depending on the model structure
    requested.
    Absence of the latent compartment entirely is not supported currently, because this would require us to calculate
    incidence from multiple flow names (which is possible, but would add complexity to the code).

    Args:
        latent_sojourn_params: The user requests relating to the latent period
        model: The summer compartmental model object to have the flows applied to it

    Returns:
        The name of the destination compartment for infection processes
        The name of the flow that transitions people into the infectious state

    """

    # The total time spent in the latent stage
    latent_sojourn = latent_sojourn_params.total_time

    # The proportion of that time spent in early latency
    latent_early_prop = latent_sojourn_params.proportion_early

    # If the latent compartment is divided into an early and a late stage
    if latent_early_prop:

        # Apply the transition between the two latent compartments
        model.add_transition_flow(
            name=FlowName.WITHIN_LATENT,
            fractional_rate=1. / latent_sojourn / latent_early_prop,
            source=Compartment.LATENT,
            dest=Compartment.LATENT_LATE,
        )

        # The parameters for the transition out of latency (through the late latent stage)
        prop_latent_late = 1. - latent_early_prop
        progress_rate = 1. / latent_sojourn / prop_latent_late
        progress_origin = Compartment.LATENT_LATE

    # If the latent stage is just one compartment
    else:

        # The parameters for transition out of the single latent compartment
        progress_origin = Compartment.LATENT
        progress_rate = 1. / latent_sojourn

    # Apply the transition out of latency flow
    model.add_transition_flow(
        name=FlowName.PROGRESSION,
        fractional_rate=progress_rate,
        source=progress_origin,
        dest=Compartment.INFECTIOUS,
    )


def add_active_transitions(
        active_sojourn_params: CompartmentSojourn,
        model: CompartmentalModel,
):
    """
    Implement the transitions through and out of the active compartment, based on the user requests regarding sojourn
    times for the active compartment.

    Args:
        active_sojourn_params: The user requests relating to the active period
        model: The summer compartmental model object to have the flows applied to it

    """

    active_sojourn = active_sojourn_params.total_time
    active_early_prop = active_sojourn_params.proportion_early

    # If the active compartment is divided into an early and a late stage
    if active_early_prop:

        # Apply the transition between the two active compartments
        model.add_transition_flow(
            name=FlowName.WITHIN_INFECTIOUS,
            fractional_rate=1. / active_sojourn / active_early_prop,
            source=Compartment.INFECTIOUS,
            dest=Compartment.INFECTIOUS_LATE,
        )

        # The parameters for the transition out of active disease (through the late active stage)
        prop_active_late = 1. - active_early_prop
        recovery_rate = 1. / active_sojourn / prop_active_late
        recovery_origin = Compartment.INFECTIOUS_LATE

    # If the active compartment is just one compartment
    else:

        # The parameters for transition out of the single active compartment
        recovery_origin = Compartment.INFECTIOUS
        recovery_rate = 1. / active_sojourn

    # Implement the recovery flow, now that we know the source and the rate
    model.add_transition_flow(
        name=FlowName.RECOVERY,
        fractional_rate=recovery_rate,
        source=recovery_origin,
        dest=Compartment.RECOVERED,
    )


def apply_reinfection_flows_without_strains(
        model: CompartmentalModel,
        infection_dest: str,
        age_groups: List[str],
        contact_rate: float,
        suscept_props: pd.Series,
):
    """
    Apply the reinfection flows in the case of a single-strain model. Note that in this case, only the late reinfection
    flow (i.e. coming out of the waned compartment) is relevant.

    Args:
        model: The SM-SIR model being adapted
        infection_dest: Where people end up first after having been infected
        age_groups: The modelled age groups
        contact_rate: The model's contact rate
        suscept_props: Adjustments to the rate of infection of susceptibles based on modelled age groups

    """

    for age_group in age_groups:
        age_adjuster = suscept_props[age_group]
        age_filter = {"agegroup": age_group}

        contact_rate_adjuster = age_adjuster
        age_contact_rate = multiply_function_or_constant(contact_rate, contact_rate_adjuster)

        model.add_infection_frequency_flow(
            FlowName.LATE_REINFECTION,
            age_contact_rate,
            Compartment.WANED,
            infection_dest,
            age_filter,
            age_filter,
        )


def build_model(
        params: dict,
        build_options: dict = None
) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.

    Args:
        params: The validated user-requested parameters
        build_options:

    Returns:
        The "SM-SIR" model, currently being used only for COVID-19

    """

    # Get the parameters and extract some of the more used ones to have simpler names
    params = Parameters(**params)

    country = params.country
    pop = params.population
    iso3 = country.iso3
    region = pop.region
    age_groups = [str(age) for age in params.age_groups]
    age_strat_params = params.age_stratification
    sojourns = params.sojourns
    detect_prop = params.detect_prop
    testing_params = params.testing_to_detection
    suscept_req = age_strat_params.susceptibility
    sympt_req = age_strat_params.prop_symptomatic
    time_params = params.time
    time_to_event_params = params.time_from_onset_to_event

    # Determine the compartments, including which are infectious
    compartment_types = get_compartments(sojourns)
    infectious_compartments = [comp for comp in compartment_types if "infectious" in comp]

    # Create the model object
    model = CompartmentalModel(
        times=(time_params.start, time_params.end),
        compartments=compartment_types,
        infectious_compartments=infectious_compartments,
        timestep=time_params.step,
        ref_date=COVID_BASE_DATETIME,
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

    # Get country population by age-group
    age_pops = pd.Series(
        inputs.get_population_by_agegroup(age_groups, iso3, region, pop.year),
        index=age_groups
    )

    # Assign the population to compartments
    assign_population(params.infectious_seed, age_pops.sum(), model)

    """
    Add intercompartmental flows
    """

    # Latency
    add_latent_transitions(sojourns.latent, model)
    infection_dest, infectious_entry_flow = Compartment.LATENT, FlowName.PROGRESSION

    # Transmission
    if params.activate_random_process:

        # Store random process as a computed value to make it available as an output
        rp_function, contact_rate = get_random_process(
            time_params,
            params.random_process,
            params.contact_rate
        )
        model.add_computed_value_process(
            "transformed_random_process",
            RandomProcessProc(rp_function)
        )

    else:
        contact_rate = params.contact_rate

    # Add the process of infecting the susceptibles for the first time
    model.add_infection_frequency_flow(
        name=FlowName.INFECTION,
        contact_rate=contact_rate,
        source=Compartment.SUSCEPTIBLE,
        dest=infection_dest,
    )

    # Active transition flows
    add_active_transitions(sojourns.active, model)

    # Add waning transition if waning being implemented
    if Compartment.WANED in compartment_types:
        model.add_transition_flow(
            name=FlowName.WANING,
            fractional_rate=1. / sojourns.recovered,
            source=Compartment.RECOVERED,
            dest=Compartment.WANED,
        )

    """
    Apply age stratification
    """

    # Preprocess age-specific parameters to match model age bands if requested in this way
    if type(suscept_req) == dict:
        suscept_adjs = convert_param_agegroups(iso3, region, suscept_req, age_groups)
    else:
        suscept_adjs = suscept_req  # In which case it should be None or a float, confirmed in parameter validation

    if type(sympt_req) == dict:
        sympt_props = convert_param_agegroups(iso3, region, sympt_req, age_groups)
        sympt_props.index = sympt_props.index.map(str)  # Change int indices to string to match model format
    else:
        sympt_props = sympt_req  # In which case it should be None or a float

    # Get the age-specific mixing matrices
    mixing_matrices = build_synthetic_matrices(
        iso3,
        params.ref_mixing_iso3,
        [int(age) for age in age_groups],
        True,  # Always age-adjust, could change this to being a parameter
        region
    )

    # Get the actual age stratification now
    age_strat = get_agegroup_strat(
        params,
        age_groups,
        age_pops,
        mixing_matrices,
        compartment_types,
        params.is_dynamic_mixing_matrix,
        suscept_adjs,
    )
    model.stratify_with(age_strat)

    """
    Testing-related processes
    """

    if testing_params or detect_prop < 1.:
        is_undetected = True
        cdr_func, non_detect_func = get_cdr_func(detect_prop, testing_params, pop, iso3)
        model.add_computed_value_process("cdr", FunctionWrapper(cdr_func))
        model.add_computed_value_process("undetected_prop", FunctionWrapper(non_detect_func))
    else:
        is_undetected, non_detect_func, cdr_func = False, None, None

    """
    Apply clinical stratification
    """

    # Apply the clinical stratification, or a None to indicate no clinical stratification to get a list for the outputs
    clinical_strat = get_clinical_strat(
        model, compartment_types, params, infectious_entry_flow, sympt_props, non_detect_func, cdr_func,
    )
    if clinical_strat:
        model.stratify_with(clinical_strat)
        clinical_strata = clinical_strat.strata

    # Need a placeholder for outputs otherwise
    else:
        clinical_strata = [""]

    """
    Apply strains stratification
    """

    voc_params = params.voc_emergence
    if params.voc_emergence:

        # Build and apply the stratification
        strain_strat = get_strain_strat(voc_params, compartment_types)
        model.stratify_with(strain_strat)

        # Seed the VoCs from the requested point in time
        seed_vocs(model, voc_params, Compartment.INFECTIOUS)

        # Keep track of the strain strata, which are needed for various purposes below
        strain_strata = strain_strat.strata

    # Need a placeholder for outputs and reinfection flows otherwise
    else:
        strain_strata = [""]

    """
    Apply the reinfection flows (knowing the strain stratification)
    """

    if voc_params:
        apply_reinfection_flows_with_strains(
            model,
            compartment_types,
            infection_dest,
            age_groups,
            params.voc_emergence,
            strain_strata,
            contact_rate,
            suscept_adjs,
        )
    else:
        # for a single-strain model, reinfection may only occur from the waned compartment
        if Compartment.WANED in compartment_types:
            apply_reinfection_flows_without_strains(
                model,
                infection_dest,
                age_groups,
                contact_rate,
                suscept_adjs,
            )

    """
    Immunity stratification
    """

    # Get the immunity stratification
    immunity_strat = get_immunity_strat(
        compartment_types,
        params.immunity_stratification,
    )

    immunity_params = params.immunity_stratification

    # Adjust infection of susceptibles for immunity status
    reinfection_flows = [FlowName.EARLY_REINFECTION] if voc_params else []
    if Compartment.WANED in compartment_types:
        reinfection_flows.append(FlowName.LATE_REINFECTION)

    immunity_low_risk_reduction = immunity_params.infection_risk_reduction.low
    immunity_high_risk_reduction = immunity_params.infection_risk_reduction.high

    if voc_params:
        # The code should run fine if VoC parameters have been submitted but the strain stratification hasn't been
        # implemented yet - but at this stage we assume we don't want it to
        msg = "Strain stratification not present in model"
        assert "strain" in [strat.name for strat in model._stratifications], msg
        adjust_susceptible_infection_with_strains(
            immunity_low_risk_reduction,
            immunity_high_risk_reduction,
            immunity_strat,
            voc_params,
        )
        adjust_reinfection_with_strains(
            immunity_low_risk_reduction,
            immunity_high_risk_reduction,
            immunity_strat,
            reinfection_flows,
            voc_params,
        )
    else:
        adjust_susceptible_infection_without_strains(
            immunity_low_risk_reduction,
            immunity_high_risk_reduction,
            immunity_strat,
        )
        adjust_reinfection_without_strains(
            immunity_low_risk_reduction,
            immunity_high_risk_reduction,
            immunity_strat,
            reinfection_flows,
        )

    # Apply the immunity stratification
    model.stratify_with(immunity_strat)

    # Implement the dynamic immunity process
    vacc_coverage_available = ["BGD", "PHL", "BTN", "VNM"]
    vacc_region_available = ["Metro Manila", "Hanoi", "Ho Chi Minh City", None]
    is_dynamic_immunity = iso3 in vacc_coverage_available and region in vacc_region_available

    if is_dynamic_immunity:
        thinning = 20 if iso3 == "BGD" else None

        if iso3 == "PHL" or iso3 == "VNM":
            apply_reported_vacc_coverage_with_booster(
                compartment_types,
                model,
                iso3,
                region,
                thinning=thinning,
                model_start_time=params.time.start,
                start_immune_prop=params.immunity_stratification.prop_immune,
                start_prop_high_among_immune=params.immunity_stratification.prop_high_among_immune,
                booster_effect_duration=params.booster_effect_duration,
                future_monthly_booster_rate=params.future_monthly_booster_rate,
                model_end_time=params.time.end
            )
        else:
            apply_reported_vacc_coverage(
                compartment_types,
                model,
                iso3,
                thinning=thinning,
                model_start_time=params.time.start,
                start_immune_prop=params.immunity_stratification.prop_immune,
                additional_immunity_points=params.additional_immunity,
            )

    """
    Get the applicable outputs
    """

    model_times = model.times

    outputs_builder = SmSirOutputsBuilder(model, compartment_types)

    if is_undetected:
        outputs_builder.request_cdr()

    # Determine what flow will be used to track disease incidence
    if Compartment.INFECTIOUS_LATE in compartment_types:
        incidence_flow = FlowName.WITHIN_INFECTIOUS
    elif Compartment.LATENT in compartment_types:
        incidence_flow = FlowName.PROGRESSION
    else:
        incidence_flow = FlowName.INFECTION
    outputs_builder.request_incidence(
        age_groups,
        clinical_strata,
        strain_strata,
        incidence_flow,
        params.request_incidence_by_age
    )

    outputs_builder.request_notifications(
        time_to_event_params.notification,
        model_times
    )
    outputs_builder.request_hospitalisations(
        model_times,
        age_groups,
        strain_strata,
        iso3,
        region,
        age_strat_params.prop_hospital,
        time_to_event_params.hospitalisation,
        params.hospital_stay.hospital_all,
        voc_params,
    )
    outputs_builder.request_icu_outputs(
        params.prop_icu_among_hospitalised,
        time_to_event_params.icu_admission,
        params.hospital_stay.icu,
        strain_strata,
        model_times,
        voc_params,
        age_groups,
    )
    outputs_builder.request_infection_deaths(
        model_times,
        age_groups,
        strain_strata,
        iso3,
        region,
        age_strat_params.cfr,
        time_to_event_params.death,
        voc_params,
    )
    outputs_builder.request_recovered_proportion(compartment_types)
    if params.activate_random_process:
        outputs_builder.request_random_process_outputs()

    # if is_dynamic_immunity:
    outputs_builder.request_immunity_props(immunity_strat.strata)

    # cumulative output requests
    cumulative_start_time = params.cumulative_start_time if params.cumulative_start_time else None
    outputs_builder.request_cumulative_outputs(
        params.requested_cumulative_outputs,
        cumulative_start_time
    )

    return model
