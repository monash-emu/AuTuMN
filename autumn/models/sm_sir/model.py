from datetime import date
from math import exp
from typing import List, Tuple

from summer import CompartmentalModel

from autumn.tools import inputs
from autumn.tools.project import Params, build_rel_path
from autumn.tools.random_process import RandomProcess
from autumn.tools.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices
from .outputs import SmSirOutputsBuilder
from .parameters import Parameters, Sojourns, CompartmentSojourn, Time, RandomProcess
from .computed_values.random_process_compute import RandomProcessProc
from .constants import BASE_COMPARTMENTS, Compartment, FlowName
from .stratifications.agegroup import get_agegroup_strat
from .stratifications.immunity import get_immunity_strat
from .stratifications.strains import get_strain_strat
from .stratifications.clinical import get_clinical_strat
from .strat_processing.strains import seed_vocs, apply_reinfection_flows
from autumn.models.sm_sir.strat_processing.agegroup import convert_param_agegroups


# Base date used to calculate mixing matrix times.
BASE_DATE = date(2019, 12, 31)
base_params = Params(build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False)


def get_compartments(
        sojourns: Sojourns
) -> List[str]:
    """
    Find the model compartments that are applicable to the parameter requests, based on the sojourn times structure.
        - Start with just the base SIR structure
        - Add in the latent compartment if there is a latent period request
        - Add in a second serial latent compartment if there is a proportion of the latent period early
        - Add in a second serial active compartment if there is a proportion of the active period early

    Args:
        sojourns: User requested sojourn times

    Returns:
        The names of the model compartments to be implemented.

    """

    # Make a copy, we really don't want to append to something that's meant to be a constant...
    compartments = BASE_COMPARTMENTS.copy()
    
    if sojourns.latent:
        compartments.append(Compartment.LATENT)
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


def get_random_process(
        time_params: Time,
        process_params: RandomProcess,
        contact_rate_value: float,
) -> Tuple[callable, callable]:
    """
    Work out the process that will contribute to the random process.

    Args:
        time_params: Start and end time of the model
        process_params: Parameters relating to the random process
        contact_rate_value: The risk of transmission per contact

    Returns:
        The random process function and the contact rate (here a summer-ready format transition function)

    """

    # build the random process, using default values and coefficients
    rp = set_up_random_process(time_params.start, time_params.end)

    # update random process details based on the model parameters
    rp.update_config_from_params(process_params)

    # Create function returning exp(W), where W is the random process
    rp_time_variant_func = rp.create_random_process_function(transform_func=lambda w: exp(w))

    # Create the time-variant contact rate that uses our computed random process
    def contact_rate_func(t, computed_values):
        return contact_rate_value * computed_values["transformed_random_process"]

    return rp_time_variant_func, contact_rate_func


def add_latent_transitions(
        latent_sojourn_params: CompartmentSojourn,
        model: CompartmentalModel,
) -> Tuple[str, str]:
    """
    Add the transition flows taking people from infection through to infectiousness, depending on the model structure
    requested.

    Args:
        latent_sojourn_params: The user requests relating to the latent period
        model: The summer compartmental model object to have the flows applied to it

    Returns:
        The name of the destination compartment for infection processes
        The name of the flow that transitions people into the infectious state

    """

    # If there is a latent stage in between infection and the onset of infectiousness
    if latent_sojourn_params:

        # The total time spent in the latent stage
        latent_sojourn = latent_sojourn_params.total_time

        # The proportion of that time spent in early latency
        latent_early_prop = latent_sojourn_params.proportion_early

        # If the latent compartment is divided into an early and a late stage
        if latent_early_prop:

            # The early latent period
            early_sojourn = latent_sojourn * latent_early_prop

            # Apply the transition between the two latent compartments
            model.add_transition_flow(
                name=FlowName.WITHIN_LATENT,
                fractional_rate=1. / early_sojourn,
                source=Compartment.LATENT,
                dest=Compartment.LATENT_LATE,
            )

            # The parameters for the transition out of latency (through the late latent stage)
            prop_latent_late = 1. - latent_early_prop
            progress_rate = 1. / latent_sojourn / prop_latent_late
            progress_origin = Compartment.LATENT_LATE

        # If the latent compartment is just one compartment
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

        # Record the infection destination and the name of the flow that takes us into the infectious compartment
        infection_dest = Compartment.LATENT
        infectious_entry_flow = FlowName.PROGRESSION

    # If no latent stage parameters are requested, infection takes us straight to the infectious compartment
    else:
        infection_dest = Compartment.INFECTIOUS
        infectious_entry_flow = FlowName.INFECTION

    return infection_dest, infectious_entry_flow


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


def get_smsir_outputs_builder(
        iso3,
        region,
        model,
        compartment_types,
        is_undetected,
        age_groups,
        clinical_strata,
        strain_strata,
        hosp_props,
        hosp_stay,
        icu_risk,
        time_to_event_params,
        ifr_props_params,
        voc_params,
        random_process,
):
    # FIXME: This function needs a docstring

    model_times = model.times

    outputs_builder = SmSirOutputsBuilder(model, compartment_types)

    if is_undetected:
        outputs_builder.request_cdr()
    outputs_builder.request_incidence(compartment_types, age_groups, clinical_strata, strain_strata)
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
        hosp_props,
        time_to_event_params.hospitalisation,
        hosp_stay.hospital_all,
        voc_params,
    )
    outputs_builder.request_icu_outputs(
        icu_risk,
        time_to_event_params.icu_admission,
        hosp_stay.icu,
        model_times,
    )
    outputs_builder.request_infection_deaths(
        model_times,
        age_groups,
        clinical_strata,
        strain_strata,
        iso3,
        region,
        ifr_props_params,
        time_to_event_params.death,
        voc_params,
    )
    outputs_builder.request_recovered_proportion(compartment_types)
    if random_process:
        outputs_builder.request_random_process_outputs()


def build_model(params: dict, build_options: dict = None) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.
    """

    params = Parameters(**params)

    # Get country/region details
    country = params.country
    pop = params.population

    # Need a placeholder for the immunity stratification a nd output loops if strains not implemented
    strain_strata = [""]

    # Preprocess age-specific parameters to match model age bands - needed for both population and age stratification
    age_groups = params.age_groups
    age_strat_params = params.age_stratification

    suscept_req = age_strat_params.susceptibility
    susc_props = convert_param_agegroups(country.iso3, pop.region, suscept_req, age_groups) if suscept_req else None
    sympt_req = age_strat_params.prop_symptomatic
    sympt_props = convert_param_agegroups(country.iso3, pop.region, sympt_req, age_groups) if sympt_req else None

    # Determine the compartments
    compartment_types = get_compartments(params.sojourns)

    # Create the model object
    model = CompartmentalModel(
        times=(params.time.start, params.time.end),
        compartments=compartment_types,
        infectious_compartments=[Compartment.INFECTIOUS],
        timestep=params.time.step,
        ref_date=BASE_DATE,
    )

    """
    Check build options.
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
    Create the total population.
    """

    # Get country population by age-group
    total_pops = inputs.get_population_by_agegroup(age_groups, country.iso3, pop.region, pop.year)

    # Assign the population to compartments
    assign_population(params.infectious_seed, sum(total_pops), model)

    """
    Add intercompartmental flows.
    """

    # Latency
    infection_dest, infectious_entry_flow = add_latent_transitions(params.sojourns.latent, model)

    # Transmission
    if params.activate_random_process:

        # Store random process as a computed value to make it available as an output
        rp_function, contact_rate = get_random_process(
            params.time,
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
    add_active_transitions(params.sojourns.active, model)

    # Add waning transition if waning being implemented
    if "waned" in compartment_types:
        model.add_transition_flow(
            name=FlowName.WANING,
            fractional_rate=1. / params.sojourns.recovered.total_time,
            source=Compartment.RECOVERED,
            dest=Compartment.WANED,
        )

    """
    Apply age stratification
    """

    mixing_matrices = build_synthetic_matrices(
        country.iso3,
        params.ref_mixing_iso3,
        age_groups,
        True,  # Always age-adjust, could change this to being a parameter
        pop.region
    )
    age_strat = get_agegroup_strat(
        params,
        total_pops,
        mixing_matrices,
        compartment_types,
        params.is_dynamic_mixing_matrix,
        susc_props,
    )
    model.stratify_with(age_strat)

    """
    Apply clinical stratification (must come after age stratification if asymptomatic props being used)
    """

    # Work out if clinical stratification needs to be applied, either because of asymptomatics or incomplete detection
    detect_prop = params.detect_prop
    is_undetected = params.testing_to_detection or detect_prop < 1.
    if sympt_props:
        msg = "Attempted to apply differential symptomatic proportions by age, but model not age stratified"
        model_stratifications = [model._stratifications[i].name for i in range(len(model._stratifications))]
        assert "agegroup" in model_stratifications, msg

    # Get and apply the clinical stratification, or a None to indicate no clinical stratification for the outputs
    if is_undetected or sympt_props:
        clinical_strat = get_clinical_strat(
            model,
            compartment_types,
            params,
            age_groups,
            infectious_entry_flow,
            detect_prop,
            is_undetected,
            sympt_props
        )
        model.stratify_with(clinical_strat)
        clinical_strata = clinical_strat.strata
    else:
        clinical_strata = None

    """
    Apply strains stratification
    """

    if params.voc_emergence:
        voc_params = params.voc_emergence

        # Build and apply stratification
        strain_strat = get_strain_strat(voc_params, compartment_types)
        model.stratify_with(strain_strat)

        # Seed the VoCs from the point in time
        seed_vocs(model, voc_params, Compartment.INFECTIOUS)

        # Keep track of the strain strata, which are needed for various purposes below
        strain_strata = strain_strat.strata

    """
    Apply the reinfection flows (knowing the strain stratification)
    """

    apply_reinfection_flows(model, compartment_types, infection_dest, params.voc_emergence, strain_strata, contact_rate)

    """
    Apply immunity stratification
    """

    immunity_strat = get_immunity_strat(
        compartment_types,
        params.immunity_stratification,
        strain_strata,
        params.voc_emergence,
    )
    model.stratify_with(immunity_strat)

    """
    Get the applicable outputs.
    """

    get_smsir_outputs_builder(
        params.country.iso3,
        params.population.region,
        model,
        compartment_types,
        is_undetected,
        age_groups,
        clinical_strata,
        strain_strata,
        params.age_stratification.prop_hospital,
        params.hospital_stay,
        params.prop_icu_among_hospitalised,
        params.time_from_onset_to_event,
        params.age_stratification.ifr,
        params.voc_emergence,
        bool(params.activate_random_process),
    )

    return model


def set_up_random_process(start_time, end_time):
    return RandomProcess(order=2, period=30, start_time=start_time, end_time=end_time)
